"""ESKF boundary condition intializer. Processes static pre-ignition
telemetry to compute the initial 15x1 state vector and P_o covariance matrix"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

project_root = Path(__file__).resolve().parent.parent.parent
phase3_dir = project_root / "source" / "phase3_nav_eskf_rts"
sys.path.append(str(phase3_dir))

import geodesy_math as mat
import kinematics_ins as kin


class staticInitializer:
    """Operates pre-ignition telemetry data to calculate operational noise-floors,
    static turn-on biases and the initial Tait-Bryan attitude angles."""

    def __init__(self, data_path: Path, config: Dict):
        self.data_path = data_path
        self.config = config

        self.imu_data = pd.DataFrame()
        self.mag_data = pd.DataFrame()
        self.gnss_data = pd.DataFrame()

        # Pre-defined global constants
        self.omega_e = self.config["omega_e"]
        self.g_local = self.config["g_local"]

        # Hard and Soft Iron calibration requirement from Magnetometer - for Attitude Extraction
        self.m_hi_vec = self.config.get("m_hi_vec", np.zeros(3, dtype=np.float64))
        self.m_si_mat = self.config.get("m_si_mat", np.eye(3, dtype=np.float64))
        # Invert Matrix to apply it into ESKF
        self.inv_m_si = np.linalg.inv(self.m_si_mat)

        # IMU Calibration Matrices
        self.s_a_ppm = self.config["s_a_ppm"]
        self.s_g_ppm = self.config["s_g_ppm"]
        self.m_a_deg = self.config["m_a_deg"]
        self.m_g_deg = self.config["m_g_deg"]

    def load_telem(self) -> None:
        """Loads the raw pre-ignition asynchronous datasets."""
        imu_path = self.data_path / "preign_data_imu.csv"
        mag_path = self.data_path / "preign_data_mag.csv"
        gnss_path = self.data_path / "preign_data_gnss.csv"

        if not (imu_path.exists() and mag_path.exists() and gnss_path.exists()):
            raise FileNotFoundError("Critical Error: Pre-ignition datasets are missing")

        self.imu_data = pd.read_csv(imu_path)
        self.mag_data = pd.read_csv(mag_path)
        self.gnss_data = pd.read_csv(gnss_path)

    def _compute_initial_att(
        self, f_mean: np.ndarray, m_mean: np.ndarray
    ) -> np.ndarray:
        """Calculates initial attitude based in Tait-Bryan Angles: roll, pitch, yaw
        by using accelerometers leveling and magnetic aligning."""
        f_x, f_y, f_z = f_mean
        m_x, m_y, m_z = m_mean

        roll0 = np.arctan2(-f_y, -f_z)
        pitch0 = np.arctan2(f_x, np.sqrt(f_y**2 + f_z**2))

        # Rotation to ENU frame: eliminates the vehicle inclination effect
        mx_h = (
            m_x * np.cos(pitch0)
            + m_y * np.sin(roll0) * np.sin(pitch0)
            + m_z * np.cos(roll0) * np.sin(pitch0)
        )
        my_h = m_y * np.cos(roll0) - m_z * np.sin(roll0)

        # Yaw is obtained by magnetic field vector
        yaw0 = np.arctan2(-my_h, mx_h)

        return np.array([roll0, pitch0, yaw0], dtype=np.float64)

    def gen_initial_state(self) -> Dict[str, np.ndarray]:
        """Constructs the 15x1 initial state vector x_0 and 15x15 covariance matrix P_0."""
        self.load_telem()
        print("Constructing initial state vector x_0 and covariance matrix P_0...")

        # Averaging time-based vector for each channel
        f_mean_raw = self.imu_data[["f_X", "f_Y", "f_Z"]].mean().values
        w_mean_raw = self.imu_data[["w_X", "w_Y", "w_Z"]].mean().values
        m_mean_raw = self.mag_data[["m_X", "m_Y", "m_Z"]].mean().values

        # Build Calibration Matrices
        calib_a = kin.calc_inver_calibr_matrix(self.s_a_ppm, self.m_a_deg)
        calib_g = kin.calc_inver_calibr_matrix(self.s_g_ppm, self.m_g_deg)

        # Applying IMU calibration
        f_mean = calib_a @ f_mean_raw
        w_mean = calib_g @ w_mean_raw

        # Applying Hard-Iron Bias and Soft-Iron Matrix (m_hi_vec, m_si_mat): Before computing Attitude Angles
        m_mean = self.inv_m_si @ (m_mean_raw - self.m_hi_vec)

        # Mean from GNSS measures
        raw_pos = self.gnss_data[["Lat", "Lon", "Alt"]].mean().values
        pos_mean = np.array(
            [np.deg2rad(raw_pos[0]), np.deg2rad(raw_pos[1]), raw_pos[2]],
            dtype=np.float64,
        )

        f_std = self.imu_data[["f_X", "f_Y", "f_Z"]].std().values
        w_std = self.imu_data[["w_X", "w_Y", "w_Z"]].std().values

        euler0 = self._compute_initial_att(f_mean, m_mean)
        w_earth_enu = np.array(
            [
                0.0,
                self.omega_e * np.cos(pos_mean[0]),
                self.omega_e * np.sin(pos_mean[0]),
            ]
        )

        # Quaternion Attitude Representation
        q0 = mat.eul2quat(euler0[0], euler0[1], euler0[2])
        C_b2n = mat.quat2dcm(q0)

        g_n = np.array([0.0, 0.0, self.g_local], dtype=np.float64)
        expected_f_b = C_b2n.T @ g_n

        b_a0 = f_mean - expected_f_b
        b_g0 = w_mean - w_earth_enu

        veloc0 = np.array([0.0, 0.0, 0.0])

        x_0 = np.zeros(15, dtype=np.float64)
        x_0[0:3] = pos_mean
        x_0[3:6] = veloc0
        x_0[6:9] = euler0
        x_0[9:12] = b_a0
        x_0[12:15] = b_g0

        P0 = np.zeros((15, 15), dtype=np.float64)
        P0[0:3, 0:3] = np.eye(3) * (self.config["sigma_pos"] ** 2)
        P0[3:6, 3:6] = np.eye(3) * (self.config["sigma_vel"] ** 2)
        P0[6, 6] = self.config["sigma_theta_phi"] ** 2
        P0[7, 7] = self.config["sigma_theta_phi"] ** 2
        P0[8, 8] = self.config["sigma_psi"] ** 2
        P0[9:12, 9:12] = np.eye(3) * (self.config["sigma_ba0"] ** 2)
        P0[12:15, 12:15] = np.eye(3) * (self.config["sigma_bg0"] ** 2)

        return {
            "x_0": x_0,
            "P_0": P0,
            "noise_floor_accel": f_std,
            "noise_floor_gyro": w_std,
        }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "raw"

    config_dir = project_root / "config"
    sys.path.append(str(project_root))
    from config.config_parser import ConfigParser

    config_path = config_dir / "config_baseline.yaml"

    if not config_path.exists():
        sys.exit(1)

    parser = ConfigParser(config_path)
    real_config = parser.parse()

    initializer = staticInitializer(data_path=data_dir, config=real_config)
    initial_data = initializer.gen_initial_state()

    print("ESKF BOUNDARY CONDITIONS (x_0, P_0)")
    print(f"State Vector x_0 Size: {initial_data['x_0'].shape}")
    print(f"Covariance P_0 Size  : {initial_data['P_0'].shape}")
    print("x_0 [Attitude in rad]     :", initial_data["x_0"][6:9].round(6))
    print("x_0 [Accel Bias in m/s^2] :", initial_data["x_0"][9:12].round(6))
    print("x_0 [Gyro Bias in rad/s]  :", initial_data["x_0"][12:15].round(6))
    print("Accel Noise [m/s^2]       :", initial_data["noise_floor_accel"].round(6))
    print("Gyro Noise  [rad/s]       :", initial_data["noise_floor_gyro"].round(6))
