import numpy as np
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Dict, Tuple


class staticInitializer:
    """Operate pre-ignition telemetry data to calculate operational noise-floors,
    static turn-on biases and the initial Tait-Bryan attitude angles"""

    def __init__(self, data_path: Path, config: Dict):
        self.data_path = data_path
        self.config = config

        self.imu_data = pd.DataFrame()
        self.mag_data = pd.DataFrame()
        self.gnss_data = pd.DataFrame()

        # Constant Parameters to use in bias correction
        self.omega_e = self.config["omega_e"]
        self.g_local = self.config["g_local"]

    def load_telem(self) -> None:
        """Loads the raw pre-ignition asynchronous datasets"""
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
        by using accelerometers niveling and magnetic aligning"""

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
        """Constructs de 15x1 initial state vector x_0 and 15x15 covariance matrix P_0"""
        self.load_telem()

        print("Constructing initial state vector x_0 and covariance matrix P_0...")

        # Averaging time-based vector for each channel
        f_mean = self.imu_data[["f_X", "f_Y", "f_Z"]].mean().values
        w_mean = self.imu_data[["w_X", "w_Y", "w_Z"]].mean().values
        m_mean = self.mag_data[["m_X", "m_Y", "m_Z"]].mean().values
        pos_mean = self.gnss_data[["Lat", "Lon", "Alt"]].mean().values

        # Noise Floor Extraction (Standard Deviation)
        f_std = self.imu_data[["f_X", "f_Y", "f_Z"]].std().values
        w_std = self.imu_data[["w_X", "w_Y", "w_Z"]].std().values

        # Attitude Extraction
        euler0 = self._compute_initial_att(f_mean, m_mean)
        w_earth_enu = np.array(
            [
                0.0,
                self.omega_e * np.cos(self.config["phi_0"]),
                self.omega_e * np.sin(self.config["phi_0"]),
            ]
        )

        b_g0 = w_mean - w_earth_enu
        b_a0 = np.array([f_mean[0], f_mean[1], f_mean[2] - self.g_local])
        veloc0 = np.array([0.0, 0.0, 0.0])

        # Initial State Vector
        x_0 = np.zeros(15, dtype=np.float64)
        x_0[:3] = pos_mean
        x_0[3:6] = np.array(veloc0)
        x_0[6:9] = euler0
        x_0[9:12] = b_a0
        x_0[12:15] = b_g0

        # Covariance Matrix P_0
        P_0 = np.zeros((15, 15), dtype=np.float64)

        # Position
        P_0[0:3, 0:3] = np.eye(3) * (self.config["sigma_pos"] ** 2)
        # Velocity
        P_0[3:6, 3:6] = np.eye(3) * (self.config["sigma_vel"] ** 2)
        #
        P_0[6, 6] = self.config["sigma_theta_phi"] ** 2
        P_0[7, 7] = self.config["sigma_theta_phi"] ** 2
        P_0[8, 8] = self.config["sigma_psi"] ** 2
        # Biases
        P_0[9:12, 9:12] = np.eye(3) * (self.config["sigma_ba0"] ** 2)
        P_0[12:15, 12:15] = np.eye(3) * (self.config["sigma_bg0"] ** 2)

        return {
            "x_0": x_0,
            "P_0": P_0,
            "noise_floor_accel": f_std,
            "noise_floor_gyro": w_std,
        }


# Test
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "raw"

    # Dictionary Test that simulates the future output of the YAML
    test_config = {
        "omega_e": 7.292115e-5,
        "g_local": 9.801,
        "phi_0": np.deg2rad(39.4811),
        "sigma_pos": 4.0,  # m
        "sigma_vel": 0.1,  # m/s
        "sigma_theta_phi": np.deg2rad(0.05),  # converted to rad
        "sigma_psi": np.deg2rad(5.0),  # converted to rad
        "sigma_ba0": 0.02,  # m/s^2
        "sigma_bg0": 0.01,  # rad/s
    }
    initializer = staticInitializer(data_path=data_dir, config=test_config)
    initial_data = initializer.gen_initial_state()

    print("ESKF BOUNDARY CONDITIONS (x_0, P_0)")
    print(f"State Vector x_0 Size: {initial_data['x_0'].shape}")
    print(f"Covariance P_0 Size  : {initial_data['P_0'].shape}")
    print("x_0 [Attitude in rad]     :", initial_data["x_0"][6:9].round(6))
    print("x_0 [Accel Bias in m/s^2] :", initial_data["x_0"][9:12].round(6))
    print("x_0 [Gyro Bias in rad/s]  :", initial_data["x_0"][12:15].round(6))
    print("Accel Noise [m/s^2]       :", initial_data["noise_floor_accel"].round(6))
    print("Gyro Noise  [rad/s]       :", initial_data["noise_floor_gyro"].round(6))
