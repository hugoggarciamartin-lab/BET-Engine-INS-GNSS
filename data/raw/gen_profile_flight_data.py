import sys
import numpy as np
import pandas as pd
import gc
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from config.config_parser import ConfigParser


def generate_flight_profile():
    config_path = project_root / "config" / "config_baseline.yaml"
    parser = ConfigParser(config_path)
    cfg = parser.parse()

    duration = 120.0
    f_imu, f_gnss, f_baro, f_mag = 400.0, 10.0, 20.0, 20.0

    gc.disable()

    t_imu = np.arange(0.0, duration, 1.0 / f_imu, dtype=np.float64)
    t_gnss = np.arange(0.1, duration, 1.0 / f_gnss, dtype=np.float64)
    t_baro = np.arange(0.05, duration, 1.0 / f_baro, dtype=np.float64)
    t_mag = np.arange(0.02, duration, 1.0 / f_mag, dtype=np.float64)

    phi_0 = np.deg2rad(39.4811)
    lam_0 = np.deg2rad(-0.3444)
    h_0 = 15.0

    a = cfg["a"]
    e2 = cfg["e2"]
    g_e = cfg["g_e"]
    k_el = cfg["k"]
    Omega_e = cfg["omega_e"]
    P0_isa = cfg["p0_isa"]
    T0_isa = cfg["t0_isa"]
    L_isa = cfg["l_isa"]
    R_air = 287.0528

    g_0 = g_e * (1 + k_el * np.sin(phi_0) ** 2) / np.sqrt(1 - e2 * np.sin(phi_0) ** 2)

    np.random.seed(202)

    def get_kinematics(t_array):
        h = np.zeros_like(t_array)
        v = np.zeros_like(t_array)
        acc = np.zeros_like(t_array)

        for i in range(1, len(t_array)):
            dt = t_array[i] - t_array[i - 1]
            t = t_array[i]

            if t < 5.0:
                acc[i] = 0.0
            elif 5.0 <= t < cfg["time_eng_off"]:
                acc[i] = 45.0
            else:
                acc[i] = -9.81 - (0.002 * v[i - 1] * np.abs(v[i - 1]))

            v[i] = v[i - 1] + acc[i] * dt
            h[i] = h[i - 1] + v[i] * dt
            if h[i] < h_0:
                h[i] = h_0

        return h, v, acc

    h_truth_imu, v_truth_imu, acc_truth_imu = get_kinematics(t_imu)
    h_truth_gnss, v_truth_gnss, _ = get_kinematics(t_gnss)
    h_truth_baro, v_truth_baro, _ = get_kinematics(t_baro)

    f_raw = np.zeros((len(t_imu), 3), dtype=np.float64)
    w_raw = np.zeros((len(t_imu), 3), dtype=np.float64)

    noise_floor_accel = 0.05
    noise_floor_gyro = 0.01

    for i, t in enumerate(t_imu):
        f_z = acc_truth_imu[i] + g_0

        n_accel = np.array(
            [
                np.random.normal(0, noise_floor_accel),
                np.random.normal(0, noise_floor_accel),
                np.random.normal(0, noise_floor_accel),
            ]
        )
        n_gyro = np.array(
            [
                np.random.normal(0, noise_floor_gyro),
                np.random.normal(0, noise_floor_gyro),
                np.random.normal(0, noise_floor_gyro),
            ]
        )

        if 5.0 <= t < cfg["time_eng_off"]:
            n_accel += np.random.normal(0, 5.0, 3)
            n_gyro += np.random.normal(0, 0.5, 3)

        f_raw[i] = [n_accel[0], n_accel[1], f_z + n_accel[2]]
        w_raw[i] = [
            n_gyro[0],
            Omega_e * np.cos(phi_0) + n_gyro[1],
            Omega_e * np.sin(phi_0) + n_gyro[2],
        ]

    sigma_p = cfg["sigma_gnss_p"]
    sigma_v = cfg["sigma_gnss_v"]

    gnss_phi = phi_0 + np.random.normal(0, sigma_p[0] / a, len(t_gnss))
    gnss_lam = lam_0 + np.random.normal(0, sigma_p[1] / a, len(t_gnss))
    gnss_h = h_truth_gnss + np.random.normal(0, sigma_p[2], len(t_gnss))

    gnss_vE = np.random.normal(0, sigma_v[0], len(t_gnss))
    gnss_vN = np.random.normal(0, sigma_v[1], len(t_gnss))
    gnss_vU = v_truth_gnss + np.random.normal(0, sigma_v[2], len(t_gnss))

    baro_P = np.zeros(len(t_baro), dtype=np.float64)
    for i, t in enumerate(t_baro):
        P_nominal = P0_isa * (1 - (L_isa * h_truth_baro[i]) / T0_isa) ** (
            g_0 / (R_air * L_isa)
        )

        venturi_drop = 0.0
        if v_truth_baro[i] > 150.0:
            venturi_drop = 0.005 * (v_truth_baro[i] ** 2)

        baro_P[i] = P_nominal - venturi_drop + np.random.normal(0, 10.0)

    m_enu = np.array([0.5, 24.5, 38.1], dtype=np.float64)
    b_hard_iron = cfg["m_hi_vec"]
    M_soft_iron = cfg["m_si_mat"]

    mag_raw = np.zeros((len(t_mag), 3), dtype=np.float64)
    roll_rate = 2.0 * np.pi * 2.0
    sigma_m = cfg["sigma_mag"]

    for i, t in enumerate(t_mag):
        roll = roll_rate * t

        C_n2b = np.array(
            [
                [np.cos(roll), np.sin(roll), 0.0],
                [-np.sin(roll), np.cos(roll), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        m_body = C_n2b @ m_enu
        m_distorted = M_soft_iron @ m_body + b_hard_iron

        mag_raw[i] = m_distorted + np.array(
            [
                np.random.normal(0, sigma_m[0]),
                np.random.normal(0, sigma_m[1]),
                np.random.normal(0, sigma_m[2]),
            ]
        )

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "time": t_imu,
            "f_X": f_raw[:, 0],
            "f_Y": f_raw[:, 1],
            "f_Z": f_raw[:, 2],
            "w_X": w_raw[:, 0],
            "w_Y": w_raw[:, 1],
            "w_Z": w_raw[:, 2],
        }
    ).to_csv(raw_dir / "flight_data_imu.csv", index=False, float_format="%.8f")

    pd.DataFrame(
        {
            "time": t_gnss,
            "Lat": gnss_phi,
            "Lon": gnss_lam,
            "Alt": gnss_h,
            "v_E": gnss_vE,
            "v_N": gnss_vN,
            "v_U": gnss_vU,
        }
    ).to_csv(raw_dir / "flight_data_gnss.csv", index=False, float_format="%.8f")

    pd.DataFrame({"time": t_baro, "P_static": baro_P}).to_csv(
        raw_dir / "flight_data_baro.csv", index=False, float_format="%.8f"
    )

    pd.DataFrame(
        {
            "time": t_mag,
            "m_X": mag_raw[:, 0],
            "m_Y": mag_raw[:, 1],
            "m_Z": mag_raw[:, 2],
        }
    ).to_csv(raw_dir / "flight_data_mag.csv", index=False, float_format="%.8f")

    gc.enable()


if __name__ == "__main__":
    generate_flight_profile()
