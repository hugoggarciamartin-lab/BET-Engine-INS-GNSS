import sys
import numpy as np
import pandas as pd
import gc
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from config.config_parser import ConfigParser


def generate_pad_telemetry():
    config_path = project_root / "config" / "config_baseline.yaml"
    parser = ConfigParser(config_path)
    cfg = parser.parse()

    duration = 60.0
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
    g_local = g_0 * (1 - (2 * h_0) / a)

    np.random.seed(101)

    noise_floor_accel = 0.05
    noise_floor_gyro = 0.01

    b_a0 = np.array([0.015, -0.012, 0.018], dtype=np.float64)
    b_g0 = np.array([0.005, -0.008, 0.004], dtype=np.float64)

    f_nominal = np.array([0.0, 0.0, g_local], dtype=np.float64)
    w_nominal = np.array(
        [0.0, Omega_e * np.cos(phi_0), Omega_e * np.sin(phi_0)], dtype=np.float64
    )

    f_raw = f_nominal + b_a0 + np.random.normal(0, noise_floor_accel, (len(t_imu), 3))
    w_raw = w_nominal + b_g0 + np.random.normal(0, noise_floor_gyro, (len(t_imu), 3))

    sigma_pos_rad = cfg["sigma_pos"] / a
    sigma_vel = cfg["sigma_vel"]

    gnss_phi = phi_0 + np.random.normal(0, sigma_pos_rad, len(t_gnss))
    gnss_lam = lam_0 + np.random.normal(0, sigma_pos_rad, len(t_gnss))
    gnss_h = h_0 + np.random.normal(0, cfg["sigma_pos"], len(t_gnss))

    gnss_vE = np.random.normal(0, sigma_vel, len(t_gnss))
    gnss_vN = np.random.normal(0, sigma_vel, len(t_gnss))
    gnss_vU = np.random.normal(0, sigma_vel, len(t_gnss))

    P_nominal = P0_isa * (1 - (L_isa * h_0) / T0_isa) ** (g_local / (R_air * L_isa))
    sigma_baro = 10.0
    baro_P = P_nominal + np.random.normal(0, sigma_baro, len(t_baro))

    m_nominal = np.array([24.5, 3.2, 38.1], dtype=np.float64)
    sigma_mag = cfg["sigma_mag"][0]

    mag_raw = m_nominal + np.random.normal(0, sigma_mag, (len(t_mag), 3))

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
    ).to_csv(raw_dir / "preign_data_imu.csv", index=False, float_format="%.8f")

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
    ).to_csv(raw_dir / "preign_data_gnss.csv", index=False, float_format="%.8f")

    pd.DataFrame({"t_baro": t_baro, "P_static": baro_P}).to_csv(
        raw_dir / "preign_data_baro.csv", index=False, float_format="%.8f"
    )

    pd.DataFrame(
        {
            "time": t_mag,
            "m_X": mag_raw[:, 0],
            "m_Y": mag_raw[:, 1],
            "m_Z": mag_raw[:, 2],
        }
    ).to_csv(raw_dir / "preign_data_mag.csv", index=False, float_format="%.8f")

    gc.enable()


if __name__ == "__main__":
    generate_pad_telemetry()
