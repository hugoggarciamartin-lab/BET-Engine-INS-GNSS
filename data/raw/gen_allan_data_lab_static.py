"""Extended static laboratory telemetry simulator (Allan Variance).
Generates long-duration synthetic IMU data governed by white noise and Coupled  Gauss-Markov processes"""

import sys
import numpy as np
import scipy.signal as signal
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from config.config_parser import ConfigParser


def generate_allan_variance_data():
    config_path = project_root / "config" / "config_baseline.yaml"
    parser = ConfigParser(config_path)
    cfg = parser.parse()

    f_s = 400.0
    dt = 1.0 / f_s
    duration_hours = 2.5
    N = int(duration_hours * 3600 * f_s)

    t_master = np.arange(0, N, dtype=np.float64) * dt
    f_raw = np.zeros((N, 3), dtype=np.float64)
    w_raw = np.zeros((N, 3), dtype=np.float64)

    phi = np.deg2rad(39.4811)
    a = cfg["a"]
    e2 = cfg["e2"]
    g_e = cfg["g_e"]
    k_el = cfg["k"]
    Omega_e = cfg["omega_e"]

    g_0 = g_e * (1 + k_el * np.sin(phi) ** 2) / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    g_local = g_0 * (1 - (2 * 15.0) / a)

    f_nominal = np.array([0.0, 0.0, g_local], dtype=np.float64)
    w_nominal = np.array(
        [0.0, Omega_e * np.cos(phi), Omega_e * np.sin(phi)], dtype=np.float64
    )

    arw_rad_s = (0.05 * np.pi / 180.0) / 60.0
    vrw_m_s = 0.02 / 60.0

    bg_instability = (5.0 * np.pi / 180.0) / 3600.0
    ba_instability = 0.1 * 9.80665 * 1e-3

    tau_g = 3600.0
    tau_a = 3600.0

    sigma_w_gyro = arw_rad_s / np.sqrt(dt)
    sigma_w_accel = vrw_m_s / np.sqrt(dt)

    phi_g = np.exp(-dt / tau_g)
    phi_a = np.exp(-dt / tau_a)

    sigma_gm_gyro = bg_instability * np.sqrt(1 - phi_g**2)
    sigma_gm_accel = ba_instability * np.sqrt(1 - phi_a**2)

    np.random.seed(42)

    for axis in range(3):
        wn_accel = np.random.normal(0, sigma_w_accel, N)
        wn_gyro = np.random.normal(0, sigma_w_gyro, N)

        gm_noise_accel = np.random.normal(0, sigma_gm_accel, N)
        gm_noise_gyro = np.random.normal(0, sigma_gm_gyro, N)

        bias_drift_accel = signal.lfilter([1.0], [1.0, -phi_a], gm_noise_accel)
        bias_drift_gyro = signal.lfilter([1.0], [1.0, -phi_g], gm_noise_gyro)

        f_raw[:, axis] = f_nominal[axis] + wn_accel + bias_drift_accel
        w_raw[:, axis] = w_nominal[axis] + wn_gyro + bias_drift_gyro

    df = pd.DataFrame(
        {
            "time": t_master,
            "f_X": f_raw[:, 0],
            "f_Y": f_raw[:, 1],
            "f_Z": f_raw[:, 2],
            "w_X": w_raw[:, 0],
            "w_Y": w_raw[:, 1],
            "w_Z": w_raw[:, 2],
        }
    )

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_filename = raw_dir / "lab_allan_variance_data.csv"
    df.to_csv(output_filename, index=False, float_format="%.8f")


if __name__ == "__main__":
    generate_allan_variance_data()
