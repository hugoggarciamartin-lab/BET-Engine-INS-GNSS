import numpy as np
import scipy.signal as signal
import pandas as pd
import gc
from pathlib import Path

raw_dir = Path(__file__).resolve().parent


def generate_allan_variance_data():
    # 1. TEMPORAL MASTER GRID ALLOCATION
    f_s = 400.0  # Hz
    dt = 1.0 / f_s
    duration_hours = 2.5
    N = int(duration_hours * 3600 * f_s)

    print(f"Allocating memory for {N} epochs...")
    gc.disable()  # Ensure deterministic memory block assignment

    # Initializing time, specficic force and angular rate vectors
    t_master = np.arange(0, N, dtype=np.float64) * dt
    f_raw = np.zeros((N, 3), dtype=np.float64)
    w_raw = np.zeros((N, 3), dtype=np.float64)

    # GEODESY
    # Coordinates - Latitude: 39.4811 deg, Altitude: 15.0 m
    phi = np.deg2rad(39.4811)
    a = 6378137.0
    e2 = 0.00669437999
    g_e = 9.7803253359
    k_el = 0.001931852652
    Omega_e = 7.292115e-5

    # Somigliana Gravity with free-air anomaly
    g_0 = g_e * (1 + k_el * np.sin(phi) ** 2) / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    g_local = g_0 * (1 - (2 * 15.0) / a)

    # Assuming Body Frame aligned with ENU for static test
    # Accelerometer senses reaction to gravity (Z points Up, so it reads +g_local)
    f_nominal = np.array([0.0, 0.0, g_local], dtype=np.float64)

    # Gyroscope senses Earth's rotation projected onto ENU
    w_nominal = np.array(
        [0.0, Omega_e * np.cos(phi), Omega_e * np.sin(phi)], dtype=np.float64
    )

    # STOCHASTIC TARGETS (Converted to standard SI units)
    # ARW: 0.05 deg/sqrt(h) -> rad/s * 1/sqrt(Hz) - Noise Coefficient
    arw_rad_s = (0.05 * np.pi / 180.0) / 60.0
    # VRW: 0.02 (m/s)/sqrt(h) -> m/s^2 * 1/sqrt(Hz)
    vrw_m_s = 0.02 / 60.0  # Noise Coefficient

    # Bias Instability: 5.0 deg/h and 0.1 mg
    bg_instability = (5.0 * np.pi / 180.0) / 3600.0  # rad/s
    ba_instability = 0.1 * 9.80665 * 1e-3  # m/s^2

    # Correlation times
    tau_g = 3600.0
    tau_a = 3600.0

    # DISCRETE NOISE GENERATION
    # White Noise (Thermo-mechanical)
    sigma_w_gyro = arw_rad_s / np.sqrt(dt)
    sigma_w_accel = vrw_m_s / np.sqrt(dt)

    # Gauss-Markov Process (Flicker Noise / Thermal Drift)
    # x[k] = phi_gm * x[k-1] + w_gm[k]
    phi_g = np.exp(-dt / tau_g)
    phi_a = np.exp(-dt / tau_a)

    # Driving noise variance for the Gauss-Markov process
    sigma_gm_gyro = bg_instability * np.sqrt(1 - phi_g**2)
    sigma_gm_accel = ba_instability * np.sqrt(1 - phi_a**2)

    # Generate isolated random sequences
    np.random.seed(42)  # Static seed for DO-330 reproducibility

    print("Injecting stochastic noise...")
    for axis in range(3):
        # Base White Noise
        wn_accel = np.random.normal(0, sigma_w_accel, N)
        wn_gyro = np.random.normal(0, sigma_w_gyro, N)

        # Gauss-Markov Driving Noise
        gm_noise_accel = np.random.normal(0, sigma_gm_accel, N)
        gm_noise_gyro = np.random.normal(0, sigma_gm_gyro, N)

        # Vectorized AR(1) Filter for Gauss-Markov drift (eliminates dynamic for-loop)
        bias_drift_accel = signal.lfilter([1.0], [1.0, -phi_a], gm_noise_accel)
        bias_drift_gyro = signal.lfilter([1.0], [1.0, -phi_g], gm_noise_gyro)

        # Combine Nominal + White Noise + Drift
        f_raw[:, axis] = f_nominal[axis] + wn_accel + bias_drift_accel
        w_raw[:, axis] = w_nominal[axis] + wn_gyro + bias_drift_gyro

    # DATA EXPORT
    print("Formatting and exporting structural array...")
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

    output_filename = raw_dir / "lab_allan_variance_data.csv"
    df.to_csv(output_filename, index=False, float_format="%.8f")
    print(f"Dataset successfully written to {output_filename.name}")

    gc.enable()


if __name__ == "__main__":
    generate_allan_variance_data()
