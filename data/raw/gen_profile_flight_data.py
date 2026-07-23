import numpy as np
import pandas as pd
import gc
from pathlib import Path

raw_dir = Path(__file__).resolve().parent


def generate_flight_profile():
    # TEMPORAL ALLOCATION
    duration = 120.0
    f_imu, f_gnss, f_baro, f_mag = 400.0, 10.0, 20.0, 20.0

    gc.disable()

    t_imu = np.arange(0.0, duration, 1.0 / f_imu, dtype=np.float64)
    t_gnss = np.arange(0.1, duration, 1.0 / f_gnss, dtype=np.float64)
    t_baro = np.arange(0.05, duration, 1.0 / f_baro, dtype=np.float64)
    t_mag = np.arange(0.02, duration, 1.0 / f_mag, dtype=np.float64)

    # BASELINE CONSTANTS
    phi_0 = np.deg2rad(39.4811)
    lam_0 = np.deg2rad(-0.3444)
    h_0 = 15.0
    a, e2, g_e, k_el = 6378137.0, 0.00669437999, 9.7803253359, 0.001931852652
    Omega_e = 7.292115e-5
    P0_isa, T0_isa, L_isa, R_air = 101325.0, 288.15, 0.0065, 287.0528

    g_0 = g_e * (1 + k_el * np.sin(phi_0) ** 2) / np.sqrt(1 - e2 * np.sin(phi_0) ** 2)

    np.random.seed(202)

    # KINEMATIC INTEGRATION (1D Vertical Approximation for synthetic targets)
    # This creates the "Truth" state to generate sensor readings.
    def get_kinematics(t_array):
        h = np.zeros_like(t_array)
        v = np.zeros_like(t_array)
        acc = np.zeros_like(t_array)

        for i in range(1, len(t_array)):
            dt = t_array[i] - t_array[i - 1]
            t = t_array[i]

            if t < 5.0:
                acc[i] = 0.0  # Pad
            elif 5.0 <= t < 25.0:
                acc[i] = 45.0  # Motor burn (~4.5G)
            else:
                acc[i] = -9.81 - (0.002 * v[i - 1] ** 2)  # Coasting + Drag

            v[i] = v[i - 1] + acc[i] * dt
            h[i] = h[i - 1] + v[i] * dt
            if h[i] < h_0:
                h[i] = h_0  # Ground limit

        return h, v, acc

    # Extract truth for all grids
    h_truth_imu, v_truth_imu, acc_truth_imu = get_kinematics(t_imu)
    h_truth_gnss, v_truth_gnss, _ = get_kinematics(t_gnss)
    h_truth_baro, v_truth_baro, _ = get_kinematics(t_baro)

    # IMU GENERATION (Injecting Vibration)
    f_raw = np.zeros((len(t_imu), 3), dtype=np.float64)
    w_raw = np.zeros((len(t_imu), 3), dtype=np.float64)

    for i, t in enumerate(t_imu):
        # Specific force: kinematic acceleration + gravity reaction
        f_z = acc_truth_imu[i] + g_0

        # Base noise
        n_accel = np.random.normal(0, 0.05, 3)
        n_gyro = np.random.normal(0, 0.01, 3)

        # Exception: Inject structural vibration during rocket burn
        if 5.0 <= t < 25.0:
            n_accel += np.random.normal(0, 5.0, 3)  # Massive variance inflation
            n_gyro += np.random.normal(0, 0.5, 3)

        f_raw[i] = [n_accel[0], n_accel[1], f_z + n_accel[2]]
        w_raw[i] = [
            n_gyro[0],
            Omega_e * np.cos(phi_0) + n_gyro[1],
            Omega_e * np.sin(phi_0) + n_gyro[2],
        ]

    # GNSS GENERATION
    gnss_phi = phi_0 + np.random.normal(0, 4.0 / a, len(t_gnss))
    gnss_lam = lam_0 + np.random.normal(0, 4.0 / a, len(t_gnss))
    gnss_h = h_truth_gnss + np.random.normal(0, 4.0, len(t_gnss))

    gnss_vE = np.random.normal(0, 0.1, len(t_gnss))
    gnss_vN = np.random.normal(0, 0.1, len(t_gnss))
    gnss_vU = v_truth_gnss + np.random.normal(0, 0.2, len(t_gnss))

    # BAROMETER GENERATION (Injecting Venturi Effect)
    baro_P = np.zeros(len(t_baro), dtype=np.float64)
    for i, t in enumerate(t_baro):
        # Standard ISA pressure
        P_nominal = P0_isa * (1 - (L_isa * h_truth_baro[i]) / T0_isa) ** (
            g_0 / (R_air * L_isa)
        )

        # Exception: Inject Venturi parasitic drop based on dynamic pressure (q = 0.5 * rho * v^2)
        # Assuming simplified rho~1.2 for transonic regime severity scaling
        venturi_drop = 0.0
        if v_truth_baro[i] > 150.0:
            venturi_drop = 0.005 * (v_truth_baro[i] ** 2)  # Suction

        baro_P[i] = P_nominal - venturi_drop + np.random.normal(0, 10.0)

    # MAGNETOMETER GENERATION
    m_nominal = np.array([24.5, 3.2, 38.1], dtype=np.float64)
    mag_raw = m_nominal + np.random.normal(0, 0.5, (len(t_mag), 3))

    # EXPORT
    print("Exporting flight profile datasets...")

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

    print("Flight datasets generated successfully.")
    gc.enable()


if __name__ == "__main__":
    generate_flight_profile()
