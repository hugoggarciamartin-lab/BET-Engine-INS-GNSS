import sys
import gc
import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

project_root = Path(__file__).resolve().parent.parent.parent
source_dir = project_root / "source"
config_dir = project_root / "config"

sys.path.append(str(project_root))
sys.path.append(str(source_dir))
sys.path.append(str(config_dir))

# Imports Arquitecture
from config.config_parser import ConfigParser
from static_initialization import staticInitializer
import kinematics_ins as kin
import eskf_predictor as pred
import eskf_measurement as meas
import geodesy_math as mat


def initialize_eskf_mem(n_epo: int) -> dict:
    """Static memory for deterministc performance = np.zeros()"""

    return {
        "x_nom": np.zeros((n_epo, 16), dtype=np.float64),
        "dx": np.zeros((n_epo, 15), dtype=np.float64),
        "P": np.zeros((n_epo, 15, 15), dtype=np.float64),
        "P_": np.zeros((n_epo, 15, 15), dtype=np.float64),
        "Phi": np.zeros((n_epo, 15, 15), dtype=np.float64),
        "z": np.zeros((n_epo, 10), dtype=np.float64),
        "Sk_diag": np.zeros((n_epo, 10), dtype=np.float64),
        "rk4_k1": np.zeros((10), dtype=np.float64),
        "rk4_k2": np.zeros((10), dtype=np.float64),
        "rk4_k3": np.zeros((10), dtype=np.float64),
        "rk4_k4": np.zeros((10), dtype=np.float64),
        "rk4_temp": np.zeros((10), dtype=np.float64),
    }  # Preassigned zeros arrays for this ESKF Parameters (fixed Buffers)
    # avoids to consume computing time (Garbage Collection pauses)


def inject_error_state(x_nom: np.ndarray, delta_x: np.ndarray) -> None:
    """Feedback Loop - Injects ESKF Error-State into nominal nonlinear state."""
    r_m, r_n = mat.calc_radii(x_nom[0])

    # Position Injection (Geodetics via radii transformation)
    x_nom[0] -= delta_x[0] / (r_m + x_nom[2])
    x_nom[1] -= delta_x[1] / ((r_n + x_nom[2]) * np.cos(x_nom[0]))
    x_nom[2] -= delta_x[2]  # Alt (Up)

    # Velocity Injection
    x_nom[3:6] -= delta_x[3:6]

    # Attitude Injection
    delta_psi = -delta_x[6:9]
    q_err = np.array(
        [1.0, 0.5 * delta_psi[0], 0.5 * delta_psi[1], 0.5 * delta_psi[2]],
        dtype=np.float64,
    )
    q_upd = mat.quat_ham_prod(q_err, x_nom[6:10])
    x_nom[6:10] = mat.norm_quat(q_upd)


def initialize_wmm_interpolators(
    mag_path: Path,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator, RegularGridInterpolator]:
    """Load and Buildsde World Magnetic Model WWM in a Grid Interpolator."""

    if not mag_path.exists():
        raise FileNotFoundError(
            f"Critical Failure: WMM Tabulated Data missing in {mag_path}"
        )

    df_wmm = pd.read_csv(mag_path)

    df_wmm.sort_values(by=["Alt_km", "Lat_deg", "Lon_deg"], inplace=True)
    alts = np.unique(df_wmm["Alt_km"])
    lats = np.unique(df_wmm["Lat_deg"])
    lons = np.unique(df_wmm["Lon_deg"])

    # Verify Grid
    expec_pts = len(alts) * len(lats) * len(lons)
    if len(df_wmm) != expec_pts:
        raise ValueError("WMM Grid has missing nodes. Interpolator will diverge")

    # Corrección de nombres de columnas (_nT) y sintaxis correcta de reshape con paréntesis
    m_E_grid = df_wmm["m_E_nT"].values.reshape((len(alts), len(lats), len(lons)))
    m_N_grid = df_wmm["m_N_nT"].values.reshape((len(alts), len(lats), len(lons)))
    m_U_grid = df_wmm["m_U_nT"].values.reshape((len(alts), len(lats), len(lons)))

    wwm_interp_E = RegularGridInterpolator((alts, lats, lons), m_E_grid)
    wwm_interp_N = RegularGridInterpolator((alts, lats, lons), m_N_grid)
    wwm_interp_U = RegularGridInterpolator((alts, lats, lons), m_U_grid)

    return wwm_interp_E, wwm_interp_N, wwm_interp_U


def main_eskf():
    # Begining of computing time
    t_start = time.perf_counter()
    k_last = 0  # Last processed epoch

    project_root = Path(__file__).resolve().parent.parent.parent

    # System Configuration
    config_path = project_root / "config" / "config_baseline.yaml"
    parser = ConfigParser(config_path)
    params = parser.parse()

    dt = 1.0 / 400.0  # Time-step fixed for 400 Hz IMU frequency

    # Master Flight Telemetry Ingestion
    telem_path = project_root / "data" / "aligned_data" / "master_flight_data.csv"
    if not telem_path.exists():
        raise FileNotFoundError(
            f"Critical Error: Master telemetry missing at {telem_path}"
        )
    df_master = pd.read_csv(telem_path)
    n_epo = len(df_master)

    # Magnetic Model Setup
    mag_path = project_root / "data" / "raw" / "wmm_mag_tab_data.csv"
    wmm_interp_E, wmm_interp_N, wmm_interp_U = initialize_wmm_interpolators(mag_path)

    # Memory Allocation
    mem = initialize_eskf_mem(n_epo)

    # Initial Static Data - Boundary Conditions (x_0, P_0)
    data_dir = project_root / "data" / "raw"

    # Only Use Static Data for: Initial Attitude, Initial IMU Bias, P_0 CoVar Matrix and IMU Noise Floor
    initializer = staticInitializer(data_path=data_dir, config=params)
    initial_data = initializer.gen_initial_state()

    # Pre-Extract variables from DataFrame
    f_X, f_Y, f_Z = (
        df_master["f_X"].values,
        df_master["f_Y"].values,
        df_master["f_Z"].values,
    )
    w_X, w_Y, w_Z = (
        df_master["w_X"].values,
        df_master["w_Y"].values,
        df_master["w_Z"].values,
    )
    gnss_lat, gnss_lon, gnss_alt = (
        df_master["gnss_Lat"].values,
        df_master["gnss_Lon"].values,
        df_master["gnss_Alt"].values,
    )
    gnss_vE, gnss_vN, gnss_vU = (
        df_master["gnss_v_E"].values,
        df_master["gnss_v_N"].values,
        df_master["gnss_v_U"].values,
    )
    mag_X, mag_Y, mag_Z = (
        df_master["mag_m_X"].values,
        df_master["mag_m_Y"].values,
        df_master["mag_m_Z"].values,
    )
    # Barometric Static Pressure
    baro_alt = df_master["baro_P_static"].values
    baro_p = df_master["baro_P_static"].values
    # Barometric Altitude by ISA atmosphere
    baro_alt = kin.calc_isa_altitude(
        baro_p, params["p0_isa"], params["t0_isa"], params["l_isa"], params["g_e"]
    )

    # Static Initialization to memory ('mem'): [lat, lon, alt, vE, vN, vU, q0, q1, q2, q3]
    mem["x_nom"][0, 0] = np.deg2rad(gnss_lat[0])
    mem["x_nom"][0, 1] = np.deg2rad(gnss_lon[0])
    mem["x_nom"][0, 2] = gnss_alt[0]

    mem["x_nom"][0, 3:6] = np.array(
        [gnss_vE[0], gnss_vN[0], gnss_vU[0]], dtype=np.float64
    )

    roll, pitch, yaw = initial_data["x_0"][6:9]
    mem["x_nom"][0, 6:10] = mat.eul2quat(roll, pitch, yaw)

    # Initial-state P_0 covariance matrix
    mem["P"][0] = initial_data["P_0"]

    bias_a = initial_data["x_0"][9:12].copy()
    bias_g = initial_data["x_0"][12:15].copy()

    # In case of 'bias_a' failure
    dcm_b2n_0 = mat.quat2dcm(mem["x_nom"][0, 6:10])
    g_0 = mat.somig_gravity(mem["x_nom"][0, 0], mem["x_nom"][0, 2])
    g_n = np.array([0.0, 0.0, g_0], dtype=np.float64)
    expected_f_b = dcm_b2n_0.T @ g_n
    # Fix it in that case
    if np.linalg.norm(bias_a) > 5.0:
        bias_a -= expected_f_b

    # Vibration Control Initailization
    noise_floor_a = initial_data["noise_floor_accel"]
    m_vib = params["m_vib_window"]
    k_vib = params["k_vib_mult"]
    accel_window = np.zeros((m_vib, 3), dtype=np.float64)

    # Sensor Calibration Matrix (Just Once)
    calib_a = kin.calc_inver_calibr_matrix(params["s_a_ppm"], params["m_a_deg"])
    calib_g = kin.calc_inver_calibr_matrix(params["s_g_ppm"], params["m_g_deg"])

    gc.disable()
    try:
        # ----- MAIN ESKF LOOP -----
        for k in range(1, n_epo):
            k_last = k
            # Current State
            x_prev = mem["x_nom"][k - 1].copy()  # Indexing begins in [0]
            P_prev = mem["P"][k - 1].copy()

            # Raw IMU Measurement
            f_raw = np.array([f_X[k - 1], f_Y[k - 1], f_Z[k - 1]], dtype=np.float64)
            w_raw = np.array([w_X[k - 1], w_Y[k - 1], w_Z[k - 1]], dtype=np.float64)

            # Apply Sensor Calibration:
            f_calib, w_calib = kin.apply_sensor_calibration(
                f_raw, w_raw, calib_a, calib_g
            )

            # Select the window for vibration checking
            accel_window[k % m_vib] = f_calib

            is_vib = False
            if k >= m_vib:
                curr_accel_std = np.std(accel_window, axis=0)
                if np.any(curr_accel_std > (noise_floor_a * k_vib)):
                    is_vib = True

            # Substract dynmamic bias
            f_b = f_calib - bias_a
            w_b = w_calib - bias_g

            # ----- ESKF PREDICTION BLOCK -----
            # Nominal State Propagation by RK4 Integration
            x_k = x_prev.copy()
            kin.rk4_integration_step(
                x_k[0:10],
                f_b,
                w_b,
                dt,
                mem["rk4_k1"],
                mem["rk4_k2"],
                mem["rk4_k3"],
                mem["rk4_k4"],
                mem["rk4_temp"],
            )  # Preassigned zeros arrays from k1 to k4 and temp_state (fixed Buffers)
            # avoids to consume computing time (Garbage Collection pauses)
            norm_q = np.linalg.norm(x_k[6:10])
            if not np.isfinite(norm_q) or norm_q < 1e-6:
                raise ValueError(
                    "Critical Failure: Quaternion norm diverged to NaN or zero."
                )
            x_k[6:10] /= norm_q

            mem["x_nom"][k] = x_k

            # Extract Geodetisc and Attitude Parameters for ESKF Matrices
            lat, lon, alt = x_k[0], x_k[1], x_k[2]
            v_n_vec = x_k[3:6]
            dcm_b2n = mat.quat2dcm(x_k[6:10])
            r_m, r_n = mat.calc_radii(lat)
            g_0 = mat.somig_gravity(lat, alt)

            # Local Mach Number
            mach_num = kin.calc_mach_number(
                v_n_vec,
                alt,
                params["p0_isa"],
                params["t0_isa"],
                params["l_isa"],
                params["g_e"],
            )

            w_ie_n = kin.earth_rate_enu(lat)
            w_en_n = kin.rate_transprt_enu(v_n_vec[0], v_n_vec[1], lat, alt, r_m, r_n)
            f_n_vec = dcm_b2n @ f_b

            # Define Q_k Penalty produced by Vibration Noise
            accel_var_penalty = np.zeros(3, dtype=np.float64)
            gyro_var_penalty = np.zeros(3, dtype=np.float64)

            if is_vib:
                # Penalty based in k_vib factor applied to static noises
                accel_var_penalty = (
                    params["k_vib_mult"] * params["sigma_accel"]
                ) ** 2 - params["sigma_accel"] ** 2
                gyro_var_penalty = (
                    params["k_vib_mult"] * params["sigma_gyro"]
                ) ** 2 - params["sigma_gyro"] ** 2

            # ----- Covariance Propagation ------
            # Dynamic System Matrix F
            F_k = pred.calc_f_matrix_continuous(
                r_m,
                r_n,
                lat,
                alt,
                v_n_vec,
                f_n_vec,
                w_ie_n,
                w_en_n,
                dcm_b2n,
                params["accel_corr_time"],
                params["gyro_corr_time"],
                g_0,
                params["a"],
            )
            Phi_k = pred.discretize_phi_matrix(F_k, dt)
            mem["Phi"][k] = Phi_k

            # Process Noise Covariance Matrix
            Q_k = pred.calc_q_matrix_discrete(
                dcm_b2n,
                dt,
                params["vrw"],
                params["arw"],
                params["accel_bias_inst"],
                params["gyro_bias_inst"],
                params["accel_corr_time"],
                params["gyro_corr_time"],
                is_vib,
                accel_var_penalty,
                gyro_var_penalty,
            )

            # Initialize P-minus using Phi, P_0 and Q_k
            P_minus = pred.predict_error_covariance(Phi_k, P_prev, Q_k)
            mem["P_"] = P_minus

            # ----- MEASUREMENT BLOCK -----
            # Set measures matching
            has_gnss = k % 40 == 0
            has_baro = k % 20 == 0
            has_mag = k % 8 == 0

            if has_gnss or has_baro or has_mag:
                # Set de limits for geodetics coord values
                alt_km = min(95.0, max(0.0, float(alt) / 1000.0))
                lat_deg = min(90.0, max(-90.0, float(np.rad2deg(lat))))
                lon_deg = float(np.rad2deg(lon)) % 360.0
                if lon_deg > 180.0:
                    lon_deg -= 360.0

                pts = np.array([[alt_km, lat_deg, lon_deg]])
                # Local Magnetic Vector from WMM Grid
                # Scale Magnetometer units to micro-Teslas
                m_n = (
                    np.array(
                        [
                            wmm_interp_E(pts)[0],
                            wmm_interp_N(pts)[0],
                            wmm_interp_U(pts)[0],
                        ],
                        dtype=np.float64,
                    )
                    * 1e-3
                )

                # Build Measure Observation Matrix
                H_k = meas.calc_h_matrix(dcm_b2n, params["r_arm_b"], m_n)

                # Measure Noise Matrix
                R_k = meas.calc_r_matrix(
                    params["sigma_gnss_p"],
                    params["sigma_gnss_v"],
                    params["sigma_baro"],
                    params["sigma_mag"],
                    params["hdop_nominal"],
                    params["vdop_nominal"],
                    mach_num,
                    is_vib,
                )

                # R_k Inflation: Mathematically decouple unmeasured channels during blind sensor epochs
                # If there is a blind sensor, we ommit that vallue by inflation de associated Measure Noise
                if not has_gnss:
                    R_k[0:6, 0:6] += np.eye(6, dtype=np.float64) * 1e9
                if not has_baro:
                    R_k[6, 6] += 1e9
                if not has_mag:
                    R_k[7:10, 7:10] += np.eye(3, dtype=np.float64) * 1e9

                pos_gnss = np.array(
                    [np.deg2rad(gnss_lat[k]), np.deg2rad(gnss_lon[k]), gnss_alt[k]],
                    dtype=np.float64,
                )
                vel_gnss = np.array(
                    [gnss_vE[k], gnss_vN[k], gnss_vU[k]], dtype=np.float64
                )
                # Hard Ironing - Calibration
                b_mag_hi = params["m_hi_vec"]
                mag_meas = (
                    np.array([mag_X[k], mag_Y[k], mag_Z[k]], dtype=np.float64)
                    - b_mag_hi
                )
                # Kinematic lever arm rotation rate
                w_bn_b = w_b - (dcm_b2n.T @ (w_ie_n + w_en_n))

                z_k = meas.calc_inno_vector(
                    np.array([lat, lon, alt]),
                    v_n_vec,
                    alt,
                    m_n,
                    pos_gnss,
                    vel_gnss,
                    baro_alt[k],
                    mag_meas,
                    r_m,
                    r_n,
                    dcm_b2n,
                    params["r_arm_b"],
                    w_bn_b,
                )

                mem["z"][k] = z_k

                # Store Innovation Covariance Matrix S_k (Theoretical) - Used in Validation Phase
                S_k = meas.calc_s_k_covariance(P_minus, H_k, R_k)
                mem["Sk_diag"][k] = np.diag(S_k)

                # Kalman Gain K_k
                delta_x, P_plus, update_valid = meas.exe_kalman_update(
                    P_minus, H_k, R_k, z_k, params["k_innov_sigma"]
                )

                # In case of feedback error into nominal state and biases
                if update_valid:
                    inject_error_state(mem["x_nom"][k], delta_x)
                    bias_a -= delta_x[9:12]
                    bias_g -= delta_x[12:15]

                # Save Current Covariance Matrix
                mem["P"][k] = P_plus

            else:
                # Blind phase: Pure inertial propagation without sparse sensor contamination
                mem["P"][k] = P_minus

            # Reset error state vector
            mem["x_nom"][0, 10:13] = bias_a
            mem["x_nom"][0, 13:16] = bias_g
            mem["dx"][k] = np.zeros(15, dtype=np.float64)
        pass
    finally:
        gc.enable()
        gc.collect()

    # Final Export to Phase 3 (RTS Smoother)
    print("ESKF Forward Pass Completed. Exporting results...")
    out_path = project_root / "data" / "results" / "eskf_output_state.npz"
    np.savez_compressed(
        out_path,
        x_nom=mem["x_nom"],
        P_minus=mem["P_"],
        P=mem["P"],
        Phi=mem["Phi"],
        dx=mem["dx"],
        z=mem["z"],
        Sk_diag=mem["Sk_diag"],
    )
    print(f"Data saved at {out_path}")

    # Total Computing Time
    t_elapsed = time.perf_counter() - t_start

    # Cálculo de métricas de rendimiento operacional
    epochs_processed = k_last + 1
    freq_exec = epochs_processed / t_elapsed if t_elapsed > 0 else 0.0

    print("COMPUTING TIME PERFORMANCE REPORT")
    print(f"Total Computing Time    : {t_elapsed:.4f} segundos")
    print(f"Calculated Epochs       : {epochs_processed} / {n_epo}")
    print(f"Effective Frequency     : {freq_exec:.3f} Hz")


if __name__ == "__main__":
    try:
        main_eskf()
    except Exception as e:
        print(f"Integration Failure: {e}")
    sys.exit(1)
