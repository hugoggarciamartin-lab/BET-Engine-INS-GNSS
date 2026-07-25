import numpy as np

# Universal WGS84 Constant
A_WGS = 6378137.0


def calc_f_matrix_continuous(
    r_m: float,
    r_n: float,
    lat_rad: float,
    alt_m: float,
    v_n_vec: np.ndarray,
    f_n_vec: np.ndarray,
    w_ie_n: np.ndarray,
    w_en_n: np.ndarray,
    c_b2n: np.ndarray,
    tau_a: np.ndarray,
    tau_g: np.ndarray,
    g_0: float,
) -> np.ndarray:
    """
    Builds the 15x15 continuous-time system dynamics Jacobian matrix F.
    State Vector: [delta_r (3), delta_v (3), delta_psi (3), delta_ba (3), delta_bg (3)]
    Accepts anisotropic 3-axis correlation time vectors (tau_a, tau_g) from Allan Variance calibration.
    """
    v_e, v_n, v_u = v_n_vec[0], v_n_vec[1], v_n_vec[2]

    # 1. Position Error Dynamics (F_rr and F_rv)
    f_rr = np.zeros((3, 3), dtype=np.float64)
    f_rv = np.array(
        [
            [0.0, 1.0 / (r_m + alt_m), 0.0],
            [1.0 / ((r_n + alt_m) * np.cos(lat_rad)), 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    # 2. Velocity Error Dynamics (F_vr, F_vv, F_vpsi)
    f_vr = np.array(
        [
            [0.0, 0.0, -v_n / ((r_m + alt_m) ** 2)],
            [
                (v_e * np.sin(lat_rad)) / ((r_n + alt_m) * (np.cos(lat_rad) ** 2)),
                0.0,
                -v_e / ((r_n + alt_m) ** 2),
            ],
            [0.0, 0.0, (2.0 * g_0) / A_WGS],
        ],
        dtype=np.float64,
    )

    w_in_n_vel = 2.0 * w_ie_n + w_en_n
    f_vv = -np.array(
        [
            [0.0, -w_in_n_vel[2], w_in_n_vel[1]],
            [w_in_n_vel[2], 0.0, -w_in_n_vel[0]],
            [-w_in_n_vel[1], w_in_n_vel[0], 0.0],
        ],
        dtype=np.float64,
    )

    f_vpsi = -np.array(
        [
            [0.0, -f_n_vec[2], f_n_vec[1]],
            [f_n_vec[2], 0.0, -f_n_vec[0]],
            [-f_n_vec[1], f_n_vec[0], 0.0],
        ],
        dtype=np.float64,
    )

    # 3. Attitude Error Dynamics (F_psipsi)
    w_in_n_att = w_ie_n + w_en_n
    f_psipsi = -np.array(
        [
            [0.0, -w_in_n_att[2], w_in_n_att[1]],
            [w_in_n_att[2], 0.0, -w_in_n_att[0]],
            [-w_in_n_att[1], w_in_n_att[0], 0.0],
        ],
        dtype=np.float64,
    )

    # 4. Sensor Bias Error Dynamics (Anisotropic Gauss-Markov diagonal matrices)
    f_ba = np.diag(-1.0 / tau_a).astype(np.float64)
    f_bg = np.diag(-1.0 / tau_g).astype(np.float64)

    # 5. Assemble 15x15 Block Matrix
    zero_3x3 = np.zeros((3, 3), dtype=np.float64)

    f_matrix = np.block(
        [
            [f_rr, f_rv, zero_3x3, zero_3x3, zero_3x3],
            [f_vr, f_vv, f_vpsi, c_b2n, zero_3x3],
            [zero_3x3, zero_3x3, f_psipsi, zero_3x3, -c_b2n],
            [zero_3x3, zero_3x3, zero_3x3, f_ba, zero_3x3],
            [zero_3x3, zero_3x3, zero_3x3, zero_3x3, f_bg],
        ]
    )

    return f_matrix


def discretize_phi_matrix(f_matrix: np.ndarray, dt: float) -> np.ndarray:
    """
    Discretizes the continuous system dynamics matrix F into State Transition Matrix Phi
    using a first-order Taylor expansion suitable for high-frequency (400 Hz) loops.
    """
    return np.eye(15, dtype=np.float64) + (f_matrix * dt)


def build_discrete_q_matrix(
    c_b2n: np.ndarray,
    dt: float,
    sigma_vrw: float,
    sigma_arw: float,
    sigma_ba_walk: float,
    sigma_bg_walk: float,
    is_vibrating: bool,
    accel_var_penalty: np.ndarray,
    gyro_var_penalty: np.ndarray,
) -> np.ndarray:
    """
    Constructs the Discrete Process Noise Covariance Matrix (Q_k) incorporating
    thermomechanical base noise and adaptive vibrational penalties.
    """
    q_v_base = (sigma_vrw**2) * np.eye(3, dtype=np.float64)
    q_psi_base = (sigma_arw**2) * np.eye(3, dtype=np.float64)

    q_v_nav = c_b2n @ q_v_base @ c_b2n.T
    q_psi_nav = c_b2n @ q_psi_base @ c_b2n.T

    if is_vibrating:
        q_v_nav += np.diag(accel_var_penalty)
        q_psi_nav += np.diag(gyro_var_penalty)

    q_v_discrete = q_v_nav * dt
    q_psi_discrete = q_psi_nav * dt

    q_ba_discrete = (sigma_ba_walk**2) * np.eye(3, dtype=np.float64) * dt
    q_bg_discrete = (sigma_bg_walk**2) * np.eye(3, dtype=np.float64) * dt

    zero_3x3 = np.zeros((3, 3), dtype=np.float64)

    q_matrix = np.block(
        [
            [zero_3x3, zero_3x3, zero_3x3, zero_3x3, zero_3x3],
            [zero_3x3, q_v_discrete, zero_3x3, zero_3x3, zero_3x3],
            [zero_3x3, zero_3x3, q_psi_discrete, zero_3x3, zero_3x3],
            [zero_3x3, zero_3x3, zero_3x3, q_ba_discrete, zero_3x3],
            [zero_3x3, zero_3x3, zero_3x3, zero_3x3, q_bg_discrete],
        ]
    )

    return q_matrix


def predict_error_covariance(
    phi_k: np.ndarray, p_prev: np.ndarray, q_k: np.ndarray
) -> np.ndarray:
    """
    Propagates the a priori error covariance matrix via the Discrete Riccati Equation.
    Enforces absolute symmetry to maintain numerical stability.
    """
    p_minus = (phi_k @ p_prev @ phi_k.T) + q_k
    return 0.5 * (p_minus + p_minus.T)
