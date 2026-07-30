"""ESKF forward prediction algorithms. Computes the continous-time
system dynamics Jacobian F, the discrete state transition matrix PHI,
and the process noise covariance matrix Q for a priori error propagation"""

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
    a_wgs: float,
) -> np.ndarray:
    """
    Builds the 15x15 continuous-time system dynamics Jacobian matrix F.
    State Vector sequence: [delta_r, delta_v, delta_psi, delta_ba, delta_bg]
    Accepts anisotropic 3-axis correlation time vectors (tau_a, tau_g).
    """
    v_e, v_n, v_u = v_n_vec[0], v_n_vec[1], v_n_vec[2]

    # Position Error Dynamics
    f_rr = np.zeros((3, 3), dtype=np.float64)
    f_rv = np.eye(3, dtype=np.float64)
    # Velocity Error Dynamics
    f_vr = np.array(
        [
            [0.0, 0.0, -v_n / ((r_m + alt_m) ** 2)],
            [
                (v_e * np.sin(lat_rad)) / ((r_n + alt_m) * (np.cos(lat_rad) ** 2)),
                0.0,
                -v_e / ((r_n + alt_m) ** 2),
            ],
            [0.0, 0.0, (2.0 * g_0) / a_wgs],
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

    # Attitude Error Dynamics
    w_in_n_att = w_ie_n + w_en_n
    f_psipsi = -np.array(
        [
            [0.0, -w_in_n_att[2], w_in_n_att[1]],
            [w_in_n_att[2], 0.0, -w_in_n_att[0]],
            [-w_in_n_att[1], w_in_n_att[0], 0.0],
        ],
        dtype=np.float64,
    )

    # Sensor Bias Error Dynamics (Anisotropic Gauss-Markov models)
    f_ba = np.diag(-1.0 / tau_a).astype(np.float64)
    f_bg = np.diag(-1.0 / tau_g).astype(np.float64)

    # F matrix assemble
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
    Discretizes the continuous system dynamics matrix F into State Transition Matrix Phi.
    Utilizes first-order Taylor expansion for real-time deterministic performance.
    """
    return np.eye(15, dtype=np.float64) + (f_matrix * dt)


def calc_q_matrix_discrete(
    c_b2n: np.ndarray,
    dt: float,
    sigma_vrw_vec: np.ndarray,
    sigma_arw_vec: np.ndarray,
    sigma_ba_walk_vec: np.ndarray,
    sigma_bg_walk_vec: np.ndarray,
    tau_a: np.ndarray,
    tau_g: np.ndarray,
    is_vibrating: bool,
    accel_var_penalty: np.ndarray,
    gyro_var_penalty: np.ndarray,
) -> np.ndarray:
    """
    Constructs the Discrete Process Noise Covariance Matrix (Q_k).
    Properly couples Bias Instability with Correlation Time for Gauss-Markov models.
    """
    # Base thermomechanical noise mapping
    q_v_base = np.diag(sigma_vrw_vec**2).astype(np.float64)
    q_psi_base = np.diag(sigma_arw_vec**2).astype(np.float64)

    # Transform body-frame noise spectral densities to navigation frame
    q_v_nav = c_b2n @ q_v_base @ c_b2n.T
    q_psi_nav = c_b2n @ q_psi_base @ c_b2n.T

    # Dynamic scale factor injection for structural resonance modes
    if is_vibrating:
        q_v_nav += np.diag(accel_var_penalty).astype(np.float64)
        q_psi_nav += np.diag(gyro_var_penalty).astype(np.float64)

    # Temporal integration (Euler method for computational bounds)
    q_v_discrete = q_v_nav * dt
    q_psi_discrete = q_psi_nav * dt

    # Gauss-Markov discrete variance formulation for dynamic biases
    q_ba_discrete = np.diag((2.0 * sigma_ba_walk_vec**2) / tau_a) * dt
    q_bg_discrete = np.diag((2.0 * sigma_bg_walk_vec**2) / tau_g) * dt

    zero_3x3 = np.zeros((3, 3), dtype=np.float64)

    # Build full covariance geometry
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
    Propagates the a priori error covariance matrix via Discrete Riccati Equation.
    Enforces strict matrix symmetry to prevent floating-point eigenvalue corruption.
    """
    p_minus = (phi_k @ p_prev @ phi_k.T) + q_k
    return 0.5 * (p_minus + p_minus.T)
