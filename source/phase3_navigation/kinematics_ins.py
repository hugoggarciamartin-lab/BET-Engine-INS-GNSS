import numpy as np
from geodesy_math import (
    calc_radii,
    somig_gravity,
    quat2dcm,
    quat_kin_matrix,
    skew_sym,
    norm_quat,
    OMEGA_E,
)


def earth_rate_enu(lat_rad: float) -> np.ndarray:
    """Calculates Mean Earth Rate Vector in ENU frame"""
    return np.array([0.0, OMEGA_E * np.cos(lat_rad), OMEGA_E * np.sin(lat_rad)])


def rate_transprt_enu(
    v_E: float, v_N: float, lat_rad: float, alt_m: float, r_m: float, r_n: float
) -> np.ndarray:
    """Calculates transport Rate along Earth Curvature in ENU frame"""

    return np.array(
        [
            -v_N / (r_m + alt_m),
            v_E / (r_n + alt_m),
            (v_E * np.tan(lat_rad) / (r_n + alt_m)),
        ],
        dtype=np.float64,
    )


def posit_lin2ang_matrix(
    lat_rad: float, alt_m: float, r_m: float, r_n: float
) -> np.ndarray:
    """Builds the diagonal matrix D that transforms lineal velocity to geodesic angular rate (phi_dot, lambda_dot, h_dot)"""

    return np.array(
        [
            [0.0, 1.0 / (r_m + alt_m), 0.0],
            [1.0 / ((r_n + alt_m) * np.cos(lat_rad)), 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def evaluate_ins_derivatives(
    state: np.ndarray, f_b: np.ndarray, w_b: np.ndarray
) -> np.ndarray:
    """Evaluates de coupled ODE system from Strapdown mecanization
    State Vector: [lat, lon, alt, v_E, v_N, v_U, q0, q1, q2, q3]"""

    lat, alt = state[0], state[2]
    v_n_vec = state[3:6]
    q_b2n = state[6:10]

    # Geodesic Parameters
    r_m, r_n = calc_radii(lat)
    g_h = somig_gravity(lat, alt)

    # Nav Angular Rates
    w_ie_n = earth_rate_enu(lat)
    w_en_n = rate_transprt_enu(v_n_vec[0], v_n_vec[1], lat, alt, r_m, r_n)
    w_in_n = w_ie_n + w_en_n

    # Attitude Transformation
    dcm_b2n = quat2dcm(q_b2n)

    # Attitude Derivative by Quaternion
    w_bn_b = w_b - (dcm_b2n.T @ w_in_n)
    omega_matrix = quat_kin_matrix(w_bn_b)
    quat_dot = 0.5 * (omega_matrix @ q_b2n)

    # Velocity Derivative
    f_n = dcm_b2n @ f_b
    coriolis = np.cross((2.0 * w_ie_n + w_en_n), v_n_vec)
    gravity = np.array([0.0, 0.0, -g_h], dtype=np.float64)
    vel_dot = f_n - coriolis + gravity

    # Position derivative
    matrix_D = posit_lin2ang_matrix(lat, alt, r_m, r_n)
    pos_dot = matrix_D @ v_n_vec

    return np.concatenate((pos_dot, vel_dot, quat_dot), dtype=np.float64)


def rk4_integration_step(
    state: np.ndarray, f_b: np.ndarray, w_b: np.ndarray, dt: float
) -> np.ndarray:
    """Execute one-step integration by 4º Order Runge-Kutta Method"""

    k1 = evaluate_ins_derivatives(state, f_b, w_b)
    k2 = evaluate_ins_derivatives(state + 0.5 * dt * k1, f_b, w_b)
    k3 = evaluate_ins_derivatives(state + 0.5 * dt * k2, f_b, w_b)
    k4 = evaluate_ins_derivatives(state + dt * k3, f_b, w_b)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    new_state[6:10] = norm_quat(new_state[6:10])  # Normalize quaternion

    return new_state
