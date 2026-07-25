import numpy as np
from geodesy_math import (
    calc_radii,
    somig_gravity,
    quat2dcm,
    quat_kin_matrix,
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
    state: np.ndarray, f_b: np.ndarray, w_b: np.ndarray, out_dot: np.ndarray
) -> None:
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

    out_dot[0:3] = pos_dot
    out_dot[3:6] = vel_dot
    out_dot[6:10] = quat_dot


def calc_inver_calibr_matrix(s_ppm: list[float], m_deg: list[float]) -> np.ndarray:
    """Builds and inverts the (I + M + S) transformation matrix from config_baseline
    Scale Factor is diagonal while Missalignment is cross-axis
    This Must Compute just Once during initialization"""

    s = np.array(s_ppm, dtype=np.float64) * 1e6
    m = np.deg2rad(np.array(m_deg, dtype=np.float64))

    s_mat = np.diag(s)

    # Missalignment as a symmetric matrix with null-elements diagonal
    m_mat = np.array(
        [[0.0, m[0], m[1]], [m[0], 0.0, m[2]], [m[1], m[2], 0.0]], dtype=np.float64
    )

    calib_matrix = np.eye(3, dtype=np.float64) + s_mat + m_mat
    return np.linalg.inv(calib_matrix)


def apply_sensor_calibration(
    f_raw: np.ndarray,
    w_raw: np.ndarray,
    inv_calib_a: np.ndarray,
    inv_calib_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply inverse calibration sensor to raw IMU mesaurements"""

    f_calib = inv_calib_a @ f_raw
    w_calib = inv_calib_g @ w_raw

    return f_calib, w_calib


def rk4_integration_step(
    state: np.ndarray,
    f_b: np.ndarray,
    w_b: np.ndarray,
    dt: float,
    k1: np.ndarray,  # inputs k1, k2, k3, k4, temp_state are, initially, np.zeros((10, 1), dtype=np.float64) arrays
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    temp_state: np.ndarray,
) -> None:
    """
    Executes standard RK4 integration using strictly pre-allocated buffers.
    Zero dynamic memory allocation. Mutates 'state' array in-place.
    """
    # k1
    evaluate_ins_derivatives(state, f_b, w_b, k1)

    # k2 (temp_state = state + 0.5*dt*k1)
    temp_state[:] = k1
    temp_state *= 0.5 * dt
    temp_state += state
    evaluate_ins_derivatives(temp_state, f_b, w_b, k2)

    # k3 (temp_state = state + 0.5*dt*k2)
    temp_state[:] = k2
    temp_state *= 0.5 * dt
    temp_state += state
    evaluate_ins_derivatives(temp_state, f_b, w_b, k3)

    # k4 (temp_state = state + dt*k3)
    temp_state[:] = k3
    temp_state *= dt
    temp_state += state
    evaluate_ins_derivatives(temp_state, f_b, w_b, k4)

    # Accumulate slopes k in 'temp_state'
    # temp_state = k1 + 2*k2 + 2*k3 + k4
    temp_state[:] = k2
    temp_state += k3
    temp_state *= 2.0
    temp_state += k1
    temp_state += k4

    # Final 'state' update: state = state + (dt/6.0) * temp_state
    temp_state *= dt / 6.0
    state += temp_state

    # Strict S3 Manifold Quaternion Normalization
    q_norm = norm_quat(state[6:10])
    state[6:10] /= q_norm
