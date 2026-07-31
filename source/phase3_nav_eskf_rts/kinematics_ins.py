"""Strapdown Inertial Navigation System INS Mechanization.
Implements continous-time coupled ODEs for position velocity
and attitude propagated via explicit 4-th Order Runge-Kutta integration"""

import numpy as np
import geodesy_math as mat


def earth_rate_enu(lat_rad: float) -> np.ndarray:
    """Calculates Mean Earth Rate Vector in ENU frame"""
    return np.array([0.0, mat.OMEGA_E * np.cos(lat_rad), mat.OMEGA_E * np.sin(lat_rad)])


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


def calc_isa_atmosphere(alt, p0=101325.0, t0=288.15, L=0.0065, g=9.80665, R=287.05287):
    """
    Computes standard atmospheric temperature and pressure at a given altitude.
    """
    if np.any(p0 <= 0):
        raise ValueError(
            "Critical Sensor Failure: Base pressure (p0) must be strictly positive."
        )

    # Tropospheric model (Valid up to 11,000 meters)
    t_loc = t0 - L * alt
    p_loc = p0 * (1.0 - (L * alt) / t0) ** ((g) / (L * R))

    return t_loc, p_loc


def calc_isa_altitude(p_loc, p0=101325.0, t0=288.15, L=0.0065, g=9.80665, R=287.05287):
    """
    Computes barometric altitude based on local static pressure using ISA equations.
    """
    if np.any(p0 <= 0) or np.any(p_loc <= 0):
        raise ValueError(
            "Critical Sensor Failure: Pressure readings must be strictly positive."
        )

    # Inverse ISA tropospheric equation
    alt = (t0 / L) * (1.0 - (p_loc / p0) ** ((L * R) / g))
    return alt


def calc_mach_number(
    v_n_vec: np.ndarray,
    alt_m: np.ndarray | float,
    p0_isa: float,
    t0_isa: float,
    l_isa: float,
    g0: float,
) -> np.ndarray | float:
    """
    Calculates Local Mach Number by considering the Atmosphere Model ISA.
    Vectorized to accept both Nx3 velocity arrays and 3x1 single vectors.
    """
    v_vec = np.atleast_2d(v_n_vec)
    # Norm along the velocity axis (handles both single vectors and arrays of vectors)
    v_tas = np.linalg.norm(v_vec, axis=1)

    t_local, _ = calc_isa_atmosphere(alt_m, p0_isa, t0_isa, l_isa, g0)

    # Local Sound Velocity array
    a_local = np.sqrt(1.4 * 287.058 * t_local)
    mach = v_tas / a_local

    if np.isscalar(alt_m) and v_vec.shape[0] == 1:
        return float(mach[0])
    return mach


def evaluate_ins_derivatives(
    state: np.ndarray, f_b: np.ndarray, w_b: np.ndarray, out_dot: np.ndarray
) -> None:
    """Evaluates de coupled ODE system from Strapdown mecanization
    State Vector: [lat, lon, alt, v_E, v_N, v_U, q0, q1, q2, q3]"""

    lat, alt = state[0], state[2]
    v_n_vec = state[3:6]
    q_b2n = state[6:10]

    # Geodesic Parameters
    r_m, r_n = mat.calc_radii(lat)
    g_h = mat.somig_gravity(lat, alt)

    # Nav Angular Rates
    w_ie_n = earth_rate_enu(lat)
    w_en_n = rate_transprt_enu(v_n_vec[0], v_n_vec[1], lat, alt, r_m, r_n)
    w_in_n = w_ie_n + w_en_n

    # Attitude Transformation
    dcm_b2n = mat.quat2dcm(q_b2n)

    # Attitude Derivative by Quaternion
    w_bn_b = w_b - (dcm_b2n.T @ w_in_n)
    omega_matrix = mat.quat_kin_matrix(w_bn_b)
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
    m_mat = mat.skew_sym(m)

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
    k1: np.ndarray,  # inputs k1, k2, k3, k4, temp_state are (First Instant) np.zeros((10, 1), dtype=np.float64) arrays
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    temp_state: np.ndarray,
) -> None:
    """
    Performs standard Runge-Kutta 4th order (RK4) numerical integration.
    Instead of creating new variables or temporary lists at every step (which
    slows down the computer and creates memory waste), this function uses
    pre-allocated memory arrays that were prepared beforehand.It directly
    updates and modifies the original 'state' array in-place, making the
    calculation extremely fast
    """

    # k1
    evaluate_ins_derivatives(state, f_b, w_b, k1)

    # k2 (temp_state = state + 0.5*dt*k1)
    temp_state[:] = k1
    temp_state *= 0.5 * dt
    temp_state += state
    temp_state[6:10] = mat.norm_quat(temp_state[6:10])
    evaluate_ins_derivatives(temp_state, f_b, w_b, k2)

    # k3 (temp_state = state + 0.5*dt*k2)
    temp_state[:] = k2
    temp_state *= 0.5 * dt
    temp_state += state
    temp_state[6:10] = mat.norm_quat(temp_state[6:10])
    evaluate_ins_derivatives(temp_state, f_b, w_b, k3)

    # k4 (temp_state = state + dt*k3)
    temp_state[:] = k3
    temp_state *= dt
    temp_state += state
    temp_state[6:10] = mat.norm_quat(temp_state[6:10])
    evaluate_ins_derivatives(temp_state, f_b, w_b, k4)

    # Final assembly for RK4 integration
    state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Last quaternion normalization
    state[6:10] = mat.norm_quat(state[6:10])
