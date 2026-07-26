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


def calc_isa_atmosphere(
    alt_m: float, p0: float, t0: float, l_isa: float, g0: float
) -> tuple[float, float]:
    """Calculates Local Temperature and Static Pressure using ISA atmosphere model.
    Valid for Troposphere and Stratosphere"""
    h = max(0.0, alt_m)
    r_air = 287.0528

    # Tropopause Constants
    h_11 = 11000.0
    t_11 = t0 - l_isa * h_11
    p_11 = p0 * (t_11 / t0) ** (g0 / (l_isa * r_air))

    if h <= h_11:
        # Troposphere: Constant Temperature Gradient
        t_local = t0 - l_isa * h
        p_local = p0 * (t_local / t0) ** (g0 / (l_isa * r_air))
    else:
        # Stratosphere: Isothermal
        t_local = t_11
        p_local = p_11 * np.exp(-g0 * (h - h_11) / (r_air * t_11))

    return float(t_local), float(p_local)


def calc_isa_altitude(
    p_local: np.ndarray, p0_isa: float, t0_isa: float, l_isa: float, g0: float
) -> np.ndarray:
    """
    Calculates Barometric Altitude from Static Pressure using the inverse ISA atmosphere model.
    Vectorized evaluate both Troposphere (h <= 11000m) and Stratosphere.
    """
    r_air = 287.0528
    h_11 = 11000.0

    # Tropopause boundary constants
    t_11 = t0_isa - l_isa * h_11
    p_11 = p0_isa * (t_11 / t0_isa) ** (g0 / (l_isa * r_air))

    # Troposphere: Adiabatic
    h_tropo = (t0_isa / l_isa) * (1.0 - (p_local / p0_isa) ** ((l_isa * r_air) / g0))

    # Stratosphere: Isothermal
    p_safe = np.maximum(p_local, 1e-10)
    h_strato = h_11 - ((r_air * t_11) / g0) * np.log(p_safe / p_11)

    # Select mathematical regime based on local pressure (higher pressure = troposphere)
    return np.where(p_local > p_11, h_tropo, h_strato)


def calc_mach_number(
    v_n_vec: np.ndarray,
    alt_m: float,
    p0_isa: float,
    t0_isa: float,
    l_isa: float,
    g0: float,
) -> float:
    """
    Calculates Local Mach Number by considering de Atmosphere Model ISA.
    """
    # ENU frame Velocity True Airspeed (Vector Module) (m/s)
    v_tas = np.linalg.norm(v_n_vec)
    # Local tempertaure (K) at an altitude called h
    t_local, _ = calc_isa_atmosphere(alt_m, p0_isa, t0_isa, l_isa, g0)
    # Local Sound Velocity
    a_local = np.sqrt(1.4 * 287.058 * t_local)

    return float(v_tas / a_local)


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
    q_norm = mat.norm_quat(state[6:10])
    state[6:10] /= q_norm
