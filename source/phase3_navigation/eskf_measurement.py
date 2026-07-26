import numpy as np
import geodesy_math as mat
import kinematics_ins as kin


def calc_h_matrix(
    r_m: float,
    r_n: float,
    lat: float,
    alt: float,
    dcm_b2n: np.ndarray,
    r_arm_b: np.ndarray,
    m_n: np.ndarray,
) -> np.ndarray:
    """
    Builds the 10x15 Observation Jacobian Matrix H_k for Closed-Loop ESKF.
    State Sequence: [delta_r(3), delta_v(3), delta_psi(3), delta_ba(3), delta_bg(3)]
    Measurements Sequence: [GNSS_Pos(3), GNSS_Vel(3), Baro_Alt(1), Mag_Body(3)]
    """
    h_k = np.zeros((10, 15), dtype=np.float64)

    # GNSS Position Block
    h_k[0:3, 0:3] = np.eye(3, dtype=np.float64)

    r_arm_n = dcm_b2n @ r_arm_b
    r_arm_n_skew = (
        mat.skew_symmetric(r_arm_n)
        if hasattr(mat, "skew_symmetric")
        else np.array(
            [
                [0.0, -r_arm_n[2], r_arm_n[1]],
                [r_arm_n[2], 0.0, -r_arm_n[0]],
                [-r_arm_n[1], r_arm_n[0], 0.0],
            ],
            dtype=np.float64,
        )
    )

    h_k[0:3, 6:9] = -r_arm_n_skew

    # GNSS Velocity Block
    h_k[3:6, 3:6] = np.eye(3, dtype=np.float64)

    # Barometric Altitude Block
    h_k[6, 2] = -1.0

    # Magnetometer Attitude Block
    m_n_skew = (
        mat.skew_symmetric(m_n)
        if hasattr(mat, "skew_symmetric")
        else np.array(
            [[0.0, -m_n[2], m_n[1]], [m_n[2], 0.0, -m_n[0]], [-m_n[1], m_n[0], 0.0]],
            dtype=np.float64,
        )
    )

    h_mag_attitude = -(dcm_b2n.T @ m_n_skew)
    h_k[7:10, 6:9] = h_mag_attitude

    return h_k


def calc_r_matrix(
    sigma_gnss_p: np.ndarray,
    sigma_gnss_v: np.ndarray,
    sigma_baro: np.ndarray,
    sigma_mag: np.ndarray,
    hdop: float,
    vdop: float,
    mach_number: float,
    is_vibrating: bool,
) -> np.ndarray:
    """Builds Measurement Noise Covariance Matrix R_k
    Includes Venturi effect in High Mach Numbers and DOP degradation in GNSS"""
    r_matrix = np.zeros((10, 10), dtype=np.float64)

    # GNSS Position
    r_matrix[0, 0] = (sigma_gnss_p[0] * hdop) ** 2
    r_matrix[1, 1] = (sigma_gnss_p[1] * hdop) ** 2
    r_matrix[2, 2] = (sigma_gnss_p[2] * vdop) ** 2

    # GNSS Velocity
    r_matrix[3:6, 3:6] = np.diag(sigma_gnss_v**2)

    # Barometer: Extract vertical component [2]
    sigma_baro_dyn = sigma_baro[2]
    if is_vibrating or mach_number > 0.3:
        sigma_baro_dyn *= max(1.0, (mach_number * 5) ** 2)

    r_matrix[6, 6] = sigma_baro_dyn**2

    # Magnetometer
    r_matrix[7:10, 7:10] = np.diag(sigma_mag**2)

    return r_matrix


def calc_inno_vector(
    pos_ins: np.ndarray,
    vel_ins: np.ndarray,
    alt_ins: float,
    mag_ins: np.ndarray,
    pos_gnss: np.ndarray,
    vel_gnss: np.ndarray,
    alt_baro: float,
    mag_meas: np.ndarray,
    r_m: float,
    r_n: float,
    dcm_b2n: np.ndarray,
    r_arm_b: np.ndarray,
    w_bn_b: np.ndarray,
) -> np.ndarray:
    """Computes 10x1 innovation vector z_k. Includes lever-arm compensation"""

    lat, lon, h = pos_ins
    pos_arm_n = dcm_b2n @ r_arm_b

    # Lever-arm Position compensation (Add extra position due to GNSS separation from IMU)
    lat_sens = lat + (pos_arm_n[0] / (r_m + h))
    lon_sens = lon + (pos_arm_n[1] / (r_n + h) * np.cos(lat))
    h_sens = h - pos_arm_n[2]

    # Lever-arm Velocity compensation
    vel_sens = vel_ins + (dcm_b2n @ np.cross(w_bn_b, r_arm_b))

    # Residual from INS vs External Measurement
    z_k = np.zeros(10, dtype=np.float64)
    z_k[0] = (lat_sens - pos_gnss[0]) * (r_m + h_sens)
    z_k[1] = (lon_sens - pos_gnss[1]) * (r_n + h_sens) * np.cos(lat_sens)
    z_k[2] = -(h_sens - pos_gnss[2])

    z_k[3:6] = vel_sens - vel_gnss
    z_k[6] = -(alt_ins - alt_baro)
    z_k[7:10] = mag_meas - dcm_b2n.T @ mag_ins

    return z_k


def calc_s_k_covariance(
    p_minus: np.ndarray, h_k: np.ndarray, r_k: np.ndarray
) -> np.ndarray:
    """
    Computes the theoretical innovation covariance envelope S_k.
    Enforces strict symmetry to prevent floating-point degradation.
    """
    s_k = (h_k @ p_minus @ h_k.T) + r_k

    return 0.5 * (s_k + s_k.T)


def exe_kalman_update(
    p_minus: np.ndarray,
    h_k: np.ndarray,
    r_k: np.ndarray,
    z_k: np.ndarray,
    k_innov_sigma: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Calculates Kalman Gain K_k, applies 50-sigma innovation trap adn returns State-Error Vector
    as well as Joseph Form Covariance (applies P_plus symmetric covariance matrix)"""

    s_k = calc_s_k_covariance(p_minus, h_k, r_k)

    try:
        s_k_inv = np.linalg.pinv(s_k)
    except np.linalg.LinAlgError:
        return np.zeros(15, dtype=np.float64), p_minus, False

    # Sigma Innovation Trap
    sigmas = np.sqrt(np.diag(s_k))
    if np.any(np.abs(z_k) > k_innov_sigma * sigmas):
        return np.zeros(15, dtype=np.float64), p_minus, False

    k_k = p_minus @ h_k.T @ s_k_inv
    delta_x = k_k @ z_k

    # Joseph Form Covariance Matrix Update
    i_kh = np.eye(15, dtype=np.float64) - (k_k @ h_k)
    p_plus = (i_kh @ p_minus @ i_kh.T) + (k_k @ r_k @ k_k.T)
    p_plus = 0.5 * (p_plus + p_plus.T)

    return delta_x, p_plus, True
