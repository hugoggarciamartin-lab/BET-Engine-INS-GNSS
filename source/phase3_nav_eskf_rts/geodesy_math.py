"""WGS84 geodetic transformations and SO(3) attitude kinematics.
Provides typed mathematical for radii curvatura, gravity corrections and quaternion operations"""

import numpy as np

A_WGS = 6378187.0
E2_WGS = 0.00669437999
OMEGA_E = 0.00007292115
G_E = 9.7803253359
K_GRAV = 0.001931852652


def calc_radii(lat_rad: np.ndarray | float) -> tuple:
    """Calculate the principal curvature radius R_M (meridian) and R_N (vertical).
    Polymorphic: handles both scalar ESKF execution and vectorized batch analysis."""
    lat = np.atleast_1d(lat_rad)
    sin2_lat = np.sin(lat) ** 2
    den = 1.0 - E2_WGS * sin2_lat

    r_m = (A_WGS * (1.0 - E2_WGS)) / (den**1.5)
    r_n = A_WGS / np.sqrt(den)

    if np.isscalar(lat_rad):
        return float(r_m[0]), float(r_n[0])
    return r_m, r_n


def somig_gravity(
    lat_rad: np.ndarray | float, alt_m: np.ndarray | float
) -> np.ndarray | float:
    """Calculate escalar local gravity applying Somigliana equation and free air anomaly correction."""
    lat = np.atleast_1d(lat_rad)
    alt = np.atleast_1d(alt_m)

    sin2_lat = np.sin(lat) ** 2
    num = 1.0 + K_GRAV * sin2_lat
    den = np.sqrt(1.0 - E2_WGS * sin2_lat)

    g_0 = G_E * (num / den)
    g_h = g_0 * (1.0 - (2.0 * alt) / A_WGS)

    if np.isscalar(lat_rad) and np.isscalar(alt_m):
        return float(g_h[0])
    return g_h


def skew_sym(v: np.ndarray) -> np.ndarray:
    """Builds an antisymetric matrix 3x3 from a 3x1 vector"""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )


def quat_kin_matrix(w: np.ndarray) -> np.ndarray:
    """Builds the ortogonal propagation matrix 4x4 for quaternion derivative"""

    return np.array(
        [
            [0.0, -w[0], -w[1], -w[2]],
            [w[0], 0.0, w[2], -w[1]],
            [w[1], -w[2], 0.0, w[0]],
            [w[2], w[1], -w[0], 0.0],
        ],
        dtype=np.float64,
    )


def quat2dcm(q: np.ndarray) -> np.ndarray:
    """ "Builds the Cosine Direction Matrix from quaternion elements"""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    c11 = q0**2 + q1**2 - q2**2 - q3**2
    c12 = 2.0 * (q1 * q2 - q0 * q3)
    c13 = 2.0 * (q1 * q3 + q0 * q2)

    c21 = 2.0 * (q1 * q2 + q0 * q3)
    c22 = q0**2 - q1**2 + q2**2 - q3**2
    c23 = 2.0 * (q2 * q3 - q0 * q1)

    c31 = 2.0 * (q1 * q3 - q0 * q2)
    c32 = 2.0 * (q2 * q3 + q0 * q1)
    c33 = q0**2 - q1**2 - q2**2 + q3**2

    return np.array(
        [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]], dtype=np.float64
    )


def quat_ham_prod(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Operates de Hamilton product between 2 quaternion from de Left"""

    p0, p1, p2, p3 = q1[0], q1[1], q1[2], q1[3]
    q_0, q_1, q_2, q_3 = q2[0], q2[1], q2[2], q2[3]

    r0 = p0 * q_0 - p1 * q_1 - p2 * q_2 - p3 * q_3
    r1 = p0 * q_1 + p1 * q_0 + p2 * q_3 - p3 * q_2
    r2 = p0 * q_2 - p1 * q_3 + p2 * q_0 + p3 * q_1
    r3 = p0 * q_3 + p1 * q_2 - p2 * q_1 + p3 * q_0

    return norm_quat(np.array([r0, r1, r2, r3], dtype=np.float64))


def eul2quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Converts Euler Angles (Tait-Bryan ZYX) to a quaternion shaped as [q0, q1, q2, q3]. Assures normalization.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    return norm_quat(np.array([q0, q1, q2, q3], dtype=np.float64))


def quat2eul(q: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion [q0, q1, q2, q3] to Euler angles (Tait-Bryan ZYX).
    Returns [roll, pitch, yaw] in radians. Handles both shape (4,) and (N, 4).
    """
    q_arr = np.atleast_2d(q)
    q0, q1, q2, q3 = q_arr[:, 0], q_arr[:, 1], q_arr[:, 2], q_arr[:, 3]

    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1**2 + q2**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2**2 + q3**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if q.ndim == 1:
        return np.array([roll[0], pitch[0], yaw[0]], dtype=np.float64)
    return np.column_stack((roll, pitch, yaw))


def norm_quat(q: np.ndarray) -> np.ndarray:
    """Force a quaternion to have module with value = 1: Belonging to S^3"""

    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (q / norm).astype(np.float64)


def geodetic_to_enu(
    lat: np.ndarray | float, lon: np.ndarray | float, h: np.ndarray | float
) -> tuple:
    """Project geodetic coordinates to local Cartesian ENU frame.
    Anchors the ENU origin to the first element of the input arrays."""
    lat_arr = np.atleast_1d(lat)
    lon_arr = np.atleast_1d(lon)
    h_arr = np.atleast_1d(h)

    N = A_WGS / np.sqrt(1.0 - E2_WGS * np.sin(lat_arr) ** 2)
    X = (N + h_arr) * np.cos(lat_arr) * np.cos(lon_arr)
    Y = (N + h_arr) * np.cos(lat_arr) * np.sin(lon_arr)
    Z = (N * (1.0 - E2_WGS) + h_arr) * np.sin(lat_arr)

    # ENU Frame Origin
    X0, Y0, Z0 = X[0], Y[0], Z[0]
    lat0, lon0 = lat_arr[0], lon_arr[0]

    dx, dy, dz = X - X0, Y - Y0, Z - Z0

    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)

    E = -slon * dx + clon * dy
    N_enu = -slat * clon * dx - slat * slon * dy + clat * dz
    U = clat * clon * dx + clat * slon * dy + slat * dz

    if np.isscalar(lat) and np.isscalar(lon):
        return float(E[0]), float(N_enu[0]), float(U[0])
    return E, N_enu, U
