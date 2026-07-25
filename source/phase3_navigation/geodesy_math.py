import numpy as np

A_WGS = 6378187.0
E2_WGS = 0.00669437999
OMEGA_E = 0.00007292115
G_E = 9.7803253359
K_GRAV = 0.001931852652


def calc_radii(lat_rad: float) -> tuple[np.float64, np.float64]:
    """Calculate the principal curvature radius R_M (meridian) and R_N (vertical)"""
    sin2_lat = np.sin(lat_rad) ** 2
    den = 1.0 - E2_WGS * sin2_lat

    r_m = (A_WGS * (1 - E2_WGS)) / (den**1.5)
    r_n = A_WGS / np.sqrt(den)

    return np.float64(r_m), np.float64(r_n)


def somig_gravity(lat_rad: float, alt_m: float) -> np.float64:
    """Calculate escalar local gravity applying Somilglian equation and free air anomaly correction"""
    sin2_lat = np.sin(lat_rad) ** 2
    num = 1.0 + K_GRAV * sin2_lat
    den = np.sqrt(1.0 - E2_WGS * sin2_lat)

    g_0 = G_E * (num / den)
    g_h = g_0 * (1.0 - (2.0 * alt_m) / A_WGS)

    return np.float64(g_h)


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


def norm_quat(q: np.ndarray) -> np.ndarray:
    """Force a quaternion to have module with value = 1: Belonging to S^3"""

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (q / norm).astype(np.float64)
