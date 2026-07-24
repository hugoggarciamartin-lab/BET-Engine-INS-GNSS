import numpy as np

A_WGS = 637187.0
E2_WGS = 0.00669437999
OMEGA_E = 0.00007292115
G_E = 9.7803253359
K_GRAV = 0.001931852652


def calc_radii(lat_rad: float) -> tuple[np.float64, np.float64]:
    """Calculate the principal curvature radius R_M (meridian) and R_N (vertical)"""
    sin2_lat = np.sin(lat_rad) ** 2
    den = 1.0 - E2_WGS * sin2_lat

    r_m = (A_WGS * (1 - E2_WGS**2)) / (den**1.5)
    r_n = A_WGS / np.sqrt(den)

    return np.float64(r_m), np.float64(r_n)


def somig_gravity(lat_rad: float, alt_m: float) -> np.float:
    """Calculate escalar local gravity applying Somilglian equation and free air anomaly correction"""
    sin2_lat = np.sin(lat_rad) ** 2
    num = 1.0 + K_GRAV * sin2_lat
    den = np.sqrt(1.0 - E2_WGS * sin2_lat)

    g_0 = G_E * (num / den)
    g_h = g_0 * (1.0 - (2.0 * alt_m) / A_WGS)

    return np.float64(g_h)
