import sys
from pathlib import Path
import numpy as np
import pytest

# Resolución absoluta de la raíz del proyecto (sube 2 niveles desde /tests/)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "source"))

from phase3_navigation.geodesy_math import calc_radii, quat2dcm, skew_sym, norm_quat


def test_calc_radii_equator():
    """Valida  la cinemática de los radios de curvatura en el ecuador (lat = 0)."""
    r_m, r_n = calc_radii(0.0)
    assert np.isclose(r_n, 6378137.0), (
        "Error Fatal: El radio del primer vertical en el ecuador debe ser exactamente el semieje mayor WGS84."
    )
    assert r_m < r_n, (
        "El radio meridiano debe ser analíticamente menor al primer vertical en el ecuador."
    )


def test_quat2dcm_identity():
    """Verifica que la condición de frontera del cuaternión nulo genera la matriz identidad pura."""
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dcm = quat2dcm(q_identity)
    expected = np.eye(3, dtype=np.float64)
    np.testing.assert_array_almost_equal(
        dcm,
        expected,
        decimal=12,
        err_msg="La transformación DCM ha perdido ortogonalidad.",
    )


def test_skew_sym_properties():
    """Asegura que la matriz antisimétrica cumple estrictamente con A = -A^T."""
    v = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    skew = skew_sym(v)
    np.testing.assert_array_almost_equal(
        skew, -skew.T, decimal=12, err_msg="La matriz generada no es antisimétrica."
    )
    assert np.isclose(skew.trace(), 0.0), (
        "La traza de una matriz antisimétrica debe ser cero."
    )


def test_norm_quat_zero_protection():
    """Garantiza la protección matemática (Failsafe) contra la división por cero en cuaterniones corruptos."""
    q_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_norm = norm_quat(q_zero)
    expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(
        q_norm,
        expected,
        err_msg="El failsafe de normalización ha fallado al manejar un vector nulo.",
    )


if __name__ == "__main__":
    test_calc_radii_equator()
    test_quat2dcm_identity()
    test_skew_sym_properties()
    test_norm_quat_zero_protection()
