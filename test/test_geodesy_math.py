"""
Validates geometrical transformations, coordinate conversions (WGS84),
and mathematical singularities handling (e.g., polar latitudes, gimbal lock).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from source.phase3_nav_eskf_rts.geodesy_math import calc_radii, quat2eul

TOLERANCE = 1e-9


def test_calc_radii_nominal_equator():
    lat_eq = 0.0
    rm, rn = calc_radii(lat_eq)
    assert np.abs(rn - 6378137.0) < TOLERANCE


def test_calc_radii_poles():
    lat_pole = np.pi / 2
    rm, rn = calc_radii(lat_pole)
    assert np.abs(rm - rn) < TOLERANCE


def test_quat2eul_scalar_input():
    q_scalar = np.array([1.0, 0.0, 0.0, 0.0])
    euler = quat2eul(q_scalar)
    assert euler.ndim == 1
    assert euler.shape == (3,)
    assert np.allclose(euler, np.zeros(3))


def test_quat2eul_vectorized_input():
    q_vec = np.array([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]])
    euler = quat2eul(q_vec)
    assert euler.ndim == 2
    assert euler.shape == (2, 3)
    assert np.abs(euler[1, 0] - (np.pi / 2)) < 1e-6


def test_quat2eul_gimbal_lock_pitch():
    q_gimbal = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    euler = quat2eul(q_gimbal)
    assert np.abs(euler[1] - (np.pi / 2)) < 1e-6
