"""
<py test \test>
Validates physical invariants, tensor vectorization, and exception handling
for atmospheric models and kinematic integrations.
"""

import sys
from pathlib import Path


import pytest
import numpy as np
from source.phase3_nav_eskf_rts.kinematics_ins import (
    calc_isa_atmosphere,
    calc_isa_altitude,
    calc_mach_number,
)

TOLERANCE = 1e-6


def test_calc_isa_atmosphere_sea_level():
    alt = 0.0
    p0, t0, L, g = 101325.0, 288.15, 0.0065, 9.80665
    t_loc, p_loc = calc_isa_atmosphere(alt, p0, t0, L, g)
    assert np.abs(t_loc - t0) < TOLERANCE
    assert np.abs(p_loc - p0) < TOLERANCE


def test_calc_isa_atmosphere_negative_pressure_handling():
    alt = 1000.0
    p0_faulty, t0, L, g = -101325.0, 288.15, 0.0065, 9.80665
    with pytest.raises(ValueError, match="strictly positive"):
        calc_isa_atmosphere(alt, p0_faulty, t0, L, g)


def test_calc_isa_altitude_negative_pressure_handling():
    p_loc_faulty = -50000.0
    with pytest.raises(ValueError, match="strictly positive"):
        calc_isa_altitude(p_loc_faulty)


def test_calc_mach_number_vectorized():
    v_n_vec = np.array([[0.0, 0.0, 0.0], [340.294, 0.0, 0.0], [0.0, -170.0, 0.0]])
    alt = np.array([0.0, 0.0, 0.0])
    mach = calc_mach_number(v_n_vec, alt)
    assert mach.ndim == 1
    assert mach.shape == (3,)
    assert np.abs(mach[0]) < TOLERANCE
    assert np.abs(mach[1] - 1.0) < 0.05
    assert mach[2] > 0
