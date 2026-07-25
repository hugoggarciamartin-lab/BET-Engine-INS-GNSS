import numpy as np
from phase3_navigation.geodesy_math import (
    calc_radii,
    skew_sym,
    quat2dcm,
    somigliana_gravity,
    A_WGS,
)

class ESKFPredictor:
    """Error-State Kalman Filter 15-State Model Prediction Core"""
    def __init__(self, dt_ float, tau_a: float, tau_g: float, ):