"""Parses de YAML baseline configuration.
Acts at the single source of truth for universal physical constantes,
sensor stocastic parameters, calibration adn algorythm constantes"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any


class ConfigParser:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.raw_data: Dict[str, Any] = {}
        self.params: Dict[str, Any] = {}

    def parse(self) -> Dict[str, Any]:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Config not found: {self.filepath}")

        with open(self.filepath, "r") as f:
            self.raw_data = yaml.safe_load(f)

        try:
            # Geodesy
            self.params["a"] = float(self.raw_data["geodesy"]["a"])
            self.params["e2"] = float(self.raw_data["geodesy"]["e2"])
            self.params["omega_e"] = float(self.raw_data["geodesy"]["omega_e"])
            self.params["g_local"] = float(self.raw_data["geodesy"]["g_e"])
            self.params["g_e"] = float(self.raw_data["geodesy"]["g_e"])
            self.params["k"] = float(self.raw_data["geodesy"]["k"])
            self.params["p0_isa"] = float(self.raw_data["geodesy"]["p0_isa"])
            self.params["t0_isa"] = float(self.raw_data["geodesy"]["t0_isa"])
            self.params["l_isa"] = float(self.raw_data["geodesy"]["l_isa"])

            # Covariance P0
            self.params["sigma_pos"] = float(
                self.raw_data["initial_uncertainty"]["sigma_pos_m"]
            )
            self.params["sigma_vel"] = float(
                self.raw_data["initial_uncertainty"]["sigma_vel_ms"]
            )
            self.params["sigma_theta_phi"] = np.deg2rad(
                float(self.raw_data["initial_uncertainty"]["sigma_theta_phi_deg"])
            )
            self.params["sigma_psi"] = np.deg2rad(
                float(self.raw_data["initial_uncertainty"]["sigma_psi_deg"])
            )
            self.params["sigma_ba0"] = float(
                self.raw_data["initial_uncertainty"]["sigma_ba0_ms2"]
            )
            self.params["sigma_bg0"] = float(
                self.raw_data["initial_uncertainty"]["sigma_bg0_rads"]
            )

            # Calibration
            self.params["r_arm_b"] = np.array(
                self.raw_data["calibration"]["r_arm_b_m"], dtype=np.float64
            )
            self.params["s_a_ppm"] = [
                float(x) for x in self.raw_data["calibration"]["s_a_ppm"]
            ]
            self.params["s_g_ppm"] = [
                float(x) for x in self.raw_data["calibration"]["s_g_ppm"]
            ]
            self.params["m_a_deg"] = [
                float(x) for x in self.raw_data["calibration"]["m_a_deg"]
            ]
            self.params["m_g_deg"] = [
                float(x) for x in self.raw_data["calibration"]["m_g_deg"]
            ]

            self.params["m_si_mat"] = np.array(
                self.raw_data["calibration"]["m_si_mat"],
                dtype=np.float64,
            )

            self.params["m_hi_vec"] = np.array(
                [float(x) for x in self.raw_data["calibration"]["m_hi_vec"]],
                dtype=np.float64,
            )

            # Sensor Quality & Base Noises
            self.params["hdop_nominal"] = float(
                self.raw_data["sensor_quality"]["hdop_nominal"]
            )
            self.params["vdop_nominal"] = float(
                self.raw_data["sensor_quality"]["vdop_nominal"]
            )

            self.params["sigma_gnss_p"] = np.array(
                [float(x) for x in self.raw_data["sensor_quality"]["sigma_gnss_p_m"]],
                dtype=np.float64,
            )
            self.params["sigma_gnss_v"] = np.array(
                [float(x) for x in self.raw_data["sensor_quality"]["sigma_gnss_v_ms"]],
                dtype=np.float64,
            )
            self.params["sigma_baro"] = np.array(
                [float(x) for x in self.raw_data["sensor_quality"]["sigma_baro_m"]],
                dtype=np.float64,
            )
            self.params["sigma_mag"] = np.array(
                [float(x) for x in self.raw_data["sensor_quality"]["sigma_mag_ut"]],
                dtype=np.float64,
            )

            # Allan Variance Parameters
            self.params["vrw"] = np.array(
                [
                    float(x)
                    for x in self.raw_data["allan_variance"]["vrw_noise_density"]
                ],
                dtype=np.float64,
            )
            self.params["accel_bias_inst"] = np.array(
                [
                    float(x)
                    for x in self.raw_data["allan_variance"]["accel_bias_instability"]
                ],
                dtype=np.float64,
            )
            self.params["accel_corr_time"] = np.array(
                [
                    float(x)
                    for x in self.raw_data["allan_variance"]["accel_correlation_time_s"]
                ],
                dtype=np.float64,
            )
            self.params["arw"] = np.array(
                [
                    float(x)
                    for x in self.raw_data["allan_variance"]["arw_noise_density"]
                ],
                dtype=np.float64,
            )
            self.params["gyro_bias_inst"] = np.array(
                [
                    float(x)
                    for x in self.raw_data["allan_variance"]["gyro_bias_instability"]
                ],
                dtype=np.float64,
            )
            self.params["gyro_corr_time"] = np.array(
                [
                    float(x)
                    for x in self.raw_data["allan_variance"]["gyro_correlation_time_s"]
                ],
                dtype=np.float64,
            )

            # Signal Processing
            self.params["filter_cutoff_hz"] = float(
                self.raw_data["signal_processing"]["filter_cutoff_hz"]
            )
            self.params["filter_order"] = int(
                self.raw_data["signal_processing"]["filter_order"]
            )

            # Tuning
            self.params["k_innov_sigma"] = float(
                self.raw_data["algorithm_tuning"]["k_innov_sigma"]
            )
            self.params["k_vib_mult"] = float(
                self.raw_data["algorithm_tuning"]["k_vib_mult"]
            )
            self.params["m_vib_window"] = int(
                self.raw_data["algorithm_tuning"]["m_vib_window"]
            )

            # Vehicle
            self.params["m_0"] = float(self.raw_data["vehicle"]["m_0_kg"])
            self.params["m_f"] = float(self.raw_data["vehicle"]["m_f_kg"])
            self.params["s_ref_m2"] = float(self.raw_data["vehicle"]["s_ref_m2"])
            self.params["time_eng_off"] = float(
                self.raw_data["vehicle"]["time_eng_off"]
            )

        except KeyError as e:
            raise KeyError(f"Missing config key: {e}")
        except ValueError as e:
            raise ValueError(f"Config type error: {e}")

        return self.params


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    path = root / "config" / "config_baseline.yaml"

    parser = ConfigParser(path)
    cfg = parser.parse()
    print("Config parsed successfully. Filter Cutoff:", cfg["filter_cutoff_hz"], "Hz")
