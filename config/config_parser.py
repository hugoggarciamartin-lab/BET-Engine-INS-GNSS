import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any


class ConfigParser:  # This Class Convert de .yaml file into an structured Dict
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

            # Launch Position
            self.params["phi_0"] = np.deg2rad(
                float(self.raw_data["launch_pad"]["lat_0_deg"])
            )
            self.params["lam_0"] = np.deg2rad(
                float(self.raw_data["launch_pad"]["lon_0_deg"])
            )
            self.params["h_0"] = float(self.raw_data["launch_pad"]["h_0_m"])

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
            self.params["m_vib_window"] = int(
                self.raw_data["algorithm_tuning"]["m_vib_window"]
            )

            # Vehicle
            self.params["m_0"] = float(self.raw_data["vehicle"]["m_0_kg"])
            self.params["m_f"] = float(self.raw_data["vehicle"]["m_f_kg"])

        except KeyError as e:
            raise KeyError(f"Missing config key: {e}")
        except ValueError as e:
            raise ValueError(f"Config type error: {e}")

        return self.params


if __name__ == "__main__":
    dir = Path(__file__).resolve().parent.parent
    path = dir / "config" / "config_baseline.yaml"

    parser = ConfigParser(path)
    cfg = parser.parse()
    print("Config parsed successfully.")
