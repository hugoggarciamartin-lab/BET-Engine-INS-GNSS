"""Flight mass evolution profile generator in time domain"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from config.config_parser import ConfigParser

project_root = Path(__file__).resolve().parent.parent.parent


def generate_mass_profile():
    config_path = project_root / "config" / "config_baseline.yaml"
    parser = ConfigParser(config_path)
    cfg = parser.parse()

    duration = 120.0
    f_mass = 10.0

    t_mass = np.arange(0.0, duration, 1.0 / f_mass, dtype=np.float64)
    m_curve = np.zeros(len(t_mass), dtype=np.float64)

    m_0 = cfg["m_0"]
    m_f = cfg["m_f"]
    burn_start = 5.0
    burn_end = cfg["time_eng_off"]

    mdot = (m_0 - m_f) / (burn_end - burn_start)

    for i, t in enumerate(t_mass):
        if t < burn_start:
            m_curve[i] = m_0
        elif burn_start <= t < burn_end:
            m_curve[i] = m_0 - mdot * (t - burn_start)
        else:
            m_curve[i] = m_f

    np.random.seed(303)
    sensor_noise = np.random.normal(0, 0.5, len(t_mass))
    m_noisy = m_curve + sensor_noise

    df = pd.DataFrame({"time": t_mass, "mass_kg": m_noisy})

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_filename = raw_dir / "flight_mass_prof_data.csv"
    df.to_csv(output_filename, index=False, float_format="%.4f")


if __name__ == "__main__":
    generate_mass_profile()
