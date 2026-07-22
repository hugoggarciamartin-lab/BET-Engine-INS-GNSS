import numpy as np
import pandas as pd
import gc


def generate_mass_profile():
    print("Generating vehicle mass profile telemetry...")

    # TEMPORAL ALLOCATION
    duration = 120.0
    f_mass = 10.0

    gc.disable()

    t_mass = np.arange(0.0, duration, 1.0 / f_mass, dtype=np.float64)
    m_curve = np.zeros(len(t_mass), dtype=np.float64)

    # MASS KINEMATICS
    m_0 = 250.0
    m_f = 100.0
    burn_start = 5.0
    burn_end = 25.0

    # Linear mass flow rate (kg/s) assuming constant thrust profile
    mdot = (m_0 - m_f) / (burn_end - burn_start)

    for i, t in enumerate(t_mass):
        if t < burn_start:
            m_curve[i] = m_0
        elif burn_start <= t < burn_end:
            m_curve[i] = m_0 - mdot * (t - burn_start)
        else:
            m_curve[i] = m_f

    # NOISE INJECTION
    # Simulating measurement uncertainty from a propellant level sensor
    np.random.seed(303)  # Strict deterministic seed
    sensor_noise = np.random.normal(0, 0.5, len(t_mass))
    m_noisy = m_curve + sensor_noise

    # EXPORT
    df = pd.DataFrame({"t_mass": t_mass, "mass_kg": m_noisy})

    output_filename = "flight_mass_prof_data.csv"
    df.to_csv(output_filename, index=False, float_format="%.4f")
    print(f"Mass profile successfully written to {output_filename}")

    gc.enable()


if __name__ == "__main__":
    generate_mass_profile()
