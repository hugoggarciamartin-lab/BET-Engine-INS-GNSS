import numpy as np
import pandas as pd


def generate_cfd_model():
    print("Generating CFD Aerodynamic lookup table...")
    # Mach number array from 0.0 to 3.0
    mach_array = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    cd_array = np.zeros_like(mach_array)

    # Synthetic transonic drag modeling
    for i, M in enumerate(mach_array):
        if M < 0.8:
            cd_array[i] = 0.45  # Subsonic baseline
        elif 0.8 <= M < 1.1:
            # Transonic drag rise (quadratic interpolation)
            cd_array[i] = 0.45 + 0.35 * ((M - 0.8) / 0.3) ** 2
        elif 1.1 <= M < 1.4:
            # Peak drag transition
            cd_array[i] = 0.80 - 0.15 * ((M - 1.1) / 0.3)
        else:
            # Supersonic decay
            cd_array[i] = 0.65 - 0.10 * ((M - 1.4) / 1.6)

    df = pd.DataFrame({"Mach": mach_array, "C_D_CFD": cd_array})

    output_filename = "cfd_drag_model.csv"
    df.to_csv(output_filename, index=False, float_format="%.4f")
    print(f"CFD table successfully written to {output_filename}")


if __name__ == "__main__":
    generate_cfd_model()
