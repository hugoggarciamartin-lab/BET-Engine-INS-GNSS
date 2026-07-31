"""Plots each innovation residual state from de EKSF z_k results, with its own theoric envelope,
calculated from innnovation covariance matrix S_k (+/- 3 *sigma).

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def analyze_innovation_residuals():
    """plots the innovation vector zk and its 3-sigma theoretical envelope."""
    plt.style.use("default")

    npz_path = project_root / "data" / "results" / "eskf_output_state.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        z = data["z"]
        sk_diag = data["Sk_diag"]

    time = np.arange(len(z)) * (1.0 / 400.0)

    labels_z = [
        "gnss pos east (m)",
        "gnss pos north (m)",
        "gnss pos up (m)",
        "gnss vel east (m/s)",
        "gnss vel north (m/s)",
        "gnss vel up (m/s)",
        "baro altitude (m)",
        "mag body x (ut)",
        "mag body y (ut)",
        "mag body z (ut)",
    ]

    print("\n--- innovation vector statistical audit ---")

    for i in range(10):
        # Aislamiento estricto: Ignorar épocas ciegas donde R_k fue inflado sumando 1e9
        valid_idx = sk_diag[:, i] < 1e6

        if np.any(valid_idx):
            z_valid = z[valid_idx, i]
            t_valid = time[valid_idx]

            mean_val = np.mean(z_valid)
            print(f"channel {i} ({labels_z[i]}): temporal mean = {mean_val:.6e}")

            sig_s_valid = np.sqrt(sk_diag[valid_idx, i])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                t_valid,
                z_valid,
                "k.",
                markersize=2,
                alpha=0.8,
                label="residual zk",
            )
            ax.plot(t_valid, 3 * sig_s_valid, "r--", lw=1, label="+3 sigma envelope")
            ax.plot(t_valid, -3 * sig_s_valid, "r--", lw=1, label="-3 sigma envelope")

            # Constrain y-axis to theoretical limits - nanmax ommits nan values from maximum value extraction process
            limit = np.nanmax(3 * sig_s_valid) * 1.5
            ax.set_ylim(-limit, limit)

            ax.set_title(f"innovation: {labels_z[i]}")
            ax.set_ylabel("error")
            ax.set_xlabel("time (s)")
            ax.grid(True, color="gray", linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize="small")
            fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    analyze_innovation_residuals()
