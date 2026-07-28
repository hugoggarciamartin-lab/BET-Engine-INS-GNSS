import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def analyze_innovation_residuals():
    """plots the innovation vector zk and its 3-sigma theoretical envelope."""
    plt.style.use("dark_background")

    npz_path = project_root / "data" / "results" / "eskf_output_state.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        z = data["z"]
        sk_diag = data["Sk_diag"]

    t = np.arange(len(z))

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
        # isolate epochs where measurements were actually ingested (ignore zoh holds)
        valid_idx = np.abs(z[:, i]) > 1e-9

        if np.any(valid_idx):
            z_valid = z[valid_idx, i]
            mean_val = np.mean(z_valid)
            print(f"channel {i} ({labels_z[i]}): temporal mean = {mean_val:.6e}")

            sig_s = np.sqrt(sk_diag[:, i])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                t[valid_idx],
                z_valid,
                "w.",
                markersize=2,
                alpha=0.8,
                label="residual zk",
            )
            ax.plot(t, 3 * sig_s, "r--", lw=1, label="+3 sigma envelope")
            ax.plot(t, -3 * sig_s, "r--", lw=1, label="-3 sigma envelope")

            # constrain y-axis to theoretical limits, ignoring R_k inflation spikes
            limit = np.nanmax(3 * sig_s[valid_idx]) * 1.5
            ax.set_ylim(-limit, limit)

            ax.set_title(f"innovation: {labels_z[i]}")
            ax.set_ylabel("error")
            ax.set_xlabel("epoch")
            ax.grid(True, color="gray", linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize="small")
            fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    analyze_innovation_residuals()
