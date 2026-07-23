import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def verify_conditioning():
    """
    Plots the raw vs. conditioned telemetry for visual validation of the
    Butterworth filter's attenuation and zero-phase characteristics.
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / "data" / "raw" / "flight_data_imu.csv"
    cond_file = (
        project_root / "data" / "aligned_data" / "conditioned_flight_data_imu.csv"
    )

    if not raw_file.exists() or not cond_file.exists():
        raise FileNotFoundError(
            "CRITICAL ERROR: Datasets missing. Execute signal_conditioning.py first."
        )

    print("Loading datasets for visual verification...")
    df_raw = pd.read_csv(raw_file)
    df_cond = pd.read_csv(cond_file)

    # Reconstruct the time vector based on the 400 Hz sampling rate
    fs = 400.0
    time_vector = df_raw.index / fs

    print("Generating overlay plots...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Specific Force (X-Axis)
    axes[0].plot(
        time_vector,
        df_raw["f_X"],
        color="cyan",
        alpha=0.4,
        label="Raw f_X (Noise)",
        linewidth=1,
    )
    axes[0].plot(
        time_vector,
        df_cond["f_X"],
        color="yellow",
        label="Conditioned f_X",
        linewidth=1.5,
    )
    axes[0].set_title("Filter Verification: Specific Force X-Axis")
    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Plot 2: Angular Rate (Y-Axis)
    axes[1].plot(
        time_vector,
        df_raw["w_Y"],
        color="magenta",
        alpha=0.4,
        label="Raw w_Y (Noise)",
        linewidth=1,
    )
    axes[1].plot(
        time_vector,
        df_cond["w_Y"],
        color="lime",
        label="Conditioned w_Y",
        linewidth=1.5,
    )
    axes[1].set_title("Filter Verification: Angular Rate Y-Axis")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular Rate (rad/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        verify_conditioning()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
