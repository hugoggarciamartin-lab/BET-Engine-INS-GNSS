import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_master_telemetry():
    print("Loading Master Flight Data for comprehensive visual inspection...")

    project_root = Path(__file__).resolve().parent.parent
    master_file = project_root / "data" / "aligned_data" / "master_flight_data.csv"

    if not master_file.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Master telemetry not found at {master_file}"
        )

    df = pd.read_csv(master_file)

    if df.isnull().values.any():
        print(
            "WARNING: NaN values detected in master telemetry. Integration will diverge."
        )

    time = df["time"].values

    # Subplots configuration
    plt.style.use("dark_background")
    fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)
    fig.suptitle("Comprehensive Master Telemetry Audit Synchronized", fontsize=16)

    # IMU (Specific Force)
    axes[0].plot(time, df["f_X"], label="f_X", color="cyan", linewidth=1)
    axes[0].plot(time, df["f_Y"], label="f_Y", color="magenta", linewidth=1)
    axes[0].plot(time, df["f_Z"], label="f_Z", color="yellow", linewidth=1)
    axes[0].set_ylabel("Specific Force\n(m/s^2)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # IMU (Gyro Rates)
    axes[1].plot(time, df["w_X"], label="w_X", color="cyan", linewidth=1)
    axes[1].plot(time, df["w_Y"], label="w_Y", color="magenta", linewidth=1)
    axes[1].plot(time, df["w_Z"], label="w_Z", color="yellow", linewidth=1)
    axes[1].set_ylabel("Angular Rate\n(rad/s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # GNSS Position (Dual Axis for Alt vs Lat/Lon)
    axes[2].plot(time, df["gnss_Lat"], label="Lat", color="lightgreen", linewidth=1.5)
    axes[2].plot(time, df["gnss_Lon"], label="Lon", color="lightblue", linewidth=1.5)
    axes[2].set_ylabel("Geodetic\n(rad)")

    ax2_alt = axes[2].twinx()  # Double Y Axis
    ax2_alt.plot(
        time, df["gnss_Alt"], label="Alt (m)", color="white", linewidth=1.5, alpha=0.7
    )
    ax2_alt.set_ylabel("Altitude\n(m)")

    axes[2].legend(loc="upper left")
    ax2_alt.legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # GNSS Veloc
    axes[3].plot(time, df["gnss_v_E"], label="v_E", color="red", linewidth=1.5)
    axes[3].plot(time, df["gnss_v_N"], label="v_N", color="green", linewidth=1.5)
    axes[3].plot(time, df["gnss_v_U"], label="v_U", color="blue", linewidth=1.5)
    axes[3].set_ylabel("GNSS Velocity\n(m/s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    # Magnetometer
    axes[4].plot(time, df["mag_m_X"], label="m_X", color="cyan", linewidth=1.5)
    axes[4].plot(time, df["mag_m_Y"], label="m_Y", color="magenta", linewidth=1.5)
    axes[4].plot(time, df["mag_m_Z"], label="m_Z", color="yellow", linewidth=1.5)
    axes[4].set_ylabel("Mag Field\n(uT)")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)

    # Barometer (Static Pressure)
    axes[5].plot(
        time,
        df["baro_P_static"],
        label="Static Pressure",
        color="orange",
        linewidth=1.5,
    )
    axes[5].set_ylabel("Pressure\n(Pa)")
    axes[5].set_xlabel("Master Time (s)")
    axes[5].legend(loc="upper right")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        plot_master_telemetry()
    except Exception as e:
        print(f"AUDIT FAILED: {e}")
        sys.exit(1)
