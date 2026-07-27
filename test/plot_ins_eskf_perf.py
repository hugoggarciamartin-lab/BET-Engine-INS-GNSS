import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_navigation_states() -> None:
    # Strict relative path to the repository topology
    base_dir = Path(__file__).resolve().parent.parent
    npz_path = base_dir / "data" / "aligned_data" / "eskf_output_state.npz"

    source_dir = base_dir / "source" / "phase3_navigation"
    sys.path.append(str(source_dir))
    from geodesy_math import quat2eul

    if not npz_path.exists():
        raise FileNotFoundError(f"Critical Error: Data file not found at {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]
        P = data["P"]
        Sk = data["Sk_diag"] if "Sk_diag" in data else None

    # 10 nominal states covariance envolopes
    for i in range(10):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Nominal state curve
        ax.plot(
            x_nom[:, i], label=f"Nominal State x[{i}]", color="black", linewidth=1.5
        )

        # State Covariance Envelope (P) - Centered on nominal
        sigma_p = np.sqrt(P[:, i, i])
        ax.plot(x_nom[:, i] + 3 * sigma_p, "r--", label=r"$+3\sigma$ ($P$)")
        ax.plot(x_nom[:, i] - 3 * sigma_p, "r--", label=r"$-3\sigma$ ($P$)")
        ax.fill_between(
            range(len(x_nom)),
            x_nom[:, i] + 3 * sigma_p,
            x_nom[:, i] - 3 * sigma_p,
            color="red",
            alpha=0.1,
        )

        # Innovation Covariance Envelope (S_k) - centered on nominal state
        if Sk is not None and i < Sk.shape[1]:
            sigma_s = np.sqrt(Sk[:, i])
            ax.plot(x_nom[:, i] + 3 * sigma_s, "b:", label=r"$+3\sigma$ ($S_k$)")
            ax.plot(x_nom[:, i] - 3 * sigma_s, "b:", label=r"$-3\sigma$ ($S_k$)")
            ax.fill_between(
                range(len(x_nom)),
                x_nom[:, i] + 3 * sigma_s,
                x_nom[:, i] - 3 * sigma_s,
                color="blue",
                alpha=0.05,
            )

        ax.set_title(f"Nominal State Audit {i} (3σ Bounds)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Operational Magnitude")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right")
        fig.tight_layout()

    # 3D Lineal Kinematics
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(
        x_nom[:, 0],
        x_nom[:, 1],
        x_nom[:, 2],
        label="Nominal INS Trajectory",
        color="midnightblue",
        linewidth=2,
    )

    ax3d.set_xlabel("Position X [m]")
    ax3d.set_ylabel("Position Y [m]")
    ax3d.set_zlabel("Position Z [m]")
    ax3d.set_title("Spatial Validation: 3D Trajectory")
    ax3d.legend()

    # Euler Angles Conversion and Plotting
    euler_angles_deg = np.zeros((len(x_nom), 3))
    for k in range(len(x_nom)):
        euler_rad = quat2eul(x_nom[k, 6:10])
        euler_angles_deg[k, :] = np.rad2deg(euler_rad)

    fig_euler, axs_euler = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs_euler[0].plot(euler_angles_deg[:, 0], color="darkorange", linewidth=1.5)
    axs_euler[0].set_title("Attitude Kinematics: Roll (\u03d5)")
    axs_euler[0].set_ylabel("Degrees (\u00b0)")
    axs_euler[0].grid(True, linestyle=":", alpha=0.7)

    axs_euler[1].plot(euler_angles_deg[:, 1], color="forestgreen", linewidth=1.5)
    axs_euler[1].set_title("Attitude Kinematics: Pitch (\u03b8)")
    axs_euler[1].set_ylabel("Degrees (\u00b0)")
    axs_euler[1].grid(True, linestyle=":", alpha=0.7)

    axs_euler[2].plot(euler_angles_deg[:, 2], color="purple", linewidth=1.5)
    axs_euler[2].set_title("Attitude Kinematics: Yaw (\u03c8)")
    axs_euler[2].set_xlabel("Epochs")
    axs_euler[2].set_ylabel("Degrees (\u00b0)")
    axs_euler[2].grid(True, linestyle=":", alpha=0.7)

    fig_euler.tight_layout()

    # --- 4. CONSOLE TRACEABILITY ---
    print(" NOMINAL STATE VECTOR MAP (x_nom) - 16 STATES")
    print("Translational Kinematics:")
    print("  x_nom[:, 0] : Position X / Latitude")
    print("  x_nom[:, 1] : Position Y / Longitude")
    print("  x_nom[:, 2] : Position Z / Altitude")
    print("  x_nom[:, 3] : Velocity V_x / V_N (m/s)")
    print("  x_nom[:, 4] : Velocity V_y / V_E (m/s)")
    print("  x_nom[:, 5] : Velocity V_z / V_D (m/s)")
    print("\nAttitude Kinematics: Quaternion (S^3):")
    print("  x_nom[:, 6] : Quaternion q0 (Scalar part)")
    print("  x_nom[:, 7] : Quaternion q1 (Vector part i)")
    print("  x_nom[:, 8] : Quaternion q2 (Vector part j)")
    print("  x_nom[:, 9] : Quaternion q3 (Vector part k)")
    print("\nInertial Calibration States:")
    print("  x_nom[:, 10:13] : Accelerometer Biases b_a (m/s^2)")
    print("  x_nom[:, 13:16] : Gyroscope Biases b_g (rad/s)")

    plt.show()


if __name__ == "__main__":
    plot_navigation_states()
