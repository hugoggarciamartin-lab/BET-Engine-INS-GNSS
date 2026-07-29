import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from source.phase3_nav_eskf_rts.geodesy_math import (
    calc_radii_vec,
    geodetic_to_enu_vec,
    quat2eul_vec,
)


def plot_nominal_kinematics(npz_path: Path):
    """plots nominal states (position, velocity, attitude) with 3-sigma tolerance bands."""
    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]
        p = data["P"]

    time = np.arange(len(x_nom)) * (1.0 / 400.0)
    lat, lon, alt = x_nom[:, 0], x_nom[:, 1], np.abs(x_nom[:, 2])

    # Position kinematics
    rm, rn = calc_radii_vec(lat)

    # project metric covariance back to angular tolerance (eq 9.3 inverted)
    sig_lat_deg = np.rad2deg(np.sqrt(p[:, 0, 0]) / (rm + alt))
    sig_lon_deg = np.rad2deg(np.sqrt(p[:, 1, 1]) / ((rn + alt) * np.cos(lat)))
    sig_alt_m = np.sqrt(p[:, 2, 2])

    pos_vars = [np.rad2deg(lat), np.rad2deg(lon), alt]
    pos_sigs = [sig_lat_deg, sig_lon_deg, sig_alt_m]
    pos_labels = ["latitude (deg)", "longitude (deg)", "altitude (m)"]

    for i in range(3):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time, pos_vars[i], "k-", lw=1.2, label="nominal")
        ax.fill_between(
            time,
            pos_vars[i] - 3 * pos_sigs[i],
            pos_vars[i] + 3 * pos_sigs[i],
            color="r",
            alpha=0.2,
            label="3-sigma",
        )
        ax.set_title(pos_labels[i])
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

    # Velocity kinematics
    vel_labels = ["velocity east (m/s)", "velocity north (m/s)", "velocity up (m/s)"]
    for i in range(3):
        fig, ax = plt.subplots(figsize=(8, 4))
        v = x_nom[:, i + 3]
        sig_v = np.sqrt(p[:, i + 3, i + 3])
        ax.plot(time, v, "k-", lw=1.2, label="nominal")
        ax.fill_between(
            time, v - 3 * sig_v, v + 3 * sig_v, color="r", alpha=0.2, label="3-sigma"
        )
        ax.set_title(vel_labels[i])
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

    # Attitude kinematics (euler)
    roll, pitch, yaw = quat2eul_vec(x_nom[:, 6:10])
    ang_vars = [np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)]
    ang_labels = ["roll (deg)", "pitch (deg)", "yaw (deg)"]

    for i in range(3):
        fig, ax = plt.subplots(figsize=(8, 4))
        sig_ang = np.rad2deg(np.sqrt(p[:, i + 6, i + 6]))
        ax.plot(time, ang_vars[i], "k-", lw=1.2, label="nominal")
        ax.fill_between(
            time,
            ang_vars[i] - 3 * sig_ang,
            ang_vars[i] + 3 * sig_ang,
            color="r",
            alpha=0.2,
            label="3-sigma",
        )
        ax.set_title(ang_labels[i])
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()


def plot_trajectory_3d(npz_path: Path):
    """projects geodetic coordinates to local cartesian frame and plots 3d path."""
    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]

    lat, lon, alt = x_nom[:, 0], x_nom[:, 1], np.abs(x_nom[:, 2])
    e, n, u = geodetic_to_enu_vec(lat, lon, alt)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(e, n, u, color="navy", lw=1.5, label="trajectory (enu)")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_zlabel("altitude relative (m)")
    ax.set_title("3d spatial trajectory")
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    npz_file = project_root / "data" / "results" / "eskf_output_state.npz"
    if npz_file.exists():
        plot_nominal_kinematics(npz_file)
        plot_trajectory_3d(npz_file)
        plt.show()
    else:
        raise FileNotFoundError(f"Error: missing file at path {npz_file}")
