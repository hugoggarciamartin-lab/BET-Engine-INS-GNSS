"""Aeropropulsive validation script. Reconstructs dynamic engine thrust
and audits empirical drag coefficient by extracting RTS smooth data"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# Grant access to repo paths to export
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Grant access to import BET Engine Libraries
phase3_dir = project_root / "source" / "phase3_nav_eskf_rts"
sys.path.append(str(phase3_dir))

import source.phase3_nav_eskf_rts.geodesy_math as mat
from config.config_parser import ConfigParser
from source.phase3_nav_eskf_rts.kinematics_ins import (
    calc_isa_atmosphere,
    calc_mach_number,
)


def interp_cfd_drag(mach_query: np.ndarray, cfd_path: Path) -> np.ndarray:
    """1d interpolation of theoretical aerodynamic drag coefficient vs mach."""
    df = pd.read_csv(cfd_path)
    interpolator = interp1d(
        df["Mach"].values, df["C_D_CFD"].values, kind="linear", fill_value="extrapolate"
    )
    return interpolator(mach_query)


def interp_mass_profile(time_query: np.ndarray, mass_path: Path) -> np.ndarray:
    """1d interpolation of dynamic vehicle mass over flight time."""
    df = pd.read_csv(mass_path)
    interpolator = interp1d(
        df["time"].values, df["mass_kg"].values, kind="linear", fill_value="extrapolate"
    )
    return interpolator(time_query)


def plot_reconstructed_thrust(
    npz_path: Path, master_csv: Path, mass_csv: Path, cfd_csv: Path, cfg: dict
):
    """reconstructs and plots theoretical vs actual engine thrust profile (eq 9.5)."""
    master_df = pd.read_csv(master_csv)
    time_arr = master_df["time"].values

    # CORRECTED: Extract longitudinal Z-axis specific force
    f_raw_z = master_df["f_Z"].values

    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]

    # CORRECTED: Extract Z-axis accelerometer bias (Index 11)
    bias_a_z = x_nom[:, 11]
    f_real_z = f_raw_z - bias_a_z

    mass_arr = interp_mass_profile(time_arr, mass_csv)

    alt = x_nom[:, 2]
    v_n_vec = x_nom[:, 3:6]
    v_tas = np.linalg.norm(v_n_vec, axis=1)

    # atmospheric calculations for aerodynamic drag isolation
    rho_arr = np.zeros_like(alt)
    mach_arr = np.zeros_like(alt)

    for i in range(len(alt)):
        t_loc, p_loc = calc_isa_atmosphere(
            alt[i], cfg["p0_isa"], cfg["t0_isa"], cfg["l_isa"], cfg["g_e"]
        )
        rho_arr[i] = p_loc / (287.0528 * t_loc)
        mach_arr[i] = calc_mach_number(
            v_n_vec[i], alt[i], cfg["p0_isa"], cfg["t0_isa"], cfg["l_isa"], cfg["g_e"]
        )

    cd_sim = interp_cfd_drag(mach_arr, cfd_csv)
    q_dyn = 0.5 * rho_arr * (v_tas**2)
    drag_sim = q_dyn * cfg["s_ref_m2"] * cd_sim

    thrust_rec_noisy = (mass_arr * f_real_z) + drag_sim

    fs = 400.0
    cutoff_hz = 2.0
    b, a = butter(4, cutoff_hz / (fs / 2.0), btype="low")
    thrust_rec_clean = filtfilt(b, a, thrust_rec_noisy)

    fig, ax = plt.subplots(figsize=(10, 5))
    idx_burn = time_arr <= cfg.get("time_eng_off", 5.0)

    ax.plot(
        time_arr[idx_burn],
        thrust_rec_noisy[idx_burn],
        color="gray",
        lw=0.5,
        alpha=0.5,
        label="raw reconstructed thrust (vibration)",
    )
    ax.plot(
        time_arr[idx_burn],
        thrust_rec_clean[idx_burn],
        "k-",
        lw=1.5,
        label="filtered thrust (macroscopic)",
    )

    ax.set_title("Engine Thrust Forensic Reconstruction (Z-Axis)")
    ax.set_ylabel("Thrust (N)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()


def plot_reconstructed_drag(
    npz_path: Path, master_csv: Path, mass_csv: Path, cfg: dict
):
    """audits empirical drag coefficient during ballistic coast phase (eq 9.6)."""
    master_df = pd.read_csv(master_csv)
    time_arr = master_df["time"].values

    # Extracción del eje longitudinal (Z)
    f_raw_z = master_df["f_Z"].values

    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]

    bias_a_z = x_nom[:, 11]
    f_real_z = f_raw_z - bias_a_z
    mass_arr = interp_mass_profile(time_arr, mass_csv)

    alt = x_nom[:, 2]
    v_n_vec = x_nom[:, 3:6]
    v_tas = np.linalg.norm(v_n_vec, axis=1)

    # Aislar estrictamente la fase balística ANTES de filtrar
    idx_eng_off = time_arr > cfg.get("time_eng_off", 5.0)

    time_coast = time_arr[idx_eng_off]
    alt_coast = alt[idx_eng_off]
    v_tas_coast = v_tas[idx_eng_off]
    v_n_coast = v_n_vec[idx_eng_off]
    mass_coast = mass_arr[idx_eng_off]
    f_real_z_coast = f_real_z[idx_eng_off]

    # Filtro Butterworth suave de fase cero solo para el ruido del sensor en caída libre
    from scipy.signal import butter, filtfilt

    fs = 400.0
    b, a = butter(2, 2.0 / (fs / 2.0), btype="low")
    if len(f_real_z_coast) > 15:
        f_real_z_coast = filtfilt(b, a, f_real_z_coast)

    cd_rec = np.zeros(len(time_coast))

    for c in range(len(time_coast)):
        t_loc, p_loc = calc_isa_atmosphere(
            alt_coast[c], cfg["p0_isa"], cfg["t0_isa"], cfg["l_isa"], cfg["g_e"]
        )
        rho = p_loc / (287.0528 * t_loc)

        # Presión dinámica
        denom = 0.5 * rho * (v_tas_coast[c] ** 2) * cfg["s_ref_m2"]

        # Protección anti-singularidad: evitar divisiones por cero en el apogeo (V ~ 0)
        if denom > 10.0:
            # El arrastre es la magnitud absoluta de la fuerza específica por la masa
            cd_rec[c] = (mass_coast[c] * np.abs(f_real_z_coast[c])) / denom
        else:
            cd_rec[c] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_coast, cd_rec, "k.", markersize=2, label="empirical drag coefficient")

    # Límite dinámico para visualizar el modelo teórico irreal sin aplastar la gráfica
    valid_cd = cd_rec[~np.isnan(cd_rec)]
    if len(valid_cd) > 0:
        ax.set_ylim(0.0, np.percentile(valid_cd, 95) * 1.2)

    ax.set_title("Aerodynamic Validation: Reconstructed Cd vs Time (Z-Axis)")
    ax.set_ylabel("Cd")
    ax.set_xlabel("Time (s)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    cfg_parser = ConfigParser(project_root / "config" / "config_baseline.yaml")
    params = cfg_parser.parse()

    npz_f = project_root / "data" / "results" / "eskf_output_state.npz"
    master_f = project_root / "data" / "aligned_data" / "master_flight_data.csv"
    mass_f = project_root / "data" / "raw" / "flight_mass_prof_data.csv"
    cfd_f = project_root / "data" / "raw" / "cfd_drag_model.csv"

    if npz_f.exists() and master_f.exists() and mass_f.exists() and cfd_f.exists():
        plot_reconstructed_thrust(npz_f, master_f, mass_f, cfd_f, params)
        plot_reconstructed_drag(npz_f, master_f, mass_f, params)
        plt.show()
