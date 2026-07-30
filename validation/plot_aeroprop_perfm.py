"""Aeropropulsive validation script. Reconstructs dynamic engine thrust
and audits empirical drag coefficient by extracting RTS smooth data"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

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
    f_raw_x = master_df["f_X"].values

    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]

    # forensic extraction of real specific force (eq 9.4)
    bias_a_x = x_nom[:, 10]
    f_real_x = f_raw_x - bias_a_x

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

    # eq 9.5: reconstruct thrust (assuming thrust and drag are collinear to x-axis)
    thrust_rec = (mass_arr * f_real_x) + drag_sim

    fig, ax = plt.subplots(figsize=(8, 4))
    idx_burn = time_arr <= cfg.get("time_eng_off", 5.0)

    ax.plot(
        time_arr[idx_burn],
        thrust_rec[idx_burn],
        "k-",
        lw=1.5,
        label="reconstructed thrust",
    )
    ax.set_title("engine thrust forensic reconstruction")
    ax.set_ylabel("thrust (n)")
    ax.set_xlabel("time (s)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()


def plot_reconstructed_drag(
    npz_path: Path, master_csv: Path, mass_csv: Path, cfg: dict
):
    """audits empirical drag coefficient during ballistic coast phase (eq 9.6)."""
    master_df = pd.read_csv(master_csv)
    time_arr = master_df["time"].values
    f_raw_x = master_df["f_X"].values

    with np.load(npz_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]

    bias_a_x = x_nom[:, 10]
    f_real_x = f_raw_x - bias_a_x
    mass_arr = interp_mass_profile(time_arr, mass_csv)

    alt = x_nom[:, 2]
    v_n_vec = x_nom[:, 3:6]
    v_tas = np.linalg.norm(v_n_vec, axis=1)

    idx_eng_off = time_arr > cfg.get("time_eng_off", 5.0)

    cd_rec = np.zeros(np.sum(idx_eng_off))
    mach_coast = np.zeros(np.sum(idx_eng_off))

    c = 0
    for i in range(len(alt)):
        if time_arr[i] > cfg.get("time_eng_off", 5.0):
            t_loc, p_loc = calc_isa_atmosphere(
                alt[i], cfg["p0_isa"], cfg["t0_isa"], cfg["l_isa"], cfg["g_e"]
            )
            rho = p_loc / (287.0528 * t_loc)
            mach = calc_mach_number(
                v_n_vec[i],
                alt[i],
                cfg["p0_isa"],
                cfg["t0_isa"],
                cfg["l_isa"],
                cfg["g_e"],
            )

            # Invert drag equation
            denom = 0.5 * rho * (v_tas[i] ** 2) * cfg["s_ref_m2"]
            if denom > 1e-3:
                cd_rec[c] = (-mass_arr[i] * f_real_x[i]) / denom
            else:
                cd_rec[c] = np.nan

            mach_coast[c] = mach
            c += 1

    time_coast = time_arr[idx_eng_off]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_coast, cd_rec, "k.", markersize=2, label="empirical drag coefficient")
    ax.set_title("aerodynamic validation: reconstructed cd vs time")
    ax.set_ylabel("cd")
    ax.set_xlabel("time (s)")
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
