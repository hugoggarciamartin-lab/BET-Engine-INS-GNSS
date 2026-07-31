import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator


def parse_wmm_to_enu_csv(input_path: str, output_path: str) -> None:
    """
    Parses the scattered WMM output and forces the creation of a structured spatial grid.
    Maps the X, Y, Z (NED) components to the ENU navigation frame.
    """
    column_names = [
        "Year",
        "Alt_km",
        "Lat_deg",
        "Lon_deg",
        "Dec_deg",
        "Inc_deg",
        "H_nT",
        "X_nT",
        "Y_nT",
        "Z_nT",
        "F_nT",
        "dD_dt",
        "dI_dt",
        "dH_dt",
        "dX_dt",
        "dY_dt",
        "dZ_dt",
        "dF_dt",
    ]

    try:
        df = pd.read_csv(input_path, sep=r"\s+", comment="#", names=column_names)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Variable extraction
    alt_raw = df["Alt_km"].values
    lat_raw = df["Lat_deg"].values
    lon_raw = df["Lon_deg"].values

    # Strict geometric mapping from NED to ENU
    m_e_raw = df["Y_nT"].values
    m_n_raw = df["X_nT"].values
    m_u_raw = -df["Z_nT"].values

    # Define the axes of a 3D structured grid (10x10x10 = 1000 sample nodes)
    # In production, these vectors must come from a Real NOAA WMM Grid.
    lat_margin = max(0.5, (lat_raw.max() - lat_raw.min()) * 0.1)
    lon_margin = max(0.5, (lon_raw.max() - lon_raw.min()) * 0.1)

    grid_alts = np.linspace(0.0, 95.0, 10)
    grid_lats = np.linspace(
        max(-90.0, lat_raw.min() - lat_margin),
        min(90.0, lat_raw.max() + lat_margin),
        10,
    )
    grid_lons = np.linspace(
        max(-180.0, lon_raw.min() - lon_margin),
        min(180.0, lon_raw.max() + lon_margin),
        10,
    )

    # Cartesian product to create the magnetic volume
    grid_A, grid_Lat, grid_Lon = np.meshgrid(
        grid_alts, grid_lats, grid_lons, indexing="ij"
    )

    # Interpolate the scattered point cloud onto the structured grid
    # NearestNDInterpolator prevents returning NaNs outside the convex hull of the original trajectory
    points = np.vstack((alt_raw, lat_raw, lon_raw)).T

    interp_E = NearestNDInterpolator(points, m_e_raw)
    interp_N = NearestNDInterpolator(points, m_n_raw)
    interp_U = NearestNDInterpolator(points, m_u_raw)

    flat_A = grid_A.flatten()
    flat_Lat = grid_Lat.flatten()
    flat_Lon = grid_Lon.flatten()

    grid_E = interp_E(flat_A, flat_Lat, flat_Lon)
    grid_N = interp_N(flat_A, flat_Lat, flat_Lon)
    grid_U = interp_U(flat_A, flat_Lat, flat_Lon)

    # Construction of the definitive DataFrame compatible with RegularGridInterpolator
    df_grid = pd.DataFrame(
        {
            "Alt_km": flat_A,
            "Lat_deg": flat_Lat,
            "Lon_deg": flat_Lon,
            "m_E_nT": grid_E,
            "m_N_nT": grid_N,
            "m_U_nT": grid_U,
        }
    )

    # Ensure lexical ordering required by SciPy
    df_grid.sort_values(by=["Alt_km", "Lat_deg", "Lon_deg"], inplace=True)

    # Export to the definitive file
    df_grid.to_csv(output_path, index=False)
    print(
        f"Success: Forced magnetic grid generated with {len(df_grid)} structured nodes."
    )


if __name__ == "__main__":
    from pathlib import Path

    # Absolute anchor to the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"

    input_file = raw_dir / "raw_wmm_data.txt"
    output_file = raw_dir / "wmm_mag_tab_data.csv"

    print(f"Searching for input file at: {input_file}")

    # Make sure to run this script independently to overwrite the WMM csv
    parse_wmm_to_enu_csv(str(input_file), str(output_file))
