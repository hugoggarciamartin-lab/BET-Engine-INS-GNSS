import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

project_root = Path(__file__).resolve().parent.parent.parent
# Add the variable "project_root" to the path as a string
sys.path.append(str(project_root))


class temporalAligner:
    """Sychronizes multi-rate telemetry onto a single hig-freq master time scale
    Uses linear interpolation to prevent spline-induced overshoots
    (Runge Phenomenon)"""

    def __init__(self, aligned_dir: Path, raw_dir: Path):
        self.aligned_dir = aligned_dir
        self.raw_dir = raw_dir
        self.fs_master = 400.0  # Hz (IMU telemetry rate)

    def _load_valid(self, filepath: Path) -> pd.DataFrame:
        if not filepath.exists():
            raise FileNotFoundError(f"Error: missing dataset at {filepath}")

        df = pd.read_csv(filepath)
        if "time" not in df.columns:
            print(
                f"Warning: there is no 'time' column in {filepath}. Assuming uniform sampling"
            )
            df["time"] = np.arange(len(df)) / self._guess_freq(filepath.name)
        return df

    def _guess_freq(self, filename: str) -> float:
        """Guess frequency if time columns are missing"""
        if "imu" in filename:
            return 400.0
        if "gnss" in filename:
            return 10.0
        if "mag" in filename:
            return 50.0
        if "baro" in filename:
            return 20.0
        return 1.0

    def align_telem(self) -> None:
        print("Initializaing Temporal Aligner")

        # CORREGIDO: Ruta y nombre de archivo exactos
        imu_file = self.aligned_dir / "conditioned_flight_data_imu.csv"

        df_imu = self._load_valid(imu_file)
        df_gnss = self._load_valid(self.raw_dir / "flight_data_gnss.csv")
        df_mag = self._load_valid(self.raw_dir / "flight_data_mag.csv")
        df_baro = self._load_valid(self.raw_dir / "flight_data_baro.csv")

        t_imu = df_imu["time"].values

        t_start = max(
            t_imu[0],
            df_gnss["time"].iloc[0],
            df_mag["time"].iloc[0],
            df_baro["time"].iloc[0],
        )

        t_end = min(
            t_imu[-1],
            df_gnss["time"].iloc[-1],
            df_mag["time"].iloc[-1],
            df_baro["time"].iloc[-1],
        )
        print(f"Master Time grid bounded: [{t_start: .3f} s, {t_end: .3f} s]")

        # Filter master time to valid window
        valid_id = (t_imu >= t_start) & (t_imu <= t_end)
        t_master = t_imu[valid_id]

        df_master = pd.DataFrame({"time": t_master})

        # Write the interpolated data from IMU channels
        for col in df_imu.columns:
            if col != "time":
                df_master[col] = df_imu[col].values[valid_id]

        def _interpolate_sensor(df_sensor: pd.DataFrame, prefix: str):
            for col in df_sensor.columns:
                if col != "time":
                    # Linear interpolation for external reference measures
                    interpol = interp1d(
                        df_sensor["time"].values,
                        df_sensor[col].values,
                        kind="linear",
                        bounds_error=True,
                    )
                    df_master[f"{prefix}_{col}"] = interpol(t_master)

        print("Applying linear interpolation to low-freq channels...")
        _interpolate_sensor(df_gnss, "gnss")
        _interpolate_sensor(df_mag, "mag")
        _interpolate_sensor(df_baro, "baro")

        # Export Master Data Array
        out_file = self.aligned_dir / "master_flight_data.csv"
        df_master.to_csv(out_file, index=False)

        print("\n TEMPORAL ALIGNMENT REPORT")
        print(f"Master Frequency     : {self.fs_master} Hz")
        print(f"Total Aligned        : {len(df_master)}")
        print(f"Total Duration       : {t_end - t_start:.2f} secs")
        print(f"Output File Name     : {out_file.name}")


if __name__ == "__main__":
    aligned_path = project_root / "data" / "aligned_data"
    raw_path = project_root / "data" / "raw"

    try:
        aligner = temporalAligner(aligned_dir=aligned_path, raw_dir=raw_path)
        aligner.align_telem()
    except Exception as e:
        print(f"Failure: {e}")
        sys.exit(1)
