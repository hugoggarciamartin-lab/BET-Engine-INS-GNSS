import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# Add to the path the variable "project_root" as a string


class temporalAligner:
    """Sychronizes multi-rate telemetry onto a single hig-freq master time scale
    Uses linear interpolation to prevent spline-induced overshoots
    (Runge Phenomenon)"""

    def __init__(self, condit_dir: Path, rawext_dir: Path):
        self.condit_dir = condit_dir
        self.rawext_dir = rawext_dir
        self.fs_master = 400.0  # Hz (IMU telemetry rate)

    def _load_valid(self, filepath: Path) -> pd.DataFrame:
        if not filepath.exists():
            raise FileNotFoundError(f"Error: missing dataset at {filepath}")

        df = pd.read_csv(filepath)
        if "time" not in df.columns:
            print(
                f"Warning: there is no 'time' column in {filepath}. Assuming uniform sampling"
            )
            df['time'] = np.arange(len(df)) /self._guess_freq(filepath.name)
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

        imu_file = self.condit_dir / "conditioned_data_flight_data_imu.csv"
        df_imu = 
