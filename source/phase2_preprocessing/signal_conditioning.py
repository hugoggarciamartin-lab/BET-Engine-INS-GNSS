import sys
import pandas as pd  # type ignore
from pathlib import Path
from scipy.signal import butter, filtfilt

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config.config_parser import ConfigParser


class signalConditioner:
    """Aplies a low-pass Butterworth filter to high-frequency IMU telemetry"""

    def __init__(self, raw_root: Path, output_dir: Path, config_file: Path):
        self.raw_root = raw_root
        self.output_dir = output_dir

        # Convert configuration file yaml into dataframe
        parser = ConfigParser(config_file)
        self.params = parser.parse()

        self.fs = 400
        self.nyq = self.fs / 2

        self.fc = self.params["filter_cutoff_hz"]
        self.order = self.params["filter_order"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _design_filter(self) -> tuple:
        """Butterworth Filter Coeffs"""
        normal_cutoff = self.fc / self.nyq

        if normal_cutoff >= 1.0:
            raise ValueError(
                "Critical error: Cutoff freq must be less than nyquist freq"
            )

        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        return b, a

    def process_imu_data(self) -> None:
        """Load filters and exports the IMU dataset"""
        imu_file = self.raw_root / "flight_data_imu.csv"

        if not imu_file.exists():
            raise FileNotFoundError(f"Error: IMU file missing {imu_file}")

        print(f"Loading raw IMU telemetry from {imu_file}")
        df_imu = pd.read_csv(imu_file)

        b, a = self._design_filter()
        cols = ["f_X", "f_Y", "f_Z", "w_X", "w_Y", "w_Z"]

        # Checkin if doesn't miss any channel
        miss_cols = [col for col in cols if col not in df_imu.columns]
        if miss_cols:
            raise KeyError(
                f"Error: There are missing columns in {imu_file}: {miss_cols} cols"
            )

        print(
            f"Applying Fouth Order Butterworth Filter (Order: {self.order}, fc: {self.fc})"
        )

        for col in cols:
            df_imu[col] = filtfilt(b, a, df_imu[col].values)

        output_file = self.output_dir / "conditioned_flight_data_imu.csv"
        df_imu.to_csv(output_file, index=False)
        print(f"Conditioned IMU data succesfully exported to {output_file.name}")


if __name__ == "__main__":
    raw_path = project_root / "data" / "raw"
    aligned_path = project_root / "data" / "aligned_data"
    config_path = project_root / "config" / "config_baseline.yaml"

    try:
        conditioner = signalConditioner(
            raw_root=raw_path, output_dir=aligned_path, config_file=config_path
        )
        conditioner.process_imu_data()

        print("SIGNAL CONDITIONING REPORT")
        print(f"Filter Type          : Zero-Phase Butterworth")
        print(f"Filter Order         : {conditioner.order}º")
        print(f"Cutoff Frequency     : {conditioner.fc} Hz")
        print(f"Target Nyquist Limit : {conditioner.nyq} Hz")

    except Exception as e:
        print(f"Failure: {e}")
        sys.exit(1)
