import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path


class AllanVarianceExt:
    """Ingest static IMU telemetry and computes Allan Variance to
    extract noise parameters: ARW, VRW, Bias Instability, Correlation Times"""

    def __init__(self, filepath: str, fs: float):
        self.filepath = filepath
        self.fs = fs
        self.dt = 1.0 / fs
        self.data = None

    def load_data(self) -> None:
        """Load the csv static Allan Variance dataset"""

        print(f"Loading telemetry : {self.filepath}")

        self.data = pd.read_csv(
            self.filepath, usecols=["f_X", "f_Y", "f_Z", "w_X", "w_Y", "w_Z"]
        )
        print(f"Loaded {len(self.data)} epochs at {self.fs} Hz")

    def _compute_allan(self, series: np.ndarray, taus: np.ndarray) -> np.ndarray:
        """Compute vectorized Allan deviation"""
        # Time integration of the acceleration and gyro rate
        theta = np.cumsum(series) * self.dt
        N = len(theta)
        deviat = np.zeros(len(taus), dtype=np.float64)

        for i, tau in enumerate(taus):
            m = max(1, int(round(tau / self.dt)))

            if N - 2 * m > 0:
                diff = theta[2 * m :] - 2 * theta[m:-m] + theta[: -2 * m]
                varian = np.mean(diff**2) / (2 * (tau) ** 2)
                deviat[i] = np.sqrt(varian)
            else:
                deviat[i] = np.nan
        return deviat

    def sensor_analysis(self, channel: str) -> Tuple[float, float, float]:
        """Extract ARW, VRW (White Noise) and Bias Instability (Flicker Noise)"""

        series = self.data[channel].values

        # Logarithmic spacing for tau in order to cover high frquency thermal drift
        taus = np.logspace(
            np.log10(self.dt),
            np.log10(
                len(series) * self.dt / 3.0
            ),  # There is a limit in the value of len(tau)
            100,
        )
        deviat = self._compute_allan(series, taus)

        # Eliminate NaN for safety
        valid = ~np.isnan(deviat)
        taus = taus[valid]
        deviat = deviat[valid]

        # ARW and VRW: Found at tau = 1.0 second (where slope is -0.5)
        random_walk = np.interp(1.0, taus, deviat)

        # Bias Instability: Found at the minimum of the curve (slope is 0.0)
        min_id = np.argmin(deviat)
        bias_instab = deviat[min_id] / 0.664  # Empirical conversion factor

        # Correlation Time
        corr_time = taus[min_id]

        return random_walk, bias_instab, corr_time

    def exe_extract(self) -> Dict[str, Dict[str, float]]:
        """Runs the extraction of the all 6 DOF IMU and returns de parameters mentioned"""
        if self.data is None:
            self.load_data()
        results = {}
        channels = ["f_X", "f_Y", "f_Z", "w_X", "w_Y", "w_Z"]

        print(
            f"Commencing Allan Variance extraction. This requires heavy CPU vectorization..."
        )
        for ch in channels:
            rw, bi, tc = self.sensor_analysis(ch)
            results[ch] = {
                "Random_Walk": rw,
                "Bias_Instability": bi,
                "Correlation_Time": tc,
            }
        return results


# Test Execution
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    # Path matching structure: data/raw/
    csv_path = project_root / "data" / "raw" / "lab_allan_variance_data.csv"

    if csv_path.exists():
        extractor = AllanVarianceExt(filepath=str(csv_path), fs=400.0)
        allan_parameters = extractor.exe_extract()

        # Formatted, audit-compliant output
        print("ALLAN VARIANCE PARAMETER EXTRACTION REPORT")
        for channel, params in allan_parameters.items():
            print(f"[{channel}]")
            print(f"  -> Random Walk (White Noise)  : {params['Random_Walk']:.6e}")
            print(f"  -> Bias Instability  : {params['Bias_Instability']:.6e}")
            print(
                f"  -> Correlation Time (tau)     : {params['Correlation_Time']:.3f} s"
            )
    else:
        raise FileNotFoundError(
            f"CRITICAL ERROR: Telemetry file not found at evaluated path: {csv_path}"
        )
