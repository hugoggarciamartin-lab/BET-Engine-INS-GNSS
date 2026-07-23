import sys
import pandas as pd
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

        parser = ConfigParser(config_file)
        self.fs = 400
        self.nyq = self.fs / 2
