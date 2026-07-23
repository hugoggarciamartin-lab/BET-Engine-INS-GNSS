import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from pathlib import Path
import sys


def run_fft_analysis():
    # Path Directory
    project_root = Path(__file__).resolve().parent.parent
    imu_file = project_root / "data" / "raw" / "flight_data_imu.csv"

    if not imu_file.exists():
        raise FileNotFoundError(f"Critical Error: Telemetry not Found at {imu_file}")

    print("Loading flight telemetry analysis")

    df = pd.read_csv(imu_file)
    fs = 400  # Sampling  Frequency
    N = len(df)

    # FFT only in the real half of the spectrums
    freqs = rfftfreq(N, 1 / fs)

    # Analyze specific force in X an gyro rate in Y, but can be done with the other channels
    f_X_fft = np.abs(rfft(df["f_X"].values)) / N
    w_y_fft = np.abs(rfft(df["w_Y"].values)) / N

    print("Generating spectral density plots...")

    time_vector = np.arange(len(df)) / fs

    plt.style.use("dark_background")
    fig1, (ax_t1, ax_t2) = plt.subplots(2, 1, figsize=(12, 6))

    ax_t1.plot(time_vector, df["f_X"], color="cyan", linewidth=0.5)
    ax_t1.set_title("Dominio Temporal: Fuerza Específica X (f_X) - BUSCA EL VUELO")
    ax_t1.set_ylabel("m/s^2")
    ax_t1.grid(True, alpha=0.2)

    ax_t2.plot(time_vector, df["w_Y"], color="magenta", linewidth=0.5)
    ax_t2.set_title("Dominio Temporal: Velocidad Angular Y (w_Y)")
    ax_t2.set_xlabel("Tiempo (s)")
    ax_t2.set_ylabel("rad/s")
    ax_t2.grid(True, alpha=0.2)

    plt.tight_layout()
    # El script se detiene aquí. Analiza la gráfica, anota el inicio y fin del vuelo, y cierra la ventana.
    plt.show()

    # Graphics
    plt.style.use("dark_background")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Accelerometer Plot
    ax1.plot(freqs, f_X_fft, color="cyan", linewidth=0.5)
    ax1.set_title("Frequency Spectrum: Acceleration X-Axis (f_X)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, fs / 2)  # Limited to half of the spectrums
    ax1.set_yscale("log")  # Log scale in order to see low and high energy armonics

    # Gyro Plot
    ax2.plot(freqs, w_y_fft, color="magenta", linewidth=0.5)
    ax2.set_title("Frequency Spectrum: Angular Rate Y-Axis (w_Y)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, fs / 2)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_fft_analysis()
