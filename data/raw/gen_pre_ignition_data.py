import numpy as np
import pandas as pd
import gc

def generate_pad_telemetry():
    # TEMPORAL ALLOCATION (ASYNCHRONOUS GRIDS)
    duration = 60.0  # seconds

    # Freq from different Sensors
    f_imu = 400.0
    f_gnss = 10.0
    f_baro = 20.0
    f_mag = 20.0
    
    gc.disable()
    
    # Generate asynchronous time arrays (simulating slight hardware start delays)
    t_imu = np.arange(0.0, duration, 1.0/f_imu, dtype=np.float64)
    t_gnss = np.arange(0.1, duration, 1.0/f_gnss, dtype=np.float64)
    t_baro = np.arange(0.05, duration, 1.0/f_baro, dtype=np.float64)
    t_mag = np.arange(0.02, duration, 1.0/f_mag, dtype=np.float64)
    
    # GEODETIC & ENVIRONMENTAL BASELINE
    phi_0 = np.deg2rad(39.4811)
    lam_0 = np.deg2rad(-0.3444)
    h_0 = 15.0
    
    a = 6378137.0
    e2 = 0.00669437999
    g_e = 9.7803253359
    k_el = 0.001931852652
    Omega_e = 7.292115e-5
    
    P0_isa = 101325.0
    T0_isa = 288.15
    L_isa = 0.0065
    R_air = 287.0528
    
    g_0 = g_e * (1 + k_el * np.sin(phi_0)**2) / np.sqrt(1 - e2 * np.sin(phi_0)**2)
    g_local = g_0 * (1 - (2 * h_0) / a)
    
    np.random.seed(101) # DO-330 deterministic seed
    
    # IMU GENERATION (400 Hz)
    # Adding platform interference (noise floor > pristine lab noise)
    noise_floor_accel = 0.05 # m/s^2 (Higher than VRW due to vehicle vibrations)
    noise_floor_gyro = 0.01  # rad/s
    
    # Static turn-on biases
    b_a0 = np.array([0.015, -0.012, 0.018], dtype=np.float64)
    b_g0 = np.array([0.005, -0.008, 0.004], dtype=np.float64)
    
    f_nominal = np.array([0.0, 0.0, g_local], dtype=np.float64)
    w_nominal = np.array([0.0, Omega_e * np.cos(phi_0), Omega_e * np.sin(phi_0)], dtype=np.float64)
    
    f_raw = f_nominal + b_a0 + np.random.normal(0, noise_floor_accel, (len(t_imu), 3))
    w_raw = w_nominal + b_g0 + np.random.normal(0, noise_floor_gyro, (len(t_imu), 3))
    
    # GNSS GENERATION (10 Hz)
    # Simulating static GNSS scatter based on expected P0 uncertainties
    sigma_pos_rad = 4.0 / a  # Rough conversion meters to radians for scatter
    sigma_vel = 0.1
    
    gnss_phi = phi_0 + np.random.normal(0, sigma_pos_rad, len(t_gnss))
    gnss_lam = lam_0 + np.random.normal(0, sigma_pos_rad, len(t_gnss))
    gnss_h = h_0 + np.random.normal(0, 4.0, len(t_gnss))
    
    gnss_vE = np.random.normal(0, sigma_vel, len(t_gnss))
    gnss_vN = np.random.normal(0, sigma_vel, len(t_gnss))
    gnss_vU = np.random.normal(0, sigma_vel, len(t_gnss))
    
    # BAROMETER GENERATION (20 Hz)
    # Inverse ISA equation to find static pressure at h_0
    P_nominal = P0_isa * (1 - (L_isa * h_0) / T0_isa) ** (g_local / (R_air * L_isa))
    sigma_baro = 10.0 # Pascals (noise floor)
    baro_P = P_nominal + np.random.normal(0, sigma_baro, len(t_baro))
    
    # MAGNETOMETER GENERATION (20 Hz)
    # Fictional local WMM field at the pad
    m_nominal = np.array([24.5, 3.2, 38.1], dtype=np.float64) # microTeslas
    sigma_mag = 0.5
    
    mag_raw = m_nominal + np.random.normal(0, sigma_mag, (len(t_mag), 3))
    
    # EXPORT DATA
    print("Exporting asynchronous launch pad datasets...")
    
    pd.DataFrame({'t_IMU': t_imu, 'f_X': f_raw[:,0], 'f_Y': f_raw[:,1], 'f_Z': f_raw[:,2],
                  'w_X': w_raw[:,0], 'w_Y': w_raw[:,1], 'w_Z': w_raw[:,2]}
                ).to_csv('preign_data_imu.csv', index=False, float_format='%.8f')
                
    pd.DataFrame({'t_GNSS': t_gnss, 'Lat': gnss_phi, 'Lon': gnss_lam, 'Alt': gnss_h,
                  'v_E': gnss_vE, 'v_N': gnss_vN, 'v_U': gnss_vU}
                ).to_csv('preign_data_gnss.csv', index=False, float_format='%.8f')
                
    pd.DataFrame({'t_baro': t_baro, 'P_static': baro_P}
                ).to_csv('preign_data_baro.csv', index=False, float_format='%.8f')
                
    pd.DataFrame({'t_mag': t_mag, 'm_X': mag_raw[:,0], 'm_Y': mag_raw[:,1], 'm_Z': mag_raw[:,2]}
                ).to_csv('preign_data_mag.csv', index=False, float_format='%.8f')

    print("Pad datasets generated successfully.")
    gc.enable()

if __name__ == "__main__":
    generate_pad_telemetry()