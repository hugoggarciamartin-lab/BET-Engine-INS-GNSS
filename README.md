# BET Engine: Best Estimated Trajectory Reconstruction

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Compliance](https://img.shields.io/badge/compliance-DO--330_Ready-orange)

## 1. Executive Summary
This repository hosts a production-grade, 15-state Error-State Kalman Filter (ESKF) and Rauch-Tung-Striebel (RTS) smoothing engine explicitly designed to reconstruct the Best Estimated Trajectory (BET) of sounding rockets and uncrewed aerospace platforms. Developed under strict software engineering principles and architected to align with aviation tool qualification frameworks (such as DO-330), the system deterministically solves the non-linear kinematics of flight by fusing high-frequency inertial telemetry (IMU) with low-frequency external observations (GNSS, Barometry, Magnetometry).

Unlike standard real-time navigation filters, the BET Engine is optimized for post-flight analysis. It enforces an immutable architectural segregation between raw data ingestion, kinematic integration (via 4th-order Runge-Kutta), and stochastic estimation. By leveraging the entire recorded flight history, the acausal RTS backward pass smooths the causal state estimates, globally minimizing covariance bounds and converging hardware errors—such as accelerometer and gyroscope biases—to their true physical values.

Furthermore, the pipeline extends beyond pure kinematics through its dedicated Aeropropulsive Validation Lane. It cross-references the optimal time-correlated spatial states with standard atmospheric models (ISA) and dynamic mass depletion curves to mathematically isolate the propulsive force vector. This enables the direct reconstruction of the engine's real-world thrust profile and the extraction of the empirical aerodynamic drag coefficient ($C_D$) during unpowered ballistic phases, providing an auditable, end-to-end traceability from raw sensor noise to physical flight performance.

---

## 2. Mathematical Framework & Stochastic Plant
The analytical core relies on an uncoupled error-state formulation, ensuring that the non-linear strapdown integration remains isolated from the linear stochastic updates.

### 2.1. Kinematic Integration (Strapdown INS)
*   **Runge-Kutta 4th Order (RK4):** The high-frequency inertial translation and rotation are integrated using an RK4 numerical solver. This module updates the nominal state vector (position, velocity, and quaternion-based attitude) in the global WGS84 frame, strictly avoiding the singularities inherent to Euler angle representations.
*   **Geodesy & Gravity:** Incorporates standard WGS84 Earth curvature and local gravity vector calculations to map ECEF coordinates to the local NED (North-East-Down) navigation frame.

### 2.2. Stochastic Estimation (ESKF & RTS)
*   **15-DOF Error-State Kalman Filter:** Executes the causal forward pass. It estimates a 15-dimensional error vector (position, velocity, attitude errors, plus accelerometer and gyroscope hardware biases). The covariance matrix is propagated via the system Jacobian and updated using the numerically stable Joseph form to maintain symmetry and positive definiteness.
*   **Rauch-Tung-Striebel Smoother:** Executes the acausal backward pass. By utilizing the entire stored flight history, the RTS algorithm smooths the causal state estimates, globally minimizing the covariance bounds and converging the sensor biases to their true physical values.

### 2.3. Aeropropulsive Extraction
*   **Zero-Phase Filtering:** Employs bidirectional Butterworth low-pass filters on the specific force vectors to purge structural vibration and acoustic noise without introducing phase shifts.
*   **Thrust & Drag Reconstruction:** Cross-references the smoothed kinematic states with an empirical mass depletion model and the ISA standard atmosphere to mathematically isolate the propulsive force and the aerodynamic drag during ballistic phases.

---S

## 3. System Architecture & Synthetic Data Generation (DO-330 Context)

To maintain strict traceability and avoid corrupting flight tests with unvalidated data, this repository enforces a rigid separation between algorithmic processing and data generation.

> [!WARNING] 
> **CRITICAL NOTICE REGARDING SYNTHETIC DATA GENERATORS:**
> Scripts responsible for generating telemetry and boundary conditions (flight profiles, Allan variance laboratory data, mass depletion curves) **DO NOT** process real-world flight data. These are Environment Simulators designed to output **strictly synthetic telemetry**. 
> Their sole purpose is to provide mathematical boundary conditions and test vectors to validate the Kalman filter without relying on physical flight logs. They must never be executed in a production pipeline analyzing real flight data.

### Subsystem Manifest
* **Phase 1: Simulators (`data/phase1_simulators/`)**: Synthetic data fabricators and environmental lookup tables.
* **Phase 2: Preprocessing**: Signal conditioning and temporal alignment of asynchronous sensors.
* **Phase 3: Navigation Core**: 15-state ESKF forward pass and RTS backward pass.
* **Phase 4: Validation**: Statistical threshold assertion and thrust reconstruction.

---

## 4. Cloning, Configuration, and Environment Setup

This software requires an isolated environment. Global Python installations are explicitly prohibited to ensure deterministic execution across different machines.

### 4.1. Repository Cloning
Clone the repository to your local machine or cloud compute instance:

    git clone https://github.com/hugoggarciamartin-lab/BET-Engine-INS-GNSS.git
    cd BET-Engine-INS-GNSS

### 4.2. Virtual Environment Provisioning
You must provision an ephemeral virtual environment before executing any mathematical operations.
Initialize a pristine Python virtual environment:

    python3 -m venv .venv

    # Activate the environment
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    .\.venv\Scripts\activate

### 4.3. Strict Dependency Installation
Install the exact library versions required for tensor operations and plotting.

    # Check if pip is updated to prevent build wheel errors
    pip install --upgrade pip

    # Install dependencies from the locked manifest
    pip install -r requirements.txt

To eliminate dependency path hacks, install the core repository as an editable Python package. Run the following command from the repository root:

    ```bash
    pip install -e .

### 4.4. Single Source of Truth (SSOT) Configuration
The system parameters (IMU noise densities, initial coordinates, temporal boundaries) are strictly controlled by a single configuration file. 
Before running the pipeline, verify the parameters in:
`config/config_baseline.yaml`

**Do not fix variables into the Python scripts.** All algorithmic adjustments must be directed through the YAML configuration file.

---

## 5. Deterministic Execution Protocol

To successfully reconstruct a trajectory, the system requires the sequential execution of four distinct phases. **Do not alter this execution order**, as each downstream process strictly depends on the artifacts generated by its predecessor.

### Phase 1: Synthetic Environment Generation
*(Skip this phase if injecting real physical flight logs into `data/raw/`)*

Execute the environment simulators to fabricate the baseline fictitious telemetry, structural models, and magnetic spatial grids. This will populate the `data/raw/` directory.

    # Generate environmental and physical models
    python data/phase1_simulators/cfd_dragcoef_data.py
    python data/phase1_simulators/wwm_magn_data.py
    python data/phase1_simulators/gen_mass_profile_data.py

    # Generate laboratory and flight telemetry
    python data/phase1_simulators/gen_allan_data_lab_static.py
    python data/phase1_simulators/gen_pre_ignition_data.py
    python data/phase1_simulators/gen_profile_flight_data.py

### Phase 2: Signal Conditioning & Alignment
Real-world sensors operate asynchronously. This phase interpolates and aligns all IMU, GNSS, and Barometric data to a common high-frequency master clock, generating the `master_flight_data.csv`.

    python source/phase2_preprocessing/signal_conditioning.py
    python source/phase2_preprocessing/temporal_aligner.py

### Phase 3: Kinematic Integration (The Core)
Run the forward causal Error-State Kalman Filter (ESKF) followed by the backward acausal Rauch-Tung-Striebel (RTS) Smoother. This generates the optimal state and covariance tensors (`.npz` format).

    python source/phase3_nav_eskf_rts/closed_loop_eskf.py
    python source/phase3_nav_eskf_rts/rts_smoother.py

### Phase 4: Analysis & Aerodynamic Reconstruction
Unpack the `.npz` tensors to assert the bounds, visualize the 3D trajectory, and compute the dynamic engine thrust.

    # Execute local plotting scripts
    python source/phase4_validation/plot_eskf_states.py
    python source/phase4_validation/plot_rts_states.py
    python source/phase4_validation/plot_innovationz_analyisis.py
    python source/phase4_validation/plot_aeroprop_perfm.py
    python source/phase4_validation/plot_master_telemetry.py

---

## 6. Certification & Requirements-Based Testing
To support tool qualification objectives, this software is continuously validated against edge cases (gimbal lock, polar singularities, atmospheric limit).

### Running the Test Suite
The repository utilizes `pytest` to assert numerical stability and enforce interface contracts.

    # Execute all unit tests and generate a Statement Coverage report
    pytest test/

---

## 7. Industrial Applications, Limitations & Future Work

### 7.1. Applications
* **Post-Flight Aerospace Analysis:** Reconstructing flight paths and detecting sensor anomalies or attitude drift for experimental vehicles and suborbital test flights.
* **Aerodynamic & Propulsive Audit:** Extracting empirical drag coefficients (C_D) and engine thrust profiles from telemetry combined with CFD tabulated data.
* **Sensor Calibration & Bias Estimation:** Estimating deterministic turn-on biases and tracking stochastic noise parameters: Random Walk (ARW, VRW) and Bias Instability(Flicker Noise) via Allan Variance.

### 7.2. Limitations
* **Non-Real-Time Processing Constraint:** The inclusion of an acausal Rauch-Tung-Striebel (RTS) backward smoother requires complete flight datasets, rendering the pipeline unsuitable for real-time onboard flight control computers.
* **Tropospheric Atmospheric Model Bounds:** The ISA atmospheric model implemented is constrained to tropospheric altitudes (h <= 11,000 meters), limiting high-altitude stratospheric trajectory accuracy unless extended.
* **Dynamic Maneuvering Assumptions:** High-dynamic maneuvers beyond the modelled error-state boundaries or severe unconsidered vibrations may cause temporary filter divergence if innovation sigma limits are exceeded.

### 7.3. Future Updatings / Missing Features
* **Extended Kalman Filter (EKF) Real-Time Mode:** Implementation of a causal forward-only operational mode for hardware-in-the-loop (HIL) testing.
* **Automated Pipeline:** Integration of GitHub Actions for automated cross-platform `pytest` execution and code coverage reporting on every commit.
* **Stratospheric & Extratropospheric Extension:** Expansion of standard atmosphere routines to cover stratosphere and mesosphere layers.
* **Aerodynamic Barometric Pressure Correction**: Current altitude updates incorporate a real-time stochastic mitigation strategy, dynamically inflating barometer measurement noise covariance based on a Mach number threshold ($M > 0.3$). Future iterations will replace this heuristic inflation with deterministic aerodynamic Static Source Error Correction (SSEC) models.This will also integrate Mach-dependent aerodynamic correction models (such as Prandtl-Glauert or empirical static source error correction tables) to compensate for dynamic pressure contamination at high subsonic and supersonic speeds.

## 8. System Architecture & Directory Structure

The current software architecture is mapped below. It separates the execution pipeline into configuration, data ingestion, core navigation processing, and post-flight validation.

```text
/BET_ENGINE
├── /config                             # Single Source of Truth (SSOT)
│   ├── config_baseline.yaml            # Base noise parameters, boundaries, and initial vehicle states.
│   └── config_parser.py                # Parses YAML to inject configuration securely into the pipeline.
│
├── /data                               # Telemetry and Artifact Storage
│   ├── /aligned_data                   
│   │   ├── conditioned_flight_data_imu.csv
│   │   └── master_flight_data.csv      # Time-aligned, synchronized master telemetry.
│   ├── /phase1_simulators              # Synthetic data generators and boundary condition modeling.
│   │   ├── cfd_dragcoef_data.py        # Generates aerodynamic CD lookup tables.
│   │   ├── gen_allan_data_lab_static.py # Generates static IMU calibration data.
│   │   ├── gen_mass_profile_data.py    # Generates dynamic mass depletion model.
│   │   ├── gen_pre_ignition_data.py    # Generates pre-ignition static telemetry.
│   │   ├── gen_profile_flight_data.py  # Generates synthetic flight telemetry.
│   │   ├── raw_wmm_data.txt
│   │   └── wwm_magn_data.py            # Generates magnetic variation tables.
│   ├── /raw                            # Immutable raw telemetry ingestion folder.
│   └── /results                        # Output artifacts from the stochastic filters.
│       ├── /eskf_csv_results
│       ├── eskf_output_state.npz       # Forward-pass optimal state tensor.
│       └── rts_eskf_output_state.npz   # Acausal smoothed BET optimal tensor.
│
├── /phase4_validation                  # Post-processing & physical capability extraction
│   ├── extract_npz_to_csv.py           # Converts binary tensors to human-readable format.
│   ├── fft_analysis.py                 # Fast Fourier Transform for signal frequency analysis.
│   ├── plot_aeroprop_perfm.py          # Isolates dynamic thrust and empirical CD.
│   ├── plot_conditioned_signal.py      # Visualizes filtered vs. raw signals.
│   ├── plot_eskf_states.py             # Visualizes causal filter state outputs.
│   ├── plot_innovationz_analyisis.py   # Analyzes filter measurement innovations/residuals.
│   ├── plot_master_telemetry.py        # Visualizes the time-aligned master dataset.
│   └── plot_rts_states.py              # Visualizes the smoothed BET trajectory constraints.
│
├── /source                             # Core execution modules
│   ├── /phase2_preprocessing           # Telemetry conditioning and alignment.
│   │   ├── allan_variance.py           # Computes Allan variance for IMU noise profiling.
│   │   ├── signal_conditioning.py      # Applies zero-phase Butterworth low-pass filtering.
│   │   └── temporal_aligner.py         # Interpolates asynchronous sensors to a master clock.
│   └── /phase3_nav_eskf_rts            # The core stochastic and kinematic solvers.
│       ├── closed_loop_eskf.py         # Main ESKF orchestrator.
│       ├── eskf_measurement.py         # Joseph form updates and observation matrices (H).
│       ├── eskf_predictor.py           # State transitions (Φ) and Jacobians (F).
│       ├── geodesy_math.py             # WGS84 coordinate transformations (ECEF/NED).
│       ├── kinematics_ins.py           # RK4 strapdown integration and ISA atmospheric math.
│       ├── rts_smoother.py             # Backwards sweep for global covariance minimization.
│       └── static_initialization.py    # Computes initial states (x0, P0) from static data.
│
├── /test                               # Automated unit testing suite
│   ├── test_geodesy_math.py
│   └── test_kinematics_ins.py
│
├── pyproject.toml                      # Build system configuration (setuptools).
├── requirements.txt                    # Python environment package dependencies.
└── run_bet_engine.py                   # Master execution entry point.
