# BET Engine: Best Estimated Trajectory Reconstruction

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Compliance](https://img.shields.io/badge/compliance-DO--330_Ready-orange)

## 1. Executive Summary
The BET (Best Estimated Trajectory) Engine is a non-real-time, high-fidelity analytical software suite designed for post-flight kinematic trajectory reconstruction. It fuses high-rate Inertial Measurement Unit (IMU) telemetry with low-rate GNSS and Barometric data via an Error-State Kalman Filter (ESKF) and an acausal Rauch-Tung-Striebel (RTS) Smoother. 

This tool is engineered for forensic trajectory validation, sensor bias estimation, and dynamic aerodynamic thrust reconstruction, adhering to strict configuration management principles.

---

## 2. System Architecture & Synthetic Data Generation (DO-330 Context)

To maintain strict traceability and avoid corrupting flight tests with unvalidated data, this repository enforces a rigid separation between algorithmic processing and data generation.

> [!WARNING] 
> **CRITICAL NOTICE REGARDING SYNTHETIC DATA GENERATORS:**
> Scripts responsible for generating telemetry and boundary conditions (e.g., flight profiles, Allan variance laboratory data, mass depletion curves) **DO NOT** process real-world flight data. These are Environment Simulators designed to output **strictly fictitious, synthetic telemetry**. 
> Their sole purpose is to provide mathematical boundary conditions and test vectors to validate the Kalman filter without relying on physical flight logs. They must never be executed in a production pipeline analyzing real flight data.

### Subsystem Manifest
* **Phase 1: Simulators (`data/raw/` or `source/phase1_simulators/`)**: Synthetic data fabricators and environmental lookup tables.
* **Phase 2: Preprocessing**: Signal conditioning and temporal alignment of asynchronous sensors.
* **Phase 3: Navigation Core**: 15-state ESKF forward pass and RTS backward pass.
* **Phase 4: Validation**: Statistical threshold assertion and thrust reconstruction.

---

## 3. Cloning, Configuration, and Environment Setup

This software requires a strictly isolated environment. Global Python installations are explicitly prohibited to prevent dependency conflicts and ensure deterministic execution across different machines.

### 3.1. Repository Cloning
Clone the repository to your local machine or cloud compute instance:
```bash
git clone [https://github.com/hugoggarciamartin-lab/BET-Engine-INS-GNSS.git](https://github.com/hugoggarciamartin-lab/BET-Engine-INS-GNSS.git)
cd BET-Engine-INS-GNSS

###3.2. Virtual Environment Provisioning
You must provision an ephemeral virtual environment before executing any mathematical operations

# Initialize a pristine Python virtual environment
python3 -m venv .venv

# Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

###3.3. Strict Dependency Installation
Install the exact library versions required for tensor operations and forensic plotting.
# Ensure pip is updated to prevent build wheel errors
pip install --upgrade pip

# Install dependencies from the locked manifest
pip install -r requirements.txt

###3.4. Single Source of Truth (SSOT) Configuration
The system parameters (e.g., IMU noise densities, initial coordinates, temporal boundaries) are strictly controlled by a single configuration file.
Before running the pipeline, verify the parameters in:
config/config_baseline.yaml

Do not hardcode variables into the Python scripts. All algorithmic adjustments must be funneled through the YAML configuration file.

##.4. Single Source of Truth (SSOT) Configuration
The system parameters (e.g., IMU noise densities, initial coordinates, temporal boundaries) are strictly controlled by a single configuration file.
Before running the pipeline, verify the parameters in:
config/config_baseline.yaml

Do not hardcode variables into the Python scripts. All algorithmic adjustments must be funneled through the YAML configuration file.

*Phase 1: Synthetic Environment Generation
(Skip this phase if injecting real physical flight logs into data/raw/)

Execute the environment simulators to fabricate the baseline fictitious telemetry, structural models, and magnetic spatial grids. This will populate the data/raw/ directory.

# Generate environmental and physical models
python source/phase1_simulators/generate_cfd_model.py
python source/phase1_simulators/parse_wmm_to_enu_csv.py
python source/phase1_simulators/generate_mass_profile.py

# Generate laboratory and flight telemetry
python source/phase1_simulators/generate_allan_variance_data.py
python source/phase1_simulators/generate_pad_telemetry.py
python source/phase1_simulators/generate_flight_profile.py

Phase 2: Signal Conditioning & Alignment
Real-world and simulated sensors operate asynchronously. This phase interpolates and aligns all IMU, GNSS, and Barometric data to a common high-frequency master clock, generating the master_flight_data.csv.

python source/phase2_preprocessing/signal_conditioning.py
python source/phase2_preprocessing/temporal_aligner.py

Phase 3: Kinematic Integration (The Core)
Run the forward causal Error-State Kalman Filter (ESKF) followed by the backward acausal Rauch-Tung-Striebel (RTS) Smoother. This generates the optimal state and covariance tensors (.npz format).

python source/phase3_nav_eskf_rts/closed_loop_eskf.py
python source/phase3_nav_eskf_rts/rts_smoother.py

Phase 4: Forensic Analysis & Aerodynamic Reconstruction
Unpack the .npz tensors to assert the bounds, visualize the 3D trajectory, and compute the dynamic engine thrust.

# Execute local plotting scripts
python source/phase4_validation/plot_rts_states.py
python source/phase4_validation/plot_aeroprop_perfm.py

5. Certification & Requirements-Based Testing
To support tool qualification objectives, this software is continuously validated against edge cases (e.g., gimbal lock, polar singularities, atmospheric limits).

Running the Test Suite
The repository utilizes pytest to assert numerical stability and enforce interface contracts.

