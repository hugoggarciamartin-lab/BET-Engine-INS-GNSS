# BET Engine: Best Estimated Trajectory Reconstruction

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Overview
This repository contains the mathematical formulation and algorithmic implementation of a high-fidelity post-processing engine designed for the reconstruction of ballistic and guided flight trajectories. 

Unlike real-time navigation systems, this software serves as an analytical validation tool. It fuses high-frequency inertial telemetry (IMU) with low-frequency GNSS/Barometric data to mitigate intrinsic sensor drift and generate a continuous, stochastically optimal **Best Estimated Trajectory (BET)**.

## 2. Core Architecture
The core algorithm is built upon a **15-State Error-State Kalman Filter (ESKF)** operating under a **Closed-Loop Loosely Coupled** configuration.

* **Kinematic Mechanization**: 4th-order Runge-Kutta (RK4) integration of specific force and angular rates, accounting for Coriolis and transport rate effects over the WGS84 ellipsoid.
* **Closed-Loop Error Injection**: Real-time feedback of estimated sensor biases $(b_a, b_g)$ into the nominal integrator to preserve first-order linearization validity.
* **Acasual Smoothing**: Implementation of a **Rauch-Tung-Striebel (RTS)** backward smoother to eliminate causal filter discontinuities and compress the covariance uncertainty bounds.
* **Adaptive Covariance**: Dynamic process noise matrix $(Q_k)$ scaling to isolate the filter from vibration-induced rectification errors and high-dynamic acoustic anomalies (e.g., Venturi effect on static pressure).

## 3. Environment Setup
This project strictly enforces dependency isolation. Global Python environments are explicitly unsupported.

```bash
# 1. Clone the repository
git clone [https://github.com/HugoGarciaMartin/BET_Engine.git](https://github.com/HugoGarciaMartin/BET_Engine.git)
cd BET_Engine

# 2. Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# 3. Install strict dependencies
pip install -r requirements.txt
