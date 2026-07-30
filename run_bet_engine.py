"""Master Pipeline execution launcher.
Enfoces OS-level memory isolation for subsystem execution and injects strict environmental paths"""

import sys
import gc
import time
import subprocess
from pathlib import Path


def execute_bet_pipeline():
    """
    Orchestrates the Best Estimated Trajectory (BET) pipeline.
    Enforces OS-level memory isolation for each critical navigation phase.
    """

    # Begining of computing time
    t_start = time.perf_counter()
    k_last = 0  # Last processed epoch

    # Repo path
    project_root = Path(__file__).resolve().parent

    # Establishing the Path to the Project Scripts
    filtering_script = (
        project_root / "source" / "phase2_preprocessing" / "signal_conditioning.py"
    )
    aligner_script = (
        project_root / "source" / "phase2_preprocessing" / "temporal_aligner.py"
    )
    eskf_script = (
        project_root / "source" / "phase3_nav_eskf_rts" / "closed_loop_eskf.py"
    )
    rts_script = project_root / "source" / "phase3_nav_eskf_rts" / "rts_smoother.py"

    rts_plot_script = project_root / "validation" / "plot_rts_states.py"

    aero_plot_script = project_root / "validation" / "plot_aeroprop_perfm.py"

    pipeline_phases = [
        ("IMU Signal Preprocessing (Phase 2)", filtering_script),
        ("Temporal Alignment (Phase 2)", aligner_script),
        ("ESKF Forward Pass (Phase 3)", eskf_script),
        ("RTS Acausal Smoother (POST)", rts_script),
        ("Plotting ESKF+RTS Analysis", rts_plot_script),
        ("Plotting Reconstructed Thrust and C_D Evolution", aero_plot_script),
    ]
    print("    BET ENGINE: PIPELINE EXECUTION INITIATING      ")

    for phase_name, script_path in pipeline_phases:
        if not script_path.exists():
            print(f"\n FAILURE: Executable missing at {script_path}")
            sys.exit(1)

        print(f"\n Launching Subsystem: {phase_name}")

        try:
            # Isolated Execution - Avoids Memory leaks
            subprocess.run(
                [sys.executable, str(script_path)], cwd=str(project_root), check=True
            )
        except subprocess.CalledProcessError as e:
            print(
                f"\n PIPELINE Failed: {phase_name} aborted with OS exit code {e.returncode}."
            )
            print("Audit the specific script logs in order to trace the problem.")
            sys.exit(1)
    print("     PIPELINE COMPLETED      ")
    gc.enable()
    gc.collect()

    # Total Computing Time
    t_elapsed = time.perf_counter() - t_start

    print("COMPUTING TIME PERFORMANCE REPORT")
    print(f"Total Computing Time    : {t_elapsed:.4f} segundos")


if __name__ == "__main__":
    execute_bet_pipeline()
