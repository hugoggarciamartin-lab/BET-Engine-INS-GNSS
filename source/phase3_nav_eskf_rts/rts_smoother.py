import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import geodesy_math as mat
from closed_loop_eskf import inject_error_state


def run_rts_smoother():
    print("Initializing RTS Acausal Smoother (Closed-Loop Architecture)...")

    # Input / Output Paths
    in_path = project_root / "data" / "aligned_data" / "eskf_output_state.npz"
    out_dir = project_root / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rts_eskf_outputs_state.npz"

    if not in_path.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Forward Pass ESKF data missing at {in_path}"
        )

    # Load Forward Pass Data
    with np.load(in_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]
        P_plus = data["P"]
        P_minus = data["P_minus"]
        Phi = data["Phi"]
        z = data["z"]
        Sk_diag = data["Sk_diag"]

    N = len(x_nom)

    # Allocate RTS Memory
    x_rts = np.copy(x_nom)
    P_rts = np.zeros_like(P_plus)
    dx_rts = np.zeros((N, 15), dtype=np.float64)

    # Boundary Condition at final epoch t = N
    P_rts[-1] = P_plus[-1]
    dx_rts[-1] = np.zeros(15, dtype=np.float64)

    print(f"Propagating certainty backwards for {N} epochs...")

    # Backward Pass Loop
    for k in range(N - 2, -1, -1):
        # 1. Smoothing Gain A_k (Eq 8.2) - Using pseudo-inverse for strict numerical stability
        try:
            P_pred_inv = np.linalg.pinv(P_minus[k + 1])
        except np.linalg.LinAlgError:
            print(
                f"Warning: Ill-conditioned covariance matrix at epoch {k}. Applying regularization."
            )
            P_pred_inv = np.linalg.pinv(P_minus[k + 1] + np.eye(15) * 1e-12)

        A_k = P_plus[k] @ Phi[k].T @ P_pred_inv

        # 2. Closed-Loop Error Recursion (Eq 8.4)
        dx_rts[k] = A_k @ dx_rts[k + 1]

        # 3. Covariance Recursion (Eq 8.6)
        P_rts[k] = P_plus[k] + A_k @ (P_rts[k + 1] - P_minus[k + 1]) @ A_k.T
        P_rts[k] = 0.5 * (P_rts[k] + P_rts[k].T)  # Force strict symmetry

        # 4. Retrospective Geometric Injection
        inject_error_state_rts(x_rts[k], dx_rts[k])

    # Exporting BET (Best Estimated Trajectory) keeping identical NPZ structure for seamless plotting
    np.savez_compressed(
        out_path,
        x_nom=x_rts,
        P=P_rts,
        z=z,  # Unchanged (Forward Pass innovations)
        Sk_diag=Sk_diag,  # Unchanged (Forward Pass theoretical envelope)
    )

    print(
        f"RTS Smoothing Complete. Absolute Trajectory (BET) exported to: {out_path.name}"
    )


if __name__ == "__main__":
    try:
        run_rts_smoother()
    except Exception as e:
        print(f"RTS SMOOTHER FAILED: {e}")
        sys.exit(1)
