import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import geodesy_math as mat
from closed_loop_eskf import inject_error_state


def run_rts_smoother():
    """Pass Backward Rauch-Tung-Striebel (RTS) Smoother
    Uses nominal state x_nom, discrete dynamic system matrix and
    posteriori/priori covariance matrices"""

    print("Initializing RTS Smoother (Closed-Loop Architecture)...")

    # Inputs and Outputs Path
    in_path = project_root / "data" / "results" / "eskf_output_state.npz"
    out_path = project_root / "data" / "results" / "rts_eskf_output_state.npz"

    if not in_path.exists():
        raise FileNotFoundError(f"Error: Forward Pass ESKF data missing at {in_path}")

    # Loading ESKF results data
    with np.load(in_path, allow_pickle=True) as data:
        x_nom = data["x_nom"]
        P_plus = data["P"]
        P_minus = data["P_minus"]
        Phi = data["Phi"]
        z = data["z"]
        Sk_diag = data["S_k_diag"]

    N = len(x_nom)

    # RTS Memory Allocation
    x_rts = np.copy(x_nom)
    P_rts = np.zeros_like(P_plus)
    dx_rts = np.zeros((N, 15), dtype=np.float64)

    # Boundary Conditionsat at final epoch when t = t_N
    P_rts[-1] = P_plus[-1]
    dx_rts[-1] = np.zeros(15, dtype=np.float64)

    print(f"Propagating backwards for {N} epochs")

    # Backwards Pass Loop
    for k in range(N - 2, -1, -1):
        # Matrix Inversion for strict numerical stablity
        try:
            # First Try: Cholesky (Requires Positive Definite Symmetric matrix)
            L = np.linalg.cholesky(P_minus[k + 1])
            L_inv = np.linalg.inv(L)
            P_pred_inv = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            print(
                f"Warning: Cholesky factorization failed at epoch {k}. Trying with pseudo-inverse..."
            )

            try:
                P_pred_inv = np.linalg.pinv(P_minus[k + 1])
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_minus[k + 1] + np.eye(15) * 1e-12)

        A_k = P_plus[k] @ Phi[k].T @ P_pred_inv

        # Closed Loop Error State Recursion
        dx_rts[k] = A_k @ dx_rts[k + 1]

        # Covariance Matrix Recursion
        P_rts[k] = P_plus[k] + A_k(P_rts[k + 1] - P_minus[k + 1]) @ A_k.T

        # Force Strict Symmetric Matrix
        P_rts[k] = 0.5 * (P_rts[k] + P_rts[k].T)

        # Delta_X State Re-injection
        inject_error_state(x_rts[k], dx_rts[k])

    # Exporting BET (Best Estimated Trajectory) into a NPZ
    np.savez_compressed(out_path, x_nom=x_rts, P=P_rts, z=z, Sk_diag=Sk_diag)

    print(f"RTS smoothing results exported to {out_path}")


if __name__ == "__main__":
    try:
        run_rts_smoother()
    except Exception as e:
        print(f"RTS Smoother Failure: {e}")
        sys.exit(1)
