import numpy as np
import pandas as pd
from pathlib import Path


def export_simulation_data() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    npz_path = base_dir / "data" / "results" / "eskf_output_state.npz"
    out_dir = base_dir / "data" / "results" / "eskf_csv_results"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file {npz_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:
        # Vectors Exportation
        vec_dict = {}
        for k in ["x_nom", "dx", "z", "Sk_diag"]:
            if k in data:
                array = np.asarray(data[k])
                if array.ndim == 1:
                    vec_dict[k] = array
                elif array.ndim == 2:
                    for i in range(array.shape[1]):
                        vec_dict[f"{k}_{i}"] = array[:, i]

        if vec_dict:
            pd.DataFrame(vec_dict).to_csv(out_dir / "eskf_vectors.csv", index=False)

        # Matrix Exportation
        matrix_dict = {}
        for j in ["P_minus", "P", "Phi"]:
            if j in data:
                array = data[j]
                if array.ndim == 3:
                    n_steps, rows, cols = array.shape
                    for r in range(rows):
                        for c in range(cols):
                            matrix_dict[f"{j}_{r}_{c}"] = array[:, r, c]

        if matrix_dict:
            pd.DataFrame(matrix_dict).to_csv(out_dir / "eskf_matrices.csv", index=False)
    print(f"Data succesfully exported in {out_dir}")


if __name__ == "__main__":
    export_simulation_data()
