"""
inverse_design.py

Given a target Poisson's ratio (nu12) and optionally a target modulus
ratio (E1_Es), search for the best-matching unit-cell geometry using:
  (a) a fine grid search over the ANALYTICAL model (ground truth), and
  (b) the trained Random Forest SURROGATE (for speed / demonstration).

This directly answers Research Question 4 from the project design.
"""

import os
import numpy as np
import pandas as pd

from generate_dataset import compute_properties, THETA_DEG_RANGE, HL_RATIO_RANGE, TL_RATIO_RANGE
from model import load_split, train_models, FEATURES

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")


def analytical_inverse_search(target_nu12: float, target_E1_Es: float = None,
                               n_theta=120, n_hl=60, n_tl=60, w_nu=1.0, w_E=1.0):
    """
    Fine grid search over the analytical model to find the geometry whose
    predicted (nu12, E1_Es) best matches the requested target(s).
    """
    theta_grid = np.linspace(*THETA_DEG_RANGE, n_theta)
    hl_grid = np.linspace(*HL_RATIO_RANGE, n_hl)
    tl_grid = np.linspace(*TL_RATIO_RANGE, n_tl)

    TH, HL, TL = np.meshgrid(theta_grid, hl_grid, tl_grid, indexing="ij")
    E1, E2, nu12, nu21, rel_density = compute_properties(TH, HL, TL)

    valid = (E1 > 0) & (rel_density > 0.01) & (rel_density < 0.9) & np.isfinite(nu12) & np.isfinite(E1)

    err_nu = (nu12 - target_nu12) ** 2
    if target_E1_Es is not None:
        # normalize E error scale roughly to nu error scale
        err_E = ((E1 - target_E1_Es) / max(target_E1_Es, 1e-6)) ** 2
        total_err = w_nu * err_nu + w_E * err_E
    else:
        total_err = err_nu

    total_err = np.where(valid, total_err, np.inf)
    idx = np.unravel_index(np.argmin(total_err), total_err.shape)

    result = {
        "theta_deg": float(TH[idx]),
        "hl_ratio": float(HL[idx]),
        "tl_ratio": float(TL[idx]),
        "predicted_nu12": float(nu12[idx]),
        "predicted_E1_Es": float(E1[idx]),
        "predicted_rel_density": float(rel_density[idx]),
        "source": "analytical_grid_search",
    }
    return result


def surrogate_inverse_search(models, target_nu12: float, target_E1_Es: float = None,
                              n_theta=60, n_hl=40, n_tl=40):
    """
    Same idea, but queries the trained Random Forest surrogate instead of
    the analytical equations directly (demonstrates the surrogate is usable
    for design queries, not just single-point prediction).
    """
    theta_grid = np.linspace(*THETA_DEG_RANGE, n_theta)
    hl_grid = np.linspace(*HL_RATIO_RANGE, n_hl)
    tl_grid = np.linspace(*TL_RATIO_RANGE, n_tl)

    combos = np.array(np.meshgrid(theta_grid, hl_grid, tl_grid, indexing="ij")).reshape(3, -1).T
    theta_c, hl_c, tl_c = combos[:, 0], combos[:, 1], combos[:, 2]

    _, _, _, _, rel_density_c = compute_properties(theta_c, hl_c, tl_c)
    X_query = pd.DataFrame({
        "theta_deg": theta_c, "hl_ratio": hl_c, "tl_ratio": tl_c, "rel_density": rel_density_c
    })[FEATURES]

    nu12_pred = models["random_forest"]["nu12"].predict(X_query)
    E1_pred = models["random_forest"]["E1_Es"].predict(X_query)

    err_nu = (nu12_pred - target_nu12) ** 2
    if target_E1_Es is not None:
        err_E = ((E1_pred - target_E1_Es) / max(target_E1_Es, 1e-6)) ** 2
        total_err = err_nu + err_E
    else:
        total_err = err_nu

    best = np.argmin(total_err)
    result = {
        "theta_deg": float(theta_c[best]),
        "hl_ratio": float(hl_c[best]),
        "tl_ratio": float(tl_c[best]),
        "predicted_nu12": float(nu12_pred[best]),
        "predicted_E1_Es": float(E1_pred[best]),
        "predicted_rel_density": float(rel_density_c[best]),
        "source": "random_forest_surrogate_search",
    }
    return result


def main():
    X_train, X_test, y_train, y_test = load_split()
    models = train_models(X_train, y_train)

    example_targets = [
        {"target_nu12": -0.3, "target_E1_Es": None},
        {"target_nu12": -1.0, "target_E1_Es": 0.01},
        {"target_nu12": -0.15, "target_E1_Es": 0.005},
    ]

    rows = []
    for t in example_targets:
        analytical = analytical_inverse_search(**t)
        surrogate = surrogate_inverse_search(models, **t)
        rows.append({"target_nu12": t["target_nu12"], "target_E1_Es": t["target_E1_Es"], **{f"analytical_{k}": v for k, v in analytical.items()}})
        rows.append({"target_nu12": t["target_nu12"], "target_E1_Es": t["target_E1_Es"], **{f"surrogate_{k}": v for k, v in surrogate.items()}})

    df = pd.DataFrame(rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "inverse_design_examples.csv")
    df.to_csv(out_path, index=False)
    print(f"[inverse_design] Saved example inverse-design results -> {out_path}")
    print(df)


if __name__ == "__main__":
    main()
