"""
sensitivity.py

Feature importance (permutation importance) and one-at-a-time (OAT)
sensitivity sweeps for the trained Random Forest surrogate models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from model import load_split, train_models, FEATURES, TARGETS
from generate_dataset import compute_properties
from plot_style import set_style, CATEGORICAL, style_axes, provenance_caption, PROVENANCE_ANALYTICAL

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")

RANDOM_SEED = 42


def permutation_importance_analysis(models, X_test, y_test):
    records = []
    for target in TARGETS:
        rf = models["random_forest"][target]
        result = permutation_importance(
            rf, X_test, y_test[target], n_repeats=15, random_state=RANDOM_SEED, n_jobs=-1
        )
        for feat, mean_imp, std_imp in zip(FEATURES, result.importances_mean, result.importances_std):
            records.append({"target": target, "feature": feat, "importance_mean": mean_imp, "importance_std": std_imp})
    return pd.DataFrame(records)


def plot_importance(imp_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    for i, target in enumerate(TARGETS):
        sub = imp_df[imp_df["target"] == target].sort_values("importance_mean")
        axes[i].barh(sub["feature"], sub["importance_mean"], xerr=sub["importance_std"], color=CATEGORICAL[1])
        axes[i].set_title(f"Feature Importance — {target}")
        axes[i].set_xlabel("Permutation importance")
        style_axes(axes[i], y_grid_only=False)
    fig.suptitle("Permutation Feature Importance (Random Forest Surrogate)")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = os.path.join(FIG_DIR, "08_feature_importance.png")
    plt.savefig(out)
    plt.close()
    print(f"[sensitivity] Saved {out}")


def one_at_a_time_sweep():
    """
    Sweep each geometric parameter individually (analytical model, not ML)
    while holding the others at their median value, to visualize direct
    physical sensitivity of nu12 and E1_Es.
    """
    theta_med, hl_med, tl_med = -35.0, 1.4, 0.10

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    theta_range = np.linspace(-80, -5, 200)
    E1, _, nu12, _, _ = compute_properties(theta_range, hl_med, tl_med)
    axes[0].plot(theta_range, nu12, color=CATEGORICAL[0])
    axes[0].axhline(0, color="#898781", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("theta (deg)")
    axes[0].set_ylabel("nu12")
    axes[0].set_title("Sensitivity: theta -> nu12")
    style_axes(axes[0], y_grid_only=False)

    hl_range = np.linspace(0.5, 2.0, 200)
    E1b, _, nu12b, _, _ = compute_properties(theta_med, hl_range, tl_med)
    axes[1].plot(hl_range, nu12b, color=CATEGORICAL[1])
    axes[1].axhline(0, color="#898781", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("h/l ratio")
    axes[1].set_ylabel("nu12")
    axes[1].set_title("Sensitivity: h/l -> nu12")
    style_axes(axes[1], y_grid_only=False)

    tl_range = np.linspace(0.02, 0.20, 200)
    E1c, _, nu12c, _, _ = compute_properties(theta_med, hl_med, tl_range)
    axes[2].plot(tl_range, E1c, color=CATEGORICAL[2])
    axes[2].set_xlabel("t/l ratio")
    axes[2].set_ylabel("E1*/Es")
    axes[2].set_title("Sensitivity: t/l -> E1*/Es")
    style_axes(axes[2], y_grid_only=False)

    fig.suptitle("One-at-a-Time Sensitivity Sweeps (Analytical Model)")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = os.path.join(FIG_DIR, "09_oat_sensitivity.png")
    plt.savefig(out)
    plt.close()
    print(f"[sensitivity] Saved {out}")


def main():
    set_style()
    X_train, X_test, y_train, y_test = load_split()
    models = train_models(X_train, y_train)

    imp_df = permutation_importance_analysis(models, X_test, y_test)
    os.makedirs(REPORT_DIR, exist_ok=True)
    imp_path = os.path.join(REPORT_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"[sensitivity] Saved {imp_path}")

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_importance(imp_df)
    one_at_a_time_sweep()

    return imp_df


if __name__ == "__main__":
    main()
