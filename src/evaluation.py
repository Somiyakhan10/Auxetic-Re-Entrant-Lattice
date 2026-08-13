"""
evaluation.py

Evaluates trained surrogate models (Linear, Ridge, Random Forest, Gradient
Boosting) on the held-out test set: MAE, RMSE, R^2; adds 5-fold
cross-validated R^2 across the full dataset as a more robust comparison;
and generates actual-vs-predicted (parity) and residual plots for the best
(Random Forest) model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate as sk_cross_validate

from model import load_split, train_models, TARGETS, FEATURES, MODEL_FACTORIES, RANDOM_SEED
from plot_style import set_style, CATEGORICAL, style_axes, provenance_caption, PROVENANCE_ANALYTICAL

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")


def evaluate(models, X_test, y_test):
    rows = []
    for model_name in MODEL_FACTORIES:
        for target in TARGETS:
            y_pred = models[model_name][target].predict(X_test)
            y_true = y_test[target]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            rows.append({
                "model": model_name, "target": target,
                "MAE": mae, "RMSE": rmse, "R2": r2
            })
    return pd.DataFrame(rows)


def cross_validate(df: pd.DataFrame, n_splits: int = 5):
    """One cross_validate() call per (model, target) computes R2 and MAE
    together in a single set of fold fits, instead of fitting twice."""
    X = df[FEATURES]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for target in TARGETS:
        y = df[target]
        for model_name, factory in MODEL_FACTORIES.items():
            result = sk_cross_validate(
                factory(), X, y, cv=kf, n_jobs=-1,
                scoring=["r2", "neg_mean_absolute_error"],
            )
            rows.append({
                "model": model_name, "target": target,
                "R2_cv_mean": result["test_r2"].mean(), "R2_cv_std": result["test_r2"].std(),
                "MAE_cv_mean": -result["test_neg_mean_absolute_error"].mean(),
                "MAE_cv_std": result["test_neg_mean_absolute_error"].std(),
            })
    return pd.DataFrame(rows)


def plot_model_comparison(cv_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    models = list(MODEL_FACTORIES.keys())
    width = 0.8 / len(models)
    x = np.arange(len(TARGETS))
    for i, model_name in enumerate(models):
        sub = cv_df[cv_df["model"] == model_name].set_index("target").loc[TARGETS]
        ax.bar(x + i * width, sub["R2_cv_mean"], width, yerr=sub["R2_cv_std"],
               label=model_name, color=CATEGORICAL[i], capsize=3)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("5-fold cross-validated R2")
    ax.set_ylim(top=1.02)
    ax.legend(title="model", loc="lower right")
    style_axes(ax)
    ax.set_title("Analytical Dataset — Surrogate Model Comparison (5-fold CV)")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(FIG_DIR, "05_model_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluation] Saved {out}")


def plot_parity(models, X_test, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, target in enumerate(TARGETS):
        y_true = y_test[target]
        y_pred_rf = models["random_forest"][target].predict(X_test)

        axes[i].scatter(y_true, y_pred_rf, s=14, alpha=0.55, color=CATEGORICAL[2], edgecolor="white", linewidth=0.3)
        lims = [min(y_true.min(), y_pred_rf.min()), max(y_true.max(), y_pred_rf.max())]
        axes[i].plot(lims, lims, "--", linewidth=1.2, color="#898781")
        axes[i].set_xlabel("Actual (analytical)")
        axes[i].set_ylabel("Predicted (Random Forest)")
        axes[i].set_title(target)
        style_axes(axes[i], y_grid_only=False)

    fig.suptitle("Actual vs. Predicted — Random Forest Surrogate")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = os.path.join(FIG_DIR, "06_parity_plots.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluation] Saved {out}")


def plot_residuals(models, X_test, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, target in enumerate(TARGETS):
        y_true = y_test[target]
        y_pred = models["random_forest"][target].predict(X_test)
        residuals = y_true - y_pred

        axes[i].scatter(y_pred, residuals, s=14, alpha=0.55, color=CATEGORICAL[0], edgecolor="white", linewidth=0.3)
        axes[i].axhline(0, color="#898781", linestyle="--", linewidth=1.2)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Residual (actual - predicted)")
        axes[i].set_title(target)
        style_axes(axes[i], y_grid_only=False)

    fig.suptitle("Residual Plots — Random Forest Surrogate (test set)")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = os.path.join(FIG_DIR, "07_residuals.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluation] Saved {out}")


def main():
    set_style()
    X_train, X_test, y_train, y_test = load_split()
    models = train_models(X_train, y_train)

    metrics_df = evaluate(models, X_test, y_test)
    os.makedirs(REPORT_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[evaluation] Saved metrics -> {metrics_path}")
    print(metrics_df.pivot(index="target", columns="model", values="R2"))

    df_full = pd.concat([pd.concat([X_train, X_test]), pd.concat([y_train, y_test])], axis=1)
    cv_df = cross_validate(df_full)
    cv_path = os.path.join(REPORT_DIR, "cv_metrics_summary.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"[evaluation] Saved cross-validated metrics -> {cv_path}")

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_model_comparison(cv_df)
    plot_parity(models, X_test, y_test)
    plot_residuals(models, X_test, y_test)

    return metrics_df, models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
