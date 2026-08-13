"""
real_data.py

Loads, cleans, explores, and models a REAL, PUBLICLY AVAILABLE experimental
dataset: the "3D Printer Dataset for Mechanical Engineers"
(Kaggle: https://www.kaggle.com/datasets/afumetto/3dprinter , collected on an
Ultimaker S5 with tensile tests on a Sincotec GmbH tester; PLA and ABS
specimens). A local CSV copy (with the original string-labeled categorical
columns) is stored at data/raw/3d_printer_dataset.csv for reproducibility.

Purpose in this project: the analytical dataset in generate_dataset.py is a
closed-form, illustrative model of auxetic UNIT-CELL GEOMETRY only — it says
nothing about how real FDM/AM process parameters (layer height, infill,
nozzle/bed temperature, print speed) affect the mechanical performance of
PRINTED PLA parts. This module supplies that missing, real-world half of the
picture: it relates real AM process settings to measured tensile strength,
elongation, and surface roughness for PLA and ABS coupons, and is treated as
an independent regression problem (process parameters -> printed-part
properties), not fused numerically with the analytical geometry model.

IMPORTANT — data provenance: this dataset IS real, measured, experimental
data (unlike the analytical auxetic dataset). It is NOT specific to
re-entrant auxetic geometry — it uses standard (grid/honeycomb) infill
coupons — so it must never be described as "auxetic experimental data".
It is used here to ground the project's AM-process reasoning in real
measurements and to demonstrate a second, independent ML pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from plot_style import (
    set_style, CATEGORICAL, SEQUENTIAL_CMAP, DIVERGING_CMAP,
    style_axes, provenance_caption, PROVENANCE_REAL,
)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "3d_printer_dataset.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "real_3d_printer_dataset_clean.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")

NUMERIC_FEATURES = [
    "layer_height", "wall_thickness", "infill_density",
    "nozzle_temperature", "bed_temperature", "print_speed", "fan_speed",
]
CATEGORICAL_FEATURES = ["infill_pattern", "material"]
FEATURES = NUMERIC_FEATURES + ["infill_pattern_honeycomb", "material_pla"]
TARGETS = ["tension_strength", "elongation", "roughness"]

RANDOM_SEED = 42


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real experimental data is messy. Remove physically impossible /
    sensor-error rows rather than imputing them:
      - roughness, tension_strength, elongation must be > 0 (a negative
        surface roughness is a logged sensor error, not a measurement)
      - bed_temperature capped at 120 C (a value of 260 C is far outside the
        operating range of any material in this dataset and is a data-entry
        error, not a real print)
      - print_speed capped at 200 mm/s for the same reason
    """
    before = len(df)
    df = df.dropna()
    df = df[(df["roughness"] > 0) & (df["tension_strength"] > 0) & (df["elongation"] > 0)]
    df = df[(df["bed_temperature"] <= 120) & (df["print_speed"] <= 200)]
    after = len(df)
    print(f"[real_data] Kept {after}/{before} rows after removing sensor-error / non-physical entries.")

    df = df.reset_index(drop=True)
    df["infill_pattern_honeycomb"] = (df["infill_pattern"] == "honeycomb").astype(int)
    df["material_pla"] = (df["material"] == "pla").astype(int)
    return df


def load_clean() -> pd.DataFrame:
    return validate_and_clean(load_raw())


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()
    cols = NUMERIC_FEATURES + TARGETS
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=CATEGORICAL[0], edgecolor="white")
        axes[i].set_title(col)
        style_axes(axes[i])
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Real 3D-Printing Dataset — Feature & Target Distributions")
    provenance_caption(fig, PROVENANCE_REAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(FIG_DIR, "10_real_distributions.png")
    plt.savefig(out)
    plt.close()
    print(f"[real_data] Saved {out}")


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = NUMERIC_FEATURES + ["infill_pattern_honeycomb", "material_pla"] + TARGETS
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0, square=True,
                linewidths=0.5, linecolor="white")
    plt.title("Real 3D-Printing Dataset — Correlation Heatmap")
    plt.figtext(0.5, 0.01, PROVENANCE_REAL, ha="center", fontsize=8.5, style="italic", color="#52514e")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(FIG_DIR, "11_real_correlation_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"[real_data] Saved {out}")


def plot_material_pattern_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    palette = {"abs": CATEGORICAL[0], "pla": CATEGORICAL[1]}
    for i, target in enumerate(TARGETS):
        sns.boxplot(data=df, x="infill_pattern", y=target, hue="material", ax=axes[i],
                     palette=palette, linewidth=1.1)
        axes[i].set_title(target)
        axes[i].set_xlabel("infill pattern")
        style_axes(axes[i])
        if i > 0:
            axes[i].legend_.remove()
    axes[0].legend(title="material", loc="upper right")
    fig.suptitle("Real 3D-Printing Dataset — Material x Infill Pattern Effects")
    provenance_caption(fig, PROVENANCE_REAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(FIG_DIR, "12_real_material_pattern_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"[real_data] Saved {out}")


def plot_process_vs_strength(df: pd.DataFrame):
    """PLA-focused view: how do real process parameters trade off against
    tensile strength? Directly relevant to the PLA-AM theme of this project."""
    pla = df[df["material"] == "pla"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sc0 = axes[0].scatter(pla["infill_density"], pla["tension_strength"],
                           c=pla["layer_height"], cmap=SEQUENTIAL_CMAP, s=70,
                           edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("infill density (%)")
    axes[0].set_ylabel("tensile strength (MPa)")
    axes[0].set_title("PLA: infill density vs. tensile strength")
    style_axes(axes[0])
    cb0 = fig.colorbar(sc0, ax=axes[0])
    cb0.set_label("layer height (mm)")

    sc1 = axes[1].scatter(pla["print_speed"], pla["elongation"],
                           c=pla["nozzle_temperature"], cmap=SEQUENTIAL_CMAP, s=70,
                           edgecolor="white", linewidth=0.6)
    axes[1].set_xlabel("print speed (mm/s)")
    axes[1].set_ylabel("elongation (%)")
    axes[1].set_title("PLA: print speed vs. elongation")
    style_axes(axes[1])
    cb1 = fig.colorbar(sc1, ax=axes[1])
    cb1.set_label("nozzle temperature (C)")

    fig.suptitle("Real Process Parameters vs. Printed-Part Properties (PLA specimens)")
    provenance_caption(fig, PROVENANCE_REAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(FIG_DIR, "13_real_process_vs_strength.png")
    plt.savefig(out)
    plt.close()
    print(f"[real_data] Saved {out}")


# --------------------------------------------------------------------------
# Modeling — small-sample (n ~ 60), so K-fold cross-validation rather than a
# single held-out split is used as the primary evidence of model quality.
# --------------------------------------------------------------------------

MODEL_FACTORIES = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=RANDOM_SEED),
}


def cross_validate_models(df: pd.DataFrame, n_splits: int = 5):
    X = df[FEATURES]
    rows = []
    preds = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for target in TARGETS:
        y = df[target]
        for model_name, factory in MODEL_FACTORIES.items():
            model = factory()
            r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
            mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
            y_pred = cross_val_predict(factory(), X, y, cv=kf)
            preds[(model_name, target)] = y_pred
            rows.append({
                "model": model_name, "target": target,
                "R2_mean": r2_scores.mean(), "R2_std": r2_scores.std(),
                "MAE_mean": mae_scores.mean(), "MAE_std": mae_scores.std(),
            })
    return pd.DataFrame(rows), preds


def plot_model_comparison(cv_df: pd.DataFrame, n_samples: int):
    fig, ax = plt.subplots(figsize=(11, 6))
    models = list(MODEL_FACTORIES.keys())
    width = 0.8 / len(models)
    x = np.arange(len(TARGETS))
    for i, model_name in enumerate(models):
        sub = cv_df[cv_df["model"] == model_name].set_index("target").loc[TARGETS]
        ax.bar(x + i * width, sub["R2_mean"], width, yerr=sub["R2_std"],
               label=model_name, color=CATEGORICAL[i], capsize=3)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("5-fold cross-validated R2")
    ax.axhline(0, color="#898781", linewidth=0.8)
    ax.legend(title="model")
    style_axes(ax)
    ax.set_title("Real 3D-Printing Dataset — Model Comparison (5-fold CV)")
    provenance_caption(fig, f"{PROVENANCE_REAL} | n={n_samples}, small-sample: wide error bars expected")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(FIG_DIR, "14_real_model_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"[real_data] Saved {out}")


def feature_importance_analysis(df: pd.DataFrame):
    X = df[FEATURES]
    records = []
    fitted = {}
    for target in TARGETS:
        y = df[target]
        rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X, y)
        fitted[target] = rf
        result = permutation_importance(rf, X, y, n_repeats=30, random_state=RANDOM_SEED, n_jobs=-1)
        for feat, mean_imp, std_imp in zip(FEATURES, result.importances_mean, result.importances_std):
            records.append({"target": target, "feature": feat, "importance_mean": mean_imp, "importance_std": std_imp})
    return pd.DataFrame(records), fitted


def plot_feature_importance(imp_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, target in enumerate(TARGETS):
        sub = imp_df[imp_df["target"] == target].sort_values("importance_mean")
        axes[i].barh(sub["feature"], sub["importance_mean"], xerr=sub["importance_std"], color=CATEGORICAL[1])
        axes[i].set_title(target)
        axes[i].set_xlabel("permutation importance")
        style_axes(axes[i], y_grid_only=False)
    fig.suptitle("Real 3D-Printing Dataset — Feature Importance (Random Forest, fit on full data)")
    provenance_caption(fig, PROVENANCE_REAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = os.path.join(FIG_DIR, "15_real_feature_importance.png")
    plt.savefig(out)
    plt.close()
    print(f"[real_data] Saved {out}")


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    df = load_clean()
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[real_data] Saved cleaned real dataset -> {PROCESSED_PATH}")
    print(df[NUMERIC_FEATURES + TARGETS].describe().T)

    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_material_pattern_comparison(df)
    plot_process_vs_strength(df)

    cv_df, _ = cross_validate_models(df)
    cv_path = os.path.join(REPORT_DIR, "real_data_cv_metrics.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"[real_data] Saved {cv_path}")
    print(cv_df.pivot(index="target", columns="model", values="R2_mean"))
    plot_model_comparison(cv_df, n_samples=len(df))

    imp_df, fitted_models = feature_importance_analysis(df)
    imp_path = os.path.join(REPORT_DIR, "real_data_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"[real_data] Saved {imp_path}")
    plot_feature_importance(imp_df)

    model_path = os.path.join(REPORT_DIR, "real_data_models.joblib")
    joblib.dump(fitted_models, model_path)
    print(f"[real_data] Saved fitted real-data models -> {model_path}")

    return df, cv_df, imp_df, fitted_models


if __name__ == "__main__":
    main()
