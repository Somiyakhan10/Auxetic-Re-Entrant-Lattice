"""
eda.py

Exploratory Data Analysis on the illustrative computational auxetic dataset.
Generates: feature/target distributions, correlation heatmap,
feature-vs-target relationship plots, and a 3D surface of the analytical
geometry -> Poisson's-ratio relationship.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import seaborn as sns

from generate_dataset import compute_properties, THETA_DEG_RANGE, HL_RATIO_RANGE
from plot_style import (
    set_style, CATEGORICAL, SEQUENTIAL_CMAP, DIVERGING_CMAP,
    style_axes, provenance_caption, PROVENANCE_ANALYTICAL,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "auxetic_dataset.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

FEATURES = ["theta_deg", "hl_ratio", "tl_ratio", "rel_density"]
TARGETS = ["E1_Es", "E2_Es", "nu12", "nu21"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def plot_distributions(df: pd.DataFrame):
    cols = FEATURES + TARGETS
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=CATEGORICAL[0], edgecolor="white")
        axes[i].set_title(col)
        style_axes(axes[i])
    fig.suptitle("Feature & Target Distributions")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(FIG_DIR, "01_distributions.png")
    plt.savefig(out)
    plt.close()
    print(f"[eda] Saved {out}")


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = FEATURES + TARGETS
    corr = df[cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0, square=True,
                linewidths=0.5, linecolor="white")
    plt.title("Correlation Heatmap — Geometry vs. Effective Properties")
    plt.figtext(0.5, 0.01, PROVENANCE_ANALYTICAL, ha="center", fontsize=8.5, style="italic", color="#52514e")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(FIG_DIR, "02_correlation_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"[eda] Saved {out}")


def plot_feature_vs_target(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc0 = axes[0].scatter(df["theta_deg"], df["nu12"], c=df["hl_ratio"], cmap=SEQUENTIAL_CMAP, s=12)
    axes[0].axhline(0, color="#898781", linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("theta_deg")
    axes[0].set_ylabel("nu12")
    axes[0].set_title("Cell Angle vs. Poisson's Ratio (nu12)")
    fig.colorbar(sc0, ax=axes[0], label="hl_ratio")
    style_axes(axes[0], y_grid_only=False)

    sc1 = axes[1].scatter(df["tl_ratio"], df["E1_Es"], c=df["theta_deg"], cmap=SEQUENTIAL_CMAP, s=12)
    axes[1].set_xlabel("tl_ratio")
    axes[1].set_ylabel("E1_Es")
    axes[1].set_title("Thickness Ratio vs. Effective Modulus (E1*/Es)")
    fig.colorbar(sc1, ax=axes[1], label="theta_deg")
    style_axes(axes[1], y_grid_only=False)

    sc2 = axes[2].scatter(df["hl_ratio"], df["nu12"], c=df["theta_deg"], cmap=SEQUENTIAL_CMAP, s=12)
    axes[2].axhline(0, color="#898781", linewidth=0.9, linestyle="--")
    axes[2].set_xlabel("hl_ratio")
    axes[2].set_ylabel("nu12")
    axes[2].set_title("h/l Ratio vs. Poisson's Ratio (nu12)")
    fig.colorbar(sc2, ax=axes[2], label="theta_deg")
    style_axes(axes[2], y_grid_only=False)

    fig.suptitle("Feature vs. Target Relationships")
    provenance_caption(fig, PROVENANCE_ANALYTICAL)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out = os.path.join(FIG_DIR, "03_feature_vs_target.png")
    plt.savefig(out)
    plt.close()
    print(f"[eda] Saved {out}")


def plot_geometry_property_surface(tl_fixed: float = 0.10):
    """
    3D surface of the ground-truth analytical relationship nu12(theta, h/l)
    at a fixed strut-thickness ratio, so the auxetic mechanism (how the
    re-entrant angle and h/l jointly drive negative Poisson's ratio) is
    visible as a single continuous surface rather than a sampled scatter.
    """
    theta_grid = np.linspace(*THETA_DEG_RANGE, 160)
    hl_grid = np.linspace(*HL_RATIO_RANGE, 160)
    TH, HL = np.meshgrid(theta_grid, hl_grid)
    _, _, NU12, _, _ = compute_properties(TH, HL, np.full_like(TH, tl_fixed))

    # Near sin(theta) -> 0 (theta -> 0) the closed-form nu12 has a removable-
    # looking but numerically huge spike; clip to the dataset's realistic
    # range so the surface isn't dominated by a single near-singular point.
    NU12 = np.clip(NU12, -8.0, 2.0)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(TH, HL, NU12, cmap=SEQUENTIAL_CMAP, linewidth=0, antialiased=True, rcount=160, ccount=160)
    ax.set_zlim(-8.0, 2.0)
    ax.contour(TH, HL, NU12, levels=[0], colors="#e34948", linewidths=2, offset=-8.0)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("h/l ratio")
    ax.set_zlabel("nu12")
    ax.set_title(f"Analytical Geometry -> Poisson's Ratio Surface (t/l = {tl_fixed:.2f})")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="nu12")
    provenance_caption(fig, PROVENANCE_ANALYTICAL + " | red contour marks nu12 = 0 (auxetic / non-auxetic boundary)")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(FIG_DIR, "04_geometry_property_surface.png")
    plt.savefig(out)
    plt.close()
    print(f"[eda] Saved {out}")


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_data()
    print(df.describe().T)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)
    plot_geometry_property_surface()


if __name__ == "__main__":
    main()
