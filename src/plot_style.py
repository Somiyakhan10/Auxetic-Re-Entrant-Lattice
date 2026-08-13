"""
plot_style.py

Shared, validated color palette and matplotlib style applied to every figure
in this project, so plots read as one consistent system instead of ad-hoc
per-script defaults.

Palette: fixed-order categorical hues (CVD-safe adjacent pairs), a single-hue
sequential ramp for magnitude, and a blue<->red diverging pair for signed
quantities (e.g. Poisson's ratio, correlation). Do not reorder the categorical
list and do not use more than 3 slots on scatter/bubble charts where every
pair of colors can appear adjacent at once.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Fixed-order categorical palette (blue, orange, aqua, yellow, magenta, green, violet, red)
CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# Single-hue sequential ramp (light -> dark), for continuous magnitude encoding
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

# Diverging pair (blue <-> red) for signed quantities, neutral midpoint = gray
DIVERGING_CMAP = "coolwarm"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def set_style():
    """Apply the shared house style to matplotlib's rcParams. Call once per script."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
    })


def cat_color(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]


def style_axes(ax, y_grid_only: bool = True):
    ax.set_axisbelow(True)
    if y_grid_only:
        ax.grid(True, axis="y", alpha=0.8)
        ax.grid(False, axis="x")


PROVENANCE_ANALYTICAL = "Illustrative computational (analytical) dataset — not experimental data"
PROVENANCE_REAL = "Real experimental dataset — Kaggle “3D Printer Dataset for Mechanical Engineers”"


def provenance_caption(fig, text: str):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=8.5, color=INK_MUTED, style="italic")
