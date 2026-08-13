"""
dynamic_charts.py

Live, interactive (Plotly) equivalents of every static figure the batch
pipeline (eda.py, evaluation.py, sensitivity.py, real_data.py) saves to
results/figures/. Used by app/app.py so the dashboard's "All Figures" tab
renders real charts computed in-process — hover tooltips, zoom, pan — instead
of embedding pre-rendered PNGs.

Every function here is a pure builder: given data/models already computed
elsewhere (and cached by Streamlit), it returns a plotly.graph_objects.Figure.
No Streamlit calls live in this module, so it stays testable/importable
outside the app.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_dataset import compute_properties, THETA_DEG_RANGE, HL_RATIO_RANGE, TL_RATIO_RANGE
from plot_style import CATEGORICAL

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
MUTED = "#898781"
DIVERGING_SCALE = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
SEQUENTIAL_SCALE = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]


def _base_layout(fig: go.Figure, title: str, height: int = 420, legend: bool = False):
    fig.update_layout(
        title=title,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=70 if not legend else 100, b=10),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0) if legend else None,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _grid_shape(n: int, max_cols: int = 5):
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


# --------------------------------------------------------------------------
# Distributions / correlation / feature-vs-target (shared shape, either dataset)
# --------------------------------------------------------------------------

def fig_distributions(df: pd.DataFrame, cols: list, title: str) -> go.Figure:
    rows, ncols = _grid_shape(len(cols), max_cols=5)
    fig = make_subplots(rows=rows, cols=ncols, subplot_titles=cols)
    for i, col in enumerate(cols):
        r, c = i // ncols + 1, i % ncols + 1
        fig.add_trace(
            go.Histogram(x=df[col], nbinsx=30, marker_color=CATEGORICAL[0],
                         marker_line_color="white", marker_line_width=0.5),
            row=r, col=c,
        )
    _base_layout(fig, title, height=240 * rows + 60)
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, cols: list, title: str) -> go.Figure:
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols, colorscale=DIVERGING_SCALE, zmid=0,
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="r"),
    ))
    _base_layout(fig, title, height=max(420, 40 * len(cols)))
    fig.update_yaxes(autorange="reversed")
    return fig


# --------------------------------------------------------------------------
# Analytical dataset (auxetic geometry)
# --------------------------------------------------------------------------

def fig_feature_vs_target(df: pd.DataFrame, ref: tuple = None) -> go.Figure:
    """ref, if given, is (theta_ref, hl_ref, tl_ref) — overlaid as a star marker
    on each panel at its corresponding analytical value, so the chart visibly
    updates as the reference-geometry controls change."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "theta vs. nu12 (color=h/l)", "t/l vs. E1*/Es (color=theta)", "h/l vs. nu12 (color=theta)",
    ])
    fig.add_trace(go.Scatter(
        x=df["theta_deg"], y=df["nu12"], mode="markers",
        marker=dict(color=df["hl_ratio"], colorscale=SEQUENTIAL_SCALE, size=5, opacity=0.6,
                    colorbar=dict(title="h/l", x=0.30)),
        hovertemplate="theta=%{x:.1f}<br>nu12=%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["tl_ratio"], y=df["E1_Es"], mode="markers",
        marker=dict(color=df["theta_deg"], colorscale=SEQUENTIAL_SCALE, size=5, opacity=0.6,
                    colorbar=dict(title="theta", x=0.65)),
        hovertemplate="t/l=%{x:.3f}<br>E1*/Es=%{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=df["hl_ratio"], y=df["nu12"], mode="markers",
        marker=dict(color=df["theta_deg"], colorscale=SEQUENTIAL_SCALE, size=5, opacity=0.6,
                    colorbar=dict(title="theta", x=1.0)),
        hovertemplate="h/l=%{x:.2f}<br>nu12=%{y:.3f}<extra></extra>",
    ), row=1, col=3)

    if ref is not None:
        theta_r, hl_r, tl_r = ref
        E1_r, _, nu12_r, _, _ = compute_properties(np.array([theta_r]), np.array([hl_r]), np.array([tl_r]))
        star = dict(color=CATEGORICAL[7], size=15, symbol="star", line=dict(color="white", width=1))
        fig.add_trace(go.Scatter(x=[theta_r], y=[nu12_r[0]], mode="markers", marker=star, name="reference"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[tl_r], y=[E1_r[0]], mode="markers", marker=star, name="reference", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[hl_r], y=[nu12_r[0]], mode="markers", marker=star, name="reference", showlegend=False), row=1, col=3)

    _base_layout(fig, "Feature vs. Target Relationships", height=420, legend=ref is not None)
    return fig


_SURFACE_TARGET_INDEX = {"E1_Es": 0, "E2_Es": 1, "nu12": 2, "nu21": 3}


def fig_geometry_surface(tl_fixed: float = 0.10, target: str = "nu12",
                          clip_range: tuple = None, ref: tuple = None) -> go.Figure:
    """3D surface of the analytical target over (theta, h/l) at a fixed t/l.
    target: one of E1_Es, E2_Es, nu12, nu21. clip_range, if given, clamps the
    z-range (these closed-form functions have a real near-singular ridge where
    h/l + sin(theta) ~ 0). ref, if given, is (theta_ref, hl_ref) and is plotted
    as a marker on the surface at its live z-value."""
    theta_grid = np.linspace(*THETA_DEG_RANGE, 120)
    hl_grid = np.linspace(*HL_RATIO_RANGE, 120)
    TH, HL = np.meshgrid(theta_grid, hl_grid)
    props = compute_properties(TH, HL, np.full_like(TH, tl_fixed))
    Z = props[_SURFACE_TARGET_INDEX[target]]

    if clip_range is not None:
        lo, hi = clip_range
        Z = np.clip(Z, lo, hi)
    else:
        lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))

    surface = go.Surface(x=theta_grid, y=hl_grid, z=Z, colorscale=SEQUENTIAL_SCALE, colorbar=dict(title=target))
    if target in ("nu12", "nu21") and lo < 0 < hi:
        surface.update(contours=dict(z=dict(show=True, start=0, end=0, size=1, color="#e34948", width=4)))
    fig = go.Figure(surface)

    subtitle = f"Geometry -> {target} Surface (t/l = {tl_fixed:.2f})"
    if target in ("nu12", "nu21"):
        subtitle += "; red contour = auxetic/non-auxetic boundary"

    if ref is not None:
        theta_r, hl_r = ref
        props_r = compute_properties(np.array([theta_r]), np.array([hl_r]), np.array([tl_fixed]))
        z_r = float(np.clip(props_r[_SURFACE_TARGET_INDEX[target]][0], lo, hi))
        fig.add_trace(go.Scatter3d(
            x=[theta_r], y=[hl_r], z=[z_r], mode="markers",
            marker=dict(size=6, color="#e34948", symbol="diamond"), name="reference",
        ))

    fig.update_layout(
        title=subtitle,
        scene=dict(
            xaxis_title="theta (deg)", yaxis_title="h/l ratio", zaxis_title=target,
            zaxis=dict(range=[lo, hi]),
        ),
        paper_bgcolor=SURFACE, margin=dict(l=10, r=10, t=60, b=10), height=560, showlegend=False,
    )
    return fig


def fig_model_comparison(cv_df: pd.DataFrame, targets: list, r2_col: str, std_col: str, title: str) -> go.Figure:
    models_order = ["linear", "ridge", "random_forest", "gradient_boosting"]
    fig = go.Figure()
    x = np.arange(len(targets))
    width = 0.8 / len(models_order)
    for i, model_name in enumerate(models_order):
        sub = cv_df[cv_df["model"] == model_name].set_index("target").reindex(targets)
        fig.add_trace(go.Bar(
            x=targets, y=sub[r2_col], name=model_name,
            error_y=dict(type="data", array=sub[std_col]),
            marker_color=CATEGORICAL[i],
        ))
    fig.update_layout(barmode="group", yaxis_title="5-fold cross-validated R2")
    _base_layout(fig, title, height=460, legend=True)
    return fig


def fig_parity(models: dict, X_test: pd.DataFrame, y_test: pd.DataFrame, targets: list) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(targets), subplot_titles=targets)
    for i, target in enumerate(targets):
        y_true = y_test[target]
        y_pred = models["random_forest"][target].predict(X_test)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker=dict(color=CATEGORICAL[2], size=5, opacity=0.55),
            hovertemplate="actual=%{x:.4f}<br>predicted=%{y:.4f}<extra></extra>",
        ), row=1, col=i + 1)
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines", line=dict(color=MUTED, dash="dash", width=1.5),
        ), row=1, col=i + 1)
        fig.update_xaxes(title_text="Actual", row=1, col=i + 1)
        fig.update_yaxes(title_text="Predicted (RF)", row=1, col=i + 1)
    _base_layout(fig, "Actual vs. Predicted — Random Forest Surrogate", height=380)
    return fig


def fig_residuals(models: dict, X_test: pd.DataFrame, y_test: pd.DataFrame, targets: list) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(targets), subplot_titles=targets)
    for i, target in enumerate(targets):
        y_true = y_test[target]
        y_pred = models["random_forest"][target].predict(X_test)
        residual = y_true - y_pred
        fig.add_trace(go.Scatter(
            x=y_pred, y=residual, mode="markers",
            marker=dict(color=CATEGORICAL[0], size=5, opacity=0.55),
            hovertemplate="predicted=%{x:.4f}<br>residual=%{y:.4f}<extra></extra>",
        ), row=1, col=i + 1)
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=i + 1)
        fig.update_xaxes(title_text="Predicted", row=1, col=i + 1)
        fig.update_yaxes(title_text="Residual", row=1, col=i + 1)
    _base_layout(fig, "Residual Plots — Random Forest Surrogate (test set)", height=380)
    return fig


def fig_feature_importance(imp_df: pd.DataFrame, targets: list) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(targets), subplot_titles=targets)
    for i, target in enumerate(targets):
        sub = imp_df[imp_df["target"] == target].sort_values("importance_mean")
        fig.add_trace(go.Bar(
            x=sub["importance_mean"], y=sub["feature"], orientation="h",
            error_x=dict(type="data", array=sub["importance_std"]),
            marker_color=CATEGORICAL[1],
        ), row=1, col=i + 1)
        fig.update_xaxes(title_text="Permutation importance", row=1, col=i + 1)
    _base_layout(fig, "Permutation Feature Importance (Random Forest)", height=380)
    return fig


def fig_oat_sensitivity(theta_ref: float = -35.0, hl_ref: float = 1.4, tl_ref: float = 0.10) -> go.Figure:
    """Sweeps each parameter individually around the given reference geometry,
    holding the other two fixed — a diamond marker shows the reference point
    on each curve, so the whole chart redraws as the reference sliders move."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "theta -> nu12", "h/l -> nu12", "t/l -> E1*/Es",
    ])
    marker = dict(color="#e34948", size=11, symbol="diamond", line=dict(color="white", width=1))

    theta_range = np.linspace(*THETA_DEG_RANGE, 200)
    _, _, nu12_a, _, _ = compute_properties(theta_range, hl_ref, tl_ref)
    _, _, nu12_a_ref, _, _ = compute_properties(np.array([theta_ref]), np.array([hl_ref]), np.array([tl_ref]))
    fig.add_trace(go.Scatter(x=theta_range, y=nu12_a, mode="lines", line=dict(color=CATEGORICAL[0], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[theta_ref], y=[nu12_a_ref[0]], mode="markers", marker=marker), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=1)

    hl_range = np.linspace(*HL_RATIO_RANGE, 200)
    _, _, nu12_b, _, _ = compute_properties(theta_ref, hl_range, tl_ref)
    _, _, nu12_b_ref, _, _ = compute_properties(np.array([theta_ref]), np.array([hl_ref]), np.array([tl_ref]))
    fig.add_trace(go.Scatter(x=hl_range, y=nu12_b, mode="lines", line=dict(color=CATEGORICAL[1], width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[hl_ref], y=[nu12_b_ref[0]], mode="markers", marker=marker), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=2)
    # h/l range crosses a near-singular geometry (h/l + sin(theta) ~ 0) for
    # some reference angles; clamp the displayed range so that spike doesn't
    # dwarf the rest of the curve (hover still shows exact values).
    fig.update_yaxes(range=[-10, 5], row=1, col=2)

    tl_range = np.linspace(*TL_RATIO_RANGE, 200)
    E1_c, _, _, _, _ = compute_properties(theta_ref, hl_ref, tl_range)
    E1_c_ref, _, _, _, _ = compute_properties(np.array([theta_ref]), np.array([hl_ref]), np.array([tl_ref]))
    fig.add_trace(go.Scatter(x=tl_range, y=E1_c, mode="lines", line=dict(color=CATEGORICAL[2], width=2.5)), row=1, col=3)
    fig.add_trace(go.Scatter(x=[tl_ref], y=[E1_c_ref[0]], mode="markers", marker=marker), row=1, col=3)

    fig.update_xaxes(title_text="theta (deg)", row=1, col=1)
    fig.update_xaxes(title_text="h/l ratio", row=1, col=2)
    fig.update_xaxes(title_text="t/l ratio", row=1, col=3)
    fig.update_yaxes(title_text="nu12", row=1, col=1)
    fig.update_yaxes(title_text="nu12", row=1, col=2)
    fig.update_yaxes(title_text="E1*/Es", row=1, col=3)
    _base_layout(
        fig,
        f"One-at-a-Time Sensitivity Sweeps (reference: theta={theta_ref:.0f} deg, h/l={hl_ref:.2f}, t/l={tl_ref:.2f})",
        height=380,
    )
    return fig


def fig_parity_residual_single(y_true, y_pred, title: str) -> go.Figure:
    """One model x one target, actual-vs-predicted + residuals side by side —
    the pairing used by the Model Playground section, redrawn on every
    dataset/target/model selection change."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Actual vs. Predicted", "Residuals"])
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=CATEGORICAL[2], size=7, opacity=0.65),
        hovertemplate="actual=%{x:.4g}<br>predicted=%{y:.4g}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color=MUTED, dash="dash", width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=y_pred, y=residual, mode="markers",
        marker=dict(color=CATEGORICAL[0], size=7, opacity=0.65),
        hovertemplate="predicted=%{x:.4g}<br>residual=%{y:.4g}<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=2)

    fig.update_xaxes(title_text="Actual", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    _base_layout(fig, title, height=420)
    return fig


# --------------------------------------------------------------------------
# Real dataset (Kaggle AM process parameters)
# --------------------------------------------------------------------------

def fig_real_material_pattern(df: pd.DataFrame, targets: list) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(targets), subplot_titles=targets)
    colors = {"abs": CATEGORICAL[0], "pla": CATEGORICAL[1]}
    for i, target in enumerate(targets):
        for material in ["abs", "pla"]:
            sub = df[df["material"] == material]
            fig.add_trace(go.Box(
                x=sub["infill_pattern"], y=sub[target], name=material,
                marker_color=colors[material], legendgroup=material,
                showlegend=(i == 0),
            ), row=1, col=i + 1)
        fig.update_xaxes(title_text="infill pattern", row=1, col=i + 1)
    fig.update_layout(boxmode="group")
    _base_layout(fig, "Real Dataset — Material x Infill Pattern Effects", height=420, legend=True)
    return fig


def fig_real_process_vs_strength(df: pd.DataFrame, material: str = "pla") -> go.Figure:
    """material: 'pla', 'abs', or 'all'. When a single material is chosen, points
    are colored by a relevant continuous process parameter; with 'all', by
    material category instead — so the chart's encoding itself changes with
    the filter, not just which rows are shown."""
    sub = df if material == "all" else df[df["material"] == material]
    label = "All materials" if material == "all" else material.upper()
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"{label}: infill density vs. tensile strength", f"{label}: print speed vs. elongation",
    ])

    if material == "all":
        for i, (mat, color) in enumerate([("abs", CATEGORICAL[0]), ("pla", CATEGORICAL[1])]):
            m = sub[sub["material"] == mat]
            fig.add_trace(go.Scatter(
                x=m["infill_density"], y=m["tension_strength"], mode="markers", name=mat,
                marker=dict(color=color, size=9), legendgroup=mat,
                hovertemplate="infill=%{x}%<br>strength=%{y} MPa<extra>" + mat + "</extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=m["print_speed"], y=m["elongation"], mode="markers", name=mat, showlegend=False,
                marker=dict(color=color, size=9), legendgroup=mat,
                hovertemplate="speed=%{x} mm/s<br>elongation=%{y}%<extra>" + mat + "</extra>",
            ), row=1, col=2)
        _base_layout(fig, "Real Process Parameters vs. Printed-Part Properties", height=420, legend=True)
    else:
        fig.add_trace(go.Scatter(
            x=sub["infill_density"], y=sub["tension_strength"], mode="markers",
            marker=dict(color=sub["layer_height"], colorscale=SEQUENTIAL_SCALE, size=9,
                        colorbar=dict(title="layer height", x=0.42)),
            hovertemplate="infill=%{x}%<br>strength=%{y} MPa<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sub["print_speed"], y=sub["elongation"], mode="markers",
            marker=dict(color=sub["nozzle_temperature"], colorscale=SEQUENTIAL_SCALE, size=9,
                        colorbar=dict(title="nozzle temp", x=1.0)),
            hovertemplate="speed=%{x} mm/s<br>elongation=%{y}%<extra></extra>",
        ), row=1, col=2)
        _base_layout(fig, f"Real Process Parameters vs. Printed-Part Properties ({label})", height=420)

    fig.update_xaxes(title_text="infill density (%)", row=1, col=1)
    fig.update_yaxes(title_text="tensile strength (MPa)", row=1, col=1)
    fig.update_xaxes(title_text="print speed (mm/s)", row=1, col=2)
    fig.update_yaxes(title_text="elongation (%)", row=1, col=2)
    return fig
