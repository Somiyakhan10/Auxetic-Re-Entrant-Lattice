"""
app.py

Streamlit dashboard for the auxetic unit-cell property surrogate, plus a
real-data explorer built on a real, publicly available Kaggle dataset of
FDM 3D-printing process parameters and measured PLA/ABS mechanical
properties.

Run with:
    streamlit run app/app.py

Nine tabs, each scoped to one kind of chart rather than one long scroll:

Analytical dataset (auxetic geometry):
  1. Forward Prediction — geometry sliders -> predicted E1*/Es, nu12
     (analytical + RF surrogate), a LIVE re-entrant unit-cell drawing, and a
     live sensitivity curve with the current point marked.
  2. Inverse Design — target nu12 (+ optional E1*/Es) -> recommended
     geometry, plotted against the full dataset.
  3. Geometry Explorer — a shared reference-geometry control (theta/h·l/t·l
     sliders + target-property picker) that redraws the 3D property surface,
     the one-at-a-time sensitivity sweeps, and a reference marker on the
     feature-vs-target scatter, all together.
  4. Analytical EDA — feature/target distributions and correlation heatmap.
  5. Analytical Models — 5-fold CV model comparison, actual-vs-predicted,
     residuals, and permutation feature importance (Random Forest).

Real dataset (Kaggle AM process parameters):
  6. Real Process Predictor — dataset snapshot + sliders for real process
     settings -> live predicted tensile strength/elongation/roughness, with
     a scatter showing where that prediction falls against real measurements.
  7. Real Data EDA — material/infill-pattern filters that redraw the
     distributions, correlation heatmap, and process-vs-property scatter.
  8. Real Data Models — 5-fold CV model comparison and permutation feature
     importance.

Cross-dataset:
  9. Model Playground — pick any (dataset, target, model) combination from
     three dropdowns and its actual-vs-predicted + residual chart, R2, and
     MAE redraw on the spot.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_dataset import compute_properties, THETA_DEG_RANGE, HL_RATIO_RANGE, TL_RATIO_RANGE
from model import load_split, train_models, FEATURES, TARGETS
from inverse_design import analytical_inverse_search, surrogate_inverse_search
from sensitivity import permutation_importance_analysis
import real_data
import dynamic_charts as dc
from plot_style import CATEGORICAL

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "auxetic_dataset.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")
ANALYTICAL_CV_PATH = os.path.join(REPORT_DIR, "cv_metrics_summary.csv")

st.set_page_config(page_title="Auxetic AM Surrogate", layout="wide")

st.title("Auxetic Re-Entrant Lattice — Computational Surrogate & Inverse Design")
st.caption(
    "Prototype computational tool combining an **illustrative analytical dataset** "
    "(closed-form re-entrant honeycomb beam theory) with a **real experimental "
    "dataset** (Kaggle: '3D Printer Dataset for Mechanical Engineers') for the "
    "AM process-parameter side of the problem. Every chart in this app is live — "
    "rendered in-process from the current data/models, not a static image."
)


@st.cache_resource
def get_models():
    X_train, X_test, y_train, y_test = load_split()
    models = train_models(X_train, y_train)
    return models, X_train, X_test, y_train, y_test


@st.cache_data
def load_analytical_df():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_analytical_importance(_models, _X_test, _y_test):
    return permutation_importance_analysis(_models, _X_test, _y_test)


@st.cache_data
def load_analytical_cv():
    if os.path.exists(ANALYTICAL_CV_PATH):
        return pd.read_csv(ANALYTICAL_CV_PATH)
    return None


@st.cache_resource
def get_real_data_models():
    df = real_data.load_clean()
    imp_df, fitted_models = real_data.feature_importance_analysis(df)
    cv_df, cv_preds = real_data.cross_validate_models(df)
    return df, fitted_models, cv_df, imp_df, cv_preds


def build_unit_cell_figure(theta_deg: float, hl_ratio: float, tl_ratio: float,
                            n_cols: int = 3, n_rows: int = 2) -> go.Figure:
    """
    Draws a schematic re-entrant honeycomb lattice (a tiled 'bowtie' unit
    cell) from the current geometry sliders. Strut length l is normalized
    to 1; h = hl_ratio, t = tl_ratio, purely for visualization. Line width
    scales with t/l so thickness changes are visible too. Not to exact
    literature scale — a qualitative sketch of how the re-entrant angle
    pinches the cell inward (the auxetic mechanism), not a measurement.
    """
    theta = np.radians(theta_deg)
    l = 1.0
    h = hl_ratio * l
    c, s = np.cos(theta), np.sin(theta)
    cell_w = 2 * l * c
    cell_h = h

    node_path = [
        (0.0, 0.0), (0.0, h), (l * c, h + l * s), (2 * l * c, h),
        (2 * l * c, 0.0), (l * c, -l * s), (0.0, 0.0),
    ]

    xs, ys = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            ox, oy = col * cell_w, row * cell_h
            for x, y in node_path:
                xs.append(x + ox)
                ys.append(y + oy)
            xs.append(None)
            ys.append(None)

    line_width = 1.5 + 14 * tl_ratio

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=CATEGORICAL[0], width=line_width),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Live re-entrant unit-cell geometry",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        margin=dict(l=10, r=10, t=40, b=10), height=340,
        showlegend=False,
    )
    return fig


def build_theta_sweep_figure(theta_deg_current: float, hl_ratio: float, tl_ratio: float) -> go.Figure:
    theta_range = np.linspace(*THETA_DEG_RANGE, 200)
    _, _, nu12_range, _, _ = compute_properties(theta_range, hl_ratio, tl_ratio)
    _, _, nu12_now, _, _ = compute_properties(
        np.array([theta_deg_current]), np.array([hl_ratio]), np.array([tl_ratio])
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta_range, y=nu12_range, mode="lines",
        line=dict(color=CATEGORICAL[0], width=2.5),
        name="nu12(theta)", hovertemplate="theta=%{x:.1f} deg<br>nu12=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#898781", line_width=1)
    fig.add_trace(go.Scatter(
        x=[theta_deg_current], y=[nu12_now[0]], mode="markers",
        marker=dict(color=CATEGORICAL[7], size=14, symbol="star", line=dict(color="white", width=1)),
        name="current setting", hovertemplate="theta=%{x:.1f} deg<br>nu12=%{y:.3f}<extra>current</extra>",
    ))
    fig.update_layout(
        title=f"Live sensitivity: nu12 vs. theta (h/l={hl_ratio:.2f}, t/l={tl_ratio:.2f} held fixed)",
        xaxis_title="theta (deg)", yaxis_title="nu12",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        margin=dict(l=10, r=10, t=40, b=10), height=340, showlegend=False,
    )
    return fig


def build_inverse_design_figure(df: pd.DataFrame, analytical: dict, surrogate: dict, target_nu12: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["theta_deg"], y=df["nu12"], mode="markers",
        marker=dict(color=df["hl_ratio"], colorscale="Blues", size=5, opacity=0.5,
                    colorbar=dict(title="h/l")),
        name="dataset", hovertemplate="theta=%{x:.1f}<br>nu12=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=target_nu12, line_dash="dash", line_color=CATEGORICAL[7],
                  annotation_text="target nu12", annotation_position="bottom right")
    fig.add_trace(go.Scatter(
        x=[analytical["theta_deg"]], y=[analytical["predicted_nu12"]], mode="markers",
        marker=dict(color=CATEGORICAL[1], size=16, symbol="diamond", line=dict(color="white", width=1)),
        name="analytical search result",
    ))
    fig.add_trace(go.Scatter(
        x=[surrogate["theta_deg"]], y=[surrogate["predicted_nu12"]], mode="markers",
        marker=dict(color=CATEGORICAL[7], size=16, symbol="star", line=dict(color="white", width=1)),
        name="surrogate search result",
    ))
    fig.update_layout(
        title="Where the recommended geometries land on the full dataset",
        xaxis_title="theta (deg)", yaxis_title="nu12",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        margin=dict(l=10, r=10, t=90, b=10), height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    return fig


def build_real_scatter_figure(real_df: pd.DataFrame, query_infill: float, query_strength: float, material: str) -> go.Figure:
    fig = px.scatter(
        real_df, x="infill_density", y="tension_strength", color="material",
        color_discrete_map={"abs": CATEGORICAL[0], "pla": CATEGORICAL[1]},
        hover_data=["layer_height", "infill_pattern"],
    )
    fig.add_trace(go.Scatter(
        x=[query_infill], y=[query_strength], mode="markers",
        marker=dict(color=CATEGORICAL[7], size=18, symbol="star", line=dict(color="white", width=1.5)),
        name="your prediction",
    ))
    fig.update_layout(
        title="Your prediction vs. real measured samples",
        xaxis_title="infill density (%)", yaxis_title="tensile strength (MPa)",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        margin=dict(l=10, r=10, t=90, b=10), height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    return fig


models, X_train, X_test, y_train, y_test = get_models()
analytical_df = load_analytical_df()

(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9,
) = st.tabs([
    "Forward Prediction", "Inverse Design", "Geometry Explorer",
    "Analytical EDA", "Analytical Models",
    "Real Process Predictor", "Real Data EDA", "Real Data Models",
    "Model Playground",
])

with tab1:
    st.caption("Illustrative computational (analytical) dataset — not experimental data.")
    st.subheader("Set unit-cell geometry")
    col1, col2, col3 = st.columns(3)
    theta = col1.slider("Cell angle, theta (deg)", float(THETA_DEG_RANGE[0]), float(THETA_DEG_RANGE[1]), -35.0)
    hl = col2.slider("h/l ratio", float(HL_RATIO_RANGE[0]), float(HL_RATIO_RANGE[1]), 1.4)
    tl = col3.slider("t/l ratio", float(TL_RATIO_RANGE[0]), float(TL_RATIO_RANGE[1]), 0.10)

    E1_a, E2_a, nu12_a, nu21_a, rel_density = compute_properties(
        np.array([theta]), np.array([hl]), np.array([tl])
    )

    X_query = pd.DataFrame({
        "theta_deg": [theta], "hl_ratio": [hl], "tl_ratio": [tl], "rel_density": [rel_density[0]]
    })[FEATURES]

    st.subheader("Predicted properties")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("E1*/Es (analytical)", f"{E1_a[0]:.5f}")
    c2.metric("E1*/Es (RF surrogate)", f"{models['random_forest']['E1_Es'].predict(X_query)[0]:.5f}")
    c3.metric("nu12 (analytical)", f"{nu12_a[0]:.3f}")
    c4.metric("nu12 (RF surrogate)", f"{models['random_forest']['nu12'].predict(X_query)[0]:.3f}")

    st.metric("Approx. relative density (rho*/rho_s)", f"{rel_density[0]:.3f}")

    if nu12_a[0] < 0:
        st.success("This geometry is AUXETIC (negative Poisson's ratio).")
    else:
        st.info("This geometry is NOT auxetic (non-negative Poisson's ratio).")

    st.subheader("Live charts")
    gcol1, gcol2 = st.columns(2)
    gcol1.plotly_chart(build_unit_cell_figure(theta, hl, tl), use_container_width=True)
    gcol2.plotly_chart(build_theta_sweep_figure(theta, hl, tl), use_container_width=True)

with tab2:
    st.caption("Illustrative computational (analytical) dataset — not experimental data.")
    st.subheader("Specify a target property")
    colA, colB, colC = st.columns(3)
    target_nu = colA.number_input("Target nu12", value=-0.3, step=0.05, format="%.2f")
    use_E_target = colB.checkbox("Also target E1*/Es?", value=False)
    target_E = colC.number_input("Target E1*/Es", value=0.01, step=0.001, format="%.4f", disabled=not use_E_target)

    if st.button("Find recommended geometry"):
        te = target_E if use_E_target else None
        analytical = analytical_inverse_search(target_nu, te)
        surrogate = surrogate_inverse_search(models, target_nu, te)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Analytical grid search (ground truth):**")
            st.json(analytical)
        with col2:
            st.markdown("**Random Forest surrogate search:**")
            st.json(surrogate)

        st.plotly_chart(
            build_inverse_design_figure(analytical_df, analytical, surrogate, target_nu),
            use_container_width=True,
        )
    else:
        st.info("Set a target and click **Find recommended geometry** to see results and a live plot.")

with tab6:
    st.caption(
        "Real experimental dataset — Kaggle *'3D Printer Dataset for Mechanical "
        "Engineers'* (Ultimaker S5, tensile-tested PLA/ABS coupons). Standard "
        "grid/honeycomb infill — NOT re-entrant auxetic geometry. Included to "
        "ground the AM-process side of this project in real measurements."
    )
    real_df, real_models, real_cv_df, real_imp_df, real_cv_preds = get_real_data_models()

    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples (after cleaning)", len(real_df))
    c2.metric("PLA samples", int((real_df["material"] == "pla").sum()))
    c3.metric("ABS samples", int((real_df["material"] == "abs").sum()))
    st.dataframe(real_df.head(10), use_container_width=True)

    st.subheader("Predict tensile strength from real process settings")
    col1, col2, col3, col4 = st.columns(4)
    layer_height = col1.slider("Layer height (mm)", 0.02, 0.20, 0.10, step=0.01)
    infill_density = col2.slider("Infill density (%)", 0, 100, 50, step=5)
    nozzle_temp = col3.slider("Nozzle temperature (C)", 195, 250, 220, step=5)
    print_speed = col4.slider("Print speed (mm/s)", 40, 120, 60, step=5)

    col5, col6, col7, col8 = st.columns(4)
    wall_thickness = col5.slider("Wall thickness (mm)", 1, 12, 6)
    bed_temp = col6.slider("Bed temperature (C)", 55, 100, 70, step=5)
    fan_speed = col7.slider("Fan speed (%)", 0, 100, 50, step=5)
    material = col8.selectbox("Material", ["pla", "abs"])
    infill_pattern = st.selectbox("Infill pattern", ["grid", "honeycomb"])

    X_real_query = pd.DataFrame({
        "layer_height": [layer_height], "wall_thickness": [wall_thickness],
        "infill_density": [infill_density], "nozzle_temperature": [nozzle_temp],
        "bed_temperature": [bed_temp], "print_speed": [print_speed], "fan_speed": [fan_speed],
        "infill_pattern_honeycomb": [1 if infill_pattern == "honeycomb" else 0],
        "material_pla": [1 if material == "pla" else 0],
    })[real_data.FEATURES]

    pred_strength = real_models["tension_strength"].predict(X_real_query)[0]
    pred_elongation = real_models["elongation"].predict(X_real_query)[0]
    pred_roughness = real_models["roughness"].predict(X_real_query)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted tensile strength (MPa)", f"{pred_strength:.1f}")
    c2.metric("Predicted elongation (%)", f"{pred_elongation:.2f}")
    c3.metric("Predicted roughness (micron)", f"{pred_roughness:.0f}")
    st.caption(f"Random Forest fit on the full cleaned dataset (n={len(real_df)}). Small sample — treat as directional, not precise.")

    st.plotly_chart(
        build_real_scatter_figure(real_df, infill_density, pred_strength, material),
        use_container_width=True,
    )

with tab3:
    st.caption(
        "Illustrative computational (analytical) dataset — not experimental data. "
        "**Live** — the reference-geometry controls below redraw every chart on this tab."
    )
    st.subheader("Reference geometry")
    rc1, rc2, rc3, rc4 = st.columns(4)
    theta_ref = rc1.slider("theta_ref (deg)", float(THETA_DEG_RANGE[0]), float(THETA_DEG_RANGE[1]), -35.0, key="theta_ref")
    hl_ref = rc2.slider("h/l_ref", float(HL_RATIO_RANGE[0]), float(HL_RATIO_RANGE[1]), 1.4, key="hl_ref")
    tl_ref = rc3.slider("t/l_ref", float(TL_RATIO_RANGE[0]), float(TL_RATIO_RANGE[1]), 0.10, key="tl_ref")
    surface_target = rc4.selectbox("3D surface property", ["nu12", "E1_Es", "E2_Es", "nu21"], key="surf_target")

    st.subheader("Feature vs. target relationships")
    st.plotly_chart(
        dc.fig_feature_vs_target(analytical_df, ref=(theta_ref, hl_ref, tl_ref)),
        use_container_width=True,
    )

    st.subheader("3D geometry surface")
    surf_clip = (float(analytical_df[surface_target].min()), float(analytical_df[surface_target].max()))
    st.plotly_chart(
        dc.fig_geometry_surface(
            tl_fixed=tl_ref, target=surface_target, clip_range=surf_clip, ref=(theta_ref, hl_ref),
        ),
        use_container_width=True,
    )

    st.subheader("One-at-a-time sensitivity sweeps")
    st.plotly_chart(
        dc.fig_oat_sensitivity(theta_ref, hl_ref, tl_ref),
        use_container_width=True,
    )

with tab4:
    st.caption("Illustrative computational (analytical) dataset — not experimental data.")
    st.subheader("Feature & target distributions")
    st.plotly_chart(
        dc.fig_distributions(analytical_df, FEATURES + TARGETS, "Feature & Target Distributions"),
        use_container_width=True,
    )
    st.subheader("Correlation heatmap")
    st.plotly_chart(
        dc.fig_correlation_heatmap(analytical_df, FEATURES + TARGETS, "Correlation Heatmap"),
        use_container_width=True,
    )

with tab5:
    st.caption("Illustrative computational (analytical) dataset — not experimental data.")
    st.subheader("Surrogate model comparison")
    analytical_cv_df = load_analytical_cv()
    if analytical_cv_df is not None:
        st.plotly_chart(
            dc.fig_model_comparison(
                analytical_cv_df, TARGETS, r2_col="R2_cv_mean", std_col="R2_cv_std",
                title="Surrogate Model Comparison (5-fold CV)",
            ),
            use_container_width=True,
        )
    else:
        st.info("Run `python src/evaluation.py` once to generate cv_metrics_summary.csv for this chart.")

    st.subheader("Actual vs. predicted (Random Forest)")
    st.plotly_chart(dc.fig_parity(models, X_test, y_test, TARGETS), use_container_width=True)

    st.subheader("Residuals (Random Forest)")
    st.plotly_chart(dc.fig_residuals(models, X_test, y_test, TARGETS), use_container_width=True)

    st.subheader("Permutation feature importance")
    with st.spinner("Computing permutation feature importance..."):
        analytical_imp_df = get_analytical_importance(models, X_test, y_test)
    st.plotly_chart(dc.fig_feature_importance(analytical_imp_df, TARGETS), use_container_width=True)

with tab7:
    st.caption(
        "Real experimental dataset — Kaggle *'3D Printer Dataset for Mechanical "
        "Engineers'*. Standard grid/honeycomb infill — NOT re-entrant auxetic geometry. "
        "**Live** — the filters below redraw the distributions, correlation heatmap, and "
        "process-vs-property scatter."
    )
    real_df, real_models, real_cv_df, real_imp_df, real_cv_preds = get_real_data_models()

    st.subheader("Filters")
    fc1, fc2 = st.columns(2)
    material_filter = fc1.selectbox("Material filter", ["all", "pla", "abs"], key="real_material_filter")
    pattern_filter = fc2.selectbox("Infill pattern filter", ["all", "grid", "honeycomb"], key="real_pattern_filter")

    filtered_real_df = real_df
    if material_filter != "all":
        filtered_real_df = filtered_real_df[filtered_real_df["material"] == material_filter]
    if pattern_filter != "all":
        filtered_real_df = filtered_real_df[filtered_real_df["infill_pattern"] == pattern_filter]

    if len(filtered_real_df) < 5:
        st.warning(f"Only {len(filtered_real_df)} rows match this filter combination — showing the full dataset instead.")
        filtered_real_df = real_df
    st.caption(f"{len(filtered_real_df)} of {len(real_df)} samples match the current filter.")

    st.subheader("Feature & target distributions")
    st.plotly_chart(
        dc.fig_distributions(filtered_real_df, real_data.NUMERIC_FEATURES + real_data.TARGETS, "Feature & Target Distributions"),
        use_container_width=True,
    )
    st.subheader("Correlation heatmap")
    st.plotly_chart(
        dc.fig_correlation_heatmap(
            filtered_real_df, real_data.NUMERIC_FEATURES + ["infill_pattern_honeycomb", "material_pla"] + real_data.TARGETS,
            "Correlation Heatmap",
        ),
        use_container_width=True,
    )
    st.subheader("Material x infill pattern effects")
    st.plotly_chart(dc.fig_real_material_pattern(real_df, real_data.TARGETS), use_container_width=True)
    st.subheader("Process parameters vs. printed-part properties")
    st.plotly_chart(dc.fig_real_process_vs_strength(real_df, material=material_filter), use_container_width=True)

with tab8:
    st.caption(
        "Real experimental dataset — Kaggle *'3D Printer Dataset for Mechanical Engineers'*. "
        "Small sample (n=69) — 5-fold cross-validated metrics carry wide uncertainty."
    )
    real_df, real_models, real_cv_df, real_imp_df, real_cv_preds = get_real_data_models()

    st.subheader("Model comparison")
    st.plotly_chart(
        dc.fig_model_comparison(
            real_cv_df, real_data.TARGETS, r2_col="R2_mean", std_col="R2_std",
            title="Model Comparison (5-fold CV)",
        ),
        use_container_width=True,
    )
    st.subheader("Permutation feature importance")
    st.plotly_chart(dc.fig_feature_importance(real_imp_df, real_data.TARGETS), use_container_width=True)

with tab9:
    st.caption(
        "Pick a dataset, target, and model — the actual-vs-predicted and residual "
        "chart, R2, and MAE below redraw immediately for that exact combination."
    )
    real_df, real_models, real_cv_df, real_imp_df, real_cv_preds = get_real_data_models()

    pc1, pc2, pc3 = st.columns(3)
    dataset_choice = pc1.selectbox(
        "Dataset", ["Analytical (auxetic geometry)", "Real (Kaggle AM process)"], key="playground_dataset",
    )
    model_options = ["linear", "ridge", "random_forest", "gradient_boosting"]

    if dataset_choice.startswith("Analytical"):
        target_choice = pc2.selectbox("Target", TARGETS, key="playground_target_a")
        model_choice = pc3.selectbox("Model", model_options, key="playground_model_a")
        y_true = y_test[target_choice]
        y_pred = models[model_choice][target_choice].predict(X_test)
        eval_note = "held-out 20% test split"
    else:
        target_choice = pc2.selectbox("Target", real_data.TARGETS, key="playground_target_r")
        model_choice = pc3.selectbox("Model", model_options, key="playground_model_r")
        y_true = real_df[target_choice]
        y_pred = real_cv_preds[(model_choice, target_choice)]
        eval_note = "5-fold out-of-fold cross-validated predictions (n=69)"

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    m1, m2 = st.columns(2)
    m1.metric("R2", f"{r2:.3f}")
    m2.metric("MAE", f"{mae:.4g}")
    st.caption(f"Evaluated on: {eval_note}.")
    st.plotly_chart(
        dc.fig_parity_residual_single(y_true, y_pred, f"{model_choice} — {target_choice}"),
        use_container_width=True,
    )

st.divider()
st.caption(
    "Computational prototype only. Analytical tabs are based on idealized linear-elastic "
    "beam theory (Gibson-Ashby-type re-entrant honeycomb formulation), not validated "
    "against experimental PLA prints or FEA. The real-data tab uses genuine measurements "
    "but from standard (non-auxetic) infill coupons on a small (n<100) sample."
)
