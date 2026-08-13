# Software Requirements Specification
## Computational Design and Machine-Learning-Based Property Prediction of Re-Entrant Auxetic PLA Lattice Structures for Additive Manufacturing

# 1. Introduction

## 1.1 Purpose
This SRS defines requirements for a computational prototype that models, predicts, and inverse-designs the mechanical response of re-entrant auxetic PLA unit cells for additive manufacturing design screening.

## 1.2 Scope
The system covers: (1) analytical auxetic-geometry dataset generation, exploratory data
analysis, ML surrogate model training/evaluation, sensitivity analysis, and inverse design;
(2) ingestion, cleaning, exploratory analysis, and ML modeling of a real, publicly available
AM process-parameter dataset (Kaggle); and (3) visualization and an interactive dashboard
covering both. It excludes FEA, physical testing, and any laboratory data acquisition by the
author — the real dataset used is third-party, pre-existing, publicly published data, not
data collected for this project.

## 1.3 Background
The professor's research group studies PLA auxetic structures produced by additive manufacturing, among other AM/SMA topics. Geometry-driven negative Poisson's ratio behavior is a known, well-studied phenomenon (classical cellular-solids mechanics), making it suitable for an analytically grounded computational study.

## 1.4 Motivation
Rapid, code-only screening of auxetic geometries reduces reliance on costly experiments/FEA during early-stage design exploration.

## 1.5 Intended Users
Student researcher (self), thesis/project advisor, potentially future lab members extending the tool.

## 1.6 Definitions and Acronyms
- **SMA** – Shape Memory Alloy
- **AM** – Additive Manufacturing
- **LPBF** – Laser Powder Bed Fusion
- **PSD** – Particle Size Distribution
- **RF** – Random Forest
- **nu (ν)** – Poisson's ratio
- **E\*** – Effective (homogenized) Young's modulus of the unit cell
- **Es** – Young's modulus of the base (strut) material

# 2. Overall Description

## 2.1 Product Perspective
A standalone, offline Python analysis pipeline with an optional Streamlit front-end; not integrated with any external lab system.

## 2.2 Product Functions
**Analytical pipeline:** Generate analytical dataset → validate/clean → EDA → train ML
surrogates (4 model types) → evaluate (holdout + cross-validation) → visualize → perform
sensitivity analysis → run inverse-design queries → export results.
**Real-data pipeline:** Ingest real Kaggle AM process dataset → validate/clean → EDA → train
ML surrogates (4 model types) → cross-validate → compute feature importance → export results.
Both pipelines feed a shared interactive dashboard.

## 2.3 User Characteristics
Single user with basic Python/ML familiarity; no domain expert input required beyond parameter-range sanity checks.

## 2.4 Operating Environment
Python 3.9+, runs locally (laptop/desktop), no GPU or special hardware required, no internet dependency after package installation.

## 2.5 Constraints
- No FEA, no COMSOL/ANSYS, no physical experiments conducted by the author.
- Real data usage is limited to a public, freely licensed, already-published dataset
  (Kaggle "3D Printer Dataset for Mechanical Engineers") — no proprietary or
  hard-to-obtain datasets.
- Must clearly and separately label each dataset by its actual provenance: the
  analytical auxetic-geometry dataset as illustrative/computational, and the Kaggle
  dataset as real/experimental-but-non-auxetic. The two must never be described as
  one dataset or numerically merged.

## 2.6 Assumptions and Dependencies
- Linear-elastic, small-deformation beam theory is an adequate approximation for the illustrative study.
- Parameter ranges (theta, h/l, t/l) are chosen to be physically realistic for printable auxetic cells but are not sourced from a specific experimental dataset.
- The real Kaggle dataset's process-parameter effects on PLA/ABS tensile properties are
  assumed to be broadly informative about FDM-printed PLA behavior in general, even
  though the coupons tested used standard (grid/honeycomb) infill rather than re-entrant
  auxetic cells.
- Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, joblib, (optional) streamlit.
  The real Kaggle dataset is vendored as a local CSV (`data/raw/3d_printer_dataset.csv`)
  rather than fetched at runtime, so no network access is required to run the pipeline.

# 3. System Requirements

## 3.1 Functional Requirements
- **FR-01** The system shall generate a parametric dataset by sweeping theta, h/l, t/l over defined ranges and computing E1*/Es, E2*/Es, nu12, nu21 via closed-form equations.
- **FR-02** The system shall validate generated samples and discard physically invalid geometries.
- **FR-03** The system shall perform exploratory data analysis (distributions, correlation heatmap).
- **FR-04** The system shall split data into train/test sets.
- **FR-05** The system shall train at least two models (Linear Regression baseline, Random Forest Regressor) to predict E*/Es and nu from geometry.
- **FR-06** The system shall evaluate models using MAE, RMSE, and R2.
- **FR-07** The system shall generate feature-importance / sensitivity results.
- **FR-08** The system shall visualize actual-vs-predicted results and parameter sensitivity curves.
- **FR-09** The system shall provide an inverse-design function: given a target nu (and optionally E*/Es), return the best-matching geometry from the analytical model or a surrogate-guided search.
- **FR-10** The system shall export figures and a results summary (CSV) to a results directory.
- **FR-11** The system shall provide a Streamlit dashboard with tabs for interactive
  parameter exploration of both the analytical and real datasets.
- **FR-12** The system shall ingest a real, publicly available AM process-parameter
  dataset (Kaggle) and validate/clean it, discarding rows with non-physical or
  sensor-error values (e.g. negative roughness).
- **FR-13** The system shall perform exploratory data analysis on the real dataset
  (distributions, correlation heatmap, material/infill-pattern comparisons).
- **FR-14** The system shall train at least four regression model types (Linear,
  Ridge, Random Forest, Gradient Boosting) on the real dataset to predict tensile
  strength, elongation, and roughness from process parameters, evaluated via
  k-fold cross-validation given the small sample size.
- **FR-15** The system shall compute permutation feature importance for the real
  dataset's models and visualize it.
- **FR-16** The system shall keep the analytical and real datasets, models, and
  reports fully separate (distinct files, distinct figure numbering, distinct
  provenance captions on every figure) — never merged into one table or model.

## 3.2 Non-Functional Requirements
- **Performance:** Each pipeline stage completes in well under 2 minutes on a
  standard laptop, except analytical model comparison (`evaluation.py`), which
  cross-validates 4 model types (including Gradient Boosting) over ~2,562 samples
  and takes on the order of 1-2 minutes. The real-data pipeline (`real_data.py`,
  n=69) completes in seconds.
- **Usability:** Single entry-point scripts per stage; clear console/plot outputs.
- **Reliability:** Deterministic dataset generation (fixed random seed) for reproducibility.
- **Maintainability:** Modular code (separate files for data generation, model, evaluation, sensitivity, inverse design, visualization).
- **Reproducibility:** All randomness seeded (seed=42); requirements pinned in requirements.txt.
- **Scalability:** Dataset size parameterizable without code changes.

# 4. Data Requirements

## 4.1 Dataset A — Analytical (auxetic geometry)

**Dataset structure:** Tabular, one row per unit-cell parameter combination.
**Features:** theta_deg, hl_ratio, tl_ratio, rel_density (derived).
**Targets:** E1_Es, E2_Es, nu12, nu21.
**Data types:** All numeric (floats).
**Missing values:** None expected; invalid/non-physical rows are filtered, not imputed.
**Preprocessing:** Range filtering (percentile-based outlier removal), no scaling required for tree-based model; features used directly.
**Train/test split:** 80/20, random_state=42, plus 5-fold cross-validation over the full set.

## 4.2 Dataset B — Real (Kaggle AM process parameters)

**Source:** Kaggle "3D Printer Dataset for Mechanical Engineers"
(https://www.kaggle.com/datasets/afumetto/3dprinter), Ultimaker S5 printer,
tensile tests on a Sincotec GmbH tester. Local copy stored at
`data/raw/3d_printer_dataset.csv` for reproducibility.
**Dataset structure:** Tabular, one row per print experiment.
**Numeric features:** layer_height, wall_thickness, infill_density, nozzle_temperature,
bed_temperature, print_speed, fan_speed.
**Categorical features (one-hot encoded):** infill_pattern (grid/honeycomb),
material (abs/pla).
**Targets:** tension_strength, elongation, roughness.
**Data types:** Numeric floats/ints; two binary-encoded categoricals.
**Missing values:** None in the source; 1 of 70 rows dropped as a non-physical
sensor-error entry (negative roughness reading, 260 C bed temperature, 360 mm/s
print speed — all far outside the operating envelope of the other 69 rows).
**Preprocessing:** Row-level physical-plausibility filtering (roughness,
tension_strength, elongation > 0; bed_temperature <= 120 C; print_speed <= 200 mm/s);
categorical one-hot encoding; no feature scaling (tree-based models used).
**Train/test approach:** 5-fold cross-validation only (n=69 is too small for a
held-out split to be statistically meaningful).

**Data provenance labeling (both datasets):**
1. Literature-derived data: Not used numerically for Dataset A; only the functional
   form of its equations is drawn from established cellular-solids theory.
2. Publicly available data: **Yes for Dataset B** — the real Kaggle "3D Printer
   Dataset for Mechanical Engineers", used as-is (cleaned, not altered
   numerically). Always labeled "Real experimental dataset — Kaggle."
3. Simulated/illustrative data: ALL of Dataset A is computed from closed-form
   analytical equations — always labeled "Illustrative computational (analytical)
   dataset." Dataset A and Dataset B are never merged or presented as one dataset.

# 5. System Architecture

```
Dataset A: Analytical (auxetic geometry)     Dataset B: Real (Kaggle AM process data)
        |                                              |
Parametric Geometry Sweep                    Kaggle "3D Printer Dataset for
(theta, h/l, t/l)                            Mechanical Engineers" (local copy)
        |                                              |
Analytical Mechanics Engine                  Data Validation and Cleaning
(Gibson-Ashby-type equations)                (drop sensor-error rows)
        |                                              |
Illustrative Computational Dataset           Exploratory Data Analysis
        |                                              |
Data Validation and Cleaning                 Categorical Encoding
        |                                              |
Exploratory Data Analysis                    ML Surrogate Training (Linear, Ridge,
        |                                    Random Forest, Gradient Boosting)
Feature Engineering (relative density)                |
        |                                    5-Fold Cross-Validation + Permutation
ML Surrogate Training (Linear, Ridge,        Feature Importance
Random Forest, Gradient Boosting)                      |
        |                                    Visualization and Export
Evaluation (holdout MAE/RMSE/R2 +                       |
5-fold CV, parity/residual plots)                       |
        |                                               |
Sensitivity / Feature Importance Analysis               |
        |                                               |
Inverse Design Query Module                             |
        |                                               |
Visualization and Export                                |
        \_______________________________________________/
                              |
                  Shared Streamlit Dashboard
        (Forward Prediction | Inverse Design | Real AM Process Data)
```

# 6. Detailed Methodology
See README.md and inline docstrings in each src/ module for full implementation detail.

**Analytical pipeline:** parametric sweep -> analytical computation -> validation/cleaning
-> EDA (incl. 3D geometry->property surface) -> feature engineering -> train/test split ->
model training (Linear, Ridge, Random Forest, Gradient Boosting) -> evaluation (holdout
MAE/RMSE/R2 + 5-fold CV + parity/residual plots) -> permutation importance + OAT sensitivity
-> inverse design (analytical grid search + surrogate-guided search) -> export.

**Real-data pipeline:** load Kaggle CSV -> validate/clean (drop sensor-error rows) -> EDA
(distributions, correlation heatmap, material/pattern comparison, process-vs-property
scatter) -> categorical encoding -> model training (Linear, Ridge, Random Forest, Gradient
Boosting) -> 5-fold cross-validation -> permutation feature importance -> export.

# 7. Machine Learning Requirements

## 7.1 Analytical dataset models
- **Features:** theta_deg, hl_ratio, tl_ratio, rel_density.
- **Targets:** E1_Es, E2_Es, nu12, nu21 (modeled separately, one regressor per target).
- **Models:** Linear Regression, Ridge, Random Forest Regressor (n_estimators=300),
  Gradient Boosting Regressor (n_estimators=300, max_depth=3, learning_rate=0.05).
- **Training process:** 80/20 split (seed=42) for holdout metrics, plus 5-fold
  cross-validation over the full dataset for a more robust comparison.
- **Evaluation metrics:** MAE, RMSE, R2 per target (holdout: `results/reports/metrics_summary.csv`;
  CV: `results/reports/cv_metrics_summary.csv`).
- **Model comparison:** Gradient Boosting and Random Forest both substantially
  outperform the linear baselines on all 4 targets, with Gradient Boosting slightly
  ahead on cross-validated R2 (see README results table and `05_model_comparison.png`).

## 7.2 Real dataset models
- **Numeric features:** layer_height, wall_thickness, infill_density, nozzle_temperature,
  bed_temperature, print_speed, fan_speed. **Encoded categorical features:**
  infill_pattern_honeycomb, material_pla.
- **Targets:** tension_strength, elongation, roughness (modeled separately).
- **Models:** Linear Regression, Ridge, Random Forest Regressor (n_estimators=300),
  Gradient Boosting Regressor (n_estimators=200, max_depth=2).
- **Training process:** 5-fold cross-validation only (n=69); a final Random Forest is
  also fit on the full cleaned dataset for the dashboard's live predictor and for
  permutation feature importance.
- **Evaluation metrics:** 5-fold CV R2 and MAE (`results/reports/real_data_cv_metrics.csv`).
- **Model comparison:** Random Forest and Gradient Boosting outperform the linear
  baselines on all 3 targets, though with much wider cross-validation uncertainty
  than the analytical dataset, reflecting real measurement noise on a small sample
  (see `14_real_model_comparison.png`).

No deep learning used on either dataset — unnecessary given the smooth, low-dimensional
analytical relationship (Dataset A) and the very small sample size (Dataset B, n=69),
where a deep model would badly overfit.

# 8. Visualization Requirements
Implemented in results/figures/, all sharing one validated color palette and house
style (src/plot_style.py) with a data-provenance caption burned into every figure:

**Analytical dataset (auxetic geometry):**
- 01_distributions.png - feature/target distributions
- 02_correlation_heatmap.png - correlation heatmap
- 03_feature_vs_target.png - feature-vs-target relationship plots
- 04_geometry_property_surface.png - 3D geometry -> nu12 surface, auxetic/non-auxetic boundary
- 05_model_comparison.png - 5-fold CV R2 across 4 models x 4 targets
- 06_parity_plots.png - actual vs predicted (Random Forest)
- 07_residuals.png - residual plots (Random Forest)
- 08_feature_importance.png - permutation feature importance
- 09_oat_sensitivity.png - one-at-a-time sensitivity sweeps

**Real dataset (Kaggle AM process data):**
- 10_real_distributions.png - feature/target distributions
- 11_real_correlation_heatmap.png - correlation heatmap
- 12_real_material_pattern_comparison.png - PLA vs ABS, grid vs honeycomb boxplots
- 13_real_process_vs_strength.png - PLA-focused process-parameter vs property scatter
- 14_real_model_comparison.png - 5-fold CV R2 across 4 models x 3 targets
- 15_real_feature_importance.png - permutation feature importance

# 9. Evaluation
**Analytical dataset:** MAE, RMSE, R2 on an 80/20 held-out split
(`results/reports/metrics_summary.csv`) plus 5-fold cross-validated R2/MAE
(`results/reports/cv_metrics_summary.csv`), reported per target property.
**Real dataset:** 5-fold cross-validated R2/MAE only, given the small sample size
(`results/reports/real_data_cv_metrics.csv`), reported per target property.

# 10. User Interface Requirements (Streamlit Dashboard)
Implemented in app/app.py, three tabs:
- **Forward Prediction** (analytical): sliders for theta, h/l, t/l -> live predicted E1*/Es and nu12 (analytical + surrogate).
- **Inverse Design** (analytical): target nu12 (+ optional target E1*/Es) input -> recommended geometry from both analytical grid search and surrogate search.
- **Real AM Process Data** (Kaggle): dataset snapshot table, cross-validated model comparison table, and sliders for real process parameters (layer height, infill density, nozzle/bed temperature, print speed, wall thickness, fan speed, material, infill pattern) -> live predicted tensile strength, elongation, and roughness from a Random Forest fit on the full cleaned real dataset.

# 11. Future Enhancements
- Replace/extend the analytical dataset with real experimental PLA **auxetic**
  measurements once available (the current real dataset uses standard infill, not
  re-entrant auxetic cells, so it complements rather than replaces Dataset A).
- Add anisotropy/build-orientation features.
- Cross-validate against FEA results.
- Extend to multi-objective optimization.
- Apply the same surrogate approach to NiTi/scan-strategy problems once literature data is compiled.

# 12. Research Integrity / Credibility Notes
- Directly inspired by the professor's research: choice of PLA re-entrant auxetic structures, framing as a computational design/screening tool.
- Author's own computational contribution: full data-generation pipeline, ML surrogate training/evaluation, sensitivity analysis, inverse-design module, and the real-data ingestion/cleaning/modeling pipeline built around a third-party dataset.
- Assumptions: linear-elastic small-deformation beam theory adequacy; chosen parameter ranges are illustrative, not measured; relative-density formula is a standard hexagonal-cell approximation; the real Kaggle dataset's process-property relationships are assumed broadly representative of FDM PLA/ABS behavior despite using standard (non-auxetic) infill.
- Requires real experimental validation before being research-grade: whether the analytical auxetic model matches real printed PLA specimens; anisotropy/defect effects; whether real re-entrant auxetic coupons behave like the standard-infill coupons in the Kaggle dataset.
- Claims that must NOT be made: the analytical (Dataset A) numbers are NOT experimental; the analytical model is NOT validated against real PLA prints; the real Kaggle dataset (Dataset B) is NOT auxetic data and must never be described as such; no numeric values are attributed to the professor's own published results; the two datasets are NOT the same study and must never be merged or implied to be.
