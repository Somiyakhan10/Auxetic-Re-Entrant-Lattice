"""
generate_dataset.py

Generates an ILLUSTRATIVE COMPUTATIONAL DATASET (not experimental data) for
re-entrant auxetic honeycomb unit cells, based on classical linear-elastic
beam theory for honeycombs (Gibson-Ashby type formulation).

Geometry convention:
    h  : vertical strut length
    l  : inclined strut length
    t  : strut thickness
    theta (deg): re-entrant cell angle, NEGATIVE for auxetic re-entrant cells
                 (the inclined struts point "inward" rather than "outward")
    b  : out-of-plane depth (cancels out of the ratios used here)

Non-dimensional inputs used as ML features:
    theta_deg   : cell angle in degrees (negative = re-entrant/auxetic)
    hl_ratio    : h / l
    tl_ratio    : t / l   (relates to strut slenderness / relative density)

Outputs (targets):
    E1_Es   : effective modulus ratio in direction 1 (E1*/Es)
    E2_Es   : effective modulus ratio in direction 2 (E2*/Es)
    nu12    : in-plane Poisson's ratio (loading direction 1)
    nu21    : in-plane Poisson's ratio (loading direction 2)
    rel_density : approximate relative density (rho*/rho_s)

IMPORTANT:
This dataset is entirely computed from closed-form analytical equations.
It is NOT derived from experiments and NOT taken from any specific paper's
measured results. It must always be labeled "illustrative computational
(analytical) dataset" in any report, plot, or presentation.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42

# Physically reasonable, illustrative parameter ranges for printable
# re-entrant auxetic unit cells (NOT taken from a specific experimental study)
THETA_DEG_RANGE = (-80.0, -5.0)   # re-entrant angle, degrees
HL_RATIO_RANGE = (0.5, 2.0)       # h/l
TL_RATIO_RANGE = (0.02, 0.20)     # t/l (thin-strut / small-deformation regime)

N_SAMPLES = 4000


def compute_properties(theta_deg: np.ndarray, hl: np.ndarray, tl: np.ndarray):
    """
    Compute effective modulus ratios and Poisson's ratios for a re-entrant
    honeycomb unit cell using standard Gibson-Ashby-type beam-theory
    equations (bending-dominated, linear-elastic, small-deformation).

    theta_deg : re-entrant angle in degrees (negative for auxetic geometry)
    hl        : h/l ratio
    tl        : t/l ratio
    """
    theta = np.radians(theta_deg)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom_1 = (hl + sin_t) * sin_t**2
    denom_2 = cos_t**3
    denom_nu12 = (hl + sin_t) * sin_t
    denom_nu21 = cos_t**2

    with np.errstate(divide="ignore", invalid="ignore"):
        E1_Es = (tl**3) * cos_t / denom_1
        E2_Es = (tl**3) * (hl + sin_t) / denom_2
        nu12 = cos_t**2 / denom_nu12
        nu21 = (hl + sin_t) * sin_t / denom_nu21

        # Standard hexagonal-cell relative density approximation
        rel_density = (tl * (hl + 2.0)) / (2.0 * cos_t * (hl + sin_t))

    return E1_Es, E2_Es, nu12, nu21, rel_density


def generate_raw_samples(n_samples: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    theta_deg = rng.uniform(*THETA_DEG_RANGE, size=n_samples)
    hl_ratio = rng.uniform(*HL_RATIO_RANGE, size=n_samples)
    tl_ratio = rng.uniform(*TL_RATIO_RANGE, size=n_samples)

    E1_Es, E2_Es, nu12, nu21, rel_density = compute_properties(theta_deg, hl_ratio, tl_ratio)

    df = pd.DataFrame({
        "theta_deg": theta_deg,
        "hl_ratio": hl_ratio,
        "tl_ratio": tl_ratio,
        "E1_Es": E1_Es,
        "E2_Es": E2_Es,
        "nu12": nu12,
        "nu21": nu21,
        "rel_density": rel_density,
    })
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-physical / degenerate rows:
      - non-finite values
      - non-positive moduli (geometrically invalid combination)
      - relative density outside a sensible printable range (0 < rho* < 1)
      - extreme outlier ratios from near-singular denominators
    """
    before = len(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df = df[(df["E1_Es"] > 0) & (df["E2_Es"] > 0)]
    df = df[(df["rel_density"] > 0.01) & (df["rel_density"] < 0.9)]

    # Clip extreme numerical outliers (near-singular geometry) using
    # percentile-based bounds rather than hard-coded magic numbers.
    for col in ["E1_Es", "E2_Es", "nu12", "nu21"]:
        lo, hi = df[col].quantile([0.005, 0.995])
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    after = len(df)
    print(f"[generate_dataset] Kept {after}/{before} samples after physical validation.")
    return df.reset_index(drop=True)


def main():
    df_raw = generate_raw_samples(N_SAMPLES)
    df_clean = validate_and_clean(df_raw)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "auxetic_dataset.csv")
    df_clean.to_csv(out_path, index=False)

    print(f"[generate_dataset] Saved illustrative computational dataset -> {out_path}")
    print(df_clean.describe().T)


if __name__ == "__main__":
    main()
