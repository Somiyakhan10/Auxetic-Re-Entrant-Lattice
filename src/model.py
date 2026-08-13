"""
model.py

Trains ML surrogate models (Linear Regression baseline + Random Forest)
to predict effective mechanical properties (E1_Es, E2_Es, nu12, nu21)
from unit-cell geometry (theta_deg, hl_ratio, tl_ratio, rel_density).

Models are saved (via joblib) for reuse by inverse_design.py and app.py.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "auxetic_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")

FEATURES = ["theta_deg", "hl_ratio", "tl_ratio", "rel_density"]
TARGETS = ["E1_Es", "E2_Es", "nu12", "nu21"]

RANDOM_SEED = 42


def load_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


MODEL_FACTORIES = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED),
    # n_jobs=1 here: these factories are called inside cross_validate(n_jobs=-1),
    # which already parallelizes across folds/targets/models — nesting a second
    # parallel pool per-tree would oversubscribe cores and slow everything down.
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=1
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED
    ),
}


def train_models(X_train, y_train):
    """Train one model of each type in MODEL_FACTORIES per target."""
    models = {name: {} for name in MODEL_FACTORIES}

    for target in TARGETS:
        for name, factory in MODEL_FACTORIES.items():
            m = factory()
            m.fit(X_train, y_train[target])
            models[name][target] = m

    return models


def save_models(models):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "surrogate_models.joblib")
    joblib.dump(models, path)
    print(f"[model] Saved trained models -> {path}")
    return path


def main():
    X_train, X_test, y_train, y_test = load_split()
    models = train_models(X_train, y_train)
    save_models(models)
    return models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
