"""
LightGBM fitting and training-only diagnostics; no test data enters fitting.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CLASS_IDS, FEATURES, RATING_FEATURES


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, columns, *, seed: int = 42):
    if set(train["y"].unique()) != set(CLASS_IDS):
        raise ValueError("Training data must contain all three outcomes")
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=400, learning_rate=0.04,
        num_leaves=7, max_depth=3, min_child_samples=80, reg_lambda=5.0,
        random_state=seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    model.fit(train[list(columns)], train["y"],
              eval_X=validation[list(columns)], eval_y=validation["y"],
              eval_metric="multi_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
    assert np.array_equal(model.classes_, CLASS_IDS)
    return model


def train_models(train: pd.DataFrame, validation: pd.DataFrame, *, seed: int = 42) -> dict:
    return {name: (fit_model(train, validation, columns, seed=seed), columns)
            for name, columns in (("rating_only", RATING_FEATURES), ("historical_features", FEATURES))}


def training_class_prior(train: pd.DataFrame) -> np.ndarray:
    return train["y"].value_counts().reindex(CLASS_IDS, fill_value=0).to_numpy() / len(train)


def get_gain_importance(model) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    assert np.isfinite(importance.gain).all() and (importance.gain >= 0).all()
    return importance
