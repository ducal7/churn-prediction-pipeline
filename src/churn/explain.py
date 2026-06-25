"""SHAP explanations: global feature importance and per-user reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .model import ChurnModel


def shap_values(model: ChurnModel, X: pd.DataFrame) -> np.ndarray:
    """Per-row, per-feature SHAP contributions for the positive (churn) class."""
    X = X[list(model.feature_columns)]
    explainer = shap.TreeExplainer(model.base)
    raw = explainer.shap_values(X)
    # LightGBM binary TreeExplainer may return a list [neg, pos]; take positive.
    if isinstance(raw, list):
        raw = raw[1]
    raw = np.asarray(raw)
    if raw.ndim == 3:  # (n, features, classes)
        raw = raw[:, :, 1]
    return raw


def global_importance(model: ChurnModel, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, descending."""
    sv = shap_values(model, X)
    imp = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(model.feature_columns), "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def reason_codes(model: ChurnModel, X: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Top-``k`` signed drivers of each user's predicted risk.

    Returns one row per input user with the calibrated probability and a compact
    human-readable string of the strongest ``+``/``-`` feature contributions.
    """
    feats = list(model.feature_columns)
    sv = shap_values(model, X)
    proba = model.predict_proba(X)

    rows = []
    values = X[feats].to_numpy()
    for i in range(len(X)):
        order = np.argsort(-np.abs(sv[i]))[:top_k]
        parts = []
        for j in order:
            sign = "+" if sv[i, j] >= 0 else "-"
            parts.append(f"{sign}{feats[j]}={values[i, j]:.2f}")
        rows.append({"churn_probability": float(proba[i]), "reason_codes": "; ".join(parts)})

    out = pd.DataFrame(rows)
    if "user_id" in X.columns:
        out.insert(0, "user_id", X["user_id"].to_numpy())
    return out
