"""Calibrated LightGBM churn model: train, persist, load, predict.

The classifier is a LightGBM gradient-boosted tree wrapped in scikit-learn's
:class:`CalibratedClassifierCV` (isotonic regression by default).  We also keep a
standalone *uncalibrated* LightGBM fitted on the full training fold purely for
SHAP attribution -- isotonic calibration is monotone, so it only re-maps the
score axis and leaves the feature-level explanation of the ranker intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import config
from .features import FEATURE_COLUMNS

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": config.SEED,
    "n_jobs": 1,
    "verbose": -1,
}


@dataclass
class ChurnModel:
    """Bundle of the calibrated classifier and the SHAP base estimator."""

    calibrated: CalibratedClassifierCV
    base: lgb.LGBMClassifier
    feature_columns: tuple[str, ...]
    calibration_method: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated churn probability for each row."""
        return self.calibrated.predict_proba(X[list(self.feature_columns)])[:, 1]

    def save(self, path=config.MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_PATH) -> ChurnModel:
        return joblib.load(path)


def train_model(train_df: pd.DataFrame, calibration_method: str = "isotonic") -> ChurnModel:
    """Fit the calibrated model + the SHAP base model on a training table."""
    X = train_df[list(FEATURE_COLUMNS)]
    y = train_df["churned"].to_numpy()

    base = lgb.LGBMClassifier(**LGBM_PARAMS)
    base.fit(X, y)

    calibrated = CalibratedClassifierCV(
        lgb.LGBMClassifier(**LGBM_PARAMS),
        method=calibration_method,
        cv=3,
    )
    calibrated.fit(X, y)

    return ChurnModel(
        calibrated=calibrated,
        base=base,
        feature_columns=FEATURE_COLUMNS,
        calibration_method=calibration_method,
    )
