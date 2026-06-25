"""Fast end-to-end smoke test on a tiny synthetic sample."""

from __future__ import annotations

import numpy as np

from churn import data, explain, features, metrics
from churn.config import CUTOFF_TRAIN, CUTOFF_VALID
from churn.model import train_model


def test_end_to_end_pipeline():
    # 1. tiny dataset
    activity, _users = data.generate(n_users=800, sim_days=181, seed=11)

    # 2. out-of-time tables
    train_df = features.build_modeling_table(activity, CUTOFF_TRAIN)
    valid_df = features.build_modeling_table(activity, CUTOFF_VALID)
    assert len(train_df) > 50 and len(valid_df) > 50

    # 3. train calibrated model
    model = train_model(train_df, calibration_method="isotonic")

    # 4. predict + evaluate out-of-time
    y_true = valid_df["churned"].to_numpy()
    y_prob = model.predict_proba(valid_df)
    assert y_prob.shape == y_true.shape
    assert ((y_prob >= 0) & (y_prob <= 1)).all()

    m = metrics.ranking_metrics(y_true, y_prob)
    # The hidden process is learnable -> comfortably better than random.
    assert m["roc_auc"] > 0.65

    # 5. SHAP explanations / reason codes
    imp = explain.global_importance(model, valid_df)
    assert len(imp) == len(model.feature_columns)
    assert (imp["mean_abs_shap"] >= 0).all()

    rc = explain.reason_codes(model, valid_df.head(20), top_k=3)
    assert len(rc) == 20
    assert rc["reason_codes"].str.len().gt(0).all()


def test_persist_round_trip(tmp_path):
    activity, _ = data.generate(n_users=600, sim_days=181, seed=5)
    train_df = features.build_modeling_table(activity, CUTOFF_TRAIN)
    model = train_model(train_df)

    path = tmp_path / "model.joblib"
    model.save(path)
    from churn.model import ChurnModel

    reloaded = ChurnModel.load(path)
    p1 = model.predict_proba(train_df)
    p2 = reloaded.predict_proba(train_df)
    assert np.allclose(p1, p2)
