"""Evaluation metrics: ranking, calibration and business lift."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC (average precision) and the Brier calibration score."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def lift_by_decile(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Lift table: users ranked by predicted risk, bucketed into deciles.

    Decile 1 is the highest-risk bucket.  ``lift`` is the bucket's churn rate
    divided by the overall churn rate; ``cumulative_lift`` is the same for the
    top-k buckets combined -- the classic "if we target the top X%" view.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    base_rate = y_true.mean()
    n = len(y_true)
    # Split into n_bins as-equal-as-possible contiguous groups.
    bin_ids = np.floor(np.arange(n) * n_bins / n).astype(int)

    rows = []
    cum_pos = 0
    cum_cnt = 0
    total_pos = y_true.sum()
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        pos = int(y_sorted[mask].sum())
        cum_pos += pos
        cum_cnt += cnt
        rate = pos / cnt if cnt else 0.0
        rows.append(
            {
                "decile": b + 1,
                "count": cnt,
                "churners": pos,
                "churn_rate": rate,
                "lift": rate / base_rate if base_rate else 0.0,
                "cumulative_churn_rate": cum_pos / cum_cnt if cum_cnt else 0.0,
                "cumulative_capture": cum_pos / total_pos if total_pos else 0.0,
                "cumulative_lift": (cum_pos / cum_cnt) / base_rate
                if base_rate and cum_cnt
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: mean predicted vs observed frequency per probability bin."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        rows.append(
            {
                "bin": b + 1,
                "count": cnt,
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
