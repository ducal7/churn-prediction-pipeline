"""Metric calculations: known-answer and property tests."""

from __future__ import annotations

import numpy as np

from churn import metrics


def test_perfect_separation_auc_is_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics.ranking_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_random_scores_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=5000)
    p = rng.random(5000)
    m = metrics.ranking_metrics(y, p)
    assert abs(m["roc_auc"] - 0.5) < 0.05


def test_brier_is_squared_error():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 0.5, 0.25])
    expected = np.mean((p - y) ** 2)
    m = metrics.ranking_metrics(y, p)
    assert abs(m["brier"] - expected) < 1e-9


def test_lift_table_shape_and_capture():
    rng = np.random.default_rng(1)
    n = 1000
    y = rng.integers(0, 2, size=n)
    # Scores positively correlated with the label.
    p = 0.3 * y + 0.7 * rng.random(n)
    table = metrics.lift_by_decile(y, p, n_bins=10)
    assert len(table) == 10
    # Cumulative capture is monotone non-decreasing and ends at 1.
    cap = table["cumulative_capture"].to_numpy()
    assert np.all(np.diff(cap) >= -1e-9)
    assert abs(cap[-1] - 1.0) < 1e-9
    # Top decile should out-perform a random decile given the correlation.
    assert table.loc[0, "lift"] > 1.0


def test_calibration_table_bins():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.05, 0.15, 0.85, 0.95, 0.92, 0.45])
    table = metrics.calibration_table(y, p, n_bins=10)
    assert {"mean_predicted", "observed_frequency", "count"} <= set(table.columns)
    assert table["count"].sum() == len(y)
