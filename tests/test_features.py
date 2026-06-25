"""Feature builder: the no-future-leakage guarantee."""

from __future__ import annotations

import pandas as pd

from churn import features
from churn.config import CUTOFF_TRAIN
from churn.data import LEAKY_COLUMNS


def test_features_ignore_the_future(tiny_dataset):
    """Features at cutoff T must be invariant to anything happening after T.

    We build features from the full log and from a log truncated at the cutoff;
    if any future row leaked into a feature the two tables would differ.
    """
    activity, _ = tiny_dataset
    full = features.build_features(activity, CUTOFF_TRAIN)

    truncated = activity[activity["day"] <= CUTOFF_TRAIN].copy()
    from_truncated = features.build_features(truncated, CUTOFF_TRAIN)

    pd.testing.assert_frame_equal(full, from_truncated)


def test_features_change_if_future_were_used(tiny_dataset):
    """Sanity check on the test itself: a deliberately leaky aggregate WOULD differ.

    This proves the truncation actually removes information, so the invariance in
    the previous test is meaningful rather than vacuous.
    """
    activity, _ = tiny_dataset
    full_future_sessions = activity.groupby("user_id")["sessions"].sum()
    truncated = activity[activity["day"] <= CUTOFF_TRAIN]
    past_sessions = truncated.groupby("user_id")["sessions"].sum()
    assert not full_future_sessions.equals(past_sessions)


def test_no_leaky_columns_in_feature_set():
    for col in LEAKY_COLUMNS:
        assert col not in features.FEATURE_COLUMNS


def test_modeling_table_has_binary_label(tiny_dataset):
    activity, _ = tiny_dataset
    table = features.build_modeling_table(activity, CUTOFF_TRAIN)
    assert set(table["churned"].unique()) <= {0, 1}
    assert 0.0 < table["churned"].mean() < 1.0  # both classes present
    assert (table["recency_days"] >= 0).all()
    assert (table["tenure_days"] >= 0).all()


def test_scope_excludes_already_churned_users(tiny_dataset):
    activity, users = tiny_dataset
    table = features.build_modeling_table(activity, CUTOFF_TRAIN)
    churn_day = users.set_index("user_id")["churn_day"]
    # Anyone in scope must have been active at/after the recency window start.
    scoped = table.set_index("user_id")
    assert (churn_day.loc[scoped.index] > CUTOFF_TRAIN - 14).all()
