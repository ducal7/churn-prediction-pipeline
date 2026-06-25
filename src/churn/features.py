"""Point-in-time feature engineering.

The cardinal rule: a feature for a user *as of* cutoff day ``T`` may depend only
on activity rows with ``day <= T``.  Everything in :func:`build_features` honours
this, which is what makes the modelling table safe to learn from.  The label, by
contrast, is computed from the *future* window ``(T, T + HORIZON]`` -- that is
allowed for the target, but never for inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# The model is only ever allowed to see these columns.
FEATURE_COLUMNS: tuple[str, ...] = (
    "tenure_days",
    "recency_days",
    "active_days_7",
    "active_days_30",
    "sessions_7",
    "sessions_30",
    "minutes_30",
    "deposit_amount_30",
    "deposit_count_30",
    "withdrawal_amount_30",
    "support_tickets_30",
    "avg_sessions_per_active_day",
    "deposit_per_active_day",
    "activity_trend",
)

LABEL_COLUMN = "churned"


def _windowed_sum(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    out = df.groupby("user_id")[col].sum()
    out.name = name
    return out


def build_features(activity: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Build the point-in-time feature table for every user active near ``cutoff``.

    Parameters
    ----------
    activity:
        The daily activity log (may contain rows after ``cutoff``; they are
        dropped here so the output is invariant to anything in the future).
    cutoff:
        The integer "as-of" day. Only ``day <= cutoff`` rows are used.

    Returns
    -------
    DataFrame indexed by a fresh range index with a ``user_id`` column, the
    :data:`FEATURE_COLUMNS`, plus ``cutoff`` and ``signup_day`` helper columns.
    Only in-scope users (sufficient tenure + recently active) are returned.
    """
    hist = activity[activity["day"] <= cutoff]
    if hist.empty:
        return pd.DataFrame(columns=["user_id", "cutoff", *FEATURE_COLUMNS])

    w30 = hist[hist["day"] > cutoff - 30]
    w7 = hist[hist["day"] > cutoff - 7]

    grp = hist.groupby("user_id")
    signup_day = grp["day"].min().rename("first_active_day")
    last_active = grp["day"].max().rename("last_active_day")

    feats = pd.DataFrame(index=last_active.index)
    feats["last_active_day"] = last_active
    feats["first_active_day"] = signup_day
    feats["recency_days"] = cutoff - last_active
    feats["tenure_days"] = cutoff - signup_day

    feats["active_days_7"] = w7.groupby("user_id")["day"].nunique()
    feats["active_days_30"] = w30.groupby("user_id")["day"].nunique()
    feats["sessions_7"] = _windowed_sum(w7, "sessions", "sessions_7")
    feats["sessions_30"] = _windowed_sum(w30, "sessions", "sessions_30")
    feats["minutes_30"] = _windowed_sum(w30, "minutes_played", "minutes_30")
    feats["deposit_amount_30"] = _windowed_sum(w30, "deposit_amount", "deposit_amount_30")
    feats["deposit_count_30"] = _windowed_sum(w30, "deposit_count", "deposit_count_30")
    feats["withdrawal_amount_30"] = _windowed_sum(w30, "withdrawal_amount", "withdrawal_amount_30")
    feats["support_tickets_30"] = _windowed_sum(w30, "support_tickets", "support_tickets_30")

    feats = feats.fillna(0.0)

    feats["avg_sessions_per_active_day"] = feats["sessions_30"] / feats["active_days_30"].replace(
        0, np.nan
    )
    feats["deposit_per_active_day"] = feats["deposit_amount_30"] / feats["active_days_30"].replace(
        0, np.nan
    )
    # Recent momentum: last-7-day rate vs the 30-day baseline rate.
    feats["activity_trend"] = feats["sessions_7"] / (feats["sessions_30"] / 4.0 + 1.0)
    feats = feats.fillna(0.0)

    feats = feats.reset_index().rename(columns={"index": "user_id"})

    # Scope: enough tenure and active within the recent window.
    in_scope = (feats["tenure_days"] >= config.MIN_TENURE_DAYS) & (
        feats["recency_days"] <= config.RECENT_WINDOW_DAYS
    )
    feats = feats[in_scope].copy()

    feats["cutoff"] = cutoff
    feats["signup_day"] = feats["first_active_day"]
    cols = ["user_id", "cutoff", "signup_day", *FEATURE_COLUMNS]
    return feats[cols].reset_index(drop=True)


def build_labels(activity: pd.DataFrame, cutoff: int, horizon: int = config.HORIZON) -> pd.Series:
    """Future churn label per user: 1 if NO activity in ``(cutoff, cutoff+horizon]``.

    This intentionally reads the future and must only be used to construct the
    training target, never as an input feature.
    """
    fut = activity[(activity["day"] > cutoff) & (activity["day"] <= cutoff + horizon)]
    active_in_horizon = set(fut["user_id"].unique())

    def _label(uid: int) -> int:
        return 0 if uid in active_in_horizon else 1

    return pd.Series({uid: _label(uid) for uid in activity["user_id"].unique()}, name=LABEL_COLUMN)


def build_modeling_table(
    activity: pd.DataFrame, cutoff: int, horizon: int = config.HORIZON
) -> pd.DataFrame:
    """Join point-in-time features to the future-derived label for one cutoff."""
    feats = build_features(activity, cutoff)
    labels = build_labels(activity, cutoff, horizon)
    feats[LABEL_COLUMN] = feats["user_id"].map(labels).astype(int)
    return feats
