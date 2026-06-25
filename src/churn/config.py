"""Central configuration: paths, seeds and the point-in-time time grid.

All knobs that define the synthetic world and the modelling protocol live here so
that the data generator, feature builder, trainer and tests share one source of
truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Filesystem layout (repo root is two levels up from this file: src/churn/..). #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

ACTIVITY_PATH = DATA_DIR / "activity_log.parquet"
USERS_PATH = DATA_DIR / "users.parquet"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SCORES_PATH = REPORTS_DIR / "scores.csv"

# --------------------------------------------------------------------------- #
# Reproducibility.                                                            #
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Synthetic world size.                                                       #
# --------------------------------------------------------------------------- #
N_USERS = 4000
SIM_DAYS = 181  # simulate day 0 .. 180 inclusive
SIGNUP_MAX_DAY = 120  # users sign up somewhere in [0, SIGNUP_MAX_DAY]
START_DATE = pd.Timestamp("2025-01-01")

# --------------------------------------------------------------------------- #
# Point-in-time protocol.                                                     #
# --------------------------------------------------------------------------- #
# We score users *as of* a cutoff day and predict whether they churn within the
# following HORIZON days.  Two cutoffs give a true out-of-time (temporal) split:
# the validation cutoff is strictly later than the training cutoff.
HORIZON = 30
CUTOFF_TRAIN = 120
CUTOFF_VALID = 150  # CUTOFF_VALID + HORIZON == 180 <= SIM_DAYS - 1

# A user is "in scope" at a cutoff only if they have enough history and were
# recently active -- i.e. they are a genuine at-risk active user, not someone
# who already left long ago.
MIN_TENURE_DAYS = 14
RECENT_WINDOW_DAYS = 14


def day_to_date(day: int | float) -> pd.Timestamp:
    """Map an integer simulation day onto a wall-clock calendar date."""
    return START_DATE + pd.Timedelta(days=int(day))
