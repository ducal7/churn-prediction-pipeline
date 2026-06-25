"""Data generator: determinism and structural sanity."""

from __future__ import annotations

import pandas as pd

from churn import data


def test_same_seed_same_data():
    a1, u1 = data.generate(n_users=300, sim_days=120, seed=123)
    a2, u2 = data.generate(n_users=300, sim_days=120, seed=123)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(u1, u2)


def test_different_seed_differs():
    a1, _ = data.generate(n_users=300, sim_days=120, seed=1)
    a2, _ = data.generate(n_users=300, sim_days=120, seed=2)
    # Astronomically unlikely to be identical for different seeds.
    assert not a1.equals(a2)


def test_no_activity_after_churn_day(tiny_dataset):
    activity, users = tiny_dataset
    churn_day = users.set_index("user_id")["churn_day"]
    last_active = activity.groupby("user_id")["day"].max()
    # Every user's last activity day is strictly before their churn day.
    common = last_active.index.intersection(churn_day.index)
    assert (last_active[common] < churn_day[common]).all()


def test_leaky_columns_present_in_users(tiny_dataset):
    _activity, users = tiny_dataset
    for col in data.LEAKY_COLUMNS:
        assert col in users.columns, f"missing trap column {col}"
