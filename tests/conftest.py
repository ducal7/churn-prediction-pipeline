"""Shared fixtures: a tiny, fast synthetic dataset for the whole suite."""

from __future__ import annotations

import pytest

from churn import data


@pytest.fixture(scope="session")
def tiny_dataset():
    """A small deterministic dataset (kept tiny so the suite runs in seconds)."""
    activity, users = data.generate(n_users=600, sim_days=181, seed=7)
    return activity, users
