"""Seeded synthetic data generator for gaming / fintech users.

The generator is fully deterministic given ``config.SEED``: same seed -> byte-for
byte identical tables.  It produces two artefacts:

* ``activity_log`` -- a daily-grain event log (one row per active user-day) that
  is the *only* legitimate input to point-in-time feature engineering.
* ``users`` -- a per-user dimension table that deliberately also carries
  **leakage traps**: columns computed over the *entire* simulation horizon
  (i.e. using information from the future relative to any scoring cutoff).  A
  naive modeller who joins these in would get a spectacular -- and completely
  invalid -- AUC.  ``features.py`` never touches them and the test-suite proves
  it.

Hidden churn process
--------------------
Each user has a latent ``engagement`` score.  High engagement means more
sessions / deposits per active day *and* a lower daily hazard of churning.
Each user draws a ``churn_day`` from a geometric distribution whose rate falls
with engagement; after that day the user emits no events.  Nothing about
``engagement`` or ``churn_day`` is exposed to the model -- it must be recovered
from the observable activity log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Columns on the ``users`` table that encode the future and must NEVER be used
# as model features.  Kept here so tests and feature code agree on the blocklist.
LEAKY_COLUMNS: tuple[str, ...] = (
    "engagement",
    "churn_day",
    "final_status",
    "lifetime_deposit_total",
    "total_active_days",
    "last_active_day",
)


def generate(
    n_users: int = config.N_USERS,
    sim_days: int = config.SIM_DAYS,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ``(activity_log, users)`` deterministically from ``seed``."""
    rng = np.random.default_rng(seed)

    user_ids = np.arange(n_users, dtype=np.int64)
    signup_day = rng.integers(0, config.SIGNUP_MAX_DAY + 1, size=n_users)

    # Latent engagement in (0, 1); Beta(2, 2) is unimodal around 0.5.
    engagement = rng.beta(2.0, 2.0, size=n_users)

    # Daily churn hazard: low-engagement users leave much sooner.
    daily_hazard = 0.002 + 0.05 * (1.0 - engagement)
    # Geometric "days until churn" measured from signup.
    days_to_churn = rng.geometric(daily_hazard)
    churn_day = np.minimum(signup_day + days_to_churn, sim_days)

    # Per-user-day session intensity, decaying slowly over a user's tenure.
    base_rate = 0.15 + 1.6 * engagement  # mean sessions on an active day

    days = np.arange(sim_days)
    # Active mask: signed up, not yet churned. Shape (n_users, sim_days).
    active_mask = (days[None, :] >= signup_day[:, None]) & (days[None, :] < churn_day[:, None])

    tenure = np.clip(days[None, :] - signup_day[:, None], 0, None)
    decay = np.exp(-tenure / 90.0)  # gentle engagement decay with tenure
    lam = base_rate[:, None] * decay * active_mask

    sessions = rng.poisson(lam).astype(np.int64)
    sessions *= active_mask  # zero out non-active days explicitly

    # Keep only rows with real activity -> sparse daily event log.
    u_idx, d_idx = np.nonzero(sessions > 0)
    n_events = u_idx.size
    sess = sessions[u_idx, d_idx]

    minutes = np.round(sess * rng.gamma(shape=3.0, scale=4.0, size=n_events), 1)

    # Deposits: a Bernoulli draw per active day, amount lognormal scaled by sessions.
    eng_ev = engagement[u_idx]
    deposit_made = rng.random(n_events) < (0.05 + 0.35 * eng_ev)
    deposit_amount = np.where(
        deposit_made,
        np.round(rng.lognormal(mean=2.6, sigma=0.7, size=n_events), 2),
        0.0,
    )
    deposit_count = deposit_made.astype(np.int64)

    withdraw_made = rng.random(n_events) < 0.04
    withdrawal_amount = np.where(
        withdraw_made,
        np.round(rng.lognormal(mean=2.4, sigma=0.6, size=n_events), 2),
        0.0,
    )

    support_tickets = rng.poisson(0.03, size=n_events).astype(np.int64)

    activity = pd.DataFrame(
        {
            "user_id": user_ids[u_idx],
            "day": d_idx.astype(np.int64),
            "date": [config.day_to_date(d) for d in d_idx],
            "sessions": sess,
            "minutes_played": minutes,
            "deposit_count": deposit_count,
            "deposit_amount": deposit_amount,
            "withdrawal_amount": withdrawal_amount,
            "support_tickets": support_tickets,
        }
    ).sort_values(["user_id", "day"], ignore_index=True)

    # ---- Build the users dimension table, including the leakage traps. ----
    # These aggregates intentionally span the WHOLE horizon (the future).
    agg = activity.groupby("user_id").agg(
        lifetime_deposit_total=("deposit_amount", "sum"),
        total_active_days=("day", "nunique"),
        last_active_day=("day", "max"),
    )
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_day": signup_day,
            "signup_date": [config.day_to_date(d) for d in signup_day],
            # --- leakage traps below this line ---
            "engagement": np.round(engagement, 4),
            "churn_day": churn_day,
            "final_status": np.where(churn_day < sim_days, "churned", "active"),
        }
    ).set_index("user_id")
    users = users.join(agg).reset_index()
    users["lifetime_deposit_total"] = users["lifetime_deposit_total"].fillna(0.0)
    users["total_active_days"] = users["total_active_days"].fillna(0).astype(np.int64)
    users["last_active_day"] = users["last_active_day"].fillna(-1).astype(np.int64)

    return activity, users


def write(activity: pd.DataFrame, users: pd.DataFrame) -> None:
    """Persist the generated tables to the (git-ignored) data directory."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    activity.to_parquet(config.ACTIVITY_PATH, index=False)
    users.to_parquet(config.USERS_PATH, index=False)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted tables, generating them first if absent."""
    if not config.ACTIVITY_PATH.exists() or not config.USERS_PATH.exists():
        activity, users = generate()
        write(activity, users)
        return activity, users
    return (
        pd.read_parquet(config.ACTIVITY_PATH),
        pd.read_parquet(config.USERS_PATH),
    )


def main() -> None:
    activity, users = generate()
    write(activity, users)
    churn_rate = (users["final_status"] == "churned").mean()
    print(
        f"Generated {len(activity):,} activity rows for {len(users):,} users "
        f"(end-of-horizon churn rate {churn_rate:.1%}).\n"
        f"  activity -> {config.ACTIVITY_PATH}\n"
        f"  users    -> {config.USERS_PATH}"
    )


if __name__ == "__main__":
    main()
