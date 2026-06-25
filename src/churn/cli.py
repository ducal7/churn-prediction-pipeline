"""Command-line interface: ``python -m churn {train,evaluate,score,data}``."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config, data, explain, features, metrics, plots
from .model import ChurnModel, train_model


def _load_modeling_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activity, _users = data.load()
    train_df = features.build_modeling_table(activity, config.CUTOFF_TRAIN)
    valid_df = features.build_modeling_table(activity, config.CUTOFF_VALID)
    return activity, train_df, valid_df


def cmd_data(_args: argparse.Namespace) -> int:
    data.main()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    _activity, train_df, _valid_df = _load_modeling_tables()
    print(
        f"Training on cutoff day {config.CUTOFF_TRAIN}: "
        f"{len(train_df):,} users, churn rate {train_df['churned'].mean():.1%}"
    )
    model = train_model(train_df, calibration_method=args.calibration)
    model.save()
    train_prob = model.predict_proba(train_df)
    m = metrics.ranking_metrics(train_df["churned"].to_numpy(), train_prob)
    print(f"  in-sample ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}")
    print(f"  saved model -> {config.MODEL_PATH}")
    return 0


def cmd_evaluate(_args: argparse.Namespace) -> int:
    _activity, _train_df, valid_df = _load_modeling_tables()
    model = ChurnModel.load()

    y_true = valid_df["churned"].to_numpy()
    y_prob = model.predict_proba(valid_df)

    m = metrics.ranking_metrics(y_true, y_prob)
    lift = metrics.lift_by_decile(y_true, y_prob)
    importance = explain.global_importance(model, valid_df)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plots.plot_roc_pr(y_true, y_prob)
    plots.plot_calibration(y_true, y_prob)
    plots.plot_lift(y_true, y_prob)
    plots.plot_shap_summary(model, valid_df)

    payload = {
        "cutoff_valid": config.CUTOFF_VALID,
        "horizon": config.HORIZON,
        "metrics": m,
        "top1_decile_lift": float(lift.loc[0, "lift"]),
        "top3_decile_capture": float(lift.loc[2, "cumulative_capture"]),
        "lift_table": lift.to_dict(orient="records"),
        "global_importance": importance.to_dict(orient="records"),
    }
    config.METRICS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"Out-of-time validation (cutoff day {config.CUTOFF_VALID}, horizon {config.HORIZON}d):")
    print(f"  ROC-AUC = {m['roc_auc']:.3f}")
    print(f"  PR-AUC  = {m['pr_auc']:.3f}  (base rate {m['base_rate']:.3f})")
    print(f"  Brier   = {m['brier']:.3f}")
    print(f"  top-decile lift = {lift.loc[0, 'lift']:.2f}x")
    print(f"  top-3-decile capture = {lift.loc[2, 'cumulative_capture']:.1%}")
    print(f"  metrics -> {config.METRICS_PATH}")
    print(f"  plots   -> {config.REPORTS_DIR}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    activity, _users = data.load()
    feats = features.build_features(activity, args.cutoff)
    if feats.empty:
        print("No in-scope users at this cutoff.", file=sys.stderr)
        return 1
    model = ChurnModel.load()
    rc = explain.reason_codes(model, feats, top_k=args.top_k)
    rc = rc.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rc.to_csv(config.SCORES_PATH, index=False)
    print(f"Scored {len(rc):,} users at cutoff day {args.cutoff}.")
    print(f"  scores -> {config.SCORES_PATH}")
    print("\nHighest-risk users:")
    with pd.option_context("display.max_colwidth", 80, "display.width", 120):
        print(rc.head(args.show).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m churn",
        description="Point-in-time churn prediction on synthetic gaming/fintech users.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="generate the synthetic dataset")
    p_data.set_defaults(func=cmd_data)

    p_train = sub.add_parser("train", help="train + calibrate the model")
    p_train.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="isotonic")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="out-of-time evaluation + plots")
    p_eval.set_defaults(func=cmd_evaluate)

    p_score = sub.add_parser("score", help="score users with SHAP reason codes")
    p_score.add_argument("--cutoff", type=int, default=config.CUTOFF_VALID)
    p_score.add_argument("--top-k", type=int, default=3, help="reason codes per user")
    p_score.add_argument("--show", type=int, default=10, help="rows to print")
    p_score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
