"""Result plots committed to ``reports/`` and embedded in the README."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_curve,
    roc_curve,
)

from . import config, metrics  # noqa: E402
from .explain import shap_values  # noqa: E402
from .model import ChurnModel  # noqa: E402


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, path=None):
    path = path or config.REPORTS_DIR / "roc_pr_curve.png"
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    m = metrics.ranking_metrics(y_true, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, color="#2b8cbe", lw=2, label=f"ROC (AUC={m['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax[0].legend(loc="lower right")

    ax[1].plot(rec, prec, color="#e6550d", lw=2, label=f"PR (AP={m['pr_auc']:.3f})")
    ax[1].axhline(
        m["base_rate"], ls="--", color="grey", lw=1, label=f"base rate={m['base_rate']:.3f}"
    )
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="upper right")

    fig.suptitle("Out-of-time validation: ranking performance")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, path=None):
    path = path or config.REPORTS_DIR / "calibration_curve.png"
    table = metrics.calibration_table(y_true, y_prob, n_bins=10)
    m = metrics.ranking_metrics(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfectly calibrated")
    ax.plot(
        table["mean_predicted"],
        table["observed_frequency"],
        "o-",
        color="#31a354",
        lw=2,
        label=f"model (Brier={m['brier']:.3f})",
    )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn frequency",
        title="Calibration (reliability) curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_lift(y_true: np.ndarray, y_prob: np.ndarray, path=None):
    path = path or config.REPORTS_DIR / "lift_chart.png"
    table = metrics.lift_by_decile(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(table["decile"], table["lift"], color="#756bb1", alpha=0.85)
    ax.axhline(1.0, ls="--", color="grey", lw=1, label="random (lift=1)")
    ax.set(
        xlabel="Risk decile (1 = highest predicted risk)",
        ylabel="Lift vs base rate",
        title="Lift by decile",
    )
    ax.set_xticks(table["decile"])
    for bar, v in zip(bars, table["lift"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_shap_summary(model: ChurnModel, X: pd.DataFrame, path=None):
    path = path or config.REPORTS_DIR / "shap_summary.png"
    feats = list(model.feature_columns)
    sv = shap_values(model, X)

    plt.figure()
    shap.summary_plot(sv, X[feats], show=False, plot_size=(8, 5))
    fig = plt.gcf()
    fig.suptitle("SHAP global feature importance", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
