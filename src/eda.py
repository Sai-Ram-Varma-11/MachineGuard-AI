"""Exploratory data analysis: summary JSON + static figures.

The Streamlit Analytics page draws interactive Plotly charts from the
processed data; this module produces the numbers and report-ready PNGs.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (
    FIGURES_DIR, FEATURE_LABELS, LEAKAGE_COLUMNS, METRICS_DIR, NUM_FEATURES,
    TARGET, TARGET_RAW,
)
from src.utils import save_json


def summarise(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Compute dataset statistics (all values come from the real data)."""
    failures = int(clean[TARGET].sum())
    n = len(clean)
    q1, q3 = clean[NUM_FEATURES].quantile(0.25), clean[NUM_FEATURES].quantile(0.75)
    iqr = q3 - q1
    outliers = ((clean[NUM_FEATURES] < q1 - 1.5 * iqr) | (clean[NUM_FEATURES] > q3 + 1.5 * iqr)).sum()
    by_type = clean.groupby("type")[TARGET].agg(["size", "sum", "mean"])
    any_mode = raw[LEAKAGE_COLUMNS].max(axis=1)
    return {
        "shape_raw": list(raw.shape),
        "shape_clean": list(clean.shape),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "missing_values_total": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "class_distribution": {
            "non_failure": n - failures, "failure": failures,
            "failure_percent": round(100 * failures / n, 3),
        },
        "type_distribution": {k: int(v) for k, v in clean["type"].value_counts().items()},
        "failure_rate_by_type": {
            t: {"records": int(r["size"]), "failures": int(r["sum"]), "rate": float(r["mean"])}
            for t, r in by_type.iterrows()
        },
        "numeric_describe": clean[NUM_FEATURES].describe().round(3).to_dict(),
        "iqr_outlier_counts": {k: int(v) for k, v in outliers.items()},
        "correlation_with_target": clean[NUM_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).round(4).to_dict(),
        "leakage_check": {
            "dropped_failure_mode_columns": LEAKAGE_COLUMNS,
            "share_of_rows_where_target_equals_any_mode_flag": float((any_mode == raw[TARGET_RAW]).mean()),
            "note": "A value near 1.0 shows these columns encode the target, so they are excluded.",
        },
    }


def make_figures(clean: pd.DataFrame) -> None:
    """Save the key EDA figures as PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    clean[TARGET].value_counts().sort_index().plot.bar(ax=axes[0], color=["#457b9d", "#e76f51"])
    axes[0].set(title="Machine failure distribution", xticklabels=["No failure", "Failure"], xlabel="")
    axes[0].tick_params(axis="x", rotation=0)
    clean["type"].value_counts().plot.bar(ax=axes[1], color="#2a9d8f")
    axes[1].set(title="Machine type distribution", xlabel="")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_distributions.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), NUM_FEATURES):
        for val, color in [(0, "#457b9d"), (1, "#e76f51")]:
            ax.hist(clean.loc[clean[TARGET] == val, col], bins=30, alpha=0.6, density=True,
                    color=color, label="Failure" if val else "No failure")
        ax.set_title(FEATURE_LABELS[col], fontsize=9)
    axes.ravel()[0].legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_feature_histograms.png", dpi=150); plt.close(fig)

    corr = clean[NUM_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Correlation heatmap")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_correlation.png", dpi=150); plt.close(fig)

    rate = clean.groupby("type")[TARGET].mean().reindex(["L", "M", "H"])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.bar(rate.index, rate.values * 100, color="#e9c46a")
    ax.set(title="Failure rate by machine type (%)", ylabel="%")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_failure_by_type.png", dpi=150); plt.close(fig)


def run_eda(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    summary = summarise(raw, clean)
    save_json(summary, METRICS_DIR / "eda_summary.json")
    make_figures(clean)
    return summary
