"""Model evaluation: metrics, curves and static figures.

WHY: Accuracy alone is misleading when failures are rare (a model that always
predicts "no failure" would look accurate). We therefore report precision,
recall, F1, ROC-AUC and PR-AUC computed from real predictions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

PLT_STYLE = {"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3}


def evaluate(model, X, y) -> dict:
    """Return metrics + confusion matrix for a fitted model on (X, y)."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _thin(a: np.ndarray, n: int = 200) -> list[float]:
    """Down-sample a curve to at most n points (keeps JSON small)."""
    idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
    return [round(float(v), 5) for v in a[idx]]


def curve_data(model, X, y) -> dict:
    """ROC and precision-recall curve points for one model."""
    prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    return {
        "roc": {"fpr": _thin(fpr), "tpr": _thin(tpr)},
        "pr": {"precision": _thin(prec), "recall": _thin(rec)},
    }


def plot_confusion_matrix(cm: list[list[int]], path: Path, title: str) -> None:
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.grid(False)
        ax.imshow(cm, cmap="Blues")
        labels = ["No failure", "Failure"]
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_curves(curves: dict, kind: str, path: Path) -> None:
    """kind = 'roc' or 'pr'; curves = {model_name: curve_data(...)}."""
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, c in curves.items():
            if kind == "roc":
                ax.plot(c["roc"]["fpr"], c["roc"]["tpr"], label=name)
            else:
                ax.plot(c["pr"]["recall"], c["pr"]["precision"], label=name)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title="ROC curves (test set)")
        else:
            ax.set(xlabel="Recall", ylabel="Precision",
                   title="Precision-Recall curves (test set)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_feature_importance(df, path: Path) -> None:
    with plt.rc_context(PLT_STYLE):
        d = df.sort_values("importance")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.barh(d["label"], d["importance"], color="#2a9d8f")
        ax.set_xlabel("Relative importance")
        ax.set_title("Features influencing the model")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
