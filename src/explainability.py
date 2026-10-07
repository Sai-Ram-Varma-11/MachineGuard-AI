"""Model explainability (global feature importance + input context).

Wording rule: we say a feature "had strong influence on the model's
prediction" - never that it "caused" a failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_LABELS, METRICS_DIR, NUM_FEATURES

IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"


def friendly_name(feature: str) -> str:
    """Readable label for a (possibly one-hot) feature name."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("type_"):
        return f"Machine Type = {feature.split('_', 1)[1]}"
    return feature


def global_importance(pipeline) -> pd.DataFrame:
    """Global importance from the fitted pipeline's classifier.

    - Tree models: ``feature_importances_`` (impurity-based).
    - Logistic Regression: absolute coefficient on standardised inputs.
    Importances are normalised to sum to 1 for easy comparison.
    """
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        raw = np.asarray(clf.feature_importances_, dtype=float)
        method = "Impurity-based feature_importances_"
    elif hasattr(clf, "coef_"):
        raw = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
        method = "Absolute standardised coefficient"
    else:
        raise ValueError("Model does not expose importances or coefficients.")
    total = raw.sum() or 1.0
    df = pd.DataFrame({"feature": names, "importance": raw / total})
    df["label"] = df["feature"].map(friendly_name)
    df["method"] = method
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def load_importance() -> pd.DataFrame:
    if not IMPORTANCE_PATH.exists():
        raise FileNotFoundError("Feature importance file missing. Run training first.")
    return pd.read_csv(IMPORTANCE_PATH)


def input_vs_training(inputs: dict, stats: dict) -> pd.DataFrame:
    """Show where each entered value sits relative to the training data."""
    rows = []
    for col in NUM_FEATURES:
        s, v = stats[col], float(inputs[col])
        position = "above typical range" if v > s["p90"] else (
            "below typical range" if v < s["p10"] else "within typical range")
        rows.append({
            "Feature": FEATURE_LABELS[col],
            "Entered value": round(v, 2),
            "Training median": round(s["median"], 2),
            "Typical range (P10-P90)": f"{s['p10']:.1f} - {s['p90']:.1f}",
            "Position": position,
        })
    return pd.DataFrame(rows)
