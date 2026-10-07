"""Preprocessing: leakage guard, ColumnTransformer and stratified split.

WHY: Machine learning models need numbers in a consistent format. Putting all
steps inside one scikit-learn object guarantees training and prediction use
EXACTLY the same transformation (no duplicated logic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CAT_FEATURES, FEATURES, ID_COLUMNS, HIGH_PERCENTILE, LEAKAGE_COLUMNS,
    NUM_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, TYPE_VALUES,
)


def assert_no_leakage(columns: list[str]) -> None:
    """Raise if identifiers, failure-mode flags or the target are in features."""
    forbidden = set(ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET, "Machine failure"])
    found = forbidden.intersection(columns)
    if found:
        raise ValueError(f"Data leakage risk - forbidden feature columns: {sorted(found)}")


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Build the ColumnTransformer.

    - Machine Type (categorical) -> OneHotEncoder with fixed categories.
    - Numeric sensors -> StandardScaler (only for scale-sensitive models such
      as Logistic Regression); tree models receive values unchanged.
    The transformer is fitted ONLY on training data (inside the Pipeline).
    """
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, NUM_FEATURES),
            ("cat", OneHotEncoder(categories=[TYPE_VALUES], handle_unknown="ignore"),
             CAT_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def split_data(df: pd.DataFrame):
    """80/20 stratified split with random_state=42."""
    X, y = df[FEATURES], df[TARGET]
    assert_no_leakage(list(X.columns))
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def compute_training_stats(X_train: pd.DataFrame) -> dict:
    """Per-feature training distribution (used for OOD warnings and rules)."""
    stats: dict = {}
    for col in NUM_FEATURES:
        s = X_train[col]
        stats[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "p10": float(np.percentile(s, 10)),
            "p90": float(np.percentile(s, HIGH_PERCENTILE)),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }
    return stats
