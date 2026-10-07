"""Reusable prediction engine (used by the Streamlit app and the tests).

Flow: validate -> out-of-distribution check -> pipeline.predict_proba ->
risk band -> importance -> recommendations. It never raises on bad user
input; it returns a result with ``valid=False`` and readable errors.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH, FEATURE_LABELS, FEATURES, NUM_FEATURES, RANGES_PATH,
    RISK_BAND_HIGH, RISK_BAND_LOW, TYPE_VALUES,
)
from src.explainability import friendly_name
from src.recommendations import generate_recommendations
from src.utils import load_json

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when the saved model is missing or unreadable."""


@dataclass
class PredictionResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    predicted_label: int | None = None
    status: str = ""
    probability: float | None = None
    risk_band: str | None = None
    top_features: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def load_model(path=BEST_MODEL_PATH):
    """Load the saved pipeline with friendly errors."""
    if not path.exists():
        raise ModelLoadError("Trained model not found. Run `python src/train_models.py` first.")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Model file could not be loaded (corrupted or "
                             f"incompatible scikit-learn version): {exc}") from exc


def load_ranges(path=RANGES_PATH) -> dict:
    try:
        return load_json(path)
    except FileNotFoundError as exc:
        raise ModelLoadError("Training ranges file missing. Run training first.") from exc


def validate_inputs(raw: dict) -> tuple[dict, list[str]]:
    """Return (clean_inputs, errors). Only impossible values are errors."""
    errors: list[str] = []
    clean: dict = {}
    for key in FEATURES:
        label = FEATURE_LABELS[key]
        if key not in raw or raw[key] is None or (isinstance(raw[key], str) and not raw[key].strip()):
            errors.append(f"{label} is required.")
            continue
        if key == "type":
            value = str(raw[key]).strip().upper()
            if value not in TYPE_VALUES:
                errors.append(f"{label} must be one of {TYPE_VALUES}.")
            else:
                clean[key] = value
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not math.isfinite(value):
            errors.append(f"{label} must be a finite number.")
        elif key in ("air_temp_k", "process_temp_k", "rot_speed_rpm") and value <= 0:
            errors.append(f"{label} must be greater than 0.")
        elif key in ("torque_nm", "tool_wear_min") and value < 0:
            errors.append(f"{label} cannot be negative.")
        else:
            clean[key] = value
    return clean, errors


def out_of_range_warnings(inputs: dict, stats: dict) -> list[str]:
    """Warn (do not block) for values outside the observed training range."""
    warnings = []
    for col in NUM_FEATURES:
        lo, hi = stats[col]["min"], stats[col]["max"]
        if inputs[col] < lo or inputs[col] > hi:
            warnings.append(
                f"Warning: {FEATURE_LABELS[col]} = {inputs[col]:g} is outside the "
                f"range observed in the training dataset ({lo:g} - {hi:g}). "
                "The prediction may be less reliable."
            )
    return warnings


def risk_band(probability: float) -> str:
    """Application-defined ILLUSTRATIVE bands (not empirically validated)."""
    if probability < RISK_BAND_LOW:
        return "Low"
    if probability < RISK_BAND_HIGH:
        return "Medium"
    return "High"


def predict_failure(raw_inputs: dict, model, stats: dict,
                    importance: pd.DataFrame | None = None) -> PredictionResult:
    """Full prediction. Never raises for bad user input."""
    inputs, errors = validate_inputs(raw_inputs)
    if errors:
        return PredictionResult(valid=False, errors=errors)
    try:
        frame = pd.DataFrame([inputs])[FEATURES]
        label = int(model.predict(frame)[0])
        prob = float(model.predict_proba(frame)[0][1]) if hasattr(model, "predict_proba") else None
    except Exception as exc:  # unexpected model/data problem
        logger.exception("Prediction failed")
        return PredictionResult(valid=False, errors=[f"Prediction could not be completed: {exc}"])

    top = []
    if importance is not None and len(importance):
        for _, row in importance.head(3).iterrows():
            top.append({"feature": row.get("label", friendly_name(row["feature"])),
                        "importance": float(row["importance"])})
    return PredictionResult(
        valid=True,
        warnings=out_of_range_warnings(inputs, stats),
        predicted_label=label,
        status="FAILURE RISK FLAGGED" if label == 1 else "NORMAL OPERATION",
        probability=prob,
        risk_band=risk_band(prob) if prob is not None else None,
        top_features=top,
        recommendations=generate_recommendations(inputs, stats, prob, label),
    )
