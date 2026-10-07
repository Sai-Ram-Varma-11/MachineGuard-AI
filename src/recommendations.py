"""Transparent rule-based maintenance recommendations.

This is NOT a machine-learning model. It applies simple, readable rules to
the ML output and to where the inputs sit in the training distribution.
"""
from __future__ import annotations

from src.config import DISCLAIMER, FEATURE_LABELS


def generate_recommendations(
    inputs: dict, stats: dict, probability: float | None, predicted_label: int
) -> list[str]:
    """Return cautious, human-readable recommendations."""
    recs: list[str] = []
    high_risk = predicted_label == 1 or (probability is not None and probability >= 0.70)

    if high_risk:
        recs.append(
            "The model flags elevated failure risk: consider scheduling an "
            "inspection / preventive-maintenance review."
        )
    if inputs["tool_wear_min"] > stats["tool_wear_min"]["p90"]:
        recs.append(
            "Tool wear is high relative to the training data: consider checking "
            "tool condition against your equipment's maintenance policy."
        )
    if inputs["process_temp_k"] > stats["process_temp_k"]["p90"] or \
            inputs["air_temp_k"] > stats["air_temp_k"]["p90"]:
        recs.append(
            "Temperature is high relative to the training data: consider "
            "reviewing process and cooling conditions."
        )
    if inputs["torque_nm"] > stats["torque_nm"]["p90"]:
        recs.append(
            "Torque is high relative to the training data: consider reviewing "
            "machine load and mechanical conditions."
        )
    if inputs["rot_speed_rpm"] < stats["rot_speed_rpm"]["p10"] and \
            inputs["torque_nm"] > stats["torque_nm"]["p90"]:
        recs.append(
            "Low rotational speed combined with high torque is an unusual "
            "operating combination in this dataset; consider a closer review."
        )
    if not recs:
        recs.append(
            "No elevated indicators found: continue routine monitoring "
            "according to your normal maintenance schedule."
        )
    recs.append(DISCLAIMER)
    return recs


def describe_indicator_features() -> dict[str, str]:
    """Which features each rule looks at (used in the About page / docs)."""
    return {
        "Tool wear": FEATURE_LABELS["tool_wear_min"],
        "Temperature": f"{FEATURE_LABELS['air_temp_k']} / {FEATURE_LABELS['process_temp_k']}",
        "Torque": FEATURE_LABELS["torque_nm"],
    }
