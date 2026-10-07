import math

from src.prediction import predict_failure, risk_band, validate_inputs


def test_valid_prediction(trained, valid_input):
    model, stats, _ = trained
    r = predict_failure(valid_input, model, stats)
    assert r.valid and r.predicted_label in (0, 1)
    assert 0.0 <= r.probability <= 1.0 and r.risk_band in {"Low", "Medium", "High"}
    assert r.recommendations


def test_missing_input_is_reported_not_raised(trained, valid_input):
    model, stats, _ = trained
    del valid_input["torque_nm"]
    r = predict_failure(valid_input, model, stats)
    assert not r.valid and any("Torque" in e for e in r.errors)


def test_invalid_inputs(trained, valid_input):
    model, stats, _ = trained
    for key, bad in [("torque_nm", "abc"), ("air_temp_k", -5), ("tool_wear_min", -1),
                     ("rot_speed_rpm", math.nan), ("type", "Z"), ("process_temp_k", None)]:
        r = predict_failure({**valid_input, key: bad}, model, stats)
        assert not r.valid and r.errors, key


def test_extreme_valid_input_still_predicts_with_ood_warning(trained, valid_input):
    model, stats, _ = trained
    r = predict_failure({**valid_input, "rot_speed_rpm": 50000}, model, stats)
    assert r.valid
    assert any("outside the range observed in the training dataset" in w for w in r.warnings)


def test_in_range_input_has_no_warning(trained, valid_input):
    model, stats, _ = trained
    assert predict_failure(valid_input, model, stats).warnings == []


def test_risk_bands():
    assert risk_band(0.1) == "Low" and risk_band(0.5) == "Medium" and risk_band(0.9) == "High"


def test_type_is_case_insensitive(valid_input):
    clean, errors = validate_inputs({**valid_input, "type": " l "})
    assert not errors and clean["type"] == "L"
