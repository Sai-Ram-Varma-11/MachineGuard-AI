from src.config import DISCLAIMER
from src.recommendations import generate_recommendations

STATS = {k: {"p10": 1, "p90": 10, "median": 5, "min": 0, "max": 20, "mean": 5, "std": 2}
         for k in ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]}
BASE = {"air_temp_k": 5, "process_temp_k": 5, "rot_speed_rpm": 5, "torque_nm": 5, "tool_wear_min": 5}


def test_normal_gets_monitoring_and_disclaimer():
    recs = generate_recommendations(BASE, STATS, 0.05, 0)
    assert any("routine monitoring" in r for r in recs) and recs[-1] == DISCLAIMER


def test_high_risk_recommends_inspection():
    assert any("inspection" in r for r in generate_recommendations(BASE, STATS, 0.9, 1))


def test_each_rule_triggers():
    assert any("tool condition" in r for r in generate_recommendations({**BASE, "tool_wear_min": 15}, STATS, 0.1, 0))
    assert any("cooling" in r for r in generate_recommendations({**BASE, "process_temp_k": 15}, STATS, 0.1, 0))
    assert any("machine load" in r for r in generate_recommendations({**BASE, "torque_nm": 15}, STATS, 0.1, 0))


def test_no_hard_instructions_language():
    text = " ".join(generate_recommendations({**BASE, "torque_nm": 15}, STATS, 0.9, 1)).lower()
    assert "shut down" not in text and "must" not in text
