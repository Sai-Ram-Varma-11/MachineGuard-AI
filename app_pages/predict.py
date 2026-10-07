import plotly.graph_objects as go
import streamlit as st

from app_pages.common import ACCENT, get_importance, get_model, get_ranges, guard, style_fig
from src.config import DISCLAIMER, FEATURE_LABELS, RISK_BAND_HIGH, RISK_BAND_LOW, TYPE_VALUES
from src.explainability import input_vs_training
from src.prediction import predict_failure, validate_inputs


def render() -> None:
    st.title("Predict Failure")
    st.write("Enter machine operating parameters. The saved model returns a **model-estimated** failure risk.")
    model, ranges, imp = guard(get_model), guard(get_ranges), guard(get_importance)

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        mtype = c1.selectbox(FEATURE_LABELS["type"], TYPE_VALUES, index=1)
        air = c1.number_input(FEATURE_LABELS["air_temp_k"], value=round(ranges["air_temp_k"]["median"], 1), step=0.1,
                              help=f"Training range: {ranges['air_temp_k']['min']:g} - {ranges['air_temp_k']['max']:g}")
        proc = c1.number_input(FEATURE_LABELS["process_temp_k"], value=round(ranges["process_temp_k"]["median"], 1), step=0.1,
                               help=f"Training range: {ranges['process_temp_k']['min']:g} - {ranges['process_temp_k']['max']:g}")
        speed = c2.number_input(FEATURE_LABELS["rot_speed_rpm"], value=float(round(ranges["rot_speed_rpm"]["median"])), step=10.0,
                                help=f"Training range: {ranges['rot_speed_rpm']['min']:g} - {ranges['rot_speed_rpm']['max']:g}")
        torque = c2.number_input(FEATURE_LABELS["torque_nm"], value=round(ranges["torque_nm"]["median"], 1), step=0.5,
                                 help=f"Training range: {ranges['torque_nm']['min']:g} - {ranges['torque_nm']['max']:g}")
        wear = c2.number_input(FEATURE_LABELS["tool_wear_min"], value=float(round(ranges["tool_wear_min"]["median"])), step=1.0,
                               help=f"Training range: {ranges['tool_wear_min']['min']:g} - {ranges['tool_wear_min']['max']:g}")
        submitted = st.form_submit_button("Predict Machine Failure Risk", type="primary")

    if not submitted:
        st.caption("Defaults are the training-data medians. Values outside the training range are allowed but trigger a reliability warning.")
        return

    raw = {"type": mtype, "air_temp_k": air, "process_temp_k": proc,
           "rot_speed_rpm": speed, "torque_nm": torque, "tool_wear_min": wear}
    res = predict_failure(raw, model, ranges, imp)
    if not res.valid:
        for err in res.errors:
            st.error(err)
        return

    for w in res.warnings:
        st.warning("⚠ " + w)

    if res.predicted_label == 1:
        st.error(f"Machine Status: **{res.status}**")
    else:
        st.success(f"Machine Status: **{res.status}**")
    m1, m2 = st.columns(2)
    if res.probability is not None:
        m1.metric("Model-estimated failure probability", f"{res.probability:.1%}")
        m2.metric("Risk band (application-defined, illustrative)", res.risk_band)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=res.probability * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT},
                   "steps": [{"range": [0, RISK_BAND_LOW * 100], "color": "#1f3b36"},
                             {"range": [RISK_BAND_LOW * 100, RISK_BAND_HIGH * 100], "color": "#4a3b1b"},
                             {"range": [RISK_BAND_HIGH * 100, 100], "color": "#4a2a1b"}]}))
        st.plotly_chart(style_fig(gauge, 260))
    st.caption("This is a model estimate learned from a synthetic dataset, **not** the actual probability of real-world failure. "
               f"Risk bands ({RISK_BAND_LOW:.0%} / {RISK_BAND_HIGH:.0%}) are illustrative and not empirically validated. "
               "Status uses the model's default 0.5 decision threshold.")

    st.subheader("Important factors considered by the model")
    for f in res.top_features:
        st.write(f"- **{f['feature']}** had a strong influence on the model's predictions overall ({f['importance']:.1%} of total importance).")
    clean, _ = validate_inputs(raw)
    st.dataframe(input_vs_training(clean, ranges), hide_index=True)

    st.subheader("Maintenance insight")
    for r in res.recommendations[:-1]:
        st.write(f"- {r}")
    st.caption(res.recommendations[-1] or DISCLAIMER)
