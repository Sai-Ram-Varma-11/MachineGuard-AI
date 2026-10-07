import pandas as pd
import plotly.express as px
import streamlit as st

from app_pages.common import FAIL_COLOR, NORMAL_COLOR, get_data, get_importance, get_json, get_model, get_ranges, guard, style_fig
from src.config import FEATURES, DISCLAIMER
from src.prediction import predict_failure


def render() -> None:
    st.title("MachineGuard AI")
    st.markdown("**Intelligent Predictive Maintenance & Machine Failure Detection**")
    st.markdown('<p class="mg-tag">Predict. Understand. Maintain.</p>', unsafe_allow_html=True)

    df = guard(get_data)
    metrics = guard(get_json, "metrics.json")
    sel = metrics["selected"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Failure Rate", f"{df['machine_failure'].mean():.2%}")
    c3.metric("Best Model", sel["model"])
    c4.metric("Model F1 Score (test)", f"{sel['f1']:.3f}")

    left, right = st.columns(2)
    with left:
        counts = df["machine_failure"].map({0: "No failure", 1: "Failure"}).value_counts().reset_index()
        counts.columns = ["Class", "Records"]
        fig = px.bar(counts, x="Class", y="Records", color="Class", text="Records", title="Failure distribution",
                     color_discrete_map={"No failure": NORMAL_COLOR, "Failure": FAIL_COLOR})
        st.plotly_chart(style_fig(fig))
    with right:
        imp = guard(get_importance)
        fig = px.bar(imp.sort_values("importance"), x="importance", y="label", orientation="h",
                     title="Features influencing the model", color_discrete_sequence=["#2DD4BF"])
        st.plotly_chart(style_fig(fig))
        st.caption(f"Method: {imp['method'].iloc[0]}. Strong influence on the model's prediction, not proof of cause.")

    st.subheader("How MachineGuard AI works")
    steps = ["Machine Data", "Preprocessing", "ML Model", "Failure Risk", "Explanation", "Maintenance Insight"]
    st.markdown(" → ".join(f'<span class="mg-step">{s}</span>' for s in steps), unsafe_allow_html=True)
    st.write("Traditional maintenance may rely on fixed schedules or reacting after a problem occurs. "
             "MachineGuard AI learns patterns from historical operating data to estimate failure risk and "
             "present understandable maintenance-oriented insights.")
    a, b, c = st.columns(3)
    a.info("**Dataset**  \nAI4I 2020 Predictive Maintenance (synthetic, UCI)")
    b.info("**Machine learning**  \nSupervised classification + unsupervised K-Means")
    c.info("**Framework**  \nStreamlit + scikit-learn + Plotly")

    st.subheader("Example prediction")
    model, ranges = guard(get_model), guard(get_ranges)
    example = {"type": df["type"].mode()[0], **{k: ranges[k]["median"] for k in ranges}}
    res = predict_failure(example, model, ranges)
    st.caption("Computed live by the saved model using the training-data median of each sensor value "
               "and the most common machine type.")
    if res.valid:
        e1, e2, e3 = st.columns(3)
        e1.metric("Machine status", res.status)
        e2.metric("Model-estimated failure probability", f"{res.probability:.1%}")
        e3.metric("Risk band (illustrative)", res.risk_band)
    st.caption(DISCLAIMER)
