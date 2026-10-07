import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_pages.common import FAIL_COLOR, NORMAL_COLOR, get_data, get_json, guard, style_fig
from src.config import FEATURE_LABELS, NUM_FEATURES, TARGET


def render() -> None:
    st.title("Analytics")
    full = guard(get_data)
    eda = guard(get_json, "eda_summary.json")
    types = st.multiselect("Filter by machine type", ["L", "M", "H"], default=["L", "M", "H"])
    df = full[full["type"].isin(types)].copy()
    if df.empty:
        st.warning("Select at least one machine type.")
        return
    df["Status"] = df[TARGET].map({0: "No failure", 1: "Failure"})
    cmap = {"No failure": NORMAL_COLOR, "Failure": FAIL_COLOR}

    t1, t2, t3, t4 = st.tabs(["Overview", "Distributions", "Relationships", "Correlation"])
    with t1:
        a, b, c, d = st.columns(4)
        a.metric("Records (filtered)", f"{len(df):,}")
        b.metric("Failures", f"{int(df[TARGET].sum()):,}")
        c.metric("Failure %", f"{df[TARGET].mean():.2%}")
        d.metric("Missing values (raw)", eda["missing_values_total"])
        left, right = st.columns(2)
        with left:
            fig = px.pie(df, names="Status", color="Status", color_discrete_map=cmap, hole=0.5,
                         title="Question: how rare are failures?")
            st.plotly_chart(style_fig(fig))
        with right:
            g = df.groupby("type")[TARGET].agg(["size", "mean"]).reset_index()
            g["Failure rate %"] = g["mean"] * 100
            fig = px.bar(g, x="type", y="Failure rate %", text=g["Failure rate %"].round(2),
                         title="Question: does failure rate differ by machine type?",
                         color_discrete_sequence=[FAIL_COLOR])
            st.plotly_chart(style_fig(fig))
        with st.expander("Dataset summary statistics"):
            st.dataframe(df[NUM_FEATURES].describe().round(2).rename(columns=FEATURE_LABELS))
            st.caption(f"IQR outlier counts (full data): {eda['iqr_outlier_counts']}")
        with st.expander("Data-leakage check"):
            lc = eda["leakage_check"]
            st.write(f"Failure-mode columns {lc['dropped_failure_mode_columns']} and identifiers were excluded from the model. "
                     f"Share of rows where the target equals 'any failure-mode flag': {lc['share_of_rows_where_target_equals_any_mode_flag']:.4f}. {lc['note']}")
    with t2:
        feat = st.selectbox("Question: how is this feature distributed for failures vs normal runs?", NUM_FEATURES,
                            format_func=lambda c: FEATURE_LABELS[c])
        fig = px.histogram(df, x=feat, color="Status", barmode="overlay", opacity=0.65, histnorm="probability density",
                           color_discrete_map=cmap, nbins=40, marginal="box")
        st.plotly_chart(style_fig(fig, 460))
    with t3:
        feat = st.selectbox("Question: how does failure rate change across the range of a feature?", NUM_FEATURES,
                            format_func=lambda c: FEATURE_LABELS[c], key="rel")
        bins = pd.qcut(df[feat], q=10, duplicates="drop")
        r = df.groupby(bins, observed=True)[TARGET].mean().mul(100).reset_index()
        r[feat] = r[feat].astype(str)
        fig = px.bar(r, x=feat, y=TARGET, labels={TARGET: "Failure rate %", feat: FEATURE_LABELS[feat] + " (decile)"},
                     color_discrete_sequence=[FAIL_COLOR])
        st.plotly_chart(style_fig(fig))
        sample = df.sample(min(len(df), 4000), random_state=42)
        fig = px.scatter(sample, x="rot_speed_rpm", y="torque_nm", color="Status", opacity=0.6, color_discrete_map=cmap,
                         title="Question: where do failures sit in the torque-speed plane?",
                         labels={"rot_speed_rpm": FEATURE_LABELS["rot_speed_rpm"], "torque_nm": FEATURE_LABELS["torque_nm"]})
        st.plotly_chart(style_fig(fig, 420))
    with t4:
        corr = df[NUM_FEATURES + [TARGET]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation heatmap (association, not causation)")
        st.plotly_chart(style_fig(fig, 480))
