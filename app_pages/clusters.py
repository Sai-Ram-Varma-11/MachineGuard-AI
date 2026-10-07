import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_pages.common import get_csv, get_json, guard, style_fig
from src.config import FEATURE_LABELS


def render() -> None:
    st.title("Machine Clusters")
    st.write("K-Means groups machine records by similar operating conditions **without using the failure label**. "
             "It is exploratory: clusters are named by number, not as 'healthy' or 'dangerous'.")
    info = guard(get_json, "clusters.json")
    ktab = guard(get_csv, "cluster_k_selection.csv")
    profile = guard(get_csv, "cluster_profile.csv")
    pts = guard(get_csv, "cluster_points.csv")

    st.info(f"Chosen K = **{info['chosen_k']}** ({info['selection_rule']}).")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ktab, x="k", y="inertia", markers=True, title="Elbow method (inertia)")
        st.plotly_chart(style_fig(fig, 320))
    with c2:
        fig = px.line(ktab, x="k", y="silhouette", markers=True, title="Silhouette score by K")
        st.plotly_chart(style_fig(fig, 320))

    ev = info["pca_explained_variance_ratio"]
    sample = pts.sample(min(len(pts), 4000), random_state=42).copy()
    sample["Cluster"] = "Cluster " + sample["cluster"].astype(str)
    fig = px.scatter(sample, x="pc1", y="pc2", color="Cluster", opacity=0.6,
                     labels={"pc1": f"PCA Component 1 ({ev[0]:.0%})", "pc2": f"PCA Component 2 ({ev[1]:.0%})"},
                     title="Clusters in PCA space")
    st.plotly_chart(style_fig(fig, 440))
    st.caption("PCA compresses the 5 scaled sensor features into 2 axes purely for visualisation; it is not the prediction model.")

    st.subheader("Cluster statistics")
    show = profile.rename(columns={**FEATURE_LABELS, "cluster": "Cluster", "count": "Count", "failure_rate": "Failure rate"})
    st.dataframe(show.style.format({c: "{:.1f}" for c in show.columns if c not in ("Cluster", "Count", "Failure rate")}
                                   | {"Failure rate": "{:.2%}"}), hide_index=True)
    fig = px.bar(profile, x="cluster", y="count", title="Cluster size", labels={"cluster": "Cluster", "count": "Machines"})
    st.plotly_chart(style_fig(fig, 300))

    st.subheader("Interpretation (generated from the data)")
    for text in info["descriptions"]:
        st.write("- " + text)
    with st.expander("K-Means in simple language"):
        st.write("Imagine dropping K pins on a map of machine records. Each record joins the nearest pin, the pins move to the middle "
                 "of their group, and this repeats until nothing changes. Features are scaled first so large-valued features like "
                 "rpm do not dominate small-valued ones.")
