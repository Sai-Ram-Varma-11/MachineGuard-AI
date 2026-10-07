import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_pages.common import get_csv, get_json, guard, style_fig


def render() -> None:
    st.title("Model Performance")
    metrics = guard(get_json, "metrics.json")
    curves = guard(get_json, "curves.json")
    table = guard(get_csv, "model_comparison.csv")
    sel = metrics["selected"]
    st.caption(f"All values are computed from the real trained models on the held-out test set "
               f"({metrics['dataset']['test_rows']:,} records; stratified 80/20 split, random_state={metrics['random_state']}).")

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_f1"]
    show = table[cols].rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                                       "f1": "F1 Score", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "cv_f1": "CV F1 (train)"})
    st.subheader("Model comparison")
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "Model"})
                 .apply(lambda r: ["background-color:#1f3b36" if r["Model"] == sel["model"] else "" for _ in r], axis=1),
                 hide_index=True)

    st.subheader(f"Selected model: {sel['model']}")
    st.write(sel["reason"])
    if sel["params"]:
        st.caption(f"Tuned hyperparameters: {sel['params']}")

    c1, c2 = st.columns(2)
    with c1:
        cm = sel["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Pred: No failure", "Pred: Failure"], y=["Actual: No failure", "Actual: Failure"],
                        color_continuous_scale="Teal", title="Confusion matrix (test set)")
        st.plotly_chart(style_fig(fig, 380))
        st.caption(f"False negatives (missed failures): {cm[1][0]} | False positives (false alarms): {cm[0][1]}")
    with c2:
        fig = go.Figure()
        for name, c in curves.items():
            fig.add_scatter(x=c["roc"]["fpr"], y=c["roc"]["tpr"], mode="lines", name=name)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance", showlegend=False)
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig, 380))

    fig = go.Figure()
    for name, c in curves.items():
        fig.add_scatter(x=c["pr"]["recall"], y=c["pr"]["precision"], mode="lines", name=name)
    fig.update_layout(title="Precision-Recall curves", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(style_fig(fig, 380))

    with st.expander("Why accuracy alone is not enough"):
        d = metrics["dataset"]
        st.write(f"Only {d['failure_percent']:.2f}% of records are failures. A model that always predicted 'no failure' would score "
                 f"about {100 - d['failure_percent']:.1f}% accuracy while detecting zero failures.")
    with st.expander("Why recall and F1 matter here"):
        st.write("Recall = share of real failures the model catches. A missed failure risk (false negative) is usually costlier than an "
                 "unnecessary inspection (false positive). F1 balances recall and precision in a single number.")
