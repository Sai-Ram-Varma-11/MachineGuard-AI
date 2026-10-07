"""MachineGuard AI - Streamlit entry point.   Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="MachineGuard AI", page_icon="🛡️", layout="wide")

from app_pages import about, analytics, clusters, common, dashboard, performance, predict  # noqa: E402

PAGES = {
    "🏠 Dashboard": dashboard.render,
    "🔮 Predict Failure": predict.render,
    "📊 Analytics": analytics.render,
    "🧠 Model Performance": performance.render,
    "🔬 Machine Clusters": clusters.render,
    "ℹ️ About": about.render,
}

st.markdown(common.CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## MachineGuard AI")
    st.caption("Predict. Understand. Maintain.")
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Educational prototype. Not a certified industrial safety or control system.")

try:
    PAGES[choice]()
except Exception:  # last-resort guard: no raw tracebacks for users
    st.error("An unexpected error occurred while rendering this page. Please refresh or re-run training.")
