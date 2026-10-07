"""Shared helpers for the Streamlit pages: cached loaders, theme, guards."""
from __future__ import annotations

import pandas as pd
import plotly.io as pio
import streamlit as st

from src.config import METRICS_DIR
from src.data_loader import load_processed
from src.explainability import load_importance
from src.prediction import ModelLoadError, load_model, load_ranges
from src.utils import load_json

ACCENT = "#2DD4BF"
NORMAL_COLOR = "#4C8DF6"
FAIL_COLOR = "#F59E0B"
pio.templates.default = "plotly_dark"

CSS = """
<style>
[data-testid="stMetric"]{background:#1C2128;border:1px solid #2D333B;border-radius:8px;padding:12px 16px;}
[data-testid="stSidebar"]{border-right:1px solid #2D333B;}
h1,h2,h3{font-weight:600;letter-spacing:.2px;}
.mg-tag{color:#8B949E;font-size:.9rem;margin-top:-.6rem;}
.mg-card{background:#1C2128;border:1px solid #2D333B;border-radius:8px;padding:14px 18px;margin-bottom:10px;}
.mg-step{display:inline-block;background:#1C2128;border:1px solid #2D333B;border-radius:6px;padding:6px 12px;margin:2px;font-size:.85rem;}
</style>
"""


def style_fig(fig, height: int = 360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_ranges() -> dict:
    return load_ranges()


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_processed()


@st.cache_data(show_spinner=False)
def get_json(name: str):
    return load_json(METRICS_DIR / name)


@st.cache_data(show_spinner=False)
def get_csv(name: str) -> pd.DataFrame:
    path = METRICS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_importance() -> pd.DataFrame:
    return load_importance()


def guard(loader, *args):
    """Run a loader; on failure show a friendly error and stop the page."""
    try:
        return loader(*args)
    except (FileNotFoundError, ModelLoadError) as exc:
        st.error(f"{exc}  \n\nRun `python src/train_models.py` from the repository root, "
                 "then refresh this page.")
    except Exception:  # never show raw tracebacks to users
        st.error("Something went wrong while loading project files. "
                 "Please re-run `python src/train_models.py`.")
    st.stop()
