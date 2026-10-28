"""
Walk-forward run dashboard.   streamlit run streamlit/dashboard.py

Page 1 compares runs (listed from MLflow, read from the run store), page 2 shows one run in detail.
The sidebar decides where the runs come from and how the VaR is read (run_store.Reading).
"""

import os

import streamlit as st

import dashboard_data as dd
import page_experiments
import page_run_comparison
import page_single_run
import page_weight_adjustment
import page_weight_comparison
import run_store as rs
import walk_forward as wf

# where the runs are: the MLflow index, its experiment (the one every training script logs to) and the run store
TRACKING_URI = f"sqlite:///{dd.ROOT / 'mlruns.db'}"
MLFLOW_EXPERIMENT = "walk_forward_horizon"
STORE_ROOT = str(dd.ROOT / "store")

os.chdir(dd.ROOT)    # the data paths a run stores (prices, additional feature CSVs) are relative to the project root, wherever streamlit was started

st.set_page_config(page_title="Walk-forward runs", layout="wide")

with st.sidebar:
    page_slot = st.container()    # on top: the page puts its own controls here (the single-run page: the run and its analysis period)

    st.header("VaR reading", help="How the VaR is read off the run's saved residuals. A reading that is not on disk yet is computed once and cached there.")
    var_confidence = st.select_slider("VaR confidence", [0.90, 0.95, 0.975, 0.99], value=0.95)
    expanding = st.checkbox("Expanding residual window", value=False)
    window_size = None if expanding else st.number_input("Residual window (days)", 20, 1000, 252, step=1)
    min_periods = st.number_input("Min periods", 5, 500, 30, step=1)
    test_confidence = st.select_slider("Test confidence", [0.90, 0.95, 0.99], value=0.95)

settings = dd.Settings(
    TRACKING_URI, MLFLOW_EXPERIMENT, STORE_ROOT,
    wf.Reading(var_confidence, None if window_size is None else int(window_size), int(min_periods), test_confidence),
)
navigation = st.navigation([
    st.Page(lambda: page_run_comparison.render(settings, page_slot), title="Run comparison", url_path="compare", default=True),
    st.Page(lambda: page_single_run.render(settings, page_slot), title="Single run", url_path="run"),
    st.Page(lambda: page_experiments.render(settings, page_slot), title="Experiments", url_path="experiments"),
    st.Page(lambda: page_weight_adjustment.render(settings, page_slot), title="Portfolio ticker weight adjustment", url_path="weights"),
    st.Page(lambda: page_weight_comparison.render(settings, page_slot), title="Model comparison with user defined weights", url_path="weight-comparison"),
])
navigation.run()
