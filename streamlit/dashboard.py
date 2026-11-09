"""
Walk-forward run dashboard.   streamlit run streamlit/dashboard.py

The pages, in order: Approach (the README), Results (the written results, linked to the experiments), Run comparison (runs listed from MLflow, read from the run
store), Single run, Experiments, Portfolio ticker weight adjustment, Model comparison with user defined weights.
The sidebar decides how the VaR is read (run_store.Reading).

When the run store is in a bucket (the setting STORE_REMOTE, see data_access.py) Approach, Results and Experiments (at the reading of the saved experiments) show the
results cache of the repository without a password; the other pages, another VaR reading and the per ticker variances ask for it and then fetch the store.
"""

import os

import streamlit as st

import dashboard_data as dd
import page_approach
import page_experiments
import page_results
import page_run_comparison
import page_single_run
import page_weight_adjustment
import page_weight_comparison
from core import results_cache as rc
from page_gates import cached_page, data_page
from core import run_store as rs
from core import walk_forward as wf

# where the runs are: the MLflow index, its experiment (the one every training script logs to) and the run store; and the results cache of the repository
# (what the Results and Experiments pages show before the password, when the store is in a bucket)
TRACKING_URI = f"sqlite:///{dd.ROOT / 'mlruns.db'}"
MLFLOW_EXPERIMENT = "walk_forward_horizon"
STORE_ROOT = str(dd.ROOT / "store")
CACHE_TRACKING_URI = f"sqlite:///{dd.ROOT / rc.CACHE_DIR / 'mlruns.db'}"
CACHE_STORE_ROOT = str(dd.ROOT / rc.CACHE_DIR / "store")

os.chdir(dd.ROOT)    # the data paths a run stores (prices, additional feature CSVs) are relative to the project root, wherever streamlit was started

st.set_page_config(page_title="Walk-forward runs", layout="wide")

with st.sidebar:
    page_slot = st.container()    # on top: the page puts its own controls here (the single-run page: the run and its analysis period)

    st.header("VaR reading", help="How the VaR is read off the run's saved residuals. A reading that is not on disk yet is computed once and cached there.")
    start = rc.CACHED_READING       # the sidebar starts at the reading the results cache holds
    var_confidence = st.select_slider("VaR confidence", [0.90, 0.95, 0.975, 0.99], value=start.var_confidence)
    expanding = st.checkbox("Expanding residual window", value=start.window_size is None)
    window_size = None if expanding else st.number_input("Residual window (days)", 20, 1000, start.window_size, step=1)
    min_periods = st.number_input("Min periods", 5, 500, start.min_periods, step=1)
    test_confidence = st.select_slider("Test confidence", [0.90, 0.95, 0.99], value=start.test_confidence)

reading = wf.Reading(var_confidence, None if window_size is None else int(window_size), int(min_periods), test_confidence)
settings = dd.Settings(TRACKING_URI, MLFLOW_EXPERIMENT, STORE_ROOT, reading)                          # the whole run store
cache_settings = dd.Settings(CACHE_TRACKING_URI, MLFLOW_EXPERIMENT, CACHE_STORE_ROOT, reading)       # the results cache


experiments_page = st.Page(
    cached_page(lambda s: page_experiments.render(s, page_slot), settings, cache_settings, any_reading=False), title="Experiments", url_path="experiments",
)
weight_comparison_page = st.Page(
    data_page(lambda s: page_weight_comparison.render(s, page_slot), settings), title="Model comparison with user defined weights", url_path="weight-comparison",
)
navigation = st.navigation([
    st.Page(lambda: page_approach.render(), title="Approach", url_path="approach", default=True),      # needs no data: open
    st.Page(
        cached_page(lambda s: page_results.render(s, page_slot, experiments_page, weight_comparison_page), settings, cache_settings, any_reading=True),
        title="Results", url_path="results",
    ),
    st.Page(data_page(lambda s: page_run_comparison.render(s, page_slot), settings), title="Run comparison", url_path="compare"),
    st.Page(data_page(lambda s: page_single_run.render(s, page_slot), settings), title="Single run", url_path="run"),
    experiments_page,
    st.Page(data_page(lambda s: page_weight_adjustment.render(s, page_slot), settings), title="Portfolio ticker weight adjustment", url_path="weights"),
    weight_comparison_page,
])
navigation.run()
