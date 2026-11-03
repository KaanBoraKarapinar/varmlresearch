"""Page 3: saved experiments. An experiment fixes the runs, the baseline, the analysis period and the VaR reading; the page shows them with our conclusion."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import streamlit as st

import core
import dashboard_data as dd
import page_run_comparison as cmp
from core import walk_forward as wf


@st.cache_data(show_spinner=False)
def _experiments(store_root: str, stamp: tuple) -> list[core.Experiment]:
    """stamp (file names + modification times) makes the cache notice a saved or edited experiment."""
    return core.list_saved_experiments(Path(store_root))


def _stamp(store_root: str) -> tuple:
    folder = core.experiment.experiments_dir(Path(store_root))
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(folder.glob("*.json"))) if folder.exists() else ()


def render(settings: dd.Settings, sidebar_top) -> None:
    st.title("Experiments")
    saved = _experiments(settings.store_root, _stamp(settings.store_root))
    if not saved:
        st.info("No experiment saved yet: create one with core.save_experiment(core.Experiment(...), store_root).")
        return
    with sidebar_top:
        st.header("Experiment")
        experiment = st.selectbox("Experiment", saved, format_func=lambda e: e.name, key="exp_pick", label_visibility="collapsed")

    reading = wf.Reading(**experiment.reading)
    # the experiment's own reading is used, not the one in the left panel: the result must not change with the sidebar
    settings = dataclasses.replace(settings, reading=reading)

    runs = cmp._runs(settings.tracking_uri, settings.experiment)
    by_id = runs.set_index("store_run_id", drop=False) if not runs.empty else runs
    missing = [i for i in experiment.run_ids if i not in by_id.index]
    if missing:
        st.error(f"{len(missing)} run(s) of this experiment are not in the MLflow experiment '{settings.experiment}': {', '.join(missing)}")
        return
    table = by_id.loc[list(experiment.run_ids)].set_index("display_name")    # same order as saved, index = the run labels
    labels = list(table.index)
    baseline = table.index[list(table["store_run_id"]).index(experiment.baseline_run_id)]

    st.subheader(experiment.name)
    if experiment.question:
        st.markdown(f"*{experiment.question}*")
    if experiment.notes:
        st.markdown(experiment.notes)
    st.markdown(
        f":blue[**EXPERIMENT:**] {len(labels)} runs · baseline {baseline} · analysis period {experiment.start} → {experiment.end} · "
        f"Var Confidence: {reading.var_confidence:g} · Min Periods: {reading.min_periods} · "
        f"Residual Window: {'expanding' if reading.window_size is None else reading.window_size} · Test Confidence: {reading.test_confidence:g}",
        help="Fixed when the experiment was saved; the VaR reading in the left panel does not change it.",
    )
    others = [e.name for e in saved if e.name != experiment.name and set(e.run_ids) & set(experiment.run_ids)]
    if others:
        st.caption("Runs of this experiment are also in: " + ", ".join(others))

    common = cmp._common(tuple(table["store_run_id"]), tuple(labels), settings.store_root, reading)
    start, end = pd.Timestamp(experiment.start), pd.Timestamp(experiment.end)
    var_dfs = {label: cmp._slice(frame, start, end) for label, frame in common["var_dfs"].items()}
    dailies = {label: cmp._slice(frame, start, end) for label, frame in common["dailies"].items()}
    days = len(next(iter(var_dfs.values())))
    if days < 30:
        st.warning(f"The runs share only {days} days in the saved analysis period.")
        return
    chosen = {label: cmp._run(table.loc[label, "store_run_id"], settings.store_root) for label in labels}
    cmp.analysis(table, chosen, var_dfs, dailies, baseline, settings, start, end, cmp._ewma_references(runs, settings.store_root))
