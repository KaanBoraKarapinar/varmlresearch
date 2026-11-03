"""
Page 5: which model works best with the weights I give. The Run comparison page (page_run_comparison.analysis, all its tabs) over runs that are
read for the user defined weights: every ticked run becomes a wf.ReweightedVirtualRun (its stored per-ticker forecasts, recombined with the
weights, nothing trained or saved) and GARCH / EWMA / historical are computed for the same weights (wf.TrainedInitialRun, not saved): they depend on
the portfolio's own return, so they are computed again for every weights that is applied.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

import dashboard_data as dd
import page_run_comparison as cmp
import page_weight_adjustment as wa
from core import walk_forward as wf


@st.cache_data(show_spinner=False)
def _reweightable(store_run_ids: tuple, store_root: str) -> tuple:
    """The store run ids whose per-ticker forecasts can be read for other weights (wf.reweight_blocker)."""
    return tuple(i for i in store_run_ids if wf.reweight_blocker(cmp._run(i, store_root)) is None)


@st.cache_resource(show_spinner=False, max_entries=300)
def _virtual(store_run_id: str, store_root: str, weights: tuple) -> wf.ReweightedVirtualRun:
    """Kept between reruns, so the readings it has computed are not computed again."""
    return wf.ReweightedVirtualRun(cmp._run(store_run_id, store_root), dict(weights))


@st.cache_resource(show_spinner="Computing GARCH, EWMA and historical for the user defined weights (about 10 s) ...", max_entries=10)
def _baseline_runs(store_run_id: str, store_root: str, weights: tuple) -> dict:
    return dd.baseline_runs(cmp._run(store_run_id, store_root), dict(weights), store_root)


def _ticker_set_key(run: wf.Run) -> str:
    """What the runs of one ticker set share: the tickers, the windows (train / forecast bars, start, end, ...), the unit and the price data; not the weights."""
    setting = {k: v for k, v in run.config["setting"].items() if k != "weights"}
    return json.dumps({"setting": setting, "data": run.config["data"]["sha256"]}, sort_keys=True, default=str)


@st.cache_data(show_spinner=False)
def _ticker_sets(store_run_ids: tuple, store_root: str) -> dict:
    """ticker-set key -> (its store run ids, label), the biggest first."""
    groups: dict[str, list[str]] = {}
    for run_id in store_run_ids:
        groups.setdefault(_ticker_set_key(cmp._run(run_id, store_root)), []).append(run_id)
    out = {}
    for key, ids in sorted(groups.items(), key=lambda item: -len(item[1])):
        run = cmp._run(ids[0], store_root)
        setting = run.config["setting"]
        names = f" ({', '.join(setting['tickers'])})" if len(setting["tickers"]) <= 6 else ""
        out[key] = (tuple(ids), f"{len(setting['tickers'])} tickers{names} · train {run.params['train_bars']} / forecast {run.params['forecast_bars']} · from {setting['start_date']} · data #{run.config['data']['sha256'][:6]}")
    return out


def render(settings: dd.Settings, sidebar_top) -> None:
    st.title("Model comparison with user defined weights")
    runs = cmp._runs(settings.tracking_uri, settings.experiment)
    if runs.empty:
        st.warning(f"No runs found in MLflow experiment '{settings.experiment}' ({settings.tracking_uri}).")
        return
    reading, store_root = settings.reading, settings.store_root
    ok_ids = set(_reweightable(tuple(runs["store_run_id"]), store_root))
    runs = runs[runs["store_run_id"].isin(ok_ids)]
    if runs.empty:
        st.warning("No run can be read for other weights yet (it needs the per-ticker forecasts and the first-window training bank).")
        return

    with sidebar_top:
        st.header("Ticker set", help="The tickers and the windows (train / forecast bars, start). The weights are yours: every run of the set is read with them, whatever weights it was trained or saved with.")
        sets = _ticker_sets(tuple(runs["store_run_id"]), store_root)
        set_key = st.selectbox("Ticker set", list(sets), format_func=lambda k: f"{sets[k][1]} ({len(sets[k][0])} runs)", key="wc_ticker_set", label_visibility="collapsed")
        in_set = runs[runs["store_run_id"].isin(sets[set_key][0])]
        st.header("Runs")
        shown = in_set.drop(columns=[c for c in in_set.columns if c.startswith("total_window_")])    # the metrics of the run's own weights
        filtered = cmp._filters(shown, prefix="wc")

    first_run = cmp._run(in_set["store_run_id"].iloc[0], store_root)
    tickers = list(first_run.config["setting"]["tickers"])
    st.header("Weights", help="The values entered are normalised so that they sum to 1. Equal weights to start with.")
    weights = wa._weights_editor(f"wc_{hashlib.sha256(set_key.encode()).hexdigest()[:8]}", pd.Series(1 / len(tickers), index=tickers))
    key = tuple(sorted(weights.round(12).items()))

    selected = cmp._runs_table(shown, filtered, prefix="wc")
    if not selected:
        st.info("Tick the runs to read with these weights. GARCH, EWMA and historical are always included.")
    ids = filtered.set_index("display_name").loc[selected, "store_run_id"] if selected else pd.Series(dtype=str)

    # the runs to compare, by label: the ticked ones read for these weights, then the three baselines of the same weights
    chosen: dict[str, wf.Run] = {label: _virtual(run_id, store_root, key) for label, run_id in ids.items()}
    baselines = _baseline_runs(first_run.run_id, store_root, key)
    suffix = f"train {first_run.params['train_bars']} f {first_run.params['forecast_bars']}"
    named = {name: f"{name} | {suffix}" for name in baselines}
    chosen.update({named[name]: run for name, run in baselines.items()})

    dates = dd.common_days(chosen, reading)
    if len(dates) < 30:
        st.warning(f"The runs share only {len(dates)} days: too few for a comparison.")
        return
    with sidebar_top:
        labels_chosen = list(chosen)
        default = next((i for i, label in enumerate(labels_chosen) if label.startswith("garch_wide")), 0)
        baseline = st.selectbox("Baseline run", labels_chosen, index=default, key="wc_baseline", help="The run the others are compared with (differences are run minus baseline, so negative is better).")
        first, last = dates[0].date(), dates[-1].date()
        with st.container(border=True):
            st.markdown("**Analysis period**", help="The days the compared runs have all forecast. Everything is re-computed on the period you pick.")
            picked = st.date_input("Analysis start – analysis end", value=(first, last), min_value=first, max_value=last, key="wc_period")
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pick the end date too.")
        st.stop()
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    with st.spinner("Reading the runs for these weights ..."):
        var_dfs = {label: cmp._slice(run.var_df(reading).loc[dates], start, end) for label, run in chosen.items()}
        dailies = {label: cmp._slice(run.daily(reading).loc[dates], start, end) for label, run in chosen.items()}
    if len(next(iter(var_dfs.values()))) < 30:
        st.warning("The analysis period has fewer than 30 days: pick a longer one.")
        st.stop()
    st.success(f"{len(chosen)} runs on {len(next(iter(var_dfs.values())))} common days: {start.date()} → {end.date()}")

    cmp.analysis(
        dd.run_table(chosen), chosen, var_dfs, dailies, baseline, settings, start, end,
        {named["ewma"]: baselines["ewma"]},
    )
