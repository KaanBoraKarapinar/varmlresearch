"""
Page 5: which model works best with the weights I give. The Run comparison page (page_run_comparison.analysis, all its tabs) over runs that are
read for the user defined weights: every ticked run becomes a wf.ReweightedVirtualRun (its stored per-ticker forecasts, recombined with the
weights, nothing trained or saved) and GARCH / EWMA / historical are computed for the same weights (wf.TrainedInitialRun, not saved): they depend on
the portfolio's own return, so they are computed again for every weights that is applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_data as dd
import page_run_comparison as cmp
import page_weight_adjustment as wa
from core import portfolios as pf
from core import walk_forward as wf

TICKER_SET_KEY = "wc_ticker_set"        # the ticker set selectbox
OPEN_SET_KEY = "wc_open_set"            # {"n_tickers", "train_bars", "forecast_bars"} of a ticker set another page (Results) wants opened here; a widget's own key would be dropped on the way


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


def _find_set(sets: dict, store_root: str, n_tickers: int, train_bars: int, forecast_bars: int) -> str | None:
    """The key of the ticker set with that many tickers and those windows (the biggest one when several), or None."""
    for key, (ids, _) in sets.items():
        run = cmp._run(ids[0], store_root)
        if len(run.config["setting"]["tickers"]) == n_tickers and run.params["train_bars"] == train_bars and run.params["forecast_bars"] == forecast_bars:
            return key
    return None


def _load_portfolio_panel(editor_id: str, tickers: list[str], store_root: str) -> None:
    """
    Load portfolio: a portfolio registered in the store (core.portfolios: the random Dirichlet portfolios, equal weights of a part of the tickers ...) that has exactly the
    tickers of the set, put in the weights table below and applied. The saved portfolios are not listed on the Run comparison or the Single run page: they are here.
    """
    registered = pf.portfolios_table(Path(store_root))
    if registered.empty:
        return
    weight_columns = [f"w_{t}" for t in tickers]
    inside = registered.reindex(columns=weight_columns).fillna(0.0)
    outside = registered[[c for c in registered.columns if c.startswith("w_") and c not in weight_columns]].fillna(0.0)
    same = registered[(outside.abs().sum(axis=1) == 0) & (inside.sum(axis=1) > 0)]       # all the weight of the portfolio is on the tickers of the set
    if same.empty:
        st.caption("No portfolio saved in the store has all its weight on these tickers.")
        return
    registered_weights = inside.loc[same.index].set_axis(tickers, axis=1)      # the weights editor is indexed by the ticker, not by the column name w_<ticker>
    with st.expander(f"Load portfolio ({len(same)} saved for these tickers)", expanded=False):
        groups = same["tag_group"].fillna("(no group)") if "tag_group" in same else pd.Series("(no group)", index=same.index)
        group = st.selectbox("Group", ["(all)", *sorted(groups.unique())], key=f"{editor_id}_group", help="The tag group of the portfolios (set when they were generated or saved).")
        shown = same if group == "(all)" else same[groups == group]
        picked = st.selectbox(
            "Portfolio", list(shown.index), key=f"{editor_id}_portfolio_{group}",
            format_func=lambda pid: f"{shown.loc[pid, 'name']}  (effective number {shown.loc[pid, 'n_eff']:.1f}, largest {shown.loc[pid, 'max_weight']:.0%})",
        )
        if st.button("Load portfolio", key=f"{editor_id}_load", type="primary", help="Fill the weights table with this portfolio's weights and apply them: the results below are for it."):
            wa.load_weights(editor_id, registered_weights.loc[picked].astype(float))
            st.rerun()


def render(settings: dd.Settings, sidebar_top) -> None:
    st.title("Model comparison with user defined weights")
    runs = cmp._runs(settings.tracking_uri, settings.experiment, settings.store_root)
    runs = runs[~runs["store_run_id"].map(dd.is_virtual_run_id)] if not runs.empty else runs      # a portfolio run is already read for its weights: only runs of the walk-forwards are re-weighted
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
        wanted = st.session_state.pop(OPEN_SET_KEY, None)
        if wanted is not None:      # arrived from another page: the ticker set it asked for, chosen before the selectbox is made
            found = _find_set(sets, store_root, **wanted)
            if found is None:
                st.warning(f"No ticker set with {wanted['n_tickers']} tickers, train {wanted['train_bars']} / forecast {wanted['forecast_bars']} among the saved runs.")
            else:
                st.session_state[TICKER_SET_KEY] = found
        set_key = st.selectbox("Ticker set", list(sets), format_func=lambda k: f"{sets[k][1]} ({len(sets[k][0])} runs)", key=TICKER_SET_KEY, label_visibility="collapsed")
        in_set = runs[runs["store_run_id"].isin(sets[set_key][0])]
        st.header("Runs")
        shown = in_set.drop(columns=[c for c in in_set.columns if c.startswith("total_window_")])    # the metrics of the run's own weights
        filtered = cmp._filters(shown, prefix="wc")

    first_run = cmp._run(in_set["store_run_id"].iloc[0], store_root)
    tickers = list(first_run.config["setting"]["tickers"])
    st.header("Weights", help="The values entered are normalised so that they sum to 1. Equal weights to start with.")
    editor_id = f"wc_{hashlib.sha256(set_key.encode()).hexdigest()[:8]}"
    _load_portfolio_panel(editor_id, tickers, store_root)
    weights = wa._weights_editor(editor_id, pd.Series(1 / len(tickers), index=tickers))
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
