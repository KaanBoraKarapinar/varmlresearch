"""
Page 4: weight adjustment. The stored per-ticker forecasts of a run, read for OTHER portfolio weights (nothing is trained or saved), and the
run's daily minimum-risk portfolio.

The VaR of the new weights is exact for GARCH, EWMA and xgboost models whose per-ticker forecasts do not depend on the weights
(wf.ReweightedVirtualRun); GARCH / EWMA / historical for the same weights are computed now (dd.baseline_runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import dashboard_data as dd
import dashboard_figures as figs
import page_run_comparison as cmp
from core import walk_forward as wf
from core.units import format_value


@st.cache_data(show_spinner="Reading the stored forecasts for the user defined weights ...")
def _reweighted(store_run_id: str, store_root: str, weights: tuple, reading: wf.Reading) -> pd.DataFrame:
    return wf.ReweightedVirtualRun(cmp._run(store_run_id, store_root), dict(weights)).var_df(reading)


@st.cache_data(show_spinner="Computing GARCH, EWMA and historical for the user defined weights (about 10 s) ...")
def _baselines(store_run_id: str, store_root: str, weights: tuple, reading: wf.Reading) -> dict:
    runs = dd.baseline_runs(cmp._run(store_run_id, store_root), dict(weights), store_root)
    return {name: run.var_df(reading) for name, run in runs.items()}


@st.cache_data(show_spinner="Solving the minimum-risk weights of every day ...")
def _minimum_risk_weights(store_run_id: str, store_root: str) -> pd.DataFrame:
    run = cmp._run(store_run_id, store_root)
    return core.daily_minimum_risk_weights(run.ewma_bank, list(run.config["setting"]["tickers"]))


@st.cache_data(show_spinner=False)
def _original_var_df(store_run_id: str, store_root: str, reading: wf.Reading) -> pd.DataFrame:
    return cmp._run(store_run_id, store_root).var_df(reading)


def load_weights(run_id: str, weights: pd.Series) -> None:
    """Put these weights in the table of _weights_editor(run_id, ...) and apply them (what the page shows from its next run on): the editor's own state, set from outside."""
    version = st.session_state.get(f"wa_version_{run_id}", 0) + 1
    st.session_state[f"wa_start_{run_id}_{version}"] = weights
    st.session_state[f"wa_version_{run_id}"] = version
    st.session_state[f"wa_applied_{run_id}"] = weights / weights.sum()


def _weights_editor(run_id: str, default: pd.Series) -> pd.Series:
    """
    The weights table (one row per ticker, editable) and its buttons. Editing the table changes nothing that is computed: the weights in use are
    the ones applied with 'Apply weights' (returned, normalised; at first the default). 'Reset weights' and 'Equal weight' only fill the
    table (zeros / equal weights). The state is kept per run_id.
    """
    applied_key, version_key = f"wa_applied_{run_id}", f"wa_version_{run_id}"
    if applied_key not in st.session_state:
        st.session_state[applied_key] = default / default.sum()
    version = st.session_state.get(version_key, 0)
    buttons = st.container()        # above the table, filled after it: a click must see the table as it is now
    start = st.session_state.get(f"wa_start_{run_id}_{version}", st.session_state[applied_key] if version == 0 else default)
    table = pd.DataFrame({"weight": start}).rename_axis("ticker")
    edited = st.data_editor(table, key=f"wa_editor_{run_id}_{version}", width="stretch", height=min(38 * (len(table) + 1), 420))
    raw = edited["weight"].astype(float)
    valid = not (raw.isna().any() or (raw < 0).any() or raw.sum() <= 0)
    with buttons:
        apply_column, reset_column, equal_column, _ = st.columns([1, 1, 1, 3])
        apply = apply_column.button("Apply weights", type="primary", key=f"wa_apply_{run_id}", help="Compute with the weights in the table.")
        reset = reset_column.button("Reset weights", key=f"wa_reset_{run_id}", help="Set every weight in the table to zero, to type the weights from scratch (they sum to zero until you fill them: nothing can be applied before that).")
        equal = equal_column.button("Equal weight", key=f"wa_equal_{run_id}", help="Fill the table with equal weights (then apply them).")
    if reset:
        st.session_state[f"wa_start_{run_id}_{version + 1}"] = pd.Series(0.0, index=default.index)
        st.session_state[version_key] = version + 1
        st.rerun()
    if equal:
        st.session_state[f"wa_start_{run_id}_{version + 1}"] = pd.Series(1 / len(default), index=default.index)
        st.session_state[version_key] = version + 1
        st.rerun()
    if apply:
        if valid:
            st.session_state[applied_key] = raw / raw.sum()
        else:
            st.error("The weights must be numbers, none negative, with a positive sum: nothing applied.")
    weights = st.session_state[applied_key]
    if valid and not np.allclose((raw / raw.sum()).to_numpy(), weights.loc[raw.index].to_numpy()):
        st.info("The table has changes that are not applied yet: press 'Apply weights'. The results below use the weights applied last.")
    st.markdown(
        f"Largest {weights.idxmax()} {weights.max():.1%} · effective number of tickers {1 / (weights ** 2).sum():.1f}",
        help="1 / Σ wᵢ² (the inverse of the Herfindahl-Hirschman index of the weights): how many equally weighted tickers the portfolio "
             "behaves like. n equal weights give n, all the weight in one ticker gives 1.",
    )
    return weights


def render(settings: dd.Settings, sidebar_top) -> None:
    st.title("Portfolio ticker weight adjustment")
    runs = cmp._runs(settings.tracking_uri, settings.experiment, settings.store_root)
    runs = runs[~runs["store_run_id"].map(dd.is_virtual_run_id)] if not runs.empty else runs      # a portfolio run is already read for its weights: only runs of the walk-forwards are re-weighted
    if runs.empty:
        st.warning(f"No runs found in MLflow experiment '{settings.experiment}' ({settings.tracking_uri}).")
        return
    runs = runs.assign(label=dd._unique_labels(runs))
    candidates = [
        row["label"] for _, row in runs.iterrows()
        if wf.reweight_blocker(cmp._run(row["store_run_id"], settings.store_root)) is None
    ]
    if not candidates:
        st.warning("No run can be re-weighted yet (it needs the per-ticker forecasts and the first-window training bank).")
        return
    with sidebar_top:
        st.header("Run")
        label = st.selectbox("Run", candidates, key="wa_run", label_visibility="collapsed")
        st.caption(f"{len(candidates)} of {len(runs)} runs can be re-weighted: the others have no per-ticker forecasts, or forecasts that depend on the weights.")
    row = runs.set_index("label").loc[label]
    store_run_id, store_root, reading = row["store_run_id"], settings.store_root, settings.reading
    run = cmp._run(store_run_id, store_root)
    own = dd.run_weights(run)
    own = own / own.sum()

    st.markdown(f":blue[**RUN:**] {label} · {len(own)} tickers · {run.params['model_family']} · unit {run.unit}")
    st.header("Weights", help="The values entered are normalised so that they sum to 1.")
    weights = _weights_editor(store_run_id, own)
    same = bool(np.allclose(weights.to_numpy(), own.loc[weights.index].to_numpy()))
    key = tuple(sorted(weights.round(12).items()))

    var_tab, minimum_risk_tab = st.tabs(["VaR with user defined weights", "Minimum-risk portfolio"])
    with var_tab:
        _var_tab(run, store_run_id, store_root, reading, weights, same, key)
    with minimum_risk_tab:
        _minimum_risk_tab(run, store_run_id, store_root, weights)


def _var_tab(run: wf.Run, store_run_id: str, store_root: str, reading: wf.Reading, weights: pd.Series, same: bool, key: tuple) -> None:
    var_dfs = {f"{run.name} · user defined weights": _reweighted(store_run_id, store_root, key, reading)}
    if not same:
        var_dfs[f"{run.name} · run's own weights"] = _original_var_df(store_run_id, store_root, reading)
    else:
        st.caption("These are the run's own weights.")
    for name, frame in _baselines(store_run_id, store_root, key, reading).items():
        var_dfs[f"{name} · user defined weights"] = frame
    common = None
    for frame in var_dfs.values():
        common = frame.index if common is None else common.intersection(frame.index)
    first, last = common[0].date(), common[-1].date()
    with st.container(border=True):
        st.markdown("**Analysis period**")
        picked = st.date_input("Analysis start – analysis end", value=(first, last), min_value=first, max_value=last, key="wa_period")
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pick the end date too.")
        return
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    var_dfs = {k: v.loc[common].loc[start:end] for k, v in var_dfs.items()}
    if len(next(iter(var_dfs.values()))) < 30:
        st.warning("The analysis period has fewer than 30 forecast days: pick a longer one.")
        return
    table = dd.var_table(var_dfs, reading)
    st.dataframe(table.style.format({"mean tick loss": "{:.5f}", "mean VaR": format_value, "expected": "{:.1f}"}), width="stretch")
    st.plotly_chart(figs.var_curves_figure(var_dfs, "VaR with user defined weights, over the realized loss"), width="stretch")
    ticks = dd.tick_losses(var_dfs, reading.var_confidence)
    st.plotly_chart(figs.cumulative_tick_loss_figure({c: ticks[c].cumsum() for c in ticks.columns}), width="stretch")


def _minimum_risk_tab(run: wf.Run, store_run_id: str, store_root: str, weights: pd.Series) -> None:
    st.caption("Every day the weights that minimise the forecast variance w'Σw of that day (long only, fully invested), from the run's own forecasts "
               "made before the day; the day's weights are applied to the day's returns. It is a portfolio to look at, not a VaR: its weights change every day.")
    min_weights = _minimum_risk_weights(store_run_id, store_root)
    result = dd.minimum_risk_summary(run, min_weights, weights)
    st.subheader("Realized risk of the daily returns")
    st.dataframe(result["realized"].style.format("{:.2f}"), width="stretch")
    cumulative = (1 + result["returns"]).cumprod()
    fig = go.Figure()
    for i, column in enumerate(cumulative.columns):
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[column], name=column, mode="lines", line=dict(width=1.8, color=figs.COLORS[i])))
    fig.update_yaxes(type="log")
    st.plotly_chart(figs._layout(fig, "Growth of 1, the weights of the day, rebalanced daily (log axis)", "growth of 1"), width="stretch")

    st.subheader("How the weights look")
    turnover = float(min_weights.diff().abs().sum(axis=1).mean() * 100)
    st.caption(f"Mean daily turnover (sum of |change of the weights|): {turnover:.1f}% of the portfolio. Sorted by the mean weight.")
    st.dataframe(result["weights"].style.format("{:.1f}"), width="stretch")
    average = st.radio("Average of the weights over", [1, 5, 21, 63], index=2, horizontal=True, format_func=lambda n: "1 day (as solved)" if n == 1 else f"{n} days",
                       help="The daily weights change a lot from day to day, so the chart is hard to read. Here each day shows the average of the last n days' "
                            "weights (they still sum to 1). Only the chart is smoothed; the tables use the real daily weights.")
    smoothed = min_weights.rolling(average, min_periods=1).mean()
    area = go.Figure()
    ordered = list(min_weights.columns[np.argsort(-min_weights.mean().to_numpy())])
    shown, folded = ordered[:len(figs.CATEGORICAL_COLORS)], ordered[len(figs.CATEGORICAL_COLORS):]
    series = {ticker: smoothed[ticker] for ticker in shown}
    colors = dict(zip(shown, figs.CATEGORICAL_COLORS))
    if folded:       # the tickers with the smallest mean weights are one grey slice: no colour is made up for a ninth series
        other = f"Other ({len(folded)} tickers)"
        series[other], colors[other] = smoothed[folded].sum(axis=1), figs.OTHER_COLOR
    for name, values in series.items():
        area.add_trace(go.Scatter(
            x=values.index, y=values * 100, name=name, mode="lines", stackgroup="w",
            fillcolor=colors[name], line=dict(width=0, color=colors[name]),      # opaque fill, no outline (an outline smears the day-to-day jumps)
        ))
    area.update_yaxes(ticksuffix="%", range=[0, 100])
    st.plotly_chart(figs._layout(area, "Minimum-risk weights over time" + ("" if average == 1 else f" ({average}-day average)"), "weight"), width="stretch")
    picked_day = st.date_input("Weights of a day", value=min_weights.index[-1].date(), min_value=min_weights.index[0].date(), max_value=min_weights.index[-1].date(), key="wa_day")
    trading_days = min_weights.index[min_weights.index <= pd.Timestamp(picked_day)]
    day = trading_days[-1]
    if day != pd.Timestamp(picked_day):
        st.caption(f"{picked_day} is not a forecast day of the run: showing {day.date()}, the day before.")
    st.dataframe(
        pd.DataFrame({"minimum risk (%)": min_weights.loc[day] * 100, "chosen weights (%)": weights * 100}).style.format("{:.2f}"),
        width="stretch",
    )
    with st.expander("All days, the weights in %"):
        st.dataframe((min_weights * 100).round(2), width="stretch")
