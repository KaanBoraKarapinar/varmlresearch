"""Page 1: compare runs on the days they share. Laid out like the single-run page: runs + analysis period in the left panel, tabs in the main area."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import core
import dashboard_data as dd
import dashboard_figures as figs
from core import run_store as rs
from core import walk_forward as wf
from core.units import format_value
from tools.graph_helpers import plot_cumulative_tick_loss_difference, plot_exceedance_days_focus, plot_var_models_vs_loss

FILTER_COLUMNS = ["portfolio", "train_bars", "model_family", "model_format", "target"]


# ---- cached loading ----------------------------------------------------------------------------------

@st.cache_data(show_spinner="Listing the runs from MLflow ...")
def _runs(tracking_uri: str, experiment: str) -> pd.DataFrame:
    return dd.list_mlflow_runs(tracking_uri, experiment)


@st.cache_resource(show_spinner=False)
def _run(store_run_id: str, store_root: str) -> wf.Run:
    return dd.load_run(store_run_id, store_root)


def _ewma_references(runs: pd.DataFrame, store_root: str) -> dict[str, wf.Run]:
    """The saved 'ewma' runs of the table by label: the plain EWMA variance the models are compared with on the variance tab."""
    ewma_runs = runs[runs["model_family"] == "ewma"].set_index("display_name")
    return {label: _run(row["store_run_id"], store_root) for label, row in ewma_runs.iterrows()}


def _runs_by_label(table: pd.DataFrame, labels: list[str], store_root: str) -> dict[str, wf.Run]:
    by_label = table.set_index("display_name")["store_run_id"]
    return {label: _run(by_label[label], store_root) for label in labels}


@st.cache_data(show_spinner="Reading the runs on their common days ...")
def _common(ids: tuple, labels: tuple, store_root: str, reading: wf.Reading) -> dict:
    """The days every run forecast, and every run's VaR frame and daily frame on them (percent returns)."""
    runs = {label: _run(i, store_root) for label, i in zip(labels, ids)}
    dates = dd.common_days(runs, reading)
    return {
        "dates": dates,
        "var_dfs": {label: run.var_df(reading).loc[dates] for label, run in runs.items()},
        "dailies": {label: run.daily(reading).loc[dates] for label, run in runs.items()},
    }


def _slice(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]


# ---- render ------------------------------------------------------------------------------------------

def render(settings: dd.Settings, sidebar_top) -> None:
    """sidebar_top: the container at the top of the left column (dashboard.py), where this page puts its run selection and period."""
    st.title("Run comparison")
    runs = _runs(settings.tracking_uri, settings.experiment)
    if runs.empty:
        st.warning(f"No runs found in MLflow experiment '{settings.experiment}' ({settings.tracking_uri}).")
        return
    reading, store_root = settings.reading, settings.store_root

    with sidebar_top:
        st.header("Runs")
        filtered = _filters(runs)

    selected = _runs_table(runs, filtered)        # the table with a checkbox per run: the selection comes from here
    if not selected:
        st.info("Tick the box of the runs you want to compare.")
        return

    table = filtered.set_index("display_name").loc[selected]
    ids, labels = tuple(table["store_run_id"]), tuple(selected)
    chosen = _runs_by_label(filtered, selected, store_root)
    portfolios: dict[tuple, list[str]] = {}
    for label, run in chosen.items():
        setting = run.config["setting"]
        key = tuple(sorted((t, round(float(setting["weights"][t]), 9)) for t in setting["tickers"]))
        portfolios.setdefault(key, []).append(label)
    if len(portfolios) > 1:        # the runs would be scored against different realized losses
        st.warning(
            f"The selected runs are of {len(portfolios)} different portfolios (tickers or weights): they cannot be compared on one set of days. "
            "Tick the runs of one portfolio (the 'portfolio' filter in the left panel narrows the table): "
            + "; ".join(f"{len(key)} tickers: {', '.join(labels_[:2])}{' ...' if len(labels_) > 2 else ''}" for key, labels_ in portfolios.items())
        )
        return
    common = _common(ids, labels, store_root, reading)
    dates = common["dates"]
    if len(dates) < 30:
        st.warning(f"The selected runs share only {len(dates)} days: too few for a comparison.")
        return

    with sidebar_top:
        default = next((i for i, l in enumerate(selected) if "garch_wide" in l), 0)
        baseline = st.selectbox("Baseline run", selected, index=default, key="cmp_baseline", help="The run the others are compared with (differences are run minus baseline, so negative is better).")
        first, last = dates[0].date(), dates[-1].date()
        with st.container(border=True):
            st.markdown("**Analysis period**", help="The days the selected runs have all forecast. Everything is re-computed on the period you pick.")
            picked = st.date_input("Analysis start – analysis end", value=(first, last), min_value=first, max_value=last, key="cmp_period")
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pick the end date too.")
        st.stop()
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    var_dfs = {label: _slice(frame, start, end) for label, frame in common["var_dfs"].items()}
    dailies = {label: _slice(frame, start, end) for label, frame in common["dailies"].items()}
    if len(next(iter(var_dfs.values()))) < 30:
        st.warning("The analysis period has fewer than 30 days: pick a longer one.")
        st.stop()
    st.success(f"{len(selected)} runs on {len(next(iter(var_dfs.values())))} common days: {start.date()} → {end.date()}")

    analysis(table, chosen, var_dfs, dailies, baseline, settings, start, end, _ewma_references(runs, store_root))


def analysis(
    table: pd.DataFrame, chosen: dict[str, wf.Run], var_dfs: dict, dailies: dict,
    baseline: str, settings: dd.Settings, start, end, ewma_references: dict[str, wf.Run],
) -> None:
    """
    The tabs for a set of runs on one analysis period (shared with the Experiments page and the user-defined-weights page). chosen: the runs
    by label (any wf.Run), table: one row per run, index = label. ewma_references: the EWMA runs the variance tab can compare with, by label.
    """
    tabs = st.tabs(["Selected runs", "VaR review", "Predicted vs realized variance", "Per ticker variance", "Exceedances / Daily VaR explaining"])
    with tabs[0]:
        _selected_tab(table, chosen)
    with tabs[1]:
        _review_tab(chosen, table, var_dfs, dailies, baseline, settings)
    with tabs[2]:
        _variance_tab(ewma_references, chosen, dailies, settings)
    with tabs[3]:
        _ticker_tab(chosen, list(chosen), start, end)
    with tabs[4]:
        _exceedances_tab(chosen, table, var_dfs, settings)


# ---- left panel -------------------------------------------------------------------------------------

def _filters(runs: pd.DataFrame, prefix: str = "cmp") -> pd.DataFrame:
    """prefix: the start of the widget keys, so two pages that use this do not share their state."""
    filtered = runs
    with st.expander("Filters", expanded=False):
        for column in FILTER_COLUMNS:
            if column in runs and runs[column].nunique(dropna=False) > 1:
                values = runs[column].astype(str).where(runs[column].notna(), "(none)")   # garch / ewma have no target
                options = sorted(values.unique())
                picked = st.multiselect(column, options, default=options, key=f"{prefix}_filter_{column}")
                filtered = filtered[values.loc[filtered.index].isin(picked)]
        if "pretrained_from" in runs and runs["pretrained_from"].notna().any():
            hide = st.checkbox(
                "Hide runs applied from a pretrained model", value=True, key=f"{prefix}_hide_pretrained",
                help="A run with pretrained_from did not train its models: it applies the models of another run (for example trained on 23 tickers) to its own portfolio.",
            )
            if hide:
                filtered = filtered[filtered["pretrained_from"].isna()]
    return filtered


def _runs_table(runs: pd.DataFrame, filtered: pd.DataFrame, prefix: str = "cmp") -> list[str]:
    """
    The runs as a table with a checkbox per row; the ticked rows are the runs to compare. The table (and so the ticks) start
    afresh when the filters change it. The buttons tick all / none by giving the table a new state.
    prefix: the start of the widget / state keys, so two pages that use this do not share their ticks.
    """
    metric_columns = [m for m in dd.HEADLINE_METRICS if m in filtered.columns]
    st.markdown(
        f"**Runs: {len(filtered)} of {len(runs)}**",
        help="Tick the box of a run to compare it. The metrics here are each run's own full period as logged to MLflow; the tabs below re-run everything on the days the selected runs share.",
    )
    start_ticked = st.session_state.get(f"{prefix}_default_ticked", False)
    nonce = st.session_state.get(f"{prefix}_editor_nonce", 0)
    left, right, _ = st.columns([1, 1, 5])
    if left.button("Tick all", key=f"{prefix}_tick_all"):
        st.session_state.update({f"{prefix}_default_ticked": True, f"{prefix}_editor_nonce": nonce + 1})
        st.rerun()
    if right.button("Tick none", key=f"{prefix}_tick_none"):
        st.session_state.update({f"{prefix}_default_ticked": False, f"{prefix}_editor_nonce": nonce + 1})
        st.rerun()

    shown = filtered[["display_name", *dd.parameter_columns(filtered), *metric_columns]].reset_index(drop=True)
    shown.insert(0, "compare", start_ticked)
    signature = hash(tuple(shown["display_name"]))
    edited = st.data_editor(
        shown, hide_index=True, width="stretch", key=f"{prefix}_editor_{nonce}_{signature}",
        disabled=[c for c in shown.columns if c != "compare"],
        column_config={"compare": st.column_config.CheckboxColumn("compare", help="Tick to compare this run.")},
    )
    return edited.loc[edited["compare"], "display_name"].tolist()


# ---- tab: selected runs ---------------------------------------------------------------------------------

def _selected_tab(table: pd.DataFrame, chosen: dict) -> None:
    details = table[[c for c in table.columns if c not in ("store_run_id", "mlflow_run_id")]].astype(str)      # one row per run
    if st.toggle("Only the columns that differ between the runs", value=True, key="cmp_only_diff"):
        details = details.loc[:, details.nunique(axis=0) > 1]
    st.dataframe(details, width="stretch")
    spans = {label: dd.run_span(run) for label, run in chosen.items()}
    st.plotly_chart(figs.timeline_figure(spans), width="stretch")
    st.dataframe(pd.DataFrame(spans).T, width="stretch")


# ---- tab: VaR review ---------------------------------------------------------------------------------

def _conservativity_style(value: str) -> str:
    """'more conservative' / 'more slack' (more unused capital) in red, 'higher tick loss' in orange, the 'less' / 'lower' ones in green."""
    if value.endswith(("more conservative", "more slack")):
        return "color: #c62828"
    if value.endswith(("higher tick loss", "higher exceedance tick loss", "higher overshoot tick loss")):
        return "color: #ef6c00"
    if value.endswith(("less conservative", "less slack", "lower tick loss", "lower exceedance tick loss", "lower overshoot tick loss")):
        return "color: #2e7d32"
    return ""


def _verdict_style(value: str) -> str:
    if value not in ("PASS", "FAIL"):
        return ""
    return "color: #2e7d32; font-weight: 600" if value == "PASS" else "color: #c62828; font-weight: 600"


TEST_ROWS_HELP = (
    "- Kupiec POF: tests whether the observed exceedance rate differs statistically from the expected one.\n"
    "- Christoffersen independence: tests whether a violation today makes one tomorrow more likely (clustering).\n"
    "- Christoffersen conditional coverage: joint test of the correct violation rate and independence."
)
CONSERVATIVITY_ROWS_HELP = (
    "- mean tick loss: the headline score of a VaR, the exceedance part plus the overshoot part; lower is better.\n"
    "- tick loss from exceedance: how much the model underestimated risk.\n"
    "- tick loss from overshoot: how much the model overestimated risk.\n"
    "- % of tick loss from exceedance / overshoot: the share of the mean tick loss that comes from each part (the two add up to 100%).\n"
    "- mean slack: average gap between the VaR and the realized loss on days without a violation; higher means more unused capital.\n"
    "- VaR / realized quantile: average VaR divided by the realized loss quantile; above 1 more conservative, below 1 more aggressive.\n"
    "- The last five rows put each run next to the baseline run: how much higher or lower its mean VaR, mean slack, mean tick loss and the two tick loss parts (exceedance, overshoot) are, in percent of the baseline's."
)


def _versus(values: pd.Series, baseline: str, higher: str, lower: str) -> pd.Series:
    """'x% <higher>' / 'x% <lower>' of each run against the baseline run's value (the baseline itself reads 'baseline')."""
    def text(label: str) -> str:
        if label == baseline:
            return "baseline"
        change = (float(values[label]) / float(values[baseline]) - 1) * 100
        return f"{abs(change):.1f}% {higher if change >= 0 else lower}"
    return pd.Series({label: text(label) for label in values.index})


def _review_tables(backtest: pd.DataFrame, level: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The test metrics and the conservativity metrics: one COLUMN per run, one row per measure (every cell is text)."""
    passed = lambda key: backtest.loc[key].map(lambda v: "PASS" if bool(v) else "FAIL")
    test = pd.DataFrame({
        "forecast days": backtest.loc["n_days"].map(lambda v: f"{int(v)}"),
        "exceedances": backtest.loc["n_exceedances"].map(lambda v: f"{int(v)}"),
        "exceedance rate": backtest.loc["exceedance_rate"].map(lambda v: f"{float(v):.2%}"),
        "Kupiec POF": passed("kupiec_pof_pass"),
        "Christoffersen independence": passed("christoffersen_ind_pass"),
        "Christoffersen conditional coverage": passed("christoffersen_cc_pass"),
    }).T
    conservativity = pd.DataFrame({
        "mean tick loss": level.loc["mean_tick_loss"].map(lambda v: f"{float(v):.6f}"),
        "tick loss from exceedance": level.loc["tick_loss_exceedance_part"].map(lambda v: f"{float(v):.6f}"),
        "tick loss from overshoot": level.loc["tick_loss_overshoot_part"].map(lambda v: f"{float(v):.6f}"),
        "% of tick loss from exceedance": (level.loc["tick_loss_exceedance_part"].astype(float) / level.loc["mean_tick_loss"].astype(float)).map(lambda v: f"{v:.1%}"),
        "% of tick loss from overshoot": (level.loc["tick_loss_overshoot_part"].astype(float) / level.loc["mean_tick_loss"].astype(float)).map(lambda v: f"{v:.1%}"),
        "mean slack": level.loc["mean_slack_non_exceedance"].map(lambda v: format_value(float(v), 3)),
        "VaR / realized quantile": level.loc["var_to_realized_quantile_ratio"].map(lambda v: f"{float(v):.3f}"),
        "mean VaR": level.loc["mean_var"].map(lambda v: format_value(float(v), 3)),
        "VaR level vs baseline": _versus(level.loc["mean_var"], baseline, "more conservative", "less conservative"),
        "unused capital vs baseline": _versus(level.loc["mean_slack_non_exceedance"], baseline, "more slack", "less slack"),
        "tick loss vs baseline": _versus(level.loc["mean_tick_loss"], baseline, "higher tick loss", "lower tick loss"),
        "tick loss from exceedance vs baseline": _versus(level.loc["tick_loss_exceedance_part"], baseline, "higher exceedance tick loss", "lower exceedance tick loss"),
        "tick loss from overshoot vs baseline": _versus(level.loc["tick_loss_overshoot_part"], baseline, "higher overshoot tick loss", "lower overshoot tick loss"),
    }).T
    return test, conservativity


def _review_tab(chosen: dict, table: pd.DataFrame, var_dfs: dict, dailies: dict, baseline: str, settings: dd.Settings) -> None:
    reading = settings.reading
    dates = next(iter(var_dfs.values())).index
    backtest, level = dd.backtest_tables(chosen, reading, dates)
    test_table, conservativity_table = _review_tables(backtest, level, baseline)

    st.markdown(
        f":blue[**VAR RESULTS UNDER:**] Var Confidence: {reading.var_confidence:g} · Min Periods: {reading.min_periods} · "
        f"Residual Window: {'expanding' if reading.window_size is None else reading.window_size} · Test Confidence: {reading.test_confidence:g}",
        help="The reading chosen in the left panel, applied to every run.",
    )
    fit = lambda frame: min((len(frame) + 1) * 35 + 3, 700)     # tall enough to show every row at once; many runs scroll inside the table
    # one ROW per run, one column per measure: any number of runs is read going down, not across
    test_table, conservativity_table = test_table.T, conservativity_table.T
    st.markdown("**Test metrics**", help=TEST_ROWS_HELP)
    st.dataframe(
        test_table.style.map(_verdict_style, subset=pd.IndexSlice[:, ["Kupiec POF", "Christoffersen independence", "Christoffersen conditional coverage"]]),
        width="stretch", height=fit(test_table),
    )
    versus_columns = [c for c in conservativity_table.columns if c.endswith("vs baseline")]          # the group of its own: each run against the baseline run
    levels = conservativity_table.drop(columns=versus_columns)
    st.markdown("**Conservativity metrics**", help=CONSERVATIVITY_ROWS_HELP)
    st.dataframe(
        levels.style.set_properties(subset=pd.IndexSlice[:, [c for c in levels.columns if c.startswith("% of tick loss")]], **{"font-weight": "bold"}),
        width="stretch", height=fit(levels),
    )
    st.markdown(
        f"**Conservativity against the baseline run**",
        help="Each run's mean VaR, unused capital (slack) and tick loss (and its two parts) in percent of the baseline run's: red = more conservative / more slack, "
             "orange = higher tick loss, green = less conservative / less slack / lower tick loss.",
    )
    st.dataframe(
        conservativity_table[versus_columns].style
        .set_properties(**{"font-weight": "bold"})
        .map(_conservativity_style),                    # more / higher red, less / lower green
        width="stretch", height=fit(conservativity_table),
    )

    st.markdown("**VaR forecasts vs realized loss**")
    banks = {label: run.combined_bank(percent=True) for label, run in chosen.items()}
    st.plotly_chart(plot_var_models_vs_loss({label: (var_dfs[label], banks[label]) for label in chosen}, title="VaR forecasts vs realized loss"), width="stretch")

    st.markdown(
        "**Cumulative tick loss**",
        help="The running sum of each run's daily tick loss. With the difference switched on, every run is shown minus the baseline run: below 0 means better than the baseline in that stretch.",
    )
    as_difference = st.toggle("Difference to the baseline run", value=False, key="cmp_cum_diff")
    if as_difference:
        st.plotly_chart(
            plot_cumulative_tick_loss_difference(var_dfs, baseline, reading.var_confidence, title=f"Cumulative tick loss difference vs {baseline} (below 0 = better)"),
            width="stretch",
        )
    else:
        st.plotly_chart(figs.cumulative_tick_loss_figure({label: core.daily_tick_loss(var_dfs[label], reading.var_confidence).cumsum() for label in chosen}), width="stretch")

    st.markdown("**Against the baseline**", help="Run minus baseline, so negative is better. dm_p_value is the Diebold-Mariano test of the daily tick-loss difference (HAC).")
    st.dataframe(core.compare_to_baseline(var_dfs, baseline, reading.var_confidence), width="stretch")
    if len(var_dfs) > 1:
        _best_run_section(var_dfs, baseline, reading)


# ---- which run is best (the logic is core.select_best_run) ------------------------------------------------

@st.cache_data(show_spinner="Comparing the runs (Model Confidence Set) ...")
def _selection(var_dfs: dict, var_confidence: float, test_confidence: float, baseline: str, alpha: float) -> core.RunSelection:
    return core.select_best_run(var_dfs, var_confidence, test_confidence, baseline=baseline, alpha=alpha)


def _verdict_color(value: str) -> str:
    return {
        "best": "color: #2e7d32; font-weight: 700", "best set": "color: #2e7d32",
        "worse": "color: #ef6c00", "not valid": "color: #c62828; font-weight: 600",
    }.get(value, "")


def _best_run_section(var_dfs: dict, baseline: str, reading: wf.Reading) -> None:
    st.markdown(
        "**Run ranking**",
        help="The tick loss alone cannot decide: a gap of a few percent is inside the noise of the period. A run is valid when it passes the Kupiec and both Christoffersen tests. "
             "The Model Confidence Set keeps the runs that cannot be told apart from the best one at the chosen level (it corrects for comparing many runs and for dependent days). "
             "The best set is the valid runs inside the set, the best run is the one of them with the lowest mean tick loss. Blocks are consecutive stretches of 252 common days: "
             "how many blocks a run wins and its mean rank (1 = best) show whether it is consistently good.",
    )
    alpha = st.select_slider("Level of the Model Confidence Set", [0.05, 0.10, 0.25], value=0.10, key="cmp_mcs_alpha",
                             help="Higher = stricter: more runs are told apart from the best (fewer stay in the set).")
    selection = _selection(var_dfs, reading.var_confidence, reading.test_confidence, baseline, alpha)
    if selection.best is None:
        st.warning("None of the selected runs passes all three tests: there is no valid run to call the best.")
    else:
        st.success(f"Top run: {selection.best}")

    board = selection.leaderboard
    shown = pd.DataFrame({
        "mean tick loss": board["mean_tick_loss"].map(lambda v: f"{v:.6f}"),
        "tests": board["valid"].map(lambda v: "PASS" if v else "FAIL"),
        "exceedance rate": board["exceedance_rate"].map(lambda v: f"{v:.2%}"),
        "in the set": board["in_mcs"].map(lambda v: "yes" if v else "no"),
        "set p-value": board["mcs_p_value"].map(lambda v: f"{v:.3f}"),
        "blocks won": board.apply(lambda r: f"{int(r['blocks_won'])} of {int(r['n_blocks'])}", axis=1),
        "mean block rank": board["mean_block_rank"].map(lambda v: f"{v:.2f}"),
        f"p vs {baseline.split(' | ')[0]}": board["dm_p_vs_baseline"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        "verdict": board["verdict"],
    })
    st.dataframe(
        shown.style.map(_verdict_color, subset=["verdict"]).map(_verdict_style, subset=["tests"]),
        width="stretch", height=(len(shown) + 1) * 35 + 3,
    )
    st.plotly_chart(figs.pairwise_matrix_figure(selection.pairwise), width="stretch")


# ---- tab: predicted vs realized variance ---------------------------------------------------------------

def _variance_tab(ewma_references: dict[str, wf.Run], chosen: dict, dailies: dict, settings: dd.Settings) -> None:
    reference = None
    if ewma_references:
        label = st.selectbox(
            "EWMA reference run", list(ewma_references), key="cmp_ewma_ref",
            help="The plain EWMA variance (decay 0.94) of the portfolio returns, the benchmark the models are compared with.",
        )
        reference = ewma_references[label]
    dates = next(iter(dailies.values())).index
    losses = next(iter(dailies.values()))["loss"]                      # the realized loss is the same in every run
    # a historical run has no variance forecast: it is left out of everything on this tab
    chosen = {label: run for label, run in chosen.items() if dd.has_variance_forecast(run)}
    dailies = {label: daily for label, daily in dailies.items() if label in chosen}
    if not chosen:
        st.info("The selected runs are 'historical' runs: their VaR is a quantile of past returns, there is no variance forecast to compare.")
        return
    variance = pd.DataFrame({label: daily["predicted_variance"] for label, daily in dailies.items()})
    baselines = pd.DataFrame({"squared return r²": losses ** 2, "realized variance (21-day mean of r²)": (losses ** 2).rolling(21, min_periods=5).mean()})
    if reference is not None:
        baselines.insert(0, "EWMA (reference run)", reference.daily(settings.reading)["predicted_variance"].reindex(dates))

    st.markdown("**Baselines**", help="Lines drawn next to the runs' variances in the chart below. They only change the chart, not the metrics.")
    chosen_baselines = [c for c in baselines.columns if st.checkbox(c, value=c.startswith("EWMA"), key=f"cmp_baseline_{c}")]

    r2 = losses ** 2
    ewma_label = "EWMA (reference run)"
    metrics_frame = variance.copy()
    if reference is not None:
        metrics_frame[ewma_label] = baselines[ewma_label]
        metrics = dd.variance_metrics(metrics_frame, r2, ewma_label)
        st.markdown("**Variance forecast metrics**", help="QLIKE is the loss of the variance forecast against the realized squared return: lower is better, it punishes a too low forecast most and it does not depend on the unit or the scale, so it is comparable between any models. qlike_diff_vs_ewma below 0 means better than the EWMA.")
    else:
        metrics = pd.DataFrame({label: {"qlike": dd.qlike(variance[label], r2), "mean_variance": float(variance[label].mean())} for label in variance}).T
        st.info("There is no 'ewma' run in this experiment: the metrics against the EWMA are left out.")
    st.dataframe(metrics, width="stretch")

    st.markdown(
        "**Window losses**",
        help="Over all windows of each run (not the analysis period), every window counted with the number of days it covers. The QLIKE columns are comparable between any runs. "
             "The loss columns are squared errors in each model's own target units: only comparable between runs with the same target and scale.",
    )
    totals = pd.DataFrame({label: run.total_window_losses for label, run in chosen.items()}).T
    st.dataframe(totals, width="stretch")

    log_y = st.toggle("Log y axis", value=True, key="cmp_var_log")
    st.plotly_chart(figs.variance_runs_figure(variance, baselines[chosen_baselines], log_y, "Predicted portfolio variance"), width="stretch")

    ratio_reference = baselines[ewma_label] if reference is not None else None
    if ratio_reference is not None:
        smooth = st.slider("Smooth the ratio (rolling days)", 1, 63, 21, key="cmp_ratio_smooth")
        ratio = variance.div(ratio_reference, axis=0).rolling(smooth, min_periods=1).mean()
        st.plotly_chart(figs.lines_figure(ratio, "Predicted variance / EWMA variance (1 = same as the EWMA)", "ratio", True), width="stretch")


# ---- tab: per ticker variance -------------------------------------------------------------------------

def _ticker_tab(chosen: dict, selected: list[str], start, end) -> None:
    with_ticker = {label: run for label, run in chosen.items() if run.has_ewma_bank}
    if not with_ticker:
        st.info("The selected runs are all wide: they predict the portfolio variance directly, there are no per-ticker variances.")
        return
    tickers = next(iter(with_ticker.values())).config["setting"]["tickers"]
    st.markdown(
        "**Variance per ticker**",
        help="Each long run's model variance for the ticker, against the EWMA's own variance of that ticker (the dashed line) on the same days. Wide runs are left out.",
    )
    ticker = st.selectbox("Ticker", tickers, key="cmp_ticker")
    frames = {label: dd.ticker_variance_frames(run) for label, run in with_ticker.items()}
    model_variance = pd.DataFrame({label: _slice(frames[label][ticker], start, end)["model variance"] for label in frames})
    first = next(iter(frames))
    own = _slice(frames[first][ticker], start, end)
    baselines = pd.DataFrame({"EWMA variance": own["EWMA variance"], "return^2": own["return^2"]})
    show_r2 = st.toggle("Show the squared return", value=False, key="cmp_ticker_r2")
    log_y = st.toggle("Log y axis", value=True, key="cmp_ticker_log")
    st.plotly_chart(
        figs.variance_runs_figure(model_variance, baselines[["EWMA variance", "return^2"] if show_r2 else ["EWMA variance"]], log_y, f"Variance of {ticker}"),
        width="stretch",
    )
    rows = {}
    for label in frames:
        f = _slice(frames[label][ticker], start, end)
        rows[label] = {
            "mean model variance": f["model variance"].mean(), "mean ratio model/EWMA": (f["model variance"] / f["EWMA variance"]).mean(),
            "median ratio model/EWMA": (f["model variance"] / f["EWMA variance"]).median(), "corr(model, EWMA)": f["model variance"].corr(f["EWMA variance"]),
            "qlike model": dd.qlike(f["model variance"], f["return^2"]),
        }
    st.dataframe(pd.DataFrame(rows).T, width="stretch")


# ---- tab: exceedances -----------------------------------------------------------------------------------

def _exceedances_tab(chosen: dict, table: pd.DataFrame, var_dfs: dict, settings: dd.Settings) -> None:
    reading = settings.reading
    days = core.exceedance_days_by_model(var_dfs)
    st.markdown("**Exceedances per calendar year**")
    st.dataframe(dd.exceedances_per_year(var_dfs), width="stretch")
    if days.empty:
        st.info("None of the runs has an exceedance in this period.")
        return
    n_runs = len(var_dfs)
    by_count = days["n_models"].value_counts().sort_index()
    label = lambda n: "only 1 run" if n == 1 else ("all runs" if n == n_runs else f"{n} runs")
    counts = pd.DataFrame({"days": by_count.values}, index=pd.Index([label(n) for n in by_count.index], name="runs exceeding the VaR that day"))
    left, right = st.columns([1, 2])
    left.markdown(
        "**Do the runs miss on the same days?**",
        help="Number of days by how many runs exceeded the VaR that day (a day's realized loss above the VaR is an exceedance, a 'miss'). Only days at least one run missed are counted.",
    )
    left.dataframe(counts, width="stretch")

    per_run, overlap = core.exceedance_overlap(var_dfs)
    right.markdown(
        "**Per run**",
        help="Exceedance days: the days the run's loss was above its VaR. Only this run: no other selected run exceeded that day. "
             "Shared with another run: at least one other did. Total difference between realized loss and VaR on the days only this run exceeded: the sum, over those days alone, "
             "of the amount by which the realized loss exceeded this run's VaR (on those days the other runs' VaR was high enough).",
    )
    right.dataframe(
        pd.DataFrame({
            "exceedance days": per_run["exceedance_days"].astype(int),
            "only this run": per_run["only_this_run"].astype(int),
            "shared with another run": per_run["shared_with_another_run"].astype(int),
            "loss − VaR difference (only-this-run days)": per_run["excess_on_only_days"].map(format_value),
        }),
        width="stretch", height=(len(per_run) + 1) * 35 + 3,
    )
    st.markdown(
        "**Overlap between runs**",
        help="The row run's exceedance days on which the column run exceeded the VaR too, in percent of the row run's exceedance days. "
             "100% means the column run always exceeds when the row run does.",
    )
    st.dataframe(
        overlap.style.format("{:.0%}", na_rep="–").background_gradient(cmap="Blues", vmin=0, vmax=1),
        width="stretch", height=(len(overlap) + 1) * 35 + 3,
    )

    shown = days.sort_values(["n_models", "loss"], ascending=False)
    st.markdown(
        f"**{len(days)} days at least one run missed**",
        help="Click a row to explain that day (nothing clicked yet: the day most runs missed). The clicked day stays selected when you change the VaR reading, even if it is no longer an exceedance.",
    )
    event = st.dataframe(shown, width="stretch", on_select="rerun", selection_mode="single-row", key="cmp_exceedance_table")
    rows = list(event.selection.rows)
    if rows and rows != st.session_state.get("cmp_selection_seen"):
        st.session_state["cmp_explain_day"] = shown.index[rows[0]]
        st.session_state["cmp_pick_day"] = shown.index[rows[0]].date()
    st.session_state["cmp_selection_seen"] = rows

    all_days = next(iter(var_dfs.values())).index
    low, high = all_days[0].date(), all_days[-1].date()
    current = st.session_state.get("cmp_explain_day", shown.index[0]).date()
    if not low <= current <= high:       # the kept day is outside the runs / period now shown (other runs, a narrower period): start over
        st.session_state["cmp_explain_day"] = shown.index[0]
        current = shown.index[0].date()
    if "cmp_pick_day" not in st.session_state or not low <= st.session_state["cmp_pick_day"] <= high:
        st.session_state["cmp_pick_day"] = current
    picked = st.date_input("Or pick any common day (it need not be an exceedance)", min_value=low, max_value=high, key="cmp_pick_day")
    if picked != current:
        st.session_state["cmp_explain_day"] = pd.Timestamp(picked)
    day = pd.Timestamp(picked)
    if day not in all_days:
        st.warning(f"{picked} is not a common day of these runs (weekend, holiday or outside the period).")
        return

    st.subheader(f"The runs on {day.date()}")
    bars = st.slider("Days on each side", 3, 40, 10, key="cmp_focus_bars")
    st.plotly_chart(plot_exceedance_days_focus(var_dfs, [day], window_bars=bars), width="stretch")
    details = core.exceedance_day_details(var_dfs, [day], labels=False).droplevel("Date")
    st.dataframe(
        details.style.format({"loss": format_value, "var_forecast": format_value, "difference": lambda v: format_value(v, signed=True), "loss_over_var": "{:.2f}"}),
        width="stretch",
    )
    which = st.selectbox("Explain the VaR of this day for", list(chosen), key="cmp_explain_run")
    explained = rs.explain_day(day, chosen[which], reading)
    explained["value"] = explained["value"].map(lambda v: f"{v:.6g}" if isinstance(v, float) else str(v))
    st.dataframe(explained, width="stretch")
