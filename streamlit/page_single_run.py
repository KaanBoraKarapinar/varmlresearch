"""Page 2: one run in detail, laid out as in the panel sketches: run box, analysis period, VaR review, VaR vs loss, predicted vs realized variance, readings comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_data as dd
import dashboard_figures as figs
import run_store as rs
import walk_forward as wf
from core.units import format_value
from graph_helpers import plot_exceedance_days_focus, plot_var_models_vs_loss, plot_var_vs_loss

READING_COLORS = [None, "red", "blue", "green", "violet", "orange"]   # None = the sidebar reading (default text color)


@st.cache_data(show_spinner="Listing the runs from MLflow ...")
def _runs(tracking_uri: str, experiment: str) -> pd.DataFrame:
    return dd.list_mlflow_runs(tracking_uri, experiment)


@st.cache_resource(show_spinner=False)
def _run(store_run_id: str, store_root: str) -> wf.Run:
    return dd.load_run(store_run_id, store_root)


@st.cache_data(show_spinner="Reading the run ...")
def _daily(store_run_id: str, store_root: str, reading: wf.Reading) -> pd.DataFrame:
    return _run(store_run_id, store_root).daily(reading)


@st.cache_data(show_spinner="Reading the run ...")
def _var_df(store_run_id: str, store_root: str, reading: wf.Reading) -> pd.DataFrame:
    return _run(store_run_id, store_root).var_df(reading)


@st.cache_data(show_spinner=False)
def _ticker_frames(store_run_id: str, store_root: str) -> dict:
    return dd.ticker_variance_frames(_run(store_run_id, store_root))


def render(settings: dd.Settings, sidebar_top) -> None:
    """sidebar_top: the container at the top of the left column (dashboard.py), where this page puts its run controls."""
    st.title("Single run")
    runs = _runs(settings.tracking_uri, settings.experiment)
    if runs.empty:
        st.warning(f"No runs found in MLflow experiment '{settings.experiment}' ({settings.tracking_uri}).")
        return

    # the left column, on top: which run, what it is, and the period everything on the page is analysed on
    with sidebar_top:
        st.header("Run")
        label = st.selectbox("Run", runs["display_name"].tolist(), key="single_run", label_visibility="collapsed")
        row = runs.set_index("display_name").loc[label]
        store_run_id, store_root = row["store_run_id"], settings.store_root
        run = _run(store_run_id, store_root)
        daily = _daily(store_run_id, store_root, settings.reading)
        _run_box(run)
        start, end = _analysis_period(daily)
    main = _slice(_var_df(store_run_id, store_root, settings.reading), start, end)

    tabs = st.tabs([
        "VaR review", "VaR settings comparison", "Predicted vs realized variance",
        "Per ticker variance", "Windows", "Exceedances / Daily VaR explaining", "Model",
    ])
    with tabs[0]:
        _review(main, settings.reading, color=None)
        st.plotly_chart(
            plot_var_vs_loss(main, run.combined_bank(percent=True), title=f"{label}: VaR vs realized loss"),
            width="stretch",
        )
        _cumulative_tick_loss(label, daily, start, end)
    with tabs[1]:
        _readings_comparison(label, run, store_run_id, settings, start, end)
    with tabs[2]:
        if dd.has_variance_forecast(run):
            _variance_panel(runs, row, run, daily, settings, start, end)
        else:
            st.info("A 'historical' run has no variance forecast: its VaR is the quantile of the last returns of the VaR reading's residual window.")
    with tabs[3]:
        if run.has_ewma_bank:
            _per_ticker(store_run_id, settings, run.params["model_family"] == "ewma")
        elif not dd.has_variance_forecast(run):
            st.info("A 'historical' run has no variance forecast, so no per-ticker variances either.")
        else:
            st.info("A wide run predicts the portfolio variance directly: there are no per-ticker variances.")
    with tabs[4]:
        _windows(run, settings.reading)
    with tabs[5]:
        _exceedances(label, run, daily, runs, row, settings)
    with tabs[6]:
        _model(run)


def _cumulative_tick_loss(label: str, daily: pd.DataFrame, start, end) -> None:
    """The running sum of the run's daily tick loss over the analysis period."""
    st.markdown(
        "**Cumulative tick loss**",
        help="The running sum of the daily tick loss (the headline score of a VaR): where in time the run costs the most. The steeper the line, the worse the stretch.",
    )
    st.plotly_chart(figs.cumulative_tick_loss_figure({label: _slice(daily, start, end)["tick_loss"].cumsum()}), width="stretch")


def _slice(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]


# ---- the run box and the analysis period ---------------------------------------------------------------

def _run_box(run: wf.Run) -> None:
    """The run's details in the left column: forecast period, tickers + weights, return unit, features, target."""
    header = dd.run_header(run)
    with st.container(border=True):
        st.markdown(f"**Forecast** {header['forecast_start'].date()} → {header['forecast_end'].date()}")
        st.markdown(f"**Tickers + weights** {header['tickers_weights']}")
        st.markdown(f"**Return unit** {header['unit']}", help="The return type the run was computed with. Everything shown here is converted to percent returns.")
        if header["target"] is None:
            st.markdown("**Features** none (reads only the returns)")
            st.markdown("**Target** none")
        else:
            st.markdown(f"**Features** `{header['features']}` · unit: {header['feature_unit']}")
            st.markdown(f"**Target** {header['target']} · scale: {header['target_scale']:g} · unit: {header['target_unit']}")


def _analysis_period(daily: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = daily.index[0].date(), daily.index[-1].date()
    with st.container(border=True):
        st.markdown("**Analysis period**")
        chosen = st.date_input("Analysis start – analysis end", value=(first, last), min_value=first, max_value=last, key="single_period")
    if not isinstance(chosen, (tuple, list)) or len(chosen) != 2:
        st.info("Pick the end date too.")
        st.stop()
    start, end = pd.Timestamp(chosen[0]), pd.Timestamp(chosen[1])
    if len(daily.loc[start:end]) < 30:
        st.warning("The analysis period has fewer than 30 forecast days: pick a longer one.")
        st.stop()
    return start, end


# ---- the VaR review table ------------------------------------------------------------------------------

def _verdict(passed: bool) -> str:
    return ":green[**PASS**]" if passed else ":red[**FAIL**]"


def _reading_text(reading: wf.Reading) -> str:
    window = "expanding" if reading.window_size is None else reading.window_size
    return (
        f"**VAR RESULTS UNDER:** Var Confidence: {reading.var_confidence:g} · Min Periods: {reading.min_periods} · "
        f"Residual Window: {window} · Test Confidence: {reading.test_confidence:g}"
    )


def _review(var_df: pd.DataFrame, reading: wf.Reading, color: str | None) -> None:
    """Statistical test results, then the loss and conservativity block, of one reading on the analysis period."""
    backtest, level = dd.review_results(var_df, reading)
    with st.container(border=True):
        st.markdown(f":{color}[{_reading_text(reading)}]" if color else _reading_text(reading))
        st.markdown("**Statistical test results**")
        days, exceedances, kupiec, independence, coverage = st.columns([1, 1.3, 1.5, 1.5, 1.5])
        days.metric("Forecast days", f"{backtest['n_days']}")
        exceedances.metric("Exceedances", f"{backtest['n_exceedances']}")
        exceedances.caption(f"Rate **{backtest['exceedance_rate']:.1%}** (expected {backtest['expected_exceedance_rate']:.0%} = 1 − confidence)")
        for column, title, key, text in (
            (kupiec, "Kupiec POF", "kupiec_pof", "Tests whether the observed exceedance rate differs statistically from the expected one."),
            (independence, "Christoffersen independence", "christoffersen_ind", "Tests whether a violation today makes one tomorrow more likely (clustering)."),
            (coverage, "Christoffersen conditional coverage", "christoffersen_cc", "Joint test of the correct violation rate and independence."),
        ):
            column.markdown(f"**{title}**", help=text)
            column.markdown(_verdict(bool(backtest[f"{key}_pass"])))
            column.caption(f"test statistic {backtest[f'{key}_stat']:.3f} · critical value {backtest[f'{key}_critical_value']:.3f}")

        st.markdown("**Loss + conservativity**")
        left, right = st.columns([1.4, 1.2])
        with left:
            total, exceed, overshoot = st.columns(3)
            total.metric(
                "Mean tick loss", f"{level['mean_tick_loss']:.6f}", format_value(level["mean_tick_loss"]), delta_color="off",
                help="The headline score of a VaR: the exceedance part plus the overshoot part. Lower is better.",
            )
            exceed.metric("= tick loss from exceedance", f"{level['tick_loss_exceedance_part']:.6f}", help="How much the model underestimated risk.")
            overshoot.metric("+ tick loss from overshoot", f"{level['tick_loss_overshoot_part']:.6f}", help="How much the model overestimated risk.")
        with right:
            slack, ratio = st.columns(2)
            slack.metric(
                "Mean slack (non-exceedance days)", format_value(level["mean_slack_non_exceedance"], 3),
                help="Average gap between the VaR and the realized loss on days without a violation. Higher means more unused capital.",
            )
            ratio.metric(
                "VaR / realized quantile ratio", f"{level['var_to_realized_quantile_ratio']:.4f}",
                help="Average VaR divided by the realized loss quantile at the VaR confidence. Above 1 the model is more conservative, below 1 more aggressive.",
            )
            ratio.caption(f"average VaR {format_value(level['mean_var'], 3)} · realized loss quantile {format_value(level['realized_loss_quantile'], 3)}")


# ---- predicted variance vs realized variance -----------------------------------------------------------

def _variance_panel(runs: pd.DataFrame, row: pd.Series, run: wf.Run, daily: pd.DataFrame, settings: dd.Settings, start, end) -> None:
    store_root, reading = settings.store_root, settings.reading
    ewma_options = _ewma_options(runs, row)
    ewma_run = None
    if not ewma_options.empty:
        ewma_label = st.selectbox(
            "EWMA reference run", ewma_options["display_name"].tolist(), key=f"single_ewma_ref_{row['walk_forward_id']}",   # one choice per walk-forward: a run starts with its own
            help="The plain EWMA variance (decay 0.94) of the portfolio returns, the benchmark the model is compared with.",
        )
        ewma_run = _run(ewma_options.set_index("display_name").loc[ewma_label, "store_run_id"], store_root)

    dates = daily.loc[start:end].index
    frame = dd.variance_baselines(run, reading, ewma_run, dates)
    explanations = {
        "EWMA (reference run)": "The reference run's EWMA variance (the no-model forecast above).",
        "squared return r²": "The day's realized squared portfolio return: the noisy one-day realized variance that every variance score is computed against.",
        "target-definition variance": "The realized return passed through the model's own target and back to a variance (what the model is trained to hit); equals r² for the squared-return target.",
        "realized variance (21-day mean of r²)": "Trailing 21-day mean of r²: a smoother realized-variance yardstick.",
    }
    baselines = [c for c in frame.columns if c != "model"]
    st.markdown("**Baselines**", help="Lines drawn next to the model's variance in the chart below. They only change the chart, not the metrics.")
    chosen = []
    for column in baselines:
        if st.checkbox(column, value=column.startswith("EWMA") or (column == baselines[0] and ewma_run is None), key=f"single_baseline_{column}", help=explanations.get(column)):
            chosen.append(column)
    frame = frame[["model", *chosen]]

    r2 = daily["loss"].loc[frame.index] ** 2
    model_variance = frame["model"]
    cols = st.columns(2)
    qlike = dd.qlike(model_variance, r2)
    delta = None
    if ewma_run is not None:
        ewma_variance = ewma_run.daily(reading)["predicted_variance"].loc[frame.index]
        delta = f"{qlike - dd.qlike(ewma_variance, r2):+.4f} vs EWMA"
    cols[0].metric(
        "QLIKE (forecast period)", f"{qlike:.4f}", delta, delta_color="inverse",
        help="The loss of the variance forecast against the realized squared return; it punishes a too low forecast more than a too high one. "
             "Lower is better. It does not depend on the unit or the scale, so it is comparable between any models.",
    )
    base = ewma_run.daily(reading)["predicted_variance"] if ewma_run is not None else None
    mse = dd.target_mse(run, model_variance, base)
    cols[1].metric(
        "MSE with target definition", "n/a" if pd.isna(mse) else f"{mse:.4f}",
        help="Squared error in the units of the model's own target (portfolio level): only comparable between models trained on the same target and scale.",
    )

    log_y = st.toggle("Log y axis", value=True, key="single_var_log")
    st.plotly_chart(figs.variance_vs_baselines_figure(frame, log_y), width="stretch")

    # the window losses are computed by the walk-forward (wf.window_losses) when the run is trained and saved with it
    totals = run.total_window_losses
    st.markdown("**Window losses**", help="Over all windows of the run (not the analysis period). Every window counts with the number of days it covers.")
    cols = st.columns(5)
    loss_definition = run.params.get("training_loss")
    cols[0].metric("Model training loss definition", str(loss_definition) if loss_definition else "n/a", help="The loss the model minimized while training.")
    own_loss_note = "Squared error in the units of the model's own target: only comparable between models trained on the same target and scale."
    for column, label, key, note in (
        (cols[1], "Total window forecast loss", "total_window_forecast_loss", "That loss on the forecast rows of every window. " + own_loss_note),
        (cols[2], "Total window training loss", "total_window_train_loss", "That loss on the training rows (in-sample) of every window. " + own_loss_note),
    ):
        column.metric(label, "n/a" if pd.isna(totals[key]) else f"{totals[key]:.4f}", help=note)
    qlike_note = "QLIKE does not depend on the unit or the scale: comparable between any models (long or wide, XGBoost, GARCH or EWMA)."
    for column, label, key, note in (
        (cols[3], "Total window QLIKE (forecast)", "total_window_forecast_qlike", "The forecast days of every window. The same number as 'QLIKE (forecast period)' when the analysis period is the whole run. " + qlike_note),
        (cols[4], "Total window QLIKE (training)", "total_window_train_qlike", "The training days (in-sample) of every window. " + qlike_note),
    ):
        column.metric(label, f"{totals[key]:.4f}", help=note)
    with st.expander("Loss per window"):
        st.dataframe(run.window_losses, width="stretch")


# ---- several VaR readings of the same run --------------------------------------------------------------

def _readings_comparison(label: str, run: wf.Run, store_run_id: str, settings: dd.Settings, start, end) -> None:
    extra: list[wf.Reading] = st.session_state.setdefault("single_extra_readings", [])
    with st.expander("Add reading", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        var_confidence = c1.select_slider("VaR confidence", [0.90, 0.95, 0.975, 0.99], value=0.95, key="add_conf")
        expanding = c2.checkbox("Expanding residual window", key="add_expanding")
        window = None if expanding else int(c2.number_input("Residual window", 20, 1000, 252, key="add_window"))
        min_periods = int(c3.number_input("Min periods", 5, 500, 30, key="add_min"))
        test_confidence = c4.select_slider("Test confidence", [0.90, 0.95, 0.99], value=0.95, key="add_test")
        if st.button("Add reading"):
            new = wf.Reading(var_confidence, window, min_periods, test_confidence)
            if new != settings.reading and new not in extra:
                extra.append(new)
                st.rerun()

    readings = [settings.reading, *extra]
    chart_runs, bank = {}, run.combined_bank(percent=True)
    for index, reading in enumerate(readings):
        color = READING_COLORS[index % len(READING_COLORS)]
        try:
            var_df = _slice(_var_df(store_run_id, settings.store_root, reading), start, end)
            _review(var_df, reading, color)
        except ValueError as error:
            st.error(f"{_reading_text(reading)}: {error}")
            continue
        chart_runs[f"c{reading.var_confidence:g} w{reading.window_size or 'exp'} m{reading.min_periods}"] = (var_df, bank)
        if index and st.button("Remove this reading", key=f"remove_reading_{index}"):
            extra.remove(reading)
            st.rerun()
    if len(chart_runs) > 1:
        st.plotly_chart(
            plot_var_models_vs_loss(chart_runs, title=f"{label}: VaR under different readings"), width="stretch",
        )


# ---- more details (kept from the first version) ----------------------------------------------------------

def _per_ticker(store_run_id: str, settings: dd.Settings, is_ewma_family: bool) -> None:
    frames = _ticker_frames(store_run_id, settings.store_root)
    st.markdown(
        "**Variance per ticker**",
        help="Model variance is what the model predicted for the ticker, EWMA variance is the EWMA's own variance of that ticker on the same day."
             + (" This is an EWMA run, so both are the same." if is_ewma_family else ""),
    )
    st.dataframe(dd.ticker_variance_summary(frames), width="stretch")
    left, right = st.columns(2)
    log_y = left.toggle("Log y axis", value=True, key="single_ticker_log")
    show_r2 = right.toggle("Show the squared return", value=False, key="single_ticker_r2")
    st.plotly_chart(figs.ticker_variance_figure(frames, log_y, show_r2), width="stretch")


def _windows(run: wf.Run, reading: wf.Reading) -> None:
    blocks = run.blocks(reading)
    st.dataframe(blocks.infer_objects(), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(figs.bar_figure(blocks["mean_tick_loss"].astype(float), "Mean tick loss per window", "mean tick loss"), width="stretch")
    right.plotly_chart(figs.bar_figure(blocks["exceedance_rate"].astype(float), "Exceedance rate per window", "rate"), width="stretch")
    with st.expander("Windows (calendars)"):
        st.dataframe(run.windows, hide_index=True, width="stretch")


def _ewma_options(runs: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """The 'ewma' runs, the ones of this run's walk-forward first."""
    ewma_runs = runs[runs["model_family"] == "ewma"]
    return pd.concat([
        ewma_runs[ewma_runs["walk_forward_id"] == row["walk_forward_id"]],
        ewma_runs[ewma_runs["walk_forward_id"] != row["walk_forward_id"]],
    ])


def _reference_run(runs: pd.DataFrame, row: pd.Series, store_root: str) -> wf.Run | None:
    """The EWMA run chosen in the variance tab (the first one of this walk-forward before it is touched)."""
    options = _ewma_options(runs, row)
    if options.empty:
        return None
    label = st.session_state.get(f"single_ewma_ref_{row['walk_forward_id']}")
    chosen = options.set_index("display_name").loc[label] if label in set(options["display_name"]) else options.iloc[0]
    return _run(chosen["store_run_id"], store_root)


def _focus_day(label: str, run: wf.Run, daily: pd.DataFrame, day: pd.Timestamp, runs: pd.DataFrame, row: pd.Series, settings: dd.Settings) -> None:
    """The days around one day: the VaR chart + table, then the portfolio variance forecast chart + table."""
    bars = st.slider("Days on each side", 3, 40, 10, key="single_focus_bars")
    reference = _reference_run(runs, row, settings.store_root)
    ewma_variance = None if reference is None else reference.daily(settings.reading)["predicted_variance"]
    var_table, variance_table = dd.day_window_tables(daily, ewma_variance, day, bars)

    st.subheader(f"VaR around {day.date()}")
    st.plotly_chart(plot_exceedance_days_focus({label: daily[["varForecast", "loss"]]}, [day], window_bars=bars), width="stretch")

    def tint(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
        """The focused day yellow, the exceedance days red."""
        def paint(r):
            if r.name == day:
                return ["background-color: #fff3b0"] * len(r)
            return ["background-color: #ffe0e0"] * len(r) if frame.loc[r.name, "exceeded"] else [""] * len(r)
        return frame.style.apply(paint, axis=1)

    st.dataframe(
        tint(var_table).format({
            "loss": format_value, "varForecast": format_value,
            "excess": lambda v: format_value(v, signed=True), "tick_loss": "{:.6f}",
        }), width="stretch",
    )

    st.subheader(f"Portfolio variance forecast around {day.date()}")
    if reference is None:
        st.info("There is no 'ewma' run in this experiment to set the forecast against.")
    log_y = st.toggle("Log y axis", value=False, key="single_focus_log")
    st.plotly_chart(figs.day_variance_figure(variance_table, day, log_y), width="stretch")

    def tint_variance(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
        return frame.style.apply(lambda r: ["background-color: #fff3b0"] * len(r) if r.name == day else [""] * len(r), axis=1)

    st.dataframe(tint_variance(variance_table).format("{:.3e}", subset=[c for c in variance_table.columns if c not in ("model / EWMA", "standardized residual", "predicted sigma")]).format("{:.3f}", subset=[c for c in ("model / EWMA", "standardized residual") if c in variance_table]), width="stretch")


def _exceedances(label: str, run: wf.Run, daily: pd.DataFrame, runs: pd.DataFrame, row: pd.Series, settings: dd.Settings) -> None:
    reading = settings.reading
    days = run.exceeding_days(reading)
    if days.empty:
        st.info("There are no exceedance days in this run.")
        return
    table = days[["loss", "varForecast", "excess", "window_id", "predicted_variance", "predicted_sigma", "quantile"]].sort_values("excess", ascending=False)
    st.markdown(
        f"**{len(days)} exceedance days** under the current reading",
        help="Click a row to explain that day (nothing clicked yet: the largest excess). The clicked day stays selected when you change "
             "the VaR reading in the left panel, even if it is no longer an exceedance.",
    )
    event = st.dataframe(table, width="stretch", on_select="rerun", selection_mode="single-row", key="single_exceedance_table")

    # The day is kept in the session (the table is rebuilt when the reading changes, the day is not). A click on a row
    # sets it; so does the picker below. A selection that stays at the same row numbers on a rebuilt table is not a new
    # click, so only a CHANGE of the selected rows counts.
    rows = list(event.selection.rows)
    if rows and rows != st.session_state.get("single_selection_seen"):
        st.session_state["single_explain_day"] = table.index[rows[0]]
        st.session_state["single_pick_day"] = table.index[rows[0]].date()      # the picker follows the click
    st.session_state["single_selection_seen"] = rows

    low, high = daily.index[0].date(), daily.index[-1].date()
    current = st.session_state.get("single_explain_day", table.index[0]).date()
    if not low <= current <= high:       # the kept day is outside the run now shown (another run with a shorter period): start over
        st.session_state["single_explain_day"] = table.index[0]
        current = table.index[0].date()
    if "single_pick_day" not in st.session_state or not low <= st.session_state["single_pick_day"] <= high:
        st.session_state["single_pick_day"] = current
    picked = st.date_input(
        "Or pick any forecast day (it need not be an exceedance)", min_value=low, max_value=high, key="single_pick_day",
    )
    if picked != current:    # the picker was moved by the user (a click sets both)
        st.session_state["single_explain_day"] = pd.Timestamp(picked)
    day = pd.Timestamp(picked)
    if day not in daily.index:
        st.warning(f"{picked} is not a forecast day of this run with a VaR under this reading (weekend, holiday, outside the period, or not enough residual history): pick another day or another reading.")
        return

    st.subheader(f"Explain the VaR of {day.date()}")
    if day in table.index:
        st.caption("Under the current reading this day IS an exceedance.")
    else:
        st.info(f"Under the current reading this day is NOT an exceedance: loss {format_value(daily.loc[day, 'loss'])} vs VaR {format_value(daily.loc[day, 'varForecast'])}.")
    explained = rs.explain_day(day, run, reading)
    # the value column mixes numbers, dates and flags: show it as text (Arrow cannot hold a mixed column)
    explained["value"] = explained["value"].map(lambda v: f"{v:.6g}" if isinstance(v, float) else str(v))
    st.dataframe(explained, width="stretch")
    _focus_day(label, run, daily, pd.Timestamp(day), runs, row, settings)


def _model(run: wf.Run) -> None:
    importance = dd.feature_importance(run)
    if importance is not None:
        st.subheader("Feature importance (XGBoost)")
        st.plotly_chart(figs.bar_figure(importance.mean().sort_values(ascending=False), "Mean importance over the windows", "importance"), width="stretch")
        st.plotly_chart(figs.heatmap_figure(importance, "Importance per window"), width="stretch")
    params = dd.garch_params(run)
    if params is not None:
        st.subheader("GARCH parameters per window")
        st.dataframe(params, width="stretch")
    with st.expander("Recipe and settings (model_settings.json)"):
        st.json(run.config)
