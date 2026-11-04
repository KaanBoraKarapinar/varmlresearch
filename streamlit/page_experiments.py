"""Page 3: saved experiments. An experiment fixes the runs, the baseline, the analysis period and the VaR reading; the page shows them with our conclusion."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import core
import dashboard_data as dd
import page_run_comparison as cmp
from core import analysis_periods as ap
from core import walk_forward as wf

PICK_KEY = "exp_pick"       # the experiment selectbox
PANEL_READING_KEY = "exp_use_panel_reading"     # the checkbox: read the experiment with the left panel's VaR reading
PERIOD_KEY = "exp_period"   # the analysis period selectbox (one per experiment: its options are the periods that experiment's runs cover)
OPEN_KEY = "exp_open"       # the slug of an experiment another page wants opened here (the Results page); a widget's own key would be dropped on the way
OPEN_PERIOD_KEY = "exp_open_period"     # ... and the label of the analysis period it should be opened on
OPEN_TAB_KEY = "exp_open_tab"           # ... the tab it should be opened on (one of cmp.TABS; None: the VaR review) ...
OPEN_SCROLL_KEY = "exp_open_scroll"     # ... and the text of the part of the page to scroll to (None: the tabs)
DEFAULT_TAB_KEY = "exp_default_tab"     # the tab the tabs are built with: the one last asked for by another page, kept so that it does not change between reruns
OWN_PERIOD = "Experiment's own period"
TABLE_PREFIX = "exp_models"     # the keys of the models table (one per experiment: its ticks are kept per experiment)
BASELINE_KEY = "exp_baseline"


def scroll_script(text: str | None) -> str:
    """
    JavaScript that scrolls the page to the first visible paragraph that contains `text` (a heading of the tab that is open), or to the tabs when text is None.
    It looks every 0.2 s for up to 5 s, as the page is still being drawn when it starts.
    """
    return f"""<script>
(() => {{
  const text = {json.dumps(text)};
  let tries = 0;
  const timer = setInterval(() => {{
    tries += 1;
    const target = text === null
      ? (document.querySelector('[data-testid="stTabs"]') || document.querySelector('[role="tablist"]'))
      : [...document.querySelectorAll('[data-testid="stMarkdownContainer"] p')].find(p => p.textContent.includes(text) && p.offsetParent !== null);
    if (target) {{ target.scrollIntoView({{behavior: "smooth", block: "start"}}); clearInterval(timer); }}
    else if (tries > 25) clearInterval(timer);
  }}, 200);
}})();
</script>"""


@st.cache_data(show_spinner=False)
def _experiments(store_root: str, stamp: tuple) -> list[core.Experiment]:
    """stamp (file names + modification times) makes the cache notice a saved or edited experiment."""
    return core.list_saved_experiments(Path(store_root))


def _stamp(store_root: str) -> tuple:
    folder = core.experiment.experiments_dir(Path(store_root))
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(folder.glob("*.json"))) if folder.exists() else ()


@st.cache_data(show_spinner=False)
def _same_portfolio(store_run_ids: tuple, reference_id: str, store_root: str) -> tuple:
    """The store run ids among `store_run_ids` with the tickers and weights of the reference run: they can be compared with it on their common days."""
    key = cmp._portfolio_key(cmp._run(reference_id, store_root))
    return tuple(i for i in store_run_ids if cmp._portfolio_key(cmp._run(i, store_root)) == key)


def _candidates(own_runs: pd.DataFrame, runs: pd.DataFrame, store_root: str) -> pd.DataFrame:
    """The experiment's runs (saved order), then the other runs of the same tickers and weights; the portfolio runs kept in the store are left out."""
    others = runs[~runs["store_run_id"].isin(own_runs["store_run_id"]) & ~runs["store_run_id"].map(dd.is_virtual_run_id) & (runs["portfolio"] == own_runs["portfolio"].iloc[0])]
    same = _same_portfolio(tuple(others["store_run_id"]), own_runs["store_run_id"].iloc[0], store_root)
    return pd.concat([own_runs, others[others["store_run_id"].isin(same)]])


def _reading_text(reading: wf.Reading) -> str:
    return (
        f"Var Confidence: {reading.var_confidence:g} · Min Periods: {reading.min_periods} · "
        f"Residual Window: {'expanding' if reading.window_size is None else reading.window_size} · Test Confidence: {reading.test_confidence:g}"
    )


def render(settings: dd.Settings, sidebar_top) -> None:
    st.title("Experiments")
    saved = _experiments(settings.store_root, _stamp(settings.store_root))
    if not saved:
        st.info("No experiment saved yet: create one with core.save_experiment(core.Experiment(...), store_root).")
        return
    opened, opened_period = st.session_state.pop(OPEN_KEY, None), st.session_state.pop(OPEN_PERIOD_KEY, None)
    opened_tab, opened_scroll = st.session_state.pop(OPEN_TAB_KEY, None), st.session_state.pop(OPEN_SCROLL_KEY, None)
    if opened is not None:
        st.session_state[DEFAULT_TAB_KEY] = opened_tab or cmp.VAR_REVIEW_TAB
    if opened is not None:
        st.session_state[PICK_KEY] = next(e for e in saved if e.slug == opened)
    with sidebar_top:
        st.header("Experiment")
        experiment = st.selectbox("Experiment", saved, format_func=lambda e: e.name, key=PICK_KEY, label_visibility="collapsed")
        use_panel_reading = st.checkbox(
            "Use the VaR reading of the left panel", key=PANEL_READING_KEY,
            help="Off: the experiment's own VaR reading, so that the result does not change with the left panel. On: the VaR confidence, residual window, min periods "
                 "and test confidence set in the left panel, to look at the same runs under another reading.",
        )

    saved_reading = wf.Reading(**experiment.reading)
    reading = settings.reading if use_panel_reading else saved_reading
    settings = dataclasses.replace(settings, reading=reading)

    runs = cmp._runs(settings.tracking_uri, settings.experiment, settings.store_root)
    by_id = runs.set_index("store_run_id", drop=False) if not runs.empty else runs
    missing = [i for i in experiment.run_ids if i not in by_id.index]
    if missing:
        st.error(f"{len(missing)} run(s) of this experiment are not in the MLflow experiment '{settings.experiment}': {', '.join(missing)}")
        return
    own_runs = by_id.loc[list(experiment.run_ids)]                           # the experiment's runs, in the saved order
    own_names = set(own_runs["display_name"])

    st.subheader(experiment.name)
    if experiment.question:
        st.markdown(f"*{experiment.question}*")
    # the models: the experiment's runs are ticked to start with, any can be unticked and the other runs of the same portfolio ticked
    candidates = _candidates(own_runs, runs, settings.store_root)
    with st.expander("Models: tick to remove or add runs", expanded=False):
        selected = cmp._runs_table(candidates, candidates, prefix=f"{TABLE_PREFIX}_{experiment.slug}", initial=own_names)
    if not selected:
        st.info("Tick at least one model.")
        return
    st.caption(f"{len(selected)} models ticked: {len(own_names & set(selected))} of the experiment's {len(own_names)} and {len(set(selected) - own_names)} added.")
    table = candidates.set_index("display_name").loc[selected]
    labels = selected
    own_baseline = by_id.loc[experiment.baseline_run_id, "display_name"]
    with sidebar_top:
        baseline = st.selectbox(
            "Baseline run", labels, index=labels.index(own_baseline) if own_baseline in labels else 0, key=f"{BASELINE_KEY}_{experiment.slug}",
            help="The run the others are compared with (differences are run minus baseline, so negative is better). The experiment's own baseline to start with.",
        )

    common = cmp._common(tuple(table["store_run_id"]), tuple(labels), settings.store_root, reading)
    # the experiment's own period, or one of the stress periods that all of its runs have forecast (core.analysis_periods)
    periods = {p.label: p for p in ap.stress_periods() if p.covered_by(common["dates"])}
    period_key = f"{PERIOD_KEY}_{experiment.slug}"
    if opened_period is not None:
        st.session_state[period_key] = opened_period
    with sidebar_top:
        with st.container(border=True):
            st.markdown("**Analysis period**", help="The experiment's own period, or a market stress period the runs of this experiment all cover. The runs stay those of the experiment.")
            picked = st.selectbox(
                "Analysis period", [OWN_PERIOD, *periods], key=period_key, label_visibility="collapsed",
                format_func=lambda label: f"{label}: {experiment.start} → {experiment.end}" if label == OWN_PERIOD else f"{label}: {periods[label].start} → {periods[label].end}",
            )
    start, end = (experiment.start, experiment.end) if picked == OWN_PERIOD else (periods[picked].start, periods[picked].end)

    own = [what for what, differs in (("analysis period", picked != OWN_PERIOD), ("VaR reading", reading != saved_reading), ("models", set(labels) != own_names), ("baseline", baseline != own_baseline)) if differs]
    if experiment.notes:
        if own:
            st.caption(f"The notes were written for the experiment's own {', '.join(own[:-1]) + ' and ' + own[-1] if len(own) > 1 else own[0]}, not for what is shown below.")
        st.markdown(experiment.notes)
    st.markdown(
        f":blue[**EXPERIMENT:**] {len(labels)} runs · baseline {baseline} · analysis period {start} → {end} · {_reading_text(reading)}",
        help="The experiment's runs are ticked to start with and can be changed in 'Models'. The VaR reading is the experiment's own unless 'Use the VaR reading of the left panel' is ticked.",
    )
    if reading != saved_reading:
        st.warning(f"This is not the VaR reading the experiment was saved with ({_reading_text(saved_reading)}): the left panel's reading is used. Untick 'Use the VaR reading of the left panel' to go back.")
    others = [e.name for e in saved if e.name != experiment.name and set(e.run_ids) & set(experiment.run_ids)]
    if others:
        st.caption("Runs of this experiment are also in: " + ", ".join(others))

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    var_dfs = {label: cmp._slice(frame, start, end) for label, frame in common["var_dfs"].items()}
    dailies = {label: cmp._slice(frame, start, end) for label, frame in common["dailies"].items()}
    days = len(next(iter(var_dfs.values())))
    if days < cmp.MIN_ANALYSIS_DAYS:
        st.warning(f"The runs share only {days} days in this analysis period; the comparison needs at least {cmp.MIN_ANALYSIS_DAYS}.")
        return
    chosen = {label: cmp._run(table.loc[label, "store_run_id"], settings.store_root) for label in labels}
    cmp.analysis(
        table, chosen, var_dfs, dailies, baseline, settings, start, end, cmp._ewma_references(runs, settings.store_root),
        id_order=list(candidates["display_name"]), default_tab=st.session_state.get(DEFAULT_TAB_KEY, cmp.VAR_REVIEW_TAB),
    )
    if opened is not None:      # arrived from another page (Results): the page is scrolled down to the part it asked for, or to the tabs, where the results are
        st.html(scroll_script(opened_scroll), unsafe_allow_javascript=True)
