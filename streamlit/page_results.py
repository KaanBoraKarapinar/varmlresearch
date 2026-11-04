"""
Page: results. The written results of the research (streamlit/results.md), shown as they are, with the markdown tables shown as Streamlit tables and, in front of
the tables that come from a saved experiment, a button that opens the experiment.

In results.md a line

    <!-- open: <experiment slug> | <analysis period> | <tab> | <text to scroll to> -->

puts a "Go to experiment" button in front of the table that follows it (or a button of its own, when no table follows): it opens that experiment on the
Experiments page with that analysis period. <experiment slug> is the file name of store/experiments/<slug>.json; <analysis period> is "Entire set" (the
experiment's own period) or the label of a core.analysis_periods period, e.g. "COVID-19 · full". The last two are optional: <tab> is the tab to open (one of
page_run_comparison.TABS; the VaR review when left out) and <text to scroll to> the text of a paragraph of that tab the page is scrolled to (the tabs when left
out).

    <!-- weights: <number of tickers> | <train bars> | <forecast bars> -->

is a button of its own, "Single portfolio specific VaR results": it opens the user defined weights page with the ticker set of that many tickers and those
windows chosen, where one of the saved portfolios can be loaded. The text of the page is edited in results.md, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

import core
import dashboard_data as dd
import page_experiments as pe
import page_run_comparison as cmp
import page_weight_comparison as pw
from core import analysis_periods as ap

RESULTS_TEXT = Path(__file__).with_name("results.md")
ENTIRE_SET = "Entire set"      # in results.md: the experiment's own analysis period
DIRECTIVE = re.compile(r"^<!--\s*open:\s*(?P<arguments>.+?)\s*-->$")
WEIGHTS_DIRECTIVE = re.compile(r"^<!--\s*weights:\s*(?P<tickers>\d+)\s*\|\s*(?P<train>\d+)\s*\|\s*(?P<forecast>\d+)\s*-->$")
WEIGHTS_LABEL = "Single portfolio specific VaR results"
CELL_SEPARATOR = re.compile(r"(?<!\\)\|")      # a | that is not the escaped \| of a cell


@dataclass(frozen=True)
class Link:
    slug: str
    period: str                     # ENTIRE_SET or the label of an analysis period
    tab: str | None = None          # one of cmp.TABS; None: the VaR review
    scroll_to: str | None = None    # the text of a paragraph of the tab; None: the tabs


def _link(arguments: str) -> Link:
    """The Link of the arguments of a directive: slug | period [| tab [| text to scroll to]]."""
    parts = [part.strip() for part in arguments.split("|")]
    if not 2 <= len(parts) <= 4:
        raise ValueError(f"A line '<!-- open: ... -->' of results.md has {len(parts)} parts, it needs 2 to 4 (slug | period | tab | text to scroll to): {arguments}")
    return Link(*parts)


@dataclass(frozen=True)
class Block:
    kind: str               # 'text', 'table' (with its link, if it has one), 'button' (a link on its own), 'weights' (the button to the user defined weights page)
    text: str = ""
    link: Link | None = None
    ticker_set: dict | None = None      # 'weights': n_tickers, train_bars, forecast_bars (what pw.OPEN_SET_KEY takes)


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _table_end(lines: list[str], start: int) -> int:
    """The index after the last line of the table that starts at `start`."""
    end = start
    while end < len(lines) and _is_table_line(lines[end]):
        end += 1
    return end


def blocks(text: str) -> list[Block]:
    """results.md as text blocks, tables (with the link written in front of them, if any) and links on their own, in order."""
    lines, out, prose = text.splitlines(), [], []

    def flush() -> None:
        if "\n".join(prose).strip():
            out.append(Block("text", "\n".join(prose)))
        prose.clear()

    i = 0
    while i < len(lines):
        found = DIRECTIVE.match(lines[i].strip())
        weights = WEIGHTS_DIRECTIVE.match(lines[i].strip())
        if weights:
            flush()
            out.append(Block("weights", ticker_set={"n_tickers": int(weights["tickers"]), "train_bars": int(weights["train"]), "forecast_bars": int(weights["forecast"])}))
            i += 1
        elif found:
            flush()
            link = _link(found["arguments"])
            nxt = i + 1
            while nxt < len(lines) and not lines[nxt].strip():
                nxt += 1
            if nxt < len(lines) and _is_table_line(lines[nxt]):
                end = _table_end(lines, nxt)
                out.append(Block("table", "\n".join(lines[nxt:end]), link))
                i = end
            else:
                out.append(Block("button", link=link))
                i += 1
        elif _is_table_line(lines[i]):
            flush()
            end = _table_end(lines, i)
            out.append(Block("table", "\n".join(lines[i:end])))
            i = end
        else:
            prose.append(lines[i])
            i += 1
    flush()
    return out


def table_frame(table: str) -> pd.DataFrame:
    """A markdown table (header, separator, rows) as a DataFrame of text; the markdown marks (**bold**, `code`) of the cells are dropped, \\| is a |."""
    def cells(line: str) -> list[str]:
        inner = line.strip()[1:-1] if line.strip().endswith("|") else line.strip()[1:]
        return [c.replace("\\|", "|").replace("**", "").replace("`", "").strip() for c in CELL_SEPARATOR.split(inner)]

    lines = table.splitlines()
    header, rows = cells(lines[0]), [cells(line) for line in lines[2:]]      # lines[1] is the |---|---| separator
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f"A row of the table with the header '{lines[0][:80]}' has {len(row)} cells, the header has {len(header)}: {' | '.join(row)[:120]}")
    return pd.DataFrame(rows, columns=header)


def _problem(link: Link, saved: dict[str, core.Experiment]) -> str | None:
    """Why the link cannot be opened (the experiment is not saved, the period or the tab does not exist), or None."""
    if link.slug not in saved:
        return f"The experiment '{link.slug}' is not saved (store/experiments/{link.slug}.json)."
    if link.period != ENTIRE_SET and link.period not in {p.label for p in ap.stress_periods()}:
        return f"'{link.period}' is not an analysis period (core.analysis_periods) nor '{ENTIRE_SET}'."
    if link.tab is not None and link.tab not in cmp.TABS:
        return f"'{link.tab}' is not a tab of the experiment page: {', '.join(cmp.TABS)}."
    return None


def _button(link: Link, number: int, saved: dict[str, core.Experiment], experiments_page, *, named: bool) -> None:
    """The button that opens the experiment on the Experiments page with the period; named: its label says which experiment."""
    problem = _problem(link, saved)
    if problem:
        st.error(problem)
        return
    experiment = saved[link.slug]
    where = "its own analysis period" if link.period == ENTIRE_SET else f"the analysis period {link.period}"
    label = f"Go to experiment: {experiment.name}" if named else "Go to experiment"
    if st.button(label, key=f"results_open_{number}", help=f"Opens '{experiment.name}' on the Experiments page with {where}."):
        st.session_state[pe.OPEN_KEY] = link.slug
        st.session_state[pe.OPEN_PERIOD_KEY] = pe.OWN_PERIOD if link.period == ENTIRE_SET else link.period
        st.session_state[pe.OPEN_TAB_KEY] = link.tab
        st.session_state[pe.OPEN_SCROLL_KEY] = link.scroll_to
        st.switch_page(experiments_page)


def _weights_button(ticker_set: dict, number: int, weights_page) -> None:
    """The button to the user defined weights page, with the ticker set chosen."""
    help_text = (
        f"Opens the page 'Model comparison with user defined weights' with the {ticker_set['n_tickers']}-ticker set (train {ticker_set['train_bars']} / forecast "
        f"{ticker_set['forecast_bars']}): load one of the saved portfolios there and see the VaR results of the models for that portfolio alone."
    )
    if st.button(WEIGHTS_LABEL, key=f"results_weights_{number}", help=help_text):
        st.session_state[pw.OPEN_SET_KEY] = ticker_set
        st.switch_page(weights_page)


def render(settings: dd.Settings, sidebar_top, experiments_page, weights_page) -> None:
    """experiments_page, weights_page: the st.Pages of the Experiments page and of the user defined weights page (dashboard.py), which the buttons switch to."""
    st.title("Results")
    saved = {e.slug: e for e in pe._experiments(settings.store_root, pe._stamp(settings.store_root))}
    for number, block in enumerate(blocks(RESULTS_TEXT.read_text(encoding="utf-8"))):
        if block.kind == "text":
            st.markdown(block.text)
        elif block.kind == "weights":
            _weights_button(block.ticker_set, number, weights_page)
        elif block.kind == "button":
            _button(block.link, number, saved, experiments_page, named=True)
        else:
            if block.link is not None:
                _button(block.link, number, saved, experiments_page, named=False)
            frame = table_frame(block.text)
            st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)
