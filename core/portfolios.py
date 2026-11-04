"""
The portfolios of the store and their names: store/Portfolios/.

    store/Portfolios/
        portfolios.jsonl        one line per portfolio, written once: portfolio_id, name, weights, tags, n_eff, ...
        portfolio_runs.jsonl    one line per (portfolio, run): which runs belong to the portfolio, wherever they are kept
                                (a virtual run under <walk_forward>/VirtualRuns/, or a baseline trained for these weights)

A portfolio is its weights (core.virtual_runs.portfolio_id: a hash of the normalised weights), so the same weights are one portfolio
however many walk-forwards or models read them. The name is a label for people and never silently changes meaning: the same portfolio under
another name, or another portfolio under a name already taken, raises. Without a name the portfolio gets a description made from its weights
(auto_name), so a run is never labelled by a hash.

tags are free provenance written with the portfolio, never needed to read it: how it was made and which group it belongs to, e.g.
{"group": "random200_v1", "generator": "dirichlet", "alpha": 0.5, "seed": 3}, or {} for weights typed by hand. Portfolios are selected by
filtering portfolios_table (one tag_<key> column per tag). Like the name, the tags of a portfolio are written once and never silently change.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .virtual_runs import portfolio_id

PORTFOLIOS_DIR = "Portfolios"
PORTFOLIOS_FILE = "portfolios.jsonl"
PORTFOLIO_RUNS_FILE = "portfolio_runs.jsonl"
RUN_KINDS = ("reweighted", "trained")
_EQUAL_TOLERANCE = 1e-9
_TOP = 3


def _normalised(weights: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if not total > 0 or any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative with a positive sum.")
    return {ticker: float(w) / total for ticker, w in weights.items()}


def effective_number(weights: Mapping[str, float]) -> float:
    """1 / sum(w_i^2): how many equally weighted tickers the portfolio behaves like (n equal weights give n, one ticker gives 1)."""
    w = np.array(list(_normalised(weights).values()))
    return float(1.0 / np.sum(w ** 2))


def _percent(weight: float) -> str:
    value = weight * 100
    return f"{value:.0f}" if value >= 1 else f"{value:.1f}"


def auto_name(weights: Mapping[str, float], name_prefix: str | None = None) -> str:
    """
    A description of the weights, in this order: '<name_prefix> · ' when one is given (e.g. 'Dir0.5' for a Dirichlet draw), then 'equal <n>'
    for equal weights, or the three largest weights and how many tickers come after them ('AAPL35·MSFT20·NVDA15·+20 · neff 5.2', neff =
    effective_number, the number of equally weighted tickers it behaves like). The parts are separated by ' · ' (with spaces), never by
    ' | ', which separates the parts of a run's label (model | portfolio | train N f M).
    """
    normalised = _normalised(weights)
    prefix = "" if name_prefix is None else f"{name_prefix} · "
    values = np.array(list(normalised.values()))
    if np.allclose(values, values[0], rtol=0, atol=_EQUAL_TOLERANCE):
        return f"{prefix}equal {len(values)}"
    ranked = sorted(normalised.items(), key=lambda item: (-item[1], item[0]))
    top = "·".join(f"{ticker}{_percent(w)}" for ticker, w in ranked[:_TOP])
    rest = len(ranked) - _TOP
    return f"{prefix}{top}{f'·+{rest}' if rest > 0 else ''} · neff {effective_number(normalised):.1f}"


def _check_name(name: str) -> str:
    if not name or not name.strip() or name != name.strip() or any(c in name for c in "/\\\n\r\t"):
        raise ValueError(f"portfolio name {name!r} must not be empty, start or end with a space, or contain '/', '\\' or a line break.")
    return name


# ---- the files ----------------------------------------------------------------------------------------

def _file(root: Path, file_name: str) -> Path:
    return Path(root) / PORTFOLIOS_DIR / file_name


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_lines(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
        handle.flush()
        os.fsync(handle.fileno())


def _append_line(path: Path, record: dict) -> None:
    _append_lines(path, [record])


def get_portfolio(root: Path, portfolio: str) -> dict | None:
    """The record of a portfolio (None when it is not registered)."""
    for record in _read_lines(_file(root, PORTFOLIOS_FILE)):
        if record["portfolio_id"] == portfolio:
            return record
    return None


def register_portfolio(
    root: Path, weights: Mapping[str, float], name: str | None = None, tags: Mapping | None = None, name_prefix: str | None = None,
) -> dict:
    """
    Register a portfolio (or return the record it already has). name=None: auto_name(weights, name_prefix); when that description is
    already the name of ANOTHER portfolio (two sets of weights can round to the same words) the portfolio id is appended.
    tags: free provenance (see the module doc), JSON values only.
    Raises when the portfolio is already registered under another name or with other tags (tags=None does not check them), or when the name
    belongs to another portfolio.
    """
    normalised = _normalised(weights)
    portfolio = portfolio_id(normalised)
    tags = None if tags is None else json.loads(json.dumps(dict(tags)))        # JSON values only, a copy
    records = _read_lines(_file(root, PORTFOLIOS_FILE))
    taken = {record["name"]: record["portfolio_id"] for record in records}
    existing = next((record for record in records if record["portfolio_id"] == portfolio), None)

    if existing is not None:
        if name is not None and name != existing["name"]:
            raise ValueError(f"portfolio {portfolio} is already registered as {existing['name']!r}, not {name!r}.")
        if tags is not None and tags != existing.get("tags", {}):
            raise ValueError(f"portfolio {portfolio} is already registered with the tags {existing.get('tags', {})!r}, not {tags!r}.")
        return existing

    if name is None:
        name = auto_name(normalised, name_prefix)
        if name in taken:
            name = f"{name} · {portfolio}"
    _check_name(name)
    if name in taken:
        raise ValueError(f"the name {name!r} is already the name of portfolio {taken[name]}; pick another name.")
    record = {
        "portfolio_id": portfolio, "name": name, "tags": tags or {},
        "weights": normalised, "n_tickers": len(normalised), "n_eff": effective_number(normalised),
        "max_weight": max(normalised.values()), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_line(_file(root, PORTFOLIOS_FILE), record)
    return record


def portfolios_by_id(root: Path) -> dict[str, dict]:
    """Every registered portfolio's record by portfolio_id: the registry read once, for a caller that looks up many."""
    return {record["portfolio_id"]: record for record in _read_lines(_file(root, PORTFOLIOS_FILE))}


def link_run(root: Path, portfolio: str, run_id: str, kind: str) -> None:
    """Record that a run belongs to a portfolio (kind: 'reweighted' for a virtual run, 'trained' for a run trained for these weights). Once is enough."""
    link_runs(root, [(portfolio, run_id, kind)])


def link_runs(root: Path, links: list[tuple[str, str, str]]) -> None:
    """link_run for many (portfolio_id, run_id, kind) at once: the registry and the links are read once, only the missing links are appended."""
    known = portfolios_by_id(root)
    present = {(r["portfolio_id"], r["run_id"]) for r in _read_lines(_file(root, PORTFOLIO_RUNS_FILE))}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for portfolio, run_id, kind in links:
        if kind not in RUN_KINDS:
            raise ValueError(f"kind must be one of {RUN_KINDS}, got {kind!r}.")
        if portfolio not in known:
            raise KeyError(f"portfolio {portfolio} is not registered.")
    new = []
    for portfolio, run_id, kind in links:
        if (portfolio, run_id) in present:
            continue
        present.add((portfolio, run_id))
        new.append({"portfolio_id": portfolio, "run_id": run_id, "kind": kind, "created_at": stamp})
    _append_lines(_file(root, PORTFOLIO_RUNS_FILE), new)


def portfolios_table(root: Path) -> pd.DataFrame:
    """
    One row per portfolio (index portfolio_id): name, n_tickers, n_eff, max_weight, created_at, one tag_<key> column per tag (a portfolio
    without that tag has NaN) and one w_<ticker> column per ticker. Select portfolios by filtering it.
    """
    records = _read_lines(_file(root, PORTFOLIOS_FILE))
    columns = ["name", "n_tickers", "n_eff", "max_weight", "created_at"]
    if not records:
        return pd.DataFrame(columns=columns).rename_axis("portfolio_id")
    rows = {
        r["portfolio_id"]: {
            **{c: r[c] for c in columns}, **{f"tag_{k}": v for k, v in r.get("tags", {}).items()},
            **{f"w_{t}": w for t, w in r["weights"].items()},
        }
        for r in records
    }
    return pd.DataFrame(rows).T.rename_axis("portfolio_id").astype({"n_tickers": int, "n_eff": float, "max_weight": float})


def portfolio_runs_table(root: Path, portfolio: str | None = None) -> pd.DataFrame:
    """Columns portfolio_id, run_id, kind, created_at: every run of every portfolio, or of one."""
    frame = pd.DataFrame(_read_lines(_file(root, PORTFOLIO_RUNS_FILE)), columns=["portfolio_id", "run_id", "kind", "created_at"])
    return frame if portfolio is None else frame[frame["portfolio_id"] == portfolio].reset_index(drop=True)
