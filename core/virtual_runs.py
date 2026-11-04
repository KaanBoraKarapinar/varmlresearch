"""
Where the portfolio readings of a walk-forward are kept: <walk_forward folder>/VirtualRuns/.

A virtual run is another run's per-ticker forecasts read for other portfolio weights (walk_forward.ReweightedVirtualRun), or a baseline
(GARCH, historical) computed for those weights. Nothing about the models is stored here, only what is needed to read the portfolio again with any VaR
reading: its combined residuals (the five RESIDUAL_COLS + window_id + part), as a partitioned parquet dataset:

    VirtualRuns/
        residuals/model=<run_name>/part-<n>.parquet     one row per (portfolio_id, Date); every portfolio of the model is in the same files
        BaselineModelDefs/<run_name>.json               the recipe a baseline name (garch_wide, historical) stands for
        _ledger.jsonl                                   one JSON line per file written (what is in it, when, by which batch), with the
                                                        definition of every run in it (run_id, kind, weights, config)

A parquet file is never changed once written: a portfolio added later goes into a NEW part-<n>.parquet (the next n), so a crash never damages what
is already there. Inside a file the rows are sorted by (portfolio_id, Date) and every portfolio is its own row group, so reading one portfolio
(a filter on portfolio_id) reads only its row groups. Pure functions on paths and frames: no dashboard, no training. `root` is always given (the
store root), there is no default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .ResidualBank import RESIDUAL_COLS

VIRTUAL_RUNS_DIR = "VirtualRuns"
MODEL_DEFINITIONS_DIR = "BaselineModelDefs"
LEDGER_FILE = "_ledger.jsonl"
RESIDUALS_DATASET = "residuals"
RESIDUAL_FILE_COLUMNS = ["portfolio_id", "Date", *RESIDUAL_COLS, "window_id", "part"]
_PART_FILE = re.compile(r"part-(\d+)\.parquet")


def portfolio_id(weights: Mapping[str, float]) -> str:
    """
    The identity of a portfolio: a hash of its normalised weights rounded to 12 digits, so the same weights give the same id however they were
    written (the same rule as the '@<digest>' of a ReweightedVirtualRun).
    """
    total = float(sum(weights.values()))
    if not total > 0 or any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative with a positive sum.")
    normalised = {ticker: round(float(w) / total, 12) for ticker, w in weights.items()}
    return hashlib.sha256(json.dumps(normalised, sort_keys=True).encode()).hexdigest()[:8]


def virtual_runs_dir(root: Path, walk_forward_id: str) -> Path:
    return Path(root) / "WalkForwardSetting" / walk_forward_id / VIRTUAL_RUNS_DIR


def _check_model_name(model: str) -> str:
    if not model or any(c in model for c in "/\\=") or model.startswith("."):
        raise ValueError(f"model (the run name) {model!r} must not be empty, start with '.' or contain '/', '\\' or '='.")
    return model


def _model_dir(root: Path, walk_forward_id: str, dataset: str, model: str) -> Path:
    return virtual_runs_dir(root, walk_forward_id) / dataset / f"model={_check_model_name(model)}"


# ---- the ledger ---------------------------------------------------------------------------------------

def read_ledger(root: Path, walk_forward_id: str) -> list[dict]:
    """Every event of the walk-forward's VirtualRuns, oldest first ([] when nothing was written yet)."""
    path = virtual_runs_dir(root, walk_forward_id) / LEDGER_FILE
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_ledger(root: Path, walk_forward_id: str, event: dict) -> None:
    path = virtual_runs_dir(root, walk_forward_id) / LEDGER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def portfolio_ids(root: Path, walk_forward_id: str, model: str, dataset: str = RESIDUALS_DATASET) -> list[str]:
    """The portfolios already written for this model, from the ledger (files marked superseded are not counted)."""
    found: list[str] = []
    for event in read_ledger(root, walk_forward_id):
        if event.get("event") == "write" and event["dataset"] == dataset and event["model"] == model and event.get("status", "active") == "active":
            found.extend(event["portfolio_ids"])
    return sorted(set(found))


# ---- the residuals ------------------------------------------------------------------------------------

def _next_part_number(folder: Path) -> int:
    numbers = [int(m.group(1)) for m in (_PART_FILE.fullmatch(p.name) for p in folder.glob("part-*.parquet")) if m]
    return max(numbers) + 1 if numbers else 0


def _residual_table(portfolio: str, frame: pd.DataFrame) -> pa.Table:
    missing = [c for c in [*RESIDUAL_COLS, "window_id", "part"] if c not in frame.columns]
    if missing:
        raise ValueError(f"portfolio {portfolio}: the residual frame misses columns {missing}.")
    if not (frame.index.is_monotonic_increasing and frame.index.is_unique):
        raise ValueError(f"portfolio {portfolio}: the residual frame is not chronological without duplicate dates.")
    out = pd.DataFrame({"portfolio_id": portfolio, "Date": pd.DatetimeIndex(frame.index).astype("datetime64[us]")})
    for column in RESIDUAL_COLS:
        out[column] = frame[column].to_numpy(dtype=float)
    out["window_id"] = frame["window_id"].to_numpy(dtype="int64")
    out["part"] = frame["part"].astype(str).to_numpy()
    return pa.Table.from_pandas(out[RESIDUAL_FILE_COLUMNS], preserve_index=False)


def save_residuals(
    root: Path, walk_forward_id: str, model: str, frames: Mapping[str, pd.DataFrame], batch: Mapping | None = None,
    definitions: Mapping[str, Mapping] | None = None,
) -> Path:
    """
    Write the combined residuals of portfolios for one model as ONE new part-<n>.parquet of VirtualRuns/residuals/model=<model>/.
    frames: {portfolio_id: frame with RESIDUAL_COLS + window_id + part, indexed by Date (what Run.combined_residuals gives)}.
    definitions: {portfolio_id: what rebuilds the run from its residuals (the weights, the run it was read from ...)}, kept in the ledger
    line and read back by read_definitions.
    batch: free provenance (what produced these files: batch id, recipe key, seed, code version ...), copied into the ledger line.
    A portfolio that is already written for this model raises (the same weights are never kept twice).
    """
    if not frames:
        raise ValueError("frames is empty: nothing to write.")
    if definitions is not None and set(definitions) != set(frames):
        raise ValueError(f"definitions must be given for exactly the portfolios of frames: {sorted(set(definitions) ^ set(frames))} differ.")
    already = set(portfolio_ids(root, walk_forward_id, model))
    duplicated = sorted(already & set(frames))
    if duplicated:
        raise ValueError(f"model {model!r}: portfolios {duplicated} are already written in {walk_forward_id}.")

    folder = _model_dir(root, walk_forward_id, RESIDUALS_DATASET, model)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"part-{_next_part_number(folder)}.parquet"
    partial = final.with_name(final.name + ".partial")
    ids = sorted(frames)
    n_rows = 0
    with pq.ParquetWriter(partial, _residual_table(ids[0], frames[ids[0]]).schema, compression="zstd") as writer:
        for portfolio in ids:                    # one write_table per portfolio: its own row group, so a filter reads only its rows
            table = _residual_table(portfolio, frames[portfolio])
            writer.write_table(table)
            n_rows += table.num_rows
    os.replace(partial, final)                   # all or nothing: a half-written file never carries the name part-<n>.parquet

    _append_ledger(root, walk_forward_id, {
        "event": "write", "status": "active", "dataset": RESIDUALS_DATASET, "model": model,
        "path": final.relative_to(virtual_runs_dir(root, walk_forward_id)).as_posix(), "n_rows": n_rows,
        "portfolio_ids": ids, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "write_id": uuid.uuid4().hex[:12], **({"batch": dict(batch)} if batch else {}),
        **({"definitions": {pid: dict(definitions[pid]) for pid in ids}} if definitions else {}),
    })
    return final


def read_definitions(root: Path, walk_forward_id: str, model: str) -> dict[str, dict]:
    """{portfolio_id: definition} of the active residual writes of this model (what save_residuals was given as definitions)."""
    found: dict[str, dict] = {}
    for event in read_ledger(root, walk_forward_id):
        if event.get("event") == "write" and event["dataset"] == RESIDUALS_DATASET and event["model"] == model and event.get("status", "active") == "active":
            found.update(event.get("definitions", {}))
    return found


def list_definitions(root: Path, walk_forward_id: str) -> list[dict]:
    """Every saved run of the walk-forward, oldest first: its definition + portfolio_id + model (the run name) + created_at."""
    found: dict[tuple[str, str], dict] = {}
    for event in read_ledger(root, walk_forward_id):
        if event.get("event") == "write" and event["dataset"] == RESIDUALS_DATASET and event.get("status", "active") == "active":
            for portfolio, definition in event.get("definitions", {}).items():
                found[(event["model"], portfolio)] = {
                    **definition, "portfolio_id": portfolio, "model": event["model"], "created_at": event["created_at"],
                }
    return list(found.values())


def find_definition(root: Path, walk_forward_id: str, run_id: str) -> dict:
    """The definition (see list_definitions) of the saved run with this run_id; KeyError when there is none."""
    for definition in list_definitions(root, walk_forward_id):
        if definition.get("run_id") == run_id:
            return definition
    raise KeyError(f"no saved virtual run {run_id} in {walk_forward_id}.")


def walk_forward_ids_with_virtual_runs(root: Path) -> list[str]:
    """The walk-forwards of the store that have a VirtualRuns ledger."""
    return sorted(p.parent.parent.name for p in (Path(root) / "WalkForwardSetting").glob(f"*/{VIRTUAL_RUNS_DIR}/{LEDGER_FILE}"))


def ensure_model_definition(root: Path, walk_forward_id: str, run_name: str, definition: Mapping) -> Path:
    """
    VirtualRuns/BaselineModelDefs/<run_name>.json: the recipe a baseline run name stands for (written once). The name is locked to its
    recipe_key, like a run name of a trained run: the same name with another recipe raises instead of silently changing meaning.
    """
    path = virtual_runs_dir(root, walk_forward_id) / MODEL_DEFINITIONS_DIR / f"{_check_model_name(run_name)}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["recipe_key"] != definition["recipe_key"]:
            raise ValueError(f"the baseline name {run_name!r} already stands for the recipe {stored['recipe_key']}, not {definition['recipe_key']}; pick another name.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(dict(definition), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)
    return path


def read_residuals(root: Path, walk_forward_id: str, model: str, portfolio: str) -> pd.DataFrame:
    """
    The combined residuals of one portfolio of one model, as Run.combined_residuals gives them: RESIDUAL_COLS + window_id + part,
    indexed by Date in time order. KeyError when the portfolio is not written for the model.
    """
    folder = _model_dir(root, walk_forward_id, RESIDUALS_DATASET, model)
    files = sorted(folder.glob("part-*.parquet"), key=lambda p: int(_PART_FILE.fullmatch(p.name).group(1))) if folder.exists() else []
    if not files:
        raise KeyError(f"no residuals of model {model!r} in {walk_forward_id}.")
    tables = [pq.read_table(f, filters=[("portfolio_id", "=", portfolio)]) for f in files]
    table = pa.concat_tables([t for t in tables if t.num_rows])  if any(t.num_rows for t in tables) else None
    if table is None:
        raise KeyError(f"portfolio {portfolio} is not written for model {model!r} in {walk_forward_id}.")
    frame = table.to_pandas().drop(columns="portfolio_id").set_index("Date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame
