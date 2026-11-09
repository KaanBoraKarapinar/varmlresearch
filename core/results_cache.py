"""
The results cache: the few files of the run store that the Results and Experiments pages need to show the saved experiments at their reading, in a folder of the
repository (results_cache/) that has the layout of the store, so the same code reads it:

    results_cache/store/            experiments/*.json, WalkForwardSetting/<walk_forward>/... (the runs of the experiments), Portfolios/portfolios.jsonl
    results_cache/mlruns.db         the MLflow index with only those runs

What a run keeps here is what the pages read at a reading: model_settings.json, the combined residual bank, window_losses.parquet, the reading's var_dataframe and its
test folder, the feature importance. What it leaves out is what only the other views read and what is big: the EWMA banks and the first-window banks (the per ticker
variance, the weights pages), the raw banks, the models, the features and the readings of other VaR settings (they are computed from the residual bank when asked
for). A virtual run (a run read for another portfolio's weights) keeps only the rows of its portfolio in its residuals files and only its events in the ledger.
build() writes it from the store, so it can be made again after a new run. Pure functions on paths: no Streamlit.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import core
from core import portfolios as pf
from core import run_store as rs
from core import virtual_runs as vr
from core import walk_forward as wf

CACHE_DIR = "results_cache"
CACHED_READING = wf.Reading()           # the VaR reading the cache holds for every run (the dashboard's own starting reading): 95 %, 252-day window, 30 min periods, test at 95 %
LEFT_OUT_FILES = {wf.EWMA_BANK_FILE, wf.FIRST_WINDOW_TRAINING_BANK_FILE}
LEFT_OUT_DIRS = {"raw_residual_banks", "raw_ewma_banks", "features"}
KEPT_IN_MODELS = {"feature_importance.parquet", "garch_params.json"}       # the other files of models/ are the trained models
ZSTD_LEVEL = 19


@dataclass
class CacheReport:
    experiments: list[str] = field(default_factory=list)
    runs: int = 0
    virtual_runs: int = 0
    files: int = 0
    bytes: int = 0
    run_ids: set[str] = field(default_factory=set)       # the (non virtual) runs that are in the cache, for the MLflow index


def _wanted(relative: Path, readings: set[wf.Reading]) -> bool:
    """Does the file of a run folder (path relative to the run folder) belong in the cache?"""
    if relative.name in LEFT_OUT_FILES or relative.parts[0] in LEFT_OUT_DIRS:
        return False
    if relative.parts[0] == "models":
        return relative.name in KEPT_IN_MODELS
    if relative.parts[0] == "readings_from_walk_forward":
        inside = relative.relative_to("readings_from_walk_forward")
        for reading in readings:
            if inside.parent == reading.var_path or reading.test_path in (inside.parent, *inside.parent.parents):
                return True
        return False
    return True


def _copy(source: Path, target: Path, report: CacheReport, dry_run: bool) -> None:
    report.files += 1
    report.bytes += source.stat().st_size
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _copy_run(store: Path, out: Path, run_id: str, readings: set[wf.Reading], report: CacheReport, dry_run: bool) -> None:
    walk_forward_id, feature_target_key, model_dir = run_id.split("/")
    run_dir = rs._run_dir(walk_forward_id, feature_target_key, model_dir, store)
    if not (run_dir / "model_settings.json").exists():
        raise FileNotFoundError(f"The run {run_id} is not in the store {store}.")
    for source in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = source.relative_to(run_dir)
        if _wanted(relative, readings):
            _copy(source, out / source.relative_to(store), report, dry_run)
    feature_target = run_dir.parents[1] / "FeatureAndTarget.json"
    if feature_target.exists():
        _copy(feature_target, out / feature_target.relative_to(store), report, dry_run)
    report.run_ids.add(run_id)


def _copy_walk_forward(store: Path, out: Path, walk_forward_id: str, report: CacheReport, dry_run: bool) -> None:
    folder = rs.walk_forward_dir(walk_forward_id, store)
    for name in ("WalkForwardSettings.json", "WindowsCreatedCalendars.parquet"):
        _copy(folder / name, out / (folder / name).relative_to(store), report, dry_run)


def _filtered_part(source: Path, target: Path, portfolios: set[str], report: CacheReport, dry_run: bool) -> int:
    """The rows of these portfolios of a residuals file, as a parquet file of their own (every portfolio its own row group, sorted, like the original)."""
    table = pq.read_table(source, filters=[("portfolio_id", "in", sorted(portfolios))])
    if not table.num_rows:
        return 0
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd", compression_level=ZSTD_LEVEL)
    report.files += 1
    report.bytes += (target.stat().st_size if not dry_run else table.nbytes)
    return table.num_rows


def _copy_virtual_runs(store: Path, out: Path, virtual_ids: set[str], report: CacheReport, dry_run: bool) -> set[str]:
    """
    The residuals and the ledger of the virtual runs: for every walk-forward, the ledger events of the wanted (model, portfolio) pairs with only those portfolios in
    them, and their residuals files with only those portfolios' rows. Returns the base runs the virtual runs read (they have to be in the cache as well).
    """
    bases: set[str] = set()
    by_walk_forward: dict[str, set[str]] = {}
    for run_id in virtual_ids:
        by_walk_forward.setdefault(run_id.split("/")[0], set()).add(run_id)
    portfolios_needed: set[str] = set()
    for walk_forward_id, run_ids in by_walk_forward.items():
        wanted = {}                                     # (model, portfolio) -> definition
        for definition in vr.list_definitions(store, walk_forward_id):
            if definition["run_id"] in run_ids:
                wanted[(definition["model"], definition["portfolio_id"])] = definition
                if definition.get("base_run_id"):
                    bases.add(definition["base_run_id"])
        missing = run_ids - {d["run_id"] for d in wanted.values()}
        if missing:
            raise KeyError(f"No saved virtual run {sorted(missing)[0]} in {walk_forward_id}.")
        events = []
        for event in vr.read_ledger(store, walk_forward_id):
            if event.get("event") != "write" or event["dataset"] != vr.RESIDUALS_DATASET or event.get("status", "active") != "active":
                continue
            keep = {p for p in event["portfolio_ids"] if (event["model"], p) in wanted}
            if not keep:
                continue
            source = vr.virtual_runs_dir(store, walk_forward_id) / event["path"]
            rows = _filtered_part(source, vr.virtual_runs_dir(out / "store", walk_forward_id) / event["path"], keep, report, dry_run)
            events.append({**event, "portfolio_ids": sorted(keep), "n_rows": rows, "definitions": {p: d for p, d in event["definitions"].items() if p in keep}})
            portfolios_needed |= keep
        if not dry_run:
            ledger = vr.virtual_runs_dir(out / "store", walk_forward_id) / vr.LEDGER_FILE
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
        definitions_dir = vr.virtual_runs_dir(store, walk_forward_id) / vr.MODEL_DEFINITIONS_DIR
        if definitions_dir.exists():
            for source in sorted(definitions_dir.glob("*.json")):
                _copy(source, vr.virtual_runs_dir(out / "store", walk_forward_id) / vr.MODEL_DEFINITIONS_DIR / source.name, report, dry_run)
    if portfolios_needed and not dry_run:
        records = [r for r in pf._read_lines(pf._file(store, pf.PORTFOLIOS_FILE)) if r["portfolio_id"] in portfolios_needed]
        target = pf._file(out / "store", pf.PORTFOLIOS_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    return bases


def build(store: Path, out: Path, readings: set[wf.Reading], *, dry_run: bool = False) -> CacheReport:
    """
    Write the cache of the saved experiments of `store` into `out` (a new folder: results_cache/). readings: the VaR readings that are kept besides those of the
    experiments themselves. The MLflow index is made apart (trim_index), because it is a copy of a database, not of files.
    """
    store, out = Path(store), Path(out)
    if out.exists() and not dry_run:
        raise FileExistsError(f"{out} exists: remove it first (it is made from the store, nothing in it is the only copy).")
    experiments = core.list_saved_experiments(store)
    if not experiments:
        raise FileNotFoundError(f"No saved experiment in {store / 'experiments'}.")
    report = CacheReport(experiments=[e.slug for e in experiments])
    readings = set(readings) | {wf.Reading(**e.reading) for e in experiments}
    ids = {i for e in experiments for i in (*e.run_ids, e.baseline_run_id)}
    virtual_ids = {i for i in ids if "@" in i}
    plain_ids = ids - virtual_ids
    plain_ids |= _copy_virtual_runs(store, out, virtual_ids, report, dry_run)
    for run_id in sorted(plain_ids):
        _copy_run(store, out / "store", run_id, readings, report, dry_run)
    for walk_forward_id in sorted({i.split("/")[0] for i in plain_ids | virtual_ids}):
        _copy_walk_forward(store, out / "store", walk_forward_id, report, dry_run)
    for experiment in experiments:
        source = core.experiment.experiments_dir(store) / f"{experiment.slug}.json"
        _copy(source, core.experiment.experiments_dir(out / "store") / source.name, report, dry_run)
    report.runs, report.virtual_runs = len(plain_ids), len(virtual_ids)
    return report


def trim_index(index_copy: Path, keep_run_ids: set[str], experiment: str) -> tuple[int, int]:
    """
    In a COPY of mlruns.db, delete (permanently) the MLflow runs whose store run id is not in keep_run_ids. Returns (kept, deleted). The runs listed in a folder
    (the virtual runs) are not in MLflow.
    """
    import subprocess
    import sys

    import mlflow
    uri = f"sqlite:///{Path(index_copy).resolve()}"
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[experiment])
    keep = runs["tags.run_id"].isin(keep_run_ids)
    for mlflow_run_id in runs.loc[~keep, "run_id"]:
        mlflow.delete_run(mlflow_run_id)
    subprocess.run([sys.executable, "-m", "mlflow", "gc", "--backend-store-uri", uri], check=True, capture_output=True)       # delete them for good
    return int(keep.sum()), int((~keep).sum())
