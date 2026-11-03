"""
Checks the three kinds of run (walk_forward.Run) against each other:

    python tools/verify_run_classes.py

1. LoadedRun: the reading computed now (Run._compute_reading) equals the reading the saved run folder holds, for every saved run that has
   the default reading on disk (nothing is written).
2. TrainedInitialRun: a training that has not been saved reads like the same training saved and loaded.
3. ReweightedVirtualRun: a run read for other weights equals the same model trained directly with those weights (GARCH, long format:
   its per-ticker forecasts do not depend on the weights), in the VaR frame, the daily frame, the ewma bank and the forecast QLIKE.
"""
import contextlib
import dataclasses
import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from core import run_store as rs
from core import walk_forward as wf
from training import train_23_experiment as e

DEFAULT_READING = wf.Reading()
TOLERANCE = 1e-12
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        return False, f"(index or columns differ: {len(a)} vs {len(b)} rows)"
    worst = 0.0
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]) and not pd.api.types.is_bool_dtype(a[column]):
            diff = (a[column].astype(float) - b[column].astype(float)).abs()
            worst = max(worst, float(diff.max(skipna=True)) if diff.notna().any() else 0.0)
            if (a[column].isna() != b[column].isna()).any():
                return False, f"(NaN differ in {column})"
        elif not a[column].equals(b[column]):
            return False, f"({column} differs)"
    return worst <= TOLERANCE, f"(max difference {worst:.1e})"


# ---- 1. every saved run: computed now == on disk -------------------------------------------------------------------
loaded = [r for r in rs.load_runs() if (r.run_dir / "readings_from_walk_forward" / DEFAULT_READING.test_path / "test_results.json").exists()]
print(f"{len(loaded)} saved runs have the default reading on disk")
bad = 0
for run in loaded:
    now = run._compute_reading(DEFAULT_READING)
    parts = {
        "var_df": (now.var_df, run.var_df(DEFAULT_READING)), "daily": (now.daily, run.daily(DEFAULT_READING)),
        "exceeding_days": (now.exceeding_days, run.exceeding_days(DEFAULT_READING)), "blocks": (now.blocks, run.blocks(DEFAULT_READING)),
    }
    for part, (computed, stored) in parts.items():
        ok, detail = frames_equal(computed, stored)
        if not ok:
            bad += 1
            check(f"{run.display_name} {part}", ok, detail)
    if now.metrics != run.metrics(DEFAULT_READING):
        bad += 1
        check(f"{run.display_name} metrics", False)
check(f"all {len(loaded)} saved runs: the reading computed now is the reading on disk", bad == 0)

# ---- 2 and 3: a small walk-forward, GARCH (long) ------------------------------------------------------------------
TICKERS = e.TICKERS_4
RECIPE = wf.ModelRecipe("long", model_family="garch", model_kwargs=dict(mean="Zero", vol="GARCH", p=1, q=1, dist="normal"))
WEIGHTS_A = {t: 0.25 for t in TICKERS}
WEIGHTS_B = {"AAPL": 0.5, "JPM": 0.1, "XOM": 0.3, "KO": 0.1}
READINGS = [wf.Reading(0.95, 252, 30, 0.95), wf.Reading(0.99, 500, 60, 0.95)]
root = Path(tempfile.mkdtemp())


def train(weights, name):
    setting = dataclasses.replace(e.setting_of(TICKERS), weights=weights)
    with contextlib.redirect_stdout(io.StringIO()):
        fits = wf.train_walk_forward(setting.make_calendars(), RECIPE)
    return setting, fits


setting_a, fits_a = train(WEIGHTS_A, "a")
setting_b, fits_b = train(WEIGHTS_B, "b")
trained_a = rs.trained_initial_run(setting_a, RECIPE, fits_a, "a")
trained_b = rs.trained_initial_run(setting_b, RECIPE, fits_b, "b")
with contextlib.redirect_stdout(io.StringIO()):
    saved_a = wf.LoadedRun(rs.save_walk_forward_run(setting_a, RECIPE, fits_a, "a", root=root, slim=True))

for reading in READINGS:
    label = f"c{reading.var_confidence:g} w{reading.window_size}"
    for part in ("var_df", "daily", "exceeding_days", "blocks"):
        ok, detail = frames_equal(getattr(trained_a, part)(reading), getattr(saved_a, part)(reading))
        check(f"TrainedInitialRun == saved LoadedRun, {part}, {label}", ok, detail)
    check(f"TrainedInitialRun == saved LoadedRun, metrics, {label}", trained_a.metrics(reading) == saved_a.metrics(reading))

check("TrainedInitialRun == saved LoadedRun, ewma_bank", frames_equal(trained_a.ewma_bank, saved_a.ewma_bank)[0])
check("TrainedInitialRun == saved LoadedRun, first_window_training_bank", frames_equal(trained_a.first_window_training_bank, saved_a.first_window_training_bank)[0])
check("TrainedInitialRun == saved LoadedRun, window_losses", frames_equal(trained_a.window_losses.astype(float), saved_a.window_losses.astype(float))[0])

for source, label in ((trained_a, "TrainedInitialRun"), (saved_a, "LoadedRun")):
    virtual = wf.ReweightedVirtualRun(source, WEIGHTS_B)
    for reading in READINGS:
        for part in ("var_df", "daily"):
            ok, detail = frames_equal(getattr(virtual, part)(reading), getattr(trained_b, part)(reading))
            check(f"ReweightedVirtualRun({label}) == trained with the weights, {part}, c{reading.var_confidence:g}", ok, detail)
    ok, detail = frames_equal(virtual.ewma_bank, trained_b.ewma_bank)
    check(f"ReweightedVirtualRun({label}) == trained with the weights, ewma_bank (portfolio variance included)", ok, detail)
    ok, detail = frames_equal(virtual.combined_residuals, trained_b.combined_residuals)
    check(f"ReweightedVirtualRun({label}) == trained with the weights, combined_residuals", ok, detail)
    forecast_a, forecast_b = virtual.window_losses["forecast_qlike"].astype(float), trained_b.window_losses["forecast_qlike"].astype(float)
    check(f"ReweightedVirtualRun({label}) == trained with the weights, forecast QLIKE of every window", bool(np.allclose(forecast_a, forecast_b, rtol=1e-9)), f"(max {float((forecast_a - forecast_b).abs().max()):.1e})")
    check(f"ReweightedVirtualRun({label}): the config carries the new weights", virtual.config["setting"]["weights"] == {t: w for t, w in zip(virtual.tickers, virtual.weights)} and virtual.config["setting"]["weights"] != saved_a.config["setting"]["weights"])

# the guards
for bad_weights, why in (({"AAPL": 1.0}, "missing tickers"), ({**WEIGHTS_B, "KO": -0.1}, "a negative weight")):
    try:
        wf.ReweightedVirtualRun(saved_a, bad_weights)
        check(f"ReweightedVirtualRun raises for {why}", False)
    except ValueError:
        check(f"ReweightedVirtualRun raises for {why}", True)
try:
    wf.Run()
    check("Run cannot be created itself", False)
except TypeError:
    check("Run cannot be created itself (abstract)", True)

print("\nFAILED:" if failures else "\nall checks passed", *failures, sep="\n  " if failures else "")
raise SystemExit(1 if failures else 0)
