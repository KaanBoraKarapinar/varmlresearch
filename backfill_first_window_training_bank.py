"""
Writes WALK_FORWARD_FIRST_WINDOW_TRAINING_BANK_FOR_RECALCULATING_RESIDUALS.parquet into the saved runs that were stored before the
file existed (new runs get it from run_store.save_walk_forward_run).

    python backfill_first_window_training_bank.py            all runs with an ewma bank and without the file
    python backfill_first_window_training_bank.py --dry-run  only list what would be done

The training days of the first window come from the run's raw_ewma_banks/ when it has them. A slim run has none, so the first window
alone is filled again with the run's own saved model (wf.fit_window; nothing is trained), and its forecast rows are checked against the
stored ewma bank before anything is written.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import numpy as np
import pandas as pd

import run_store as rs
import train_23_experiment as e
import walk_forward as wf

TOLERANCE = 1e-9


def from_raw_banks(run: wf.LoadedRun) -> pd.DataFrame:
    raw = run.raw_ewma_bank(0)
    forecast_dates = run.ewma_bank.index[run.ewma_bank["window_id"] == 0]
    training = raw.loc[raw.index.difference(forecast_dates)].copy()
    stored = run.ewma_bank.loc[forecast_dates]
    if not np.allclose(raw.loc[forecast_dates][stored.drop(columns="window_id").columns].to_numpy(), stored.drop(columns="window_id").to_numpy(), rtol=TOLERANCE, equal_nan=True):
        raise ValueError("the raw window-0 bank and the combined ewma bank disagree on the forecast days.")
    training["window_id"] = 0
    return training


def from_first_window_fit(run: wf.LoadedRun) -> pd.DataFrame:
    setting = rs.load_walk_forward(run.walk_forward_id, e.prices)
    calendar = setting.make_calendars()[0]
    recipe = run.recipe
    if recipe.model_family == "xgboost":
        fit = wf.fit_window(0, calendar, recipe, existing_model=run.model(0), trained_columns=run.feature_columns())
    else:
        fit = wf.fit_window(0, calendar, recipe)
    rebuilt = wf.ewma_bank_to_frame(fit.ewma_bank, calendar.forecast_dates)
    stored = run.ewma_bank[run.ewma_bank["window_id"] == 0].drop(columns="window_id")
    if not (rebuilt.index.equals(stored.index) and np.allclose(rebuilt[stored.columns].to_numpy(), stored.to_numpy(), rtol=TOLERANCE, equal_nan=True)):
        raise ValueError("the rebuilt first window does not reproduce the stored forecast rows of window 0.")
    return wf.build_first_window_training_bank([fit])


def main() -> None:
    dry = "--dry-run" in sys.argv
    runs = rs.load_runs()
    todo = [r for r in runs if (r.has_ewma_bank and not r.has_first_window_training_bank]
    print(f"{len(runs)} runs, {len(todo)} to do", flush=True)
    failed = []
    for run in todo:
        source = "raw_ewma_banks" if (run.run_dir / "raw_ewma_banks").exists() else "first window refilled"
        print(f"{run.display_name[:70]:70s} {source}", flush=True)
        if dry:
            continue
        try:
            frame = from_raw_banks(run) if source == "raw_ewma_banks" else from_first_window_fit(run)
            rs._to_parquet(frame, run.run_dir / wf.FIRST_WINDOW_TRAINING_BANK_FILE, rs._parquet_metadata(run.config))
        except Exception as error:      # reported at the end, the other runs go on
            failed.append((run.display_name, repr(error)))
    print(f"done, {len(failed)} failed")
    for name, error in failed:
        print("FAILED", name, error)


if __name__ == "__main__":
    main()
