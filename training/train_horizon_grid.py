"""
The training window x 1-day forecast grid: the same models with train windows of 252, 504, 756 and 1008 days, refitted every day.

    python training/train_horizon_grid.py --workers 8                         # all four windows, all models
    python training/train_horizon_grid.py --dry-run                           # what would be trained, nothing is
    python training/train_horizon_grid.py --trains 756 --only xgb_F3_drv --workers 8
    python training/train_horizon_grid.py --root /mnt/results/store --no-mlflow --workers 14       # on a server: its own store, no MLflow
    python training/train_horizon_grid.py --log-only                          # index the runs already in the store in MLflow (after copying a server's store here)

Every model is trained on the 23 tickers with equal weights, forecast 1 day (the model is refitted every day), from 2016-03-01, so each train window starts forecasting on its
own first day: 252 on 2017-03-01 (2346 days), 504 on 2018-03-01 (2094), 756 on 2019-03-04 (1842), 1008 on 2020-03-03 (1590); the forecasts of different train windows are
compared on the days they share. The RV data starts on 2016-01-04, which is why a 1008-day window cannot forecast before 2020-03.

The models: the five of the dashboard's best group and their relatives (xgb_F1_drv, xgb_F3_drv, xgb_F3_rv, xgb_F3_ratio: the recipes of train_feature_target_grid.py), garch_wide,
ewma and historical. Windows are independent, so --workers processes fit them at the same time (core.run_store.train_and_save_walk_forward_run, workers=): the runs are saved
slim and window by window reduced, so a run of 2000 windows needs hundreds of MB, not tens of GB. A saved run is skipped: run the script again after an interruption.
About 6 minutes of one core per XGBoost run (measured), so with 8 workers about 1 to 2 minutes a run including the saving.
"""
import argparse
import dataclasses
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core import run_store as rs
from core import walk_forward as wf
from training import process_portfolios as pp
from training import train_23_experiment as e
from training import train_feature_target_grid as grid

TRAINS = (252, 504, 756, 1008)
FORECAST_BARS = 1
START_DATE = "2016-03-01"
XGBOOST_MODELS = ("xgb_F1_drv", "xgb_F3_drv", "xgb_F3_rv", "xgb_F3_ratio")
BASELINES = ("garch_wide", "ewma", "historical")
RECIPES = {**{name: grid.RECIPES[name] for name in XGBOOST_MODELS}, **{name: wf.BASELINE_RECIPES[name] for name in BASELINES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trains", nargs="+", type=int, choices=TRAINS, default=list(TRAINS), help="the train windows (days)")
    parser.add_argument("--forecast", type=int, default=FORECAST_BARS, help="days between two fits (the forecast days of a window): 1 refits every day, 21 about every month")
    parser.add_argument("--only", nargs="+", choices=list(RECIPES), help="only these models")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="processes that fit windows at the same time (default: the cores minus 2)")
    parser.add_argument("--root", type=Path, default=rs.STORE_ROOT, help="the run store to save in (default: store/)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end-date", help="the last forecast day (for a trial: the walk-forward is another one, use a throw-away --root)")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log the runs to MLflow (on a server)")
    parser.add_argument("--log-only", action="store_true", help="only log the runs already in the store to MLflow, train nothing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = args.only or list(RECIPES)
    plan = []
    for train in args.trains:
        setting = pp.base_setting(train, args.forecast, args.start)
        if args.end_date:
            setting = dataclasses.replace(setting, end_date=args.end_date)
        table = setting.window_table()
        print(f"train {train} / forecast {args.forecast}: {len(table)} windows, forecast {table['forecast_start'].iloc[0].date()} .. {table['forecast_end'].iloc[-1].date()}\n  {setting.walk_forward_id}")
        plan.append((train, setting))
    print(f"models: {names}; {args.workers} workers; store {args.root}")
    if args.dry_run:
        return

    saved = []
    started = time.time()
    jobs = [(train, setting, name) for train, setting in plan for name in names]
    seconds_per_window: dict[bool, list[float]] = {True: [], False: []}      # of the finished runs, by XGBoost / not: the basis of the estimate of what is left
    for number, (train, setting, name) in enumerate(jobs, start=1):
        t = time.time()
        if args.log_only:
            run_id = rs._run_id(setting.walk_forward_id, RECIPES[name])
            saved.append(rs.load_walk_forward_run(run_id, args.root))
            continue
        n_windows = len(setting.window_table())
        label = f"[{number:2d}/{len(jobs)}] train {train:4d} {name:13s}"
        reported = {"decile": -1}

        def progress(done: int, total: int, label=label, t=t, reported=reported) -> None:
            decile = done * 10 // total
            if decile > reported["decile"]:         # a line at every 10% of the windows
                reported["decile"] = decile
                elapsed = time.time() - t
                eta = elapsed / done * (total - done)
                print(f"{label} windows {done:5d}/{total} ({done / total:4.0%}) | {elapsed / 60:5.1f} min, about {eta / 60:4.1f} min to go", flush=True)

        print(f"{label} starts: {n_windows} windows", flush=True)
        run = rs.train_and_save_walk_forward_run(setting, RECIPES[name], name, args.root, slim=True, workers=args.workers, progress=progress)
        saved.append(run)
        took = time.time() - t
        if reported["decile"] >= 0:                 # a run that was fitted now (a saved one is skipped at once and has nothing to report)
            seconds_per_window[RECIPES[name].model_family == "xgboost"].append(took / n_windows)
        left = sum(
            len(s.window_table()) * (sum(seconds_per_window[RECIPES[nm].model_family == "xgboost"]) / len(seconds_per_window[RECIPES[nm].model_family == "xgboost"]))
            for _, s, nm in jobs[number:] if seconds_per_window[RECIPES[nm].model_family == "xgboost"]
        )
        unknown = sum(1 for _, _, nm in jobs[number:] if not seconds_per_window[RECIPES[nm].model_family == "xgboost"])
        estimate = f"about {left / 60:4.1f} min left" + (f" (+{unknown} runs of a kind not timed yet)" if unknown else "")
        print(f"{label} done in {took / 60:5.1f} min (saved) | {number}/{len(jobs)} runs, {(time.time() - started) / 60:5.1f} min since the start, {estimate}", flush=True)
    if not args.no_mlflow:
        for run in saved:
            rs.log_walk_forward_run_to_mlflow(run, e.READING, experiment=e.MLFLOW_EXPERIMENT)
        print(f"{len(saved)} runs logged to MLflow ({e.MLFLOW_EXPERIMENT})")
    print("done", flush=True)


if __name__ == "__main__":
    main()
