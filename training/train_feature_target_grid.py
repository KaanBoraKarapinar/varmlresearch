"""
Experiment: which feature set and which target, on the 23 tickers with equal weights (the base walk-forward every portfolio is read from).

    python training/train_feature_target_grid.py                       # trains and saves the 13 xgboost runs + the three baselines
    python training/train_feature_target_grid.py --tickers 4          # the same runs on AAPL, JPM, XOM, KO (equal weights), trained on those 4 tickers only
    python training/train_feature_target_grid.py --dry-run            # lists the runs
    python training/train_feature_target_grid.py --only xgb_F1_ratio xgb_F3_rv

One walk-forward: the 23 tickers, equal weights, train 756 / forecast 21 from 2016-03-01 (the RV starts 2016-01-04), forecast days 2019-03-04 .. 2026-06-30
(the setting of train_23_experiment.py). The XGBoost parameters are the defaults for every run, so that the runs differ only in the features and the target.

FEATURE SETS: nested and PORTABLE (no MarketEwmaVol / CrossSectionVol / TickerOneHot: those depend on the portfolio's tickers, and a model that has them
cannot be read for other weights or applied to another portfolio), each adds to the one before it in this order:

    F0  Returns, AbsReturn, EwmaVol                           the plainest
    F1  F0 + RollingStats (5 / 10 / 21 days)                  the base set the project used so far
    F2  F1 + macro (VIX, S&P 500 return, three yield spreads) what the macro data gives, without RV
    F3  F1 + RvHar (realized variance of the day, its week and month mean)   what RV as a feature gives, without macro
    F4  F3 + macro                                            everything

TARGETS (the name of a run is xgb_<feature set>_<target>):

    ratio  RatioToBase(clip 20)   r^2 against the EWMA variance: a multiplicative correction of the EWMA (the target of the feature ladder F0..F4)
    sq     SquaredReturn          r^2 itself: noisy, and a squared-error tree can predict <= 0 (the variance floor): kept on purpose, count the floor days
    ln     LogVariance            ln(r^2 + eps): the prediction goes through exp(), so it is always positive
    rv     RealizedVariance       the next day's realized variance (5-minute data)
    drv    RelativeRvChange       the change of the realized variance against today's

    the ladder      F0..F4 with ratio                                      5 runs
    the targets     sq, ln, rv, drv  x  F1 (no RV in the features), F3 (RV in the features)    8 runs (ratio x F1 and ratio x F3 are in the ladder)

The runs are saved slim (no per-window feature files: with 23 tickers they are most of the disk) in the run store and logged to MLflow. A saved run is reused:
run the script again after an interruption.
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core import run_store as rs
from core import walk_forward as wf
from core.feature_blocks import AbsReturn, EwmaVol, FeatureSpec, Returns, RvHar, Sp500Return, Vix, YieldSpread10y3m, YieldSpread10y5y, YieldSpread30y10y
from core.targets import LogVariance, RatioToBase, RealizedVariance, RelativeRvChange, SquaredReturn
from training import train_23_experiment as e

UNIT = e.UNIT
UNIVERSES = {"23": e.TICKERS_23, "4": e.TICKERS_4}      # equal weights in both; the 4 tickers are trained on their own, the models do not see the other 19
MACRO = (Vix(changes=(5,)), Sp500Return(), YieldSpread10y3m(), YieldSpread10y5y(), YieldSpread30y10y())
F1 = FeatureSpec.default(unit=UNIT).portable()
FEATURE_SETS = {
    "F0": FeatureSpec((Returns(), AbsReturn(), EwmaVol(21)), unit=UNIT),
    "F1": F1,
    "F2": F1.with_blocks(*MACRO),
    "F3": F1.with_blocks(RvHar()),
    "F4": F1.with_blocks(RvHar(), *MACRO),
}
TARGETS = {
    "ratio": RatioToBase(clip=20.0, unit=UNIT),
    "sq": SquaredReturn(e.SCALE, unit=UNIT),
    "ln": LogVariance(unit=UNIT),
    "rv": RealizedVariance(e.SCALE, unit=UNIT),
    "drv": RelativeRvChange(1.0, unit=UNIT),
}
GRID = [(f, "ratio") for f in FEATURE_SETS] + [(f, t) for t in ("sq", "ln", "rv", "drv") for f in ("F1", "F3")]
RECIPES = {f"xgb_{f}_{t}": wf.ModelRecipe("long", features=FEATURE_SETS[f], target=TARGETS[t]) for f, t in GRID}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers", choices=list(UNIVERSES), default="23", help="the universe: 23 tickers (default) or the 4 tickers AAPL, JPM, XOM, KO; equal weights, the models see only these tickers")
    parser.add_argument("--only", nargs="+", help="train only these runs (names as listed by --dry-run)")
    parser.add_argument("--no-baselines", action="store_true", help="leave out garch_wide, ewma and historical")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log the runs to MLflow")
    parser.add_argument("--dry-run", action="store_true", help="list what would be trained, train nothing")
    args = parser.parse_args()
    unknown = [n for n in (args.only or []) if n not in RECIPES]
    if unknown:
        parser.error(f"unknown runs {unknown}; known: {list(RECIPES)}.")

    setting = e.setting_of(UNIVERSES[args.tickers])
    table = setting.window_table()
    print(f"S{args.tickers}: {len(setting.tickers)} tickers, {len(table)} windows, forecast {table['forecast_start'].iloc[0].date()} .. {table['forecast_end'].iloc[-1].date()}\n  {setting.walk_forward_id}")
    chosen = {name: recipe for name, recipe in RECIPES.items() if not args.only or name in args.only}
    baselines = {} if args.no_baselines else wf.BASELINE_RECIPES
    for name, recipe in chosen.items():
        blocks = ", ".join(type(b).__name__ for b in recipe.features.blocks)
        print(f"  {name:18s} target {type(recipe.target).__name__:17s} features: {blocks}")
    print(f"  baselines: {list(baselines)}")
    if args.dry_run:
        return

    saved = []
    for name, recipe in baselines.items():
        saved.append(e.timed(f"S{args.tickers} {name}", lambda: rs.train_and_save_walk_forward_run(setting, recipe, name, slim=True)))
    for name, recipe in chosen.items():
        saved.append(e.timed(f"S{args.tickers} {name}", lambda: rs.train_and_save_walk_forward_run(setting, recipe, name, slim=True)))
    if not args.no_mlflow:
        for run in saved:
            rs.log_walk_forward_run_to_mlflow(run, e.READING, experiment=e.MLFLOW_EXPERIMENT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
