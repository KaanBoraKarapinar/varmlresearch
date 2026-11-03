"""
Experiment: models trained ONCE on 23 tickers, applied to other calendars (other portfolios), against models trained on that portfolio itself.

    python training/train_23_experiment.py

All walk-forwards: train 756 / forecast 21 from 2016-03-01 (the RV starts 2016-01-04), forecast days 2019-03-04 .. 2026-06-30.
Needs data_general/23_ticker_data_2010_2026.csv (downloads/download_prices_yahoo.py), RV_sip_23tickers.csv (downloads/alpaca_calculate_rv.py), vix, sp500, us3m/5y/10y/30y.

The features are PORTABLE (FeatureSpec.portable(): no MarketEwmaVol, which is the portfolio's own volatility), so a model trained on the 23 tickers
can be applied to a calendar with another portfolio (ModelRecipe.pretrained_from, see walk_forward._check_pretrained).

    S23  the 23 tickers, equal weight 1/23: the models are TRAINED here (pooled over 23 tickers); slim runs (no per-window feature files)
    S4   AAPL, JPM, XOM, KO, 0.25 each: the S23 models APPLIED (names ..._from23) and, as the reference, the same models TRAINED on the 4 tickers
    S13  the 13 tickers of the first batch, 1/13 each: the S23 models applied (slim)

The runs are saved in the run store (a saved run is reused: the script can be run again after an interruption) and logged to MLflow.
"""
import dataclasses
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

import core
from core import run_store as rs
from core import walk_forward as wf
from core.feature_blocks import FeatureSpec, RvHar, Sp500Return, Vix, YieldSpread10y3m, YieldSpread10y5y, YieldSpread30y10y
from core.targets import RealizedVariance, RelativeRvChange, SquaredReturn

DATA_PATH = './data_general/23_ticker_data_2010_2026.csv'
UNIT = 'percent'
MLFLOW_EXPERIMENT = 'walk_forward_horizon'
READING = wf.Reading(var_confidence=0.95, window_size=252, min_periods=30, test_confidence=0.95)

TICKERS_4 = ['AAPL', 'JPM', 'XOM', 'KO']
TICKERS_13 = ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "NFLX", "AMZN", "HD", "MCD", "PG", "KO", "WMT"]
TICKERS_23 = TICKERS_13 + ["XOM", "CVX", "JPM", "GS", "BAC", "V", "JNJ", "UNH", "LLY", "MRK"]

prices = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')       # META is empty before 2012-05-18: the walk-forwards start in 2016


def market_source(tag: str) -> core.AdditionalFeatureSource:
    return core.AdditionalFeatureSource(tag, f"data_general/{tag}.csv", "market", column="Close", available_after_bars=0)


SOURCES = (
    core.AdditionalFeatureSource("rv", "data_general/RV_sip_23tickers.csv", "tickers", available_after_bars=0),    # each setting reads the columns of its own tickers
    market_source("vix"), market_source("sp500"),
    market_source("us3m"), market_source("us5y"), market_source("us10y"), market_source("us30y"),
)


def setting_of(tickers: list[str]) -> wf.WalkForwardSetting:
    return wf.WalkForwardSetting(
        prices, tickers, {t: 1 / len(tickers) for t in tickers}, train_bars=756, forecast_bars=21, unit=UNIT, data_path=DATA_PATH,
        start_date="2016-03-01", additional_feature_sources=SOURCES,
    )


SCALE = 10000
BASE = FeatureSpec.default(unit=UNIT).portable()                  # Returns, AbsReturn, RollingStats, EwmaVol
RV_HAR = BASE.with_blocks(RvHar())
MACRO = RV_HAR.with_blocks(Vix(changes=(5,)), Sp500Return(), YieldSpread10y3m(), YieldSpread10y5y(), YieldSpread30y10y())
XGB_RECIPES = {
    'xgb_long':         wf.ModelRecipe('long', features=BASE, target=SquaredReturn(SCALE, unit=UNIT)),
    'xgb_long_rv_har':  wf.ModelRecipe('long', features=RV_HAR, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_har': wf.ModelRecipe('long', features=RV_HAR, target=RelativeRvChange(1.0, unit=UNIT)),
    'xgb_long_rv_macro':  wf.ModelRecipe('long', features=MACRO, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_macro': wf.ModelRecipe('long', features=MACRO, target=RelativeRvChange(1.0, unit=UNIT)),
}
BASELINES = wf.BASELINE_RECIPES


def timed(label: str, function):
    t0 = time.time()
    run = function()
    print(f"{label:55s} {time.time() - t0:6.0f} s", flush=True)
    return run


def main() -> None:
    s23, s4, s13 = setting_of(TICKERS_23), setting_of(TICKERS_4), setting_of(TICKERS_13)
    for name, s in (("S23", s23), ("S4", s4), ("S13", s13)):
        table = s.window_table()
        print(f"{name}: {len(s.tickers)} tickers, {len(table)} windows, forecast {table['forecast_start'].iloc[0].date()} .. {table['forecast_end'].iloc[-1].date()} | {s.walk_forward_id[-40:]}", flush=True)

    # 1. the 23 tickers: the baselines and the models trained once (slim: the per-window feature files of 23 tickers are most of the disk)
    trained = {}
    for name, recipe in BASELINES.items():
        timed(f"S23 {name}", lambda: rs.train_and_save_walk_forward_run(s23, recipe, name, slim=True))
    for name, recipe in XGB_RECIPES.items():
        trained[name] = timed(f"S23 {name} (trained on 23 tickers)", lambda: rs.train_and_save_walk_forward_run(s23, recipe, name, slim=True))

    # 2. the 4 tickers: the baselines, the same models trained on the 4 tickers (the reference), and the 23-ticker models applied
    for name, recipe in BASELINES.items():
        timed(f"S4  {name}", lambda: rs.train_and_save_walk_forward_run(s4, recipe, name))
    for name, recipe in XGB_RECIPES.items():
        timed(f"S4  {name} (trained on the 4 tickers)", lambda: rs.train_and_save_walk_forward_run(s4, recipe, name))
    for name, recipe in XGB_RECIPES.items():
        applied = dataclasses.replace(recipe, pretrained_from=trained[name].run_id)
        timed(f"S4  {name}_from23 (the 23-ticker model applied)", lambda: rs.train_and_save_walk_forward_run(s4, applied, f"{name}_from23"))

    # 3. the 13 tickers: the baselines and the 23-ticker models applied
    for name, recipe in BASELINES.items():
        timed(f"S13 {name}", lambda: rs.train_and_save_walk_forward_run(s13, recipe, name, slim=True))
    for name, recipe in XGB_RECIPES.items():
        applied = dataclasses.replace(recipe, pretrained_from=trained[name].run_id)
        timed(f"S13 {name}_from23 (the 23-ticker model applied)", lambda: rs.train_and_save_walk_forward_run(s13, applied, f"{name}_from23", slim=True))

    walk_forward_ids = {s23.walk_forward_id, s4.walk_forward_id, s13.walk_forward_id}
    for run in [r for r in rs.load_runs(train_bars=756, forecast_bars=21) if r.walk_forward_id in walk_forward_ids]:
        rs.log_walk_forward_run_to_mlflow(run, READING, experiment=MLFLOW_EXPERIMENT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
