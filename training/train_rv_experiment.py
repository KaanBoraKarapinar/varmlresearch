"""
Experiment: the realized-variance (RV) targets against the squared-return model, train 756 / forecast 21 from 2016-03.

    python train_rv_experiment.py

Needs data_general/RV_sip.csv (alpaca_calculate_rv.py). The RV starts 2016-01-04 and the windows read 30 warmup bars before
the first train bar, so the walk-forward starts on 2016-03-01.

Runs: the baselines (garch_wide, ewma, historical), the squared-return and ratio models of the standard set, and three RV models:
    xgb_long_rv       default features, target RealizedVariance (the next day's RV)
    xgb_long_rv_har   + RvHar features (rv, rv_week, rv_month), target RealizedVariance
    xgb_long_drv_har  + RvHar features, target RelativeRvChange (dRV_rel_next = RV_next / RV - 1)
"""
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import pandas as pd

import core
import run_store as rs
import walk_forward as wf
from core.feature_blocks import FeatureSpec, RvHar, TickerOneHot
from core.targets import RatioToBase, RealizedVariance, RelativeRvChange, SquaredReturn

TICKERS = ['AAPL', 'JPM', 'XOM', 'KO']
WEIGHTS = {t: 0.25 for t in TICKERS}
DATA_PATH = './data_general/4_ticker_data_2010_2026.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
UNIT = 'percent'
READING = wf.Reading(var_confidence=0.95, window_size=252, min_periods=30, test_confidence=0.95)
MLFLOW_EXPERIMENT = 'walk_forward_horizon'

sources = (core.AdditionalFeatureSource("rv", "data_general/RV_sip.csv", "tickers", available_after_bars=0),)
setting = wf.WalkForwardSetting(
    df, TICKERS, WEIGHTS, train_bars=756, forecast_bars=21, unit=UNIT, data_path=DATA_PATH,
    start_date="2016-03-01", additional_feature_sources=sources,
)
table = setting.window_table()
print(setting.walk_forward_id, len(table), "windows | first forecast", table["forecast_start"].iloc[0].date(),
      "| last", table["forecast_end"].iloc[-1].date(), flush=True)

GARCH_KWARGS = dict(mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
FEATURES = FeatureSpec.default(unit=UNIT)                     # the current feature set
FEATURES_RV = FEATURES.with_blocks(RvHar())                   # + rv, rv_week, rv_month
SCALE = 10000
RECIPES = {
    'garch_wide':            wf.ModelRecipe('wide', model_family='garch', model_kwargs=GARCH_KWARGS),
    'ewma':                  wf.ModelRecipe('wide', model_family='ewma'),
    'historical':            wf.ModelRecipe('wide', model_family='historical'),     # plain historical VaR (the window comes from the VaR reading)
    'xgb_long':              wf.ModelRecipe('long', features=FEATURES, target=SquaredReturn(SCALE, unit=UNIT)),
    'xgb_long_ratio_ticker': wf.ModelRecipe('long', features=FEATURES.with_blocks(TickerOneHot()), target=RatioToBase(clip=20.0, unit=UNIT)),
    'xgb_long_rv':           wf.ModelRecipe('long', features=FEATURES, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_rv_har':       wf.ModelRecipe('long', features=FEATURES_RV, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_har':      wf.ModelRecipe('long', features=FEATURES_RV, target=RelativeRvChange(1.0, unit=UNIT)),
}

if __name__ == "__main__":
    for name, recipe in RECIPES.items():
        t0 = time.time()
        rs.train_and_save_walk_forward_run(setting, recipe, name)
        print(name, round(time.time() - t0), "s", flush=True)

    for run in [r for r in rs.load_runs(train_bars=756, forecast_bars=21) if r.walk_forward_id == setting.walk_forward_id]:
        rs.log_walk_forward_run_to_mlflow(run, READING, experiment=MLFLOW_EXPERIMENT)
    print("done", flush=True)
