"""
Experiment: the RV models with the market features added (VIX, S&P 500 return, Treasury yield spreads), train 756 / forecast 21 from 2016-03.

    python training/train_macro_experiment.py

Needs data_general/RV_sip.csv, vix.csv, sp500.csv, us3m.csv, us5y.csv, us10y.csv, us30y.csv.
The additional data is part of the walk-forward's identity, so the baselines are trained again on this walk-forward.

    xgb_long_rv_har / xgb_long_drv_har      RV features only (the reference)
    xgb_long_rv_macro / xgb_long_drv_macro  + Vix(changes=(5,)), Sp500Return, YieldSpread10y3m / 10y5y / 30y10y
    ..._lags                                 + RvLags((1, 2, 3)): rv_lag_1..3 (rv_har_lags: RV features + lags without the market features)
    ..._ticker                               + TickerOneHot (drv_har_ticker, rv_macro_ticker, drv_macro_ticker)
"""
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
from core.feature_blocks import FeatureSpec, RvHar, RvLags, Sp500Return, TickerOneHot, Vix, YieldSpread10y3m, YieldSpread10y5y, YieldSpread30y10y
from core.targets import RealizedVariance, RelativeRvChange

TICKERS = ['AAPL', 'JPM', 'XOM', 'KO']
WEIGHTS = {t: 0.25 for t in TICKERS}
DATA_PATH = './data_general/4_ticker_data_2010_2026.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
UNIT = 'percent'
READING = wf.Reading(var_confidence=0.95, window_size=252, min_periods=30, test_confidence=0.95)
MLFLOW_EXPERIMENT = 'walk_forward_horizon'


def market_source(tag: str) -> core.AdditionalFeatureSource:
    """A market-wide series of data_general/<tag>.csv (column Close), known at the close of its own day."""
    return core.AdditionalFeatureSource(tag, f"data_general/{tag}.csv", "market", column="Close", available_after_bars=0)


sources = (
    core.AdditionalFeatureSource("rv", "data_general/RV_sip.csv", "tickers", available_after_bars=0),
    market_source("vix"), market_source("sp500"),
    market_source("us3m"), market_source("us5y"), market_source("us10y"), market_source("us30y"),
)
setting = wf.WalkForwardSetting(
    df, TICKERS, WEIGHTS, train_bars=756, forecast_bars=21, unit=UNIT, data_path=DATA_PATH,
    start_date="2016-03-01", additional_feature_sources=sources,
)
table = setting.window_table()
print(setting.walk_forward_id, len(table), "windows | first forecast", table["forecast_start"].iloc[0].date(), flush=True)

GARCH_KWARGS = dict(mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
FEATURES = FeatureSpec.default(unit=UNIT)
FEATURES_RV = FEATURES.with_blocks(RvHar())
FEATURES_MACRO = FEATURES_RV.with_blocks(
    Vix(changes=(5,)), Sp500Return(), YieldSpread10y3m(), YieldSpread10y5y(), YieldSpread30y10y(),
)
RV_LAGS = RvLags((1, 2, 3))                                   # rv_lag_1..3: the RV of yesterday and the two days before
FEATURES_RV_LAGS = FEATURES_RV.with_blocks(RV_LAGS)
FEATURES_MACRO_LAGS = FEATURES_MACRO.with_blocks(RV_LAGS)
SCALE = 10000
RECIPES = {
    'garch_wide':         wf.ModelRecipe('wide', model_family='garch', model_kwargs=GARCH_KWARGS),
    'ewma':               wf.ModelRecipe('wide', model_family='ewma'),
    'historical':         wf.ModelRecipe('wide', model_family='historical'),     # plain historical VaR (the window comes from the VaR reading)
    'xgb_long_rv_har':    wf.ModelRecipe('long', features=FEATURES_RV, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_har':   wf.ModelRecipe('long', features=FEATURES_RV, target=RelativeRvChange(1.0, unit=UNIT)),
    'xgb_long_rv_macro':  wf.ModelRecipe('long', features=FEATURES_MACRO, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_macro': wf.ModelRecipe('long', features=FEATURES_MACRO, target=RelativeRvChange(1.0, unit=UNIT)),
    # + the ticker tag (TickerOneHot: a level per ticker for the pooled model), on the features of the winners
    'xgb_long_drv_har_ticker':   wf.ModelRecipe('long', features=FEATURES_RV.with_blocks(TickerOneHot()), target=RelativeRvChange(1.0, unit=UNIT)),
    'xgb_long_rv_macro_ticker':  wf.ModelRecipe('long', features=FEATURES_MACRO.with_blocks(TickerOneHot()), target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_macro_ticker': wf.ModelRecipe('long', features=FEATURES_MACRO.with_blocks(TickerOneHot()), target=RelativeRvChange(1.0, unit=UNIT)),
    # + the lagged RV (RvLags)
    'xgb_long_rv_har_lags':     wf.ModelRecipe('long', features=FEATURES_RV_LAGS, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_rv_macro_lags':   wf.ModelRecipe('long', features=FEATURES_MACRO_LAGS, target=RealizedVariance(SCALE, unit=UNIT)),
    'xgb_long_drv_macro_lags':  wf.ModelRecipe('long', features=FEATURES_MACRO_LAGS, target=RelativeRvChange(1.0, unit=UNIT)),
}

if __name__ == "__main__":
    for name, recipe in RECIPES.items():
        t0 = time.time()
        rs.train_and_save_walk_forward_run(setting, recipe, name)
        print(name, round(time.time() - t0), "s", flush=True)

    for run in [r for r in rs.load_runs(train_bars=756, forecast_bars=21) if r.walk_forward_id == setting.walk_forward_id]:
        rs.log_walk_forward_run_to_mlflow(run, READING, experiment=MLFLOW_EXPERIMENT)
    print("done", flush=True)
