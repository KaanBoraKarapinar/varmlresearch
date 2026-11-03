"""
Experiment: the seven standard runs on the walk-forward train 756 / forecast 21 (all data, from the first possible bar).

    python train_756_f21.py

Trains and saves the runs into the run store (a run that is already saved is reused) and logs them to MLflow.
"""
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import pandas as pd

import run_store as rs
import walk_forward as wf
from core.feature_blocks import FeatureSpec, TickerOneHot
from core.targets import RatioToBase, SquaredReturn

TICKERS = ['AAPL', 'JPM', 'XOM', 'KO']
WEIGHTS = {t: 0.25 for t in TICKERS}
DATA_PATH = './data_general/4_ticker_data_2010_2026.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
UNIT = 'percent'
READING = wf.Reading(var_confidence=0.95, window_size=252, min_periods=30, test_confidence=0.95)
MLFLOW_EXPERIMENT = 'walk_forward_horizon'

setting = wf.WalkForwardSetting(df, TICKERS, WEIGHTS, train_bars=756, forecast_bars=21, unit=UNIT, data_path=DATA_PATH)
print(setting.walk_forward_id, len(setting.window_table()), "windows", flush=True)

GARCH_KWARGS = dict(mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
FEATURES = FeatureSpec.default(unit=UNIT)
SCALE = 10000
RECIPES = {
    'garch_wide':            wf.ModelRecipe('wide', model_family='garch', model_kwargs=GARCH_KWARGS),
    'ewma':                  wf.ModelRecipe('wide', model_family='ewma'),
    'historical':            wf.ModelRecipe('wide', model_family='historical'),     # plain historical VaR (the window comes from the VaR reading)
    'xgb_wide':              wf.ModelRecipe('wide', features=FEATURES, target=SquaredReturn(SCALE, unit=UNIT)),
    'xgb_long':              wf.ModelRecipe('long', features=FEATURES, target=SquaredReturn(SCALE, unit=UNIT)),
    'xgb_long_n300':         wf.ModelRecipe('long', features=FEATURES, target=SquaredReturn(SCALE, unit=UNIT), model_kwargs=dict(n_estimators=300)),
    'xgb_long_ratio':        wf.ModelRecipe('long', features=FEATURES, target=RatioToBase(clip=20.0, unit=UNIT)),
    'xgb_long_ratio_ticker': wf.ModelRecipe('long', features=FEATURES.with_blocks(TickerOneHot()), target=RatioToBase(clip=20.0, unit=UNIT)),
}

if __name__ == "__main__":
    for name, recipe in RECIPES.items():
        t0 = time.time()
        rs.train_and_save_walk_forward_run(setting, recipe, name)
        print(name, round(time.time() - t0), "s", flush=True)

    for run in [r for r in rs.load_runs(train_bars=756, forecast_bars=21) if r.walk_forward_id == setting.walk_forward_id]:
        rs.log_walk_forward_run_to_mlflow(run, READING, experiment=MLFLOW_EXPERIMENT)
    print("done", flush=True)
