"""
Experiment: the 23-ticker models applied to portfolios with RANDOM FIXED weights, against GARCH, EWMA and the historical VaR of the same portfolio.

    python training/train_random_weights_experiment.py          (after training/train_23_experiment.py: it applies the models that script trained)

For every random weight vector over the 23 tickers one walk-forward is built (train 756 / forecast 21 from 2016-03-01, as in train_23_experiment.py):
    GARCH, EWMA and the historical VaR are computed for that portfolio (they read only its return series and, for the EWMA, the covariance),
    the five portable XGBoost models trained once on the 23 tickers (S23 of train_23_experiment.py) are applied to it (ModelRecipe.pretrained_from).
The runs of portfolio k are called <model>_rw<k>. All runs are slim and logged to MLflow.

The weights come from a Dirichlet distribution with a fixed seed: alpha 1 spreads them, alpha 0.3 and 0.1 concentrate the portfolio on a few
tickers (far from the market, where the per-ticker forecasts should count more).
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

import numpy as np

from core import run_store as rs
from core import walk_forward as wf
from training import train_23_experiment as e

PORTFOLIOS = {                      # name: (Dirichlet alpha, seed)
    "rw1": (1.0, 11),
    "rw2": (1.0, 22),
    "rw3": (0.3, 33),
    "rw4": (0.3, 44),
    "rw5": (0.1, 55),
}


def random_weights(alpha: float, seed: int) -> dict[str, float]:
    weights = np.random.default_rng(seed).dirichlet(alpha * np.ones(len(e.TICKERS_23)))
    return {ticker: float(w) for ticker, w in zip(e.TICKERS_23, weights)}


def setting_with(weights: dict[str, float]) -> wf.WalkForwardSetting:
    return wf.WalkForwardSetting(
        e.prices, e.TICKERS_23, weights, train_bars=756, forecast_bars=21, unit=e.UNIT, data_path=e.DATA_PATH,
        start_date="2016-03-01", additional_feature_sources=e.SOURCES,
    )


def describe(weights: dict[str, float]) -> str:
    w = np.array(list(weights.values()))
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
    return f"effective number of tickers {1 / (w ** 2).sum():.1f} | largest weight {w.max():.2f} | top: " + ", ".join(f"{t} {v:.2f}" for t, v in top)


def main() -> None:
    s23 = e.setting_of(e.TICKERS_23)
    sources = {r.name: r for r in rs.load_runs(train_bars=756, forecast_bars=21) if r.walk_forward_id == s23.walk_forward_id and r.name in e.XGB_RECIPES}
    if set(sources) != set(e.XGB_RECIPES):
        raise SystemExit(f"run train_23_experiment.py first: the 23-ticker models {sorted(set(e.XGB_RECIPES) - set(sources))} are not in the run store.")

    walk_forward_ids = []
    for portfolio, (alpha, seed) in PORTFOLIOS.items():
        weights = random_weights(alpha, seed)
        setting = setting_with(weights)
        walk_forward_ids.append(setting.walk_forward_id)
        print(f"\n{portfolio} (Dirichlet alpha {alpha}, seed {seed}): {describe(weights)}", flush=True)
        for name, recipe in e.BASELINES.items():
            t0 = time.time()
            rs.train_and_save_walk_forward_run(setting, recipe, f"{name}_{portfolio}", slim=True)
            print(f"  {name}_{portfolio:4s} {time.time() - t0:5.0f} s", flush=True)
        for name, recipe in e.XGB_RECIPES.items():
            t0 = time.time()
            applied = dataclasses.replace(recipe, pretrained_from=sources[name].run_id)
            rs.train_and_save_walk_forward_run(setting, applied, f"{name}_from23_{portfolio}", slim=True)
            print(f"  {name}_from23_{portfolio:4s} {time.time() - t0:5.0f} s", flush=True)

    for run in [r for r in rs.load_runs(train_bars=756, forecast_bars=21) if r.walk_forward_id in set(walk_forward_ids)]:
        rs.log_walk_forward_run_to_mlflow(run, e.READING, experiment=e.MLFLOW_EXPERIMENT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
