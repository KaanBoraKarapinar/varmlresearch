"""
The HAR-RV baseline: the realized-variance regression of Corsi (2009) as a run of the same pipeline, to compare the XGBoost models with.

    python training/train_har_rv_baseline.py                          # log-HAR-RV, train 756 / forecast 21, 23 tickers, equal weights
    python training/train_har_rv_baseline.py --tickers 4              # the same on AAPL, JPM, XOM, KO
    python training/train_har_rv_baseline.py --variant level          # the HAR-RV in levels (kept for the record, see below)
    python training/train_har_rv_baseline.py --train 504 --forecast 21 --workers 8

HAR-RV forecasts a ticker's next-day realized variance as a LINEAR function of its realized variance today, its mean over the last 5 days and its mean over the last 22 days.
Here the regression is pooled over the tickers (one set of coefficients, refitted in every window), like the XGBoost models it is compared with. XGBoost's linear booster (gblinear)
does the fit: with enough rounds it finds the OLS coefficients (checked on the first window against numpy's least squares). The per-ticker variances then go through the same EWMA
correlation, the same filtered historical simulation and the same readings as every other run, so the VaR is comparable with theirs.

Two variants (a run name each):

    har_rv_log  (default)  features LogRvHar (the logs of the day's RV and of its 5- and 22-day means), target LogRealizedVariance: the log-HAR-RV, the textbook form. The forecast
                           exp(prediction) is always positive. 5000 rounds of gblinear, learning rate 0.5: the same fit as OLS (correlation 0.99998 on the first window).
    har_rv      (level)    features RvHar, target RealizedVariance, in levels. Its forecast can be NEGATIVE after a spike of the RV (April 2020): the variance floor (1e-12)
                           turns it into a standardized residual of 1e6, and the 99% quantile of the next 252 days is ruined (on the 4 tickers a VaR of 44,300%, on the 23 tickers
                           4 days above 20%). Kept in the store for the record; use har_rv_log.

Not the textbook exactly: one coefficient set for all tickers (the textbook fits each asset).
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core import run_store as rs
from core import walk_forward as wf
from core.feature_blocks import FeatureSpec, LogRvHar, RvHar
from core.targets import LogRealizedVariance, RealizedVariance
from training import process_portfolios as pp
from training import train_23_experiment as e

UNIVERSES = {"23": e.TICKERS_23, "4": e.TICKERS_4}      # equal weights in both
RECIPES = {
    "har_rv_log": wf.ModelRecipe(
        "long", features=FeatureSpec((LogRvHar(),), unit=e.UNIT), target=LogRealizedVariance(1.0, unit=e.UNIT),
        model_kwargs={"booster": "gblinear", "n_estimators": 5000, "learning_rate": 0.5},
    ),
    "har_rv": wf.ModelRecipe(
        "long", features=FeatureSpec((RvHar(),), unit=e.UNIT), target=RealizedVariance(e.SCALE, unit=e.UNIT),
        model_kwargs={"booster": "gblinear", "n_estimators": 2000, "learning_rate": 0.5},
    ),
}
VARIANTS = {"log": "har_rv_log", "level": "har_rv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=list(VARIANTS), default="log")
    parser.add_argument("--tickers", choices=list(UNIVERSES), default="23", help="the universe: 23 tickers (default) or the 4 tickers AAPL, JPM, XOM, KO; equal weights")
    parser.add_argument("--train", type=int, default=756)
    parser.add_argument("--forecast", type=int, default=21)
    parser.add_argument("--start", default="2016-03-01")
    parser.add_argument("--workers", type=int, default=None, help="fit windows in this many processes (none: one after the other)")
    parser.add_argument("--root", type=Path, default=rs.STORE_ROOT)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    name = VARIANTS[args.variant]
    setting = pp.base_setting(args.train, args.forecast, args.start, UNIVERSES[args.tickers])
    print(f"{name}: {args.tickers} tickers, train {args.train} / forecast {args.forecast}, {len(setting.window_table())} windows\n  {setting.walk_forward_id}")
    run = rs.train_and_save_walk_forward_run(setting, RECIPES[name], name, args.root, slim=True, workers=args.workers)
    print(f"saved: {run.run_id}")
    if not args.no_mlflow:
        rs.log_walk_forward_run_to_mlflow(run, e.READING, experiment=e.MLFLOW_EXPERIMENT)
        print(f"logged to MLflow ({e.MLFLOW_EXPERIMENT})")


if __name__ == "__main__":
    main()
