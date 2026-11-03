"""
Analysis of train_random_weights_experiment.py: the 23-ticker models applied to portfolios with random fixed weights, against GARCH, EWMA and the
historical VaR of the same portfolio.

    python training/analyze_random_weights.py   prints the tables and saves one experiment per random portfolio (store/experiments)

For every portfolio (and for the equal-weight 1/23 one as the reference): the mean tick loss of every model, its difference to the GARCH of the same
portfolio in % (negative = better), the Diebold-Mariano p-value (HAC) of that difference and its Holm correction over the 7 comparisons of the portfolio.
The summary over the 5 random portfolios counts in how many of them a model is ahead of GARCH: the portfolios are drawn from the same 23 stocks, so
they are not independent samples.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import core
from core import run_store as rs
from core import walk_forward as wf
from core.run_comparison import _dm_p_value
from training import train_23_experiment as e
from training import train_random_weights_experiment as t

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
PERIODS = {"whole": ("2019-03-04", "2100-01-01"), "crash": ("2020-02-19", "2020-06-30"), "after": ("2020-07-01", "2100-01-01")}
MODELS = ["garch_wide", "ewma", "historical", "xgb_long", "xgb_long_rv_har", "xgb_long_drv_har", "xgb_long_rv_macro", "xgb_long_drv_macro"]
READING = dict(var_confidence=0.95, window_size=252, min_periods=30, test_confidence=0.95)
STORE = ROOT / "store"


def portfolio_settings() -> dict:
    out = {"eq": (e.setting_of(e.TICKERS_23), "", None, None)}
    for name, (alpha, seed) in t.PORTFOLIOS.items():
        weights = t.random_weights(alpha, seed)
        out[name] = (t.setting_with(weights), name, alpha, seed)
    return out


def runs_of(all_runs, setting, tag) -> dict:
    by = {r.name: r for r in all_runs if r.walk_forward_id == setting.walk_forward_id}
    if tag == "":
        return {m: by[m] for m in MODELS}
    return {m: by[f"{m}_{tag}"] if m in ("garch_wide", "ewma", "historical") else by[f"{m}_from23_{tag}"] for m in MODELS}


def diagonal_share(run, setting) -> float:
    """The share of the portfolio variance that the ticker variances give alone (sum w_i^2 sigma_i^2 / w' Sigma w), averaged over the days."""
    bank, tickers = run.ewma_bank, list(setting.tickers)
    d = np.sqrt(bank[[f"sigma_{x}" for x in tickers]].to_numpy())
    cov = np.stack([bank[[f"cov_{i}_{j}" for j in tickers]].to_numpy() for i in tickers], axis=1)
    std = np.sqrt(np.einsum("dii->di", cov))
    sigma = d[:, :, None] * (cov / (std[:, :, None] * std[:, None, :])) * d[:, None, :]
    w = np.array([setting.weights[x] for x in tickers])
    return float((np.einsum("i,di->d", w ** 2, d ** 2) / np.einsum("i,dij,j->d", w, sigma, w)).mean())


def evaluate(runs: dict, conf: float, period: str) -> pd.DataFrame:
    reading = wf.Reading(var_confidence=conf, window_size=252, min_periods=30, test_confidence=conf)
    dates = rs.common_dates(list(runs.values()), reading)
    dates = dates[(dates >= PERIODS[period][0]) & (dates <= PERIODS[period][1])]
    var_dfs = {m: r.var_df(reading).loc[dates] for m, r in runs.items()}
    losses = core.daily_tick_losses(var_dfs, conf)
    mean = losses.mean()
    table = pd.DataFrame({"tick": mean, "vs garch %": (mean / mean["garch_wide"] - 1) * 100})
    if len(dates) > 600:
        p = pd.Series({m: _dm_p_value((losses[m] - losses["garch_wide"]).to_numpy(), 5) for m in MODELS if m != "garch_wide"})
        table["dm_p"], table["holm_p"] = p, pd.Series(multipletests(p.to_numpy(), method="holm")[1], index=p.index)
    table["exceed"] = pd.Series({m: int((v["loss"] > v["varForecast"]).sum()) for m, v in var_dfs.items()})
    table["kupiec_ok"] = pd.Series({m: bool(core.run_full_backtest_suite_single_window(v, v.index[0], v.index[-1], conf, conf)["kupiec_pof_pass"]) for m, v in var_dfs.items()})
    table.attrs["days"] = len(dates)
    return table


def notes_of(name, alpha, seed, setting, tables) -> str:
    w = pd.Series(setting.weights)
    top = ", ".join(f"{k} {v:.2f}" for k, v in w.sort_values(ascending=False).head(5).items())
    fmt = lambda table: ", ".join(f"`{m}` {table.loc[m, 'tick']:.5f} ({table.loc[m, 'vs garch %']:+.1f}%)" for m in table.sort_values("tick").index if m != "garch_wide")

    def significant(table):
        found = [f"`{m}` (corrected p = {table.loc[m, 'holm_p']:.3f})" for m in table.index if m != "garch_wide" and table.loc[m, "holm_p"] < 0.05]
        return ", ".join(found) if found else "none"
    return f"""\
**Random fixed weights over the 23 tickers: the 23-ticker models, trained once, applied to this portfolio, against GARCH, EWMA and the historical VaR of the same portfolio.**

- **The portfolio.** Dirichlet(alpha {alpha}) draw, seed {seed}: effective number of tickers {1 / (w ** 2).sum():.1f}, largest weight {w.max():.2f}; the largest: {top}. The portfolio is fixed over the whole period.
  The walk-forward is train 756 / forecast 21 from 2016-03-01 (forecast days 2019-03-04 to 2026-06-30, 1842 days). The XGBoost models are the five portable models trained on the 23 tickers (`train_23_experiment.py`); GARCH, EWMA and the historical VaR are computed for this portfolio.
- **95%, whole period, mean tick loss** (GARCH {tables['whole95'].loc['garch_wide', 'tick']:.5f}; the difference to it in brackets): {fmt(tables['whole95'])}.
  Significantly different from GARCH after the Holm correction (7 comparisons): {significant(tables['whole95'])}.
- **99%, whole period** (GARCH {tables['whole99'].loc['garch_wide', 'tick']:.5f}): {fmt(tables['whole99'])}. Significant after the correction: {significant(tables['whole99'])}.
  Kupiec fails at 99% for: {', '.join(f'`{m}`' for m in tables['whole99'].index if not tables['whole99'].loc[m, 'kupiec_ok']) or 'no model'}.
- **Covid crash (93 days, about 5 exceedances expected: descriptive only), 95%** (GARCH {tables['crash95'].loc['garch_wide', 'tick']:.5f}): {fmt(tables['crash95'])}.
- **After the crash, 95%** (GARCH {tables['after95'].loc['garch_wide', 'tick']:.5f}): {fmt(tables['after95'])}.
- **Reading.** One portfolio is one sample: the tests are conditional on it. The summary over the five random portfolios is in `Research Results.md`.
"""


def main() -> None:
    all_runs = rs.load_runs(train_bars=756, forecast_bars=21)
    settings = portfolio_settings()
    relative, info_rows = {}, []
    for pname, (setting, tag, alpha, seed) in settings.items():
        runs = runs_of(all_runs, setting, tag)
        w = np.array([setting.weights[x] for x in setting.tickers])
        info_rows.append({"portfolio": pname, "eff. tickers": round(1 / (w ** 2).sum(), 1), "max weight": round(float(w.max()), 2), "diag share": round(diagonal_share(runs["xgb_long_drv_har"], setting), 2)})
        tables = {"whole95": evaluate(runs, 0.95, "whole"), "whole99": evaluate(runs, 0.99, "whole"), "crash95": evaluate(runs, 0.95, "crash"), "after95": evaluate(runs, 0.95, "after")}
        print(f"\n######## {pname}: {info_rows[-1]}")
        for key in ("whole95", "whole99"):
            print(f"--- {key} | {tables[key].attrs['days']} days"); print(tables[key].round(5).to_string(float_format=lambda x: f"{x:.5f}"))
        for key, (conf, period) in {"whole95": (0.95, "whole"), "whole99": (0.99, "whole"), "crash95": (0.95, "crash"), "after95": (0.95, "after")}.items():
            for m in MODELS:
                relative[(pname, conf, period, m)] = tables[key].loc[m, "vs garch %"]
        if pname != "eq":
            experiment = core.Experiment(
                name=f"Random portfolio {pname}: 23-ticker models against GARCH, EWMA, historical (train 756 f 21)",
                question=f"Do the 23-ticker XGBoost models beat GARCH on a portfolio with random fixed weights ({pname})?",
                run_ids=tuple(runs[m].run_id for m in MODELS), baseline_run_id=runs["garch_wide"].run_id,
                start="2019-03-04", end="2026-06-30", reading=READING, notes=notes_of(pname, alpha, seed, setting, tables),
            )
            print("saved", core.save_experiment(experiment, STORE, overwrite=True).name)

    print("\n######## the portfolios"); print(pd.DataFrame(info_rows).set_index("portfolio").to_string())
    rel = pd.Series(relative).rename_axis(["portfolio", "conf", "period", "model"]).unstack("portfolio")[list(settings)]
    for (conf, period), group in rel.groupby(level=["conf", "period"]):
        g = group.droplevel(["conf", "period"]).drop(index="garch_wide")
        randoms = [c for c in g.columns if c != "eq"]
        g["mean"] = g[randoms].mean(axis=1)
        g["ahead of garch in"] = (g[randoms] < 0).sum(axis=1)
        print(f"\n=== tick loss against GARCH of the same portfolio, % (negative = better), VaR {conf:g}, {period}"); print(g.round(1).to_string())


if __name__ == "__main__":
    main()
