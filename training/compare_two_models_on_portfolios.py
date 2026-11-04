"""
Two models head to head on the portfolios of one walk-forward: is the difference between them significant?

    python training/compare_two_models_on_portfolios.py --a xgb_F3_drv --b xgb_F1_drv --group random200_v1 --train 756 --forecast 21
    ... --limit 3        # look before the full run

For every portfolio of the group and every reading (VaR 95% / 99%) it reads only the two models (nothing is trained), takes the daily tick loss of each and tests the
daily difference a - b (Diebold-Mariano, HAC standard error). Negative = a is cheaper than b. It writes, in store/Results/<group>/:

  * <a>_vs_<b>_train<T>_f<F>_per_portfolio.csv    a row per portfolio, reading and period: the tick losses, the relative difference, the DM statistic and p-value
  * <a>_minus_<b>_daily_tick_loss_<reading>.csv    the daily tick loss difference of every portfolio (days x portfolios), so that any other test can be run later without reading
                                                  the VaR again

and prints, for the whole group and for each Dirichlet alpha, the test on the AVERAGE daily difference of its portfolios (one p-value per group). The portfolios come from the
same stocks and days, so their p-values must not be combined as if they were independent; testing the group's average difference keeps the dependence between them.
Readings and periods are those of collect_portfolio_backtests.py.
"""
import argparse
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
from core import portfolios as pf
from core import run_store as rs
from core.run_comparison import _dm_p_value        # the Diebold-Mariano p-value the dashboard uses (HAC, two-sided)
from training import process_portfolios as pp
from training.collect_portfolio_backtests import READINGS, periods_of

STORE = Path("store")
HAC_LAGS = 5


def dm_statistic_and_p(difference: pd.Series) -> tuple[float, float]:
    """The t statistic and the two-sided p-value of mean(difference) = 0 with a Newey-West standard error."""
    import statsmodels.api as sm

    fit = sm.OLS(difference.to_numpy(dtype=float), pd.Series(1.0, index=difference.index).to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, help="a run name, e.g. xgb_F3_drv")
    parser.add_argument("--b", required=True, help="the run it is compared with")
    parser.add_argument("--group", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--forecast", type=int, required=True)
    parser.add_argument("--start", default="2016-03-01")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    walk_forward_id = pp.base_setting(args.train, args.forecast, args.start).walk_forward_id
    table = rs.runs_table(STORE)
    table = table[table["walk_forward_id"] == walk_forward_id]
    base = {name: run_id for run_id, name in table["name"].items()}
    missing = [n for n in (args.a, args.b) if n not in base]
    if missing:
        sys.exit(f"The walk-forward has no saved run named {missing}: {sorted(base)}")

    portfolios = pf.portfolios_table(STORE)
    portfolios = portfolios[portfolios["tag_group"] == args.group]
    links = pf.portfolio_runs_table(STORE)
    ids = list(portfolios.index)[: args.limit]
    out_dir = STORE / "Results" / args.group
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, daily, started = [], {tag: {} for tag in READINGS}, time.time()
    for n, pid in enumerate(ids, start=1):
        run_ids = {name: f"{base[name]}@{pid}" for name in (args.a, args.b)}
        known = set(links.loc[links["portfolio_id"] == pid, "run_id"])
        if not set(run_ids.values()) <= known:
            sys.exit(f"Portfolio {pid} has no saved runs {sorted(set(run_ids.values()) - known)}: run process_portfolios.py first.")
        runs = {name: rs.load_virtual_run(run_id, STORE) for name, run_id in run_ids.items()}
        for tag, reading in READINGS.items():
            dates = rs.common_dates(list(runs.values()), reading)
            var_dfs = {name: run.var_df(reading).loc[dates] for name, run in runs.items()}
            losses = core.run_comparison.daily_tick_losses(var_dfs, reading.var_confidence)
            difference = losses[args.a] - losses[args.b]
            daily[tag][pid] = difference
            for period, (start, end) in periods_of(dates).items():
                window = difference.loc[start:end]
                stat, p = dm_statistic_and_p(window)
                rows.append({
                    "portfolio_id": pid, "alpha": portfolios.loc[pid, "tag_alpha"], "n_eff": portfolios.loc[pid, "n_eff"], "reading": tag, "period": period, "days": len(window),
                    f"tick_loss_{args.a}": losses[args.a].loc[start:end].mean(), f"tick_loss_{args.b}": losses[args.b].loc[start:end].mean(),
                    "relative_difference": window.mean() / losses[args.b].loc[start:end].mean(), "dm_stat": stat, "dm_p": p,
                })
        if n % 10 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} portfolios, {time.time() - started:.0f} s", flush=True)

    result = pd.DataFrame(rows)
    stem = f"{args.a}_vs_{args.b}_train{args.train}_f{args.forecast}"
    result.to_csv(out_dir / f"{stem}_per_portfolio.csv", index=False)
    for tag, series in daily.items():
        pd.DataFrame(series).to_csv(out_dir / f"{args.a}_minus_{args.b}_daily_tick_loss_{tag}.csv")
    print(f"-> {out_dir}/{stem}_per_portfolio.csv and two daily difference files ({len(ids)} portfolios)")

    print(f"\n{args.a} against {args.b}; negative = {args.a} cheaper. DM on the average daily difference of the group (HAC, {HAC_LAGS} lags), and the per-portfolio counts.")
    alpha = portfolios.loc[ids, "tag_alpha"].astype(float)
    for tag in READINGS:
        frame = pd.DataFrame(daily[tag])
        for label, members in [("all", list(frame.columns))] + [(f"alpha {a:g}", list(alpha.index[alpha == a])) for a in sorted(alpha.unique(), reverse=True)]:
            mean_difference = frame[members].mean(axis=1)
            stat, p = dm_statistic_and_p(mean_difference)
            part = result[(result["reading"] == tag) & (result["period"] == "entire") & (result["portfolio_id"].isin(members))]
            cheaper = int((part["relative_difference"] < 0).sum())
            significant = int(((part["dm_p"] < 0.05) & (part["relative_difference"] < 0)).sum()), int(((part["dm_p"] < 0.05) & (part["relative_difference"] > 0)).sum())
            print(f"VaR {tag}% {label:10s}: mean relative difference {part['relative_difference'].mean():+.2%} | DM on the group average: t {stat:+.2f}, p {p:.3f} | "
                  f"{args.a} cheaper in {cheaper} of {len(members)}, significantly cheaper in {significant[0]}, significantly dearer in {significant[1]}")


if __name__ == "__main__":
    main()
