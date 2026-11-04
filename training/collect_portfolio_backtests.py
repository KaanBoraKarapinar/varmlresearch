"""
The backtests (Kupiec POF, Christoffersen independence and conditional coverage) of the portfolios of one walk-forward, for every model, reading and period.

    python training/collect_portfolio_backtests.py --train 756 --forecast 21 --group random200_v1
    python training/collect_portfolio_backtests.py --train 756 --forecast 21 --group random200_v1 --limit 2     # look before the full run

Reads the portfolio runs kept by process_portfolios.py (nothing is trained) and writes one long csv, store/Results/<group>/train<T>_f<F>_portfolio_backtests.csv:
a row per portfolio, reading (95 / 99), period and model with the exceedances, the three test statistics, their p-values (chi-square: 1, 1 and 2 degrees of freedom)
and whether the test passed at the reading's test confidence. The readings and the periods are those of the results table of the same group: the VaR 95% / 99% with a
residual window of 252 and min periods 30; the whole forecast span, the COVID-19 decline, and the full COVID-19, 2022 bear market and 2025 tariff shock episodes
(core/analysis_periods.py).

About 6 seconds a portfolio (the VaR readings are computed from the saved residuals), so 200 portfolios take about 20 minutes. Output is a file, not the dashboard.
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
from scipy.stats import chi2

import core
from core import analysis_periods as ap
from core import portfolios as pf
from core import run_store as rs
from core import walk_forward as wf
from training import process_portfolios as pp

STORE = Path("store")
READINGS = {"95": wf.Reading(0.95, 252, 30, 0.95), "99": wf.Reading(0.99, 252, 30, 0.99)}
TESTS = {"kupiec_pof": 1, "christoffersen_ind": 1, "christoffersen_cc": 2}       # test -> degrees of freedom of its chi-square


def periods_of(dates: pd.DatetimeIndex) -> dict[str, tuple[str, str]]:
    """The whole span, the COVID-19 decline (30 days) and the full phase of every stress episode the days cover."""
    stress = [p for p in ap.stress_periods() if p.covered_by(dates)]
    out = {"entire": (str(dates[0].date()), str(dates[-1].date()))}
    out.update({"covid_decline": (p.start, p.end) for p in stress if p.episode == "COVID-19" and p.phase == "decline"})
    out.update({p.label: (p.start, p.end) for p in stress if p.phase == "full"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--forecast", type=int, required=True)
    parser.add_argument("--group", required=True, help="the portfolios with this tag group=...")
    parser.add_argument("--start", default="2016-03-01")
    parser.add_argument("--limit", type=int, help="only the first N portfolios")
    args = parser.parse_args()

    walk_forward_id = pp.base_setting(args.train, args.forecast, args.start).walk_forward_id
    portfolios = pf.portfolios_table(STORE)
    portfolios = portfolios[portfolios["tag_group"] == args.group]
    if portfolios.empty:
        sys.exit(f"No registered portfolio has the tag group={args.group}.")
    links = pf.portfolio_runs_table(STORE)
    links = links[links["run_id"].str.startswith(walk_forward_id)]
    ids = list(portfolios.index)[: args.limit]
    out = STORE / "Results" / args.group / f"train{args.train}_f{args.forecast}_portfolio_backtests.csv"
    print(f"{len(ids)} portfolios of group {args.group}, walk-forward {walk_forward_id}\n-> {out}", flush=True)

    rows, skipped, started = [], [], time.time()
    for n, pid in enumerate(ids, start=1):
        run_ids = list(links.loc[links["portfolio_id"] == pid, "run_id"])
        if not run_ids:
            skipped.append(pid)
            continue
        runs = {run.name: run for run in (rs.load_virtual_run(i, STORE) for i in run_ids)}
        for tag, reading in READINGS.items():
            dates = rs.common_dates(list(runs.values()), reading)
            var_dfs = {name: run.var_df(reading).loc[dates] for name, run in runs.items()}
            for period, (start, end) in periods_of(dates).items():
                for model, var_df in var_dfs.items():
                    window = var_df.loc[start:end]
                    suite = core.run_full_backtest_suite_single_window(window, window.index[0], window.index[-1], reading.var_confidence, reading.test_confidence)
                    row = {
                        "portfolio_id": pid, "alpha": portfolios.loc[pid, "tag_alpha"], "n_eff": portfolios.loc[pid, "n_eff"], "reading": tag, "period": period,
                        "days": suite["n_days"], "model": model, "n_exceedances": suite["n_exceedances"], "exceedance_rate": suite["exceedance_rate"],
                        "expected_exceedance_rate": suite["expected_exceedance_rate"],
                    }
                    for test, degrees in TESTS.items():
                        stat = suite.get(f"{test}_stat")
                        row[f"{test}_stat"] = stat
                        row[f"{test}_p"] = float(chi2.sf(stat, degrees)) if stat is not None and pd.notna(stat) else float("nan")
                        row[f"{test}_pass"] = suite.get(f"{test}_pass")
                    rows.append(row)
        if n % 10 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} portfolios, {time.time() - started:.0f} s", flush=True)
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{frame['portfolio_id'].nunique()} portfolios, {len(frame)} rows -> {out} | without runs: {len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
