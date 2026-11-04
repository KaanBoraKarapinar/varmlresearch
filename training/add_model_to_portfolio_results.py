"""
Adds one more model to the results table of the portfolios of a group (the rows of store/Results/<group>/train<T>_f<F>_portfolio_results.csv), without computing the others again.

    python training/add_model_to_portfolio_results.py --model har_rv_log --group random200_v1 --train 756 --forecast 21
    ... --limit 3 --dry-run        # look before the full run: reads 3 portfolios, writes nothing

For every portfolio of the group, reading (VaR 95% / 99%) and period of the table it reads the model's saved run of the portfolio (nothing is trained) and the portfolio's
`garch_wide` (the reference of `rel_diff` and `dm_p`), and appends the model's rows with the same columns as the table (mean_tick_loss, rel_diff, dm_p, n_exceedances). The
table is refused when the model is already in it, and every portfolio and reading of the table must get its rows: nothing is added silently or in part.
The readings and the periods are those of the table (collect_portfolio_backtests.periods_of).
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
from training import process_portfolios as pp
from training.collect_portfolio_backtests import READINGS, periods_of

STORE = Path("store")
REFERENCE = "garch_wide"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="the name of the saved run to add, e.g. har_rv_log")
    parser.add_argument("--group", required=True, help="the portfolios with this tag group=...")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--forecast", type=int, required=True)
    parser.add_argument("--start", default="2016-03-01")
    parser.add_argument("--limit", type=int, help="only the first N portfolios (with --dry-run: to look at it)")
    parser.add_argument("--dry-run", action="store_true", help="compute, print, write nothing")
    args = parser.parse_args()

    results_path = STORE / "Results" / args.group / f"train{args.train}_f{args.forecast}_portfolio_results.csv"
    table = pd.read_csv(results_path)
    if args.model in set(table["model"]):
        sys.exit(f"{results_path} already has the rows of '{args.model}'.")
    walk_forward_id = pp.base_setting(args.train, args.forecast, args.start).walk_forward_id
    runs_table = rs.runs_table(STORE)
    base = {name: run_id for run_id, name in runs_table[runs_table["walk_forward_id"] == walk_forward_id]["name"].items()}
    missing = [name for name in (args.model, REFERENCE) if name not in base]
    if missing:
        sys.exit(f"The walk-forward {walk_forward_id} has no saved run named {missing}: {sorted(base)}")

    portfolios = pf.portfolios_table(STORE)
    portfolios = portfolios[portfolios["tag_group"] == args.group]
    ids = list(portfolios.index)[: args.limit]
    if args.limit is None and set(ids) != set(table["portfolio_id"]):
        sys.exit(f"The portfolios of group {args.group} are not those of {results_path.name} ({len(ids)} against {table['portfolio_id'].nunique()}): not adding rows to a part of them.")
    links = pf.portfolio_runs_table(STORE)
    known = set(links["run_id"])

    rows, started = [], time.time()
    for n, pid in enumerate(ids, start=1):
        run_ids = {name: f"{base[name]}@{pid}" for name in (args.model, REFERENCE)}
        if not set(run_ids.values()) <= known:
            sys.exit(f"Portfolio {pid} has no saved runs {sorted(set(run_ids.values()) - known)}: run apply_models_to_portfolio.py / process_portfolios.py first.")
        runs = {name: rs.load_virtual_run(run_id, STORE) for name, run_id in run_ids.items()}
        for tag, reading in READINGS.items():
            dates = rs.common_dates(list(runs.values()), reading)
            var_dfs = {name: run.var_df(reading).loc[dates] for name, run in runs.items()}
            for period, (start, end) in periods_of(dates).items():
                window = {name: frame.loc[start:end] for name, frame in var_dfs.items()}
                row = core.compare_to_baseline(window, REFERENCE, reading.var_confidence).loc[args.model]
                rows.append({
                    "portfolio_id": pid, "alpha": portfolios.loc[pid, "tag_alpha"], "n_eff": portfolios.loc[pid, "n_eff"], "reading": tag, "period": period,
                    "days": len(window[REFERENCE]), "model": args.model, "mean_tick_loss": row["mean_tick_loss"], "rel_diff": row["tick_loss_relative_diff"],
                    "dm_p": row["dm_p_value"], "n_exceedances": row["n_exceedances"],
                })
        if n % 10 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} portfolios, {time.time() - started:.0f} s", flush=True)

    added = pd.DataFrame(rows).astype({"reading": table["reading"].dtype})      # the csv reads 95 / 99 as numbers: the same type as the table's
    expected = table[table["model"] == REFERENCE][["portfolio_id", "reading", "period"]]
    expected = expected[expected["portfolio_id"].isin(ids)]
    got = added[["portfolio_id", "reading", "period"]]
    if len(expected.merge(got, how="outer", indicator=True).query("_merge != 'both'")):
        sys.exit("The rows of the new model do not cover exactly the portfolio / reading / period rows of the table (the periods differ): nothing written.")
    print(added[added["period"] == "entire"].groupby("reading")[["mean_tick_loss", "rel_diff", "n_exceedances"]].mean().to_string())
    if args.dry_run:
        print("dry run: nothing written")
        return
    pd.concat([table, added[table.columns]], ignore_index=True).to_csv(results_path, index=False)
    print(f"{len(added)} rows of '{args.model}' appended -> {results_path}")


if __name__ == "__main__":
    main()
