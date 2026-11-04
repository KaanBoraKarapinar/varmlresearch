"""
Runs the registered portfolios through ONE walk-forward: for every selected portfolio it keeps the portfolio's runs of that walk-forward in the store.

    python training/process_portfolios.py --train 756 --forecast 21 --group random200_v1
    python training/process_portfolios.py --train 756 --forecast 21 --tag group=random200_v1 --tag alpha=0.5
    python training/process_portfolios.py --train 756 --forecast 21 --group random200_v1 --limit 3 --dry-run       # look before running
    python training/process_portfolios.py --train 252 --forecast 21 --group random200_v1 --models xgb_long ewma    # only these saved runs

The portfolios come from the registry (training/generate_dirichlet_portfolios.py fills it), selected by their tags: --group g is the same as
--tag group=g, several tags must all match, and at least one is needed (or --all). Only portfolios of exactly the 23 tickers are used.

The walk-forward is the one of the 23 tickers with equal weights (the same inputs as train_23_experiment.py: prices, additional feature data,
start date), with the --train / --forecast bars given. Its runs have to be saved already (they are not trained here):

  * the saved runs that can be read for other weights (the long xgboost runs and ewma: their per-ticker forecasts do not depend on the weights;
    --models picks some of them, by run name, otherwise every such run of the walk-forward) are read for each portfolio. Nothing is trained: the
    portfolio's residuals are kept under <walk-forward>/VirtualRuns/ (run_store.save_virtual_run);
  * the baselines (--baselines, default garch_wide historical; --baselines alone for none) depend on the portfolio's own return, so they are
    trained again for every portfolio, a second or two each, and kept the same way (run_store.save_baseline_runs).

What is kept for a portfolio is never written twice: run the script again after an interruption, or with more portfolios, and it carries on.
The portfolios are written in chunks (--chunk, default 50): every model gets ONE new file per chunk with all the portfolios of the chunk in it, so
200 portfolios and 8 models are about 32 files (a file per portfolio and model would be 1600).
"""
import argparse
import datetime
import json
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

from core import portfolios as pf
from core import run_store as rs
from core import virtual_runs as vr
from core import walk_forward as wf
from training import train_23_experiment as e


def base_setting(train_bars: int, forecast_bars: int, start_date: str, tickers: list[str] | None = None) -> wf.WalkForwardSetting:
    """
    The walk-forward of the 23 tickers (or of `tickers`) with equal weights: the inputs of train_23_experiment.setting_of with other train / forecast bars.
    Built from the inputs, not by dataclasses.replace of another setting: that normalises the weights a second time and moves 1/23 by one unit in the last digit,
    which is another walk-forward id's inputs for run_store (save_walk_forward refuses).
    """
    tickers = e.TICKERS_23 if tickers is None else tickers
    return wf.WalkForwardSetting(
        e.prices, tickers, {t: 1 / len(tickers) for t in tickers}, train_bars=train_bars, forecast_bars=forecast_bars,
        unit=e.UNIT, data_path=e.DATA_PATH, start_date=start_date, additional_feature_sources=e.SOURCES,
    )


def parse_tag(text: str) -> tuple[str, object]:
    """'alpha=0.5' -> ('alpha', 0.5); a value that is not JSON stays text ('group=random200_v1')."""
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"{text!r} is not key=value.")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def select_portfolios(root: Path, filters: list[tuple[str, object]], tickers: list[str]) -> pd.DataFrame:
    """The rows of the portfolio table whose tags all match, for exactly these tickers (a portfolio of other tickers cannot be read here)."""
    table = pf.portfolios_table(root)
    if table.empty:
        raise SystemExit(f"no portfolio is registered under {root}: run training/generate_dirichlet_portfolios.py first.")
    for key, value in filters:
        column = f"tag_{key}"
        if column not in table.columns:
            known = sorted(c.removeprefix("tag_") for c in table.columns if c.startswith("tag_"))
            raise SystemExit(f"no registered portfolio has the tag {key!r}; the tags in the registry: {known}.")
        table = table[table[column] == value]
    same_tickers = table["n_tickers"].eq(len(tickers)) & table.reindex(columns=[f"w_{t}" for t in tickers]).notna().all(axis=1)
    return table[same_tickers]


def find_base_runs(root: Path, walk_forward_id: str, wanted: list[str] | None) -> dict[str, wf.Run]:
    """
    The saved runs of the walk-forward that can be read for other weights, by run name. wanted=None: all of them (the others are listed with
    the reason); wanted: exactly these, and one that is not saved or cannot be re-weighted stops the script.
    """
    table = rs.runs_table(root)
    table = table[table["walk_forward_id"] == walk_forward_id] if "walk_forward_id" in table.columns else table.iloc[0:0]
    if table.empty:
        raise SystemExit(
            f"no run is saved for this walk-forward ({walk_forward_id}) under {root}: train and save its runs first "
            "(for the equal-weight 23 tickers, with the same train / forecast bars)."
        )
    runs = {row["name"]: rs.load_walk_forward_run(run_id, root) for run_id, row in table.iterrows()}
    blockers = {name: wf.reweight_blocker(run) for name, run in runs.items()}
    if wanted is not None:
        missing = [n for n in wanted if n not in runs]
        blocked = {n: blockers[n] for n in wanted if n in blockers and blockers[n]}
        if missing or blocked:
            raise SystemExit(f"cannot use --models: not saved {missing}; cannot be re-weighted {blocked}. Saved runs: {sorted(runs)}.")
        return {n: runs[n] for n in wanted}
    for name, blocker in blockers.items():
        if blocker:
            print(f"  not read for other weights: {name} ({blocker})")
    return {name: runs[name] for name, blocker in blockers.items() if not blocker}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=int, required=True, help="train bars of the walk-forward (e.g. 756)")
    parser.add_argument("--forecast", type=int, required=True, help="forecast bars of the walk-forward = the days between two trainings (e.g. 21)")
    parser.add_argument("--start", default="2016-03-01", help="start date of the walk-forward (default 2016-03-01: the RV data starts 2016-01-04)")
    parser.add_argument("--group", help="select the portfolios with this tag 'group'")
    parser.add_argument("--tag", action="append", type=parse_tag, default=[], metavar="KEY=VALUE", help="select by a tag (repeatable, all must match), e.g. alpha=0.5")
    parser.add_argument("--all", action="store_true", help="select every registered portfolio (needed when no --group / --tag is given)")
    parser.add_argument("--models", nargs="+", help="names of the saved runs to read for the portfolios (default: every run of the walk-forward that can be)")
    parser.add_argument("--baselines", nargs="*", default=["garch_wide", "historical"], help="baselines trained for every portfolio (default garch_wide historical; none: --baselines)")
    parser.add_argument("--limit", type=int, help="only the first N selected portfolios (to try the script)")
    parser.add_argument("--chunk", type=int, default=50, help="portfolios written together (default 50): one file per model per chunk, and what an interruption can lose")
    parser.add_argument("--root", type=Path, default=rs.STORE_ROOT, help=f"the store root (default {rs.STORE_ROOT})")
    parser.add_argument("--dry-run", action="store_true", help="show what would be done, write nothing")
    args = parser.parse_args()

    filters = ([("group", args.group)] if args.group else []) + args.tag
    if not filters and not args.all:
        parser.error("give --group, --tag key=value, or --all: the script never takes every portfolio by accident.")
    if args.chunk < 1:
        parser.error("--chunk must be at least 1.")
    unknown = [b for b in args.baselines if b not in wf.BASELINE_RECIPES]
    if unknown:
        parser.error(f"unknown baselines {unknown}; known: {list(wf.BASELINE_RECIPES)}.")

    setting = base_setting(args.train, args.forecast, args.start)
    walk_forward_id = setting.walk_forward_id
    windows = setting.window_table()
    print(f"walk-forward: train {args.train} / forecast {args.forecast}, {len(windows)} windows, forecast days "
          f"{windows['forecast_start'].iloc[0].date()} .. {windows['forecast_end'].iloc[-1].date()}\n  {walk_forward_id}")

    table = select_portfolios(args.root, filters, e.TICKERS_23)
    if args.limit:
        table = table.iloc[: args.limit]
    if table.empty:
        raise SystemExit(f"no portfolio of the 23 tickers matches {filters}.")
    bases = find_base_runs(args.root, walk_forward_id, args.models)
    models = [*bases, *args.baselines]
    print(f"{len(table)} portfolios selected by {filters or 'all'}; read from saved runs: {list(bases)}; trained per portfolio: {args.baselines}")

    kept = {model: set(vr.portfolio_ids(args.root, walk_forward_id, model)) for model in models}      # read once, not once per portfolio
    todo = [p for p in table.index if not all(p in kept[model] for model in models)]
    print(f"{len(table) - len(todo)} of them are already complete for these models, {len(todo)} to do, in {-(-len(todo) // args.chunk)} chunk(s) of {args.chunk}.")
    if args.dry_run or not todo:
        print("dry run: nothing written." if args.dry_run else "nothing to do.")
        return

    registry = pf.portfolios_by_id(args.root)
    batch = {"script": "process_portfolios", "argv": sys.argv[1:], "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
    start = time.time()
    for first in range(0, len(todo), args.chunk):
        chunk, tick = todo[first: first + args.chunk], time.time()
        # the portfolios of a chunk are written together: one new file per model for the whole chunk (what an interruption loses is one chunk)
        rs.save_virtual_runs(
            [wf.ReweightedVirtualRun(base, registry[p]["weights"], registry[p]) for p in chunk for base in bases.values()], args.root, batch=batch,
        )
        if args.baselines:
            rs.save_baseline_runs_for(setting, [registry[p] for p in chunk], args.root, names=tuple(args.baselines), batch=batch)
        print(f"{first + len(chunk):4d}/{len(todo)}  chunk of {len(chunk)} portfolios, {time.time() - tick:6.1f} s ({(time.time() - tick) / len(chunk):.1f} s per portfolio)", flush=True)
    print(f"done: {len(todo)} portfolios written, {len(table) - len(todo)} were already there, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
