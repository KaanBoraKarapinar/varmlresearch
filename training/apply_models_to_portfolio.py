"""
The saved models of one walk-forward read for ONE portfolio that is a part of its tickers (nothing is trained).

    python training/apply_models_to_portfolio.py --tickers AAPL JPM XOM KO --name "equal 4 · AAPL JPM XOM KO" --group equal4_from23 --train 756 --forecast 21

The models trained on the 23 tickers are read for the portfolio that holds only the given tickers, equal weights and 0 for the others (core.walk_forward.ReweightedVirtualRun:
the per-ticker variances of the stored forecasts are put into the covariance of the portfolio's weights, so the covariance is the EWMA one of those tickers). The portfolio is
registered (core.portfolios, tag group=<group>) and every run of the walk-forward that can be read for other weights is kept for it, like the portfolios of process_portfolios.py.
It is what "a model trained on a larger universe applied to a smaller portfolio" means: the models of the 4-ticker portfolio itself are a walk-forward of their own.
"""
import argparse
import datetime
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core import portfolios as pf
from core import run_store as rs
from core import walk_forward as wf
from training import process_portfolios as pp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers", nargs="+", required=True, help="the tickers the portfolio holds (equal weights); a part of the walk-forward's tickers")
    parser.add_argument("--name", required=True, help="the name the portfolio is registered and shown under")
    parser.add_argument("--group", required=True, help="the tag group=... of the portfolio")
    parser.add_argument("--train", type=int, default=756)
    parser.add_argument("--forecast", type=int, default=21)
    parser.add_argument("--start", default="2016-03-01")
    parser.add_argument("--root", type=Path, default=rs.STORE_ROOT)
    args = parser.parse_args()

    setting = pp.base_setting(args.train, args.forecast, args.start)
    unknown = [t for t in args.tickers if t not in setting.tickers]
    if unknown:
        raise SystemExit(f"{unknown} are not tickers of the walk-forward {setting.tickers}.")
    weights = {t: (1 / len(args.tickers) if t in args.tickers else 0.0) for t in setting.tickers}
    record = pf.register_portfolio(args.root, weights, name=args.name, tags={"group": args.group, "generator": "equal weights of a part of the tickers"})
    print(f"portfolio {record['portfolio_id']}: {record['name']} (n_eff {record['n_eff']:.1f})")
    bases = pp.find_base_runs(args.root, setting.walk_forward_id, None)
    batch = {"script": "apply_models_to_portfolio", "argv": sys.argv[1:], "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
    run_ids = rs.save_virtual_runs([wf.ReweightedVirtualRun(base, record["weights"], record) for base in bases.values()], args.root, batch=batch)
    print(f"{len(run_ids)} runs kept for the portfolio ({', '.join(bases)}):")
    for run_id in run_ids:
        print("  ", run_id)


if __name__ == "__main__":
    main()
