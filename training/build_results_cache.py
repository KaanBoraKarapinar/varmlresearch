"""
Make the results cache (results_cache/, see core/results_cache.py): the files of the run store that the Results and Experiments pages need to show the saved
experiments, with the MLflow index of those runs only. The dashboard shows them without the password and without the bucket.

    python training/build_results_cache.py --dry-run     what would be written (a size estimate)
    python training/build_results_cache.py                write results_cache/ (it must not exist: remove it first)

Start it again after a new experiment is saved (store/experiments/) or a saved run changes, and commit results_cache/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import results_cache as rc

MLFLOW_EXPERIMENT = "walk_forward_horizon"      # the experiment every training script logs to (the dashboard's too)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=ROOT / "store")
    parser.add_argument("--index", type=Path, default=ROOT / "mlruns.db", help="the MLflow index to copy and trim")
    parser.add_argument("--out", type=Path, default=ROOT / rc.CACHE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = rc.build(args.store, args.out, {rc.CACHED_READING}, dry_run=args.dry_run)
    print(f"{len(report.experiments)} experiments, {report.runs} runs + {report.virtual_runs} virtual runs: {report.files} files, {report.bytes / 1e6:,.1f} MB")
    if args.dry_run:
        return
    shutil.copyfile(args.index, args.out / "mlruns.db")
    kept, deleted = rc.trim_index(args.out / "mlruns.db", report.run_ids, MLFLOW_EXPERIMENT)
    print(f"index: {kept} runs kept, {deleted} deleted; {(args.out / 'mlruns.db').stat().st_size / 1e6:.1f} MB")
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{args.out}: {total / 1e6:,.1f} MB")


if __name__ == "__main__":
    main()
