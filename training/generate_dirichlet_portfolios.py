"""
Draws random portfolios (weights over the 23 tickers) from Dirichlet distributions and registers them in the portfolio registry. Done once.

    python training/generate_dirichlet_portfolios.py --group random200_v1                              # 4 alphas x 50 portfolios
    python training/generate_dirichlet_portfolios.py --group random200_v1 --alphas 5 1 --per-alpha 10  # fewer
    python training/generate_dirichlet_portfolios.py --group random200_v1 --dry-run                    # only shows what would be registered

The registry (store/Portfolios/portfolios.jsonl) belongs to no walk-forward: the same portfolios are read by every walk-forward
(training/process_portfolios.py). Each portfolio carries tags: the group, the generator, the alpha, the draw number and the seed, so that
process_portfolios.py can select them (--group random200_v1, --tag alpha=0.5).

The same command always draws the same portfolios (the generator of draw k of alpha number a is seeded with [seed, a, k]): running it again
registers nothing new. A different --seed draws other portfolios, so give another --group with it.

Alpha is the concentration of the Dirichlet distribution: a large alpha gives near equal weights, a small alpha puts most of the weight on a few
tickers (alpha 0.1: about 3 tickers carry the portfolio).
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os

ROOT = Path(__file__).resolve().parent.parent       # the project root: the scripts live one folder down
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from core import portfolios as pf
from core import run_store as rs
from core.virtual_runs import portfolio_id
from training import train_23_experiment as e

GENERATOR = "dirichlet"


def draw_weights(alphas: list[float], per_alpha: int, seed: int):
    """Yield (alpha, draw number, weights): the weights of draw k of alpha number a come from a generator seeded with [seed, a, k]."""
    tickers = e.TICKERS_23
    for a, alpha in enumerate(alphas):
        for k in range(per_alpha):
            rng = np.random.default_rng([seed, a, k])
            yield alpha, k, dict(zip(tickers, (float(w) for w in rng.dirichlet(alpha * np.ones(len(tickers))))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", required=True, help="the name of this set of portfolios, stored as the tag 'group' (e.g. random200_v1)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[5.0, 1.0, 0.5, 0.1], help="Dirichlet alphas (default: 5 1 0.5 0.1)")
    parser.add_argument("--per-alpha", type=int, default=50, help="portfolios per alpha (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="the seed of the whole draw (default: 0)")
    parser.add_argument("--root", type=Path, default=rs.STORE_ROOT, help=f"the store root (default: {rs.STORE_ROOT})")
    parser.add_argument("--dry-run", action="store_true", help="draw and show, register nothing")
    args = parser.parse_args()
    if args.per_alpha < 1 or any(a <= 0 for a in args.alphas):
        parser.error("--per-alpha must be at least 1 and every alpha must be positive.")

    # look at everything first: a portfolio that is already registered with other tags (typically: the same seed under another group name, which draws
    # the very same portfolios) stops the script before anything is written
    registry = pf.portfolios_by_id(args.root)
    drawn = []
    for alpha, draw, weights in draw_weights(args.alphas, args.per_alpha, args.seed):
        tags = {"group": args.group, "generator": GENERATOR, "alpha": alpha, "draw": draw, "seed": args.seed}
        drawn.append((alpha, weights, tags, registry.get(portfolio_id(weights))))
    conflicts = [(tags, record) for _, _, tags, record in drawn if record is not None and record["tags"] != tags]
    if conflicts:
        tags, record = conflicts[0]
        raise SystemExit(
            f"{len(conflicts)} of the {len(drawn)} drawn portfolios are already registered with other tags, nothing was written. Example: draw {tags['draw']} of "
            f"alpha {tags['alpha']:g} is {record['name']!r}, registered with {record['tags']}. The same seed and alphas draw the same portfolios whatever "
            f"the group is called: use another --seed for new portfolios, or the group the portfolios are registered with ({record['tags'].get('group')!r})."
        )

    new, existing, rows = 0, 0, []
    for alpha, weights, tags, record in drawn:
        if not args.dry_run:
            pf.register_portfolio(args.root, weights, name_prefix=f"Dir{alpha:g}", tags=tags)
        existing += record is not None
        new += record is None
        rows.append({"alpha": alpha, "n_eff": pf.effective_number(weights), "max_weight": max(weights.values()), "new": record is None})

    print(f"group {args.group!r}: {len(rows)} portfolios drawn, {new} new, {existing} already registered" + (" (dry run: nothing written)" if args.dry_run else ""))
    print(f"{'alpha':>7} {'portfolios':>11} {'neff min / mean / max':>26} {'largest weight, mean':>22}")
    for alpha in args.alphas:
        part = [r for r in rows if r["alpha"] == alpha]
        n_eff = np.array([r["n_eff"] for r in part])
        print(f"{alpha:>7g} {len(part):>11} {n_eff.min():>10.1f} {n_eff.mean():>7.1f} {n_eff.max():>7.1f} {np.mean([r['max_weight'] for r in part]):>22.2f}")
    if not args.dry_run:
        print(f"registry: {args.root / pf.PORTFOLIOS_DIR / pf.PORTFOLIOS_FILE}")


if __name__ == "__main__":
    main()
