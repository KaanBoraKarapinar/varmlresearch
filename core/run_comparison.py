"""
Which of several runs is the best one? The decision rests on the daily tick loss of each run's VaR on the days they ALL
forecast (the same days, so the realized losses are the same and the comparison is paired), and it never rests on the
tick loss alone:

    1. validity     the run passes the Kupiec and both Christoffersen tests (a VaR with clustering violations is not trusted
                    just because its mean cost is low)
    2. significance the Model Confidence Set (Hansen, Lunde, Nason 2011; arch.bootstrap.MCS): the runs that cannot be told
                    apart from the best one at level alpha. It corrects for comparing many runs at once and for the days being
                    dependent (block bootstrap)
    3. consistency  the common days are cut into blocks (default 252 days): how many blocks each run wins and its mean rank
    + the pairwise matrix: row vs column, mean tick-loss difference and the Diebold-Mariano p-value (HAC standard error)
      with a Holm correction for the number of pairs

    select_best_run(var_dfs, var_confidence, test_confidence)  -> RunSelection (leaderboard, pairwise matrix, the best run)

The best set is the valid runs inside the MCS; the best run is the one of them with the lowest mean tick loss (None when no
run is valid). var_dfs: {run label: var_df} with the varForecast and loss columns, on the SAME dates and realized loss
(the same check as compare_to_baseline). Pure functions on frames: no dashboard, no store.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .statistical_var_backtest_utils import LOSS_COL, run_full_backtest_suite_single_window
from .var_level_and_volatility_utils import LOSS_AGREEMENT_TOLERANCE, daily_tick_loss


@dataclass(frozen=True)
class PairwiseMatrix:
    """Row run against column run. mean_difference = mean(tick loss of the row - tick loss of the column): negative = the row is better."""
    mean_difference: pd.DataFrame
    p_value: pd.DataFrame        # Diebold-Mariano p-value of each pair, Holm-corrected over all pairs
    verdict: pd.DataFrame        # 'better' / 'worse' / 'same' of the ROW run at the chosen alpha (the diagonal is '')


@dataclass(frozen=True)
class RunSelection:
    leaderboard: pd.DataFrame    # one row per run, best first (see select_best_run)
    pairwise: PairwiseMatrix
    best: str | None             # the valid run of the Model Confidence Set with the lowest mean tick loss
    alpha: float                 # the level of the Model Confidence Set and of the pairwise verdicts


def exceedance_overlap(var_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Do the runs exceed the VaR on the same days? On the days they all share:
      per run  exceedance_days, only_this_run (no other run exceeded that day), shared_with_another_run (at least one other
               did), excess_on_only_days (the sum of the difference between the realized loss and the VaR over the days
               ONLY this run exceeded: how much this run alone got wrong, the others' VaR was high enough on those days)
      matrix   row run vs column run: the share of the ROW run's exceedance days on which the COLUMN run exceeded too
               (the diagonal is NaN); 100% = the column run always exceeds when the row run does
    """
    names = list(var_dfs)
    first = var_dfs[names[0]]
    for name, var_df in var_dfs.items():
        if not var_df.index.equals(first.index):
            raise ValueError(f"var_dfs[{name!r}] is on different dates than var_dfs[{names[0]!r}]: use the days they all share.")
    exceeded = pd.DataFrame({n: v[LOSS_COL].astype(float) > v["varForecast"].astype(float) for n, v in var_dfs.items()})
    others = {n: exceeded.drop(columns=n).any(axis=1) for n in names}
    per_run = pd.DataFrame({
        "exceedance_days": exceeded.sum(),
        "only_this_run": pd.Series({n: int((exceeded[n] & ~others[n]).sum()) for n in names}),
        "shared_with_another_run": pd.Series({n: int((exceeded[n] & others[n]).sum()) for n in names}),
        "excess_on_only_days": pd.Series({
            n: float((var_dfs[n][LOSS_COL].astype(float) - var_dfs[n]["varForecast"].astype(float))[exceeded[n] & ~others[n]].sum()) for n in names
        }),
    })
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b and exceeded[a].any():
                matrix.loc[a, b] = float((exceeded[a] & exceeded[b]).sum() / exceeded[a].sum())
    return per_run, matrix


def daily_tick_losses(var_dfs: dict[str, pd.DataFrame], var_confidence: float) -> pd.DataFrame:
    """The daily tick loss of every run (columns = runs), after checking they are on the same dates and the same realized loss."""
    if len(var_dfs) < 2:
        raise ValueError("Need at least two runs to compare.")
    names = list(var_dfs)
    first = var_dfs[names[0]]
    for name, var_df in var_dfs.items():
        if not var_df.index.equals(first.index):
            raise ValueError(f"var_dfs[{name!r}] is on different dates than var_dfs[{names[0]!r}]: use the days they all share.")
        if (var_df[LOSS_COL].astype(float) - first[LOSS_COL].astype(float)).abs().max() > LOSS_AGREEMENT_TOLERANCE:
            raise ValueError(f"var_dfs[{name!r}] disagrees on the realized loss.")
    return pd.DataFrame({name: daily_tick_loss(var_df, var_confidence) for name, var_df in var_dfs.items()})


def _dm_p_value(difference: np.ndarray, hac_lags: int) -> float:
    """Two-sided p-value of mean(difference) = 0 with a Newey-West (HAC) standard error; NaN for an all-zero difference."""
    import statsmodels.api as sm

    if not np.any(difference):
        return float("nan")
    fit = sm.OLS(difference, np.ones(len(difference))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return float(fit.pvalues[0])


def pairwise_tick_loss_matrix(losses: pd.DataFrame, alpha: float = 0.10, hac_lags: int = 5) -> PairwiseMatrix:
    """Every run against every other on the daily tick-loss difference (Diebold-Mariano, Holm-corrected over the pairs)."""
    from statsmodels.stats.multitest import multipletests

    names = list(losses.columns)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    raw = np.array([_dm_p_value((losses[a] - losses[b]).to_numpy(), hac_lags) for a, b in pairs])
    corrected = np.full(len(pairs), np.nan)
    valid = ~np.isnan(raw)
    if valid.any():
        corrected[valid] = multipletests(raw[valid], method="holm")[1]

    mean_difference = pd.DataFrame(0.0, index=names, columns=names)
    p_value = pd.DataFrame(np.nan, index=names, columns=names)
    verdict = pd.DataFrame("", index=names, columns=names)
    for (a, b), p in zip(pairs, corrected):
        difference = float((losses[a] - losses[b]).mean())
        mean_difference.loc[a, b], mean_difference.loc[b, a] = difference, -difference
        p_value.loc[a, b] = p_value.loc[b, a] = p
        significant = (not np.isnan(p)) and p < alpha
        verdict.loc[a, b] = ("better" if difference < 0 else "worse") if significant else "same"
        verdict.loc[b, a] = ("worse" if difference < 0 else "better") if significant else "same"
    return PairwiseMatrix(mean_difference, p_value, verdict)


def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    """
    The Model Confidence Set of the runs (arch.bootstrap.MCS, stationary block bootstrap): one row per run with its
    p_value (the run with p_value >= alpha stays) and in_set. A run is in the set when it cannot be told apart from the best.
    """
    from arch.bootstrap import MCS

    class _MCS(MCS):
        # with pandas 3 the model-index column arch builds is an object column of numpy ints, and arch then fails to use it as an
        # indexer (an IndexError once many runs are eliminated): hand it plain ints
        def _format_pvalues(self, eliminated):
            first = lambda value: np.asarray(value).reshape(-1)[0]      # a scalar, or a one-element array
            return super()._format_pvalues([(int(first(index)), float(first(p_value))) for index, p_value in eliminated])

    # Runs with exactly the same daily losses (a baseline trained in two walk-forwards, same model and days) cannot be told apart:
    # their difference is always 0, the test statistic is undefined and arch fails. Test one run per group and give the whole
    # group its result.
    key = {name: losses[name].to_numpy().tobytes() for name in losses.columns}
    first_of_key = {}
    for name in losses.columns:
        first_of_key.setdefault(key[name], name)
    representative = {name: first_of_key[key[name]] for name in losses.columns}
    unique = list(dict.fromkeys(representative.values()))
    if len(unique) == 1:
        return pd.DataFrame({"p_value": 1.0, "in_set": True}, index=losses.columns)

    mcs = _MCS(losses[unique], size=alpha, reps=reps, seed=seed)
    mcs.compute()
    pvalues = mcs.pvalues["Pvalue"].reindex(unique)
    included = set(mcs.included)
    return pd.DataFrame({
        "p_value": [float(pvalues[representative[name]]) for name in losses.columns],
        "in_set": [representative[name] in included for name in losses.columns],
    }, index=losses.columns)


def block_ranking(losses: pd.DataFrame, block_days: int = 252) -> pd.DataFrame:
    """
    The common days cut into consecutive blocks of `block_days` days (the last one may be shorter): the mean tick loss of
    every run per block, then blocks_won (lowest in the block) and mean_block_rank (1 = best). Does not depend on the
    runs' own windows, so runs of different walk-forwards are comparable.
    """
    block = np.arange(len(losses)) // block_days
    per_block = losses.groupby(block).mean()
    ranks = per_block.rank(axis=1)
    return pd.DataFrame({
        "blocks_won": per_block.eq(per_block.min(axis=1), axis=0).sum().astype(int),
        "mean_block_rank": ranks.mean(),
        "n_blocks": len(per_block),
    })


def select_best_run(
    var_dfs: dict[str, pd.DataFrame], var_confidence: float, test_confidence: float = 0.95,
    baseline: str | None = None, alpha: float = 0.10, block_days: int = 252, reps: int = 1000, hac_lags: int = 5,
) -> RunSelection:
    """
    The runs ranked, with everything the choice rests on. Leaderboard columns:
      mean_tick_loss, valid (Kupiec + both Christoffersen pass), exceedance_rate, in_mcs / mcs_p_value (Model Confidence
      Set at `alpha`), blocks_won / mean_block_rank (consistency over blocks of `block_days` days),
      dm_p_vs_baseline (Diebold-Mariano p-value against `baseline`, if given), verdict:
        'best'        the valid run of the set with the lowest tick loss
        'best set'    valid and not significantly worse than the best
        'worse'       significantly worse than the best run of the set (outside the Model Confidence Set)
        'not valid'   fails one of the three tests
    """
    losses = daily_tick_losses(var_dfs, var_confidence)
    first = next(iter(var_dfs.values()))
    tests = {
        name: run_full_backtest_suite_single_window(var_df, first.index[0], first.index[-1], var_confidence, test_confidence)
        for name, var_df in var_dfs.items()
    }
    leaderboard = pd.DataFrame({
        "mean_tick_loss": losses.mean(),
        "valid": pd.Series({n: bool(t["kupiec_pof_pass"] and t["christoffersen_ind_pass"] and t["christoffersen_cc_pass"]) for n, t in tests.items()}),
        "exceedance_rate": pd.Series({n: float(t["exceedance_rate"]) for n, t in tests.items()}),
    })
    mcs = model_confidence_set(losses, alpha, reps)
    leaderboard = leaderboard.join(mcs.rename(columns={"p_value": "mcs_p_value", "in_set": "in_mcs"})).join(block_ranking(losses, block_days))
    if baseline is not None:
        leaderboard["dm_p_vs_baseline"] = {
            name: (float("nan") if name == baseline else _dm_p_value((losses[name] - losses[baseline]).to_numpy(), hac_lags))
            for name in losses.columns
        }

    candidates = leaderboard[leaderboard["valid"] & leaderboard["in_mcs"]]
    best = candidates["mean_tick_loss"].idxmin() if len(candidates) else None
    verdict = pd.Series("worse", index=leaderboard.index)
    verdict[leaderboard["in_mcs"]] = "best set"
    verdict[~leaderboard["valid"]] = "not valid"
    if best is not None:
        verdict[best] = "best"
    leaderboard["verdict"] = verdict
    return RunSelection(leaderboard.sort_values("mean_tick_loss"), pairwise_tick_loss_matrix(losses, alpha, hac_lags), best, alpha)
