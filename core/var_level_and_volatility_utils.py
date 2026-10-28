"""
Descriptive VaR quality suite: how HIGH a model puts its VaR (conservativeness) and how much the VaR MOVES
(volatility), plus the tick (pinball) loss that scores both together.

No hypothesis tests here -- those live in statistical_var_backtest_utils.py. The statistical tests only say
whether the exceedance rate/pattern fits the VaR confidence; this suite shows what the model costs
(a needlessly high VaR) and how jumpy it is, which the tests cannot see (a VaR of 100% passes the
"too few exceedances" side trivially, so read these numbers together with exceedance_rate).

Loss convention (same as the rest of the project): `loss` = -portfolio_return, `varForecast` is positive, an
exceedance is loss > varForecast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .statistical_var_backtest_utils import LOSS_COL, VAR_COL, _period_var_df

# Runs that are compared must have the same realized loss. The same portfolio read from two downloads of the prices differs in the
# last digits (about 1e-6 of a return), another portfolio differs a thousand times that on some day: this tells them apart.
LOSS_AGREEMENT_TOLERANCE = 1e-5

if TYPE_CHECKING:
    from .GroundTruthCalendar import GroundTruthCalendar


def run_level_and_volatility_suite_single_window(forecast_df: pd.DataFrame, var_confidence: float) -> dict:
    """
    One contiguous window (date index, "varForecast" + "loss" columns) -> a flat dict.

    Tick (pinball) loss of a quantile forecast at level a = var_confidence, u = loss - VaR:
        rho_a(u) = a * u            if u > 0   (exceedance: the VaR was too LOW)
                 = (1 - a) * (-u)   otherwise  (the VaR was higher than needed: conservative)
    It is a proper scoring rule (the true quantile minimizes it), so it is the headline score; it is reported
    split in these two parts, which add up to `mean_tick_loss`.
    """
    if not 0 < var_confidence < 1:
        raise ValueError(f"var_confidence must be in (0, 1), got {var_confidence!r}")
    var = forecast_df[VAR_COL].to_numpy(dtype=float)
    loss = forecast_df[LOSS_COL].to_numpy(dtype=float)
    n = len(var)
    if n < 2:
        raise ValueError("Need at least 2 days to measure the VaR's day-to-day change.")

    exceeded = loss > var
    above_var = np.maximum(loss - var, 0.0)      # how far the loss went above the VaR (0 on non-exceedance days)
    below_var = np.maximum(var - loss, 0.0)      # unused buffer (0 on exceedance days)

    tick_exceedance_part = float((var_confidence * above_var).mean())
    tick_overshoot_part = float(((1 - var_confidence) * below_var).mean())

    mean_var = float(var.mean())
    realized_quantile = float(np.quantile(loss, var_confidence))   # hindsight yardstick: the window's own quantile
    daily_change = np.abs(np.diff(var))
    var_p05, var_p95 = np.quantile(var, [0.05, 0.95])

    return {
        "var_confidence": var_confidence,
        "n_days": n,
        "n_exceedances": int(exceeded.sum()),
        "exceedance_rate": float(exceeded.mean()),
        "expected_exceedance_rate": 1 - var_confidence,
        # score
        "mean_tick_loss": tick_exceedance_part + tick_overshoot_part,
        "tick_loss_exceedance_part": tick_exceedance_part,
        "tick_loss_overshoot_part": tick_overshoot_part,
        # conservativeness (level)
        "mean_var": mean_var,
        "median_var": float(np.median(var)),
        "realized_loss_quantile": realized_quantile,
        "var_to_realized_quantile_ratio": mean_var / realized_quantile if realized_quantile > 0 else np.nan,
        "mean_slack_non_exceedance": float(below_var[~exceeded].mean()) if (~exceeded).any() else np.nan,
        # tail severity
        "mean_exceedance_size": float(above_var[exceeded].mean()) if exceeded.any() else np.nan,
        "max_exceedance_size": float(above_var[exceeded].max()) if exceeded.any() else np.nan,
        # volatility of the VaR
        "var_std": float(var.std(ddof=1)),
        "var_cv": float(var.std(ddof=1) / mean_var) if mean_var > 0 else np.nan,
        "mean_abs_daily_change": float(daily_change.mean()),
        "relative_mean_abs_daily_change": float(daily_change.mean() / mean_var) if mean_var > 0 else np.nan,
        "var_p95_p5_ratio": float(var_p95 / var_p05) if var_p05 > 0 else np.nan,
        "var_max_min_ratio": float(var.max() / var.min()) if var.min() > 0 else np.nan,
    }


def exceedance_days_by_model(var_dfs: dict[str, pd.DataFrame], labels: bool = True) -> pd.DataFrame:
    """
    Do the models blow up on the same days? One row per day on which AT LEAST ONE model has an exceedance
    (loss > varForecast), one column per model, plus the realized `loss` and `n_models` (how many
    models exceeded that day: n_models == len(var_dfs) means every model missed it, 1 means only one did).
    The model columns read "exceeded" / "not exceeded" (labels=True) or True / False (labels=False).

    var_dfs: {model_name: var_df} of the SAME calendar period (same dates and same realized loss), as
    calendar.var_from_residual_bank_*_period returns them. Both are checked and raise otherwise.
    """
    if not var_dfs:
        raise ValueError("var_dfs is empty.")
    names = list(var_dfs)
    first = var_dfs[names[0]]
    for name, var_df in var_dfs.items():
        if not var_df.index.equals(first.index):
            raise ValueError(f"var_dfs[{name!r}] is on different dates than var_dfs[{names[0]!r}]: use the same period.")
        if (var_df[LOSS_COL].astype(float) - first[LOSS_COL].astype(float)).abs().max() > LOSS_AGREEMENT_TOLERANCE:
            raise ValueError(f"var_dfs[{name!r}] disagrees on the realized loss: they must come from the same calendar.")

    exceeded = pd.DataFrame({
        name: var_df[LOSS_COL].astype(float) > var_df[VAR_COL].astype(float) for name, var_df in var_dfs.items()
    })
    table = exceeded.replace({True: "exceeded", False: "not exceeded"}) if labels else exceeded.copy()
    table.insert(0, "loss", first[LOSS_COL].astype(float))
    table["n_models"] = exceeded.sum(axis=1)
    return table.loc[table["n_models"] > 0].sort_index()


def exceedance_day_details(
    var_dfs: dict[str, pd.DataFrame], days, labels: bool = True,
) -> pd.DataFrame:
    """
    For the given days (e.g. the exceedance_days_by_model table, or any list of dates) and every model: the
    realized loss, the VaR forecast and their difference. One row per (day, model), indexed by (Date, model).

    Columns: loss, var_forecast, difference (= loss - VaR; positive = the VaR was exceeded by that much),
    loss_over_var (loss / VaR), exceeded ("exceeded" / "not exceeded", or True / False with labels=False).
    `days` is a DataFrame (its index is used), a DatetimeIndex or a list of dates; each must be in every var_df.
    """
    if isinstance(days, pd.DataFrame):
        days = days.index
    days = pd.DatetimeIndex(days)

    parts = []
    for name, var_df in var_dfs.items():
        missing = days.difference(var_df.index)
        if len(missing):
            raise KeyError(f"var_dfs[{name!r}] has no row for: {[d.date().isoformat() for d in missing]}")
        part = var_df.loc[days, [LOSS_COL, VAR_COL]].astype(float).rename(columns={VAR_COL: "var_forecast"})
        part["difference"] = part[LOSS_COL] - part["var_forecast"]
        part["loss_over_var"] = part[LOSS_COL] / part["var_forecast"]
        part["exceeded"] = part["difference"] > 0
        part["model"] = name
        parts.append(part.rename(columns={LOSS_COL: "loss"}))

    table = pd.concat(parts)
    if labels:
        table["exceeded"] = table["exceeded"].map({True: "exceeded", False: "not exceeded"})
    table.index.name = "Date"
    table = table.reset_index().set_index(["Date", "model"])
    model_order = {name: i for i, name in enumerate(var_dfs)}
    order = np.lexsort((table.index.get_level_values("model").map(model_order), table.index.get_level_values("Date")))
    return table.iloc[order][["loss", "var_forecast", "difference", "loss_over_var", "exceeded"]]


def exceedance_day_summary(var_dfs: dict[str, pd.DataFrame], days) -> pd.DataFrame:
    """
    One row per day (in the order of `days`) across all the models, computed from exceedance_day_details:
    loss (realized), n_models, n_missed (models whose VaR the loss exceeded), and the average / minimum /
    maximum of difference = loss - VaR over the models (positive = exceeded). plot_exceedance_days_focus
    reads its day titles and ring labels from this table.
    """
    if isinstance(days, pd.DataFrame):
        days = days.index
    days = pd.DatetimeIndex(days)

    details = exceedance_day_details(var_dfs, days, labels=False)
    by_day = details.groupby(level="Date")
    summary = pd.DataFrame({
        "loss": by_day["loss"].first(),
        "n_models": by_day.size(),
        "n_missed": by_day["exceeded"].sum().astype(int),
        "avg_difference": by_day["difference"].mean(),
        "min_difference": by_day["difference"].min(),
        "max_difference": by_day["difference"].max(),
    })
    return summary.loc[days.unique()]


def daily_tick_loss(forecast_df: pd.DataFrame, var_confidence: float) -> pd.Series:
    """
    The tick (pinball) loss of every day (the mean of this series is run_level_and_volatility_suite_single_window's
    mean_tick_loss): a * (loss - VaR) on an exceedance day, (1 - a) * (VaR - loss) otherwise, a = var_confidence.
    """
    if not 0 < var_confidence < 1:
        raise ValueError(f"var_confidence must be in (0, 1), got {var_confidence!r}")
    var = forecast_df[VAR_COL].astype(float)
    u = forecast_df[LOSS_COL].astype(float) - var
    return pd.Series(np.where(u > 0, var_confidence * u, -(1 - var_confidence) * u), index=forecast_df.index, name="tick_loss")


def qlike(variance, realized_squared_return, floor: float = 1e-10) -> float:
    """
    The QLIKE loss of variance forecasts against the realized squared return r^2: mean(r^2/h - ln(r^2/h) - 1), h = the
    forecast. Lower is better, 0 would be a forecast equal to r^2 every day, a too LOW forecast is punished more than a
    too high one. It does not depend on the unit or the scale of the returns, so it is comparable between models
    (unlike a squared error in a model's own target units).
    """
    ratio = np.maximum(np.asarray(realized_squared_return, dtype=float), floor) / np.asarray(variance, dtype=float)
    return float(np.mean(ratio - np.log(ratio) - 1))


def compare_to_baseline(
    var_dfs: dict[str, pd.DataFrame], baseline: str, var_confidence: float, hac_lags: int = 5,
) -> pd.DataFrame:
    """
    Every run against one baseline run (e.g. a GARCH), on the same days. One row per run, indexed by name.

    var_dfs: {run_name: var_df} on the SAME dates and realized loss (checked, like exceedance_days_by_model).
    Columns (difference = run - baseline, so NEGATIVE means the run is better than the baseline):
        mean_tick_loss, tick_loss_diff, tick_loss_relative_diff (diff / baseline's mean tick loss),
        days_better_share (share of days the run's tick loss is below the baseline's),
        n_exceedances, exceedance_diff,
        dm_t_stat, dm_p_value: Diebold-Mariano test, the mean of the daily tick-loss difference tested against 0
            with a HAC (Newey-West, `hac_lags`) standard error, two-sided; NaN for the baseline itself.
    A small p-value says the difference is not just day-to-day noise; it does not say the run is correct.
    """
    if baseline not in var_dfs:
        raise KeyError(f"baseline {baseline!r} is not in var_dfs: {list(var_dfs)}")
    base_df = var_dfs[baseline]
    for name, var_df in var_dfs.items():
        if not var_df.index.equals(base_df.index):
            raise ValueError(f"var_dfs[{name!r}] is on different dates than the baseline {baseline!r}: use the same period.")
        if (var_df[LOSS_COL].astype(float) - base_df[LOSS_COL].astype(float)).abs().max() > LOSS_AGREEMENT_TOLERANCE:
            raise ValueError(f"var_dfs[{name!r}] disagrees with the baseline on the realized loss.")

    import statsmodels.api as sm

    base_tick = daily_tick_loss(base_df, var_confidence)
    base_exceed = int((base_df[LOSS_COL].astype(float) > base_df[VAR_COL].astype(float)).sum())
    rows = {}
    for name, var_df in var_dfs.items():
        tick = daily_tick_loss(var_df, var_confidence)
        diff = tick - base_tick
        n_exceed = int((var_df[LOSS_COL].astype(float) > var_df[VAR_COL].astype(float)).sum())
        row = {
            "mean_tick_loss": float(tick.mean()),
            "tick_loss_diff": float(diff.mean()),
            "tick_loss_relative_diff": float(diff.mean() / base_tick.mean()),
            "days_better_share": float((diff < 0).mean()),
            "n_exceedances": n_exceed,
            "exceedance_diff": n_exceed - base_exceed,
            "dm_t_stat": np.nan,
            "dm_p_value": np.nan,
        }
        if name == baseline:
            row["days_better_share"] = np.nan
        elif diff.abs().max() > 0:
            fit = sm.OLS(diff.to_numpy(), np.ones(len(diff))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
            row["dm_t_stat"], row["dm_p_value"] = float(fit.tvalues[0]), float(fit.pvalues[0])
        rows[name] = row
    return pd.DataFrame(rows).T


def _level_and_volatility_period(
    period_dates: pd.DatetimeIndex, var_df: pd.DataFrame, var_confidence: float, label: str,
) -> pd.Series:
    forecast_df = _period_var_df(period_dates, var_df, label, caller=f"level_and_volatility_{label}_period")
    return pd.Series(run_level_and_volatility_suite_single_window(forecast_df, var_confidence), name=label)


def level_and_volatility_forecast_period(
    calendar: "GroundTruthCalendar", var_df: pd.DataFrame, var_confidence: float,
) -> pd.Series:
    """
    run_level_and_volatility_suite_single_window on the FORECAST dates of var_df (the calendar's index is used,
    warns if var_df does not cover it). var_confidence must be the one var_df was produced with.
    Returns a Series named "forecast" (one column when put into a DataFrame).
    """
    return _level_and_volatility_period(calendar.forecast_dates, var_df, var_confidence, "forecast")


def level_and_volatility_train_period(
    calendar: "GroundTruthCalendar", var_df: pd.DataFrame, var_confidence: float,
) -> pd.Series:
    """Same on the TRAIN dates of var_df (in-sample: optimistic, the model saw these targets). Named "train"."""
    return _level_and_volatility_period(calendar.train_data.index, var_df, var_confidence, "train")


def level_and_volatility_entire_period(
    calendar: "GroundTruthCalendar", var_df: pd.DataFrame, var_confidence: float,
) -> pd.Series:
    """Same on every train + forecast date of var_df, as ONE window. Named "entire"."""
    period_dates = calendar.train_data.index.append(calendar.forecast_dates)
    return _level_and_volatility_period(period_dates, var_df, var_confidence, "entire")


def level_and_volatility_periods_side_by_side(
    calendar: "GroundTruthCalendar",
    var_confidence: float,
    *,
    forecast_var_df: pd.DataFrame | None = None,
    train_var_df: pd.DataFrame | None = None,
    entire_var_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    level_and_volatility_forecast_period / _train_period / _entire_period for the var_dfs you pass, as one
    table: one column per period ("forecast", "train", "entire"), one row per result field.
    """
    columns = []
    if forecast_var_df is not None:
        columns.append(level_and_volatility_forecast_period(calendar, forecast_var_df, var_confidence))
    if train_var_df is not None:
        columns.append(level_and_volatility_train_period(calendar, train_var_df, var_confidence))
    if entire_var_df is not None:
        columns.append(level_and_volatility_entire_period(calendar, entire_var_df, var_confidence))
    if not columns:
        raise ValueError("Pass at least one of forecast_var_df, train_var_df, entire_var_df.")
    return pd.concat(columns, axis=1)
