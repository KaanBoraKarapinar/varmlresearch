"""
Statistical VaR backtest suite (hypothesis tests: binomial, Basel traffic light, Kupiec POF/TUFF, Haas TBF,
Christoffersen) -- copied from 08_backtest_trafficlight_kupiec_christoffersen.ipynb, logic unchanged.
The descriptive side (how high / how volatile the VaR is, tick loss) lives in var_level_and_volatility_utils.py.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats as stats

if TYPE_CHECKING:
    from .GroundTruthCalendar import GroundTruthCalendar

VAR_COL = "varForecast"
LOSS_COL = "loss"


def loss_exceeding_days(
    forecast_df: pd.DataFrame,
    start_date,
    end_date,
) -> dict:
    result = {}

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    forecast_df = forecast_df.loc[start_date:end_date]
    number_of_days_in_df = len(forecast_df)

    days_breach_status = forecast_df[LOSS_COL] > forecast_df[VAR_COL]

    indexes_of_days_exceeded = np.where(days_breach_status)[0]

    days_between_exceedances = np.diff(indexes_of_days_exceeded)

    if len(indexes_of_days_exceeded) > 0:
        first_breach = np.argmax(days_breach_status.values) + 1
        days_between_exceedances = np.insert(days_between_exceedances, 0, first_breach)
    else:
        # no breach at all in the window: leave empty rather than crash on argmax-of-all-False
        days_between_exceedances = np.array([], dtype=int)

    result["days_exceeded_boolean"] = days_breach_status
    result["X_days_exceeded_number"] = int(days_breach_status.sum())
    result["N_window_size_by_weekdays_in_df"] = number_of_days_in_df
    result["days_between_exceedances"] = days_between_exceedances

    return result


def binomial_test_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
    test_confidence: float = 0.95,
) -> bool:
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    X_days_exceeded_number = preprocess["X_days_exceeded_number"]
    N_window_size = preprocess["N_window_size_by_weekdays_in_df"]

    p = 1 - var_confidence
    expected_breach = N_window_size * p
    test_statistic = (X_days_exceeded_number - expected_breach) / np.sqrt(expected_breach * (1 - p))
    test_statistic = np.abs(test_statistic)

    alpha = 1 - test_confidence
    threshold = stats.norm.ppf(1 - alpha / 2)

    return not (test_statistic > threshold)


def basel_traffic_light_test_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
) -> int:
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    X_days_exceeded_number = preprocess["X_days_exceeded_number"]
    N_window_size = preprocess["N_window_size_by_weekdays_in_df"]

    p = 1 - var_confidence
    cumulative_binomial_probability = stats.binom.cdf(X_days_exceeded_number, N_window_size, p)

    threshold_green = 0.95
    threshold_red = 0.9999

    if cumulative_binomial_probability < threshold_green:
        return 1
    elif threshold_green <= cumulative_binomial_probability < threshold_red:
        return 2
    else:
        return 3


def kupiec_pof_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
    test_confidence: float = 0.95,
    detailed_output: bool = False,
):
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    X_days_exceeded_number = preprocess["X_days_exceeded_number"]
    N_window_size = preprocess["N_window_size_by_weekdays_in_df"]

    p = 1 - var_confidence
    p_hat = X_days_exceeded_number / N_window_size

    log_numerator = (N_window_size - X_days_exceeded_number) * np.log(1 - p) + X_days_exceeded_number * np.log(p)

    if X_days_exceeded_number == 0:
        log_denominator = N_window_size * np.log(1 - p_hat)
    elif X_days_exceeded_number == N_window_size:
        log_denominator = X_days_exceeded_number * np.log(p_hat)
    else:
        log_denominator = (N_window_size - X_days_exceeded_number) * np.log(1 - p_hat) + X_days_exceeded_number * np.log(p_hat)

    test_statistic = -2 * (log_numerator - log_denominator)

    alpha = 1 - test_confidence
    threshold = stats.chi2.ppf(1 - alpha, df=1)

    reject_h0 = test_statistic > threshold

    if detailed_output:
        return {
            "test_statistic": test_statistic,
            "threshold": threshold,
            "p_hat": p_hat,
            "X_days_exceeded_number": X_days_exceeded_number,
            "N_window_size": N_window_size,
            "reject_h0": reject_h0,
        }

    return not reject_h0


def kupiec_tuff_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
    test_confidence: float = 0.95,
) -> bool:
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    X_days_exceeded_number = preprocess["X_days_exceeded_number"]

    if X_days_exceeded_number == 0:
        raise ValueError("Kupiec TUFF test is undefined when there is no exceedance")

    n = preprocess["days_between_exceedances"][0]

    p = 1 - var_confidence

    log_numerator = np.log(p) + (n - 1) * np.log(1 - p)

    if n == 1:
        log_denominator = np.log(1 / n)
    else:
        log_denominator = np.log(1 / n) + (n - 1) * np.log(1 - 1 / n)

    test_statistic = -2 * (log_numerator - log_denominator)

    alpha = 1 - test_confidence
    threshold = stats.chi2.ppf(1 - alpha, df=1)

    return test_statistic <= threshold


def haas_tbf_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
    test_confidence: float = 0.95,
) -> bool:
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    X_days_exceeded_number = preprocess["X_days_exceeded_number"]

    n_t = preprocess["days_between_exceedances"]

    if len(n_t) == 0:
        raise ValueError("Haas TBF test is undefined when there is no exceedance")

    p = 1 - var_confidence

    log_numerator = np.log(p) + (n_t - 1) * np.log(1 - p)

    with np.errstate(divide="ignore", invalid="ignore"):
        # np.where evaluates both branches for every element before selecting, so the
        # n_t==1 elements harmlessly produce log(0)=-inf / 0*-inf=nan here and get
        # discarded by the selection below -- suppressed rather than fixed structurally,
        # to keep this identical in behavior to the original notebook's logic.
        log_denominator = np.where(
            n_t == 1,
            np.log(1 / n_t),
            np.log(1 / n_t) + (n_t - 1) * np.log(1 - 1 / n_t),
        )

    LR_TBFI = -2 * np.sum(log_numerator - log_denominator)

    LR_POF = kupiec_pof_single_window(
        start_date, end_date, forecast_df, var_confidence, test_confidence,
        detailed_output=True,
    )["test_statistic"]

    test_statistic = LR_TBF = LR_TBFI + LR_POF

    alpha = 1 - test_confidence
    degrees_of_freedom = X_days_exceeded_number + 1
    threshold = stats.chi2.ppf(1 - alpha, df=degrees_of_freedom)

    return not (test_statistic > threshold)


def christoffersen_independence_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
    test_confidence: float = 0.95,
    detailed_output: bool = False,
):
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    breach = preprocess["days_exceeded_boolean"].values.astype(int)

    today = breach[1:]
    yesterday = breach[:-1]

    n00 = np.sum((yesterday == 0) & (today == 0))
    n01 = np.sum((yesterday == 0) & (today == 1))
    n10 = np.sum((yesterday == 1) & (today == 0))
    n11 = np.sum((yesterday == 1) & (today == 1))

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    denom = n00 + n01 + n10 + n11
    pi = (n01 + n11) / denom if denom > 0 else 0.0

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    log_numerator = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    log_denominator = (
        n00 * safe_log(1 - pi0) + n01 * safe_log(pi0) +
        n10 * safe_log(1 - pi1) + n11 * safe_log(pi1)
    )

    test_statistic = -2 * (log_numerator - log_denominator)

    alpha = 1 - test_confidence
    threshold = stats.chi2.ppf(1 - alpha, df=1)

    reject_h0 = test_statistic > threshold

    if detailed_output:
        return {
            "test_statistic": test_statistic,
            "threshold": threshold,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi0": pi0, "pi1": pi1, "pi": pi,
            "reject_h0": reject_h0,
        }

    return not reject_h0


def christoffersen_conditional_coverage_single_window(
    start_date,
    end_date,
    forecast_df: pd.DataFrame,
    var_confidence: float,
    test_confidence: float = 0.95,
    detailed_output: bool = False,
):
    LR_POF = kupiec_pof_single_window(
        start_date, end_date, forecast_df,
        var_confidence, test_confidence, detailed_output=True,
    )["test_statistic"]

    LR_IND = christoffersen_independence_single_window(
        start_date, end_date, forecast_df,
        var_confidence, test_confidence, detailed_output=True,
    )["test_statistic"]

    test_statistic = LR_POF + LR_IND

    alpha = 1 - test_confidence
    threshold = stats.chi2.ppf(1 - alpha, df=2)

    reject_h0 = test_statistic > threshold

    if detailed_output:
        return {
            "test_statistic": test_statistic,
            "threshold": threshold,
            "LR_POF": LR_POF,
            "LR_independence": LR_IND,
            "reject_h0": reject_h0,
        }

    return not reject_h0


def run_full_backtest_suite_single_window(
    forecast_df: pd.DataFrame,
    start_date,
    end_date,
    var_confidence: float,
    test_confidence: float = 0.95,
) -> dict:
    """Runs every test in the suite once, on ONE `forecast_df` (date index,
    "varForecast" + "loss" columns) covering a single contiguous daily sequence,
    and returns a flat dict of results, convenient for logging as MLflow
    metrics/params.

    See `run_full_backtest_suite_all_windows` below for running this across many
    (possibly overlapping) windows, e.g. the list produced by
    `ROLLING_TRAINER_single_model_multi_day_FOR_FORECAST_LENGTH`."""
    preprocess = loss_exceeding_days(forecast_df, start_date, end_date)
    x = preprocess["X_days_exceeded_number"]
    n = preprocess["N_window_size_by_weekdays_in_df"]

    out = {
        "var_confidence": var_confidence,
        "test_confidence": test_confidence,
        "n_days": n,
        "n_exceedances": x,
        "exceedance_rate": x / n if n > 0 else np.nan,
        "expected_exceedance_rate": 1 - var_confidence,
        "binomial_pass": binomial_test_single_window(
            start_date, end_date, forecast_df, var_confidence, test_confidence),
        "traffic_light_zone": basel_traffic_light_test_single_window(
            start_date, end_date, forecast_df, var_confidence),
    }

    kupiec = kupiec_pof_single_window(
        start_date, end_date, forecast_df, var_confidence, test_confidence,
        detailed_output=True,
    )
    out["kupiec_pof_stat"] = kupiec["test_statistic"]
    out["kupiec_pof_critical_value"] = kupiec["threshold"]    # the stat must stay at or below this to pass
    out["kupiec_pof_pass"] = not kupiec["reject_h0"]

    if x > 0:
        out["kupiec_tuff_pass"] = kupiec_tuff_single_window(
            start_date, end_date, forecast_df, var_confidence, test_confidence)
        out["haas_tbf_pass"] = haas_tbf_single_window(
            start_date, end_date, forecast_df, var_confidence, test_confidence)
    else:
        out["kupiec_tuff_pass"] = np.nan
        out["haas_tbf_pass"] = np.nan

    christ_ind = christoffersen_independence_single_window(
        start_date, end_date, forecast_df, var_confidence, test_confidence,
        detailed_output=True,
    )
    out["christoffersen_ind_stat"] = christ_ind["test_statistic"]
    out["christoffersen_ind_critical_value"] = christ_ind["threshold"]
    out["christoffersen_ind_pass"] = not christ_ind["reject_h0"]

    christ_cc = christoffersen_conditional_coverage_single_window(
        start_date, end_date, forecast_df, var_confidence, test_confidence,
        detailed_output=True,
    )
    out["christoffersen_cc_stat"] = christ_cc["test_statistic"]
    out["christoffersen_cc_critical_value"] = christ_cc["threshold"]
    out["christoffersen_cc_pass"] = not christ_cc["reject_h0"]

    return out


def _period_var_df(
    period_dates: pd.DatetimeIndex, var_df: pd.DataFrame, label: str, caller: str,
) -> pd.DataFrame:
    """
    The varForecast/loss rows of var_df on the calendar's `period_dates` (shared by every per-period suite, e.g.
    level_and_volatility_*_period). Only the calendar's date index is used (the losses come from var_df);
    warns if var_df does not cover all of the period's dates. `caller` is only used in the messages.
    """
    dates = period_dates[period_dates.isin(var_df.index)]
    if len(dates) != len(period_dates):
        warnings.warn(
            f"{caller}: var_df covers {len(dates)} of the {len(period_dates)} {label}-period "
            f"calendar dates; it runs on those {len(dates)}.", stacklevel=4,
        )
    if dates.empty:
        raise ValueError(f"{caller}: var_df has none of the {label}-period calendar dates.")

    # sequence-dependent measures (TUFF, Haas, Christoffersen, daily VaR change) read neighbouring rows as
    # consecutive bars: no bar may be missing
    if (np.diff(period_dates.get_indexer(dates)) != 1).any():
        raise ValueError(f"{caller}: the {label}-period dates in var_df are not consecutive bars of the calendar.")

    forecast_df = var_df.loc[dates, [VAR_COL, LOSS_COL]]
    if forecast_df.isna().any().any():
        raise ValueError(f"{caller}: var_df has NaN in {VAR_COL}/{LOSS_COL} in the {label} period.")
    return forecast_df


def _backtest_period(
    period_dates: pd.DatetimeIndex, var_df: pd.DataFrame, var_confidence: float, test_confidence: float, label: str,
) -> pd.Series:
    """The rows of var_df on the calendar's `period_dates` (see _period_var_df), backtested as ONE window."""
    forecast_df = _period_var_df(period_dates, var_df, label, caller=f"backtest_{label}_period")
    result = run_full_backtest_suite_single_window(
        forecast_df, forecast_df.index[0], forecast_df.index[-1], var_confidence, test_confidence)
    return pd.Series(result, name=label)


def backtest_forecast_period(
    calendar: "GroundTruthCalendar", var_df: pd.DataFrame, var_confidence: float, test_confidence: float = 0.95,
) -> pd.Series:
    """
    run_full_backtest_suite_single_window on the FORECAST dates of var_df: the out-of-sample, honest test.
    var_df: 'varForecast' + 'loss' columns, date index (calendar.var_from_residual_bank_*_period output).
    var_confidence must be the one var_df was produced with (it is not stored in var_df); both confidences are
    embedded in the result. Returns a Series named "forecast" (one column when put into a DataFrame).
    """
    return _backtest_period(calendar.forecast_dates, var_df, var_confidence, test_confidence, "forecast")


def backtest_train_period(
    calendar: "GroundTruthCalendar", var_df: pd.DataFrame, var_confidence: float, test_confidence: float = 0.95,
) -> pd.Series:
    """
    Same on the TRAIN dates of var_df (in-sample). Optimistic by construction (the model saw these targets):
    a sanity check / overfitting gauge, not performance evidence. Series named "train".
    """
    return _backtest_period(calendar.train_data.index, var_df, var_confidence, test_confidence, "train")


def backtest_entire_period(
    calendar: "GroundTruthCalendar", var_df: pd.DataFrame, var_confidence: float, test_confidence: float = 0.95,
) -> pd.Series:
    """
    Same on every train + forecast date of var_df, as ONE window (the in-sample part makes it optimistic, see
    backtest_train_period). Series named "entire".
    """
    period_dates = calendar.train_data.index.append(calendar.forecast_dates)
    return _backtest_period(period_dates, var_df, var_confidence, test_confidence, "entire")


def backtest_periods_side_by_side(
    calendar: "GroundTruthCalendar",
    var_confidence: float,
    test_confidence: float = 0.95,
    *,
    forecast_var_df: pd.DataFrame | None = None,
    train_var_df: pd.DataFrame | None = None,
    entire_var_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    backtest_forecast_period / backtest_train_period / backtest_entire_period for the var_dfs you pass, as
    one table: one column per period ("forecast", "train", "entire"), one row per result field.
    """
    columns = []
    if forecast_var_df is not None:
        columns.append(backtest_forecast_period(calendar, forecast_var_df, var_confidence, test_confidence))
    if train_var_df is not None:
        columns.append(backtest_train_period(calendar, train_var_df, var_confidence, test_confidence))
    if entire_var_df is not None:
        columns.append(backtest_entire_period(calendar, entire_var_df, var_confidence, test_confidence))
    if not columns:
        raise ValueError("Pass at least one of forecast_var_df, train_var_df, entire_var_df.")
    return pd.concat(columns, axis=1)


def _pass_rate(flags: pd.Series) -> float:
    """Fraction of non-NaN entries that are truthy -- NaN entries (e.g.
    kupiec_tuff_pass/haas_tbf_pass on a window with zero exceedances) are excluded
    from both the numerator and denominator rather than counted as failures."""
    return pd.to_numeric(flags, errors="coerce").mean()


def run_full_backtest_suite_all_windows(
    windows: list[dict],
    var_confidence: float,
    test_confidence: float = 0.95,
) -> dict:
    """
    Runs `run_full_backtest_suite_single_window` once per window in `windows` (the
    list produced by `ROLLING_TRAINER_single_model_multi_day_FOR_FORECAST_LENGTH`:
    each window's own `forecast_df` is one contiguous, non-overlapping-within-
    itself daily sequence, so every test in the suite -- including the sequence-
    dependent ones, Kupiec TUFF/Haas TBF/Christoffersen -- stays statistically
    meaningful at the per-window level, even though the windows overlap with EACH
    OTHER), then aggregates across windows to answer "how good was this
    (model_name, training_window) decision", not just "how did this one window do":

    * pass/fail flags -> pass RATE (fraction of windows that passed; NaN-skipped
      for tests that are undefined without an exceedance, kupiec_tuff/haas_tbf)
    * day/exceedance counts -> summed, then `pooled_exceedance_rate` = total
      exceedances / total days (correct weighting if window lengths ever differ;
      here every surviving window contributes exactly `forecast_period` days, so
      it coincides with the plain mean)
    * `traffic_light_zone` (ordinal 1/2/3) -> % of windows landing in each zone
    * continuous test statistics -> mean across windows (a magnitude/severity
      signal, secondary to the pass rates)

    Since windows overlap with their neighbors (each shares `forecast_period - 1`
    of its days with the next), these per-window results are NOT independent
    samples -- treat the aggregate as descriptive, not as input to a fresh
    significance test.

    Returns a dict with the aggregate summary above, plus `"per_window_results"`:
    the raw per-window DataFrame (one row per window -- `window_start`/`window_end`
    plus everything `run_full_backtest_suite_single_window` returns) for drilling
    into which specific windows/dates failed.
    """
    if not windows:
        raise ValueError("windows is empty -- nothing to backtest.")

    per_window_rows = []

    for window in windows:
        forecast_df = window["forecast_df"]
        result = run_full_backtest_suite_single_window(
            forecast_df=forecast_df,
            start_date=forecast_df.index.min(),
            end_date=forecast_df.index.max(),
            var_confidence=var_confidence,
            test_confidence=test_confidence,
        )
        result["window_start"] = window["window_start"]
        result["window_end"] = window["window_end"]
        per_window_rows.append(result)

    per_window_df = pd.DataFrame(per_window_rows)

    total_days = int(per_window_df["n_days"].sum())
    total_exceedances = int(per_window_df["n_exceedances"].sum())
    zone_rate = per_window_df["traffic_light_zone"].value_counts(normalize=True)

    return {
        "n_windows": len(per_window_df),
        "total_days": total_days,
        "total_exceedances": total_exceedances,
        "pooled_exceedance_rate": total_exceedances / total_days if total_days > 0 else np.nan,
        "expected_exceedance_rate": 1 - var_confidence,
        "mean_exceedance_rate": per_window_df["exceedance_rate"].mean(),
        "binomial_pass_rate": _pass_rate(per_window_df["binomial_pass"]),
        "pct_windows_green": float(zone_rate.get(1, 0.0)),
        "pct_windows_yellow": float(zone_rate.get(2, 0.0)),
        "pct_windows_red": float(zone_rate.get(3, 0.0)),
        "kupiec_pof_pass_rate": _pass_rate(per_window_df["kupiec_pof_pass"]),
        "mean_kupiec_pof_stat": per_window_df["kupiec_pof_stat"].mean(),
        "kupiec_tuff_pass_rate": _pass_rate(per_window_df["kupiec_tuff_pass"]),
        "haas_tbf_pass_rate": _pass_rate(per_window_df["haas_tbf_pass"]),
        "christoffersen_ind_pass_rate": _pass_rate(per_window_df["christoffersen_ind_pass"]),
        "mean_christoffersen_ind_stat": per_window_df["christoffersen_ind_stat"].mean(),
        "christoffersen_cc_pass_rate": _pass_rate(per_window_df["christoffersen_cc_pass"]),
        "mean_christoffersen_cc_stat": per_window_df["christoffersen_cc_stat"].mean(),
        "per_window_results": per_window_df,
    }
