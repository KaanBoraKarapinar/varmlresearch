import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Literal, NamedTuple
import plotly.graph_objects as go

# %%



from MLProject.var_backtest_utils import run_full_backtest_suite_single_window


# %%
class TrainingFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    feature_dates: pd.DatetimeIndex
    target_scale: float
    frequency: pd.Timedelta
    feature_cols: list[str]
    type: str  # "single" | "wide" | "long"

    ##important part:

    X_in_sample: pd.DataFrame
    X_out_sample_last_index: pd.DataFrame

    y_in_sample: pd.DataFrame

    DANGEROUS_r_next_in_sample: pd.Series


    #only for long format
    X_in_sample_id_cols: pd.DataFrame | None
    X_out_sample_last_index_id_cols: pd.DataFrame | None


class ForecastFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    feature_dates: pd.DatetimeIndex
    target_scale: float
    frequency: pd.Timedelta
    feature_cols: list[str]
    type: str  # "single" | "wide" | "long"

    ##important part:

    X_forecast_sample: pd.DataFrame
    y_forecast_sample: pd.DataFrame

    DANGEROUS_r_next_forecast_sample: pd.Series

    # only for long format
    id_cols: pd.DataFrame | None

    def features_at(self, date) -> pd.Series:
        date = pd.Timestamp(date)
        if date not in self.X_forecast_sample.index:
            raise KeyError(f"{date} not in ForecastFeature range [{self.start} .. {self.end}]")
        return self.X_forecast_sample.loc[date]
# %%





class VarPredictionRun:
    """One named run of VaR forecasts: Date -> varForecast."""

    def __init__(self, key: str, forecast_dates: pd.DatetimeIndex):
        self.key = key
        self.forecast_dates = forecast_dates
        self.data = pd.DataFrame({"varForecast": np.nan}, index=forecast_dates)

    def set(self, date, var_estimate: float) -> "VarPredictionRun":
        date = pd.Timestamp(date)
        if date not in self.forecast_dates:
            raise KeyError(f"{date} not in this run's forecast_dates for key={self.key!r}")
        self.data.loc[date, "varForecast"] = float(var_estimate)
        return self

    def set_batch(self, var_series: pd.Series) -> "VarPredictionRun":
        missing = pd.DatetimeIndex(var_series.index).difference(self.forecast_dates)
        if len(missing):
            raise KeyError(f"dates outside forecast_dates for key={self.key!r}: {list(missing)}")
        self.data.loc[var_series.index, "varForecast"] = var_series.to_numpy(dtype=float)
        return self

    def dropna(self) -> pd.DataFrame:
        return self.data.dropna()








class GroundTruthCalendar:
    def __init__(
        self,
        frequency: pd.Timedelta,
        tickers_raw_price_data: pd.DataFrame,
        tickers: list[str],
        weights: dict[str, float] | pd.Series,
        train_start,
        train_end,
        forecast_start,
        forecast_end,
        drop_market_closed: bool = True,
    ):
        if not isinstance(frequency, pd.Timedelta):
            raise TypeError(f"frequency must be a pd.Timedelta, got {type(frequency)}")
        if frequency <= pd.Timedelta(0):
            raise ValueError(f"frequency must be positive: {frequency}")

        missing = set(tickers) - set(tickers_raw_price_data.columns)
        if missing:
            raise ValueError(f"tickers_data is missing columns: {sorted(missing)}")

        self.frequency = frequency
        self.tickers = list(tickers)
        self.weights = self._validate_weights(self.tickers, weights)

        pct_returns = self._calculate_pct_return(tickers_raw_price_data)

        full_data = self._compute_portfolio_return_and_loss_FOR_PCT_RETURN(pct_returns, self.tickers, self.weights)

        self.train_start, self.train_end, self.forecast_start, self.forecast_end = self._validate_windows(
            full_data, train_start, train_end, forecast_start, forecast_end
        )

        self.train_data = self._slice_with_gap_check(full_data, self.train_start, self.train_end, "train", drop_market_closed)
        self.forecast_data = self._slice_with_gap_check(full_data, self.forecast_start, self.forecast_end, "forecast", drop_market_closed)
        self.forecast_dates = self.forecast_data.index

        # DANGEROUS: train_data + forecast_data concatenated. Only for feature builders
        # that need rolling-window history to reach into train_data at the start of the forecast window (warmup)
        self.DANGEROUS_merged_warmedup_FORECAST_data = pd.concat([self.train_data, self.forecast_data]).sort_index()


        self.var_prediction_runs: dict[str, VarPredictionRun] = {}

    @staticmethod
    def _validate_weights(tickers: list[str], weights: dict[str, float] | pd.Series) -> pd.Series:
        weights = pd.Series(weights, dtype=float)
        missing_weights = set(tickers) - set(weights.index)
        if missing_weights:
            raise ValueError(f"Missing weights for tickers: {sorted(missing_weights)}")
        if not np.isfinite(weights.to_numpy()).all():
            raise ValueError("Weights must be finite.")
        if np.isclose(weights.sum(), 0.0):
            raise ValueError("Weights sum to zero.")
        return weights.loc[tickers] / weights.loc[tickers].sum()

    @staticmethod
    def _calculate_pct_return(tickers_raw_price_data: pd.DataFrame) -> pd.DataFrame:
        return tickers_raw_price_data.sort_index().pct_change()


    @staticmethod
    def _compute_portfolio_return_and_loss_FOR_PCT_RETURN(tickers_pct_change_data: pd.DataFrame, tickers: list[str], weights: pd.Series) -> pd.DataFrame:
        full_data = tickers_pct_change_data.sort_index().copy()
        portfolio_return = full_data[tickers].mul(weights, axis="columns").sum(axis=1, min_count=len(tickers))
        full_data["portfolio_return"] = portfolio_return
        full_data["portfolio_loss"] = -portfolio_return
        return full_data

    @staticmethod
    def _warn_if_wrong_weekday(ts: pd.Timestamp, label: str, expect: str) -> None:
        actual = ts.day_name()
        if actual != expect:
            print(f"[WARNING] {label}={ts.date().isoformat()} falls on {actual}, expected {expect} (using it as-is).")

    def _slice_with_gap_check(self, full_data, start, end, label, drop_market_closed):
        sliced = full_data.loc[start:end]
        n_business_days_dropped = 0

        if self.frequency == pd.Timedelta(hours=24):
            expected_business_days = pd.date_range(start, end, freq="B")
            missing_business_days = expected_business_days.difference(sliced.index)
            n_business_days_dropped = len(missing_business_days)

            if n_business_days_dropped:
                dates_str = ", ".join(d.isoformat() for d in missing_business_days.date[:5])  # pyright: ignore[reportAttributeAccessIssue]
                suffix = f" and {n_business_days_dropped - 5} more" if n_business_days_dropped > 5 else ""
                day_word = "day" if n_business_days_dropped == 1 else "days"
                msg = (
                    f"[{label}] {n_business_days_dropped} weekday {day_word} had no data and were skipped "
                    f"(likely market holidays): {dates_str}{suffix}"
                )

                if drop_market_closed:
                    print(f"[WARNING] {msg}")
                else:
                    raise ValueError(msg)

        if sliced.empty:
            raise ValueError(f"[{label}] no bars left in [{start} .. {end}] after slicing.")

        bar_word = "bar" if len(sliced) == 1 else "bars"
        print(
            f"[INFO] {label} set: {len(sliced)} {bar_word}, {n_business_days_dropped} weekday {'day' if n_business_days_dropped == 1 else 'days'} skipped")

        return sliced.copy()

    @classmethod
    def _validate_windows(cls, full_data, train_start, train_end, forecast_start, forecast_end) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        train_start = pd.Timestamp(train_start)
        train_end = pd.Timestamp(train_end)
        forecast_start = pd.Timestamp(forecast_start)
        forecast_end = pd.Timestamp(forecast_end)

        bounds = (full_data.index[0], full_data.index[-1]) if len(full_data) else (None, None)

        for label, ts in (
            ("train_start", train_start), ("train_end", train_end),
            ("forecast_start", forecast_start), ("forecast_end", forecast_end),
        ):
            if ts not in full_data.index:
                raise KeyError(f"{label}={ts} is not a bar in tickers_data (range: {bounds})")

        if not (train_start <= train_end):
            raise ValueError(f"train_start {train_start} is after train_end {train_end}")
        if not (train_end < forecast_start):
            raise ValueError(f"train_end {train_end} must be strictly before forecast_start {forecast_start} (no look-ahead)")
        if not (forecast_start <= forecast_end):
            raise ValueError(f"forecast_start {forecast_start} is after forecast_end {forecast_end}")

        cls._warn_if_wrong_weekday(train_start, "train_start", expect="Monday")
        cls._warn_if_wrong_weekday(train_end, "train_end", expect="Friday")
        cls._warn_if_wrong_weekday(forecast_start, "forecast_start", expect="Monday")
        cls._warn_if_wrong_weekday(forecast_end, "forecast_end", expect="Friday")

        return train_start, train_end, forecast_start, forecast_end

    def describe_split(self) -> pd.DataFrame:
        train_rows = pd.DataFrame({"split": "train"}, index=self.train_data.index)
        forecast_rows = pd.DataFrame({"split": "forecast"}, index=self.forecast_data.index)
        return pd.concat([train_rows, forecast_rows]).sort_index()

    def next_bars(self, dates) -> pd.DatetimeIndex:
        """
        Position-based next bar for each date in `dates`, looked up against the full
        train+forecast calendar (DANGEROUS_merged_warmedup_FORECAST_data) -- so a date
        sitting at the train/forecast boundary (e.g. train_end) still resolves to its
        true next bar in forecast_data. Position-based, not frequency-based, so it
        correctly skips weekends/holidays instead of naively adding a fixed offset.
        """
        index = self.DANGEROUS_merged_warmedup_FORECAST_data.index
        dates = pd.DatetimeIndex(dates)
        positions = index.get_indexer(dates)
        if (positions == -1).any():
            bad = dates[positions == -1]
            raise KeyError(f"Dates not found in calendar: {list(bad)}")
        if (positions + 1 >= len(index)).any():
            bad = dates[positions + 1 >= len(index)]
            raise KeyError(f"Dates have no following bar in calendar: {list(bad)}")
        return index[positions + 1]

    def usable_train_dates(self, candidate_dates, min_periods: int) -> pd.DatetimeIndex:
        """
        Intersects `candidate_dates` (e.g. a TrainingFeature's own common_dates --
        already restricted by that feature's own per-ticker NaN/warmup drops, which the
        calendar has no visibility into) with train_data dates that have >= min_periods
        prior train bars available (survived burn-in). Prints the comparison so a caller
        can see, e.g., that feature warmup already pushed the effective train start later
        than train_data's own start, or how many candidate dates got cut by burn-in.
        """
        candidate_dates = pd.DatetimeIndex(candidate_dates)
        burned_in = self.train_data.index[min_periods:]
        usable_dates = candidate_dates[candidate_dates.isin(burned_in)]

        n_candidates = len(candidate_dates)
        n_usable = len(usable_dates)
        n_dropped = n_candidates - n_usable
        burn_in_start = burned_in[0].date() if len(burned_in) else "N/A"
        print(
            f"[INFO] usable_train_dates: {n_usable}/{n_candidates} candidate dates survived "
            f"burn-in (min_periods={min_periods}, train_data starts {self.train_data.index[0].date()}, "
            f"burn-in cutoff {burn_in_start}, {n_dropped} candidate date(s) dropped)."
        )

        if usable_dates.empty:
            raise ValueError(f"usable_train_dates: no candidate dates survived burn-in (min_periods={min_periods}).")
        return usable_dates

    def forecast_pairs(self) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Usable indices for forecast-phase scoring: feature_dates[i] is the bar
        immediately before outcome_dates[i], both drawn from forecast_dates by position
        (not frequency subtraction, for the same skip-weekends/holidays reason as
        next_bars). forecast_dates[0] is never an outcome here -- its own feature bar is
        the last TRAINING bar, which no ForecastFeature carries (that boundary case is
        instead backfilled at training-seed time via next_bars/train_seed_dates).
        """
        if len(self.forecast_dates) < 2:
            empty = self.forecast_dates[:0]
            return empty, empty
        return self.forecast_dates[:-1], self.forecast_dates[1:]

    #def _lookup(self, date) -> pd.Series:
        #date = pd.Timestamp(date)
        #if date in self.train_data.index:
            #return self.train_data.loc[date]
        #if date in self.forecast_data.index:
            #return self.forecast_data.loc[date]
        #raise KeyError(f"{date} is not in train_data or forecast_data (embargo gap, dropped non-trading day, or out of range).")


    #def ticker_returns(self) -> pd.DataFrame:
        #return pd.concat([self.train_data[self.tickers], self.forecast_data[self.tickers]]).sort_index()

    #def realized_return(self, date) -> float:
        #return float(self._lookup(date)["portfolio_return"])

    #def realized_loss(self, date) -> float:
        #return float(self._lookup(date)["portfolio_loss"])

    def _get_or_create_run(self, key: str) -> VarPredictionRun:
        if key not in self.var_prediction_runs:
            self.var_prediction_runs[key] = VarPredictionRun(key, self.forecast_dates)
        return self.var_prediction_runs[key]

    def add_predicted_var(self, key: str, date, var_estimate: float) -> "GroundTruthCalendar":
        self._get_or_create_run(key).set(date, var_estimate)
        return self

    def add_predicted_var_batch(self, key: str, var_series: pd.Series) -> "GroundTruthCalendar":
        self._get_or_create_run(key).set_batch(var_series)
        return self

    def backtest(self, key: str, var_confidence: float, test_confidence: float = 0.95) -> dict:
        if key not in self.var_prediction_runs:
            raise KeyError(f"No run recorded under key={key!r}. Recorded keys: {list(self.var_prediction_runs)}")

        run_df = self.var_prediction_runs[key].dropna()
        if run_df.empty:
            raise ValueError(f"var_prediction_runs[{key!r}] is empty, nothing to backtest.")

        forecast_df = run_df.join(self.forecast_data["portfolio_loss"].rename("loss"))

        return run_full_backtest_suite_single_window(
            forecast_df=forecast_df,
            start_date=forecast_df.index.min(),
            end_date=forecast_df.index.max(),
            var_confidence=var_confidence,
            test_confidence=test_confidence,
        )


class DailyGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed: bool = True):
        super().__init__(pd.Timedelta(hours=24), tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed)


class HourlyGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed: bool = True):
        super().__init__(pd.Timedelta(hours=1), tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed)


class MinuteGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed: bool = True):
        super().__init__(pd.Timedelta(minutes=1), tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed)

# %%
def plot_var_runs(calendar: GroundTruthCalendar, keys: list[str], title: str = "VaR Forecast vs Loss") -> "go.Figure":
    import plotly.graph_objects as go

    fig = go.Figure()

    for key in keys:
        if key not in calendar.var_prediction_runs:
            raise KeyError(f"No run recorded under key={key!r}. Recorded keys: {list(calendar.var_prediction_runs)}")

        run_df = calendar.var_prediction_runs[key].dropna()
        fig.add_scatter(x=run_df.index, y=run_df["varForecast"], mode="lines", name=f"VaR Forecast ({key})")

    reference_key = keys[0]
    loss_df = calendar.var_prediction_runs[reference_key].dropna()
    loss_series = calendar.data.loc[loss_df.index, "portfolio_loss"]
    fig.add_scatter(x=loss_series.index, y=loss_series, mode="lines", name="Loss")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
    )
    return fig
# %%
### DATA QUALITY CHECKS

def DATA_QUALITY_check_variance_floor(predicted_variances, dates=None, context: str = "") -> np.ndarray:
    VARIANCE_FLOOR = 1e-12

    values = np.atleast_1d(np.asarray(predicted_variances, dtype=float))
    bad_mask = values <= VARIANCE_FLOOR

    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        if dates is not None:
            dates = pd.DatetimeIndex(np.atleast_1d(dates))
            bad_dates = list(dates[bad_mask][:5])
            location = f" at dates (first 5): {bad_dates}"
        else:
            location = ""

        prefix = f"[{context}] " if context else ""
        warnings.warn(
            f"{prefix}DEGENERATE VARIANCE: {n_bad} value(s) <= {VARIANCE_FLOOR}{location}.",
            stacklevel=2,
        )

    return np.maximum(values, VARIANCE_FLOOR)


def DATA_QUALITY_replace_inf_with_warning(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Replace inf/-inf with NaN, warning loudly if it actually triggers (near-zero sigma blowing up a division)."""
    inf_mask = np.isinf(df.to_numpy(dtype=float))
    if inf_mask.any():
        n_bad = int(inf_mask.sum())
        prefix = f"[{context}] " if context else ""
        warnings.warn(
            f"{prefix}INF VALUES DETECTED: {n_bad} cell(s) were inf/-inf, likely from dividing by "
            f"a near-zero sigma. These rows will be dropped.",
            stacklevel=2,
        )
    return df.replace([np.inf, -np.inf], np.nan)


# %%

import warnings
import numpy as np
import pandas as pd

RESIDUAL_COLS = ["actual_return", "predicted_variance", "predicted_sigma", "mu_hat", "standardized_residual"]
VARIANCE_FLOOR = 1e-12



class ResidualBank:
    def __init__(
        self,
        start_timestamp: pd.Timestamp,
        frequency: pd.Timedelta,
        default_mu_hat: float = 0.0,
        mu_mode: Literal["rolling", "constant"] = "constant",
        rolling_mu_hat_window_size: int | None = None,
    ):
        if not isinstance(frequency, pd.Timedelta):
            raise TypeError(f"frequency must be a pd.Timedelta, got: {type(frequency)}")
        if frequency <= pd.Timedelta(0):
            raise ValueError(f"frequency must be positive: {frequency}")
        if mu_mode not in ("rolling", "constant"):
            raise ValueError(f"mu_mode must be 'rolling' or 'constant', got: {mu_mode!r}")
        if mu_mode == "rolling":
            if rolling_mu_hat_window_size is None or rolling_mu_hat_window_size <= 0:
                raise ValueError(
                    f"mu_mode='rolling' requires a positive rolling_mu_hat_window_size, got: {rolling_mu_hat_window_size!r}"
                )

        self.start_timestamp = start_timestamp
        self.frequency = frequency
        self.default_mu_hat = float(default_mu_hat)
        self.mu_mode = mu_mode
        # Fixed at construction, never re-specified per call -- mu_mode="constant" leaves
        # this None since there's no window to speak of (default_mu_hat is used instead).
        self.rolling_mu_hat_window_size = rolling_mu_hat_window_size
        self.residuals = pd.DataFrame(columns=RESIDUAL_COLS, index=pd.DatetimeIndex([], name="Date"))

        # stateful mu_hat used to standardize the NEXT incoming residual.
        # Deliberately NOT recalculated until step_2, so step_1 never uses information
        # from the bar it is currently pricing (no look-ahead). In mu_mode="constant",
        # step_2/general_bulk_insert never touch this again -- it stays pinned at default_mu_hat.
        self._current_single_stateful_mu = float(default_mu_hat)

    @property
    def current_mu_hat(self) -> float:
        return self._current_single_stateful_mu

    def DANGEROUS_add_residual(self, date, actual_return: float, predicted_variance: float, mu_hat: float) -> "ResidualBank":
        """
        NO STATEFUL MU TRACKING, RISK OF LOOK FORWARD BIAS
        """
        date = pd.Timestamp(date)
        predicted_variance = float(DATA_QUALITY_check_variance_floor(predicted_variance, dates=[date], context="DANGEROUS_add_residual")[0])
        sigma = np.sqrt(predicted_variance)
        standardized_residual = (float(actual_return) - float(mu_hat)) / sigma

        if not np.isfinite(standardized_residual):
            warnings.warn(
                f"[DANGEROUS_add_residual] non-finite standardized_residual at {date} (sigma={sigma}), skipping this row.",
                stacklevel=2,
            )
            return self

        self.residuals.loc[date] = {
            "actual_return": float(actual_return),
            "predicted_variance": predicted_variance,
            "predicted_sigma": sigma,
            "mu_hat": float(mu_hat),
            "standardized_residual": standardized_residual,
        }
        return self

    def step_1_add_residual_with_prev_stateful_mu(
        self, date, actual_return: float, predicted_variance: float
    ) -> "ResidualBank":
        """
        Standardize and insert the new residual using the mu_hat carried over from BEFORE
        this bar's return was realized. Pins mu_hat to the current stateful value.
        """
        return self.DANGEROUS_add_residual(
            date=date,
            actual_return=actual_return,
            predicted_variance=predicted_variance,
            mu_hat=self._current_single_stateful_mu,
        )

    def step_2_change_stateful_mu(self, as_of: pd.Timestamp | None = None) -> "ResidualBank":
        """
        Recompute the stateful mu_hat from the (now updated) windowed history. Call this
        AFTER step_1, once per bar, so the next bar's step_1 sees this bar's return.
        No-op in mu_mode="constant" -- mu_hat stays pinned at default_mu_hat.
        """
        if self.mu_mode == "constant":
            return self

        window = self.windowed_read(self.rolling_mu_hat_window_size, as_of)
        if window.empty:
            raise ValueError(
                f"Cannot update mu_hat: residual window is empty "
                f"(rolling_mu_hat_window_size={self.rolling_mu_hat_window_size}, as_of={as_of})."
            )

        self._current_single_stateful_mu = float(window["actual_return"].mean())
        return self

    def _bootstrap_sequential_prefix(self, dates, actual_returns, predicted_variances) -> None:
        """
        One bar at a time: step_1 adds the residual using whatever mu_hat has
        accumulated so far (starting from default_mu_hat), then step_2 refreshes
        mu_hat from up to the last rolling_mu_hat_window_size bars now available
        (windowed_read tolerates a partial window). No-look-ahead, but slow -- O(n)
        individual pandas .loc writes. Only meant for the <= rolling_mu_hat_window_size
        bars training_safe_bulk_insert needs to ramp up mu_hat from scratch; everything
        after that goes through the vectorized general_bulk_insert instead.
        """
        for date, actual_return, predicted_variance in zip(dates, actual_returns, predicted_variances):
            self.step_1_add_residual_with_prev_stateful_mu(date, actual_return, predicted_variance)
            self.step_2_change_stateful_mu(as_of=date)

    def training_safe_bulk_insert(self, dates, actual_returns, predicted_variances) -> "ResidualBank":
        """
        Seeds a bank with a batch of historical bars, no-look-ahead throughout.
        general_bulk_insert (vectorized, fast) can't compute a rolling
        mu_hat for a bar without rolling_mu_hat_window_size bars of PRIOR history
        already in the bank, which a from-scratch bank can never have for its own
        first rolling_mu_hat_window_size bars -- so training_safe_bulk_insert only pays
        the slow sequential bootstrap (_bootstrap_sequential_prefix) for however many
        MORE bars are needed to reach rolling_mu_hat_window_size bars of total history,
        then hands the (usually much larger) remainder to general_bulk_insert.
        Numerically identical to bootstrapping every bar sequentially, just
        without paying O(n) individual pandas writes for all of it.

        In mu_mode="constant" there is no rolling-history requirement at all
        (every row's mu_hat is default_mu_hat), so this skips straight to
        general_bulk_insert for the whole batch.

        This is purely about mu_hat -- multi_day_var's quantile window is a separate
        parameter (multi_day_var_calculation_window_size), never touched here.
        """
        dates = pd.DatetimeIndex(dates, name="Date")
        actual_returns = np.asarray(actual_returns, dtype=float)
        predicted_variances = np.asarray(predicted_variances, dtype=float)

        if self.mu_mode == "constant":
            return self.general_bulk_insert(dates, actual_returns, predicted_variances)

        n_bootstrap = min(max(0, self.rolling_mu_hat_window_size - len(self.residuals)), len(dates))
        if n_bootstrap:
            self._bootstrap_sequential_prefix(
                dates[:n_bootstrap], actual_returns[:n_bootstrap], predicted_variances[:n_bootstrap]
            )
        if n_bootstrap < len(dates):
            self.general_bulk_insert(
                dates[n_bootstrap:], actual_returns[n_bootstrap:], predicted_variances[n_bootstrap:]
            )
        return self

    def general_bulk_insert(self, dates, actual_returns, predicted_variances) -> "ResidualBank":
        """
        mu_mode="rolling": mu_hat_t = mean(actual_return) over the
        rolling_mu_hat_window_size bars STRICTLY BEFORE t (t's own return excluded).
        Same no-look-ahead rule as step_1/step_2, vectorized.
        mu_mode="constant": mu_hat_t = default_mu_hat for every t -- no history
        requirement, no rolling/NaN check.
        """
        dates = pd.DatetimeIndex(dates, name="Date")
        actual_returns = np.asarray(actual_returns, dtype=float)
        predicted_variances = DATA_QUALITY_check_variance_floor(predicted_variances, dates=dates,
                                                                context="general_bulk_insert")

        if self.mu_mode == "constant":
            mu_hat_new = pd.Series(self.default_mu_hat, index=dates)
        else:
            combined_returns = pd.concat(
                [self.residuals["actual_return"], pd.Series(actual_returns, index=dates)]).sort_index()
            combined_returns = combined_returns[~combined_returns.index.duplicated(keep="last")]

            mu_hat_new = combined_returns.rolling(self.rolling_mu_hat_window_size).mean().shift(1).loc[dates]
            if mu_hat_new.isna().any():
                bad = mu_hat_new[mu_hat_new.isna()].index.tolist()
                raise ValueError(
                    f"Not enough history for rolling mu_hat "
                    f"(rolling_mu_hat_window_size={self.rolling_mu_hat_window_size}) at dates: {bad}"
                )

        sigma = np.sqrt(predicted_variances)
        standardized_residual = (actual_returns - mu_hat_new.to_numpy()) / sigma

        new_rows = pd.DataFrame(
            {
                "actual_return": actual_returns,
                "predicted_variance": predicted_variances,
                "predicted_sigma": sigma,
                "mu_hat": mu_hat_new.to_numpy(),
                "standardized_residual": standardized_residual,
            },
            index=dates,
        )
        new_rows = DATA_QUALITY_replace_inf_with_warning(new_rows, context="general_bulk_insert").dropna()

        self.residuals = pd.concat([self.residuals.drop(index=new_rows.index, errors="ignore"), new_rows]).sort_index()
        if self.mu_mode == "constant":
            self._current_single_stateful_mu = self.default_mu_hat
        else:
            self._current_single_stateful_mu = float(self.residuals["actual_return"].tail(self.rolling_mu_hat_window_size).mean())
        return self


    def windowed_read(self, window_size: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Generic last-N-residuals reader, purpose-agnostic -- callers pass whichever window (rolling_mu_hat_window_size or multi_day_var_calculation_window_size) is relevant to them."""
        df = self.residuals if as_of is None else self.residuals.loc[:as_of]
        return df.tail(window_size)

    def empirical_var_for_target(
            self,
            target_timestamp: pd.Timestamp,
            predicted_variance: float,
            var_confidence: float,
            multi_day_var_calculation_window_size: int,
            enforce_window_size: bool = False,
    ) -> pd.Series:
        """
        Compute empirical VaR for a specific target bar. Requires the feature bar
        (target_timestamp - frequency) to already be present in the residual history.

        multi_day_var_calculation_window_size and self.rolling_mu_hat_window_size are
        independent -- the quantile window (VaR tail estimate) and the mu_hat window
        (drift estimate, only read in mu_mode="rolling") don't have to be the same
        length, so they're read separately via two windowed_read calls. The mu_hat
        window is fixed on the bank at construction time (not passed here) since
        mu_mode/its window is a bank-wide policy, not something that varies per call.

        :param enforce_window_size: if True, raise unless the QUANTILE window contains
            exactly multi_day_var_calculation_window_size residuals. If False (default),
            any non-empty window up to that size is accepted. Only gates the quantile
            window, not the mu_hat window.
        :raises KeyError: if target_timestamp - frequency is not in self.residuals.
        :raises ValueError: if the quantile window is empty, or (when
            enforce_window_size=True) shorter than multi_day_var_calculation_window_size.
        :return: pd.Series indexed by target_timestamp, with fields varForecast, mu_hat,
            predicted_variance, last_feature_timestamp, n_residuals_used (quantile window size).
        """
        target_timestamp = pd.Timestamp(target_timestamp)
        last_feature_timestamp = target_timestamp - self.frequency

        if last_feature_timestamp not in self.residuals.index:
            raise KeyError(
                f"empirical_var_for_target: {last_feature_timestamp} (= target {target_timestamp} - "
                f"frequency {self.frequency}) is not in residual history. "
                f"Cannot price {target_timestamp} without its immediately preceding feature bar."
            )

        predicted_variance = float(
            DATA_QUALITY_check_variance_floor(predicted_variance, dates=[target_timestamp], context="empirical_var_for_target")[0])
        quantile_window = self.windowed_read(multi_day_var_calculation_window_size, as_of=last_feature_timestamp)

        if quantile_window.empty:
            raise ValueError(
                f"Residual window is empty (multi_day_var_calculation_window_size={multi_day_var_calculation_window_size}, "
                f"as_of={last_feature_timestamp})."
            )

        if enforce_window_size and len(quantile_window) < multi_day_var_calculation_window_size:
            raise ValueError(
                f"enforce_window_size=True but only {len(quantile_window)}/{multi_day_var_calculation_window_size} "
                f"residuals available as of {last_feature_timestamp} (target={target_timestamp})."
            )

        if self.mu_mode == "constant":
            mu_hat = self.default_mu_hat
        else:
            mu_window = self.windowed_read(self.rolling_mu_hat_window_size, as_of=last_feature_timestamp)
            if mu_window.empty:
                raise ValueError(
                    f"Cannot compute mu_hat: residual window is empty "
                    f"(rolling_mu_hat_window_size={self.rolling_mu_hat_window_size}, as_of={last_feature_timestamp})."
                )
            mu_hat = float(mu_window["actual_return"].mean())

        print(
            f"[empirical_var_for_target] target={target_timestamp}, using {len(quantile_window)} residual(s) "
            f"(quantile window) up to {last_feature_timestamp}"
        )

        q = np.quantile(quantile_window["standardized_residual"], 1 - var_confidence)
        sigma_hat = np.sqrt(predicted_variance)
        var_estimate = -(mu_hat + sigma_hat * q)

        return pd.Series(
            {
                "varForecast": var_estimate,
                "mu_hat": mu_hat,
                "predicted_variance": predicted_variance,
                "last_feature_timestamp": last_feature_timestamp,
                "n_residuals_used": len(quantile_window),
            },
            name=target_timestamp,
        )

    def multi_day_var(self, dates, var_confidence: float, multi_day_var_calculation_window_size: int) -> pd.DataFrame:
        """
        Pure calculation: dates must already be in self.residuals (bulk_insert first).
        VaR for date t uses standardized_residual from the
        multi_day_var_calculation_window_size bars STRICTLY BEFORE t (t excluded).
        mu_hat is read straight off the stored column (written by general_bulk_insert
        using rolling_mu_hat_window_size at insert time), not recomputed here.
        """
        dates = pd.DatetimeIndex(dates, name="Date")
        missing = dates.difference(self.residuals.index)
        if len(missing):
            raise KeyError(f"multi_day_var: dates missing from residual history: {list(missing)}")

        q_series = (
            self.residuals["standardized_residual"]
            .rolling(multi_day_var_calculation_window_size)
            .apply(lambda w: np.quantile(w, 1 - var_confidence), raw=True)
            .shift(1)
        )

        result = self.residuals.loc[dates, ["mu_hat", "predicted_sigma"]].copy()
        result["quantile"] = q_series.loc[dates]

        if result["quantile"].isna().any():
            bad = result.index[result["quantile"].isna()].tolist()
            raise ValueError(
                f"Not enough prior residual history for quantile "
                f"(multi_day_var_calculation_window_size={multi_day_var_calculation_window_size}) at dates: {bad}"
            )

        result["varForecast"] = -(result["mu_hat"] + result["predicted_sigma"] * result["quantile"])
        return result[["varForecast"]]


    def copy(self) -> "ResidualBank":
        new_bank = ResidualBank(
            self.start_timestamp, self.frequency, self.default_mu_hat, self.mu_mode, self.rolling_mu_hat_window_size
        )
        new_bank.residuals = self.residuals.copy()
        new_bank._current_single_stateful_mu = self._current_single_stateful_mu
        return new_bank

    def to_dataframe(self) -> pd.DataFrame:
        return self.residuals



# %%



class TrainedModel(ABC):
    """
    A frozen, trained model plus everything needed to score OOS bars with it.
    Each model type implements its own predict_variance.
    """

    def __init__(
        self,
        training_feature: TrainingFeature,
        underlying_model: object,
        model_name: str,
        target_scale: float,
        feature_cols: list[str],
        residual_bank_in_sample: ResidualBank,
        ewma_variance_bank_in_sample: "EwmaVarianceBank | None" = None,
    ):
        self.training_feature = training_feature
        self.underlying_model = underlying_model
        self.model_name = model_name
        self.target_scale = target_scale
        self.feature_cols = feature_cols
        self.residual_bank_in_sample = residual_bank_in_sample
        self.ewma_variance_bank_in_sample = ewma_variance_bank_in_sample

    @abstractmethod
    def predict_variance_single_wide(self, forecast_feature: "ForecastFeature", feature_date: pd.Timestamp) -> float:
        """Predict the variance for the bar immediately AFTER feature_date (wide/single-ticker feature)."""

    @abstractmethod
    def predict_variance_batch_wide(self, forecast_feature: "ForecastFeature", feature_dates: pd.Timestamp) -> pd.Series:
        """Predict the variances for the given feature dates (wide/single-ticker feature)."""

    def predict_variance_batch_long(
        self, X_long: pd.DataFrame, id_cols: pd.DataFrame, feature_dates
    ) -> dict[str, pd.Series]:
        """
        Splits a long-format (one row per (Date, ticker)) feature block into one
        date-indexed wide view per ticker via id_cols, and scores each with
        predict_variance_batch_wide.
        """
        feature_dates = pd.DatetimeIndex(feature_dates)
        result: dict[str, pd.Series] = {}
        for ticker, rows in id_cols.groupby("ticker").groups.items():
            ticker_X = X_long.loc[rows, self.feature_cols].set_axis(
                id_cols.loc[rows, "Date"].to_numpy(), axis=0
            ).sort_index()
            ticker_view = ForecastFeature(
                start=ticker_X.index.min(), end=ticker_X.index.max(), feature_dates=ticker_X.index,
                target_scale=self.target_scale, frequency=self.training_feature.frequency,
                feature_cols=self.feature_cols, type="wide",
                X_forecast_sample=ticker_X, y_forecast_sample=pd.DataFrame(),
                DANGEROUS_r_next_forecast_sample=pd.Series(dtype=float), id_cols=None,
            )
            result[ticker] = self.predict_variance_batch_wide(ticker_view, feature_dates)
        return result


class ForecastResult(NamedTuple):
    forecast_feature: ForecastFeature
    trained_model: object
    model_name: str
    target_scale: float
    feature_cols: list[str]
    residual_bank_oos_sample: ResidualBank

    ##important part:
    VarPredictions: pd.DataFrame

# %%

def multi_day_var_from_pretrained_model(
    trained_model: TrainedModel,
    forecast_feature: "ForecastFeature",
    calendar: GroundTruthCalendar,
    multi_day_var_calculation_window_size: int,
    var_confidence: float,
    run_key: str | None = None,
) -> pd.DataFrame:
    """
    forecast_dates is no longer a parameter, calendar.forecast_dates is the
    single source of truth for which bars get scored. calendar.forecast_data's
    index IS calendar.forecast_dates by construction, so actual returns/losses
    are read straight off it -- no train_data ever touched here.

    feature_dates/outcome_dates come from calendar.forecast_pairs() -- see that
    method for why forecast_dates[0] itself is never scored here.

    trained_model.residual_bank_in_sample already carries its rolling_mu_hat_window_size
    (fixed at construction, see ResidualBank.__init__) so general_bulk_insert reads it off
    the bank instead of taking it here. multi_day_var_calculation_window_size feeds the
    VaR quantile rolling window (multi_day_var) -- that one's independent and still passed
    per call.
    """
    bank = trained_model.residual_bank_in_sample.copy()
    feature_dates, outcome_dates = calendar.forecast_pairs()

    if len(outcome_dates) == 0:
        return pd.DataFrame(columns=["varForecast", "loss"])

    predicted_variances = trained_model.predict_variance_batch_wide(forecast_feature, feature_dates)
    predicted_variances.index = outcome_dates

    actual_returns = calendar.forecast_data.loc[outcome_dates, "portfolio_return"].to_numpy()

    bank.general_bulk_insert(outcome_dates, actual_returns, predicted_variances.to_numpy())

    var_df = bank.multi_day_var(outcome_dates, var_confidence, multi_day_var_calculation_window_size)
    var_df["loss"] = calendar.forecast_data.loc[var_df.index, "portfolio_loss"]

    calendar.add_predicted_var_batch(run_key or trained_model.model_name, var_df["varForecast"])

    return var_df

# %%

class EwmaVarianceBank:
    """
    Portfolio-variance bank for pooled long-format models: diagonal (per-ticker
    variance) comes from the ML model, off-diagonal (correlation) from a
    RiskMetrics-style EWMA on raw ticker returns. Portfolio variance is the
    full quadratic form w^T Sigma_hat w.

    Mirrors ResidualBank's lifecycle exactly: training_safe_bulk_insert()+add() during
    training builds the *_in_sample bank, then .copy() carries it into the
    forecast phase where add()/general_bulk_insert() keep extending it one date at a time.

    No-look-ahead by construction: add(date, ...) rolls the EWMA covariance
    forward using the PREVIOUS call's return (self._prev_return), never the
    return being recorded in the same call -- same discipline as ResidualBank's
    stateful mu_hat.
    """

    def __init__(self, tickers: list[str], weights: pd.Series, ewma_lam: float = 0.94, ewma_min_periods: int = 30):
        self.tickers = list(tickers)
        self.weights = weights.loc[self.tickers].to_numpy()
        self.ewma_lam = ewma_lam
        self.ewma_min_periods = ewma_min_periods

        self.ticker_returns = pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], name="Date"))
        self.sigma_diag = pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], name="Date"))
        self.portfolio_variance = pd.Series(dtype=float, name="predicted_variance")
        self.cov_by_date: dict[pd.Timestamp, np.ndarray] = {}

        self._current_sigma: np.ndarray | None = None
        self._prev_return: np.ndarray | None = None

    def training_safe_bulk_insert(self, warmup_returns: pd.DataFrame) -> "EwmaVarianceBank":
        """
        TRAIN-ONLY burn-in: >= ewma_min_periods bars, no portfolio-variance output yet.

        The seed covariance (plain sample cov over the first ewma_min_periods bars)
        already encodes every one of those bars, INCLUDING the seed window's own last
        bar -- so when exactly ewma_min_periods bars are given, _prev_return is left
        None rather than that last bar's return. Setting it would make the very next
        add()/general_bulk_insert() call blend that same bar's outer product back into
        the seed cov a second time. Only bars processed AFTER the seed loop
        (len(R) > ewma_min_periods) leave a genuine "pending" return to chain forward,
        since those were walked one at a time via the recursion, not folded into the
        batch np.cov.
        """
        R = warmup_returns[self.tickers].to_numpy(dtype=float)
        if len(R) < self.ewma_min_periods:
            raise ValueError(f"Need >= {self.ewma_min_periods} warmup bars, got {len(R)}.")
        self._current_sigma = np.atleast_2d(np.cov(R[:self.ewma_min_periods], rowvar=False))
        for i in range(self.ewma_min_periods, len(R)):
            r_prev = R[i - 1]
            self._current_sigma = self.ewma_lam * self._current_sigma + (1 - self.ewma_lam) * np.outer(r_prev, r_prev)
        self._prev_return = R[-1] if len(R) > self.ewma_min_periods else None
        return self

    @staticmethod
    def _portfolio_variance_from_cov(cov: np.ndarray, sigma_diag: np.ndarray, weights: np.ndarray) -> float:
        """Diagonal = sigma_diag (model forecast), off-diagonal = cov's own correlation. w^T Sigma_hat w."""
        d_ = np.sqrt(np.diag(cov))
        d_ = np.where(d_ == 0, np.nan, d_)
        corr = cov / np.outer(d_, d_)
        np.fill_diagonal(corr, 1.0)
        sigma_hat = corr * np.sqrt(np.outer(sigma_diag, sigma_diag))
        np.fill_diagonal(sigma_hat, sigma_diag)
        return float(weights @ sigma_hat @ weights)

    def add(self, date, ticker_return_row: pd.Series, sigma_diag_row: pd.Series) -> float:
        """
        Own-day rule: date holds the ticker returns realized ON date, and the
        model's per-ticker variance forecast made FOR date (from date's own
        preceding feature bar). Returns the resulting portfolio variance.
        """
        if self._current_sigma is None:
            raise ValueError("EwmaVarianceBank: call training_safe_bulk_insert(...) before add(...).")

        date = pd.Timestamp(date)
        r_t = ticker_return_row.loc[self.tickers].to_numpy(dtype=float)
        sigma_diag = sigma_diag_row.loc[self.tickers].to_numpy(dtype=float)

        if self._prev_return is not None:
            self._current_sigma = self.ewma_lam * self._current_sigma + (1 - self.ewma_lam) * np.outer(
                self._prev_return, self._prev_return
            )
        cov = self._current_sigma.copy()
        self.cov_by_date[date] = cov

        sigma_diag = DATA_QUALITY_check_variance_floor(sigma_diag, dates=[date], context="EwmaVarianceBank.add")
        portfolio_variance = float(DATA_QUALITY_check_variance_floor(
            self._portfolio_variance_from_cov(cov, sigma_diag, self.weights), dates=[date], context="EwmaVarianceBank.add"
        )[0])

        self.ticker_returns.loc[date] = ticker_return_row.loc[self.tickers]
        self.sigma_diag.loc[date] = sigma_diag
        self.portfolio_variance.loc[date] = portfolio_variance
        self._prev_return = r_t
        return portfolio_variance

    def general_bulk_insert(self, dates, ticker_returns_df: pd.DataFrame, sigma_diag_df: pd.DataFrame) -> pd.Series:
        """
        dates MUST be chronologically sorted -- this is a stateful recursion, same
        math as looping add() one date at a time. Unlike that loop, this runs the
        recursion over plain numpy arrays and writes ticker_returns/sigma_diag/
        portfolio_variance/cov_by_date ONCE at the end via pd.concat, instead of
        one slow pandas .loc assignment per date -- the EWMA update itself can't
        be vectorized away (each date's covariance genuinely depends on the
        previous one), but the per-date pandas bookkeeping can be.
        """
        if self._current_sigma is None:
            raise ValueError("EwmaVarianceBank: call training_safe_bulk_insert(...) before general_bulk_insert(...).")

        dates = pd.DatetimeIndex(dates)
        R = ticker_returns_df.loc[dates, self.tickers].to_numpy(dtype=float)
        S = sigma_diag_df.loc[dates, self.tickers].to_numpy(dtype=float)
        n_dates, n_tickers = len(dates), len(self.tickers)

        sigma = self._current_sigma
        prev_return = self._prev_return
        covs = np.empty((n_dates, n_tickers, n_tickers))
        sigma_diag_out = np.empty((n_dates, n_tickers))
        portfolio_variances = np.empty(n_dates)

        for i in range(n_dates):
            if prev_return is not None:
                sigma = self.ewma_lam * sigma + (1 - self.ewma_lam) * np.outer(prev_return, prev_return)
            covs[i] = sigma
            sigma_diag_out[i] = DATA_QUALITY_check_variance_floor(S[i], dates=[dates[i]], context="EwmaVarianceBank.general_bulk_insert")
            portfolio_variances[i] = self._portfolio_variance_from_cov(sigma, sigma_diag_out[i], self.weights)
            prev_return = R[i]

        portfolio_variances = DATA_QUALITY_check_variance_floor(portfolio_variances, dates=dates, context="EwmaVarianceBank.general_bulk_insert")

        self.ticker_returns = pd.concat([self.ticker_returns, pd.DataFrame(R, index=dates, columns=self.tickers)])
        self.sigma_diag = pd.concat([self.sigma_diag, pd.DataFrame(sigma_diag_out, index=dates, columns=self.tickers)])
        self.portfolio_variance = pd.concat(
            [self.portfolio_variance, pd.Series(portfolio_variances, index=dates, name="predicted_variance")]
        )
        self.cov_by_date.update(zip(dates, covs))

        self._current_sigma = sigma
        self._prev_return = prev_return
        return pd.Series(portfolio_variances, index=dates, name="predicted_variance")

    def copy(self) -> "EwmaVarianceBank":
        new_bank = EwmaVarianceBank(self.tickers, pd.Series(self.weights, index=self.tickers), self.ewma_lam, self.ewma_min_periods)
        new_bank.ticker_returns = self.ticker_returns.copy()
        new_bank.sigma_diag = self.sigma_diag.copy()
        new_bank.portfolio_variance = self.portfolio_variance.copy()
        new_bank.cov_by_date = dict(self.cov_by_date)
        new_bank._current_sigma = None if self._current_sigma is None else self._current_sigma.copy()
        new_bank._prev_return = None if self._prev_return is None else self._prev_return.copy()
        return new_bank

    def to_dataframe(self) -> pd.DataFrame:
        """
        Combines ticker_returns, sigma_diag, and portfolio_variance into a single
        wide-format DataFrame. Columns are multi-indexed: (returns, TICKER),
        (sigma_diag, TICKER), (portfolio_variance, ''). cov_by_date is deliberately
        excluded -- pull that separately if needed.
        """
        return pd.concat(
            {
                "returns": self.ticker_returns,
                "sigma_diag": self.sigma_diag,
                "portfolio_variance": self.portfolio_variance.to_frame("portfolio_variance"),
            },
            axis=1,
        )


def multi_day_var_from_long_format_model(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    long_forecast_feature: "ForecastFeature",
    multi_day_var_calculation_window_size: int,
    var_confidence: float,
    initial_residual_bank: ResidualBank,
    initial_ewma_variance_bank: EwmaVarianceBank,
    run_key: str | None = None,
) -> pd.DataFrame:
    # initial_residual_bank already carries its rolling_mu_hat_window_size (fixed at
    # construction, see ResidualBank.__init__) -- general_bulk_insert below reads it off
    # the bank, not passed here.
    bank = initial_residual_bank.copy()
    ewma_bank = initial_ewma_variance_bank.copy()

    # see calendar.forecast_pairs() for why forecast_dates[0] itself is never scored here.
    feature_dates, outcome_dates = calendar.forecast_pairs()

    if len(outcome_dates) == 0:
        return pd.DataFrame(columns=["varForecast", "loss"])

    diag_by_ticker = pooled_model.predict_variance_batch_long(
        long_forecast_feature.X_forecast_sample, long_forecast_feature.id_cols, feature_dates
    )
    # own-day rule: the variance forecast made from feature_dates[i] is FOR outcome_dates[i]
    sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in calendar.tickers}, index=outcome_dates)

    predicted_variances = ewma_bank.general_bulk_insert(outcome_dates, calendar.forecast_data[calendar.tickers], sigma_diag_df)

    actual_returns = calendar.forecast_data.loc[outcome_dates, "portfolio_return"].to_numpy()

    bank.general_bulk_insert(outcome_dates, actual_returns, predicted_variances.to_numpy())

    var_df = bank.multi_day_var(outcome_dates, var_confidence, multi_day_var_calculation_window_size)
    var_df["loss"] = calendar.forecast_data.loc[var_df.index, "portfolio_loss"]

    calendar.add_predicted_var_batch(run_key or pooled_model.model_name, var_df["varForecast"])

    return var_df, bank, ewma_bank

