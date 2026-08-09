import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple
import plotly.graph_objects as go

# %%



from MLProject.var_backtest_utils import run_full_backtest_suite_single_window


# %%
class TrainingFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
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
    id_cols: pd.DataFrame | None


class ForecastFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    target_scale: float
    frequency: pd.Timedelta
    feature_cols: list[str]
    type: str  # "single" | "wide" | "long"

    ##important part:

    X_forecast_sample: pd.DataFrame
    y_forecast_sample: pd.DataFrame

    DANGEROUS_r_next_out_sample: pd.Series

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
        tickers_data: pd.DataFrame,
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

        missing = set(tickers) - set(tickers_data.columns)
        if missing:
            raise ValueError(f"tickers_data is missing columns: {sorted(missing)}")

        self.frequency = frequency
        self.tickers = list(tickers)
        self.weights = self._validate_weights(self.tickers, weights)

        full_data = self._compute_portfolio_return_and_loss(tickers_data, self.tickers, self.weights)

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
    def _compute_portfolio_return_and_loss(tickers_data: pd.DataFrame, tickers: list[str], weights: pd.Series) -> pd.DataFrame:
        full_data = tickers_data.sort_index().copy()
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
    def __init__(self, tickers_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed: bool = True):
        super().__init__(pd.Timedelta(hours=24), tickers_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed)


class HourlyGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed: bool = True):
        super().__init__(pd.Timedelta(hours=1), tickers_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed)


class MinuteGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed: bool = True):
        super().__init__(pd.Timedelta(minutes=1), tickers_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, drop_market_closed)

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
    def __init__(self, start_timestamp: pd.Timestamp, frequency: pd.Timedelta, initial_mu_hat: float = 0.0):
        if not isinstance(frequency, pd.Timedelta):
            raise TypeError(f"frequency must be a pd.Timedelta, got: {type(frequency)}")
        if frequency <= pd.Timedelta(0):
            raise ValueError(f"frequency must be positive: {frequency}")

        self.start_timestamp = start_timestamp
        self.frequency = frequency
        self.residuals = pd.DataFrame(columns=RESIDUAL_COLS, index=pd.DatetimeIndex([], name="Date"))

        # stateful mu_hat used to standardize the NEXT incoming residual.
        # Deliberately NOT recalculated until step_2, so step_1 never uses information
        # from the bar it is currently pricing (no look-ahead).
        self._current_single_stateful_mu = float(initial_mu_hat)

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

    def step_2_change_stateful_mu(self, window_size: int, as_of: pd.Timestamp | None = None) -> "ResidualBank":
        """
        Recompute the stateful mu_hat from the (now updated) windowed history. Call this
        AFTER step_1, once per bar, so the next bar's step_1 sees this bar's return.
        """
        window = self.windowed_read(window_size, as_of)
        if window.empty:
            raise ValueError(f"Cannot update mu_hat: residual window is empty (window_size={window_size}, as_of={as_of}).")

        self._current_single_stateful_mu = float(window["actual_return"].mean())
        return self

    def bulk_insert_respecting_order(self, dates, actual_returns, predicted_variances,
                                     window_size: int) -> "ResidualBank":
        """
        mu_hat_t = mean(actual_return) over the window_size bars STRICTLY BEFORE t
        (t's own return excluded). Same no-look-ahead rule as step_1/step_2, vectorized.
        """
        dates = pd.DatetimeIndex(dates, name="Date")
        actual_returns = np.asarray(actual_returns, dtype=float)
        predicted_variances = DATA_QUALITY_check_variance_floor(predicted_variances, dates=dates,
                                                                context="bulk_insert_respecting_order")

        combined_returns = pd.concat(
            [self.residuals["actual_return"], pd.Series(actual_returns, index=dates)]).sort_index()
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep="last")]

        mu_hat_new = combined_returns.rolling(window_size).mean().shift(1).loc[dates]
        if mu_hat_new.isna().any():
            bad = mu_hat_new[mu_hat_new.isna()].index.tolist()
            raise ValueError(f"Not enough history for rolling mu_hat (window_size={window_size}) at dates: {bad}")

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
        new_rows = DATA_QUALITY_replace_inf_with_warning(new_rows, context="bulk_insert_respecting_order").dropna()

        self.residuals = pd.concat([self.residuals.drop(index=new_rows.index, errors="ignore"), new_rows]).sort_index()
        self._current_single_stateful_mu = float(self.residuals["actual_return"].tail(window_size).mean())
        return self


    def windowed_read(self, window_size: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        df = self.residuals if as_of is None else self.residuals.loc[:as_of]
        return df.tail(window_size)

    def empirical_var_for_target(
            self,
            target_timestamp: pd.Timestamp,
            predicted_variance: float,
            var_confidence: float,
            window_size: int,
            enforce_window_size: bool = False,
    ) -> pd.Series:
        """
        Compute empirical VaR for a specific target bar. Requires the feature bar
        (target_timestamp - frequency) to already be present in the residual history,


        :param enforce_window_size: if True, raise unless the window contains exactly
            window_size residuals If False (default), any non-empty window up to window_size is accepted.
        :raises KeyError: if target_timestamp - frequency is not in self.residuals.
        :raises ValueError: if the window is empty, or (when enforce_window_size=True)
            shorter than window_size.
        :return: pd.Series indexed by target_timestamp, with fields varForecast, mu_hat,
            predicted_variance, last_feature_timestamp, n_residuals_used.
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
        window = self.windowed_read(window_size, as_of=last_feature_timestamp)

        print(
            f"[empirical_var_for_target] target={target_timestamp}, using {len(window)} residual(s) up to {last_feature_timestamp}")

        if window.empty:
            raise ValueError(f"Residual window is empty (window_size={window_size}, as_of={last_feature_timestamp}).")

        if enforce_window_size and len(window) < window_size:
            raise ValueError(
                f"enforce_window_size=True but only {len(window)}/{window_size} residuals "
                f"available as of {last_feature_timestamp} (target={target_timestamp})."
            )

        mu_hat = float(window["actual_return"].mean())
        q = np.quantile(window["standardized_residual"], 1 - var_confidence)
        sigma_hat = np.sqrt(predicted_variance)
        var_estimate = -(mu_hat + sigma_hat * q)

        return pd.Series(
            {
                "varForecast": var_estimate,
                "mu_hat": mu_hat,
                "predicted_variance": predicted_variance,
                "last_feature_timestamp": last_feature_timestamp,
                "n_residuals_used": len(window),
            },
            name=target_timestamp,
        )

    def multi_day_var(self, dates, var_confidence: float, window_size: int) -> pd.DataFrame:
        """
        Pure calculation: dates must already be in self.residuals (bulk_insert first).
        VaR for date t uses standardized_residual from the window_size bars STRICTLY
        BEFORE t (t excluded).
        """
        dates = pd.DatetimeIndex(dates, name="Date")
        missing = dates.difference(self.residuals.index)
        if len(missing):
            raise KeyError(f"multi_day_var: dates missing from residual history: {list(missing)}")

        q_series = (
            self.residuals["standardized_residual"]
            .rolling(window_size)
            .apply(lambda w: np.quantile(w, 1 - var_confidence), raw=True)
            .shift(1)
        )

        result = self.residuals.loc[dates, ["mu_hat", "predicted_sigma"]].copy()
        result["quantile"] = q_series.loc[dates]

        if result["quantile"].isna().any():
            bad = result.index[result["quantile"].isna()].tolist()
            raise ValueError(
                f"Not enough prior residual history for quantile (window_size={window_size}) at dates: {bad}")

        result["varForecast"] = -(result["mu_hat"] + result["predicted_sigma"] * result["quantile"])
        return result[["varForecast"]]


    def copy(self) -> "ResidualBank":
        new_bank = ResidualBank(self.start_timestamp, self.frequency, self._current_single_stateful_mu)
        new_bank.residuals = self.residuals.copy()
        return new_bank


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
    ):
        self.training_feature = training_feature
        self.underlying_model = underlying_model
        self.model_name = model_name
        self.target_scale = target_scale
        self.feature_cols = feature_cols
        self.residual_bank_in_sample = residual_bank_in_sample

    @abstractmethod
    def predict_variance(self, forecast_feature: "ForecastFeature", feature_date: pd.Timestamp) -> float:
        """Predict the variance for the bar immediately AFTER feature_date."""

    @abstractmethod
    def predict_variance_batch(self, forecast_feature: "ForecastFeature", feature_dates: pd.Timestamp) -> float:
        """Predict the variances for the given feature dates ."""


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
    window_size: int,
    var_confidence: float,
) -> pd.DataFrame:
    """
    forecast_dates is no longer a parameter, calendar.forecast_dates is the
    single source of truth for which bars get scored.
    """
    bank = trained_model.residual_bank_in_sample.copy()
    forecast_dates = calendar.forecast_dates

    if len(forecast_dates) == 0:
        return pd.DataFrame(columns=["varForecast", "loss"])

    feature_dates = forecast_dates - calendar.frequency
    predicted_variances = trained_model.predict_variance_batch(forecast_feature, feature_dates)
    predicted_variances.index = forecast_dates

    actual_returns = np.array([calendar.realized_return(d) for d in forecast_dates])

    bank.bulk_insert_respecting_order(forecast_dates, actual_returns, predicted_variances.to_numpy(), window_size)

    var_df = bank.multi_day_var(forecast_dates, var_confidence, window_size)
    var_df["loss"] = [calendar.realized_loss(d) for d in var_df.index]

    for date, var_estimate in var_df["varForecast"].items():
        calendar.add_predicted_var(date, var_estimate)

    return var_df


def multi_day_var_from_long_format_model(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    per_ticker_forecast_feature: dict[str, "ForecastFeature"],
    window_size: int,
    var_confidence: float,
    initial_residual_bank: ResidualBank,
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
) -> pd.DataFrame:
    bank = initial_residual_bank.copy()
    forecast_dates = calendar.forecast_dates

    if len(forecast_dates) == 0:
        return pd.DataFrame(columns=["varForecast", "loss"])

    feature_dates = forecast_dates - calendar.frequency
    predicted_variances = predict_portfolio_variance_batch_with_ewma(
        pooled_model, calendar, per_ticker_forecast_feature, feature_dates, ewma_lam, ewma_min_periods
    )
    predicted_variances.index = forecast_dates

    actual_returns = np.array([calendar.realized_return(d) for d in forecast_dates])

    bank.bulk_insert_respecting_order(forecast_dates, actual_returns, predicted_variances.to_numpy(), window_size)

    var_df = bank.multi_day_var(forecast_dates, var_confidence, window_size)
    var_df["loss"] = [calendar.realized_loss(d) for d in var_df.index]

    for date, var_estimate in var_df["varForecast"].items():
        calendar.add_predicted_var(date, var_estimate)

    return var_df

# %%

def _build_ewma_cov_by_date(returns_df: pd.DataFrame, lam: float = 0.94, min_periods: int = 30) -> dict[pd.Timestamp, np.ndarray]:
    """RiskMetrics-style rolling EWMA covariance. Date t's matrix uses returns up to t-1 only."""
    R = returns_df.to_numpy(dtype=float)
    dates = returns_df.index

    sigma = np.atleast_2d(np.cov(R[:min_periods], rowvar=False))

    cov_by_date = {}
    for t in range(min_periods, len(R)):
        r_prev = R[t - 1]  # last realized return, valid one-step-ahead for date t
        sigma = lam * sigma + (1 - lam) * np.outer(r_prev, r_prev)
        cov_by_date[dates[t]] = sigma.copy()
    return cov_by_date


def _correlation_at(cov_by_date: dict[pd.Timestamp, np.ndarray], date, ewma_min_periods: int) -> np.ndarray:
    date = pd.Timestamp(date)
    if date not in cov_by_date:
        raise KeyError(f"No EWMA covariance at {date} (min_periods={ewma_min_periods} not reached, or date out of range).")
    sigma = cov_by_date[date]
    d = np.sqrt(np.diag(sigma))
    d[d == 0] = np.nan
    corr = sigma / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def predict_portfolio_variance_batch_with_ewma(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    per_ticker_forecast_feature: dict[str, "ForecastFeature"],
    feature_dates,
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
) -> pd.Series:
    """
    Diagonal (per-ticker variance) from pooled_model, off-diagonal (correlation)
    from EWMA on calendar.ticker_returns. Portfolio variance = w^T Sigma w.
    """
    feature_dates = pd.DatetimeIndex(feature_dates)

    missing_tickers = set(calendar.tickers) - set(per_ticker_forecast_feature.keys())
    if missing_tickers:
        raise KeyError(f"per_ticker_forecast_feature missing tickers: {sorted(missing_tickers)}")

    cov_by_date = _build_ewma_cov_by_date(calendar.ticker_returns, lam=ewma_lam, min_periods=ewma_min_periods)

    diag_by_ticker = {
        t: pooled_model.predict_variance_batch(per_ticker_forecast_feature[t], feature_dates)
        for t in calendar.tickers
    }

    w = calendar.weights.loc[calendar.tickers].to_numpy()
    variances = []
    for date in feature_dates:
        sigma_diag = np.array([diag_by_ticker[t].loc[date] for t in calendar.tickers])
        corr = _correlation_at(cov_by_date, date, ewma_min_periods)
        sigma_hat = corr * np.sqrt(np.outer(sigma_diag, sigma_diag))
        np.fill_diagonal(sigma_hat, sigma_diag)
        variances.append(float(w @ sigma_hat @ w))

    variances = DATA_QUALITY_check_variance_floor(
        np.array(variances), dates=feature_dates, context="predict_portfolio_variance_batch_with_ewma"
    )
    return pd.Series(variances, index=feature_dates, name="predicted_variance")


def multi_day_var_from_long_format_model(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    per_ticker_forecast_feature: dict[str, "ForecastFeature"],
    forecast_dates: pd.DatetimeIndex,
    window_size: int,
    var_confidence: float,
    initial_residual_bank: ResidualBank,
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
) -> pd.DataFrame:
    bank = initial_residual_bank.copy()
    forecast_dates = pd.DatetimeIndex(forecast_dates)

    if len(forecast_dates) == 0:
        return pd.DataFrame(columns=["varForecast", "loss"])

    feature_dates = forecast_dates - calendar.frequency
    predicted_variances = predict_portfolio_variance_batch_with_ewma(
        pooled_model, calendar, per_ticker_forecast_feature, feature_dates, ewma_lam, ewma_min_periods
    )
    predicted_variances.index = forecast_dates

    actual_returns = np.array([calendar.realized_return(d) for d in forecast_dates])

    bank.bulk_insert_respecting_order(forecast_dates, actual_returns, predicted_variances.to_numpy(), window_size)

    var_df = bank.multi_day_var(forecast_dates, var_confidence, window_size)
    var_df["loss"] = [calendar.realized_loss(d) for d in var_df.index]

    for date, var_estimate in var_df["varForecast"].items():
        calendar.add_predicted_var(date, var_estimate)

    return var_df

