import warnings
from typing import Literal

import numpy as np
import pandas as pd

from .data_quality import DATA_QUALITY_check_variance_floor, DATA_QUALITY_replace_inf_with_warning

RESIDUAL_COLS = ["actual_return", "predicted_variance", "predicted_sigma", "mu_hat", "standardized_residual"]
VARIANCE_FLOOR = 1e-12
DEFAULT_EXPANDING_MIN_PERIODS = 30


class ResidualBank:
    """
    Standardized-residual history of the PORTFOLIO plus the settings that decide how a residual is
    written: default_mu_hat, mu_mode, rolling_mu_hat_window_size -- how mu_hat is formed when a residual
    is inserted (baked into every stored row, so changing them needs a new training run).

    Everything about READING the VaR (var_confidence, the quantile window, its min_periods) is passed
    to multi_day_var, so one filled bank can be read any way you like without retraining.

    start_timestamp / frequency are bound to a calendar when training starts (see start_training), so a bank
    built by hand does not need them. A bank is single-use: training fills it in place and marks it
    `trained`, after which only the forecast phase adds to it (on a copy), see
    calculate_multi_day_standart_residuals_from_*.
    """

    def __init__(
        self,
        default_mu_hat: float = 0.0,
        mu_mode: Literal["rolling", "constant"] = "constant",
        rolling_mu_hat_window_size: int | None = None,
        start_timestamp: pd.Timestamp | None = None,
        frequency: pd.Timedelta | None = None,
    ):
        if frequency is not None:
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
        self.residuals =pd.DataFrame(columns=RESIDUAL_COLS, index=pd.DatetimeIndex([], name="Date"))
        # False until training starts to fill this bank (see start_training); a bank is single-use
        self.trained = False

        # stateful mu_hat used to standardize the NEXT incoming residual.
        # Deliberately NOT recalculated until step_2, so step_1 never uses information
        # from the bar it is currently pricing (no look-ahead). In mu_mode="constant",
        # step_2/general_bulk_insert never touch this again -- it stays pinned at default_mu_hat.
        self._current_single_stateful_mu = float(default_mu_hat)

    @property
    def current_mu_hat(self) -> float:
        return self._current_single_stateful_mu

    def _check_dates_follow_history(self, dates: pd.DatetimeIndex) -> None:
        """
        Rows are only ever appended in chronological order: dates written together must be sorted without
        duplicates, and the first one must come after the last date already in the bank. Going back in time or
        re-writing a date raises instead of silently overwriting a row.
        """
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError("ResidualBank: dates must be chronologically sorted, without duplicates.")
        if len(self.residuals) and dates[0] <= self.residuals.index.max():
            raise ValueError(
                f"ResidualBank: {dates[0]} is not after the last date already in the bank "
                f"({self.residuals.index.max()}); dates must be added in chronological order."
            )

    def DANGEROUS_add_residual(self, date, actual_return: float, predicted_variance: float, mu_hat: float) -> "ResidualBank":
        """
        NO STATEFUL MU TRACKING, RISK OF LOOK FORWARD BIAS
        """
        date = pd.Timestamp(date)
        self._check_dates_follow_history(pd.DatetimeIndex([date]))
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

        dates must be chronologically sorted and come after the bank's last date (checked, see
        _check_dates_follow_history): rows are appended, never overwritten.
        """
        dates = pd.DatetimeIndex(dates, name="Date")
        self._check_dates_follow_history(dates)
        actual_returns = np.asarray(actual_returns, dtype=float)
        predicted_variances = DATA_QUALITY_check_variance_floor(predicted_variances, dates=dates,
                                                                context="general_bulk_insert")

        if self.mu_mode == "constant":
            mu_hat_new = pd.Series(self.default_mu_hat, index=dates)
        else:
            combined_returns = pd.concat(
                [self.residuals["actual_return"], pd.Series(actual_returns, index=dates)]).sort_index()

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

        self.residuals = pd.concat([self.residuals, new_rows])
        if self.mu_mode == "constant":
            self._current_single_stateful_mu = self.default_mu_hat
        else:
            self._current_single_stateful_mu = float(self.residuals["actual_return"].tail(self.rolling_mu_hat_window_size).mean())
        return self


    def windowed_read(self, window_size: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Generic last-N-residuals reader, purpose-agnostic -- callers pass whichever window (rolling_mu_hat_window_size or multi_day_var_calculation_window_size) is relevant to them."""
        df = self.residuals if as_of is None else self.residuals.loc[:as_of]
        return df.tail(window_size)

    @staticmethod
    def _resolve_quantile_window(
        multi_day_var_calculation_window_size: int | None, multi_day_var_min_periods: int | None
    ) -> tuple[bool, int]:
        """
        The quantile-window rules shared by every VaR reader (multi_day_var, empirical_var_for_target), so
        they cannot drift apart: window int = rolling (the last N residuals), None = expanding (all of them);
        min_periods None defaults to N for the rolling window (the window must be full) and to
        DEFAULT_EXPANDING_MIN_PERIODS for the expanding window. Returns (expanding, min_periods).
        """
        if multi_day_var_calculation_window_size is not None and multi_day_var_calculation_window_size <= 0:
            raise ValueError(
                f"multi_day_var_calculation_window_size must be positive or None (expanding), got: {multi_day_var_calculation_window_size!r}"
            )
        if (
            multi_day_var_calculation_window_size is not None and multi_day_var_min_periods is not None
            and multi_day_var_min_periods > multi_day_var_calculation_window_size
        ):
            raise ValueError(
                f"multi_day_var_min_periods ({multi_day_var_min_periods}) cannot exceed "
                f"multi_day_var_calculation_window_size ({multi_day_var_calculation_window_size})."
            )
        expanding = multi_day_var_calculation_window_size is None
        min_periods = multi_day_var_min_periods
        if min_periods is None:
            min_periods = DEFAULT_EXPANDING_MIN_PERIODS if expanding else multi_day_var_calculation_window_size
        return expanding, min_periods

    def empirical_var_for_target(
            self,
            target_timestamp: pd.Timestamp,
            last_feature_timestamp: pd.Timestamp,
            predicted_variance: float,
            var_confidence: float,
            multi_day_var_calculation_window_size: int | None,
            multi_day_var_min_periods: int | None = None,
    ) -> pd.Series:
        """
        Compute empirical VaR for a specific target bar. last_feature_timestamp is the bar immediately
        BEFORE the target (calendar.previous_bars gives it by bar position, skipping weekends/holidays) and
        must already be present in the residual history.

        multi_day_var_calculation_window_size and self.rolling_mu_hat_window_size are
        independent -- the quantile window (VaR tail estimate) and the mu_hat window
        (drift estimate, only read in mu_mode="rolling") don't have to be the same
        length, so they're read separately via two windowed_read calls. The mu_hat
        window is fixed on the bank at construction time (not passed here) since
        mu_mode/its window is a bank-wide policy, not something that varies per call.

        :param multi_day_var_calculation_window_size, multi_day_var_min_periods: exactly the rules of
            multi_day_var (shared through _resolve_quantile_window): int window = the last N residuals up to
            last_feature_timestamp, None = all of them (expanding); min_periods None defaults to N for the
            rolling window and to DEFAULT_EXPANDING_MIN_PERIODS for the expanding one. Only gates the
            quantile window, not the mu_hat window.
        :raises KeyError: if last_feature_timestamp is not in self.residuals.
        :raises ValueError: if last_feature_timestamp is not before target_timestamp, if the window settings
            are invalid, or if fewer than min_periods residuals are available.
        :return: pd.Series indexed by target_timestamp, with fields varForecast, mu_hat,
            predicted_variance, last_feature_timestamp, n_residuals_used (quantile window size).
        """
        target_timestamp = pd.Timestamp(target_timestamp)
        last_feature_timestamp = pd.Timestamp(last_feature_timestamp)
        if last_feature_timestamp >= target_timestamp:
            raise ValueError(
                f"empirical_var_for_target: last_feature_timestamp {last_feature_timestamp} must be before "
                f"target {target_timestamp}."
            )
        expanding, min_periods = self._resolve_quantile_window(
            multi_day_var_calculation_window_size, multi_day_var_min_periods
        )

        if last_feature_timestamp not in self.residuals.index:
            raise KeyError(
                f"empirical_var_for_target: {last_feature_timestamp} (the bar before target {target_timestamp}) "
                f"is not in residual history. Cannot price {target_timestamp} without its immediately "
                f"preceding feature bar."
            )

        predicted_variance = float(
            DATA_QUALITY_check_variance_floor(predicted_variance, dates=[target_timestamp], context="empirical_var_for_target")[0])
        history = self.residuals.loc[:last_feature_timestamp]
        quantile_window = history if expanding else history.tail(multi_day_var_calculation_window_size)

        if len(quantile_window) < min_periods:
            raise ValueError(
                f"Not enough prior residual history for quantile "
                f"(multi_day_var_calculation_window_size={multi_day_var_calculation_window_size}, "
                f"min_periods={min_periods}) at dates: [{target_timestamp}]"
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

    def multi_day_var(
        self,
        dates,
        var_confidence: float,
        multi_day_var_calculation_window_size: int | None,
        multi_day_var_min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Pure calculation: dates must already be in self.residuals (bulk_insert first).
        VaR for date t uses standardized_residual from the bars STRICTLY BEFORE t
        (t excluded). mu_hat is read straight off the stored column (written by
        general_bulk_insert using rolling_mu_hat_window_size at insert time), not
        recomputed here.

        All of the reading is chosen per call, so one filled bank can be read any way you like:
            var_confidence: the quantile level is 1 - var_confidence.
            multi_day_var_calculation_window_size:
                int  -> rolling window: the last N residuals before t.
                None -> expanding window: ALL residuals before t (use every bar available).
            multi_day_var_min_periods: fewest prior residuals a date needs to get a VaR (fewer ->
                ValueError). None defaults to N for the rolling window (the window must be full) and to
                DEFAULT_EXPANDING_MIN_PERIODS for the expanding window (a quantile over a handful of
                residuals is meaningless).

        :return: DataFrame indexed by dates with varForecast and n_residuals_used (how
            many residuals the quantile of that date was computed from).
        """
        expanding, min_periods = self._resolve_quantile_window(
            multi_day_var_calculation_window_size, multi_day_var_min_periods
        )
        dates = pd.DatetimeIndex(dates, name="Date")
        missing = dates.difference(self.residuals.index)
        if len(missing):
            raise KeyError(f"multi_day_var: dates missing from residual history: {list(missing)}")

        standardized = self.residuals["standardized_residual"]
        roller = (
            standardized.expanding(min_periods=min_periods) if expanding
            else standardized.rolling(multi_day_var_calculation_window_size, min_periods=min_periods)
        )
        q_series = roller.apply(lambda w: np.quantile(w, 1 - var_confidence), raw=True).shift(1)

        result = self.residuals.loc[dates, ["mu_hat", "predicted_sigma"]].copy()
        result["quantile"] = q_series.loc[dates]

        if result["quantile"].isna().any():
            bad = result.index[result["quantile"].isna()].tolist()
            raise ValueError(
                f"Not enough prior residual history for quantile "
                f"(multi_day_var_calculation_window_size={multi_day_var_calculation_window_size}, "
                f"min_periods={min_periods}) at dates: {bad}"
            )

        result["varForecast"] = -(result["mu_hat"] + result["predicted_sigma"] * result["quantile"])

        # residuals strictly before each date == its row position (index is sorted and unique)
        n_prior = self.residuals.index.get_indexer(dates)
        result["n_residuals_used"] = n_prior if expanding else np.minimum(n_prior, multi_day_var_calculation_window_size)
        return result[["varForecast", "n_residuals_used"]]


    def copy(self) -> "ResidualBank":
        new_bank = ResidualBank(
            self.default_mu_hat, self.mu_mode, self.rolling_mu_hat_window_size,
            self.start_timestamp, self.frequency,
        )
        new_bank.trained = self.trained
        new_bank.residuals = self.residuals.copy()
        new_bank._current_single_stateful_mu = self._current_single_stateful_mu
        return new_bank

    def start_training(self, start_timestamp: pd.Timestamp, frequency: pd.Timedelta) -> "ResidualBank":
        """
        Called once by training right before it fills this bank: binds the calendar's start/frequency and
        marks the bank as trained. A bank is single-use: handing a trained bank to another training run
        raises, so one run's residuals can never leak into the next (create a new ResidualBank per run).
        """
        if self.trained:
            raise ValueError("This ResidualBank was already used for training; create a new ResidualBank.")
        self.start_timestamp = start_timestamp
        self.frequency = frequency
        self.trained = True
        return self

    def to_dataframe(self) -> pd.DataFrame:
        return self.residuals
