from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .additional_feature_data import check_additional_feature_data
from .EwmaVarianceBank import EwmaVarianceBank
from .units import check_unit

if TYPE_CHECKING:
    from .ResidualBank import ResidualBank


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
        unit: str,
        drop_market_closed: bool = True,
        warmup_bars: int = 0,
        additional_feature_data: pd.DataFrame | None = None,
    ):
        self.unit = check_unit(unit)   # the unit every return of this calendar (ticker and portfolio) is expressed in
        if not isinstance(warmup_bars, (int, np.integer)) or warmup_bars < 0:
            raise ValueError(f"warmup_bars must be a non-negative int, got {warmup_bars!r}")
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

        # only the raw prices come in: the unit decides which return is computed from them
        if self.unit == "log":
            log_returns = self._calculate_log_return(tickers_raw_price_data)
            full_data = self._compute_portfolio_return_and_loss_FOR_LOG_RETURN(log_returns, self.tickers, self.weights)
        else:   # percent: the simple (pct_change) return, as a fraction
            pct_returns = self._calculate_pct_return(tickers_raw_price_data)
            full_data = self._compute_portfolio_return_and_loss_FOR_PCT_RETURN(pct_returns, self.tickers, self.weights)

        self.train_start, self.train_end, self.forecast_start, self.forecast_end = self._validate_windows(
            full_data, train_start, train_end, forecast_start, forecast_end
        )

        self.train_data = self._slice_with_gap_check(full_data, self.train_start, self.train_end, "train", drop_market_closed)
        self.forecast_data = self._slice_with_gap_check(full_data, self.forecast_start, self.forecast_end, "forecast", drop_market_closed)
        self.forecast_dates = self.forecast_data.index

        # Warmup: `warmup_bars` real bars strictly BEFORE train_start. Past data only (no look-ahead): it is used to
        # warm up rolling features and to seed the EWMA covariance so that the very first train bar already has a
        # feature row and a covariance to be priced against, instead of dropping the first ~50 train bars.
        # train_data itself stays pure (what you asked to train on); warmup bars are never training rows.
        self.warmup_bars = int(warmup_bars)
        before_train = full_data.loc[:self.train_start].iloc[:-1]
        if len(before_train) < self.warmup_bars:
            raise ValueError(
                f"warmup_bars={self.warmup_bars} but only {len(before_train)} bar(s) exist before "
                f"train_start={self.train_start.date()} in the price data."
            )
        self.warmup_data = before_train.iloc[len(before_train) - self.warmup_bars:].copy()
        if self.warmup_bars and self.warmup_data[self.tickers + ["portfolio_return"]].isna().any().any():
            raise ValueError(
                f"warmup_bars={self.warmup_bars} reaches the first price bar (which has no return): "
                f"use a smaller warmup_bars or a later train_start."
            )
        self.train_data_with_warmup = pd.concat([self.warmup_data, self.train_data])

        # The last TRAIN bar's (target-less) feature row prices forecast_dates[0]. Defined once here so the
        # feature builders (which feature rows to compute) and forecast_pairs (which pairs to form) agree.
        self.train_data_last_index = self.train_data.index[-1:]
        self.forecast_feature_dates = self.train_data_last_index.append(self.forecast_dates)

        # Forecast-phase scoring pairs (feature_dates, outcome_dates), built once: feature_dates[i] is the bar
        # immediately before outcome_dates[i], drawn by position (not frequency subtraction, for the same
        # skip-weekends/holidays reason as next_bars). outcome_dates is every forecast date; feature_dates is the
        # last TRAIN bar followed by forecast_dates[:-1], so forecast_dates[0] is priced from the last train bar's
        # feature row and n forecast bars give n pairs.
        self.forecast_pairs = (self.forecast_feature_dates[:-1], self.forecast_dates)

        # DANGEROUS: (warmup +) train_data + forecast_data concatenated. Only for feature builders
        # that need rolling-window history to reach into train_data at the start of the forecast window (warmup).
        # It starts at the same bar as train_data_with_warmup, so a train-side and a forecast-side feature row of the
        # same date are computed from the same history (the train_end row is identical on both sides).
        self.DANGEROUS_merged_warmedup_FORECAST_data = pd.concat(
            [self.warmup_data, self.train_data, self.forecast_data]
        ).sort_index()

        # The additional feature data (columns (tag, scope), see additional_feature_data.py) is cut the same two ways as the
        # price data, so the train rows of a feature can never see a forecast-period value:
        #   train_additional_data_with_warmup                       <-> train_data_with_warmup
        #   DANGEROUS_merged_warmedup_FORECAST_additional_data      <-> DANGEROUS_merged_warmedup_FORECAST_data
        self.train_additional_data_with_warmup = None
        self.DANGEROUS_merged_warmedup_FORECAST_additional_data = None
        if additional_feature_data is not None:
            check_additional_feature_data(additional_feature_data, self.tickers)
            self.train_additional_data_with_warmup = self._additional_rows_on(
                additional_feature_data, self.train_data_with_warmup.index, "warmup + train")
            self.DANGEROUS_merged_warmedup_FORECAST_additional_data = self._additional_rows_on(
                additional_feature_data, self.DANGEROUS_merged_warmedup_FORECAST_data.index, "warmup + train + forecast")

    @staticmethod
    def _additional_rows_on(additional: pd.DataFrame, bars: pd.DatetimeIndex, label: str) -> pd.DataFrame:
        """The rows of the additional feature data on exactly these bars."""
        missing = bars.difference(additional.index)
        if len(missing):
            raise ValueError(
                f"additional_feature_data has no row for {len(missing)} bar(s) of the {label} part "
                f"(first: {missing[0].date()}): it must cover every bar of the calendar."
            )
        return additional.loc[bars]

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
    def _calculate_log_return(tickers_raw_price_data: pd.DataFrame) -> pd.DataFrame:
        return np.log(tickers_raw_price_data.sort_index()).diff()

    @staticmethod
    def _compute_portfolio_return_and_loss_FOR_PCT_RETURN(
        tickers_pct_change_data: pd.DataFrame, tickers: list[str], weights: pd.Series,
    ) -> pd.DataFrame:
        """The portfolio's percent return is the weighted sum of the tickers' percent returns (all as fractions)."""
        full_data = tickers_pct_change_data.sort_index().copy()
        portfolio_return = full_data[tickers].mul(weights, axis="columns").sum(axis=1, min_count=len(tickers))
        full_data["portfolio_return"] = portfolio_return
        full_data["portfolio_loss"] = -portfolio_return
        return full_data

    @staticmethod
    def _compute_portfolio_return_and_loss_FOR_LOG_RETURN(
        tickers_log_return_data: pd.DataFrame, tickers: list[str], weights: pd.Series,
    ) -> pd.DataFrame:
        """
        The portfolio's log return is the log of its simple return, log(1 + sum w_i * (exp(r_i) - 1)), NOT the weighted
        sum of the tickers' log returns (log returns do not add across assets).
        """
        full_data = tickers_log_return_data.sort_index().copy()
        simple_returns = np.expm1(full_data[tickers])
        portfolio_simple_return = simple_returns.mul(weights, axis="columns").sum(axis=1, min_count=len(tickers))
        full_data["portfolio_return"] = np.log1p(portfolio_simple_return)
        full_data["portfolio_loss"] = -full_data["portfolio_return"]
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

    def previous_bars(self, dates) -> pd.DatetimeIndex:
        """
        Position-based previous bar for each date in `dates`, the mirror of next_bars: looked up against the
        full train+forecast calendar (DANGEROUS_merged_warmedup_FORECAST_data), so it skips weekends/holidays
        instead of subtracting a fixed offset (a Monday's previous bar is the Friday, not the Sunday).
        """
        index = self.DANGEROUS_merged_warmedup_FORECAST_data.index
        dates = pd.DatetimeIndex(dates)
        positions = index.get_indexer(dates)
        if (positions == -1).any():
            bad = dates[positions == -1]
            raise KeyError(f"Dates not found in calendar: {list(bad)}")
        if (positions - 1 < 0).any():
            bad = dates[positions - 1 < 0]
            raise KeyError(f"Dates have no preceding bar in calendar: {list(bad)}")
        return index[positions - 1]

    def train_dates_past_burn_in(self, candidate_dates, burn_in_bars: int) -> pd.DatetimeIndex:
        """
        Drops the `candidate_dates` (e.g. a TrainingFeature's feature_dates) that still sit INSIDE the burn-in:
        the first `burn_in_bars` bars of warmup + train. Only train dates that come after those bars have a
        covariance built from strictly earlier data to be priced against.

        With warmup_bars >= burn_in_bars the burn-in is covered by the warmup bars, so every train date is
        returned and nothing is dropped; with warmup_bars = 0 it cuts the first burn_in_bars train bars.
        Prints how many candidates survived, and raises if none did.
        """
        candidate_dates = pd.DatetimeIndex(candidate_dates)
        past_burn_in = self.train_data_with_warmup.index[burn_in_bars:]
        kept_dates = candidate_dates[candidate_dates.isin(past_burn_in)]

        n_candidates = len(candidate_dates)
        n_kept = len(kept_dates)
        cutoff = past_burn_in[0].date() if len(past_burn_in) else "N/A"
        print(
            f"[INFO] train_dates_past_burn_in: {n_kept}/{n_candidates} candidate dates are past the burn-in "
            f"(burn_in_bars={burn_in_bars}, warmup_bars={self.warmup_bars}, train_data starts "
            f"{self.train_data.index[0].date()}, burn-in ends {cutoff}, {n_candidates - n_kept} date(s) dropped)."
        )

        if kept_dates.empty:
            raise ValueError(f"train_dates_past_burn_in: no candidate dates are past the burn-in (burn_in_bars={burn_in_bars}).")
        return kept_dates

    def ewma_realized_variance_bank(self, ewma_lam: float = 0.94, ewma_min_periods: int = 30) -> EwmaVarianceBank:
        """
        A plain EwmaVarianceBank filled with the REALIZED returns only (no model forecast mixed in), as the
        realized-variance reference for the model outputs: bank.portfolio_variance is the pure EWMA
        w^T Sigma w, bank.cov_by_date the matrices (plot_ewma_bank works on it).

        Warmed up and walked the same way training walks the model's bank: the seed is the `ewma_min_periods`
        bars ending at seed_end = max(warmup_bars, ewma_min_periods - 1) on warmup + train, then every later
        bar of train + forecast is inserted. So the dates are the model banks' outcome dates, and each
        date's variance is built from strictly earlier returns (the same information the model had).

        It reads forecast-period returns: an EVALUATION/plot reference only, never a model input.
        """
        returns = self.DANGEROUS_merged_warmedup_FORECAST_data[self.tickers]
        seed_end = max(len(self.warmup_data), ewma_min_periods - 1)
        if seed_end + 1 >= len(returns):
            raise ValueError(f"ewma_realized_variance_bank: need more than {seed_end + 1} bars (warmup + train + forecast), got {len(returns)}.")

        bank = EwmaVarianceBank(self, ewma_lam=ewma_lam, ewma_min_periods=ewma_min_periods)
        bank.start_training(self).training_safe_bulk_insert(returns.iloc[seed_end - ewma_min_periods + 1: seed_end + 1])
        bank.general_bulk_insert(returns.index[seed_end + 1:], returns)
        return bank

    def ewma_realized_variance(self, ewma_lam: float = 0.94, ewma_min_periods: int = 30) -> pd.Series:
        """
        The REALIZED EWMA variance of every bar: v_d already contains r_d (the return realized ON d), unlike
        ewma_realized_variance_bank, whose value at d is built from the returns BEFORE d.

        So v_d is the bank's value of the next bar (v_d = w^T Sigma_(d+1) w, Sigma_(d+1) = lam * Sigma_d +
        (1 - lam) * r_d r_d^T); the very last bar has no next bar in the bank, so its step is taken here.
        The first value (the seed bar) is the seed covariance itself, which already contains that bar.
        Indexed seed bar .. last forecast bar. EVALUATION/plot reference only, it reads forecast-period returns.
        """
        bank = self.ewma_realized_variance_bank(ewma_lam, ewma_min_periods)
        last_date = bank.portfolio_variance.index[-1]
        r_last = bank.ticker_returns.loc[last_date].to_numpy(dtype=float)
        cov_with_last = ewma_lam * bank.cov_by_date[last_date].to_numpy() + (1 - ewma_lam) * np.outer(r_last, r_last)
        last_variance = float(bank.weights @ cov_with_last @ bank.weights)

        seed_end = max(len(self.warmup_data), ewma_min_periods - 1)
        index = self.DANGEROUS_merged_warmedup_FORECAST_data.index[seed_end:]
        return pd.Series(np.append(bank.portfolio_variance.to_numpy(), last_variance), index=index, name="ewma_realized_variance")

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

    def var_from_residual_bank_forecast_period(
        self,
        bank: "ResidualBank",
        var_confidence: float,
        multi_day_var_calculation_window_size: int | None,
        multi_day_var_min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Reads the VaR of every forecast date off a bank that already holds the forecast window's
        residuals (see calculate_multi_day_standart_residuals_from_*) and adds the realized loss.
        How to read is chosen here, per call: var_confidence, the quantile window (None = expanding) and
        its min_periods, see ResidualBank.multi_day_var.
        """
        var_df = bank.multi_day_var(
            self.forecast_dates, var_confidence, multi_day_var_calculation_window_size, multi_day_var_min_periods
        )
        var_df["loss"] = self.forecast_data.loc[var_df.index, "portfolio_loss"]
        return var_df

    def var_from_residual_bank_train_period(
        self,
        bank: "ResidualBank",
        var_confidence: float,
        multi_day_var_calculation_window_size: int | None,
        multi_day_var_min_periods: int,
    ) -> pd.DataFrame:
        """
        The VaR of the TRAIN dates the bank holds (in-sample), with the realized loss. The first
        multi_day_var_min_periods bank dates are dropped: they do not have that many prior residuals yet
        (multi_day_var would raise).
        """
        train_dates = bank.residuals.index[bank.residuals.index.isin(self.train_data.index)]
        var_df = bank.multi_day_var(
            train_dates[multi_day_var_min_periods:], var_confidence,
            multi_day_var_calculation_window_size, multi_day_var_min_periods,
        )
        var_df["loss"] = self.train_data.loc[var_df.index, "portfolio_loss"]
        return var_df

    def var_from_residual_bank_entire_period(
        self,
        bank: "ResidualBank",
        var_confidence: float,
        multi_day_var_calculation_window_size: int | None,
        multi_day_var_min_periods: int,
    ) -> pd.DataFrame:
        """
        The VaR of every train + forecast date the bank holds, with the realized loss. Same as
        var_from_residual_bank_train_period: the first multi_day_var_min_periods bank dates are dropped.
        """
        entire_dates = bank.residuals.index[bank.residuals.index.isin(self.train_data.index.append(self.forecast_dates))]
        var_df = bank.multi_day_var(
            entire_dates[multi_day_var_min_periods:], var_confidence,
            multi_day_var_calculation_window_size, multi_day_var_min_periods,
        )
        var_df["loss"] = self.DANGEROUS_merged_warmedup_FORECAST_data.loc[var_df.index, "portfolio_loss"]
        return var_df


class DailyGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, unit: str, drop_market_closed: bool = True, warmup_bars: int = 0, additional_feature_data: pd.DataFrame | None = None):
        super().__init__(pd.Timedelta(hours=24), tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, unit, drop_market_closed, warmup_bars, additional_feature_data)


class HourlyGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, unit: str, drop_market_closed: bool = True, warmup_bars: int = 0, additional_feature_data: pd.DataFrame | None = None):
        super().__init__(pd.Timedelta(hours=1), tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, unit, drop_market_closed, warmup_bars, additional_feature_data)


class MinuteGroundTruthCalendar(GroundTruthCalendar):
    def __init__(self, tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, unit: str, drop_market_closed: bool = True, warmup_bars: int = 0, additional_feature_data: pd.DataFrame | None = None):
        super().__init__(pd.Timedelta(minutes=1), tickers_raw_price_data, tickers, weights, train_start, train_end, forecast_start, forecast_end, unit, drop_market_closed, warmup_bars, additional_feature_data)
