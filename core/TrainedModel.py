import pandas as pd
from abc import ABC, abstractmethod

from .feature_types import TrainingFeature, ForecastFeature
from .GroundTruthCalendar import GroundTruthCalendar
from .ResidualBank import ResidualBank
from .EwmaVarianceBank import EwmaVarianceBank


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
        residual_bank_in_sample: ResidualBank | None = None,
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

    def predict_variance_batch_long_by_ticker(
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

    def long_format_calculate_training_standartized_residuals(
        self,
        calendar: GroundTruthCalendar,
        training_feature: TrainingFeature,
        residual_bank: ResidualBank,
        ewma_bank: EwmaVarianceBank,
    ) -> tuple[ResidualBank, EwmaVarianceBank]:
        """
        Builds the portfolio-level in-sample ResidualBank + EwmaVarianceBank for a
        pooled long-format model, to be used as residual_bank_in_sample /
        ewma_variance_bank_in_sample.

        Diagonal (per-ticker variance) comes from this model via
        predict_variance_batch_long_by_ticker, off-diagonal (correlation) from EWMA on
        calendar.train_data_with_warmup ONLY (never touches forecast-period returns, so no
        leakage). Portfolio variance = w^T Sigma_hat w, computed by
        EwmaVarianceBank.general_bulk_insert().

        Bank entries are labeled by the OUTCOME day (own-day rule): Date=X holds the
        actual_return that REALLY happened on X, and the predicted_variance that was
        produced from X's own preceding feature bar.

        training_feature.X_out_sample_last_index (the window's final feature row, no known target
        yet) is NOT seeded here: its outcome is forecast_dates[0], which the forecast phase prices
        itself through calendar.forecast_pairs.

        residual_bank / ewma_bank are SINGLE-USE: they carry the settings (mu_mode, default_mu_hat,
        rolling_mu_hat_window_size, quantile window, ewma_lam, ewma_min_periods, ...), are filled IN PLACE here and
        marked `trained` (start_training). A bank that was already used raises instead of leaking into a new run.
        The returned banks are the same objects that were passed in.
        """
        if training_feature.type != "long":
            raise ValueError(f"long_format_calculate_training_standartized_residuals requires a long-format TrainingFeature, got type={training_feature.type!r}")
        if training_feature.X_in_sample_id_cols is None:
            raise ValueError("long_format_calculate_training_standartized_residuals requires training_feature.X_in_sample_id_cols (Date/ticker per row).")
        # checked up front so a refused run leaves neither bank half-filled
        if residual_bank.trained or ewma_bank.trained:
            raise ValueError("residual_bank / ewma_bank were already used for training; create new banks for every training run.")

        tickers = calendar.tickers

        # feature_dates is the feature builder's responsibility (TrainingFeature.feature_dates),
        # not recomputed here from X_in_sample_id_cols.
        feature_dates = training_feature.feature_dates
        diag_by_ticker = self.predict_variance_batch_long_by_ticker(training_feature.X_in_sample, training_feature.X_in_sample_id_cols, feature_dates)
        sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in tickers}, index=feature_dates)

        # ---- EWMA correlation: warmup + TRAIN returns only (never forecast-period returns) ----
        # warmup bars (before train_start, past data) are only ever read by the seed covariance below; they are
        # never an outcome date, so they never become a row of either bank.
        returns = calendar.train_data_with_warmup[tickers].sort_index()
        ewma_min_periods = ewma_bank.ewma_min_periods
        # Seed window = the ewma_min_periods bars ENDING at seed_end: the train_start bar when there is enough warmup,
        # else the ewma_min_periods-th bar (= the first ewma_min_periods train bars when warmup_bars = 0).
        seed_end = max(len(calendar.warmup_data), ewma_min_periods - 1)
        ewma_bank.start_training(calendar).training_safe_bulk_insert(
            returns.iloc[seed_end - ewma_min_periods + 1: seed_end + 1]
        )
        # usable_dates: feature_dates that are also past the burn-in (a covariance, built from strictly earlier
        # bars, to price against). Threshold is ewma_min_periods - 1, not ewma_min_periods: the seed window's own
        # last bar is a valid feature address too -- the seed cov already encodes that bar (see
        # EwmaVarianceBank.training_safe_bulk_insert). With enough warmup nothing is dropped. X_in_sample never
        # contains the last train date (its target/r_next is NaN by construction), so usable_dates always has a
        # following bar -- calendar.next_bars gives outcome_dates, their paired own-day OUTCOME bar.
        usable_dates = calendar.train_dates_past_burn_in(feature_dates, ewma_min_periods - 1)
        outcome_dates = calendar.next_bars(usable_dates)

        # own-day rule: sigma_diag_df is forecast made FROM usable_dates[i]'s features, which is
        # FOR outcome_dates[i] -- re-key it to outcome_dates so general_bulk_insert's EWMA
        # recursion (keyed/stepped by outcome day) lines up with the diagonal it's paired with,
        # same convention calculate_multi_day_standart_residuals_from_long_format_model uses for the forecast phase.
        sigma_diag_outcome = sigma_diag_df.loc[usable_dates].set_axis(outcome_dates)
        portfolio_variances = ewma_bank.general_bulk_insert(outcome_dates, returns, sigma_diag_outcome)

        # training_safe_bulk_insert (not general_bulk_insert) because the bank starts empty here:
        # general_bulk_insert needs rolling_mu_hat_window_size bars of PRIOR history to already be
        # IN the bank before it can compute a rolling mu_hat, which a from-scratch seed can
        # never satisfy for its own first rolling_mu_hat_window_size dates. training_safe_bulk_insert
        # ramps the stateful mu_hat up from scratch instead (step_1/step_2), so it works from empty.
        residual_bank.start_training(outcome_dates[0], calendar.frequency)
        residual_bank.training_safe_bulk_insert(
            dates=outcome_dates,
            actual_returns=calendar.train_data.loc[outcome_dates, "portfolio_return"].to_numpy(),
            predicted_variances=portfolio_variances.to_numpy(),
        )

        return residual_bank, ewma_bank

    def calculate_multi_day_standart_residuals_from_wide_format_model(
        self,
        calendar: GroundTruthCalendar,
        forecast_feature: ForecastFeature,
    ) -> ResidualBank:
        """
        Fills a copy of self.residual_bank_in_sample with the standardized residuals of the
        whole forecast window and returns it. No VaR is computed here and no confidence is involved:
        read the VaR afterwards with calendar.var_from_residual_bank(bank, var_confidence, run_key)
        (or bank.multi_day_var) at any confidence.

        calendar.forecast_dates is the single source of truth for which bars get scored.
        calendar.forecast_data's index IS calendar.forecast_dates by construction, so actual returns are
        read straight off it -- no train_data ever touched here.

        feature_dates/outcome_dates come from calendar.forecast_pairs -- see where it is built
        in GroundTruthCalendar.__init__ for how forecast_dates[0] is priced from the last train bar's feature row.

        The returned bank carries every setting it was created with (mu_mode, rolling_mu_hat_window_size,
        quantile window, ...), so none of them is passed here. The model's own bank is not modified.
        """
        bank = self.residual_bank_in_sample.copy()
        feature_dates, outcome_dates = calendar.forecast_pairs

        predicted_variances = self.predict_variance_batch_wide(forecast_feature, feature_dates)
        predicted_variances.index = outcome_dates

        actual_returns = calendar.forecast_data.loc[outcome_dates, "portfolio_return"].to_numpy()

        bank.general_bulk_insert(outcome_dates, actual_returns, predicted_variances.to_numpy())
        return bank

    def calculate_multi_day_standart_residuals_from_long_format_model(
        self,
        calendar: GroundTruthCalendar,
        long_forecast_feature: ForecastFeature,
    ) -> tuple[ResidualBank, EwmaVarianceBank]:
        """
        Fills copies of self's two in-sample banks with the whole forecast window and returns
        (residual_bank, ewma_variance_bank): the EWMA bank gets the per-ticker model variances + realized
        ticker returns and yields the portfolio variances, which standardize the realized portfolio
        returns into the residual bank. No VaR is computed here and no confidence is involved: read the
        VaR afterwards with calendar.var_from_residual_bank(bank, var_confidence, run_key) (or
        bank.multi_day_var) at any confidence.

        The banks come from the model (self.residual_bank_in_sample /
        self.ewma_variance_bank_in_sample) and carry every setting they were created with, so none of
        them is passed here. The model's own banks are not modified.
        """
        bank = self.residual_bank_in_sample.copy()
        ewma_bank = self.ewma_variance_bank_in_sample.copy()

        # see calendar.forecast_pairs: every forecast date is an outcome, the first one priced from the last train bar.
        feature_dates, outcome_dates = calendar.forecast_pairs

        diag_by_ticker = self.predict_variance_batch_long_by_ticker(
            long_forecast_feature.X_forecast_sample, long_forecast_feature.id_cols, feature_dates
        )
        # own-day rule: the variance forecast made from feature_dates[i] is FOR outcome_dates[i]
        sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in calendar.tickers}, index=outcome_dates)

        predicted_variances = ewma_bank.general_bulk_insert(outcome_dates, calendar.forecast_data[calendar.tickers], sigma_diag_df)

        actual_returns = calendar.forecast_data.loc[outcome_dates, "portfolio_return"].to_numpy()

        bank.general_bulk_insert(outcome_dates, actual_returns, predicted_variances.to_numpy())

        return bank, ewma_bank
