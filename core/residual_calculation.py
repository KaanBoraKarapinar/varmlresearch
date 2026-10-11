import pandas as pd

from .feature_types import ForecastFeature
from .GroundTruthCalendar import GroundTruthCalendar
from .ResidualBank import ResidualBank
from .EwmaVarianceBank import EwmaVarianceBank
from .TrainedModel import TrainedModel


def calculate_multi_day_standart_residuals_from_wide_format_model(
    trained_model: TrainedModel,
    calendar: GroundTruthCalendar,
    forecast_feature: "ForecastFeature",
) -> ResidualBank:
    """
    Fills a copy of trained_model.residual_bank_in_sample with the standardized residuals of the
    whole forecast window and returns it. No VaR is computed here and no confidence is involved:
    read the VaR afterwards with calendar.var_from_residual_bank(bank, var_confidence, run_key)
    (or bank.multi_day_var) at any confidence.

    calendar.forecast_dates is the single source of truth for which bars get scored.
    calendar.forecast_data's index IS calendar.forecast_dates by construction, so actual returns are
    read straight off it -- no train_data ever touched here.

    feature_dates/outcome_dates come from calendar.forecast_pairs -- see where it is built
    in GroundTruthCalendar.__init__ for how forecast_dates[0] is priced from the last train bar's feature row.

    The returned bank carries every setting it was created with (mu_mode, rolling_mu_hat_window_size,
    quantile window, ...), so none of them is passed here.
    """
    bank = trained_model.residual_bank_in_sample.copy()
    feature_dates, outcome_dates = calendar.forecast_pairs

    predicted_variances = trained_model.predict_variance_batch_wide(forecast_feature, feature_dates)
    predicted_variances.index = outcome_dates

    actual_returns = calendar.forecast_data.loc[outcome_dates, "portfolio_return"].to_numpy()

    bank.general_bulk_insert(outcome_dates, actual_returns, predicted_variances.to_numpy())
    return bank


def calculate_multi_day_standart_residuals_from_long_format_model(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    long_forecast_feature: "ForecastFeature",
) -> tuple[ResidualBank, EwmaVarianceBank]:
    """
    Fills copies of pooled_model's two in-sample banks with the whole forecast window and returns
    (residual_bank, ewma_variance_bank): the EWMA bank gets the per-ticker model variances + realized
    ticker returns and yields the portfolio variances, which standardize the realized portfolio
    returns into the residual bank. No VaR is computed here and no confidence is involved: read the
    VaR afterwards with calendar.var_from_residual_bank(bank, var_confidence, run_key) (or
    bank.multi_day_var) at any confidence.

    The banks come from the model (pooled_model.residual_bank_in_sample /
    .ewma_variance_bank_in_sample) and carry every setting they were created with, so none of
    them is passed here. The model's own banks are not modified.
    """
    bank = pooled_model.residual_bank_in_sample.copy()
    ewma_bank = pooled_model.ewma_variance_bank_in_sample.copy()

    # see calendar.forecast_pairs: every forecast date is an outcome, the first one priced from the last train bar.
    feature_dates, outcome_dates = calendar.forecast_pairs

    diag_by_ticker = pooled_model.predict_variance_batch_long_by_ticker(
        long_forecast_feature.X_forecast_sample, long_forecast_feature.id_cols, feature_dates
    )
    # own-day rule: the variance forecast made from feature_dates[i] is FOR outcome_dates[i]
    sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in calendar.tickers}, index=outcome_dates)

    predicted_variances = ewma_bank.general_bulk_insert(outcome_dates, calendar.forecast_data[calendar.tickers], sigma_diag_df)

    actual_returns = calendar.forecast_data.loc[outcome_dates, "portfolio_return"].to_numpy()

    bank.general_bulk_insert(outcome_dates, actual_returns, predicted_variances.to_numpy())

    return bank, ewma_bank
