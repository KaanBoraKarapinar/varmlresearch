"""
GARCH baseline on the same model API as tree_models.py: estimation and the one-step variance forecasts both
come from the `arch` package, this module only wires them into TrainedModel / the single-use banks.

    garch_train(calendar, training_feature, residual_bank)               wide: one GARCH on the portfolio return
    garch_train(calendar, training_feature, residual_bank, ewma_bank)    long: one GARCH per ticker + EWMA correlation

The feature objects are the ones feature_builders already makes: GARCH only reads the `portfolio_return` column
(wide) or the `r` column per (Date, ticker) (long). Parameters are estimated on the train rows only; the
forecast-period returns enter only as realized past, through arch's one-step forecast of the fixed model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from core import (
    DATA_QUALITY_check_variance_floor,
    EwmaVarianceBank,
    ForecastFeature,
    GroundTruthCalendar,
    ResidualBank,
    TrainedModel,
    TrainingFeature,
)

GARCH_DEFAULTS = dict(mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
# returns are multiplied by this for the optimizer (arch recommends percent), variances are divided back by scale**2
DEFAULT_SCALE = 100.0


def _fit_params(returns: pd.Series, arch_kwargs: dict, scale: float, label: str) -> pd.Series:
    result = arch_model(returns * scale, rescale=False, **arch_kwargs).fit(disp="off")
    if result.convergence_flag != 0:
        warnings.warn(
            f"[garch] {label}: optimizer did not converge (flag {result.convergence_flag}) on "
            f"{returns.index[0].date()} .. {returns.index[-1].date()}.",
            stacklevel=2,
        )
    return result.params


def _one_step_variance(returns: pd.Series, params: pd.Series, arch_kwargs: dict, scale: float) -> pd.Series:
    """
    arch's one-step forecast of the model fixed at `params`: for every date of `returns` (the origin), the
    variance of the bar AFTER it, made from returns up to and including that date. Real variance units.
    """
    fixed = arch_model(returns * scale, rescale=False, **arch_kwargs).fix(params)
    variance = fixed.forecast(horizon=1, start=0, reindex=False).variance["h.1"]
    return variance / scale ** 2


def check_chronological_order_and_combine_dataframes(train_returns: pd.Series, forecast_returns: pd.Series) -> pd.Series:
    """The train series followed by the rows of forecast_returns that are after it (rows already in train are dropped)."""
    forecast_returns = forecast_returns.sort_index()
    after = forecast_returns.loc[forecast_returns.index > train_returns.index[-1]]
    series = pd.concat([train_returns, after.astype(float)])
    if not (series.index.is_monotonic_increasing and series.index.is_unique):
        raise ValueError("garch: train and forecast returns do not form one chronological series.")
    return series


class TrainedGarchModel(TrainedModel):
    """
    params / train_returns: a Series each for wide, a {ticker: Series} each for long. The same params are used
    in-sample (train rows alone) and out of sample (train rows followed by the forecast rows).
    """

    def __init__(
        self, training_feature: TrainingFeature, params, train_returns, arch_kwargs: dict, scale: float,
        feature_cols: list[str], residual_bank_in_sample: ResidualBank | None = None,
    ):
        super().__init__(
            training_feature=training_feature, underlying_model=params, model_name="garch",
            target=training_feature.target, feature_cols=feature_cols,
            residual_bank_in_sample=residual_bank_in_sample,
        )
        self.params = params
        self.train_returns = train_returns
        self.arch_kwargs = arch_kwargs
        self.scale = scale

    # ---- wide: the portfolio return -----------------------------------------------------------------

    def predict_variance_batch_wide(self, forecast_feature: ForecastFeature, feature_dates) -> pd.Series:
        feature_dates = pd.DatetimeIndex(feature_dates)
        X = forecast_feature.X_forecast_sample
        missing = feature_dates.difference(X.index)
        if len(missing):
            raise KeyError(f"feature_dates missing from forecast_feature.X_forecast_sample: {list(missing)}")

        series = check_chronological_order_and_combine_dataframes(self.train_returns, X["portfolio_return"].astype(float))
        variance = _one_step_variance(series, self.params, self.arch_kwargs, self.scale)
        values = DATA_QUALITY_check_variance_floor(
            variance.loc[feature_dates].to_numpy(), dates=feature_dates, context="TrainedGarchModel.predict_variance_batch_wide",
        )
        return pd.Series(values, index=feature_dates, name="predicted_variance")

    def predict_variance_single_wide(self, forecast_feature: ForecastFeature, feature_date) -> float:
        return float(self.predict_variance_batch_wide(forecast_feature, [pd.Timestamp(feature_date)]).iloc[0])

    # ---- long: one GARCH per ticker -----------------------------------------------------------------

    def predict_variance_batch_long_by_ticker(
        self, X_long: pd.DataFrame, id_cols: pd.DataFrame, feature_dates, base_variance: pd.Series | None = None
    ) -> dict[str, pd.Series]:
        feature_dates = pd.DatetimeIndex(feature_dates)
        result: dict[str, pd.Series] = {}
        for ticker, rows in id_cols.groupby("ticker").groups.items():
            returns = pd.Series(X_long.loc[rows, "r"].to_numpy(dtype=float), index=id_cols.loc[rows, "Date"].to_numpy())
            series = check_chronological_order_and_combine_dataframes(self.train_returns[ticker], returns)
            variance = _one_step_variance(series, self.params[ticker], self.arch_kwargs, self.scale)
            missing = feature_dates.difference(variance.index)
            if len(missing):
                raise KeyError(f"garch {ticker}: feature_dates missing from the series: {list(missing)}")
            values = DATA_QUALITY_check_variance_floor(
                variance.loc[feature_dates].to_numpy(), dates=feature_dates,
                context=f"TrainedGarchModel.predict_variance_batch_long_by_ticker ({ticker})",
            )
            result[ticker] = pd.Series(values, index=feature_dates, name="predicted_variance")
        return result


def _garch_train_wide(
    training_feature: TrainingFeature, residual_bank: ResidualBank, arch_kwargs: dict, scale: float,
) -> TrainedGarchModel:
    X = training_feature.X_in_sample
    train_returns = X["portfolio_return"].astype(float).sort_index()
    params = _fit_params(train_returns, arch_kwargs, scale, "portfolio")

    model = TrainedGarchModel(
        training_feature, params, train_returns, arch_kwargs, scale, feature_cols=["portfolio_return"],
    )

    # same bank filling as tree_models._tree_train_wide, only the variances come from GARCH: the variance forecast
    # made from feature date f is FOR the bar after f, so the row is labeled by that outcome bar (own-day rule)
    actual_next = training_feature.DANGEROUS_r_next_in_sample
    variance = _one_step_variance(train_returns, params, arch_kwargs, scale)
    predicted_variances = DATA_QUALITY_check_variance_floor(
        variance.loc[X.index].to_numpy(), dates=X.index, context="garch_train (wide)",
    )

    outcome_dates = training_feature.training_feature_residual_outcome_dates
    residual_bank.start_training(outcome_dates[0], training_feature.frequency)
    residual_bank.training_safe_bulk_insert(
        dates=outcome_dates,
        actual_returns=actual_next.to_numpy(),
        predicted_variances=predicted_variances,
    )
    model.residual_bank_in_sample = residual_bank
    return model


def _garch_train_long(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, residual_bank: ResidualBank,
    ewma_bank: EwmaVarianceBank, arch_kwargs: dict, scale: float,
) -> TrainedGarchModel:
    id_cols = training_feature.X_in_sample_id_cols
    if id_cols is None:
        raise ValueError("garch_train (long) requires training_feature.X_in_sample_id_cols (Date/ticker per row).")
    X = training_feature.X_in_sample

    train_returns, params = {}, {}
    for ticker in calendar.tickers:
        rows = id_cols.index[id_cols["ticker"] == ticker]
        returns = pd.Series(X.loc[rows, "r"].to_numpy(dtype=float), index=id_cols.loc[rows, "Date"].to_numpy()).sort_index()
        train_returns[ticker] = returns
        params[ticker] = _fit_params(returns, arch_kwargs, scale, ticker)

    model = TrainedGarchModel(training_feature, params, train_returns, arch_kwargs, scale, feature_cols=["r"])
    # the per-ticker variances come from predict_variance_batch_long_by_ticker above, correlation from the EWMA bank
    model.residual_bank_in_sample, model.ewma_variance_bank_in_sample = model.long_format_calculate_training_standartized_residuals(
        calendar, training_feature, residual_bank, ewma_bank,
    )
    return model


def garch_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, residual_bank: ResidualBank,
    ewma_bank: EwmaVarianceBank | None = None,
    **arch_kwargs,
) -> TrainedGarchModel:
    """
    Same call shape as tree_models.xgboost_train. residual_bank / ewma_bank are SINGLE-USE (see tree_train);
    ewma_bank is only used by the long path (None -> EwmaVarianceBank(calendar)), passing one for wide raises.
    arch_kwargs go to arch.arch_model on top of GARCH_DEFAULTS (e.g. o=1 for GJR, dist="t"); `scale` is the
    return multiplier for the optimizer (default 100 = percent).
    """
    scale = float(arch_kwargs.pop("scale", DEFAULT_SCALE))
    arch_kwargs = {**GARCH_DEFAULTS, **arch_kwargs}

    if training_feature.type == "long":
        if ewma_bank is None:
            ewma_bank = EwmaVarianceBank(calendar)
        return _garch_train_long(calendar, training_feature, residual_bank, ewma_bank, arch_kwargs, scale)
    if ewma_bank is not None:
        raise ValueError("ewma_bank is only used by the long-format path.")
    return _garch_train_wide(training_feature, residual_bank, arch_kwargs, scale)
