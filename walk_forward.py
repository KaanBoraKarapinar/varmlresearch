"""
Walk-forward VaR: many (train window -> forecast window) steps, one model per step, one stitched VaR series.

Dataclasses only hold decisions or results, plain functions do the work:

    WalkForwardSetting.make_calendars(...)   calendar decisions  -> list of calendars
    train_walk_forward(calendars, recipe)    calendars + recipe  -> list of WalkForwardSingleCalendarFit
    build_combined_residual_bank_WITHOUT_CALC(fits)   all windows -> ONE combined ResidualBank (no VaR)
    create_combined_residuals_bank_and_read_var_df(fits, var_confidence, window, ...)
                                             fits + any reading -> the VaR of every forecast day
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

import garch_models as gm
import tree_models as tm
from core import (
    DailyGroundTruthCalendar,
    EwmaVarianceBank,
    FeatureSpec,
    ResidualBank,
    SquaredReturn,
    Target,
    TrainedModel,
    build_features,
)


# --------------------------------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardSetting:
    """
    Calendar decisions.

    Windows advance by forecast_bars, so the forecast windows tile the data with no gap and no overlap
    (that is what lets them be stitched into one series). Each forecast window becomes part of the
    train window of the next steps: rolling keeps train_bars bars, expanding keeps everything since the start.
    """
    train_bars: int
    forecast_bars: int = 21
    expanding: bool = False
    warmup_bars: int = 30

    def __post_init__(self):
        if self.train_bars <= 0 or self.forecast_bars <= 0:
            raise ValueError("train_bars and forecast_bars must be positive.")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be non-negative.")

    def make_calendars(
        self, tickers_raw_price_data: pd.DataFrame, tickers: list[str], weights: dict[str, float] | pd.Series,
    ) -> list[DailyGroundTruthCalendar]:
        """
        One DailyGroundTruthCalendar per window, in time order (list position = window id).
        Windows are cut by bar position on the price index. The first bar has no return (pct_change), so the
        first train bar is at position warmup_bars + 1 and the warmup bars never reach it.
        """
        index = tickers_raw_price_data.sort_index().index
        first_train_pos = self.warmup_bars + 1
        train_end_pos = first_train_pos + self.train_bars - 1

        calendars = []
        # the last forecast bar (train_end_pos + forecast_bars) must exist
        while train_end_pos + self.forecast_bars < len(index):
            train_start_pos = first_train_pos if self.expanding else train_end_pos - self.train_bars + 1
            calendars.append(DailyGroundTruthCalendar(
                tickers_raw_price_data, tickers, weights,
                train_start=index[train_start_pos], train_end=index[train_end_pos],
                forecast_start=index[train_end_pos + 1], forecast_end=index[train_end_pos + self.forecast_bars],
                warmup_bars=self.warmup_bars,
            ))
            train_end_pos += self.forecast_bars

        if not calendars:
            raise ValueError(
                f"No window fits: {len(index)} bars, need warmup {self.warmup_bars} + 1 + train {self.train_bars} "
                f"+ forecast {self.forecast_bars}."
            )
        return calendars


@dataclass(frozen=True)
class ModelRecipe:
    """
    How every window's model is trained. The two bank groups are baked into the stored residuals at
    training time, so they live here and not in the VaR reading (create_combined_residuals_bank_and_read_var_df).

    model_family: 'xgboost' (features + target below), 'garch' (arch package, reads only the returns) or
    'ewma' (no model at all: the pure RiskMetrics EWMA variance, the "do nothing" baseline; model_format,
    features, target and model_kwargs are ignored).
    """
    model_format: Literal["long", "wide"]
    model_kwargs: dict = field(default_factory=dict)  # passed to xgboost, or to arch.arch_model for the garch family
    model_family: Literal["xgboost", "garch", "ewma"] = "xgboost"
    # what the model sees and what it is trained to predict (core/feature_blocks.py / core/targets.py)
    features: FeatureSpec = field(default_factory=FeatureSpec.default)
    target: Target = SquaredReturn(1.0)
    # ResidualBank
    mu_mode: Literal["rolling", "constant"] = "constant"
    default_mu_hat: float = 0.0
    rolling_mu_hat_window_size: int | None = None
    # EwmaVarianceBank (long only)
    ewma_lam: float = 0.94
    ewma_min_periods: int = 30

    def __post_init__(self):
        if self.model_format not in ("long", "wide"):
            raise ValueError(f"model_format must be 'long' or 'wide', got {self.model_format!r}")
        if self.model_family not in ("xgboost", "garch", "ewma"):
            raise ValueError(f"model_family must be 'xgboost', 'garch' or 'ewma', got {self.model_family!r}")


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardSingleCalendarFit:
    """
    What is left after training one window. residual_bank holds the model's in-sample residuals for the train
    dates followed by its out-of-sample residuals for the forecast dates.
    """
    window_id: int
    calendar: DailyGroundTruthCalendar
    trained_model: TrainedModel | None  # None for the 'ewma' family (no model)
    residual_bank: ResidualBank
    ewma_bank: EwmaVarianceBank | None = None  # long format only


def train_walk_forward(
    calendars: list[DailyGroundTruthCalendar], recipe: ModelRecipe,
) -> list[WalkForwardSingleCalendarFit]:
    """One model per calendar (fresh single-use banks every time). No VaR and no residual sharing here."""
    fits = []
    for window_id, calendar in enumerate(calendars):
        if recipe.model_family == "ewma":
            fits.append(_ewma_fit(window_id, calendar, recipe))
            continue
        training_feature, forecast_feature = build_features(calendar, recipe.model_format, recipe.features, recipe.target)
        residual_bank = ResidualBank(
            default_mu_hat=recipe.default_mu_hat, mu_mode=recipe.mu_mode,
            rolling_mu_hat_window_size=recipe.rolling_mu_hat_window_size,
        )
        train = tm.xgboost_train if recipe.model_family == "xgboost" else gm.garch_train
        if recipe.model_format == "long":
            model = train(
                calendar, training_feature, residual_bank,
                ewma_bank=EwmaVarianceBank(calendar, ewma_lam=recipe.ewma_lam, ewma_min_periods=recipe.ewma_min_periods),
                **recipe.model_kwargs,
            )
            bank, ewma_bank = model.calculate_multi_day_standart_residuals_from_long_format_model(calendar, forecast_feature)
        else:
            model = train(calendar, training_feature, residual_bank, **recipe.model_kwargs)
            bank = model.calculate_multi_day_standart_residuals_from_wide_format_model(calendar, forecast_feature)
            ewma_bank = None

        fits.append(WalkForwardSingleCalendarFit(window_id, calendar, model, bank, ewma_bank))
    return fits


def _ewma_fit(window_id: int, calendar: DailyGroundTruthCalendar, recipe: ModelRecipe) -> WalkForwardSingleCalendarFit:
    """
    The no-model baseline: the portfolio variance is the pure EWMA w' Sigma w of the realized returns (each
    date's value is built from strictly earlier returns), standardized and read exactly like a model's.
    Nothing is fitted, so train and forecast residuals are equally honest.
    """
    ewma_bank = calendar.ewma_realized_variance_bank(recipe.ewma_lam, recipe.ewma_min_periods)
    variance = ewma_bank.portfolio_variance
    dates = variance.index[variance.index.isin(calendar.train_data.index.append(calendar.forecast_dates))]
    returns = calendar.DANGEROUS_merged_warmedup_FORECAST_data["portfolio_return"].loc[dates]

    residual_bank = ResidualBank(
        default_mu_hat=recipe.default_mu_hat, mu_mode=recipe.mu_mode,
        rolling_mu_hat_window_size=recipe.rolling_mu_hat_window_size,
    )
    residual_bank.start_training(dates[0], calendar.frequency)
    residual_bank.training_safe_bulk_insert(dates, returns.to_numpy(), variance.loc[dates].to_numpy())
    return WalkForwardSingleCalendarFit(window_id, calendar, None, residual_bank, ewma_bank)


# --------------------------------------------------------------------------------------------------
# one residual bank for the whole walk-forward, read with any VaR reading
# --------------------------------------------------------------------------------------------------

def build_combined_residual_bank_WITHOUT_CALC(fits: list[WalkForwardSingleCalendarFit]) -> ResidualBank:
    """
    ONE ResidualBank for the whole walk-forward, in time order: the first window's bank as it is (its in-sample
    train residuals, then its out-of-sample forecast ones, the usual single-calendar start so the quantile
    window is full from the first day), followed by only the out-of-sample (forecast-date) rows of every later
    window. Once the window has moved past the first train, every residual it reads is out-of-sample.

    Each row keeps the predicted variance / sigma / mu_hat of the model that forecast that day. The bank is for
    reading (create_combined_residuals_bank_and_read_var_df, graphs), not to train or extend. Checks that the windows tile: each window's train ends
    on the previous window's last forecast day.
    """
    if not fits:
        raise ValueError("fits is empty.")
    for previous, current in zip(fits[:-1], fits[1:]):
        if current.calendar.train_end != previous.calendar.forecast_dates[-1]:
            raise ValueError(
                f"windows {previous.window_id} and {current.window_id} do not tile: window {current.window_id} train "
                f"ends {current.calendar.train_end.date()}, window {previous.window_id} forecast ends "
                f"{previous.calendar.forecast_dates[-1].date()}."
            )

    bank = fits[0].residual_bank.copy()
    bank.residuals = pd.concat(
        [fits[0].residual_bank.residuals]
        + [fit.residual_bank.residuals.loc[fit.calendar.forecast_dates] for fit in fits[1:]]
    )
    if not (bank.residuals.index.is_monotonic_increasing and bank.residuals.index.is_unique):
        raise ValueError("The combined residual bank is not chronological without duplicates.")
    return bank


def create_combined_residuals_bank_and_read_var_df(
    fits: list[WalkForwardSingleCalendarFit],
    var_confidence: float,
    multi_day_var_calculation_window_size: int | None,
    multi_day_var_min_periods: int | None = None,
) -> pd.DataFrame:
    """
    The VaR of every forecast date of the walk-forward, read off the combined bank (build_combined_residual_bank_WITHOUT_CALC(fits), built
    here), like calendar.var_from_residual_bank_forecast_period does for one calendar: how to read is chosen here,
    per call (var_confidence, the quantile window with None = expanding, its min_periods, see
    ResidualBank.multi_day_var), so the same fits can be read again any other way. Columns: varForecast,
    n_residuals_used, loss, window_id (the window whose model forecast that day).

    It feeds run_full_backtest_suite_single_window / run_level_and_volatility_suite_single_window directly (they
    read varForecast and loss).
    """
    bank = build_combined_residual_bank_WITHOUT_CALC(fits)
    forecast_dates = fits[0].calendar.forecast_dates
    for fit in fits[1:]:
        forecast_dates = forecast_dates.append(fit.calendar.forecast_dates)

    var_df = bank.multi_day_var(
        forecast_dates, var_confidence, multi_day_var_calculation_window_size, multi_day_var_min_periods
    )
    var_df["loss"] = pd.concat([fit.calendar.forecast_data["portfolio_loss"] for fit in fits]).loc[var_df.index]
    var_df["window_id"] = pd.concat(
        [pd.Series(fit.window_id, index=fit.calendar.forecast_dates) for fit in fits]
    ).loc[var_df.index]
    # the bank's frame starts as object columns, so the VaR columns can come back as object: make them numbers
    var_df = var_df.astype({"varForecast": float, "loss": float, "n_residuals_used": int, "window_id": int})
    if var_df[["varForecast", "loss"]].isna().any().any():
        raise ValueError("The var_df has NaN in varForecast / loss.")
    return var_df


def var_df_windows(var_df: pd.DataFrame) -> list[dict]:
    """The stitched var_df cut back into its windows, the format run_full_backtest_suite_all_windows reads."""
    return [
        {"window_id": window_id, "window_start": part.index[0], "window_end": part.index[-1], "forecast_df": part}
        for window_id, part in var_df.groupby("window_id", sort=True)
    ]
