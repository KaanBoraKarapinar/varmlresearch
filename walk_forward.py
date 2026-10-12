"""
Walk-forward VaR: many (train window -> forecast window) steps, one model per step, one stitched VaR series.

Dataclasses only hold decisions or results, plain functions do the work:

    WalkForwardSetting.make_calendars(...)   calendar decisions  -> list of calendars
    train_walk_forward(calendars, recipe)    calendars + recipe  -> list of WalkForwardSingleCalendarFit
    share_residuals(fits, residual_sharing)  residual decision   -> list of fits (banks possibly shared)
    var_by_period(fit, var_setting)          one fit             -> forecast / train / entire VaR
    build_var_df(fits, var_setting, ...)     everything          -> the stitched forecast var_df
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import pandas as pd

import feature_builders as fb
import garch_models as gm
import tree_models as tm
from core import (
    DEFAULT_EXPANDING_MIN_PERIODS,
    DailyGroundTruthCalendar,
    EwmaVarianceBank,
    ResidualBank,
    TrainedModel,
)


# --------------------------------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardSetting:
    """
    Calendar decisions and the residual-sharing decision.

    Windows advance by forecast_bars, so the forecast windows tile the data with no gap and no overlap
    (that is what lets them be stitched into one series). Each forecast window becomes part of the
    train window of the next steps: rolling keeps train_bars bars, expanding keeps everything since the start.

    residual_sharing:
        'none'       every window's VaR quantile reads its own model's residuals (in-sample ones for the train dates)
        'carry_over' for a date that was a forecast day of an earlier window, the earlier model's out-of-sample
                     residual replaces the new model's (optimistic, in-sample) one
    """
    train_bars: int
    forecast_bars: int = 21
    expanding: bool = False
    warmup_bars: int = 30
    residual_sharing: Literal["none", "carry_over"] = "none"

    def __post_init__(self):
        if self.train_bars <= 0 or self.forecast_bars <= 0:
            raise ValueError("train_bars and forecast_bars must be positive.")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be non-negative.")
        if self.residual_sharing not in ("none", "carry_over"):
            raise ValueError(f"residual_sharing must be 'none' or 'carry_over', got {self.residual_sharing!r}")

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
    training time, so they live here and not in VarSetting.
    """
    model_format: Literal["long", "wide"]
    model_kwargs: dict = field(default_factory=dict)  # passed to xgboost, or to arch.arch_model for the garch family
    model_family: Literal["xgboost", "garch"] = "xgboost"
    # features
    windows: tuple = (5, 10, 21)
    ewma_span: int = 21
    sq_return_lags: tuple | None = None
    target_scale: float = 1.0
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
        if self.model_family not in ("xgboost", "garch"):
            raise ValueError(f"model_family must be 'xgboost' or 'garch', got {self.model_family!r}")


@dataclass(frozen=True)
class VarSetting:
    """
    How the VaR is read off a residual bank (see ResidualBank.multi_day_var). window None = expanding.
    min_periods = fewest prior residuals a day needs to get a VaR; 30 (like the single-calendar notebooks) so a
    train of exactly `window` bars still works. None would mean "the window must be full".
    """
    var_confidence: float
    window: int | None
    min_periods: int | None = DEFAULT_EXPANDING_MIN_PERIODS


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
    trained_model: TrainedModel
    residual_bank: ResidualBank
    ewma_bank: EwmaVarianceBank | None = None  # long format only


def train_walk_forward(
    calendars: list[DailyGroundTruthCalendar], recipe: ModelRecipe,
) -> list[WalkForwardSingleCalendarFit]:
    """One model per calendar (fresh single-use banks every time). No VaR and no residual sharing here."""
    build_features = fb.build_long_features if recipe.model_format == "long" else fb.build_wide_features

    fits = []
    for window_id, calendar in enumerate(calendars):
        training_feature, forecast_feature = build_features(
            calendar, windows=recipe.windows, ewma_span=recipe.ewma_span,
            sq_return_lags=recipe.sq_return_lags, target_scale=recipe.target_scale,
        )
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


# --------------------------------------------------------------------------------------------------
# VaR
# --------------------------------------------------------------------------------------------------

def var_by_period(fit: WalkForwardSingleCalendarFit, var_setting: VarSetting) -> dict[str, pd.DataFrame]:
    """
    One fit's VaR for its forecast, train and entire period, all with the same VaR setting, read off
    fit.residual_bank only (no cross-window logic here, see share_residuals). Keys: forecast / train / entire.
    The first min_periods bank dates of train / entire carry no VaR (not enough prior residuals).
    """
    _, min_periods = ResidualBank._resolve_quantile_window(var_setting.window, var_setting.min_periods)
    calendar, bank = fit.calendar, fit.residual_bank
    return {
        "forecast": calendar.var_from_residual_bank_forecast_period(
            bank, var_setting.var_confidence, var_setting.window, f"walk_forward_{fit.window_id}", min_periods,
        ),
        "train": calendar.var_from_residual_bank_train_period(
            bank, var_setting.var_confidence, var_setting.window, min_periods,
        ),
        "entire": calendar.var_from_residual_bank_entire_period(
            bank, var_setting.var_confidence, var_setting.window, min_periods,
        ),
    }


# --------------------------------------------------------------------------------------------------
# residual sharing + the final var_df
# --------------------------------------------------------------------------------------------------

def share_residuals(
    fits: list[WalkForwardSingleCalendarFit], residual_sharing: Literal["none", "carry_over"],
) -> list[WalkForwardSingleCalendarFit]:
    """
    'none': the fits unchanged.
    'carry_over': in time order, every date of a fit's bank that was a FORECAST day of an earlier window
    gets the earlier model's out-of-sample row (all columns) instead of the new model's in-sample row. The
    first window has no earlier model, so it is unchanged. Returns new fits; the given ones are not modified.
    """
    if residual_sharing == "none":
        return list(fits)
    if residual_sharing != "carry_over":
        raise ValueError(f"residual_sharing must be 'none' or 'carry_over', got {residual_sharing!r}")

    out_of_sample_rows = pd.DataFrame()
    shared = []
    for fit in fits:
        bank = fit.residual_bank.copy()
        if len(out_of_sample_rows):
            overlap = bank.residuals.index.intersection(out_of_sample_rows.index)
            bank.residuals.loc[overlap] = out_of_sample_rows.loc[overlap]
        shared.append(replace(fit, residual_bank=bank))

        # the ORIGINAL fit's forecast rows are honest out-of-sample residuals of this model
        forecast_rows = fit.residual_bank.residuals.loc[fit.calendar.forecast_dates]
        out_of_sample_rows = pd.concat([out_of_sample_rows, forecast_rows]) if len(out_of_sample_rows) else forecast_rows
    return shared


def build_var_df(
    fits: list[WalkForwardSingleCalendarFit], var_setting: VarSetting,
    residual_sharing: Literal["none", "carry_over"] = "none",
) -> pd.DataFrame:
    """
    The stitched out-of-sample VaR: every window's forecast-period var_df (after the residual-sharing
    decision), one after the other. Columns: varForecast, n_residuals_used, loss, window_id.

    Checks that the windows tile: each window's train ends on the previous window's last forecast day, and the
    result has every forecast date exactly once, without NaN. It feeds run_full_backtest_suite_single_window /
    run_level_and_volatility_suite_single_window directly (they read varForecast and loss).
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

    parts = []
    for fit in share_residuals(fits, residual_sharing):
        forecast_df = var_by_period(fit, var_setting)["forecast"].copy()
        forecast_df["window_id"] = fit.window_id
        parts.append(forecast_df)
    # the bank's frame starts as object columns, so the VaR columns can come back as object: make them numbers
    var_df = pd.concat(parts).astype({"varForecast": float, "loss": float, "n_residuals_used": int})

    expected_dates = fits[0].calendar.forecast_dates
    for fit in fits[1:]:
        expected_dates = expected_dates.append(fit.calendar.forecast_dates)
    if not var_df.index.equals(expected_dates):
        raise ValueError("The stitched var_df does not cover every forecast date exactly once, in order.")
    if var_df[["varForecast", "loss"]].isna().any().any():
        raise ValueError("The stitched var_df has NaN in varForecast / loss.")
    return var_df


def stitched_residual_bank(fits: list[WalkForwardSingleCalendarFit]) -> ResidualBank:
    """
    One ResidualBank holding each window's out-of-sample (forecast-date) residual rows, in time order: what the
    graphs read for the hover data (predicted_variance, sigma, mu_hat, standardized residual) and for the
    model-output-vs-realized-variance plot. For reading only, it is not a bank to train or extend.
    """
    if not fits:
        raise ValueError("fits is empty.")
    bank = fits[0].residual_bank.copy()
    bank.residuals = pd.concat([fit.residual_bank.residuals.loc[fit.calendar.forecast_dates] for fit in fits])
    return bank


def var_df_windows(var_df: pd.DataFrame) -> list[dict]:
    """The stitched var_df cut back into its windows, the format run_full_backtest_suite_all_windows reads."""
    return [
        {"window_id": window_id, "window_start": part.index[0], "window_end": part.index[-1], "forecast_df": part}
        for window_id, part in var_df.groupby("window_id", sort=True)
    ]
