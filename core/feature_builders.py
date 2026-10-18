"""
Calendar -> (TrainingFeature, ForecastFeature).

What the features are comes from a FeatureSpec (feature_blocks.py), what the model is trained to predict from a
Target (targets.py); wide and long are two views of the same blocks:

    build_features(calendar, "long", spec, target)
    build_long_features(calendar, windows, ewma_span, sq_return_lags, target_scale)   # the default spec, kept
    build_wide_features(...)                                                          # for the old call sites
"""

from __future__ import annotations

import pandas as pd

from .feature_blocks import FeatureSpec
from .feature_types import TrainingFeature, ForecastFeature
from .GroundTruthCalendar import GroundTruthCalendar
from .targets import SquaredReturn, Target


def build_features(
        calendar: GroundTruthCalendar,
        model_format: str,
        spec: FeatureSpec | None = None,
        target: Target | None = None,
) -> tuple[TrainingFeature, ForecastFeature]:
    """
    spec None -> FeatureSpec.default(), target None -> SquaredReturn(1.0).

    The blocks are computed twice, so the train rows can never see a forecast-period return:
    train rows from the train side only (warmup + train), forecast rows from the merged data (warmup + train +
    forecast: the last train bar + the forecast dates, see calendar.forecast_feature_dates, need the history).
    """
    if model_format not in ("wide", "long"):
        raise ValueError(f"model_format must be 'wide' or 'long', got {model_format!r}")
    spec = spec or FeatureSpec.default()
    target = target or SquaredReturn(1.0)

    tickers = list(calendar.tickers)
    columns = tickers + ["portfolio_return"]
    train_frame = _frame(calendar.train_data_with_warmup[columns], tickers, spec, target, model_format)
    forecast_frame = _frame(calendar.DANGEROUS_merged_warmedup_FORECAST_data[columns], tickers, spec, target, model_format)

    feature_cols = [c for c in train_frame.columns if c not in ("Date", "ticker", "r_next", "ewma_base_variance")]
    if not feature_cols:
        raise ValueError("The FeatureSpec produced no feature for this view.")

    # the train frame ends on the last train bar, so that bar's r_next (the first forecast return) is NaN by itself
    train_full = _cut(train_frame, calendar.train_data.index, feature_cols, target, model_format, f"{model_format} train")
    forecast_full = _cut(forecast_frame, calendar.forecast_feature_dates, feature_cols, target, model_format,
                         f"{model_format} forecast")

    if train_full.empty:
        raise ValueError(
            f"No usable {model_format}-format rows in train window [{calendar.train_start} .. {calendar.train_end}].")
    if forecast_full.empty:
        raise ValueError(
            f"No usable {model_format}-format rows in forecast window [{calendar.forecast_start} .. {calendar.forecast_end}].")

    clean_mask = train_full["r_next"].notna()
    train_clean = train_full.loc[clean_mask]
    train_last = train_full.loc[~clean_mask]
    if train_clean.empty:
        raise ValueError(
            f"No usable (target-known) {model_format}-format rows in train window [{calendar.train_start} .. {calendar.train_end}].")

    needs_base = target.needs_base
    if model_format == "wide":
        # train_clean.index can start later than calendar.train_start (rolling-window warmup has no history
        # before the first bar), so start/end/feature_dates come from the rows that really survived.
        feature_dates_train = train_clean.index
        feature_dates_forecast = forecast_full.index
        id_train = id_train_last = id_forecast = None
    else:
        train_clean = train_clean.reset_index(drop=True)
        train_last = train_last.reset_index(drop=True)
        forecast_full = forecast_full.reset_index(drop=True)
        feature_dates_train = pd.DatetimeIndex(train_clean["Date"].unique()).sort_values()
        feature_dates_forecast = pd.DatetimeIndex(forecast_full["Date"].unique()).sort_values()
        id_train = train_clean[["Date", "ticker"]]
        id_train_last = train_last[["Date", "ticker"]]
        id_forecast = forecast_full[["Date", "ticker"]]

    training_feature = TrainingFeature(
        start=feature_dates_train.min(), end=feature_dates_train.max(), feature_dates=feature_dates_train,
        feature_cols=feature_cols,
        target=target, frequency=calendar.frequency, type=model_format,
        X_in_sample=train_clean[feature_cols],
        X_out_sample_last_index=train_last[feature_cols],
        y_in_sample=_y(train_clean, target),
        DANGEROUS_r_next_in_sample=train_clean["r_next"],
        X_in_sample_id_cols=id_train,
        X_out_sample_last_index_id_cols=id_train_last,
        base_variance=train_clean["ewma_base_variance"] if needs_base else None,
        training_feature_residual_outcome_dates=calendar.next_bars(feature_dates_train),
    )
    forecast_feature = ForecastFeature(
        start=feature_dates_forecast.min(), end=feature_dates_forecast.max(), feature_dates=feature_dates_forecast,
        feature_cols=feature_cols,
        target=target, frequency=calendar.frequency, type=model_format,
        X_forecast_sample=forecast_full[feature_cols],
        y_forecast_sample=_y(forecast_full, target),
        DANGEROUS_r_next_forecast_sample=forecast_full["r_next"],
        id_cols=id_forecast,
        base_variance=forecast_full["ewma_base_variance"] if needs_base else None,
    )
    return training_feature, forecast_feature


def _frame(returns, tickers, spec, target, model_format) -> pd.DataFrame:
    """The features, r_next and (if the target needs it) the EWMA base variance of every row of `returns`."""
    features = spec.compute(returns, tickers)
    ewma_base_variance = target.ewma_base_variance(returns)   # None unless the target needs a base variance
    r_next = returns.shift(-1)

    if model_format == "wide":
        return _wide_frame(returns, features, r_next["portfolio_return"],
                           None if ewma_base_variance is None else ewma_base_variance["portfolio_return"])
    return _long_frame(returns, tickers, features, r_next, ewma_base_variance)


def _wide_frame(returns, features, r_next_portfolio, ewma_base_variance_portfolio) -> pd.DataFrame:
    """Every series' features side by side: '<series>_<feature>' (the raw return keeps the series name)."""
    columns = {}
    for name, (values, long_only) in features.items():
        if long_only:
            continue
        for s in returns.columns:
            columns[s if name == "r" else f"{s}_{name}"] = values[s]
    frame = pd.DataFrame(columns, index=returns.index)
    frame["r_next"] = r_next_portfolio
    if ewma_base_variance_portfolio is not None:
        frame["ewma_base_variance"] = ewma_base_variance_portfolio
    return frame


def _long_frame(returns, tickers, features, r_next, ewma_base_variance) -> pd.DataFrame:
    """One row per (Date, ticker), sorted by Date then ticker: every feature's column of that ticker."""
    parts = []
    for ticker in tickers:
        columns = {name: values[ticker] for name, (values, long_only) in features.items()}
        part = pd.DataFrame(columns, index=returns.index)
        part["r_next"] = r_next[ticker]
        if ewma_base_variance is not None:
            part["ewma_base_variance"] = ewma_base_variance[ticker]
        part.insert(0, "ticker", ticker)
        part.insert(0, "Date", returns.index)
        # a bar without a return (the very first price row) is not a row; the date index would clash with the Date column
        parts.append(part.loc[returns[ticker].notna().to_numpy()].reset_index(drop=True))

    return pd.concat(parts).sort_values(["Date", "ticker"]).reset_index(drop=True)


def _cut(frame, keep_index, feature_cols, target, model_format, label) -> pd.DataFrame:
    """The rows of `frame` on keep_index that have every feature (and the base variance, if the target needs it)."""
    date = frame.index if model_format == "wide" else pd.DatetimeIndex(frame["Date"])
    in_keep = pd.Series(date.isin(keep_index), index=frame.index)
    n_before = int(in_keep.sum())

    subset = feature_cols + (["ewma_base_variance"] if target.needs_base else [])
    frame = frame.loc[in_keep].dropna(subset=subset)
    n_after = len(frame)
    n_dropped = n_before - n_after
    drop_pct = (n_dropped / n_before * 100) if n_before else 0.0
    unit = "bars" if model_format == "wide" else "rows"
    print(f"[INFO] {label}: {n_before} {unit} -> {n_after} after dropna ({n_dropped} dropped, {drop_pct:.1f}%)")
    return frame


def _y(rows: pd.DataFrame, target: Target) -> pd.DataFrame:
    ewma_base_variance = rows["ewma_base_variance"].to_numpy() if "ewma_base_variance" in rows.columns else None
    return pd.DataFrame({"target": target.forward_make_y(rows["r_next"].to_numpy(), ewma_base_variance)}, index=rows.index)


def build_wide_features(
        calendar: GroundTruthCalendar,
        windows: tuple = (5, 10, 21),
        ewma_span: int = 21,
        sq_return_lags: tuple | None = None,
        target_scale: float = 1.0,
        spec: FeatureSpec | None = None,
        target: Target | None = None,
) -> tuple[TrainingFeature, ForecastFeature]:
    """One row per bar, every series' features side by side. Target: the portfolio's next-bar return."""
    spec = spec or FeatureSpec.default(windows, ewma_span, sq_return_lags)
    return build_features(calendar, "wide", spec, target or SquaredReturn(target_scale))


def build_long_features(
        calendar: GroundTruthCalendar,
        windows: tuple = (5, 10, 21),
        ewma_span: int = 21,
        sq_return_lags: tuple | None = None,
        target_scale: float = 1.0,
        spec: FeatureSpec | None = None,
        target: Target | None = None,
) -> tuple[TrainingFeature, ForecastFeature]:
    """One row per (bar, ticker), pooled across tickers. Target: that ticker's next-bar return."""
    spec = spec or FeatureSpec.default(windows, ewma_span, sq_return_lags)
    return build_features(calendar, "long", spec, target or SquaredReturn(target_scale))
