from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from core import TrainingFeature, ForecastFeature, GroundTruthCalendar


# %%



def _rolling_feature_block(returns_df: pd.DataFrame, windows: tuple, ewma_span: int,
                           sq_return_lags: tuple | None) -> pd.DataFrame:
    abs_r = returns_df.abs()
    df = returns_df.copy()
    df = df.join(abs_r.add_suffix("_abs_r"))

    for w in windows:
        df = df.join(returns_df.rolling(w).std().add_suffix(f"_real_vol_{w}"))
        df = df.join(abs_r.rolling(w).mean().add_suffix(f"_mean_abs_{w}"))
        df = df.join(returns_df.rolling(w).skew().add_suffix(f"_skew_{w}"))

    df = df.join(returns_df.ewm(span=ewma_span).std().add_suffix("_ewma_vol"))

    if sq_return_lags:
        r_sq = returns_df ** 2
        for lag in sq_return_lags:
            df = df.join(r_sq.shift(lag).add_suffix(f"_r2_lag_{lag}"))

    return df


def build_wide_features(
        calendar: GroundTruthCalendar,
        windows: tuple = (5, 10, 21),
        ewma_span: int = 21,
        sq_return_lags: tuple | None = None,
        target_scale: float = 1.0,
) -> tuple[TrainingFeature, ForecastFeature]:
    def _returns(df: pd.DataFrame) -> pd.DataFrame:
        r = df[calendar.tickers].copy()
        r["portfolio_return"] = df["portfolio_return"]
        return r

    train_returns = _returns(calendar.train_data)
    merged_returns = _returns(calendar.DANGEROUS_merged_warmedup_FORECAST_data)

    def _build_common(returns_df: pd.DataFrame, keep_index: pd.Index, label: str) -> tuple[pd.DataFrame, list[str]]:
        df = _rolling_feature_block(returns_df, windows, ewma_span, sq_return_lags)
        feature_cols = list(df.columns)

        df = df.copy()
        df["target"] = returns_df["portfolio_return"].shift(-1) ** 2

        n_before = len(df.loc[df.index.isin(keep_index)])
        df = df.dropna(subset=feature_cols)
        df = df.loc[df.index.isin(keep_index)]
        n_after = len(df)
        n_dropped = n_before - n_after
        drop_pct = (n_dropped / n_before * 100) if n_before else 0.0
        print(f"[INFO] {label}: {n_before} bars -> {n_after} after dropna ({n_dropped} dropped, {drop_pct:.1f}%)")

        return df, feature_cols

    train_full, feature_cols = _build_common(train_returns, calendar.train_data.index, "wide train")
    r_next_train = train_returns["portfolio_return"].shift(-1)

    clean_mask = train_full["target"].notna()
    train_clean = train_full.loc[clean_mask]
    train_last = train_full.loc[~clean_mask]

    if train_clean.empty:
        raise ValueError(
            f"No usable (target-known) rows in train window [{calendar.train_start} .. {calendar.train_end}].")

    # calendar.forecast_feature_dates = last train bar + forecast dates (the last train bar's feature row prices forecast_dates[0])
    forecast_df, _ = _build_common(merged_returns, calendar.forecast_feature_dates, "wide forecast")
    if forecast_df.empty:
        raise ValueError(f"No usable rows in forecast window [{calendar.forecast_start} .. {calendar.forecast_end}].")
    r_next_forecast = merged_returns["portfolio_return"].shift(-1)

    training_feature = TrainingFeature(
        # train_clean.index can start later than calendar.train_start -- rolling-window
        # warmup (e.g. a 21-bar std) has no history before train_start to draw from, so
        # dropna(subset=feature_cols) in _build_common already cuts those bars. Deriving
        # start/end/feature_dates from the actual surviving index (instead of trusting
        # calendar.train_start/train_end) keeps them truthful about what X_in_sample covers.
        start=train_clean.index.min(), end=train_clean.index.max(), feature_dates=train_clean.index,
        feature_cols=feature_cols,
        target_scale=target_scale, frequency=calendar.frequency, type="wide",
        X_in_sample=train_clean[feature_cols],
        X_out_sample_last_index=train_last[feature_cols],
        y_in_sample=train_clean[["target"]] * target_scale,
        DANGEROUS_r_next_in_sample=r_next_train.loc[train_clean.index],
        X_in_sample_id_cols=None,
        X_out_sample_last_index_id_cols=None,
    )
    forecast_feature = ForecastFeature(
        # forecast_df is built off DANGEROUS_merged_warmedup_FORECAST_data, so unlike the
        # train side it has train-period history to draw rolling windows from and should
        # already span exactly calendar.forecast_data.index -- still derived from the
        # actual index rather than assumed, for the same reason as training_feature above.
        start=forecast_df.index.min(), end=forecast_df.index.max(), feature_dates=forecast_df.index,
        feature_cols=feature_cols,
        target_scale=target_scale, frequency=calendar.frequency, type="wide",
        X_forecast_sample=forecast_df[feature_cols],
        y_forecast_sample=forecast_df[["target"]] * target_scale,
        DANGEROUS_r_next_forecast_sample=r_next_forecast.loc[forecast_df.index],
        id_cols=None,
    )
    return training_feature, forecast_feature


def build_long_features(
        calendar: GroundTruthCalendar,
        windows: tuple = (5, 10, 21),
        ewma_span: int = 21,
        sq_return_lags: tuple | None = None,
        target_scale: float = 1.0,
) -> tuple[TrainingFeature, ForecastFeature]:
    """One row per (bar, ticker), pooled across tickers. Target = that ticker's next-bar squared return."""

    def _build_long(df: pd.DataFrame) -> pd.DataFrame:
        long = df[calendar.tickers].stack().rename("r").reset_index()
        long.columns = ["Date", "ticker", "r"]
        long = long.sort_values(["Date", "ticker"]).reset_index(drop=True)

        g = long.groupby("ticker")["r"]
        long["abs_r"] = long["r"].abs()

        for w in windows:
            long[f"real_vol_{w}"] = g.transform(lambda x: x.rolling(w).std())
            long[f"mean_abs_{w}"] = long.groupby("ticker")["abs_r"].transform(lambda x: x.rolling(w).mean())
            long[f"skew_{w}"] = g.transform(lambda x: x.rolling(w).skew())

        long["ewma_vol"] = g.transform(lambda x: x.ewm(span=ewma_span).std())

        if sq_return_lags:
            long["r_sq"] = long["r"] ** 2
            for lag in sq_return_lags:
                long[f"r2_lag_{lag}"] = long.groupby("ticker")["r_sq"].transform(lambda x: x.shift(lag))
            long = long.drop(columns="r_sq")

        long["r_next"] = g.transform(lambda x: x.shift(-1))
        long["target"] = long["r_next"] ** 2

        market_vol = df["portfolio_return"].ewm(span=ewma_span).std().rename("market_ewma_vol")
        long = long.merge(market_vol, left_on="Date", right_index=True)

        return long

    train_long = _build_long(calendar.train_data)
    merged_long = _build_long(calendar.DANGEROUS_merged_warmedup_FORECAST_data)

    feature_cols = [c for c in train_long.columns if c not in ("Date", "ticker", "r_next", "target")]

    def _finalize(long_df: pd.DataFrame, keep_index: pd.Index, label: str) -> pd.DataFrame:
        scoped = long_df.loc[long_df["Date"].isin(keep_index)]
        n_before = len(scoped)

        long_df = long_df.dropna(subset=feature_cols)
        result = long_df.loc[long_df["Date"].isin(keep_index)].reset_index(drop=True)
        n_after = len(result)
        n_dropped = n_before - n_after
        drop_pct = (n_dropped / n_before * 100) if n_before else 0.0
        print(f"[INFO] {label}: {n_before} rows -> {n_after} after dropna ({n_dropped} dropped, {drop_pct:.1f}%)")

        return result

    train_full = _finalize(train_long, calendar.train_data.index, "long train")
    # calendar.forecast_feature_dates = last train bar + forecast dates (the last train bar's feature rows price forecast_dates[0])
    forecast_full = _finalize(merged_long, calendar.forecast_feature_dates, "long forecast")

    if train_full.empty:
        raise ValueError(
            f"No usable long-format rows in train window [{calendar.train_start} .. {calendar.train_end}].")
    if forecast_full.empty:
        raise ValueError(
            f"No usable long-format rows in forecast window [{calendar.forecast_start} .. {calendar.forecast_end}].")

    id_cols_list = ["Date", "ticker"]

    clean_mask = train_full["target"].notna()
    train_clean = train_full.loc[clean_mask].reset_index(drop=True)
    train_last = train_full.loc[~clean_mask].reset_index(drop=True)

    if train_clean.empty:
        raise ValueError(
            f"No usable (target-known) long-format rows in train window [{calendar.train_start} .. {calendar.train_end}].")

    # feature_dates: distinct dates X_in_sample/X_forecast_sample actually cover (one row
    # per (Date, ticker) in long format, so this collapses back to one entry per date).
    # Same burn-in-honesty reasoning as build_wide_features -- train_clean's dates can
    # start later than calendar.train_start because of rolling-window warmup.
    train_feature_dates = pd.DatetimeIndex(train_clean["Date"].unique()).sort_values()
    forecast_feature_dates = pd.DatetimeIndex(forecast_full["Date"].unique()).sort_values()

    training_feature = TrainingFeature(
        start=train_feature_dates.min(), end=train_feature_dates.max(), feature_dates=train_feature_dates,
        feature_cols=feature_cols,
        target_scale=target_scale, frequency=calendar.frequency, type="long",
        X_in_sample=train_clean[feature_cols],
        X_out_sample_last_index=train_last[feature_cols],
        y_in_sample=train_clean[["target"]].assign(target=train_clean["target"] * target_scale),
        DANGEROUS_r_next_in_sample=train_clean["r_next"],
        X_in_sample_id_cols=train_clean[id_cols_list],
        X_out_sample_last_index_id_cols=train_last[id_cols_list],
    )
    forecast_feature = ForecastFeature(
        start=forecast_feature_dates.min(), end=forecast_feature_dates.max(), feature_dates=forecast_feature_dates,
        feature_cols=feature_cols,
        target_scale=target_scale, frequency=calendar.frequency, type="long",
        X_forecast_sample=forecast_full[feature_cols],
        y_forecast_sample=forecast_full[["target"]].assign(target=forecast_full["target"] * target_scale),
        DANGEROUS_r_next_forecast_sample=forecast_full["r_next"],
        id_cols=forecast_full[id_cols_list],
    )
    return training_feature, forecast_feature
# %%