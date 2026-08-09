from data_management import *
import pandas as pd
import numpy as np


class TrainedTreeModel(TrainedModel):

    def predict_variance(self, forecast_feature: ForecastFeature, feature_date) -> float:
        feature_date = pd.Timestamp(feature_date)
        X = forecast_feature.X_forecast_sample

        if feature_date not in X.index:
            raise KeyError(f"{feature_date} not in forecast_feature.X_out_sample")

        row = X.loc[[feature_date], self.feature_cols].to_numpy(dtype=np.float32)
        variance = float(DATA_QUALITY_check_variance_floor(self.underlying_model.predict(row), context="TreeModel.predict_variance")[0])
        return variance

    def predict_variance_batch(self, forecast_feature: ForecastFeature, feature_dates) -> pd.Series:
        feature_dates = pd.DatetimeIndex(feature_dates)
        X = forecast_feature.X_forecast_sample
        missing = feature_dates.difference(X.index)
        if len(missing):
            raise KeyError(f"feature_dates missing from forecast_feature.X_out_sample: {list(missing)}")

        rows = X.loc[feature_dates, self.feature_cols].to_numpy(dtype=np.float32)
        variances = DATA_QUALITY_check_variance_floor(self.underlying_model.predict(rows), dates=feature_dates, context=f"TreeModel.predict_variance_batch with target_scale = {forecast_feature.target_scale}" )
        return pd.Series(variances, index=feature_dates, name="predicted_variance")





def long_format_train_seed(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    training_feature: TrainingFeature,
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
) -> ResidualBank:
    """
    Builds the portfolio-level in-sample ResidualBank for a pooled long-format model,
    to be passed as initial_residual_bank / used as residual_bank_in_sample.

    Diagonal (per-ticker variance) comes from pooled_model, off-diagonal (correlation)
    from EWMA on calendar.train_data ONLY (never touches forecast-period returns, so
    no leakage). Portfolio variance = w^T Sigma w.

    Bank entries are labeled by the OUTCOME day (own-day rule): Date=X holds the
    actual_return that REALLY happened on X, and the predicted_variance that was
    produced from X's own preceding feature bar.

    Also seeds a pending prediction for training_feature.X_out_sample_last_index
    (one row per ticker, the window's final feature row with no known target yet),
    whose outcome is the first bar the forecast phase will price.
    """
    if training_feature.type != "long":
        raise ValueError(f"long_format_train_seed requires a long-format TrainingFeature, got type={training_feature.type!r}")
    if training_feature.id_cols is None:
        raise ValueError("long_format_train_seed requires training_feature.id_cols (Date/ticker per row).")

    feature_cols = pooled_model.feature_cols
    tickers = calendar.tickers
    weights = calendar.weights.loc[tickers].to_numpy()

    # ---- per-ticker "forecast-like" views of the TRAINING X, so pooled_model.predict_variance_batch can be reused ----
    per_ticker_train_view: dict[str, ForecastFeature] = {}
    for t in tickers:
        row_mask = training_feature.id_cols["ticker"] == t
        ticker_X = training_feature.X_in_sample.loc[row_mask, feature_cols]
        ticker_dates = training_feature.id_cols.loc[row_mask, "Date"].to_numpy()
        ticker_X = ticker_X.set_axis(ticker_dates, axis=0)
        per_ticker_train_view[t] = ForecastFeature(
            start=training_feature.start, end=training_feature.end,
            target_scale=training_feature.target_scale, frequency=training_feature.frequency,
            feature_cols=feature_cols, type="wide",
            X_forecast_sample=ticker_X, y_forecast_sample=pd.DataFrame(),
            DANGEROUS_r_next_forecast_sample=pd.Series(dtype=float), id_cols=None,
        )

    common_dates = per_ticker_train_view[tickers[0]].X_forecast_sample.index
    for t in tickers[1:]:
        common_dates = common_dates.intersection(per_ticker_train_view[t].X_forecast_sample.index)
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    diag_by_ticker = {
        t: pooled_model.predict_variance_batch(per_ticker_train_view[t], common_dates)
        for t in tickers
    }

    # ---- EWMA correlation, TRAIN-ONLY returns ----
    train_returns = calendar.train_data[tickers].sort_index()
    R = train_returns.to_numpy(dtype=float)
    dates_idx = train_returns.index

    sigma_ewma = np.atleast_2d(np.cov(R[:ewma_min_periods], rowvar=False))
    cov_by_date: dict[pd.Timestamp, np.ndarray] = {}
    for i in range(ewma_min_periods, len(R)):
        r_prev = R[i - 1]
        sigma_ewma = ewma_lam * sigma_ewma + (1 - ewma_lam) * np.outer(r_prev, r_prev)
        cov_by_date[dates_idx[i]] = sigma_ewma.copy()

    def _portfolio_variance(sigma_diag: np.ndarray, cov: np.ndarray) -> float:
        d_ = np.sqrt(np.diag(cov))
        d_[d_ == 0] = np.nan
        corr = cov / np.outer(d_, d_)
        np.fill_diagonal(corr, 1.0)
        sigma_hat = corr * np.sqrt(np.outer(sigma_diag, sigma_diag))
        np.fill_diagonal(sigma_hat, sigma_diag)
        return float(weights @ sigma_hat @ weights)

    portfolio_variances, feature_dates_used = [], []
    for date in common_dates:
        if date not in cov_by_date:
            continue
        sigma_diag = np.array([diag_by_ticker[t].loc[date] for t in tickers])
        portfolio_variances.append(_portfolio_variance(sigma_diag, cov_by_date[date]))
        feature_dates_used.append(date)

    if not feature_dates_used:
        raise ValueError(f"long_format_train_seed: no dates survived EWMA burn-in (min_periods={ewma_min_periods}).")

    feature_dates_used = pd.DatetimeIndex(feature_dates_used)
    portfolio_variances = DATA_QUALITY_check_variance_floor(np.array(portfolio_variances), dates=feature_dates_used, context="long_format_train_seed")

    outcome_dates = _next_bars(dates_idx, feature_dates_used)
    actual_returns = np.array([calendar.realized_return(d) for d in outcome_dates])

    bank = ResidualBank(outcome_dates[0], calendar.frequency)
    bank.bulk_load(dates=outcome_dates, actual_returns=actual_returns, predicted_variances=portfolio_variances, mu_hat=float(actual_returns.mean()))
    # bulk_load already refreshed bank._current_single_stateful_mu from actual_return.mean()

    # ---- X_out_sample_last_index: one row per ticker, pending prediction for the first forecast bar ----
    last_index_df = training_feature.X_out_sample_last_index
    if not last_index_df.empty:
        last_id_cols = training_feature.id_cols.reindex(last_index_df.index)
        last_dates = last_id_cols["Date"].unique()
        if len(last_dates) != 1:
            raise ValueError(f"Expected all tickers' last-index rows to share one Date, got: {list(last_dates)}")
        last_feature_date = pd.Timestamp(last_dates[0])

        sigma_diag = []
        for t in tickers:
            row_mask = last_id_cols["ticker"] == t
            if not row_mask.any():
                raise ValueError(f"Ticker {t} has no row in X_out_sample_last_index for {last_feature_date}.")
            row_X = last_index_df.loc[row_mask, feature_cols].to_numpy(dtype=np.float32)
            sigma_diag.append(float(pooled_model.underlying_model.predict(row_X)[0]))
        sigma_diag = np.array(sigma_diag)

        wide_index = calendar.DANGEROUS_merged_warmedup_FORECAST_data.index
        merged_returns = calendar.DANGEROUS_merged_warmedup_FORECAST_data[tickers]
        R_full = merged_returns.to_numpy(dtype=float)
        full_dates = merged_returns.index
        pos = full_dates.get_loc(last_feature_date)

        sigma_full = np.atleast_2d(np.cov(R_full[:ewma_min_periods], rowvar=False))
        for i in range(ewma_min_periods, pos + 1):
            r_prev = R_full[i - 1]
            sigma_full = ewma_lam * sigma_full + (1 - ewma_lam) * np.outer(r_prev, r_prev)

        last_index_variance = _portfolio_variance(sigma_diag, sigma_full)
        last_index_variance = float(DATA_QUALITY_check_variance_floor(
            last_index_variance, dates=[last_feature_date], context="long_format_train_seed_last_index"
        )[0])

        pending_outcome_date = _next_bars(wide_index, pd.DatetimeIndex([last_feature_date]))[0]
        bank.bulk_load(
            dates=[pending_outcome_date],
            actual_returns=[calendar.realized_return(pending_outcome_date)],
            predicted_variances=[last_index_variance],
            mu_hat=bank.current_mu_hat,
        )

    return bank


def _tree_train_long(calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str, model_kwargs: dict) -> TrainedTreeModel:
    if training_feature.id_cols is None:
        raise ValueError("_tree_train_long requires training_feature.id_cols (Date/ticker per row).")

    X = training_feature.X_in_sample.to_numpy(dtype=np.float32)
    y = training_feature.y_in_sample["target"].to_numpy(dtype=np.float32)

    model = model_constructor(**model_kwargs)
    model.fit(X, y)

    trained_model = TrainedTreeModel(
        training_feature=training_feature, underlying_model=model, model_name=model_name,
        target_scale=training_feature.target_scale, feature_cols=training_feature.feature_cols,
        residual_bank_in_sample=ResidualBank(training_feature.start, training_feature.frequency),
    )

    trained_model.residual_bank_in_sample = long_format_train_seed(trained_model, calendar, training_feature)
    return trained_model





####

def tree_train(calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str, model_kwargs: dict | None = None) -> TrainedTreeModel:
    model_kwargs = model_kwargs or {}
    if training_feature.type == "long":
        return _tree_train_long(calendar, training_feature, model_constructor, model_name, model_kwargs)
    return _tree_train_wide(calendar, training_feature, model_constructor, model_name, model_kwargs)


def xgboost_train(calendar: GroundTruthCalendar, training_feature: TrainingFeature, **xgb_kwargs) -> TrainedTreeModel:
    import xgboost as xgb
    defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, n_jobs=1)
    defaults.update(xgb_kwargs)
    return tree_train(calendar, training_feature, xgb.XGBRegressor, "xgboost", defaults)


def random_forest_train(calendar: GroundTruthCalendar, training_feature: TrainingFeature, **rf_kwargs) -> TrainedTreeModel:
    from sklearn.ensemble import RandomForestRegressor
    defaults = dict(n_estimators=200, max_depth=5, random_state=0)
    defaults.update(rf_kwargs)
    return tree_train(calendar, training_feature, RandomForestRegressor, "random_forest", defaults)






