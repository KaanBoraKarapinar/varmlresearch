from data_management import *
import pandas as pd
import numpy as np


class TrainedTreeModel(TrainedModel):

    def predict_variance_single_wide(self, forecast_feature: ForecastFeature, feature_date) -> float:
        feature_date = pd.Timestamp(feature_date)
        X = forecast_feature.X_forecast_sample

        if feature_date not in X.index:
            raise KeyError(f"{feature_date} not in forecast_feature.X_out_sample")

        row = X.loc[[feature_date], self.feature_cols].to_numpy(dtype=np.float32)
        variance = float(DATA_QUALITY_check_variance_floor(self.underlying_model.predict(row), context="TreeModel.predict_variance_single_wide")[0])
        return variance

    def predict_variance_batch_wide(self, forecast_feature: ForecastFeature, feature_dates) -> pd.Series:
        feature_dates = pd.DatetimeIndex(feature_dates)
        X = forecast_feature.X_forecast_sample
        missing = feature_dates.difference(X.index)
        if len(missing):
            raise KeyError(f"feature_dates missing from forecast_feature.X_out_sample: {list(missing)}")

        rows = X.loc[feature_dates, self.feature_cols].to_numpy(dtype=np.float32)
        variances = DATA_QUALITY_check_variance_floor(self.underlying_model.predict(rows), dates=feature_dates, context=f"TreeModel.predict_variance_batch_wide with target_scale = {forecast_feature.target_scale}" )
        return pd.Series(variances, index=feature_dates, name="predicted_variance")


def _next_bars(index: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each date in `dates` (a subset of `index`), the immediately following bar in
    `index` -- position-based, so it correctly skips weekends/holidays instead of naively
    adding a fixed frequency offset."""
    positions = index.get_indexer(dates)
    if (positions == -1).any():
        bad = dates[positions == -1]
        raise KeyError(f"Dates not found in index: {list(bad)}")
    if (positions + 1 >= len(index)).any():
        bad = dates[positions + 1 >= len(index)]
        raise KeyError(f"Dates have no following bar in index: {list(bad)}")
    return index[positions + 1]





def long_format_train_seed(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    training_feature: TrainingFeature,
    window_size: int,
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
) -> tuple[ResidualBank, EwmaVarianceBank]:
    """
    Builds the portfolio-level in-sample ResidualBank + EwmaVarianceBank for a
    pooled long-format model, to be used as residual_bank_in_sample /
    ewma_variance_bank_in_sample.

    Diagonal (per-ticker variance) comes from pooled_model via
    predict_variance_batch_long, off-diagonal (correlation) from EWMA on
    calendar.train_data ONLY (never touches forecast-period returns, so no
    leakage). Portfolio variance = w^T Sigma_hat w, computed by
    EwmaVarianceBank.add()/bulk_add().

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

    tickers = calendar.tickers

    common_dates = pd.DatetimeIndex(training_feature.id_cols["Date"].unique()).sort_values()
    diag_by_ticker = pooled_model.predict_variance_batch_long(training_feature.X_in_sample, training_feature.id_cols, common_dates)
    sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in tickers}, index=common_dates)

    # ---- EWMA correlation, TRAIN-ONLY returns ----
    train_returns = calendar.train_data[tickers].sort_index()
    ewma_bank = EwmaVarianceBank(tickers, calendar.weights, ewma_lam, ewma_min_periods).seed_covariance(
        train_returns.iloc[:ewma_min_periods]
    )

    # only common_dates past the EWMA burn-in window have a covariance to price against
    usable_dates = common_dates[common_dates.isin(train_returns.index[ewma_min_periods:])]
    if usable_dates.empty:
        raise ValueError(f"long_format_train_seed: no dates survived EWMA burn-in (min_periods={ewma_min_periods}).")

    portfolio_variances = ewma_bank.bulk_add(usable_dates, train_returns, sigma_diag_df)

    # X_in_sample never contains the last train date (its target/r_next is NaN by
    # construction), so usable_dates always has a following bar within train_returns.
    outcome_dates = _next_bars(train_returns.index, usable_dates)

    # bulk_load (not bulk_insert_respecting_order) because the bank starts empty here:
    # bulk_insert_respecting_order needs window_size bars of PRIOR history to already be
    # IN the bank before it can compute a rolling mu_hat, which a from-scratch seed can
    # never satisfy for its own first window_size dates. bulk_load ramps the stateful
    # mu_hat up from scratch instead (step_1/step_2), so it works from empty.
    bank = ResidualBank(outcome_dates[0], calendar.frequency)
    bank.bulk_load(
        dates=outcome_dates,
        actual_returns=calendar.train_data.loc[outcome_dates, "portfolio_return"].to_numpy(),
        predicted_variances=portfolio_variances.to_numpy(),
        window_size=window_size,
    )

    # ---- X_out_sample_last_index: one row per ticker, pending prediction for the first forecast bar ----
    last_index_df = training_feature.X_out_sample_last_index
    if not last_index_df.empty:
        if training_feature.id_cols_last_index is None:
            raise ValueError("long_format_train_seed requires training_feature.id_cols_last_index for X_out_sample_last_index.")
        last_id_cols = training_feature.id_cols_last_index
        last_dates = last_id_cols["Date"].unique()
        if len(last_dates) != 1:
            raise ValueError(f"Expected all tickers' last-index rows to share one Date, got: {list(last_dates)}")
        last_feature_date = pd.Timestamp(last_dates[0])

        last_diag_by_ticker = pooled_model.predict_variance_batch_long(last_index_df, last_id_cols, [last_feature_date])
        last_sigma_diag = pd.Series({t: last_diag_by_ticker[t].loc[last_feature_date] for t in tickers})

        merged = calendar.DANGEROUS_merged_warmedup_FORECAST_data
        pending_outcome_date = _next_bars(merged.index, pd.DatetimeIndex([last_feature_date]))[0]
        pending_variance = ewma_bank.add(pending_outcome_date, merged.loc[pending_outcome_date, tickers], last_sigma_diag)

        bank.bulk_load(
            dates=[pending_outcome_date],
            actual_returns=[merged.loc[pending_outcome_date, "portfolio_return"]],
            predicted_variances=[pending_variance],
            window_size=window_size,
        )

    return bank, ewma_bank


def _tree_train_long(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    model_kwargs: dict, window_size: int,
) -> TrainedTreeModel:
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

    trained_model.residual_bank_in_sample, trained_model.ewma_variance_bank_in_sample = long_format_train_seed(
        trained_model, calendar, training_feature, window_size
    )
    return trained_model


def _tree_train_wide(
    training_feature: TrainingFeature, model_constructor, model_name: str, model_kwargs: dict, window_size: int,
) -> TrainedTreeModel:
    feature_cols = training_feature.feature_cols
    actual_next = training_feature.DANGEROUS_r_next_in_sample
    common_index = training_feature.X_in_sample.index.intersection(actual_next.index).intersection(training_feature.y_in_sample.index)
    common_index = common_index.sort_values()  # seed_index = common_index[window_size:] below relies on chronological order

    X = training_feature.X_in_sample.loc[common_index, feature_cols].to_numpy(dtype=np.float32)
    y = training_feature.y_in_sample.loc[common_index, "target"].to_numpy(dtype=np.float32)

    model = model_constructor(**model_kwargs)
    model.fit(X, y)

    predicted_variances = DATA_QUALITY_check_variance_floor(model.predict(X), dates=common_index, context=f"{model_name}_train")

    # bulk_load (not bulk_insert_respecting_order): the bank starts empty, and
    # bulk_insert_respecting_order can never seed a from-scratch bank (see
    # long_format_train_seed for the full explanation). bulk_load ramps the
    # stateful mu_hat up from scratch instead.
    bank = ResidualBank(common_index[0], training_feature.frequency)
    bank.bulk_load(
        dates=common_index,
        actual_returns=actual_next.loc[common_index].to_numpy(),
        predicted_variances=predicted_variances,
        window_size=window_size,
    )

    return TrainedTreeModel(
        training_feature=training_feature, underlying_model=model, model_name=model_name,
        target_scale=training_feature.target_scale, feature_cols=feature_cols, residual_bank_in_sample=bank,
    )



####

def tree_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    window_size: int, model_kwargs: dict | None = None,
) -> TrainedTreeModel:
    model_kwargs = model_kwargs or {}
    if training_feature.type == "long":
        return _tree_train_long(calendar, training_feature, model_constructor, model_name, model_kwargs, window_size)
    return _tree_train_wide(training_feature, model_constructor, model_name, model_kwargs, window_size)


def xgboost_train(calendar: GroundTruthCalendar, training_feature: TrainingFeature, window_size: int, **xgb_kwargs) -> TrainedTreeModel:
    import xgboost as xgb
    defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, n_jobs=1)
    defaults.update(xgb_kwargs)
    return tree_train(calendar, training_feature, xgb.XGBRegressor, "xgboost", window_size, defaults)


def random_forest_train(calendar: GroundTruthCalendar, training_feature: TrainingFeature, window_size: int, **rf_kwargs) -> TrainedTreeModel:
    from sklearn.ensemble import RandomForestRegressor
    defaults = dict(n_estimators=200, max_depth=5, random_state=0)
    defaults.update(rf_kwargs)
    return tree_train(calendar, training_feature, RandomForestRegressor, "random_forest", window_size, defaults)






