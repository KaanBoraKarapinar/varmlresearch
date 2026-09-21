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




def long_format_train_seed(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    training_feature: TrainingFeature,
    rolling_mu_hat_window_size: int,
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
    residual_bank_setting_mu_mode: Literal["rolling", "constant"] = "constant",
    residual_bank_setting_default_mu_hat: float = 0.0,
) -> tuple[ResidualBank, EwmaVarianceBank]:
    """
    Builds the portfolio-level in-sample ResidualBank + EwmaVarianceBank for a
    pooled long-format model, to be used as residual_bank_in_sample /
    ewma_variance_bank_in_sample.

    Diagonal (per-ticker variance) comes from pooled_model via
    predict_variance_batch_long, off-diagonal (correlation) from EWMA on
    calendar.train_data ONLY (never touches forecast-period returns, so no
    leakage). Portfolio variance = w^T Sigma_hat w, computed by
    EwmaVarianceBank.add()/general_bulk_insert().

    Bank entries are labeled by the OUTCOME day (own-day rule): Date=X holds the
    actual_return that REALLY happened on X, and the predicted_variance that was
    produced from X's own preceding feature bar.

    Also seeds a pending prediction for training_feature.X_out_sample_last_index
    (one row per ticker, the window's final feature row with no known target yet),
    whose outcome is the first bar the forecast phase will price.

    rolling_mu_hat_window_size only ever feeds ResidualBank's mu_hat seeding here --
    this function never touches multi_day_var's own (separate) quantile window.
    """
    if training_feature.type != "long":
        raise ValueError(f"long_format_train_seed requires a long-format TrainingFeature, got type={training_feature.type!r}")
    if training_feature.X_in_sample_id_cols is None:
        raise ValueError("long_format_train_seed requires training_feature.X_in_sample_id_cols (Date/ticker per row).")

    tickers = calendar.tickers

    # feature_dates is now the feature builder's responsibility (TrainingFeature.feature_dates),
    # not recomputed here from X_in_sample_id_cols.
    feature_dates = training_feature.feature_dates
    diag_by_ticker = pooled_model.predict_variance_batch_long(training_feature.X_in_sample, training_feature.X_in_sample_id_cols, feature_dates)
    sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in tickers}, index=feature_dates)

    # ---- EWMA correlation, TRAIN-ONLY returns ----
    train_returns = calendar.train_data[tickers].sort_index()
    ewma_bank = EwmaVarianceBank(tickers, calendar.weights, ewma_lam, ewma_min_periods).training_safe_bulk_insert(
        train_returns.iloc[:ewma_min_periods]
    )
    # usable_dates: feature_dates that are also past the calendar's own EWMA burn-in
    # window (a covariance to price against). Threshold is ewma_min_periods - 1, not
    # ewma_min_periods: the seed window's own last bar (position ewma_min_periods - 1)
    # is a valid feature address too -- the seed cov it produces already encodes that
    # bar (see EwmaVarianceBank.training_safe_bulk_insert), so it needs no extra prior
    # history beyond the seed itself. X_in_sample never contains the last
    # train date (its target/r_next is NaN by construction), so usable_dates always
    # has a following bar -- calendar.next_bars gives outcome_dates, their paired
    # own-day OUTCOME bar.
    usable_dates = calendar.usable_train_dates(feature_dates, ewma_min_periods - 1)
    outcome_dates = calendar.next_bars(usable_dates)

    # own-day rule: sigma_diag_df is forecast made FROM usable_dates[i]'s features, which is
    # FOR outcome_dates[i] -- re-key it to outcome_dates so general_bulk_insert's EWMA
    # recursion (keyed/stepped by outcome day) lines up with the diagonal it's paired with,
    # same convention multi_day_var_from_long_format_model uses for the forecast phase.
    sigma_diag_outcome = sigma_diag_df.loc[usable_dates].set_axis(outcome_dates)
    portfolio_variances = ewma_bank.general_bulk_insert(outcome_dates, train_returns, sigma_diag_outcome)

    # training_safe_bulk_insert (not general_bulk_insert) because the bank starts empty here:
    # general_bulk_insert needs rolling_mu_hat_window_size bars of PRIOR history to already be
    # IN the bank before it can compute a rolling mu_hat, which a from-scratch seed can
    # never satisfy for its own first rolling_mu_hat_window_size dates. training_safe_bulk_insert
    # ramps the stateful mu_hat up from scratch instead (step_1/step_2), so it works from empty.
    bank = ResidualBank(
        outcome_dates[0], calendar.frequency,
        residual_bank_setting_default_mu_hat, residual_bank_setting_mu_mode, rolling_mu_hat_window_size,
    )
    bank.training_safe_bulk_insert(
        dates=outcome_dates,
        actual_returns=calendar.train_data.loc[outcome_dates, "portfolio_return"].to_numpy(),
        predicted_variances=portfolio_variances.to_numpy(),
    )

    # ---- X_out_sample_last_index: one row per ticker, pending prediction for the first forecast bar ----
    last_index_df = training_feature.X_out_sample_last_index
    if not last_index_df.empty:
        if training_feature.X_out_sample_last_index_id_cols is None:
            raise ValueError("long_format_train_seed requires training_feature.X_out_sample_last_index_id_cols for X_out_sample_last_index.")
        last_id_cols = training_feature.X_out_sample_last_index_id_cols
        last_dates = last_id_cols["Date"].unique()
        if len(last_dates) != 1:
            raise ValueError(f"Expected all tickers' last-index rows to share one Date, got: {list(last_dates)}")
        last_feature_date = pd.Timestamp(last_dates[0])

        last_diag_by_ticker = pooled_model.predict_variance_batch_long(last_index_df, last_id_cols, [last_feature_date])
        last_sigma_diag = pd.Series({t: last_diag_by_ticker[t].loc[last_feature_date] for t in tickers})

        merged = calendar.DANGEROUS_merged_warmedup_FORECAST_data
        pending_outcome_date = calendar.next_bars([last_feature_date])[0]
        pending_variance = ewma_bank.add(pending_outcome_date, merged.loc[pending_outcome_date, tickers], last_sigma_diag)

        bank.training_safe_bulk_insert(
            dates=[pending_outcome_date],
            actual_returns=[merged.loc[pending_outcome_date, "portfolio_return"]],
            predicted_variances=[pending_variance],
        )

    return bank, ewma_bank


def _tree_train_long(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    model_kwargs: dict, rolling_mu_hat_window_size: int,
    residual_bank_setting_mu_mode: Literal["rolling", "constant"] = "constant",
    residual_bank_setting_default_mu_hat: float = 0.0,
) -> TrainedTreeModel:
    if training_feature.X_in_sample_id_cols is None:
        raise ValueError("_tree_train_long requires training_feature.X_in_sample_id_cols (Date/ticker per row).")

    X = training_feature.X_in_sample.to_numpy(dtype=np.float32)
    y = training_feature.y_in_sample["target"].to_numpy(dtype=np.float32)

    model = model_constructor(**model_kwargs)
    model.fit(X, y)

    trained_model = TrainedTreeModel(
        training_feature=training_feature, underlying_model=model, model_name=model_name,
        target_scale=training_feature.target_scale, feature_cols=training_feature.feature_cols,
        residual_bank_in_sample=ResidualBank(
            training_feature.start, training_feature.frequency,
            residual_bank_setting_default_mu_hat, residual_bank_setting_mu_mode,
        ),
    )

    trained_model.residual_bank_in_sample, trained_model.ewma_variance_bank_in_sample = long_format_train_seed(
        trained_model, calendar, training_feature, rolling_mu_hat_window_size,
        residual_bank_setting_mu_mode=residual_bank_setting_mu_mode,
        residual_bank_setting_default_mu_hat=residual_bank_setting_default_mu_hat,
    )
    return trained_model


def _tree_train_wide(
    training_feature: TrainingFeature, model_constructor, model_name: str, model_kwargs: dict, rolling_mu_hat_window_size: int,
    residual_bank_setting_mu_mode: Literal["rolling", "constant"] = "constant",
    residual_bank_setting_default_mu_hat: float = 0.0,
) -> TrainedTreeModel:
    feature_cols = training_feature.feature_cols
    actual_next = training_feature.DANGEROUS_r_next_in_sample
    common_index = training_feature.X_in_sample.index.intersection(actual_next.index).intersection(training_feature.y_in_sample.index)
    common_index = common_index.sort_values()  # training_safe_bulk_insert's bootstrap/general split relies on chronological order

    X = training_feature.X_in_sample.loc[common_index, feature_cols].to_numpy(dtype=np.float32)
    y = training_feature.y_in_sample.loc[common_index, "target"].to_numpy(dtype=np.float32)

    model = model_constructor(**model_kwargs)
    model.fit(X, y)

    predicted_variances = DATA_QUALITY_check_variance_floor(model.predict(X), dates=common_index, context=f"{model_name}_train")

    # training_safe_bulk_insert (not general_bulk_insert): the bank starts empty, and
    # general_bulk_insert can never seed a from-scratch bank (see
    # long_format_train_seed for the full explanation). training_safe_bulk_insert ramps the
    # stateful mu_hat up from scratch instead.
    bank = ResidualBank(
        common_index[0], training_feature.frequency,
        residual_bank_setting_default_mu_hat, residual_bank_setting_mu_mode,
    )
    bank.training_safe_bulk_insert(
        dates=common_index,
        actual_returns=actual_next.loc[common_index].to_numpy(),
        predicted_variances=predicted_variances,
        rolling_mu_hat_window_size=rolling_mu_hat_window_size,
    )

    return TrainedTreeModel(
        training_feature=training_feature, underlying_model=model, model_name=model_name,
        target_scale=training_feature.target_scale, feature_cols=feature_cols, residual_bank_in_sample=bank,
    )



####

def tree_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    rolling_mu_hat_window_size: int,
    residual_bank_setting_mu_mode: Literal["rolling", "constant"] = "constant",
    residual_bank_setting_default_mu_hat: float = 0.0,
    model_kwargs: dict | None = None,
) -> TrainedTreeModel:
    model_kwargs = model_kwargs or {}
    if training_feature.type == "long":
        return _tree_train_long(
            calendar, training_feature, model_constructor, model_name, model_kwargs, rolling_mu_hat_window_size,
            residual_bank_setting_mu_mode, residual_bank_setting_default_mu_hat,
        )
    return _tree_train_wide(
        training_feature, model_constructor, model_name, model_kwargs, rolling_mu_hat_window_size,
        residual_bank_setting_mu_mode, residual_bank_setting_default_mu_hat,
    )


def xgboost_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, rolling_mu_hat_window_size: int,
    residual_bank_setting_mu_mode: Literal["rolling", "constant"] = "constant",
    residual_bank_setting_default_mu_hat: float = 0.0,
    **xgb_kwargs,
) -> TrainedTreeModel:
    import xgboost as xgb
    defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, n_jobs=1)
    defaults.update(xgb_kwargs)
    return tree_train(
        calendar, training_feature, xgb.XGBRegressor, "xgboost", rolling_mu_hat_window_size,
        residual_bank_setting_mu_mode, residual_bank_setting_default_mu_hat, defaults,
    )


def random_forest_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, rolling_mu_hat_window_size: int,
    residual_bank_setting_mu_mode: Literal["rolling", "constant"] = "constant",
    residual_bank_setting_default_mu_hat: float = 0.0,
    **rf_kwargs,
) -> TrainedTreeModel:
    from sklearn.ensemble import RandomForestRegressor
    defaults = dict(n_estimators=200, max_depth=5, random_state=0)
    defaults.update(rf_kwargs)
    return tree_train(
        calendar, training_feature, RandomForestRegressor, "random_forest", rolling_mu_hat_window_size,
        residual_bank_setting_mu_mode, residual_bank_setting_default_mu_hat, defaults,
    )