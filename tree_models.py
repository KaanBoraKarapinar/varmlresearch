import pandas as pd
import numpy as np

from core import (
    DATA_QUALITY_check_variance_floor,
    EwmaVarianceBank,
    ForecastFeature,
    GroundTruthCalendar,
    ResidualBank,
    TrainedModel,
    TrainingFeature,
)


class TrainedTreeModel(TrainedModel):

    def predict_variance_single_wide(self, forecast_feature: ForecastFeature, feature_date) -> float:
        feature_date = pd.Timestamp(feature_date)
        X = forecast_feature.X_forecast_sample

        if feature_date not in X.index:
            raise KeyError(f"{feature_date} not in forecast_feature.X_out_sample")

        row = X.loc[[feature_date], self.feature_cols].to_numpy(dtype=np.float32)
        # the model predicts in target_scale units; divide back to real variance units
        variance = float(DATA_QUALITY_check_variance_floor(self.underlying_model.predict(row) / self.target_scale, context="TreeModel.predict_variance_single_wide")[0])
        return variance

    def predict_variance_batch_wide(self, forecast_feature: ForecastFeature, feature_dates) -> pd.Series:
        feature_dates = pd.DatetimeIndex(feature_dates)
        X = forecast_feature.X_forecast_sample
        missing = feature_dates.difference(X.index)
        if len(missing):
            raise KeyError(f"feature_dates missing from forecast_feature.X_out_sample: {list(missing)}")

        rows = X.loc[feature_dates, self.feature_cols].to_numpy(dtype=np.float32)
        # the model predicts in target_scale units (y = r_next^2 * target_scale); divide back to real variance units
        variances = DATA_QUALITY_check_variance_floor(self.underlying_model.predict(rows) / self.target_scale, dates=feature_dates, context=f"TreeModel.predict_variance_batch_wide with target_scale = {forecast_feature.target_scale}" )
        return pd.Series(variances, index=feature_dates, name="predicted_variance")




def long_format_train_seed(
    pooled_model: TrainedModel,
    calendar: GroundTruthCalendar,
    training_feature: TrainingFeature,
    residual_bank: ResidualBank,
    ewma_bank: EwmaVarianceBank,
) -> tuple[ResidualBank, EwmaVarianceBank]:
    """
    Builds the portfolio-level in-sample ResidualBank + EwmaVarianceBank for a
    pooled long-format model, to be used as residual_bank_in_sample /
    ewma_variance_bank_in_sample.

    Diagonal (per-ticker variance) comes from pooled_model via
    predict_variance_batch_long_by_ticker, off-diagonal (correlation) from EWMA on
    calendar.train_data ONLY (never touches forecast-period returns, so no
    leakage). Portfolio variance = w^T Sigma_hat w, computed by
    EwmaVarianceBank.add()/general_bulk_insert().

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
        raise ValueError(f"long_format_train_seed requires a long-format TrainingFeature, got type={training_feature.type!r}")
    if training_feature.X_in_sample_id_cols is None:
        raise ValueError("long_format_train_seed requires training_feature.X_in_sample_id_cols (Date/ticker per row).")
    # checked up front so a refused run leaves neither bank half-filled
    if residual_bank.trained or ewma_bank.trained:
        raise ValueError("residual_bank / ewma_bank were already used for training; create new banks for every training run.")

    tickers = calendar.tickers

    # feature_dates is now the feature builder's responsibility (TrainingFeature.feature_dates),
    # not recomputed here from X_in_sample_id_cols.
    feature_dates = training_feature.feature_dates
    diag_by_ticker = pooled_model.predict_variance_batch_long_by_ticker(training_feature.X_in_sample, training_feature.X_in_sample_id_cols, feature_dates)
    sigma_diag_df = pd.DataFrame({t: diag_by_ticker[t].to_numpy() for t in tickers}, index=feature_dates)

    # ---- EWMA correlation, TRAIN-ONLY returns ----
    train_returns = calendar.train_data[tickers].sort_index()
    ewma_min_periods = ewma_bank.ewma_min_periods
    ewma_bank.start_training(calendar).training_safe_bulk_insert(
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
    # same convention calculate_multi_day_standart_residuals_from_long_format_model uses for the forecast phase.
    sigma_diag_outcome = sigma_diag_df.loc[usable_dates].set_axis(outcome_dates)
    portfolio_variances = ewma_bank.general_bulk_insert(outcome_dates, train_returns, sigma_diag_outcome)

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


def _tree_train_long(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    model_kwargs: dict, residual_bank: ResidualBank, ewma_bank: EwmaVarianceBank,
) -> TrainedTreeModel:
    if training_feature.X_in_sample_id_cols is None:
        raise ValueError("_tree_train_long requires training_feature.X_in_sample_id_cols (Date/ticker per row).")

    X = training_feature.X_in_sample.to_numpy(dtype=np.float32)
    y = training_feature.y_in_sample["target"].to_numpy(dtype=np.float32)

    model = model_constructor(**model_kwargs)
    model.fit(X, y)

    # both banks are seeded below, they need the trained model to score the training bars
    trained_model = TrainedTreeModel(
        training_feature=training_feature, underlying_model=model, model_name=model_name,
        target_scale=training_feature.target_scale, feature_cols=training_feature.feature_cols,
    )

    trained_model.residual_bank_in_sample, trained_model.ewma_variance_bank_in_sample = long_format_train_seed(
        trained_model, calendar, training_feature, residual_bank, ewma_bank,
    )
    return trained_model


def _tree_train_wide(
    training_feature: TrainingFeature, model_constructor, model_name: str, model_kwargs: dict, residual_bank: ResidualBank,
) -> TrainedTreeModel:
    feature_cols = training_feature.feature_cols
    actual_next = training_feature.DANGEROUS_r_next_in_sample
    common_index = training_feature.X_in_sample.index.intersection(actual_next.index).intersection(training_feature.y_in_sample.index)
    common_index = common_index.sort_values()  # training_safe_bulk_insert's bootstrap/general split relies on chronological order

    X = training_feature.X_in_sample.loc[common_index, feature_cols].to_numpy(dtype=np.float32)
    y = training_feature.y_in_sample.loc[common_index, "target"].to_numpy(dtype=np.float32)

    model = model_constructor(**model_kwargs)
    model.fit(X, y)

    # the model predicts in target_scale units; divide back to real variance units, same as predict_variance_batch_wide
    predicted_variances = DATA_QUALITY_check_variance_floor(model.predict(X) / training_feature.target_scale, dates=common_index, context=f"{model_name}_train")

    # training_safe_bulk_insert (not general_bulk_insert): the bank starts empty, and
    # general_bulk_insert can never seed a from-scratch bank (see
    # long_format_train_seed for the full explanation). training_safe_bulk_insert ramps the
    # stateful mu_hat up from scratch instead.
    # residual_bank is SINGLE-USE: bound to this run and filled in place, see long_format_train_seed
    residual_bank.start_training(common_index[0], training_feature.frequency)
    residual_bank.training_safe_bulk_insert(
        dates=common_index,
        actual_returns=actual_next.loc[common_index].to_numpy(),
        predicted_variances=predicted_variances,
    )

    return TrainedTreeModel(
        training_feature=training_feature, underlying_model=model, model_name=model_name,
        target_scale=training_feature.target_scale, feature_cols=feature_cols, residual_bank_in_sample=residual_bank,
    )



####

def tree_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    residual_bank: ResidualBank,
    ewma_bank: EwmaVarianceBank | None = None,
    model_kwargs: dict | None = None,
) -> TrainedTreeModel:
    """
    residual_bank / ewma_bank carry every setting of the two banks and are SINGLE-USE: training fills them in
    place and marks them `trained`, handing a trained bank to another training run raises (create new banks
    for every run). ewma_bank is only used by the long path (None -> EwmaVarianceBank(calendar) with its
    defaults); passing one for the wide path raises. An ewma_bank is built from a calendar and training
    checks it is the same calendar's tickers/weights.
    """
    model_kwargs = model_kwargs or {}
    if training_feature.type == "long":
        if ewma_bank is None:
            ewma_bank = EwmaVarianceBank(calendar)
        return _tree_train_long(calendar, training_feature, model_constructor, model_name, model_kwargs, residual_bank, ewma_bank)
    if ewma_bank is not None:
        raise ValueError("ewma_bank is only used by the long-format path.")
    return _tree_train_wide(training_feature, model_constructor, model_name, model_kwargs, residual_bank)


def xgboost_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, residual_bank: ResidualBank,
    ewma_bank: EwmaVarianceBank | None = None,
    **xgb_kwargs,
) -> TrainedTreeModel:
    import xgboost as xgb
    defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, n_jobs=1)
    defaults.update(xgb_kwargs)
    return tree_train(calendar, training_feature, xgb.XGBRegressor, "xgboost", residual_bank, ewma_bank, defaults)


def random_forest_train(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, residual_bank: ResidualBank,
    ewma_bank: EwmaVarianceBank | None = None,
    **rf_kwargs,
) -> TrainedTreeModel:
    from sklearn.ensemble import RandomForestRegressor
    defaults = dict(n_estimators=200, max_depth=5, random_state=0)
    defaults.update(rf_kwargs)
    return tree_train(calendar, training_feature, RandomForestRegressor, "random_forest", residual_bank, ewma_bank, defaults)
