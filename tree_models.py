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

    trained_model.residual_bank_in_sample, trained_model.ewma_variance_bank_in_sample = trained_model.long_format_calculate_training_standartized_residuals(
        calendar, training_feature, residual_bank, ewma_bank,
    )
    return trained_model


def _tree_train_wide(
    calendar: GroundTruthCalendar, training_feature: TrainingFeature, model_constructor, model_name: str,
    model_kwargs: dict, residual_bank: ResidualBank,
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
    # TrainedModel.long_format_calculate_training_standartized_residuals for the full explanation). training_safe_bulk_insert ramps the
    # stateful mu_hat up from scratch instead.
    # residual_bank is SINGLE-USE: bound to this run and filled in place, see TrainedModel.long_format_calculate_training_standartized_residuals
    # own-day rule: the variance forecast made from feature date f is FOR the bar after f, and so is the return in
    # actual_next, so the row is labeled by that outcome bar (same as the long path and the forecast phase)
    outcome_dates = calendar.next_bars(common_index)
    residual_bank.start_training(outcome_dates[0], training_feature.frequency)
    residual_bank.training_safe_bulk_insert(
        dates=outcome_dates,
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
    return _tree_train_wide(calendar, training_feature, model_constructor, model_name, model_kwargs, residual_bank)


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
