from data_management import *

# %%
def _next_bars(index: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each date in `dates` (a subset of `index`), the immediately following bar in `index`."""
    positions = index.get_indexer(dates)
    if (positions == -1).any():
        bad = dates[positions == -1]
        raise KeyError(f"Dates not found in index: {list(bad)}")
    if (positions + 1 >= len(index)).any():
        bad = dates[positions + 1 >= len(index)]
        raise KeyError(f"Dates have no following bar in index: {list(bad)}")
    return index[positions + 1]



# %%
class TrainedGarchModel(TrainedModel):

    def predict_variance(self, feature_date) -> float:
        feature_date = pd.Timestamp(feature_date)
        fitted_index = self.underlying_model.conditional_volatility.index

        if feature_date not in fitted_index:
            raise KeyError(f"{feature_date} not in fitted GARCH index")

        horizon = fitted_index.get_loc(feature_date) - len(fitted_index) + 2  # bars after last fit point
        garch_fc = self.underlying_model.forecast(horizon=horizon, reindex=False)
        variance = float(garch_fc.variance.values[-1, -1]) / 10_000.0

        return float(DATA_QUALITY_check_variance_floor(variance, dates=[feature_date],
                                                       context="TrainedGarchModel.predict_variance")[0])

    def predict_variance_batch(self, forecast_feature, feature_dates) -> pd.Series:
        def predict_variance_batch(self, feature_dates) -> pd.Series:
            feature_dates = pd.DatetimeIndex(feature_dates)
            garch_fc = self.underlying_model.forecast(horizon=len(feature_dates), reindex=False)
            variances = garch_fc.variance.values[-1, :] / 10_000.0

            result = pd.Series(variances, index=feature_dates, name="predicted_variance")
            return pd.Series(
                DATA_QUALITY_check_variance_floor(result.to_numpy(), dates=feature_dates,
                                                  context="TrainedGarchModel.predict_variance_batch"),
                index=feature_dates,
                name="predicted_variance",
            )



class TrainedTreeModel(TrainedModel):
    """Shared by xgboost and random_forest, both are sklearn-style .predict(X)."""

    def predict_variance(self, forecast_feature, feature_date) -> float:
        feature_date = pd.Timestamp(feature_date)
        X = forecast_feature.X_forecast_sample

        if feature_date not in X.index:
            raise KeyError(f"{feature_date} not in forecast_feature.X_out_sample")

        row = X.loc[[feature_date], self.feature_cols].to_numpy(dtype=np.float32)
        variance = float(
            DATA_QUALITY_check_variance_floor(self.underlying_model.predict(row),
                                              context="TreeModel.predict_variance")[0])
        return variance

    def predict_variance_batch(self, forecast_feature, feature_dates) -> pd.Series:
        """All feature_dates scored in ONE .predict() call, no loop."""
        feature_dates = pd.DatetimeIndex(feature_dates)
        X = forecast_feature.X_forecast_sample
        missing = feature_dates.difference(X.index)
        if len(missing):
            raise KeyError(f"feature_dates missing from forecast_feature.X_out_sample: {list(missing)}")

        rows = X.loc[feature_dates, self.feature_cols].to_numpy(dtype=np.float32)
        variances = DATA_QUALITY_check_variance_floor(self.underlying_model.predict(rows), dates=feature_dates,
                                                      context="TreeModel.predict_variance_batch")
        return pd.Series(variances, index=feature_dates, name="predicted_variance")


class TrainedLstmModel(TrainedModel):
    def __init__(
        self,
        training_feature: TrainingFeature,
        underlying_model: object,
        model_name: str,
        target_scale: float,
        feature_cols: list[str],
        residual_bank_in_sample: ResidualBank,
        sequence_length: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        super().__init__(training_feature, underlying_model, model_name, target_scale, feature_cols, residual_bank_in_sample)
        self.sequence_length = sequence_length
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def _build_sequence_batch(self, forecast_feature, feature_dates):
        import torch

        X = forecast_feature.X_forecast_sample
        feature_dates = pd.DatetimeIndex(feature_dates)

        missing = feature_dates.difference(X.index)
        if len(missing):
            raise KeyError(f"feature_dates missing from forecast_feature.X_out_sample: {list(missing)}")

        positions = np.array([X.index.get_loc(d) for d in feature_dates])
        starts = positions - self.sequence_length + 1
        if (starts < 0).any():
            bad = feature_dates[starts < 0]
            raise ValueError(f"Not enough rows before these dates for sequence_length={self.sequence_length}: {list(bad)}")

        values = X[self.feature_cols].to_numpy(dtype=np.float32)
        sequences = np.stack([values[s:p + 1] for s, p in zip(starts, positions)])
        normalized = (sequences - self.feature_mean) / self.feature_std

        device = next(self.underlying_model.parameters()).device
        return torch.tensor(normalized, dtype=torch.float32, device=device)

    def predict_variance_batch(self, forecast_feature, feature_dates) -> pd.Series:
        import torch

        feature_dates = pd.DatetimeIndex(feature_dates)
        batch = self._build_sequence_batch(forecast_feature, feature_dates)

        self.underlying_model.eval()
        with torch.no_grad():
            predicted_scaled = self.underlying_model(batch).squeeze(-1).cpu().numpy()

        variances = predicted_scaled.astype(float) / self.target_scale
        return pd.Series(
            DATA_QUALITY_check_variance_floor(variances, dates=feature_dates, context="TrainedLstmModel.predict_variance_batch"),
            index=feature_dates,
        )

    def predict_variance(self, forecast_feature, feature_date) -> float:
        return float(self.predict_variance_batch(forecast_feature, [feature_date]).iloc[0])