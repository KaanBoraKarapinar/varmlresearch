import pandas as pd
from abc import ABC, abstractmethod

from .feature_types import TrainingFeature, ForecastFeature
from .ResidualBank import ResidualBank
from .EwmaVarianceBank import EwmaVarianceBank


class TrainedModel(ABC):
    """
    A frozen, trained model plus everything needed to score OOS bars with it.
    Each model type implements its own predict_variance.
    """

    def __init__(
        self,
        training_feature: TrainingFeature,
        underlying_model: object,
        model_name: str,
        target_scale: float,
        feature_cols: list[str],
        residual_bank_in_sample: ResidualBank | None = None,
        ewma_variance_bank_in_sample: "EwmaVarianceBank | None" = None,
    ):
        self.training_feature = training_feature
        self.underlying_model = underlying_model
        self.model_name = model_name
        self.target_scale = target_scale
        self.feature_cols = feature_cols
        self.residual_bank_in_sample = residual_bank_in_sample
        self.ewma_variance_bank_in_sample = ewma_variance_bank_in_sample

    @abstractmethod
    def predict_variance_single_wide(self, forecast_feature: "ForecastFeature", feature_date: pd.Timestamp) -> float:
        """Predict the variance for the bar immediately AFTER feature_date (wide/single-ticker feature)."""

    @abstractmethod
    def predict_variance_batch_wide(self, forecast_feature: "ForecastFeature", feature_dates: pd.Timestamp) -> pd.Series:
        """Predict the variances for the given feature dates (wide/single-ticker feature)."""

    def predict_variance_batch_long_by_ticker(
        self, X_long: pd.DataFrame, id_cols: pd.DataFrame, feature_dates
    ) -> dict[str, pd.Series]:
        """
        Splits a long-format (one row per (Date, ticker)) feature block into one
        date-indexed wide view per ticker via id_cols, and scores each with
        predict_variance_batch_wide.
        """
        feature_dates = pd.DatetimeIndex(feature_dates)
        result: dict[str, pd.Series] = {}
        for ticker, rows in id_cols.groupby("ticker").groups.items():
            ticker_X = X_long.loc[rows, self.feature_cols].set_axis(
                id_cols.loc[rows, "Date"].to_numpy(), axis=0
            ).sort_index()
            ticker_view = ForecastFeature(
                start=ticker_X.index.min(), end=ticker_X.index.max(), feature_dates=ticker_X.index,
                target_scale=self.target_scale, frequency=self.training_feature.frequency,
                feature_cols=self.feature_cols, type="wide",
                X_forecast_sample=ticker_X, y_forecast_sample=pd.DataFrame(),
                DANGEROUS_r_next_forecast_sample=pd.Series(dtype=float), id_cols=None,
            )
            result[ticker] = self.predict_variance_batch_wide(ticker_view, feature_dates)
        return result
