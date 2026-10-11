import pandas as pd
from typing import NamedTuple


class TrainingFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    feature_dates: pd.DatetimeIndex
    target_scale: float
    frequency: pd.Timedelta
    feature_cols: list[str]
    type: str  # "single" | "wide" | "long"

    ##important part:

    X_in_sample: pd.DataFrame
    X_out_sample_last_index: pd.DataFrame

    y_in_sample: pd.DataFrame

    DANGEROUS_r_next_in_sample: pd.Series


    #only for long format
    X_in_sample_id_cols: pd.DataFrame | None
    X_out_sample_last_index_id_cols: pd.DataFrame | None


class ForecastFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    feature_dates: pd.DatetimeIndex
    target_scale: float
    frequency: pd.Timedelta
    feature_cols: list[str]
    type: str  # "single" | "wide" | "long"

    ##important part:

    X_forecast_sample: pd.DataFrame
    y_forecast_sample: pd.DataFrame

    DANGEROUS_r_next_forecast_sample: pd.Series

    # only for long format
    id_cols: pd.DataFrame | None

    def features_at(self, date) -> pd.Series:
        date = pd.Timestamp(date)
        if date not in self.X_forecast_sample.index:
            raise KeyError(f"{date} not in ForecastFeature range [{self.start} .. {self.end}]")
        return self.X_forecast_sample.loc[date]
