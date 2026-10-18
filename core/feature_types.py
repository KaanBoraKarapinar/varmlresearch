import pandas as pd
from typing import NamedTuple


class TrainingFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    feature_dates: pd.DatetimeIndex
    target: object  # targets.Target: forward() made y_in_sample, inverse() turns a prediction into a variance
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

    # per-row EWMA variance known at the feature date (same index as X_in_sample); only built for a
    # target that needs it (targets.RatioToBase), None otherwise
    base_variance: pd.Series | None = None

    # own-day rule: training_feature_residual_outcome_dates[i] is the bar AFTER feature_dates[i] (calendar.next_bars),
    # the day the return DANGEROUS_r_next of that feature date really happens on; residual bank rows are labeled with it
    training_feature_residual_outcome_dates: pd.DatetimeIndex | None = None


class ForecastFeature(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    feature_dates: pd.DatetimeIndex
    target: object  # targets.Target, see TrainingFeature
    frequency: pd.Timedelta
    feature_cols: list[str]
    type: str  # "single" | "wide" | "long"

    ##important part:

    X_forecast_sample: pd.DataFrame
    y_forecast_sample: pd.DataFrame

    DANGEROUS_r_next_forecast_sample: pd.Series

    # only for long format
    id_cols: pd.DataFrame | None

    # per-row EWMA variance known at the feature date (same index as X_forecast_sample), see TrainingFeature
    base_variance: pd.Series | None = None

    def features_at(self, date) -> pd.Series:
        date = pd.Timestamp(date)
        if date not in self.X_forecast_sample.index:
            raise KeyError(f"{date} not in ForecastFeature range [{self.start} .. {self.end}]")
        return self.X_forecast_sample.loc[date]
