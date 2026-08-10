"""
End-to-end smoke test for the GroundTruthCalendar / TrainingFeature / TrainedModel
pipeline in data_management.py, feature_builders.py, tree_models.py.

Run from anywhere with: python3 MLProject/smoke_test_var_pipeline.py
(or `python3 smoke_test_var_pipeline.py` from inside MLProject/).

Exercises both the wide and long paths end to end:
    build_*_features -> xgboost_train(window_size=...) -> residual_bank_in_sample
    (+ ewma_variance_bank_in_sample for long) -> multi_day_var_from_*
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_HERE), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

import data_management as dm
import feature_builders as fb
import tree_models as tm

TICKERS = ["AAPL", "JPM", "XOM", "KO"]
WEIGHTS = {"AAPL": 0.25, "JPM": 0.25, "XOM": 0.25, "KO": 0.25}

WINDOW_SIZE = 20
VAR_CONFIDENCE = 0.95

_df = pd.read_csv(_HERE / "data_general" / "4_ticker_data.csv", parse_dates=["Date"], index_col="Date")
TICKERS_RETURNS = _df[TICKERS].pct_change().dropna()


def make_calendar() -> dm.DailyGroundTruthCalendar:
    return dm.DailyGroundTruthCalendar(
        TICKERS_RETURNS, TICKERS, WEIGHTS,
        train_start="2024-06-03", train_end="2024-11-29",
        forecast_start="2024-12-02", forecast_end="2024-12-31",
    )


def run_wide() -> pd.DataFrame:
    print("\n=== WIDE path ===")
    calendar = make_calendar()
    training_feature, forecast_feature = fb.build_wide_features(calendar)

    model = tm.xgboost_train(calendar, training_feature, WINDOW_SIZE)
    assert not model.residual_bank_in_sample.residuals.empty, "wide residual_bank_in_sample is empty"
    print(f"residual_bank_in_sample seeded with {len(model.residual_bank_in_sample.residuals)} rows")

    var_df = dm.multi_day_var_from_pretrained_model(model, forecast_feature, calendar, WINDOW_SIZE, VAR_CONFIDENCE)
    assert not var_df.empty, "multi_day_var_from_pretrained_model returned an empty DataFrame"
    assert var_df["varForecast"].notna().all(), "NaN in wide varForecast"
    assert var_df["loss"].notna().all(), "NaN in wide loss"
    print(var_df.head())
    print(f"-> {len(var_df)} wide VaR forecasts produced")
    return var_df


def run_long() -> pd.DataFrame:
    print("\n=== LONG path ===")
    calendar = make_calendar()
    training_feature, forecast_feature = fb.build_long_features(calendar)

    model = tm.xgboost_train(calendar, training_feature, WINDOW_SIZE)
    assert not model.residual_bank_in_sample.residuals.empty, "long residual_bank_in_sample is empty"
    assert model.ewma_variance_bank_in_sample is not None, "ewma_variance_bank_in_sample was not seeded"
    print(f"residual_bank_in_sample seeded with {len(model.residual_bank_in_sample.residuals)} rows")
    print(f"ewma_variance_bank_in_sample seeded with {len(model.ewma_variance_bank_in_sample.portfolio_variance)} rows")

    var_df = dm.multi_day_var_from_long_format_model(
        model, calendar, forecast_feature, WINDOW_SIZE, VAR_CONFIDENCE,
        model.residual_bank_in_sample, model.ewma_variance_bank_in_sample,
    )
    assert not var_df.empty, "multi_day_var_from_long_format_model returned an empty DataFrame"
    assert var_df["varForecast"].notna().all(), "NaN in long varForecast"
    assert var_df["loss"].notna().all(), "NaN in long loss"
    print(var_df.head())
    print(f"-> {len(var_df)} long VaR forecasts produced")
    return var_df


if __name__ == "__main__":
    run_wide()
    run_long()
    print("\nSmoke test passed.")
