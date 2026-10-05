# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import optuna

# %% [markdown]
# # Project Overview
#
# This project builds a framework for creating ** Value-at-Risk (VaR)** forecasts with 3 different models and compares the performansce of hyperparameter/ data actuality and density decisions.
#
# All four models plug into the same VaR pipeline: fit, predict
# next-day variance, convert to VaR via the empirical quantile of standardized
# residuals (inspired by calculating VaR via GARCH residuals), backtest with the Kupiec/Christoffersen/traffic-light test suite.
#
#  Project is divided in 3 different files:
# * `var_lstm_xgboost.py` -- the modeling library: feature engineering and models
# * `var_backtest_utils.py` -- the backtest test suite (Kupiec, Christoffersen,
#   traffic light, binomial),
# * 12_ml_models_benchmarking.py` (this file) -- the benchmarking/tuning layer:
#   which `training_window` and which hyperparameters produce the best-calibrated
#   VaR forecasts
#
# ## Feature Building
# Currently, the framework only uses the closing price differnece of tickers for each day. I'm planning to expend use multi day data, but as "hacky" guardrails implememented to check whether any date function is really inside the feature dataframe/ a weekday where trades happened needs general structural change to accept and clean single trading day with 6,5 days.
# * (especially for GARCH based VaR forecast using multi day horizont, using the difference of
#
# Target: squared percent return as a proxy for variance
#
# * Portfolio weights can be rearranged via notebook wide definition of `WEIGHTS`. Currenttly an equal weight porfolio is assumed.
#
# * According to goal, long or wide format features can be created.
#     * if single tickers VaR  wanted to be predicted with available data `build_ml_features_portfolio_long_format` is used,
#     * For portfolio wide var calculations 'build_ml_features_portfolio_wide_format' is used.
#         * For tree models to capture the autocorrelative nature for volatility, lagged values of previous returns can be added.
#
# For each day:
# * return
# * absolute_return
# * in windows of 5,10 and 21 days (to capture short/meidum/long term effects)
#     * realized volatility
#     * mean return
#     * SKEWNESS of return
#         * to capture
# * EWMA Volatility for 21 days
# * laged values
#
# are created as features
#
# ## Helper methods / classes
#
# * `calculate_empirical_var` calculates var from given residuals
# * `MLFeatureResult` saves the built features' dataframe with all tag references of feature targets
#     * as r_next squared (as target) given
#     * and r_next itself for further VaR calculations also added into the dataframe,
#         * AND USING IT AS A FEATURE IS A DIREKT INFORMATION LEAKAGE
#
# * ForecastResult holds
#     * trained model
#     * residuals dataframe
#         *  created by a helper function 'create_residuals_dataframe'
#     * mu_hat, average of every r_next other than last out of sample day, to calculate var
#     * oos_variance = predicted target, "squared percent return as a proxy for variance"
#     *

# %%

# %% [markdown]
# # VaR  Comparison

# %% [markdown]
# ## Configuration

# %%
# resolved against this file's own location (not the caller's cwd) so the path
# stays correct whether this file is %run from Jupyter or imported as a module
# from anywhere else.
DATA_PATH = Path(__file__).resolve().parent.parent / "data_general" / "4_ticker_data.csv"
TICKERS = ["AAPL", "JPM", "XOM", "KO"]
TICKERS_WITH_PORTFOLIO = TICKERS + ["PORTFOLIO"]

WEIGHTS = {
    "AAPL": 0.25,
    "JPM": 0.25,
    "XOM": 0.25,
    "KO": 0.25,

}

# demo/data-loading side effects -- guarded so a plain `import` of this file only
# defines functions/classes, it doesn't read the CSV or build a global dataframe.
# `%run`-ing this file from Jupyter still executes this block exactly as before,
# since %run sets __name__ to "__main__".
if __name__ == "__main__":
    raw = pd.read_csv(DATA_PATH)

    close_price = raw.copy()
    close_price["Date"] = pd.to_datetime(close_price["Date"])
    close_price = close_price.set_index("Date").sort_index()

    percent_return_with_portfolio = close_price.pct_change()
    percent_return_with_portfolio["PORTFOLIO"] = percent_return_with_portfolio[TICKERS].mul(WEIGHTS).sum(axis=1)
    display(percent_return_with_portfolio.tail())



# %% [markdown]
# Project settings (DEMO_SCALE, WINDOW_SIZE, PERIOD_OF_FORECAST, VALIDATION_PERIOD,
# N_TRIALS, VAR_CONFIDENCE, MODELS, TUNE_GARCH) and MLflow tracking setup now live in
# `12_ml_models_benchmarking.py`, which `%run`s this file -- nothing below this point
# depends on them, only `12_ml_models_benchmarking.py`'s own cells do.

# %% [markdown]
# # Defining LSTM Model

# %% [markdown]
# ## Feature Creation
# To feed into the the models, `build_ml_features` recieves the log_return dataframe
#
#
#
# ###Features
# * return
# * absolute_return
# * in windows of 5,10 and 21 days the (to capture short/meidum/long term effects)
#     * realized volatility
#     * mean return
#     * SKEWNESS of return
#         * to capture
# * EWMA Volatility for 21 days
#
#
# ### Target
# * Models trying to predict squared return of **NEXT_DAY** to derive volatility from that proxy
#     * as we can not observe "volatility"
# * the very last row's target is intentionally left NaN (it's the out-of-sample day being forecast)
#
#
# ## Wide-Long Format Discussion
# * Initially started creating the wide version, but upon research i wasnt sure whether long version would be better suiting.
# * Answer: it depends
#
#
# * **When the target is shared/aggregate (e.g., portfolio volatility)**:
#     * use a wide format. The model must observe all tickers within the same row, at the same point
#
# * When the target is ticker-specific (each ticker forecasts its own future): a long format with ticker as a categorical feature is more defensible. This yields a larger effective sample size and allows shared patterns to be learned across tickers.
# *

# %% [markdown]
# ###### pandas: `.stack()` to wide -> long format

# %%
if __name__ == "__main__":
    ## example wide -> long format change pandas, to later create portolio_wide feature creation_formula
    long = percent_return_with_portfolio[TICKERS].stack().rename("r").reset_index()
    display(long.head(10))
    # stack makes the name of level as level_1, we have to tag with ticker
    long.columns = ["Date", "ticker", "r"]

# %% [markdown]
# ###### pandas: 'group.by' object and calling transform on it
#
# long.groupby("ticker") like in sql GROUP-BY statement, creates groups for every unique value of ticker
#     (think like there are 4 different datasets)
#         **ENABLES US TO TAKE AT R_NEXT NEXT TICKER VALUE OF APPL, NOT THE DIRECT SHIFT (FOR EXAMPLE JPMS Nan VALUE)**
# g = long.groupby("ticker")["r"], keeps the grouping logic, but just the "r" column is stored
# as its lasily evaluated, effects can not be seen before transform is called
# * (sql window functions logic)

# %%
#g = long.groupby("ticker")["r"]
#g

# %%
#r_next = g.transform(lambda x: x.shift(-1))
#long["r_next"] = r_next
#display(long.head(10))

# %%

# %%
from typing import NamedTuple

class MLFeatureResult(NamedTuple):
    feature_cols: list[str]
    df: pd.DataFrame
    type: str  # "single" | "wide" | "long"
    portfolio_df: pd.DataFrame
    # One row per trading date. Always contains:
    #   r       = realized portfolio return on date t
    #   r_next  = realized portfolio return on date t+1
    #   target  = r_next ** 2




# %%

def _validated_portfolio_weights() -> pd.Series:
    """Return normalized portfolio weights and fail loudly on missing config."""
    weights = pd.Series(
        {ticker: WEIGHTS.get(ticker, np.nan) for ticker in TICKERS},
        dtype=float,
    )

    if weights.isna().any():
        missing = weights[weights.isna()].index.tolist()
        raise ValueError(f"Missing portfolio weights for: {missing}")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite.")

    total = float(weights.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Portfolio weights sum to zero.")

    return weights / total


def _build_portfolio_frame(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the canonical one-row-per-trading-day portfolio calendar carried
    inside MLFeatureResult.
    """
    weights = _validated_portfolio_weights()

    missing = [ticker for ticker in TICKERS if ticker not in returns_df.columns]
    if missing:
        raise KeyError(f"Missing ticker return columns: {missing}")

    portfolio_r = (
        returns_df[TICKERS]
        .mul(weights, axis="columns")
        .sum(axis=1, min_count=len(TICKERS))
        .rename("r")
    )

    portfolio_df = portfolio_r.to_frame()
    portfolio_df["r_next"] = portfolio_df["r"].shift(-1)
    portfolio_df["target"] = portfolio_df["r_next"] ** 2
    return portfolio_df.sort_index()


def build_ml_features_single_ticker(
    r: pd.Series,
    windows=(5, 10, 21),
    sq_return_lags: tuple | None = None,
) -> MLFeatureResult:
    """
    :param r:  returns series for single tickr
    :param windows: window size for feature calculation
    :param sq_return_lags: optional tuple of lag days for lagged squared-return
        features (e.g. (1, 5, 63) for yesterday / 1 week ago / ~3 months ago).
        If None or empty, no lagged squared-return features are added.
    :return:
        MLFeatureResult(feature_cols, df) where:
        * feature_cols: list of feature column names
        * df: DataFrame containing, for day t as feature:
            * return
            * absolute return
            * skewness
            * for given windows volatility, average return and skewness
            * EWMA Volatility for 21 days
            * (optional) lagged squared returns r_{t-lag}^2 for each lag in sq_return_lags

          and for day t+1 as label:
            * squared_next day return
            * (and next day return for calculation)


    """
    df = pd.DataFrame({"r": r, "abs_r": r.abs()})
    for w in windows:
        df[f"real_vol_{w}"] = r.rolling(w).std()
        df[f"mean_abs_{w}"] = df["abs_r"].rolling(w).mean()
        df[f"skew_{w}"] = r.rolling(w).skew()
    df["ewma_vol"] = r.ewm(span=21).std()

    if sq_return_lags:
        r_sq = df["r"] ** 2
        for lag in sq_return_lags:
            df[f"r2_lag_{lag}"] = r_sq.shift(lag)

    df["r_next"] = r.shift(-1)          # signed next-day return, for std_resid
    df["target"] = df["r_next"] ** 2    # squared next-day return, the regression target

    feature_cols = [c for c in df.columns if c not in ("r_next", "target")]
    portfolio_df = pd.DataFrame({"r": r}).sort_index()
    portfolio_df["r_next"] = portfolio_df["r"].shift(-1)
    portfolio_df["target"] = portfolio_df["r_next"] ** 2
    return MLFeatureResult(
        feature_cols,
        df.dropna(subset=feature_cols),
        type="single",
        portfolio_df=portfolio_df,
    )


def build_ml_features_portfolio_long_format(
    returns_df: pd.DataFrame,
    windows=(5, 10, 21),
    sq_return_lags: tuple | None = None,
) -> MLFeatureResult:
    """
    :param r: percent_return_with_portfolio dataset
    :param windows: window size for feature calculation
    :param sq_return_lags: optional tuple of lag days for lagged squared-return
        features (e.g. (1, 5, 63) for yesterday / 1 week ago / ~3 months ago),
        computed per-ticker. If None or empty, no lagged squared-return
        features are added.
    :return:
        MLFeatureResult(feature_cols, df) where:
        * feature_cols: list of feature column names
        * df: DataFrame containing, for day t as feature:
            * return
            * absolute return
            * skewness
            * for given windows volatility, average return and skewness
            * EWMA Volatility for 21 days
            * (optional) lagged squared returns r_{t-lag}^2 for each lag in sq_return_lags, per ticker
            * market-level EWMA volatility (from PORTFOLIO), shared across all ticker rows

          and for day t+1 as label:
            * squared_next day return
            * (and next day return for calculation)

    """
    long = returns_df[TICKERS].stack().rename("r").reset_index()
    long.columns = ["Date", "ticker", "r"]
    long = long.sort_values(["Date", "ticker"]).reset_index(drop=True)

    g = long.groupby("ticker")["r"]

    long["abs_r"] = long["r"].abs()
    for w in windows:
        long[f"real_vol_{w}"] = g.transform(lambda x: x.rolling(w).std())
        long[f"mean_abs_{w}"] = long.groupby("ticker")["abs_r"].transform(lambda x: x.rolling(w).mean())
        long[f"skew_{w}"]     = g.transform(lambda x: x.rolling(w).skew())

    long["ewma_vol"] = g.transform(lambda x: x.ewm(span=21).std())

    if sq_return_lags:
        long["r_sq"] = long["r"] ** 2
        for lag in sq_return_lags:
            long[f"r2_lag_{lag}"] = long.groupby("ticker")["r_sq"].transform(lambda x: x.shift(lag))
        long = long.drop(columns="r_sq")

    long["r_next"]   = g.transform(lambda x: x.shift(-1))
    long["target"]   = long["r_next"] ** 2

    market_vol = returns_df["PORTFOLIO"].ewm(span=21).std().rename("market_ewma_vol")
    long = long.merge(market_vol, left_on="Date", right_index=True)

    feature_cols = [c for c in long.columns if c not in ("Date", "ticker", "r_next", "target")]
    portfolio_df = _build_portfolio_frame(returns_df)
    return MLFeatureResult(
        feature_cols,
        long.dropna(subset=feature_cols),
        type="long",
        portfolio_df=portfolio_df,
    )


def build_ml_features_portfolio_wide_format(
    returns_df: pd.DataFrame,
    windows=(5, 10, 21),
    sq_return_lags: tuple | None = None,
    multiple_targets=False,
) -> MLFeatureResult:
    """
    :param r: percent_return_with_portfolio dataset
    :param windows: window size for feature calculation
    :param sq_return_lags: optional tuple of lag days for lagged squared-return
        features (e.g. (1, 5, 63) for yesterday / 1 week ago / ~3 months ago),
        computed per column (ticker + portfolio). If None or empty, no lagged
        squared-return features are added.
    :param multiple_targets: if True, target/r_next built per-ticker (suffix per ticker);
                              if False, only a single "target" / "r_next"

    :return:
        MLFeatureResult(feature_cols, df) where:
        * feature_cols: list of feature column names
        * df: DataFrame containing, for day t as feature:
            * return
            * absolute return
            * skewness
            * for given windows volatility, average return and skewness
            * EWMA Volatility for 21 days
            * (optional) lagged squared returns r_{t-lag}^2 for each lag in sq_return_lags, per ticker/portfolio
            * market-level EWMA volatility (from PORTFOLIO), shared across all ticker rows

          and for day t+1 as label:
            * squared_next day return
            * (and next day return for calculation)

    """
    abs_r = returns_df[TICKERS_WITH_PORTFOLIO].abs()

    df = returns_df[TICKERS_WITH_PORTFOLIO].copy()
    df = df.join(abs_r.add_suffix("_abs_r"))

    for w in windows:
        df = df.join(returns_df[TICKERS_WITH_PORTFOLIO].rolling(w).std().add_suffix(f"_real_vol_{w}"))
        df = df.join(abs_r.rolling(w).mean().add_suffix(f"_mean_abs_{w}"))
        df = df.join(returns_df[TICKERS_WITH_PORTFOLIO].rolling(w).skew().add_suffix(f"_skew_{w}"))

    df = df.join(returns_df[TICKERS_WITH_PORTFOLIO].ewm(span=21).std().add_suffix("_ewma_vol"))

    if sq_return_lags:
        r_sq = returns_df[TICKERS_WITH_PORTFOLIO] ** 2
        for lag in sq_return_lags:
            df = df.join(r_sq.shift(lag).add_suffix(f"_r2_lag_{lag}"))

    # market-level feature, from PORTFOLIO, shared across all ticker rows
    market_ewma_vol = returns_df["PORTFOLIO"].ewm(span=21).std().rename("market_ewma_vol")
    df = df.join(market_ewma_vol)

    if multiple_targets:
        r_next = returns_df[TICKERS_WITH_PORTFOLIO].shift(-1)
        df = df.join(r_next.add_suffix("_r_next"))
        df = df.join((r_next ** 2).add_suffix("_target"))

        exclude_cols = [f"{t}_r_next" for t in TICKERS_WITH_PORTFOLIO] + [f"{t}_target" for t in TICKERS_WITH_PORTFOLIO]
    else:
        port_r_next = returns_df["PORTFOLIO"].shift(-1).rename("r_next")
        port_target = (port_r_next ** 2).rename("target")
        df = df.join(port_r_next)
        df = df.join(port_target)

        exclude_cols = ["r_next", "target"]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    portfolio_df = _build_portfolio_frame(returns_df)
    return MLFeatureResult(
        feature_cols,
        df.dropna(subset=feature_cols),
        type="wide",
        portfolio_df=portfolio_df,
    )


# %%
if __name__ == "__main__":
    display(build_ml_features_single_ticker(percent_return_with_portfolio["AAPL"]).df.head(10))

# %%
if __name__ == "__main__":
    display(build_ml_features_portfolio_long_format(percent_return_with_portfolio).df.head(10))


# %%
if __name__ == "__main__":
    display(build_ml_features_portfolio_wide_format(percent_return_with_portfolio).df.head(10))


# %% [markdown]
# # LSTM Forecast
#
# ## walk-forward logic from our wrapper
#
# * As we wanna try different window lengths for our models to see to test their performance, main wrapper function to call models have following parameters:
#
# * **window size** = e.g. 100
#     * how many days we will fit into the model
# * **periood_of_forecast** = e.g. 3
#     * how many days are we gonna predict with that data,
#
# If we have data for **300 days**, this applies:
#
# * Example;
#
# ```
# model1: create a new model fit a model with data of days [1..100], predict with that model VaR for day 101+102+103, compare with realised loss
#     * DATA OF DAYS [1-101], (n=window size +1) data is loaded by wrapper, as models drop last item, "out_of_sample_row"
#
# model 2: fit with [2..102], predict VaR for day 103+104+105, compare with realised loss
# model 3: fit with [3..103], predict day 104+..., compare with realised loss
# .
#
# model 199: fit with [199..299], predict day 300, compare with realised loss of day 300 (301,302 not defined)
#
# ```
#
#
# ## feature creation and lag windows
#
# * There are 2 seperate aggregation logics applied during the feature creations
#
# **feature_windows** = system adds  average return and volatility of those windows in dataframe defined here.
#     * standart = FEATURE_WINDOWS = (5, 10, 21)
#
# **sq_return_lags** = for tree models, squared returns with X day lags (of X day prior) defitinions can be done via this parameter
#     * thereby they can track/ recreate volatiliy clustering effects
#
# ## LSTM SPECIFIC sequence length
#
# **sequence_length** : LSTMs have internally the same "meta" walk-forward logic created for the framework.
#
# if seq_length = 3
#
# and
#
# * **window size** = e.g. 100
#     * how many days we will fit into the model
# * **periood_of_forecast** = e.g. 3
#     * how many days are we gonna predict with that data,
#
#
# ```
# * model1: create a new model fit a model with data of days [1..100], predict with that model VaR for day 101+102+103, compare with realised loss.
#     * Data for days[1-101] is loaded:
#     INSIDE THE LSTM:
#
#     # seq1: X: features[day1,day2,day3] y: target[day3], as the target of day3 is day4s value
#     # seq2: X: features[day2,day3,day4] y: target[day4], as the target of day4 is day5s value
#     ....
#     # seq98   X: features[day98,day99,day100] y: target[day100] as the target of day4 is day5s value
# ```
#
#
#

# %% [markdown]
#

# %%
class ForecastResult(NamedTuple):
    model: object
    model_name: str

    predicted_oos_next_day_variance: float
    mu_hat: float
    residuals_in_sample: pd.DataFrame

    #r_next: pd.Series -> functionality delegated to residual calculation
    variance_in_sample: pd.Series
    feature_cols: list[str]

    sequence_length: int | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    target_scale: float = 1.0


##### RESIDUAL CALCULATION HELPER FUNCTIONS ####

def create_residuals_dataframe(
    residual_dates,
    realized_returns,
    predicted_variances,
    mu_hat: float,
) -> pd.DataFrame:
    """
    CALCULATES STANDARTIZED RESIDUALS, TURNS RESIDUAL_DATES INTO INDEXES

    """
    dates = pd.DatetimeIndex(residual_dates, name="Date")
    realized_returns = np.asarray(realized_returns, dtype=float)
    predicted_variances = np.maximum(np.asarray(predicted_variances, dtype=float), 1e-12)

    if not (len(dates) == len(realized_returns) == len(predicted_variances)):
        raise ValueError(
            "residual_dates, realized_returns, and predicted_variances must have equal length."
        )

    predicted_sigmas = np.sqrt(predicted_variances)
    standardized_residuals = (realized_returns - float(mu_hat)) / predicted_sigmas

    return (
        pd.DataFrame(
            {
                "actual_return": realized_returns,
                "predicted_variance": predicted_variances,
                "predicted_sigma": predicted_sigmas,
                "mu_hat": float(mu_hat),
                "standardized_residual": standardized_residuals,
            },
            index=dates,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )


def append_residual_observation(
    residuals_df: pd.DataFrame,
    date: pd.Timestamp,
    actual_return: float,
    predicted_variance: float,
    mu_hat: float,
    window_size: int,
) -> pd.DataFrame:
    """

    CALCULATES STANDART RESIDUALS FOR GIVEN NEW PARAMETERS AND APPENDS TO A PREVIOUSLY EXISTING
    RESIDUALS DATAFRAME, WHICH HAS DIFFERENT MU_HAT AND PREDICTED_VARIANCE VALUES,
    TO PREVENT LOOK-AHEAD-BIAS



    :return:
    """
    new_row = create_residuals_dataframe(
        residual_dates=[date],
        realized_returns=[actual_return],
        predicted_variances=[predicted_variance],
        mu_hat=mu_hat,
    )
    return (
        pd.concat([residuals_df.drop(index=date, errors="ignore"), new_row])
        .sort_index()
        .tail(window_size)
    )



# %%
def garch_forecast(features: MLFeatureResult, p: int = 1, q: int = 1, dist: str = "normal") -> ForecastResult | None:
    """

    CALCULATES THE GARCH FORECAST WITH THE VALUES IN ROW "r" "MLFeatureResult"
    IF THERE IS NO ROW R (FOR WIDE FORMAT PORTFOLIO), THE ROW "PORTFOLIO" IS USED AS RETURN OF PORTFOLIO.


    """


    from arch import arch_model

    # single/long-ticker features carry a plain "r" column; wide-format portfolio
    # features don't (only per-ticker columns + "PORTFOLIO") -- fall back to that
    return_col = "r" if "r" in features.df.columns else "PORTFOLIO"
    returns = features.df[return_col].dropna()
    returns_pct = returns * 100.0
    if len(returns_pct) < 30:
        return None

    try:
        model = arch_model(returns_pct, p=p, q=q, mean="constant", vol="garch", dist=dist)
        res = model.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=1, reindex=False)
    except Exception:
        return None

    mu_hat = float(res.params["mu"]) / 100.0
    variance_hat_next = float(fc.variance.values[-1, 0]) / 10_000.0
    variance_in_sample = (res.conditional_volatility.dropna() ** 2) / 10_000.0
    realized_returns = features.df.loc[variance_in_sample.index, return_col]

    residuals_df = create_residuals_dataframe(
        residual_dates=variance_in_sample.index,
        realized_returns=realized_returns.to_numpy(),
        predicted_variances=variance_in_sample.to_numpy(),
        mu_hat=mu_hat,
    )

    return ForecastResult(
        model=res,
        model_name="garch",
        predicted_oos_next_day_variance=variance_hat_next,
        mu_hat=mu_hat,
        residuals_in_sample=residuals_df,
        #r_next=residuals_df["actual_return"],
        variance_in_sample=residuals_df["predicted_variance"],
        feature_cols=features.feature_cols,
    )

# %%
import torch
import torch.nn as nn


class TinyLSTM:

    def __init__(self, n_features: int, hidden: int = 8):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
                self.softplus = nn.Softplus()  # variance / squared return cannot be negative

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                raw_output = self.fc(h[-1]).squeeze(-1)
                return self.softplus(raw_output) + 1e-8

        self.net = Net()


def lstm_forecast(
    features: MLFeatureResult,
    sequence_length: int = 10,
    hidden: int = 8,
    epochs: int = 30,
    lr: float = 0.01,
) -> ForecastResult | None:
    TARGET_SCALE = 10_000.0

    vals_X = features.df[features.feature_cols].to_numpy(dtype=np.float32)
    target_y = features.df["target"].to_numpy(dtype=np.float32) * TARGET_SCALE
    r_next = features.df["r_next"].to_numpy(dtype=np.float32)
    n = len(features.df)

    if n <= sequence_length:
        return None

    X_seq, y_seq, r_next_seq, target_dates = [], [], [], []

    for i in range(sequence_length - 1, n):
        X_seq.append(vals_X[i - sequence_length + 1:i + 1])
        y_seq.append(target_y[i])
        r_next_seq.append(r_next[i])
        target_dates.append(features.df.index[i + 1] if i + 1 < n else pd.NaT)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    r_next_seq = np.asarray(r_next_seq, dtype=np.float32)
    target_dates = pd.DatetimeIndex(target_dates)

    X_train_seq = X_seq[:-1]
    X_oos_seq = X_seq[-1:]
    y_train_seq = y_seq[:-1]
    r_next_train = r_next_seq[:-1]
    residual_dates = target_dates[:-1]

    if len(X_train_seq) == 0:
        return None

    mean_features = X_train_seq.reshape(-1, X_train_seq.shape[-1]).mean(axis=0)
    standard_deviation_features = X_train_seq.reshape(-1, X_train_seq.shape[-1]).std(axis=0) + 1e-8

    X_train_s = (X_train_seq - mean_features) / standard_deviation_features
    X_oos_s = (X_oos_seq - mean_features) / standard_deviation_features
    torch.manual_seed(0)

    X_train_torch = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train_seq, dtype=torch.float32)
    X_oos_torch = torch.tensor(X_oos_s, dtype=torch.float32)

    model = TinyLSTM(n_features=X_train_torch.shape[-1], hidden=hidden).net
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train_torch)
        loss = loss_fn(pred, y_train_torch)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_in_sample = model(X_train_torch).cpu().numpy()
        pred_oos = float(model(X_oos_torch).cpu().numpy()[0])

    variance_in_sample_values = np.maximum(pred_in_sample / TARGET_SCALE, 1e-12)
    variance_oos = max(pred_oos / TARGET_SCALE, 1e-12)
    mu_hat = float(np.mean(r_next_train))

    residuals_df = create_residuals_dataframe(
        residual_dates=residual_dates,
        realized_returns=r_next_train,
        predicted_variances=variance_in_sample_values,
        mu_hat=mu_hat,
    )

    return ForecastResult(
        model=model,
        model_name="lstm",
        predicted_oos_next_day_variance=variance_oos,
        mu_hat=mu_hat,
        residuals_in_sample=residuals_df,
        #r_next=residuals_df["actual_return"],
        variance_in_sample=residuals_df["predicted_variance"],
        feature_cols=features.feature_cols,
        sequence_length=sequence_length,
        feature_mean=mean_features,
        feature_std=standard_deviation_features,
        target_scale=TARGET_SCALE,
    )


# %%
if __name__ == "__main__":
    feature_df = build_ml_features_portfolio_wide_format(percent_return_with_portfolio)

# %%
# demo call -- commented out so `%run`-ing this file from 12_ml_models_benchmarking.py
# doesn't retrain an LSTM every time; uncomment to inspect a single fit interactively
# forecast = lstm_forecast(feature_df)
# forecast

# %%
# demo call -- see note above
# display(forecast.residuals_in_sample["standardized_residual"].describe())
# display(forecast.residuals_in_sample["standardized_residual"].skew())
# display(forecast.residuals_in_sample["standardized_residual"].kurtosis())


# %% [markdown]
# # Tree Based Models

# %%
def _tree_forecast(
    features: MLFeatureResult,
    model_constructor,
    model_name,
    model_kwargs: dict,
) -> ForecastResult | None:
    """
    Fit a tree variance model without leaking the latest long-format date.

    In long format there are multiple ticker rows per feature date, so all
    rows from the latest feature date must be held out together.
    """
    df = features.df

    if features.type == "long":
        if "Date" not in df.columns:
            raise ValueError("Long-format features must contain a Date column.")

        oos_feature_date = pd.Timestamp(df["Date"].max())

        train_df = (
            df.loc[df["Date"] < oos_feature_date]
            .dropna(subset=["target", "r_next"])
            .copy()
        )
        oos_df = (
            df.loc[df["Date"] == oos_feature_date]
            .dropna(subset=features.feature_cols)
            .copy()
        )

        if train_df.empty or oos_df.empty:
            return None

        missing_tickers = set(TICKERS).difference(set(oos_df["ticker"]))
        if missing_tickers:
            raise ValueError(
                f"OOS feature date {oos_feature_date.date()} is missing "
                f"ticker rows: {sorted(missing_tickers)}"
            )

        train_X = train_df[features.feature_cols].to_numpy(dtype=np.float32)
        train_y = train_df["target"].to_numpy(dtype=np.float32)
        r_next_train = train_df["r_next"].to_numpy(dtype=np.float32)
        oos_X = oos_df[features.feature_cols].to_numpy(dtype=np.float32)

        model = model_constructor(**model_kwargs)
        model.fit(train_X, train_y)

        variance_in_sample_values = np.maximum(
            model.predict(train_X),
            1e-12,
        )
        ticker_oos_variances = np.maximum(
            model.predict(oos_X),
            1e-12,
        )

        # Long-format portfolio variance is later constructed ticker by ticker
        # in predict_variance_for_date(). This is only an interface-compatible
        # scalar summary.
        variance_oos = float(np.mean(ticker_oos_variances))
        mu_hat = float(np.mean(r_next_train))

        calendar_index = features.portfolio_df.sort_index().index
        next_date_by_feature_date = pd.Series(
            calendar_index[1:],
            index=calendar_index[:-1],
        )
        mapped_dates = train_df["Date"].map(next_date_by_feature_date)
        valid = mapped_dates.notna().to_numpy()

        residuals_df = create_residuals_dataframe(
            residual_dates=pd.DatetimeIndex(mapped_dates[valid]),
            realized_returns=r_next_train[valid],
            predicted_variances=variance_in_sample_values[valid],
            mu_hat=mu_hat,
        )

    else:
        clean_df = df.dropna(subset=["target", "r_next"])

        if len(clean_df) < 2:
            return None

        X = clean_df[features.feature_cols].to_numpy(dtype=np.float32)
        y = clean_df["target"].to_numpy(dtype=np.float32)
        r_next = clean_df["r_next"].to_numpy(dtype=np.float32)

        train_X = X[:-1]
        oos_row_X = X[[-1]]
        train_y = y[:-1]
        r_next_train = r_next[:-1]

        model = model_constructor(**model_kwargs)
        model.fit(train_X, train_y)

        variance_in_sample_values = np.maximum(
            model.predict(train_X),
            1e-12,
        )
        variance_oos = max(
            float(model.predict(oos_row_X)[0]),
            1e-12,
        )
        mu_hat = float(np.mean(r_next_train))
        residual_dates = clean_df.index[1:len(r_next_train) + 1]

        residuals_df = create_residuals_dataframe(
            residual_dates=residual_dates,
            realized_returns=r_next_train,
            predicted_variances=variance_in_sample_values,
            mu_hat=mu_hat,
        )

    return ForecastResult(
        model=model,
        model_name=model_name,
        predicted_oos_next_day_variance=variance_oos,
        mu_hat=mu_hat,
        residuals_in_sample=residuals_df,
        variance_in_sample=residuals_df["predicted_variance"],
        feature_cols=features.feature_cols,
    )


def diagnose_long_holdout(features: MLFeatureResult) -> pd.DataFrame:
    """
    Show which rows belong to the latest OOS feature date.

    None of these rows may enter model.fit().
    """
    if features.type != "long":
        raise ValueError("This diagnostic is only for long-format features.")

    oos_feature_date = pd.Timestamp(features.df["Date"].max())
    cols = ["Date", "ticker", "r", "r_next", "target"]
    return (
        features.df.loc[features.df["Date"] == oos_feature_date, cols]
        .sort_values("ticker")
        .reset_index(drop=True)
    )


# %%
def random_forest_forecast(features: MLFeatureResult, **rf_kwargs) -> ForecastResult | None:
    from sklearn.ensemble import RandomForestRegressor
    defaults = dict(n_estimators=200, max_depth=5, random_state=0)
    defaults.update(rf_kwargs)
    return _tree_forecast(features, RandomForestRegressor, model_name="random_forest", model_kwargs=defaults)


def xgboost_forecast(features: MLFeatureResult, **xgb_kwargs) -> ForecastResult | None:
    import xgboost as xgb
    defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, n_jobs=1)
    defaults.update(xgb_kwargs)
    return _tree_forecast(features, xgb.XGBRegressor, model_name="xgboost", model_kwargs=defaults)


# %%
# demo call -- commented out, see note above the lstm demo call
# random_forest_forecast(feature_df)

# %%
# demo call -- see note above
# xgboost_forecast(feature_df)


# %% [markdown]
# # Portfolio Covariance (EWMA / RiskMetrics) -- long format
#
# Long format (`build_ml_features_portfolio_long_format`) trains ONE
# `garch`/`xgboost`/`random_forest`/`lstm` model on the pooled, stacked
# per-ticker rows (larger effective sample than a single portfolio series).
# At prediction time (see `predict_variance_for_date` below), that ONE
# trained model is applied once per ticker to get each ticker's own
# predicted variance sigma_i^2 -- the DIAGONAL of the portfolio's covariance
# matrix. A single-ticker model can never predict the OFF-diagonal
# (cross-ticker covariance), so that comes from a RiskMetrics-style EWMA
# correlation matrix of the raw ticker returns instead:
#     Sigma_hat[i, j] = rho_ij * sqrt(sigma_i^2 * sigma_j^2),  i != j
#     Sigma_hat[i, i] = sigma_i^2                              (the ML forecast)
# Portfolio variance is then the full quadratic form
# `portfolio_variance = w @ Sigma_hat @ w` -- NEVER `sum(w_i * sigma_i^2)`,
# which silently drops every cross-term (correlation) and is systematically
# wrong for a correlated portfolio.

# %%
def ewma_covariance_matrix(
    returns: pd.DataFrame,
    lam: float = 0.94,
    min_periods: int = 30,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Rolling EWMA covariance matrix estimation, RiskMetrics-style.

    :param returns: DataFrame, index=date, columns=tickers. Each row r_t must
        already be realized (this function does NOT shift anything;
        caller is responsible for alignment).
    :param lam: decay factor (0.94 daily is the RiskMetrics default)
    :param min_periods: rows to burn in before covariance is considered valid
    :return:
        dict mapping date -> (n_assets, n_assets) covariance matrix.
        The matrix keyed at date t is Sigma_hat_t: it only uses
        information up to and including row t-1, so it is a valid one-step
        ahead forecast FOR date t. Do not key it as Sigma_{t-1} or you will
        misalign it against r_next later.
    """
    tickers = returns.columns
    n = len(tickers)
    dates = returns.index

    R = returns.to_numpy(dtype=np.float64)

    # seed with simple sample covariance over the first min_periods rows
    sigma = np.cov(R[:min_periods], rowvar=False)
    if n == 1:
        sigma = np.array([[sigma]])

    cov_by_date: dict[pd.Timestamp, np.ndarray] = {}

    for t in range(min_periods, len(R)):
        r_t = R[t - 1]  # last REALIZED return row, not R[t]
        outer = np.outer(r_t, r_t)
        sigma = lam * sigma + (1 - lam) * outer

        # this sigma uses info up to date t-1 (inclusive), so it is the forecast for dates[t]
        cov_by_date[dates[t]] = sigma.copy()

    return cov_by_date


def covariance_to_correlation(sigma: np.ndarray) -> np.ndarray:
    """Normalize a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.diag(sigma))
    d[d == 0] = np.nan
    corr = sigma / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def portfolio_variance_from_ewma(
    weights: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """w^T Sigma w, given a single date's covariance matrix."""
    return float(weights @ sigma @ weights)


# %% [markdown]
#

# %% [markdown]
# # Creating Var From Forecast

# %%
def calculate_empirical_var(
    standardized_residuals,
    mu_hat: float,
    predicted_variance: float,
    var_confidence: float,
) -> float:
    """Calculate empirical VaR in loss space from a standardized residual sample."""
    q = np.quantile(standardized_residuals, 1 - var_confidence)
    sigma_hat = np.sqrt(max(float(predicted_variance), 1e-12))
    predicted_return_quantile = float(mu_hat) + sigma_hat * q
    return -predicted_return_quantile


def empirical_var_from_forecast(forecast: ForecastResult, var_confidence: float) -> float:
    return calculate_empirical_var(
        standardized_residuals=forecast.residuals_in_sample["standardized_residual"],
        mu_hat=forecast.mu_hat,
        predicted_variance=forecast.predicted_oos_next_day_variance,
        var_confidence=var_confidence,
    )


# %% [markdown]
# # Wrapper for Creating Models with Window Data
#
#

# %%

MODEL_REGISTRY = {
    "garch": garch_forecast,
    "xgboost": xgboost_forecast,
    "random_forest": random_forest_forecast,
    "lstm": lstm_forecast,
}



# %% [markdown]
# # LSTM Forecast
#
# ## walk-forward logic from our wrapper
#
# * As we wanna try different window lengths for our models to see to test their performance, main wrapper function to call models have following parameters:
#
# * **window size** = e.g. 100
#     * how many days we will fit into the model
# * **periood_of_forecast** = e.g. 3
#     * how many days are we gonna predict with that data,
#
# If we have data for **300 days**, this applies:
#
# * Example;
#
# ```
# model1: create a new model fit a model with data of days [1..100], predict with that model VaR for day 101+102+103, compare with realised loss
#     * DATA OF DAYS [1-101], (n=window size +1) data is loaded by wrapper, as models drop last item, "out_of_sample_row"
#
# model 2: fit with [2..102], predict VaR for day 103+104+105, compare with realised loss
# model 3: fit with [3..103], predict day 104+..., compare with realised loss
# .
#
# model 199: fit with [199..299], predict day 300, compare with realised loss of day 300 (301,302 not defined)
#
# ```
#
#
# ## feature creation and lag windows
#
# * There are 2 seperate aggregation logics applied during the feature creations
#
# **feature_windows** = system adds  average return and volatility of those windows in dataframe defined here.
#     * standart = FEATURE_WINDOWS = (5, 10, 21)
#
# **sq_return_lags** = for tree models, squared returns with X day lags (of X day prior) defitinions can be done via this parameter
#     * thereby they can track/ recreate volatiliy clustering effects
#
# ## LSTM SPECIFIC sequence length
#
# **sequence_length** : LSTMs have internally the same "meta" walk-forward logic created for the framework.
#
# if seq_length = 3
#
# and
#
# * **window size** = e.g. 100
#     * how many days we will fit into the model
# * **periood_of_forecast** = e.g. 3
#     * how many days are we gonna predict with that data,
#
#
# ```
# * model1: create a new model fit a model with data of days [1..100], predict with that model VaR for day 101+102+103, compare with realised loss.
#     * Data for days[1-101] is loaded:
#     INSIDE THE LSTM:
#
#     # seq1: X: features[day1,day2,day3] y: target[day3], as the target of day3 is day4s value
#     # seq2: X: features[day2,day3,day4] y: target[day4], as the target of day4 is day5s value
#     ....
#     # seq98   X: features[day98,day99,day100] y: target[day100] as the target of day4 is day5s value
# ```
#
#
#

# %% [markdown]
# ### Creating multiday VaR Forecasts
#
# o create VaR forecast for out of sample days, for the time period defined with `period_of_forecast`,
# * for each day of prediction:
#     * system predicts "the day before" with the pretrained model
#     * updates the variance and mu_hat value
#     * creates residual for the day before
#     * recalcualtes the VaR with previously calculated residuals + new residual
#
#
# TO NOT LEAK THE FOLLOWING DAYS INFORMATION into residuals, for every iteration new residual is calculated
#

# %%
def build_lstm_sequence_for_date(
    forecast: ForecastResult,
    features_df: pd.DataFrame,
    feature_date: pd.Timestamp,
) -> torch.Tensor:
    """
    HELPER METHOD FOR `predict_variance_for_date_single` to feed data into LSTM

    Builds the normalized LSTM sequence ending on `feature_date`.
    """
    if forecast.sequence_length is None:
        raise ValueError("sequence_length is missing from ForecastResult.")

    if forecast.feature_mean is None or forecast.feature_std is None:
        raise ValueError(
            "LSTM normalization parameters are missing from ForecastResult."
        )

    position = features_df.index.get_loc(feature_date)

    if not isinstance(position, (int, np.integer)):
        raise ValueError("features_df must have a unique index.")

    start = position - forecast.sequence_length + 1

    if start < 0:
        raise ValueError(
            f"Not enough feature rows to build an LSTM sequence for {feature_date}."
        )

    sequence = (
        features_df
        .iloc[start:position + 1][forecast.feature_cols]
        .to_numpy(dtype=np.float32)
    )

    normalized = (
        sequence - forecast.feature_mean
    ) / forecast.feature_std

    device = next(forecast.model.parameters()).device

    return torch.tensor(
        normalized,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def predict_variance_for_date_single(
    forecast: ForecastResult,
    features_df: pd.DataFrame,
    feature_date: pd.Timestamp,
) -> float:
    """

    RETURNS THE VARIANCE PREDICTION RESULT OF pretrained model in forecast object for the feature_date


    :param forecast: FORECAST RESULT OBJECT WITH TRAINED MODEL
    :param features_df: created features for day single day "t", including day t+1s r_next value
    :param feature_date: single day to use at prediction
    :return: VARIANCE PREDICTION FOR DAY "T+1"
    """
    if feature_date not in features_df.index:
        raise KeyError(
            f"Feature date {feature_date} is missing from features_df."
        )

    feature_position = features_df.index.get_loc(feature_date)

    if not isinstance(feature_position, (int, np.integer)):
        raise ValueError("features_df must have a unique index.")

    target_position = feature_position + 1

    if target_position >= len(features_df):
        raise ValueError(
            f"No target date exists after feature date {feature_date}."
        )

    target_date = features_df.index[target_position]

    if forecast.model_name == "garch":
        # last prediction in forecast object:
        last_prediction_date = (
            forecast.residuals_in_sample
            .sort_index()
            .last_valid_index()
        )

        if last_prediction_date is None:
            raise ValueError("GARCH residual history is empty.")

        if last_prediction_date not in features_df.index:
            raise KeyError(
                f"Last GARCH residual date {last_prediction_date} "
                "is missing from features_df."
            )

        last_prediction_position = features_df.index.get_loc(
            last_prediction_date
        )

        if not isinstance(last_prediction_position, (int, np.integer)):
            raise ValueError("features_df must have a unique index.")

        # Number of observed trading rows between the last in-sample date
        # and the target date. This avoids counting weekends and holidays.
        horizon = target_position - last_prediction_position

        if horizon < 1:
            raise ValueError(
                f"Target date {target_date} must be after "
                f"the last prediction date {last_prediction_date}."
            )

        garch_fc = forecast.model.forecast(
            horizon=horizon,
            reindex=False,
        )

        # Last column corresponds to the requested forecast horizon.
        garch_variance_forecast_for_target_date = float(
            garch_fc.variance.values[-1, horizon - 1]
        )

        # Keep this conversion only when GARCH was trained on returns * 100.
        garch_variance_forecast_for_target_date /= 10_000.0

        return max(
            garch_variance_forecast_for_target_date,
            1e-12,
        )

    if forecast.model_name in {"xgboost", "random_forest"}:
        row = features_df.loc[[feature_date],forecast.feature_cols]

        return max(
            float(forecast.model.predict(row)[0]),
            1e-12,
        )

    if forecast.model_name == "lstm":
        sequence = build_lstm_sequence_for_date(
            forecast,
            features_df,
            feature_date,
        )

        forecast.model.eval()

        with torch.no_grad():
            predicted_scaled_variance = float(
                forecast.model(sequence).squeeze().item()
            )

        return max(
            predicted_scaled_variance / forecast.target_scale,
            1e-12,
        )

    raise ValueError(
        f"Unknown model_name: {forecast.model_name}"
    )


def predict_variance_for_date(
    forecast: ForecastResult,
    features_df: pd.DataFrame,
    feature_date: pd.Timestamp,
    features_type: str = "wide",
    ewma_lam: float = 0.94,
    ewma_min_periods: int = 30,
) -> float:
    """
    For "wide"/"single" features_type, calls `predict_variance_for_date_single`
    directly -- unchanged behavior.

    For "long" features_type, `forecast` is the ONE model fit on the pooled,
    stacked per-ticker training rows (e.g. `xgboost_forecast(long_features)`).
    That same model is applied once per ticker here via
    `predict_variance_for_date_single` to get each ticker's own predicted
    variance sigma_i^2 -- the DIAGONAL of the portfolio's covariance matrix.
    A single-ticker prediction can never carry cross-ticker covariance, so
    the OFF-diagonal comes from a RiskMetrics EWMA correlation matrix of the
    raw ticker returns up to `feature_date`:
        Sigma_hat[i, j] = rho_ij * sqrt(sigma_i^2 * sigma_j^2),  i != j
        Sigma_hat[i, i] = sigma_i^2                              (ML forecast)
    Portfolio variance is the full quadratic form
    `w @ Sigma_hat @ w` -- NEVER a naive `sum(w_i * sigma_i^2)`, which drops
    every cross-term (correlation) and is systematically wrong.

    :param features_df: for "long", the RAW long dataframe (Date/ticker/r/...
        columns, as built by `build_ml_features_portfolio_long_format`) --
        NOT a pre-pivoted frame.
    """
    if features_type != "long":
        return predict_variance_for_date_single(forecast, features_df, feature_date)

    tickers = list(TICKERS)
    weights = np.array([WEIGHTS.get(t, 0.0) for t in tickers], dtype=float)

    # diagonal: the SAME pretrained (pooled) model, applied once per ticker
    ticker_variances = []
    for ticker in tickers:
        ticker_df = (
            features_df.loc[features_df["ticker"] == ticker]
            .set_index("Date")
            .sort_index()
        )
        ticker_variances.append(
            predict_variance_for_date_single(forecast, ticker_df, feature_date)
        )
    sigma_diag = np.array(ticker_variances, dtype=float)

    # off-diagonal: EWMA correlation of raw ticker returns, up to feature_date
    returns_wide = (
        features_df
        .pivot(index="Date", columns="ticker", values="r")[tickers]
        .dropna()
        .sort_index()
    )
    returns_upto = returns_wide.loc[:feature_date]

    cov_by_date = ewma_covariance_matrix(returns_upto, lam=ewma_lam, min_periods=ewma_min_periods)
    if not cov_by_date:
        raise ValueError(f"Not enough EWMA correlation history through {feature_date}.")
    corr = covariance_to_correlation(cov_by_date[max(cov_by_date)])

    sigma_hat = corr * np.sqrt(np.outer(sigma_diag, sigma_diag))
    np.fill_diagonal(sigma_hat, sigma_diag)

    return max(portfolio_variance_from_ewma(weights, sigma_hat), 1e-12)


def _portfolio_calendar(features: MLFeatureResult) -> pd.DataFrame:
    """Return the canonical one-row-per-trading-day portfolio frame."""
    calendar = features.portfolio_df.sort_index()

    required = {"r", "r_next", "target"}
    missing = required.difference(calendar.columns)
    if missing:
        raise ValueError(
            f"MLFeatureResult.portfolio_df is missing columns: {sorted(missing)}"
        )

    if not calendar.index.is_unique:
        raise ValueError("MLFeatureResult.portfolio_df index must be unique.")

    return calendar


def _long_format_in_sample_residuals(
    forecast: ForecastResult,
    raw_features_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    training_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Rebuilds a PORTFOLIO-level in-sample residual history for a model fit on
    pooled long-format rows. `forecast.residuals_in_sample` can NOT be used
    for this directly: it's built by the underlying forecaster (e.g.
    `_tree_forecast`) straight off the long dataframe's own "r_next" column,
    which is each ROW's own TICKER's next-day return (long format has no
    PORTFOLIO row at all) -- and dated by the long dataframe's meaningless
    RangeIndex, which `create_residuals_dataframe` silently reinterprets as
    nanoseconds-since-epoch. Sorting/tailing that gives a residual sample
    dominated by whichever ticker sorts last, not the portfolio.

    Instead, walks `training_dates` (the SAME calendar window the model was
    fit on) day by day using the identical `predict_variance_for_date` +
    `calendar_df["r_next"]` machinery the walk-forward loop itself uses, so
    the seed is self-consistent with every later step. Mirrors
    `_tree_forecast`'s own "exclude the window's last row's r_next" rule
    (`train_y = y[:-1]`) -- training_dates[-1]'s r_next is the return
    realized on the first FORECAST day, not yet known at seed time.
    """
    mu_hat = float(calendar_df.loc[training_dates[:-1], "r_next"].mean())

    residual_dates, actual_returns, predicted_variances = [], [], []

    for i in range(1, len(training_dates) - 1):
        feature_date = training_dates[i - 1]
        target_date = training_dates[i]

        actual_return = float(calendar_df.loc[feature_date, "r_next"])
        if not np.isfinite(actual_return):
            continue

        predicted_variance = predict_variance_for_date(
            forecast=forecast,
            features_df=raw_features_df,
            feature_date=feature_date,
            features_type="long",
        )

        residual_dates.append(target_date)
        actual_returns.append(actual_return)
        predicted_variances.append(max(float(predicted_variance), 1e-12))

    return create_residuals_dataframe(
        residual_dates=residual_dates,
        realized_returns=actual_returns,
        predicted_variances=predicted_variances,
        mu_hat=mu_hat,
    )


def rolling_var_forecasting_from_pretrained_model(
    forecast: ForecastResult,
    full_features: MLFeatureResult,
    forecast_dates: pd.DatetimeIndex,
    window_size: int,
    var_confidence: float,
) -> pd.DataFrame:
    """
    Generate sequential multi-day VaR forecasts using a frozen pretrained model.

    For each forecast date:
    1. Predict variance using information available before the target date.
    2. Calculate VaR using the existing residual history.
    3. Observe the realized target-date return.
    4. Calculate and append the new standardized residual.
    """

    raw_features_df = full_features.df  # passed to predict_variance_for_date as-is (may be long/stacked)
    features_df = _portfolio_calendar(full_features)
    forecast_dates = pd.DatetimeIndex(forecast_dates)

    if len(forecast_dates) == 0:
        return pd.DataFrame(columns=["varForecast"])

    missing_dates = forecast_dates.difference(features_df.index)

    if len(missing_dates):
        raise KeyError(
            f"Forecast dates missing from features: {list(missing_dates)}"
        )


    # MAIN DATA OBJECT IN FUNCTION
    # Residuals that were already known at the end of training.

    if full_features.type == "long":
        # forecast.residuals_in_sample is NOT usable here -- see
        # _long_format_in_sample_residuals's docstring. Rebuild the seed from
        # the training window's own dates, PORTFOLIO-return-based, using the
        # same machinery the walk-forward loop below uses.
        first_forecast_pos = features_df.index.get_loc(forecast_dates[0])
        training_dates = features_df.index[max(0, first_forecast_pos - window_size):first_forecast_pos]
        residual_bank = _long_format_in_sample_residuals(
            forecast=forecast,
            raw_features_df=raw_features_df,
            calendar_df=features_df,
            training_dates=training_dates,
        ).tail(window_size)
    else:
        residual_bank = (
            forecast.residuals_in_sample
            .sort_index()
            .tail(window_size)
            .copy()
        )

    if residual_bank.empty:
        raise ValueError("No in-sample residuals are available.")

    mu_hat = float(
        residual_bank["actual_return"].mean()
    )

    var_rows = {}

    # LOOP
    # for every day in forecast_dates
    for target_date in forecast_dates:
        target_position = features_df.index.get_loc(target_date)

        if not isinstance(target_position, (int, np.integer)):
            raise ValueError(
                f"Feature date is not unique: {target_date}"
            )

        if target_position == 0:
            raise ValueError(
                f"No previous feature row exists for {target_date}."
            )

        # feature date, which we use for predictions is the ONE INDEX BEFORE
        feature_date_one_day_before = features_df.index[
            target_position - 1
        ]


        ######## CALCULATING VaR EXPECTATION FOR TARGET DATE "T" ############
        # R_NEXT IS NOT REALIZED HERE YET!!!!
        # WE USE THE CALCULATED VARIANCE FOR "T" USING DATA/ RESIDUALS TILL "T-1"


        # calling the trained model
        predicted_variance = predict_variance_for_date(
            forecast=forecast,
            features_df=raw_features_df,
            feature_date=feature_date_one_day_before,
            features_type=full_features.type,
        )

        predicted_variance = max(float(predicted_variance),1e-12)

        # calculate var value with calculate_empirical_var
        var_rows[target_date] = calculate_empirical_var(
            standardized_residuals=(
                residual_bank["standardized_residual"]
            ),
            mu_hat=mu_hat,
            predicted_variance=predicted_variance,
            var_confidence=var_confidence,
        )

         ######## NOW R_NEXT IS REALIZED, TARGET DATE IS OVER ############

        # get target dates return data, from previous days r_next
        actual_return = float(
            features_df.loc[feature_date_one_day_before,"r_next", ]
        )

        if not np.isfinite(actual_return):
            raise ValueError(
                f"Missing realized return for target date {target_date}."
            )

        #calculate standart deviation from models output/prediction
        predicted_sigma = np.sqrt(predicted_variance)

        # calculate the new residual,
        # USING THE UNUPDATED MU HAT/AVERAGE
        # IN EVERY MODEL WE USED R_NEXT, WHICH LACKED THE FINAL OUT OF SAMPLE DAYS R_NEXT,
        # IT WASNT INCLUDED IN MU_HAT VALUE
        new_residual = (
            actual_return - mu_hat
        ) / predicted_sigma

        new_residual_row = pd.DataFrame(
            {
                "actual_return": [actual_return],
                "predicted_variance": [predicted_variance],
                "predicted_sigma": [predicted_sigma],
                "mu_hat": [mu_hat],
                "standardized_residual": [new_residual],
            },
            index=pd.DatetimeIndex(
                [target_date],
                name="Date",
            ),
        )


        # ADD TO RESIDUALS COLLECTION THE NEW CALCULATED RESIDUAL, FOR NEXT LOOP
        residual_bank = (
            pd.concat(
                [
                    residual_bank.drop(
                        index=target_date,
                        errors="ignore",
                    ),
                    new_residual_row,
                ]
            )
            .sort_index()
            .tail(window_size)
        )

        # AFTER RESIDUAL CALCULATION, UPDATE THE MU HAT WITH THE REALIZED RETURN OF DAY T FOR NEXT LOOP
        mu_hat = float(residual_bank["actual_return"].mean())

    return (
        pd.Series(var_rows,name="varForecast",).sort_index().to_frame()
    )


def NON_ROLLING_TRAINER_single_model_multi_day_var(
    features: MLFeatureResult,
    model_name: str,
    window_size: int = 250,
    period_of_forecast: int = 20,
    var_confidence: float = 0.95,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Fit once, then produce `period_of_forecast` sequential VaR forecasts.

    Returns a single DataFrame indexed by date, with columns "varForecast"
    and "loss" (the realized loss for that same date).
    """
    model_kwargs = model_kwargs or {}
    forecaster = MODEL_REGISTRY[model_name]
    full_df = features.df
    calendar_df = _portfolio_calendar(features)

    if len(calendar_df) <= window_size + period_of_forecast:
        raise ValueError(
            "Not enough history for window_size and period_of_forecast."
        )


    #start from the LAST WINDOW possible
    first_target_pos = len(calendar_df) - period_of_forecast
    training_end_pos = first_target_pos - 1
    training_start_pos = training_end_pos - window_size + 1

    training_start_date = calendar_df.index[training_start_pos]
    training_end_date = calendar_df.index[training_end_pos]

    if features.type == "long":
        training_df = full_df[full_df["Date"].between(training_start_date, training_end_date)]
    else:
        training_df = full_df.loc[training_start_date:training_end_date]

    training_features = MLFeatureResult(
        feature_cols=features.feature_cols,
        df=training_df,
        type=features.type,
        portfolio_df=calendar_df.loc[training_start_date:training_end_date],
    )

    forecast = forecaster(training_features,**model_kwargs,)

    if forecast is None:
        raise RuntimeError(
            f"{model_name} could not be fitted."
        )

    forecast_dates = calendar_df.index[
        first_target_pos:first_target_pos + period_of_forecast
    ]

    var_df = rolling_var_forecasting_from_pretrained_model(
        forecast=forecast,
        full_features=features,
        forecast_dates=forecast_dates,
        window_size=window_size,
        var_confidence=var_confidence,
    )

    realized_returns = []

    for target_date in var_df.index:
        target_position = calendar_df.index.get_loc(target_date)

        if not isinstance(target_position, (int, np.integer)):
            raise ValueError(
                f"Feature date is not unique: {target_date}"
            )

        if target_position == 0:
            raise ValueError(
                f"No previous feature row exists for {target_date}."
            )

        feature_date_one_day_before = calendar_df.index[
            target_position - 1
        ]

        realized_returns.append(
            float(
                calendar_df.loc[
                    feature_date_one_day_before,
                    "r_next",
                ]
            )
        )

    loss_df = pd.DataFrame(
        {
            "loss": -np.asarray(
                realized_returns,
                dtype=float,
            )
        },
        index=var_df.index,
    )

    return var_df.join(loss_df)


# %%
def ROLLING_TRAINER_single_model_multi_day_var_FOR_NEXT_DAY(
    features: MLFeatureResult,
    model_name: str,
    window_size: int = 250,
    period_of_forecast: int = 252,
    var_confidence: float = 0.95,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Refit the model for every forecast date and produce a one-day-ahead
    portfolio VaR. Window sizes are always measured in trading dates, even
    when the model features are stored in long format.
    """
    model_kwargs = model_kwargs or {}
    forecaster = MODEL_REGISTRY[model_name]

    full_df = features.df
    calendar_df = _portfolio_calendar(features)
    n_dates = len(calendar_df)

    period_of_forecast = min(
        period_of_forecast,
        n_dates - window_size - 1,
    )
    if period_of_forecast <= 0:
        raise ValueError(
            "Not enough history for the requested window_size/period_of_forecast."
        )

    start_target_pos = n_dates - period_of_forecast
    var_rows, loss_rows = {}, {}

    for target_pos in range(start_target_pos, n_dates):
        target_date = calendar_df.index[target_pos]
        feature_date = calendar_df.index[target_pos - 1]

        training_end_pos = target_pos - 1
        training_start_pos = training_end_pos - window_size + 1

        training_start_date = calendar_df.index[training_start_pos]
        training_end_date = calendar_df.index[training_end_pos]

        if features.type == "long":
            window_df = full_df[
                full_df["Date"].between(training_start_date, training_end_date)
            ]
        else:
            window_df = full_df.loc[training_start_date:training_end_date]

        window_features = MLFeatureResult(
            feature_cols=features.feature_cols,
            df=window_df,
            type=features.type,
            portfolio_df=calendar_df.loc[training_start_date:training_end_date],
        )

        forecast = forecaster(window_features, **model_kwargs)
        if forecast is None:
            continue

        if features.type == "long":
            residual_bank = _long_format_in_sample_residuals(
                forecast=forecast,
                raw_features_df=full_df,
                calendar_df=calendar_df,
                training_dates=calendar_df.index[
                    training_start_pos:training_end_pos + 1
                ],
            ).tail(window_size)

            if residual_bank.empty:
                continue

            predicted_variance = predict_variance_for_date(
                forecast=forecast,
                features_df=full_df,
                feature_date=feature_date,
                features_type="long",
            )

            var_loss = calculate_empirical_var(
                standardized_residuals=residual_bank["standardized_residual"],
                mu_hat=float(residual_bank["actual_return"].mean()),
                predicted_variance=predicted_variance,
                var_confidence=var_confidence,
            )
        else:
            var_loss = empirical_var_from_forecast(
                forecast,
                var_confidence,
            )

        actual_return = float(calendar_df.loc[target_date, "r"])
        if not np.isfinite(actual_return):
            continue

        var_rows[target_date] = float(var_loss)
        loss_rows[target_date] = -actual_return

    return (
        pd.DataFrame(
            {
                "varForecast": pd.Series(var_rows, dtype=float),
                "loss": pd.Series(loss_rows, dtype=float),
            }
        )
        .sort_index()
    )


# %%
# demo call -- commented out (252 refits, ~seconds), see note above the lstm demo call
# ROLLING_TRAINER_single_model_multi_day_var_FOR_NEXT_DAY(feature_df,"xgboost")

# %%
def ROLLING_TRAINER_single_model_multi_day_FOR_FORECAST_LENGTH(
    features: MLFeatureResult,
    model_name: str,
    training_window: int = 250,
    forecast_period: int = 252,
    var_confidence: float = 0.95,
    model_kwargs: dict | None = None,
    forecast_start: str | pd.Timestamp | None = None,
    forecast_end: str | pd.Timestamp | None = None,
) -> list[dict]:
    """
    Rolls a (fit once, forecast `forecast_period` days) window forward ONE DAY AT A
    TIME across `features.df`: fit on `training_window` days starting at day i,
    forecast the following `forecast_period` days from that single frozen fit (via
    `rolling_var_forecasting_from_pretrained_model` -- no retraining within a
    window), package the forecast with the realized returns, then advance to day
    i+1 and repeat. Windows therefore overlap heavily with their neighbors (each
    shares `forecast_period - 1` of its `forecast_period` forecasted days with the
    next window) -- this is by design, to get a forecast for every possible
    `training_window`-sized slice of history, not just a non-overlapping tiling of
    it. A window whose fit fails entirely (e.g. GARCH non-convergence) is skipped,
    not fatal -- the remaining windows are still returned. Trailing windows that
    don't have enough remaining real data to pair a full `forecast_period`-day
    forecast with are dropped, not padded/truncated.

    `forecast_start`/`forecast_end` (optional) narrow which windows get produced,
    down to only those whose FIRST forecasted day falls within
    `[forecast_start, forecast_end]` (inclusive; either bound can be left `None`).
    Left as `None` (the default), every possible window is produced: starting from
    `features.df`'s very first `training_window` days, through to wherever there's
    still enough remaining history for a full `forecast_period`-day-ahead forecast.

    Counterpart of `NON_ROLLING_TRAINER_single_model_multi_day_var`, which does only
    ONE such window (the last one, at the end of the dataset).

    Returns a list of dicts, one per window (the per-window `forecast_df`s are NOT
    concatenated across windows -- each window's `forecast_period` days overlap
    with its neighbors', so merging them into one date-indexed frame would
    silently produce duplicate/competing forecasts for the same calendar date):
        {"window_start": <training window's first date>,
         "window_end": <training window's last date>,
         "forecast_df": <DataFrame indexed by date, "varForecast" + "loss"
             columns, one row per forecasted day>}
    See `run_full_backtest_suite_all_windows` in `var_backtest_utils.py` for
    backtesting this list (each window's own `forecast_period` days are a valid,
    non-overlapping-*within-itself* daily sequence, so the full test suite -- Kupiec
    TUFF/Haas TBF/Christoffersen included -- stays meaningful per window; only
    pooling windows together needs to account for the overlap).
    """
    model_kwargs = model_kwargs or {}
    forecaster = MODEL_REGISTRY[model_name]
    full_df = features.df
    calendar_df = _portfolio_calendar(features)
    n = len(calendar_df)

    if n <= training_window + forecast_period:
        raise ValueError("Not enough history for even a single training_window/forecast_period window.")

    # bound which windows get produced (by the date of each window's FIRST
    # forecasted day) -- defaults to every window the available history allows
    first_target_pos_min = training_window
    first_target_pos_max = n - forecast_period

    if forecast_start is not None:
        first_target_pos_min = max(
            first_target_pos_min,
            calendar_df.index.searchsorted(pd.Timestamp(forecast_start)),
        )

    if forecast_end is not None:
        first_target_pos_max = min(
            first_target_pos_max,
            calendar_df.index.searchsorted(pd.Timestamp(forecast_end), side="right") - 1,
        )

    windows = []

    # LOOP
    # slide a (training_window, forecast_period) window forward ONE DAY at a time
    for first_target_pos in range(first_target_pos_min, first_target_pos_max + 1):
        training_end_pos = first_target_pos - 1
        training_start_pos = training_end_pos - training_window + 1

        training_start_date = calendar_df.index[training_start_pos]
        training_end_date = calendar_df.index[training_end_pos]

        if features.type == "long":
            training_df = full_df[full_df["Date"].between(training_start_date, training_end_date)]
        else:
            training_df = full_df.loc[training_start_date:training_end_date]

        training_features = MLFeatureResult(
            feature_cols=features.feature_cols,
            df=training_df,
            type=features.type,
        )

        forecast = forecaster(training_features, **model_kwargs)

        if forecast is None:
            # this window's fit failed entirely (e.g. GARCH convergence) -- skip it
            continue

        forecast_dates = calendar_df.index[first_target_pos:first_target_pos + forecast_period]

        var_df = rolling_var_forecasting_from_pretrained_model(
            forecast=forecast,
            full_features=features,
            forecast_dates=forecast_dates,
            window_size=training_window,
            var_confidence=var_confidence,
        )

        realized_returns = []
        for target_date in var_df.index:
            target_position = calendar_df.index.get_loc(target_date)
            feature_date_one_day_before = calendar_df.index[target_position - 1]
            realized_returns.append(float(calendar_df.loc[feature_date_one_day_before, "r_next"]))

        loss_df = pd.DataFrame(
            {"loss": -np.asarray(realized_returns, dtype=float)},
            index=var_df.index,
        )

        windows.append({
            "window_start": training_start_date,
            "window_end": training_end_date,
            "forecast_df": var_df.join(loss_df),
        })

    return windows
