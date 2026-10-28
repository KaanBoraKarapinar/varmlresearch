import streamlit as st

st.set_page_config(page_title="VaR Framework Docs", layout="wide")


# ---------------------------------------------------------------- sections
def overview():
    st.header("Overview")
    st.markdown(
        """
The framework forecasts portfolio **Value at Risk (VaR)** with filtered historical simulation (FHS):

1. A model predicts the **next-day portfolio variance** (tree models, GARCH or pure EWMA).
2. Realized returns are divided by the predicted volatility to get **standardized residuals**, kept in a `ResidualBank`.
3. The `(1 - confidence)` quantile of recent residuals, rescaled by the new variance forecast, is the VaR.
4. A walk-forward experiment repeats this over many train/forecast windows and a backtest suite judges the result.

| Building block | Role |
|---|---|
| `GroundTruthCalendar` | Single source of truth for dates and realized prices, prevents look-ahead bias |
| `ResidualBank` | Stores standardized residuals, turns them into VaR |
| `EwmaVarianceBank` | Combines per-ticker model variances with EWMA correlations into a portfolio variance |
| Feature blocks / targets | Define model inputs and what the model learns |
| Models | XGBoost-style trees, GARCH (baseline), pure EWMA |
| Walk-forward + backtest suite | Out-of-sample evaluation over many windows |
"""
    )


def calendar():
    st.header("Calendar (GroundTruthCalendar)")
    st.markdown(
        """
The calendar exists to **prevent look-ahead bias** and to be the one object that holds the realized end-of-day prices.
Only trading days are stored, so a forecast looks at the previous **index** (bar), not at "day - 1".
"""
    )
    st.code(
        """calendar = dm.DailyGroundTruthCalendar(
    df, TICKERS, WEIGHTS,
    train_start="2024-06-03", train_end="2024-11-29",
    forecast_start="2024-12-02", forecast_end="2024-12-31",
)""",
        language="python",
    )
    st.subheader("Warm-up bars")
    st.markdown(
        "Extra bars before `train_start` can be included. They are used only to compute rolling functions "
        "(feature burn-in, first EWMA covariance), so the first training rows are not thrown away. "
        "Their own returns never appear in the data. Default is 0."
    )
    st.subheader("Core rule: own-day labelling")
    st.info(
        "In the banks, the date of a row is the **outcome day**. The row `Date = X` holds the return **realized on X** "
        "and the variance forecast made from the features of the **bar before X**."
    )
    st.subheader("Dates")
    st.markdown(
        """
| Term | Meaning |
|---|---|
| `TrainingFeature.feature_dates` | Dates of feature rows with a known target, after feature warm-up is dropped |
| `ForecastFeature.feature_dates` | Dates the forecast feature matrix covers (last train bar plus all forecast days) |
| `calendar.forecast_dates` | All bars from `forecast_start` to `forecast_end` (no holidays or weekends) |
| `calendar.forecast_pairs` | Each forecast date paired with the next one, since a forecast on day d prices day d+1 |
| `train_dates_past_burn_in(candidates, burn_in_bars)` | Drops the first `burn_in_bars` of the calendar, returns the candidates that remain |
"""
    )
    st.caption("Example: calendar [1..8], burn_in_bars = 3, candidates [2, 4, 5, 8] gives [4, 5, 8].")
    st.subheader("Data")
    st.markdown(
        """
| Attribute | Content |
|---|---|
| `train_data` | Data between `train_start` and `train_end` |
| `forecast_data` | Data of the forecast period only (does not carry the last train bar) |
| `train_data_with_warmup` | Warm-up plus train data |
| `DANGEROUS_merged_warmedup_FORECAST_data` | Warm-up + train + forecast, used only when building `ForecastFeature` |
"""
    )
    st.subheader("Functions")
    st.markdown(
        """
- `var_from_residual_bank_X_period`: VaR for the train, forecast or full period.
- `ewma_realized_variance_bank(ewma_lam=0.94, ewma_min_periods=30)`: variance/covariance computed from **realized returns**, not from model forecasts. Seeded with warm-up or training data, then updated with EWMA.
"""
    )


def residual_bank():
    st.header("Residual Bank")
    st.latex(
        r"z_t = \frac{r_t - \hat{\mu}}{\sqrt{\hat{\sigma}^2_t}}"
    )
    st.markdown(
        """
In filtered historical simulation, quantiles of the standardized residuals are the proxy for the VaR. After training,
the residuals of the training period are available in the model's `initial_residual_bank`.
"""
    )
    st.code(
        """residual_bank = dm.ResidualBank(
    mu_mode="constant",          # "constant" or "rolling"
    default_mu_hat=0.0,
    rolling_mu_hat_window_size=50,   # only read when mu_mode="rolling"
)""",
        language="python",
    )
    st.markdown(
        """
- **constant** (default): every row uses `default_mu_hat`.
- **rolling**: `mu_hat_t` is the mean of the previous N returns, excluding day t itself (no look-ahead).
"""
    )
    st.subheader("Columns")
    st.markdown(
        """
| Column | Meaning |
|---|---|
| `actual_return` | Realized portfolio return of that day |
| `predicted_variance` | Portfolio variance forecast (from the previous bar's features) |
| `predicted_sigma` | `sqrt(predicted_variance)` |
| `mu_hat` | Mean return estimate used that day (default 0) |
| `standardized_residual` | `(actual_return - mu_hat) / predicted_sigma` |
"""
    )
    st.subheader("Lifecycle")
    st.markdown(
        "A bank is **single use**: once `trained = True` it cannot be trained again (`ValueError`), so rows of one "
        "training never leak into the next. For forecasting the bank is **copied**, so the model's own bank stays in its training state. "
        "Dates must be chronological and without duplicates."
    )
    st.code(
        """ResidualBank(...)                 trained = False
  | start_training(...)          trained = True
  | training_safe_bulk_insert()  96 training rows
  | .copy()                      model bank stays as trained
  | general_bulk_insert()        +21 forecast rows -> 117
  | multi_day_var(...)           VaR""",
        language="text",
    )
    st.subheader("Writing residuals (rolling mu_hat safety)")
    st.markdown(
        """
| Function | What it does |
|---|---|
| `DANGEROUS_add_residual` | Writes one row. Takes `mu_hat` from outside, so look-ahead is possible. Skips non-finite results with a warning |
| `step_1_add_residual_with_prev_stateful_mu` | Calls the function above with the **previous** stateful `mu_hat`, so today's return is not yet inside it |
| `step_2_change_stateful_mu` | Runs after step 1, updates `mu_hat` with the last N bars (today included) for the next step 1 |
| `_bootstrap_sequential_prefix` | Step 1 + step 2 per date. Slow, used for the first N bars with too little history |
| `general_bulk_insert` | Fast vectorized insert (`rolling(N).mean().shift(1)` for rolling mode), appends rows |
| `training_safe_bulk_insert` | Uses bootstrap for the first N bars, then bulk insert. Same result, faster |
"""
    )
    st.subheader("Reading the VaR")
    st.markdown(
        "`multi_day_var(dates, var_confidence, window, min_periods)` takes, for each date, the `(1 - confidence)` quantile "
        "of the previous N residuals (with `shift(1)`) and returns `VaR = -(mu_hat + sigma * q)`. Output columns: "
        "`varForecast` and `n_residuals_used`. Insufficient history raises `ValueError`."
    )
    st.markdown(
        """
| Setting | Meaning |
|---|---|
| `window = N` | At most the last N residuals (**rolling**) |
| `window = None` | All residuals up to that day (**expanding**) |
| `min_periods` | Minimum residuals needed for a VaR (default 30 for expanding, N for rolling) |

**Rolling**: recent data weighs more. Useful as an upper bound, e.g. `window=180, min_periods=30`.

**Expanding**: the early forecast days rest entirely on training residuals, which are in-sample and can be smaller than real
out-of-sample errors, so the quantile may be too narrow. A short window (N=20) forgets this within a month, a long or expanding window reacts slowly.
"""
    )
    st.markdown(
        "`empirical_var_for_target(...)` computes the VaR of a **single day** for a new `predicted_variance`."
    )
    st.subheader("Two windows, do not mix them up")
    st.markdown(
        """
| Parameter | Used when | Effective when | If changed |
|---|---|---|---|
| `rolling_mu_hat_window_size` | Creating `ResidualBank` | Rows are written (`mu_mode="rolling"`) | Retraining needed |
| `multi_day_var_calculation_window_size` | Reading VaR | VaR is read | Nothing needed |
"""
    )


def ewma_bank():
    st.header("EWMA Variance Bank")
    st.warning("The EWMA formula assumes a zero mean return. Keep this in mind for VaR.")
    st.markdown(
        """
In the long format the model predicts one variance per ticker. To get a **portfolio** variance that reacts quickly to shocks,
the framework uses EWMA correlations instead of a plain weighted average:

- Correlations between tickers come from **realized returns** (EWMA).
- The diagonal (variances) comes from the **ML model's forecasts**.
"""
    )
    st.latex(r"\sigma_p^2 = w^\top \Sigma\, w")
    st.latex(r"\Sigma_t^{EWMA} = \lambda\, \Sigma_{t-1}^{EWMA} + (1-\lambda)\, r_{t-1} r_{t-1}^{\top}")
    st.code(
        """ewma_bank = dm.EwmaVarianceBank(
    ewma_lam=0.94,        # RiskMetrics decay, larger = forgets slower
    ewma_min_periods=30,  # first N train bars seed the covariance
)""",
        language="python",
    )
    st.subheader("Data points")
    st.markdown(
        """
| Field | Content |
|---|---|
| `portfolio_variance` | Daily portfolio variances, the bank's main output (goes to `ResidualBank` as `predicted_variance`) |
| `cov_by_date` | Covariance matrix of every day |
| `sigma_diag` | Per-ticker variance forecasts of the model (diagonal) |
| `_current_sigma` | Current covariance matrix, tomorrow's is computed from it |
| `_prev_return` | Yesterday's returns, needed for the next update |
"""
    )
    st.code(
        """EwmaVarianceBank(ewma_lam, ewma_min_periods)   trained = False
  | start_training(tickers, weights)         trained = True
  | training_safe_bulk_insert(first 30 bars) initial covariance
  | general_bulk_insert(...)                 training rows + portfolio variances
  | .copy() -> general_bulk_insert(...)      forecast period""",
        language="text",
    )
    st.markdown(
        "`training_safe_bulk_insert(warmup_returns)` builds **one** standard covariance matrix from the first `ewma_min_periods` bars "
        "(default 30) and stores it in `current_sigma`. `general_bulk_insert` can take the diagonal from outside or compute it itself."
    )


def features():
    st.header("Features, Targets and Feature Blocks")
    st.subheader("Default features")
    st.markdown(
        """
| Feature | Computation | Captures |
|---|---|---|
| `r` | Return of the day | Latest shock |
| `abs_r` | Absolute return | Shock size |
| `real_vol_w` | Std of last w returns (w = 5, 10, 21) | Recent volatility |
| `mean_abs_w` | Mean absolute return over w days | Robust volatility |
| `skew_w` | Skewness over w days | Asymmetry of falls and rises |
| `ewma_vol` | Exponentially weighted std (span 21) | Volatility with more weight on recent days |
| `r2_lag_k` | Squared return k days ago (optional) | Volatility clustering (GARCH style) |

| Parameter | Effect |
|---|---|
| `windows=(5,10,21)` | 3 features per window. The largest window costs that many bars minus one (21 gives 20 bars lost) |
| `ewma_span=21` | Speed of `ewma_vol` and `market_ewma_vol` |
| `sq_return_lags=None` | e.g. `(1,2,3)` adds lagged squared returns, more bars lost |
| `target_scale=1.0` | Multiplies the target only |
"""
    )
    st.markdown(
        "**Long format**: one row per (day, ticker), target is that ticker's next-day squared return. "
        "**Wide format**: one row per day, target is the portfolio's next-day squared return."
    )
    st.subheader("Target scaling")
    st.markdown(
        "Squared percent returns are tiny, so models struggle to learn them. The target is multiplied by a scale, stored in "
        "`TrainingFeature.target_scale`, and divided back after prediction."
    )
    st.subheader("TrainingFeature fields")
    st.markdown(
        """
| Field | Meaning |
|---|---|
| `X_in_sample`, `y_in_sample` | Rows with known target and their scaled targets |
| `X_out_sample_last_index` | Last train day, no target |
| `X_in_sample_id_cols` | `Date` and `ticker` per row (long only) |
| `DANGEROUS_r_next_in_sample` | Next-day real return (unscaled, not squared), for evaluation only |
| `base_variance` | Base variance for targets that need one |
"""
    )
    st.subheader("Example dates")
    st.markdown(
        """
| Item | Range | Bars |
|---|---|---|
| Train | 2024-06-03 to 2024-11-29 | 126 |
| Forecast | 2024-12-02 to 2024-12-31 | 21 |
| Training `feature_dates` (after 20-bar warm-up) | 2024-07-02 to 2024-11-27 | 105 |
| Forecast `feature_dates` (last train bar + forecast) | 2024-11-29 to 2024-12-31 | 22 |
"""
    )
    st.subheader("Targets")
    st.markdown(
        "A target is a class that holds a **formula**, not data. It defines how the model's output turns into a variance."
    )
    st.markdown(
        """
| Target | Model learns `y` | Back to variance | Needs EWMA base |
|---|---|---|---|
| `SquaredReturn` | `r^2 * scale` | `pred / scale` | no |
| `RatioToBase` | `min(r^2 / EWMA var, clip) * scale` | `(pred / scale) * EWMA var` | **yes** (`base_lam=0.94`, `min_periods=30`, `clip=None`) |
| `LogVariance` | `log(r^2 + eps) * scale` | `exp(pred / scale)` | no (`eps=1e-8`) |
| `AbsReturn` | `abs(r) * scale` | `(pred / scale)^2` | no |

`forward_make_y(r_next, base_variance)` turns the next return into the learning target, `inverse_to_variance(prediction, base_variance)` is its inverse.
`clip` caps the target on extremely volatile days.
"""
    )
    st.subheader("Base variance idea")
    st.markdown("If the mean return is about 0, then `r_{f+1}^2` is a noisy observation of the true variance. Defining")
    st.latex(r"y_f = \frac{r_{f+1}^2}{\hat{\sigma}^2_{f+1,\,base}}")
    st.latex(r"\mathbb{E}[y_f \mid \mathcal{F}_f] = \frac{\sigma_{f+1}^2}{\hat{\sigma}^2_{f+1,\,base}} = m_f")
    st.markdown(
        "the model learns **how many times larger the true variance is than the EWMA variance** (`m_f`). If `m_f = 1`, the EWMA is already right. "
        "`emwa_base_variance` copies the EWMA bank logic; for targets with a base, the `returns ** 2` part must be adapted to the target."
    )
    st.subheader("Feature blocks and FeatureSpec")
    st.markdown(
        """
- **FeatureBlock**: mathematical definition of one feature column, with `compute(returns, tickers)` returning `(name, values)`. `long_only = True` blocks are skipped in wide format.
- **FeatureSpec**: ordered tuple of blocks. `default(...)` gives the early default set, `with_blocks(...)` adds blocks to an existing spec, `compute(...)` produces the values for the wide/long views.
- **`fb.build_features`**: takes spec, target, calendar and format, builds the features from the merged warm-up + train + forecast data and creates `r_next` with `returns.shift(-1)`.
"""
    )
    st.code(
        """spec = FeatureSpec((
    Returns(), AbsReturn(), RollingVol((5, 21)),
    EwmaVol(21), EwmaVariance(0.94), TickerOneHot(),   # one-hot: long only
))
target = RatioToBase(clip=20)
training_feature, forecast_feature = fb.build_features(calendar, "long", spec, target)""",
        language="python",
    )


def models():
    st.header("Models and Training")
    st.subheader("TrainedModel (abstract class)")
    st.markdown(
        """
Constructor fields: `training_feature`, `underlying_model` (e.g. `XGBRegressor`), `model_name`, `target_scale`,
`feature_cols` (column order used at prediction), and `residual_bank_in_sample` / `ewma_variance_bank_in_sample`
(both **`None` at first**, filled after the model is trained and scored).

| Method | Purpose |
|---|---|
| `predict_variance_single_wide(forecast_feature, feature_date)` | Abstract. Variance of the **next bar** for one date (portfolio variance for wide models) |
| `predict_variance_batch_wide(forecast_feature, feature_dates)` | Same for many dates with one `.predict()` call, returns a date-indexed Series |
| `predict_variance_batch_long_by_ticker(X_long, id_cols, feature_dates)` | Not abstract. Splits the long block by ticker, builds a wide view per ticker, returns `dict[ticker, Series]` |
"""
    )
    st.subheader("Training residuals")
    st.markdown(
        "The model is currently trained once on the whole training period. Its predictions on that same period (it has seen all of it) "
        "give the training residuals."
    )
    st.code(
        """long_format_train_seed(pooled_model, calendar, training_feature, residual_bank, ewma_bank)
    -> tuple[ResidualBank, EwmaVarianceBank]""",
        language="python",
    )
    st.markdown(
        """
1. Predict variances for `training_feature.feature_dates` with `predict_variance_batch_long_by_ticker`.
2. Decide `usable_dates` (where the EWMA recursion runs): if the calendar has enough warm-up bars for `ewma_min_periods`, they start at `feature_dates`; otherwise a burn-in is cut first (`train_dates_past_burn_in`).
3. `training_safe_bulk_insert` builds the first covariance matrix from `ewma_min_periods` bars.
4. `general_bulk_insert` runs the EWMA recursion and computes portfolio variances, which are inserted into the residual bank.
5. Both banks are marked as trained, so training data cannot be written again.
"""
    )
    st.subheader("Forecast residuals")
    st.markdown(
        "`calculate_multi_day_standart_residuals_from_wide_format_model` and `..._from_long_format_model`:"
    )
    st.markdown(
        """
1. Copy the model's training bank.
2. Take `(feature_date, outcome_date)` pairs from `calendar.forecast_pairs`. The first forecast day is priced from the last train bar's features.
3. Predict the variance for each feature day.
4. Read the realized portfolio returns of the outcome days from `calendar.forecast_data`.
5. `bank.general_bulk_insert(...)` adds the standardized residuals. The filled bank is returned.
"""
    )
    st.subheader("GARCH models (baseline)")
    st.latex(r"\sigma^2_{t+1} = \omega + \alpha\, r_t^2 + \beta\, \sigma^2_t")
    st.markdown(
        """
`omega, alpha, beta` are fitted by maximum likelihood on training data. The **wide GARCH** (portfolio returns only) is the main baseline.

- `_fit_params`: fits GARCH on returns (with scale) and returns `result.params`.
- `_one_step_variance(returns, params)`: builds a fixed model with `.fix(params)` and calls `forecast(horizon=1, start=0, reindex=False)` to get a one-step variance for **every day**.
- `check_chronological_order_and_combine_dataframes`: joins train and forecast returns, because GARCH must also see the training period when forecasting (expanding logic). Marked `#possiblebug` in the source notes.
- **Wide**: trains on portfolio returns. **Long**: trains per ticker, parameters in `TrainedGarchModel.params[ticker]`, then combines with the EWMA bank like the tree models.
"""
    )


def walk_forward():
    st.header("Walk-Forward Experiment")
    st.markdown(
        "Creates many calendars from the same data, trains the same model on each window and judges the model over all periods together. "
        "Forecasted days are not skipped, they become part of the next period's forecast set. One experiment produces **one VaR DataFrame**."
    )
    st.code(
        """setting = wf.WalkForwardSetting(
    df, TICKERS, WEIGHTS,
    train_bars=756, forecast_bars=252, expanding=False, warmup_bars=30,
    start_date=None, end_date=None,
)
calendars = setting.make_calendars()""",
        language="python",
    )
    st.markdown(
        """
| Parameter | Meaning |
|---|---|
| `train_bars` | Bars of training data per window |
| `forecast_bars` | Bars to forecast (default 21) |
| `warmup_bars` | Look-back bars for rolling features, not part of training rows (default 30) |
| `expanding` | Let the train window grow (not recommended) or keep it fixed |
| `residual_sharing` | `"none"` or `"carry_over"` |
| `start_date`, `end_date` | `None` = start / end of the data |
"""
    )
    st.subheader("Model recipes")
    st.markdown(
        """
`ModelRecipe` fields: `model_family` (`'xgboost'`, `'garch'`, or `'ewma'` = pure EWMA, the "do nothing" baseline),
`model_format` (`'long'` or `'wide'`), `features`, `target`, and the bank settings (`mu_mode`, `default_mu_hat`,
`ewma_lam`, `ewma_min_periods`), which are baked into the residuals at training time.
"""
    )
    st.code(
        """recipe_garch = wf.ModelRecipe('wide', model_family='garch',
                              model_kwargs=dict(mean='Zero', vol='GARCH', p=1, q=1, dist='normal'))
recipe_ewma  = wf.ModelRecipe('wide', model_family='ewma')
recipe_ratio = wf.ModelRecipe('long', target=RatioToBase(clip=20),
                              features=FeatureSpec.default().with_blocks(TickerOneHot()))""",
        language="python",
    )
    st.markdown(
        "`train_walk_forward(calendars, recipe)` trains one model per calendar. Each `WalkForwardSingleCalendarFit` holds the trained model, "
        "its residual bank, its EWMA bank and its `window_id`. "
        "`build_combined_residual_bank_WITHOUT_CALC` merges the forecast parts of all fits (the first fit's training residuals are put in front so every date has residuals)."
    )


def backtest():
    st.header("Backtest Suite")
    st.markdown(
        "`run_full_backtest_suite_single_window(forecast_df, ...)` takes a date-indexed `forecast_df` with `varForecast` and `loss`. "
        "An **exceedance** is a day with `loss > varForecast`. With `p = 1 - var_confidence`, a correct VaR has an exceedance probability of p. "
        "`test_confidence` sets how strictly the test decides: `alpha = 1 - test_confidence`, so 0.95 to 0.99 lowers the risk of wrongly rejecting a correct model from 5% to 1%."
    )
    st.subheader("Statistical tests")
    st.markdown(
        """
| Test | Question | Statistic | Output |
|---|---|---|---|
| Binomial | Does the exceedance count match expectation | z = (X - Np) / sqrt(Np(1-p)), reject if abs(z) > 1.96 | `binomial_pass` |
| Basel traffic light | Which Basel zone is the count in | P(Binom(N,p) <= X): < 0.95 green, < 0.9999 yellow, else red | `traffic_light_zone` |
| Kupiec POF | Is the exceedance **rate** equal to p (unconditional coverage) | Likelihood ratio, chi2(1) | `kupiec_pof_stat`, `kupiec_pof_pass` |
| Kupiec TUFF | Is the time to the first exceedance plausible | LR on first exceedance day, chi2(1) | `kupiec_tuff_pass` |
| Haas TBF | Are all waiting times between exceedances plausible | LR_TBFI + LR_POF, chi2(X+1) | `haas_tbf_pass` |
| Christoffersen independence | Do exceedances cluster | 1st order Markov chain LR, chi2(1) | `christoffersen_ind_stat/_pass` |
| Christoffersen conditional coverage | Rate **and** independence together | LR_POF + LR_IND, chi2(2) | `christoffersen_cc_stat/_pass` |
"""
    )
    st.caption(
        "Short windows (21 days) have little statistical power, long windows (252 days) are far more informative. "
        "The source notes on this comparison are still unfinished."
    )
    st.subheader("Model-based metrics")
    st.markdown(
        """
| Group | Metric | Meaning |
|---|---|---|
| Score | `mean_tick_loss` | Headline metric, lower is better |
| | `tick_loss_exceedance_part` | Mean of `a*(L-V)+`: cost of a VaR that is too low |
| | `tick_loss_overshoot_part` | Mean of `(1-a)*(V-L)+`: cost of a VaR that is too high |
| Conservativeness | `mean_var`, `median_var` | Raw VaR level |
| | `var_to_realized_quantile_ratio` | `mean_var / empirical a-quantile of L` (> 1 = conservative) |
| | `mean_slack_non_exceedance` | Mean `V - L` on non-exceedance days (unused buffer) |
| Tail severity | `mean_exceedance_size`, `max_exceedance_size` | How bad the misses are |
| Volatility | `var_std`, `var_cv` | Scale-free spread of the VaR |
| | `mean_abs_daily_change`, `relative_mean_abs_daily_change` | Day-to-day jumpiness |
| | `var_p95_p5_ratio`, `var_max_min_ratio` | Range |
| Context | `exceedance_rate`, `expected_exceedance_rate`, `n_days`, `var_confidence` | Read conservativeness together with coverage (VaR = 100% is "conservative" and useless) |
"""
    )
    st.markdown("**Baseline:** the wide GARCH model.")


# ---------------------------------------------------------------- navigation
PAGES = {
    "Overview": overview,
    "Calendar": calendar,
    "Residual Bank": residual_bank,
    "EWMA Variance Bank": ewma_bank,
    "Features and Targets": features,
    "Models and Training": models,
    "Walk-Forward": walk_forward,
    "Backtest Suite": backtest,
}

st.sidebar.title("VaR Framework")
choice = st.sidebar.radio("Section", list(PAGES), label_visibility="collapsed")
st.title("VaR Forecasting Framework: Documentation")
PAGES[choice]()
