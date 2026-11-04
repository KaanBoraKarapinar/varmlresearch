## Research questions

**Primary question:** Does XGBoost-based variance forecasting, combined with Filtered Historical Simulation (FHS), produce better-calibrated Value-at-Risk (VaR) estimates than FHS with GARCH or EWMA variance forecasts, or than historical simulation?

**Secondary question:** How do the models perform over the whole period from 2010 to 2026, with particular emphasis on market downturns such as the COVID-19 pandemic and the February 2018 correction?

1. For long models, does the explanatory power come mainly from the EWMA covariance or from the models' variance predictions?
2. Do more complex feature sets consistently improve the out-of-sample VaR forecasts, or do simpler models do as well?
3. To what extent does realized variance, calculated from five-minute price data, improve the models when it is used as a target and as a feature?
4. For simple but noisy targets (such as squared returns) and for more complex targets (such as realized variance and the ratio to the base variance), how do the models respond to different training periods (252, 504, 756 and 1008 days) and to different refit frequencies (daily and every 21 days)?
5. Can XGBoost models trained on a larger stock universe keep or improve their forecasting performance when they are applied to a smaller portfolio without retraining?

## Portfolios and universes

- Equal-weight portfolio of 4 stocks: AAPL, JPM, XOM, KO
  - with models that only see the data of these 4 tickers,
  - with models that see the data of 23 tickers.
- Equal-weight portfolio of 23 stocks: AAPL, MSFT, NVDA, AVGO, GOOGL, META, NFLX, AMZN, HD, MCD, PG, KO, WMT, XOM, CVX, JPM, GS, BAC, V, JNJ, UNH, LLY, MRK
  - Because of hedging between the stocks, the daily realized loss of this portfolio is centered around 0%.
- 200 random-weight portfolios of the same 23 stocks
  - 50 portfolios for each of the Dirichlet parameters alpha = 5, 1, 0.5 and 0.1.

## Experiment 1: differences between features

Five feature sets are defined; F0 is the base of all of them:

| Feature set | Definition | Number of features |
|---|---|---|
| F0 | Returns (`r`), absolute returns (`abs_r`), EWMA volatility (21) | 3 |
| F1 | F0 + `RollingStats(5, 10, 21)` | 12 |
| F2 | F1 + macro | 20 |
| F3 | F1 + `RvHar()` | 15 |
| F4 | F3 + macro | 23 |

and five targets:

| Target | y | Abbreviation |
|---|---|---|
| SquaredReturn | next day's r² × scale | sq |
| RatioToBase | next day's r² / EWMA variance | ratio |
| RealizedVariance | next day's realized variance (RV) × scale | rv |
| RelativeRvChange | RV_next / RV_today − 1 | drv |
| LogVariance | log(r² + ε) | ln |

A model is named by its feature set and its target: `xgb_F3_drv` is XGBoost with the features of F3 and the target `drv`.

## Equal-weight portfolio of 23 tickers

- **The target matters more than the feature set.** With the same features the order is drv, rv, then sq and ln (about equal), then ratio:
  - mean tick loss (x10^-3) with F1: drv 1.206, rv 1.266, sq 1.284, ln 1.287, ratio 1.312;
  - with F3: drv 1.210, rv 1.231, sq 1.241, ln 1.243, ratio 1.276.
- **Realized variance as a feature** lowers the tick loss for four of the five targets, by 2.7% to 3.4%, and not for drv:
  - ln 1.287 to 1.243, sq 1.284 to 1.241, rv 1.266 to 1.231, ratio 1.312 to 1.276, drv 1.206 to 1.210 (slightly worse).
  - The best run, `xgb_F1_drv`, does not use RV as a feature, only as (the change of) the target.
- **Macro variables do not help.** F1 to F2: 1.312 to 1.303 (almost nothing); F3 to F4: 1.276 to 1.297 (worse). Both for the ratio target.
- **RollingStats add nothing.** F0 1.314, F1 1.312 (ratio target).

The five models with the highest Model Confidence Set p-value over the entire period, and the log HAR-RV (`har_rv_log`, M17: a linear regression of the next day's log realized variance on the logs of today's RV and of its 5- and 22-day means, the standard benchmark for realized-variance forecasts):

<!-- open: feature_and_target_importance | Entire set -->
| Run | mean tick loss (x10^-3) | tests | exceedance rate | in the set | ↓ set p-value | blocks won | mean block rank | p vs M01 · garch_wide | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M15 · xgb_F1_drv \| train 756 f 21 \| 23 tickers | 1.206 | PASS | 5.37% | yes | 1.000 | 5 of 8 | 3.88 | 0.222 | **best** |
| M16 · xgb_F3_drv \| train 756 f 21 \| 23 tickers | 1.210 | PASS | 5.75% | yes | 0.515 | 1 of 8 | 3.88 | 0.256 | best set |
| M01 · garch_wide \| train 756 f 21 \| 23 tickers | 1.231 | PASS | 5.32% | yes | 0.339 | 1 of 8 | 7.38 | | best set |
| M07 · xgb_F3_ratio \| train 756 f 21 \| 23 tickers | 1.276 | PASS | 5.48% | yes | 0.213 | 0 of 8 | 8.12 | 0.076 | best set |
| M14 · xgb_F3_rv \| train 756 f 21 \| 23 tickers | 1.231 | PASS | 5.59% | yes | 0.213 | 0 of 8 | 5.88 | 0.992 | best set |
| M17 · har_rv_log \| train 756 f 21 \| 23 tickers | 1.232 | PASS | 5.75% | yes | 0.209 | 0 of 8 | 4.75 | 0.939 | best set |

*The five models are ordered by the "↓ set p-value" column; the log HAR-RV is the 8th of the 17 runs by that column and is added at the end.*

- EWMA is not among the best models. GARCH ranks third overall: it beats most of the XGBoost models, and only the two drv models are ahead of it.
- The log HAR-RV is level with GARCH (1.232 against 1.231, Diebold-Mariano p = 0.94) and with `xgb_F3_rv`, which has the same inputs and the same target; only the two drv models are clearly ahead of it.
- The advantage of the two drv models over GARCH is not statistically significant on this portfolio (Diebold-Mariano p = 0.22 and 0.26).

**COVID-19.** GARCH is the best model for the equal-weight portfolio, followed by `xgb_F1_drv` and `xgb_F3_drv`. The historical simulation fails the exceedance tests. The log HAR-RV is behind GARCH here (3.336 against 3.033), but not significantly (p = 0.11).

<!-- open: feature_and_target_importance | COVID-19 · full -->
| Run | mean tick loss (x10^-3) | tests | exceedance rate | in the set | ↓ set p-value | blocks won | mean block rank | p vs M01 · garch_wide | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M01 · garch_wide \| train 756 f 21 \| 23 tickers | 3.033 | PASS | 5.51% | yes | 1.000 | 1 of 1 | 1.00 | | **best** |
| M15 · xgb_F1_drv \| train 756 f 21 \| 23 tickers | 3.118 | PASS | 5.51% | yes | 0.834 | 0 of 1 | 2.00 | 0.697 | best set |
| M16 · xgb_F3_drv \| train 756 f 21 \| 23 tickers | 3.138 | PASS | 6.30% | yes | 0.834 | 0 of 1 | 3.00 | 0.619 | best set |
| M14 · xgb_F3_rv \| train 756 f 21 \| 23 tickers | 3.314 | PASS | 7.09% | yes | 0.564 | 0 of 1 | 4.00 | 0.425 | best set |
| M03 · historical \| train 756 f 21 \| 23 tickers | 5.115 | FAIL | 12.60% | yes | 0.502 | 0 of 1 | 17.00 | 0.150 | **not valid** |
| M17 · har_rv_log \| train 756 f 21 \| 23 tickers | 3.336 | PASS | 6.30% | yes | 0.157 | 0 of 1 | 5.00 | 0.111 | best set |

**2022 bear market and the recovery after it.** `xgb_F3_drv` (with the variance features) is ahead of `xgb_F1_drv`. Pure EWMA becomes relevant again, and the ratio models, which capture deviations from the EWMA, get close, but they do not pass `xgb_F3_drv`. The best run is M16 · xgb_F3_drv. The log HAR-RV (1.235) is level with GARCH (1.237). The table shows the five models with the highest set p-value and the log HAR-RV.

<!-- open: feature_and_target_importance | 2022 bear market · full -->
| Run | mean tick loss (x10^-3) | tests | exceedance rate | in the set | ↓ set p-value | blocks won | mean block rank | p vs M01 · garch_wide | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M16 · xgb_F3_drv \| train 756 f 21 \| 23 tickers | 1.227 | PASS | 6.23% | yes | 1.000 | 1 of 3 | 8.00 | 0.644 | **best** |
| M01 · garch_wide \| train 756 f 21 \| 23 tickers | 1.237 | PASS | 5.25% | yes | 0.971 | 1 of 3 | 6.00 | | best set |
| M02 · ewma \| train 756 f 21 \| 23 tickers | 1.248 | PASS | 5.25% | yes | 0.971 | 1 of 3 | 7.00 | 0.649 | best set |
| M04 · xgb_F0_ratio \| train 756 f 21 \| 23 tickers | 1.246 | PASS | 5.25% | yes | 0.971 | 0 of 3 | 6.33 | 0.693 | best set |
| M05 · xgb_F1_ratio \| train 756 f 21 \| 23 tickers | 1.242 | PASS | 4.86% | yes | 0.971 | 0 of 3 | 5.67 | 0.762 | best set |
| M17 · har_rv_log \| train 756 f 21 \| 23 tickers | 1.235 | PASS | 6.03% | yes | 0.952 | 0 of 3 | 6.67 | 0.941 | best set |

### Do the models miss on the same days?

For the three best models and the log HAR-RV over the entire period, the exceedances mostly occur on the same days: of the 124 days on which at least one of the four runs exceeds its VaR, 79 are days on which all of them do.

<!-- open: feature_and_target_importance | Entire set | Exceedances / Daily VaR explaining | Do the runs miss on the same days? -->
| runs exceeding the VaR that day | days |
|---|---|
| only 1 run | 16 |
| 2 runs | 10 |
| 3 runs | 19 |
| all runs | 79 |

Per run:

<!-- open: feature_and_target_importance | Entire set | Exceedances / Daily VaR explaining | Do the runs miss on the same days? -->
| Run | exceedance days | only this run | shared with another run | loss − VaR difference (only-this-run days) |
|---|---|---|---|---|
| M01 · garch_wide \| train 756 f 21 \| 23 tickers | 98 | 8 | 90 | 0.96% |
| M15 · xgb_F1_drv \| train 756 f 21 \| 23 tickers | 99 | 5 | 94 | 0.40% |
| M16 · xgb_F3_drv \| train 756 f 21 \| 23 tickers | 106 | 1 | 105 | 0.00% |
| M17 · har_rv_log \| train 756 f 21 \| 23 tickers | 106 | 2 | 104 | 0.69% |

The same pattern holds in the COVID-19 period:

<!-- open: feature_and_target_importance | COVID-19 · full | Exceedances / Daily VaR explaining | Do the runs miss on the same days? -->
| runs exceeding the VaR that day | days |
|---|---|
| only 1 run | 1 |
| 2 runs | 1 |
| 3 runs | 1 |
| all runs | 6 |

Per run:

<!-- open: feature_and_target_importance | COVID-19 · full | Exceedances / Daily VaR explaining | Do the runs miss on the same days? -->
| Run | exceedance days | only this run | shared with another run | loss − VaR difference (only-this-run days) |
|---|---|---|---|---|
| M01 · garch_wide \| train 756 f 21 \| 23 tickers | 7 | 0 | 7 | 0.00% |
| M15 · xgb_F1_drv \| train 756 f 21 \| 23 tickers | 7 | 0 | 7 | 0.00% |
| M16 · xgb_F3_drv \| train 756 f 21 \| 23 tickers | 8 | 0 | 8 | 0.00% |
| M17 · har_rv_log \| train 756 f 21 \| 23 tickers | 8 | 1 | 7 | 0.66% |

## Random-weight portfolios of 23 tickers

Because of the hedging between the stocks, the realized loss of the equal-weight portfolio is concentrated near 0%, and that makes it hard to tell the models apart. To evaluate the models on portfolios with more varied risk, 200 fixed-weight portfolios are drawn from a Dirichlet distribution, 50 for each concentration parameter alpha = 5, 1, 0.5 and 0.1 (seed 0).

**A large alpha gives weights that are nearly equal; a small alpha gives portfolios dominated by a few stocks.**

<!-- weights: 23 | 756 | 21 -->

| alpha | mean effective number of stocks | mean largest weight |
|---|---|---|
| 5 | 19.5 | 9% |
| 1 | 12.5 | 16% |
| 0.5 | 8.6 | 23% |
| 0.1 | 3.4 | 48% |

The more concentrated the portfolio, the wider the distribution of its daily realized loss:

| Portfolio | Std of daily loss | Days inside ±1% | Days inside ±2% |
|---|---|---|---|
| Equal weight, 23 stocks | 1.19% | 75.5% | 94.0% |
| Dirichlet alpha = 5 (close to equal weight) | 1.20% | 75.1% | 93.8% |
| alpha = 1 | 1.22% | 73.9% | 93.4% |
| alpha = 0.5 | 1.30% | 70.7% | 91.9% |
| alpha = 0.1 (close to a single dominant stock) | 1.50% | 63.5% | 87.8% |
| Equal weight, 4 stocks (AAPL, JPM, XOM, KO) | 1.30% | 71.7% | 92.6% |

**Findings**

- **Models that use realized-variance information beat GARCH on nearly every portfolio, but by a small margin.** At the 95% level the best models (`xgb_F1_drv`, `xgb_F3_drv`) have a 2.6% to 2.7% lower mean tick loss than GARCH, and they beat it on 195 and 199 of the 200 portfolios.
  - Both models use the target RelativeRvChange (`drv`). `xgb_F3_drv` is significantly better than GARCH on more portfolios than `xgb_F1_drv` (76 against 58 of 200).
- **At the 99% level the gap roughly doubles** (to about −6%) for the models built on realized variance (`drv`, `rv`, `ln` and `sq` with the F3 features).
- **The advantage grows with the concentration.** From alpha = 5 (near-equal weights) to alpha = 0.1, the advantage of the best model at 95% grows from −2.2% to −3.7%, and the number of portfolios with a significant Diebold-Mariano (DM) result grows from 3 to 32 of 50. GARCH falls from rank 5 to rank 9 of 17 (the 17 models are the 13 XGBoost runs, the log HAR-RV, EWMA, GARCH and the historical simulation).
  - In near-equal-weight portfolios, where the hedging keeps the losses near 0%, GARCH is about as good as the best models. At 95%, `xgb_F3_sq` and `xgb_F3_ln` do not beat GARCH.
- **In the COVID-19 crash GARCH is as good as XGBoost or better at 95%.** GARCH is rank 1 for alpha = 5 and 1; XGBoost is ahead only for the concentrated portfolios (−0.4% at alpha = 0.5, −6.5% at alpha = 0.1). **At 99% the realized-variance models are 13% to 19% better** (the log HAR-RV 10% to 12%).
  - **However,** the DM test does not confirm the 99% COVID-19 result: 0 to 10 of 50 portfolios are significant, with a median p-value of about 0.2. The period has 127 days, so a 99% VaR has very few expected exceedances (about 1).
- **The log HAR-RV, a linear regression with no trees, gets a large part of the gain.** At 95% over the entire period it is the 4th of 17 models (tick loss 1.351, 1.4% below GARCH; better than GARCH on 162 of the 200 portfolios, significantly on 37) and is close to `xgb_F3_rv`, which has the same inputs and target (1.350); only the two drv models are clearly ahead of it. Its advantage over GARCH grows with the concentration too: −0.1% at alpha = 5, −0.7% at alpha = 1, −1.6% at alpha = 0.5 and −3.1% at alpha = 0.1.
  - In the COVID-19 crash at 95% it is worse than GARCH for the portfolios of alpha 0.5, 1 and 5 (+6% to +10%) and level with it for alpha = 0.1. At 99% it is 10.7% better than GARCH (188 of the 200 portfolios, significantly on 10), 2 to 5 percentage points less than the three best XGBoost runs.
  - So most of what the XGBoost models gain over GARCH comes from the realized variance itself, which a linear regression uses as well; what the trees add is the `drv` target at 95% and the `ln` and `rv` targets at 99% in the crash, and that is 2 to 5 percentage points at most.

**VaR 95%, entire period, all 200 portfolios**

| Portfolios | Rank | Model | Model tick loss (x10^-3) | GARCH tick loss (x10^-3) | GARCH minus model (x10^-3) | Relative difference to GARCH | Portfolios where the model beats GARCH | Portfolios where it is significantly better (Diebold-Mariano, p < 0.05) | Median DM p |
|---|---|---|---|---|---|---|---|---|---|
| All 200 portfolios | 1 | xgb_F1_drv | 1.334 | 1.373 | +0.040 | -2.7% | 195 of 200 | 58 of 200 | 0.13 |
| | 2 | xgb_F3_drv | 1.334 | 1.373 | +0.039 | -2.6% | 199 of 200 | 76 of 200 | 0.10 |
| | 3 | xgb_F3_rv | 1.350 | 1.373 | +0.023 | -1.5% | 174 of 200 | 30 of 200 | 0.57 |
| | 4 | har_rv_log | 1.351 | 1.373 | +0.022 | -1.4% | 162 of 200 | 37 of 200 | 0.49 |
| | 7 | GARCH | 1.373 | 1.373 | 0 | 0 | reference | reference | |

**VaR 99%, COVID-19 (2020-02-19 to 2020-08-18, 127 days), all 200 portfolios**

The realized-variance models with F3 features are the best, but the DM test confirms the difference on only a few portfolios.

| Portfolios | Rank | Model | Model tick loss (x10^-3) | GARCH tick loss (x10^-3) | GARCH minus model (x10^-3) | Relative difference to GARCH | Portfolios where the model beats GARCH | Portfolios where it is significantly better (Diebold-Mariano, p < 0.05) | Median DM p |
|---|---|---|---|---|---|---|---|---|---|
| All 200 portfolios | 1 | xgb_F3_ln | 0.858 | 1.021 | +0.163 | -16.0% | 189 of 200 | 10 of 200 | 0.22 |
| | 2 | xgb_F3_rv | 0.869 | 1.021 | +0.152 | -14.9% | 188 of 200 | 18 of 200 | 0.18 |
| | 3 | xgb_F1_drv | 0.884 | 1.021 | +0.137 | -13.2% | 188 of 200 | 16 of 200 | 0.26 |
| | 6 | har_rv_log | 0.910 | 1.021 | +0.111 | -10.7% | 188 of 200 | 10 of 200 | 0.29 |
| | 11 | GARCH | 1.021 | 1.021 | 0 | 0 | reference | reference | |

## Equal-weight portfolio of 4 tickers

The portfolio of AAPL, JPM, XOM and KO, 0.25 each, is less diversified than the 23 stocks: the standard deviation of its daily loss is 1.30% against 1.19%, and 71.7% of its days lie inside ±1% against 75.5% (see the table of the random portfolios below). Same setup as above: train 756 / forecast 21, the forecast days 2019-03-04 to 2026-06-30 (1842 days), VaR 95% (and 99%), residual window 252, min periods 30. Two questions are answered: how the models do when they are trained on these 4 tickers alone, and whether the models trained on the 23 tickers do as well on this portfolio without retraining (research question 5).

### The 23-stock and the 4-stock portfolio side by side

Each model is trained on the tickers of its own portfolio. Difference to the GARCH of the same portfolio (negative = cheaper than GARCH) and its Diebold-Mariano p-value (not corrected for the number of comparisons), entire period. Tick loss x10^-3.

VaR 95%:

| Model | 23 stocks: tick loss | 23 stocks: vs GARCH | 23 stocks: DM p | 4 stocks: tick loss | 4 stocks: vs GARCH | 4 stocks: DM p |
|---|---|---|---|---|---|---|
| xgb_F1_drv | 1.206 | -2.0% | 0.22 | 1.272 | -4.5% | 0.035 |
| xgb_F3_drv | 1.210 | -1.7% | 0.26 | 1.285 | -3.6% | 0.063 |
| xgb_F3_rv | 1.231 | +0.0% | 0.99 | 1.309 | -1.8% | 0.48 |
| xgb_F3_ratio | 1.276 | +3.6% | 0.076 | 1.358 | +1.9% | 0.33 |
| xgb_F3_sq | 1.241 | +0.8% | 0.75 | 1.391 | +4.4% | 0.32 |
| xgb_F3_ln | 1.243 | +1.0% | 0.64 | 1.386 | +4.1% | 0.42 |
| har_rv_log | 1.232 | +0.1% | 0.94 | 1.287 | -3.4% | 0.030 |
| GARCH | 1.231 | | | 1.332 | | |
| EWMA | 1.315 | +6.8% | 0.002 | 1.405 | +5.5% | 0.002 |
| historical | 1.473 | +19.6% | 0.026 | 1.523 | +14.3% | 0.077 |

VaR 99%:

| Model | 23 stocks: tick loss | 23 stocks: vs GARCH | 23 stocks: DM p | 4 stocks: tick loss | 4 stocks: vs GARCH | 4 stocks: DM p |
|---|---|---|---|---|---|---|
| xgb_F1_drv | 0.3586 | -6.9% | 0.040 | 0.3712 | -3.2% | 0.34 |
| xgb_F3_drv | 0.3618 | -6.0% | 0.033 | 0.3653 | -4.8% | 0.10 |
| xgb_F3_rv | 0.3589 | -6.8% | 0.029 | 0.3688 | -3.8% | 0.10 |
| xgb_F3_ratio | 0.3702 | -3.8% | 0.39 | 0.4124 | +7.5% | 0.017 |
| xgb_F3_sq | 0.3535 | -8.2% | 0.015 | 0.4522 | +17.9% | 0.042 |
| xgb_F3_ln | 0.3590 | -6.8% | 0.049 | 0.4582 | +19.5% | 0.097 |
| har_rv_log | 0.3628 | -5.8% | 0.036 | 0.3763 | -1.9% | 0.54 |
| GARCH | 0.3850 | | | 0.3835 | | |
| EWMA | 0.3867 | +0.4% | 0.93 | 0.4101 | +6.9% | 0.010 |
| historical | 0.5200 | +35.1% | 0.090 | 0.5481 | +42.9% | 0.040 |

- **At 95% the realized-variance models are further ahead of GARCH on the 4-stock portfolio**: `xgb_F1_drv` is 4.5% better (p = 0.035), `xgb_F3_drv` 3.6% (p = 0.063) and the log HAR-RV 3.4% (p = 0.030), against 2.0% and 1.7% better (p = 0.22 and 0.26) and 0.1% worse (p = 0.94) on the 23 stocks. This fits the result of the random portfolios, where the advantage grows as the portfolio gets more concentrated.
- **At 99% the picture is the other way round for the significance.** On the 23 stocks the models that use realized variance (drv, rv, sq, ln and the log HAR-RV) are 6% to 8% better than GARCH, with p-values from 0.015 to 0.05. On the 4 stocks the good ones (drv, rv and the HAR-RV) are 2% to 5% better and none is significant (p from 0.10); at 99% only about 18 exceedances are expected in 1842 days, so the tests have little power.
- **The noisy targets are the weak point when only 4 tickers are used**: trained on the 4 tickers, `xgb_F3_sq`, `xgb_F3_ln` and `xgb_F3_ratio` are 7.5% to 19.5% worse than GARCH at 99%, while on the 23 stocks they are as good as GARCH or better. The section below looks at that.
- EWMA and the historical simulation are worse than GARCH on both portfolios at 95%; at 99% the historical simulation is 35% to 43% worse, and EWMA is level with GARCH on the 23 stocks and 6.9% worse on the 4.

### Models trained on the 4 tickers alone

The five models with the highest set p-value over the entire period (the log HAR-RV is among them), and GARCH, the baseline:

<!-- open: feature_and_target_importance_4_tickers | Entire set -->
| Run | mean tick loss (x10^-3) | tests | exceedance rate | in the set | ↓ set p-value | blocks won | mean block rank | p vs M01 · garch_wide | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M15 · xgb_F1_drv \| train 756 f 21 \| 4 tickers | 1.272 | PASS | 5.10% | yes | 1.000 | 3 of 8 | 6.62 | 0.034 | **best** |
| M16 · xgb_F3_drv \| train 756 f 21 \| 4 tickers | 1.285 | PASS | 5.16% | yes | 0.537 | 0 of 8 | 5.12 | 0.063 | best set |
| M17 · har_rv_log \| train 756 f 21 \| 4 tickers | 1.287 | PASS | 5.27% | yes | 0.537 | 0 of 8 | 4.50 | 0.030 | best set |
| M07 · xgb_F3_ratio \| train 756 f 21 \| 4 tickers | 1.358 | PASS | 5.59% | yes | 0.351 | 2 of 8 | 6.62 | 0.331 | best set |
| M14 · xgb_F3_rv \| train 756 f 21 \| 4 tickers | 1.309 | PASS | 5.10% | yes | 0.351 | 0 of 8 | 6.50 | 0.477 | best set |
| M01 · garch_wide \| train 756 f 21 \| 4 tickers | 1.332 | PASS | 5.10% | yes | 0.229 | 0 of 8 | 7.25 | | best set |

- **The best models are the same two drv models as on the 23 stocks, and here the log HAR-RV is as good as `xgb_F3_drv`**; the three are 3.4% to 4.5% below GARCH. Of the 17 runs, 15 pass the exceedance tests; the two that fail are the historical simulation and `xgb_F1_sq`, the most fragile model (see the last section).
- **The target matters, and here the order of the noisy ones differs.** With F3 features: drv 1.285, rv 1.309, ratio 1.358, ln 1.386, sq 1.391; with F1 features: drv 1.272, rv 1.350, ratio 1.408, ln 1.468, sq 1.591. On the 23 stocks ratio was the worst target; on the 4 stocks sq and ln are.
- **Realized variance as a feature (F1 to F3) lowers the tick loss for four of the five targets, by more than on the 23 stocks**: rv 3.0%, ratio 3.5%, ln 5.5%, sq 12.6% (against 2.7% to 3.4%), and not for drv (+1.0%). The features carry more weight when the model has only 4 x 756 rows to learn from.
- **Macro does not help** (ratio target: F1 1.408 to F2 1.398, F3 1.358 to F4 1.367). **RollingStats help a little here** (ratio: F0 1.445 to F1 1.408, -2.6%), which they did not on the 23 stocks (1.314 to 1.312).

COVID-19 (2020-02-19 to 2020-08-18, 127 days):

<!-- open: feature_and_target_importance_4_tickers | COVID-19 · full -->
| Run | mean tick loss (x10^-3) | tests | exceedance rate | in the set | ↓ set p-value | blocks won | mean block rank | p vs M01 · garch_wide | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M15 · xgb_F1_drv \| train 756 f 21 \| 4 tickers | 3.515 | PASS | 5.51% | yes | 1.000 | 1 of 1 | 1.00 | 0.638 | **best** |
| M01 · garch_wide \| train 756 f 21 \| 4 tickers | 3.647 | PASS | 4.72% | yes | 0.608 | 0 of 1 | 2.00 | | best set |
| M08 · xgb_F4_ratio \| train 756 f 21 \| 4 tickers | 3.808 | PASS | 5.51% | yes | 0.500 | 0 of 1 | 5.00 | 0.449 | best set |
| M14 · xgb_F3_rv \| train 756 f 21 \| 4 tickers | 3.834 | PASS | 6.30% | yes | 0.500 | 0 of 1 | 6.00 | 0.665 | best set |
| M17 · har_rv_log \| train 756 f 21 \| 4 tickers | 3.666 | PASS | 4.72% | yes | 0.500 | 0 of 1 | 3.00 | 0.929 | best set |

- In the crash `xgb_F1_drv` is the best model on the 4 stocks (3.515, 3.6% below GARCH), whereas on the 23 stocks GARCH was. The difference is not significant (p = 0.64) and the period is one block of 127 days: it describes the crash, it does not test anything. The log HAR-RV (3.666) is level with GARCH (3.647).

2022 bear market and the recovery after it (2022-01-03 to 2024-01-19). The best model by set p-value, GARCH, `xgb_F3_drv` and the log HAR-RV:

<!-- open: feature_and_target_importance_4_tickers | 2022 bear market · full -->
| Run | mean tick loss (x10^-3) | tests | exceedance rate | in the set | ↓ set p-value | blocks won | mean block rank | p vs M01 · garch_wide | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M07 · xgb_F3_ratio \| train 756 f 21 \| 4 tickers | 1.141 | PASS | 4.86% | yes | 1.000 | 2 of 3 | 5.67 | 0.243 | **best** |
| M01 · garch_wide \| train 756 f 21 \| 4 tickers | 1.171 | PASS | 4.67% | yes | 0.564 | 0 of 3 | 6.67 | | best set |
| M16 · xgb_F3_drv \| train 756 f 21 \| 4 tickers | 1.170 | PASS | 5.06% | yes | 0.564 | 0 of 3 | 5.33 | 0.957 | best set |
| M17 · har_rv_log \| train 756 f 21 \| 4 tickers | 1.157 | PASS | 5.25% | yes | 0.564 | 0 of 3 | 2.67 | 0.550 | best set |

- In 2022 the ratio model, which captures the deviation from the EWMA, is the best on the 4 stocks (1.141, 2.6% below GARCH); on the 23 stocks the ratio models became relevant too, but did not pass `xgb_F3_drv`. The differences between the models shown are small (1.141 to 1.171) and none is significant against GARCH (p from 0.24).

### Models trained on the 23 tickers, applied to the 4-ticker portfolio

Nothing is trained again: the per-ticker variance forecasts that the models trained on the 23 tickers stored are put into the covariance matrix of this portfolio (weights 0.25 for the 4 tickers and 0 for the others; the covariance is the EWMA one of the 4 tickers, the same for every model). The realized loss is the same for both sets of models (checked: the two paths are identical). Mean tick loss x10^-3; the difference is "trained on 23" against "trained on 4", negative = the 23-ticker model is cheaper, and its Diebold-Mariano p-value (HAC, not corrected for the number of comparisons). The models are ordered by their tick loss at 95% when trained on the 4 tickers.

<!-- open: models_of_the_23_tickers_applied_to_the_4_ticker_portfolio | Entire set -->
| Model | 95%: trained on 4 | 95%: trained on 23 | 95%: difference | 95%: DM p | 99%: trained on 4 | 99%: trained on 23 | 99%: difference | 99%: DM p |
|---|---|---|---|---|---|---|---|---|
| xgb_F1_drv | 1.272 | 1.264 | -0.6% | 0.022 | 0.3712 | 0.3701 | -0.3% | 0.77 |
| xgb_F3_drv | 1.285 | 1.274 | -0.8% | 0.045 | 0.3653 | 0.3608 | -1.2% | 0.115 |
| har_rv_log | 1.287 | 1.285 | -0.1% | 0.070 | 0.3763 | 0.3752 | -0.3% | 0.17 |
| xgb_F3_rv | 1.309 | 1.286 | -1.7% | 0.063 | 0.3688 | 0.3593 | -2.6% | 0.048 |
| xgb_F1_rv | 1.350 | 1.335 | -1.1% | 0.20 | 0.3807 | 0.3728 | -2.1% | 0.26 |
| xgb_F3_ratio | 1.358 | 1.328 | -2.2% | 0.025 | 0.4124 | 0.3875 | -6.0% | <0.001 |
| xgb_F4_ratio | 1.367 | 1.379 | +0.9% | 0.48 | 0.4112 | 0.4052 | -1.5% | 0.75 |
| xgb_F3_ln | 1.386 | 1.297 | -6.4% | 0.078 | 0.4582 | 0.3711 | -19.0% | 0.056 |
| xgb_F3_sq | 1.391 | 1.301 | -6.5% | 0.006 | 0.4522 | 0.3686 | -18.5% | 0.006 |
| xgb_F2_ratio | 1.398 | 1.392 | -0.4% | 0.78 | 0.4213 | 0.4136 | -1.8% | 0.63 |
| xgb_F1_ratio | 1.408 | 1.391 | -1.2% | 0.091 | 0.4181 | 0.4000 | -4.3% | 0.010 |
| xgb_F0_ratio | 1.445 | 1.392 | -3.7% | <0.001 | 0.4227 | 0.3984 | -5.8% | 0.017 |
| xgb_F1_ln | 1.468 | 1.363 | -7.1% | 0.016 | 0.4527 | 0.3801 | -16.0% | 0.066 |
| xgb_F1_sq | 1.591 | 1.352 | -15.0% | 0.002 | 0.6707 | 0.3793 | -43.4% | 0.001 |

GARCH, EWMA and the historical simulation have no model to train on the larger set: they are those of the 4-ticker portfolio (GARCH 1.332 and 0.3835, EWMA 1.405 and 0.4101, historical 1.523 and 0.5481 at 95% and 99%).

- **The model trained on the 23 tickers is the cheaper one in 13 of the 14 models at 95% and in all 14 at 99%.** The only exception at 95% is `xgb_F4_ratio` (+0.9%, p = 0.48).
- **For the best models the gain is small**: -0.1% to -0.8% at 95% and -0.3% to -1.2% at 99% for `xgb_F1_drv`, `xgb_F3_drv` and the log HAR-RV, and only partly significant (95%: p = 0.022 and 0.045 for the two drv models, 0.070 for the HAR-RV; 99%: none).
- **For the fragile models the gain is large**: the noisy targets `sq` and `ln` gain 6% to 15% at 95% and 16% to 43% at 99% (`xgb_F1_sq` 15.0% and 43.4%, p = 0.002 and 0.001). A model fitted on only 4 x 756 rows is fragile: `xgb_F1_sq` trained on the 4 tickers has 32 exceedances at 99% (18 are expected in 1842 days), the 23-ticker one 21.
- **The log HAR-RV, a linear regression with three coefficients, hardly changes** (-0.1% and -0.3%). That fits the explanation that what the larger universe gives the flexible models is less overfitting, and not information about these 4 stocks that the 4 tickers alone do not have.
- **Against the GARCH of the portfolio** the best model trained on the 4 tickers is 4.5% (95%) and 4.8% (99%) better, the best one trained on the 23 tickers 5.1% and 6.3% better.
- **Answer to research question 5:** a model trained on a larger universe can be applied to a smaller portfolio without retraining and does not lose; for the best models the gain is small, for the fragile ones it is large. Limits: one portfolio, one training window and refit (756 / 21) and one period; the p-values are not corrected for the number of comparisons.

## Training periods and refit frequency

The four best XGBoost models (`xgb_F1_drv`, `xgb_F3_drv`, `xgb_F3_rv`, `xgb_F3_ratio`) are used to test whether a longer training period, or a more frequent refit of the model, improves the forecasts. Neither does.

### Refit every 21 days against every day

Both are trained on 756 days and forecast the same 1,842 days (2019-03-04 to 2026-06-30): 88 windows with a refit every 21 days, 1,842 windows with a daily refit.

Mean tick loss (x10^-3):

<!-- open: refit_frequency_every_21_days_against_every_day | Entire set -->
| Model | 95%: 21 days | 95%: daily | 95%: difference | 99%: 21 days | 99%: daily | 99%: difference |
|---|---|---|---|---|---|---|
| xgb_F1_drv | 1.206 | 1.209 | +0.2% | 0.359 | 0.360 | +0.3% |
| xgb_F3_drv | 1.210 | 1.216 | +0.4% | 0.362 | 0.363 | +0.3% |
| xgb_F3_rv | 1.231 | 1.237 | +0.4% | 0.359 | 0.363 | +1.0% |
| xgb_F3_ratio | 1.276 | 1.263 | -1.0% | 0.370 | 0.374 | +1.1% |
| GARCH | 1.231 | 1.230 | -0.1% | 0.385 | 0.388 | +0.8% |
| EWMA | 1.315 | 1.315 | 0 | 0.387 | 0.387 | 0 |
| historical | 1.473 | 1.473 | 0 | 0.520 | 0.520 | 0 |

- A positive difference means that the daily refit is worse. **The daily refit does not help: all differences are at most 1.1%, and the daily refit is worse in 8 of the 10 comparisons that differ.** It is better only for `xgb_F3_ratio` and GARCH at 95%.
- It also costs 21 times more windows: 41 minutes for the whole grid on 8 cores, against about 2 minutes for the refit every 21 days.

**Why.** A refit only updates the learned function. The new information of each day already enters through the inputs of the model (the day's return, the EWMA volatility, the RV). The daily variance forecasts of the 21-day and the daily models are therefore almost the same:

<!-- open: refit_frequency_every_21_days_against_every_day | Entire set -->
| Model | Correlation of the log variance forecasts | Median absolute difference | 95th percentile | Mean difference, day 0-4 of a 21-day window | Day 16-20 |
|---|---|---|---|---|---|
| xgb_F1_drv | 0.9992 | 0.5% | 3.2% | 0.6% | 1.5% |
| xgb_F3_drv | 0.9996 | 0.6% | 3.2% | 0.6% | 1.4% |
| xgb_F3_rv | 0.9972 | 0.6% | 3.6% | 1.1% | 2.2% |
| xgb_F3_ratio | 0.9974 | 1.2% | 6.4% | 1.6% | 2.8% |
| GARCH | 0.9989 | 0.9% | 6.9% | 1.0% | 2.5% |

### Training period of 252, 504, 756 and 1008 days

The models are refit every day and forecast the day after (t+1). The tests cover the 1,590 days that all four training periods forecast (2020-03-03 to 2026-06-30).

**The XGBoost tick loss is nearly constant across the training periods** (the `drv` models at 95%: 1.220 to 1.224).

VaR 95%, mean tick loss (x10^-3) and, in brackets, the difference to the GARCH with the same training period:

<!-- open: training_period_252_504_756_and_1008_days | Entire set -->
| Model | T = 252 | T = 504 | T = 756 | T = 1008 |
|---|---|---|---|---|
| xgb_F1_drv | 1.222 (-2.5%) | 1.224 (-0.7%) | 1.223 (-1.0%) | 1.223 (-1.0%) |
| xgb_F3_drv | 1.223 (-2.4%) | 1.223 (-0.8%) | 1.223 (-1.0%) | 1.220 (-1.2%) |
| xgb_F3_rv | 1.242 (-0.9%) | 1.245 (+1.0%) | 1.244 (+0.7%) | 1.245 (+0.8%) |
| xgb_F3_ratio | 1.265 (+0.9%) | 1.277 (+3.5%) | 1.271 (+2.9%) | 1.278 (+3.5%) |
| GARCH | 1.253 | 1.233 | 1.235 | 1.235 |
| EWMA | 1.321 | 1.321 | 1.321 | 1.321 |
| historical | 1.482 | 1.482 | 1.482 | 1.482 |

The XGBoost models hardly change with the training period (the range over the four periods is 0.003 to 0.004 for the drv and rv models and 0.013 for the ratio model). What changes is GARCH: it is worse with 252 days (1.253 against 1.233 to 1.235), so the advantage of `drv` over GARCH is largest at T = 252 (−2.5%).
