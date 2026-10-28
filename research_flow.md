# Research flow

The decisions of the research in order: what we tried, what came out, what we decided. One entry per decision, newest at the bottom.
Every entry has the same fields so the file can be read by a script later (to draw the flow).

Entry format:

```
## R<number>: <short title>
- Status: tried | decided | dropped
- Date: <YYYY-MM-DD> (when it was written down)
- Question: ...
- Tried: ...
- Result: ...
- Decision: ...
- Next: R<number>, ... (the entries that follow from this one)
- Evidence: runs / experiments / notebooks
```

---

## R1: Simple features, several models, tried blindly
- Status: decided
- Date: 2026-10-25
- Question: Can an XGBoost variance model on simple features beat the plain baselines (GARCH(1,1), EWMA) for the 95% / 99% VaR of the 4-ticker portfolio?
- Tried: Seven runs per walk-forward (train 252 / 504 / 756, forecast 252; train 756 with forecast 21): `garch_wide`, `ewma` (baselines), `xgb_wide`, `xgb_long`, `xgb_long_n300`, `xgb_long_ratio`, `xgb_long_ratio_ticker`. The features were the same simple default set for all XGBoost runs (`FeatureSpec.default`); the models differ in the format (wide / long), the number of trees, and the target (squared return or ratio to the base variance, with the ticker one-hot or without). No feature search: the variations were tried blindly.
- Result: None of the XGBoost runs beats `garch_wide`; `ewma` is as good. After the Covid shock (2020-07 onwards, train 756) all seven runs are valid and stay in the Model Confidence Set at 95%, and the tick-loss gaps are small (0.00117 to 0.00127). Training every 21 days instead of every 252 days changes nothing. Within the XGBoost runs, the ratio-target ones (`xgb_long_ratio`, `xgb_long_ratio_ticker`) are the best and are tied with each other; `xgb_wide` and `xgb_long_n300` are the worst. At 99% `xgb_long_n300` breaks down on one near-zero variance forecast (2020-03-16), which shows how fragile the filtered historical simulation is to a tiny predicted variance.
- Decision: The simple features with a blind model search do not beat the baselines. Go on with the ratio-target long models (`xgb_long_ratio`, `xgb_long_ratio_ticker`) as the XGBoost candidates and keep `garch_wide` and `ewma` as the benchmarks; drop `xgb_wide` and `xgb_long_n300`. (Written down from the results above: confirm or correct which models we carried forward.)
- Next: R2
- Evidence: experiment `Post-Covid (2020-07 onwards), train 756 f 252` (store/experiments); runs `train 756 f 21` for the 21-day retraining; notebook `walk_forward_runs.ipynb`.

---

## R2: Whole sample and the Covid crash, read separately
- Status: decided
- Date: 2026-10-25
- Question: Does the R1 result (no XGBoost run beats the baselines) hold on the whole sample, and what happens in the Covid crash itself (2020-02-19 to 2020-06-30, 93 days)?
- Tried: The same seven runs on all common days (train 252 / 504 / 756 with forecast 252, and train 756 with forecast 21), at VaR 95% and 99%; and on the Covid window alone for train 756 (forecast 252 and 21).
- Result:
  - Whole sample, 95%: `garch_wide` has the lowest mean tick loss in every setup (0.00117 to 0.00122). The best XGBoost run is always a ratio-target long model (0.00120 to 0.00128), and its gap to `garch_wide` is significant (Diebold-Mariano p between 0.0005 and 0.03). A longer training window helps the ratio models (0.00128, 0.00121, 0.00120 for 252, 504, 756 days). At train 756 / f 252 `garch_wide` and `ewma` fail the Christoffersen independence test, so among the valid runs `xgb_long_ratio_ticker` ranks first there. `xgb_long` and `xgb_long_n300` often fail independence (violations cluster).
  - Whole sample, 99%: `garch_wide` is the best in every setup (0.00035 to 0.00038). At train 252 / 504 `xgb_long`, `xgb_long_n300` and `xgb_wide` fail Kupiec (too many exceedances). `xgb_long_n300` breaks down at train 756 (mean tick loss about 0.26).
  - Covid crash, 95% (only about 4.7 exceedances expected, so the tests are weak): `garch_wide` is the best (tick loss 0.00443, 6 exceedances). The ratio models and `ewma` follow (7 to 8 exceedances, slightly conservative, VaR about 17-22% above the realized quantile). The squared-return XGBoost runs (`xgb_wide`, `xgb_long`, `xgb_long_n300`) under-estimate the risk (13 to 16 exceedances, fail Kupiec, VaR 25-30% below the realized quantile). Retraining every 21 days does not help in the crash.
  - The Covid crash is where the models separate; after it (R1) the gaps almost vanish.
- Decision: The conclusion of R1 stands on the whole sample: simple features do not beat GARCH or EWMA. The ratio-target long models are the only XGBoost candidates to carry on; the squared-return targets are dropped. Because the models differ mostly in the crash, further work should look at what helps in a volatility shock, not at the calm periods. (Written down from the results above: confirm or correct.)
- Next: (to be filled)
- Evidence: the analysis of the whole sample and the Covid window on the `train 252 / 504 / 756 f 252` and `train 756 f 21` runs (95% and 99%); the post-Covid experiment of R1.
