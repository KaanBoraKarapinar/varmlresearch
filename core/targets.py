"""
What a model is trained to predict, and how its output becomes a variance again.

A target is the forward / inverse pair around the model:

    y        = target.forward_make_y(r_next, base_variance)           # feature builder, training rows
    variance = target.inverse_to_variance(prediction, base_variance)  # every model, at predict time

so a model never needs to know which target it was trained on: it keeps the Target object (carried in
TrainingFeature / ForecastFeature) and calls inverse_to_variance.

One class per target, two methods each. Only RatioToBase uses base_variance (the per-row EWMA variance known
at the feature date for the bar after it); the other targets accept it and ignore it, so the callers always
pass it and do not care which target they hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .units import check_unit


@dataclass(frozen=True)
class Target:
    """
    Base class: forward_make_y(r_next) -> y and inverse_to_variance(prediction) -> variance, one subclass per target.

    scale (every target): what the model sees is y * scale, and the prediction is divided by scale again before
    it becomes a variance. It only makes the numbers friendlier to a tree (r^2 is ~1e-4) and does not change
    the variance that comes out.
    """
    scale: float = 1.0
    # the unit of the returns the target is made from (percent | log, see core/units.py): required, and it
    # must be the calendar's unit (build_features checks)
    unit: str = field(kw_only=True)

    needs_base = False
    # A target made from the calendar's additional feature data (not from the return alone) says so, gives the next
    # bar's value in next_outcome and, if the wide view has no meaning for it, supports_wide = False.
    needs_additional_data = False
    supports_wide = True

    def __post_init__(self):
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {self.scale!r}")
        check_unit(self.unit)

    def forward_make_y(self, r_next, base_variance=None) -> np.ndarray:
        raise NotImplementedError

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        raise NotImplementedError

    def ewma_base_variance(self, returns: pd.DataFrame, additional: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """None: this target has no base variance (RatioToBase and RelativeRvChange override)."""
        return None

    def next_outcome(self, returns: pd.DataFrame, additional: pd.DataFrame | None) -> pd.DataFrame | None:
        """
        None: forward_make_y is made from the next return (r_next), like every target above. A target made from the
        additional feature data returns here, per row, the value of the NEXT bar that forward_make_y takes in place of r_next
        (same columns as `returns`; the portfolio column may be NaN).
        """
        return None


def _additional_tickers_frame(returns: pd.DataFrame, additional: pd.DataFrame | None, tag: str) -> pd.DataFrame:
    """The 'tickers' scope series `tag` of the additional feature data on the dates of `returns`, with a NaN portfolio column."""
    if additional is None:
        raise ValueError(f"The target needs the additional feature {tag!r} but the calendar has no additional_feature_data.")
    if tag not in additional.columns.get_level_values(0):
        raise ValueError(f"No additional feature {tag!r} (has {list(additional.columns.get_level_values(0).unique())}).")
    part = additional[tag]
    tickers = [c for c in returns.columns if c != "portfolio_return"]
    if set(part.columns) != set(tickers):
        raise ValueError(f"The additional feature {tag!r} has the scope {list(part.columns)}; the target needs one series per ticker {tickers}.")
    frame = part.reindex(returns.index)[tickers].copy()
    frame["portfolio_return"] = np.nan
    return frame


@dataclass(frozen=True)
class SquaredReturn(Target):
    """y = r_next^2 * scale (the original target)."""

    def forward_make_y(self, r_next, base_variance=None) -> np.ndarray:
        return (np.asarray(r_next, dtype=float) ** 2) * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        return np.asarray(prediction, dtype=float) / self.scale


@dataclass(frozen=True)
class RatioToBase(Target):
    """
    y = r_next^2 / base_variance: the model learns a multiplicative correction to the EWMA variance, so the
    level (and the scale difference between tickers) comes from the base and a prediction of 1 IS the EWMA.
    clip: the ratio is capped at this value for TRAINING only (it is heavy tailed: one huge move on a quiet
    day gives a ratio of 50+); the inverse is not changed.
    """
    needs_base = True

    base_lam: float = 0.94
    min_periods: int = 30
    clip: float | None = None

    def ewma_base_variance(self, returns: pd.DataFrame, additional: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        One-step EWMA variance per series: row f is the variance forecast made at f for the bar f+1
        (RiskMetrics recursion s2_{f+1} = lam * s2_f + (1 - lam) * r_f^2, i.e. ewm(alpha=1-lam, adjust=False)
        of r^2). Past data only, so no look-ahead; the first `min_periods` rows are NaN (warmup bars warm it up).
        """
        return (returns ** 2).ewm(alpha=1 - self.base_lam, adjust=False, min_periods=self.min_periods).mean()

    def forward_make_y(self, r_next, base_variance=None) -> np.ndarray:
        ratio = (np.asarray(r_next, dtype=float) ** 2) / self._need(base_variance)
        ratio = ratio if self.clip is None else np.minimum(ratio, self.clip)
        return ratio * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        return (np.asarray(prediction, dtype=float) / self.scale) * self._need(base_variance)

    @staticmethod
    def _need(base_variance) -> np.ndarray:
        if base_variance is None:
            raise ValueError("RatioToBase needs base_variance (it was not built for these rows).")
        return np.asarray(base_variance, dtype=float)


@dataclass(frozen=True)
class RealizedVariance(Target):
    """
    y = the next bar's realized variance (additional feature data tag 'rv': the sum of squared 5-minute returns of that
    day, data_general/RV_sip.csv) * scale; variance = prediction / scale. Long format only (there is no RV of the portfolio).
    The RV covers the trading session only, so it is a little below the variance of the close-to-close return; the
    quantile of the standardized residuals absorbs that factor, as it does for LogVariance.
    """
    needs_additional_data = True
    supports_wide = False
    tag = "rv"

    def next_outcome(self, returns, additional):
        return _additional_tickers_frame(returns, additional, self.tag).shift(-1)

    def forward_make_y(self, next_rv, base_variance=None) -> np.ndarray:
        return np.asarray(next_rv, dtype=float) * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        return np.asarray(prediction, dtype=float) / self.scale


@dataclass(frozen=True)
class LogRealizedVariance(Target):
    """
    y = log of the next bar's realized variance * scale; variance = exp(prediction / scale), so it is always positive (a target in levels can be forecast below zero, which the
    variance floor turns into a standardized residual of 1e6 and a VaR of hundreds of percent). exp of a mean of logs is below the mean variance by a roughly constant factor:
    the standardized-residual quantile absorbs it, as for LogVariance. Long format only.
    """
    needs_additional_data = True
    supports_wide = False
    tag = "rv"

    def next_outcome(self, returns, additional):
        return _additional_tickers_frame(returns, additional, self.tag).shift(-1)

    def forward_make_y(self, next_rv, base_variance=None) -> np.ndarray:
        return np.log(np.asarray(next_rv, dtype=float)) * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        return np.exp(np.clip(np.asarray(prediction, dtype=float) / self.scale, -50.0, 50.0))


@dataclass(frozen=True)
class RelativeRvChange(Target):
    """
    dRV_rel_next: y = RV_next / RV - 1 (the relative change of the realized variance from today to the next bar) * scale,
    where RV is today's, known at the feature date (the base). The model learns the change; variance = RV * (1 + prediction / scale).
    A prediction below -scale would give a negative variance: the variance floor check catches it. Long format only.
    """
    needs_base = True
    needs_additional_data = True
    supports_wide = False
    tag = "rv"

    def ewma_base_variance(self, returns, additional=None):
        """Not an EWMA here: the base is today's RV (the name is shared with RatioToBase)."""
        return _additional_tickers_frame(returns, additional, self.tag)

    def next_outcome(self, returns, additional):
        return _additional_tickers_frame(returns, additional, self.tag).shift(-1)

    def forward_make_y(self, next_rv, base_variance=None) -> np.ndarray:
        if base_variance is None:
            raise ValueError("RelativeRvChange needs base_variance (today's RV).")
        return (np.asarray(next_rv, dtype=float) / np.asarray(base_variance, dtype=float) - 1.0) * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        if base_variance is None:
            raise ValueError("RelativeRvChange needs base_variance (today's RV).")
        return np.asarray(base_variance, dtype=float) * (1.0 + np.asarray(prediction, dtype=float) / self.scale)


@dataclass(frozen=True)
class LogVariance(Target):
    """
    y = log(r_next^2 + eps). E[log r^2] is below log E[r^2] by a roughly constant amount, which only moves
    every predicted variance by the same factor: the standardized-residual quantile absorbs that, the VaR does not
    need a correction (a variance-accuracy score like QLIKE would).
    """
    eps: float = 1e-8

    def forward_make_y(self, r_next, base_variance=None) -> np.ndarray:
        return np.log(np.asarray(r_next, dtype=float) ** 2 + self.eps) * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        return np.exp(np.clip(np.asarray(prediction, dtype=float) / self.scale, -50.0, 50.0))


@dataclass(frozen=True)
class AbsReturn(Target):
    """y = |r_next| (a volatility, less heavy tailed than r^2); variance = prediction^2."""

    def forward_make_y(self, r_next, base_variance=None) -> np.ndarray:
        return np.abs(np.asarray(r_next, dtype=float)) * self.scale

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        return (np.asarray(prediction, dtype=float) / self.scale) ** 2
