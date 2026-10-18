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

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Target:
    """
    Base class: forward_make_y(r_next) -> y and inverse_to_variance(prediction) -> variance, one subclass per target.

    scale (every target): what the model sees is y * scale, and the prediction is divided by scale again before
    it becomes a variance. It only makes the numbers friendlier to a tree (r^2 is ~1e-4) and does not change
    the variance that comes out.
    """
    scale: float = 1.0

    needs_base = False

    def __post_init__(self):
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {self.scale!r}")

    def forward_make_y(self, r_next, base_variance=None) -> np.ndarray:
        raise NotImplementedError

    def inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:
        raise NotImplementedError

    def ewma_base_variance(self, returns: pd.DataFrame) -> pd.DataFrame | None:
        """None: this target has no base variance (RatioToBase overrides)."""
        return None


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

    def ewma_base_variance(self, returns: pd.DataFrame) -> pd.DataFrame:
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
