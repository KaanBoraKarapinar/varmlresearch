"""
Feature definitions as small blocks. ONE spec makes both the wide and the long feature matrix:

    spec = FeatureSpec.default(windows=(5, 10, 21), ewma_span=21, sq_return_lags=None)
    spec = FeatureSpec((Returns(), AbsReturn(), RollingStats((5, 21)), EwmaVol(21), TickerOneHot()))

Every block is computed on the date x series returns frame (the tickers and "portfolio_return") and yields
named features; feature_builders turns them into the wide view (every series' features side by side,
"<series>_<feature>") or the long view (one row per (Date, ticker), plain feature names).

Blocks only look at the past (rolling / ewm / shift(+)), so computing them once on warmup + train + forecast
and slicing gives the same train rows as a train-only computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FeatureBlock:
    """
    compute -> [(feature_name, DataFrame)], the DataFrame has the same columns as returns (one per series):
    a ticker's long row takes its own column, the wide view puts every column side by side.

    long_only: the block is left out of the wide view. True when the wide view already has the information
    (MarketEwmaVol is the portfolio's own ewma vol, wide has it as portfolio_return_ewma_vol) or when it has no
    meaning there (TickerOneHot: a wide row is a whole day, not a ticker).
    """
    long_only: ClassVar[bool] = False

    def compute(self, returns: pd.DataFrame, tickers: list[str]) -> list:
        raise NotImplementedError


@dataclass(frozen=True)
class Returns(FeatureBlock):
    """The raw return itself, named `r` (wide: the series name, e.g. "AAPL", "portfolio_return")."""

    def compute(self, returns, tickers):
        return [("r", returns)]


@dataclass(frozen=True)
class AbsReturn(FeatureBlock):
    def compute(self, returns, tickers):
        return [("abs_r", returns.abs())]


@dataclass(frozen=True)
class RollingVol(FeatureBlock):
    """Realized volatility: the std of the last w returns, for every window."""
    windows: tuple = (5, 10, 21)

    def compute(self, returns, tickers):
        return [(f"real_vol_{w}", returns.rolling(w).std()) for w in self.windows]


@dataclass(frozen=True)
class RollingMeanAbs(FeatureBlock):
    """The mean of the last w absolute returns, for every window."""
    windows: tuple = (5, 10, 21)

    def compute(self, returns, tickers):
        return [(f"mean_abs_{w}", returns.abs().rolling(w).mean()) for w in self.windows]


@dataclass(frozen=True)
class RollingSkew(FeatureBlock):
    """The skew of the last w returns, for every window."""
    windows: tuple = (5, 10, 21)

    def compute(self, returns, tickers):
        return [(f"skew_{w}", returns.rolling(w).skew()) for w in self.windows]


@dataclass(frozen=True)
class RollingStats(FeatureBlock):
    """
    RollingVol + RollingMeanAbs + RollingSkew in the old order (per window: vol, mean |r|, skew). The default
    spec uses it only to keep the column order of the old builders; to try them one by one, put the three
    separate blocks in the spec instead.
    """
    windows: tuple = (5, 10, 21)

    def compute(self, returns, tickers):
        out = []
        for w in self.windows:
            for block in (RollingVol((w,)), RollingMeanAbs((w,)), RollingSkew((w,))):
                out += block.compute(returns, tickers)
        return out


@dataclass(frozen=True)
class EwmaVol(FeatureBlock):
    span: int = 21

    def compute(self, returns, tickers):
        return [("ewma_vol", returns.ewm(span=self.span).std())]


@dataclass(frozen=True)
class SquaredReturnLags(FeatureBlock):
    lags: tuple = (1, 2, 3)

    def compute(self, returns, tickers):
        squared = returns ** 2
        return [(f"r2_lag_{lag}", squared.shift(lag)) for lag in self.lags]


@dataclass(frozen=True)
class EwmaVariance(FeatureBlock):
    """The RiskMetrics EWMA variance (the same base RatioToBase divides by) as a feature."""
    lam: float = 0.94
    min_periods: int = 30

    def compute(self, returns, tickers):
        ewma = (returns ** 2).ewm(alpha=1 - self.lam, adjust=False, min_periods=self.min_periods).mean()
        return [("ewma_var", ewma)]


@dataclass(frozen=True)
class CrossSectionVol(FeatureBlock):
    """
    Rolling vol of the OTHER tickers (their mean): what the rest of the market is doing. For the portfolio
    column it is the mean over all tickers.
    """
    window: int = 21

    def compute(self, returns, tickers):
        vol = returns[tickers].rolling(self.window).std()
        out = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for ticker in tickers:
            others = [t for t in tickers if t != ticker]
            out[ticker] = vol[others].mean(axis=1) if others else vol[ticker]
        out["portfolio_return"] = vol.mean(axis=1)
        return [(f"xs_vol_{self.window}", out)]


@dataclass(frozen=True)
class MarketEwmaVol(FeatureBlock):
    """The portfolio's ewma vol: the same value in every column, so every ticker row gets it (long only)."""
    long_only: ClassVar[bool] = True
    span: int = 21

    def compute(self, returns, tickers):
        market = returns["portfolio_return"].ewm(span=self.span).std()
        return [("market_ewma_vol", pd.DataFrame({series: market for series in returns.columns}))]


@dataclass(frozen=True)
class TickerOneHot(FeatureBlock):
    """ticker_<NAME> = 1 on that ticker's rows, 0 on the others (long only): a level per ticker for a pooled model."""
    long_only: ClassVar[bool] = True

    def compute(self, returns, tickers):
        out = []
        for ticker in tickers:
            one_hot = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
            one_hot[ticker] = 1.0
            out.append((f"ticker_{ticker}", one_hot))
        return out


@dataclass(frozen=True)
class FeatureSpec:
    blocks: tuple

    @staticmethod
    def default(windows: tuple = (5, 10, 21), ewma_span: int = 21, sq_return_lags: tuple | None = None) -> "FeatureSpec":
        """The feature set the project used before blocks existed (same columns, same order)."""
        blocks = [Returns(), AbsReturn(), RollingStats(tuple(windows)), EwmaVol(ewma_span)]
        if sq_return_lags:
            blocks.append(SquaredReturnLags(tuple(sq_return_lags)))
        blocks.append(MarketEwmaVol(ewma_span))
        return FeatureSpec(tuple(blocks))

    def with_blocks(self, *extra: FeatureBlock) -> "FeatureSpec":
        return FeatureSpec(self.blocks + tuple(extra))

    def compute(self, returns: pd.DataFrame, tickers: list[str]) -> dict:
        """-> {feature_name: (DataFrame, long_only)} in block order."""
        features = {}
        for block in self.blocks:
            for name, values in block.compute(returns, tickers):
                if name in features:
                    raise ValueError(f"feature {name!r} is produced by two blocks.")
                features[name] = (values, block.long_only)
        return features
