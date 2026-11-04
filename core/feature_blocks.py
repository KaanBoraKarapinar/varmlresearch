"""
Feature definitions as small blocks. ONE spec makes both the wide and the long feature matrix:

    spec = FeatureSpec.default(unit="percent", windows=(5, 10, 21), ewma_span=21, sq_return_lags=None)
    spec = FeatureSpec((Returns(), AbsReturn(), RollingStats((5, 21)), EwmaVol(21), TickerOneHot()), unit="percent")

unit (percent | log) is required and must be the calendar's unit.

Every block is computed on the date x series returns frame (the tickers and "portfolio_return") and yields
named features; feature_builders turns them into the wide view (every series' features side by side,
"<series>_<feature>") or the long view (one row per (Date, ticker), plain feature names).

Blocks only look at the past (rolling / ewm / shift(+)), so computing them once on warmup + train + forecast
and slicing gives the same train rows as a train-only computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

import numpy as np
import pandas as pd

from .units import check_unit

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
    # the block's value depends on WHICH tickers and weights the calendar has (the portfolio's own return, the other tickers, the set of
    # tickers): a model trained with it cannot be applied to a calendar with another portfolio (see walk_forward.train_walk_forward, pretrained)
    portfolio_dependent: ClassVar[bool] = False
    # the block reads the calendar's additional feature data: compute(returns, tickers, additional)
    needs_additional_data: ClassVar[bool] = False
    # the wide view gets ONE column (the feature's name) instead of one per series: for a feature that is the same for every series
    market_wide: ClassVar[bool] = False

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
    portfolio_dependent: ClassVar[bool] = True      # the other tickers are the calendar's tickers
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
    portfolio_dependent: ClassVar[bool] = True      # the portfolio is the calendar's tickers with its weights
    span: int = 21

    def compute(self, returns, tickers):
        market = returns["portfolio_return"].ewm(span=self.span).std()
        return [("market_ewma_vol", pd.DataFrame({series: market for series in returns.columns}))]


@dataclass(frozen=True)
class TickerOneHot(FeatureBlock):
    """ticker_<NAME> = 1 on that ticker's rows, 0 on the others (long only): a level per ticker for a pooled model."""
    long_only: ClassVar[bool] = True
    portfolio_dependent: ClassVar[bool] = True      # one column per ticker of the calendar: another set of tickers is another set of columns

    def compute(self, returns, tickers):
        out = []
        for ticker in tickers:
            one_hot = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
            one_hot[ticker] = 1.0
            out.append((f"ticker_{ticker}", one_hot))
        return out


def _additional_part(additional, tag: str, scope_columns: list, returns: pd.DataFrame) -> pd.DataFrame:
    """The columns `scope_columns` of one tag of the additional feature data, on the dates of `returns`."""
    if additional is None:
        raise ValueError(f"The block asks for the additional feature {tag!r} but the calendar has no additional_feature_data.")
    if tag not in additional.columns.get_level_values(0):
        raise ValueError(f"No additional feature {tag!r} (has {list(additional.columns.get_level_values(0).unique())}).")
    part = additional[tag]
    if set(part.columns) != set(scope_columns):
        raise ValueError(f"The additional feature {tag!r} has the scope {list(part.columns)}, this block needs {list(scope_columns)}.")
    return part.reindex(returns.index)[scope_columns]


@dataclass(frozen=True)
class AdditionalFeatureBlock(FeatureBlock):
    """
    A feature made from the calendar's additional feature data. A subclass IS the feature (Vix, Volume, YieldSpread10y3m): it
    names the `tags` it reads (the series in the additional data; a class constant or, when it depends on the block's
    settings, a property) and their `scope`, and `features(values)` makes the feature columns from them.

    scope "market": one series per tag for the whole portfolio. Each frame in `values` has the same series in every
        column (so the long view gives it to every ticker row); the wide view gets ONE column per feature.
    scope "tickers": one series per ticker. Each frame in `values` has one column per ticker; the long view takes the
        ticker's own column, the wide view has '<TICKER>_<feature>' (no portfolio column).

    values: {tag: DataFrame}, the series as they are known on each bar (available_after_bars is already applied), on the
    bars of `returns`; use only the past (shift, rolling, pct_change) like every block.
    """
    needs_additional_data: ClassVar[bool] = True
    scope: ClassVar[str]       # "market" | "tickers"
    market_wide: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "scope"):
            if cls.scope not in ("market", "tickers"):
                raise ValueError(f"{cls.__name__}.scope must be 'market' or 'tickers', got {cls.scope!r}.")
            cls.market_wide = cls.scope == "market"

    def features(self, values: dict) -> list:
        """-> [(feature_name, DataFrame shaped like a frame of values)]"""
        raise NotImplementedError

    def compute(self, returns, tickers, additional):
        values = {}
        for tag in self.tags:
            if self.scope == "market":
                series = _additional_part(additional, tag, ["market"], returns)["market"]
                values[tag] = pd.DataFrame({name: series for name in returns.columns})
            else:
                values[tag] = _additional_part(additional, tag, list(tickers), returns)
        return self.features(values)


@dataclass(frozen=True)
class Vix(AdditionalFeatureBlock):
    """
    The VIX (additional data tag 'vix', scope market): its level as `vix`, and the percent change over each bar count in
    `changes` as `vix_change_<n>` (e.g. changes=(5,): the change against 5 bars earlier).
    """
    tags: ClassVar[tuple] = ("vix",)
    scope: ClassVar[str] = "market"
    level: bool = True
    changes: tuple = ()

    def features(self, values):
        vix = values["vix"]
        out = [("vix", vix)] if self.level else []
        return out + [(f"vix_change_{n}", vix.pct_change(n)) for n in self.changes]


@dataclass(frozen=True)
class Sp500Return(AdditionalFeatureBlock):
    """
    The S&P 500 index return (additional data tag 'sp500', the index level, scope market) over each bar count in `bars`:
    `sp500_return_<n>`, the simple (percent change) return against n bars earlier.
    """
    tags: ClassVar[tuple] = ("sp500",)
    scope: ClassVar[str] = "market"
    bars: tuple = (1, 5, 21)

    def features(self, values):
        sp500 = values["sp500"]
        return [(f"sp500_return_{n}", sp500.pct_change(n)) for n in self.bars]


@dataclass(frozen=True)
class RvHar(AdditionalFeatureBlock):
    """
    The daily realized variance of each ticker (additional data tag 'rv', scope tickers: the sum of squared 5-minute
    returns of that day, data_general/RV_sip.csv) and its two HAR averages: `rv` (the day), `rv_week` (mean of the last
    `week` days), `rv_month` (mean of the last `month` days). All of them end on the bar's own day, so they are known at its close.
    The change to the NEXT day's RV (dRV_rel_next) is not a feature (it looks forward): it is the target RelativeRvChange.
    """
    tags: ClassVar[tuple] = ("rv",)
    scope: ClassVar[str] = "tickers"
    week: int = 5
    month: int = 22

    def features(self, values):
        rv = values["rv"]
        return [("rv", rv), ("rv_week", rv.rolling(self.week).mean()), ("rv_month", rv.rolling(self.month).mean())]


@dataclass(frozen=True)
class LogRvHar(AdditionalFeatureBlock):
    """
    The HAR regressors of RvHar in logs: `log_rv` (the log of the day's realized variance), `log_rv_week` and `log_rv_month` (the logs of its mean over the last `week` / `month`
    days). With the target LogRealizedVariance a linear model on these is the log-HAR-RV of Corsi (2009), whose forecast exp(prediction) is always positive (a linear HAR in
    levels can forecast a negative variance after a spike, which the variance floor turns into a standardized residual of 1e6).
    """
    tags: ClassVar[tuple] = ("rv",)
    scope: ClassVar[str] = "tickers"
    week: int = 5
    month: int = 22

    def features(self, values):
        rv = values["rv"]
        return [("log_rv", np.log(rv)), ("log_rv_week", np.log(rv.rolling(self.week).mean())), ("log_rv_month", np.log(rv.rolling(self.month).mean()))]


@dataclass(frozen=True)
class RvLags(AdditionalFeatureBlock):
    """
    The realized variance of each ticker n bars earlier (additional data tag 'rv'): `rv_lag_<n>` for every n in `lags`
    (lags=(1, 2, 3): yesterday and the two days before). Use it next to RvHar, which has the day itself and the averages.
    """
    tags: ClassVar[tuple] = ("rv",)
    scope: ClassVar[str] = "tickers"
    lags: tuple = (1, 2, 3)

    def features(self, values):
        rv = values["rv"]
        return [(f"rv_lag_{n}", rv.shift(n)) for n in self.lags]


@dataclass(frozen=True)
class YieldSpread10y3m(AdditionalFeatureBlock):
    """10-year minus 3-month Treasury yield, in percentage points (tags us10y, us3m)."""
    tags: ClassVar[tuple] = ("us10y", "us3m")
    scope: ClassVar[str] = "market"

    def features(self, values):
        return [("yield_spread_10y_3m", values["us10y"] - values["us3m"])]


@dataclass(frozen=True)
class YieldSpread10y5y(AdditionalFeatureBlock):
    """10-year minus 5-year Treasury yield, in percentage points (tags us10y, us5y)."""
    tags: ClassVar[tuple] = ("us10y", "us5y")
    scope: ClassVar[str] = "market"

    def features(self, values):
        return [("yield_spread_10y_5y", values["us10y"] - values["us5y"])]


@dataclass(frozen=True)
class YieldSpread30y10y(AdditionalFeatureBlock):
    """30-year minus 10-year Treasury yield, in percentage points (tags us30y, us10y)."""
    tags: ClassVar[tuple] = ("us30y", "us10y")
    scope: ClassVar[str] = "market"

    def features(self, values):
        return [("yield_spread_30y_10y", values["us30y"] - values["us10y"])]


@dataclass(frozen=True)
class Volume(AdditionalFeatureBlock):
    """
    The trading volume of each ticker (additional data tag 'volume', scope tickers). The level drifts over the years, so it is
    not a feature on its own: `volume_ratio_to_mean_<n>` is the volume over its mean of the last n bars, `volume_change_<n>`
    the percent change against n bars earlier.
    """
    tags: ClassVar[tuple] = ("volume",)
    scope: ClassVar[str] = "tickers"
    ratio_to_mean: tuple = (21,)
    changes: tuple = ()

    def features(self, values):
        volume = values["volume"]
        out = [(f"volume_ratio_to_mean_{n}", volume / volume.rolling(n).mean()) for n in self.ratio_to_mean]
        return out + [(f"volume_change_{n}", volume.pct_change(n)) for n in self.changes]


@dataclass(frozen=True)
class FeatureSpec:
    blocks: tuple
    # the unit of the returns the features are made from (percent | log, see core/units.py): required, and
    # it must be the calendar's unit (build_features checks)
    unit: str = field(kw_only=True)

    def __post_init__(self):
        check_unit(self.unit)

    @staticmethod
    def default(
        *, unit: str, windows: tuple = (5, 10, 21), ewma_span: int = 21, sq_return_lags: tuple | None = None,
    ) -> "FeatureSpec":
        """The feature set the project used before blocks existed (same columns, same order)."""
        blocks = [Returns(), AbsReturn(), RollingStats(tuple(windows)), EwmaVol(ewma_span)]
        if sq_return_lags:
            blocks.append(SquaredReturnLags(tuple(sq_return_lags)))
        blocks.append(MarketEwmaVol(ewma_span))
        return FeatureSpec(tuple(blocks), unit=unit)

    def with_blocks(self, *extra: FeatureBlock) -> "FeatureSpec":
        return FeatureSpec(self.blocks + tuple(extra), unit=self.unit)

    @property
    def portfolio_dependent_blocks(self) -> tuple[str, ...]:
        """The names of the blocks whose value depends on the calendar's tickers / weights (a model trained with them is tied to that portfolio)."""
        return tuple(type(block).__name__ for block in self.blocks if block.portfolio_dependent)

    def portable(self) -> "FeatureSpec":
        """The same spec without the portfolio dependent blocks: a model trained with it can be applied to a calendar with another portfolio."""
        return FeatureSpec(tuple(block for block in self.blocks if not block.portfolio_dependent), unit=self.unit)

    def describe(self) -> str:
        """The blocks with their settings, the one place this text is made: Returns+RollingStats(windows=(5,10,21))+EwmaVol(span=21)+..."""
        def block_text(block) -> str:
            block_fields = fields(block)
            if not block_fields:
                return type(block).__name__
            arguments = ",".join(f"{f.name}={getattr(block, f.name)!r}" for f in block_fields).replace(" ", "")
            return f"{type(block).__name__}({arguments})"
        return "+".join(block_text(block) for block in self.blocks)

    def compute(self, returns: pd.DataFrame, tickers: list[str], additional: pd.DataFrame | None = None) -> dict:
        """-> {feature_name: (DataFrame, long_only, market_wide)} in block order. additional: the calendar's additional feature data for these bars."""
        features = {}
        for block in self.blocks:
            produced = block.compute(returns, tickers, additional) if block.needs_additional_data else block.compute(returns, tickers)
            for name, values in produced:
                if name in features:
                    raise ValueError(f"feature {name!r} is produced by two blocks.")
                features[name] = (values, block.long_only, block.market_wide)
        return features
