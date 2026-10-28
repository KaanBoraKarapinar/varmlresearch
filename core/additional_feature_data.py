"""
Additional feature data: daily data that is not a price (VIX, a rate, volume, ...), read from CSV files into ONE
DataFrame with two-level columns (tag, scope), aligned to the price bars.

    sources = [
        AdditionalFeatureSource("vix", "data_general/vix.csv", "market", column="Close", available_after_bars=0),
        AdditionalFeatureSource("volume", "data_general/volume.csv", "tickers", available_after_bars=0),
    ]
    additional = load_additional_feature_data(sources, price_index, tickers)

    additional["vix"]       one column "market"
    additional["volume"]    one column per ticker

scope "market": one series for the whole portfolio (VIX, a rate). scope "tickers": one series per ticker, the CSV has
one column per ticker (named like the price columns).

The CSV: a `Date` column (ISO dates), one tag per file, one value column ("market") or one column per ticker ("tickers").

available_after_bars says how long after its date a value is known: 0 = at that date's close (it can be a feature of
that date's row), 1 = one bar later (the value of date t is a feature of the next bar). It has no default on purpose:
a wrong 0 lets a feature see a number that is not published yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SCOPES = ("market", "tickers")
MARKET = "market"                  # the scope label of a market-wide series (the second column level)


@dataclass(frozen=True)
class AdditionalFeatureSource:
    tag: str                       # the name of the series; it is the feature name
    path: str                      # the CSV file
    scope: str                     # "market" | "tickers"
    column: str | None = None      # scope "market": the CSV column to use (optional when the file has only one)
    # how many bars after its date the value is known (0 = at that date's close); required
    available_after_bars: int = field(kw_only=True)

    def __post_init__(self):
        if not self.tag or not str(self.tag).strip():
            raise ValueError("An additional feature source needs a tag.")
        if self.scope not in SCOPES:
            raise ValueError(f"{self.tag}: scope must be one of {SCOPES}, got {self.scope!r}.")
        if isinstance(self.available_after_bars, bool) or not isinstance(self.available_after_bars, int) or self.available_after_bars < 0:
            raise ValueError(f"{self.tag}: available_after_bars must be an int >= 0, got {self.available_after_bars!r}.")
        if self.scope == "tickers" and self.column is not None:
            raise ValueError(f"{self.tag}: scope 'tickers' reads one column per ticker, `column` is only for scope 'market'.")


def check_additional_feature_data(additional: pd.DataFrame, tickers: list[str]) -> None:
    """The shape every additional feature table must have: a sorted date index, columns (tag, scope), each tag 'market' or exactly the tickers."""
    if not isinstance(additional.index, pd.DatetimeIndex) or not additional.index.is_monotonic_increasing or not additional.index.is_unique:
        raise ValueError("additional feature data needs a sorted, unique DatetimeIndex.")
    if not isinstance(additional.columns, pd.MultiIndex) or additional.columns.nlevels != 2:
        raise ValueError("additional feature data needs two-level columns (tag, scope).")
    for tag in additional.columns.get_level_values(0).unique():
        scopes = set(additional[tag].columns)
        if scopes != {MARKET} and scopes != set(tickers):
            raise ValueError(f"{tag}: the scope must be '{MARKET}' or exactly the tickers {sorted(tickers)}, got {sorted(scopes)}.")


def _read_source(source: AdditionalFeatureSource, tickers: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(source.path, parse_dates=["Date"], index_col="Date")
    if not frame.index.is_unique:
        raise ValueError(f"{source.tag}: {source.path} has a date twice.")
    frame = frame.sort_index()
    if source.scope == "market":
        if source.column is not None:
            if source.column not in frame.columns:
                raise ValueError(f"{source.tag}: column {source.column!r} is not in {source.path} (has {list(frame.columns)}).")
            series = frame[source.column]
        elif frame.shape[1] == 1:
            series = frame.iloc[:, 0]
        else:
            raise ValueError(f"{source.tag}: {source.path} has several columns {list(frame.columns)}: say which one with column=.")
        frame = series.rename(MARKET).to_frame()
    else:
        missing = [t for t in tickers if t not in frame.columns]
        if missing:
            raise ValueError(f"{source.tag}: {source.path} has no column for {missing}; scope 'tickers' needs one column per ticker.")
        frame = frame[list(tickers)]
    try:
        return frame.astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{source.tag}: {source.path} has a value that is not a number ({error}).") from error


def _align(source: AdditionalFeatureSource, frame: pd.DataFrame, price_index: pd.DatetimeIndex, needed_from: pd.Timestamp) -> pd.DataFrame:
    """
    The value known as of each price bar (forward fill only, never backward), then moved available_after_bars bars later.
    The series must cover the bars from needed_from (the first bar any calendar reads) to the last price bar; before
    needed_from it may be missing (NaN), such bars are never read.
    """
    if frame.index[0] > needed_from:
        raise ValueError(
            f"{source.tag}: starts {frame.index[0].date()}, after {needed_from.date()}, the first bar the windows read; "
            f"the early windows would have no value (start the walk-forward later, or use a longer series)."
        )
    if frame.index[-1] < price_index[-1]:
        raise ValueError(
            f"{source.tag}: ends {frame.index[-1].date()}, before the last price bar {price_index[-1].date()}; "
            f"forward filling would carry a stale value."
        )
    aligned = frame.reindex(frame.index.union(price_index)).ffill().reindex(price_index)
    if aligned.loc[needed_from:].isna().any().any():
        raise ValueError(f"{source.tag}: has missing values that forward filling cannot fill (an all-empty column).")
    return aligned.shift(source.available_after_bars)


def load_additional_feature_data(
    sources, price_index: pd.DatetimeIndex, tickers: list[str], needed_from=None,
) -> pd.DataFrame:
    """
    The CSV files of `sources` as one table: index = price_index, columns = (tag, scope).
    needed_from: the first bar the windows read (None: the first price bar). A series that starts later than the
    price data is fine as long as it starts by then; its bars before it stay NaN.
    """
    sources = list(sources)
    tags = [s.tag for s in sources]
    if len(set(tags)) != len(tags):
        raise ValueError(f"A tag is used twice: {sorted({t for t in tags if tags.count(t) > 1})}.")
    price_index = pd.DatetimeIndex(price_index).sort_values()
    needed_from = price_index[0] if needed_from is None else pd.Timestamp(needed_from)
    parts = {s.tag: _align(s, _read_source(s, tickers), price_index, needed_from) for s in sources}
    additional = pd.concat(parts, axis=1)
    additional.columns.names = ["tag", "scope"]
    check_additional_feature_data(additional, tickers)
    return additional
