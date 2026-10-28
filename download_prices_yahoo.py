"""
Daily adjusted close prices from Yahoo Finance, one CSV like data_general/4_ticker_data_2010_2026.csv
(Date index, one column per ticker, adjusted for splits and dividends).

    python download_prices_yahoo.py

The 5-minute bars and the realized variance of the same tickers come from alpaca_calculate_rv.py.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path(__file__).resolve().parent / "data_general"
START, END = "2010-01-04", "2026-07-01"          # END is exclusive: the last bar is 2026-06-30 (the end of the 4-ticker file)
UNIVERSES = {
    "13tickers": ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "NFLX", "AMZN", "HD", "MCD", "PG", "KO", "WMT"],
}
UNIVERSES["23tickers"] = UNIVERSES["13tickers"] + ["XOM", "CVX", "JPM", "GS", "BAC", "V", "JNJ", "UNH", "LLY", "MRK"]   # the 13 + 10 more
UNIVERSE = "23tickers"


def download_prices(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    prices = raw[tickers].copy()                  # the order of `tickers`, not yfinance's alphabetical order
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices.dropna(how="all")


if __name__ == "__main__":
    tickers = UNIVERSES[UNIVERSE]
    prices = download_prices(tickers)
    path = OUT_DIR / f"{len(tickers)}_ticker_data_2010_2026.csv"
    prices.to_csv(path)
    print(f"{prices.shape[0]} days x {prices.shape[1]} tickers -> {path}")
    first_valid = prices.apply(lambda column: column.first_valid_index().date())
    print("first day of each ticker:", first_valid.to_dict())
    print("missing values per ticker:", prices.isna().sum().to_dict())
