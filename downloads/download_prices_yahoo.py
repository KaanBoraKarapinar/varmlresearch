"""
Daily adjusted close prices from Yahoo Finance, one CSV like data_general/4_ticker_data_2010_2026.csv
(Date index, one column per ticker, adjusted for splits and dividends).

    python downloads/download_prices_yahoo.py

The 5-minute bars and the realized variance of the same tickers come from downloads/alpaca_calculate_rv.py.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path(__file__).resolve().parent.parent / "data_general"       # the project root's data_general (this script lives in downloads/)
START, END = "2010-01-04", "2026-07-01"          # END is exclusive: the last bar is 2026-06-30 (the end of the 4-ticker file)
UNIVERSES = {
    "13tickers": ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "NFLX", "AMZN", "HD", "MCD", "PG", "KO", "WMT"],
}
UNIVERSES["23tickers"] = UNIVERSES["13tickers"] + ["XOM", "CVX", "JPM", "GS", "BAC", "V", "JNJ", "UNH", "LLY", "MRK"]   # the 13 + 10 more
UNIVERSES["49tickers"] = UNIVERSES["23tickers"] + [      # the 23 + 26 more (the same list as in alpaca_calculate_rv.py; no LIN: its realized variance starts on 2018-10-31)
    "ORCL", "CRM", "AMD", "CSCO", "DIS", "TMUS", "TSLA", "LOW", "PEP", "COST", "COP", "BRK.B", "MS", "MA", "ABBV", "TMO", "ABT",
    "CAT", "GE", "RTX", "HON", "UNP", "NEE", "DUK", "SHW", "PLD",
]
UNIVERSE = "49tickers"
BASE_FILES = {"49tickers": "50_ticker_data_2010_2026.csv"}      # the tickers of this file (the 49 + LIN, downloaded first) are kept as they are (adjusted prices change a little with every new dividend): only the others are downloaded
YAHOO_SYMBOL = {"BRK.B": "BRK-B"}                              # Yahoo writes the class B share with a dash; the files keep the Alpaca name


def download_prices(tickers: list[str]) -> pd.DataFrame:
    symbols = [YAHOO_SYMBOL.get(t, t) for t in tickers]
    raw = yf.download(symbols, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    prices = raw[symbols].copy()                  # the order of `tickers`, not yfinance's alphabetical order
    prices.columns = tickers
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices.dropna(how="all")


if __name__ == "__main__":
    tickers = UNIVERSES[UNIVERSE]
    base_path = OUT_DIR / BASE_FILES[UNIVERSE] if UNIVERSE in BASE_FILES else None
    base = pd.read_csv(base_path, parse_dates=["Date"], index_col="Date") if base_path is not None and base_path.exists() else pd.DataFrame()
    kept = [t for t in tickers if t in base.columns]
    to_download = [t for t in tickers if t not in kept]
    new = download_prices(to_download) if to_download else pd.DataFrame(index=base.index)
    prices = pd.concat([base[kept], new], axis=1).loc[:, tickers] if kept else new
    print(f"kept {len(kept)} tickers of {base_path.name if base_path is not None else '-'}, downloaded {len(tickers) - len(kept)}")
    path = OUT_DIR / f"{len(tickers)}_ticker_data_2010_2026.csv"
    prices.to_csv(path)
    print(f"{prices.shape[0]} days x {prices.shape[1]} tickers -> {path}")
    first_valid = prices.apply(lambda column: column.first_valid_index().date())
    print("first day of each ticker:", first_valid.to_dict())
    print("missing values per ticker:", prices.isna().sum().to_dict())
