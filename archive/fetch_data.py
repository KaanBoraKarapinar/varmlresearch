"""
Downloads daily adjusted close prices from Yahoo Finance and writes them in the same
layout as data_general/4_ticker_data.csv (Date index, one column per ticker), so the
output can be passed straight to GroundTruthCalendar as raw price data.

Usage:
    python fetch_data.py
    python fetch_data.py --start 2010-01-01 --end 2026-06-30 --out data_general/4_ticker_data_2010_2026.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "JPM", "KO", "XOM"]
START = "2010-01-01"
END = "2026-06-30"  # inclusive; yfinance's own `end` is exclusive, handled in fetch_prices
OUT_PATH = Path(__file__).resolve().parent / "data_general" / "4_ticker_data_2010_2026.csv"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # auto_adjust=True: Close is already split/dividend adjusted
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {tickers} in [{start} .. {end}].")

    prices = raw["Close"][sorted(tickers)].copy()
    prices.index = pd.DatetimeIndex(prices.index.tz_localize(None) if prices.index.tz else prices.index, name="Date")
    return prices.sort_index()


def check_prices(prices: pd.DataFrame) -> None:
    nan_counts = prices.isna().sum()
    if nan_counts.any():
        print(f"[WARNING] NaN cells per ticker:\n{nan_counts[nan_counts > 0]}")
    print(f"[INFO] {len(prices)} bars, {prices.index.min().date()} .. {prices.index.max().date()}")
    print(f"[INFO] first valid bar per ticker:\n{prices.apply(pd.Series.first_valid_index)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    prices = fetch_prices(args.tickers, args.start, args.end)
    check_prices(prices)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out)
    print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
