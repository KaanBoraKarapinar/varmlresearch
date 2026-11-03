"""
5-minute bars from Alpaca -> daily realized volatility (RV) and HAR features.
Saves one wide CSV: RV.csv (rows = date, columns = tickers).
Derived variables (RV_w, RV_m, dRV_rel_next) are NOT computed here.

Setup:
    pip install alpaca-py pandas numpy python-dotenv
    .env file in the project root (it is not tracked by git):
        APCA_API_KEY_ID=...
        APCA_API_SECRET_KEY=...

    python downloads/alpaca_calculate_rv.py        (the paths below are relative to the project root; the script moves there itself)
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent       # the project root: this script lives in downloads/
os.chdir(ROOT)

# ----------------------------- CONFIG -----------------------------
UNIVERSES = {
    "4tickers": ["AAPL", "JPM", "XOM", "KO"],     # 4_ticker_data_2010_2026.csv; its files keep the plain names (RV_sip.csv, bars_5min_sip.parquet)
    "13tickers": ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "NFLX", "AMZN", "HD", "MCD", "PG", "KO", "WMT"],   # files: RV_sip_13tickers.csv, ...
}
UNIVERSES["23tickers"] = UNIVERSES["13tickers"] + ["XOM", "CVX", "JPM", "GS", "BAC", "V", "JNJ", "UNH", "LLY", "MRK"]   # the 13 + 10 more
BASE_UNIVERSES = {"23tickers": "13tickers"}      # tickers already downloaded for the base universe are read from its bar file, not fetched again
UNIVERSE = "23tickers"
TICKERS = UNIVERSES[UNIVERSE]
SUFFIX = "" if UNIVERSE == "4tickers" else f"_{UNIVERSE}"
START = "2016-01-01"         # YYYY-MM-DD. IEX has nothing before 2020-07-27; SIP reaches back to 2016 (checked on this account)
END = "2026-07-01"           # the last price bar is 2026-06-30; None = yesterday (free plan: newest data must be 15+ min old)
FEED = DataFeed.SIP          # SIP = all US venues (IEX is a few % of the volume); the files are named after the feed
OUT_DIR = Path("./data_general")
SAVE_RAW_BARS = True        # if True: one parquet file with all 5-min bars (ticker column), kept apart from data_general
RAW_BARS_PATH = Path(f"./data_raw/bars_5min_{FEED.name.lower()}{SUFFIX}.parquet")
RV_PATH = OUT_DIR / f"RV_{FEED.name.lower()}{SUFFIX}.csv"
TZ = "America/New_York"
MIN_BARS_PER_DAY = 36        # a full session has 78 five-minute bars (SIP: half days 44-76, IEX: 42); the holes of the SIP feed have 0, 1 or 15 bars
# ------------------------------------------------------------------


def fetch_bars(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """5-min regular-session bars, MultiIndex (symbol, timestamp in New York time)."""
    client = StockHistoricalDataClient(
        os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    )
    end_dt = (
        datetime.strptime(end, "%Y-%m-%d")
        if end
        else datetime.now() - timedelta(days=1)
    )
    parts = []
    for ticker in tickers:      # one ticker at a time: the extended-hours bars of all of them together would be several million rows
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=end_dt,
            adjustment=Adjustment.ALL,
            feed=FEED,
        )
        df = client.get_stock_bars(req).df  # SDK handles pagination
        if df.empty:
            print(f"{ticker}: no bars", flush=True)
            continue
        for sym, g in df.groupby(level="symbol"):
            g = g.droplevel("symbol")
            g.index = g.index.tz_convert(TZ)
            g = g.between_time("09:30", "15:55")  # regular session, bar start times
            g.index = pd.MultiIndex.from_product([[sym], g.index], names=["symbol", "timestamp"])
            parts.append(g)
            print(f"{ticker}: {len(g)} regular-session bars {g.index.get_level_values('timestamp').min().date()} -> {g.index.get_level_values('timestamp').max().date()}", flush=True)
    return pd.concat(parts)


def daily_rv(bars: pd.DataFrame) -> pd.Series:
    """RV_t = sum of squared intraday 5-min log returns (variance, not sqrt). One ticker."""
    bars = bars.copy()
    bars["day"] = bars.index.date
    r = np.log(bars["close"]).groupby(bars["day"]).diff()
    # First bar of each day: open -> close, so the overnight gap does not leak in
    first = bars.groupby("day").cumcount() == 0
    r[first] = np.log(bars.loc[first, "close"] / bars.loc[first, "open"])
    rv = (r ** 2).groupby(bars["day"]).sum()
    # a day with too few bars is a hole in the feed (some days arrive with 1 or 15 bars), not a quiet day: no RV
    n_bars = bars.groupby("day").size()
    rv[n_bars < MIN_BARS_PER_DAY] = np.nan
    rv.index = pd.to_datetime(rv.index)
    return rv


def main() -> None:
    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts, reused = [], []
    base_universe = BASE_UNIVERSES.get(UNIVERSE)
    if base_universe:
        base_path = Path(f"./data_raw/bars_5min_{FEED.name.lower()}_{base_universe}.parquet")
        if base_path.exists():
            base = pd.read_parquet(base_path)
            base["date"] = pd.to_datetime(base["date"])
            reused = [t for t in TICKERS if t in set(base["ticker"])]
            parts.append(base[base["ticker"].isin(reused)].rename(columns={"ticker": "symbol", "date": "timestamp"}).set_index(["symbol", "timestamp"]))
            print(f"reused from {base_path.name}: {len(reused)} tickers, not fetched again", flush=True)
    parts.append(fetch_bars([t for t in TICKERS if t not in reused], START, END))
    bars = pd.concat(parts)
    syms = bars.index.get_level_values("symbol").unique()
    missing = [t for t in TICKERS if t not in syms]
    if missing:
        print(f"No data returned for: {missing}")

    rv = pd.DataFrame({s: daily_rv(bars.xs(s, level="symbol")) for s in TICKERS if s in syms})     # the order of TICKERS
    rv.index.name = "Date"      # like every other file of data_general (the additional feature loader reads "Date")
    rv = rv.sort_index()

    path = RV_PATH
    rv.to_csv(path)
    print(f"RV: {rv.shape[0]} days x {rv.shape[1]} tickers -> {path}")

    if SAVE_RAW_BARS:
        raw = bars.reset_index().rename(columns={"symbol": "ticker", "timestamp": "date"})
        RAW_BARS_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(RAW_BARS_PATH, index=False)
        print(f"raw bars: {len(raw)} rows -> {RAW_BARS_PATH}")


if __name__ == "__main__":
    main()