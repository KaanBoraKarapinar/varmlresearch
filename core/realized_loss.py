"""
How the realized loss of a portfolio is distributed: its level, its spread and how tightly it is centered around 0.

The realized loss is the loss the VaR is scored against (-portfolio return). Like every return-sized number of the project it is a fraction: 0.0153 is
a 1.53% loss (core.units). A diversified portfolio has days that mostly cancel out between its stocks (hedging), so its loss is centered heavily
around 0: this module puts numbers on that. Pure functions on frames: no dashboard, no store.

    realized_loss_statistics(losses)    one row per portfolio (the columns of `losses`), one column per statistic
    portfolio_losses(returns, weights)  the daily loss of every portfolio from the tickers' returns
"""

from __future__ import annotations

import pandas as pd

CONCENTRATION_BANDS = (0.005, 0.01, 0.02)       # the share of days with |loss| inside +-band (0.5%, 1%, 2%)


def portfolio_losses(ticker_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    The daily realized loss of every portfolio: -(returns . weights), the weights held fixed (the calendar's definition of the portfolio return).
    ticker_returns: days x tickers, simple returns as fractions (core.units.return_to_percent of a run's own returns); weights: tickers x portfolios
    (one column per portfolio). Returns days x portfolios.
    """
    missing = set(weights.index) - set(ticker_returns.columns)
    if missing:
        raise ValueError(f"The returns have no column for the tickers {sorted(missing)}.")
    return pd.DataFrame(-(ticker_returns[list(weights.index)].to_numpy(dtype=float) @ weights.to_numpy(dtype=float)), index=ticker_returns.index, columns=weights.columns)


def realized_loss_statistics(losses: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    One row per column of `losses` (the daily realized loss of a portfolio, as fractions): days, mean, median, std, mean |loss|, the 1 / 5 / 95 / 99%
    quantiles, the worst and the best day, the share of days with |loss| inside +-0.5 / 1 / 2 percent, skew and excess kurtosis.
    """
    frame = losses.to_frame() if isinstance(losses, pd.Series) else losses
    if frame.isna().any().any():
        raise ValueError("The realized loss has missing days.")
    out = pd.DataFrame({
        "days": frame.count(), "mean": frame.mean(), "median": frame.median(), "std": frame.std(),
        "mean |loss|": frame.abs().mean(),
        "1% quantile": frame.quantile(0.01), "5% quantile": frame.quantile(0.05), "95% quantile": frame.quantile(0.95), "99% quantile": frame.quantile(0.99),
        "worst day": frame.max(), "best day": frame.min(),
        **{f"|loss| <= {band:.1%}": (frame.abs() <= band).mean() for band in CONCENTRATION_BANDS},
        "skew": frame.skew(), "excess kurtosis": frame.kurt(),
    })
    out["days"] = out["days"].astype(int)
    return out
