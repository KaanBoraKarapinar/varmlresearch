# %%
import numpy as np
pandas as pd


# %%
TICKERS = ["AAPL", "JPM", "XOM", "KO"]
TICKERS_WITH_PORTFOLIO = TICKERS + ["PORTFOLIO"]

WEIGHTS = {
    "AAPL": 0.25,
    "JPM": 0.25,
    "XOM": 0.25,
    "KO": 0.25,
}


def _validated_portfolio_weights() -> pd.Series:
    """Return normalized portfolio weights and fail loudly on missing config."""
    weights = pd.Series(
        {ticker: WEIGHTS.get(ticker, np.nan) for ticker in TICKERS},
        dtype=float,
    )

    if weights.isna().any():
        missing = weights[weights.isna()].index.tolist()
        raise ValueError(f"Missing portfolio weights for: {missing}")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite.")

    total = float(weights.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Portfolio weights sum to zero.")

    return weights / total

