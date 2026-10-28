"""
The daily minimum-risk portfolio of a stored run: every day the weights that minimise the forecast variance w' Sigma w of that day,
Sigma = D R D (D = the model's per-ticker volatility forecast, R = the correlation of the day's EWMA covariance), long only and fully
invested. Nothing is trained: the per-ticker forecasts come from the run's combined ewma bank (Run.ewma_bank). The weights of a day
use only that day's forecasts, made before the day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .units import return_to_percent


def sigma_matrices(bank: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """Sigma_d = D R D for every day of the bank (shape days x tickers x tickers)."""
    d = np.sqrt(bank[[f"sigma_{t}" for t in tickers]].to_numpy(dtype=float))
    cov = np.stack([bank[[f"cov_{i}_{j}" for j in tickers]].to_numpy(dtype=float) for i in tickers], axis=1)
    diag = np.sqrt(np.einsum("dii->di", cov))
    corr = cov / (diag[:, :, None] * diag[:, None, :])
    return d[:, :, None] * corr * d[:, None, :]


def min_variance_weights(sigma: np.ndarray, start: np.ndarray) -> np.ndarray:
    """min w' S w  s.t. sum w = 1, w >= 0 (SLSQP; S is divided by its mean diagonal so the numbers are about 1). The optimum is unique."""
    s = sigma / (np.trace(sigma) / len(sigma))
    result = minimize(
        lambda w: w @ s @ w, start, jac=lambda w: 2 * s @ w, bounds=[(0, 1)] * len(start),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}],
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"the minimum-variance optimisation did not converge: {result.message}")
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def daily_minimum_risk_weights(bank: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """The minimum-risk weights of every day of the bank (index Date, one column per ticker, rows sum to 1)."""
    sigmas = sigma_matrices(bank, tickers)
    w = np.full(len(tickers), 1 / len(tickers))
    rows = []
    for sigma in sigmas:
        w = min_variance_weights(sigma, w)       # the solution of the day before is the starting point
        rows.append(w)
    return pd.DataFrame(rows, index=bank.index, columns=tickers)


def weighted_simple_returns(bank: pd.DataFrame, tickers: list[str], weights: pd.DataFrame | np.ndarray, unit: str) -> pd.Series:
    """
    The simple (percent) return of a portfolio per day from the bank's ticker returns: the day's weights applied to the day's
    returns. weights: a day x ticker frame (or one vector for every day). The bank stores returns in the run's unit; a log run's are
    turned into simple returns first.
    """
    simple = return_to_percent(bank[[f"return_{t}" for t in tickers]], unit).to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float) if isinstance(weights, pd.DataFrame) else np.asarray(weights, dtype=float)
    return pd.Series((simple * w).sum(axis=1), index=bank.index, name="portfolio_return")


def portfolio_forecast_variance(bank: pd.DataFrame, tickers: list[str], weights: pd.DataFrame | np.ndarray) -> pd.Series:
    """w' Sigma_d w per day (the weights of the day, or one vector for every day)."""
    sigmas = sigma_matrices(bank, tickers)
    w = weights.to_numpy(dtype=float) if isinstance(weights, pd.DataFrame) else np.broadcast_to(np.asarray(weights, dtype=float), (len(bank), len(tickers)))
    return pd.Series(np.einsum("di,dij,dj->d", w, sigmas, w), index=bank.index, name="forecast_variance")
