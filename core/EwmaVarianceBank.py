from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data_quality import DATA_QUALITY_check_variance_floor

if TYPE_CHECKING:
    from .GroundTruthCalendar import GroundTruthCalendar


class EwmaVarianceBank:
    """
    Portfolio-variance bank for pooled long-format models: diagonal (per-ticker
    variance) comes from the ML model, off-diagonal (correlation) from a
    RiskMetrics-style EWMA on raw ticker returns. Portfolio variance is the
    full quadratic form w^T Sigma_hat w.

    training_safe_bulk_insert()+add() during
    training builds the *_in_sample bank, then .copy() carries it into the
    forecast phase where add()/general_bulk_insert() keep extending it one date at a time.

    No-look-ahead by construction: add(date, ...) rolls the EWMA covariance
    forward using the PREVIOUS call's return (self._prev_return), never the
    return being recorded in the same call -- same discipline as ResidualBank's
    stateful mu_hat.
    """

    def __init__(self, calendar: "GroundTruthCalendar", ewma_lam: float = 0.94, ewma_min_periods: int = 30):
        self.ewma_lam = ewma_lam
        self.ewma_min_periods = ewma_min_periods
        # tickers and (normalized) weights come from the calendar, so the bank cannot disagree with it
        self.tickers = list(calendar.tickers)
        self.weights = calendar.weights.loc[self.tickers].to_numpy()

        self.ticker_returns = pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], name="Date"))
        self.sigma_diag = pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], name="Date"))
        self.portfolio_variance = pd.Series(dtype=float, name="predicted_variance")
        # {date: (tickers x tickers) EWMA covariance as a labeled DataFrame}, the computation itself runs on numpy
        self.cov_by_date: dict[pd.Timestamp, pd.DataFrame] = {}

        self._current_sigma: np.ndarray | None = None
        self._prev_return: np.ndarray | None = None
        # False until training starts to fill this bank (see start_training); a bank is single-use
        self.trained = False

    def start_training(self, calendar: "GroundTruthCalendar") -> "EwmaVarianceBank":
        """
        Called once by training right before it fills this bank: checks that the calendar is the one this bank
        was built from (same tickers and weights) and marks the bank as trained. A bank is single-use: handing
        a trained bank to another training run raises, so one run's state can never leak into the next
        (create a new EwmaVarianceBank per run).
        """
        if self.trained:
            raise ValueError("This EwmaVarianceBank was already used for training; create a new EwmaVarianceBank.")
        if set(calendar.tickers) != set(self.tickers) or not np.allclose(
            calendar.weights.loc[self.tickers].to_numpy(), self.weights
        ):
            raise ValueError("This EwmaVarianceBank was built from a calendar with different tickers/weights.")
        self.trained = True
        return self

    def training_safe_bulk_insert(self, warmup_returns: pd.DataFrame) -> "EwmaVarianceBank":
        """
        TRAIN-ONLY burn-in: >= ewma_min_periods bars, no portfolio-variance output yet.

        The seed covariance (plain sample cov over the first ewma_min_periods bars)
        already encodes every one of those bars, INCLUDING the seed window's own last
        bar -- so when exactly ewma_min_periods bars are given, _prev_return is left
        None rather than that last bar's return. Setting it would make the very next
        add()/general_bulk_insert() call blend that same bar's outer product back into
        the seed cov a second time. Only bars processed AFTER the seed loop
        (len(R) > ewma_min_periods) leave a genuine "pending" return to chain forward,
        since those were walked one at a time via the recursion, not folded into the
        batch np.cov.
        """
        R = warmup_returns[self.tickers].to_numpy(dtype=float)
        if len(R) < self.ewma_min_periods:
            raise ValueError(f"Need >= {self.ewma_min_periods} warmup bars, got {len(R)}.")
        self._current_sigma = np.atleast_2d(np.cov(R[:self.ewma_min_periods], rowvar=False))
        for i in range(self.ewma_min_periods, len(R)):
            r_prev = R[i - 1]
            self._current_sigma = self.ewma_lam * self._current_sigma + (1 - self.ewma_lam) * np.outer(r_prev, r_prev)
        self._prev_return = R[-1] if len(R) > self.ewma_min_periods else None
        return self

    @staticmethod
    def _portfolio_variance_from_cov(cov: np.ndarray, sigma_diag: np.ndarray, weights: np.ndarray) -> float:
        """Diagonal = sigma_diag (model forecast), off-diagonal = cov's own correlation. w^T Sigma_hat w."""
        d_ = np.sqrt(np.diag(cov))
        d_ = np.where(d_ == 0, np.nan, d_)
        corr = cov / np.outer(d_, d_)
        np.fill_diagonal(corr, 1.0)
        sigma_hat = corr * np.sqrt(np.outer(sigma_diag, sigma_diag))
        np.fill_diagonal(sigma_hat, sigma_diag)
        return float(weights @ sigma_hat @ weights)

    def _check_dates_follow_history(self, dates: pd.DatetimeIndex) -> None:
        """
        The EWMA recursion is stateful: dates inserted together must be sorted without duplicates, and the
        first one must come after the last date already in the bank. Going back in time or re-inserting a
        date would run the recursion a second time and silently give wrong covariances, so this raises.
        """
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError("EwmaVarianceBank: dates must be chronologically sorted, without duplicates.")
        if len(self.portfolio_variance) and dates[0] <= self.portfolio_variance.index.max():
            raise ValueError(
                f"EwmaVarianceBank: {dates[0]} is not after the last date already in the bank "
                f"({self.portfolio_variance.index.max()}); dates must be added in chronological order."
            )

    def add(self, date, ticker_return_row: pd.Series, sigma_diag_row: pd.Series) -> float:
        """
        Own-day rule: date holds the ticker returns realized ON date, and the
        model's per-ticker variance forecast made FOR date (from date's own
        preceding feature bar). Returns the resulting portfolio variance.
        """
        if self._current_sigma is None:
            raise ValueError("EwmaVarianceBank: call training_safe_bulk_insert(...) before add(...).")

        date = pd.Timestamp(date)
        self._check_dates_follow_history(pd.DatetimeIndex([date]))
        r_t = ticker_return_row.loc[self.tickers].to_numpy(dtype=float)
        sigma_diag = sigma_diag_row.loc[self.tickers].to_numpy(dtype=float)

        if self._prev_return is not None:
            self._current_sigma = self.ewma_lam * self._current_sigma + (1 - self.ewma_lam) * np.outer(
                self._prev_return, self._prev_return
            )
        cov = self._current_sigma.copy()
        self.cov_by_date[date] = pd.DataFrame(cov, index=self.tickers, columns=self.tickers)

        sigma_diag = DATA_QUALITY_check_variance_floor(sigma_diag, dates=[date], context="EwmaVarianceBank.add")
        portfolio_variance = float(DATA_QUALITY_check_variance_floor(
            self._portfolio_variance_from_cov(cov, sigma_diag, self.weights), dates=[date], context="EwmaVarianceBank.add"
        )[0])

        self.ticker_returns.loc[date] = ticker_return_row.loc[self.tickers]
        self.sigma_diag.loc[date] = sigma_diag
        self.portfolio_variance.loc[date] = portfolio_variance
        self._prev_return = r_t
        return portfolio_variance

    def general_bulk_insert(self, dates, ticker_returns_df: pd.DataFrame, sigma_diag_df: pd.DataFrame | None = None) -> pd.Series:
        """
        dates must be chronologically sorted and come after the bank's last date (checked, see
        _check_dates_follow_history) -- this is a stateful recursion, same
        math as looping add() one date at a time. Unlike that loop, this runs the
        recursion over plain numpy arrays and writes ticker_returns/sigma_diag/
        portfolio_variance/cov_by_date ONCE at the end via pd.concat, instead of
        one slow pandas .loc assignment per date -- the EWMA update itself can't
        be vectorized away (each date's covariance genuinely depends on the
        previous one), but the per-date pandas bookkeeping can be.
        """
        if self._current_sigma is None:
            raise ValueError("EwmaVarianceBank: call training_safe_bulk_insert(...) before general_bulk_insert(...).")

        dates = pd.DatetimeIndex(dates)
        self._check_dates_follow_history(dates)
        R = ticker_returns_df.loc[dates, self.tickers].to_numpy(dtype=float)
        S = None if sigma_diag_df is None else sigma_diag_df.loc[dates, self.tickers].to_numpy(dtype=float)
        n_dates, n_tickers = len(dates), len(self.tickers)

        sigma = self._current_sigma
        prev_return = self._prev_return
        covs = np.empty((n_dates, n_tickers, n_tickers))
        sigma_diag_out = np.empty((n_dates, n_tickers))
        portfolio_variances = np.empty(n_dates)

        for i in range(n_dates):
            if prev_return is not None:
                sigma = self.ewma_lam * sigma + (1 - self.ewma_lam) * np.outer(prev_return, prev_return)
            covs[i] = sigma
            # sigma_diag_df=None: no model forecast is mixed in, the covariance's OWN diagonal is used, so the
            # portfolio variance is the pure EWMA w^T Sigma w of the realized returns
            # (what GroundTruthCalendar.ewma_realized_variance_bank does).
            if S is None:
                sigma_diag_out[i] = np.diag(sigma)
                portfolio_variances[i] = self.weights @ sigma @ self.weights
            else:
                sigma_diag_out[i] = DATA_QUALITY_check_variance_floor(S[i], dates=[dates[i]], context="EwmaVarianceBank.general_bulk_insert")
                portfolio_variances[i] = self._portfolio_variance_from_cov(sigma, sigma_diag_out[i], self.weights)
            prev_return = R[i]

        portfolio_variances = DATA_QUALITY_check_variance_floor(portfolio_variances, dates=dates, context="EwmaVarianceBank.general_bulk_insert")

        self.ticker_returns = pd.concat([self.ticker_returns, pd.DataFrame(R, index=dates, columns=self.tickers)])
        self.sigma_diag = pd.concat([self.sigma_diag, pd.DataFrame(sigma_diag_out, index=dates, columns=self.tickers)])
        self.portfolio_variance = pd.concat(
            [self.portfolio_variance, pd.Series(portfolio_variances, index=dates, name="predicted_variance")]
        )
        self.cov_by_date.update(
            (d, pd.DataFrame(cov, index=self.tickers, columns=self.tickers)) for d, cov in zip(dates, covs)
        )

        self._current_sigma = sigma
        self._prev_return = prev_return
        return pd.Series(portfolio_variances, index=dates, name="predicted_variance")

    def copy(self) -> "EwmaVarianceBank":
        return deepcopy(self)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Combines ticker_returns, sigma_diag, and portfolio_variance into a single
        wide-format DataFrame. Columns are multi-indexed: (returns, TICKER),
        (sigma_diag, TICKER), (portfolio_variance, ''). cov_by_date is deliberately
        excluded -- pull that separately if needed.
        """
        return pd.concat(
            {
                "returns": self.ticker_returns,
                "sigma_diag": self.sigma_diag,
                "portfolio_variance": self.portfolio_variance.to_frame("portfolio_variance"),
            },
            axis=1,
        )
