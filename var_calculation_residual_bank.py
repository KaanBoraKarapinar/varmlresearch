import numpy as np
import pandas as pd
import warnings




import numpy as np
import pandas as pd

# %%

class CalendarFreqManager:
    pass




# %%
### DATA QUALITY CHECKS

def DATA_QUALITY_check_variance_floor(predicted_variances, dates=None, context: str = "") -> np.ndarray:
    VARIANCE_FLOOR = 1e-12

    values = np.atleast_1d(np.asarray(predicted_variances, dtype=float))
    bad_mask = values <= VARIANCE_FLOOR

    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        if dates is not None:
            dates = pd.DatetimeIndex(np.atleast_1d(dates))
            bad_dates = list(dates[bad_mask][:5])
            location = f" at dates (first 5): {bad_dates}"
        else:
            location = ""

        prefix = f"[{context}] " if context else ""
        warnings.warn(
            f"{prefix}DEGENERATE VARIANCE: {n_bad} value(s) <= {VARIANCE_FLOOR}{location}.",
            stacklevel=2,
        )


def DATA_QUALITY_replace_inf_with_warning(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Replace inf/-inf with NaN, warning loudly if it actually triggers (near-zero sigma blowing up a division)."""
    inf_mask = np.isinf(df.to_numpy(dtype=float))
    if inf_mask.any():
        n_bad = int(inf_mask.sum())
        prefix = f"[{context}] " if context else ""
        warnings.warn(
            f"{prefix}INF VALUES DETECTED: {n_bad} cell(s) were inf/-inf, likely from dividing by "
            f"a near-zero sigma. These rows will be dropped.",
            stacklevel=2,
        )
    return df.replace([np.inf, -np.inf], np.nan)


# %%

import warnings
import numpy as np
import pandas as pd

RESIDUAL_COLS = ["actual_return", "predicted_variance", "predicted_sigma", "mu_hat", "standardized_residual"]
VARIANCE_FLOOR = 1e-12



class ResidualBank:
    def __init__(self, start_timestamp: pd.Timestamp, frequency: pd.Timedelta, initial_mu_hat: float = 0.0):
        if not isinstance(frequency, pd.Timedelta):
            raise TypeError(f"frequency must be a pd.Timedelta, got: {type(frequency)}")
        if frequency <= pd.Timedelta(0):
            raise ValueError(f"frequency must be positive: {frequency}")

        self.start_timestamp = start_timestamp
        self.frequency = frequency
        self.residuals = pd.DataFrame(columns=RESIDUAL_COLS, index=pd.DatetimeIndex([], name="Date"))

        # stateful mu_hat used to standardize the NEXT incoming residual.
        # Deliberately NOT recalculated until step_2, so step_1 never uses information
        # from the bar it is currently pricing (no look-ahead).
        self._current_single_stateful_mu = float(initial_mu_hat)

    @property
    def current_mu_hat(self) -> float:
        return self._current_single_stateful_mu

    def DANGEROUS_add_residual(self, date, actual_return: float, predicted_variance: float, mu_hat: float) -> "ResidualBank":
        """
        NO STATEFUL MU TRACKING, RISK OF LOOK FORWARD BIAS
        """
        date = pd.Timestamp(date)
        predicted_variance = float(DATA_QUALITY_check_variance_floor(predicted_variance, dates=[date], context="DANGEROUS_add_residual")[0])
        sigma = np.sqrt(predicted_variance)
        standardized_residual = (float(actual_return) - float(mu_hat)) / sigma

        if not np.isfinite(standardized_residual):
            warnings.warn(
                f"[DANGEROUS_add_residual] non-finite standardized_residual at {date} (sigma={sigma}), skipping this row.",
                stacklevel=2,
            )
            return self

        self.residuals.loc[date] = {
            "actual_return": float(actual_return),
            "predicted_variance": predicted_variance,
            "predicted_sigma": sigma,
            "mu_hat": float(mu_hat),
            "standardized_residual": standardized_residual,
        }
        return self

    def step_1_add_residual_with_prev_stateful_mu(
        self, date, actual_return: float, predicted_variance: float
    ) -> "ResidualBank":
        """
        Standardize and insert the new residual using the mu_hat carried over from BEFORE
        this bar's return was realized. Pins mu_hat to the current stateful value.
        """
        return self.DANGEROUS_add_residual(
            date=date,
            actual_return=actual_return,
            predicted_variance=predicted_variance,
            mu_hat=self._current_single_stateful_mu,
        )

    def step_2_change_stateful_mu(self, window_size: int, as_of: pd.Timestamp | None = None) -> "ResidualBank":
        """
        Recompute the stateful mu_hat from the (now updated) windowed history. Call this
        AFTER step_1, once per bar, so the next bar's step_1 sees this bar's return.
        """
        window = self.windowed_read(window_size, as_of)
        if window.empty:
            raise ValueError(f"Cannot update mu_hat: residual window is empty (window_size={window_size}, as_of={as_of}).")

        self._current_single_stateful_mu = float(window["actual_return"].mean())
        return self

    def windowed_read(self, window_size: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        df = self.residuals if as_of is None else self.residuals.loc[:as_of]
        return df.tail(window_size)

    def empirical_var_for_target(
            self,
            target_timestamp: pd.Timestamp,
            predicted_variance: float,
            var_confidence: float,
            window_size: int,
            enforce_window_size: bool = False,
    ) -> pd.Series:
        """
        Compute empirical VaR for a specific target bar. Requires the feature bar
        (target_timestamp - frequency) to already be present in the residual history,


        :param enforce_window_size: if True, raise unless the window contains exactly
            window_size residuals If False (default), any non-empty window up to window_size is accepted.
        :raises KeyError: if target_timestamp - frequency is not in self.residuals.
        :raises ValueError: if the window is empty, or (when enforce_window_size=True)
            shorter than window_size.
        :return: pd.Series indexed by target_timestamp, with fields varForecast, mu_hat,
            predicted_variance, last_feature_timestamp, n_residuals_used.
        """
        target_timestamp = pd.Timestamp(target_timestamp)
        last_feature_timestamp = target_timestamp - self.frequency

        if last_feature_timestamp not in self.residuals.index:
            raise KeyError(
                f"empirical_var_for_target: {last_feature_timestamp} (= target {target_timestamp} - "
                f"frequency {self.frequency}) is not in residual history. "
                f"Cannot price {target_timestamp} without its immediately preceding feature bar."
            )

        predicted_variance = float(
            DATA_QUALITY_check_variance_floor(predicted_variance, dates=[target_timestamp], context="empirical_var_for_target")[0])
        window = self.windowed_read(window_size, as_of=last_feature_timestamp)

        print(
            f"[empirical_var_for_target] target={target_timestamp}, using {len(window)} residual(s) up to {last_feature_timestamp}")

        if window.empty:
            raise ValueError(f"Residual window is empty (window_size={window_size}, as_of={last_feature_timestamp}).")

        if enforce_window_size and len(window) < window_size:
            raise ValueError(
                f"enforce_window_size=True but only {len(window)}/{window_size} residuals "
                f"available as of {last_feature_timestamp} (target={target_timestamp})."
            )

        mu_hat = float(window["actual_return"].mean())
        q = np.quantile(window["standardized_residual"], 1 - var_confidence)
        sigma_hat = np.sqrt(predicted_variance)
        var_estimate = -(mu_hat + sigma_hat * q)

        return pd.Series(
            {
                "varForecast": var_estimate,
                "mu_hat": mu_hat,
                "predicted_variance": predicted_variance,
                "last_feature_timestamp": last_feature_timestamp,
                "n_residuals_used": len(window),
            },
            name=target_timestamp,
        )