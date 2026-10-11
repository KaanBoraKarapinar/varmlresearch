import warnings

import numpy as np
import pandas as pd

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

    return np.maximum(values, VARIANCE_FLOOR)


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
