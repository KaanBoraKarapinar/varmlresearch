"""
The unit is the RETURN TYPE the calendar computes and the records of a run are made of. Declared by the user (it is a
required argument everywhere, never defaulted) and carried by the calendar, the feature spec, the target and every saved run,
so nothing can silently mix return types.

    percent   the simple percent-change return, price[t] / price[t-1] - 1 (pct_change), stored as a fraction:
              0.0153 is a 1.53% move (the numbers are never multiplied by 100, "%" is only a display format)
    log       the log return, log(price[t] / price[t-1])

The calendar takes only the raw prices and computes the return its unit asks for (GroundTruthCalendar:
_FOR_PCT_RETURN for percent, _FOR_LOG_RETURN for log). The intermediate records of a run stay in its own unit; the VaR
READINGS (what is tested, compared and shown) are converted at the last step to percent returns by the functions below, so a
log run and a percent run are comparable and shown the same way.
"""

import numpy as np

UNITS = ("percent", "log")


def check_unit(unit: str) -> str:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    return unit


def loss_to_percent(loss, unit: str):
    """
    A loss or a VaR (a positive number = a loss of that size) of a run in `unit`, as a percent-return loss.
    percent: as is. log: a log loss l = -log(1 + R) is the simple loss 1 - exp(-l). The log case is exact for a VaR too: a
    quantile of the log return maps to the quantile of the simple return (the map is monotone), so an exceedance
    (loss > VaR) is the same event in both units.
    """
    check_unit(unit)
    return loss if unit == "percent" else 1.0 - np.exp(-loss)


def return_to_percent(value, unit: str):
    """A return (or a mean forecast mu_hat) of a run in `unit`, as a percent return (log: exp(r) - 1)."""
    check_unit(unit)
    return value if unit == "percent" else np.expm1(value)


def scale_to_percent(value, unit: str, power: int = 1):
    """
    A sigma (power 1) or a variance (power 2) of a run in `unit` as percent-return values. A log return and a simple return differ
    only in the second order, so this is the identity for both units (the one place the conversion is first-order).
    """
    check_unit(unit)
    return value


def format_value(value: float, digits: int = 2, signed: bool = False) -> str:
    """A return-sized number (a VaR, a loss, a sigma) for display: 0.0153 -> '1.53%'."""
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}%}"


def plot_format() -> dict:
    """Plotly number formats of a return-sized axis / hover: tickformat, ticksuffix, hover format."""
    return {"tickformat": ".1%", "ticksuffix": "", "hoverformat": ".2%", "hoversuffix": ""}
