"""
The tables of the analysis report: what was tried (tickers, weights, features, targets) and what came out. Every function returns a
DataFrame (or a dict of them) built from the run store and the saved experiments, so the report notebook and the later Streamlit page call the
same functions and show the same numbers. Nothing here trains anything: the runs are read as they are saved.

    universes() / portfolios() / model_catalog() / feature_matrix()      what was run: tickers, windows, weights, models
    glossary_features() / glossary_targets()                              what a feature block / a target is
    experiment_overview() / experiment_table(name)                        the saved experiments and one experiment's result table
    window_table(...)                                                     the same runs on the whole sample / the Covid crash / after it
    vs_baseline_across(...) / holm_across(...)                            the 23-ticker models on the equal-weight and the random portfolios
    diagonal_share_table()                                                portfolio concentration against the advantage over GARCH
    minimum_risk_tables()                                                 the daily minimum-risk portfolio of the 23 tickers
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import core
from core import run_store as rs
from core import walk_forward as wf
from core.run_comparison import _dm_p_value

STORE_ROOT = rs.STORE_ROOT
PERIODS = {                                  # the analysis windows of the report (forecast days)
    "whole period": ("2019-03-04", "2100-01-01"),
    "Covid crash (2020-02-19 to 2020-06-30)": ("2020-02-19", "2020-06-30"),
    "after the crash (2020-07-01 on)": ("2020-07-01", "2100-01-01"),
}
MODELS_23 = ["garch_wide", "ewma", "historical", "xgb_long", "xgb_long_rv_har", "xgb_long_drv_har", "xgb_long_rv_macro", "xgb_long_drv_macro"]


def styled(frame: pd.DataFrame, digits: int | None = None):
    """
    The table as the report shows it: tick losses with 5 decimals, p-values with 3, percentages with 1, other numbers with 2 (NaN left empty);
    digits: every number with this many decimals instead.
    """
    def format_of(column) -> str:
        if digits is not None:
            return f"{{:.{digits}f}}"
        name = str(column[-1] if isinstance(column, tuple) else column).lower()
        if "tick loss" in name and "vs baseline" not in name:
            return "{:.5f}"
        if name.endswith(" p") or "p-value" in name:
            return "{:.3f}"
        if "%" in name or "vs" in name:
            return "{:.1f}"
        return "{:.2f}"
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    return frame.style.format({c: format_of(c) for c in numeric}, na_rep="")


def reading(confidence: float = 0.95) -> wf.Reading:
    """The reading of the report: the quantile of the last 252 standardized residuals (at least 30), tested at the VaR level."""
    return wf.Reading(var_confidence=confidence, window_size=252, min_periods=30, test_confidence=confidence)


# ---- the saved runs and experiments ----------------------------------------------------------------------------

def all_runs(store_root: Path = STORE_ROOT) -> list[wf.LoadedRun]:
    return rs.load_runs(Path(store_root))


def find_experiment(part: str, store_root: Path = STORE_ROOT) -> core.Experiment:
    """The saved experiment whose name contains `part` (exactly one)."""
    found = [e for e in core.list_saved_experiments(Path(store_root)) if part.lower() in e.name.lower()]
    if len(found) != 1:
        raise ValueError(f"{len(found)} saved experiments contain {part!r}: {[e.name for e in found]}")
    return found[0]


def experiment_runs(experiment: core.Experiment, store_root: Path = STORE_ROOT) -> dict[str, wf.LoadedRun]:
    """The runs of an experiment by run name, in the order they were saved."""
    runs = [rs.load_walk_forward_run(i, Path(store_root)) for i in experiment.run_ids]
    return {run.name: run for run in runs}


def model_name(run: wf.Run) -> str:
    """The model behind a run name: the portfolio suffix (_rw1 ...) and '_from23' (the model trained on 23 tickers, applied) removed."""
    return re.sub(r"_from23$", "", re.sub(r"_rw\d$", "", run.name))


def _effective_tickers(weights: pd.Series) -> float:
    weights = weights / weights.sum()
    return float(1 / (weights ** 2).sum())


def _weights_of(run: wf.Run) -> pd.Series:
    setting = run.config["setting"]
    return pd.Series({t: float(setting["weights"][t]) for t in setting["tickers"]})


# ---- what was run: tickers, windows, weights, models -----------------------------------------------------------

def universes(store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """One row per walk-forward setting without its weights: the tickers and the windows the runs share."""
    rows: dict[tuple, dict] = {}
    for run in all_runs(store_root):
        setting = run.config["setting"]
        key = (tuple(setting["tickers"]), run.params["train_bars"], run.params["forecast_bars"], setting["start_date"], run.config["data"]["sha256"][:6])
        windows = run.windows
        row = rows.setdefault(key, {
            "tickers": ", ".join(setting["tickers"]) if len(setting["tickers"]) <= 6 else f"{len(setting['tickers'])} tickers",
            "n tickers": len(setting["tickers"]), "train bars": run.params["train_bars"], "forecast bars": run.params["forecast_bars"],
            "first forecast day": pd.Timestamp(windows["forecast_start"].iloc[0]).date(), "last forecast day": pd.Timestamp(windows["forecast_end"].iloc[-1]).date(),
            "windows": len(windows), "forecast days": int(windows["n_forecast"].sum()), "price data": key[4], "runs": 0,
        })
        row["runs"] += 1
    table = pd.DataFrame(rows.values())
    return table.sort_values(["n tickers", "train bars", "forecast bars"]).reset_index(drop=True)


def portfolio_draws(store_root: Path = STORE_ROOT) -> dict[str, tuple[float, int]]:
    """{'rw1': (alpha, seed), ...} from the notes of the saved random-portfolio experiments."""
    draws = {}
    for experiment in core.list_saved_experiments(Path(store_root)):
        match = re.search(r"Random portfolio (rw\d)", experiment.name)
        found = re.search(r"Dirichlet\(alpha ([0-9.]+)\) draw, seed (\d+)", experiment.notes)
        if match and found:
            draws[match.group(1)] = (float(found.group(1)), int(found.group(2)))
    return draws


def portfolios(store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """The portfolios of the experiments: tickers and fixed weights (equal, or a seeded Dirichlet draw over the 23 tickers)."""
    draws = portfolio_draws(store_root)
    entries = [
        ("4 tickers, equal", "Pool size"), ("23 tickers, equal (1/23)", "23 tickers, equal weight"),
        *[(label, f"Random portfolio {label}") for label in draws],
    ]
    rows = []
    for label, part in entries:
        run = next(iter(experiment_runs(find_experiment(part, store_root), store_root).values()))
        weights = _weights_of(run)
        weights = weights / weights.sum()
        alpha, seed = draws.get(label, (np.nan, np.nan))
        rows.append({
            "portfolio": label, "n tickers": len(weights), "effective tickers": round(_effective_tickers(weights), 1),
            "largest weight": f"{weights.idxmax()} {weights.max():.2f}", "smallest weight": round(float(weights.min()), 3),
            "Dirichlet alpha": alpha, "seed": seed,
        })
    return pd.DataFrame(rows).set_index("portfolio")


def model_catalog(store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """One row per model (name, features, target): its family, format, target, scale and on how many tickers it exists."""
    rows: dict[tuple, dict] = {}
    for run in all_runs(store_root):
        key = (model_name(run), run.params.get("feature_blocks"), run.params.get("target"))
        row = rows.setdefault(key, {
            "model": key[0], "family": run.params["model_family"], "format": run.params["model_format"], "target": key[2] or "-",
            "scale": run.params.get("target_scale"), "portable features": ("MarketEwmaVol" not in str(key[1])) if key[1] else None,
            "ticker counts": set(),
        })
        row["ticker counts"].add(len(run.config["setting"]["tickers"]))
    table = pd.DataFrame(rows.values())
    table["ticker counts"] = table["ticker counts"].map(lambda s: ", ".join(str(n) for n in sorted(s)))
    order = {name: i for i, name in enumerate(["garch_wide", "ewma", "historical", "xgb_wide", "xgb_long", "xgb_long_n300", "xgb_long_ratio", "xgb_long_ratio_ticker"])}
    return table.assign(_o=table["model"].map(order).fillna(99)).sort_values(["_o", "model", "portable features"]).drop(columns="_o").reset_index(drop=True)


def feature_matrix(store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """Which feature blocks each XGBoost model reads (a block of the 4-ticker model with MarketEwmaVol and the portable one are two rows)."""
    seen: dict[tuple, dict] = {}
    for run in all_runs(store_root):
        blocks = run.params.get("feature_blocks")
        if not blocks:
            continue
        key = (model_name(run), "MarketEwmaVol" in blocks)
        seen.setdefault(key, {"model": key[0] + ("" if key[1] else " (portable)"), **{re.sub(r"\(.*", "", b): "x" for b in re.findall(r"[A-Za-z0-9]+(?:\([^)]*\))?", blocks)}})
    table = pd.DataFrame(seen.values()).set_index("model").fillna("")
    first = ["Returns", "AbsReturn", "RollingStats", "EwmaVol", "MarketEwmaVol", "TickerOneHot", "RvHar", "RvLags", "Vix", "Sp500Return", "YieldSpread10y3m", "YieldSpread10y5y", "YieldSpread30y10y"]
    return table[[c for c in first if c in table.columns]].sort_index()


GLOSSARY_FEATURES = {
    "Returns": "the day's return of the ticker",
    "AbsReturn": "its absolute value",
    "RollingStats": "volatility, mean absolute return and skew over the last 5, 10 and 21 days",
    "EwmaVol": "the RiskMetrics EWMA volatility of the ticker (span 21)",
    "MarketEwmaVol": "the EWMA volatility of the PORTFOLIO, the same value on every ticker row (depends on the weights, so a model that has it cannot be applied to another portfolio)",
    "TickerOneHot": "one 0/1 column per ticker: a level per ticker for the pooled model",
    "RvHar": "the ticker's daily realized variance (the sum of squared 5-minute returns), its 5-day and 22-day means (HAR)",
    "RvLags": "the realized variance of the ticker 1, 2 and 3 days earlier",
    "Vix": "the VIX level and its 5-day percent change",
    "Sp500Return": "the S&P 500 return over 1, 5 and 21 days",
    "YieldSpread10y3m": "10-year minus 3-month Treasury yield",
    "YieldSpread10y5y": "10-year minus 5-year Treasury yield",
    "YieldSpread30y10y": "30-year minus 10-year Treasury yield",
}
GLOSSARY_TARGETS = {
    "SquaredReturn": "the next day's squared return times a scale: the plain variance target",
    "RatioToBase": "the next day's squared return divided by the EWMA variance: the model learns a multiplicative correction of the EWMA",
    "RealizedVariance": "the next day's realized variance (sum of squared 5-minute returns), times a scale: a much less noisy target than r²",
    "RelativeRvChange": "the relative change of the realized variance from today to the next day, RV_next / RV - 1",
}


def glossary(entries: dict[str, str], column: str) -> pd.DataFrame:
    return pd.DataFrame({column: list(entries), "what it is": list(entries.values())}).set_index(column)


def glossary_features() -> pd.DataFrame:
    return glossary(GLOSSARY_FEATURES, "feature block")


def glossary_targets() -> pd.DataFrame:
    return glossary(GLOSSARY_TARGETS, "target")


# ---- the saved experiments -------------------------------------------------------------------------------------

def experiment_overview(store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """The saved experiments: the question, the tickers, how many runs and the period."""
    rows = []
    for experiment in core.list_saved_experiments(Path(store_root)):
        first = next(iter(experiment_runs(experiment, store_root).values()))
        weights = _weights_of(first)
        rows.append({
            "experiment": experiment.name, "question": experiment.question, "tickers": len(weights),
            "effective tickers": round(_effective_tickers(weights), 1), "runs": len(experiment.run_ids), "from": experiment.start, "to": experiment.end,
        })
    return pd.DataFrame(rows).set_index("experiment")


def compare(runs: dict[str, wf.Run], baseline: str, read: wf.Reading, start: str = PERIODS["whole period"][0], end: str = PERIODS["whole period"][1]) -> pd.DataFrame:
    """
    The runs of one portfolio on the days they all forecast inside [start, end]: mean tick loss (the score of a VaR, lower is better), the difference to
    the baseline run in %, the exceedances against the number expected, whether the run passes the three tests (Kupiec, Christoffersen
    independence and conditional coverage), the Diebold-Mariano p-value of the difference to the baseline (HAC), its Holm correction over the
    comparisons made, and whether the run is in the 90% Model Confidence Set.
    """
    dates = rs.common_dates(list(runs.values()), read)
    dates = dates[(dates >= start) & (dates <= end)]
    var_dfs = {label: run.var_df(read).loc[dates] for label, run in runs.items()}
    board = core.select_best_run(var_dfs, read.var_confidence, read.test_confidence, baseline=baseline, alpha=0.10).leaderboard
    others = [label for label in board.index if label != baseline and pd.notna(board.loc[label, "dm_p_vs_baseline"])]
    holm = pd.Series(multipletests(board.loc[others, "dm_p_vs_baseline"].to_numpy(), method="holm")[1], index=others) if others else pd.Series(dtype=float)
    table = pd.DataFrame({
        "mean tick loss": board["mean_tick_loss"],
        "vs baseline %": (board["mean_tick_loss"] / board.loc[baseline, "mean_tick_loss"] - 1) * 100,
        "exceedances": pd.Series({label: int((v["loss"] > v["varForecast"]).sum()) for label, v in var_dfs.items()}),
        "expected": len(dates) * (1 - read.var_confidence),
        "tests": board["valid"].map({True: "pass", False: "fail"}),
        "DM p": board["dm_p_vs_baseline"], "Holm p": holm,
        "in MCS": board["in_mcs"].map({True: "yes", False: "no"}),
    })
    table.attrs["days"] = len(dates)
    return table.sort_values("mean tick loss")


def experiment_table(part: str, store_root: Path = STORE_ROOT, confidence: float | None = None) -> pd.DataFrame:
    """The result table of a saved experiment, read as it was saved (its reading, period and baseline); confidence overrides the VaR level."""
    experiment = find_experiment(part, store_root)
    runs = experiment_runs(experiment, store_root)
    baseline = next(label for label, run in runs.items() if run.run_id == experiment.baseline_run_id)
    read = reading(confidence) if confidence is not None else wf.Reading(**experiment.reading)
    return compare(runs, baseline, read, experiment.start, experiment.end)


FULL_SAMPLE_PERIODS = {                        # for the walk-forwards that start in 2010: the whole sample, then the crash and after it
    "full sample": ("1900-01-01", "2100-01-01"), **{k: v for k, v in PERIODS.items() if k != "whole period"},
}


def walk_forward_runs(first_forecast_year: int, train_bars: int, forecast_bars: int, n_tickers: int = 4, store_root: Path = STORE_ROOT) -> dict[str, wf.LoadedRun]:
    """The runs (by model name) of the walk-forward whose first forecast day is in `first_forecast_year` and that has these bars and this many tickers."""
    found = {}
    for run in all_runs(store_root):
        windows = run.windows
        if (len(run.config["setting"]["tickers"]) == n_tickers and run.params["train_bars"] == train_bars and run.params["forecast_bars"] == forecast_bars
                and pd.Timestamp(windows["forecast_start"].iloc[0]).year == first_forecast_year):
            found[run.name] = run
    return found


def window_table(runs: dict[str, wf.Run], baseline: str, confidence: float = 0.95, periods: dict[str, tuple[str, str]] = PERIODS) -> pd.DataFrame:
    """The mean tick loss of every run on each period (the whole period, the Covid crash, after it), and the difference to the baseline in %."""
    read = reading(confidence)
    dates = rs.common_dates(list(runs.values()), read)
    columns = {}
    for period, (start, end) in periods.items():
        days = dates[(dates >= start) & (dates <= end)]
        if len(days) < 30:
            continue
        ticks = core.daily_tick_losses({label: run.var_df(read).loc[days] for label, run in runs.items()}, confidence).mean()
        columns[(period, "mean tick loss")] = ticks
        columns[(period, "vs baseline %")] = (ticks / ticks[baseline] - 1) * 100
    return pd.DataFrame(columns).sort_values(next(iter(columns)))


# ---- the 23-ticker models on the equal-weight and the random portfolios ----------------------------------------

def _portfolio_runs(store_root: Path = STORE_ROOT) -> dict[str, dict[str, wf.LoadedRun]]:
    """{'eq': {model: run}, 'rw1': ..., ...}: the eight models of the 23-ticker experiments by model name."""
    out = {"eq": {model_name(r): r for r in experiment_runs(find_experiment("23 tickers, equal weight", store_root), store_root).values()}}
    for label in portfolio_draws(store_root):
        out[label] = {model_name(r): r for r in experiment_runs(find_experiment(f"Random portfolio {label}", store_root), store_root).values()}
    return out


def vs_baseline_across(confidence: float = 0.95, period: str = "whole period", baseline: str = "garch_wide", store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """
    Mean tick loss of every model against the baseline of the SAME portfolio, in % (negative = better), for the equal-weight portfolio and the random
    ones; the last columns are the mean over the random portfolios and in how many of them the model is ahead of the baseline.
    """
    read, (start, end) = reading(confidence), PERIODS[period]
    columns = {}
    for label, runs in _portfolio_runs(store_root).items():
        dates = rs.common_dates(list(runs.values()), read)
        dates = dates[(dates >= start) & (dates <= end)]
        ticks = core.daily_tick_losses({m: r.var_df(read).loc[dates] for m, r in runs.items()}, confidence).mean()
        columns[label] = (ticks / ticks[baseline] - 1) * 100
    table = pd.DataFrame(columns).drop(index=baseline).reindex([m for m in MODELS_23 if m != baseline])
    random = [c for c in table.columns if c != "eq"]
    table["mean (random)"] = table[random].mean(axis=1)
    table["ahead in"] = (table[random] < 0).sum(axis=1).map(lambda n: f"{n} of {len(random)}")
    return table.sort_values("mean (random)")


def holm_across(confidence: float = 0.95, baseline: str = "garch_wide", store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """Holm-corrected Diebold-Mariano p-values of every model against the baseline of the same portfolio (whole period)."""
    read, (start, end) = reading(confidence), PERIODS["whole period"]
    columns = {}
    for label, runs in _portfolio_runs(store_root).items():
        dates = rs.common_dates(list(runs.values()), read)
        dates = dates[(dates >= start) & (dates <= end)]
        losses = core.daily_tick_losses({m: r.var_df(read).loc[dates] for m, r in runs.items()}, confidence)
        models = [m for m in MODELS_23 if m != baseline]
        raw = pd.Series({m: _dm_p_value((losses[m] - losses[baseline]).to_numpy(), 5) for m in models})
        columns[label] = pd.Series(multipletests(raw.to_numpy(), method="holm")[1], index=models)
    return pd.DataFrame(columns).reindex([m for m in MODELS_23 if m != baseline])


def diagonal_share(run: wf.Run) -> float:
    """The share of the portfolio variance that the ticker variances give alone, sum w_i^2 sigma_i^2 / w' Sigma w, averaged over the days."""
    bank, tickers = run.ewma_bank, list(run.config["setting"]["tickers"])
    weights = _weights_of(run).loc[tickers].to_numpy()
    weights = weights / weights.sum()
    sigma = core.sigma_matrices(bank, tickers)
    own = np.einsum("i,di->d", weights ** 2, np.einsum("dii->di", sigma))
    return float((own / np.einsum("i,dij,j->d", weights, sigma, weights)).mean())


def diagonal_share_table(model: str = "xgb_long_drv_har", store_root: Path = STORE_ROOT) -> pd.DataFrame:
    """Concentration of every 23-ticker portfolio, the diagonal share of its variance, and the model's tick loss against GARCH (95%, whole period, in %)."""
    table = portfolios(store_root)
    advantage = vs_baseline_across(0.95, "whole period", store_root=store_root).loc[model].drop(["mean (random)", "ahead in"])
    rows = {}
    for label, runs in _portfolio_runs(store_root).items():
        name = "23 tickers, equal (1/23)" if label == "eq" else label
        rows[name] = {
            "effective tickers": table.loc[name, "effective tickers"], "largest weight": table.loc[name, "largest weight"],
            "diagonal share": round(diagonal_share(runs[model]), 2), f"{model} vs GARCH %": round(float(advantage[label]), 1),
        }
    return pd.DataFrame(rows).T.rename_axis("portfolio")


def minimum_risk_tables(model: str = "xgb_long_drv_har", store_root: Path = STORE_ROOT) -> dict[str, pd.DataFrame]:
    """
    The daily minimum-risk portfolio of the 23 tickers (the weights that minimise the forecast variance of the day, long only), from the stored
    forecasts of `model`: how the weights look, and the realized risk of its daily return against the equal-weight portfolio.
    """
    run = _portfolio_runs(store_root)["eq"][model]
    bank, tickers = run.ewma_bank, list(run.config["setting"]["tickers"])
    weights = core.daily_minimum_risk_weights(bank, tickers)
    equal = np.full(len(tickers), 1 / len(tickers))
    returns = {"minimum risk (daily weights)": core.weighted_simple_returns(bank, tickers, weights, run.unit), "equal weight (1/23)": core.weighted_simple_returns(bank, tickers, equal, run.unit)}
    forecast = {"minimum risk (daily weights)": core.portfolio_forecast_variance(bank, tickers, weights), "equal weight (1/23)": core.portfolio_forecast_variance(bank, tickers, equal)}
    realized = pd.DataFrame({
        label: {
            "volatility (% / year)": float(r.std() * np.sqrt(252) * 100), "forecast volatility (% / year)": float(np.sqrt(forecast[label].mean() * 252) * 100),
            "realized r² / forecast variance": float((r ** 2).mean() / forecast[label].mean()),
            "worst day (%)": float(r.min() * 100), "mean of the worst 5% days (%)": float(r[r <= r.quantile(0.05)].mean() * 100),
        }
        for label, r in returns.items()
    }).T
    held = pd.DataFrame({
        "mean weight (%)": weights.mean() * 100, "max weight (%)": weights.max() * 100, "days with weight 0 (%)": (weights < 1e-6).mean() * 100,
    }).sort_values("mean weight (%)", ascending=False)
    summary = pd.Series({
        "days": len(weights), "mean daily turnover (% of the portfolio)": float(weights.diff().abs().sum(axis=1).mean() * 100),
        "tickers with a weight above 1% (mean per day)": float((weights > 0.01).sum(axis=1).mean()), "largest weight of the day (mean)": float(weights.max(axis=1).mean()),
    }, name="value").to_frame()
    return {"realized risk": realized, "weights by ticker": held, "summary": summary}
