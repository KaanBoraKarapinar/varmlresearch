"""
Data access and tables for the Streamlit dashboard (no Streamlit import, so it can be tested alone).

Runs are LISTED from MLflow (the index) and READ from the run store (the parquet folders):
a MLflow run carries the tag `run_id` that run_store.load_walk_forward_run takes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import numpy as np
import pandas as pd

import core
from core import run_store as rs
from core import virtual_runs as vr
from core import walk_forward as wf

@dataclass(frozen=True)
class Settings:
    """What the sidebar decides: where the runs are listed from, where the data is, how the VaR is read."""
    tracking_uri: str
    experiment: str
    store_root: str
    reading: wf.Reading


HEADLINE_METRICS = [
    "total_window_var_metrics.n_exceedances", "total_window_var_metrics.exceedance_rate",
    "total_window_var_metrics.mean_tick_loss", "total_window_var_metrics.mean_var",
]
BACKTEST_ROWS = [
    "n_days", "n_exceedances", "exceedance_rate", "expected_exceedance_rate", "traffic_light_zone",
    "kupiec_pof_stat", "kupiec_pof_pass", "christoffersen_ind_pass", "christoffersen_cc_pass",
]
LEVEL_ROWS = [
    "mean_tick_loss", "tick_loss_exceedance_part", "tick_loss_overshoot_part", "mean_var", "median_var",
    "var_to_realized_quantile_ratio", "mean_exceedance_size", "max_exceedance_size", "var_cv",
    "relative_mean_abs_daily_change", "var_p95_p5_ratio",
]


# --------------------------------------------------------------------------------------------------
# MLflow (the list of runs)
# --------------------------------------------------------------------------------------------------

def list_experiments(tracking_uri: str) -> list[str]:
    mlflow.set_tracking_uri(tracking_uri)
    return [e.name for e in mlflow.search_experiments() if e.name != "Default"]


def _numeric_if_possible(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


def list_mlflow_runs(tracking_uri: str, experiment: str) -> pd.DataFrame:
    """
    One row per saved run (the newest MLflow run when a run was logged more than once): names, ids, every logged
    param as a column (numbers as numbers) and the logged metrics (own full period, see the comparison page for
    common days).
    """
    mlflow.set_tracking_uri(tracking_uri)
    raw = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time DESC"])
    if raw.empty or "tags.run_id" not in raw.columns:
        return pd.DataFrame()
    raw = raw.dropna(subset=["tags.run_id"]).drop_duplicates("tags.run_id")
    out = pd.DataFrame({
        "display_name": raw["tags.mlflow.runName"], "name": raw["params.run_name"],
        "store_run_id": raw["tags.run_id"], "walk_forward_id": raw["tags.walk_forward_id"],
        "store_path": raw["tags.store_path"], "mlflow_run_id": raw["run_id"],
    })
    for column in raw.columns:
        if column.startswith("params.") and column not in ("params.run_name", "params.reading"):
            out[column.removeprefix("params.")] = _numeric_if_possible(raw[column])
    for column in raw.columns:
        if column.startswith("metrics."):
            out[column.removeprefix("metrics.")] = raw[column]
    out["portfolio"] = out["walk_forward_id"].map(portfolio_label)
    out["display_name"] = _unique_labels(out)
    return out.sort_values(["train_bars", "name"]).reset_index(drop=True)


def portfolio_label(walk_forward_id: str) -> str:
    """'4 tickers' / '23 tickers' from a walk-forward id (<from..>_<to..>_<train>_<forecast>_<tickers with weights or n tickers>_<hash>)."""
    block = re.search(r"forecast\d+bars_(.+)_[0-9a-f]{8}$", walk_forward_id).group(1)
    count = re.fullmatch(r"(\d+)tickers", block)
    return f"{count.group(1)} tickers" if count else f"{block.count('-') + 1} tickers"


def _unique_labels(runs: pd.DataFrame) -> pd.Series:
    """
    The run labels the pages use as keys, so they must be unique. A display name that several walk-forwards share
    (same model, same train / forecast bars, other portfolio, other start or other data) gets, in this order and only as far
    as it takes to be unique: the number of tickers, the walk-forward's first bar, the hash of its id. Names that are
    already unique stay as they are.
    """
    labels = runs["display_name"].copy()
    shared = runs.groupby("display_name")["walk_forward_id"].transform("nunique") > 1
    labels[shared] = runs.loc[shared, "display_name"] + " | " + runs.loc[shared, "portfolio"]
    still = labels.groupby(labels).transform("size") > 1
    start = runs["walk_forward_id"].str.extract(r"^from(\d{4}-\d{2}-\d{2})")[0]
    labels[still] = labels[still] + " | from " + start[still]
    still = labels.groupby(labels).transform("size") > 1
    labels[still] = labels[still] + " #" + runs.loc[still, "walk_forward_id"].str[-8:]
    return labels


def list_virtual_runs(store_root: str) -> pd.DataFrame:
    """
    One row per portfolio run kept under <walk_forward>/VirtualRuns/ (run_store.save_virtual_run / save_baseline_runs), with the columns of
    list_mlflow_runs, built from the definitions in the ledgers (no run is loaded): label '<model> | <portfolio> | train N f M', 'portfolio' = the
    portfolio's name, 'kind' = reweighted | trained, and one tag_<key> column per tag of the portfolios (a filter for the group, the generator, ...).
    They are not in MLflow (no metrics).
    """
    root, rows = Path(store_root), []
    for walk_forward_id in vr.walk_forward_ids_with_virtual_runs(root):
        for definition in vr.list_definitions(root, walk_forward_id):
            config, portfolio = definition["config"], definition["config"]["portfolio"]
            rows.append({
                "display_name": config["display_name"], "name": config["name"], "store_run_id": config["run_id"],
                "walk_forward_id": walk_forward_id, "store_path": str(vr.virtual_runs_dir(root, walk_forward_id)), "mlflow_run_id": None,
                **config["params"], "portfolio": portfolio["name"], "kind": definition["kind"].replace("trained_baseline", "trained"),
                **{f"tag_{key}": value for key, value in portfolio.get("tags", {}).items()},
            })
    return pd.DataFrame(rows)


def virtual_runs_stamp(store_root: str) -> tuple:
    """The ledgers and the portfolio registry with their modification times: a cache key that changes when a portfolio run is saved."""
    root = Path(store_root)
    files = [root / "WalkForwardSetting" / w / "VirtualRuns" / "_ledger.jsonl" for w in vr.walk_forward_ids_with_virtual_runs(root)]
    files += [root / "Portfolios" / "portfolios.jsonl"]
    return tuple((str(p), p.stat().st_mtime_ns) for p in files if p.exists())


def list_runs(tracking_uri: str, experiment: str, store_root: str) -> pd.DataFrame:
    """What the pages list: the runs of MLflow and the portfolio runs of the store (list_virtual_runs), one table, one unique label each."""
    logged, saved = list_mlflow_runs(tracking_uri, experiment), list_virtual_runs(store_root)
    if saved.empty:
        return logged
    out = pd.concat([logged, saved], ignore_index=True)
    out["kind"] = out["kind"].fillna("trained")
    out["display_name"] = _unique_labels(out)
    return out.sort_values(["train_bars", "name"]).reset_index(drop=True)


def is_virtual_run_id(store_run_id: str) -> bool:
    """A run kept under VirtualRuns/ has '@<portfolio_id>' at the end of its id."""
    return "@" in store_run_id


def parameter_columns(runs: pd.DataFrame) -> list[str]:
    """The logged params that vary between the runs (what is worth grouping or comparing by)."""
    base = {"display_name", "name", "store_run_id", "walk_forward_id", "store_path", "mlflow_run_id"}
    metric_like = {c for c in runs.columns if c.startswith("total_window_")}
    return [c for c in runs.columns if c not in base | metric_like and runs[c].astype(str).nunique() > 1]


# --------------------------------------------------------------------------------------------------
# the run store (the data)
# --------------------------------------------------------------------------------------------------

def load_run(store_run_id: str, store_root: str) -> wf.Run:
    """A saved run (LoadedRun), or a portfolio run kept under VirtualRuns/ (id with '@<portfolio_id>'): StoredVirtualRun | StoredBaselineRun."""
    if is_virtual_run_id(store_run_id):
        return rs.load_virtual_run(store_run_id, Path(store_root))
    return rs.load_walk_forward_run(store_run_id, Path(store_root))


def run_span(run: wf.Run) -> dict:
    """The time the run covers: its first train window, its forecast days and how many windows."""
    windows = run.windows
    return {
        "first_train_start": windows["train_start"].iloc[0], "first_train_end": windows["train_end"].iloc[0],
        "first_forecast": windows["forecast_start"].iloc[0], "last_forecast": windows["forecast_end"].iloc[-1],
        "n_windows": len(windows), "n_forecast_days": int(windows["n_forecast"].sum()),
        "train_bars": int(run.params["train_bars"]),
    }


def has_variance_forecast(run: wf.Run) -> bool:
    """False for the 'historical' family: its VaR is a quantile of past returns, there is no variance forecast to show or score."""
    return run.params["model_family"] != "historical"


def common_days(runs: dict[str, wf.Run], reading: wf.Reading) -> pd.DatetimeIndex:
    """The days every run has forecast."""
    return rs.common_dates(list(runs.values()), reading)


def _var_dfs(runs: dict[str, wf.Run], reading: wf.Reading, dates) -> dict[str, pd.DataFrame]:
    return {label: run.var_df(reading).loc[dates] for label, run in runs.items()}


def backtest_tables(runs: dict[str, wf.Run], reading: wf.Reading, dates) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The backtest suite and the level / volatility suite, re-run on `dates` for every run (rows = items)."""
    backtest, level = {}, {}
    for label, var_df in _var_dfs(runs, reading, dates).items():
        backtest[label] = core.run_full_backtest_suite_single_window(
            var_df, var_df.index[0], var_df.index[-1], reading.var_confidence, reading.test_confidence,
        )
        level[label] = core.run_level_and_volatility_suite_single_window(var_df, reading.var_confidence)
    return pd.DataFrame(backtest), pd.DataFrame(level)


def compare_to_baseline(runs: dict[str, wf.Run], reading: wf.Reading, dates, baseline: str) -> pd.DataFrame:
    return core.compare_to_baseline(_var_dfs(runs, reading, dates), baseline, reading.var_confidence)


def exceedances_per_year(var_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Exceedance count per calendar year and run."""
    return pd.DataFrame({
        label: (v["loss"] > v["varForecast"]).groupby(v.index.year).sum() for label, v in var_dfs.items()
    }).rename_axis("year")


# --------------------------------------------------------------------------------------------------
# variance forecasts against the EWMA
# --------------------------------------------------------------------------------------------------

def variance_frame(runs: dict[str, wf.Run], reading: wf.Reading, dates) -> pd.DataFrame:
    """The portfolio variance each run predicted for `dates` (columns = runs)."""
    return pd.DataFrame({label: run.daily(reading)["predicted_variance"].loc[dates] for label, run in runs.items()})


def realized_squared_return(run: wf.Run, reading: wf.Reading, dates) -> pd.Series:
    """r^2 of the portfolio on `dates` (the noisy one-day realized variance every variance score is computed against)."""
    return (run.daily(reading)["loss"].loc[dates] ** 2).rename("realized_r2")


qlike = core.qlike   # lives in core (the walk-forward window losses use it too)


def variance_metrics(variance: pd.DataFrame, r2: pd.Series, ewma_label: str) -> pd.DataFrame:
    """
    One row per run on the same days: QLIKE (and its difference to the EWMA, negative = better than the EWMA),
    correlation of the predicted variance with r^2 and with the EWMA variance, mean / median ratio to the EWMA.
    """
    ewma = variance[ewma_label]
    base = qlike(ewma, r2)
    rows = {}
    for label in variance:
        h = variance[label]
        rows[label] = {
            "qlike": qlike(h, r2), "qlike_diff_vs_ewma": qlike(h, r2) - base,
            "corr_with_r2": float(h.corr(r2)), "corr_with_ewma": float(h.corr(ewma)),
            "mean_ratio_to_ewma": float((h / ewma).mean()), "median_ratio_to_ewma": float((h / ewma).median()),
            "mean_variance": float(h.mean()),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------------------------------

def ticker_variance_frames(run: wf.Run, dates=None) -> dict[str, pd.DataFrame]:
    """Per ticker: the model's variance, the EWMA's own variance and the squared return (long / ewma-family runs)."""
    bank = run.ewma_bank
    if dates is not None:
        bank = bank.loc[bank.index.intersection(dates)]
    frames = {}
    for ticker in run.config["setting"]["tickers"]:
        frames[ticker] = pd.DataFrame({
            "model variance": bank[f"sigma_{ticker}"], "EWMA variance": bank[f"cov_{ticker}_{ticker}"],
            "return^2": core.return_to_percent(bank[f"return_{ticker}"], run.unit) ** 2,
        })
    return frames


def ticker_returns(run: wf.Run) -> pd.DataFrame:
    """The simple return of every ticker of a long / ewma-family run on its forecast days (days x tickers, fractions): what any weights are applied to."""
    tickers = run.config["setting"]["tickers"]
    bank = run.ewma_bank
    return core.return_to_percent(bank[[f"return_{ticker}" for ticker in tickers]].set_axis(tickers, axis=1), run.unit)


def ticker_variance_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per ticker: mean variances, the ratio model / EWMA and the correlation of the two."""
    rows = {}
    for ticker, f in frames.items():
        rows[ticker] = {
            "mean model variance": f["model variance"].mean(), "mean EWMA variance": f["EWMA variance"].mean(),
            "mean ratio model/EWMA": (f["model variance"] / f["EWMA variance"]).mean(),
            "median ratio model/EWMA": (f["model variance"] / f["EWMA variance"]).median(),
            "corr(model, EWMA)": f["model variance"].corr(f["EWMA variance"]),
            "qlike model": qlike(f["model variance"], f["return^2"]), "qlike EWMA": qlike(f["EWMA variance"], f["return^2"]),
        }
    return pd.DataFrame(rows).T


def day_window_tables(
    daily: pd.DataFrame, ewma_variance: pd.Series | None, day, bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The `bars` days on each side of `day` (the day included): the VaR table and the portfolio-variance table."""
    position = daily.index.get_loc(pd.Timestamp(day))
    window = daily.iloc[max(position - bars, 0): position + bars + 1]
    var_table = window[["loss", "varForecast", "excess", "exceeded", "tick_loss", "n_residuals_used"]].copy()
    variance = pd.DataFrame({"predicted variance": window["predicted_variance"], "predicted sigma": window["predicted_sigma"]})
    if ewma_variance is not None:
        variance["EWMA variance"] = ewma_variance.reindex(window.index)
        variance["model / EWMA"] = variance["predicted variance"] / variance["EWMA variance"]
    variance["return²"] = window["loss"] ** 2
    variance["standardized residual"] = window["standardized_residual"]
    return var_table, variance


def run_header(run: wf.Run) -> dict:
    """What the 'Run' box shows: forecast period, tickers + weights, features, target (all declared in the run, units included)."""
    span = run_span(run)
    setting = run.config["setting"]
    header = {
        "forecast_start": span["first_forecast"], "forecast_end": span["last_forecast"],
        "tickers_weights": " · ".join(f"{t} {setting['weights'][t]:.0%}" for t in setting["tickers"]),
        "unit": run.unit, "features": None, "feature_unit": None, "target": None, "target_scale": None, "target_unit": None,
    }
    if run.params["model_family"] == "xgboost":
        header.update(
            features=run.params["feature_blocks"], feature_unit=run.params["feature_unit"],
            target=run.params["target"], target_scale=run.params["target_scale"], target_unit=run.params["target_unit"],
        )
    return header


def review_results(var_df: pd.DataFrame, reading: wf.Reading) -> tuple[dict, dict]:
    """The backtest suite and the level / volatility suite of one VaR frame (the analysis period of the page)."""
    backtest = core.run_full_backtest_suite_single_window(
        var_df, var_df.index[0], var_df.index[-1], reading.var_confidence, reading.test_confidence,
    )
    return backtest, core.run_level_and_volatility_suite_single_window(var_df, reading.var_confidence)


def unit_loss(run: wf.Run) -> pd.Series:
    """The realized portfolio loss in the run's OWN unit (the target is defined there); the readings are in percent returns."""
    residuals = run.combined_residuals
    return -residuals.loc[residuals["part"] == "forecast", "actual_return"]


def target_mse(run: wf.Run, variance: pd.Series, base: pd.Series | None = None) -> float:
    """
    Squared error in the model's own target units on the portfolio: the target of the realized return (in the run's own
    unit) against the target value a variance forecast corresponds to (a return of size sqrt(variance), ratio clip off).
    Only comparable between models with the same target and scale. NaN without a target, or for a ratio target without
    a base variance.
    """
    if run.params["model_family"] != "xgboost":
        return float("nan")
    target = run.recipe.target
    if target.needs_base and base is None:
        return float("nan")
    base_values = None if base is None else base.loc[variance.index].to_numpy()
    realized = target.forward_make_y(unit_loss(run).loc[variance.index].to_numpy(), base_values)
    predicted_target = dataclasses.replace(target, clip=None) if hasattr(target, "clip") else target
    predicted = predicted_target.forward_make_y(np.sqrt(variance.to_numpy()), base_values)
    return float(np.mean((predicted - realized) ** 2))


def variance_baselines(
    run: wf.Run, reading: wf.Reading, ewma_run: wf.Run | None, dates,
) -> pd.DataFrame:
    """The variance lines a model is set against, on `dates`: the model, the EWMA run, r², the target-definition variance, a 21-day realized variance."""
    daily = run.daily(reading)
    loss = daily["loss"]
    frame = pd.DataFrame({"model": daily["predicted_variance"]})
    if ewma_run is not None:
        frame["EWMA (reference run)"] = ewma_run.daily(reading)["predicted_variance"]
    frame["squared return r²"] = loss ** 2
    if run.params["model_family"] == "xgboost":
        target = run.recipe.target
        base = frame["EWMA (reference run)"].to_numpy() if target.needs_base and ewma_run is not None else None
        if not target.needs_base or base is not None:
            frame["target-definition variance"] = target.inverse_to_variance(
                target.forward_make_y(unit_loss(run).loc[daily.index].to_numpy(), base), base,
            )
    frame["realized variance (21-day mean of r²)"] = (loss ** 2).rolling(21, min_periods=5).mean()
    return frame.loc[frame.index.intersection(dates)]


def _models_folder(run: wf.Run) -> Path | None:
    """
    The folder of the saved models of a run: its own, or, for a portfolio run read off another run's per-ticker forecasts (the models do not depend on the
    weights), the base run's. None when no models are kept (a baseline kept under VirtualRuns/ is its residuals only, a training that is not saved has none).
    """
    if isinstance(run, wf.ReweightedVirtualRun):
        return _models_folder(run.base)
    run_dir = getattr(run, "run_dir", None)
    return None if run_dir is None else run_dir / "models"


def feature_importance(run: wf.Run) -> pd.DataFrame | None:
    folder = _models_folder(run)
    path = None if folder is None else folder / "feature_importance.parquet"
    return pd.read_parquet(path) if path is not None and path.exists() else None


def garch_params(run: wf.Run) -> pd.DataFrame | None:
    folder = _models_folder(run)
    path = None if folder is None else folder / "garch_params.json"
    if path is None or not path.exists():
        return None
    params = rs._read_json(path)
    first = next(iter(params.values()))
    if isinstance(next(iter(first.values())), dict):   # long: {window: {ticker: {param: value}}}
        return pd.concat({int(w): pd.DataFrame(p).T for w, p in params.items()}, names=["window_id", "ticker"])
    return pd.DataFrame(params).T.rename_axis("window_id")


# --------------------------------------------------------------------------------------------------
# weight adjustment: the same stored forecasts, other portfolio weights
# --------------------------------------------------------------------------------------------------

def run_table(runs: dict[str, wf.Run]) -> pd.DataFrame:
    """One row per run (index = its label) from its config: what the 'Selected runs' tab shows, for runs that are not in MLflow."""
    rows = {label: {**run.params, "store_run_id": run.run_id, "walk_forward_id": run.walk_forward_id} for label, run in runs.items()}
    return pd.DataFrame(rows).T.rename_axis("display_name")


def run_weights(run: wf.Run) -> pd.Series:
    setting = run.config["setting"]
    return pd.Series({t: float(setting["weights"][t]) for t in setting["tickers"]}, name="weight")


def baseline_runs(run: wf.Run, weights: dict[str, float], store_root: str) -> dict[str, wf.TrainedInitialRun]:
    """
    GARCH, EWMA and the historical VaR of the run's walk-forward for OTHER weights, computed now and not saved: their variance depends on
    the portfolio's return, so they cannot be re-weighted from stored rows. They are runs like any other (TrainedInitialRun).
    """
    stored = rs._read_json(rs.walk_forward_dir(run.walk_forward_id, Path(store_root)) / "WalkForwardSettings.json")
    prices = pd.read_csv(ROOT / stored["data"]["data_path"], parse_dates=["Date"], index_col="Date")
    setting = rs.load_walk_forward(run.walk_forward_id, prices, Path(store_root))
    setting = dataclasses.replace(setting, weights=dict(weights))
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):         # the calendars print a warning for every market holiday they skip
        calendars = setting.make_calendars()
        for name, recipe in wf.BASELINE_RECIPES.items():
            out[name] = rs.trained_initial_run(setting, recipe, wf.train_walk_forward(calendars, recipe), name)
    return out


def tick_losses(var_dfs: dict[str, pd.DataFrame], var_confidence: float) -> pd.DataFrame:
    """The daily tick loss of every series (columns). Not core.daily_tick_losses: portfolios with other weights have another realized loss."""
    return pd.DataFrame({label: core.daily_tick_loss(var_df, var_confidence) for label, var_df in var_dfs.items()})


def var_table(var_dfs: dict[str, pd.DataFrame], reading: wf.Reading) -> pd.DataFrame:
    """One row per VaR series on the same days: exceedances against the expected number, Kupiec, mean tick loss and mean VaR."""
    ticks = tick_losses(var_dfs, reading.var_confidence)
    rows = {}
    for label, var_df in var_dfs.items():
        suite = core.run_full_backtest_suite_single_window(var_df, var_df.index[0], var_df.index[-1], reading.var_confidence, reading.test_confidence)
        rows[label] = {
            "days": int(suite["n_days"]), "exceedances": int(suite["n_exceedances"]),
            "expected": round(float(len(var_df) * (1 - reading.var_confidence)), 1), "Kupiec ok": bool(suite["kupiec_pof_pass"]),
            "mean tick loss": float(ticks[label].mean()), "mean VaR": float(var_df["varForecast"].mean()),
        }
    return pd.DataFrame(rows).T


def minimum_risk_summary(run: wf.Run, weights: pd.DataFrame, fixed: pd.Series) -> dict[str, pd.DataFrame]:
    """
    The daily minimum-risk portfolio of the run against the portfolio with `fixed` weights and the equal-weight one: how the weights look
    (summary) and the realized risk of each portfolio's daily return (the day's weights applied to that day's returns).
    """
    bank, tickers = run.ewma_bank, list(weights.columns)
    equal = np.full(len(tickers), 1 / len(tickers))
    returns = {
        "minimum risk (daily weights)": core.weighted_simple_returns(bank, tickers, weights, run.unit),
        "the weights above": core.weighted_simple_returns(bank, tickers, fixed.loc[tickers].to_numpy(), run.unit),
        "equal weight": core.weighted_simple_returns(bank, tickers, equal, run.unit),
    }
    forecast = {
        "minimum risk (daily weights)": core.portfolio_forecast_variance(bank, tickers, weights),
        "the weights above": core.portfolio_forecast_variance(bank, tickers, fixed.loc[tickers].to_numpy()),
        "equal weight": core.portfolio_forecast_variance(bank, tickers, equal),
    }
    realized = pd.DataFrame({
        label: {
            "volatility (% / year)": float(r.std() * np.sqrt(252) * 100), "forecast volatility (% / year)": float(np.sqrt(forecast[label].mean() * 252) * 100),
            "realized r^2 / forecast variance": float((r ** 2).mean() / forecast[label].mean()),
            "worst day (%)": float(r.min() * 100), "mean of the worst 5% days (%)": float(r[r <= r.quantile(0.05)].mean() * 100),
        }
        for label, r in returns.items()
    }).T
    held = pd.DataFrame({
        "mean weight (%)": weights.mean() * 100, "min weight (%)": weights.min() * 100, "max weight (%)": weights.max() * 100,
        "days with 0 weight (%)": (weights < 1e-6).mean() * 100, "days with weight > 50% (%)": (weights > 0.5).mean() * 100,
    }).sort_values("mean weight (%)", ascending=False)
    return {"returns": pd.DataFrame(returns), "realized": realized, "weights": held}
