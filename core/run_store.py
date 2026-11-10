"""
Walk-forward run artifacts: every walk-forward run writes its intermediate outputs to disk once (parquet = the
source of truth), MLflow indexes them, and the loader + views read from disk, never from the training session.

    store/WalkForwardSetting/<walk_forward_id>/
        WalkForwardSettings.json, WindowsCreatedCalendars.parquet      the walk-forward (windows = calendars)
        FeatureAndTargets/<feature_target_key>/                        'none' for garch / ewma (no features)
            FeatureAndTarget.json
            WalkForwardTrainedModels/<family>-<format>-<model_key>/    = one RUN (a recipe on this walk-forward)
                model_settings.json                  name, recipe (rebuildable), flat params, keys, code, versions
                WALK_FORWARD_FORECAST_RESIDUAL_BANK.parquet   the combined residual bank (first window's train rows + every forecast row)
                WALK_FORWARD_EWMA_BANK.parquet       forecast rows of every window's ewma bank (long / ewma family)
                raw_residual_banks/calendar000.parquet ...   every window's own bank (train in-sample + forecast)
                raw_ewma_banks/calendar000.parquet ...       every window's own ewma bank
                models/                              calendar000.ubj + feature_importance.parquet | garch_params.json
                features/                            calendar000_train.parquet, calendar000_forecast.parquet (xgboost)
                readings_from_walk_forward/c0.95/window_size_252/minlength_30/
                    var_dataframe.parquet
                    testconfidence_0.95/  test_results.json, test_dataframe.parquet, exceeding_days.parquet,
                                          window_blocks.parquet

Terms (walk_forward.py): a WINDOW is one calendar (calendarNNN, NNN = window_id), a WALK-FORWARD is the set of
windows of one WalkForwardSetting, a RUN is one ModelRecipe trained on one walk-forward, a READING is how the VaR is
read off the combined residuals (it is an argument of the evaluation, not part of the run).

    train_and_save_walk_forward_run(setting, recipe, name)   train + save (or reuse an identical run)
    load_walk_forward_run(run_id)                            -> LoadedRun
    save_virtual_run(virtual) / save_baseline_runs(setting, weights)   a portfolio's runs kept under <walk_forward>/VirtualRuns/ (residuals only)
    load_virtual_run(run_id)                                 -> StoredVirtualRun | StoredBaselineRun
    trained_initial_run(setting, recipe, fits, name)         -> TrainedInitialRun (a training that is not saved)
    runs_table / load_runs                                   select runs by params
    leaderboard / common_dates / factor_table / window_matrix / explain_window / explain_day / exceedance_days
    log_walk_forward_run_to_mlflow(run, reading, experiment) index a run in MLflow (the folder stays the truth)
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import core
from core import EwmaVarianceBank
from core import walk_forward as wf
from core.ResidualBank import RESIDUAL_COLS
from core import portfolios
from core import virtual_runs as vr
# the run read back from disk is walk_forward.LoadedRun; run_store keeps the layout on disk, the saving and the views over runs
from core.walk_forward import (
    LoadedRun, Reading, Run, _calendar_name, _json_default, _parquet_metadata, _read_json, _to_parquet, _write_json,
    config_to_dict,
)

STORE_ROOT = Path("store")
_SETTINGS_DIR = "WalkForwardSetting"
_REPO_DIR = Path(__file__).resolve().parent.parent       # the project root (this file lives in core/): the git commands of the code info run there


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------

def _hash8(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=_json_default).encode()).hexdigest()[:8]


def _date_str(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _setting_config(setting: wf.WalkForwardSetting) -> dict:
    windows = setting.window_table()
    config = {
        "tickers": setting.tickers, "weights": setting.weights, "train_bars": setting.train_bars, "unit": setting.unit,
        "forecast_bars": setting.forecast_bars, "expanding": setting.expanding, "warmup_bars": setting.warmup_bars,
        "drop_market_closed": setting.drop_market_closed,
        # resolved dates: they rebuild the same windows (and the same id) without the None defaults
        "start_date": _date_str(windows["train_start"].iloc[0]), "end_date": _date_str(windows["forecast_end"].iloc[-1]),
    }
    if setting.additional_feature_sources:     # only then, so the config of a walk-forward without it stays as it was
        config["additional_feature_sources"] = [dataclasses.asdict(s) for s in setting.additional_feature_sources]
    return config


def _setting_from_config(config: dict, prices: pd.DataFrame, data_path: str | None = None) -> wf.WalkForwardSetting:
    config = dict(config)
    if "additional_feature_sources" in config:
        config["additional_feature_sources"] = tuple(core.AdditionalFeatureSource(**s) for s in config["additional_feature_sources"])
    return wf.WalkForwardSetting(prices, data_path=data_path, **config)


def _flat_params(setting: wf.WalkForwardSetting, recipe: wf.ModelRecipe) -> dict:
    params = {
        "train_bars": setting.train_bars, "forecast_bars": setting.forecast_bars, "expanding": setting.expanding,
        "unit": setting.unit,
        "model_family": recipe.model_family, "model_format": recipe.model_format,
        "mu_mode": recipe.mu_mode, "default_mu_hat": recipe.default_mu_hat,
        "ewma_lam": recipe.ewma_lam, "ewma_min_periods": recipe.ewma_min_periods,
    }
    if recipe.model_family == "xgboost":
        params["target"] = type(recipe.target).__name__
        params.update({f"target_{f.name}": getattr(recipe.target, f.name) for f in dataclasses.fields(recipe.target)})
        params["feature_blocks"] = recipe.features.describe()
        params["feature_unit"] = recipe.features.unit
    params.update({f"model_{k}": v for k, v in recipe.model_kwargs.items()})
    if recipe.pretrained_from is not None:
        params["pretrained_from"] = recipe.pretrained_from
    return params


def _feature_target_key(recipe: wf.ModelRecipe) -> str:
    """Hash of the features + target; garch / ewma read neither, so they share the folder 'none'."""
    if recipe.model_family != "xgboost":
        return "none"
    return _hash8([config_to_dict(recipe.features), config_to_dict(recipe.target)])


def _model_key(recipe: wf.ModelRecipe) -> str:
    """Hash of everything else that changes the model or its banks: family, format, model_kwargs, bank settings."""
    parts = [
        recipe.model_family, recipe.model_format, recipe.model_kwargs, recipe.mu_mode, recipe.default_mu_hat,
        recipe.rolling_mu_hat_window_size, recipe.ewma_lam, recipe.ewma_min_periods,
    ]
    if recipe.pretrained_from is not None:     # only then, so the keys of every run that trained its own models stay as they were
        parts.append(recipe.pretrained_from)
    return _hash8(parts)


def _model_dir_name(recipe: wf.ModelRecipe) -> str:
    return f"{recipe.model_family}-{recipe.model_format}-{_model_key(recipe)}"


def _keys(setting: wf.WalkForwardSetting, recipe: wf.ModelRecipe) -> dict:
    return {
        "walk_forward_id": setting.walk_forward_id, "train_bars": setting.train_bars,
        "feature_target_key": _feature_target_key(recipe), "model_key": _model_key(recipe),
    }


def _default_run_name(recipe: wf.ModelRecipe) -> str:
    if recipe.model_family == "xgboost":
        return f"xgboost_{recipe.model_format}_{type(recipe.target).__name__}"
    return f"{recipe.model_family}_{recipe.model_format}"


def _display_name(name: str, setting_config: dict) -> str:
    """'<name> | train N f M', and ' | log' appended for a log run (percent is the standard and stays unmarked)."""
    suffix = "" if setting_config["unit"] == "percent" else f" | {setting_config['unit']}"
    return f"{name} | train {setting_config['train_bars']} f {setting_config['forecast_bars']}{suffix}"


def _code_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=_REPO_DIR, capture_output=True, text=True).stdout
    status = git("status", "--porcelain", "--", "*.py").strip()
    return {
        "git_commit": git("rev-parse", "HEAD").strip(), "dirty": bool(status),
        "diff": git("diff", "HEAD", "--", "*.py") if status else "",
    }


def _versions() -> dict:
    out = {}
    for package in ("xgboost", "arch", "pandas", "numpy", "pyarrow"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


# --------------------------------------------------------------------------------------------------
# the walk-forward folder
# --------------------------------------------------------------------------------------------------

def walk_forward_dir(walk_forward_id: str, root: Path = STORE_ROOT) -> Path:
    return Path(root) / _SETTINGS_DIR / walk_forward_id


def _run_dir(walk_forward_id: str, feature_target_key: str, model_dir_name: str, root: Path = STORE_ROOT) -> Path:
    return (
        walk_forward_dir(walk_forward_id, root) / "FeatureAndTargets" / feature_target_key
        / "WalkForwardTrainedModels" / model_dir_name
    )


def _model_settings_paths(root: Path, walk_forward_id: str = "*") -> list[Path]:
    """Every finished run's model_settings.json (a run still being written is called <dir>.partial)."""
    pattern = f"{_SETTINGS_DIR}/{walk_forward_id}/FeatureAndTargets/*/WalkForwardTrainedModels/*/model_settings.json"
    return [p for p in sorted(Path(root).glob(pattern)) if not p.parent.name.endswith(".partial")]


def save_walk_forward(setting: wf.WalkForwardSetting, root: Path = STORE_ROOT) -> Path:
    """WalkForwardSettings.json + WindowsCreatedCalendars.parquet. An existing folder of the same id must hold the same inputs."""
    folder = walk_forward_dir(setting.walk_forward_id, root)
    index = setting.tickers_raw_price_data.sort_index().index
    content = {
        "walk_forward_id": setting.walk_forward_id,
        "setting": _setting_config(setting),
        "data": {
            "data_path": setting.data_path, "sha256": setting.data_sha256,
            "first_date": _date_str(index[0]), "last_date": _date_str(index[-1]),
        },
    }
    if folder.exists():
        stored = _read_json(folder / "WalkForwardSettings.json")
        if stored["walk_forward_id"] != content["walk_forward_id"] or stored["setting"] != json.loads(
            json.dumps(content["setting"], default=_json_default)
        ) or stored["data"]["sha256"] != content["data"]["sha256"]:
            raise ValueError(f"{folder} holds a walk-forward with other inputs than its id says; refusing to overwrite.")
        return folder
    folder.mkdir(parents=True)
    _write_json(folder / "WalkForwardSettings.json", content)
    _to_parquet(
        setting.window_table(), folder / "WindowsCreatedCalendars.parquet",
        {"walk_forward_id": setting.walk_forward_id, "unit": setting.unit},
    )
    return folder


def load_walk_forward(
    walk_forward_id: str, tickers_raw_price_data: pd.DataFrame, root: Path = STORE_ROOT, *, check_data_values: bool = True,
) -> wf.WalkForwardSetting:
    """
    Rebuild the setting from WalkForwardSettings.json + the price data, checking the hash, the windows and the data.

    check_data_values=False: the last part of the id (the hash8, which includes a hash of the parsed VALUES of the price and the additional feature CSVs) may differ.
    The same CSV parses to other last bits on another processor (pandas' default float parser works in the platform's long double, which is 64 bit on an arm64 Mac
    and 80 bit on x86 Linux), so a store trained on one machine does not reproduce its hash on the other. Everything the id spells out (dates, bar counts, tickers
    and weights) and the saved windows must still match. For a machine that only reads a store (the deployed dashboard); training keeps the strict check.
    """
    folder = walk_forward_dir(walk_forward_id, root)
    stored = _read_json(folder / "WalkForwardSettings.json")
    setting = _setting_from_config(stored["setting"], tickers_raw_price_data, stored["data"]["data_path"])
    if setting.walk_forward_id != walk_forward_id and (check_data_values or setting.walk_forward_id.rpartition("_")[0] != walk_forward_id.rpartition("_")[0]):
        raise ValueError(
            f"The price data does not reproduce {walk_forward_id} (got {setting.walk_forward_id}): other data or other inputs."
        )
    saved_windows = pd.read_parquet(folder / "WindowsCreatedCalendars.parquet")
    pd.testing.assert_frame_equal(setting.window_table(), saved_windows, check_dtype=False)
    return setting


# --------------------------------------------------------------------------------------------------
# saving one run
# --------------------------------------------------------------------------------------------------

def _float_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    """The bank's frame starts as object columns: numbers on disk."""
    return residuals[RESIDUAL_COLS].astype(float)


def _write_combined_residuals(path: Path, fits: list[wf.WalkForwardSingleCalendarFit], metadata: dict) -> None:
    _to_parquet(wf.build_combined_residuals_frame(fits), path, metadata)


def _write_raw_residual_banks(folder: Path, fits: list[wf.WalkForwardSingleCalendarFit], metadata: dict) -> None:
    folder.mkdir()
    for fit in fits:
        frame = _float_residuals(fit.residual_bank.residuals)
        frame["part"] = np.where(frame.index.isin(fit.calendar.forecast_dates), "forecast", "train")
        _to_parquet(frame, folder / f"{_calendar_name(fit.window_id)}.parquet", metadata)


def _write_ewma(run_dir: Path, fits: list[wf.WalkForwardSingleCalendarFit], metadata: dict, keep_raw: bool = True) -> None:
    if fits[0].ewma_bank is None:
        return
    _to_parquet(wf.build_combined_ewma_bank_WITHOUT_CALC(fits), run_dir / wf.EWMA_BANK_FILE, metadata)
    _to_parquet(wf.build_first_window_training_bank(fits), run_dir / wf.FIRST_WINDOW_TRAINING_BANK_FILE, metadata)
    if not keep_raw:
        return
    (run_dir / "raw_ewma_banks").mkdir()
    for fit in fits:
        _to_parquet(wf.ewma_bank_to_frame(fit.ewma_bank), run_dir / "raw_ewma_banks" / f"{_calendar_name(fit.window_id)}.parquet", metadata)


def _write_models(folder: Path, fits: list[wf.WalkForwardSingleCalendarFit], recipe: wf.ModelRecipe, metadata: dict) -> None:
    if recipe.model_family in ("ewma", "historical"):
        return
    folder.mkdir()
    if recipe.model_family == "xgboost":
        importance = {}
        for fit in fits:
            model = fit.trained_model
            model.underlying_model.save_model(str(folder / f"{_calendar_name(fit.window_id)}.ubj"))
            importance[fit.window_id] = pd.Series(model.underlying_model.feature_importances_, index=model.feature_cols)
        _to_parquet(pd.DataFrame(importance).T.rename_axis("window_id"), folder / "feature_importance.parquet", metadata)
    else:
        def as_json(params):
            return {t: p.to_dict() for t, p in params.items()} if isinstance(params, dict) else params.to_dict()
        _write_json(folder / "garch_params.json", {str(f.window_id): as_json(f.trained_model.params) for f in fits})


def _feature_rows(X, ids, y, r_next, base_variance, outcome_dates=None, feature_dates=None) -> pd.DataFrame:
    """One feature matrix as one frame: Date (+ ticker), the features, y, r_next, base_variance, outcome date."""
    if ids is None:
        frame = X.rename_axis("Date").reset_index()
    else:
        frame = X.reset_index(drop=True)
        frame.insert(0, "ticker", ids["ticker"].to_numpy())
        frame.insert(0, "Date", ids["Date"].to_numpy())
    n = len(frame)
    for name, values in (("y", y), ("r_next", r_next), ("base_variance", base_variance)):
        if values is None:
            continue
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != n:
            raise ValueError(f"feature column {name} has {len(values)} rows, X has {n}.")
        frame[name] = values
    if outcome_dates is not None:
        frame["residual_outcome_date"] = frame["Date"].map(dict(zip(feature_dates, outcome_dates)))
    return frame


def _write_features(folder: Path, fits: list[wf.WalkForwardSingleCalendarFit], recipe: wf.ModelRecipe, metadata: dict) -> None:
    if recipe.model_family != "xgboost":
        return
    folder.mkdir()
    for fit in fits:
        tf = fit.trained_model.training_feature
        ff = fit.forecast_feature
        calendar_name = _calendar_name(fit.window_id)
        _to_parquet(_feature_rows(
            tf.X_in_sample, tf.X_in_sample_id_cols, tf.y_in_sample, tf.DANGEROUS_r_next_in_sample, tf.base_variance,
            tf.training_feature_residual_outcome_dates, tf.feature_dates,
        ), folder / f"{calendar_name}_train.parquet", metadata)
        _to_parquet(_feature_rows(
            ff.X_forecast_sample, ff.id_cols, ff.y_forecast_sample, ff.DANGEROUS_r_next_forecast_sample, ff.base_variance,
        ), folder / f"{calendar_name}_forecast.parquet", metadata)


def _model_settings(setting: wf.WalkForwardSetting, recipe: wf.ModelRecipe, name: str, fits, slim: bool = False) -> dict:
    setting_config = _setting_config(setting)
    frequency = fits[0].residual_bank.frequency
    params = _flat_params(setting, recipe)
    if fits[0].trained_model is not None:   # the loss the model minimized, kept by the trained model itself
        params["training_loss"] = fits[0].trained_model.loss_definition
    if slim:       # only then, so the settings of every other run stay as they were
        params["slim"] = True
    return {
        "run_id": _run_id(setting.walk_forward_id, recipe), "name": name,
        "display_name": _display_name(name, setting_config),
        "walk_forward_id": setting.walk_forward_id,
        "setting": setting_config, "recipe": config_to_dict(recipe),
        "params": params, "keys": _keys(setting, recipe),
        "data": {"data_path": setting.data_path, "sha256": setting.data_sha256},
        "frequency": None if frequency is None else str(frequency),
        "code": _code_info(), "versions": _versions(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def trained_initial_run(
    setting: wf.WalkForwardSetting, recipe: wf.ModelRecipe, fits: list[wf.WalkForwardSingleCalendarFit], name: str,
) -> wf.TrainedInitialRun:
    """A training that has not been saved, read like a saved run (nothing is written): the same model_settings a saved run gets."""
    return wf.TrainedInitialRun(_model_settings(setting, recipe, name, fits), setting.window_table(), fits)


def _run_id(walk_forward_id: str, recipe: wf.ModelRecipe) -> str:
    return f"{walk_forward_id}/{_feature_target_key(recipe)}/{_model_dir_name(recipe)}"


def save_walk_forward_run(
    setting: wf.WalkForwardSetting, recipe: wf.ModelRecipe, fits: list[wf.WalkForwardSingleCalendarFit],
    name: str, root: Path = STORE_ROOT, slim: bool = False,
) -> Path:
    """
    Write the run folder (all or nothing: written aside as <dir>.partial, then renamed).
    slim: leave out the per-window feature files and the per-window raw EWMA banks, which are most of the bytes of a run with
    many tickers (the models, the combined banks, the residual banks and every reading stay): run.features() and run.raw_ewma_bank() are not available then.
    """
    save_walk_forward(setting, root)
    final = _run_dir(setting.walk_forward_id, _feature_target_key(recipe), _model_dir_name(recipe), root)
    feature_target_json = final.parents[1] / "FeatureAndTarget.json"
    if not feature_target_json.exists():
        feature_target_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json(feature_target_json, (
            {"features": config_to_dict(recipe.features), "target": config_to_dict(recipe.target)}
            if recipe.model_family == "xgboost" else {"features": None, "target": None}
        ))
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    settings_content = _model_settings(setting, recipe, name, fits, slim)
    metadata = _parquet_metadata(settings_content)
    _write_json(partial / "model_settings.json", settings_content)
    _write_combined_residuals(partial / "WALK_FORWARD_FORECAST_RESIDUAL_BANK.parquet", fits, metadata)
    _write_raw_residual_banks(partial / "raw_residual_banks", fits, metadata)
    _write_ewma(partial, fits, metadata, keep_raw=not slim)
    _to_parquet(wf.window_losses(fits), partial / "window_losses.parquet", metadata)
    _write_models(partial / "models", fits, recipe, metadata)
    if not slim:
        _write_features(partial / "features", fits, recipe, metadata)
    partial.rename(final)
    return final


def train_and_save_walk_forward_run(
    setting: wf.WalkForwardSetting, recipe: wf.ModelRecipe, name: str | None = None, root: Path = STORE_ROOT, slim: bool = False,
    workers: int | None = None, progress=None,
) -> LoadedRun:
    """
    Train the recipe on the walk-forward and save it. The recipe decides the folder (feature+target key, model key),
    `name` is its label. The same recipe is reused (nothing is trained); the same name for another recipe, or another
    name for a saved recipe, raises: a name never silently changes meaning. slim: see save_walk_forward_run.
    workers (needs slim): fit the windows `workers` processes at a time, each window reduced (wf.slim_fit) as it is made, so a walk-forward with thousands of windows (a
    1-day forecast) fits in memory; the run saved is the one slim=True saves without workers. None: the windows are fitted one after the other, all kept in full.
    progress(windows_done, n_windows): called while the windows are fitted (with workers), to follow a long training.
    """
    name = name or _default_run_name(recipe)
    if "/" in name:
        raise ValueError(f"run name {name!r} must not contain '/'.")
    run_id = _run_id(setting.walk_forward_id, recipe)
    for path in _model_settings_paths(root, setting.walk_forward_id):
        stored = _read_json(path)
        if stored["run_id"] == run_id and stored["name"] != name:
            raise ValueError(f"this recipe is already saved as {stored['name']!r}, not {name!r}.")
        if stored["name"] == name and stored["run_id"] != run_id:
            raise ValueError(f"the name {name!r} is already used by another recipe of this walk-forward; pick another name.")
        if stored["run_id"] == run_id:
            return load_walk_forward_run(run_id, root)
    pretrained = None if recipe.pretrained_from is None else load_walk_forward_run(recipe.pretrained_from, root)
    if workers is not None:
        if not slim:
            raise ValueError("workers needs slim=True: the windows are reduced to what a slim run keeps as they are made.")
        fits = wf.train_walk_forward_slim(setting, recipe, workers, progress=progress)
    else:
        fits = wf.train_walk_forward(setting.make_calendars(), recipe, pretrained)
    save_walk_forward_run(setting, recipe, fits, name, root, slim)
    return load_walk_forward_run(run_id, root)


def load_walk_forward_run(run_id: str, root: Path = STORE_ROOT) -> LoadedRun:
    """run_id = <walk_forward_id>/<feature_target_key>/<model dir>, as in runs_table's index."""
    walk_forward_id, feature_target_key, model_dir_name = run_id.split("/")
    run_dir = _run_dir(walk_forward_id, feature_target_key, model_dir_name, root)
    if not (run_dir / "model_settings.json").exists():
        raise FileNotFoundError(f"no run {run_id} under {root}")
    return LoadedRun(run_dir)


def save_virtual_run(
    virtual: wf.ReweightedVirtualRun, root: Path = STORE_ROOT, batch: dict | None = None,
    name: str | None = None, tags: dict | None = None, name_prefix: str | None = None,
) -> str:
    """
    Keep a ReweightedVirtualRun: its combined residuals go to <walk_forward>/VirtualRuns/ (core.virtual_runs), with the weights and the run
    it was read from, so load_virtual_run opens it later with any VaR reading. The portfolio is registered (core.portfolios) under `name`
    (None: described by its weights, after `name_prefix`) with its `tags`, and the run is linked to it; the registry is checked first, so a
    name that is taken writes nothing.
    A portfolio of a base run is kept once: saving it again changes nothing. batch: free provenance for the ledger.
    Returns the run_id (<base run_id>@<portfolio_id>).
    """
    weights = {ticker: float(w) for ticker, w in zip(virtual.tickers, virtual.weights)}
    portfolios.register_portfolio(root, weights, name, tags, name_prefix)
    return save_virtual_runs([virtual], root, batch)[0]


def _ledger_config(config: dict) -> dict:
    """
    A run's model_settings for the ledger: JSON values, without the git diff of the code (it is the size of the working tree's uncommitted changes,
    ~100 KB in every ledger line); the commit and whether the tree was dirty stay.
    """
    config = json.loads(json.dumps(config, default=_json_default))
    if "code" in config:
        config["code"] = {k: v for k, v in config["code"].items() if k != "diff"}
    return config


def save_virtual_runs(virtuals: list[wf.ReweightedVirtualRun], root: Path = STORE_ROOT, batch: dict | None = None) -> list[str]:
    """
    Keep the virtual runs of SEVERAL portfolios at once: the portfolios of one base run go into ONE new part file (not a file per portfolio), which
    is what keeps a few hundred portfolios at a few files per model. The portfolios have to be registered already (save_virtual_run registers one; the
    label of a run carries the registered name). A run that is already kept is not written again. Returns the run_ids in the order given.
    """
    groups: dict[tuple[str, str], list[wf.ReweightedVirtualRun]] = {}
    for virtual in virtuals:
        groups.setdefault((virtual.base.walk_forward_id, virtual.base.name), []).append(virtual)
    records = portfolios.portfolios_by_id(root)
    links = []
    for (walk_forward_id, model), members in groups.items():
        kept = set(vr.portfolio_ids(root, walk_forward_id, model))
        frames, definitions = {}, {}
        for virtual in members:
            weights = {ticker: float(w) for ticker, w in zip(virtual.tickers, virtual.weights)}
            portfolio = vr.portfolio_id(weights)
            if portfolio not in records:
                raise KeyError(f"portfolio {portfolio} is not registered: register it first (core.portfolios.register_portfolio).")
            links.append((portfolio, virtual.run_id, "reweighted"))
            if portfolio in kept or portfolio in frames:
                continue
            labelled = wf.ReweightedVirtualRun(virtual.base, weights, records[portfolio])      # the run as the dashboard lists it: its label carries the portfolio name
            frames[portfolio] = virtual.combined_residuals
            definitions[portfolio] = {
                "kind": "reweighted", "run_id": virtual.run_id, "weights": weights, "base_run_id": virtual.base.run_id,
                "config": _ledger_config(labelled.config),
            }
        if frames:
            vr.save_residuals(root, walk_forward_id, model, frames, batch=batch, definitions=definitions)
    portfolios.link_runs(root, links)
    return [virtual.run_id for virtual in virtuals]


def save_baseline_runs(
    base_setting: wf.WalkForwardSetting, weights: dict[str, float], root: Path = STORE_ROOT,
    names: tuple[str, ...] = ("garch_wide", "historical"), name: str | None = None, tags: dict | None = None,
    name_prefix: str | None = None, batch: dict | None = None,
) -> dict[str, str]:
    """
    Train baselines (wf.BASELINE_RECIPES) for ONE portfolio: base_setting with these weights, its windows and data. They depend on the portfolio's own
    return, so they are trained again for every portfolio, and only their combined residuals are kept, under the BASE walk-forward's VirtualRuns/
    (core.virtual_runs): no folder per portfolio, no per-window files. The recipes are kept once as VirtualRuns/BaselineModelDefs/<name>.json and the
    complete model_settings of each run in the ledger, so a run can be read, listed and, if wanted, trained again as a full run.
    A baseline already kept for the portfolio is not trained again. Returns {run name: run_id}, run_id = <base walk-forward>/<ft key>/<model dir>@<portfolio_id>.
    """
    record = portfolios.register_portfolio(root, weights, name, tags, name_prefix)
    return save_baseline_runs_for(base_setting, [record], root, names, batch)[record["portfolio_id"]]


def save_baseline_runs_for(
    base_setting: wf.WalkForwardSetting, records: list[dict], root: Path = STORE_ROOT,
    names: tuple[str, ...] = ("garch_wide", "historical"), batch: dict | None = None,
) -> dict[str, dict[str, str]]:
    """
    save_baseline_runs for SEVERAL registered portfolios (records: their registry records, core.portfolios): each baseline is trained for every portfolio
    that does not have it yet and the residuals of ALL of them are written together, ONE new part file per baseline, so a few hundred portfolios stay
    at a few files per baseline. Nothing is written until the baselines of every portfolio are trained: the number of portfolios per call is the
    size of what an interruption loses. Returns {portfolio_id: {run name: run_id}}.
    """
    unknown = [n for n in names if n not in wf.BASELINE_RECIPES]
    if unknown:
        raise ValueError(f"unknown baselines {unknown}; known: {list(wf.BASELINE_RECIPES)}.")
    base_id = base_setting.walk_forward_id
    save_walk_forward(base_setting, root)               # the walk-forward folder with its windows (an existing one must hold the same inputs)
    run_ids = {
        record["portfolio_id"]: {
            n: f"{base_id}/{_feature_target_key(wf.BASELINE_RECIPES[n])}/{_model_dir_name(wf.BASELINE_RECIPES[n])}@{record['portfolio_id']}" for n in names
        } for record in records
    }
    kept = {n: set(vr.portfolio_ids(root, base_id, n)) for n in names}
    frames: dict[str, dict] = {n: {} for n in names}
    definitions: dict[str, dict] = {n: {} for n in names}
    for record in records:
        portfolio = record["portfolio_id"]
        todo = [n for n in names if portfolio not in kept[n]]
        if not todo:
            continue
        setting = dataclasses.replace(base_setting, weights={t: float(record["weights"][t]) for t in base_setting.tickers})
        with contextlib.redirect_stdout(io.StringIO()):      # the calendars print a line for every market holiday they skip
            calendars = setting.make_calendars()
        for run_name in todo:
            recipe = wf.BASELINE_RECIPES[run_name]
            with contextlib.redirect_stdout(io.StringIO()):
                fits = wf.train_walk_forward(calendars, recipe)
            config = _model_settings(setting, recipe, run_name, fits)
            config["trained_on_walk_forward_id"] = config["walk_forward_id"]        # the id a full run for these weights would get
            config["walk_forward_id"] = config["keys"]["walk_forward_id"] = base_id
            config["run_id"] = run_ids[portfolio][run_name]
            config["display_name"] = wf._label_with_portfolio(run_name, config["display_name"], record["name"])
            config["portfolio"] = {k: record[k] for k in ("portfolio_id", "name", "tags")}
            frames[run_name][portfolio] = wf.build_combined_residuals_frame(fits)
            definitions[run_name][portfolio] = {
                "kind": "trained_baseline", "run_id": run_ids[portfolio][run_name], "weights": record["weights"],
                "recipe_key": _model_key(recipe), "config": _ledger_config(config),
            }
    for run_name in names:
        if not frames[run_name]:
            continue
        recipe = wf.BASELINE_RECIPES[run_name]
        vr.ensure_model_definition(root, base_id, run_name, {
            "run_name": run_name, "recipe": config_to_dict(recipe), "recipe_key": _model_key(recipe), "versions": _versions(),
        })
        vr.save_residuals(root, base_id, run_name, frames[run_name], batch=batch, definitions=definitions[run_name])
    portfolios.link_runs(root, [(portfolio, run_id, "trained") for portfolio, by_name in run_ids.items() for run_id in by_name.values()])
    return run_ids


def load_virtual_run(run_id: str, root: Path = STORE_ROOT) -> wf.Run:
    """
    A run kept under VirtualRuns/ (run_id = <walk_forward>/<ft key>/<model dir>@<portfolio_id>): a StoredVirtualRun (another run's per-ticker
    forecasts read for the portfolio's weights, save_virtual_run) or a StoredBaselineRun (a baseline trained for the weights, save_baseline_runs).
    """
    _, separator, portfolio = run_id.rpartition("@")
    if not separator or not portfolio:
        raise ValueError(f"{run_id!r} is not a virtual run id (<base run_id>@<portfolio_id>).")
    try:
        definition = vr.find_definition(root, run_id.split("/")[0], run_id)
    except KeyError:
        raise FileNotFoundError(f"no saved virtual run {run_id} under {root}") from None
    if definition.get("kind") == "trained_baseline":
        windows = pd.read_parquet(walk_forward_dir(run_id.split("/")[0], root) / "WindowsCreatedCalendars.parquet")
        return wf.StoredBaselineRun(definition["config"], windows, root, definition["model"])
    base = load_walk_forward_run(definition["base_run_id"], root)
    return wf.StoredVirtualRun(base, definition["weights"], root, portfolios.get_portfolio(root, portfolio))


# --------------------------------------------------------------------------------------------------
# selecting runs and the views
# --------------------------------------------------------------------------------------------------

def runs_table(root: Path = STORE_ROOT, **filters) -> pd.DataFrame:
    """
    One row per saved run (index run_id): its name + keys + flat params + created_at. filters: column=value or
    column=[values], e.g. runs_table(train_bars=756, model_family='xgboost', target='RatioToBase').
    """
    rows = {}
    for path in _model_settings_paths(root):
        config = _read_json(path)
        rows[config["run_id"]] = {
            "name": config["name"], "display_name": config["display_name"],
            **config["keys"], **config["params"], "created_at": config["created_at"],
        }
    table = pd.DataFrame(rows).T.rename_axis("run_id")
    for column, wanted in filters.items():
        if column not in table.columns:
            raise KeyError(f"unknown filter column {column!r}; columns: {list(table.columns)}")
        wanted = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        table = table[table[column].isin(wanted)]
    return table


def load_runs(root: Path = STORE_ROOT, **filters) -> list[LoadedRun]:
    return [load_walk_forward_run(run_id, root) for run_id in runs_table(root, **filters).index]


def _by_walk_forward(runs: list[Run]) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.walk_forward_id, []).append(run)
    return groups


def common_dates(runs: list[Run], reading: Reading) -> pd.DatetimeIndex:
    """The days every given run has forecast (computed on the fly from the saved test_dataframe files)."""
    dates = None
    for run in runs:
        index = run.daily(reading).index
        dates = index if dates is None else dates.intersection(index)
    return dates.sort_values()


def _names(runs: list[Run]) -> list[str]:
    """The run names when the runs share one walk-forward, 'name | train N f M' otherwise."""
    return [r.name for r in runs] if len(_by_walk_forward(runs)) == 1 else [r.display_name for r in runs]


_METRICS = {
    "mean_tick_loss": lambda daily: daily["tick_loss"].mean(),
    "exceedance_rate": lambda daily: daily["exceeded"].mean(),
    "mean_var": lambda daily: daily["varForecast"].mean(),
}


def leaderboard(
    runs: list[Run], reading: Reading, baseline: str, last_days: int | None = None,
) -> pd.DataFrame:
    """
    Every run against the run named `baseline` of the SAME walk-forward, on the days that group's runs share
    (core.compare_to_baseline: differences are run - baseline, NEGATIVE = better; DM test with HAC).
    last_days: only the last N common days (e.g. to look at a holdout once at the end).
    """
    out = {}
    for walk_forward_id, group in _by_walk_forward(runs).items():
        dates = common_dates(group, reading)
        if last_days is not None:
            dates = dates[-last_days:]
        var_dfs = {run.name: run.var_df(reading).loc[dates] for run in group}
        if baseline not in var_dfs:
            raise KeyError(f"{walk_forward_id}: no run named {baseline!r} among {list(var_dfs)}")
        out[walk_forward_id] = core.compare_to_baseline(var_dfs, baseline, reading.var_confidence)
    return pd.concat(out, names=["walk_forward_id", "run"])


def factor_table(
    runs: list[Run], factor: str, reading: Reading, metric: str = "mean_tick_loss",
) -> pd.DataFrame:
    """A metric by one param (e.g. 'model_n_estimators') x train_bars, each walk-forward on its own common days."""
    rows = []
    for group in _by_walk_forward(runs).values():
        dates = common_dates(group, reading)
        for run in group:
            if factor in run.params:
                rows.append({
                    factor: run.params[factor], "train_bars": run.params["train_bars"],
                    metric: float(_METRICS[metric](run.daily(reading).loc[dates])),
                })
    return pd.DataFrame(rows).pivot_table(index=factor, columns="train_bars", values=metric)


def window_matrix(runs: list[Run], reading: Reading, metric: str = "mean_tick_loss") -> pd.DataFrame:
    """windows x runs of a per-window metric (a column of window_blocks): adding a run adds a column."""
    return pd.DataFrame({name: run.blocks(reading)[metric].astype(float) for name, run in zip(_names(runs), runs)})


def explain_window(window_id: int, runs: list[Run], reading: Reading) -> pd.DataFrame:
    """The window's forecast days with the runs side by side: (run, varForecast | loss | exceeded)."""
    parts = {}
    for name, run in zip(_names(runs), runs):
        daily = run.daily(reading)
        parts[name] = daily.loc[daily["window_id"] == window_id, ["varForecast", "loss", "exceeded"]]
    return pd.concat(parts, axis=1)


def exceedance_days(runs: list[Run], reading: Reading) -> pd.DataFrame:
    """core.exceedance_days_by_model on the common days of the given runs (use runs of ONE walk-forward)."""
    dates = common_dates(runs, reading)
    return core.exceedance_days_by_model({n: r.var_df(reading).loc[dates] for n, r in zip(_names(runs), runs)})


def explain_day(date, run: Run, reading: Reading) -> pd.DataFrame:
    """
    One table that decomposes the VaR of one forecast day: the model variance (long / ewma: the per-ticker
    variances, the EWMA correlation and the weights that gave the portfolio variance), mu_hat, sigma, the quantile
    and the residual window it came from, VaR = -(mu_hat + sigma * q), the loss and how it was exceeded.
    """
    date = pd.Timestamp(date)
    daily = run.daily(reading)
    if date not in daily.index:
        raise KeyError(f"{date.date()} is not a forecast day of {run.display_name}")
    day = daily.loc[date]
    rows: list[tuple[str, object, str]] = [("window_id", int(day["window_id"]), "the window whose model forecast this day")]

    if run.has_ewma_bank:
        tickers = run.config["setting"]["tickers"]
        weights = np.array([run.config["setting"]["weights"][t] for t in tickers])
        ewma_row = run.ewma_bank.loc[date]
        sigma_diag = np.array([ewma_row[f"sigma_{t}"] for t in tickers], dtype=float)
        cov = np.array([[ewma_row[f"cov_{i}_{j}"] for j in tickers] for i in tickers], dtype=float)
        rebuilt = EwmaVarianceBank._portfolio_variance_from_cov(cov, sigma_diag, weights)
        for t, w, s, c in zip(tickers, weights, sigma_diag, np.diag(cov)):
            rows += [(f"weight[{t}]", w, ""), (f"model_variance[{t}]", s, "the diagonal of the portfolio covariance"),
                     (f"ewma_variance[{t}]", c, "the EWMA's own diagonal (not used by a model, only its correlation)")]
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        for a in range(len(tickers)):
            for b in range(a + 1, len(tickers)):
                rows.append((f"ewma_corr[{tickers[a]},{tickers[b]}]", corr[a, b], "off-diagonal"))
        rows.append(("portfolio_variance_rebuilt", rebuilt, "w' D R D w from the saved parts"))
    rows += [
        ("predicted_variance", day["predicted_variance"], "the model's portfolio variance for this day"),
        ("predicted_sigma", day["predicted_sigma"], "sqrt of it"), ("mu_hat", day["mu_hat"], ""),
    ]

    residuals = run.combined_residuals
    position = residuals.index.get_loc(date)
    n_used = int(day["n_residuals_used"])
    used = residuals.iloc[position - n_used:position]
    unit_row = residuals.iloc[position]     # the run's own unit: the VaR is rebuilt there and then converted like the reading
    rebuilt_var = core.loss_to_percent(-(unit_row["mu_hat"] + unit_row["predicted_sigma"] * day["quantile"]), run.unit)
    rows += [
        ("quantile_window_n", n_used, "standardized residuals strictly before this day"),
        ("quantile_window_first", used.index[0].date(), ""), ("quantile_window_last", used.index[-1].date(), ""),
        ("quantile_q", day["quantile"], f"{1 - reading.var_confidence:g}-quantile of that window"),
        ("varForecast", day["varForecast"], "-(mu_hat + sigma * q)"),
        ("varForecast_rebuilt", rebuilt_var, ""),
        ("loss", day["loss"], "-portfolio_return"), ("exceeded", bool(day["exceeded"]), "loss > varForecast"),
        ("excess", day["excess"], "loss - varForecast"), ("tick_loss", day["tick_loss"], ""),
    ]
    return pd.DataFrame(rows, columns=["item", "value", "note"]).set_index("item")


# --------------------------------------------------------------------------------------------------
# MLflow (an index over the folders; the folder stays the truth)
# --------------------------------------------------------------------------------------------------

def log_walk_forward_run_to_mlflow(
    run: Run, reading: Reading, experiment: str, tracking_uri: str = "sqlite:///mlruns.db",
) -> str:
    """
    One MLflow run named '<name> | train N f M': params = the flat params, metrics in two groups over all windows:
    total_window_var_metrics.* (everything about the VaR: tests, level, tick loss, tail, how much it moves) and
    total_window_volatility_metrics.* (the variance forecast: train / forecast QLIKE and the model's own loss, day-weighted),
    tags = keys / git / data hash / store_path (the run folder relative to the store root, e.g.
    WalkForwardSetting/<walk_forward_id>/FeatureAndTargets/...; the folder is not copied into MLflow),
    artifacts = model_settings.json, test_results.json and exceeding_days.json (the exceedance days as a table the
    UI shows: date, loss, VaR, excess, window_id).
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=Path("mlartifacts").resolve().as_uri())
    mlflow.set_experiment(experiment)

    metrics = run.metrics(reading)
    scalars = {
        f"{suite}.{key}": float(value)
        for suite, values in metrics.items() if suite != "reading" for key, value in values.items()
        if isinstance(value, (int, float, bool, np.number)) and not isinstance(value, str) and np.isfinite(float(value))
    }
    exceeding = run.exceeding_days(reading)[["loss", "varForecast", "excess", "window_id"]].reset_index()
    exceeding["Date"] = exceeding["Date"].dt.strftime("%Y-%m-%d")
    config = run.config
    with mlflow.start_run(run_name=run.display_name) as active:
        mlflow.log_params({k: str(v)[:500] for k, v in config["params"].items()})
        mlflow.log_params({"reading": reading.key, "run_name": run.name})
        # the variance forecast numbers (QLIKE and the model's own loss): total_window_volatility_metrics.<train|forecast>_<qlike|loss>
        volatility = {
            f"total_window_volatility_metrics.{k.removeprefix('total_window_')}": v
            for k, v in run.total_window_losses.items() if np.isfinite(v)
        }
        mlflow.log_metrics({**scalars, **volatility})
        mlflow.set_tags({
            **config["keys"], "run_id": run.run_id, "git_commit": config["code"]["git_commit"],
            "git_dirty": str(config["code"]["dirty"]), "data_sha256": config["data"]["sha256"],
            # relative to the store root (like a data-lake path): the root is configuration (STORE_ROOT), not run data
            "store_path": run.run_dir.relative_to(run.run_dir.parents[5]).as_posix(),
        })
        # the folder is NOT copied (store/ stays the single source); only the small readable files are artifacts
        mlflow.log_artifact(str(run.run_dir / "model_settings.json"))
        mlflow.log_artifact(str(run.run_dir / "readings_from_walk_forward" / reading.test_path / "test_results.json"))
        mlflow.log_table(exceeding, artifact_file="exceeding_days.json")
        return active.info.run_id
