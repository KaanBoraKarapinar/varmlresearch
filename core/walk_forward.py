"""
Walk-forward VaR: many (train window -> forecast window) steps, one model per step, one stitched VaR series.

Dataclasses only hold decisions or results, plain functions do the work:

    WalkForwardSetting.make_calendars()      the setting (data, universe, windows) -> list of calendars
    train_walk_forward(calendars, recipe)    calendars + recipe  -> list of WalkForwardSingleCalendarFit
    build_combined_residual_bank_WITHOUT_CALC(fits)   all windows -> ONE combined ResidualBank (no VaR)
    create_combined_residuals_bank_and_read_var_df(fits, var_confidence, window, ...)
                                             fits + any reading -> the VaR of every forecast day
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import core
from core.models import garch_models as gm
from core.models import tree_models as tm
from core import (
    DailyGroundTruthCalendar,
    EwmaVarianceBank,
    FeatureSpec,
    GroundTruthCalendar,
    ResidualBank,
    SquaredReturn,
    Target,
    TrainedModel,
    build_features,
    check_unit,
    qlike,
    return_to_percent,
)
from core import feature_blocks, targets
from core.ResidualBank import RESIDUAL_COLS
from core.feature_blocks import Returns
from core.units import return_to_percent

# the training days of the FIRST window only (per-ticker sigma / returns / EWMA covariance): the forecast days' quantile
# reads these standardized residuals until it has a full window of out-of-sample ones, so re-weighting needs them
FIRST_WINDOW_TRAINING_BANK_FILE = "WALK_FORWARD_FIRST_WINDOW_TRAINING_BANK_FOR_RECALCULATING_RESIDUALS.parquet"


# --------------------------------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class WalkForwardSetting:
    """
    Everything a calendar takes (the price data, the tickers, the weights, drop_market_closed, warmup_bars) plus the
    window decisions, so the setting knows the whole universe: one setting = one walk-forward (its table of windows).

    Windows tile forward from start_date and advance by forecast_bars, so the forecast windows tile the data with no
    gap and no overlap (that is what lets them be stitched into one series). Each forecast window becomes part of
    the train window of the next steps: rolling keeps train_bars bars, expanding keeps everything since the start.

    start_date:str the first TRAIN bar of window 0 (like the calendar's train_start; the warmup bars must exist before
        it). None: the first bar that has warmup_bars bars (and a return) before it.
    end_date: the last forecast bar (like the calendar's forecast_end). None: the last bar of the data. The last
        window is shorter than forecast_bars when the span is not a multiple of it, so end_date is honoured.
    data_path: only a label for walk_forward.json, not part of the identity (the data itself is hashed).
    """
    tickers_raw_price_data: pd.DataFrame = field(repr=False)
    tickers: list[str]
    weights: dict[str, float] | pd.Series
    train_bars: int
    # the return type every record is made of (percent | log, core/units.py): required, handed to every calendar
    unit: str = field(kw_only=True)
    forecast_bars: int = 21
    expanding: bool = False
    warmup_bars: int = 30
    start_date: str | pd.Timestamp | None = None
    end_date: str | pd.Timestamp | None = None
    drop_market_closed: bool = True
    data_path: str | None = None
    # CSV files of additional daily feature data (core.AdditionalFeatureSource): read once, aligned to the price bars, handed to every calendar
    additional_feature_sources: tuple = ()

    def __post_init__(self):
        if self.train_bars <= 0 or self.forecast_bars <= 0:
            raise ValueError("train_bars and forecast_bars must be positive.")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be non-negative.")
        check_unit(self.unit)
        object.__setattr__(self, "additional_feature_sources", tuple(self.additional_feature_sources))
        missing = set(self.tickers) - set(self.tickers_raw_price_data.columns)
        if missing:
            raise ValueError(f"tickers_raw_price_data is missing columns: {sorted(missing)}")
        object.__setattr__(self, "tickers", list(self.tickers))
        # the normalized weights, exactly what the calendar uses
        normalized = GroundTruthCalendar._validate_weights(self.tickers, self.weights)
        object.__setattr__(self, "weights", {t: float(w) for t, w in normalized.items()})
        if not self.expanding and self.train_bars % self.forecast_bars:
            warnings.warn(
                f"train_bars={self.train_bars} is not a multiple of forecast_bars={self.forecast_bars}: the window "
                f"boundaries of walk-forwards with other train lengths will not coincide with these.",
                stacklevel=2,
            )
        self.additional_feature_data   # read the CSV files now, so a bad file fails here and not in the middle of a training

    @cached_property
    def _index(self) -> pd.DatetimeIndex:
        return self.tickers_raw_price_data.sort_index().index

    def _positions(self) -> list[tuple[int, int, int]]:
        """(train_start_pos, train_end_pos, forecast_end_pos) of every window, by bar position on the price index."""
        index = self._index
        # the first bar has no return (pct_change), so a train bar needs warmup_bars + 1 bars before it
        first_train_pos = self.warmup_bars + 1
        if self.start_date is not None:
            start = pd.Timestamp(self.start_date)
            if start not in index:
                raise KeyError(f"start_date={start} is not a bar in the price data (range: {index[0]} .. {index[-1]})")
            first_train_pos = int(index.get_loc(start))
            if first_train_pos < self.warmup_bars + 1:
                raise ValueError(
                    f"start_date={start.date()} is bar {first_train_pos}: warmup_bars={self.warmup_bars} needs "
                    f"{self.warmup_bars + 1} bars before the first train bar."
                )
        last_pos = len(index) - 1
        if self.end_date is not None:
            end = pd.Timestamp(self.end_date)
            if end not in index:
                raise KeyError(f"end_date={end} is not a bar in the price data (range: {index[0]} .. {index[-1]})")
            last_pos = int(index.get_loc(end))

        positions = []
        train_end_pos = first_train_pos + self.train_bars - 1
        while train_end_pos < last_pos:   # at least one forecast bar after the train window
            train_start_pos = first_train_pos if self.expanding else train_end_pos - self.train_bars + 1
            positions.append((train_start_pos, train_end_pos, min(train_end_pos + self.forecast_bars, last_pos)))
            train_end_pos += self.forecast_bars

        if not positions:
            raise ValueError(
                f"No window fits: bars {first_train_pos}..{last_pos} of the price data, need train {self.train_bars} "
                f"+ at least 1 forecast bar."
            )
        return positions

    def window_table(self) -> pd.DataFrame:
        """One row per window: window_id, train_start, train_end, forecast_start, forecast_end, n_train, n_forecast."""
        index = self._index
        return pd.DataFrame([
            {
                "window_id": window_id,
                "train_start": index[train_start_pos], "train_end": index[train_end_pos],
                "forecast_start": index[train_end_pos + 1], "forecast_end": index[forecast_end_pos],
                "n_train": train_end_pos - train_start_pos + 1, "n_forecast": forecast_end_pos - train_end_pos,
            }
            for window_id, (train_start_pos, train_end_pos, forecast_end_pos) in enumerate(self._positions())
        ])

    def make_calendars(self) -> list[DailyGroundTruthCalendar]:
        """One DailyGroundTruthCalendar per window, in time order (list position = window id)."""
        index = self._index
        return [
            DailyGroundTruthCalendar(
                self.tickers_raw_price_data, self.tickers, self.weights,
                train_start=index[train_start_pos], train_end=index[train_end_pos],
                forecast_start=index[train_end_pos + 1], forecast_end=index[forecast_end_pos],
                unit=self.unit, warmup_bars=self.warmup_bars, drop_market_closed=self.drop_market_closed,
                additional_feature_data=self.additional_feature_data,
            )
            for train_start_pos, train_end_pos, forecast_end_pos in self._positions()
        ]

    @cached_property
    def additional_feature_data(self) -> pd.DataFrame | None:
        """The additional feature CSVs as one table on the price bars (columns (tag, scope)); None without sources."""
        if not self.additional_feature_sources:
            return None
        # the first bar any calendar reads: the warmup of the first window; a series may start later than the price data
        first_read_pos = self._positions()[0][0] - self.warmup_bars
        return core.load_additional_feature_data(
            self.additional_feature_sources, self._index, self.tickers, needed_from=self._index[first_read_pos],
        )

    @cached_property
    def additional_feature_identity(self) -> dict | None:
        """What identifies the additional data: the declarations (not the paths) and a hash of the values; None without sources."""
        if not self.additional_feature_sources:
            return None
        values = pd.util.hash_pandas_object(self.additional_feature_data, index=True).to_numpy().tobytes()
        return {
            "sha256": hashlib.sha256(values).hexdigest(),
            "sources": [
                {"tag": s.tag, "scope": s.scope, "column": s.column, "available_after_bars": s.available_after_bars}
                for s in self.additional_feature_sources
            ],
        }

    @cached_property
    def data_sha256(self) -> str:
        """sha256 of the index and the price values of the used tickers (the data identity)."""
        frame = self.tickers_raw_price_data[self.tickers].sort_index()
        return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()

    @cached_property
    def walk_forward_id(self) -> str:
        """
        The identity of this walk-forward, readable: blocks joined by "_", parts inside a block by "-":
        from<START>_to<END>_train<N>bars_forecast<M>bars_<TICKER><weight>-..._<hash8>
        The readable blocks show what people look at; <hash8> hashes EVERYTHING that defines the windows and the data
        (tickers, weights, data, dates, bar counts, expanding, warmup_bars, drop_market_closed), so it also tells
        apart what the readable blocks do not show. Beyond 6 tickers the ticker block is <n>tickers.
        """
        positions = self._positions()
        index = self._index
        start = index[positions[0][0]].date().isoformat()
        end = index[positions[-1][2]].date().isoformat()
        identity = {
            "tickers": self.tickers, "weights": {t: round(w, 12) for t, w in self.weights.items()},
            "data_sha256": self.data_sha256, "start": start, "end": end, "train_bars": self.train_bars,
            "forecast_bars": self.forecast_bars, "expanding": self.expanding, "warmup_bars": self.warmup_bars,
            "drop_market_closed": self.drop_market_closed, "unit": self.unit,
        }
        if self.additional_feature_sources:     # only then, so the ids of walk-forwards without it do not change
            identity["additional_feature_data"] = self.additional_feature_identity
        hash8 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
        tickers_block = (
            "-".join(f"{t}{self.weights[t]:.4g}" for t in self.tickers) if len(self.tickers) <= 6
            else f"{len(self.tickers)}tickers"
        )
        train_block = f"train{self.train_bars}bars" + ("-expanding" if self.expanding else "")
        return f"from{start}_to{end}_{train_block}_forecast{self.forecast_bars}bars_{tickers_block}_{hash8}"


@dataclass(frozen=True)
class ModelRecipe:
    """
    How every window's model is trained. The two bank groups are baked into the stored residuals at
    training time, so they live here and not in the VaR reading (create_combined_residuals_bank_and_read_var_df).

    model_family: 'xgboost' (features + target below), 'garch' (arch package, reads only the returns),
    'ewma' (no model at all: the pure RiskMetrics EWMA variance, the "do nothing" baseline) or
    'historical' (plain historical VaR: the quantile of the last window_size portfolio returns, no variance forecast;
    the window comes from the VaR reading). For 'ewma' and 'historical' model_format, features, target and
    model_kwargs are ignored.
    """
    model_format: Literal["long", "wide"]
    model_kwargs: dict = field(default_factory=dict)  # passed to xgboost, or to arch.arch_model for the garch family
    model_family: Literal["xgboost", "garch", "ewma", "historical"] = "xgboost"
    # what the model sees and what it is trained to predict (core/feature_blocks.py / core/targets.py)
    # (the xgboost family needs both, each declares its unit; garch / ewma read only the returns and take neither)
    features: FeatureSpec | None = None
    target: Target | None = None
    # ResidualBank
    mu_mode: Literal["rolling", "constant"] = "constant"
    default_mu_hat: float = 0.0
    rolling_mu_hat_window_size: int | None = None
    # EwmaVarianceBank (long only)
    ewma_lam: float = 0.94
    ewma_min_periods: int = 30
    # the run_id of an already trained run whose models are used as they are (no fit): this recipe then only scores them on
    # the calendars of the walk-forward it is trained on. features / target must be the source's, see _check_pretrained.
    pretrained_from: str | None = None

    def __post_init__(self):
        if self.model_format not in ("long", "wide"):
            raise ValueError(f"model_format must be 'long' or 'wide', got {self.model_format!r}")
        if self.model_family not in ("xgboost", "garch", "ewma", "historical"):
            raise ValueError(f"model_family must be 'xgboost', 'garch', 'ewma' or 'historical', got {self.model_family!r}")
        if self.model_family == "xgboost" and (self.features is None or self.target is None):
            raise ValueError("an xgboost recipe needs both features and target (each declares its unit).")
        if self.pretrained_from is not None and (self.model_family != "xgboost" or self.model_format != "long"):
            raise ValueError("pretrained_from works for a long xgboost recipe only.")


# the baselines every comparison uses, computed for the portfolio itself (wide): GARCH(1,1), the EWMA and the plain historical VaR
BASELINE_RECIPES = {
    "garch_wide": ModelRecipe("wide", model_family="garch", model_kwargs=dict(mean="Zero", vol="GARCH", p=1, q=1, dist="normal")),
    "ewma": ModelRecipe("wide", model_family="ewma"),
    "historical": ModelRecipe("wide", model_family="historical"),
}


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardSingleCalendarFit:
    """
    What is left after training one window. residual_bank holds the model's in-sample residuals for the train
    dates followed by its out-of-sample residuals for the forecast dates.
    """
    window_id: int
    calendar: DailyGroundTruthCalendar
    trained_model: TrainedModel | None  # None for the 'ewma' and 'historical' families (no model)
    residual_bank: ResidualBank
    ewma_bank: EwmaVarianceBank | None = None  # long format only
    forecast_feature: object | None = None  # the forecast-side features (None for the 'ewma' family)


def _check_pretrained(calendars: list[DailyGroundTruthCalendar], recipe: ModelRecipe, pretrained: "LoadedRun") -> None:
    """
    A trained run can be applied to the calendars of another walk-forward only when nothing the model saw changes silently:
    the same windows (every model belongs to its own train / forecast dates), the same features and target, the same unit and,
    when a feature depends on the portfolio (FeatureSpec.portfolio_dependent_blocks), the same tickers and weights.
    """
    if pretrained.run_id != recipe.pretrained_from:
        raise ValueError(f"the recipe asks for {recipe.pretrained_from!r} but the run given is {pretrained.run_id!r}.")
    source = pretrained.recipe
    if source.model_family != "xgboost" or source.model_format != "long":
        raise ValueError(f"{pretrained.display_name} is not a long xgboost run: only those can be applied to another calendar.")
    for name in ("features", "target"):
        if config_to_dict(getattr(recipe, name)) != config_to_dict(getattr(source, name)):
            raise ValueError(f"the {name} of the recipe are not the {name} the model was trained with ({pretrained.display_name}).")
    if (recipe.mu_mode, recipe.default_mu_hat) != (source.mu_mode, source.default_mu_hat):
        raise ValueError("the mu_hat settings of the recipe are not the ones of the trained run.")
    windows = pretrained.windows
    if len(windows) != len(calendars):
        raise ValueError(f"the trained run has {len(windows)} windows, these calendars {len(calendars)}: the models do not belong to these windows.")
    for window_id, (calendar, (_, row)) in enumerate(zip(calendars, windows.iterrows())):
        dates = (calendar.train_start, calendar.train_end, calendar.forecast_start, calendar.forecast_end)
        saved = tuple(pd.Timestamp(row[c]) for c in ("train_start", "train_end", "forecast_start", "forecast_end"))
        if dates != saved:
            raise ValueError(f"window {window_id} is {dates} here and {saved} in the trained run: a model only fits its own window.")
        if calendar.unit != pretrained.unit:
            raise ValueError(f"the calendar is in unit {calendar.unit!r}, the model was trained in {pretrained.unit!r}.")
    blocks = recipe.features.portfolio_dependent_blocks
    if blocks:
        setting = pretrained.config["setting"]
        trained = {t: float(setting["weights"][t]) for t in setting["tickers"]}
        here = {t: float(w) for t, w in calendars[0].weights.items()}
        if set(trained) != set(here) or any(abs(trained[t] - here[t]) > 1e-9 for t in trained):
            raise ValueError(
                f"the features of this model depend on the portfolio ({', '.join(blocks)}), which is different here "
                f"(trained: {len(trained)} tickers, here: {len(here)} tickers or other weights). Train the model with "
                f"FeatureSpec.portable() to apply it to another portfolio."
            )


def train_walk_forward(
    calendars: list[DailyGroundTruthCalendar], recipe: ModelRecipe, pretrained: "LoadedRun | None" = None,
) -> list[WalkForwardSingleCalendarFit]:
    """
    One model per calendar (fresh single-use banks every time). No VaR and no residual sharing here.
    pretrained (a LoadedRun, recipe.pretrained_from): no model is fitted, window k takes the model of window k of that run
    and only the banks of these calendars are filled with it (_check_pretrained says what has to be the same).
    """
    if (recipe.pretrained_from is None) != (pretrained is None):
        raise ValueError("recipe.pretrained_from and the pretrained run go together: give both or neither.")
    if pretrained is not None:
        _check_pretrained(calendars, recipe, pretrained)
        trained_columns = pretrained.feature_columns()
    return [
        fit_window(window_id, calendar, recipe, None if pretrained is None else pretrained.model(window_id),
                   None if pretrained is None else trained_columns)
        for window_id, calendar in enumerate(calendars)
    ]


def fit_window(
    window_id: int, calendar: DailyGroundTruthCalendar, recipe: ModelRecipe,
    existing_model=None, trained_columns: list[str] | None = None,
) -> WalkForwardSingleCalendarFit:
    """
    One window of train_walk_forward. existing_model (with the columns it was trained on): no model is fitted, the banks of
    this calendar are filled with that model (xgboost only).
    """
    if recipe.model_family == "ewma":
        return _ewma_fit(window_id, calendar, recipe)
    if recipe.model_family == "historical":
        return _historical_fit(window_id, calendar, recipe)
    if recipe.model_family == "xgboost":
        features, target = recipe.features, recipe.target
    else:   # garch reads only the returns: the minimal spec / target in the calendar's own unit
        features = FeatureSpec((Returns(),), unit=calendar.unit)
        target = SquaredReturn(1.0, unit=calendar.unit)
    training_feature, forecast_feature = build_features(calendar, recipe.model_format, features, target)
    residual_bank = ResidualBank(
        default_mu_hat=recipe.default_mu_hat, mu_mode=recipe.mu_mode,
        rolling_mu_hat_window_size=recipe.rolling_mu_hat_window_size,
    )
    train = tm.xgboost_train if recipe.model_family == "xgboost" else gm.garch_train
    existing = {}
    if existing_model is not None:
        if list(training_feature.feature_cols) != trained_columns:
            raise ValueError(
                f"window {window_id}: the features of this calendar are {list(training_feature.feature_cols)}, "
                f"the model was trained on {trained_columns}."
            )
        existing = {"existing_model": existing_model}
    if recipe.model_format == "long":
        model = train(
            calendar, training_feature, residual_bank,
            ewma_bank=EwmaVarianceBank(calendar, ewma_lam=recipe.ewma_lam, ewma_min_periods=recipe.ewma_min_periods),
            **existing, **recipe.model_kwargs,
        )
        bank, ewma_bank = model.calculate_multi_day_standart_residuals_from_long_format_model(calendar, forecast_feature)
    else:
        model = train(calendar, training_feature, residual_bank, **recipe.model_kwargs)
        bank = model.calculate_multi_day_standart_residuals_from_wide_format_model(calendar, forecast_feature)
        ewma_bank = None
    return WalkForwardSingleCalendarFit(window_id, calendar, model, bank, ewma_bank, forecast_feature)


def _ewma_fit(window_id: int, calendar: DailyGroundTruthCalendar, recipe: ModelRecipe) -> WalkForwardSingleCalendarFit:
    """
    The no-model baseline: the portfolio variance is the pure EWMA w' Sigma w of the realized returns (each
    date's value is built from strictly earlier returns), standardized and read exactly like a model's.
    Nothing is fitted, so train and forecast residuals are equally honest.
    """
    ewma_bank = calendar.ewma_realized_variance_bank(recipe.ewma_lam, recipe.ewma_min_periods)
    variance = ewma_bank.portfolio_variance
    dates = variance.index[variance.index.isin(calendar.train_data.index.append(calendar.forecast_dates))]
    returns = calendar.DANGEROUS_merged_warmedup_FORECAST_data["portfolio_return"].loc[dates]

    residual_bank = ResidualBank(
        default_mu_hat=recipe.default_mu_hat, mu_mode=recipe.mu_mode,
        rolling_mu_hat_window_size=recipe.rolling_mu_hat_window_size,
    )
    residual_bank.start_training(dates[0], calendar.frequency)
    residual_bank.training_safe_bulk_insert(dates, returns.to_numpy(), variance.loc[dates].to_numpy())
    return WalkForwardSingleCalendarFit(window_id, calendar, None, residual_bank, ewma_bank)


def _historical_fit(window_id: int, calendar: DailyGroundTruthCalendar, recipe: ModelRecipe) -> WalkForwardSingleCalendarFit:
    """
    Plain historical VaR as a run: every day's variance is set to 1, so the standardized residual IS the portfolio return
    and the filtered historical simulation (VaR = -(mu + sigma * q), q the quantile of the last window_size residuals)
    is the quantile of the last window_size returns, i.e. historical simulation. The window is the VaR reading's.
    There is no variance forecast: the predicted_variance / predicted_sigma columns of this run are the constant 1.
    """
    dates = calendar.train_data.index.append(calendar.forecast_dates)
    returns = calendar.DANGEROUS_merged_warmedup_FORECAST_data["portfolio_return"].loc[dates]
    residual_bank = ResidualBank(
        default_mu_hat=recipe.default_mu_hat, mu_mode=recipe.mu_mode,
        rolling_mu_hat_window_size=recipe.rolling_mu_hat_window_size,
    )
    residual_bank.start_training(dates[0], calendar.frequency)
    residual_bank.training_safe_bulk_insert(dates, returns.to_numpy(), np.ones(len(dates)))
    return WalkForwardSingleCalendarFit(window_id, calendar, None, residual_bank, None)


# --------------------------------------------------------------------------------------------------
# one residual bank for the whole walk-forward, read with any VaR reading
# --------------------------------------------------------------------------------------------------

def build_combined_residual_bank_WITHOUT_CALC(fits: list[WalkForwardSingleCalendarFit]) -> ResidualBank:
    """
    ONE ResidualBank for the whole walk-forward, in time order: the first window's bank as it is (its in-sample
    train residuals, then its out-of-sample forecast ones, the usual single-calendar start so the quantile
    window is full from the first day), followed by only the out-of-sample (forecast-date) rows of every later
    window. Once the window has moved past the first train, every residual it reads is out-of-sample.

    Each row keeps the predicted variance / sigma / mu_hat of the model that forecast that day. The bank is for
    reading (create_combined_residuals_bank_and_read_var_df, graphs), not to train or extend. Checks that the windows tile: each window's train ends
    on the previous window's last forecast day.
    """
    if not fits:
        raise ValueError("fits is empty.")
    for previous, current in zip(fits[:-1], fits[1:]):
        if current.calendar.train_end != previous.calendar.forecast_dates[-1]:
            raise ValueError(
                f"windows {previous.window_id} and {current.window_id} do not tile: window {current.window_id} train "
                f"ends {current.calendar.train_end.date()}, window {previous.window_id} forecast ends "
                f"{previous.calendar.forecast_dates[-1].date()}."
            )

    bank = fits[0].residual_bank.copy()
    bank.residuals = pd.concat(
        [fits[0].residual_bank.residuals]
        + [fit.residual_bank.residuals.loc[fit.calendar.forecast_dates] for fit in fits[1:]]
    )
    if not (bank.residuals.index.is_monotonic_increasing and bank.residuals.index.is_unique):
        raise ValueError("The combined residual bank is not chronological without duplicates.")
    return bank


def ewma_bank_to_frame(ewma_bank: EwmaVarianceBank, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    One EwmaVarianceBank as one frame indexed by Date: portfolio_variance, sigma_<ticker> (the model's per-ticker
    variance), return_<ticker>, and cov_<i>_<j> (the EWMA covariance of that day, every pair).
    dates: only these days of the bank (default: all of them). Building the frame costs a matrix per day, so a caller that
    needs the forecast days of a window should not build the training days too.
    """
    dates = ewma_bank.portfolio_variance.index if dates is None else pd.DatetimeIndex(dates)
    missing = dates.difference(pd.DatetimeIndex(list(ewma_bank.cov_by_date)))
    if len(missing):
        raise ValueError(f"The ewma bank has no covariance for dates: {list(missing)}")
    tickers = list(ewma_bank.tickers)
    matrices = [ewma_bank.cov_by_date[date] for date in dates]
    if not all(list(m.index) == tickers and list(m.columns) == tickers for m in matrices):    # the bank builds them in this order: a check, not a reorder
        raise ValueError("A covariance matrix of the ewma bank is not in the order of its tickers.")
    cov = np.stack([m.to_numpy(dtype=float) for m in matrices])
    columns = {"portfolio_variance": ewma_bank.portfolio_variance.astype(float).loc[dates].to_numpy()}
    for ticker in tickers:
        columns[f"sigma_{ticker}"] = ewma_bank.sigma_diag[ticker].astype(float).loc[dates].to_numpy()
    for ticker in tickers:
        columns[f"return_{ticker}"] = ewma_bank.ticker_returns[ticker].astype(float).loc[dates].to_numpy()
    for a, i in enumerate(tickers):
        for b, j in enumerate(tickers):
            columns[f"cov_{i}_{j}"] = cov[:, a, b]
    return pd.DataFrame(columns, index=dates).rename_axis("Date")       # one frame built at once: hundreds of columns inserted one by one are slow


def build_combined_ewma_bank_WITHOUT_CALC(fits: list[WalkForwardSingleCalendarFit]) -> pd.DataFrame:
    """
    The EWMA banks of the whole walk-forward as ONE frame: only the forecast-date rows of every window, cut out of
    that window's bank (ewma_bank_to_frame) and pasted in time order, plus window_id. Unlike the combined residual
    bank it does not keep the first window's train rows: it is the per-ticker variances / covariances behind every
    forecast day's portfolio variance, which is what explaining a VaR needs.
    """
    if not fits or fits[0].ewma_bank is None:
        raise ValueError("The fits have no ewma banks (only long-format and ewma-family fits do).")
    parts = []
    for fit in fits:
        missing = fit.calendar.forecast_dates.difference(fit.ewma_bank.portfolio_variance.index)
        if len(missing):
            raise ValueError(f"window {fit.window_id}: the ewma bank misses forecast dates {list(missing)}")
        forecast = ewma_bank_to_frame(fit.ewma_bank, fit.calendar.forecast_dates)
        forecast["window_id"] = fit.window_id
        parts.append(forecast)
    combined = pd.concat(parts)
    if not (combined.index.is_monotonic_increasing and combined.index.is_unique):
        raise ValueError("The combined ewma bank is not chronological without duplicates.")
    return combined


def build_first_window_training_bank(fits: list[WalkForwardSingleCalendarFit]) -> pd.DataFrame:
    """
    The training days of the FIRST window's ewma bank as one frame (same columns as the combined ewma bank, window_id 0):
    the in-sample per-ticker variances, returns and EWMA covariances. Together with the combined ewma bank (forecast
    days of every window) it holds every day the combined residual bank has, so the residuals can be recomputed for other weights.
    """
    if not fits or fits[0].ewma_bank is None:
        raise ValueError("The fits have no ewma banks (only long-format and ewma-family fits do).")
    first = fits[0]
    train_dates = first.ewma_bank.portfolio_variance.index.difference(first.calendar.forecast_dates)
    if not len(train_dates):
        raise ValueError("The first window's ewma bank has no training days.")
    frame = ewma_bank_to_frame(first.ewma_bank, train_dates)
    frame["window_id"] = first.window_id
    return frame


def reweighted_residuals(
    training_bank: pd.DataFrame,
    forecast_bank: pd.DataFrame,
    tickers: list[str],
    weights: np.ndarray,
    unit: str,
    mu_mode: str,
    default_mu_hat: float,
    rolling_mu_hat_window_size: int | None,
) -> pd.DataFrame:
    """
    The combined residual bank (RESIDUAL_COLS, the same rows as build_combined_residual_bank_WITHOUT_CALC) for OTHER weights,
    from the stored per-ticker rows: the first window's training days (first_window_training_bank) followed by the forecast
    days of every window (ewma_bank). Nothing is trained or forecast again:
      Sigma_t = D R D (D = the model's per-ticker sigma, R = the correlation of the EWMA covariance), variance = w' Sigma_t w,
      actual_return = the portfolio return of the unit (percent: w . r_t; log: log(1 + w . (exp(r_t) - 1)), as the calendar forms it),
      mu_hat as the ResidualBank forms it (constant, or the mean of the previous window bars).
    weights: in the order of tickers, already normalised.
    """
    check_unit(unit)
    frame = pd.concat([training_bank, forecast_bank])
    if not (frame.index.is_monotonic_increasing and frame.index.is_unique):
        raise ValueError("The training bank and the ewma bank do not join into one chronological series without duplicates.")
    weights = np.asarray(weights, dtype=float)
    n = len(tickers)
    if weights.shape != (n,):
        raise ValueError(f"weights has shape {weights.shape}, expected ({n},) in the order of tickers.")

    d = frame[[f"sigma_{t}" for t in tickers]].to_numpy(dtype=float)
    cov = np.stack([frame[[f"cov_{i}_{j}" for j in tickers]].to_numpy(dtype=float) for i in tickers], axis=1)    # (T, n, n)
    std = np.sqrt(np.einsum("tii->ti", cov))
    corr = cov / (std[:, :, None] * std[:, None, :])
    corr[:, np.arange(n), np.arange(n)] = 1.0
    sigma = corr * np.sqrt(d[:, :, None] * d[:, None, :])
    sigma[:, np.arange(n), np.arange(n)] = d       # the same construction as EwmaVarianceBank._portfolio_variance_from_cov
    variance = np.einsum("i,tij,j->t", weights, sigma, weights)
    returns = frame[[f"return_{t}" for t in tickers]].to_numpy(dtype=float)
    actual = returns @ weights if unit == "percent" else np.log1p(np.expm1(returns) @ weights)

    if mu_mode == "constant":
        mu = np.full(len(frame), float(default_mu_hat))
    elif mu_mode == "rolling":
        previous = pd.Series(actual).rolling(rolling_mu_hat_window_size, min_periods=1).mean().shift(1)
        mu = previous.fillna(float(default_mu_hat)).to_numpy()      # the first bar has no history: default_mu_hat, as the bank starts
    else:
        raise ValueError(f"mu_mode must be 'rolling' or 'constant', got {mu_mode!r}")

    root = np.sqrt(variance)
    out = pd.DataFrame({
        "actual_return": actual, "predicted_variance": variance, "predicted_sigma": root,
        "mu_hat": mu, "standardized_residual": (actual - mu) / root,
    }, index=frame.index).rename_axis("Date")
    return out


def build_combined_residuals_frame(fits: list[WalkForwardSingleCalendarFit]) -> pd.DataFrame:
    """
    The combined residual bank as one frame of floats (what a saved run keeps as WALK_FORWARD_FORECAST_RESIDUAL_BANK.parquet): RESIDUAL_COLS, window_id
    (the window that forecast the day; the first window's training days: 0) and part ('forecast' | 'train').
    """
    frame = build_combined_residual_bank_WITHOUT_CALC(fits).residuals[RESIDUAL_COLS].astype(float)
    forecast_window = pd.concat([pd.Series(fit.window_id, index=fit.calendar.forecast_dates) for fit in fits])
    frame["window_id"] = forecast_window.reindex(frame.index).fillna(0).astype(int)
    frame["part"] = np.where(frame.index.isin(forecast_window.index), "forecast", "train")
    return frame


def create_combined_residuals_bank_and_read_var_df(
    fits: list[WalkForwardSingleCalendarFit],
    var_confidence: float,
    multi_day_var_calculation_window_size: int | None,
    multi_day_var_min_periods: int | None = None,
) -> pd.DataFrame:
    """
    The VaR of every forecast date of the walk-forward, read off the combined bank (build_combined_residual_bank_WITHOUT_CALC(fits), built
    here), like calendar.var_from_residual_bank_forecast_period does for one calendar: how to read is chosen here,
    per call (var_confidence, the quantile window with None = expanding, its min_periods, see
    ResidualBank.multi_day_var), so the same fits can be read again any other way. Columns: varForecast,
    n_residuals_used, loss, window_id (the window whose model forecast that day).

    It feeds run_full_backtest_suite_single_window / run_level_and_volatility_suite_single_window directly (they
    read varForecast and loss).
    """
    bank = build_combined_residual_bank_WITHOUT_CALC(fits)
    forecast_dates = fits[0].calendar.forecast_dates
    for fit in fits[1:]:
        forecast_dates = forecast_dates.append(fit.calendar.forecast_dates)

    var_df = bank.multi_day_var(
        forecast_dates, var_confidence, multi_day_var_calculation_window_size, multi_day_var_min_periods
    )
    var_df["loss"] = pd.concat([fit.calendar.forecast_data["portfolio_loss"] for fit in fits]).loc[var_df.index]
    var_df["window_id"] = pd.concat(
        [pd.Series(fit.window_id, index=fit.calendar.forecast_dates) for fit in fits]
    ).loc[var_df.index]
    # the bank's frame starts as object columns, so the VaR columns can come back as object: make them numbers
    var_df = var_df.astype({"varForecast": float, "loss": float, "n_residuals_used": int, "window_id": int})
    if var_df[["varForecast", "loss"]].isna().any().any():
        raise ValueError("The var_df has NaN in varForecast / loss.")
    return var_df


def _target_unit_mse(model: tm.TrainedTreeModel, X: pd.DataFrame, y: pd.Series, dates: pd.Index) -> tuple[float, int]:
    """(mean squared error in the model's own target units on the rows with a target, how many distinct days those rows cover)."""
    keep = y.notna().to_numpy()
    if not keep.any():
        return float("nan"), 0
    error = model.underlying_model.predict(X.to_numpy(dtype="float32")[keep]) - y.to_numpy(dtype=float)[keep]
    return float(np.mean(error ** 2)), int(pd.Index(dates)[keep].nunique())


def window_losses(fits: list[WalkForwardSingleCalendarFit]) -> pd.DataFrame:
    """
    How well every window's model did on its own TRAIN days (in-sample) and on its FORECAST days, one row per window:
      train_qlike / forecast_qlike   QLIKE of the portfolio variance forecasts against r^2 (core.qlike), r on percent returns:
                                     independent of the scale, so comparable between ANY models; days = train_days / forecast_days
      train_loss / forecast_loss     the model's squared error in its own target units (XGBoost only, NaN otherwise),
                                     over the rows it was fitted / forecast on (one row per day and ticker for a long model);
                                     only comparable between models with the same target and scale; days = *_loss_days
    The in-sample rows are the training days of the window's bank (the same rolling windows overlap: a day is in many windows).
    """
    rows = {}
    for fit in fits:
        residuals = fit.residual_bank.residuals[["actual_return", "predicted_variance"]].astype(float)
        # the realized return as a percent return, so a log run and a percent run are scored against the same r^2
        residuals["actual_return"] = return_to_percent(residuals["actual_return"], fit.calendar.unit)
        is_forecast = residuals.index.isin(fit.calendar.forecast_dates)
        row = {}
        has_variance_forecast = fit.trained_model is not None or fit.ewma_bank is not None     # the historical family has none
        for part, mask in (("train", ~is_forecast), ("forecast", is_forecast)):
            part_rows = residuals[mask]
            row[f"{part}_days"] = len(part_rows)
            row[f"{part}_qlike"] = (
                qlike(part_rows["predicted_variance"], part_rows["actual_return"] ** 2) if has_variance_forecast else np.nan
            )
        row.update(train_loss=np.nan, train_loss_days=0, forecast_loss=np.nan, forecast_loss_days=0)
        if isinstance(fit.trained_model, tm.TrainedTreeModel):
            tf, ff = fit.trained_model.training_feature, fit.forecast_feature
            train_dates = tf.X_in_sample.index if tf.X_in_sample_id_cols is None else tf.X_in_sample_id_cols["Date"]
            forecast_dates = ff.X_forecast_sample.index if ff.id_cols is None else ff.id_cols["Date"]
            row["train_loss"], row["train_loss_days"] = _target_unit_mse(
                fit.trained_model, tf.X_in_sample, tf.y_in_sample["target"], train_dates)
            row["forecast_loss"], row["forecast_loss_days"] = _target_unit_mse(
                fit.trained_model, ff.X_forecast_sample, ff.y_forecast_sample["target"], forecast_dates)
        rows[fit.window_id] = row
    return pd.DataFrame(rows).T.rename_axis("window_id")


def total_window_losses(table: pd.DataFrame) -> dict[str, float]:
    """
    The window losses as four numbers: every window's value averaged with a weight of the number of days it covers,
    so a short last window counts for what it is. (Averaging the windows' means weighted by their days is the same as
    pooling their days, as long as the days are the rows each mean was taken over.)
    """
    def weighted(value: str, days: str) -> float:
        valid = table[value].notna() & (table[days] > 0)
        return float(np.average(table.loc[valid, value].astype(float), weights=table.loc[valid, days].astype(float))) if valid.any() else float("nan")

    return {
        "total_window_train_qlike": weighted("train_qlike", "train_days"),
        "total_window_forecast_qlike": weighted("forecast_qlike", "forecast_days"),
        "total_window_train_loss": weighted("train_loss", "train_loss_days"),
        "total_window_forecast_loss": weighted("forecast_loss", "forecast_loss_days"),
    }


def var_df_windows(var_df: pd.DataFrame) -> list[dict]:
    """The stitched var_df cut back into its windows, the format run_full_backtest_suite_all_windows reads."""
    return [
        {"window_id": window_id, "window_start": part.index[0], "window_end": part.index[-1], "forecast_df": part}
        for window_id, part in var_df.groupby("window_id", sort=True)
    ]

# --------------------------------------------------------------------------------------------------
# a run read back from disk (the files are written by run_store)
# --------------------------------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime.date)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _calendar_name(window_id: int) -> str:
    return f"calendar{window_id:03d}"



# --------------------------------------------------------------------------------------------------
# config round trip (no pickle): dataclass <-> dict with a class registry
# --------------------------------------------------------------------------------------------------

def _build_registry() -> dict[str, type]:
    registry = {"walk_forward.ModelRecipe": ModelRecipe, "feature_blocks.FeatureSpec": feature_blocks.FeatureSpec}
    for module in (feature_blocks, targets):
        for value in vars(module).values():
            if isinstance(value, type) and dataclasses.is_dataclass(value) and value.__module__ == module.__name__:
                registry[f"{module.__name__.split('.')[-1]}.{value.__name__}"] = value
    return registry


_REGISTRY = _build_registry()


def _class_key(obj) -> str:
    return f"{type(obj).__module__.split('.')[-1]}.{type(obj).__name__}"


def config_to_dict(obj):
    """A dataclass tree (ModelRecipe / FeatureSpec / blocks / Target) -> JSON-able dict, tuples are tagged."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        key = _class_key(obj)
        if key not in _REGISTRY:
            raise TypeError(f"{key} is not in the config registry.")
        return {"__class__": key, **{f.name: config_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, tuple):
        return {"__tuple__": [config_to_dict(x) for x in obj]}
    if isinstance(obj, list):
        return [config_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): config_to_dict(v) for k, v in obj.items()}
    return obj


def config_from_dict(data):
    """Inverse of config_to_dict."""
    if isinstance(data, dict):
        if "__class__" in data:
            cls = _REGISTRY[data["__class__"]]
            return cls(**{k: config_from_dict(v) for k, v in data.items() if k != "__class__"})
        if "__tuple__" in data:
            return tuple(config_from_dict(x) for x in data["__tuple__"])
        return {k: config_from_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [config_from_dict(x) for x in data]
    return data


def recipe_from_config(config: dict) -> ModelRecipe:
    """The ModelRecipe stored in a run's model_settings.json."""
    return config_from_dict(config["recipe"])


def _to_parquet(frame: pd.DataFrame, path: Path, metadata: dict) -> None:
    """Every parquet of a run carries the run's identity, unit, feature / target / loss definitions in its file metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame)
    merged = {**(table.schema.metadata or {}), **{k.encode(): str(v).encode() for k, v in metadata.items()}}
    pq.write_table(table.replace_schema_metadata(merged), path)


def _parquet_metadata(config: dict) -> dict:
    """What goes into the parquet file metadata, from a run's model_settings.json."""
    params = config["params"]
    keys = ("unit", "model_family", "model_format", "feature_blocks", "feature_unit", "target", "target_scale", "target_unit", "training_loss")
    return {
        "run_id": config["run_id"], "name": config["name"], "walk_forward_id": config["walk_forward_id"],
        **{key: params[key] for key in keys if key in params},
    }


@dataclass(frozen=True)
class Reading:
    """How the VaR is read off the combined residuals (var_confidence, quantile window, min_periods) + the test level."""
    var_confidence: float = 0.95
    window_size: int | None = 252      # None = expanding
    min_periods: int | None = 30
    test_confidence: float = 0.95

    @property
    def var_path(self) -> Path:
        """readings_from_walk_forward/<this>: c0.95/window_size_252/minlength_30"""
        window = "expanding" if self.window_size is None else self.window_size
        min_periods = "auto" if self.min_periods is None else self.min_periods
        return Path(f"c{self.var_confidence:g}") / f"window_size_{window}" / f"minlength_{min_periods}"

    @property
    def test_path(self) -> Path:
        return self.var_path / f"testconfidence_{self.test_confidence:g}"

    @property
    def key(self) -> str:
        return self.test_path.as_posix()


@dataclass(frozen=True)
class ReadingResult:
    """What one Reading of a run gives, all in percent returns: the VaR frame, the day-by-day frame, its exceedance days, the per-window blocks and the test numbers."""
    var_df: pd.DataFrame
    daily: pd.DataFrame
    exceeding_days: pd.DataFrame
    blocks: pd.DataFrame
    metrics: dict


class Run(ABC):
    """
    A walk-forward run as the dashboard and the comparisons see it: one recipe on one walk-forward. A subclass provides the SOURCES
    (the config, the windows, the combined residuals, the ewma banks, the window losses); everything derived from them (the readings,
    the bank for the plots, the totals) is written once, here.

      LoadedRun               the sources are the files of a saved run folder (run_store.load_walk_forward_run)
      TrainedInitialRun       the sources are the fits of a training that has not been saved
      ReweightedVirtualRun    another run's per-ticker forecasts read for other portfolio weights: nothing is trained, nothing is saved
    """

    # ---- the sources ----------------------------------------------------------------------------

    @property
    @abstractmethod
    def config(self) -> dict:
        """model_settings.json: run_id, name, display_name, walk_forward_id, setting, params, frequency, ..."""

    @property
    @abstractmethod
    def windows(self) -> pd.DataFrame:
        """The windows table of the walk-forward (train / forecast dates of every window)."""

    @property
    @abstractmethod
    def combined_residuals(self) -> pd.DataFrame:
        """RESIDUAL_COLS + window_id + part ('train' for the first window's training days, 'forecast'), in time order."""

    @property
    @abstractmethod
    def window_losses(self) -> pd.DataFrame:
        """Per window: QLIKE and the model's own loss on its train and forecast days (window_losses)."""

    @property
    @abstractmethod
    def has_ewma_bank(self) -> bool:
        """True for the long and ewma families: they keep the per-ticker variances, returns and EWMA covariances."""

    @property
    @abstractmethod
    def ewma_bank(self) -> pd.DataFrame:
        """The forecast rows of every window's ewma bank (long and ewma family); FileNotFoundError when has_ewma_bank is False."""

    @property
    @abstractmethod
    def has_first_window_training_bank(self) -> bool:
        """True when first_window_training_bank can be read."""

    @property
    @abstractmethod
    def first_window_training_bank(self) -> pd.DataFrame:
        """The training days of the first window's ewma bank (the rows before ewma_bank starts), see build_first_window_training_bank."""

    # ---- what the config says --------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.display_name})"

    @property
    def run_id(self) -> str:
        return self.config["run_id"]

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def display_name(self) -> str:
        return self.config["display_name"]

    @property
    def walk_forward_id(self) -> str:
        return self.config["walk_forward_id"]

    @property
    def unit(self) -> str:
        """The return type the run's own records are made of (percent | log); its readings are always percent returns."""
        return self.config["setting"]["unit"]

    @property
    def params(self) -> dict:
        return self.config["params"]

    @property
    def recipe(self) -> ModelRecipe:
        return recipe_from_config(self.config)

    @property
    def total_window_losses(self) -> dict:
        """The four day-weighted totals of window_losses (total_window_losses)."""
        return total_window_losses(self.window_losses)

    # ---- the bank for the plots ----------------------------------------------------------------

    def combined_bank(self, percent: bool = False) -> ResidualBank:
        """
        A ResidualBank rebuilt like build_combined_residual_bank_WITHOUT_CALC, so the graph helpers work on it. In the run's
        own unit (the VaR is read off it); percent=True is a display-only copy with the returns and mu_hat as percent returns
        (for the hover data of the plots, which sit next to the percent-return readings).
        """
        recipe = self.recipe
        frequency = self.config["frequency"]
        bank = ResidualBank(
            recipe.default_mu_hat, recipe.mu_mode, recipe.rolling_mu_hat_window_size,
            start_timestamp=pd.Timestamp(self.windows["train_start"].iloc[0]),
            frequency=None if frequency is None else pd.Timedelta(frequency),
        )
        bank.trained = True
        bank.residuals = self.combined_residuals[RESIDUAL_COLS].copy()
        if percent:
            bank.residuals["actual_return"] = core.return_to_percent(bank.residuals["actual_return"], self.unit)
            bank.residuals["mu_hat"] = core.return_to_percent(bank.residuals["mu_hat"], self.unit)
        return bank

    # ---- the readings, derived from the combined residuals ----------------------------------------

    @cached_property
    def _readings(self) -> dict:
        return {}

    def _reading_result(self, reading: "Reading") -> ReadingResult:
        if reading not in self._readings:
            self._readings[reading] = self._compute_reading(reading)
        return self._readings[reading]

    def var_df(self, reading: "Reading") -> pd.DataFrame:
        return self._reading_result(reading).var_df.copy()

    def daily(self, reading: "Reading") -> pd.DataFrame:
        """var_df + the model's outputs of that day + quantile, exceeded, excess, tick_loss."""
        return self._reading_result(reading).daily.copy()

    def exceeding_days(self, reading: "Reading") -> pd.DataFrame:
        return self._reading_result(reading).exceeding_days.copy()

    def metrics(self, reading: "Reading") -> dict:
        return copy.deepcopy(self._reading_result(reading).metrics)

    def blocks(self, reading: "Reading") -> pd.DataFrame:
        return self._reading_result(reading).blocks.copy()

    def _compute_reading(self, reading: "Reading") -> ReadingResult:
        """
        Read the run's combined residuals with `reading`: var_dataframe (per VaR reading) and, per test confidence, the daily frame,
        the exceedance days, the window blocks and the test numbers.
        Same reading as create_combined_residuals_bank_and_read_var_df (the loss is -actual_return of the bank row), then
        converted to PERCENT returns (a no-op for a percent run): VaR and loss exactly (core.loss_to_percent), sigma / variance as
        they are, mu_hat as a return. The quantile of the standardized residuals does not depend on the unit.
        """
        unit = self.unit
        residuals = self.combined_residuals
        forecast_rows = residuals[residuals["part"] == "forecast"]
        var_unit = self.combined_bank().multi_day_var(
            forecast_rows.index, reading.var_confidence, reading.window_size, reading.min_periods,
        )
        quantile = -(var_unit["varForecast"].astype(float) + forecast_rows["mu_hat"]) / forecast_rows["predicted_sigma"]   # in the run's unit space

        var_df = var_unit.copy()
        var_df["varForecast"] = core.loss_to_percent(var_unit["varForecast"].astype(float), unit)
        var_df["loss"] = core.loss_to_percent(-forecast_rows["actual_return"], unit)
        var_df["window_id"] = forecast_rows["window_id"]
        var_df = var_df.astype({"varForecast": float, "loss": float, "n_residuals_used": int, "window_id": int})

        daily = var_df.join(forecast_rows[["predicted_variance", "predicted_sigma", "mu_hat", "standardized_residual"]])
        daily["predicted_variance"] = core.scale_to_percent(daily["predicted_variance"], unit, 2)
        daily["predicted_sigma"] = core.scale_to_percent(daily["predicted_sigma"], unit, 1)
        daily["mu_hat"] = core.return_to_percent(daily["mu_hat"], unit)
        daily["quantile"] = quantile
        daily["exceeded"] = daily["loss"] > daily["varForecast"]
        daily["excess"] = daily["loss"] - daily["varForecast"]
        daily["tick_loss"] = core.daily_tick_loss(var_df, reading.var_confidence)

        block_rows = {}
        for window_id, part in var_df.groupby("window_id", sort=True):
            suite = core.run_full_backtest_suite_single_window(
                part, part.index[0], part.index[-1], reading.var_confidence, reading.test_confidence,
            )
            block_rows[window_id] = {
                "start": part.index[0], "end": part.index[-1],
                **{k: suite[k] for k in (
                    "n_days", "n_exceedances", "exceedance_rate", "traffic_light_zone",
                    "kupiec_pof_pass", "christoffersen_ind_pass", "christoffersen_cc_pass",
                )},
                "mean_tick_loss": float(daily.loc[part.index, "tick_loss"].mean()),
                "mean_var": float(part["varForecast"].mean()),
            }

        backtest = core.run_full_backtest_suite_single_window(
            var_df, var_df.index[0], var_df.index[-1], reading.var_confidence, reading.test_confidence,
        )
        level_volatility = core.run_level_and_volatility_suite_single_window(var_df, reading.var_confidence)
        metrics = {
            "reading": dataclasses.asdict(reading),
            # every number about the VaR itself (the tests, the level, the cost, the tail, how much it moves) over all windows;
            # the numbers about the variance forecast are total_window_volatility_metrics (total_window_losses)
            "total_window_var_metrics": {**backtest, **level_volatility},
        }
        blocks = pd.DataFrame(block_rows).T.rename_axis("window_id").infer_objects()      # typed columns, as when read back from parquet
        blocks[["start", "end"]] = blocks[["start", "end"]].astype("datetime64[us]")
        return ReadingResult(
            var_df=var_df, daily=daily, exceeding_days=daily[daily["exceeded"]],
            blocks=blocks,
            metrics=json.loads(json.dumps(metrics, default=_json_default)),     # the same types as when it is read back from test_results.json
        )


class LoadedRun(Run):
    """
    A saved run, read back from its folder (run_store.load_walk_forward_run). Everything is lazy and file by file; nothing needs
    the price data or the training session. A reading that is not on disk yet is derived once and written there.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    @cached_property
    def config(self) -> dict:
        return _read_json(self.run_dir / "model_settings.json")

    @cached_property
    def windows(self) -> pd.DataFrame:
        # <walk_forward_id>/FeatureAndTargets/<key>/WalkForwardTrainedModels/<this run>
        return pd.read_parquet(self.run_dir.parents[3] / "WindowsCreatedCalendars.parquet")

    @cached_property
    def combined_residuals(self) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir / "WALK_FORWARD_FORECAST_RESIDUAL_BANK.parquet")

    @cached_property
    def window_losses(self) -> pd.DataFrame:
        """Per window: QLIKE and the model's own loss on its train and forecast days (window_losses)."""
        return pd.read_parquet(self.run_dir / "window_losses.parquet")

    def raw_residual_bank(self, window_id: int) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir / "raw_residual_banks" / f"{_calendar_name(window_id)}.parquet")

    @property
    def has_ewma_bank(self) -> bool:
        return (self.run_dir / "WALK_FORWARD_EWMA_BANK.parquet").exists()

    @cached_property
    def ewma_bank(self) -> pd.DataFrame:
        path = self.run_dir / "WALK_FORWARD_EWMA_BANK.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{self.display_name} has no ewma bank (not a long / ewma run).")
        return pd.read_parquet(path)

    @property
    def has_first_window_training_bank(self) -> bool:
        return (self.run_dir / FIRST_WINDOW_TRAINING_BANK_FILE).exists()

    @cached_property
    def first_window_training_bank(self) -> pd.DataFrame:
        path = self.run_dir / FIRST_WINDOW_TRAINING_BANK_FILE
        if not path.exists():
            raise FileNotFoundError(f"{self.display_name} has no first-window training bank (not a long / ewma run, or saved before it existed).")
        return pd.read_parquet(path)

    def raw_ewma_bank(self, window_id: int) -> pd.DataFrame:
        path = self.run_dir / "raw_ewma_banks" / f"{_calendar_name(window_id)}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{self.display_name} has no raw ewma banks (a slim run, or not a long / ewma run).")
        return pd.read_parquet(path)

    def model(self, window_id: int):
        """xgboost: the loaded XGBRegressor. garch: the parameters. ewma: None."""
        family = self.config["params"]["model_family"]
        if family == "xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(self.run_dir / "models" / f"{_calendar_name(window_id)}.ubj"))
            return model
        if family == "garch":
            return _read_json(self.run_dir / "models" / "garch_params.json")[str(window_id)]
        return None

    def feature_columns(self) -> list[str]:
        """The feature columns the models of this run were trained on, in the order they take them (from the saved feature importances)."""
        path = self.run_dir / "models" / "feature_importance.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{self.display_name} has no saved xgboost models.")
        return list(pd.read_parquet(path).columns)

    def features(self, window_id: int, part: str = "train") -> pd.DataFrame:
        path = self.run_dir / "features" / f"{_calendar_name(window_id)}_{part}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{self.display_name} was saved slim (without the feature files); train it again without slim to have them.")
        return pd.read_parquet(path)

    # ---- readings on disk ---------------------------------------------------------------------------------

    def _reading_dir(self, reading: Reading) -> Path:
        return self.run_dir / "readings_from_walk_forward" / reading.var_path

    def _test_dir(self, reading: Reading) -> Path:
        """The folder of a finished reading; test_results.json is written last, so its presence means finished."""
        folder = self.run_dir / "readings_from_walk_forward" / reading.test_path
        if not (folder / "test_results.json").exists():
            self.derive_reading(reading)
        return folder

    def var_df(self, reading: Reading) -> pd.DataFrame:
        self._test_dir(reading)
        return pd.read_parquet(self._reading_dir(reading) / "var_dataframe.parquet")

    def daily(self, reading: Reading) -> pd.DataFrame:
        """var_df + the model's outputs of that day + quantile, exceeded, excess, tick_loss (test_dataframe.parquet)."""
        return pd.read_parquet(self._test_dir(reading) / "test_dataframe.parquet")

    def exceeding_days(self, reading: Reading) -> pd.DataFrame:
        return pd.read_parquet(self._test_dir(reading) / "exceeding_days.parquet")

    def metrics(self, reading: Reading) -> dict:
        return _read_json(self._test_dir(reading) / "test_results.json")

    def blocks(self, reading: Reading) -> pd.DataFrame:
        return pd.read_parquet(self._test_dir(reading) / "window_blocks.parquet")

    # ---- the reading, derived from the combined residuals ----------------------------------------------

    def derive_reading(self, reading: Reading) -> Path:
        """
        Read the run's combined residuals with `reading` (Run._compute_reading) and write var_dataframe.parquet (per VaR reading) and,
        per test confidence, test_dataframe / exceeding_days / window_blocks / test_results.
        """
        result = self._compute_reading(reading)
        var_folder = self._reading_dir(reading)
        test_folder = self.run_dir / "readings_from_walk_forward" / reading.test_path
        test_folder.mkdir(parents=True, exist_ok=True)
        metadata = {**_parquet_metadata(self.config), "reading": reading.key, "values_in": "percent_return"}
        _to_parquet(result.var_df, var_folder / "var_dataframe.parquet", metadata)
        _to_parquet(result.daily, test_folder / "test_dataframe.parquet", metadata)
        _to_parquet(result.exceeding_days, test_folder / "exceeding_days.parquet", metadata)
        _to_parquet(result.blocks, test_folder / "window_blocks.parquet", metadata)
        _write_json(test_folder / "test_results.json", result.metrics)      # last: its presence marks a finished reading
        return test_folder


class TrainedInitialRun(Run):
    """
    A training that has not been saved: its fits read like a saved run, nothing is written (the readings stay in memory).
    config is what run_store.trained_initial_run builds (the same model_settings a saved run would get), windows is setting.window_table().
    """

    def __init__(self, config: dict, windows: pd.DataFrame, fits: list[WalkForwardSingleCalendarFit]):
        if not fits:
            raise ValueError("fits is empty.")
        self._config, self._windows, self._fits = config, windows, fits

    @property
    def config(self) -> dict:
        return self._config

    @property
    def windows(self) -> pd.DataFrame:
        return self._windows

    @cached_property
    def combined_residuals(self) -> pd.DataFrame:
        return build_combined_residuals_frame(self._fits)

    @cached_property
    def window_losses(self) -> pd.DataFrame:
        return window_losses(self._fits)

    @property
    def has_ewma_bank(self) -> bool:
        return self._fits[0].ewma_bank is not None

    @cached_property
    def ewma_bank(self) -> pd.DataFrame:
        if not self.has_ewma_bank:
            raise FileNotFoundError(f"{self.display_name} has no ewma bank (not a long / ewma run).")
        return build_combined_ewma_bank_WITHOUT_CALC(self._fits)

    @property
    def has_first_window_training_bank(self) -> bool:
        return self.has_ewma_bank

    @cached_property
    def first_window_training_bank(self) -> pd.DataFrame:
        return build_first_window_training_bank(self._fits)


def reweight_blocker(run: Run) -> str | None:
    """Why the run's per-ticker forecasts cannot be read for other weights (None = they can)."""
    if not run.has_ewma_bank:
        return "it has no per-ticker forecasts (a wide model or the historical VaR)"
    if not run.has_first_window_training_bank:
        return "its first-window training bank is not saved (run tools/backfill_first_window_training_bank.py)"
    recipe = run.recipe
    if recipe.model_family == "xgboost" and "MarketEwmaVol" in recipe.features.portfolio_dependent_blocks:
        return "its MarketEwmaVol feature is the portfolio's own ewma vol, so its forecasts depend on the weights"
    return None


class ReweightedVirtualRun(Run):
    """
    Another run's per-ticker forecasts read for OTHER portfolio weights: Sigma_t = D R D of the stored rows, the residuals, the quantile
    and the loss are recomputed with the weights given (reweighted_residuals). Nothing is trained and nothing is saved.
    Exact for GARCH, EWMA and the xgboost models with portable features; reweight_blocker says when it is not.
    base: any Run with an ewma bank. weights: ticker -> weight for exactly the base's tickers, normalised here.
    """

    def __init__(self, base: Run, weights: dict[str, float]):
        blocker = reweight_blocker(base)
        if blocker:
            raise ValueError(f"{base.display_name} cannot be re-weighted: {blocker}.")
        tickers = list(base.config["setting"]["tickers"])
        if set(weights) != set(tickers):
            raise ValueError(f"weights must cover exactly the run's tickers {tickers}, got {sorted(weights)}.")
        w = np.array([weights[t] for t in tickers], dtype=float)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum.")
        self.base, self.tickers, self.weights = base, tickers, w / w.sum()

    @cached_property
    def config(self) -> dict:
        config = copy.deepcopy(self.base.config)
        config["setting"]["weights"] = {t: float(w) for t, w in zip(self.tickers, self.weights)}
        digest = hashlib.sha256(json.dumps({t: round(float(w), 12) for t, w in zip(self.tickers, self.weights)}, sort_keys=True).encode()).hexdigest()[:8]
        config["run_id"] = f"{self.base.run_id}@{digest}"
        config["display_name"] = f"{self.base.display_name} | user defined weights"
        config["reweighted_from"] = self.base.run_id
        return config

    @property
    def windows(self) -> pd.DataFrame:
        return self.base.windows

    @cached_property
    def combined_residuals(self) -> pd.DataFrame:
        recipe = self.base.recipe
        frame = reweighted_residuals(
            self.base.first_window_training_bank, self.base.ewma_bank, self.tickers, self.weights, self.unit,
            recipe.mu_mode, recipe.default_mu_hat, recipe.rolling_mu_hat_window_size,
        )
        forecast_window = self.base.ewma_bank["window_id"]
        frame["window_id"] = forecast_window.reindex(frame.index).fillna(0).astype(int)
        frame["part"] = np.where(frame.index.isin(forecast_window.index), "forecast", "train")
        return frame

    @cached_property
    def window_losses(self) -> pd.DataFrame:
        """
        The base's table with the QLIKE of the forecast days recomputed for these weights. The in-sample QLIKE (train_qlike) is NaN: only
        the first window's training days are stored, so it cannot be recomputed. The model's own losses (target units, per ticker) do not
        depend on the weights and are the base's.
        """
        table = self.base.window_losses.copy()
        residuals = self.combined_residuals
        forecast = residuals[residuals["part"] == "forecast"]
        r2 = return_to_percent(forecast["actual_return"], self.unit) ** 2
        table["train_qlike"] = np.nan
        for window_id, rows in forecast.groupby("window_id"):
            table.loc[window_id, "forecast_days"] = len(rows)
            table.loc[window_id, "forecast_qlike"] = qlike(rows["predicted_variance"], r2.loc[rows.index])
        return table

    @property
    def has_ewma_bank(self) -> bool:
        return True

    def _with_portfolio_variance(self, bank: pd.DataFrame) -> pd.DataFrame:
        """The base's per-ticker rows with the portfolio variance of these weights (the other columns do not depend on the weights)."""
        bank = bank.copy()
        bank["portfolio_variance"] = self.combined_residuals["predicted_variance"].loc[bank.index]
        return bank

    @cached_property
    def ewma_bank(self) -> pd.DataFrame:
        return self._with_portfolio_variance(self.base.ewma_bank)

    @property
    def has_first_window_training_bank(self) -> bool:
        return True

    @cached_property
    def first_window_training_bank(self) -> pd.DataFrame:
        return self._with_portfolio_variance(self.base.first_window_training_bank)
