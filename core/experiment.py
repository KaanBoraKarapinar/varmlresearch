"""
An experiment: a named, saved comparison of runs (which runs, which baseline, which analysis period, which VaR reading, what we concluded).

The experiment refers to runs by their run_id; a run knows nothing about experiments, so one run can be in any number of them.
One experiment = one json file, <store root>/experiments/<slug>.json.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

EXPERIMENTS_DIR = "experiments"
READING_KEYS = ("var_confidence", "window_size", "min_periods", "test_confidence")


@dataclass(frozen=True)
class Experiment:
    name: str
    question: str                  # what the experiment asks, one line
    run_ids: tuple[str, ...]       # store run ids (Run.run_id)
    baseline_run_id: str           # one of run_ids
    start: str                     # analysis period, ISO dates
    end: str
    reading: dict                  # var_confidence, window_size (None = expanding), min_periods, test_confidence
    notes: str = ""                # our own conclusion, markdown

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("An experiment needs a name.")
        if len(self.run_ids) < 2:
            raise ValueError("An experiment compares at least two runs.")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("A run is listed twice.")
        if self.baseline_run_id not in self.run_ids:
            raise ValueError("The baseline must be one of the experiment's runs.")
        if set(self.reading) != set(READING_KEYS):
            raise ValueError(f"reading needs exactly {READING_KEYS}.")
        if self.start > self.end:
            raise ValueError("The analysis period ends before it starts.")

    @property
    def slug(self) -> str:
        return slugify(self.name)


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"'{name}' has no usable letters for a file name.")
    return slug


def experiments_dir(root: Path) -> Path:
    return Path(root) / EXPERIMENTS_DIR


def save_experiment(experiment: Experiment, root: Path, overwrite: bool = False) -> Path:
    path = experiments_dir(root) / f"{experiment.slug}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Experiment '{experiment.name}' already exists ({path.name}).")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**dataclasses.asdict(experiment), "run_ids": list(experiment.run_ids)}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_experiment(slug: str, root: Path) -> Experiment:
    data = json.loads((experiments_dir(root) / f"{slug}.json").read_text())
    return Experiment(**{**data, "run_ids": tuple(data["run_ids"])})


def list_experiments(root: Path) -> list[Experiment]:
    """Every saved experiment, by name."""
    folder = experiments_dir(root)
    if not folder.exists():
        return []
    return sorted((load_experiment(p.stem, root) for p in folder.glob("*.json")), key=lambda e: e.name.lower())


def experiments_containing(run_id: str, root: Path) -> list[Experiment]:
    """The experiments a run is part of."""
    return [e for e in list_experiments(root) if run_id in e.run_ids]
