"""
Analysis periods: the S&P 500 stress episodes, each in three phases, defined once so that any experiment can be read on any of them.

An experiment fixes runs, baseline, reading and its own analysis period (core.Experiment); these periods are not saved per experiment: the Experiments page
offers them next to the experiment's own period, and the Results page reads every experiment on every one of them.

    decline    peak   -> trough
    recovery   trough -> new high
    full       peak   -> new high

The dates are the table of the S&P 500 stress episodes; "new high" is approximate where that table has a ~.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STRESS_PHASES = ("decline", "recovery", "full")


@dataclass(frozen=True)
class Episode:
    name: str
    peak: str
    trough: str
    new_high: str
    decline: tuple[str, str] | None = None      # the decline's dates when they are not peak -> trough (COVID-19, see EPISODES)

    def phase(self, phase: str) -> tuple[str, str]:
        dates = {"decline": self.decline or (self.peak, self.trough), "recovery": (self.trough, self.new_high), "full": (self.peak, self.new_high)}
        if phase not in dates:
            raise ValueError(f"phase must be one of {STRESS_PHASES}, not '{phase}'.")
        return dates[phase]


@dataclass(frozen=True)
class AnalysisPeriod:
    episode: str
    phase: str
    start: str      # ISO dates
    end: str

    @property
    def label(self) -> str:
        return f"{self.episode} · {self.phase}"


    def covered_by(self, days: pd.DatetimeIndex) -> bool:
        """Whether every day of the period lies inside the days the runs forecast: a crisis read on its last weeks only is not that crisis."""
        return days[0] <= pd.Timestamp(self.start) and pd.Timestamp(self.end) <= days[-1]


EPISODES = (
    Episode("European Debt Crisis, wave 1", "2010-04-23", "2010-07-02", "2010-11-04"),
    Episode("European Debt Crisis, wave 2", "2011-04-29", "2011-10-03", "2012-02-24"),
    Episode("China slowdown", "2015-05-21", "2016-02-11", "2016-07-11"),
    Episode("February 2018 correction", "2018-01-26", "2018-02-08", "2018-08-24"),
    Episode("December 2018 selloff", "2018-09-20", "2018-12-24", "2019-04-23"),
    # peak -> trough is 24 trading days, under the 30 the comparison pages need: the decline is 3 trading days wider at each end (2020-02-13 .. 2020-03-26)
    Episode("COVID-19", "2020-02-19", "2020-03-23", "2020-08-18", decline=("2020-02-13", "2020-03-26")),
    Episode("2022 bear market", "2022-01-03", "2022-10-12", "2024-01-19"),
    Episode("2025 tariff shock", "2025-02-19", "2025-04-08", "2025-06-27"),
)


def stress_periods() -> list[AnalysisPeriod]:
    """Every phase of every episode, in time order (the episodes by their peak, then decline, recovery, full)."""
    return [AnalysisPeriod(e.name, phase, *e.phase(phase)) for e in EPISODES for phase in STRESS_PHASES]
