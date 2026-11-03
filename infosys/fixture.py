"""Deterministic, fully synthetic event store for tests and the CPU check.

    python -m infosys.fixture --out infosys/fixtures/tiny
"""

from __future__ import annotations

import argparse
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from infosys.store import BarArrays, Split, Story, store_files, write_bars, write_manifest, write_stories

TICKERS = ("AAA", "BBB", "CCC", "DDD")
_NY = ZoneInfo("America/New_York")
# split -> (trading days, story count)
_PLAN: dict[Split, tuple[list[date], int]] = {
    "train": ([date(2023, 6, d) for d in range(5, 10)], 60),
    "validation": ([date(2023, 7, d) for d in range(10, 15)], 120),
    "holdout": ([date(2024, 7, d) for d in range(8, 11)], 60),
}
_TEMPLATES = {
    1: ["{t} beats earnings estimates", "{t} raises full-year guidance", "{t} wins large contract"],
    -1: ["{t} misses earnings estimates", "{t} cuts full-year guidance", "{t} faces product recall"],
    0: ["{t} schedules investor day", "{t} names new board member", "{t} files routine report"],
}
_CLUSTER_SPLIT, _CLUSTER_SIZE = "validation", 6
_DRIFT_PER_MIN, _DRIFT_MINUTES = 0.05, 10


def _utc(day: date, hour: int, minute: int, second: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=_NY).astimezone(UTC)


def _stories(rng: random.Random, split: Split) -> tuple[list[Story], dict[str, int]]:
    days, count = _PLAN[split]
    stories, sign_of = [], {}
    cluster_at = _utc(days[1], 10, 15, 30)
    for i in range(count):
        day = days[i % len(days)]
        if split == _CLUSTER_SPLIT and i < _CLUSTER_SIZE:
            published = cluster_at
        elif i % 10 == 9:  # pre-market or after-hours
            hour = rng.choice([7, 8, 17, 18])
            published = _utc(day, hour, rng.randrange(60), rng.randrange(60))
        else:  # leave room for a 10-minute forward window before the close
            published = _utc(day, 9, 31, 0) + timedelta(seconds=rng.randrange(0, 5 * 3600 + 25 * 60))
        n_tickers = rng.choice([2, 3]) if i % 5 == 2 else 1
        tickers = tuple(sorted(rng.sample(TICKERS, n_tickers)))
        sign = rng.choice([1, -1, 0])
        headline = rng.choice(_TEMPLATES[sign]).format(t=tickers[0])
        sid = f"{split[:2]}-{i:04d}"
        sign_of[sid] = sign
        stories.append(
            Story(
                story_id=sid,
                source="edgar" if i % 5 == 4 else "benzinga",
                published_at=published,
                tickers=tickers,
                headline=headline,
                summary=f"{headline}. Synthetic summary {i}.",
            )
        )
    return stories, sign_of


def _bars(rng: random.Random, split: Split, stories: list[Story], sign_of: dict[str, int]) -> dict[str, BarArrays]:
    out: dict[str, list[tuple[int, float, float]]] = {t: [] for t in TICKERS}
    price = {t: 100.0 for t in TICKERS}
    for day in _PLAN[split][0]:
        start = int(_utc(day, 9, 30, 0).timestamp())
        starts = [start + 60 * m for m in range(390)]
        drift = {t: np.zeros(390) for t in TICKERS}
        for s in stories:
            minute = (int(s.published_at.timestamp()) - start) // 60 + 1
            for t in s.tickers:
                if 0 <= minute < 390:
                    drift[t][minute : minute + _DRIFT_MINUTES] += sign_of[s.story_id] * _DRIFT_PER_MIN
        for t in TICKERS:
            for m, ts in enumerate(starts):
                open_ = price[t]
                price[t] = round(open_ + rng.gauss(0, 0.02) + drift[t][m], 4)
                out[t].append((ts, open_, price[t]))
    return {
        t: (
            np.array([r[0] for r in rows], np.int64),
            np.array([r[1] for r in rows], np.float64),
            np.array([r[2] for r in rows], np.float64),
        )
        for t, rows in out.items()
    }


def build_fixture(out: Path) -> None:
    rng = random.Random(7)
    out.mkdir(parents=True, exist_ok=True)
    for split in _PLAN:
        stories, sign_of = _stories(rng, split)
        stories_name, bars_name = store_files(split)
        write_stories(out / stories_name, stories)
        write_bars(out / bars_name, _bars(rng, split, stories, sign_of))
    write_manifest(out, {"fixture": "tiny"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    build_fixture(parser.parse_args().out)
