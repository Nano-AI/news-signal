"""Event-store format: gzip JSON-lines stories, npz minute bars, sha256 manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Literal, get_args
from zoneinfo import ZoneInfo

import numpy as np

Split = Literal["train", "validation", "holdout"]
SPLITS: tuple[Split, ...] = get_args(Split)
SPLIT_STARTS: dict[Split, datetime] = {
    "train": datetime(2020, 1, 1, tzinfo=UTC),
    "validation": datetime(2023, 7, 1, tzinfo=UTC),
    "holdout": datetime(2024, 7, 1, tzinfo=UTC),
}
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Story:
    story_id: str
    source: Literal["benzinga", "edgar"]
    published_at: datetime  # tz-aware UTC, whole seconds
    tickers: tuple[str, ...]
    headline: str
    summary: str


def split_of(published_at: datetime) -> Split:
    if published_at < SPLIT_STARTS["train"]:
        raise ValueError(f"{published_at.isoformat()} is before {SPLIT_STARTS['train'].date()}")
    if published_at >= SPLIT_STARTS["holdout"]:
        return "holdout"
    if published_at >= SPLIT_STARTS["validation"]:
        return "validation"
    return "train"


def _gzip_write(path: Path, data: bytes) -> None:
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(data)


def _check_utc(when: datetime, *, whole_seconds: bool = False) -> None:
    if when.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    if whole_seconds and when.microsecond:
        raise ValueError(f"{when.isoformat()} is not a whole second")


def write_stories(path: Path, stories: Iterable[Story]) -> int:
    ordered = sorted(stories, key=lambda s: (s.published_at, s.story_id))
    for s in ordered:
        _check_utc(s.published_at, whole_seconds=True)
    ids = [s.story_id for s in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate story_id")
    lines = [
        json.dumps(
            {
                "story_id": s.story_id,
                "source": s.source,
                "published_at": s.published_at.astimezone(UTC).strftime(_TS_FORMAT),
                "tickers": list(s.tickers),
                "headline": s.headline,
                "summary": s.summary,
            },
            ensure_ascii=False,
        )
        for s in ordered
    ]
    _gzip_write(path, "".join(line + "\n" for line in lines).encode())
    return len(lines)


def read_stories(path: Path) -> list[Story]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [
        Story(
            story_id=r["story_id"],
            source=r["source"],
            published_at=datetime.strptime(r["published_at"], _TS_FORMAT).replace(tzinfo=UTC),
            tickers=tuple(r["tickers"]),
            headline=r["headline"],
            summary=r["summary"],
        )
        for r in rows
    ]


# (minute-start epoch seconds int64, open float64, close float64), sorted and unique by time
BarArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def write_bars(path: Path, bars: Mapping[str, BarArrays]) -> int:
    arrays: dict[str, np.ndarray] = {}
    rows = 0
    for ticker, (t, o, c) in sorted(bars.items()):
        t, o, c = np.asarray(t, np.int64), np.asarray(o, np.float64), np.asarray(c, np.float64)
        if not (len(t) == len(o) == len(c)):
            raise ValueError(f"{ticker}: bar arrays differ in length")
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(c)) and np.all(o > 0) and np.all(c > 0)):
            raise ValueError(f"{ticker}: bar prices must be finite and positive")
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError(f"{ticker}: bar times are not sorted and unique")
        arrays[f"{ticker}.t"], arrays[f"{ticker}.o"], arrays[f"{ticker}.c"] = t, o, c
        rows += len(t)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    return rows


class Bars:
    def __init__(self, data: Mapping[str, BarArrays]):
        self._data = dict(data)

    def tickers(self) -> list[str]:
        return sorted(self._data)

    def arrays(self, ticker: str) -> BarArrays:
        return self._data[ticker]

    def _bar_before(self, ticker: str, t: datetime, max_lookback_s: int) -> tuple[int, float] | None:
        """(start, close) of the latest bar that has ended by `t`, or None if stale or unknown."""
        _check_utc(t)
        if ticker not in self._data:
            return None
        times, _, closes = self._data[ticker]
        latest_start = math.floor(t.timestamp()) - 60  # a bar ends at start + 60
        i = int(np.searchsorted(times, latest_start, side="right")) - 1
        if i < 0 or times[i] < latest_start - max_lookback_s:
            return None
        return int(times[i]), float(closes[i])

    def close_before(self, ticker: str, t: datetime, max_lookback_s: int = 1800) -> float | None:
        """Close of the latest bar that has ended by `t`; None if stale or unknown."""
        bar = self._bar_before(ticker, t, max_lookback_s)
        return None if bar is None else bar[1]

    def forward_return(self, ticker: str, published_at: datetime, horizon_min: int) -> float | None:
        """Return over `horizon_min` minutes inside one regular New York session, else None.

        The start price is the close of a bar that ended at or before `published_at`, at most 30 minutes
        old. The end price is the close of a bar that ended in (`published_at`, `published_at` + H]; if
        none did (early close, data gap), the result is None rather than a stale price.
        """
        _check_utc(published_at)
        if horizon_min <= 0:
            raise ValueError("horizon_min must be positive")
        end_at = published_at + timedelta(minutes=horizon_min)
        p, q = published_at.astimezone(_NY), end_at.astimezone(_NY)
        if p.date() != q.date() or p.time() < time(9, 31) or q.time() > time(16, 0):
            return None
        start = self._bar_before(ticker, published_at, 1800)
        end = self._bar_before(ticker, end_at, 1800)
        if start is None or end is None or end[0] + 60 <= math.floor(published_at.timestamp()):
            return None
        return end[1] / start[1] - 1


def read_bars(path: Path) -> Bars:
    with np.load(path) as z:
        tickers = sorted({k.rsplit(".", 1)[0] for k in z.files})
        return Bars({t: (z[f"{t}.t"], z[f"{t}.o"], z[f"{t}.c"]) for t in tickers})


def store_files(split: Split) -> tuple[str, str]:
    return f"stories-{split}.jsonl.gz", f"bars-{split}.npz"


def load_split(data_dir: Path, split: Split) -> tuple[list[Story], Bars]:
    stories_name, bars_name = store_files(split)
    return read_stories(data_dir / stories_name), read_bars(data_dir / bars_name)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _row_count(path: Path) -> int:
    if path.suffix == ".npz":
        with np.load(path) as z:
            return sum(len(z[k]) for k in z.files if k.endswith(".t"))
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_manifest(data_dir: Path, extra: Mapping[str, object]) -> dict:
    """`extra` must carry `tickers`, the store's universe; it may not override `format` or `files`."""
    if "tickers" not in extra:
        raise ValueError("manifest extra must list the store's `tickers`")
    if {"format", "files"} & set(extra):
        raise ValueError("manifest extra may not set `format` or `files`")
    files = {}
    for split in SPLITS:
        for name in store_files(split):
            path = data_dir / name
            files[name] = {"sha256": sha256_file(path), "rows": _row_count(path)}
    manifest = {"format": 1, "files": files, **extra}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class StoreMismatch(Exception):
    """A needed store file is missing or differs from the pinned digest. Message is the file name."""


def verify_store(data_dir: Path, pinned: Mapping, splits: Iterable[Split]) -> None:
    for split in splits:
        for name in store_files(split):
            path = data_dir / name
            expected = pinned["files"].get(name, {}).get("sha256")
            if expected is None or not path.is_file() or sha256_file(path) != expected:
                raise StoreMismatch(name)
