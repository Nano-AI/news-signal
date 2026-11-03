"""Replay planning: pure functions from stories to a timed delivery schedule."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from infosys.store import Story

MAX_DELAY_MS = 2000.0
MAX_DUP_GAP_S = 0.5


@dataclass(frozen=True)
class FeedSpec:
    name: str
    transport: Literal["ws", "poll"]
    real: bool
    sources: frozenset[str]
    base_ms: float
    jitter_ms: float
    drop: float
    dup: float


_BOTH = frozenset({"benzinga", "edgar"})
FEEDS: tuple[FeedSpec, ...] = (
    FeedSpec("benzinga", "ws", True, frozenset({"benzinga"}), 60, 40, 0, 0),
    FeedSpec("edgar", "poll", True, frozenset({"edgar"}), 200, 100, 0, 0),
    FeedSpec("syn-0", "ws", False, _BOTH, 20, 150, 0.05, 0.03),
    FeedSpec("syn-1", "poll", False, _BOTH, 0, 300, 0.05, 0.03),
    FeedSpec("syn-2", "ws", False, _BOTH, 80, 30, 0.10, 0.01),
    FeedSpec("syn-3", "poll", False, _BOTH, 40, 60, 0.02, 0.05),
)


@dataclass(frozen=True)
class Delivery:
    at_s: float
    feed: str
    seq: int
    story: Story


@dataclass(frozen=True)
class ReplayPlan:
    deliveries: tuple[Delivery, ...]
    measured: tuple[Story, ...]
    warmup_ids: frozenset[str]
    release_s: Mapping[str, float]
    available_s: Mapping[str, float]
    last_s: float


def _order(story: Story) -> tuple:
    return (story.published_at, story.story_id)


def compress(stories: Sequence[Story], gap_cap_s: float) -> list[float]:
    """Release offsets: the first story at 0, then each gap capped at `gap_cap_s`."""
    out: list[float] = []
    for i, story in enumerate(stories):
        if i == 0:
            out.append(0.0)
        else:
            gap = (story.published_at - stories[i - 1].published_at).total_seconds()
            out.append(out[-1] + min(gap, gap_cap_s))
    return out


def burst_windows(
    stories: Sequence[Story], *, window_s: int = 300, budget_s: float = 180.0, gap_cap_s: float = 2.0
) -> list[Story]:
    """Stories of the busiest `window_s` buckets whose compressed replay fits in `budget_s`."""
    buckets: dict[int, list[Story]] = defaultdict(list)
    for story in sorted(stories, key=_order):
        buckets[math.floor(story.published_at.timestamp() / window_s)].append(story)
    # most stories first; ties go to the earlier bucket
    ranked = sorted(buckets, key=lambda b: (-len(buckets[b]), b))
    total, chosen = 0.0, []
    for b in ranked:
        cost = compress(buckets[b], gap_cap_s)[-1] + (gap_cap_s if chosen else 0.0)
        if total + cost <= budget_s:
            total += cost
            chosen.append(b)
    return [s for b in sorted(chosen) for s in buckets[b]]


def sample_stories(stories: Sequence[Story], n: int, seed: int) -> list[Story]:
    """A seeded uniform subset of `n` stories in input (time) order, or all if there are at most `n`."""
    if len(stories) <= n:
        return list(stories)
    picked = sorted(random.Random(seed).sample(range(len(stories)), n))
    return [stories[i] for i in picked]


def _feed_deliveries(feed: FeedSpec, story: Story, release: float, seed: int) -> list[tuple[float, Story]]:
    rng = random.Random(f"{seed}:{feed.name}:{story.story_id}")
    if not feed.real and rng.random() < feed.drop:
        return []
    jitter = rng.expovariate(1 / feed.jitter_ms) if feed.jitter_ms > 0 else 0.0
    at = release + min(feed.base_ms + jitter, MAX_DELAY_MS) / 1000
    out = [(at, story)]
    if not feed.real and rng.random() < feed.dup:
        out.append((at + rng.uniform(0, MAX_DUP_GAP_S), story))
    return out


def plan_pass(
    measured: Sequence[Story],
    warmup: Sequence[Story],
    *,
    gap_cap_s: float,
    seed: int,
    feeds: Sequence[FeedSpec] = FEEDS,
    warmup_rate_hz: float = 10.0,
    warmup_gap_s: float = 2.0,
) -> ReplayPlan:
    """Schedule warmup stories at a fixed rate, then the measured ones compressed, on every feed.

    Warmup ends one slot after its last story (`len(warmup) / warmup_rate_hz`).
    """
    measured = sorted(measured, key=_order)
    warmup = sorted(warmup, key=_order)
    if {s.story_id for s in measured} & {s.story_id for s in warmup}:
        raise ValueError("a story is in both the measured and the warmup set")
    release: dict[str, float] = {s.story_id: i / warmup_rate_hz for i, s in enumerate(warmup)}
    start = len(warmup) / warmup_rate_hz + warmup_gap_s
    release.update({s.story_id: start + r for s, r in zip(measured, compress(measured, gap_cap_s), strict=True)})

    raw: dict[str, list[tuple[float, Story]]] = {f.name: [] for f in feeds}
    for story in [*warmup, *measured]:
        for feed in feeds:
            if story.source in feed.sources:
                raw[feed.name] += _feed_deliveries(feed, story, release[story.story_id], seed)

    deliveries: list[Delivery] = []
    for name, items in raw.items():
        items.sort(key=lambda it: (it[0], it[1].published_at, it[1].story_id))
        deliveries += [Delivery(at, name, i, story) for i, (at, story) in enumerate(items, start=1)]
    deliveries.sort(key=lambda d: (d.at_s, d.feed, d.seq))

    available: dict[str, float] = {}
    for d in deliveries:
        available.setdefault(d.story.story_id, d.at_s)  # deliveries are time-sorted
    missing = release.keys() - available.keys()
    if missing:
        raise ValueError(f"no feed carries story {sorted(missing)[0]}")
    return ReplayPlan(
        deliveries=tuple(deliveries),
        measured=tuple(measured),
        warmup_ids=frozenset(s.story_id for s in warmup),
        release_s=release,
        available_s=available,
        last_s=deliveries[-1].at_s if deliveries else 0.0,
    )
