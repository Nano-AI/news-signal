"""Score one replay pass: per-pair latency, misses, duplicates, and prediction quality."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from infosys.newsd import PassLog, ReceivedSignal
from infosys.replay import ReplayPlan
from infosys.store import Bars

MIN_IC_PAIRS = 30
# A pass is valid iff send lag p90 <= 5 ms and p99 <= 50 ms. Systematic lateness (a starved server) moves p90;
# rare node stalls only move the tail. Above either bound the latencies are not trustworthy.
MAX_SEND_LAG_P90_MS = 5.0
MAX_SEND_LAG_P99_MS = 50.0


@dataclass(frozen=True)
class PassScore:
    stories: int
    pairs: int
    signaled: int
    missed: int
    duplicates: int
    invalid: int
    latency_ms: dict[str, float | None]  # None when there are no pairs
    throughput_per_s: float
    scored_pairs: int
    ic: float | None
    ic_se: float | None
    hit_rate: float | None
    send_lag_ms: dict[str, float]  # p50, p90, p99, max of newsd's send lateness; zeros when nothing was sent
    valid: bool = True
    invalid_reason: str | None = None
    # Record only. Per feed: {"pairs", "p50", "p99", "first"}; see _feed_stats. Empty when the log has no feeds.
    feed_latency_ms: dict[str, dict[str, float | int | None]] = field(default_factory=dict)


@dataclass(frozen=True)
class Fidelity:
    """How closely a pass reproduces a reference signal set (the baseline FinBERT's), pair by pair."""

    pairs: int  # measured pairs; each is in the reference
    agreed: int  # pairs whose signal direction equals the reference direction; a missed pair never agrees
    agreement: float  # agreed / pairs
    both: int  # pairs the candidate signalled
    non_neutral: int  # pairs where the candidate's and the reference's signal are both non-zero
    spearman: float | None  # rank correlation of the two signals over the non-neutral pairs; None if undefined
    # Record only, never gated: rank correlation of |signal| over the pairs non-zero and same-sign on both sides
    # (the within-sign confidence agreement); None below 30 such pairs.
    confidence_spearman: float | None = None


def _ic(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < MIN_IC_PAIRS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = float(spearmanr(x, y).statistic)
    return None if np.isnan(r) else r


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    p50, p90, p99 = (float(np.percentile(values, q)) for q in (50, 90, 99))
    return {"p50": p50, "p90": p90, "p99": p99, "max": float(max(values))}


def _feed_stats(plan: ReplayPlan, log: PassLog, first: dict[tuple[str, str], ReceivedSignal]) -> dict:
    """Per feed, over measured pairs with a signal that the feed carried: signal receipt minus that feed's own
    hand-over instant of the story (negative when another feed's copy was acted on before this feed delivered).
    "first" counts measured stories this feed handed over earliest (ties go to the feed name that sorts first)."""
    lat: dict[str, list[float]] = {feed: [] for feed in log.feed_sent_s}
    firsts = dict.fromkeys(lat, 0)
    for story in plan.measured:
        carried = [(stamps[story.story_id], feed) for feed, stamps in log.feed_sent_s.items() if story.story_id in stamps]
        if carried:
            firsts[min(carried)[1]] += 1
        for ticker in story.tickers:
            sig = first.get((story.story_id, ticker))
            if sig is not None:
                for at, feed in carried:
                    lat[feed].append((sig.received_s - at) * 1000)
    out = {}
    for feed, values in lat.items():
        p = _percentiles(values)
        out[feed] = {"pairs": len(values), "p50": p["p50"], "p99": p["p99"], "first": firsts[feed]}
    return out


def first_signals(plan: ReplayPlan, log: PassLog) -> tuple[dict[tuple[str, str], ReceivedSignal], int]:
    """The first signal of each measured pair (log order is receipt order), and the count of later ones."""
    measured_pairs = {(s.story_id, t) for s in plan.measured for t in s.tickers}
    first: dict[tuple[str, str], ReceivedSignal] = {}
    duplicates = 0
    for sig in log.signals:
        key = (sig.story_id, sig.ticker)
        if key not in measured_pairs:
            continue
        if key in first:
            duplicates += 1
        else:
            first[key] = sig
    return first, duplicates


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def fidelity(plan: ReplayPlan, log: PassLog, reference: Mapping[tuple[str, str], float]) -> Fidelity:
    """Agreement and signal Spearman against `reference`, which maps (story_id, ticker) to direction * confidence.

    Direction is the sign of the signal, the same `direction * confidence` as the IC: -1, 0 (neutral, or a zero
    confidence) or +1. Neutral agrees only with neutral. The Spearman is over the pairs where both signals are non-zero:
    the neutral zeros are ties that would otherwise dominate the ranks and hide a faked confidence; it is None
    (undefined, so a gate fails) below 30 such pairs. A signed 0.95 needs only a within-sign confidence correlation
    of about 0.8; `confidence_spearman` reports that within-sign correlation, for the record. Raises ValueError if a measured pair is not in the reference."""
    first, _ = first_signals(plan, log)
    keys = [(s.story_id, t) for s in plan.measured for t in s.tickers]
    absent = sum(k not in reference for k in keys)
    if absent:
        raise ValueError(f"{absent} measured pairs are not in the reference")
    cand, ref, nn_cand, nn_ref, ss_cand, ss_ref = [], [], [], [], [], []
    agreed = 0
    for key in keys:
        sig = first.get(key)
        if sig is not None:
            cand.append(sig.direction * sig.confidence)
            ref.append(reference[key])
            agreed += _sign(cand[-1]) == _sign(ref[-1])
            if cand[-1] and ref[-1]:
                nn_cand.append(cand[-1])
                nn_ref.append(ref[-1])
                if cand[-1] * ref[-1] > 0:
                    ss_cand.append(abs(cand[-1]))
                    ss_ref.append(abs(ref[-1]))
    return Fidelity(
        pairs=len(keys), agreed=agreed, agreement=agreed / len(keys) if keys else 0.0, both=len(cand), non_neutral=len(nn_cand),
        spearman=_ic(np.array(nn_cand), np.array(nn_ref)),
        confidence_spearman=_ic(np.array(ss_cand), np.array(ss_ref)),
    )


def score_pass(
    plan: ReplayPlan, log: PassLog, bars: Bars, *, horizon_min: int, bootstrap: int = 1000, bootstrap_seed: int = 0
) -> PassScore:
    """Measured pairs only; the first signal of a pair counts, later ones are duplicates."""
    first, duplicates = first_signals(plan, log)

    latencies: list[float] = []
    pairs = signaled = 0
    xs: list[list[float]] = []  # per story with a scored pair: signal strength
    ys: list[list[float]] = []  # and forward return
    directions: list[int] = []
    for story in plan.measured:
        avail = log.sent_s.get(story.story_id, plan.available_s[story.story_id])  # actual send, else scheduled
        sx, sy = [], []
        for ticker in story.tickers:
            pairs += 1
            sig = first.get((story.story_id, ticker))
            if sig is None:
                latencies.append((log.end_s - avail) * 1000)
                continue
            signaled += 1
            latencies.append((sig.received_s - avail) * 1000)
            ret = bars.forward_return(ticker, story.published_at, horizon_min)
            if ret is not None:
                sx.append(sig.direction * sig.confidence)
                sy.append(ret)
                directions.append(sig.direction)
        if sx:
            xs.append(sx)
            ys.append(sy)

    x = np.array([v for s in xs for v in s])
    y = np.array([v for s in ys for v in s])
    d = np.array(directions)
    ic = _ic(x, y)
    hit_rate = None
    traded = (d != 0) & (y != 0)
    if traded.any():
        hit_rate = float(np.mean(np.sign(d[traded]) == np.sign(y[traded])))

    ic_se = None
    if ic is not None and bootstrap > 1:
        rng = np.random.default_rng(bootstrap_seed)
        sx_arrays, sy_arrays = [np.array(s) for s in xs], [np.array(s) for s in ys]
        samples = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(sx_arrays), len(sx_arrays))
            r = _ic(np.concatenate([sx_arrays[i] for i in idx]), np.concatenate([sy_arrays[i] for i in idx]))
            if r is not None:
                samples.append(r)
        if len(samples) > 1:
            ic_se = float(np.std(samples, ddof=1))

    span = (
        max(s.received_s for s in first.values()) - min(log.sent_s.get(s.story_id, plan.available_s[s.story_id]) for s in plan.measured)
        if first
        else 0.0
    )
    lag = log.send_lag_ms
    send_lag = {k: v or 0.0 for k, v in _percentiles(lag).items()}
    failed = [
        f"send lag {k} {send_lag[k]:.1f} ms exceeds {bound} ms"
        for k, bound in (("p90", MAX_SEND_LAG_P90_MS), ("p99", MAX_SEND_LAG_P99_MS))
        if send_lag[k] > bound
    ]
    reason = "; ".join(failed) or None
    return PassScore(
        stories=len(plan.measured),
        pairs=pairs,
        signaled=signaled,
        missed=pairs - signaled,
        duplicates=duplicates,
        invalid=sum(log.invalid.values()) + log.schema_errors,
        latency_ms=_percentiles(latencies),
        throughput_per_s=signaled / span if span > 0 else 0.0,
        scored_pairs=len(x),
        ic=ic,
        ic_se=ic_se,
        hit_rate=hit_rate,
        send_lag_ms=send_lag,
        valid=reason is None,
        invalid_reason=reason,
        feed_latency_ms=_feed_stats(plan, log, first),
    )
