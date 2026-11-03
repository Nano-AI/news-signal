"""Score one replay pass: per-pair latency, misses, duplicates, and prediction quality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from infosys.newsd import PassLog, ReceivedSignal
from infosys.replay import ReplayPlan
from infosys.store import Bars

MIN_IC_PAIRS = 30


@dataclass(frozen=True)
class PassScore:
    stories: int
    pairs: int
    signaled: int
    missed: int
    duplicates: int
    invalid: int
    latency_ms: dict[str, float]
    throughput_per_s: float
    scored_pairs: int
    ic: float | None
    ic_se: float | None
    hit_rate: float | None
    send_lag_p99_ms: float


def _ic(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < MIN_IC_PAIRS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = float(spearmanr(x, y).statistic)
    return None if np.isnan(r) else r


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    p50, p90, p99 = (float(np.percentile(values, q)) for q in (50, 90, 99))
    return {"p50": p50, "p90": p90, "p99": p99, "max": float(max(values))}


def score_pass(
    plan: ReplayPlan, log: PassLog, bars: Bars, *, horizon_min: int, bootstrap: int = 1000, bootstrap_seed: int = 0
) -> PassScore:
    """Measured pairs only; the first signal of a pair counts, later ones are duplicates."""
    measured_pairs = {(s.story_id, t) for s in plan.measured for t in s.tickers}
    first: dict[tuple[str, str], ReceivedSignal] = {}
    duplicates = 0
    for sig in log.signals:  # log order is receipt order
        key = (sig.story_id, sig.ticker)
        if key not in measured_pairs:
            continue
        if key in first:
            duplicates += 1
        else:
            first[key] = sig

    latencies: list[float] = []
    pairs = signaled = 0
    xs: list[list[float]] = []  # per story with a scored pair: signal strength
    ys: list[list[float]] = []  # and forward return
    directions: list[int] = []
    for story in plan.measured:
        avail = plan.available_s[story.story_id]
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
        max(s.received_s for s in first.values()) - min(plan.available_s[s.story_id] for s in plan.measured)
        if first
        else 0.0
    )
    lag = log.send_lag_ms
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
        send_lag_p99_ms=float(np.percentile(lag, 99)) if lag else 0.0,
    )
