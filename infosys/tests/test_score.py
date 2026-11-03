from datetime import UTC, datetime

import numpy as np
import pytest

from infosys.newsd import PassLog, ReceivedSignal
from infosys.replay import ReplayPlan
from infosys.score import MAX_SEND_LAG_P90_MS, MAX_SEND_LAG_P99_MS, score_pass
from infosys.store import Bars, Story

P = datetime(2024, 3, 11, 14, 0, tzinfo=UTC)  # 10:00 EDT


def _plan(stories: list[Story], available: list[float], warmup: tuple[Story, ...] = ()) -> ReplayPlan:
    stamps = {s.story_id: a for s, a in zip([*warmup, *stories], [*([0.0] * len(warmup)), *available], strict=True)}
    return ReplayPlan((), tuple(stories), frozenset(s.story_id for s in warmup), stamps, stamps, max(stamps.values(), default=0.0))


def _log(signals: list[ReceivedSignal], end_s: float = 100.0) -> PassLog:
    return PassLog(0.0, signals, {}, 0, [1.0, 2.0], end_s)


def _sig(story_id, ticker="AAA", received=1.0, direction=1, confidence=1.0) -> ReceivedSignal:
    return ReceivedSignal(story_id, ticker, direction, confidence, received)


def _story(i: int, tickers=("AAA",)) -> Story:
    return Story(f"s{i}", "benzinga", P, tickers, "h", "b")


def _bars(returns: dict[str, float]) -> Bars:
    """One start bar before P and one ending at P + 5 min per ticker, with the given forward return."""
    t0 = int(P.timestamp())
    times = np.array([t0 - 60, t0 + 240], np.int64)
    return Bars({k: (times, np.array([100.0, 100 * (1 + r)]), np.array([100.0, 100 * (1 + r)])) for k, r in returns.items()})


def _ranked(n: int, reverse=False):
    stories = [_story(i, (f"T{i:02d}",)) for i in range(n)]
    rets = {f"T{i:02d}": (n - i if reverse else i) / 1000 for i in range(n)}
    signals = [_sig(f"s{i}", f"T{i:02d}", confidence=(i + 1) / n) for i in range(n)]
    return _plan(stories, [1.0] * n), _log(signals), _bars(rets)


def test_latency_missed_duplicates():
    s1, s2 = _story(1), _story(2, ("AAA", "BBB"))
    plan = _plan([s1, s2], [1.0, 2.0])
    log = _log([_sig("s1", received=1.25), _sig("s2", received=2.5), _sig("s2", received=2.6)], end_s=5.0)
    score = score_pass(plan, log, _bars({}), horizon_min=5)
    expected = [250, 500, 3000]
    assert (score.pairs, score.signaled, score.missed, score.duplicates) == (3, 2, 1, 1)
    assert score.latency_ms["max"] == 3000
    assert score.latency_ms["p99"] == np.percentile(expected, 99)
    assert score.latency_ms["p50"] == np.percentile(expected, 50)
    assert score.throughput_per_s == 2 / (2.5 - 1.0)
    assert score.send_lag_ms == pytest.approx({"p50": 1.5, "p90": 1.9, "p99": 1.99, "max": 2.0})


def test_invalid_counts_rejections_and_schema_errors():
    log = PassLog(0.0, [], {"unknown_ticker": 2, "not_yet_available": 1}, 4, [], 1.0)
    assert score_pass(_plan([_story(1)], [1.0]), log, _bars({}), horizon_min=5).invalid == 7


def test_ic_perfect_rank():
    score = score_pass(*_ranked(40), horizon_min=5)
    assert score.ic == pytest.approx(1.0) and score.hit_rate == 1.0 and score.scored_pairs == 40
    plan, log, bars = _ranked(40, reverse=True)
    assert score_pass(plan, log, bars, horizon_min=5).ic == pytest.approx(-1.0)


def test_ic_none_below_30():
    score = score_pass(*_ranked(29), horizon_min=5)
    assert score.ic is None and score.ic_se is None and score.scored_pairs == 29


def test_ic_none_when_signals_are_constant():
    plan, log, bars = _ranked(40)
    flat = _log([_sig(f"s{i}", f"T{i:02d}", confidence=0.5) for i in range(40)])
    assert score_pass(plan, flat, bars, horizon_min=5).ic is None


def test_bootstrap_deterministic():
    rng = np.random.default_rng(0)
    n = 60
    stories = [_story(i, (f"T{i:02d}",)) for i in range(n)]
    bars = _bars({f"T{i:02d}": float(rng.normal(0, 0.01)) for i in range(n)})
    signals = [_sig(f"s{i}", f"T{i:02d}", confidence=float(rng.uniform(0.1, 1))) for i in range(n)]
    plan, log = _plan(stories, [1.0] * n), _log(signals)
    a = score_pass(plan, log, bars, horizon_min=5, bootstrap=200, bootstrap_seed=5)
    b = score_pass(plan, log, bars, horizon_min=5, bootstrap=200, bootstrap_seed=5)
    assert a.ic_se == b.ic_se and a.ic_se > 0
    assert score_pass(plan, log, bars, horizon_min=5, bootstrap=0).ic_se is None


def test_warmup_excluded():
    warm = Story("w0", "benzinga", P, ("AAA",), "h", "b")
    plan = _plan([_story(1)], [1.0], warmup=(warm,))
    log = _log([_sig("w0", received=0.5), _sig("w0", received=0.6), _sig("s1", received=1.5)])
    score = score_pass(plan, log, _bars({}), horizon_min=5)
    assert (score.stories, score.pairs, score.signaled, score.duplicates) == (1, 1, 1, 0)
    assert score.latency_ms["max"] == 500


def test_latency_runs_from_actual_send():
    plan = _plan([_story(1)], [1.0])
    log = _log([_sig("s1", received=1.25)])
    log.sent_s["s1"] = 1.2
    assert score_pass(plan, log, _bars({}), horizon_min=5).latency_ms["p50"] == pytest.approx(50)


def test_no_pairs_reports_null_percentiles():
    score = score_pass(_plan([], []), _log([]), _bars({}), horizon_min=5)
    assert score.latency_ms == {"p50": None, "p90": None, "p99": None, "max": None}


def _lagged(fractions: dict[float, float], n: int = 1000):
    """Score a pass whose deliveries are `fraction` late by `ms`, the rest 0.1 ms late."""
    lag = [0.1] * n
    at = 0
    for ms, frac in fractions.items():
        k = round(n * frac)
        lag[at : at + k] = [ms] * k
        at += k
    plan = _plan([_story(1)], [1.0])
    return score_pass(plan, PassLog(0.0, [_sig("s1")], {}, 0, lag, 100.0), _bars({}), horizon_min=5)


def test_pass_valid_without_lag():
    ok = score_pass(_plan([_story(1)], [1.0]), _log([_sig("s1")]), _bars({}), horizon_min=5)
    assert ok.valid and ok.invalid_reason is None


def test_rare_stalls_do_not_invalidate():
    score = _lagged({20.0: 0.02})  # 2% of deliveries 20 ms late: p90 small, p99 20 ms
    assert score.valid and score.send_lag_ms["p99"] == 20.0 and score.send_lag_ms["max"] == 20.0


def test_systematic_lateness_is_invalid_on_p90():
    score = _lagged({8.0: 0.15})
    assert not score.valid and "p90 8.0 ms exceeds 5.0" in score.invalid_reason and "p99" not in score.invalid_reason


def test_long_tail_is_invalid_on_p99():
    score = _lagged({60.0: 0.02})
    assert not score.valid and "p99 60.0 ms exceeds 50.0" in score.invalid_reason and "p90" not in score.invalid_reason


def test_bounds_are_inclusive_and_both_named():
    assert _lagged({MAX_SEND_LAG_P99_MS: 0.02}).valid
    both = _lagged({60.0: 0.2})
    assert "p90" in both.invalid_reason and "p99" in both.invalid_reason
    assert MAX_SEND_LAG_P90_MS == 5.0


def test_send_lag_fields_are_in_the_score():
    fields = _lagged({}).send_lag_ms
    assert set(fields) == {"p50", "p90", "p99", "max"}
