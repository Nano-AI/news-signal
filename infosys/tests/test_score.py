from datetime import UTC, datetime

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from infosys.newsd import PassLog, ReceivedSignal
from infosys.replay import ReplayPlan
from infosys.score import MAX_SEND_LAG_P90_MS, MAX_SEND_LAG_P99_MS, fidelity, score_pass
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


def _feed_log(signals, feed_sent, sent_s=None, end_s=100.0) -> PassLog:
    return PassLog(0.0, signals, {}, 0, [1.0], end_s, sent_s or {}, feed_sent)


def test_per_feed_latency_and_first_counts():
    s1, s2 = _story(1), _story(2, ("AAA", "BBB"))
    plan = _plan([s1, s2], [1.0, 2.0])
    feed_sent = {"fast": {"s1": 1.0, "s2": 2.0}, "slow": {"s1": 1.5, "s2": 1.75}, "idle": {}}
    log = _feed_log([_sig("s1", received=1.25), _sig("s2", received=2.5)], feed_sent)  # (s2, BBB) is missed
    feeds = score_pass(plan, log, _bars({}), horizon_min=5).feed_latency_ms
    assert feeds["fast"] == {"pairs": 2, "p50": np.percentile([250, 500], 50), "p99": np.percentile([250, 500], 99), "first": 1}
    # slow delivered s1 after the signal was already in: negative, and it was first for s2
    assert feeds["slow"] == {"pairs": 2, "p50": np.percentile([-250, 750], 50), "p99": np.percentile([-250, 750], 99), "first": 1}
    assert feeds["idle"] == {"pairs": 0, "p50": None, "p99": None, "first": 0}


def test_per_feed_first_ties_go_to_the_first_name_and_warmup_is_ignored():
    warm, s1 = _story(0), _story(1)
    plan = _plan([s1], [1.0], warmup=(warm,))
    feed_sent = {"b": {"s1": 1.0, "s0": 0.1}, "a": {"s1": 1.0, "s0": 0.1}}
    feeds = score_pass(plan, _feed_log([_sig("s1", received=2.0), _sig("s0", received=0.5)], feed_sent), _bars({}), horizon_min=5).feed_latency_ms
    assert (feeds["a"]["first"], feeds["b"]["first"], feeds["a"]["pairs"]) == (1, 0, 1)


@given(st.lists(st.tuples(st.floats(0, 50), st.floats(0, 50)), min_size=1, max_size=20))
def test_per_feed_first_counts_sum_to_the_stories_sent(offsets):
    stories = [_story(i) for i in range(len(offsets))]
    plan = _plan(stories, [0.0] * len(stories))
    feed_sent = {"x": {f"s{i}": a for i, (a, _) in enumerate(offsets)}, "y": {f"s{i}": b for i, (_, b) in enumerate(offsets)}}
    log = _feed_log([_sig(f"s{i}", received=100.0) for i in range(len(stories))], feed_sent)
    feeds = score_pass(plan, log, _bars({}), horizon_min=5).feed_latency_ms
    assert feeds["x"]["first"] + feeds["y"]["first"] == len(stories)
    assert feeds["x"]["pairs"] == feeds["y"]["pairs"] == len(stories)
    assert feeds["x"]["first"] == sum(a <= b for a, b in offsets)


# -- fidelity to a reference signal set ------------------------------------------------------------------


def _fid_case(n: int, flips: int = 0, skip: int = 0, scale: float = 1.0):
    """n pairs whose reference signal is a strictly increasing ramp over both signs; the candidate repeats it, except
    `flips` pairs with the opposite direction and `skip` pairs with no signal at all."""
    stories = [_story(i, (f"T{i:02d}",)) for i in range(n)]
    ref = {(f"s{i}", f"T{i:02d}"): (i - n / 2 + 0.25) / n for i in range(n)}  # never exactly 0
    signals = []
    for i in range(skip, n):
        r = ref[(f"s{i}", f"T{i:02d}")]
        direction = (1 if r > 0 else -1) * (-1 if skip <= i < skip + flips else 1)
        signals.append(_sig(f"s{i}", f"T{i:02d}", direction=direction, confidence=abs(r) * scale))
    return _plan(stories, [1.0] * n), _log(signals), ref


def test_fidelity_is_perfect_for_the_reference_itself():
    f = fidelity(*_fid_case(50))
    assert (f.pairs, f.agreed, f.agreement, f.both) == (50, 50, 1.0, 50) and f.spearman == pytest.approx(1.0)


def test_fidelity_counts_flips_and_missed_pairs_as_disagreement():
    f = fidelity(*_fid_case(100, flips=3, skip=0))
    assert f.agreement == 0.97 and f.both == 100
    f = fidelity(*_fid_case(100, skip=2))
    assert (f.agreed, f.pairs, f.both) == (98, 100, 98) and f.agreement == 0.98


@given(n=st.integers(40, 120), data=st.data())
def test_fidelity_agreement_is_the_agreeing_fraction(n, data):
    flips = data.draw(st.integers(0, n // 2))
    skip = data.draw(st.integers(0, n // 4))
    f = fidelity(*_fid_case(n, flips=flips, skip=skip))
    assert f.agreement == (n - skip - flips) / n and f.pairs == n


def test_fidelity_spearman_ignores_a_monotone_rescale_and_sees_a_constant_confidence():
    assert fidelity(*_fid_case(60, scale=0.5)).spearman == pytest.approx(1.0)
    plan, log, ref = _fid_case(60)
    constant = _log([_sig(g.story_id, g.ticker, direction=g.direction, confidence=0.9) for g in log.signals])
    f = fidelity(plan, constant, ref)
    assert f.agreement == 1.0 and f.spearman is not None and f.spearman < 0.95


def test_fidelity_neutral_agrees_only_with_neutral():
    stories = [_story(i, (f"T{i:02d}",)) for i in range(40)]
    ref = {(f"s{i}", f"T{i:02d}"): 0.0 if i < 10 else 0.5 + i / 100 for i in range(40)}
    signals = [_sig(f"s{i}", f"T{i:02d}", direction=0 if i < 9 else 1, confidence=0.9) for i in range(40)]
    f = fidelity(_plan(stories, [1.0] * 40), _log(signals), ref)
    assert f.agreed == 39  # pair 9 is a reference neutral that the candidate called positive
    zero_conf = [_sig(f"s{i}", f"T{i:02d}", direction=1, confidence=0.0 if i < 10 else 0.9) for i in range(40)]
    assert fidelity(_plan(stories, [1.0] * 40), _log(zero_conf), ref).agreed == 40  # confidence 0 is neutral


def test_fidelity_ignores_duplicates_and_unmeasured_signals_and_refuses_an_uncovered_pair():
    plan, log, ref = _fid_case(40)
    extra = _log([*log.signals, _sig("s0", direction=-1, received=5.0), _sig("nope", "ZZZ")])
    assert fidelity(plan, extra, ref).agreement == 1.0  # the later duplicate and the foreign signal do not count
    with pytest.raises(ValueError, match="not in the reference"):
        fidelity(plan, log, {k: v for k, v in ref.items() if k != ("s3", "T03")})
