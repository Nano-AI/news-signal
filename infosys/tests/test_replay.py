from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from infosys.replay import FEEDS, burst_windows, compress, plan_pass, sample_stories
from infosys.store import Story

EPOCH = datetime(2024, 3, 11, 14, 0, tzinfo=UTC)  # 14:00 is aligned to 300 s


def _story(i: int, offset_s: float, source="benzinga", tickers=("AAA",)) -> Story:
    return Story(f"s{i:03d}", source, EPOCH + timedelta(seconds=offset_s), tickers, f"h{i}", f"b{i}")


def _stories(offsets: list[float]) -> list[Story]:
    return [_story(i, o) for i, o in enumerate(offsets)]


def test_compress_caps_gaps():
    assert compress(_stories([0, 0, 1, 10, 10.5]), 2.0) == [0, 0, 1, 3, 3.5]


def test_burst_windows_budget():
    a = [_story(i, i) for i in range(5)]  # window at 0 s: duration 4
    b = [_story(10 + i, 600 + i) for i in range(3)]  # duration 2
    c = [_story(20 + i, 1200 + 0.5 * i) for i in range(8)]  # duration 3.5
    got = burst_windows([*c, *a, *b], window_s=300, budget_s=10, gap_cap_s=2.0)
    assert got == a + c


def test_sample_stories_is_seeded_and_ordered():
    stories = _stories(list(range(50)))
    got = sample_stories(stories, 10, seed=3)
    assert got == sample_stories(stories, 10, seed=3)
    assert len(got) == 10 and got == sorted(got, key=lambda s: s.published_at)
    assert sample_stories(stories, 50, seed=3) == stories


story_specs = st.lists(
    st.tuples(
        st.sampled_from([0, 0, 0, 1, 1, 2, 5, 30, 500]),  # gaps, many same-second clusters
        st.sampled_from(["benzinga", "edgar"]),
        st.integers(1, 3),
    ),
    min_size=1,
    max_size=40,
)


@given(specs=story_specs, seed=st.integers(0, 2**32), warm=st.integers(0, 5), cap=st.sampled_from([0.0, 0.5, 2.0]))
def test_plan_invariants(specs, seed, warm, cap):
    offset, stories = 0, []
    for i, (gap, source, k) in enumerate(specs):
        offset += gap
        stories.append(_story(i, offset, source, ("AAA", "BBB", "CCC")[:k]))
    warm = min(warm, len(stories))
    plan = plan_pass(stories[warm:], stories[:warm], gap_cap_s=cap, seed=seed)
    again = plan_pass(stories[warm:], stories[:warm], gap_cap_s=cap, seed=seed)

    assert plan == again
    by_id = {s.story_id: s for s in stories}
    assert plan.warmup_ids == {s.story_id for s in stories[:warm]}
    assert [d.at_s for d in plan.deliveries] == sorted(d.at_s for d in plan.deliveries)
    seqs: dict[str, list[int]] = {}
    for d in plan.deliveries:
        seqs.setdefault(d.feed, []).append(d.seq)
        assert d.story == by_id[d.story.story_id]
        assert d.at_s <= plan.release_s[d.story.story_id] + 2.5
    assert all(s == list(range(1, len(s) + 1)) for s in seqs.values())
    for story in stories:
        real = [d for d in plan.deliveries if d.feed == story.source and d.story is story]
        assert len(real) == 1  # real feeds never drop or duplicate
        assert plan.available_s[story.story_id] >= plan.release_s[story.story_id]
        assert plan.available_s[story.story_id] == min(d.at_s for d in plan.deliveries if d.story is story)
    assert plan.last_s == max(d.at_s for d in plan.deliveries)


def test_same_second_burst():
    stories = [_story(i, 0) for i in range(50)]
    plan = plan_pass(stories, [], gap_cap_s=2.0, seed=7)
    assert len({plan.release_s[s.story_id] for s in stories}) == 1
    real = [d for d in plan.deliveries if d.feed == "benzinga"]
    assert sorted(d.seq for d in real) == list(range(1, 51))
    assert sorted(d.story.story_id for d in real) == [s.story_id for s in stories]  # none dropped
    assert plan == plan_pass(list(reversed(stories)), [], gap_cap_s=2.0, seed=7)  # input order is irrelevant


def test_synthetic_feeds_drop_and_duplicate_but_real_feeds_do_not():
    stories = [_story(i, i, "edgar") for i in range(400)]
    plan = plan_pass(stories, [], gap_cap_s=1.0, seed=1)
    counts = {f.name: sum(d.feed == f.name for d in plan.deliveries) for f in FEEDS}
    assert counts["edgar"] == 400 and counts["benzinga"] == 0
    assert counts["syn-2"] < 400 and counts["syn-3"] > 380
