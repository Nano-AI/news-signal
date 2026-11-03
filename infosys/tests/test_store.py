from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from infosys.store import (
    Bars,
    Story,
    StoreMismatch,
    read_bars,
    read_stories,
    sha256_bytes,
    sha256_file,
    split_of,
    store_files,
    verify_store,
    write_bars,
    write_manifest,
    write_stories,
)

START = datetime(2024, 3, 11, 13, 30, tzinfo=UTC)  # 09:30 EDT


def _utc(*a) -> datetime:
    return datetime(*a, tzinfo=UTC)


def _ramp(start: datetime, n: int = 390) -> Bars:
    t = np.array([int(start.timestamp()) + 60 * i for i in range(n)], np.int64)
    close = 100.0 + np.arange(n, dtype=np.float64)
    return Bars({"AAA": (t, close, close)})


stories_st = st.lists(
    st.builds(
        Story,
        story_id=st.text(min_size=1, max_size=8),
        source=st.sampled_from(["benzinga", "edgar"]),
        published_at=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 1, 1)).map(
            lambda d: d.replace(microsecond=0, tzinfo=UTC)
        ),
        tickers=st.lists(st.sampled_from(["AAA", "BBB"]), min_size=1, max_size=2, unique=True).map(tuple),
        headline=st.text(max_size=30),
        summary=st.text(max_size=30),
    ),
    unique_by=lambda s: s.story_id,
    max_size=8,
)


@given(stories_st)
def test_stories_roundtrip(tmp_path_factory, stories):
    path = tmp_path_factory.mktemp("s") / "stories.jsonl.gz"
    assert write_stories(path, stories) == len(stories)
    assert read_stories(path) == sorted(stories, key=lambda s: (s.published_at, s.story_id))
    again = path.with_name("again.jsonl.gz")
    write_stories(again, reversed(stories))
    assert sha256_file(path) == sha256_file(again)  # deterministic bytes, input order irrelevant


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (_utc(2023, 6, 30, 23, 59, 59), "train"),
        (_utc(2023, 7, 1), "validation"),
        (_utc(2024, 6, 30, 23, 59, 59), "validation"),
        (_utc(2024, 7, 1), "holdout"),
    ],
)
def test_split_boundaries(when, expected):
    assert split_of(when) == expected


def test_split_before_window_raises():
    with pytest.raises(ValueError):
        split_of(_utc(2019, 12, 31, 23, 59, 59))


def test_forward_return_edt():
    bars = _ramp(START)
    assert bars.forward_return("AAA", _utc(2024, 3, 11, 13, 35, 20), 5) == pytest.approx(109 / 104 - 1)


def test_forward_return_est_session():
    bars = _ramp(_utc(2024, 3, 8, 14, 30))  # 09:30 EST
    assert bars.forward_return("AAA", _utc(2024, 3, 8, 14, 31), 5) is not None
    assert bars.forward_return("AAA", _utc(2024, 3, 8, 13, 35), 5) is None  # 08:35 EST


def test_forward_return_past_close():
    bars = _ramp(START)
    assert bars.forward_return("AAA", _utc(2024, 3, 11, 19, 57), 5) is None  # 15:57 + 5 min > 16:00


def test_forward_return_gap_and_unknown_ticker():
    t = np.array([int(START.timestamp())], np.int64)
    bars = Bars({"AAA": (t, np.array([100.0]), np.array([100.0]))})
    assert bars.forward_return("AAA", START + timedelta(minutes=40), 5) is None  # >31 min since last bar
    assert bars.forward_return("ZZZ", START + timedelta(minutes=5), 5) is None


def test_close_before_uses_only_ended_bars():
    bars = _ramp(START)
    assert bars.close_before("AAA", START + timedelta(seconds=59)) is None
    assert bars.close_before("AAA", START + timedelta(seconds=60)) == 100.0


def test_bars_roundtrip_and_validation(tmp_path):
    t = np.array([60, 120, 180], np.int64)
    data = {"AAA": (t, np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))}
    assert write_bars(tmp_path / "b.npz", data) == 3
    back = read_bars(tmp_path / "b.npz")
    assert back.tickers() == ["AAA"]
    assert all(np.array_equal(a, b) for a, b in zip(back.arrays("AAA"), data["AAA"]))
    bad = {"AAA": (np.array([60, 60], np.int64), np.zeros(2), np.zeros(2))}
    with pytest.raises(ValueError):
        write_bars(tmp_path / "bad.npz", bad)


def _tiny_store(data_dir: Path) -> dict:
    story = Story("s1", "benzinga", _utc(2024, 3, 11, 14), ("AAA",), "h", "s")
    for split in ("train", "validation", "holdout"):
        stories_name, bars_name = store_files(split)
        write_stories(data_dir / stories_name, [story])
        write_bars(data_dir / bars_name, {"AAA": _ramp(START).arrays("AAA")})
    return write_manifest(data_dir, {"note": "x"})


def test_manifest_detects_change(tmp_path):
    manifest = _tiny_store(tmp_path)
    assert manifest["format"] == 1 and manifest["note"] == "x"
    assert manifest["files"]["bars-train.npz"]["rows"] == 390
    assert str(tmp_path) not in (tmp_path / "manifest.json").read_text()
    verify_store(tmp_path, manifest, ["train", "validation", "holdout"])
    path = tmp_path / "bars-validation.npz"
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    verify_store(tmp_path, manifest, ["train"])  # unrelated split still fine
    with pytest.raises(StoreMismatch) as err:
        verify_store(tmp_path, manifest, ["validation"])
    assert str(err.value) == "bars-validation.npz"
    path.unlink()
    with pytest.raises(StoreMismatch, match="bars-validation.npz"):
        verify_store(tmp_path, manifest, ["validation"])


def test_sha256_helpers_agree(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    assert sha256_bytes(b"abc") == sha256_file(tmp_path / "f")
    assert sha256_bytes(b"abc").startswith("ba7816bf")
