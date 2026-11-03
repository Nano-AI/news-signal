from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import json

from infosys.fixture import TEMPLATES, TICKERS, build_fixture
from infosys.store import SPLITS, load_split, verify_store

CHECKED_IN = Path(__file__).resolve().parents[1] / "fixtures" / "tiny"
COUNTS = {"train": 60, "validation": 120, "holdout": 60}


def test_fixture_regenerates_identically(tmp_path):
    build_fixture(tmp_path)
    for split in SPLITS:
        stories_a, bars_a = load_split(tmp_path, split)
        stories_b, bars_b = load_split(CHECKED_IN, split)
        assert stories_a == stories_b
        assert bars_a.tickers() == bars_b.tickers()
        for ticker in bars_a.tickers():
            assert all(np.array_equal(x, y) for x, y in zip(bars_a.arrays(ticker), bars_b.arrays(ticker)))


def test_fixture_shape():
    everything = []
    for split in SPLITS:
        stories, bars = load_split(CHECKED_IN, split)
        assert len(stories) == COUNTS[split]
        assert bars.tickers() == sorted(TICKERS)
        everything += stories
    ny = ZoneInfo("America/New_York")
    assert max(sum(s.published_at == p for s in everything) for p in {s.published_at for s in everything}) >= 5
    assert sum(len(s.tickers) > 1 for s in everything) / len(everything) >= 0.15
    assert all(len(s.tickers) <= 3 for s in everything)
    outside = [s for s in everything if not time(9, 31) <= s.published_at.astimezone(ny).time() <= time(16, 0)]
    assert len(outside) >= 10
    assert 0.15 <= sum(s.source == "edgar" for s in everything) / len(everything) <= 0.25
    assert sum(p.stat().st_size for p in CHECKED_IN.iterdir()) < 1_000_000


def test_checked_in_manifest_matches_files():
    manifest = json.loads((CHECKED_IN / "manifest.json").read_text())
    assert manifest["tickers"] == list(TICKERS)
    verify_store(CHECKED_IN, manifest, SPLITS)


def test_checked_in_fixture_is_synthetic():
    templates = {t for group in TEMPLATES.values() for t in group}
    for split in SPLITS:
        stories, _ = load_split(CHECKED_IN, split)
        for s in stories:
            assert set(s.tickers) <= set(TICKERS)
            assert any(s.headline == t.format(t=s.tickers[0]) for t in templates), s.story_id
            assert s.summary == f"{s.headline}. Synthetic summary {int(s.story_id[3:])}."
