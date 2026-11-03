import json
from datetime import UTC, date, datetime
from pathlib import Path

from infosys.build import build
from infosys.fetch.alpaca import BARS_URL, NEWS_URL
from infosys.fetch.edgar import SUBMISSIONS_URL, TICKERS_URL
from infosys.fetch.http import CachedClient
from infosys.store import load_split, read_stories

END = date(2024, 1, 1)


def _write_cache(cache: Path, records: list[tuple[str, dict, dict]]) -> None:
    """Fill `cache` through the real client, with a transport that serves each body in turn."""
    queue = [json.dumps(body).encode() for _, _, body in records]
    client = CachedClient(cache, min_interval_s=0, transport=lambda url, headers: (200, queue.pop(0)))
    for url, params, _ in records:
        client.get_json(url, params, {})


def _item(id_, created, symbols, headline="h", summary="s"):
    return {"id": id_, "created_at": created, "headline": headline, "summary": summary, "symbols": symbols}


def _news(*items, page=0):
    return (NEWS_URL, {"page": str(page)}, {"news": list(items), "next_page_token": None})


def _stories(out: Path, split: str):
    return {s.story_id: s for s in read_stories(out / f"stories-{split}.jsonl.gz")}


def _run(tmp_path, records):
    _write_cache(tmp_path / "cache", records)
    return build(tmp_path / "cache", tmp_path / "out", END), tmp_path / "out"


def test_filters_and_dropped_counts(tmp_path):
    manifest, out = _run(
        tmp_path,
        [
            _news(
                _item(1, "2023-01-02T15:00:00Z", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]),  # 6: roundup
                _item(2, "2023-01-02T15:00:01Z", ["AAPL", "ZZZZ"]),
                _item(3, "2023-01-02T15:00:02Z", ["ZZZZ"]),
                _item(4, "2023-01-02T15:00:03Z", ["MSFT"]),
                _item(4, "2023-01-02T15:00:03Z", ["MSFT"]),  # same id again
                page=0,
            ),
            _news(_item(4, "2023-01-02T15:00:03Z", ["MSFT"]), page=1),  # and across pages
        ],
    )
    got = _stories(out, "train")
    assert {k: v.tickers for k, v in got.items()} == {"bz:2": ("AAPL",), "bz:4": ("MSFT",)}
    assert manifest["dropped"] == {"roundups": 1, "no_universe_ticker": 1, "duplicates": 2}
    assert manifest["universe"] == manifest["tickers"] and manifest["window_end"] == "2024-01-01"


def test_split_boundary(tmp_path):
    _, out = _run(
        tmp_path,
        [_news(_item(1, "2023-06-30T23:59:59Z", ["AAPL"]), _item(2, "2023-07-01T00:00:00Z", ["AAPL"]))],
    )
    assert set(_stories(out, "train")) == {"bz:1"}
    assert set(_stories(out, "validation")) == {"bz:2"}
    assert not _stories(out, "holdout")


def test_summary_cleaned(tmp_path):
    _, out = _run(tmp_path, [_news(_item(1, "2023-01-03T15:00:00Z", ["AAPL"], "A &amp; B", "<p>x &amp; y</p>"))])
    (story,) = _stories(out, "train").values()
    assert (story.summary, story.headline, story.source) == ("x & y", "A & B", "benzinga")


def _edgar_records(recent):
    cik = "CIK0000320193"
    return [
        (TICKERS_URL, {}, {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}, "1": {"cik_str": 1, "ticker": "OTHER"}}),
        (SUBMISSIONS_URL + f"{cik}.json", {}, {"name": "Apple Inc.", "filings": {"recent": recent, "files": []}}),
    ]


def test_edgar_items_and_amendments(tmp_path):
    recent = {
        "accessionNumber": ["a-1", "a-2", "a-3"],
        "form": ["8-K", "8-K/A", "8-K"],
        "acceptanceDateTime": ["2023-01-05T21:30:00.000Z"] * 3,
        "items": ["2.02,9.01", "2.02", "9.99"],
    }
    _, out = _run(tmp_path, _edgar_records(recent))
    got = _stories(out, "train")
    assert set(got) == {"edgar:a-1", "edgar:a-3"}
    assert got["edgar:a-1"].headline == (
        "Apple Inc. 8-K: Results of Operations and Financial Condition; Financial Statements and Exhibits"
    )
    assert got["edgar:a-3"].headline == "Apple Inc. 8-K: Item 9.99"
    assert (got["edgar:a-1"].source, got["edgar:a-1"].tickers, got["edgar:a-1"].summary) == ("edgar", ("AAPL",), "")


def test_edgar_history_file_and_window(tmp_path):
    recent = {"accessionNumber": [], "form": [], "acceptanceDateTime": [], "items": []}
    history = {
        "accessionNumber": ["old", "new"],
        "form": ["8-K", "8-K"],
        "acceptanceDateTime": ["2019-12-31T12:00:00.000Z", "2023-01-05T21:30:00.000Z"],
        "items": ["8.01", "8.01"],
    }
    records = _edgar_records(recent) + [(SUBMISSIONS_URL + "CIK0000320193-submissions-001.json", {}, history)]
    _, out = _run(tmp_path, records)
    assert set(_stories(out, "train")) == {"edgar:new"}  # the 2019 filing is before the window


def test_bars_split_deduped_and_manifest(tmp_path):
    def bar(t, p):
        return {"t": t, "o": p, "h": p, "l": p, "c": p, "v": 1}

    june = [bar("2023-06-30T23:59:00Z", 10.0), bar("2023-07-01T00:00:00Z", 11.0)]
    july = [bar("2023-07-01T00:00:00Z", 11.0), bar("2023-07-01T00:01:00Z", 0.0)]  # repeat + bad price
    records = [
        (BARS_URL, {"m": "6"}, {"bars": {"AAPL": june}, "next_page_token": None}),
        (BARS_URL, {"m": "7"}, {"bars": {"AAPL": july}, "next_page_token": None}),
        (BARS_URL, {"m": "8"}, {"bars": {}, "next_page_token": None}),
    ]
    manifest, out = _run(tmp_path, records)
    train = load_split(out, "train")[1].arrays("AAPL")
    val = load_split(out, "validation")[1].arrays("AAPL")
    assert train[0].tolist() == [int(datetime(2023, 6, 30, 23, 59, tzinfo=UTC).timestamp())]
    assert val[0].tolist() == [int(datetime(2023, 7, 1, tzinfo=UTC).timestamp())] and val[2].tolist() == [11.0]
    assert manifest["files"]["bars-validation.npz"]["rows"] == 1
    assert load_split(out, "holdout")[1].tickers() == load_split(out, "train")[1].tickers() and len(manifest["tickers"]) == 25


def test_tmp_files_in_cache_are_ignored(tmp_path):
    _write_cache(tmp_path / "cache", [_news(_item(1, "2023-01-03T15:00:00Z", ["AAPL"]))])
    junk = next((tmp_path / "cache").rglob("*.json")).with_suffix(".json.tmp")
    junk.write_text("{half")
    build(tmp_path / "cache", tmp_path / "out", END)
