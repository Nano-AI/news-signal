import json
from datetime import date
from pathlib import Path

import pytest

from infosys.fetch import alpaca, edgar
from infosys.fetch.http import CachedClient, FetchError

SENTINEL = "SECRET-SENTINEL-123"
URL = "https://example.test/api"


class FakeTransport:
    """Serves queued (status, body) answers per call and records every call."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        status, body = self.answers.pop(0) if self.answers else (200, {"ok": True})
        return status, json.dumps(body).encode()


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def _client(tmp_path, transport, time=None, **kw):
    time = time or FakeTime()
    return CachedClient(tmp_path / "cache", transport=transport, sleep=time.sleep, clock=time.clock, **kw)


def test_secret_never_written(tmp_path, capsys):
    headers = {"APCA-API-KEY-ID": SENTINEL, "APCA-API-SECRET-KEY": SENTINEL}
    client = _client(tmp_path, FakeTransport([(200, {"a": 1}), (404, {})]))
    client.get_json(URL, {"p": "1"}, headers)
    client.get_json(URL, {"p": "1"}, headers)
    with pytest.raises(FetchError) as err:
        client.get_json(URL, {"p": "2"}, headers)
    assert SENTINEL not in str(err.value)
    for f in (tmp_path / "cache").rglob("*"):
        if f.is_file():
            assert SENTINEL not in f.read_text()
    out = capsys.readouterr()
    assert SENTINEL not in out.out + out.err


def test_cache_hit_skips_network(tmp_path):
    t = FakeTransport([(200, {"a": 1})])
    client = _client(tmp_path, t)
    assert client.get_json(URL, {"b": "2", "a": "1"}, {}) == {"a": 1}
    assert client.get_json(URL, {"a": "1", "b": "2"}, {"X": "other"}) == {"a": 1}  # order, headers irrelevant
    assert client.network_requests == 1 and len(t.calls) == 1


def test_rate_limit_spacing(tmp_path):
    time = FakeTime()
    starts: list[float] = []

    def transport(url, headers):
        starts.append(time.now)
        return 200, b"{}"

    client = _client(tmp_path, transport, time)
    for i in range(5):
        client.get_json(URL, {"i": str(i)}, {})
    assert all(b - a >= 0.4 - 1e-9 for a, b in zip(starts, starts[1:], strict=False))


def test_retry_on_429_then_ok(tmp_path):
    time = FakeTime()
    t = FakeTransport([(429, {}), (503, {}), (200, {"v": 1})])
    assert _client(tmp_path, t, time).get_json(URL, {}, {}) == {"v": 1}
    assert len(t.calls) == 3
    assert [s for s in time.sleeps if s >= 2] == [2, 4]


def test_retry_exhaustion_raises(tmp_path):
    time = FakeTime()
    t = FakeTransport([(500, {})] * 6)
    with pytest.raises(FetchError):
        _client(tmp_path, t, time).get_json(URL, {}, {})
    assert [s for s in time.sleeps if s >= 2] == [2, 4, 8, 16, 32]
    assert len(t.calls) == 6


def test_4xx_raises_without_headers(tmp_path):
    t = FakeTransport([(403, {})])
    with pytest.raises(FetchError) as err:
        _client(tmp_path, t).get_json(URL, {"q": "1"}, {"APCA-API-KEY-ID": SENTINEL})
    assert err.value.status == 403
    assert SENTINEL not in str(err.value) and "APCA" not in str(err.value)
    assert len(t.calls) == 1


def test_partial_tmp_file_ignored(tmp_path):
    t = FakeTransport([(200, {"fresh": True})])
    client = _client(tmp_path, t)
    client.get_json(URL, {"x": "1"}, {})
    (record,) = (tmp_path / "cache").rglob("*.json")
    fresh = record.read_text()
    record.unlink()
    record.with_suffix(".json.tmp").write_text('{"url": "half-writ')  # a crash mid-write
    t2 = FakeTransport([(200, {"fresh": True})])
    assert _client(tmp_path, t2).get_json(URL, {"x": "1"}, {}) == {"fresh": True}
    assert len(t2.calls) == 1 and record.read_text() == fresh
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def _news_pages(n):
    return [
        (200, {"news": [{"id": i}], "next_page_token": f"tok{i}" if i < n else None}) for i in range(1, n + 1)
    ]


def test_resume_replays_page_chain(tmp_path):
    day = date(2023, 5, 1)
    first = FakeTransport(_news_pages(3))
    c1 = _client(tmp_path, first)
    assert alpaca.fetch_news(c1, day, date(2023, 6, 1), headers={}) == 3
    assert [c[0].count("page_token") for c in first.calls] == [0, 1, 1]
    second = FakeTransport()
    c2 = _client(tmp_path, second)
    assert alpaca.fetch_news(c2, day, date(2023, 6, 1), headers={}) == 3
    assert c2.network_requests == 0 and c2.cache_hits == 3 and not second.calls


def test_interrupted_chain_resumes_from_cache(tmp_path):
    t = FakeTransport(_news_pages(3)[:2] + [(500, {})] * 6)
    with pytest.raises(FetchError):
        alpaca.fetch_news(_client(tmp_path, t), date(2023, 5, 1), date(2023, 6, 1), headers={})
    rest = FakeTransport(_news_pages(3)[2:])
    c = _client(tmp_path, rest)
    assert alpaca.fetch_news(c, date(2023, 5, 1), date(2023, 6, 1), headers={}) == 3
    assert c.cache_hits == 2 and len(rest.calls) == 1


def test_news_and_bars_queries(tmp_path):
    t = FakeTransport()
    c = _client(tmp_path, t)
    alpaca.fetch_news(c, date(2023, 12, 15), date(2024, 2, 10), headers={"K": "v"})
    assert len(t.calls) == 3  # Dec, Jan, Feb cut at month edges
    url, headers = t.calls[0]
    assert "start=2023-12-15T00%3A00%3A00Z" in url and "end=2024-01-01T00%3A00%3A00Z" in url
    assert "limit=50" in url and "sort=asc" in url and headers == {"K": "v"}
    assert "end=2024-02-10T00%3A00%3A00Z" in t.calls[2][0]
    t.calls.clear()
    alpaca.fetch_bars(c, "AAPL", date(2023, 1, 1), date(2023, 2, 1), headers={})
    assert "feed=iex" in t.calls[0][0] and "timeframe=1Min" in t.calls[0][0] and "symbols=AAPL" in t.calls[0][0]


def test_load_keys(tmp_path):
    env = tmp_path / "e.env"
    env.write_text(f"ALPACA_API_KEY=abc&def\nALPACA_SECRET_KEY={SENTINEL}\n")
    assert alpaca.load_keys(env) == {"APCA-API-KEY-ID": "abc&def", "APCA-API-SECRET-KEY": SENTINEL}
    env.write_text("ALPACA_API_KEY=abc\n")
    with pytest.raises(ValueError, match="ALPACA_SECRET_KEY") as err:
        alpaca.load_keys(env)
    assert "abc" not in str(err.value)


def test_edgar_user_agent_has_contact_and_is_not_cached(tmp_path):
    answers = {
        "company_tickers": {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}},
        "CIK0000320193.json": {
            "name": "Apple Inc.",
            "filings": {
                "recent": {},
                "files": [
                    {"name": "CIK0000320193-submissions-001.json", "filingFrom": "1994-01-01", "filingTo": "2015-01-01"},
                    {"name": "CIK0000320193-submissions-002.json", "filingFrom": "2016-01-01", "filingTo": "2019-12-31"},
                ],
            },
        },
    }

    def transport(url, headers):
        calls.append((url, dict(headers)))
        key = next(k for k in answers if k in url)
        return 200, json.dumps(answers[key]).encode()

    calls: list[tuple[str, dict]] = []
    c = _client(tmp_path, transport)
    n = edgar.fetch_8k(c, ["AAPL"], date(2020, 1, 1), date(2024, 1, 1), "me@example.test")
    assert n == 2  # tickers map + main file; both history files end before the window
    assert all(h["User-Agent"] == "news-signal research me@example.test" for _, h in calls)
    assert all("me@example.test" not in f.read_text() for f in (tmp_path / "cache").rglob("*.json"))
    assert calls[1][0] == "https://data.sec.gov/submissions/CIK0000320193.json"


def test_contact_email_override():
    assert edgar.contact_email("a@b.test") == "a@b.test"
