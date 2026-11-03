"""Alpaca news (Benzinga) and IEX minute bars, one query chain per calendar month."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

from infosys.fetch.http import CachedClient
from infosys.universe import TICKERS

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def load_keys(env_file: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    keys = {"ALPACA_API_KEY": "APCA-API-KEY-ID", "ALPACA_SECRET_KEY": "APCA-API-SECRET-KEY"}
    missing = [name for name in keys if not values.get(name)]
    if missing:
        raise ValueError(f"{env_file.name}: missing variable {', '.join(missing)}")
    return {header: values[name] for name, header in keys.items()}


def _months(start: date, end: date) -> Iterator[tuple[date, date]]:
    """[start, end) cut at calendar-month boundaries."""
    cur = start
    while cur < end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, min(nxt, end)
        cur = nxt


def _z(d: date) -> str:
    return datetime(d.year, d.month, d.day, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chain(client: CachedClient, url: str, params: dict[str, str], headers: dict[str, str]) -> int:
    pages = 0
    while True:
        body = client.get_json(url, params, headers)
        pages += 1
        token = body.get("next_page_token")
        if not token:
            return pages
        params = {**params, "page_token": token}


def fetch_news(client: CachedClient, start: date, end: date, *, headers: dict[str, str]) -> int:
    pages = 0
    for lo, hi in _months(start, end):
        params = {
            "symbols": ",".join(TICKERS),
            "start": _z(lo),
            "end": _z(hi),
            "limit": "50",
            "sort": "asc",
            "include_content": "false",
            "exclude_contentless": "false",
        }
        pages += _chain(client, NEWS_URL, params, headers)
    return pages


def fetch_bars(client: CachedClient, ticker: str, start: date, end: date, *, headers: dict[str, str]) -> int:
    pages = 0
    for lo, hi in _months(start, end):
        params = {
            "symbols": ticker,
            "timeframe": "1Min",
            "start": _z(lo),
            "end": _z(hi),
            "feed": "iex",
            "adjustment": "raw",
            "limit": "10000",
            "sort": "asc",
        }
        pages += _chain(client, BARS_URL, params, headers)
    return pages
