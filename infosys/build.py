"""Build the event store (stories, bars, manifest) from a fetch cache."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from infosys.fetch.alpaca import BARS_URL, NEWS_URL
from infosys.fetch.edgar import ITEM_TITLES, SUBMISSIONS_URL, TICKERS_URL, acceptance_utc
from infosys.store import (
    SPLIT_STARTS,
    SPLITS,
    BarArrays,
    Story,
    split_of,
    store_files,
    write_bars,
    write_manifest,
    write_stories,
)
from infosys.universe import MAX_TICKERS_PER_STORY, TICKERS

_TAG = re.compile(r"<[^>]*>")


def _records(cache: Path) -> tuple[list[dict], int]:
    """Every readable cache record in path order, and the number of corrupt ones skipped."""
    records, corrupt = [], 0
    for path in sorted(cache.glob("*/*.json")):  # `.json.tmp` files never match
        try:
            rec = json.loads(path.read_text())
            rec["url"], rec["body"]
        except (ValueError, KeyError, TypeError):
            corrupt += 1
            continue
        records.append(rec)
    return records, corrupt


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC).replace(microsecond=0)


def _clean_summary(text: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", text)).split())


def _eight_k_stories(filings: dict, ticker: str, name: str) -> Iterator[Story]:
    for acc, form, accepted, items in zip(
        filings["accessionNumber"], filings["form"], filings["acceptanceDateTime"], filings["items"], strict=True
    ):
        if form != "8-K":  # 8-K/A amends an earlier filing
            continue
        titles = [ITEM_TITLES.get(i.strip(), f"Item {i.strip()}") for i in items.split(",") if i.strip()]
        headline = f"{name} 8-K: {'; '.join(titles)}" if titles else f"{name} 8-K"
        yield Story(f"edgar:{acc}", "edgar", acceptance_utc(accepted), (ticker,), headline, "")


def _bar_columns(rows: list[dict]) -> BarArrays:
    return (
        np.array([int(_utc(r["t"]).timestamp()) for r in rows], np.int64),
        np.array([r["o"] for r in rows], np.float64),
        np.array([r["c"] for r in rows], np.float64),
    )


def _merge_bars(parts: list[BarArrays]) -> BarArrays:
    """Concatenate chunks, sort by time and drop repeated minutes (month edges overlap) and bad prices."""
    if not parts:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    t, o, c = (np.concatenate(col) for col in zip(*parts, strict=True))
    t, first = np.unique(t, return_index=True)
    o, c = o[first], c[first]
    ok = (o > 0) & (c > 0)
    return t[ok], o[ok], c[ok]


def build(cache: Path, out: Path, end: date) -> dict:
    start = SPLIT_STARTS["train"]
    stop = datetime(end.year, end.month, end.day, tzinfo=UTC)
    news: list[dict] = []
    bar_parts: dict[str, list[BarArrays]] = defaultdict(list)
    cik_ticker: dict[int, str] = {}
    names: dict[int, str] = {}
    filings: list[tuple[int, dict]] = []
    records, corrupt = _records(cache)
    if corrupt:
        print(f"skipped corrupt cache entries: {corrupt}")
    for rec in records:
        url, body = rec["url"], rec["body"]
        if url == NEWS_URL:
            news.extend(body["news"])
        elif url == BARS_URL:
            for ticker, rows in (body.get("bars") or {}).items():
                bar_parts[ticker].append(_bar_columns(rows))
        elif url == TICKERS_URL:
            cik_ticker.update({int(r["cik_str"]): r["ticker"] for r in body.values() if r["ticker"] in TICKERS})
        elif url.startswith(SUBMISSIONS_URL):
            cik = int(url.rsplit("/", 1)[1].removeprefix("CIK")[:10])
            if "filings" in body:  # main file: recent filings inline
                names[cik] = body["name"]
                filings.append((cik, body["filings"]["recent"]))
            else:  # history file: the filings arrays only
                filings.append((cik, body))

    dropped = Counter(roundups=0, no_universe_ticker=0, duplicates=0)
    stories: dict[str, Story] = {}

    def add(story: Story) -> None:
        if not start <= story.published_at < stop:
            return
        if story.story_id in stories:
            dropped["duplicates"] += 1
        else:
            stories[story.story_id] = story

    for item in news:
        symbols = item["symbols"]
        if len(symbols) > MAX_TICKERS_PER_STORY:
            dropped["roundups"] += 1
            continue
        tickers = tuple(dict.fromkeys(s for s in symbols if s in TICKERS))
        if not tickers:
            dropped["no_universe_ticker"] += 1
            continue
        summary = _clean_summary(item.get("summary") or "")
        headline = html.unescape(item["headline"])
        add(Story(f"bz:{item['id']}", "benzinga", _utc(item["created_at"]), tickers, headline, summary))
    for cik, block in filings:
        if cik in cik_ticker:
            for story in _eight_k_stories(block, cik_ticker[cik], names.get(cik, "")):
                add(story)

    out.mkdir(parents=True, exist_ok=True)
    bars = {t: _merge_bars(bar_parts.get(t, [])) for t in TICKERS}
    starts = [SPLIT_STARTS[s].timestamp() for s in SPLITS] + [stop.timestamp()]
    counts: dict[str, dict[str, int]] = {}
    for split, lo, hi in zip(SPLITS, starts, starts[1:], strict=False):
        in_split = [s for s in stories.values() if split_of(s.published_at) == split]
        window_bars = {}
        for ticker, (t, o, c) in bars.items():
            keep = (t >= lo) & (t < hi)
            window_bars[ticker] = (t[keep], o[keep], c[keep])
        stories_name, bars_name = store_files(split)
        counts[split] = {
            "stories": write_stories(out / stories_name, in_split),
            "pairs": sum(len(s.tickers) for s in in_split),
            "bars": write_bars(out / bars_name, window_bars),
        }

    manifest = write_manifest(
        out,
        {
            "tickers": list(TICKERS),
            "universe": list(TICKERS),
            "window_end": end.isoformat(),
            "dropped": dict(dropped),
        },
    )
    for split, c in counts.items():
        print(f"{split}: stories={c['stories']} pairs={c['pairs']} bars={c['bars']}")
    print(f"dropped: {dict(dropped)}")
    for ticker, (t, _, _) in bars.items():
        first_last = [datetime.fromtimestamp(int(t[i]), UTC).strftime("%Y-%m-%d") for i in (0, -1)] if len(t) else []
        print(f"bars {ticker}: " + (" .. ".join(first_last) or "none"))
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m infosys.build")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date(), help="exclusive")
    args = ap.parse_args(argv)
    build(args.cache, args.out, args.end)


if __name__ == "__main__":
    main()
