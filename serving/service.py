"""Baseline service: subscribe to every feed, classify each new story, post one signal per ticker."""

import asyncio
import sys
from collections.abc import Callable

import aiohttp

_STORY_KEYS = {"type", "feed", "seq", "story_id", "source", "published_at", "tickers", "headline", "summary"}


def parse_story(msg: dict) -> dict:
    if set(msg) != _STORY_KEYS or msg["type"] != "story":
        raise ValueError(f"bad story message keys: {sorted(msg)}")
    return msg


def build_signal(story_id: str, ticker: str, direction: int, confidence: float) -> dict:
    return {"story_id": story_id, "ticker": ticker, "direction": direction, "confidence": confidence}


async def _ws_feed(session, base: str, feed: dict, queue: asyncio.Queue, opened: asyncio.Event):
    async with session.ws_connect(base.replace("http", "ws", 1) + feed["path"]) as ws:
        opened.set()
        async for frame in ws:
            msg = frame.json()
            if msg["type"] == "end":
                await queue.put(None)
                return
            await queue.put(parse_story(msg))
    raise RuntimeError(f"ws feed {feed['name']} closed without end")


async def _poll_feed(session, base: str, feed: dict, queue: asyncio.Queue):
    after = 0
    while True:
        async with session.get(base + feed["path"], params={"after": after, "wait_ms": 1000, "limit": 500}) as r:
            r.raise_for_status()
            body = await r.json()
        for msg in body["messages"]:
            await queue.put(parse_story(msg))
        after = body["next"]
        if body["end"]:
            break
    await queue.put(None)


async def _post(session, base: str, story_id: str, tickers: list, direction: int, confidence: float) -> bool:
    """POST one story's signals in a single batch; True if newsd answered 409 (run finished)."""
    sigs = [build_signal(story_id, t, direction, confidence) for t in tickers]
    async with session.post(base + "/signals", json={"signals": sigs}) as r:
        if r.status == 409:
            return True
        if r.status != 200:
            raise RuntimeError(f"POST /signals -> {r.status}: {await r.text()}")
    return False


async def _worker(session, base: str, queue: asyncio.Queue, n_feeds: int, classify):
    seen: set[str] = set()  # ponytail: naive dedup, grows without bound
    pending: set[asyncio.Task] = set()
    stopped = False

    def reap(done):
        nonlocal stopped
        for t in done:
            pending.discard(t)
            if t.result():
                stopped = True

    ended = 0
    try:
        while ended < n_feeds and not stopped:
            story = await queue.get()
            if story is None:
                ended += 1
                continue
            if story["story_id"] in seen:
                continue
            seen.add(story["story_id"])
            direction, confidence = await asyncio.to_thread(classify, story["headline"])
            pending.add(asyncio.create_task(
                _post(session, base, story["story_id"], story["tickers"], direction, confidence)))
            reap({t for t in pending if t.done()})  # surface errors early
        if pending:
            done, _ = await asyncio.wait(pending)
            reap(done)
    finally:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def run(newsd_url: str, classify: Callable[[str], tuple[int, float]]) -> None:
    base = newsd_url.rstrip("/")
    queue: asyncio.Queue = asyncio.Queue()
    async with aiohttp.ClientSession() as session:
        async with session.get(base + "/feeds") as r:
            r.raise_for_status()
            feeds = (await r.json())["feeds"]
        tasks, opened = [], []
        for feed in feeds:
            if feed["transport"] == "ws":
                ev = asyncio.Event()
                opened.append(ev)
                tasks.append(asyncio.create_task(_ws_feed(session, base, feed, queue, ev)))
            else:
                tasks.append(asyncio.create_task(_poll_feed(session, base, feed, queue)))
        tasks.append(asyncio.create_task(_worker(session, base, queue, len(feeds), classify)))
        waiter = asyncio.gather(*(ev.wait() for ev in opened))
        try:
            # a feed task dying before its socket opens must not hang the wait
            done, _ = await asyncio.wait([waiter, *tasks], return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t is not waiter:
                    t.result()
            await waiter
            async with session.post(base + "/ready") as r:
                r.raise_for_status()
            worker, pending = tasks[-1], set(tasks)
            while worker in pending:  # worker ends at all feeds ended, or on 409
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    t.result()
        except BaseException as e:
            print(f"serving: {e!r}", file=sys.stderr)
            raise
        finally:
            waiter.cancel()
            for t in tasks:
                t.cancel()
            await asyncio.gather(waiter, *tasks, return_exceptions=True)
