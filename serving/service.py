"""Baseline service: subscribe to every feed, classify each new story, post one signal per ticker."""

import asyncio
import json
import queue
import sys
import threading
from collections.abc import Callable

import aiohttp

_STORY_KEYS = {"type", "feed", "seq", "story_id", "source", "published_at", "tickers", "headline", "summary"}


def parse_story(msg: dict) -> dict:
    if set(msg) != _STORY_KEYS or msg["type"] != "story":
        raise ValueError(f"bad story message keys: {sorted(msg)}")
    return msg


def build_signal(story_id: str, ticker: str, direction: int, confidence: float) -> dict:
    return {"story_id": story_id, "ticker": ticker, "direction": direction, "confidence": confidence}


MAX_BATCH = 64  # stories per forward pass
MAX_POST = 1024  # pairs per POST /signals


async def _ws_feed(session, base: str, feed: dict, on_story, opened: asyncio.Event):
    async with session.ws_connect(base.replace("http", "ws", 1) + feed["path"]) as ws:
        opened.set()
        async for frame in ws:
            msg = frame.json()
            if msg["type"] == "end":
                return
            on_story(parse_story(msg))
    raise RuntimeError(f"ws feed {feed['name']} closed without end")


async def _poll_feed(session, base: str, feed: dict, on_story):
    after = 0
    while True:
        params = {"after": after, "wait_ms": 1000, "limit": 500}
        async with session.get(base + feed["path"], params=params) as r:
            r.raise_for_status()
            body = await r.json()
        for msg in body["messages"]:
            on_story(parse_story(msg))
        after = body["next"]
        if body["end"]:
            return


def _infer_loop(inq: queue.Queue, classify_batch, loop, on_batch, finished: asyncio.Future):
    """The only thread that touches the model: classify whatever is queued, as soon as it is free."""
    try:
        done = False
        while not done:
            batch = [inq.get()]
            if batch[0] is None:
                break
            while len(batch) < MAX_BATCH:
                try:
                    item = inq.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    done = True
                    break
                batch.append(item)
            results = classify_batch([ids for _, ids in batch])
            loop.call_soon_threadsafe(on_batch, [s for s, _ in batch], results)
    except BaseException as e:
        loop.call_soon_threadsafe(finished.set_exception, e)
        return
    loop.call_soon_threadsafe(finished.set_result, None)


_JSON = {"Content-Type": "application/json"}


async def _post(session, base: str, body: bytes) -> bool:
    """POST one pre-serialised batch; False when the run is already finished (409)."""
    async with session.post(base + "/signals", data=body, headers=_JSON) as r:
        if r.status == 409:
            return False
        if r.status != 200:
            raise RuntimeError(f"POST /signals -> {r.status}: {await r.text()}")
        rejected = (await r.json()).get("rejected")
        if rejected:
            raise RuntimeError(f"POST /signals rejected {len(rejected)} pairs: {rejected[:3]}")
    return True


async def run(newsd_url: str, classify_batch: Callable[[list], list[tuple[int, float]]],
              tokenize: Callable[[str], object] = lambda h: h) -> None:
    """tokenize runs at arrival in the event loop; classify_batch receives its outputs."""
    sys.setswitchinterval(0.0002)
    base = newsd_url.rstrip("/")
    loop = asyncio.get_running_loop()
    inq: queue.Queue = queue.Queue()
    seen: set[str] = set()  # checked synchronously in the event loop: first arrival wins
    posts: set[asyncio.Task] = set()
    stopped = asyncio.Event()  # a 409 or a failed POST ends the loop below
    post_errors: list[BaseException] = []

    def on_story(story: dict) -> None:
        if story["story_id"] not in seen:
            seen.add(story["story_id"])
            inq.put((story, tokenize(story["headline"])))

    async def post_task(body):
        if not await _post(session, base, body):
            stopped.set()

    def on_post_done(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception() is not None:
            post_errors.append(t.exception())
            stopped.set()

    def on_batch(stories, results) -> None:
        sigs = [
            build_signal(s["story_id"], t, d, c)
            for s, (d, c) in zip(stories, results)
            for t in s["tickers"]
        ]
        for i in range(0, len(sigs), MAX_POST):
            task = asyncio.create_task(
                post_task(json.dumps({"signals": sigs[i : i + MAX_POST]}).encode())
            )
            posts.add(task)
            task.add_done_callback(posts.discard)
            task.add_done_callback(on_post_done)

    finished: asyncio.Future = loop.create_future()
    thread = threading.Thread(
        target=_infer_loop, args=(inq, classify_batch, loop, on_batch, finished), daemon=True
    )
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async with session.get(base + "/feeds") as r:
            r.raise_for_status()
            feeds = (await r.json())["feeds"]
        tasks, opened = [], []
        for feed in feeds:
            if feed["transport"] == "ws":
                ev = asyncio.Event()
                opened.append(ev)
                tasks.append(asyncio.create_task(_ws_feed(session, base, feed, on_story, ev)))
            else:
                tasks.append(asyncio.create_task(_poll_feed(session, base, feed, on_story)))
        thread.start()
        waiter = asyncio.gather(*(ev.wait() for ev in opened))
        stop_wait = asyncio.create_task(stopped.wait())
        try:
            # a feed task dying before its socket opens must not hang the wait
            pending = {waiter, *tasks, finished, stop_wait}
            ready_sent = False
            feeds_done = False
            while not stopped.is_set():
                if not ready_sent and waiter.done():
                    await waiter
                    async with session.post(base + "/ready") as r:
                        r.raise_for_status()
                    ready_sent = True
                if not feeds_done and all(t.done() for t in tasks):
                    feeds_done = True
                    inq.put(None)  # every feed ended: let the inference thread finish
                if feeds_done and finished.done():
                    finished.result()
                    if not posts:
                        break
                watch = [x for x in pending if not x.done()] + list(posts)
                done, _ = await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t is not stop_wait and not t.cancelled():
                        t.result()
            if post_errors:
                raise post_errors[0]
        except BaseException as e:
            print(f"serving: {e!r}", file=sys.stderr)
            raise
        finally:
            inq.put(None)
            waiter.cancel()
            stop_wait.cancel()
            for t in [*tasks, *posts]:
                t.cancel()
            await asyncio.gather(waiter, stop_wait, *tasks, *posts, return_exceptions=True)
