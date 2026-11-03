import asyncio
import contextlib
import dataclasses
import json
import socket
import time
from datetime import UTC, datetime, timedelta

import aiohttp
import pytest

from infosys.newsd import PassLog, ReadyTimeout, serve_pass
from infosys.replay import FEEDS, FeedSpec, plan_pass
from infosys.store import Story

BOUND_S = 10  # only guards against hangs
BASE = datetime(2024, 3, 11, 14, 0, tzinfo=UTC)
FAST_FEEDS = tuple(dataclasses.replace(f, base_ms=5, jitter_ms=10) for f in FEEDS)


def _story(i: int, offset_s: float, source="benzinga", tickers=("AAA",)) -> Story:
    return Story(f"s{i:02d}", source, BASE + timedelta(seconds=offset_s), tickers, f"head {i}", f"body {i}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def _bounded(coro):
    return await asyncio.wait_for(coro, BOUND_S)


class Server:
    """Runs serve_pass in a task; `start` returns once it listens (and `on_listening` has finished)."""

    def __init__(self, plan, *, on_listening=None, drain_s=0.05, ready_timeout_s=BOUND_S, clock=time.monotonic):
        self.port, self.listening = _free_port(), asyncio.Event()
        self.url = f"http://127.0.0.1:{self.port}"

        async def hook():
            self.listening.set()
            if on_listening is not None:
                await on_listening(self)

        self.task = asyncio.create_task(
            serve_pass(
                plan,
                port=self.port,
                ready_timeout_s=ready_timeout_s,
                drain_s=drain_s,
                on_listening=hook,
                clock=clock,
            )
        )

    async def start(self) -> "Server":
        await _bounded(self.listening.wait())
        return self

    async def result(self) -> PassLog:
        return await _bounded(self.task)

    async def stop(self):
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, 3 * BOUND_S))


def _signal(story_id="s00", ticker="AAA", **extra):
    return {"story_id": story_id, "ticker": ticker, "direction": 1, "confidence": 0.5, **extra}


async def _post(session, url, signals, path="/signals"):
    async with session.post(url + path, json={"signals": signals}) as r:
        return r.status, await r.json()


def test_pass_end_to_end_never_early():
    stories = [_story(i, i * 0.5, "edgar" if i % 3 == 0 else "benzinga", ("AAA", "BBB")[: 1 + i % 2]) for i in range(8)]
    plan = plan_pass(
        stories[2:], stories[:2], gap_cap_s=0.05, seed=4, feeds=FAST_FEEDS, warmup_rate_hz=100, warmup_gap_s=0.05
    )
    first_seen: dict[str, float] = {}
    posted: set[tuple[str, str]] = set()

    async def sight(session, url, msg):
        first_seen.setdefault(msg["story_id"], time.monotonic())
        for ticker in msg["tickers"]:
            if (msg["story_id"], ticker) not in posted:
                posted.add((msg["story_id"], ticker))
                status, body = await _post(session, url, [_signal(msg["story_id"], ticker)])
                assert status == 200 and body["rejected"] == [], body

    async def ws_reader(session, url, path, connected):
        async with session.ws_connect(url + path) as ws:
            connected.set()
            async for frame in ws:
                msg = json.loads(frame.data)
                if msg["type"] == "end":
                    return
                await sight(session, url, msg)

    async def poll_reader(session, url, path, connected):
        connected.set()
        after = 0
        while True:
            async with session.get(url + path, params={"after": after, "wait_ms": 200}) as r:
                body = await r.json()
            for msg in body["messages"]:
                await sight(session, url, msg)
            after = body["next"]
            if body["end"]:
                return

    async def main():
        readers: list[asyncio.Task] = []
        session = aiohttp.ClientSession()

        async def subscribe(server):
            async with session.get(server.url + "/feeds") as r:
                feeds = (await r.json())["feeds"]
            events = []
            for f in feeds:
                events.append(asyncio.Event())
                reader = ws_reader if f["transport"] == "ws" else poll_reader
                readers.append(asyncio.create_task(reader(session, server.url, f["path"], events[-1])))
            await _bounded(asyncio.gather(*(e.wait() for e in events)))
            async with session.post(server.url + "/ready") as r:
                assert (await r.json()) == {"ok": True}

        server = Server(plan, on_listening=subscribe, drain_s=0.1)
        try:
            log = await server.result()
            await _bounded(asyncio.gather(*readers))
        finally:
            await session.close()
            await server.stop()
        return log

    log = _run(main())
    pairs = {(s.story_id, t) for s in stories for t in s.tickers}
    assert {(s.story_id, s.ticker) for s in log.signals} == pairs
    assert len(log.signals) == len(pairs)
    assert log.invalid == {} and log.schema_errors == 0
    for story in stories:
        assert first_seen[story.story_id] - log.t0 >= plan.available_s[story.story_id] - 0.001


def _manual_plan():
    """A: available at 0 s; F: available at 50 s. The fake clock decides which are visible."""
    a, f = _story(0, 0, tickers=("AAA", "BBB")), _story(1, 50)
    plan = plan_pass([a, f], [], gap_cap_s=100, seed=0, feeds=FAST_FEEDS[:2], warmup_gap_s=0.0)
    return plan, a, f


def test_signal_validation():
    plan, a, f = _manual_plan()
    clock = FakeClock()

    async def main():
        server = await Server(plan, clock=clock).start()
        async with aiohttp.ClientSession() as s:
            # before /ready everything is not yet available
            assert await _post(s, server.url, [_signal("s00")]) == (
                200,
                {"accepted": 0, "rejected": [{"index": 0, "reason": "not_yet_available"}]},
            )
            async with s.post(server.url + "/ready"):
                pass
            clock.now += 1.5  # t0 = 101.0, so plan time is 0.5 s

            status, body = await _post(s, server.url, [_signal(price=1)])
            assert status == 422 and body["error"].startswith("signals.0.price:")
            status, body = await _post(s, server.url, [_signal(direction=2)])
            assert status == 422 and body["error"].startswith("signals.0.direction:")
            async with s.post(server.url + "/signals", data=b"[1]") as r:
                assert r.status == 400 and await r.json() == {"error": "malformed JSON body"}

            batch = [
                _signal(),
                _signal("nope"),
                _signal("s00", "ZZZ"),
                _signal(f.story_id),
                _signal(),
            ]
            status, body = await _post(s, server.url, batch)
            assert status == 200 and body["accepted"] == 2
            assert body["rejected"] == [
                {"index": 1, "reason": "not_yet_available"},
                {"index": 2, "reason": "unknown_ticker"},
                {"index": 3, "reason": "not_yet_available"},
            ]
            clock.now += 1000
            return await server.result()

    log = _run(main())
    assert log.schema_errors == 2
    assert log.invalid == {"not_yet_available": 3, "unknown_ticker": 1}  # one of them before /ready
    assert [(x.story_id, x.ticker, x.received_s) for x in log.signals] == [("s00", "AAA", 0.5)] * 2


def test_poll_params():
    plan, _, _ = _manual_plan()

    async def main():
        server = await Server(plan).start()
        try:
            async with aiohttp.ClientSession() as s:
                for query, name in [("wait_ms=1001", "wait_ms"), ("after=-1", "after"), ("limit=0", "limit"), ("after=x", "after")]:
                    async with s.get(f"{server.url}/feeds/edgar/poll?{query}") as r:
                        assert r.status == 422 and (await r.json())["error"].startswith(f"{name}:"), query
                async with s.get(f"{server.url}/feeds/edgar/poll?after=3&wait_ms=300") as r:
                    body = await _bounded(r.json())  # holds for wait_ms, then returns empty
                assert body == {"messages": [], "next": 3, "end": False}
                for path, status in [("/feeds/benzinga/poll", 404), ("/feeds/edgar/ws", 404), ("/nope", 404), ("/ready", 405)]:
                    async with s.get(server.url + path) as r:
                        assert r.status == status and "error" in await r.json(), path
        finally:
            await server.stop()

    _run(main())


def test_ready_timeout():
    plan, _, _ = _manual_plan()

    async def main():
        with pytest.raises(ReadyTimeout):
            await _bounded(serve_pass(plan, port=_free_port(), ready_timeout_s=0.2, drain_s=0.0))

    _run(main())


def test_end_broadcast():
    plan = plan_pass([_story(0, 0)], [], gap_cap_s=1, seed=0, feeds=FAST_FEEDS, warmup_gap_s=0.0)
    clock = FakeClock()

    async def poll_to_end(s, url):
        after, polls = 0, []
        while True:
            async with s.get(f"{url}/feeds/edgar/poll", params={"after": after, "wait_ms": 100}) as r:
                body = await r.json()
            polls.append(body)
            after = body["next"]
            if body["end"]:
                return polls

    async def main():
        server = await Server(plan, clock=clock).start()
        async with aiohttp.ClientSession() as s:
            ws = await s.ws_connect(server.url + "/feeds/benzinga/ws")
            async with s.post(server.url + "/ready"):
                pass
            polling = asyncio.create_task(poll_to_end(s, server.url))
            clock.now += 1000
            frames = []
            while True:
                frame = await _bounded(ws.receive())
                if frame.type != aiohttp.WSMsgType.TEXT:
                    break
                frames.append(json.loads(frame.data))
            polls = await _bounded(polling)
            await server.result()
        return frames, polls

    frames, polls = _run(main())
    assert [m["type"] for m in frames] == ["story", "end"]
    assert polls[-1] == {"messages": [], "next": 0, "end": True}
    assert not any(p["end"] for p in polls[:-1])
