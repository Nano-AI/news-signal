import asyncio
import contextlib
import dataclasses
import json
import re
import socket
import time
from datetime import UTC, datetime, timedelta

import aiohttp
import numpy as np
import pytest

from infosys.newsd import PassLog, ReadyTimeout, ReceivedSignal, serve_pass
from infosys.replay import FEEDS, FeedSpec, plan_pass
from infosys.score import score_pass
from infosys.store import Bars, Story

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

    def __init__(self, plan, *, on_listening=None, drain_s=0.05, ready_timeout_s=BOUND_S, clock=time.monotonic, **serve_kwargs):
        self.port, self.listening = _free_port(), asyncio.Event()
        self.url = f"http://127.0.0.1:{self.port}"

        async def hook():
            if on_listening is not None:
                await on_listening(self)
            self.listening.set()

        self.task = asyncio.create_task(
            serve_pass(
                plan,
                port=self.port,
                ready_timeout_s=ready_timeout_s,
                drain_s=drain_s,
                on_listening=hook,
                clock=clock,
                **serve_kwargs,
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
    """Steps a fake clock across every release: nothing is visible before its instant, all of it after."""
    stories = [_story(i, i * 0.5, "edgar" if i % 3 == 0 else "benzinga", ("AAA", "BBB")[: 1 + i % 2]) for i in range(8)]
    plan = plan_pass(
        stories[2:], stories[:2], gap_cap_s=0.05, seed=4, feeds=FAST_FEEDS, warmup_rate_hz=100, warmup_gap_s=0.05
    )
    clock = FakeClock()
    first_seen: dict[str, float] = {}
    posted: set[tuple[str, str]] = set()

    async def sight(session, url, msg):
        first_seen.setdefault(msg["story_id"], clock())
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

    async def released(session, url) -> int:
        async with session.get(url + "/status") as r:
            return (await r.json())["released"]

    async def main():
        readers: list[asyncio.Task] = []
        session = aiohttp.ClientSession()
        t0 = clock() + 1.0

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

        server = Server(plan, on_listening=subscribe, drain_s=0.1, clock=clock)
        try:
            await server.start()
            measured = {s.story_id for s in plan.measured}
            for at in sorted({plan.available_s[i] for i in measured}):
                before = sum(plan.available_s[i] < at for i in measured)
                clock.now = t0 + at - 0.001
                await asyncio.sleep(0.05)  # lets an early delivery happen if the scheduler were wrong
                assert await released(session, server.url) == before
                clock.now = t0 + at
                for _ in range(2000):  # hang guard: ~10 s
                    if await released(session, server.url) > before:
                        break
                    await asyncio.sleep(0.005)
                else:
                    raise AssertionError("delivery never happened")
            clock.now = t0 + plan.last_s + 1.0
            log = await server.result()
            await _bounded(asyncio.gather(*readers))
        finally:
            await session.close()
            await server.stop()
        return log, t0

    log, t0 = _run(main())
    pairs = {(s.story_id, t) for s in stories for t in s.tickers}
    assert {(s.story_id, s.ticker) for s in log.signals} == pairs
    assert len(log.signals) == len(pairs)
    assert log.invalid == {} and log.schema_errors == 0 and log.t0 == t0
    for story in stories:
        assert first_seen[story.story_id] - t0 >= plan.available_s[story.story_id] - 1e-9
        assert log.sent_s[story.story_id] >= plan.available_s[story.story_id] - 1e-9


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


def test_drain_accepts_signals_then_done_rejects_with_409():
    plan, _, _ = _manual_plan()
    clock = FakeClock()

    async def status(s, url):
        async with s.get(url + "/status") as r:
            return await r.json()

    async def main():
        server = await Server(plan, clock=clock, drain_s=100.0).start()
        async with aiohttp.ClientSession() as s:
            async with s.post(server.url + "/ready"):
                pass
            async with s.get(server.url + "/feeds/edgar/poll"):  # a polled feed keeps the server up after done
                pass
            assert (await status(s, server.url))["state"] == "warmup"
            clock.now += 1.0 + plan.last_s + 1.0  # past the last release, inside the drain
            assert (await status(s, server.url))["state"] == "draining"
            assert (await status(s, server.url))["total"] == 2
            assert await _post(s, server.url, [_signal()]) == (200, {"accepted": 1, "rejected": []})
            clock.now += 1000
            for _ in range(2000):  # hang guard
                if (await status(s, server.url))["state"] == "done":
                    break
                await asyncio.sleep(0.005)
            assert await _post(s, server.url, [_signal()]) == (409, {"error": "run finished"})
            async with s.get(server.url + "/feeds/edgar/poll") as r:  # collect the end so the server exits
                assert (await r.json())["end"] is True
            return await server.result()

    log = _run(main())
    assert [x.story_id for x in log.signals] == ["s00"]


def _big_plan(n=80, feeds=FAST_FEEDS[:1]):
    big = "x" * 250_000  # n * 250 kB: far beyond what socket buffers hold for a reader that never reads
    stories = [dataclasses.replace(_story(i, 0.02 * i), summary=big) for i in range(n)]
    return stories, plan_pass(stories, [], gap_cap_s=0.02, seed=2, feeds=feeds, warmup_gap_s=0.0)


def test_slow_subscriber_does_not_stall_other_subscribers():
    stories, plan = _big_plan()

    async def main():
        server = await Server(plan, send_timeout_s=0.2, subscriber_queue_max=50).start()
        received, stuck_frames = [], []
        async with aiohttp.ClientSession() as s:
            url = server.url + "/feeds/benzinga/ws"
            fast, stuck = await s.ws_connect(url), await s.ws_connect(url)  # the second is never read
            async with s.post(server.url + "/ready"):
                pass
            async for frame in fast:  # no JSON parsing: it would share the event loop with the scheduler
                if frame.data.startswith('{"type": "end"'):
                    break
                received.append(re.search(r'"story_id": "([^"]+)"', frame.data[:400]).group(1))
            log = await server.result()
            while True:  # whatever the stuck subscriber got before it was cut off
                frame = await _bounded(stuck.receive())
                if frame.type != aiohttp.WSMsgType.TEXT:
                    break
                stuck_frames.append(json.loads(frame.data))
        return received, stuck_frames, log

    received, stuck_frames, log = _run(main())
    assert sorted(received) == [s.story_id for s in stories]  # jitter may swap neighbours
    assert np.percentile(log.send_lag_ms, 99) < 5.0  # the stuck reader did not delay the scheduler
    assert all(m["type"] == "story" for m in stuck_frames)  # disconnected: it never got the end
    delivered = {m["story_id"] for m in stuck_frames}
    assert len(delivered) < len(stories)
    signals = [ReceivedSignal(i, "AAA", 1, 1.0, 5.0) for i in delivered]  # the stuck candidate signals what it saw
    score = score_pass(plan, PassLog(0.0, signals, {}, 0, [], 100.0, log.sent_s), Bars({}), horizon_min=5)
    assert score.missed == len(stories) - len(delivered) > 0


def test_non_reading_candidate_cannot_lower_its_latency():
    """Reading late must not move the stamps: they are taken when newsd hands the story over."""
    stories, plan = _big_plan(60)

    async def run(read_late: bool) -> PassLog:
        clock = FakeClock()
        t0 = clock() + 1.0
        server = await Server(
            plan, clock=clock, send_timeout_s=60, subscriber_queue_max=1000, drain_s=500.0
        ).start()
        async with aiohttp.ClientSession() as s:
            ws = await s.ws_connect(server.url + "/feeds/benzinga/ws")

            async def drain():
                async for frame in ws:
                    if json.loads(frame.data)["type"] == "end":
                        return

            reader = None if read_late else asyncio.create_task(drain())
            async with s.post(server.url + "/ready"):
                pass
            for at in sorted(set(plan.available_s.values())):
                clock.now = t0 + at
                want = sum(a <= at for a in plan.available_s.values())
                for _ in range(2000):  # hang guard
                    async with s.get(server.url + "/status") as r:
                        if (await r.json())["released"] >= want:
                            break
                    await asyncio.sleep(0.005)
                else:
                    raise AssertionError("delivery never happened")
            clock.now = t0 + plan.last_s + 1000  # the candidate dawdles, then starts reading
            await _bounded(reader or drain())
            return await server.result()

    reading, dawdling = _run(run(False)), _run(run(True))
    assert dawdling.sent_s.keys() == reading.sent_s.keys() == plan.available_s.keys()
    for story_id, scheduled in plan.available_s.items():
        assert dawdling.sent_s[story_id] == pytest.approx(reading.sent_s[story_id], abs=1e-6)
        assert dawdling.sent_s[story_id] == pytest.approx(scheduled, abs=1e-6)
