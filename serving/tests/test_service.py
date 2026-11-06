import asyncio

import pytest
from aiohttp import web

from serving.service import run

STORIES = [
    ("s1", ["AAA", "BBB"], "up up"),
    ("s2", ["CCC"], "down down"),
    ("s3", ["DDD"], "flat flat"),
]
FAKE = {"up up": (1, 0.9), "down down": (-1, 0.8), "flat flat": (0, 0.6)}


def story(feed, seq, sid, tickers, headline):
    return {"type": "story", "feed": feed, "seq": seq, "story_id": sid, "source": "benzinga",
            "published_at": "2023-07-10T14:31:05Z", "tickers": tickers, "headline": headline, "summary": "x"}


class FakeNewsd:
    def __init__(self):
        self.events, self.signals, self.ws_open = [], [], 0
        self.polls, self.post_sizes = [], []
        self.ready, self.both_ws = asyncio.Event(), asyncio.Event()
        self.app = web.Application()
        self.app.add_routes([
            web.get("/feeds", self.feeds), web.get("/feeds/{n}/ws", self.ws),
            web.get("/feeds/poll/poll", self.poll), web.post("/ready", self.on_ready),
            web.post("/signals", self.on_signals),
        ])

    async def feeds(self, _):
        return web.json_response({"feeds": [
            {"name": "w1", "transport": "ws", "path": "/feeds/w1/ws"},
            {"name": "w2", "transport": "ws", "path": "/feeds/w2/ws"},
            {"name": "poll", "transport": "poll", "path": "/feeds/poll/poll"}]})

    async def ws(self, req):
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self.ws_open += 1
        self.events.append("ws")
        if self.ws_open == 2:
            self.both_ws.set()
        await self.ready.wait()
        name = req.match_info["n"]
        for i, (sid, t, h) in enumerate(STORIES):
            await ws.send_json(story(name, i + 1, sid, t, h))
        await ws.send_json({"type": "end"})
        await ws.close()
        return ws

    async def poll(self, req):
        after = int(req.query["after"])
        self.polls.append(dict(req.query))
        if not self.ready.is_set():
            await asyncio.sleep(0.02)  # stand-in for the server-side hold
        msgs = [story("poll", i + 1, *s) for i, s in enumerate(STORIES) if i + 1 > after] if self.ready.is_set() else []
        return web.json_response({"messages": msgs, "next": after + len(msgs), "end": bool(msgs)})

    async def on_ready(self, _):
        self.events.append("ready")
        assert self.both_ws.is_set()
        self.ready.set()
        return web.json_response({"ok": True})

    async def on_signals(self, req):
        body = await req.json()
        self.post_sizes.append(sorted(s["ticker"] for s in body["signals"]))
        self.signals += body["signals"]
        return web.json_response({"accepted": len(body["signals"]), "rejected": []})


async def _scenario():
    fake = FakeNewsd()
    runner = web.AppRunner(fake.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        await asyncio.wait_for(run(f"http://127.0.0.1:{port}", lambda h: FAKE[h]), timeout=10)
    finally:
        await runner.cleanup()
    return fake


def test_service_end_to_end():
    fake = asyncio.run(_scenario())
    assert fake.events == ["ws", "ws", "ready"]
    pairs = sorted((s["story_id"], s["ticker"]) for s in fake.signals)
    assert pairs == [("s1", "AAA"), ("s1", "BBB"), ("s2", "CCC"), ("s3", "DDD")]
    assert sorted(fake.post_sizes) == [["AAA", "BBB"], ["CCC"], ["DDD"]]  # one POST per story
    assert fake.polls and all(int(q["wait_ms"]) > 0 and 1 <= int(q["limit"]) <= 500 for q in fake.polls)
    assert {s["story_id"]: (s["direction"], s["confidence"]) for s in fake.signals}["s2"] == (-1, 0.8)


# ---- failure paths ----------------------------------------------------------------------------
class Variant(FakeNewsd):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.posts = 0

    async def ws(self, req):
        if self.mode != "ws_drop":
            return await super().ws(req)
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self.ws_open += 1
        if self.ws_open == 2:
            self.both_ws.set()
        await self.ready.wait()
        await ws.close()  # no end frame
        return ws

    async def on_signals(self, req):
        self.posts += 1
        if self.mode in ("409", "500"):
            return web.json_response({"error": "x"}, status=int(self.mode))
        return await super().on_signals(req)


async def _failing(mode, classify):
    fake = Variant(mode)
    runner = web.AppRunner(fake.app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        await asyncio.wait_for(run(f"http://127.0.0.1:{port}", classify), timeout=10)
    finally:
        await runner.cleanup()
    return fake


def _boom(h):
    raise KeyError("boom")


@pytest.mark.parametrize(
    "mode,exc,msg",
    [("500", RuntimeError, "500"), ("ws_drop", RuntimeError, "closed without end"), ("ok", KeyError, "boom")],
)
def test_failures_raise(mode, exc, msg):
    with pytest.raises(exc, match=msg):
        asyncio.run(_failing(mode, _boom if mode == "ok" else (lambda h: FAKE[h])))


def test_409_stops_cleanly():
    assert asyncio.run(_failing("409", lambda h: FAKE[h])).posts >= 1
