"""newsd: serves one replay pass over the wire protocol in docs/protocol.md and logs what comes back."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aiohttp import web
from pydantic import ValidationError

from infosys.protocol import FeedInfo, FeedList, Rejection, SignalBatch, SignalReply, Status, StoryMessage
from infosys.replay import FEEDS, Delivery, ReplayPlan
from infosys.store import Story

READY_LEAD_S = 1.0  # t0 = first /ready + this, so every client has time to settle
MAX_POLL_WAIT_MS = 1000
MAX_POLL_LIMIT = 500
_SLEEP_CAP_S = 0.05
SEND_TIMEOUT_S = 1.0  # a single ws send stuck this long disconnects the subscriber
SUBSCRIBER_QUEUE_MAX = 1000  # undelivered messages per ws subscriber before it is disconnected
QUIET_S = 5.0  # the pass ends early once every measured pair has a signal and this long passes with none
_END_LINGER_S = 5.0  # guard: how long to wait for pollers to collect their end


@dataclass(frozen=True)
class ReceivedSignal:
    story_id: str
    ticker: str
    direction: int
    confidence: float
    received_s: float  # seconds since t0, taken after the request body was read


@dataclass
class PassLog:
    t0: float
    signals: list[ReceivedSignal]  # accepted signals, duplicates included
    invalid: dict[str, int]  # rejection reason -> count
    schema_errors: int
    send_lag_ms: list[float]  # actual - scheduled send instant, per delivery of a measured story, in ms
    end_s: float
    # story_id -> earliest instant newsd handed the story to a subscriber's queue (ws) or made it pollable
    # (poll), in seconds since t0. Latency is measured from it.
    sent_s: dict[str, float] = field(default_factory=dict)


class ReadyTimeout(Exception):
    """No client called POST /ready in time."""


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


@web.middleware
async def _json_errors(request: web.Request, handler: Callable) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPNotFound:
        return _error(404, "not found")
    except web.HTTPMethodNotAllowed:
        return _error(405, "method not allowed")


def _int_param(request: web.Request, name: str, default: int, lo: int, hi: int | None) -> int:
    raw = request.query.get(name)
    if raw is None:
        return default
    value = int(raw) if re.fullmatch(r"-?[0-9]+", raw) else None
    bound = f"{lo}..{hi}" if hi is not None else f">= {lo}"
    if value is None or value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name}: must be {bound}")
    return value


class _Sub:
    """One ws subscriber: its own bounded queue and sender task, so a slow reader stalls nobody else."""

    def __init__(self, ws: web.WebSocketResponse, transport, maxsize: int):
        self.ws, self.transport = ws, transport
        self.queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize)
        self.task: asyncio.Task | None = None


class _Pass:
    def __init__(self, plan: ReplayPlan, clock: Callable[[], float], send_timeout_s: float, queue_max: int):
        self.plan, self.clock = plan, clock
        self.send_timeout_s, self.queue_max = send_timeout_s, queue_max
        self.measured_ids = {s.story_id for s in plan.measured}
        self.feeds: dict[str, str] = {f.name: f.transport for f in FEEDS}  # name -> transport
        unknown = {d.feed for d in plan.deliveries} - self.feeds.keys()
        if unknown:
            raise ValueError(f"plan uses feeds outside replay.FEEDS: {sorted(unknown)}")
        self.stories: dict[str, Story] = {d.story.story_id: d.story for d in plan.deliveries}
        self.buffers: dict[str, list[dict]] = {name: [] for name in self.feeds}
        self.subscribers: dict[str, set[_Sub]] = {name: set() for name in self.feeds}
        self.new_data = asyncio.Condition()
        self.ready = asyncio.Event()
        self.t0: float | None = None
        self.done = False
        self.released: set[str] = set()
        self.run_from = min((plan.release_s[s.story_id] for s in plan.measured), default=0.0)
        self.polled: set[str] = set()
        self.senders: set[asyncio.Task] = set()
        self.ended: set[str] = set()
        self.ended_seen = asyncio.Event()
        self.pending = {(s.story_id, t) for s in plan.measured for t in s.tickers}  # no first signal yet
        self.last_signal_s = 0.0
        self.log = PassLog(0.0, [], {}, 0, [], plan.last_s)

    # -- state ---------------------------------------------------------------------------------

    def state(self) -> str:
        if self.t0 is None:
            return "waiting"
        if self.done:
            return "done"
        elapsed = self.clock() - self.t0
        if elapsed > self.plan.last_s:
            return "draining"
        return "running" if elapsed >= self.run_from else "warmup"

    async def sleep_until(self, plan_s: float) -> None:
        assert self.t0 is not None
        while (gap := self.t0 + plan_s - self.clock()) > 0:
            await asyncio.sleep(min(max(gap, 0.0005), _SLEEP_CAP_S))

    # -- scheduler -----------------------------------------------------------------------------

    def _stamp(self, story_id: str) -> None:
        assert self.t0 is not None
        now = self.clock() - self.t0
        if now < self.log.sent_s.get(story_id, now + 1):
            self.log.sent_s[story_id] = now

    def _drop(self, feed: str, sub: _Sub) -> None:
        self.subscribers[feed].discard(sub)
        if sub.transport is not None:
            sub.transport.abort()  # also ends the handler's read loop

    async def _sender(self, feed: str, sub: _Sub) -> None:
        try:
            while (item := await sub.queue.get()) is not None:
                _, text = item
                async with asyncio.timeout(self.send_timeout_s):
                    await sub.ws.send_str(text)
            async with asyncio.timeout(self.send_timeout_s):
                await sub.ws.send_str(json.dumps({"type": "end"}))
                await sub.ws.close()
        except Exception:  # stuck (timeout) or broken connection
            self._drop(feed, sub)

    async def deliver(self, d: Delivery) -> None:
        assert self.t0 is not None
        await self.sleep_until(d.at_s)
        story = d.story
        if story.story_id in self.measured_ids:  # warmup is where the candidate cold-starts: not the server falling behind
            self.log.send_lag_ms.append((self.clock() - (self.t0 + d.at_s)) * 1000)
        msg = StoryMessage(
            type="story",
            feed=d.feed,
            seq=d.seq,
            story_id=story.story_id,
            source=story.source,
            published_at=story.published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            tickers=list(story.tickers),
            headline=story.headline,
            summary=story.summary,
        ).model_dump(mode="json")
        if story.story_id in self.measured_ids:
            self.released.add(story.story_id)
        text = json.dumps(msg)
        for sub in list(self.subscribers[d.feed]):
            try:
                sub.queue.put_nowait((story.story_id, text))
                self._stamp(story.story_id)  # handed to the subscriber: its own backpressure cannot delay this
            except asyncio.QueueFull:
                self._drop(d.feed, sub)
        if self.feeds[d.feed] == "poll":
            self._stamp(story.story_id)
        async with self.new_data:
            self.buffers[d.feed].append(msg)
            self.new_data.notify_all()

    async def run(self, drain_s: float) -> None:
        for d in self.plan.deliveries:
            await self.deliver(d)
        self.last_signal_s = self.clock() - self.t0
        deadline = self.plan.last_s + drain_s
        while (now := self.clock() - self.t0) < deadline:
            if not self.pending and now - self.last_signal_s >= QUIET_S:
                self.log.end_s = now  # every pair is in and the line has gone quiet
                break
            await asyncio.sleep(min(max(deadline - now, 0.0005), _SLEEP_CAP_S))
        self.done = True
        async with self.new_data:
            self.new_data.notify_all()
        for feed, subs in self.subscribers.items():
            for sub in list(subs):
                try:
                    sub.queue.put_nowait(None)
                except asyncio.QueueFull:
                    self._drop(feed, sub)
        await asyncio.gather(*self.senders, return_exceptions=True)
        if self.polled - self.ended:
            try:
                await asyncio.wait_for(self.ended_seen.wait(), _END_LINGER_S)
            except TimeoutError:
                pass

    # -- handlers ------------------------------------------------------------------------------

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def feed_list(self, request: web.Request) -> web.Response:
        infos = [FeedInfo(name=n, transport=t, path=f"/feeds/{n}/{t}") for n, t in self.feeds.items()]
        return web.json_response(FeedList(feeds=infos).model_dump(mode="json"))

    async def ready_handler(self, request: web.Request) -> web.Response:
        if self.t0 is None:
            self.t0 = self.clock() + READY_LEAD_S
            self.log.t0 = self.t0
            self.ready.set()
        return web.json_response({"ok": True})

    async def status(self, request: web.Request) -> web.Response:
        body = Status(state=self.state(), released=len(self.released), total=len(self.plan.measured))
        return web.json_response(body.model_dump(mode="json"))

    def _feed(self, request: web.Request, transport: str) -> str | None:
        name = request.match_info["name"]
        return name if self.feeds.get(name) == transport else None

    async def ws_handler(self, request: web.Request) -> web.StreamResponse:
        name = self._feed(request, "ws")
        if name is None:
            return _error(404, f"unknown feed: {request.match_info['name']}")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.done:
            await ws.send_str(json.dumps({"type": "end"}))
            await ws.close()
            return ws
        sub = _Sub(ws, request.transport, self.queue_max)
        sub.task = asyncio.create_task(self._sender(name, sub))
        self.senders.add(sub.task)
        self.subscribers[name].add(sub)
        try:
            async for _ in ws:  # frames from the client are ignored
                pass
        finally:
            self.subscribers[name].discard(sub)
            if not self.done:
                sub.task.cancel()
        return ws

    async def poll_handler(self, request: web.Request) -> web.Response:
        name = self._feed(request, "poll")
        if name is None:
            return _error(404, f"unknown feed: {request.match_info['name']}")
        try:
            after = _int_param(request, "after", 0, 0, None)
            wait_ms = _int_param(request, "wait_ms", 0, 0, MAX_POLL_WAIT_MS)
            limit = _int_param(request, "limit", 100, 1, MAX_POLL_LIMIT)
        except ValueError as e:
            return _error(422, str(e))
        self.polled.add(name)
        buf = self.buffers[name]
        if wait_ms and len(buf) <= after and not self.done:
            try:
                async with asyncio.timeout(wait_ms / 1000):
                    async with self.new_data:
                        await self.new_data.wait_for(lambda: len(buf) > after or self.done)
            except TimeoutError:
                pass
        messages = buf[after : after + limit]
        nxt = messages[-1]["seq"] if messages else after
        end = self.done and nxt >= len(buf)
        if end:
            self.ended.add(name)
            if self.ended >= self.polled:
                self.ended_seen.set()
        return web.json_response({"messages": messages, "next": nxt, "end": end})

    async def signals_handler(self, request: web.Request) -> web.Response:
        body = await request.read()
        now = self.clock()
        received = None if self.t0 is None else now - self.t0
        if self.done:
            return _error(409, "run finished")
        try:
            obj = json.loads(body)
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            return _error(400, "malformed JSON body")
        try:
            batch = SignalBatch.model_validate(obj)
        except ValidationError as e:
            self.log.schema_errors += 1
            first = e.errors()[0]
            return _error(422, f"{'.'.join(str(p) for p in first['loc']) or 'body'}: {first['msg']}")
        rejected = []
        for i, sig in enumerate(batch.signals):
            story = self.stories.get(sig.story_id)
            reason = None
            if story is None or received is None or now < self.t0 + self.plan.available_s[sig.story_id]:
                reason = "not_yet_available"
            elif sig.ticker not in story.tickers:
                reason = "unknown_ticker"
            if reason:
                rejected.append(Rejection(index=i, reason=reason))
                self.log.invalid[reason] = self.log.invalid.get(reason, 0) + 1
            else:
                self.log.signals.append(
                    ReceivedSignal(sig.story_id, sig.ticker, sig.direction, sig.confidence, received)
                )
                self.pending.discard((sig.story_id, sig.ticker))
                self.last_signal_s = received
        reply = SignalReply(accepted=len(batch.signals) - len(rejected), rejected=rejected)
        return web.json_response(reply.model_dump(mode="json"))


def _app(p: _Pass) -> web.Application:
    app = web.Application(middlewares=[_json_errors])
    app.add_routes(
        [
            web.get("/health", p.health),
            web.get("/feeds", p.feed_list),
            web.get("/feeds/{name}/ws", p.ws_handler),
            web.get("/feeds/{name}/poll", p.poll_handler),
            web.post("/ready", p.ready_handler),
            web.post("/signals", p.signals_handler),
            web.get("/status", p.status),
        ]
    )
    return app


async def serve_pass(
    plan: ReplayPlan,
    *,
    port: int,
    ready_timeout_s: float,
    drain_s: float,
    on_listening: Callable[[], Awaitable[None]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    send_timeout_s: float = SEND_TIMEOUT_S,
    subscriber_queue_max: int = SUBSCRIBER_QUEUE_MAX,
) -> PassLog:
    """Serve `plan` on 127.0.0.1:`port` until it finishes; raise ReadyTimeout if nobody calls /ready."""
    p = _Pass(plan, clock, send_timeout_s, subscriber_queue_max)
    p.log.end_s = plan.last_s + drain_s
    runner = web.AppRunner(_app(p), access_log=None, shutdown_timeout=1.0)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
        if on_listening is not None:
            await on_listening()
        try:
            await asyncio.wait_for(p.ready.wait(), ready_timeout_s)
        except TimeoutError:
            raise ReadyTimeout(f"no /ready within {ready_timeout_s}s") from None
        await p.run(drain_s)
    finally:
        await runner.cleanup()
    return p.log
