# Wire protocol

`infosys` (trusted) and `serving` (the candidate) share only this document. Every JSON shape below is
defined by a pydantic model in `infosys/protocol.py` and has one valid example in
`docs/protocol-examples.json`. All models reject unknown keys and use strict types (a JSON string is not
accepted where a number is expected).

## Terms

- **Story**: one news item. `story_id` is canonical and identical on every feed that carries it.
- **Pair**: a `(story_id, ticker)` combination. A story with k tickers expects k signals. Latency, missed,
  duplicate and prediction scores are all per pair.
- **Feed**: a named delivery channel with its own `seq` counter, discovered through `GET /feeds`.
- Timestamps are UTC strings `YYYY-MM-DDTHH:MM:SSZ`, whole seconds.

## newsd HTTP API

Base URL `http://127.0.0.1:<port>`. For WebSocket, replace the scheme: `ws://127.0.0.1:<port>`.

### Errors

Every error is JSON `{"error": "<message>"}` with the status below. Nothing else is an error: polling or
posting before `POST /ready` is not one.

| status | when | message |
| --- | --- | --- |
| 400 | body of `POST /signals` is not valid JSON or not a JSON object | `malformed JSON body` |
| 404 | unknown feed name, a `ws` path on a `poll` feed (or the reverse), or an unknown path | `unknown feed: <name>` or `not found` |
| 405 | wrong HTTP method for a known path | `method not allowed` |
| 409 | `POST /signals` while state is `done` | `run finished` |
| 413 | body of `POST /signals` is over 1 MiB | `request body too large` |
| 422 | a poll query parameter is out of range or not an integer | `<param>: <reason>`, e.g. `wait_ms: must be 0..1000` |
| 422 | a `SignalBatch` fails schema validation | `<path>: <reason>` for the first error, `<path>` dotted from the body root, e.g. `signals.0.price: extra forbidden`, `signals.0.direction: ...`, or `signals: ...` for a bad list length |

### `GET /health` -> `{"ok": true}`

### `GET /feeds` -> `FeedList`

`{"feeds": [FeedInfo]}` with `FeedInfo = {"name": str, "transport": "ws" | "poll", "path": str}`.
`path` is relative to the base URL (use the `ws://` scheme for `ws`): `/feeds/{name}/ws` for `ws`,
`/feeds/{name}/poll` for `poll`. Subscribe to
every feed listed; names are not fixed.

### `GET /feeds/{name}/ws`

WebSocket upgrade. The server sends text frames, each a JSON `StoryMessage` or `{"type": "end"}`. Frames
from the client are ignored. A subscriber receives only messages delivered after it connects, so open every
subscription before calling `POST /ready`. After `end`, the server closes the socket. Each subscriber has its own bounded queue (1000 messages) and
sender: a subscriber whose queue overflows, or whose single send is stuck for more than 1 s, is disconnected
and receives no more messages (its undelivered pairs count as missed unless another feed delivers them).
One slow subscriber never delays another feed or subscriber.

`StoryMessage`:

| key | type | rule |
| --- | --- | --- |
| `type` | `"story"` | |
| `feed` | string | name of the delivering feed |
| `seq` | int | >= 1, per feed, see below |
| `story_id` | string | canonical |
| `source` | `"benzinga"` or `"edgar"` | |
| `published_at` | string | the story's publication time |
| `tickers` | list of string | 1 to 5 entries, from the replayed split's universe (see Universe) |
| `headline` | string | |
| `summary` | string | |

`seq` is per feed, starts at 1, and increases by exactly 1 for every message the feed delivers (including
duplicates, excluding drops). The same `story_id` may arrive more than once on a feed or on different feeds
with different `seq` values. Messages released at the same instant keep a deterministic order, by
`(published_at, story_id)`.

### `GET /feeds/{name}/poll?after=<int>&wait_ms=<int>&limit=<int>` -> `PollResponse`

`{"messages": [StoryMessage], "next": int, "end": bool}`.

- `after`: integer >= 0, default 0. Messages with `seq > after` are returned in seq order.
- `wait_ms`: integer 0..1000, default 0. When no message is available, the server holds the request up to
  `wait_ms`, then returns an empty list.
- `limit`: integer 1..500, default 100. Maximum messages per response.
- `next`: the last seq returned, or `after` when the list is empty. Pass it as the next `after`.
- Before `POST /ready` the response is empty with `next` = `after` and `end` false (the request still holds up
  to `wait_ms`).
- `end`: true once the feed has delivered everything and the response includes the final message (or none
  remain). When `end` is true the feed is finished; there is no `{"type": "end"}` message in poll feeds.
- Any out-of-range or non-integer parameter gives 422 `{"error": "..."}` naming the parameter.

### `POST /ready` -> `{"ok": true}`

Starts the pass. Idempotent: later calls do nothing and still return `{"ok": true}`. Send it after all
subscriptions are open.

### `POST /signals`

Body `SignalBatch`: `{"signals": [Signal]}` with 1 to 1024 items.

`Signal`: `story_id` (string), `ticker` (string), `direction` (`-1`, `0` or `1`), `confidence` (finite float,
`0 <= c <= 1`).

A schema error (unknown key, wrong type, value out of range, 0 or more than 1024 items) gives 422 (see
Errors) and the whole batch is rejected. Otherwise the reply is 200 `SignalReply`:
`{"accepted": int, "rejected": [{"index": int, "reason": ...}]}`. `index` is the item's position in the batch.
Each item is checked in this order and gets the first reason that applies:

1. `not_yet_available`: the story has not been made available on any feed yet, or `story_id` is not in the
   replayed split. The two cases are deliberately indistinguishable, so a candidate cannot probe for future
   stories. Before `POST /ready` every item gets this reason.
2. `unknown_ticker`: the story is available but `ticker` is not one of its tickers.

`accepted` counts items without a rejection, including duplicates (a duplicate is accepted but scored as a
duplicate). The first signal for a `(story_id, ticker)` pair counts; later ones are duplicates.

### `GET /status` -> `Status`

`{"state": "waiting" | "warmup" | "running" | "draining" | "done", "released": int, "total": int}`.
`waiting` is before `POST /ready`; `warmup` is after it, before the first story is released; `running`
releases stories; `draining` is after the last release, while late signals are still accepted; `done` is
the end. `released` is the number of stories released on any feed so far; `total` is the number of replayed (measured) stories, warmup excluded; `released` counts only those.

## Universe, horizon and labels

- **Universe**: the store's tickers are listed in its `manifest.json` under `tickers`. The real store uses the
  25 tickers of `infosys.universe.TICKERS`; the synthetic fixture uses `AAA BBB CCC DDD`. A story's `tickers`
  are a subset of the replayed store's universe. newsd checks signal tickers against the story's own
  `tickers`, never against a built-in list, and the candidate must not hard-code a universe either.
- **Horizon H**: 5 minutes. Forward return of a pair is
  `close(p + H) / close(p) - 1`, with `p` the story's `published_at`, both prices raw (unadjusted), from one
  regular session.
- **Labels**: the return is defined only when `p` and `p + H` fall on the same New York date with
  `09:31 <= local(p)` and `local(p + H) <= 16:00`; the start price comes from a one-minute bar that ended at
  or before `p` and is at most 30 minutes old; the end price comes from a bar that ended in `(p, p + H]`.
  Otherwise the return is `null`. A bar's close is its price at the bar's end (start + 60 s).
- **Latency** of a pair: `receipt - min over feeds of send`, where `send` is the instant newsd actually handed
  that story to the candidate on a feed: the instant it enters the subscriber's queue on a `ws` feed (so a
  candidate that stops reading gains nothing), the moment the message becomes pollable on a `poll` feed. Receipt is the first accepted signal for the pair. Both are on newsd's
  clock. A pair never signaled counts as missed, with latency `end - min send` (`end` is the end of the drain). The drain lasts at most `drain_s` after the last release; it ends early, once every measured pair has a first signal and 5 s then pass with no accepted signal. After the end, `POST /signals` gets 409 and nothing is counted: duplicate or invalid signals sent after the end are not scored, so a service must send every signal it intends to send while the pass is running.
  A story never sent on any feed uses its scheduled instant (`release + feed delay`) instead.
- **Invalid pass**: newsd also records, for every delivery of a measured story (warmup excluded), its lateness (actual send minus scheduled send).
  The pass is valid if and only if the 90th percentile of that lateness is at most 5 ms and the 99th is at
  most 50 ms; otherwise it is marked invalid in the score (`valid: false`, the reason names each bound that
  failed with its value) and its latencies are not to be trusted. The score reports the p50, p90, p99 and max
  lateness. Systematic server lateness, which is what a CPU-starving candidate causes, moves p90, while rare
  node stalls only move the tail: an idle GPU node showed intermittent 15-20 ms stalls, while measured passes kept p90 around 1.5 ms.
  A stall bunches late stories, which makes the candidate's job harder, not easier, and latency is measured
  from the enqueue instant, so stalls never flatter the candidate.

## Semantics

- Every feed copy of a story carries the canonical `story_id`.
- Synthetic feeds copy content exactly and change only timing, drops and duplicates.
- Real feeds (Benzinga, EDGAR) never drop a story; drops happen only on synthetic copies. So every
  `(story_id, ticker)` pair becomes available through its real feed.
- Receipt time is newsd's clock when the request body has been read.
- After `end` (ws) or `end: true` (poll), the server closes.

## Process contract (harness -> candidate)

Run from the repo root with the harness's own interpreter.

```
python -m serving.prepare --train FILE --out DIR --device {cuda,cpu}
python -m serving --newsd URL --model MODEL --device {cuda,cpu} --prepared DIR
```

- `serving.prepare`: `FILE` is gzip JSON lines of `TrainRecord`, the latest at most 50,000 train stories.
  `returns` holds the forward return at H (see Labels) per ticker, or `null` when it is not defined; its keys
  equal `tickers`. Lines are in ascending `(published_at, story_id)` order, so "latest" means the tail.
  `TrainRecord = {story_id, source, published_at, tickers, headline, summary, returns: {ticker: float | null}}`.
  Time limit 1200 s. `DIR` receives whatever the service needs later.
- `serving` (the service): `URL` is the newsd base URL. It subscribes to every feed, posts signals, and exits
  with code 0 after the `end` of every feed.
- The environment has `HF_HUB_OFFLINE=1`, no `NEWS_SIGNAL_*` variables, and no network guarantee. `MODEL` is
  `ProsusAI/finbert` (from the Hugging Face cache) or a local directory.
