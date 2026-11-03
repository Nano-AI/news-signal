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

Base URL `http://127.0.0.1:<port>`. Errors are JSON `{"error": "<message>"}`. A 422 message names the
offending parameter or key.

### `GET /health` -> `{"ok": true}`

### `GET /feeds` -> `FeedList`

`{"feeds": [FeedInfo]}` with `FeedInfo = {"name": str, "transport": "ws" | "poll", "path": str}`.
`path` is relative to the base URL: `/feeds/{name}/ws` for `ws`, `/feeds/{name}/poll` for `poll`. Subscribe to
every feed listed; names are not fixed.

### `GET /feeds/{name}/ws`

WebSocket upgrade. The server sends text frames, each a JSON `StoryMessage` or `{"type": "end"}`. Frames
from the client are ignored. A subscriber receives only messages delivered after it connects, so open every
subscription before calling `POST /ready`. After `end`, the server closes the socket.

`StoryMessage`:

| key | type | rule |
| --- | --- | --- |
| `type` | `"story"` | |
| `feed` | string | name of the delivering feed |
| `seq` | int | >= 1, per feed, see below |
| `story_id` | string | canonical |
| `source` | `"benzinga"` or `"edgar"` | |
| `published_at` | string | the story's publication time |
| `tickers` | list of string | 1 to 5 entries, all from the universe |
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

A schema error (unknown key, wrong type, value out of range, 0 or more than 1024 items) gives 422 naming the
key, and the whole batch is rejected. Otherwise the reply is 200 `SignalReply`:
`{"accepted": int, "rejected": [{"index": int, "reason": ...}]}`. `index` is the item's position in the batch.
Each item is checked in this order and gets the first reason that applies:

1. `unknown_story`: `story_id` is not in the replayed split.
2. `unknown_ticker`: `ticker` is not one of the story's tickers.
3. `not_yet_available`: no feed has made the story available yet.

`accepted` counts items without a rejection, including duplicates (a duplicate is accepted but scored as a
duplicate). The first signal for a `(story_id, ticker)` pair counts; later ones are duplicates.

### `GET /status` -> `Status`

`{"state": "waiting" | "warmup" | "running" | "draining" | "done", "released": int, "total": int}`.
`waiting` is before `POST /ready`; `warmup` is after it, before the first story is released; `running`
releases stories; `draining` is after the last release, while late signals are still accepted; `done` is
the end. `released` is the number of stories released on any feed so far; `total` is the split's story count.

## Semantics

- Every feed copy of a story carries the canonical `story_id`.
- Synthetic feeds copy content exactly and change only timing, drops and duplicates.
- Receipt time is newsd's clock when the request body has been read.
- After `end` (ws) or `end: true` (poll), the server closes.

## Process contract (harness -> candidate)

Run from the repo root with the harness's own interpreter.

```
python -m serving.prepare --train FILE --out DIR --device {cuda,cpu}
python -m serving --newsd URL --model MODEL --device {cuda,cpu} --prepared DIR
```

- `serving.prepare`: `FILE` is gzip JSON lines of `TrainRecord`, the latest at most 50,000 train stories.
  `returns` holds the forward return at the gate's horizon per ticker, or `null` when it is not defined.
  `TrainRecord = {story_id, source, published_at, tickers, headline, summary, returns: {ticker: float | null}}`.
  Time limit 1200 s. `DIR` receives whatever the service needs later.
- `serving` (the service): `URL` is the newsd base URL. It subscribes to every feed, posts signals, and exits
  with code 0 after the `end` of every feed.
- The environment has `HF_HUB_OFFLINE=1`, no `NEWS_SIGNAL_*` variables, and no network guarantee. `MODEL` is
  `ProsusAI/finbert` (from the Hugging Face cache) or a local directory.
