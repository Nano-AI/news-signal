# news-signal

Financial news in, trading signal out. Two independent codebases share only the
wire protocol in `docs/protocol.md`:

- `infosys/`: the trusted information system. It builds a frozen event store
  (news stories and minute bars), replays a split through `newsd`, and scores
  latency and prediction quality.
- `serving/`: a FinBERT news-to-signal service. It subscribes to the feeds,
  scores stories, and posts signals back. This is the code a benchmark
  optimizes.

`serving/` never imports `infosys`, and `infosys` never imports `serving`
(`tests/test_boundaries.py`).

## Layout

- `docs/protocol.md`, `docs/protocol-examples.json`: the contract.
- `infosys/fixtures/tiny/`: a small synthetic event store used by tests.
- `scripts/scan_publish.py`: pre-publish scan for paths, secrets and data files.

## Tests

```
uv run pytest -q
```
