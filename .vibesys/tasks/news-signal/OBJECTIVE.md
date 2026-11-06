# Objective: news-to-signal latency on 1x MI210

Lower the **p99 latency from a news story reaching the service to its trading signal**, on 1x AMD MI210
(gfx90a, 64 GB), in the bursts replay (the busiest five-minute windows of the validation split, replayed fast
over six feeds). The candidate is the `serving/` package: a service that subscribes to every news feed,
classifies each story with FinBERT, and posts one signal per `(story, ticker)` pair back to `newsd`.
`docs/protocol.md` defines the contract.

## Scoring and quality bar

The bursts benchmark reports `p99_latency_ms` (the objective) and `p50_latency_ms`; missed or duplicate pairs
beyond 0.1 % or any invalid signal make the pass an error. An accuracy gate replays validation stories at a
high rate. Both are scored by `infosys/`.

Quality bar: agree with the original FinBERT (fp32 on CPU, hidden reference). The same direction on at least
99 % of hidden validation pairs, and signal Spearman at least 0.95. Any implementation is allowed if it meets
the quality bar and the rules below.

## Languages and toolchains

- `serving/` may be written in any language, e.g. Python or C++.
- `python -m serving` may launch a program built during `serving.prepare`. Build output goes in the prepare
  work directory, which persists; the service runs it from there.
- Toolchains on the GPU cluster's login node (compute nodes are expected to match): ROCm 7.2 with `hipcc`,
  g++ 12, cmake 3.25, make. Not present: standalone clang++, ninja, Rust, CUDA. The installed torch is a ROCm
  build and ships its C++ headers.
- The CPU check runs on a machine without a GPU, and the service must start there (`--device cpu`).

If a rule blocks a better design, say so in your notes. You may propose anything; the scorer stays untouchable.

## Rules

- Only `serving/` changes. The scoring code and data (`infosys/`) are read-only and off limits, as are
  `docs/protocol.md`, `pyproject.toml`, `uv.lock` and `.vibesys/`.
- `python -m serving.prepare` and `python -m serving` must keep working as the process contract in
  `docs/protocol.md` describes.
- No new Python dependencies (the pinned `pyproject.toml` and `uv.lock`): the evaluation environment is shared.
- Do not read the hidden data, environment variables, or files beyond the contract. Do not infer future
  stories from ids or order, and do not signal a story before receiving it.
- Behave the same in every pass. Do not detect which pass you are in (story count, spacing, rate, the parent's
  command line, anything else) and change the model or output accordingly.
- Do not use logs to move data: failed-pass logs are truncated and are not a channel.
- Train only in `serving.prepare`, on the train file it is given. Persist nothing derived from a pass or from
  the stories (memoised outputs, logs of headlines, network calls).

## CPU check

Run `.vibesys/tasks/news-signal/cpu_check/run.sh` (about 20 s, no GPU); it prints `PASS` when the service
starts and signals every pair. Measurement internals, enforcement details and the baseline record are in
`REFERENCE.md` beside this file.
