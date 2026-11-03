# Objective: news-to-signal latency on 1x MI210

Minimize the **p99 latency from a news story reaching the service to its trading signal**, on 1x AMD MI210
(gfx90a, 64 GB), while the accuracy gate holds.

The candidate is the `serving/` package: a service that subscribes to every news feed, classifies each story
with FinBERT, and posts one signal per `(story, ticker)` pair back to `newsd`. It starts as a plain Hugging
Face FinBERT classifier, one headline per call. Everything the service may rely on is in
`docs/protocol.md`: read it first.

## Candidate rules

- Only `serving/` changes. The harness runs `python -m serving.prepare` and `python -m serving` from the repo
  root (see "Process contract" in `docs/protocol.md`), so both entry points must keep working.
- `infosys/`, `docs/protocol.md`, `pyproject.toml`, `uv.lock` and `.vibesys/` are read-only. Digests of `infosys/`
  and of the other three files are pinned in the task; both gates fail, without measuring, if any differs. The
  environment is fixed for this trial: the service may build extensions or vendor code inside `serving/` using
  packages already installed, but may not add dependencies. (The gates run with `uv run --frozen` where uv is
  used; on the cluster the shared, pre-built environment plays that role and is outside the tree.)
- No weights and no data in git: a file over 5 MB, or with a suffix in `.safetensors .bin .pt .pth .onnx .gguf
  .npz .parquet` anywhere outside `infosys/` and `.vibesys/`, fails both gates.
- Only packages installed in the evaluation environment may be imported. The only model available offline is
  `ProsusAI/finbert` (the run sets `HF_HUB_OFFLINE=1`).
- Training happens only in `serving.prepare`, on the train file it is given (at most 50,000 stories) and
  within 1200 s. Whatever it writes to `--out` is available to the service through `--prepared`.
- The service may not read environment variables or files beyond the contract (in particular not the scoring
  data directory, nor `/proc` entries of its parent), may not infer future stories from ids or order, and may
  not signal a story before it has received it. `newsd` refuses a signal for a story before its first
  scheduled availability. The runtime boundary around the scoring data is rule-based beyond what the harness
  enforces (the parent is not dumpable, the child's environment is stripped of `NEWS_SIGNAL_*`).
- The service must behave the same in every pass. Detecting which pass it is in (by story count, spacing, rate,
  the parent's command line, or anything else) and changing its model or output accordingly is forbidden.
- Log output on a failed pass reaches you truncated (20 lines of 200 characters); do not use it to move data.

## Measurement

Two commands, both run from the repo root on the MI210 node.

**Benchmark** (`benchmark/run.py --mode bursts`): replays the busiest five-minute windows of the validation
split, up to 180 s of compressed replay (gaps capped at 2 s), on six feeds with different delays, drops and
duplicates. Latency of a pair is the time from `newsd` handing the story to the service to the first accepted
signal for the pair. Reported: `p99_latency_ms` and `p50_latency_ms` (the objectives), `p90_latency_ms`,
`throughput_signals_per_s`, `missed`, `duplicates`, `invalid` and `newsd_send_lag_p99_ms`. A pass with any
missed pair beyond 0.1 %, duplicate beyond 0.1 %, or any invalid signal produces an error, not a result. So does
a pass in which `newsd` itself fell behind (send lag p99 over 5 ms): its latencies are not trustworthy, so rerun.
The pass also scores prediction quality: it fails if the service emits fewer than two distinct directions, or if
its IC is below the baseline bursts IC minus 3 standard errors of the pass. The service process runs at nice 10
so its CPU threads cannot starve `newsd`'s scheduler.

**Accuracy gate** (`accuracy_checker/checker.py`): replays a fixed sample of up to 30,000 validation stories at
a high rate and scores prediction quality as the rank correlation (IC) between `direction * confidence` and
the 5-minute forward return. It passes only when the pass is valid, IC is at least the frozen `ic_floor` in
`gate.toml` (baseline IC minus two bootstrap standard errors), and missed, duplicate and invalid counts are
within the gate. The pass ends once every pair has a signal and 5 s pass with none (at most 10 minutes after
the last story), so a keeping-up service finishes in about the replay time (up to 5 minutes) plus startup.

The score is computed by `infosys/`; the held-out holdout split exists only for the final report and is never
used in a round to tune a knob or choose between candidates.

## Fast CPU check before submitting

The trusted runs are slow to start (model load, replay, drain). Catch lifecycle bugs in about 20 s on the
editor host, which has no GPU. From the candidate root:

```bash
.vibesys/tasks/news-signal/cpu_check/run.sh
```

It builds a tiny random BERT classifier, replays the synthetic fixture through your service
(`python -m serving.prepare`, then `python -m serving --device cpu`), and prints `PASS` only if no pair was
missed or duplicated and no signal was rejected; it also prints the latency line. It catches crashes, missed
pairs, duplicate signals, a service that never calls `POST /ready`, and a service that exits early. Run it once
the service starts; once it passes, submit, and do not rerun it. Passing is necessary, not sufficient: the
accuracy gate on real data stays the gate. Keep `--device cpu` working in `serving/`.

## Baseline

Pending: the baseline has not been frozen. The numbers (IC, IC floor, p99 latency of the unmodified
`serving/` on two runs, validation story count, scored pairs) are written here and into `gate.toml` by
`scripts/freeze_baseline.py`.
