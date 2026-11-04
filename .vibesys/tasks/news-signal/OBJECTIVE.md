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
  packages already installed, but may not add dependencies. (The pinned `pyproject.toml` and `uv.lock` keep the
  dependencies fixed; on the cluster the shared, pre-built environment is outside the tree.)
- No weights and no data in git: a file over 5 MB, or with a suffix in `.safetensors .bin .pt .pth .onnx .gguf
  .npz .parquet` anywhere outside `infosys/` and `.vibesys/`, fails both gates.
- Only packages installed in the evaluation environment may be imported. The only model available offline is
  `ProsusAI/finbert` (the run sets `HF_HUB_OFFLINE=1`).
- Training happens only in `serving.prepare`, on the train file it is given (at most 50,000 stories) and
  within 1200 s. Whatever it writes to `--out` is available to the service through `--prepared`.
- The service may not read environment variables or files beyond the contract (in particular not the scoring
  data directory, nor `/proc` entries of its parent), may not infer future stories from ids or order, and may
  not signal a story before it has received it. `newsd` refuses a signal for a story before its first
  scheduled availability. On every real-data pass
  the harness runs both `serving.prepare` and the service in an unprivileged user and mount namespace, then drops
  every capability and sets no-new-privs before exec (so the mounts cannot be undone and a nested namespace gains
  nothing). An empty tmpfs hides the scoring data directory. The pass is refused, with no fallback, if the
  namespace, a mount or the capability drop cannot be set up. The parent is not dumpable and the child's
  environment is stripped of `NEWS_SIGNAL_*`. The fixture check runs without the namespace.
- Nothing the service writes during the serving phase survives the attempt (prepare outputs do persist: it
  never sees stories). Enforced, for the service only: `/tmp`, the repo checkout, the prepare work directory
  (including `prepared/`), `$HOME` (so `~/.cache`, `~/.triton`), `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME` and every
  directory named by an environment variable ending in `CACHE`, `_HOME`, `_DIR` or `_PATH_DB` are read-only;
  `/var/tmp` and `/dev/shm` are fresh tmpfs; `TMPDIR`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `MIOPEN_USER_DB_PATH` and `PYTHONPYCACHEPREFIX` point at a scratch tmpfs that is
  discarded. Prepare may warm caches into its work directory and the service reads them there. The service must
  not persist or transmit anything derived from the stories (memoised outputs, logs of headlines, a network
  call) by any path the wrapper does not cover either; the network is not restricted by the harness.
- `serving.prepare` is not write-confined, so its outputs persist across rounds. It may persist only artifacts
  derived from the train split and the model, never anything derived from a pass.
- The service must behave the same in every pass. Detecting which pass it is in (by story count, spacing, rate,
  the parent's command line, or anything else) and changing its model or output accordingly is forbidden.
- Log output on a failed pass reaches you truncated (20 lines of 200 characters); do not use it to move data.

## Measurement

Two commands, both run from the repo root on the MI210 node.

**Benchmark** (`benchmark/run.py --mode bursts`): replays the busiest five-minute windows of the validation
split, up to 180 s of compressed replay (gaps capped at 0.25 s), on six feeds with different delays, drops and
duplicates. Latency of a pair is the time from `newsd` handing the story to the service to the first accepted
signal for the pair. Reported: `p99_latency_ms` and `p50_latency_ms` (the objectives), `p90_latency_ms`,
`throughput_signals_per_s`, `missed`, `duplicates`, `invalid`, `newsd_send_lag_p50_ms`, `_p90_ms`, `_p99_ms` and `_max_ms`. Also recorded, and decisive for
nothing: `max_latency_ms`; `cold_start_s` (spawn of the service to its `POST /ready`, as the harness sees it);
`gpu_mem_peak_mib` (peak whole-device VRAM while the service runs, sampled by the harness from outside the
namespace every 250 ms, absent when the host exposes none); and, in `<vs-output>.stats.json`, per-feed p50 and p99
of `receipt - that feed's send` with how often each feed sent a story first. Throughput is measured pairs with a
signal divided by the time from the first measured send to the last first-signal receipt. A pass with any
missed pair beyond 0.1 %, duplicate beyond 0.1 %, or any invalid signal produces an error, not a result. So does
a pass in which `newsd` itself fell behind (send lag p90 over 5 ms or p99 over 50 ms) twice: the harness reruns an invalid pass once
and uses the rerun, reporting `reruns` and the first pass's send lag.
Systematic server lateness, which is what a CPU-starving candidate causes, moves p90, while rare node stalls only
move the tail (an idle GPU node showed intermittent 15-20 ms stalls; measured passes kept p90 around 1.5 ms). A stall bunches late
stories, which makes the candidate's job harder, not easier, and latency is measured from the enqueue instant, so
stalls never flatter the candidate.
The pass also scores prediction quality against the hidden reference (see "Quality" below): it fails if the
service emits fewer than two distinct directions, if its direction agreement is under 0.99, or if its signal
Spearman is under 0.95. The result also reports `direction_agreement` and `signal_spearman`; `ic` and `ic_se` are
recorded for information only. The service process runs at nice 10
so its CPU threads cannot starve `newsd`'s scheduler.

**Accuracy gate** (`accuracy_checker/checker.py`): replays a fixed sample of up to 30,000 validation stories at
a high rate and scores prediction quality as fidelity to the original FinBERT (see "Quality" below). It passes only
when the pass is valid, direction agreement is at least 0.99 and signal Spearman at least 0.95 (`agreement_floor`
and `spearman_floor` in `gate.toml`), and missed, duplicate and invalid counts are within the gate. It also
prints the IC of the 5-minute forward return, which decides nothing. The pass ends once every pair has a signal and 5 s pass with none (at most 10 minutes after
the last story), so a keeping-up service finishes in about the replay time (up to 5 minutes) plus startup.

The score is computed by `infosys/`; the held-out holdout split exists only for the final report and is never
used in a round to tune a knob or choose between candidates.

## Quality

Quality means **agreeing with the original FinBERT**. The reference is the signal the unmodified `ProsusAI/finbert`
produces for every `(story, ticker)` pair of the validation split, in fp32 on CPU; it is hidden from you in both
prepare and serve, like the rest of the scoring data. A pass must reproduce it:

- **Direction agreement >= 0.99**: the fraction of reference pairs where your signal's direction equals the
  reference's. The direction of a signal is the sign of `direction * confidence`: -1, +1, or 0 for a neutral
  signal (`direction = 0`, or `confidence = 0`). Neutral agrees only with neutral. A pair you do not signal counts as
  a disagreement.
- **Signal Spearman >= 0.95**: the rank correlation of your `direction * confidence` with the reference's over the
  pairs you signalled. This guards against keeping the directions but faking the confidence (for example a constant
  confidence). A faithful fp16 or int8 model scores above 0.99.

Precision changes, fused kernels, batching, compilation and distillation are all
allowed if the signals still match. The IC (rank correlation with the forward return) is still computed and
reported, for information only: on this data the baseline's IC is indistinguishable from zero, so it cannot say
whether a faster model is as good.

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

Pending: the baseline has not been frozen. The numbers (the reference pair count and digest, the baseline's own
agreement and Spearman on the GPU accuracy pass and on both bursts passes, IC (record only), p99 latency of the
unmodified `serving/` on two runs, validation story count, scored pairs, and a record-only block with p90 and max latency,
throughput, send lag, model load time, peak GPU memory and per-feed latency) are written here and into `gate.toml` by
`scripts/freeze_baseline.py`.
