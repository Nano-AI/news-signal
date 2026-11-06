# Run record: 20261106-090202

A summary of one VibeSys optimization run on this task, starting from `main` (df868f7). The run
ended at its 8 h deadline with an adopted candidate, 3c435f9. Branches on this repository hold the code.

## Setup

| Item | Value |
|---|---|
| Hardware | 1x AMD MI210 |
| Planner and judge | Opus 5.5 |
| Implementer | Sonnet 5.5 |
| Wall time | 8 h 01 m |
| Cost | $11.61 (tokens) |
| Ideas tried | 9 (8 produced a candidate, 1 cancelled at the deadline) |
| Adopted | 3c435f9, "Native libtorch C++ hot path" |

## Results

Latency of the bursts benchmark, milliseconds. Lower is better.

| Split | Code | p99 | p50 | Speedup (p99 / p50) |
|---|---|---|---|---|
| Validation | Original (frozen baseline) | 88.5 | 14.6 | |
| Validation | Winner 3c435f9 | 2.87 | 2.17 | about 31x / 6.7x |
| Holdout (mean of 2 runs) | Original | 88.3 | 16.1 | |
| Holdout (mean of 2 runs) | Winner 3c435f9 | 3.54 | 2.50 | about 25x / 6.4x |

Holdout winner: 0 missed, 0 duplicate and 0 invalid pairs.

| Fidelity to the original model | Agreement | Spearman |
|---|---|---|
| Validation (gate: 0.99 / 0.95) | 1.0000 | 0.99999 |
| Holdout | not measured yet | |

Notes:

- The run never benchmarked the starting code, so the app showed "input not measured". The baseline
  figures above come from the frozen baseline measured before the run.
- A scorer floor check (a do-nothing service) gave a p50 of about 1.6 ms on the original client
  structure, so the winner's holdout median of 2.50 ms is within about 1 ms of the transport floor.
- An independent review of the winner found no rule break. Low-severity notes: no fallback after the
  process hands over to the native binary, fp16 is not self-checked during prepare, and the C++
  tokenizer parity check only covered characters present in the train headlines.

## Ideas

Each idea builds on a parent commit. "Candidate" means it produced a commit that passed correctness,
benchmark and local validation. Only the winner was marked eligible for adoption.

| # | Idea | Parent | Commit | Outcome |
|---|---|---|---|---|
| 1 | Remove queueing and polling stalls in the ingest, classify and post pipeline (long-poll ingest, micro-batching, batched posts) | original | 3253230 | Candidate. p99 about 87 to 30.2 ms, p50 14.5 to 12.0 ms. |
| 2 | fp16 FinBERT with HIP graphs per batch and length bucket | 3253230 | 072efbc | Candidate. Built on idea 1. |
| 3 | Fused BERT forward | 072efbc | 6d6fab6 | Candidate. |
| 4 | Per-bucket autotuned graphs | 6d6fab6 | 062320f | Candidate. |
| 5 | Host-path overlap: tokenization and HTTP overlapped with GPU work, less GIL handoff | 062320f | 9363ddc | Candidate. p99 3.82 to 3.43 ms, p50 2.89 to 2.58 ms (about the noise bar). |
| 6 | Fused HIP kernels (attention, residual add plus LayerNorm, embeddings) built in prepare | 9363ddc (via 4e79b55) | 9514c5d | Candidate, not retained as a gain: p99 3.53 vs 3.43 ms, within noise. |
| 7 | Dual-lane concurrent replay on separate HIP streams | 9363ddc | e244815 | Candidate, disproven: concurrent small graphs do not overlap usefully. |
| 8 | Native libtorch C++ hot path: epoll networking, C++ WordPiece tokenizer checked against the HF tokenizer in prepare, graph replay, batches up to 64 | 9363ddc | 3c435f9 | Adopted. p99 2.87 ms, p50 2.17 ms on validation. |
| 9 | Native GEMM tuning and epilogues | 3c435f9 | e732118 (partial) | Cancelled at the deadline, no candidate. |

## Branches

Code for each idea is on https://github.com/Nano-AI/news-signal:

- Winner (3c435f9): https://github.com/Nano-AI/news-signal/tree/vibesys/run-20261106-090202/winner
- Idea 9, partial work (e732118): https://github.com/Nano-AI/news-signal/tree/vibesys/run-20261106-090202/e732118
- Idea 6 (9514c5d): https://github.com/Nano-AI/news-signal/tree/vibesys/run-20261106-090202/9514c5d
- Idea 7 (e244815): https://github.com/Nano-AI/news-signal/tree/vibesys/run-20261106-090202/e244815
- A second, separate run (20261106-201316) was stopped by hand after one candidate, which was not
  adopted: https://github.com/Nano-AI/news-signal/tree/vibesys/run-20261106-201316/fc9db6f

Earlier candidates (ideas 1 to 5) are ancestors of the winner and are in its history.
