"""Latency benchmark: replay the busiest stretches of a split and report news-to-signal latency.

Writes an evaluator result-protocol-2 record stream to --vs-output (see sdk/vs-evaluator/PROTOCOL.md).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # passrun

from passrun import Mode, PassError, load_gate, run_pass  # noqa: E402

HELLO = {
    "kind": "hello",
    "protocol": 2,
    "metrics": {
        "p99_latency_ms": {"unit": "ms", "direction": "min"},
        "p50_latency_ms": {"unit": "ms", "direction": "min"},
        "p90_latency_ms": {"unit": "ms", "direction": "min"},
        "throughput_signals_per_s": {"unit": "signals/s", "direction": "max"},
        "missed": {"direction": "min"},
        "duplicates": {"direction": "min"},
        "invalid": {"direction": "min"},
        "stories": {},
        "newsd_send_lag_p50_ms": {"unit": "ms", "direction": "min", "required": False},
        "newsd_send_lag_p90_ms": {"unit": "ms", "direction": "min", "required": False},
        "newsd_send_lag_p99_ms": {"unit": "ms", "direction": "min", "required": False},
        "newsd_send_lag_max_ms": {"unit": "ms", "direction": "min", "required": False},
        "max_latency_ms": {"unit": "ms", "direction": "min", "required": False},
        "cold_start_s": {"unit": "s", "direction": "min", "required": False},
        "gpu_mem_peak_mib": {"unit": "MiB", "direction": "min", "required": False},
        "reruns": {"direction": "min", "required": False},
        "first_pass_send_lag_p90_ms": {"unit": "ms", "required": False},
        "first_pass_send_lag_p99_ms": {"unit": "ms", "required": False},
        "ic": {"direction": "max", "required": False},
        "ic_se": {"required": False},
    },
}


def error_record(message: str, *, partial: dict | None = None, requirement: str | None = None) -> dict:
    rec: dict = {"kind": "error", "message": message}
    if requirement:
        rec |= {"error_kind": "requirement_not_met", "requirement": requirement}
    if partial:
        rec["partial"] = partial
    return rec


def _partial_p99(p99: float | None) -> dict | None:
    if p99 is None or not math.isfinite(p99):
        return None
    return {"name": "p99_latency_ms", "value": p99, "direction": "min", "unit": "ms"}


IC_SE_MARGIN = 3.0


def quality_failure(score, gate: dict, directions: set[int]) -> str | None:
    """A degenerate or much worse signal set fails the benchmark, so a fast model that predicts nothing cannot win."""
    if len(directions) < 2:
        return f"the signals use {len(directions)} direction(s); a real model emits at least two"
    base = gate["baseline"]
    if not base["frozen"]:
        return None  # the baseline run that freezes bursts_ic
    if score.ic is None:
        return "ic is undefined on the bursts pass"
    bound = base["bursts_ic"] - IC_SE_MARGIN * (score.ic_se or 0.0)
    if score.ic < bound:
        return f"ic {score.ic:.4f} is below the baseline {base['bursts_ic']:.4f} minus {IC_SE_MARGIN:g} se"
    return None


def records_for(score, gate: dict, directions: set[int], first_invalid=None, serve=None) -> list[dict]:
    """HELLO plus one result, or an error naming the first failing check and carrying the p99 reached.

    Result protocol 2 has no details map: a result row holds only finite numbers under metrics declared in HELLO,
    and unknown keys are rejected (sdk/vs-evaluator/PROTOCOL.md). So the record-only scalars are optional declared
    metrics (absent when unknown), and the per-feed numbers go to the sidecar of `stats_for`."""
    lat = score.latency_ms
    partial = _partial_p99(lat["p99"]) if score.valid else None  # an invalid pass's p99 is not trustworthy
    if not score.valid:
        first = f"; first pass: {first_invalid.invalid_reason}" if first_invalid else ""
        return [HELLO, error_record(f"invalid pass after a rerun: {score.invalid_reason}{first}" if first_invalid
                                    else f"invalid pass: {score.invalid_reason}")]
    for name, count, limit in (
        ("missed", score.missed, gate["max_missed_frac"] * score.pairs),
        ("duplicates", score.duplicates, gate["max_duplicate_frac"] * score.pairs),
        ("invalid", score.invalid, gate["max_invalid"]),
    ):
        if count > limit:
            return [HELLO, error_record(f"{name} {count} exceeds the gate ({limit:g})", partial=partial)]
    if lat["p99"] is None:
        return [HELLO, error_record("the pass produced no latency samples")]
    failure = quality_failure(score, gate, directions)
    if failure:
        return [HELLO, error_record(f"prediction quality: {failure}", partial=partial)]
    values = {
        "p99_latency_ms": lat["p99"],
        "p50_latency_ms": lat["p50"],
        "p90_latency_ms": lat["p90"],
        "throughput_signals_per_s": score.throughput_per_s,
        "missed": score.missed,
        "duplicates": score.duplicates,
        "invalid": score.invalid,
        "stories": score.stories,
        "newsd_send_lag_p50_ms": score.send_lag_ms["p50"],
        "newsd_send_lag_p90_ms": score.send_lag_ms["p90"],
        "newsd_send_lag_p99_ms": score.send_lag_ms["p99"],
        "newsd_send_lag_max_ms": score.send_lag_ms["max"],
    }
    record_only = {"max_latency_ms": lat["max"], "cold_start_s": getattr(serve, "cold_start_s", None),
                   "gpu_mem_peak_mib": getattr(serve, "gpu_mem_peak_mib", None)}  # fmt: skip
    values |= {k: v for k, v in record_only.items() if v is not None}
    if first_invalid is not None:  # the first pass was invalid (newsd fell behind) and the rerun is reported
        values["reruns"] = 1
        values["first_pass_send_lag_p90_ms"] = first_invalid.send_lag_ms["p90"]
        values["first_pass_send_lag_p99_ms"] = first_invalid.send_lag_ms["p99"]
    else:
        values["reruns"] = 0
    if score.ic is not None:
        values["ic"] = score.ic
    if score.ic_se is not None:
        values["ic_se"] = score.ic_se
    return [HELLO, {"kind": "result", "label": "", "values": values}]


def stats_for(score, serve=None) -> dict:
    """Every record-only number of a pass, including what a result row cannot hold (null, per-feed maps)."""
    serve = serve or {}
    return {
        "latency_ms": score.latency_ms,
        "throughput_signals_per_s": score.throughput_per_s,
        "send_lag_ms": score.send_lag_ms,
        "feeds": score.feed_latency_ms,
        "serve": dataclasses.asdict(serve) if dataclasses.is_dataclass(serve) else serve,
    }


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))


def _fmt(value: float | None) -> str:
    return "null" if value is None else f"{value:.1f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchmark")
    p.add_argument("--mode", choices=[Mode.BURSTS.value], required=True)
    p.add_argument("--split", choices=["validation", "holdout"], default="validation")
    p.add_argument("--vs-output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args(argv)
    try:
        gate = load_gate()
        result = run_pass(Mode.BURSTS, a.split, device=a.device)
        records = records_for(result.score, gate, {sig.direction for sig in result.log.signals}, result.first_invalid,
                              result.serve)
    except PassError as e:
        records = [HELLO, error_record(e.message, requirement=e.requirement)]
    except Exception as e:  # the stream must exist; the type alone, since messages can carry data text
        records = [HELLO, error_record(f"unexpected {type(e).__name__} in the benchmark harness")]
    write_records(a.vs_output, records)
    if records[-1]["kind"] == "result":  # sidecar next to the stream: <vs-output>.stats.json
        with contextlib.suppress(OSError):  # record only: it must never fail a round whose result is written
            a.vs_output.with_name(a.vs_output.name + ".stats.json").write_text(
                json.dumps(stats_for(result.score, result.serve), indent=2) + "\n"
            )
    final = records[-1]
    if final["kind"] == "error":
        print(f"benchmark failed: {final['message']}")
        return 1
    v = final["values"]
    print(
        f"bursts: stories={v['stories']} p50={v['p50_latency_ms']:.1f}ms p90={v['p90_latency_ms']:.1f}ms "
        f"p99={v['p99_latency_ms']:.1f}ms max={_fmt(v.get('max_latency_ms'))}ms throughput={v['throughput_signals_per_s']:.1f}/s "
        f"send_lag_p90={v['newsd_send_lag_p90_ms']:.2f}ms p99={v['newsd_send_lag_p99_ms']:.2f}ms reruns={v['reruns']} "
        f"cold_start={_fmt(v.get('cold_start_s'))}s gpu_mem_peak={_fmt(v.get('gpu_mem_peak_mib'))}MiB"
    )
    for feed, f in result.score.feed_latency_ms.items():
        print(f"feed {feed}: pairs={f['pairs']} p50={_fmt(f['p50'])}ms p99={_fmt(f['p99'])}ms first={f['first']}")
    if result.first_invalid is not None:
        print(f"note: first pass was invalid ({result.first_invalid.invalid_reason}); the rerun is reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
