"""Latency benchmark: replay the busiest stretches of a split and report news-to-signal latency.

Writes an evaluator result-protocol-2 record stream to --vs-output (see sdk/vs-evaluator/PROTOCOL.md).
"""

from __future__ import annotations

import argparse
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
        "newsd_send_lag_p99_ms": {"unit": "ms", "direction": "min", "required": False},
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


def records_for(score, gate: dict) -> list[dict]:
    """HELLO plus one result, or an error naming the first failing count and carrying the p99 reached."""
    lat = score.latency_ms
    partial = _partial_p99(lat["p99"])
    if not score.valid:
        return [HELLO, error_record(f"invalid pass: {score.invalid_reason}", partial=partial)]
    for name, count, limit in (
        ("missed", score.missed, gate["max_missed_frac"] * score.pairs),
        ("duplicates", score.duplicates, gate["max_duplicate_frac"] * score.pairs),
        ("invalid", score.invalid, gate["max_invalid"]),
    ):
        if count > limit:
            return [HELLO, error_record(f"{name} {count} exceeds the gate ({limit:g})", partial=partial)]
    if lat["p99"] is None:
        return [HELLO, error_record("the pass produced no latency samples")]
    values = {
        "p99_latency_ms": lat["p99"],
        "p50_latency_ms": lat["p50"],
        "p90_latency_ms": lat["p90"],
        "throughput_signals_per_s": score.throughput_per_s,
        "missed": score.missed,
        "duplicates": score.duplicates,
        "invalid": score.invalid,
        "stories": score.stories,
        "newsd_send_lag_p99_ms": score.send_lag_p99_ms,
    }
    return [HELLO, {"kind": "result", "label": "", "values": values}]


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchmark")
    p.add_argument("--mode", choices=[Mode.BURSTS.value], required=True)
    p.add_argument("--split", choices=["validation", "holdout"], default="validation")
    p.add_argument("--vs-output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args(argv)
    try:
        gate = load_gate()
        score = run_pass(Mode.BURSTS, a.split, device=a.device).score
        records = records_for(score, gate)
    except PassError as e:
        records = [HELLO, error_record(e.message, requirement=e.requirement)]
    write_records(a.vs_output, records)
    final = records[-1]
    if final["kind"] == "error":
        print(f"benchmark failed: {final['message']}")
        return 1
    v = final["values"]
    print(
        f"bursts: stories={v['stories']} p50={v['p50_latency_ms']:.1f}ms p90={v['p90_latency_ms']:.1f}ms "
        f"p99={v['p99_latency_ms']:.1f}ms throughput={v['throughput_signals_per_s']:.1f}/s "
        f"send_lag_p99={v['newsd_send_lag_p99_ms']:.2f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
