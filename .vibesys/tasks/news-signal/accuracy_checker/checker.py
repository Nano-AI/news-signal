"""Accuracy gate: replay a validation sample through the candidate and compare its IC with the baseline.

Gate mode (no --report) passes only if the pass is valid, the baseline is frozen, IC holds the frozen floor, and
missed, duplicate and invalid counts stay within the gate. --report prints the numbers and always exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # passrun

from passrun import Mode, PassError, load_gate, rescore, run_pass  # noqa: E402


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


NOT_FROZEN = "baseline is not frozen (gate.toml baseline.frozen = false)"


def evaluate_gate(score, gate: dict) -> str | None:
    """The first failing check as a message, or None if the gate passes."""
    base = gate["baseline"]
    if not base["frozen"]:
        return NOT_FROZEN
    if not score.valid:
        return f"invalid pass: {score.invalid_reason}"
    if score.ic is None or score.ic < base["ic_floor"]:
        return f"ic {_fmt(score.ic)} is below the floor {base['ic_floor']}"
    if score.missed > gate["max_missed_frac"] * score.pairs:
        return f"missed {score.missed} of {score.pairs} pairs exceeds {gate['max_missed_frac']}"
    if score.duplicates > gate["max_duplicate_frac"] * score.pairs:
        return f"duplicates {score.duplicates} of {score.pairs} pairs exceeds {gate['max_duplicate_frac']}"
    if score.invalid > gate["max_invalid"]:
        return f"invalid {score.invalid} exceeds {gate['max_invalid']}"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="checker")
    p.add_argument("--split", choices=["validation", "holdout"], default="validation")
    p.add_argument("--report", action="store_true")
    p.add_argument("--horizons")
    p.add_argument("--drain-s", type=float)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--json-out", type=Path)
    a = p.parse_args(argv)
    if not a.report and (a.drain_s is not None or a.horizons is not None or a.split != "validation"):
        print("FAIL: gate mode refuses --drain-s, --horizons and a holdout split (use --report)")
        return 1
    try:
        horizons = [int(h) for h in a.horizons.split(",")] if a.horizons else None
    except ValueError:
        print("FAIL: --horizons must be comma-separated integers")
        return 1
    try:
        gate = load_gate()
        if not a.report and not gate["baseline"]["frozen"]:
            print(f"FAIL: {NOT_FROZEN}")
            return 1
        result = run_pass(Mode.ACCURACY, a.split, device=a.device, drain_s=a.drain_s)
    except PassError as e:
        print(e.message)
        return 1

    s = result.score
    print(
        f"accuracy: stories={s.stories} pairs={s.pairs} scored_pairs={s.scored_pairs} ic={_fmt(s.ic)} "
        f"ic_se={_fmt(s.ic_se)} hit_rate={_fmt(s.hit_rate)} missed={s.missed} duplicates={s.duplicates} "
        f"invalid={s.invalid}"
    )
    per_horizon = {}
    for h in horizons or [gate["h_minutes"]]:
        hs = s if h == gate["h_minutes"] else rescore(result, h)
        per_horizon[h] = {"ic": hs.ic, "ic_se": hs.ic_se, "scored_pairs": hs.scored_pairs}
        print(f"ic@{h}={_fmt(hs.ic)} se={_fmt(hs.ic_se)}")
    lat = s.latency_ms
    print(
        f"latency_ms: p50={_fmt(lat['p50'], 1)} p90={_fmt(lat['p90'], 1)} p99={_fmt(lat['p99'], 1)} max={_fmt(lat['max'], 1)} "
        f"throughput={s.throughput_per_s:.1f}/s send_lag_ms: p50={s.send_lag_ms['p50']:.2f} p90={s.send_lag_ms['p90']:.2f} "
        f"p99={s.send_lag_ms['p99']:.2f} max={s.send_lag_ms['max']:.2f}"
    )
    sv = result.serve
    print(
        f"serve: cold_start_s={_fmt(sv.cold_start_s, 1)} gpu_mem_idle_mib={_fmt(sv.gpu_mem_idle_mib, 0)} "
        f"gpu_mem_peak_mib={_fmt(sv.gpu_mem_peak_mib, 0)}"
    )
    for feed, f in s.feed_latency_ms.items():
        print(f"feed {feed}: pairs={f['pairs']} p50={_fmt(f['p50'], 1)}ms p99={_fmt(f['p99'], 1)}ms first={f['first']}")
    if result.first_invalid is not None:
        print(f"note: first pass was invalid ({result.first_invalid.invalid_reason}); the rerun is used")
    if not s.valid:
        print(f"note: invalid pass: {s.invalid_reason}")
    if a.json_out:
        a.json_out.write_text(
            json.dumps(
                {
                    "score": asdict(s),
                    "horizons": per_horizon,
                    "serve": asdict(sv),
                    "first_pass": asdict(result.first_invalid) if result.first_invalid else None,
                },
                indent=2,
            )
            + "\n"
        )
    if a.report:
        return 0
    failure = evaluate_gate(s, gate)
    print("PASS" if failure is None else f"FAIL: {failure}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    sys.exit(main())
