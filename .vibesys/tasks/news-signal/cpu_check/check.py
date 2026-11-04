"""CPU check: one smoke pass over the synthetic fixture with a tiny random model. Takes about 20 s."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # passrun
sys.path.insert(0, str(HERE))  # tiny_model

from passrun import Mode, PassError, run_pass  # noqa: E402
from tiny_model import build_tiny_model  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ns-tiny-") as tmp:
        model = build_tiny_model(Path(tmp) / "model")
        try:
            result = run_pass(Mode.SMOKE, "validation", device="cpu", model=str(model))
        except PassError as e:
            print(f"FAIL: {e.message}")
            return 1
    score, sv = result.score, result.serve
    lat = score.latency_ms
    print(
        f"stories={score.stories} pairs={score.pairs} missed={score.missed} duplicates={score.duplicates} "
        f"invalid={score.invalid} p50_ms={lat['p50']} p99_ms={lat['p99']} send_lag_p90_ms={score.send_lag_ms['p90']:.2f} send_lag_p99_ms={score.send_lag_ms['p99']:.2f}"
    )
    print(
        f"p90_ms={lat['p90']} max_ms={lat['max']} throughput_per_s={score.throughput_per_s:.1f} "
        f"send_lag_p50_ms={score.send_lag_ms['p50']:.2f} send_lag_max_ms={score.send_lag_ms['max']:.2f} "
        f"cold_start_s={sv.cold_start_s} gpu_mem_peak_mib={sv.gpu_mem_peak_mib}"
    )
    for feed, f in score.feed_latency_ms.items():
        print(f"feed {feed}: pairs={f['pairs']} p50_ms={f['p50']} p99_ms={f['p99']} first={f['first']}")
    if not score.valid:
        print(f"note: timing not trustworthy on this host: {score.invalid_reason}")
    bad = {"missed": score.missed, "duplicates": score.duplicates, "invalid": score.invalid}
    failing = [f"{k}={v}" for k, v in bad.items() if v]
    if failing:
        print("FAIL: " + ", ".join(failing))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
