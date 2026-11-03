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
            score = run_pass(Mode.SMOKE, "validation", device="cpu", model=str(model)).score
        except PassError as e:
            print(f"FAIL: {e.message}")
            return 1
    lat = score.latency_ms
    print(
        f"stories={score.stories} pairs={score.pairs} missed={score.missed} duplicates={score.duplicates} "
        f"invalid={score.invalid} p50_ms={lat['p50']} p99_ms={lat['p99']} send_lag_p90_ms={score.send_lag_ms['p90']:.2f} send_lag_p99_ms={score.send_lag_ms['p99']:.2f}"
    )
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
