"""GPU probe of the real-data wrapper, run by `freeze_baseline.py --stages probe` from the checkout root.

Prints WRAPPED/PLAIN fixture passes (the wrapped one forces the real-data wrapper), UMOUNT and NESTED results
from inside the wrapper, and FinBERT latency inside and outside it. Lines are data-free.
"""

from __future__ import annotations

import ctypes
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path[:0] = [".", ".vibesys/tasks/news-signal"]


def child(kind: str, data: str) -> None:
    if kind == "umount":
        libc = ctypes.CDLL(None, use_errno=True)
        r = libc.umount2(data.encode(), 0)
        print("UMOUNT", "rc", r, "errno", ctypes.get_errno(), "entries", len(os.listdir(data)), flush=True)
    elif kind == "nested":
        r = subprocess.run(["unshare", "-Urm", "sh", "-c", f"umount {data} 2>&1; ls {data} | wc -l"], capture_output=True, text=True)
        print("NESTED", "rc", r.returncode, "out", " ".join(r.stdout.split())[:80], flush=True)
    elif kind == "lat":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        m = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").cuda().eval()
        b = {k: v.cuda() for k, v in tok("Apple beats earnings expectations", return_tensors="pt").items()}
        ts = []
        with torch.inference_mode():
            for _ in range(30):
                m(**b)
            for _ in range(300):
                torch.cuda.synchronize()
                t = time.perf_counter()
                m(**b)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000)
        print("LAT", sys.argv[4], "cuda", torch.cuda.is_available(), "median_ms %.2f" % statistics.median(ts),
              "p99_ms %.2f" % sorted(ts)[296], "entries", len(os.listdir(data)), flush=True)


def main() -> None:
    import infosys  # noqa: F401
    import passrun

    for label, force in (("WRAPPED", True), ("PLAIN", False)):
        r = passrun.run_pass(passrun.Mode.SMOKE, "validation", device="cuda", force_hide=force)
        s = r.score
        print(label, "valid", s.valid, "missed", s.missed, "dup", s.duplicates, "invalid", s.invalid,
              "p50_ms %.1f" % s.latency_ms["p50"], "p99_ms %.1f" % s.latency_ms["p99"],
              "send_lag_p99_ms %.2f" % s.send_lag_p99_ms, "rerun", r.first_invalid is not None, flush=True)
    root = Path(".").resolve()
    data = root / "infosys" / "fixtures" / "tiny"
    env = passrun._sanitized_env()
    with tempfile.TemporaryDirectory(prefix="ns-probe-") as tmp:
        protect = passrun.protected_paths(root, Path(tmp), env)
        for kind in ("umount", "nested", "lat"):
            for mode, prot in (("inside", protect), ("hidden-only", None), ("outside", "none")):
                if kind != "lat" and mode != "inside":
                    continue
                cmd = [sys.executable, __file__, "--child", kind, str(data), mode]
                if prot != "none":
                    cmd = passrun.hidden(cmd, data, prot)
                r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=root)
                out = [ln for ln in r.stdout.splitlines() if ln.startswith(("UMOUNT", "NESTED", "LAT"))]
                print("\n".join(out) if out else f"HIDDEN {kind} {mode} rc {r.returncode} (no output)", flush=True)
    print("PROBE_DONE")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
