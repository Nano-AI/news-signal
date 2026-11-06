"""Prepare: builds the optional fused HIP kernel library (nothing is trained)."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "kernels" / "bert_fused.hip"


def build_kernels(out: Path) -> bool:
    hipcc = shutil.which("hipcc") or ("/opt/rocm/bin/hipcc" if Path("/opt/rocm/bin/hipcc").exists() else None)
    if hipcc is None:
        print("prepare: hipcc not found; skipping fused kernels", file=sys.stderr)
        return False
    cmd = [hipcc, "-O3", "--offload-arch=gfx90a", "-shared", "-fPIC", str(SRC), "-o", str(out / "bert_fused.so")]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"prepare: fused kernel build failed: {getattr(e, 'stderr', e)}", file=sys.stderr)
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(prog="serving.prepare")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", required=True, choices=["cuda", "cpu"])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda":
        build_kernels(out)


if __name__ == "__main__":
    main()
