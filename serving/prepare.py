"""Prepare: no training; builds and verifies the native runtime (failure only disables it)."""

import argparse
from pathlib import Path

from serving.native import prepare_native


def main() -> None:
    p = argparse.ArgumentParser(prog="serving.prepare")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", required=True, choices=["cuda", "cpu"])
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    prepare_native(a.train, a.out, a.device)


if __name__ == "__main__":
    main()
