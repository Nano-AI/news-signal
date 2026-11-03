"""No-op prepare: the baseline does not train."""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="serving.prepare")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", required=True, choices=["cuda", "cpu"])
    Path(p.parse_args().out).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
