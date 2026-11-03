"""Pre-publish scan. Prints `file:line: rule`, never the matched text. Exit 1 on any finding.

    scan_publish.py [--deny FILE] [--env-file FILE] [--history]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ABS_PATH = re.compile(r"(^|[^\w.~])/(Users|home|home1|work|work1|scratch)/")
DATA_SUFFIXES = (".jsonl.gz", ".npz", ".parquet", ".csv")
MAX_BYTES = 1_000_000
MIN_SECRET_LEN = 8
SELF = "scripts/scan_publish.py"


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def _line_rules(text: str, deny: list[str], secrets: list[str], paths: bool) -> list[tuple[int, str]]:
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if paths and ABS_PATH.search(line):
            found.append((n, "absolute-path"))
        if any(d in low for d in deny):
            found.append((n, "deny-pattern"))
        if any(s in line for s in secrets):
            found.append((n, "env-value"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deny", type=Path)
    ap.add_argument("--env-file", type=Path)
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    deny = [x.strip().lower() for x in args.deny.read_text().splitlines() if x.strip()] if args.deny else []
    secrets = []
    if args.env_file:
        from dotenv import dotenv_values

        secrets = [v for v in dotenv_values(args.env_file).values() if v and len(v) >= MIN_SECRET_LEN]

    findings = []
    for name in _git("ls-files", "-z").decode().split("\0"):
        path = Path(name)
        if not name or not path.is_file():
            continue
        if name.endswith(DATA_SUFFIXES) and not name.startswith("infosys/fixtures/"):
            findings.append(f"{name}:0: data-file-outside-fixtures")
        if path.stat().st_size > MAX_BYTES:
            findings.append(f"{name}:0: file-over-1MB")
        text = path.read_bytes().decode("latin-1")  # 1:1 bytes, so binary files are scanned too
        findings += [f"{name}:{n}: {r}" for n, r in _line_rules(text, deny, secrets, name != SELF)]

    if args.history:
        log = _git("log", "-p", "--all", "--no-color").decode("latin-1")
        findings += [f"history:{n}: {r}" for n, r in _line_rules(log, deny, secrets, True)]

    print("\n".join(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
