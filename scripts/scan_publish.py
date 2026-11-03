"""Pre-publish scan. Prints `file:line: rule`, never the matched text. Exit 1 on any finding.

    scan_publish.py [--deny FILE] [--history]

Rules: absolute paths, deny tokens (whole words), key-shaped secrets, data files outside the fixtures,
files over 1 MB.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

ABS_PATH = re.compile(r"(^|[^\w.~])/(Users|home|home1|work|work1|scratch)/")
ALPACA_KEY_ID = re.compile(r"(?<![A-Za-z0-9])(PK|AK)[A-Z0-9]{16,}(?![A-Za-z0-9])")
BASE62_40 = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9]{40}(?![A-Za-z0-9+/_-])")
SECRET_ASSIGNMENT = re.compile(
    r"\w*(?:key|secret|token|passw(?:or)?d)\w*\s*[:=]\s*[\"']?([A-Za-z0-9+/_=-]{20,})", re.IGNORECASE
)
MIN_ENTROPY = 3.5  # bits per character
DATA_SUFFIXES = (".jsonl.gz", ".npz", ".parquet", ".csv")
MAX_BYTES = 1_000_000
SELF = "scripts/scan_publish.py"


def _entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def _deny_regex(token: str) -> re.Pattern[str]:
    """Whole-word match: a letter may not touch either end, so a token inside a longer word passes."""
    before = r"(?<![A-Za-z])" if token[0].isalpha() else ""
    after = r"(?![A-Za-z])" if token[-1].isalpha() else ""
    return re.compile(before + re.escape(token) + after, re.IGNORECASE)


def _secret_shaped(line: str) -> bool:
    if ALPACA_KEY_ID.search(line):
        return True
    for m in BASE62_40.finditer(line):
        v = m.group()
        if v.lower() != v and v.upper() != v and any(ch.isdigit() for ch in v):
            return True
    return any(
        _entropy(m.group(1)) >= MIN_ENTROPY and any(c.isdigit() for c in m.group(1))
        for m in SECRET_ASSIGNMENT.finditer(line)
    )


def scan_text(text: str, deny: Iterable[str] = (), check_paths: bool = True) -> list[tuple[int, str]]:
    patterns = [_deny_regex(d) for d in deny if d.strip()]
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        if check_paths and ABS_PATH.search(line):
            found.append((n, "absolute-path"))
        if any(p.search(line) for p in patterns):
            found.append((n, "deny-pattern"))
        if _secret_shaped(line):
            found.append((n, "secret-shaped"))
    return found


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deny", type=Path)
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    deny = [x.strip() for x in args.deny.read_text().splitlines() if x.strip()] if args.deny else []

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
        findings += [f"{name}:{n}: {r}" for n, r in scan_text(text, deny, check_paths=name != SELF)]

    if args.history:
        log = _git("log", "-p", "--all", "--no-color").decode("latin-1")
        findings += [f"git-log:{n}: {r}" for n, r in scan_text(log, deny)]

    print("\n".join(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
