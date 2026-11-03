"""Pull Alpaca news and bars plus EDGAR submissions into a resumable cache. Prints counts only."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

from infosys.fetch import alpaca, edgar
from infosys.fetch.http import CachedClient
from infosys.universe import TICKERS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m infosys.fetch")
    ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date(), help="exclusive")
    ap.add_argument("--contact-email")
    ap.add_argument("--skip-edgar", action="store_true")
    args = ap.parse_args(argv)

    headers = alpaca.load_keys(args.env_file)
    client = CachedClient(args.cache)
    email = None
    if not args.skip_edgar:
        email = edgar.contact_email(args.contact_email)
        edgar.preflight(client, email)  # fail now, not hours in

    def log(line: str) -> None:
        print(f"{line} requests={client.network_requests} hits={client.cache_hits}", flush=True)

    alpaca.fetch_news(client, args.start, args.end, headers=headers, log=log)
    for ticker in TICKERS:
        log(f"bars {ticker}: pages={alpaca.fetch_bars(client, ticker, args.start, args.end, headers=headers)}")
    if email is not None:
        log(f"edgar: docs={edgar.fetch_8k(client, TICKERS, args.start, args.end, email)}")
    print("fetch complete")


if __name__ == "__main__":
    main()
