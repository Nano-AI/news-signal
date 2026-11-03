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
    args = ap.parse_args(argv)

    headers = alpaca.load_keys(args.env_file)
    email = edgar.contact_email(args.contact_email)  # EDGAR runs last so a block there cannot stop the Alpaca pull
    client = CachedClient(args.cache)

    def progress(label: str, pages: int) -> None:
        print(f"{label}: pages={pages} requests={client.network_requests} hits={client.cache_hits}", flush=True)

    progress("news", alpaca.fetch_news(client, args.start, args.end, headers=headers))
    for ticker in TICKERS:
        progress(f"bars {ticker}", alpaca.fetch_bars(client, ticker, args.start, args.end, headers=headers))
    progress("edgar", edgar.fetch_8k(client, TICKERS, args.start, args.end, email))
    print("fetch complete")


if __name__ == "__main__":
    main()
