"""SEC EDGAR: ticker-to-CIK map and per-company submissions (8-K filings)."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from infosys.fetch.http import CachedClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"

# UNVERIFIED (Step 5 blocked by SEC 403): rule to confirm is "acceptanceDateTime wall-clock is in this zone".
ACCEPTANCE_TZ = "UTC"

ITEM_TITLES: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a Registrant",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "5.05": "Amendments to the Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or Other External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}  # fmt: skip


def contact_email(override: str | None) -> str:
    if override:
        return override
    out = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, check=False)
    email = out.stdout.strip()
    if not email:
        raise ValueError("no contact email: pass --contact-email or set git config user.email")
    return email


def acceptance_utc(acceptance: str, tz: str = ACCEPTANCE_TZ) -> datetime:
    """Convert a submissions `acceptanceDateTime` to UTC, reading its wall-clock part in `tz`."""
    naive = datetime.fromisoformat(acceptance.removesuffix("Z")).replace(tzinfo=None, microsecond=0)
    return naive.replace(tzinfo=ZoneInfo(tz)).astimezone(UTC)


def fetch_8k(client: CachedClient, tickers: Sequence[str], start: date, end: date, email: str) -> int:
    """Fetch every needed submissions file; returns the number of JSON documents read.

    `start`/`end` only skip history files that cannot overlap the window; build applies the exact window.
    """
    headers = {"User-Agent": f"news-signal research {email}"}
    wanted = set(tickers)
    ciks = {
        int(row["cik_str"])
        for row in client.get_json(TICKERS_URL, {}, headers).values()
        if row["ticker"] in wanted
    }
    docs = 1
    for cik in sorted(ciks):
        main = client.get_json(f"{SUBMISSIONS_URL}CIK{cik:010d}.json", {}, headers)
        docs += 1
        for f in main["filings"]["files"]:
            if date.fromisoformat(f["filingTo"]) < start or date.fromisoformat(f["filingFrom"]) >= end:
                continue
            client.get_json(f"{SUBMISSIONS_URL}{f['name']}", {}, headers)
            docs += 1
    return docs
