"""Query-keyed JSON cache with rate limiting and retries. Stdlib only; headers are never stored."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BACKOFF_S = (2, 4, 8, 16, 32)
Transport = Callable[[str, Mapping[str, str]], tuple[int, bytes]]


class FetchError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


def urllib_transport(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
    """GET `url`. Connection failures map to 599 so the client retries them like a 5xx."""
    try:
        with urlopen(Request(url, headers=dict(headers)), timeout=60) as r:  # noqa: S310
            return r.status, r.read()
    except HTTPError as e:
        return e.code, b""
    except OSError:  # URLError, timeouts, resets
        return 599, b""


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        min_interval_s: float = 0.4,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._dir = cache_dir
        self._min_interval_s = min_interval_s
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._last_start: float | None = None
        self.network_requests = 0
        self.cache_hits = 0

    def _pace(self) -> None:
        if self._last_start is not None:
            wait = self._min_interval_s - (self._clock() - self._last_start)
            if wait > 0:
                self._sleep(wait)
        self._last_start = self._clock()

    def get_json(self, url: str, params: Mapping[str, str], headers: Mapping[str, str]) -> dict:
        query = urlencode(sorted(params.items()))
        key = hashlib.sha256(f"{url}?{query}".encode()).hexdigest()
        path = self._dir / key[:2] / f"{key}.json"
        if path.is_file():  # only a finished record has this name; `.tmp` files never match
            self.cache_hits += 1
            return json.loads(path.read_text())["body"]
        full = f"{url}?{query}" if query else url
        for delay in (*BACKOFF_S, None):
            self._pace()
            self.network_requests += 1
            status, raw = self._transport(full, headers)
            if status == 200:
                break
            if delay is None or not (status == 429 or status >= 500):
                raise FetchError(status, full)
            self._sleep(delay)
        record = {"url": url, "params": dict(params), "status": status, "body": json.loads(raw)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, path)
        return record["body"]
