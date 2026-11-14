"""RawTree I/O layer: receiving requests (`pull`), deduplicating (`SeenWindow`), and batch `push`."""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from state_manager import load_env

API, TEAM, API_KEY = load_env()

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
http.mount("https://", _adapter)
http.mount("http://", _adapter)

Record = dict[str, Any]

# Connect timeout 0.5s, read timeout 1.2s so a stalled socket never eats the 3s budget
FAST_TIMEOUT: tuple[float, float] = (0.5, 1.2)

SALES = """
SELECT product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'{sales}
GROUP BY product_id"""


def query(sql: str, timeout: float | tuple[float, float] = FAST_TIMEOUT, retries: int = 2) -> list[Record]:
    """Execute a RawTree SQL query with short connect/read timeouts and quick retries."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(0.05)
    assert last_err is not None
    raise last_err


def best_sellers(until: str | None = None) -> list[str]:
    """Every product ever bought, best-selling first (kept for offline_eval.ipynb)."""
    sales_cut = f" AND ts < '{until}'" if until else ""
    sales = sorted(
        ((int(r["n"]), r["product_id"]) for r in query(SALES.format(sales=sales_cut), timeout=15.0)),
        reverse=True,
    )
    return [product for _, product in sales]


def normalize_request(raw: Record) -> Record:
    """Normalize dotted context fields (`context.device`) into a nested `context` dict."""
    ctx = raw.get("context")
    if not isinstance(ctx, dict):
        ctx = {
            "device": raw.get("context.device", ""),
            "region": raw.get("context.region", ""),
            "page": raw.get("context.page", ""),
        }
    recent = raw.get("recent_views")
    return {
        "request_id": str(raw["request_id"]),
        "ts": str(raw.get("ts", "")),
        "user_id": str(raw.get("user_id", "")),
        "is_new_user": bool(raw.get("is_new_user", False)),
        "context": ctx,
        "recent_views": recent if isinstance(recent, list) else [],
    }


class SeenWindow:
    """Rolling deduplication window so memory stays bounded."""

    def __init__(self, maxlen: int = 20000) -> None:
        self._queue: deque[str] = deque()
        self.processed_request_ids: set[str] = set()
        self._maxlen = maxlen

    def add_if_new(self, req_id: str) -> bool:
        if req_id in self.processed_request_ids:
            return False
        if len(self._queue) >= self._maxlen:
            oldest = self._queue.popleft()
            self.processed_request_ids.discard(oldest)
        self._queue.append(req_id)
        self.processed_request_ids.add(req_id)
        return True


# Always filter to the last 3 seconds so:
# 1. After any network blip we never waste time answering already-expired (>3s old) requests.
# 2. Slight out-of-order arrivals within the 3s window are never skipped (SeenWindow deduplicates in O(1)).
SQL_PULL_LIVE = """
SELECT r.request_id::String AS request_id,
       toString(r.ts::DateTime64(3)) AS ts,
       r.user_id::String AS user_id,
       r.is_new_user::Bool AS is_new_user,
       r.context.device::String AS `context.device`,
       r.context.region::String AS `context.region`,
       r.context.page::String AS `context.page`,
       r.recent_views AS recent_views
FROM shop.requests AS r
WHERE r.ts::DateTime64(3) > now() - INTERVAL 3 SECOND
ORDER BY r.ts ASC
LIMIT 60
"""


def pull(seen: SeenWindow, last_ts: str = "") -> tuple[list[Record], str]:
    """Fetch fresh (<3s old), unseen requests from shop.requests."""
    rows = query(SQL_PULL_LIVE, timeout=FAST_TIMEOUT, retries=2)
    fresh: list[Record] = []
    newest_ts = last_ts
    seen_add = seen.add_if_new
    append = fresh.append
    for raw in rows:
        req = normalize_request(raw)
        ts = req["ts"]
        if ts > newest_ts:
            newest_ts = ts
        if seen_add(req["request_id"]):
            append(req)
    return fresh, newest_ts


def push(answers: list[Record], retries: int = 2) -> None:
    """Batch insert answers into team_NN.submissions with fast timeout and retry."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = http.post(
                f"{API}/v1/tables/submissions",
                params={"database": TEAM},
                json=answers,
                timeout=FAST_TIMEOUT,
            )
            r.raise_for_status()
            return
        except Exception as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(0.05)
    assert last_err is not None
    raise last_err
