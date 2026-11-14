"""Shared RawTree HTTP client.

Every module (cold_start, bandit, jury_report, ...) talks to RawTree through this
thin wrapper so that auth, timeouts and error handling live in exactly one place.

RawTree conventions (from https://rawtree.com/llms.txt and starter/recommender.py):
  * POST {API}/v1/query  with {"sql": "..."}           -> read/query
  * POST {API}/v1/tables/{table}?database=team_NN        -> insert rows (list of dicts)
  * Auth: `Authorization: Bearer $RAWTREE_API_KEY` header
  * Query response: {"meta": [...], "data": [ {col: value, ...}, ... ], "rows": N, ...}
  * Schemaless columns: filters/ORDER BY mostly work as-is, but JOIN keys and some
    aggregates need explicit casts, e.g. `product_id::String`, `ts::DateTime64(3)`.

Credentials come from the environment; the project root `.env` is loaded first
(same pattern as starter/recommender.py, no extra dependency beyond python-dotenv).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests
import requests

# Load `.env` from the project root (parent of src/) if present. If python-dotenv
# is missing on some machine, fall back to the same minimal parsing the starter uses.
_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ModuleNotFoundError:
    _env_file = _ROOT / ".env"
    for _line in _env_file.read_text().splitlines() if _env_file.exists() else []:
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip())

Record = dict[str, Any]
"""One RawTree row, as returned by the Query API: {"column": value, ...}."""


class RawTreeError(RuntimeError):
    """Raised when RawTree answers with a non-2xx status or a malformed body."""


def _from_env(key: str, default: str | None = None) -> str:
    """Read a config value from the environment, falling back to .env defaults."""
    value = os.environ.get(key, default)
    if value is None:
        raise RawTreeError(f"missing required environment variable {key!r}")
    return value


class RawTreeClient:
    """Minimal, retrying client for the RawTree Query + Ingest APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        team_id: str | None = None,
        api_url: str | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self.api_url = (api_url or os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")).rstrip("/")
        self.team_id = team_id or _from_env("TEAM_ID")
        self.timeout = timeout
        self.max_retries = max_retries

        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {api_key or _from_env('RAWTREE_API_KEY')}"

    # ------------------------------------------------------------------ reads
    def query(self, sql: str, *, retries: int | None = None) -> list[Record]:
        """Run a read-only SQL query, return the `data` rows as list of dicts.

        Retries transient failures (5xx / connection errors) with a short linear
        backoff. 4xx errors (bad SQL, auth) raise immediately: retrying those
        would just burn time.
        """
        attempts = self.max_retries if retries is None else retries
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                r = self.http.post(f"{self.api_url}/v1/query", json={"sql": sql}, timeout=self.timeout)
                if r.status_code >= 400:
                    raise RawTreeError(f"query failed [{r.status_code}]: {r.text[:300]} for sql: {sql[:120]}...")
                return r.json()["data"]
            except (requests.ConnectionError, requests.Timeout) as err:  # transient
                last_err = err
                time.sleep(0.5 * (attempt + 1))
        raise RawTreeError(f"query failed after {attempts} attempts: {last_err}") from last_err

    def query_df(self, sql: str):
        """Same as query() but returns a pandas DataFrame (for offline analysis)."""
        import pandas as pd  # imported lazily: the live hot path never needs pandas

        return pd.DataFrame(self.query(sql))

    # ---------------------------------------------------------------- writes
    def insert(self, table: str, rows: list[Record], *, database: str | None = None) -> None:
        """Batch-insert rows into `<database>.{table}` (defaults to the team DB)."""
        if not rows:
            return
        db = database or self.team_id
        r = self.http.post(
            f"{self.api_url}/v1/tables/{table}",
            params={"database": db},
            json=rows,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise RawTreeError(f"insert into {db}.{table} failed [{r.status_code}]: {r.text[:300]}")


def default_client() -> RawTreeClient:
    """Client built from environment/.env — the one entry point most modules use."""
    return RawTreeClient()


if __name__ == "__main__":  # tiny smoke test: `py src/client.py`
    c = default_client()
    print("catalog size:", c.query("SELECT count() AS n FROM shop.catalog"))
