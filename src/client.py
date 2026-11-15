"""Shared RawTree HTTP client (Person 3 / M1).

Small, dependency-light wrapper used by every `src/` module:
    * query(sql)         -> list[dict] rows (POST /v1/query, retrying)
    * query_df(sql)      -> pandas.DataFrame (lazy pandas import)
    * insert(table, rows)-> POST /v1/tables/{table}?database=team_NN
    * team_id            -> from TEAM_ID in .env (used to build SQL)

Credentials come from the repo-root `.env` (RAWTREE_API_URL, TEAM_ID,
RAWTREE_API_KEY). A tiny fallback parser runs after python-dotenv so the
client also works when dotenv is unavailable.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

try:  # python-dotenv is installed, but stay importable without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

Record = dict[str, Any]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API = "https://api.rawtree.com"


class RawTreeError(RuntimeError):
    """Raised after all retries for a query, or on a failed insert."""



def _load_env() -> None:
    """Load the repo-root .env; plain parser fallback keeps env untouched."""
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


class RawTreeClient:
    """Retrying RawTree REST client shared by all team modules."""

    def __init__(self, api_url: str | None = None, team_id: str | None = None,
                 api_key: str | None = None) -> None:
        _load_env()
        self.api = (api_url or os.environ.get("RAWTREE_API_URL", DEFAULT_API)).rstrip("/")
        self.team_id = (team_id or os.environ.get("TEAM_ID", "")).strip()
        self.api_key = (api_key or os.environ.get("RAWTREE_API_KEY", "")).strip()
        if not self.team_id or not self.api_key:
            raise RuntimeError("TEAM_ID / RAWTREE_API_KEY missing from .env")
        self.queries_made = 0
        self.inserts_made = 0
        self._sess = requests.Session()
        self._sess.headers["Authorization"] = f"Bearer {self.api_key}"
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        self._sess.mount("https://", adapter)
        self._sess.mount("http://", adapter)

    # ---- core -------------------------------------------------------------- #
    def query(self, sql: str, *, timeout: float = 20.0,
              retries: int = 3) -> list[Record]:
        """Run one SQL statement; return the `data` rows (retrying on errors)."""
        last_err: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                r = self._sess.post(f"{self.api}/v1/query", json={"sql": sql},
                                    timeout=timeout)
                r.raise_for_status()
                self.queries_made += 1
                return r.json().get("data", [])
            except Exception as exc:  # noqa: BLE001 -- retry any transport error
                last_err = exc
                if attempt + 1 < max(1, retries):
                    time.sleep(0.15 * (attempt + 1))
        assert last_err is not None
        raise RawTreeError(f"query failed after {retries} attempts: {last_err}") from last_err

    def query_df(self, sql: str, *, timeout: float = 20.0, retries: int = 3):
        """Same as `query` but as a pandas DataFrame (pandas imported lazily)."""
        import pandas as pd

        return pd.DataFrame(self.query(sql, timeout=timeout, retries=retries))

    def insert(self, table: str, rows: list[Record],
               *, timeout: float = 10.0) -> None:
        """Append rows to `{team_id}.{table}` (one batch POST, no retry loop —
        callers implement their own retry semantics)."""
        if not rows:
            return
        r = self._sess.post(
            f"{self.api}/v1/tables/{table}",
            params={"database": self.team_id},
            json=list(rows),
            timeout=timeout,
        )
        r.raise_for_status()
        self.inserts_made += 1

    def close(self) -> None:
        self._sess.close()


def default_client() -> RawTreeClient:
    """One shared client per process (used by scripts/probe_*.py)."""
    global _DEFAULT_CLIENT
    try:
        return _DEFAULT_CLIENT
    except NameError:
        _DEFAULT_CLIENT = RawTreeClient()
        return _DEFAULT_CLIENT
