"""Experiment logging + offline IPS evaluation (Person 3, Milestone 4).

Two jobs, one module:

1. OFFLINE EVALUATION -- the `starter/offline_eval.ipynb` metrics, reusable
   The held-out set is the **last 7 days** of `shop.history_sessions`
   (policies must be built from data before the cut), exactly as in the
   notebook.  `evaluate` / `evaluate_policy` reproduce its estimators:

     * hit_rate@10     -- of the impressions where the shopper clicked, how
                          often the clicked product is in the policy's top-10
                          (easy, but biased towards popular products);
     * ips_ctr_slot1   -- inverse-propensity estimate of the slot-1 click rate:
                          keep only impressions where the policy's first
                          product equals the one the OLD system put in slot 1,
                          weight each by 1 / propensity[0];
     * snips_ctr_slot1 -- self-normalised IPS (lower variance);
     * slot1_matches   -- how many impressions contributed (few matches = a
                          noisy IPS: always report it next to the estimate).

   One deliberate divergence from the notebook: it divides by propensity[0]
   even on no-match rows, so any matched row with propensity 0 yields
   0/0 = NaN (the RuntimeWarning seen in the notebook) and poisons the whole
   IPS column.  Here a no-match row contributes weight 0 WITHOUT dividing;
   a matched row with zero propensity simply carries weight 0 as well.

   Notebook wiring (two lines at the top of a cell):
       import sys; sys.path.insert(0, "..")
       from src.experiment_logger import ExperimentLogger, evaluate

2. EXPERIMENT LOG -- logs/experiments.jsonl + team_12.scratch
   `ExperimentLogger` appends one JSON line per event, so offline runs and
   live KPI snapshots stay comparable across the hackathon.  With a
   RawTreeClient attached the same record is ALSO inserted into
   team_12.scratch (kind-tagged rows; best-effort: insert failures are
   swallowed; the local JSONL stays the source of truth).

3. LIVE KPIS -- team_NN.feedback, read-only SQL (never the request path)
   `fetch_live_kpis(client)` aggregates a trailing window: conversion,
   revenue, latency reliability and served_by / fallback splits.  Every query
   is independently guarded so one rejected dialect detail cannot blank the
   rest of the report.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .client import RawTreeClient, Record

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = ROOT / "logs" / "experiments.jsonl"
DEFAULT_TABLE = "scratch"

# --------------------------------------------------------------------------- #
# Offline evaluation (identical semantics to starter/offline_eval.ipynb)       #
# --------------------------------------------------------------------------- #
HOLDOUT_DAYS = 7              # the notebook's train/test cut
DEFAULT_TEST_LIMIT = 100_000  # random impressions sampled after the cut
_EPS = 1e-12

LAST_HISTORY_DAY_SQL = (
    "SELECT toDate(max(ts::DateTime64(3))) AS d FROM shop.history_sessions"
)

EVAL_SESSIONS_SQL = """
SELECT request_id, ts, user_id, context, recent_views,
       shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE ts >= '{cut}'
ORDER BY rand()
LIMIT {limit}
"""
"""The held-out evaluation slice: the notebook's own sampling query."""


def heldout_cut(client: RawTreeClient) -> str:
    """'YYYY-MM-DD' splitting history into train (< cut) and test (>= cut)."""
    rows = client.query(LAST_HISTORY_DAY_SQL)
    last_day = str((rows[0] or {}).get("d") or "")[:10]
    cut = (datetime.strptime(last_day, "%Y-%m-%d").date()
           - timedelta(days=HOLDOUT_DAYS))
    return cut.isoformat()


def eval_sessions_sql(cut: str, limit: int = DEFAULT_TEST_LIMIT) -> str:
    """The held-out sampling SQL for one cut date."""
    cut = str(cut).replace("'", "")
    return EVAL_SESSIONS_SQL.format(cut=cut, limit=int(limit))


def fetch_eval_sessions(client: RawTreeClient, *, cut: str | None = None,
                        limit: int = DEFAULT_TEST_LIMIT) -> list[Record]:
    """Held-out sessions, same columns as the notebook's `test` frame."""
    return client.query(eval_sessions_sql(cut or heldout_cut(client), limit))


def _clean_outcome(value: Any) -> str | None:
    """'' / None / NaN -> None (the notebook: df.clicked.replace('', None))."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _pf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_policy(recs: list, shown: list, propensity: list,
                    clicked: list) -> dict[str, Any]:
    """The notebook's `evaluate` math over parallel per-impression columns.

    `recs[i]` is one policy answer (ordered product ids), `shown[i]` what the
    old system displayed, `propensity[i]` its slot propensities, `clicked[i]`
    the clicked product (or ''/None). IPS divides nothing on no-match rows,
    so zero propensities can no longer produce NaN.
    """
    n = len(recs)
    hit_num = hit_den = matches = 0
    w_sum = wr_sum = 0.0
    for i in range(n):
        rec = [str(p) for p in _as_list(recs[i])]
        shown_i = [str(p) for p in _as_list(shown[i])]
        props = [_pf(p) for p in _as_list(propensity[i])]
        clicked_i = _clean_outcome(clicked[i]) if i < len(clicked) else None

        if clicked_i is not None:
            hit_den += 1
            if clicked_i in rec:
                hit_num += 1

        first = shown_i[0] if shown_i else ""
        if rec and rec[0] == first:
            matches += 1
            prop = props[0] if props else 0.0
            weight = (1.0 / prop) if prop > _EPS else 0.0
            reward = 1.0 if clicked_i == first else 0.0
            w_sum += weight
            wr_sum += weight * reward
    return {
        "n_test": n,
        "hit_rate@10": (hit_num / hit_den) if hit_den else math.nan,
        "ips_ctr_slot1": (wr_sum / n) if n else math.nan,
        "snips_ctr_slot1": (wr_sum / max(w_sum, 1e-9)) if n else math.nan,
        "slot1_matches": matches,
    }


def _field(row: Any, key: str) -> Any:
    """Read a column from a dict row or a pandas itertuples row."""
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def evaluate(policy: Callable[[Any], list[str]],
             rows: Iterable[Any]) -> dict[str, Any]:
    """Notebook-style entry point: one policy callable, one iterable of rows.

    Works with the notebook's `df.itertuples()` frames AND with the raw record
    lists returned by `fetch_eval_sessions`, so offline runs share one code
    path with the notebook instead of a copied formula.
    """
    rows = list(rows)
    return evaluate_policy(
        [policy(r) for r in rows],
        [_field(r, "shown") for r in rows],
        [_field(r, "propensity") for r in rows],
        [_field(r, "clicked") for r in rows],
    )


# --------------------------------------------------------------------------- #
# LIVE KPIS (read-only SQL over team_NN.feedback -- reports/background use)    #
# --------------------------------------------------------------------------- #
DEFAULT_WINDOW_HOURS = 24.0
GRACE_MS = 3_000          # answer deadline the leaderboard enforces
FEEDBACK_WARMUP_LIMIT = 2_000  # rows folded for the report's bandit KPIs

_KPI_FROM_WHERE = """FROM {team}.feedback
WHERE request_id != '__init__'
  AND ts::DateTime64(3) > toDateTime64('{since}', 3)"""

KPI_MAIN_SQL = """
SELECT served_by::String AS served_by,
       count() AS requests,
       countIf(clicked::String != '') AS clicks,
       countIf(purchased::String != '') AS purchases,
       sum(revenue_eur::Float64) AS revenue_eur
""" + _KPI_FROM_WHERE + "\nGROUP BY served_by"
"""Conversion + revenue per served_by bucket (validated dialect patterns)."""

KPI_LATENCY_SQL = """
SELECT served_by::String AS served_by,
       quantile(0.5)(latency_ms::Float64)  AS latency_p50_ms,
       quantile(0.95)(latency_ms::Float64) AS latency_p95_ms,
       quantile(0.99)(latency_ms::Float64) AS latency_p99_ms,
       countIf(latency_ms::Float64 IS NOT NULL) AS with_latency,
       countIf(latency_ms::Float64 < 3000) AS on_time
""" + _KPI_FROM_WHERE + "\nGROUP BY served_by"
"""Reliability per served_by bucket; latency_ms is NULL for missed requests."""

KPI_FALLBACK_SQL = """
SELECT fallback_reason::String AS reason, count() AS requests
""" + _KPI_FROM_WHERE + "\nGROUP BY reason\nORDER BY requests DESC"
"""Why the baseline served our shoppers (empty reason = we served)."""

KPI_HOURLY_SQL = """
SELECT toStartOfHour(ts::DateTime64(3)) AS hour,
       count() AS requests,
       countIf(clicked::String != '') AS clicks,
       countIf(purchased::String != '') AS purchases,
       sum(revenue_eur::Float64) AS revenue_eur
""" + _KPI_FROM_WHERE + "\nGROUP BY hour\nORDER BY hour DESC\nLIMIT 48"
"""Hourly conversion trend for the report's trend table."""

KPI_COLD_SQL = """
SELECT r.is_new_user::String AS is_new_user,
       f.served_by::String AS served_by,
       count() AS requests,
       countIf(f.clicked::String != '') AS clicks,
       countIf(f.purchased::String != '') AS purchases
FROM {team}.feedback AS f
INNER JOIN shop.requests AS r
        ON r.request_id::String = f.request_id::String
WHERE f.request_id != '__init__'
  AND f.ts::DateTime64(3) > toDateTime64('{since}', 3)
GROUP BY is_new_user, served_by
"""
"""Cold-start lift: feedback joined to shop.requests.is_new_user, so the jury
sees our conversion on genuinely cold shoppers vs the baseline's."""


SLOT_CTR_SQL = """
SELECT slot, count() AS impressions,
       countIf(pid != '' AND pid == clicked) AS clicks
FROM
(
    SELECT
        arrayMap(x -> coalesce(x, ''),
                 CAST(shown, 'Array(String)')) AS shown_arr,
        coalesce(clicked::String, '') AS clicked
    FROM shop.history_sessions
    WHERE ts::DateTime64(3) >= toDateTime64('{since}', 3)
)
ARRAY JOIN shown_arr AS pid, arrayEnumerate(shown_arr) AS slot
GROUP BY slot
ORDER BY slot
"""
"""Per-slot position CTR over history (position bias that the bandit's slot
decay mirrors). Uses the M1-validated cast pattern for schemaless arrays."""


def feedback_since_utc(hours: float) -> str:
    """'YYYY-MM-DD HH:MM:SS' N hours back, the format toDateTime64 accepts."""
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    return start.strftime("%Y-%m-%d %H:%M:%S")


def _q(text: str) -> str:
    return str(text).replace("'", "")


def kpi_main_sql(team: str, since: str) -> str:
    return KPI_MAIN_SQL.format(team=team, since=_q(since))


def kpi_latency_sql(team: str, since: str) -> str:
    return KPI_LATENCY_SQL.format(team=team, since=_q(since))


def kpi_fallback_sql(team: str, since: str) -> str:
    return KPI_FALLBACK_SQL.format(team=team, since=_q(since))


def kpi_hourly_sql(team: str, since: str) -> str:
    return KPI_HOURLY_SQL.format(team=team, since=_q(since))


def kpi_cold_sql(team: str, since: str) -> str:
    return KPI_COLD_SQL.format(team=team, since=_q(since))


def history_slot_ctr_sql(since: str) -> str:
    return SLOT_CTR_SQL.format(since=_q(since))


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def fetch_live_kpis(client, *, window_hours: float = DEFAULT_WINDOW_HOURS,
                    team: str | None = None) -> dict[str, Any]:
    """Trailing-window KPIs from team_NN.feedback; degraded, never raising.

    Returns flat totals (requests, ctr, cvr, revenue_per_request, on_time_rate,
    our_share / fallback_share, ...) plus `by_served_by` buckets with latency
    percentiles, `fallback_reasons` counts and an `hourly` trend.  Query
    failures are collected in `errors` instead of aborting.
    """
    team = team or client.team_id
    since = feedback_since_utc(window_hours)
    out: dict[str, Any] = {"team": team, "window_hours": float(window_hours),
                           "window_start_utc": since, "grace_ms": GRACE_MS,
                           "errors": []}
    buckets: dict[str, dict[str, Any]] = {}
    totals = {"requests": 0, "clicks": 0, "purchases": 0, "revenue_eur": 0.0,
              "on_time": 0, "with_latency": 0}

    try:
        for row in client.query(kpi_main_sql(team, since)):
            sb = str(row.get("served_by") or "unknown")
            bucket = buckets.setdefault(sb, {"served_by": sb})
            n = _int(row.get("requests"))
            c = _int(row.get("clicks"))
            p = _int(row.get("purchases"))
            rev = _pf(row.get("revenue_eur"))
            bucket.update(requests=n, clicks=c, purchases=p, revenue_eur=rev)
            bucket["ctr"] = round(c / n, 4) if n else 0.0
            bucket["cvr"] = round(p / n, 4) if n else 0.0
            bucket["revenue_per_request"] = round(rev / n, 4) if n else 0.0
            totals["requests"] += n
            totals["clicks"] += c
            totals["purchases"] += p
            totals["revenue_eur"] += rev
    except Exception as exc:  # noqa: BLE001 -- report it, do not crash
        out["errors"].append(f"main: {exc}")

    try:
        for row in client.query(kpi_latency_sql(team, since)):
            sb = str(row.get("served_by") or "unknown")
            bucket = buckets.setdefault(sb, {"served_by": sb})
            with_latency = _int(row.get("with_latency"))
            on_time = _int(row.get("on_time"))
            bucket["latency_p50_ms"] = _pf(row.get("latency_p50_ms"))
            bucket["latency_p95_ms"] = _pf(row.get("latency_p95_ms"))
            bucket["latency_p99_ms"] = _pf(row.get("latency_p99_ms"))
            bucket["with_latency"] = with_latency
            bucket["on_time"] = on_time
            bucket["on_time_rate"] = ((on_time / with_latency)
                                      if with_latency else math.nan)
            totals["with_latency"] += with_latency
            totals["on_time"] += on_time
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"latency: {exc}")

    out["by_served_by"] = buckets
    req = totals["requests"]
    clicks = totals["clicks"]
    purchases = totals["purchases"]
    fallback_n = buckets.get("fallback", {}).get("requests", 0)
    out.update(requests=req, clicks=clicks, purchases=purchases,
               revenue_eur=round(totals["revenue_eur"], 2),
               ctr=round(clicks / req, 4) if req else 0.0,
               cvr=round(purchases / req, 4) if req else 0.0,
               revenue_per_request=(round(totals["revenue_eur"] / req, 4)
                                    if req else 0.0),
               with_latency=totals["with_latency"],
               on_time=totals["on_time"],
               on_time_rate=(totals["on_time"] / totals["with_latency"]
                             if totals["with_latency"] else math.nan),
               fallback_requests=fallback_n,
               our_requests=req - fallback_n,
               our_share=round((req - fallback_n) / req, 4) if req else 0.0,
               fallback_share=round(fallback_n / req, 4) if req else 0.0)

    reasons: dict[str, int] = {}
    try:
        for row in client.query(kpi_fallback_sql(team, since)):
            reasons[str(row.get("reason") or "")] = _int(row.get("requests"))
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"fallback: {exc}")
    out["fallback_reasons"] = reasons

    hourly: list[dict[str, Any]] = []
    try:
        for row in client.query(kpi_hourly_sql(team, since)):
            n = _int(row.get("requests"))
            hourly.append({"hour": str(row.get("hour") or ""),
                           "requests": n,
                           "clicks": _int(row.get("clicks")),
                           "purchases": _int(row.get("purchases")),
                           "ctr": round(_int(row.get("clicks")) / n, 4) if n else 0.0,
                           "revenue_eur": round(_pf(row.get("revenue_eur")), 2)})
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"hourly: {exc}")
    out["hourly"] = hourly
    return out


def fetch_cold_lift(client: RawTreeClient, *,
                    window_hours: float = DEFAULT_WINDOW_HOURS,
                    team: str | None = None) -> dict[str, Any]:
    """Cold vs warm conversion, our policy vs the baseline (chart data).

    Joins the trailing feedback window to `shop.requests.is_new_user` -- the
    only live source of "this shopper was cold".  Guarded like the other KPI
    queries: a rejected join dialect degrades to `errors`, never a crash.
    """
    team = team or client.team_id
    since = feedback_since_utc(window_hours)
    out: dict[str, Any] = {"team": team, "window_hours": float(window_hours),
                           "window_start_utc": since, "cells": {}, "errors": []}
    try:
        rows = client.query(kpi_cold_sql(team, since))
    except Exception as exc:  # noqa: BLE001 -- chart skips, report stays up
        out["errors"].append(f"cold join: {exc}")
        return out
    for row in rows:
        cold = str(row.get("is_new_user") or "").strip().lower() == "true"
        sb = str(row.get("served_by") or "unknown")
        n = _int(row.get("requests"))
        c = _int(row.get("clicks"))
        p = _int(row.get("purchases"))
        cell = out["cells"].setdefault(("cold" if cold else "warm", sb),
                                       {"segment": "cold" if cold else "warm",
                                        "served_by": sb, "requests": 0,
                                        "clicks": 0, "purchases": 0})
        cell["requests"] += n
        cell["clicks"] += c
        cell["purchases"] += p
    for cell in out["cells"].values():
        n = cell["requests"]
        cell["ctr"] = round(cell["clicks"] / n, 4) if n else 0.0
        cell["cvr"] = round(cell["purchases"] / n, 4) if n else 0.0
    return out


def fetch_slot_ctr(client: RawTreeClient, *, since: str = "2000-01-01"
                   ) -> list[dict[str, Any]]:
    """Per-slot position CTR over history (1 row per slot 1..10, ascending).

    Full history by default: the position-bias curve is a stable property of
    the marketplace, not a trailing window. Guarded: errors -> empty list.
    """
    out: list[dict[str, Any]] = []
    try:
        rows = client.query(history_slot_ctr_sql(since))
    except Exception:  # noqa: BLE001 -- charts degrade silently
        return out
    for row in rows:
        impressions = _int(row.get("impressions"))
        clicks = _int(row.get("clicks"))
        out.append({"slot": _int(row.get("slot")),
                    "impressions": impressions, "clicks": clicks,
                    "ctr": round(clicks / impressions, 6) if impressions else 0.0})
    return out

# --------------------------------------------------------------------------- #
# Experiment log: JSONL source of truth + optional team_12.scratch push         #
# --------------------------------------------------------------------------- #
class ExperimentLogger:
    """Append-only log for offline runs and live KPI snapshots.

    Every `log*` call writes one JSON line to `logs/experiments.jsonl` (the
    source of truth, works fully offline) and -- when a RawTreeClient is
    attached -- inserts the same record into `team_12.scratch` (kind-tagged) so
    experiment timeline is queryable with plain SQL from any team laptop.
    RawTree insert problems are counted (`push_errors`, `last_push_error`)
    and never propagate: logging must never break an experiment.
    """

    def __init__(self, path=DEFAULT_LOG_PATH, client: RawTreeClient | None = None,
                 table: str = DEFAULT_TABLE) -> None:
        self.path = Path(path)
        self.client = client
        self.table = table
        self.push_errors = 0
        self.last_push_error = None
        self._table_ensured = False

    # ---- write side ----------------------------------------------------------
    def log(self, kind: str, name: str, metrics: dict | None = None,
            params: dict | None = None, tags=None,
            ts: str | None = None) -> dict[str, Any]:
        """Append one record; return it (so callers can reuse the ts)."""
        record: dict[str, Any] = {
            "ts": ts or _utc_iso(),
            "kind": kind,
            "type": kind,  # alias: the team spec queries scratch by `type`
            "name": str(name),
            "metrics": dict(metrics or {}),
            "params": dict(params or {}),
            "tags": [str(t) for t in (tags or ())],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # even a read-only disk must not kill an experiment run
        if self.client is not None:
            try:
                self._insert_with_retry(dict(record))
            except Exception as exc:  # noqa: BLE001 -- local log stays truth
                self.push_errors += 1
                self.last_push_error = exc
        return record

    def log_experiment(self, name: str, metrics: dict | None = None,
                       params: dict | None = None, tags=None) -> dict[str, Any]:
        """One offline run: IPS metrics, config, dataset fingerprint..."""
        return self.log("experiment", name, metrics, params, tags)

    def log_live_snapshot(self, kpis: dict, name: str = "live_snapshot",
                          tags=("live",)) -> dict[str, Any]:
        """One live-KPI snapshot (fetch_live_kpis output or BanditState.summary)."""
        return self.log("live_snapshot", name, metrics=kpis, tags=tags)

    def _insert_with_retry(self, row: dict[str, Any]) -> None:
        """Insert one record; if the table is missing, create it once, retry.

        RawTree does not auto-create tables for keys without automatic
        table-creation rights, and the Query API only accepts
        `CREATE TABLE [IF NOT EXISTS] <table> ENGINE = ReplicatedRawMergeTree`
        (raw table, no column definitions; needs an admin-permission key).
        """
        try:
            self.client.insert(self.table, [row])
        except Exception as first_err:  # noqa: BLE001 -- local log stays truth
            if self._ensure_table():
                try:
                    self.client.insert(self.table, [row])
                    return
                except Exception as exc:  # noqa: BLE001
                    self.push_errors += 1
                    self.last_push_error = exc
            else:
                self.push_errors += 1
                self.last_push_error = first_err

    def _ensure_table(self) -> bool:
        """One-time CREATE TABLE IF NOT EXISTS attempt (best effort)."""
        if self._table_ensured:
            return False
        self._table_ensured = True
        try:
            self.client.query(
                f"CREATE TABLE IF NOT EXISTS {self.client.team_id}.{self.table} "
                "ENGINE = ReplicatedRawMergeTree")
            return True
        except Exception:
            return False

    # ---- read side ------------------------------------------------------------
    def load(self) -> list[dict[str, Any]]:
        """All records (oldest first); malformed lines are skipped quietly."""
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            return records
        for line in self.path.read_text(encoding="utf-8",
                                         errors="replace").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except ValueError:
                continue
        return records

    def to_dataframe(self):
        """All records as a pandas DataFrame (lazy import, analysis only)."""
        import pandas as pd

        return pd.DataFrame(self.load())

    def best(self, name: str | None = None, metric: str = "snips_ctr_slot1",
             kind: str = "experiment") -> dict[str, Any] | None:
        """The logged record with the highest `metric` (NaNs ignored)."""
        best_record = None
        best_value = -math.inf
        for record in self.load():
            if record.get("kind") != kind:
                continue
            if name is not None and record.get("name") != name:
                continue
            try:
                number = float((record.get("metrics") or {}).get(metric))
            except (TypeError, ValueError):
                continue
            if number != number:  # NaN
                continue
            if number > best_value:
                best_record, best_value = record, number
        return best_record

    def last(self, kind: str | None = None) -> dict[str, Any] | None:
        """The most recent record (optionally of one kind)."""
        for record in reversed(self.load()):
            if kind is None or record.get("kind") == kind:
                return record
        return None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

