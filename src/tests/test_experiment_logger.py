"""Offline unit tests for src/experiment_logger.py + src/jury_report.py (M4).

No network: RawTree calls go through a fake client that dispatches canned
rows by SQL substring and records inserts.

Run from the project root:  py -m pytest src/tests/test_experiment_logger.py -v
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bandit import BanditState, ingest_feedback_rows  # noqa: E402
from src.experiment_logger import (  # noqa: E402
    DEFAULT_TABLE,
    ExperimentLogger,
    eval_sessions_sql,
    evaluate,
    evaluate_policy,
    fetch_live_kpis,
    feedback_since_utc,
    kpi_latency_sql,
    heldout_cut,
    kpi_main_sql,
)
from src.jury_report import build_jury_report  # noqa: E402
from src.charts import build_charts, collect_chart_data  # noqa: E402
from src.experiment_logger import (  # noqa: E402
    fetch_cold_lift,
    fetch_slot_ctr,
    history_slot_ctr_sql,
    kpi_cold_sql,
)


class FakeClient:
    """RawTreeClient stand-in: canned rows by SQL substring, records writes."""

    def __init__(self, routes=(), team_id="team_12"):
        self.routes = list(routes)  # (substring, rows), first match wins
        self.queries = []
        self.inserted = []
        self.team_id = team_id

    def query(self, sql):
        self.queries.append(sql)
        for needle, rows in self.routes:
            if needle in sql:
                return rows
        return []

    def insert(self, table, rows, database=None):
        self.inserted.append((table, list(rows), database))


class BrokenClient(FakeClient):
    """Every query and insert fails -- the API is having a bad day."""

    def query(self, sql):
        self.queries.append(sql)
        raise RuntimeError("boom")

    def insert(self, table, rows, database=None):
        self.inserted.append((table, list(rows), database))
        raise RuntimeError("insert boom")


def _hist(i, shown, prop, clicked="", recs=None):
    """One shop.history_sessions-style row (+ optional policy answer)."""
    return {"request_id": f"r{i}", "shown": list(shown),
            "propensity": list(prop), "clicked": clicked}


# ------------------------------------------------------------ evaluation ---- #
def test_eval_matches_hand_computation() -> None:
    recs = [["a", "x"], ["b", "y"], ["z", "d"], ["f", "h"]]
    shown = [["a", "b"], ["b", "c"], ["d", "e"], ["f", "g"]]
    prop = [[0.5, 0.5], [0.25, 0.75], [0.0, 0.5], [0.1, 0.9]]
    clicked = ["", "b", "z", "f"]
    m = evaluate_policy(recs, shown, prop, clicked)
    # slot-1 matches: rows 0, 1, 3 (key: rec[0] == shown[0])
    # weights: 1/0.5=2 (no reward), 1/0.25=4 (reward), --, 1/0.1=10 (reward)
    assert m["n_test"] == 4
    assert m["hit_rate@10"] == pytest.approx(1.0)      # clicked rows: all 3 hit
    assert m["ips_ctr_slot1"] == pytest.approx(14.0 / 4)
    assert m["snips_ctr_slot1"] == pytest.approx(14.0 / 16.0)
    assert m["slot1_matches"] == 3


def test_zero_propensity_matched_row_carries_zero_weight_not_nan() -> None:
    m = evaluate_policy([["a", "x"]], [["a", "b"]], [[0.0, 1.0]], ["a"])
    assert m["slot1_matches"] == 1
    assert m["ips_ctr_slot1"] == 0.0
    assert m["snips_ctr_slot1"] == 0.0
    assert not any(isinstance(v, float) and math.isnan(v) for v in m.values())


def test_no_match_row_never_divides_by_propensity() -> None:
    m = evaluate_policy([["z", "q"]], [["a", "b"]], [[0.0, 0.0]], ["z"])
    assert m["slot1_matches"] == 0
    assert m["hit_rate@10"] == 1.0    # clicked z, z is in our top-10
    assert m["ips_ctr_slot1"] == 0.0


def test_empty_inputs_return_nan_not_crash() -> None:
    m = evaluate_policy([], [], [], [])
    assert m["n_test"] == 0
    assert math.isnan(m["hit_rate@10"])
    assert math.isnan(m["ips_ctr_slot1"])


def test_notebook_formula_equivalence_on_random_rows() -> None:
    """When no propensity is 0, we must match the notebook formula exactly."""
    rng = np.random.default_rng(7)
    n, prods = 200, 40
    ids = [f"p_{i:02d}" for i in range(prods)]
    shown, prop, clicked, recs = [], [], [], []
    for i in range(n):
        s = list(rng.choice(ids, size=10, replace=False))
        p = list(rng.uniform(0.01, 1.0, size=10))
        p[0] = max(p)  # notebook-like: slot 1 often the largest
        r = list(rng.choice(ids, size=10, replace=False))
        if rng.random() < 0.5:
            r = s  # half the answers agree with the log
        c = ""
        if rng.random() < 0.6:
            c = str(s[0]) if rng.random() < 0.4 else str(rng.choice(s[1:] if len(s) > 1 else s))
        shown.append(s); prop.append(p); clicked.append(c); recs.append(r)

    got = evaluate_policy(recs, shown, prop, clicked)
    first_shown = [s[0] for s in shown]
    match = np.array([r[0] == fs for r, fs in zip(recs, first_shown)], dtype=float)
    w = match / np.array([p[0] for p in prop])
    reward = np.array([
        1.0 if (clicked[i] not in ("", None) and clicked[i] == first_shown[i]) else 0.0
        for i in range(n)
    ])
    has_click = np.array([c not in ("", None) for c in clicked])
    hit = float(np.mean([clicked[i] in recs[i] for i in range(n) if has_click[i]]))
    assert got["hit_rate@10"] == pytest.approx(hit, abs=1e-12)
    assert got["ips_ctr_slot1"] == pytest.approx(float((w * reward).mean()), abs=1e-9)
    assert got["snips_ctr_slot1"] == pytest.approx(
        float((w * reward).sum() / max(w.sum(), 1e-9)), abs=1e-9)
    assert got["slot1_matches"] == int(match.sum())


def test_evaluate_wrapper_accepts_dicts_and_namespace_rows() -> None:
    policy_ns = lambda row: [str(row.shown[0]), "x"]  # noqa: E731
    dicts = [{"shown": ["a", "b"], "propensity": [0.5, 0.5], "clicked": "a"}]
    ns = [SimpleNamespace(shown=["a", "b"], propensity=[0.5, 0.5], clicked="a")]
    policy_dict = lambda row: [str(row["shown"][0]), "x"]  # noqa: E731
    md = evaluate(policy_dict, dicts)
    ms = evaluate(policy_ns, ns)
    assert md == ms
    assert md["ips_ctr_slot1"] == pytest.approx(2.0)   # w=2, reward=1
    assert md["slot1_matches"] == 1


# ---------------------------------------------------------------- SQL ------- #
def test_heldout_cut_and_eval_sql() -> None:
    client = FakeClient(routes=[("toDate(max", [{"d": "2026-11-14"}])])
    assert heldout_cut(client) == "2026-11-07"
    sql = eval_sessions_sql("2026-11-07", 5)
    assert "shop.history_sessions" in sql
    assert "ts >= '2026-11-07'" in sql
    assert "LIMIT 5" in sql
    assert "propensity" in sql and "clicked" in sql


def test_kpi_sql_contains_feedback_guards() -> None:
    sql = kpi_main_sql("team_12", "2026-11-14 10:00:00")
    assert "team_12.feedback" in sql
    assert "request_id != '__init__'" in sql
    assert "toDateTime64('2026-11-14 10:00:00', 3)" in sql
    assert "served_by::String" in sql


def test_feedback_since_utc_format() -> None:
    since = feedback_since_utc(24.0)
    assert len(since) == 19
    datetime.strptime(since, "%Y-%m-%d %H:%M:%S")  # raises if wrong

# ------------------------------------------------------ live-KPI fetch ------ #
_TEAM_MAIN = [
    {"served_by": "team", "requests": 10, "clicks": 3, "purchases": 1,
     "revenue_eur": 12.5},
    {"served_by": "fallback", "requests": 4, "clicks": 0, "purchases": 0,
     "revenue_eur": 0.0},
]
_TEAM_LATENCY = [
    {"served_by": "team", "latency_p50_ms": 100.0, "latency_p95_ms": 900.0,
     "with_latency": 10, "on_time": 9},
    {"served_by": "fallback", "latency_p50_ms": None, "latency_p95_ms": None,
     "with_latency": 0, "on_time": 0},
]
_REASONS = [{"reason": "missing", "requests": 3},
            {"reason": "", "requests": 1}]  # keys as the SQL aliases them
_HOURLY = [{"hour": "2026-11-14 10:00:00", "requests": 7, "clicks": 1,
            "purchases": 0, "revenue_eur": 2.5}]


def _kpi_client():
    return FakeClient(routes=[
        ("quantile", _TEAM_LATENCY),
        ("fallback_reason", _REASONS),
        ("toStartOfHour", _HOURLY),
        ("countIf(clicked", _TEAM_MAIN),
    ])


def test_fetch_live_kpis_aggregates_totals() -> None:
    client = _kpi_client()
    kpis = fetch_live_kpis(client)
    assert kpis["errors"] == []
    assert kpis["requests"] == 14
    assert kpis["clicks"] == 3 and kpis["purchases"] == 1
    assert kpis["ctr"] == round(3 / 14, 4)
    assert kpis["cvr"] == round(1 / 14, 4)
    assert kpis["revenue_per_request"] == round(12.5 / 14, 4)
    assert kpis["our_requests"] == 10 and kpis["fallback_requests"] == 4
    assert kpis["our_share"] == round(10 / 14, 4)
    assert kpis["on_time_rate"] == pytest.approx(0.9)
    team = kpis["by_served_by"]["team"]
    assert team["latency_p95_ms"] == 900.0
    assert team["on_time_rate"] == pytest.approx(0.9)
    assert kpis["fallback_reasons"] == {"missing": 3, "": 1}
    assert len(kpis["hourly"]) == 1
    assert kpis["hourly"][0]["ctr"] == round(1 / 7, 4)
    assert all("team_12.feedback" in q for q in client.queries)


def test_fetch_live_kpis_survives_total_outage() -> None:
    kpis = fetch_live_kpis(BrokenClient(), team="team_12")
    assert len(kpis["errors"]) == 4  # main + latency + fallback + hourly
    assert kpis["requests"] == 0
    assert kpis["ctr"] == 0.0
    assert math.isnan(kpis["on_time_rate"])


# ---------------------------------------------------------------- logger ---- #
def test_logger_writes_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "experiments.jsonl"
    lg = ExperimentLogger(path)
    lg.log_experiment("baseline", metrics={"snips_ctr_slot1": 0.10, "n_test": 5},
                      params={"cut": "2026-11-07"}, tags=["m4"])
    lg.log_experiment("coview", metrics={"snips_ctr_slot1": 0.20, "n_test": 5})
    lg.log_live_snapshot({"ctr": 0.01})
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3
    loaded = lg.load()
    assert [rec["name"] for rec in loaded] == ["baseline", "coview", "live_snapshot"]
    assert loaded[0]["kind"] == "experiment"
    assert loaded[0]["params"]["cut"] == "2026-11-07"
    assert loaded[0]["tags"] == ["m4"]
    assert lg.best(metric="snips_ctr_slot1")["name"] == "coview"
    assert lg.best(name="baseline")["name"] == "baseline"
    assert lg.best(name="nope") is None
    assert lg.last(kind="live_snapshot")["metrics"] == {"ctr": 0.01}


def test_logger_skips_malformed_lines(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_text('{"kind": "experiment", "name": "ok"}\n{broken\n\n',
                    encoding="utf-8")
    assert len(ExperimentLogger(path).load()) == 1


def test_logger_to_dataframe(tmp_path) -> None:
    import pandas as pd

    lg = ExperimentLogger(tmp_path / "e.jsonl")
    lg.log_experiment("x", metrics={"snips_ctr_slot1": 0.5})
    df = lg.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1 and df.iloc[0]["name"] == "x"


def test_logger_pushes_to_rawtree_and_survives_failures(tmp_path) -> None:
    fake = FakeClient()
    lg = ExperimentLogger(tmp_path / "e.jsonl", client=fake)
    rec = lg.log_experiment("x", metrics={"a": 1})
    assert fake.inserted == [(DEFAULT_TABLE, [rec], None)]

    broken = ExperimentLogger(tmp_path / "e2.jsonl", client=BrokenClient())
    broken.log_experiment("y", metrics={"a": 2})
    assert broken.push_errors == 1
    assert isinstance(broken.last_push_error, RuntimeError)
    assert (tmp_path / "e2.jsonl").read_text(encoding="utf-8").strip()


# ----------------------------------------------------------- jury report ---- #
class NoTableClient(FakeClient):
    """Inserts fail with the real 'table not found' error until DDL runs."""

    def __init__(self):
        super().__init__()
        self.table_created = False

    def query(self, sql):
        self.queries.append(sql)
        if "CREATE TABLE" in sql:
            self.table_created = True
            return []
        return []

    def insert(self, table, rows, database=None):
        self.inserted.append((table, list(rows), database))
        if not self.table_created:
            raise RuntimeError("insert into team_12.experiments failed [400]: "
                               '{"error":"rawtree_error",'
                               '"message":"Table experiments not found."}')


def test_logger_creates_missing_table_and_retries(tmp_path) -> None:
    client = NoTableClient()
    lg = ExperimentLogger(tmp_path / "e.jsonl", client=client)
    lg.log_experiment("bootstrap", metrics={"a": 1})
    assert client.table_created                       # DDL issued once
    assert "CREATE TABLE IF NOT EXISTS team_12.scratch" in client.queries[0]
    assert lg.push_errors == 0                        # retry succeeded
    assert len(client.inserted) == 2                  # failed try + retry


def _fb(rid, shown, clicked="", purchased="", served_by="team",
        ts="2026-11-14 10:00:00"):
    return {"request_id": rid, "ts": ts, "shown": list(shown),
            "clicked": clicked, "purchased": purchased, "revenue_eur": 0.0,
            "latency_ms": 100.0, "served_by": served_by, "fallback_reason": ""}


def _jury_logger(tmp_path) -> ExperimentLogger:
    lg = ExperimentLogger(tmp_path / "e.jsonl")
    lg.log_experiment("baseline", metrics={"hit_rate@10": 0.4,
                                          "ips_ctr_slot1": math.nan,
                                          "snips_ctr_slot1": 0.05,
                                          "slot1_matches": 300,
                                          "n_test": 100000},
                      params={"cut": "2026-11-07"}, tags=["baseline"])
    lg.log_experiment("coview_boost", metrics={"hit_rate@10": 0.55,
                                               "ips_ctr_slot1": 0.021,
                                               "snips_ctr_slot1": 0.019,
                                               "slot1_matches": 510,
                                               "n_test": 100000},
                      params={"new_product_boost": 1.25}, tags=["m4"])
    return lg


def _bandit_state() -> BanditState:
    state = BanditState()
    ingest_feedback_rows(state, [
        _fb("r1", ["p_a", "p_b"], clicked="p_a"),
        _fb("r2", ["p_c"], purchased="p_c"),
    ])
    return state


def test_jury_report_renders_all_sections(tmp_path) -> None:
    md, data = build_jury_report(client=_kpi_client(),
                                 bandit_state=_bandit_state(),
                                 logger=_jury_logger(tmp_path))
    assert "# Jury report" in md
    for section in ("## 1. What the system does", "## 2. Offline validation",
                    "## 3. Live KPIs", "## 4. Design decisions",
                    "## 5. Known limitations"):
        assert section in md, section
    assert "coview_boost" in md and "0.0190" in md
    assert "served by our policy" in md
    assert "10 (71.4%)" in md
    assert "team (ours)" in md
    assert "requests folded 2" in md
    assert "exposure 2.6" in md
    assert "active price drops 0" in md
    assert data["live_kpis"]["requests"] == 14
    assert data["bandit"]["products_tracked"] == 3
    assert len(data["experiments"]) == 2
    assert data["warnings"] == []


def test_jury_report_degrades_without_client(tmp_path) -> None:
    md, data = build_jury_report(client=None, logger=_jury_logger(tmp_path))
    assert "Live KPIs were not collected" in md
    assert data["live_kpis"] is None and data["bandit"] is None
    assert data["warnings"] == []
    assert "n/a" in md


def test_jury_report_warms_bandit_from_client_when_state_missing(tmp_path) -> None:
    client = FakeClient(routes=[
        ("ORDER BY ts DESC", [_fb("r1", ["p_a"], clicked="p_a")]),
        ("quantile", _TEAM_LATENCY),
        ("fallback_reason", _REASONS),
        ("toStartOfHour", _HOURLY),
        ("countIf(clicked", _TEAM_MAIN),
    ])
    md, data = build_jury_report(client=client, logger=_jury_logger(tmp_path))
    assert data["bandit"]["requests_folded"] == 1
    assert "requests folded 1" in md


# --------------------------------------- M4 addenda: type tag + chart layer -- #
def test_logged_records_carry_kind_and_type_tags(tmp_path) -> None:
    fake = FakeClient()
    lg = ExperimentLogger(tmp_path / "e.jsonl", client=fake)
    rec = lg.log_experiment("x", metrics={"a": 1})
    assert rec["kind"] == "experiment" and rec["type"] == "experiment"
    row = fake.inserted[0][1][0]
    assert row["kind"] == "experiment" and row["type"] == "experiment"
    lg.log_live_snapshot({"ctr": 0.1})
    assert fake.inserted[1][1][0]["type"] == "live_snapshot"


def test_new_kpi_sqls_cover_p99_cold_join_and_slot_curve() -> None:
    latency = kpi_latency_sql("team_12", "2026-11-14 10:00:00")
    assert "quantile(0.99)(latency_ms::Float64)" in latency
    assert "team_12.feedback" in latency and "request_id != '__init__'" in latency

    cold = kpi_cold_sql("team_12", "2026-11-14 10:00:00")
    assert "INNER JOIN shop.requests" in cold
    assert "r.request_id::String = f.request_id::String" in cold
    assert "r.is_new_user::String" in cold
    assert "request_id != '__init__'" in cold
    assert "toDateTime64('2026-11-14 10:00:00', 3)" in cold

    slot = history_slot_ctr_sql("2026-11-01")
    assert "ARRAY JOIN shown_arr AS pid, arrayEnumerate(shown_arr) AS slot" in slot
    assert "CAST(shown, 'Array(String)')" in slot
    assert "countIf(pid != '' AND pid == clicked)" in slot
    assert "shop.history_sessions" in slot
    assert "toDateTime64('2026-11-01', 3)" in slot


_COLD_ROWS = [
    {"is_new_user": "true", "served_by": "team",
     "requests": 8, "clicks": 2, "purchases": 1},
    {"is_new_user": "true", "served_by": "fallback",
     "requests": 4, "clicks": 1, "purchases": 1},
    {"is_new_user": "false", "served_by": "team",
     "requests": 10, "clicks": 3, "purchases": 0},
]


def test_fetch_cold_lift_aggregates_segment_cells() -> None:
    client = FakeClient(routes=[("INNER JOIN shop.requests", _COLD_ROWS)])
    cold = fetch_cold_lift(client)
    assert cold["errors"] == []
    cold_cell = cold["cells"][("cold", "team")]
    assert cold_cell["cvr"] == round(1 / 8, 4) and cold_cell["ctr"] == 0.25
    assert cold["cells"][("cold", "fallback")]["cvr"] == 0.25
    assert cold["cells"][("warm", "team")]["clicks"] == 3
    assert all("shop.requests" in q or "team_12.feedback" in q
               for q in client.queries)


def test_fetch_cold_lift_survives_a_rejected_join() -> None:
    cold = fetch_cold_lift(BrokenClient())
    assert cold["cells"] == {} and len(cold["errors"]) == 1


_SLOT_ROWS = [{"slot": 1, "impressions": 1000, "clicks": 100},
              {"slot": 2, "impressions": 1000, "clicks": 40}]


def test_fetch_slot_ctr_converts_history_rows() -> None:
    client = FakeClient(routes=[("ARRAY JOIN", _SLOT_ROWS)])
    rows = fetch_slot_ctr(client)
    assert [r["slot"] for r in rows] == [1, 2]
    assert rows[0]["ctr"] == pytest.approx(0.1)
    assert rows[1]["ctr"] == pytest.approx(0.04)
    assert fetch_slot_ctr(BrokenClient()) == []   # silent degradation

def _full_kpis() -> dict[str, Any]:
    return {"grace_ms": 3000, "by_served_by": {
        "team": {"served_by": "team", "requests": 10, "clicks": 3, "purchases": 1,
                 "ctr": 0.3, "cvr": 0.1, "with_latency": 10, "on_time": 9,
                 "latency_p50_ms": 100.0, "latency_p95_ms": 900.0,
                 "latency_p99_ms": 1500.0},
        "fallback": {"served_by": "fallback", "requests": 4, "clicks": 0,
                     "purchases": 0, "ctr": 0.0, "cvr": 0.0,
                     "with_latency": 4, "on_time": 4,
                     "latency_p50_ms": 50.0, "latency_p95_ms": 400.0,
                     "latency_p99_ms": 600.0},
    }}


def test_build_charts_renders_all_four_pngs(tmp_path) -> None:
    data = {
        "kpis": _full_kpis(),
        "cold": {"errors": [], "cells": {
            ("cold", "team"): {"cvr": 0.0125},
            ("cold", "fallback"): {"cvr": 0.005},
            ("warm", "team"): {"cvr": 0.02},
            ("warm", "fallback"): {"cvr": 0.02}}},
        "slot": [{"slot": 1, "impressions": 1000, "clicks": 100, "ctr": 0.1},
                 {"slot": 2, "impressions": 1000, "clicks": 40, "ctr": 0.04},
                 {"slot": 3, "impressions": 1000, "clicks": 20, "ctr": 0.02}],
        "warnings": [],
    }
    paths, warnings = build_charts(data, out_dir=tmp_path / "reports")
    assert warnings == []
    assert {p.name for p in paths} == {"cvr.png", "cold_start_lift.png",
                                       "latency_percentiles.png",
                                       "slot_ctr_curve.png"}
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_jury_report_embeds_charts_and_warnings(tmp_path) -> None:
    chart_paths = [Path("reports") / "cvr.png",
                   Path("reports") / "slot_ctr_curve.png"]
    md, data = build_jury_report(client=None, logger=_jury_logger(tmp_path),
                                 charts=(chart_paths,
                                         ["cold-start data: join rejected"]))
    assert "## Charts" in md
    assert "(reports/cvr.png)" in md and "(reports/slot_ctr_curve.png)" in md
    assert "join rejected" in md
    assert data["warnings"] == ["cold-start data: join rejected"]


def test_chart_pipeline_degrades_without_client(tmp_path) -> None:
    data = collect_chart_data(None)
    paths, warnings = build_charts(data, out_dir=tmp_path / "r")
    assert paths == []
    assert warnings and "no RawTree client" in warnings[0]


def test_build_charts_empty_data_skips_each_chart(tmp_path) -> None:
    data = {"kpis": {"by_served_by": {}}, "cold": {"cells": {}},
            "slot": [], "warnings": []}
    paths, warnings = build_charts(data, out_dir=tmp_path / "r")
    assert paths == []
    assert len(warnings) == 4  # one graceful skip per chart
    assert any("cvr.png skipped" in w for w in warnings)
    assert any("slot_ctr_curve.png skipped" in w for w in warnings)
