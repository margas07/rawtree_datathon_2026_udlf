"""M4 live probe: validate the jury/KPI SQL against the real stream, verify
the team_12.scratch round trip, smoke-evaluate the baseline policy on the
held-out slice, and render the first data-backed jury_report.md.

Read-only against shop.*; writes ONE probe row into team_12.scratch.
Launch detached from the project root (flaky shell integration):
    Start-Process py -ArgumentList '-X','utf8','scripts/probe_jury.py'
Report: jury_probe_report.txt. Page: jury_report.md.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bandit import BanditState, feedback_recent_sql, ingest_feedback_rows  # noqa: E402
from src.client import RawTreeClient  # noqa: E402
from src.experiment_logger import (  # noqa: E402
    ExperimentLogger,
    eval_sessions_sql,
    evaluate,
    fetch_live_kpis,
    heldout_cut,
    fetch_cold_lift,
    fetch_slot_ctr,
)
from src.charts import build_charts  # noqa: E402
from src.jury_report import REPORT_PATH, build_jury_report  # noqa: E402

REPORT = ROOT / "jury_probe_report.txt"
EVAL_SMOKE_N = 300

_out: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _out.append(msg)


def main() -> int:
    t0 = time.perf_counter()
    ok = False
    try:
        ok = _run()
    except Exception as exc:  # noqa: BLE001 -- always land the report on disk
        log(f"FATAL: {exc!r}")
        log(traceback.format_exc())
    log(f"=== M4 probe {'PASS' if ok else 'FAIL'} "
        f"({time.perf_counter() - t0:.1f}s) ===")
    REPORT.write_text("\n".join(_out), encoding="utf-8")
    return 0 if ok else 1


def _run() -> bool:
    log(f"=== M4 jury/KPI probe -- {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    client = RawTreeClient()

    # 1. live KPI SQL (the four guarded queries) --------------------------- #
    kpis = fetch_live_kpis(client, window_hours=24.0)
    log(f"KPI collection errors: {kpis['errors'] or 'none'}")
    for key in ("requests", "our_requests", "our_share", "fallback_requests",
                "ctr", "cvr", "revenue_eur", "revenue_per_request",
                "on_time_rate"):
        log(f"  {key}: {kpis.get(key)}")
    team_kpis = kpis["by_served_by"].get("team", {})
    log(f"  team latency p50/p95/p99: {team_kpis.get('latency_p50_ms')}/"
        f"{team_kpis.get('latency_p95_ms')}/{team_kpis.get('latency_p99_ms')} ms")
    log(f"  by_served_by: {sorted(kpis['by_served_by'])}")
    log(f"  fallback_reasons: {kpis['fallback_reasons']}")
    log(f"  hourly rows: {len(kpis['hourly'])}")
    kpi_ok = not kpis["errors"]

    # 2. bandit state folded from the recent feedback window --------------- #
    state = BanditState()
    rows = client.query(feedback_recent_sql(client.team_id, 2000))
    folded = ingest_feedback_rows(state, rows)
    log(f"bandit warm-up: fetched {len(rows)}, folded {folded}")
    log(f"  summary: {json.dumps(state.summary())}")

    # 2b. cold-start join + slot CTR (chart inputs) -------------------------- #
    cold = fetch_cold_lift(client, window_hours=24.0)
    log(f"cold-start join: cells={len(cold['cells'])}; errors={cold['errors'] or 'none'}")
    for key, cell in sorted(cold["cells"].items()):
        log(f"  {key}: requests={cell['requests']} "
            f"ctr={cell['ctr']} cvr={cell['cvr']}")
    slot = fetch_slot_ctr(client)
    log("slot CTR: " + " ".join(f"s{r['slot']}={r['ctr']:.4f}" for r in slot))

    # 3. team_12.scratch round trip ----------------------------------------- #
    logger = ExperimentLogger(ROOT / "logs" / "experiments.jsonl", client=client)
    logger.log_experiment("m4_probe", metrics={"ping": 1.0},
                          params={"probe": "jury"}, tags=["probe"])
    log(f"logger push_errors: {logger.push_errors}; "
        f"last_push_error: {logger.last_push_error}")
    found = None
    found_type = 0
    try:
        rows = client.query(
            "SELECT count() AS n FROM team_12.scratch "
            "WHERE __raw_data.name::String = 'm4_probe'")
        rows_type = client.query(
            "SELECT count() AS n FROM team_12.scratch "
            "WHERE __raw_data.type::String = 'experiment' "
            "AND __raw_data.name::String = 'm4_probe'")
        found_type = rows_type[0].get("n") if rows_type else 0
        found = rows[0].get("n") if rows else 0
    except Exception as exc:  # noqa: BLE001
        log(f"experiments read-back failed: {exc}")
    log(f"team_12.scratch rows named m4_probe: {found}; "
        f"typed-experiment m4_probe: {found_type}")
    push_ok = (logger.push_errors == 0 and found is not None
               and int(found) >= 1 and int(found_type or 0) >= 1)

    # 4. offline smoke evaluation on the held-out slice (baseline policy) -- #
    cut = heldout_cut(client)
    sessions = client.query(eval_sessions_sql(cut, EVAL_SMOKE_N))
    t1 = time.perf_counter()
    best_rows = client.query(
        "SELECT product_id::String AS product_id FROM shop.history_events "
        "WHERE event_type::String = 'purchase' "
        f"AND ts::DateTime64(3) < toDateTime64('{cut} 00:00:00', 3) "
        "GROUP BY product_id ORDER BY count() DESC")
    best = [r["product_id"] for r in best_rows]
    log(f"offline smoke: cut={cut}, sessions={len(sessions)}, "
        f"best-sellers={len(best)} ({time.perf_counter() - t1:.1f}s)")
    metrics = evaluate(lambda _row: best[:10], sessions)
    log(f"  baseline eval: {json.dumps(metrics, allow_nan=True)}")
    logger.log_experiment("baseline_heldout_smoke", metrics=metrics,
                          params={"cut": cut, "n_sessions": len(sessions)},
                          tags=["baseline", "m4", "smoke"])
    log(f"  logger push_errors after eval row: {logger.push_errors}")
    offline_ok = metrics["slot1_matches"] > 0

    # 5. render jury report + charts ---------------------------------------- #
    chart_paths, chart_warnings = build_charts(
        {"kpis": kpis, "cold": cold, "slot": slot, "warnings": []})
    log(f"charts: {[p.name for p in chart_paths]}; "
        f"warnings={chart_warnings or 'none'}")
    charts_ok = len(chart_paths) == 4 and not chart_warnings
    md, data = build_jury_report(client=client, bandit_state=state,
                                 logger=logger,
                                 charts=(chart_paths, chart_warnings))
    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"jury report: {len(md.splitlines())} lines -> {REPORT_PATH.name}; "
        f"experiments={len(data['experiments'])}; "
        f"warnings={data['warnings'] or 'none'}")

    log(f"verdict: kpi_ok={kpi_ok} push_ok={push_ok} "
        f"offline_ok={offline_ok} charts_ok={charts_ok} "
        f"report={REPORT_PATH.exists()}")
    return (kpi_ok and push_ok and offline_ok and charts_ok
            and REPORT_PATH.exists())


if __name__ == "__main__":
    raise SystemExit(main())
