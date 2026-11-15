"""M3 live probe: validate bandit.py's two unverified SQL paths against the
real RawTree API (read-only):

1. FEEDBACK_RECENT_SQL / team_12.feedback recent window (startup poll path);
2. FEEDBACK_SINCE_SQL with a toDateTime64 watermark (incremental poll path);
3. PRICE_CHANGES_SQL (argMax/max over shop.catalog_updates price changes) and
   the drop fractions update_price_drops derives from it;
4. a FeedbackIngestor.run_once() smoke round on the live stream.

Writes its own report file -- shell output capture is unreliable (M1/M2 finding).

Run from the project root:  py -X utf8 scripts/probe_bandit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bandit import (  # noqa: E402
    BanditState,
    FeedbackIngestor,
    PRICE_CHANGES_SQL,
    feedback_recent_sql,
    feedback_since_sql,
    ingest_feedback_rows,
    update_price_drops,
)
from src.client import RawTreeClient  # noqa: E402

REPORT = ROOT / "bandit_probe_report.txt"

_out: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _out.append(msg)


def main() -> int:
    try:
        code = _run()
    except Exception as exc:  # noqa: BLE001 -- always land the report on disk
        import traceback
        log(f"FATAL: {exc!r}")
        log(traceback.format_exc())
        code = 1
    REPORT.write_text("\n".join(_out), encoding="utf-8")
    log(f"(report written to {REPORT.name})")
    return code


def _run() -> int:
    log(f"=== M3 bandit SQL probe -- {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    client = RawTreeClient()
    state = BanditState()
    ok = True

    # ---- 1. recent-window feedback poll (startup path) --------------------- #
    sql = feedback_recent_sql(client.team_id, limit=500)
    log(f"\n[1] recent feedback window:\n{sql.strip()}")
    try:
        t0 = time.perf_counter()
        rows = client.query(sql)
        log(f"OK: {len(rows)} rows in {time.perf_counter() - t0:.1f}s")
        if rows:
            sample = rows[0]
            log(f"  first row keys: {sorted(sample)}")
            served = {}
            for r in rows:
                served[str(r.get("served_by"))] = served.get(str(r.get("served_by")), 0) + 1
            log(f"  served_by mix: {served}")
            folded = ingest_feedback_rows(state, rows)
            log(f"  folded {folded}/{len(rows)} rows (fallback/init excluded)")
        else:
            log("  (empty feedback table)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        log(f"FAILED: {exc!r}")

    # ---- 2. incremental poll with toDateTime64 watermark ------------------- #
    watermark = state.watermark_sql()
    log(f"\n[2] incremental poll (watermark = {watermark!r}):")
    if watermark:
        try:
            t0 = time.perf_counter()
            fresh = client.query(feedback_since_sql(client.team_id, watermark))
            log(f"OK: {len(fresh)} new rows in {time.perf_counter() - t0:.1f}s")
            folded = ingest_feedback_rows(state, fresh)
            log(f"  folded {folded} new rows; total_requests={state.total_requests}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            log(f"FAILED (feed Ingestor recent-window fallback): {exc!r}")
    else:
        log("SKIPPED: no rows in the recent window -> no watermark yet")

    # ---- 3. price changes + drop map --------------------------------------- #
    log(f"\n[3] price changes:\n{PRICE_CHANGES_SQL.strip()}")
    try:
        t0 = time.perf_counter()
        price_rows = client.query(PRICE_CHANGES_SQL)
        log(f"OK: {len(price_rows)} products with price_change history "
            f"({time.perf_counter() - t0:.1f}s)")
        if price_rows:
            log(f"  sample row: {price_rows[0]}")
        n_drops = update_price_drops(state, price_rows)
        top = sorted(state.price_drops.items(), key=lambda kv: -kv[1])[:5]
        log(f"  active price drops: {n_drops}; deepest 5: {top}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        log(f"FAILED: {exc!r}")

    # ---- 4. FeedbackIngestor.run_once on the live stream -------------------- #
    log("\n[4] FeedbackIngestor.run_once() live smoke:")
    try:
        ing = FeedbackIngestor(client, state)
        folded = ing.run_once()
        log(f"OK: run_once folded {folded} rows; errors={ing.errors}")
        log(f"  state summary: {state.summary()}")
        ing.stop()
    except Exception as exc:  # noqa: BLE001
        ok = False
        log(f"FAILED: {exc!r}")

    log(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- bandit SQL paths "
        f"{'all validated live' if ok else 'have failures (see above)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
