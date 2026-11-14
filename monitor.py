"""Live health & performance monitor reading team_NN.feedback (Fase 3).

Usage:
    python monitor.py          # refreshes every 60s
    python monitor.py --once   # prints once and exits
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import requests

from state_manager import load_env


def query(sess: requests.Session, api: str, sql: str) -> list[dict[str, Any]]:
    r = sess.post(f"{api}/v1/query", json={"sql": sql}, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])


def print_report(sess: requests.Session, api: str, team: str) -> None:
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== RawTree Monitor ({team}) @ {now_str} ===")

    reliability = query(
        sess,
        api,
        f"""
        SELECT served_by::String AS served_by,
               fallback_reason::String AS fallback_reason,
               count() AS requests,
               round(quantile(0.5)(latency_ms::Float64), 1) AS p50_ms,
               round(quantile(0.95)(latency_ms::Float64), 1) AS p95_ms
        FROM {team}.feedback
        WHERE ts > now() - INTERVAL 15 MINUTE AND request_id != '__init__'
        GROUP BY served_by, fallback_reason
        ORDER BY requests DESC
        """,
    )
    total_15m = sum(int(r["requests"]) for r in reliability)
    team_15m = sum(int(r["requests"]) for r in reliability if r["served_by"] == "team")
    on_time_pct = (100.0 * team_15m / total_15m) if total_15m else 0.0
    print(f"Last 15m: total={total_15m} | team={team_15m} ({on_time_pct:.1f}% on-time)")
    for row in reliability:
        print(
            f"  - served_by={row['served_by']:8s} reason={row['fallback_reason'] or '-':8s} "
            f"n={row['requests']:5d} p50={row['p50_ms']}ms p95={row['p95_ms']}ms"
        )
        if row["served_by"] == "team" and float(row["p95_ms"] or 0) > 1500:
            print(f"    [ALERT] p95 latency above 1500ms ({row['p95_ms']}ms)")

    if total_15m > 0 and on_time_pct < 90.0:
        print(f"  [ALERT] On-time rate below 90% ({on_time_pct:.1f}%) — red status light risk!")

    perf = query(
        sess,
        api,
        f"""
        SELECT f.served_by::String AS served_by,
               count() AS requests,
               round(100.0 * countIf(f.clicked != '') / count(), 2) AS ctr_pct,
               round(100.0 * countIf(f.purchased != '') / count(), 2) AS cvr_pct,
               round(sum(f.revenue_eur::Float64) / count(), 4) AS rev_per_req
        FROM {team}.feedback AS f
        WHERE f.ts > now() - INTERVAL 60 MINUTE AND f.request_id != '__init__'
        GROUP BY served_by
        ORDER BY requests DESC
        """,
    )
    print("Last 60m conversion:")
    for row in perf:
        print(
            f"  - served_by={row['served_by']:8s} n={row['requests']:5d} "
            f"CTR={row['ctr_pct']}% CVR={row['cvr_pct']}% rev/req={row['rev_per_req']} EUR"
        )

    by_version = query(
        sess,
        api,
        f"""
        SELECT s.model_version::String AS model_version,
               count() AS requests,
               round(100.0 * countIf(f.clicked != '') / count(), 2) AS ctr_pct,
               round(100.0 * countIf(f.purchased != '') / count(), 2) AS cvr_pct,
               round(sum(f.revenue_eur::Float64) / count(), 4) AS rev_per_req
        FROM {team}.feedback AS f
        INNER JOIN {team}.submissions AS s ON s.request_id::String = f.request_id::String
        WHERE f.ts > now() - INTERVAL 60 MINUTE
          AND f.request_id != '__init__'
          AND f.served_by = 'team'
        GROUP BY model_version
        ORDER BY requests DESC
        """,
    )
    if by_version:
        print("Last 60m by model_version:")
        for row in by_version:
            print(
                f"  - {row['model_version']}: n={row['requests']} "
                f"CTR={row['ctr_pct']}% CVR={row['cvr_pct']}% rev/req={row['rev_per_req']} EUR"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor RawTree team feedback.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds.")
    args = parser.parse_args()

    api, team, key = load_env()
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {key}"

    try:
        while True:
            try:
                print_report(sess, api, team)
            except Exception as e:
                print(f"[monitor] query error: {e}")
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        sess.close()


if __name__ == "__main__":
    main()
