"""READ-ONLY dry run of the fully wired pipeline (no answers submitted).

Run from the project root:  py -X utf8 scripts/dry_run.py

Verifies, against the LIVE system but without touching team_12.submissions:
  1. LiveCache.initialize() against real shop.* tables (scratch writes off).
  2. Cold-start features built from the merged catalog (+ new products).
  3. BanditState warmed from real team feedback, price drops refreshed with
     P1's cached baseline prices.
  4. recommend_live() with the full wiring (cold start -> blend -> bandit ->
     validate) on ~200 recent real shop.requests rows: latency percentiles,
     10-distinct-active guarantee, cold/new-product coverage.

Everything verbose goes to dry_run_report.txt; stdout stays concise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predictor import recommend_live  # noqa: E402
from request_io import normalize_request  # noqa: E402
from src.bandit import (  # noqa: E402
    PRICE_CHANGES_SQL,
    BanditState,
    ingest_feedback_rows,
    update_price_drops,
)
from src.client import RawTreeClient  # noqa: E402
from src.cold_start import build_catalog_features, fetch_segment_stats  # noqa: E402
from state_manager import LiveCache  # noqa: E402

N_REQUESTS = 200
REPORT = ROOT / "dry_run_report.txt"
SAMPLE_REQUESTS_SQL = """
SELECT r.request_id::String AS request_id,
       toString(r.ts::DateTime64(3)) AS ts,
       r.user_id::String AS user_id,
       r.is_new_user::Bool AS is_new_user,
       r.context.device::String AS `context.device`,
       r.context.region::String AS `context.region`,
       r.context.page::String AS `context.page`,
       r.recent_views AS recent_views
FROM shop.requests AS r
ORDER BY r.ts::DateTime64(3) DESC
LIMIT {n}
"""

_lines: list[str] = []


def log(msg: str = "") -> None:
    _lines.append(msg)


def main() -> int:
    t_start = time.perf_counter()
    log(f"=== Wired pipeline dry run (READ-ONLY) -- "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # ---- 1. LiveCache without scratch writes ------------------------------- #
    cache = LiveCache(scratch_interval=1e18)  # huge interval -> no POSTs
    t0 = time.perf_counter()
    cache.initialize()
    log(f"LiveCache.initialize: {time.perf_counter() - t0:.1f}s | "
        f"active={len(cache.active_products)} new={len(cache.new_products)} "
        f"baseline_prices={len(cache.baseline_prices)}")
    snap = cache.get_snapshot()

    # ---- 2. cold-start features (merged catalog includes new products) ----- #
    client = RawTreeClient()
    updates_rows = [{"product_id": pid, "change": "new_product"}
                    for pid in cache.new_products]
    t0 = time.perf_counter()
    cache.cold_features = build_catalog_features(
        list(cache.catalog.values()), updates_rows, fetch_segment_stats(client))
    log(f"cold_features built in {time.perf_counter() - t0:.1f}s "
        f"({len(cache.catalog)} products, {len(cache.new_products)} new)")

    # ---- 3. bandit warm-up + price drops with P1's baseline prices --------- #
    cache.bandit_state = BanditState()
    try:
        rows = client.query(
            "SELECT request_id::String AS request_id, ts, shown, "
            "clicked::String AS clicked, purchased::String AS purchased, "
            "served_by::String AS served_by, "
            "fallback_reason::String AS fallback_reason "
            f"FROM {client.team_id}.feedback WHERE request_id != '__init__' "
            "ORDER BY ts DESC LIMIT 5000")
        folded = ingest_feedback_rows(cache.bandit_state, rows)
    except Exception as e:  # noqa: BLE001
        folded = 0
        log(f"feedback ingest skipped: {e}")
    try:
        n_drops = update_price_drops(cache.bandit_state,
                                      client.query(PRICE_CHANGES_SQL),
                                      baseline_prices=cache.baseline_prices)
    except Exception as e:  # noqa: BLE001
        n_drops = 0
        log(f"price-drop refresh skipped: {e}")
    log(f"bandit warm-up: folded={folded} requests, price_drops={n_drops}, "
        f"state={cache.bandit_state.summary()}")
    return _score_requests(cache, snap, client, t_start)


def _pids(req) -> list[str]:
    return [v.get("product_id") for v in (req.get("recent_views") or [])
            if isinstance(v, dict)]


def _score_requests(cache, snap, client, t_start: float) -> int:
    raw_rows = client.query(SAMPLE_REQUESTS_SQL.format(n=N_REQUESTS))
    reqs = [normalize_request(r) for r in raw_rows]
    log(f"fetched {len(reqs)} recent shop.requests rows (read-only)")

    lat: list[float] = []
    cold_n = warm_injected = valid_n = 0
    samples: list[str] = []
    cold_samples = 0
    for i, req in enumerate(reqs):
        seen = set(_pids(req))
        is_cold = bool(req["is_new_user"]) or any(
            p in snap["new_products"] for p in seen)
        t0 = time.perf_counter()
        items = recommend_live(req, snap, cache)
        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

        ok = (len(items) == 10 and len(set(items)) == 10
              and all(p in snap["active_products"] for p in items))
        valid_n += ok
        cold_n += is_cold
        if not is_cold and any(p in snap["new_products"] for p in items):
            warm_injected += 1
        if i < 3 or (is_cold and cold_samples < 3 and i > 3):
            if is_cold:
                cold_samples += 1
            samples.append(
                f"request_id={req['request_id']} "
                f"is_new_user={req['is_new_user']} views={len(_pids(req))} "
                f"cold={is_cold} {dt:.1f}ms ok={ok}\n  -> {items}")
        if not ok:
            log(f"INVALID OUTPUT for {req['request_id']}: {items}")
    return _report(lat, valid_n, cold_n, warm_injected, samples, t_start)


def _report(lat, valid_n, cold_n, warm_injected, samples, t_start) -> int:
    ordered = sorted(lat)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    log("")
    log("=== RESULTS ===")
    log(f"requests scored: {len(ordered)}")
    log(f"valid 10-distinct-active outputs: {valid_n}/{len(ordered)}")
    log(f"cold requests: {cold_n}")
    log(f"warm outputs with new-product injection: {warm_injected}")
    log(f"latency ms  p50={pct(0.50):.1f}  p90={pct(0.90):.1f}  "
        f"p95={pct(0.95):.1f}  p99={pct(0.99):.1f}  max={ordered[-1]:.1f}")
    log("")
    log("=== SAMPLE OUTPUTS ===")
    for s in samples:
        log(s)
    log("")
    log(f"total wall time: {time.perf_counter() - t_start:.1f}s "
        f"(initialize + features + {len(ordered)} recommendations)")
    log("NOTE: no INSERT was made into team_12.submissions (read-only run).")

    print(f"dry run: {valid_n}/{len(ordered)} valid | "
          f"p50={pct(0.50):.1f}ms p95={pct(0.95):.1f}ms | "
          f"cold={cold_n} injected(warm)={warm_injected} | "
          f"report -> dry_run_report.txt")
    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = main()
    finally:
        REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    sys.exit(code)

