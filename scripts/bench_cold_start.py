"""M2b benchmark: build CatalogFeatures from the REAL catalog + measure the
per-request latency of get_cold_start_candidates on REAL shop.requests rows.

Also live-validates SEGMENT_STATS_SQL (the one M2 piece never run against the
API). Writes its own report file -- shell output capture is unreliable.

Run from the project root:  py -X utf8 scripts/bench_cold_start.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.client import RawTreeClient  # noqa: E402
from src.cold_start import (  # noqa: E402
    build_catalog_features,
    fetch_segment_stats,
    get_cold_start_candidates,
)

REPORT = ROOT / "bench_report.txt"
LAT_BUDGET_MS = 50.0
N_REQUESTS = 300

# Minimal casts (M1 findings: ::String casts are safe; numeric/bool conversion
# is done in Python below to avoid unverified dialect functions).
CATALOG_SQL = """
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       description::String AS description, price_eur, is_active
FROM shop.catalog
"""

UPDATES_SQL = """
SELECT product_id::String AS product_id, change::String AS change,
       ts::DateTime64(3) AS ts, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       description::String AS description, price_eur, is_active
FROM shop.catalog_updates
ORDER BY ts DESC
LIMIT 5000
"""

REQUESTS_SQL = """
SELECT request_id::String AS request_id,
       is_new_user, context.device::String AS device,
       context.region::String AS region, context.page::String AS page,
       recent_views, ts::DateTime64(3) AS ts
FROM shop.requests
ORDER BY ts DESC
LIMIT 600
"""

_out: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _out.append(msg)


def _is_active(value) -> bool:
    return str(value).lower() in ("true", "1", "yes")


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
    log(f"=== M2b cold-start benchmark -- {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    client = RawTreeClient()

    # ---- 1. fetch real catalog + updates + segment stats ------------------- #
    t0 = time.perf_counter()
    catalog = client.query(CATALOG_SQL)
    t_cat = time.perf_counter() - t0
    log(f"catalog rows: {len(catalog)} (query {t_cat:.1f}s)")

    t0 = time.perf_counter()
    updates = client.query(UPDATES_SQL)
    t_upd = time.perf_counter() - t0
    new_tags = sum(1 for r in updates if str(r.get("change")) == "new_product")
    log(f"catalog_updates rows: {len(updates)} ({new_tags} new_product) "
        f"(query {t_upd:.1f}s)")

    # Merged catalog view (what P1's cache should hold): base catalog with
    # newest update snapshot per product applied (updates fetched DESC ->
    # first write wins, so new products land with their snapshot fields).
    merged: dict[str, dict] = {str(r["product_id"]): dict(r) for r in catalog}
    for r in updates:
        pid = str(r.get("product_id") or "")
        if pid and pid not in merged:
            merged[pid] = dict(r)
    catalog_merged = list(merged.values())
    log(f"merged catalog rows: {len(catalog_merged)} "
        f"(+{len(catalog_merged) - len(catalog)} new products)")
    return _continue_bench(client, catalog_merged, updates)


def _continue_bench(client, catalog, updates) -> int:
    t0 = time.perf_counter()
    try:
        segments = fetch_segment_stats(client)  # validates SEGMENT_STATS_SQL
        t_seg = time.perf_counter() - t0
        log(f"SEGMENT_STATS_SQL OK: {len(segments)} rows ({t_seg:.1f}s)")
        if segments:
            log(f"  sample: {dict(list(segments[0].items())[:5])}")
    except Exception as exc:  # noqa: BLE001 -- report and continue without
        t_seg = time.perf_counter() - t0
        log(f"SEGMENT_STATS_SQL FAILED after {t_seg:.1f}s: {exc!r}")
        segments = None
    return _build_and_measure(client, catalog, updates, segments)


def _build_and_measure(client, catalog, updates, segments) -> int:
    # ---- 2. build features (background-phase cost, seconds are fine) ------- #
    t0 = time.perf_counter()
    feats = build_catalog_features(catalog, updates,
                                   segments if segments is not None else None)
    t_build = time.perf_counter() - t0
    log(f"build_catalog_features: {t_build:.1f}s "
        f"({len(feats.ids)} products, matrix {feats.matrix.shape}, "
        f"{len(feats.new_products)} new products, "
        f"{len(feats.segment_rankings)} segment keys)")
    # active set: everything in the catalog marked active (P1 owns the real one)
    active = {str(r["product_id"]) for r in catalog if _is_active(r.get("is_active"))}
    log(f"active products: {len(active)}")
    cache = SimpleNamespace(active=active, trending={}, cold_features=feats)

    # ---- 3. per-request latency on real requests --------------------------- #
    t0 = time.perf_counter()
    requests = client.query(REQUESTS_SQL)
    t_req = time.perf_counter() - t0
    log(f"requests fetched: {len(requests)} (query {t_req:.1f}s)")

    branches = {"A_blank": 0, "B_new_with_views": 0, "C_inject_only": 0,
                "warm_no_views": 0, "no_candidates": 0}
    cand_counts: list[int] = []
    timings: list[float] = []
    examples: list[str] = []
    for req in requests[:N_REQUESTS]:
        viewed = [v.get("product_id") for v in (req.get("recent_views") or [])
                  if isinstance(v, dict) and v.get("product_id")]
        t0 = time.perf_counter()
        scores = get_cold_start_candidates(req, cache)
        timings.append((time.perf_counter() - t0) * 1000.0)
        cand_counts.append(len(scores))
        if not viewed:
            branches["warm_no_views" if not req.get("is_new_user")
                     else "A_blank"] += 1
        elif req.get("is_new_user"):
            branches["B_new_with_views"] += 1
        else:
            branches["C_inject_only"] += 1
        if not scores:
            branches["no_candidates"] += 1
        elif len(examples) < 3 and viewed:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
            examples.append(f"  req={req.get('request_id')} "
                            f"new_user={req.get('is_new_user')} "
                            f"views={viewed[-3:]}\n    -> {top}")

    timings = np.array(timings)
    cand_counts = np.array(cand_counts)
    p50, p95, p99 = np.percentile(timings, [50, 95, 99])
    log(f"\nbranch mix over {len(timings)} requests: {branches}")
    log(f"candidates/request: mean={cand_counts.mean():.1f} "
        f"max={cand_counts.max()}")
    log(f"latency ms: mean={timings.mean():.3f} p50={p50:.3f} p95={p95:.3f} "
        f"p99={p99:.3f} max={timings.max():.3f}")
    for line in examples:
        log(line)

    ok = timings.max() < LAT_BUDGET_MS
    log(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- request path "
        f"{'<' if ok else '>='} {LAT_BUDGET_MS:.0f} ms budget")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

