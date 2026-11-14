"""Orchestrator loop for RawTree Datathon.

    python starter/recommender.py

Connects the three layers:
1. `request_io.py`: receives new requests (`pull`) and sends batch answers (`push`).
2. `predictor.py`: runs team algorithms (`recommend_live`) and strict validation (`validate`).
3. `state_manager.py`: keeps `LiveCache` updated in background threads.

Also re-exports `best_sellers` and `recommend` for `starter/offline_eval.ipynb`.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictor import recommend, recommend_live, validate  # noqa: E402, F401
from request_io import (  # noqa: E402, F401
    TEAM,
    SeenWindow,
    best_sellers,
    normalize_request,
    pull,
    push,
    query,
)
from state_manager import LiveCache  # noqa: E402

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1-live-cache")


def _latency_ms(ts_str: str, now_utc: float) -> float | None:
    if not ts_str:
        return None
    try:
        iso = ts_str[:-1] + "+00:00" if ts_str.endswith("Z") else ts_str + "+00:00"
        dt = datetime.fromisoformat(iso)
        return max(0.0, (now_utc - dt.timestamp()) * 1000.0)
    except Exception:
        return None


def main() -> None:
    cache = LiveCache(catalog_interval=1.5, trends_interval=60.0)
    print(f"[recommender] initializing LiveCache for {TEAM} ({MODEL_VERSION})...")
    while True:
        try:
            cache.start()
            break
        except Exception as e:
            print(f"[recommender] LiveCache init failed, retrying in 2s: {e}")
            time.sleep(2)
    print(f"[recommender] LiveCache ready: {len(cache.get_active())} active products")

    seen = SeenWindow(maxlen=5000)
    last_ts = ""
    latencies: deque[float] = deque(maxlen=500)
    last_stats_log = time.time()

    while True:
        loop_t0 = time.monotonic()
        try:
            reqs, last_ts = pull(seen, last_ts)
            if reqs:
                snap = cache.get_snapshot()
                answers = [
                    {
                        "request_id": req["request_id"],
                        "items": recommend_live(req, snap),
                        "model_version": MODEL_VERSION,
                    }
                    for req in reqs
                ]
                push(answers)
                now_utc = time.time()
                for req in reqs:
                    lat = _latency_ms(req["ts"], now_utc)
                    if lat is not None:
                        latencies.append(lat)
                print(f"answered {len(answers)}")

            if time.time() - last_stats_log >= 30.0 and latencies:
                ordered = sorted(latencies)
                p50 = ordered[len(ordered) // 2]
                p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
                print(f"[latency] n={len(ordered)} p50={p50:.0f}ms p95={p95:.0f}ms")
                last_stats_log = time.time()
        except Exception as e:
            print(f"[recommender] loop error: {e}")
            time.sleep(0.15)

        elapsed = time.monotonic() - loop_t0
        time.sleep(max(0.0, 0.15 - elapsed))


if __name__ == "__main__":
    main()
