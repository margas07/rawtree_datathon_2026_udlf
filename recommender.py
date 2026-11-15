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
from src.bandit import (  # noqa: E402
    PRICE_CHANGES_SQL,
    BanditState,
    FeedbackIngestor,
    update_price_drops,
)
from src.client import RawTreeClient  # noqa: E402
from src.cold_start import build_catalog_features, fetch_segment_stats  # noqa: E402

MODEL_VERSION = os.environ.get("MODEL_VERSION", "plutus-v1-cold")

COLD_FEATURES_REFRESH_S = float(os.environ.get("COLD_FEATURES_REFRESH_S", 300))


class _BaselineFeedbackIngestor(FeedbackIngestor):
    """FeedbackIngestor that also passes P1's cached baseline prices.

    shop.catalog_updates has no old_price column, so when a price_change row
    carries no usable ref_price we fall back to the ORIGINAL shop.catalog
    price captured in LiveCache.baseline_prices at startup.
    """

    def __init__(self, client, state, baseline_prices, **kw) -> None:
        super().__init__(client, state, **kw)
        self.baseline_prices = baseline_prices

    def run_once(self, *, force_prices: bool = False) -> int:
        """Poll feedback once; price refresh uses the cached baseline prices."""
        folded = self._poll_feedback()
        now = time.monotonic()
        if force_prices or (now - self._last_price_refresh
                            >= self.price_refresh_seconds):
            self._last_price_refresh = now
            try:
                update_price_drops(self.state,
                                   self.client.query(PRICE_CHANGES_SQL),
                                   baseline_prices=self.baseline_prices)
            except Exception as exc:  # noqa: BLE001 -- keep the loop alive
                self._record_error(exc)
        return folded


def _build_cold_features(cache: LiveCache, client: RawTreeClient) -> None:
    """Build cold-start CatalogFeatures from the merged catalog (background)."""
    catalog_rows = list(cache.catalog.values())
    updates_rows = [{"product_id": pid, "change": "new_product"}
                    for pid in cache.new_products]
    try:
        segment_rows = fetch_segment_stats(client)
    except Exception as e:  # noqa: BLE001
        print(f"[recommender] segment stats skipped: {e}")
        segment_rows = None
    cache.cold_features = build_catalog_features(catalog_rows, updates_rows,
                                                 segment_rows)
    print(f"[recommender] cold_features built ({len(catalog_rows)} products, "
          f"{len(cache.new_products)} new)")


def _cold_feature_loop(cache: LiveCache, client: RawTreeClient) -> None:
    import threading
    import time as _time

    def loop() -> None:
        while True:
            try:
                _build_cold_features(cache, client)
            except Exception as e:  # noqa: BLE001
                print(f"[recommender] cold_features error: {e}")
            _time.sleep(COLD_FEATURES_REFRESH_S)

    threading.Thread(target=loop, name="cold-features-builder",
                     daemon=True).start()


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

    # ---- Optional background threads (disabled by default to maintain <100ms p95 latency) ---- #
    enable_bandit = os.environ.get("ENABLE_BANDIT", "0") == "1"
    if enable_bandit:
        try:
            client = RawTreeClient()
        except Exception as e:  # noqa: BLE001
            print(f"[recommender] RawTreeClient unavailable ({e}); bandit disabled")
            client = None
        if client is not None:
            cache.bandit_state = BanditState()
            _BaselineFeedbackIngestor(
                client, cache.bandit_state, cache.baseline_prices,
            ).start()
            _cold_feature_loop(cache, client)
            print("[recommender] bandit state + feedback ingestor + cold-feature builder started")

    seen = SeenWindow(maxlen=5000)
    last_ts = ""
    latencies: deque[float] = deque(maxlen=500)
    last_stats_log = time.time()

    while True:
        loop_t0 = time.monotonic()
        reqs = []
        try:
            reqs, last_ts = pull(seen, last_ts)
            if reqs:
                snap = cache.get_snapshot()
                answers = [
                    {
                        "request_id": req["request_id"],
                        "items": recommend_live(req, snap, cache),
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
        if not reqs:
            time.sleep(max(0.0, 0.08 - elapsed))


if __name__ == "__main__":
    main()
