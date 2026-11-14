"""Gestor de estado en tiempo real (state_manager.py).

Mantiene en memoria los datos de la tienda organizados en diccionarios listos
para usar en el ranking (`catalog`, `prices`, `price_drops`, `trending_by_category`,
`trending_by_region`, `top_selling`, `top_viewed`, `velocity_spikes`), actualizados
en segundo plano sin bloquear `recommender.py`, y persiste features en `team_NN.scratch`.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

SQL_CATALOG = """
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, description::String AS description,
       is_active::Bool AS is_active
FROM shop.catalog"""

SQL_UPDATES = """
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, description::String AS description,
       is_active::Bool AS is_active, change::String AS change,
       toString(ts::DateTime64(3)) AS ts
FROM shop.catalog_updates
{where}
ORDER BY ts ASC"""

# 60-day baseline purchases, organic CVR + total events per hour (for best_sellers, CVR & velocity)
SQL_HISTORY_BASELINE = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'purchase') AS purchases,
       countIf(event_type = 'purchase') / greatest(countIf(event_type = 'view'), 1) AS organic_cvr,
       count() / (60.0 * 24.0) AS events_per_hour_60d
FROM shop.history_events
GROUP BY product_id
ORDER BY purchases DESC, product_id DESC"""

SQL_TREND_GLOBAL = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'purchase') AS purchases,
       countIf(event_type = 'view') AS views,
       count() AS total_events_60m,
       5 * countIf(event_type = 'purchase')
       + 2 * countIf(event_type = 'add_to_cart')
       + countIf(event_type = 'view') AS score
FROM shop.events
WHERE ts > now() - INTERVAL 60 MINUTE
GROUP BY product_id
ORDER BY score DESC
LIMIT 200"""

SQL_TREND_CATEGORY = """
SELECT c.category::String AS category, e.product_id::String AS product_id,
       5 * countIf(e.event_type = 'purchase') + countIf(e.event_type = 'view') AS score
FROM shop.events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE e.ts > now() - INTERVAL 60 MINUTE
GROUP BY category, product_id
ORDER BY score DESC
LIMIT 20 BY category"""

SQL_TREND_REGION = """
SELECT region::String AS region, product_id::String AS product_id,
       5 * countIf(event_type = 'purchase') + countIf(event_type = 'view') AS score
FROM shop.events
WHERE ts > now() - INTERVAL 60 MINUTE AND region != ''
GROUP BY region, product_id
ORDER BY score DESC
LIMIT 20 BY region"""

SQL_SPIKES_RECENT = """
SELECT product_id::String AS product_id,
       countIf(ts > now() - INTERVAL 15 MINUTE) AS views_15m,
       countIf(ts <= now() - INTERVAL 15 MINUTE) / 12.0 AS avg_15m_before,
       views_15m / greatest(avg_15m_before, 1.0) AS lift
FROM shop.events
WHERE event_type = 'view' AND ts > now() - INTERVAL 195 MINUTE
GROUP BY product_id
HAVING views_15m >= 3
ORDER BY lift DESC, views_15m DESC
LIMIT 30"""

SQL_COVISITED_HISTORY = """
SELECT p, groupArray(20)(q) AS top_q
FROM (
    SELECT a.product_id AS p, b.product_id AS q, count() AS n
    FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
          FROM shop.history_events WHERE event_type = 'view') AS a
    INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
                FROM shop.history_events WHERE event_type = 'view') AS b
      ON a.session_id = b.session_id
    WHERE a.product_id != b.product_id
    GROUP BY p, q
    ORDER BY p ASC, n DESC
    LIMIT 20 BY p
)
GROUP BY p"""

SQL_COVISITED_LIVE = """
SELECT p, groupArray(20)(q) AS top_q
FROM (
    SELECT a.product_id AS p, b.product_id AS q, count() AS n
    FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
          FROM shop.events WHERE event_type = 'view' AND ts > now() - INTERVAL 120 MINUTE) AS a
    INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
                FROM shop.events WHERE event_type = 'view' AND ts > now() - INTERVAL 120 MINUTE) AS b
      ON a.session_id = b.session_id
    WHERE a.product_id != b.product_id
    GROUP BY p, q
    ORDER BY p ASC, n DESC
    LIMIT 20 BY p
)
GROUP BY p"""


def load_env() -> tuple[str, str, str]:
    """Carga RAWTREE_API_URL, TEAM_ID y RAWTREE_API_KEY desde .env (raíz del repo) o entorno."""
    root_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(root_env)
    if root_env.exists():
        for line in root_env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    return (
        os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
        os.environ["TEAM_ID"].strip(),
        os.environ["RAWTREE_API_KEY"].strip(),
    )


def _group_by_key(rows: list[dict[str, Any]], key_col: str) -> dict[str, list[str]]:
    """Agrupa filas SQL en un diccionario {clave: [product_id, ...]}."""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row[key_col]), []).append(str(row["product_id"]))
    return grouped


class LiveCache:
    """Caché en memoria basada en diccionarios para alimentar el ranking en <1ms."""

    def __init__(
        self,
        catalog_interval: float = 1.5,
        trends_interval: float = 60.0,
        scratch_interval: float = 120.0,
    ) -> None:
        self.api, self.team, self.api_key = load_env()
        self.catalog_interval = catalog_interval
        self.trends_interval = trends_interval
        self.scratch_interval = scratch_interval

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_update_ts = ""
        self._last_scratch_write = 0.0

        # Diccionarios principales de estado
        self.catalog: dict[str, dict[str, Any]] = {}
        self.prices: dict[str, float] = {}
        self.price_drops: dict[str, float] = {}
        self.baseline_rate_60d: dict[str, float] = {}

        self.organic_cvr_60d: dict[str, float] = {}
        self.velocity: dict[str, float] = {}
        self.covisited: dict[str, list[str]] = {}
        self.active_products: set[str] = set()
        self.new_products: set[str] = set()
        # P3 integration: bandit state + cold-start features (filled by the
        # orchestrator at startup) and the ORIGINAL shop.catalog prices
        # captured before any catalog_updates row is folded in — used as the
        # baseline for bandit price-drop detection.
        self.bandit_state: Any = None
        self.cold_features: Any = None
        self.baseline_prices: dict[str, float] = {}
        self.trends: dict[str, Any] = {
            "global": [],
            "top_selling": [],
            "top_viewed": [],
            "by_category": {},
            "by_region": {},
            "spikes": [],
            "best_sellers": [],
        }
        self._snapshot: dict[str, Any] = self._build_snapshot()

    def _new_session(self) -> requests.Session:
        sess = requests.Session()
        sess.headers["Authorization"] = f"Bearer {self.api_key}"
        return sess

    def _query(self, sess: requests.Session, sql: str, timeout: float = 10.0) -> list[dict[str, Any]]:
        r = sess.post(f"{self.api}/v1/query", json={"sql": sql}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    def _update_product_dict(self, row: dict[str, Any], is_update: bool = False) -> None:
        """Actualiza los diccionarios catalog, prices, price_drops y active_products."""
        pid = str(row["product_id"])
        prev = self.catalog.get(pid, {})
        prev_price = prev.get("price_eur")
        new_price = float(row["price_eur"]) if row.get("price_eur") is not None else (prev_price or 0.0)

        change = str(row.get("change", "")) if is_update else ""
        if change == "out_of_stock":
            in_stock = False
        elif change in ("back_in_stock", "new_product"):
            in_stock = True
        else:
            in_stock = prev.get("in_stock", True)

        is_active = bool(row.get("is_active", True))
        self.catalog[pid] = {
            "product_id": pid,
            "category": row.get("category", ""),
            "subcategory": row.get("subcategory", ""),
            "brand": row.get("brand", ""),
            "price_eur": new_price,
            "description": row.get("description", ""),
            "is_active": is_active,
            "in_stock": in_stock,
            "price_drop_flag": 1.0 if pid in self.price_drops else 0.0,
            "organic_cvr": self.organic_cvr_60d.get(pid, prev.get("organic_cvr", 0.0)),
            "velocity": self.velocity.get(pid, prev.get("velocity", 0.0)),
        }
        self.prices[pid] = new_price
        self.covisited.setdefault(pid, [])

        if is_active and in_stock:
            self.active_products.add(pid)
        else:
            self.active_products.discard(pid)

        if change == "new_product":
            self.new_products.add(pid)
        elif change == "price_change" and prev_price and new_price < prev_price:
            drop = (prev_price - new_price) / prev_price
            if drop >= 0.05:
                self.price_drops[pid] = round(drop, 4)
            else:
                self.price_drops.pop(pid, None)

        ts = str(row.get("ts", ""))
        if ts > self._last_update_ts:
            self._last_update_ts = ts

    def _merge_covisited_rows(self, rows: list[dict[str, Any]]) -> None:
        """Actualiza self.covisited[p] priorizando las nuevas co-visitas y creciendo hasta máx 20."""
        for row in rows:
            pid = str(row["p"])
            incoming = [str(q) for q in (row.get("top_q") or []) if str(q) != pid]
            existing = self.covisited.get(pid, [])
            merged: list[str] = []
            seen: set[str] = set()
            for q in incoming + existing:
                if q not in seen and q != pid:
                    merged.append(q)
                    seen.add(q)
                    if len(merged) == 20:
                        break
            assert isinstance(merged, list) and len(merged) <= 20 and len(merged) == len(seen) and pid not in seen
            self.covisited[pid] = merged

    def _build_snapshot(self) -> dict[str, Any]:
        """Construye el diccionario único que reciben predictor.py y los modelos."""
        active = frozenset(self.active_products)
        covisited_snap: dict[str, list[str]] = {}
        for pid, q_list in self.covisited.items():
            filtered = [q for q in q_list if q in active and q != pid][:20]
            covisited_snap[pid] = filtered
            assert (
                isinstance(filtered, list)
                and len(filtered) <= 20
                and len(filtered) == len(set(filtered))
                and pid not in filtered
            ), f"Invalid covisited list for {pid}"
            covisited_snap[pid] = filtered

        catalog_snapshot: dict[str, dict[str, Any]] = {}
        for pid, item in self.catalog.items():
            entry = dict(item)
            entry["price_drop_flag"] = 1.0 if pid in self.price_drops else 0.0
            entry["organic_cvr"] = self.organic_cvr_60d.get(pid, entry.get("organic_cvr", 0.0))
            entry["velocity"] = self.velocity.get(pid, entry.get("velocity", 0.0))
            catalog_snapshot[pid] = entry



        assert isinstance(covisited_snap, dict)
        return {
            "active_products": active,
            "prices": dict(self.prices),
            "catalog": catalog_snapshot,
            "new_products": frozenset(self.new_products),
            "price_drops": dict(self.price_drops),
            "covisited": covisited_snap,
            "trending_global": [p for p in self.trends["global"] if p in active],
            "top_selling": [p for p in self.trends["top_selling"] if p in active],
            "top_viewed": [p for p in self.trends["top_viewed"] if p in active],
            "trending_by_category": {
                cat: [p for p in pids if p in active]
                for cat, pids in self.trends["by_category"].items()
            },
            "trending_by_region": {
                reg: [p for p in pids if p in active]
                for reg, pids in self.trends["by_region"].items()
            },
            "velocity_spikes": [p for p in self.trends["spikes"] if p in active],
            "best_sellers": [p for p in self.trends["best_sellers"] if p in active],
            "updated_at": time.time(),
        }

    def _publish_snapshot(self) -> None:
        snap = self._build_snapshot()
        with self._lock:
            self._snapshot = snap

    def initialize(self) -> None:
        """Carga inicial de catálogo, stock, baseline de 60 días y tendencias."""
        sess = self._new_session()
        try:
            for row in self._query(sess, SQL_CATALOG, timeout=15.0):
                self._update_product_dict(row, is_update=False)
            # Baseline prices = the pre-update catalog snapshot; every later
            # catalog_updates row overwrites self.prices, so capture now.
            self.baseline_prices = dict(self.prices)

            try:
                for row in self._query(sess, SQL_UPDATES.format(where=""), timeout=15.0):
                    self._update_product_dict(row, is_update=True)
            except Exception as e:
                print(f"[state_manager] catalog_updates init skipped: {e}")

            hist_rows = self._query(sess, SQL_HISTORY_BASELINE, timeout=20.0)
            self.trends["best_sellers"] = [
                str(r["product_id"]) for r in hist_rows if int(r.get("purchases", 0)) > 0
            ]
            self.baseline_rate_60d = {
                str(r["product_id"]): float(r.get("events_per_hour_60d", 0.0)) for r in hist_rows
            }
            self.organic_cvr_60d = {
                str(r["product_id"]): float(r.get("organic_cvr", 0.0)) for r in hist_rows
            }
            self.velocity = dict(self.baseline_rate_60d)

            try:
                cov_hist = self._query(sess, SQL_COVISITED_HISTORY, timeout=25.0)
                self._merge_covisited_rows(cov_hist)
            except Exception as e:
                print(f"[state_manager] covisited history init skipped: {e}")

            try:
                cov_hist = self._query(sess, SQL_COVISITED_HISTORY, timeout=25.0)
                self._merge_covisited_rows(cov_hist)
            except Exception as e:
                print(f"[state_manager] covisited history init skipped: {e}")

            try:
                self._refresh_trends(sess)
            except Exception as e:
                print(f"[state_manager] initial trends skipped: {e}")

            self._publish_snapshot()
        finally:
            sess.close()

    def _refresh_catalog(self, sess: requests.Session) -> bool:
        where = f"WHERE ts::DateTime64(3) > '{self._last_update_ts}'" if self._last_update_ts else ""
        rows = self._query(sess, SQL_UPDATES.format(where=where), timeout=4.0)
        for row in rows:
            self._update_product_dict(row, is_update=True)
        return bool(rows)

    def _refresh_trends(self, sess: requests.Session) -> bool:
        global_rows = self._query(sess, SQL_TREND_GLOBAL, timeout=8.0)
        if global_rows:
            self.trends["global"] = [str(r["product_id"]) for r in global_rows]
            by_purchases = sorted(global_rows, key=lambda r: int(r.get("purchases", 0)), reverse=True)
            by_views = sorted(global_rows, key=lambda r: int(r.get("views", 0)), reverse=True)
            self.trends["top_selling"] = [str(r["product_id"]) for r in by_purchases[:100]]
            self.trends["top_viewed"] = [str(r["product_id"]) for r in by_views[:100]]
            for r in global_rows:
                self.velocity[str(r["product_id"])] = float(r.get("total_events_60m", 0.0))

        by_cat = _group_by_key(self._query(sess, SQL_TREND_CATEGORY, timeout=8.0), "category")
        # Asegura que los new_products en tendencia también entren en su categoría
        for pid in self.trends["global"]:
            cat = self.catalog.get(pid, {}).get("category")
            if cat and pid not in by_cat.get(cat, []):
                by_cat.setdefault(cat, []).append(pid)
        if by_cat:
            self.trends["by_category"] = by_cat

        by_reg = _group_by_key(self._query(sess, SQL_TREND_REGION, timeout=8.0), "region")
        if by_reg:
            self.trends["by_region"] = by_reg

        # Velocity spikes: combina picos de 15m vs 3h y picos de 60m vs baseline de 60 días
        spikes_recent = [str(r["product_id"]) for r in self._query(sess, SQL_SPIKES_RECENT, timeout=8.0)]
        spikes_60d: list[tuple[float, str]] = []
        for r in global_rows:
            pid = str(r["product_id"])
            ev_60m = float(r.get("total_events_60m", 0.0))
            base_60d = max(self.baseline_rate_60d.get(pid, 0.0), 1.0)
            if ev_60m >= 5 and (ev_60m / base_60d) >= 2.0:
                spikes_60d.append((ev_60m / base_60d, pid))
        spikes_60d.sort(reverse=True)

        merged_spikes: list[str] = []
        for pid in [p for _, p in spikes_60d[:20]] + spikes_recent:
            if pid not in merged_spikes:
                merged_spikes.append(pid)
        if merged_spikes:
            self.trends["spikes"] = merged_spikes[:30]

        try:
            cov_live = self._query(sess, SQL_COVISITED_LIVE, timeout=15.0)
            self._merge_covisited_rows(cov_live)
        except Exception as e:
            print(f"[state_manager] covisited live refresh skipped: {e}")

        self._write_to_scratch(sess)
        return True

    def _write_to_scratch(self, sess: requests.Session) -> None:
        """Persiste features calculadas en SQL dentro de team_NN.scratch (criterio del jurado)."""
        now = time.time()
        if now - self._last_scratch_write < self.scratch_interval or not self.trends["global"]:
            return
        try:
            payload = [
                {
                    "kind": "live_store_features",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    "active_count": len(self.active_products),
                    "new_products_count": len(self.new_products),
                    "price_drops_count": len(self.price_drops),
                    "trending_global_top20": self.trends["global"][:20],
                    "top_selling_top20": self.trends["top_selling"][:20],
                    "velocity_spikes_top20": self.trends["spikes"][:20],
                }
            ]
            sess.post(
                f"{self.api}/v1/tables/scratch",
                params={"database": self.team},
                json=payload,
                timeout=5.0,
            ).raise_for_status()
            self._last_scratch_write = now
        except Exception as e:
            print(f"[state_manager] scratch write skipped: {e}")

    def _start_worker(self, interval: float, refresh_fn: Callable[[requests.Session], bool], name: str) -> None:
        def loop() -> None:
            sess = self._new_session()
            try:
                while not self._stop_event.is_set():
                    try:
                        if refresh_fn(sess):
                            self._publish_snapshot()
                    except Exception as e:
                        print(f"[state_manager] {name} error: {e}")
                    self._stop_event.wait(interval)
            finally:
                sess.close()

        threading.Thread(target=loop, name=name, daemon=True).start()

    def start(self) -> None:
        """Carga el estado inicial y lanza los dos hilos de refresco."""
        self.initialize()
        self._start_worker(self.catalog_interval, self._refresh_catalog, "catalog-refresher")
        self._start_worker(self.trends_interval, self._refresh_trends, "trends-refresher")

    def stop(self) -> None:
        self._stop_event.set()

    # Métodos de acceso para recommender.py y modelos del equipo
    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot

    def get_active(self) -> frozenset[str]:
        return self.get_snapshot()["active_products"]

    def get_prices(self) -> dict[str, float]:
        return self.get_snapshot()["prices"]

    def get_trending(self, category: str | None = None, region: str | None = None) -> list[str]:
        snap = self.get_snapshot()
        if category and snap["trending_by_category"].get(category):
            return snap["trending_by_category"][category]
        if region and snap["trending_by_region"].get(region):
            return snap["trending_by_region"][region]
        return snap["trending_global"] or snap["best_sellers"]


if __name__ == "__main__":
    cache = LiveCache()
    t0 = time.monotonic()
    cache.initialize()
    snap = cache.get_snapshot()
    print(
        f"LiveCache ready in {time.monotonic() - t0:.2f}s | "
        f"active={len(snap['active_products'])} | "
        f"new={len(snap['new_products'])} | "
        f"trending={len(snap['trending_global'])} | "
        f"top_selling={len(snap['top_selling'])} | "
        f"spikes={len(snap['velocity_spikes'])}"
    )
