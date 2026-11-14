"""Milestone-1 data probe: read-only exploration to pin down the schemas and
statistics that cold_start.py / bandit.py / jury_report.py will rely on.

Run from the project root:
    py scripts/probe_data.py

Everything here is read-only against the shared `shop` database.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src...`

from src.client import default_client  # noqa: E402

c = default_client()

PROBES: list[tuple[str, str]] = [
    # --- shape of one historical session record (fields cold_start will mimic) ---
    ("session record", "SELECT * FROM shop.history_sessions LIMIT 1"),
    # --- catalog_updates event vocabulary: is `price_change` a separate type? ---
    ("catalog_updates events", "SELECT change::String AS k, count() AS n FROM shop.catalog_updates GROUP BY k ORDER BY n DESC"),
    # --- live stream tables still empty until Saturday? ---
    ("requests count", "SELECT count() AS n FROM shop.requests"),
    ("events count", "SELECT count() AS n FROM shop.events"),
    # --- categories / subcategories / brands: vocabulary sizes for one-hot encoding ---
    ("categories", "SELECT category::String AS category, count() AS n FROM shop.catalog GROUP BY category ORDER BY n DESC"),
    ("subcategories", "SELECT count(DISTINCT subcategory::String) AS n FROM shop.catalog"),
    ("brands", "SELECT count(DISTINCT brand::String) AS n FROM shop.catalog"),
    ("price stats", "SELECT min(price_eur) AS lo, max(price_eur) AS hi, avg(price_eur::Float64) AS avg FROM shop.catalog"),
    # --- co-purchase density: how many products were ever purchased? (item2item base) ---
    ("purchased products", "SELECT count(DISTINCT product_id::String) AS n FROM shop.history_events WHERE event_type = 'purchase'"),
    # --- new-user proxy: sessions with <=1 view in recent_views (cold users in history) ---
    ("cold-ish sessions", "SELECT count() AS n FROM shop.history_sessions WHERE length(recent_views) <= 1"),
    # --- slot CTR curve from history (position bias, needed to weight slots) ---
    ("slot ctr", """
        SELECT indexOf(shown, clicked) AS slot, count() AS clicks
        FROM shop.history_sessions WHERE clicked IS NOT NULL
        GROUP BY slot ORDER BY slot
    """),
]

for name, sql in PROBES:
    try:
        rows = c.query(sql)
        print(f"\n=== {name} ===")
        print(json.dumps(rows[:12] if len(rows) > 12 else rows, default=str, indent=1)[:1600])
    except Exception as e:  # keep probing even if one query fails
        print(f"\n=== {name} === FAILED: {e}")
