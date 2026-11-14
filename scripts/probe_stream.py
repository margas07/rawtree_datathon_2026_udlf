"""Probe 2: live-stream record shapes needed before coding cold_start/bandit.

Read-only. Run:  py -X utf8 scripts/probe_stream.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.client import default_client  # noqa: E402

c = default_client()

PROBES = [
    ("latest live request", "SELECT * FROM shop.requests ORDER BY ts DESC LIMIT 2"),
    ("new-user requests", "SELECT count() AS n FROM shop.requests WHERE is_new_user = True"),
    ("price_change rows",
     "SELECT * FROM shop.catalog_updates WHERE change::String = 'price_change' ORDER BY ts DESC LIMIT 3"),
    ("new_product rows",
     "SELECT * FROM shop.catalog_updates WHERE change::String = 'new_product' ORDER BY ts DESC LIMIT 2"),
    ("live event sample", "SELECT * FROM shop.events ORDER BY ts DESC LIMIT 2"),
    ("feedback table (our team)", "SELECT * FROM team_12.feedback LIMIT 3"),
    ("submissions table (our team)", "SELECT count() AS n FROM team_12.submissions"),
]

for name, sql in PROBES:
    try:
        rows = c.query(sql)
        print(f"\n=== {name} ===")
        print(json.dumps(rows, default=str, indent=1)[:2500])
    except Exception as e:
        print(f"\n=== {name} === FAILED: {e}")
