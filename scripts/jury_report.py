"""Runner: render the jury report from live data and log a live snapshot.

    Start-Process py -ArgumentList '-X','utf8','scripts/jury_report.py'

Writes jury_report.md at the project root (and logs/experiments.jsonl +
team_12.scratch). Works offline too: missing data degrades to n/a.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.jury_report import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
