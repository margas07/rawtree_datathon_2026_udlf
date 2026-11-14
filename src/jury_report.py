"""Jury-facing report generator (Person 3, Milestone 4).

`build_jury_report` renders one self-contained Markdown page a product
manager can read in two minutes -- and the jury too:

  1. what the system does and why it is built this way;
  2. the offline validation table -- experiments logged with
     src/experiment_logger.ExperimentLogger (the notebook IPS metrics);
  3. live KPIs of the trailing window: team_NN.feedback SQL (conversion,
     revenue, latency, fallback) plus `BanditState.summary()` counters;
  4. the design decisions (tunables) with one-line justifications;
  5. known limitations, stated honestly.

`scripts/jury_report.py` is the runner: `py -X utf8 scripts/jury_report.py`
writes `jury_report.md` at the project root and records a live_snapshot row
in the experiment log.  Everything is BEST-EFFORT: a missing client, an
unreachable RawTree or an empty experiment log degrade to "n/a" placeholders
-- this page must always land on disk, even on the morning of the final round.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bandit import (
    EXPLORE_MAX,
    PERF_EXPONENT,
    PRICE_DROP_WEIGHT,
    PURCHASE_REWARD,
    SLOT_DECAY,
    UCB_EXPLORATION,
    BanditState,
    feedback_recent_sql,
    ingest_feedback_rows,
)
from .charts import build_charts, collect_chart_data
from .client import RawTreeClient
from .cold_start import (
    NEW_PRODUCT_BOOST,
    SEGMENT_TOP_N,
    SEGMENT_WEIGHT,
    TOPK_NEIGHBORS,
    TRENDING_WEIGHT,
    VIEW_HALF_LIFE_HOURS,
)
from .experiment_logger import (
    DEFAULT_LOG_PATH,
    DEFAULT_WINDOW_HOURS,
    FEEDBACK_WARMUP_LIMIT,
    ExperimentLogger,
    fetch_live_kpis,
)

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "jury_report.md"

# (parameter, value, one-line justification) -- values read from the source
# constants at import time, so the report can never drift from the code.
_TUNABLES: tuple[tuple[str, Any, str], ...] = (
    ("slot exposure decay (bandit)", SLOT_DECAY,
     "history slot-CTR falls ~x0.6 per slot (M1) -- bandit impressions are weighted the same way"),
    ("purchase value in the reward", PURCHASE_REWARD,
     "a purchase counts as 3 clicks: buying matters more than curiosity"),
    ("UCB exploration width", UCB_EXPLORATION,
     "under-shown products gain a bonus; capped so exploration never more than doubles a score"),
    ("exploitation exponent", PERF_EXPONENT,
     "sqrt-damped: a product must convert ~4x the average to earn a 2x boost"),
    ("price-drop multiplier weight", PRICE_DROP_WEIGHT,
     "a 20% discount lifts the score by +10% -- visible, never dominant"),
    ("UCB exploration cap", EXPLORE_MAX,
     "bounds the exploration bonus for never-shown products"),
    ("segment vs trending blend", f"{SEGMENT_WEIGHT} / {TRENDING_WEIGHT}",
     "blank-session new users: history conversion leads, live trending breaks ties"),
    ("new-product boost", NEW_PRODUCT_BOOST,
     "new products join the pools of shoppers browsing their most similar items"),
    ("content neighbours per product", TOPK_NEIGHBORS,
     "top-50 established neighbours precomputed per new product at build time"),
    ("session-view half-life", VIEW_HALF_LIFE_HOURS,
     "recent views weigh more when seeding item-to-item similarity"),
    ("segment candidates pooled", SEGMENT_TOP_N,
     "top converters pulled into the candidate pool per segment ranking"),
)


def _num(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{nd}f}"


def _pct1(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{100 * v:.1f}%"


def _int_str(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{int(round(v)):,}"


def _euro(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"EUR {v:.2f}"


def _what_we_built() -> list[str]:
    return [
        "## 1. What the system does",
        "",
        "Every `shop.requests` row gets an ordered top-10 carousel within the 3 s deadline. "
        "The request path touches only precomputed dicts / NumPy arrays (no SQL, no model "
        "calls): slow signals are rebuilt in Person 1's background loop, and every RawTree "
        "query sits behind the shared `src/client.py`.",
        "",
        "- **Cold start, new shoppers (`src/cold_start.py`)** -- no session views: rank "
        "products by historical conversion inside the shopper's segment (device x region x "
        "page, 4-level wildcard fallback) blended with the live 1-hour trending list. Has "
        "session views: pure item-to-item content similarity seeded by the most recent "
        "views (recency-weighted, no popularity blending).",
        "- **Cold start, new products (`src/cold_start.py`)** -- TF-IDF descriptions + "
        "category/subcategory/brand one-hot + log-price define a content space; every NEW "
        "product is pre-mapped to its top-50 most similar established products and injected "
        "into the pools of shoppers browsing those (personalisation for items with zero "
        "sales history).",
        "- **Online adaptation (`src/bandit.py`)** -- a daemon thread folds `team_NN.feedback` "
        "into a thread-safe per-product state (slot-weighted impressions, clicks, purchases, "
        "price drops). Each request's candidate scores are multiplied by (a) a UCB exploration "
        "bonus, (b) a smoothed per-product performance multiplier, (c) a price-drop boost. "
        "With no state or no feedback yet, scores pass through unchanged -- wiring is safe "
        "at any time.",
        "- **Evaluation closed loop (this milestone)** -- offline runs are scored with the "
        "starter notebook's estimators (hit-rate@10, IPS / SNIPS slot-1 CTR) and every live "
        "KPI snapshot lands in the same experiment log (`logs/experiments.jsonl`, mirrored "
        "into team_12.scratch, kind-tagged), so design decisions stay traceable before and after "
        "going live.",
        "- **Guard rails** -- never recommend out-of-stock items (Person 1's cache, their "
        "final guard); bandit learning excludes rows the baseline served, because those "
        "shoppers never saw our arms.",
        "",
    ]


def _offline_section(records: list[dict[str, Any]]) -> list[str]:
    lines = ["## 2. Offline validation (held-out last 7 days of shop.history_sessions)",
             "",
             "Experiments are scored with the exact estimators from `starter/offline_eval.ipynb`: "
             "hit-rate@10, and the inverse-propensity estimate of the slot-1 click rate "
             "(weight 1/propensity, self-normalised variant = SNIPS). Policies are built from "
             "data before the cut and evaluated after it.", ""]
    rows: list[list[str]] = []
    for record in records:
        if record.get("kind") != "experiment":
            continue
        m = record.get("metrics") or {}
        rows.append([
            str(record.get("name") or "(unnamed)"),
            _num(m.get("hit_rate@10")),
            _num(m.get("ips_ctr_slot1")),
            _num(m.get("snips_ctr_slot1")),
            _int_str(m.get("slot1_matches")),
            _int_str(m.get("n_test")),
            str(record.get("ts") or "")[:19],
        ])
    if not rows:
        lines += ["No offline runs logged yet. Wire the notebook in two lines:", "",
                  "```python",
                  "import sys; sys.path.insert(0, '..')",
                  "from src.experiment_logger import ExperimentLogger, evaluate",
                  "```", ""]
        return lines
    lines += ["| policy | hit_rate@10 | IPS slot-1 | SNIPS slot-1 | slot-1 matches | n_test | logged (UTC) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for row in list(reversed(rows))[-12:]:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              "`slot1_matches` is the IPS sample size -- few matches means a noisy IPS; "
              "compare every estimate with the old system's observed slot-1 CTR on the same "
              "test slice before drawing conclusions.", ""]
    return lines


def _live_section(kpis: dict[str, Any] | None, summary: dict[str, Any] | None,
                  window_hours: float) -> list[str]:
    lines = [f"## 3. Live KPIs (last {_num(window_hours, 0)} h of team feedback)", ""]
    if kpis is None:
        return lines + ["_Live KPIs were not collected in this run (no RawTree client)._", ""]
    since = str(kpis.get("window_start_utc") or "")
    lines += [f"Window starts {since} UTC (source: `team_NN.feedback`, read-only SQL).", "",
              "| metric | value |",
              "|---|---:|"]
    lines += [
        f"| requests with feedback | {_int_str(kpis.get('requests'))} |",
        f"| served by our policy | {_int_str(kpis.get('our_requests'))} ({_pct1(kpis.get('our_share'))}) |",
        f"| served by the fallback baseline | {_int_str(kpis.get('fallback_requests'))} ({_pct1(kpis.get('fallback_share'))}) |",
        f"| click-through rate | {_pct1(kpis.get('ctr'))} |",
        f"| conversion rate | {_pct1(kpis.get('cvr'))} |",
        f"| revenue total | {_euro(kpis.get('revenue_eur'))} |",
        f"| revenue per request | {_euro(kpis.get('revenue_per_request'))} |",
    ]
    ours = (kpis.get("by_served_by") or {}).get("team", {})
    lines += [
        f"| our latency p50 / p95 | {_num(ours.get('latency_p50_ms'), 0)} / {_num(ours.get('latency_p95_ms'), 0)} ms |",
        f"| answered within 3 s (ours) | {_pct1(ours.get('on_time_rate'))} |",
        f"| hours with traffic | {_int_str(len(kpis.get('hourly') or []))} |",
        "",
        "Reasons the baseline had to serve (fallback_reason, top 3): "
        + (_fallback_text(kpis.get("fallback_reasons") or {}) or "none -- we served everyone"),
        "",
        "| bucket | requests | CTR | CVR | EUR/request | p50 ms | p95 ms | in 3 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    buckets = sorted((kpis.get("by_served_by") or {}).values(),
                     key=lambda b: b.get("requests", 0), reverse=True)
    for b in buckets:
        label = "team (ours)" if b.get("served_by") == "team" else str(b.get("served_by"))
        lines.append(
            f"| {label} | {_int_str(b.get('requests'))} | {_pct1(b.get('ctr'))} | "
            f"{_pct1(b.get('cvr'))} | {_euro(b.get('revenue_per_request'))} | "
            f"{_num(b.get('latency_p50_ms'), 0)} | {_num(b.get('latency_p95_ms'), 0)} | "
            f"{_pct1(b.get('on_time_rate'))} |")
    lines.append("")
    if summary:
        lines += [
            "**Bandit state** (slot-weighted counters folded from live feedback):",
            "",
            f"- requests folded {summary.get('requests_folded')} -- products tracked "
            f"{summary.get('products_tracked')} -- exposure {summary.get('exposure')}",
            f"- clicks {summary.get('clicks')} (CTR {_pct1(summary.get('ctr'))}) -- "
            f"purchases {summary.get('purchases')} (CVR {_pct1(summary.get('cvr'))})",
            f"- active price drops {summary.get('price_drops')} -- last feedback "
            f"{summary.get('last_ts') or 'n/a'}",
            "",
        ]
    errors = kpis.get("errors") or []
    if errors:
        lines += ["Collection warnings (degraded, not failures):"]
        lines += [f"- {err}" for err in errors]
        lines.append("")
    return lines


def _fallback_text(reasons: dict[str, int]) -> str:
    parts = []
    for reason, count in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        parts.append(f"`{reason or '(blank)'}` x{count}")
    return ", ".join(parts)


_CHART_CAPTIONS: dict[str, str] = {
    "cvr.png": "CTR and CVR per serving system (trailing window).",
    "cold_start_lift.png": "Cold-start lift: CVR on new-user shoppers, ours vs the baseline.",
    "latency_percentiles.png": "Answer latency p50/p95/p99 vs the 3 s deadline.",
    "slot_ctr_curve.png": "Slot-position CTR over history -- the position bias our bandit weights mirror.",
}


def _charts_section(charts: tuple[list[Path], list[str]] | None) -> list[str]:
    if not charts:
        return []
    paths, chart_warnings = charts
    lines: list[str] = []
    if paths:
        lines += ["## Charts", "",
                  "Rendered into `reports/` from the same guarded queries "
                  "(`src/charts.py`); the PNGs sit next to this file.", ""]
        for path in paths:
            caption = _CHART_CAPTIONS.get(path.name, "")
            lines.append(f"![{caption}](reports/{path.name})")
            if caption:
                lines += ["", f"_{caption}_", ""]
    if chart_warnings:
        lines += ["Chart notes (degraded, not failures):"]
        lines += [f"- {w}" for w in chart_warnings]
        lines.append("")
    return lines


def _tunables_section() -> list[str]:
    lines = ["## 4. Design decisions (tunables)", "",
             "All constants live in the modules themselves, so the report cannot drift "
             "from the code:", "",
             "| parameter | value | why |", "|---|---:|---|"]
    for name, value, why in _TUNABLES:
        lines.append(f"| {name} | {_num(value) if isinstance(value, float) else value} | {why} |")
    lines.append("")
    return lines


def _limitations_section() -> list[str]:
    return [
        "## 5. Known limitations",
        "",
        "- The offline log has no `is_new_user`; blank cold sessions are proxied by "
        "`length(recent_views) <= 1` (~45% of history sessions), so offline cold-start "
        "numbers carry that approximation.",
        "- IPS is slot-1 only (the starter evaluation's target); `slot1_matches` governs "
        "its noise, and hit-rate@10 is popularity-biased in the old system's favour.",
        "- Price-drop detection uses the historical max price as the reference, because "
        "`shop.catalog_updates` carries no old-price field; boosts clear automatically "
        "when a price recovers.",
        "- Latency figures come from the platform-measured `latency_ms` in "
        "`team_NN.feedback`; rows with NULL latency were missed answers.",
        "- Bandit learning excludes fallback rows on purpose: the baseline carousel's "
        "outcomes say nothing about our arms (they are still deduplicated/watermarked).",
        "",
    ]


def build_jury_report(client: RawTreeClient | None = None,
                      bandit_state: BanditState | None = None,
                      logger: ExperimentLogger | None = None,
                      *, window_hours: float = DEFAULT_WINDOW_HOURS,
                      charts: tuple[list[Path], list[str]] | None = None
                      ) -> tuple[str, dict[str, Any]]:
    """Render the report page; return (markdown, data-dict for tests/logging).

    `bandit_state` may be omitted: with a client we fold the most recent
    feedback window into a fresh BanditState just for the report (read-only).
    """
    records = logger.load() if logger is not None else []
    kpis: dict[str, Any] | None = None
    warnings: list[str] = []
    if client is not None:
        try:
            kpis = fetch_live_kpis(client, window_hours=window_hours)
        except Exception as exc:  # noqa: BLE001 -- degrade, never crash
            warnings.append(f"live KPIs unavailable: {exc}")
        if bandit_state is None:
            try:
                state = BanditState()
                rows = client.query(
                    feedback_recent_sql(client.team_id, FEEDBACK_WARMUP_LIMIT))
                ingest_feedback_rows(state, rows)
                bandit_state = state
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"bandit warm-up failed: {exc}")
    summary: dict[str, Any] | None = None
    if bandit_state is not None:
        try:
            summary = bandit_state.summary()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"bandit summary failed: {exc}")

    if charts:
        warnings.extend(charts[1])
    now = datetime.now(timezone.utc)
    md = "\n".join([
        "# Jury report -- team_12 -- RawTree Challenge (Datathon FME 2026)",
        "",
        f"_Generated {now:%Y-%m-%d %H:%M} UTC -- live window: last "
        f"{_num(window_hours, 0)} h._",
        "",
        *_what_we_built(),
        *_offline_section(records),
        *_live_section(kpis, summary, window_hours),
        *_charts_section(charts),
        *_tunables_section(),
        *_limitations_section(),
    ])
    if warnings:
        md += "## Collection warnings\n\n" + "\n".join(f"- {w}" for w in warnings) + "\n"
    data = {"live_kpis": kpis, "bandit": summary,
            "experiments": [r for r in records if r.get("kind") == "experiment"],
            "warnings": warnings}
    return md, data


def main() -> int:
    """Runner used by scripts/jury_report.py (never run this file directly)."""
    try:
        client = RawTreeClient()
    except Exception as exc:  # noqa: BLE001
        print(f"no RawTree client ({exc}); rendering placeholders")
        client = None
    logger = ExperimentLogger(DEFAULT_LOG_PATH, client=client)
    chart_paths, chart_warnings = build_charts(collect_chart_data(client))
    md, data = build_jury_report(client=client, logger=logger,
                                 charts=(chart_paths, chart_warnings))
    REPORT_PATH.write_text(md, encoding="utf-8")
    if data["live_kpis"] is not None:
        k = data["live_kpis"]
        logger.log_live_snapshot({
            "requests": k.get("requests"),
            "our_share": k.get("our_share"),
            "fallback_share": k.get("fallback_share"),
            "ctr": k.get("ctr"),
            "cvr": k.get("cvr"),
            "revenue_per_request": k.get("revenue_per_request"),
            "on_time_rate": k.get("on_time_rate"),
            "bandit_requests_folded": (data["bandit"] or {}).get("requests_folded"),
            "collection_errors": k.get("errors") or [],
        })
    print(f"jury report written to {REPORT_PATH}; charts: {len(chart_paths)}")
    return 0
