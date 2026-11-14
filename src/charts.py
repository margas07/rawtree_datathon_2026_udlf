"""PNG charts for the jury report (Person 3, Milestone 4).

`collect_chart_data(client)` gathers the guarded data sources from
`src/experiment_logger.py` (trailing-window KPIs, cold-vs-warm conversion,
per-slot history CTR) and `build_charts(data)` renders four PNGs into
`reports/`:

  1. `cvr.png`                 -- CTR / CVR per served_by bucket;
  2. `cold_start_lift.png`     -- CVR for new-user vs warm shoppers, our
                                  policy vs the baseline (the cold-start lift);
  3. `latency_percentiles.png` -- latency p50/p95/p99 vs the 3 s deadline;
  4. `slot_ctr_curve.png`      -- history slot-1..10 CTR (position bias).

Everything is best-effort: a missing client, an unreachable RawTree or empty
data simply skips the chart and appends a warning -- the report must always
land on disk.  `matplotlib.use("Agg")` keeps this display-free and CI-safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must happen before pyplot import)
import matplotlib.pyplot as plt  # noqa: E402

from .client import RawTreeClient  # noqa: E402
from .experiment_logger import (  # noqa: E402
    DEFAULT_WINDOW_HOURS,
    fetch_cold_lift,
    fetch_live_kpis,
    fetch_slot_ctr,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

_FORCE = 3_000  # answer deadline; fallback for a kpis dict without grace_ms


def _served_color(served_by: str) -> str:
    return {"team": "#2a7f62", "fallback": "#b0413e"}.get(served_by, "#9e9e9e")


def _served_label(served_by: str) -> str:
    return "team (ours)" if served_by == "team" else served_by


def _buckets(kpis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """served_by buckets sorted by traffic, highest first."""
    buckets = sorted(((kpis or {}).get("by_served_by") or {}).values(),
                     key=lambda b: b.get("requests", 0), reverse=True)
    return [b for b in buckets if b.get("requests")]


def collect_chart_data(client: RawTreeClient | None, *,
                       window_hours: float = DEFAULT_WINDOW_HOURS
                       ) -> dict[str, Any]:
    """All chart inputs in one guarded bundle (never raises)."""
    data: dict[str, Any] = {"window_hours": float(window_hours), "warnings": []}
    if client is None:
        data["warnings"].append("no RawTree client -- charts skipped")
        return data
    data["kpis"] = fetch_live_kpis(client, window_hours=window_hours)
    data["cold"] = fetch_cold_lift(client, window_hours=window_hours)
    data["slot"] = fetch_slot_ctr(client)
    for err in ((data["cold"] or {}).get("errors") or []):
        data["warnings"].append(f"cold-start data: {err}")
    for err in ((data["kpis"] or {}).get("errors") or []):
        data["warnings"].append(f"live KPIs: {err}")
    return data


def _chart_conversion(data: dict[str, Any], out_dir: Path,
                      warnings: list[str]) -> Path | None:
    buckets = _buckets(data.get("kpis"))
    if not buckets:
        warnings.append("chart cvr.png skipped: no served_by data")
        return None
    names = [_served_label(str(b.get("served_by") or "unknown")) for b in buckets]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = list(range(len(buckets)))
    width = 0.38
    for offset, metric, label, color in (
            (-width / 2, "ctr", "CTR (clicks/req)", "#a8c5bf"),
            (width / 2, "cvr", "CVR (purchases/req)", "#2a7f62")):
        values = [float(b.get(metric) or 0.0) for b in buckets]
        bars = ax.bar([xi + offset for xi in x], values, width=width, color=color,
                      label=label)
        ax.bar_label(bars, labels=[f"{100 * v:.2f}%" for v in values], fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel("rate")
    ax.set_title("Conversion per serving system (trailing window)")
    ax.legend(loc="upper right", frameon=False)
    ax.margins(y=0.2)
    fig.tight_layout()
    path = out_dir / "cvr.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _cvr_of(cell: dict[str, Any] | None) -> float:
    return float((cell or {}).get("cvr") or 0.0)


def _cell_label(value: float, cell: dict[str, Any] | None) -> str:
    return f"{100 * value:.2f}%" if cell else "no data"


def _chart_cold_lift(data: dict[str, Any], out_dir: Path,
                     warnings: list[str]) -> Path | None:
    cells = ((data.get("cold") or {}).get("cells")) or {}
    ours = [cells.get(("cold", "team")), cells.get(("warm", "team"))]
    base = [cells.get(("cold", "fallback")), cells.get(("warm", "fallback"))]
    if not any(ours) and not any(base):
        warnings.append("chart cold_start_lift.png skipped: no cold/warm join data")
        return None
    labels = ["cold\n(new users)", "warm"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, cell_list, color in (("team (ours)", ours, "#2a7f62"),
                                   ("fallback baseline", base, "#b0413e")):
        values = [_cvr_of(c) for c in cell_list]
        texts = [_cell_label(v, c) for v, c in zip(values, cell_list)]
        bars = ax.bar(labels, values, width=0.38, color=color, label=name)
        ax.bar_label(bars, labels=texts, fontsize=8)
    team_cold, base_cold = _cvr_of(ours[0]), _cvr_of(base[0])
    if team_cold and base_cold:
        ax.set_title(f"Cold-start lift: our CVR on new users is "
                     f"{team_cold / base_cold:.2f}x the baseline's")
    else:
        ax.set_title("Cold-start CVR: ours vs the baseline")
    ax.set_ylabel("CVR")
    ax.legend(loc="upper right", frameon=False)
    ax.margins(y=0.2)
    fig.tight_layout()
    path = out_dir / "cold_start_lift.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _chart_latency(data: dict[str, Any], out_dir: Path,
                   warnings: list[str]) -> Path | None:
    kpis = data.get("kpis")
    buckets = [b for b in _buckets(kpis) if b.get("with_latency")]
    if not buckets:
        warnings.append("chart latency_percentiles.png skipped: no latency data")
        return None
    grace = float((kpis or {}).get("grace_ms") or _FORCE)
    percents = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.8 / len(buckets)
    for i, bucket in enumerate(buckets):
        values = [float(bucket.get(p) or 0.0) for p in percents]
        xs = [j + (i - (len(buckets) - 1) / 2) * width for j in range(3)]
        bars = ax.bar(xs, values, width=width * 0.92,
                      color=_served_color(str(bucket.get("served_by") or "")),
                      label=_served_label(str(bucket.get("served_by") or "")))
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], fontsize=8)
    ax.axhline(grace, color="#b0413e", linestyle="--", linewidth=1.2)
    ax.text(len(percents) - 0.55, grace, f"{grace:.0f} ms deadline",
            va="center", ha="left", color="#b0413e", fontsize=8)
    ax.set_xticks(range(3), ["p50", "p95", "p99"])
    ax.set_ylabel("latency (ms)")
    ax.set_title("Answer latency percentiles (platform-measured)")
    ax.legend(loc="upper left", frameon=False)
    ax.margins(y=0.2)
    fig.tight_layout()
    path = out_dir / "latency_percentiles.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _chart_slot_ctr(data: dict[str, Any], out_dir: Path,
                    warnings: list[str]) -> Path | None:
    rows = [r for r in (data.get("slot") or [])
            if r.get("slot") and r.get("impressions")]
    if len(rows) < 2:
        warnings.append("chart slot_ctr_curve.png skipped: no slot CTR data")
        return None
    slots = [int(r["slot"]) for r in rows]
    ctrs = [100 * float(r.get("ctr") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(slots, ctrs, marker="o", color="#2a7f62", linewidth=1.8)
    ax.fill_between(slots, ctrs, color="#2a7f62", alpha=0.12)
    for slot, ctr in zip(slots, ctrs):
        ax.annotate(f"{ctr:.2f}%", (slot, ctr), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(slots)
    ax.set_xlabel("carousel slot (1 = first shown)")
    ax.set_ylabel("CTR (%)")
    ax.set_title("Slot CTR over history -- position bias the bandit mirrors")
    ax.margins(y=0.25)
    fig.tight_layout()
    path = out_dir / "slot_ctr_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


_CHARTS: tuple[tuple[str, Callable], ...] = (
    ("cvr.png", _chart_conversion),
    ("cold_start_lift.png", _chart_cold_lift),
    ("latency_percentiles.png", _chart_latency),
    ("slot_ctr_curve.png", _chart_slot_ctr),
)


def build_charts(data: dict[str, Any], *,
                 out_dir: Path = REPORTS_DIR) -> tuple[list[Path], list[str]]:
    """Render every well-supplied chart; return ([paths], [warnings])."""
    warnings = list(data.get("warnings") or [])
    paths: list[Path] = []
    collected = ("kpis" in data or "cold" in data or "slot" in data)
    if not collected:
        return paths, warnings  # nothing was collected at all
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warnings.append(f"cannot create {out_dir}: {exc}")
        return paths, warnings
    for _, builder in _CHARTS:
        try:
            path = builder(data, out_dir, warnings)
        except Exception as exc:  # noqa: BLE001 -- one bad chart keeps the rest
            warnings.append(f"chart build failed: {type(exc).__name__}: {exc}")
            continue
        if path is not None:
            paths.append(path)
    return paths, warnings

