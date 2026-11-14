"""Prediction & ultra-fast set-based validation layer."""
from __future__ import annotations

from typing import Any, Iterable

Record = dict[str, Any]

# Optional team modules (Fase 4 integration: cold_start.py, ranker.py, bandit.py)
try:
    import cold_start as _cold_start  # type: ignore
except ImportError:
    _cold_start = None

try:
    import ranker as _ranker  # type: ignore
except ImportError:
    _ranker = None

try:
    import bandit as _bandit  # type: ignore
except ImportError:
    _bandit = None


def recommend(recent_views: list[Record], best: list[str], unavailable: set[str] = frozenset()) -> list[str]:
    """Baseline signature kept compatible with starter/offline_eval.ipynb."""
    return [product for product in best if product not in unavailable][:10]


def _sort_dict_scores(scores: dict[str, float]) -> list[str]:
    return [ str(k) for k, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True) ]


def _extract_recent_pids(recent_views: Any) -> list[str]:
    """Safely extract product_id strings from recent_views regardless of element format."""
    if not isinstance(recent_views, list) or not recent_views:
        return []
    pids: list[str] = []
    for v in recent_views:
        if isinstance(v, dict):
            pid = v.get("product_id")
            if pid:
                pids.append(str(pid))
        elif isinstance(v, str) and v:
            pids.append(v)
    return pids


def validate(
    items: Iterable[Any] | None,
    active_products: frozenset[str] | set[str],
    fallback_sources: Iterable[str] | tuple[Iterable[str], ...],
    k: int = 10,
) -> list[str]:
    """Guarantee exactly k distinct, active product IDs with C-level set fast-path and early returns."""
    if items is None:
        items = ()

    # Fast-path: since active_products only contains valid str IDs, issuperset(top) + len(set(top)) == k
    # verifies type, stock status, and uniqueness in a single C-level check (~0.5 us).
    if isinstance(items, (list, tuple)) and len(items) >= k:
        top = items[:k]
        if active_products.issuperset(top) and len(set(top)) == k:
            return list(top) if isinstance(top, tuple) else top

    valid: list[str] = []
    seen: set[str] = set()
    append = valid.append
    add = seen.add

    for raw_pid in items:
        pid = raw_pid if isinstance(raw_pid, str) else str(raw_pid)
        if pid in active_products and pid not in seen:
            append(pid)
            add(pid)
            if len(valid) == k:
                return valid  # Early return as soon as k items are collected

    sources: tuple[Iterable[str], ...]
    if isinstance(fallback_sources, tuple) and fallback_sources and not isinstance(fallback_sources[0], str):
        sources = fallback_sources  # type: ignore[assignment]
    else:
        sources = (fallback_sources,)  # type: ignore[assignment]

    for src in sources:
        for pid in src:
            if pid in active_products and pid not in seen:
                append(pid)
                add(pid)
                if len(valid) == k:
                    return valid  # Early return

    for pid in active_products:
        if pid not in seen:
            append(pid)
            add(pid)
            if len(valid) == k:
                return valid  # Early return

    assert len(valid) == k and len(seen) == k, f"Validator failed: got {len(valid)} items"
    return valid


def _fallback_sources(req: Record, snap: dict[str, Any]) -> tuple[Iterable[str], ...]:
    """Return tuple of fallback references (zero list copying)."""
    region = req.get("context", {}).get("region")
    by_reg = snap["trending_by_region"]
    if region and region in by_reg:
        return (by_reg[region], snap["trending_global"], snap["best_sellers"])
    return (snap["trending_global"], snap["best_sellers"])


def recommend_live(req: Record, snap: dict[str, Any]) -> list[str]:
    """Generate 10 validated recommendations in-memory without copying fallback lists."""
    fallbacks = _fallback_sources(req, snap)
    active = snap["active_products"]
    try:
        candidates: list[str] = list(fallbacks[0])
        new_prods = snap["new_products"]
        recent_pids = _extract_recent_pids(req.get("recent_views"))
        is_cold = bool(req.get("is_new_user")) or (
            bool(new_prods) and any(pid in new_prods for pid in recent_pids)
        )
        if is_cold and _cold_start is not None and hasattr(_cold_start, "get_candidates"):
            candidates = _cold_start.get_candidates(req, snap)

        if _ranker is not None and hasattr(_ranker, "rank"):
            ranked: Any = _ranker.rank(req, snap, candidates=candidates)
        elif candidates is not None:
            ranked = _sort_dict_scores(candidates) if isinstance(candidates, dict) else candidates
        else:
            ranked = ()
            if recent_pids:
                last_pid = recent_pids[-1]
                cov_items = snap.get("covisited", {}).get(last_pid, [])
                last_cat = snap.get("catalog", {}).get(last_pid, {}).get("category")
                cat_items = snap.get("trending_by_category", {}).get(last_cat, ()) if last_cat else ()
                ranked = [p for p in (*cov_items, *cat_items) if p != last_pid]

        if isinstance(ranked, dict):
            ranked = _sort_dict_scores(ranked)

        if _bandit is not None and hasattr(_bandit, "adjust"):
            ranked = _bandit.adjust(req, list(ranked) if ranked is not None else [], snap)
            if isinstance(ranked, dict):
                ranked = _sort_dict_scores(ranked)

        return validate(ranked, active, fallbacks)
    except Exception as e:
        print(f"[predictor] fail-safe fallback for {req.get('request_id')}: {e}")
        return validate((), active, fallbacks)
