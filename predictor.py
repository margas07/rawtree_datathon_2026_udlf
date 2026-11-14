"""Prediction & ultra-fast set-based validation layer."""
from __future__ import annotations

from typing import Any, Iterable

Record = dict[str, Any]

# Person 3 modules (live in the src/ package; optional so the predictor also
# works offline with only the snapshot).
try:
    from src.cold_start import get_cold_start_candidates as _cs_get  # type: ignore
except ImportError:  # pragma: no cover
    _cs_get = None

try:
    from src.bandit import apply_online_adjustments as _bandit_adjust  # type: ignore
except ImportError:  # pragma: no cover
    _bandit_adjust = None

# Optional Person 2 warm-user ranker (kept duck-typed; absent today).
try:
    import ranker as _ranker  # type: ignore
except ImportError:
    _ranker = None

# Blend weights: cold users follow cold-start candidates, warm users follow the
# warm list and only receive new-product injections from cold start.
COLD_WARM_WEIGHT = 0.25     # warm-list contribution when the user is cold
WARM_COLD_WEIGHT = 0.50     # cold-start contribution when the user is warm
WARM_SLOT_DECAY = 0.7       # positional weight decay of the warm list


class _CacheView:
    """Attribute-style view over the snapshot + LiveCache extras.

    Person 3's functions read attributes (`active`, `trending`, `cold_features`,
    `bandit_state`); the pipeline hands around a dict snapshot, so per request
    we build one tiny view object (dict lookups only, microseconds).
    """

    __slots__ = ("active", "active_products", "in_stock", "trending",
                 "cold_features", "bandit_state")

    def __init__(self, snap: dict[str, Any], cache: Any) -> None:
        active = snap["active_products"]
        self.active = active
        self.active_products = active
        self.cold_features = getattr(cache, "cold_features", None)
        self.bandit_state = getattr(cache, "bandit_state", None)
        trending = getattr(cache, "trending", None)
        if not isinstance(trending, dict):
            trending = _rank_scores(snap["trending_global"])
        self.trending = trending


def _rank_scores(pids: Any, decay: float = 1.0 / 200.0) -> dict[str, float]:
    """Positional scores for a ranked pid list (slot 1 highest, geometric)."""
    scores: dict[str, float] = {}
    w = 1.0
    for pid in pids:
        if pid not in scores:
            scores[pid] = w
        w *= 1.0 - decay
    return scores


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


def recommend_live(req: Record, snap: dict[str, Any], cache: Any = None) -> list[str]:
    """Generate 10 validated recommendations in-memory without copying fallback lists.

    `cache` is Person 1's LiveCache (or any object exposing `cold_features`,
    `bandit_state`, `trending`); when absent the pipeline degrades gracefully
    to the snapshot-only warm path.
    """
    fallbacks = _fallback_sources(req, snap)
    active = snap["active_products"]
    try:
        recent_pids = _extract_recent_pids(req.get("recent_views"))
        new_prods = snap["new_products"]
        is_cold = bool(req.get("is_new_user")) or (
            bool(new_prods) and any(pid in new_prods for pid in recent_pids)
        )

        # Person 3 cold-start candidates (dict scores; {} when not applicable).
        cold_scores: dict[str, float] = {}
        view = None
        if cache is not None and (_cs_get is not None or _bandit_adjust is not None):
            view = _CacheView(snap, cache)
            if _cs_get is not None and view.cold_features is not None:
                try:
                    cold_scores = _cs_get(req, view) or {}
                except Exception:  # noqa: BLE001 -- cold start must never kill a request
                    cold_scores = {}

        # Warm list: Person 2's ranker when present, else the category-trending
        # fallback driven by the most recent view.
        warm_list: list[str] = []
        if _ranker is not None and hasattr(_ranker, "rank"):
            ranked: Any = _ranker.rank(req, snap, candidates=cold_scores)
            if isinstance(ranked, dict):
                ranked = _sort_dict_scores(ranked)
            warm_list = list(ranked or ())
        elif recent_pids:
            last_pid = recent_pids[-1]
            last_cat = snap["catalog"].get(last_pid, {}).get("category")
            if last_cat:
                cat_items = snap["trending_by_category"].get(last_cat, ())
                # Avoid putting the exact item currently being viewed in slot 1
                warm_list = [p for p in cat_items if p != last_pid] if last_pid in cat_items else cat_items

        warm_scores = _rank_scores(warm_list, WARM_SLOT_DECAY)

        def _normalise(d: dict[str, float]) -> dict[str, float]:
            m = max(d.values()) if d else 0.0
            return {k: v / m for k, v in d.items()} if m > 0 else {}

        if is_cold:
            merged = _normalise(cold_scores)
            for pid, s in _normalise(warm_scores).items():
                merged[pid] = merged.get(pid, 0.0) + COLD_WARM_WEIGHT * s
        else:
            merged = _normalise(warm_scores)
            for pid, s in _normalise(cold_scores).items():
                merged[pid] = merged.get(pid, 0.0) + WARM_COLD_WEIGHT * s

        # Person 3 bandit: UCB exploration + exploitation + price-drop boost
        # (pass-through copy until feedback exists, so early behavior is stable).
        if view is not None and _bandit_adjust is not None and merged:
            try:
                merged = _bandit_adjust(merged, view) or merged
            except Exception:  # noqa: BLE001 -- bandit must never kill a request
                pass

        ranked_pids = sorted(merged, key=merged.__getitem__, reverse=True)
        return validate(ranked_pids, active, fallbacks)
    except Exception as e:
        print(f"[predictor] fail-safe fallback for {req.get('request_id')}: {e}")
        return validate((), active, fallbacks)
