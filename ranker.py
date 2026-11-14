import lightgbm as lgb
import numpy as np
from collections import Counter
from typing import Any

# Load model into memory globally so it doesn't reload on every request
ranker_model: lgb.Booster = lgb.Booster(model_file='ranker_v2_model.txt')

def get_raw_coview_scores(recent_views: list[Any], covisit_matrix: dict[str, list[str]]) -> dict[str, float]:
    """Calculates exponential decay co-view scores."""
    score: Counter[str] = Counter()
    for k, view in enumerate(reversed(recent_views)):
        if isinstance(view, dict):
            product_id: str = str(view.get("product_id", ""))
        else:
            product_id: str = str(view)
        if product_id:
            for rank, candidate_id in enumerate(covisit_matrix.get(product_id, [])):
                score[candidate_id] += (0.7 ** k) / (rank + 1)
    return dict(score)

def build_features(
    candidate_ids: list[str], 
    recent_views: list[Any], 
    live_catalog_cache: dict[str, dict[str, Any]], 
    coview_scores: dict[str, float]
) -> np.ndarray:
    """Vectorized feature construction for LightGBM."""
    feature_matrix: np.ndarray = np.zeros((len(candidate_ids), 6), dtype=np.float64)
    
    recent_pids: list[str] = [
        str(v.get("product_id", "")) if isinstance(v, dict) else str(v)
        for v in recent_views
    ]
    recent_prices: list[float] = [float(live_catalog_cache.get(pid, {}).get("price_eur", 0.0)) for pid in recent_pids if pid]
    avg_session_price: float = float(np.mean(recent_prices)) if recent_prices else 0.0
    recent_subcats: list[str] = [str(live_catalog_cache.get(pid, {}).get("subcategory", "")) for pid in recent_pids if pid]
    recent_len: int = len(recent_subcats)

    for i, pid in enumerate(candidate_ids):
        catalog_info: dict[str, Any] = live_catalog_cache.get(pid, {})
        cand_price: float = float(catalog_info.get("price_eur", 0.0))
        cand_subcat: str = str(catalog_info.get("subcategory", ""))
        
        feature_matrix[i, 0] = coview_scores.get(pid, 0.0)
        
        feature_matrix[i, 1] = (cand_price / avg_session_price) if avg_session_price > 0.0 else 1.0
        
        if recent_len > 0:
            matches: int = sum(1 for sub in recent_subcats if sub == cand_subcat)
            feature_matrix[i, 2] = float(matches) / recent_len
        else:
            feature_matrix[i, 2] = 0.0
            
        feature_matrix[i, 3] = float(catalog_info.get("price_drop_flag", 0.0))
        feature_matrix[i, 4] = float(catalog_info.get("organic_cvr", 0.0))
        feature_matrix[i, 5] = float(catalog_info.get("velocity", 0.0))

    return feature_matrix

def rank_candidates(
    request_row: dict[str, Any], 
    covisit_matrix: dict[str, list[str]], 
    live_catalog_cache: dict[str, dict[str, Any]], 
    baseline_fallback: list[str] | dict[str, float],
    active_products: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Generates features, scores candidates via LightGBM, and returns top 10 IDs."""
    recent_views: list[Any] = request_row.get("recent_views", [])
    
    coview_scores: dict[str, float] = get_raw_coview_scores(recent_views, covisit_matrix)
    
    fallback_list: list[str] = list(baseline_fallback.keys()) if isinstance(baseline_fallback, dict) else (baseline_fallback or [])
    initial_candidates: list[str] = list(coview_scores.keys())[:50] + fallback_list[:20]
    
    seen: set[str] = set()
    candidate_ids: list[str] = []
    for pid in initial_candidates:
        if pid and pid not in seen:
            seen.add(pid)
            if active_products is not None:
                if pid in active_products:
                    candidate_ids.append(pid)
            else:
                info = live_catalog_cache.get(pid, {})
                if info.get("is_active", False) and info.get("in_stock", True):
                    candidate_ids.append(pid)
    
    if not candidate_ids:
        return baseline_fallback[:10]

    X_predict: np.ndarray = build_features(candidate_ids, recent_views, live_catalog_cache, coview_scores)
    probabilities: np.ndarray = ranker_model.predict(X_predict)
    
    scored_candidates: list[tuple[str, float]] = sorted(
        zip(candidate_ids, probabilities.tolist()), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [pid for pid, _ in scored_candidates][:10]


def rank(req: dict[str, Any], snap: dict[str, Any], candidates: list[str]) -> list[str]:
    request_row = req.copy()
    covisit_matrix: dict[str, list[str]] = snap.get("covisited", {})
    live_catalog_cache: dict[str, dict[str, Any]] = snap.get("catalog", {})
    baseline_fallback = candidates or []

    return rank_candidates(
        request_row,
        covisit_matrix,
        live_catalog_cache,
        baseline_fallback,
        active_products=snap.get("active_products"),
    )
