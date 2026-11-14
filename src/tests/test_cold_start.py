"""Unit tests for src/cold_start.py -- fully offline (mock catalog + cache).

Run from the project root:  py -m pytest src/tests/test_cold_start.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.cold_start import (  # noqa: E402
    CatalogFeatures,
    SEGMENT_WEIGHT,
    build_catalog_features,
    get_cold_start_candidates,
)

# ---- tiny mock catalog: two near-identical pet shampoos, a jacket, etc. ----- #
CATALOG = [
    {"product_id": "p_a", "category": "Pets", "subcategory": "Pet Shampoo",
     "brand": "Rixa", "price_eur": 9.99, "is_active": True,
     "description": "Gentle pet shampoo for dogs and cats with oatmeal extract."},
    {"product_id": "p_b", "category": "Pets", "subcategory": "Pet Shampoo",
     "brand": "Rixa", "price_eur": 11.99, "is_active": True,
     "description": "Gentle pet shampoo for dogs and cats with oatmeal and aloe."},
    {"product_id": "p_c", "category": "Fashion", "subcategory": "Jackets",
     "brand": "Zore", "price_eur": 89.99, "is_active": True,
     "description": "Wireless travel-ready winter jacket with a minimalist design."},
    {"product_id": "p_d", "category": "Electronics", "subcategory": "Cameras",
     "brand": "Nexa", "price_eur": 199.0, "is_active": True,
     "description": "Compact action camera with 4k recording and waterproof body."},
    {"product_id": "p_out", "category": "Pets", "subcategory": "Pet Shampoo",
     "brand": "Vedo", "price_eur": 7.99, "is_active": False,  # out of stock!
     "description": "Organic pet shampoo with extra durability for sensitive skin."},
]
UPDATES = [
    {"product_id": "p_new", "change": "new_product", "category": "Pets",
     "subcategory": "Pet Shampoo", "brand": "Rixa", "price_eur": 12.99,
     "is_active": True,
     "description": "Gentle pet shampoo for dogs and cats with oatmeal formula."},
    {"product_id": "p_a", "change": "price_change"},   # must NOT count as new
]
# p_new is included in the MERGED catalog view (Person 1 applies updates)
CATALOG_MERGED = CATALOG + [
    {"product_id": "p_new", "category": "Pets", "subcategory": "Pet Shampoo",
     "brand": "Rixa", "price_eur": 12.99, "is_active": True,
     "description": "Gentle pet shampoo for dogs and cats with oatmeal formula."},
]
SEGMENTS = [
    {"device": "web", "region": "Catalonia", "page": "home",
     "product_id": "p_c", "impressions": 100, "purchases": 10},
    {"device": "web", "region": "Catalonia", "page": "home",
     "product_id": "p_a", "impressions": 1000, "purchases": 50},
    {"device": "ios", "region": "Galicia", "page": "product",
     "product_id": "p_d", "impressions": 500, "purchases": 100},
]
ACTIVE = {r["product_id"] for r in CATALOG_MERGED if r["is_active"]}
CACHE = SimpleNamespace(active=ACTIVE, trending={"p_d": 5.0, "p_c": 2.0})


@pytest.fixture(scope="module")
def features() -> CatalogFeatures:
    return build_catalog_features(CATALOG_MERGED, UPDATES, SEGMENTS)


def _ranked(scores: dict[str, float]) -> list[str]:
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------- build ----- #
def test_build_structure(features: CatalogFeatures) -> None:
    assert set(features.ids) == {r["product_id"] for r in CATALOG_MERGED}
    assert features.new_products == {"p_new"}          # price_change not "new"
    assert all(nb != pid for pid, nbs in features.neighbors.items()
               for nb, _ in nbs)                       # no self-neighbours
    assert all(0.0 <= sim <= 1.0 for nbs in features.neighbors.values()
               for _, sim in nbs)


def test_new_product_maps_to_established(features: CatalogFeatures) -> None:
    """New product must be linked to its most similar ESTABLISHED products."""
    top_established = [pid for pid, _ in features.neighbors["p_new"]
                       if pid not in features.new_products]
    assert top_established, "new product has no established neighbours"
    assert top_established[0] in ("p_a", "p_b", "p_out")  # pet shampoos
    # reverse map: browsing a similar established product can surface p_new
    assert any(new_pid == "p_new" for new_pid, _ in
               features.established_to_new.get(top_established[0], ()))


# ---------------------------------------------------------- branch A -------- #
def test_new_user_no_views_segment_blend(features: CatalogFeatures) -> None:
    req = {"is_new_user": True, "recent_views": [], "ts": "2026-11-14 13:00:00",
           "context.device": "web", "context.region": "Catalonia",
           "context.page": "home"}
    scores = get_cold_start_candidates(req, CACHE, features)
    assert scores, "empty-view new user must produce candidates"
    ranked = _ranked(scores)
    assert "p_out" not in scores, "inactive product leaked into candidates"
    assert ranked[0] == "p_c", "segment top-converter (10/100) must rank first"
    assert scores["p_d"] <= SEGMENT_WEIGHT + 1e-9  # trending-only item capped


def test_new_user_no_views_hierarchical_fallback(
        features: CatalogFeatures) -> None:
    """Unknown exact segment falls back through the hierarchy to global."""
    req = {"is_new_user": True, "recent_views": [],
           "context.device": "android", "context.region": "Nowhere",
           "context.page": "category"}
    scores = get_cold_start_candidates(req, CACHE, features)
    assert scores, "hierarchy fallback must reach the global ranking"


# ------------------------------------------------- branches B and C --------- #
def test_new_user_with_views_pure_similarity(features: CatalogFeatures) -> None:
    req = {"is_new_user": True,
           "recent_views": [{"product_id": "p_a", "ts": "2026-11-14 12:59:00"}],
           "ts": "2026-11-14 13:00:00",
           "context.device": "web", "context.region": "Catalonia",
           "context.page": "product"}
    scores = get_cold_start_candidates(req, CACHE, features)
    ranked = _ranked(scores)
    assert "p_a" not in scores, "already-viewed product must be excluded"
    # p_d IS a top-50 neighbour of p_a in this tiny catalog, but purely via
    # near-zero content similarity (~0.003). Trending blending would instead
    # add >= 0.25 to it, so a low score proves the view branch stays pure.
    assert scores.get("p_d", 0.0) < 0.1, \
        "trending must NOT leak into the view branch"
    # Branch C also injects the new product (by design); among ESTABLISHED
    # items the pure content similarity must put p_b (near-identical) first.
    ranked_established = [pid for pid in ranked
                          if pid not in features.new_products]
    assert ranked_established[0] == "p_b", \
        "most similar established pet shampoo must rank first"


def test_new_user_viewing_new_product_uses_content_index(
        features: CatalogFeatures) -> None:
    """Session seed IS a new product -> its established neighbours carry over."""
    req = {"is_new_user": True,
           "recent_views": [{"product_id": "p_new", "ts": "2026-11-14 12:59:00"}],
           "ts": "2026-11-14 13:00:00"}
    scores = get_cold_start_candidates(req, CACHE, features)
    assert scores and set(scores) <= ACTIVE
    assert "p_new" not in scores, "the viewed new product itself is excluded"


def test_warm_user_gets_new_product_injection(features: CatalogFeatures) -> None:
    """NOT a new user: only the new-product injection runs (Person 2 leads)."""
    req = {"is_new_user": False,
           "recent_views": [{"product_id": "p_a", "ts": "2026-11-14 12:59:00"}],
           "ts": "2026-11-14 13:00:00"}
    scores = get_cold_start_candidates(req, CACHE, features)
    assert scores, "browsing a pet shampoo should surface the new pet shampoo"
    assert "p_new" in scores, "similar established browse must inject p_new"
    assert "p_out" not in scores, "inactive product leaked via injection"


def test_inactive_never_returned(features: CatalogFeatures) -> None:
    req = {"is_new_user": True,
           "recent_views": [{"product_id": "p_d", "ts": "2026-11-14 12:59:00"}],
           "ts": "2026-11-14 13:00:00"}
    scores = get_cold_start_candidates(req, CACHE, features)
    assert "p_out" not in scores


def test_missing_features_returns_empty() -> None:
    cache = SimpleNamespace(active=ACTIVE)  # no cold_features attached yet
    req = {"is_new_user": True, "recent_views": []}
    assert get_cold_start_candidates(req, cache) == {}


def test_duplicate_and_unknown_views_robust(features: CatalogFeatures) -> None:
    req = {"is_new_user": True, "ts": "2026-11-14 13:00:00",
           "recent_views": [{"product_id": "p_a", "ts": "2026-11-14 12:59:00"},
                            {"product_id": "p_a", "ts": "2026-11-14 12:58:00"},
                            {"product_id": "p_ghost", "ts": "2026-11-14 12:57:00"}]}
    scores = get_cold_start_candidates(req, CACHE, features)
    assert "p_b" in scores  # duplicate p_a counted once, ghost id ignored


# ---------------------------------------------------------------- latency --- #
def test_request_path_latency_with_synthetic_5k_catalog() -> None:
    """Real-workload simulation: 5,000 products, top-50 neighbours, 300
    requests cycling through every branch; every call must stay < 50 ms."""
    rng = np.random.default_rng(7)
    n, d, k = 5000, 256, 50
    ids = [f"p_{i:05d}" for i in range(n)]
    mat = rng.normal(size=(n, d)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)
    neighbors = {}
    for i, pid in enumerate(ids):
        idx = np.argpartition(-sims[i], k - 1)[:k]
        neighbors[pid] = [(ids[j], float(sims[i, j])) for j in idx]
    feats = CatalogFeatures(
        ids=ids, id_to_idx={p: i for i, p in enumerate(ids)}, matrix=mat,
        neighbors=neighbors, new_products={"p_00001"},
        established_to_new={"p_00010": [("p_00001", 0.9)]},
        segment_rankings={("*", "*", "*"): [(pid, 0.02) for pid in ids[:40]]},
        global_ranking=[(pid, 0.02) for pid in ids[:40]],
    )
    cache = SimpleNamespace(active=set(ids),
                            trending={ids[i]: 1.0 for i in range(50)})
    new_empty = {"is_new_user": True, "recent_views": [],
                 "ts": "2026-11-14 13:00:00", "context.device": "web",
                 "context.region": "Catalonia", "context.page": "home"}
    new_views = {"is_new_user": True, "ts": "2026-11-14 13:00:00",
                 "recent_views": [{"product_id": ids[i]} for i in range(5)]}
    warm_views = {"is_new_user": False, "ts": "2026-11-14 13:00:00",
                  "recent_views": [{"product_id": ids[i]} for i in range(10, 15)]}

    timings = []
    for i in range(300):
        req = [new_empty, new_views, warm_views][i % 3]
        t0 = time.perf_counter()
        scores = get_cold_start_candidates(req, cache, feats)
        timings.append((time.perf_counter() - t0) * 1000.0)
        assert scores
    timings = np.array(timings)
    p50, p95, p99 = np.percentile(timings, [50, 95, 99])
    print(f"\nlatency ms: mean={timings.mean():.2f} p50={p50:.2f} "
          f"p95={p95:.2f} p99={p99:.2f} max={timings.max():.2f}")
    assert timings.max() < 50.0, \
        f"request path exceeded 50 ms budget: {timings.max():.2f}"
