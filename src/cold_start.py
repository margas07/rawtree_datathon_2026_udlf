"""Cold-start handling: brand-new users and brand-new products (Person 3).

Two problems, one module:

1. NEW USERS (`request["is_new_user"] == True`)
   * No session views  -> rank products by historical conversion inside the
     shopper's context segment (`context.device`, `context.region`,
     `context.page`), with a graceful hierarchy fallback
     (exact -> device x region -> device -> global), blended with the live
     1-hour trending list maintained by Person 1.
   * Has session views -> PURE item-to-item similarity from the session seeds
     (no popularity blending, by design: the session already tells us the taste).

2. NEW PRODUCTS (`change == 'new_product'` in `shop.catalog_updates`)
   * Every product gets a lightweight content vector: TF-IDF over `description`
     + one-hot `category` / `subcategory` / `brand` + min-max scaled log price.
     All vectors are L2-normalised, so cosine similarity is a dot product.
   * At build time each new product is mapped to its top-K most similar
     ESTABLISHED products, so it immediately inherits their co-purchase
     neighbourhood (Person 2's signals).
   * Reverse index: whenever a shopper browses an established product, relevant
     new products are injected into the candidate pool (boosted slightly because
     they are under-explored).

LATENCY CONTRACT
----------------
`build_catalog_features` / `fetch_segment_stats` are the SLOW parts (SQL +
linear algebra) and run ONCE per refresh in Person 1's background loop.
`get_cold_start_candidates` is the ONLY function in the per-request path: it
touches precomputed dicts/arrays only, well under the 50 ms budget (benchmarked
in scripts/bench_cold_start.py).

INTEGRATION CONTRACT (Person 1 / Person 2)
------------------------------------------
    # background refresh (Person 1):
    live_cache.active    = set of active, in-stock product_ids        # required
    live_cache.trending  = {product_id: trending_score}               # optional
    live_cache.cold_features = build_catalog_features(...)            # once
    # per request (Person 1's recommender.py):
    scores = get_cold_start_candidates(request, live_cache)   # {pid: score}
    # ...merge with Person 2's warm-user scores, keep top-10 distinct active.

`active` filtering is best-effort here (missing attribute -> filter skipped);
Person 1's output rule (10 distinct ACTIVE items) is the final guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .client import Record

# ---- Tunables (single source of truth for the jury slide) ------------------ #
TOPK_NEIGHBORS = 50          # stored neighbours per product (content space)
MAX_SEED_VIEWS = 8           # most-recent session views that drive similarity
VIEW_HALF_LIFE_HOURS = 2.0   # recency decay of session views (ts-based)
VIEW_DECAY_FACTOR = 0.6      # recency decay per step back (position-based)
SEGMENT_TOP_N = 40           # segment candidates pulled into the pool
SEGMENT_WEIGHT = 0.75        # segment conversion weight (empty-view branch)
TRENDING_WEIGHT = 0.25       # live trending weight (empty-view branch)
NEW_PRODUCT_BOOST = 1.25     # slight exploration boost for injected new products
SMOOTH_PRIOR_N = 50.0        # Bayesian smoothing impressions for segment rates

# content-vector feature weights
TEXT_WEIGHT = 1.0
CATEGORY_WEIGHT = 0.8
SUBCATEGORY_WEIGHT = 1.2
BRAND_WEIGHT = 0.6
PRICE_WEIGHT = 0.4


# --------------------------------------------------------------------------- #
# Data structures                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class CatalogFeatures:
    """Everything the per-request path needs, precomputed once per refresh."""

    ids: list[str]
    """Catalog product ids; row i of `matrix` corresponds to `ids[i]`."""

    id_to_idx: dict[str, int]

    matrix: np.ndarray
    """(n_products, D) float32, rows L2-normalised -> cosine = dot product."""

    neighbors: dict[str, list[tuple[str, float]]]
    """product_id -> [(similar_product_id, cosine_sim), ...] sorted desc."""

    new_products: set[str]
    """Products introduced via `new_product` events in shop.catalog_updates."""

    established_to_new: dict[str, list[tuple[str, float]]]
    """established_id -> [(new_id, sim), ...]: reverse of new->established map."""

    segment_rankings: dict[tuple[str, str, str], list[tuple[str, float]]]
    """(device, region, page) -> [(product_id, smoothed_conv_rate), ...] desc.
    '*' wildcards mark the hierarchical fallback levels; global is ('*','*','*')."""

    global_ranking: list[tuple[str, float]] = field(default_factory=list)
    """[(product_id, smoothed_conv_rate), ...] over all segments, desc."""


# --------------------------------------------------------------------------- #
# SQL fetchers (background refresh only -- never in the request path)         #
# --------------------------------------------------------------------------- #
SEGMENT_STATS_SQL = """
SELECT context.device::String AS device,
       context.region::String AS region,
       context.page::String   AS page,
       shown_item::String     AS product_id,
       count()                             AS impressions,
       countIf(purchased::String = shown_item) AS purchases
FROM shop.history_sessions
ARRAY JOIN arrayMap(x -> coalesce(toString(x), ''), CAST(shown, 'Array(String)')) AS shown_item
GROUP BY device, region, page, product_id
"""
"""Per-context-segment carousel impressions & purchases from the 60-day history.
ARRAY JOIN explodes the 10 `shown` slots -> one row per (segment, product).
`shown` holds nullable elements -> raw ARRAY JOIN fails with
"Nullable ... not allowed inside Variant type", hence the arrayMap(toStr)."""


def fetch_segment_stats(client) -> list[Record]:
    """Fetch segment conversion stats (ONE heavy query, startup/refresh only)."""
    return client.query(SEGMENT_STATS_SQL)


# --------------------------------------------------------------------------- #
# Build: slow, runs once per catalog refresh                                  #
# --------------------------------------------------------------------------- #
def build_catalog_features(
    catalog_rows: list[Record],
    updates_rows: list[Record] | None = None,
    segment_rows: list[Record] | None = None,
    *,
    topk: int = TOPK_NEIGHBORS,
) -> CatalogFeatures:
    """Precompute the content-similarity index and segment conversion rankings.

    Args:
        catalog_rows: current merged catalogue (shop.catalog + applied
            catalog_updates), one record per product with fields
            product_id / description / category / subcategory / brand / price_eur.
        updates_rows: raw shop.catalog_updates rows (tags new products).
        segment_rows: rows from fetch_segment_stats(); optional (skips rankings).
    """
    # ---- dedupe ids (a merged cache may contain the same product twice) ---- #
    ids: list[str] = []
    id_to_idx: dict[str, int] = {}
    for row in catalog_rows:
        pid = str(row["product_id"])
        if pid not in id_to_idx:
            id_to_idx[pid] = len(ids)
            ids.append(pid)

    new_products = {str(r["product_id"]) for r in (updates_rows or [])
                    if str(r.get("change", "")) == "new_product"} & set(ids)

    matrix = _content_matrix(catalog_rows, ids)
    neighbors = _topk_neighbors(matrix, ids, topk)

    # new product -> established neighbours (inherit co-purchase signals);
    # stored inverted: established -> [(new_id, sim), ...] for fast injection.
    est_to_new: dict[str, list[tuple[str, float]]] = {}
    for new_pid in new_products:
        for est_pid, sim in neighbors.get(new_pid, ()):
            if est_pid not in new_products:
                est_to_new.setdefault(est_pid, []).append((new_pid, sim))
    for pairs in est_to_new.values():
        pairs.sort(key=lambda p: -p[1])  # reverse map sorted by similarity

    seg_rankings, global_ranking = _segment_rankings(segment_rows or [])

    return CatalogFeatures(
        ids=ids,
        id_to_idx=id_to_idx,
        matrix=matrix,
        neighbors=neighbors,
        new_products=new_products,
        established_to_new=est_to_new,
        segment_rankings=seg_rankings,
        global_ranking=global_ranking,
    )


def _content_matrix(catalog_rows: list[Record], ids: list[str]) -> np.ndarray:
    """TF-IDF text + one-hot categoricals + scaled price -> L2-normalised rows."""
    n = len(ids)
    by_id = {str(r["product_id"]): r for r in catalog_rows}

    descriptions = [str(by_id.get(pid, {}).get("description") or "") for pid in ids]
    text = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    blocks = [text.fit_transform(descriptions) * TEXT_WEIGHT]

    def onehot(column: str, weight: float) -> sparse.csr_matrix:
        values = [str(by_id.get(pid, {}).get(column) or "?") for pid in ids]
        vocab = {v: i for i, v in enumerate(sorted(set(values)))}
        mat = sparse.csr_matrix(
            (np.full(n, weight, dtype=np.float32),
             (np.arange(n), np.array([vocab[v] for v in values]))),
            shape=(n, len(vocab)),
        )
        return mat

    blocks.append(onehot("category", CATEGORY_WEIGHT))
    blocks.append(onehot("subcategory", SUBCATEGORY_WEIGHT))
    blocks.append(onehot("brand", BRAND_WEIGHT))

    # price: log-scale (prices span 2-644 EUR), then min-max into [0, 1]
    prices = np.array([float(by_id.get(pid, {}).get("price_eur") or np.nan)
                       for pid in ids])
    if np.isnan(prices).any():
        median = float(np.nanmedian(prices)) if np.isfinite(prices).any() else 0.0
        prices = np.where(np.isnan(prices), median, prices)
    log_p = np.log1p(prices)
    lo, hi = float(log_p.min()), float(log_p.max())
    scaled = (log_p - lo) / max(hi - lo, 1e-9)
    blocks.append(sparse.csr_matrix(scaled.reshape(-1, 1) * PRICE_WEIGHT))

    combined = sparse.hstack(blocks, format="csr").astype(np.float32)
    return normalize(combined).toarray()  # dense (n, D); cosine == dot product


def _topk_neighbors(matrix: np.ndarray, ids: list[str], topk: int
                    ) -> dict[str, list[tuple[str, float]]]:
    """Dense all-pairs cosine on normalised rows, keep top-K per product."""
    sims = matrix @ matrix.T  # rows are unit-norm -> dot product = cosine
    np.fill_diagonal(sims, -1.0)  # never match a product with itself
    k = min(topk, sims.shape[1] - 1)
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for i, pid in enumerate(ids):
        row = sims[i]
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        neighbors[pid] = [(ids[j], float(row[j])) for j in idx]
    return neighbors


def _segment_rankings(segment_rows: list[Record]
                      ) -> tuple[dict[tuple[str, str, str], list[tuple[str, float]]],
                                 list[tuple[str, float]]]:
    """Smoothed conversion rate per (segment, product) + hierarchical fallback.

    rate = (purchases + 1) / (impressions + 50): Bayesian smoothing, so a
    product bought 3/5 times does not outrank one bought 300/10,000.
    """
    exact: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in segment_rows:
        key = (str(row["device"]), str(row["region"]), str(row["page"]))
        cell = exact.setdefault(key, {}).setdefault(
            str(row["product_id"]), [0.0, 0.0])
        cell[0] += float(row["impressions"])
        cell[1] += float(row["purchases"])

    def rank(stats: dict[str, list[float]]) -> list[tuple[str, float]]:
        scored = [(pid, (pur + 1.0) / (imp + SMOOTH_PRIOR_N))
                  for pid, (imp, pur) in stats.items()]
        scored.sort(key=lambda p: -p[1])
        return scored

    def merge(keys: list[tuple[str, str, str]]) -> dict[str, list[float]]:
        merged: dict[str, list[float]] = {}
        for key in keys:
            for pid, cell in exact.get(key, {}).items():
                acc = merged.setdefault(pid, [0.0, 0.0])
                acc[0] += cell[0]
                acc[1] += cell[1]
        return merged

    def pages(device: str, region: str) -> set[str]:
        return {p for d, r, p in exact if d == device and r == region}

    rankings: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    devices = {d for d, _, _ in exact}
    regions = {r for _, r, _ in exact}
    for device, region, page in exact:
        rankings[(device, region, page)] = rank(exact[(device, region, page)])
        rankings.setdefault((device, region, "*"), rank(
            merge([(device, region, p) for p in pages(device, region)])))
    for device in devices:
        rankings.setdefault((device, "*", "*"), rank(
            merge([(device, r, p) for r in regions for p in pages(device, r)])))
    global_ranking = rank(merge(list(exact)))
    rankings[("*", "*", "*")] = global_ranking
    return rankings, global_ranking


# --------------------------------------------------------------------------- #
# Request path: MUST stay <50 ms -- dicts and small array slices only         #
# --------------------------------------------------------------------------- #
def get_cold_start_candidates(
    request: Record,
    live_cache: Any,
    features: CatalogFeatures | None = None,
) -> dict[str, float]:
    """Candidate scores for ONE recommendation request. Called every request.

    Args:
        request: raw shop.requests row (flattened context keys:
            'context.device', 'context.region', 'context.page',
            'is_new_user', 'recent_views', 'ts').
        live_cache: Person 1's cache; reads `active` (set), `trending`
            (dict) and `cold_features` (CatalogFeatures) when present.
        features: explicit CatalogFeatures override (tests / benchmarks).

    Returns:
        {product_id: score}; only ACTIVE products, never an already-viewed
        product. Empty dict when there is nothing useful to add (warm request
        handled by Person 2, or features not built yet).
    """
    feats = features if features is not None else getattr(
        live_cache, "cold_features", None)
    if feats is None:
        return {}
    active = _active_set(live_cache)
    trending = getattr(live_cache, "trending", None) or {}
    viewed = _viewed_products(request)
    is_new_user = bool(request.get("is_new_user", False))
    req_ts = _parse_ts(request.get("ts"))

    # Branch A: brand-new user, blank session -> segment x trending blend.
    if is_new_user and not viewed:
        return _segment_candidates(request, feats, active, trending)

    if not viewed:
        return {}  # warm user, no session signal: Person 2's territory

    weights = _view_weights(request, viewed, req_ts)

    # Branch B (new user WITH views): PURE session item-to-item similarity.
    scores: dict[str, float] = {}
    if is_new_user:
        scores = _session_similarity(request, viewed, weights, feats, active)

    # Branch C (EVERY request with views): inject relevant NEW products.
    return _inject_new_products(viewed, weights, feats, active, scores)


def _active_set(live_cache: Any) -> set[str] | None:
    """Best-effort active/in-stock set from Person 1's cache (None = unknown)."""
    for attr in ("active", "active_products", "in_stock"):
        value = getattr(live_cache, attr, None)
        if isinstance(value, (set, frozenset)):
            return set(value)
    return None


def _viewed_products(request: Record) -> list[str]:
    """Session views ordered oldest -> newest, duplicates dropped."""
    seen: list[str] = []
    for entry in request.get("recent_views") or []:
        pid = entry.get("product_id") if isinstance(entry, dict) else entry
        pid = str(pid) if pid is not None else ""
        if pid and pid not in seen:
            seen.append(pid)
    return seen


def _parse_ts(value: Any) -> datetime | None:
    """RawTree timestamps look like '2026-11-14 13:25:38.291000000'."""
    try:
        return datetime.fromisoformat(str(value)[:26])
    except (ValueError, TypeError):
        return None


def _view_weights(request: Record, viewed: list[str],
                  req_dt: datetime | None) -> list[float]:
    """Recency weight per view (aligned with deduped `viewed`); newest ~1.0."""
    if req_dt is not None:
        weights: list[float] = []
        seen: set[str] = set()
        for entry in request.get("recent_views") or []:
            pid = entry.get("product_id") if isinstance(entry, dict) else entry
            dt = _parse_ts(entry.get("ts") if isinstance(entry, dict) else None)
            if pid is None or dt is None or str(pid) in seen:
                continue  # dedupe exactly like _viewed_products
            seen.add(str(pid))
            hours = max((req_dt - dt).total_seconds() / 3600.0, 0.0)
            weights.append(0.5 ** (hours / VIEW_HALF_LIFE_HOURS))
        if len(weights) == len(viewed):
            return weights
    # fallback: position decay, oldest -> newest so the LAST view weighs 1.0
    n = len(viewed)
    return [VIEW_DECAY_FACTOR ** (n - 1 - i) for i in range(n)]


def _session_similarity(request: Record, viewed: list[str],
                        weights: list[float], feats: CatalogFeatures,
                        active: set[str] | None) -> dict[str, float]:
    """Branch B: sum content similarity over the most-recent session seeds.

    Pure item-to-item: no popularity, no trending -- the session is the signal.
    """
    scores: dict[str, float] = {}
    viewed_set = set(viewed)
    for pid, w in zip(viewed[-MAX_SEED_VIEWS:], weights[-MAX_SEED_VIEWS:]):
        for nb, sim in feats.neighbors.get(pid, ()):
            if nb in viewed_set:  # never recommend what they already saw
                continue
            if active is not None and nb not in active:
                continue
            scores[nb] = scores.get(nb, 0.0) + w * sim
    return scores


def _inject_new_products(viewed: list[str], weights: list[float],
                         feats: CatalogFeatures, active: set[str] | None,
                         scores: dict[str, float]) -> dict[str, float]:
    """Branch C: push relevant NEW products into the pool for any shopper.

    Uses the precomputed reverse map established -> [(new_id, sim), ...]; a
    shopper browsing similar items gets the new product with a small boost
    (under-explored). Only ever ADDS or UPGRADES a score, never lowers it.
    """
    viewed_set = set(viewed)
    for pid, w in zip(viewed[-MAX_SEED_VIEWS:], weights[-MAX_SEED_VIEWS:]):
        for new_pid, sim in feats.established_to_new.get(pid, ()):
            if new_pid in viewed_set:
                continue
            if active is not None and new_pid not in active:
                continue
            boosted = w * sim * NEW_PRODUCT_BOOST
            if boosted > scores.get(new_pid, 0.0):
                scores[new_pid] = boosted
    return scores


def _segment_candidates(request: Record, feats: CatalogFeatures,
                        active: set[str] | None,
                        trending: dict[str, float]) -> dict[str, float]:
    """Branch A: blank-session new users -> best converters in their context
    segment (hierarchical fallback), blended with the live trending list."""
    device = str(request.get("context.device") or "*")
    region = str(request.get("context.region") or "*")
    page = str(request.get("context.page") or "*")
    ranking: list[tuple[str, float]] | None = None
    for key in ((device, region, page),   # exact segment
                (device, region, "*"),    # any page
                (device, "*", "*"),       # any region
                ("*", "*", "*")):         # global
        candidate = feats.segment_rankings.get(key)
        if candidate:
            ranking = candidate
            break
    if ranking is None:
        ranking = feats.global_ranking
    if not ranking:
        return {}

    scores: dict[str, float] = {}
    top = ranking[:SEGMENT_TOP_N]
    max_rate = top[0][1] or 1.0
    for pid, rate in top:  # segment score normalised to [0, 1]
        if active is None or pid in active:
            scores[pid] = SEGMENT_WEIGHT * (rate / max_rate)

    if trending:  # trending scores normalised into [0, 1] before blending
        t_max = max(trending.values()) or 1.0
        for pid, tv in trending.items():
            if active is None or pid in active:
                scores[pid] = scores.get(pid, 0.0) + TRENDING_WEIGHT * (tv / t_max)
    return scores
