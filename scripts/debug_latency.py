"""Debug: reproduce the empty-scores case from the latency test."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cold_start import CatalogFeatures, get_cold_start_candidates

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
    established_to_new={"p_00002": [("p_00001", 0.9)]},
    segment_rankings={("*", "*", "*"): [(pid, 0.02) for pid in ids[:40]]},
    global_ranking=[(pid, 0.02) for pid in ids[:40]],
)
cache = SimpleNamespace(active=set(ids), trending={ids[i]: 1.0 for i in range(50)})
new_empty = {"is_new_user": True, "recent_views": [], "ts": "2026-11-14 13:00:00",
             "context.device": "web", "context.region": "Catalonia",
             "context.page": "home"}
new_views = {"is_new_user": True, "ts": "2026-11-14 13:00:00",
             "recent_views": [{"product_id": ids[i]} for i in range(5)]}
warm_views = {"is_new_user": False, "ts": "2026-11-14 13:00:00",
              "recent_views": [{"product_id": ids[i]} for i in range(5)]}

for name, req in [("new_empty", new_empty), ("new_views", new_views),
                  ("warm_views", warm_views)]:
    s = get_cold_start_candidates(req, cache, feats)
    print(name, "->", len(s), "candidates", list(s.items())[:3])
