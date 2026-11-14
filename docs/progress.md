# Progress / Memory Bank — Person 3 (Cold Start, Online Adaptation & Jury Metrics)

> **Purpose:** persistent context so any future session can resume without re-probing.
> Update after every milestone. Last updated: 2026-11-14, Milestone 4 COMPLETE (incl. charts addendum).

## 0. Task summary (carry-over constraints)
- **Goal (RawTree Challenge, Datathon FME 2026):** recommend 10 distinct ACTIVE
  product_ids per `shop.requests` row within 3 s; conversion scored slot-weighted
  (slots 1–3 get the most attention).
- **My components (Person 3):** `src/cold_start.py` (new users + new products),
  `src/bandit.py` (feedback-driven UCB + price-drop multiplier),
  `experiment_logger.py` / `jury_report.py` (offline IPS metrics + live KPIs).
- **Hard rules:** request path < 50 ms (precomputed dicts/NumPy only, no
  per-request SQL); never return out-of-stock items; clean typed interfaces
  (`get_cold_start_candidates(request, live_cache)`,
  `apply_online_adjustments(scored_candidates, live_cache)`); teammates own
  polling/cache (P1) and warm-user co-visitation ranker (P2).
- **Operating rule:** one milestone at a time, test each, wait for confirmation.

## 1. Environment facts
- Windows, PowerShell; **Python only via `py`** (3.13.7, no `python` on PATH).
- Installed: requests, numpy, pandas, scikit-learn, scipy, matplotlib, pytest,
  python-dotenv 1.2.4 (was missing; installed during M1).
- Shell output capture is flaky → prefer writing output to a file and reading
  it, or pytest `--junitxml`.
- `.env`: `TEAM_ID=team_12`, `RAWTREE_API_KEY=...` (verified working; never share).
- RawTree API: `POST https://api.rawtree.com/v1/query` `{"sql": ...}`, Bearer
  auth → `{"meta":[...], "data":[rows]}`; insert:
  `POST /v1/tables/{table}?database=team_NN`.
- SQL dialect: schemaless Dynamic columns; casts needed for joins/aggregates
  (`product_id::String`, `ts::DateTime64(3)`, `change::String`); `ARRAY JOIN`
  works; `LIMIT n BY` works.

## 2. Schema findings from live probes (Milestone 1, read-only queries)
1. **Live stream IS running** (day 1): `shop.requests` had 47,736 rows,
   `shop.events` ~1.06M at first probe (2026-11-14 ~13:30 UTC+1).
2. **shop.requests row (flattened keys):** `request_id, ts, user_id,
   is_new_user (bool), context.device, context.page, context.region,
   recent_views [{product_id, ts}]`. **~30% of requests have is_new_user=True.**
3. **shop.catalog_updates rows carry the FULL product snapshot** (product_id,
   category, subcategory, brand, description, price_eur = NEW price, is_active,
   change, ts). **No old_price field** → price-drop detection = compare update
   `price_eur` vs cached baseline price. Event counts at probe time:
   price_change 1,241 · new_product 100 · out_of_stock 162 · back_in_stock 126.
4. **shop.events row:** `product_id, session_id, user_id, event_type, device,
   region, price_eur, ts` (flat, not nested).
5. **team_NN.feedback row:** `request_id, ts, shown[], clicked ('' or id),
   purchased ('' or id), revenue_eur, latency_ms (null when missed),
   fallback_reason (''/missing/...), served_by ('init'/'fallback'/...)` plus one
   `request_id == '__init__'` row that must be filtered out.
6. **History (offline) stats:** `shop.history_sessions` 2M rows (same fields as
   live requests + `shown[10]`, `propensity[10]`, `clicked`, `purchased`,
   `revenue_eur`; **no is_new_user** → offline cold-user proxy:
   `length(recent_views) <= 1`, ~45% of sessions). `shop.history_events` 12M rows
   (10.35M view / 1.09M add_to_cart / 548k purchase; 4,809/5,000 products ever
   purchased). Catalog: 10 categories (Home & Garden 640 … Beauty & Health 322),
   89 subcategories, 80 brands, price €1.99–€643.99 (avg €47.19).
   Slot CTR decays hard: clicks slot1 21,483 → slot10 3,260.
7. `team_12.submissions` had 797 rows while feedback showed
   `served_by='fallback', fallback_reason='missing'` → most shoppers were getting
   the baseline; integrate fast.

## 3. Files created so far
- `src/client.py` — shared `RawTreeClient` (`query` retrying, `query_df`,
  `insert`), `.env` loading with fallback parser. Smoke-tested OK.
- `src/__init__.py`, `src/tests/__init__.py`.
- `scripts/probe_data.py` — history/catalog stats probe (read-only, reusable).
- `scripts/probe_stream.py` — live-stream record shapes probe (read-only).
- `src/cold_start.py` — M2 module (see §4).
- `src/tests/test_cold_start.py` — 11 offline unit tests + 5k-product latency test.
- `scripts/bench_cold_start.py` — M2b benchmark (real catalog end-to-end:
  fetch catalog/updates/segment stats, build features, latency over 300 real
  `shop.requests` rows). Writes `bench_report.txt` itself; run detached:
  `Start-Process py -ArgumentList '-X','utf8','scripts\bench_cold_start.py'`.
- `bench_report.txt` — latest benchmark output (M2 PASS evidence).
- `src/bandit.py` — M3 module (see §4).
- `src/tests/test_bandit.py` — 21 offline unit tests (synthetic feedback rows,
  fake client for the background thread, 5k-candidate latency test).
- `scripts/probe_bandit.py` — live SQL probe for the bandit queries
  (read-only, writes `bandit_probe_report.txt` itself).
- `bandit_probe_report.txt` — latest live-probe output (M3 PASS evidence).
- `src/experiment_logger.py` — M4 module (see §4): notebook-identical offline
  IPS evaluation + `ExperimentLogger` (JSONL + team_12.scratch push) +
  `fetch_live_kpis` (4 guarded read-only feedback queries).
- `src/jury_report.py` — M4 module: `build_jury_report()` renders the PM/jury
  Markdown page; `scripts/jury_report.py` is the runner.
- `src/tests/test_experiment_logger.py` — 28 offline unit tests (fake client),
  incl. the M4 addenda below (dual kind/type tag, p99, cold join, slot CTR,
  chart builder, report chart section).
- `scripts/probe_jury.py` — M4 live probe; writes `jury_probe_report.txt`
  itself; run detached like the others.
- `scripts/probe_ddl.py` — one-off probe proving CREATE TABLE needs admin
  permission (403); documents the scratch-table decision.
- `jury_probe_report.txt` — latest M4 probe output (PASS evidence).
- `src/charts.py` — M4 addendum: `collect_chart_data` + `build_charts` render
  four best-effort PNGs into `reports/` (cvr, cold_start_lift,
  latency_percentiles with p50/p95/p99 + 3 s line, slot_ctr_curve); Agg
  backend, one chart failing never blocks the others.
- `reports/` — the generated PNGs (24–30 kB each, regenerated by the
  runner/probe); `jury_report.md` links them from its "## Charts" section.
- `jury_report.md` — the rendered jury page (regenerated by the runner/probe).

## 4. Milestone status
- **M1 DONE** — client + probes, findings in §2.
- **M2 DONE (2026-11-14)** — `cold_start.py` complete, 11/11 tests green,
  real-catalog benchmark PASS. Details:
  - Design as planned: `CatalogFeatures` (TF-IDF text ×1.0 + one-hot category
    0.8 / subcategory 1.2 / brand 0.6 + min-max log price 0.4, L2-normalised →
    cosine = dot), top-50 neighbours, `new_products` set, reverse map
    `established_to_new`, segment rankings with 4-level wildcard fallback
    `(d,r,p) → (d,r,*) → (d,*,*) → (*,*,*)`, smoothed rate
    `(purchases+1)/(impressions+50)`.
  - Request path: A) new user no views → segment × trending (0.75/0.25);
    B) new user with views → pure session item-to-item (recency half-life 2 h,
    position-decay fallback 0.6); C) any request with views → new-product
    injection via `established_to_new` (boost 1.25). Viewed items excluded;
    inactive filtered via `live_cache.active`.
  - **Tests: 11/11 green** (`pytest_report.xml`). Note: the "trending must not
    leak" test checks `scores.get("p_d", 0) < 0.1` instead of absence — in a
    tiny catalog every product is a top-50 neighbour, so weak content
    similarity (~0.003) is expected; trending blending would add ≥0.25.
  - **`SEGMENT_STATS_SQL` live-validated**: raw `ARRAY JOIN shown` fails with
    "Nullable ... not allowed inside Variant"; `shown` is a Dynamic column so
    it must be cast first. Working form:
    `ARRAY JOIN arrayMap(x -> coalesce(toString(x), ''), CAST(shown, 'Array(String)')) AS shown_item`
    → 237,815 rows in 2.4 s, 124 segment keys built.
  - **Benchmark (bench_report.txt, 300 real requests):** build 3.4 s for 5,200
    products (matrix 5200×423, 200 new products — all from catalog_updates
    snapshots merged into shop.catalog); request path mean 0.38 ms, p50 0.36,
    p95 0.57, p99 0.77, max 1.02 ms → **PASS, 49× headroom**. Branch mix:
    20 A-blank / 69 B-new-with-views / 172 C-inject-only / 39 warm-no-views;
    no_candidates 59/300 (warm-no-views are empty BY DESIGN — P2's territory).
- **M3 DONE (2026-11-14)** — `bandit.py` complete, 33/33 tests green
  (21 new + 12 cold-start regression), live probe PASS. Details:
  - `BanditState` (thread-safe, `threading.Lock`): per-product slot-weighted
    `impressions` (float, slot i counts `0.6**i` — matches the measured slot-CTR
    decay), `clicks`, `purchases`, `price_drops {pid: frac}`, ordered dedup
    dict `seen_requests` (evicts oldest at 1M), ts watermark; `snapshot()` =
    shallow copies for the lock-light request path; `summary()` KPI dict.
  - `ingest_feedback_rows()` — skips `__init__`, duplicate request_ids and
    `served_by='fallback'` rows by default (baseline outcomes are not OUR
    arms; they still advance watermark+dedup); `include_fallback=True` opt-in.
  - UCB adjustments (`apply_online_adjustments`, M3 entry point, never
    mutates input): exploration `1 + 0.30*sqrt(log(T+2)/(n+1))` capped ×2.0;
    exploitation `(rate_i/g_rate)**0.5` clamped [0.5, 2.0] with purchase =
    3 clicks and Bayesian smoothing `rate=(succ+g*100)/(n+100)` (a product at
    exactly the global rate stays ×1.0; 0-success shrinkage is intended);
    price multiplier `1 + 0.5*drop_frac` (drop ≥3% counts, frac capped 0.5 →
    max ×1.25). Pass-through when no state, no feedback yet, or nothing at all
    (so early on we never uniformly inflate our scores vs P2's).
  - Price drops: since catalog_updates has NO old_price, reference price =
    historical max per product (`PRICE_CHANGES_SQL`: `argMax(price_eur, ts)`
    vs `max(price_eur)` per product — ONE query, validated live); map rebuilt
    on every refresh so recovered prices clear their boost automatically.
  - `FeedbackIngestor` — daemon thread (P1 starts it once):
    `FeedbackIngestor(client, live_cache.bandit_state).start()`; feedback
    polled every 30 s incremental via ts watermark
    `ts::DateTime64(3) > toDateTime64('...')` (VALIDATED LIVE), auto-fallback
    to a recent-window `ORDER BY ts DESC LIMIT 5000` poll with request_id
    dedup if the cast SQL is ever rejected; price map refreshed every 60 s;
    all errors swallowed + counted (`errors`, `last_error`).
  - P1 wiring (one line each): `live_cache.bandit_state = BanditState()`
    (aliases `bandit`/`online_state`/`feedback_state` accepted) before the
    loop; `FeedbackIngestor(client, live_cache.bandit_state).start()` in the
    background; in the request path call
    `apply_online_adjustments(scores, live_cache)` right after
    `get_cold_start_candidates`.
  - **Tests: 33/33 green** (`pytest_report.xml`): counts/slot-weights, dedup,
    init+fallback filtering, watermark, stringified-array `shown`, exact UCB
    multiplier math incl. both clamps, purchase>click, exact price-drop/cap/
    recovery math, no-mutation, aliases, manual run_once, incremental
    watermark poll, background thread with fake client (clean stop; survives
    3 failing feedback polls + always-failing price query and still ingests),
    and latency: 5k tracked products × 5k candidates × 300 calls →
    mean 8.5 ms, p50 8.1, p95 11.4, p99 12.4, max 15.9 ms → PASS (~3×
    headroom; dominated by the snapshot dict copies, ~8 ms at 5k products —
    negligible next to the 3 s answer deadline and the 50 ms CPU budget).
  - **Live probe (bandit_probe_report.txt): PASS** — recent window 500 rows
    in 1.3 s (served_by mix 484 team / 16 fallback → 484 folded), incremental
    toDateTime64 poll OK (0 new rows in 0.2 s), PRICE_CHANGES_SQL OK (1,286
    ever-discounted products, 0 active drops right now), run_once folded 3
    fresh rows → 487 requests tracked, exposure 1,210, CTR 1.24%, CVR 0.33%.
- **M4 DONE (2026-11-14)** — `experiment_logger.py` + `jury_report.py` complete,
  52/52 tests green, live probe PASS. Details:
  - `ExperimentLogger` — append-only JSONL at `logs/experiments.jsonl`
    (source of truth, works offline) + best-effort insert of the same record
    into `team_12.scratch` (`kind='experiment'/'live_snapshot'`; P1's
    `live_trends_snapshot` rows share the table — filter by kind). Insert
    failures counted (`push_errors`, `last_push_error`), never fatal; a
    missing table triggers ONE `CREATE TABLE IF NOT EXISTS` attempt + retry.
    API: `log/log_experiment/log_live_snapshot/load/to_dataframe/
    best(name, metric='snips_ctr_slot1')/last(kind)`. Record shape:
    `{ts, kind, name, metrics{}, params{}, tags[]}`.
  - Offline evaluation = the notebook's math, reusable:
    `evaluate(policy, rows)` / `evaluate_policy(recs, shown, propensity,
    clicked)` → `hit_rate@10, ips_ctr_slot1, snips_ctr_slot1,
    slot1_matches, n_test`. Held-out slice = last 7 days of
    `shop.history_sessions` (`heldout_cut`, `fetch_eval_sessions`,
    `ORDER BY rand() LIMIT 100000`, notebook columns). Deliberate
    divergence: zero-propensity rows carry weight 0 (the notebook divides
    → 0/0 NaN poisons the whole IPS column); no-match rows never divide.
  - `fetch_live_kpis(client)` — 4 independently-guarded read-only queries
    over `team_NN.feedback` (per-`served_by` conversion/revenue;
    latency `quantile(0.5/0.95)` + on-time vs 3000 ms — **quantile over a
    cast LIVE-VALIDATED**; `fallback_reasons`; hourly trend). Filter
    `request_id != '__init__'`; casts `served_by::String`,
    `ts::DateTime64(3)`, `revenue_eur::Float64`, `latency_ms::Float64`.
    Query failures collect into `errors`, never raise.
  - `jury_report.build_jury_report(client, bandit_state, logger)` →
    `(markdown_str, data_dict)`. Sections: PM-friendly system overview,
    offline experiments table (from the log), live KPI table +
    `BanditState.summary()`, tunables with justifications (values READ
    FROM SOURCE CONSTANTS so the report cannot drift), known limitations.
    All inputs degrade to `n/a`; `client=None` works; the bandit state is
    auto-warmed from the recent feedback window when not passed.
    `scripts/jury_report.py` writes `jury_report.md` at the repo root and
    logs one `live_snapshot` row.
  - **Live probe PASS (jury_probe_report.txt, 18:08 UTC)**: KPI SQL 0
    errors (71.8k requests/24 h, our_share 35.3%, CTR 2.55%, CVR 0.79%,
    on-time 99.76%, our p50/p95 1069/1934 ms); scratch round trip OK
    (push_errors 0, probe row read back); offline smoke eval on 300
    held-out sessions (cut 2026-11-07) works; `jury_report.md` rendered.
  - **Tests: 52/52 green** — 19 new M4 tests: hand-computed IPS/SNIPS,
    notebook-equivalence on random data, zero-propensity guard, JSONL
    round-trip/`best()`/`last()`, push-failure + missing-table retry,
    fake-client live-KPI aggregation + total-outage degradation, jury
    report sections + `client=None` resilience.

- probe verdict: kpi_ok push_ok offline_ok charts_ok report (all True required).
- **M4 addendum (2026-11-14, evening)** — verified against the spec's
  `"type": "experiment"` wording and added the PNG chart layer:
  - `ExperimentLogger` now tags every record with BOTH `kind` and
    `type` (identical values), so scratch rows are queryable either way;
    live read-back confirmed `WHERE __raw_data.type::String = 'experiment'`
    returns the probe row (older pre-addendum rows only carry `kind`).
  - `KPI_LATENCY_SQL` now also returns `quantile(0.99)` — live p50/p95/p99
    (team): 1060 / 1965 / 2902 ms — p99 is 200 ms under the 3 s deadline.
  - `KPI_COLD_SQL` (NEW, live-validated): `team_NN.feedback` INNER JOIN
    `shop.requests` on `request_id::String`, grouped by `is_new_user::String`
    × `served_by` → cold CVR 1.74% (ours) vs 1.34% (fallback) ≈ 1.3x lift;
    warm 0.56% vs 0.46%. `fetch_cold_lift()` guards it like the other queries.
  - `SLOT_CTR_SQL` (NEW, live-validated): per-slot CTR over
    `shop.history_sessions` via `ARRAY JOIN shown_arr AS pid,
    arrayEnumerate(shown_arr) AS slot` with the nullable-safe
    `arrayMap(x -> coalesce(x,''), CAST(shown,'Array(String)'))` — s1 1.07%
    → s10 0.16%.
  - Charts (src/charts.py, see §3) + `build_jury_report(charts=...)` image
    section; runner `scripts/jury_report.py` regenerates everything in one
    run; `scripts/probe_jury.py` verdict now includes charts_ok.
  - **Tests: 61/61 green** (28 in test_experiment_logger.py); probe PASS in
    5.8 s with zero warnings on all four charts.

## 5. Interface contract agreed with P1/P2 (do not break)
```python
# P1 background refresh:
live_cache.active = set[str]              # active + in stock (final guard is P1's)
live_cache.trending = dict[str, float]    # 1-hour trending
live_cache.cold_features = build_catalog_features(catalog_rows, updates_rows,
                                                  segment_rows)
# P1 request path:
scores = get_cold_start_candidates(request, live_cache)  # -> dict[pid, float]
# P3 (M3): apply_online_adjustments(scored_candidates, live_cache) -> dict[pid, float]
```
`cold_start` reads cache attrs `active` / `active_products` / `in_stock`
(best-effort), `trending`, `cold_features`; explicit `features=` kwarg
overrides for tests/benchmarks. Warm users get only new-product injection
(P2's ranker leads for warm traffic).

M3 addition (same rules): P1 creates `live_cache.bandit_state = BanditState()`
once, starts `FeedbackIngestor(client, live_cache.bandit_state)` in the
background loop, and calls `apply_online_adjustments(scores, live_cache)`
immediately after `get_cold_start_candidates` in the request path. Missing
state / zero feedback => scores pass through unchanged, so wiring is safe at
any time. Latency added by the bandit step at 5k products/candidates:
mean ~8.5 ms, max ~16 ms (snapshot copies) -> still well inside the budget.

## 6. Gotchas learned
- PowerShell mangles inline JSON quoting → use `Invoke-RestMethod` or a Python
  script file, never `curl -d '{...}'` inline.
- PS `>` redirect writes UTF-16 (readable but noisy) → prefer `--junitxml` or
  `py -X utf8` scripts writing their own files.
- pytest console output often lost in shell integration → always use
  `--junitxml=pytest_report.xml` and read the XML.
- **Long-running scripts get KILLED by the next foreground terminal command**
  (flaky shell integration) → launch via `Start-Process py ... -WindowStyle
  Hidden` and poll the output file instead of sleeping in foreground.
- `ARRAY JOIN` on a schemaless/Dynamic array column needs
  `CAST(col, 'Array(String)')` (plus nullable-safe `arrayMap(coalesce(toString))`
  if elements can be null) — raw `ARRAY JOIN shown` → 400 "Nullable ... not
  allowed inside Variant type".
- `shop.catalog` does NOT contain `new_product` items — they only exist as
  full-snapshot rows in `shop.catalog_updates`; merge newest-snapshot-per-id
  into the catalog view before `build_catalog_features` (200 new products as of
  2026-11-14).
- TF-IDF on ~5k one-liner descriptions is small (fast build); dense 5000×D
  float32 matrix + 5000×5000 similarity matrix ≈ 100 MB — acceptable; rebuild
  per refresh takes seconds (background only, 3.4 s measured).
- Minor: `src/client.py` has a duplicated `import requests` (lines 24-25) —
  harmless, P1 may clean up.
