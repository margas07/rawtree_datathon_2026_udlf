"""Online adaptation: feedback-driven UCB bandit + price-drop boosts (Person 3).

Learns from OUR OWN live feedback (team_NN.feedback) what the static rankers
cannot see: which products convert *right now*. Three pieces:

1. INGEST (background thread only, never the request path)
   A daemon thread polls team_NN.feedback every few seconds and folds each row
   into a BanditState: per-product slot-weighted impressions, clicks and
   purchases (carousel slot i counts 0.6**i, matching the hard slot-CTR decay
   measured on shop.history_sessions in M1). Rows served by the baseline
   (`served_by == 'fallback'`) are NOT our policy's arms, so they are ignored
   by default; the `__init__` placeholder row is always skipped. Polling is
   incremental (ts watermark); if the cast SQL is rejected the thread falls
   back to a recent-window poll and relies on request_id dedup, so repeated
   polls are idempotent. The same thread periodically refreshes the price-drop
   map from shop.catalog_updates.

2. UCB-SHAPED EXPLORATION + EXPLOITATION
   apply_online_adjustments multiplies each candidate score by
     * an exploration bonus  1 + UCB_EXPLORATION*sqrt(log(T+2)/(n+1)), capped
       at EXPLORE_MAX: under-shown products get promoted (UCB shape);
     * an exploitation multiplier  (rate_i / global_rate) ** PERF_EXPONENT,
       clamped to [PERF_MULT_MIN, PERF_MULT_MAX], where rate_i is the
       slot-weighted success rate smoothed with PRIOR_N pseudo-impressions at
       the global rate (a purchase counts as PURCHASE_REWARD clicks);
     * a price-drop multiplier  1 + PRICE_DROP_WEIGHT * drop_fraction.

3. PRICE-DROP MULTIPLIER
   shop.catalog_updates carries NO old_price field (M1 finding), so a drop is
   detected by comparing the latest `price_change` price against the product's
   reference price (historical max = the "cached baseline"). A 20% discount
   lifts the score by PRICE_DROP_WEIGHT*0.2 (+10%): enough to break ties and
   lift near-equal candidates, never enough to bury a much better product.
   The map is rebuilt on every refresh, so a recovered price clears its boost.

REQUEST-PATH CONTRACT (agreed with P1/P2, docs/progress.md section 5):
    scores = get_cold_start_candidates(request, live_cache)   # M2
    scores = apply_online_adjustments(scores, live_cache)     # M3 (this file)
    # then merge with Person 2's warm-user scores, keep top-10 distinct ACTIVE
apply_online_adjustments only touches precomputed dicts (no SQL, no numpy),
so the 50 ms budget is safe even for 5k-product catalogs (tested). The state
is read from the cache attribute `bandit_state` (aliases accepted); when it is
missing - or there is no feedback yet - candidates pass through unchanged.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from .client import Record

# ---- Tunables (single source of truth for the jury slide) ------------------ #
SLOT_DECAY = 0.6            # exposure weight of carousel slot i = 0.6**i
PURCHASE_REWARD = 3.0       # a purchase counts as 3 clicks in the reward
PRIOR_N = 100.0             # Bayesian prior impressions when smoothing rates
UCB_EXPLORATION = 0.30      # width of the UCB exploration bonus
EXPLORE_MAX = 2.0           # cap on the exploration multiplier
PERF_EXPONENT = 0.5         # sqrt-damped exploitation multiplier
PERF_MULT_MIN = 0.5         # clamp of the exploitation multiplier ...
PERF_MULT_MAX = 2.0         # ... so a fluke can never dominate the ranker
PRICE_DROP_MIN_REL = 0.03   # >= 3% below the reference price counts as a drop
PRICE_DROP_WEIGHT = 0.5     # multiplier = 1 + WEIGHT * drop_fraction
PRICE_DROP_MAX_FRAC = 0.5   # drop fraction capped (a 70% sale counts as 50%)
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_PRICE_REFRESH_SECONDS = 60.0
FEEDBACK_WINDOW = 5000      # rows fetched by the recent-window fallback poll
MAX_TRACKED_REQUESTS = 1_000_000

# --------------------------------------------------------------------------- #
# SQL fetchers (background thread only -- never in the request path)          #
# --------------------------------------------------------------------------- #
_FEEDBACK_SELECT = """SELECT request_id::String AS request_id, ts,
       shown, clicked::String AS clicked, purchased::String AS purchased,
       served_by::String AS served_by, fallback_reason::String AS fallback_reason
FROM {team}.feedback"""


def feedback_recent_sql(team: str, limit: int = FEEDBACK_WINDOW) -> str:
    """Most recent feedback rows, newest first (startup + dialect fallback)."""
    return (_FEEDBACK_SELECT.format(team=team)
            + f"\nWHERE request_id != '__init__'\nORDER BY ts DESC\nLIMIT {int(limit)}")


def feedback_since_sql(team: str, since: str) -> str:
    """Incremental poll: every feedback row with ts after `since`.

    `since` is 'YYYY-MM-DD HH:MM:SS.mmm' (BanditState.watermark_sql). If the
    dialect rejects toDateTime64 the caller falls back to feedback_recent_sql
    and request_id dedup keeps the statistics correct.
    """
    since = str(since).replace("'", "")
    return (_FEEDBACK_SELECT.format(team=team)
            + "\nWHERE request_id != '__init__'"
            + f" AND ts::DateTime64(3) > toDateTime64('{since}', 3)")


PRICE_CHANGES_SQL = """
SELECT product_id::String AS product_id,
       argMax(price_eur, ts::DateTime64(3)) AS latest_price,
       max(price_eur)                     AS ref_price
FROM shop.catalog_updates
WHERE change::String = 'price_change'
GROUP BY product_id
"""
"""One row per ever-discounted product: current price + reference price.
M1 finding: catalog_updates has no old_price, so `ref_price` (the historical
max) plays the role of the cached baseline price for drop detection."""


# --------------------------------------------------------------------------- #
# Small parsing helpers                                                       #
# --------------------------------------------------------------------------- #
def _parse_ts(value: Any) -> datetime | None:
    """RawTree timestamps look like '2026-11-14T16:02:16.900Z' or
    '2026-11-14 13:25:38.291000000'; the first 26 chars always parse."""
    try:
        return datetime.fromisoformat(str(value)[:26])
    except (ValueError, TypeError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_id_list(value: Any) -> list[str]:
    """`shown` is an Array(String) in API responses, but schemaless Dynamic
    columns can occasionally arrive stringified -- accept both."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    text = str(value or "").strip()
    if not text or text == "[]":
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return [str(v) for v in parsed if v is not None and str(v) != ""]
    return [text]


# --------------------------------------------------------------------------- #
# State (shared by the background thread and the request path)                #
# --------------------------------------------------------------------------- #
@dataclass
class BanditState:
    """Thread-safe per-product online statistics.

    `impressions` are slot-weighted: carousel slot i counts 0.6**i, so a
    product shown in slot 1 earns a full impression while one in slot 10 earns
    almost nothing (the history slot-CTR decays exactly like this). Clicks and
    purchases are plain counts; in the reward used by the UCB formulas a
    purchase weighs PURCHASE_REWARD times a click.
    """

    impressions: dict[str, float] = field(default_factory=dict)
    clicks: dict[str, int] = field(default_factory=dict)
    purchases: dict[str, int] = field(default_factory=dict)
    price_drops: dict[str, float] = field(default_factory=dict)
    seen_requests: dict[str, None] = field(default_factory=dict)  # ordered dedup
    total_requests: int = 0
    last_ts: datetime | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def snapshot(self) -> SimpleNamespace:
        """Cheap consistent copy for the request path (shallow dict copies)."""
        with self.lock:
            return SimpleNamespace(
                impressions=dict(self.impressions),
                clicks=dict(self.clicks),
                purchases=dict(self.purchases),
                price_drops=dict(self.price_drops),
                total_requests=self.total_requests,
                total_exposure=sum(self.impressions.values()),
                total_clicks=sum(self.clicks.values()),
                total_purchases=sum(self.purchases.values()),
            )

    def watermark_sql(self) -> str | None:
        """High-water SQL timestamp for incremental polling (None = cold start)."""
        with self.lock:
            dt = self.last_ts
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if dt is not None else None

    def summary(self) -> dict[str, Any]:
        """Tiny KPI dict for logs / the jury demo."""
        with self.lock:
            exposure = sum(self.impressions.values())
            clicks = sum(self.clicks.values())
            purchases = sum(self.purchases.values())
            counts = (self.total_requests, len(self.impressions), self.last_ts)
        return {
            "requests_folded": counts[0],
            "products_tracked": counts[1],
            "exposure": round(exposure, 1),
            "clicks": clicks,
            "purchases": purchases,
            "ctr": round(clicks / exposure, 4) if exposure else 0.0,
            "cvr": round(purchases / exposure, 4) if exposure else 0.0,
            "price_drops": len(self.price_drops),
            "last_ts": counts[2].isoformat() if counts[2] else None,
        }


# --------------------------------------------------------------------------- #
# Ingest: pure row -> state folding (unit-tested offline with synthetic rows) #
# --------------------------------------------------------------------------- #
def ingest_feedback_rows(state: BanditState, rows: list[Record] | None, *,
                         include_fallback: bool = False) -> int:
    """Fold team_NN.feedback rows into `state`; return how many were folded.

    Skipped rows:
      * the `__init__` placeholder row (M1 finding: it must be filtered out);
      * duplicate request_ids (repeated / overlapping polls are idempotent);
      * when `include_fallback` is False (the default), rows with
        `served_by == 'fallback'`: those shoppers saw the baseline carousel,
        not our arms, so their outcomes say nothing about OUR products.

    Skipped rows still advance the watermark + dedup bookkeeping so that
    incremental polls make progress. Impression counts are slot-weighted with
    SLOT_DECAY; slot 0 is the head of `shown`.
    """
    folded = 0
    with state.lock:
        for row in rows or []:
            rid = str(row.get("request_id") or "")
            if not rid or rid == "__init__" or rid in state.seen_requests:
                continue
            state.seen_requests[rid] = None
            dt = _parse_ts(row.get("ts"))
            if dt is not None and (state.last_ts is None or dt > state.last_ts):
                state.last_ts = dt
            served = str(row.get("served_by") or "")
            if not include_fallback and served == "fallback":
                continue

            shown = _as_id_list(row.get("shown"))
            clicked = str(row.get("clicked") or "")
            purchased = str(row.get("purchased") or "")
            for slot, pid in enumerate(shown):
                state.impressions[pid] = (state.impressions.get(pid, 0.0)
                                          + SLOT_DECAY ** slot)
            if clicked:
                state.clicks[clicked] = state.clicks.get(clicked, 0) + 1
            if purchased:
                state.purchases[purchased] = state.purchases.get(purchased, 0) + 1
            state.total_requests += 1
            folded += 1

        if len(state.seen_requests) > MAX_TRACKED_REQUESTS:
            for _ in range(MAX_TRACKED_REQUESTS // 10):
                state.seen_requests.popitem(last=False)  # dicts are ordered
    return folded


def update_price_drops(state: BanditState, rows: list[Record] | None,
                       baseline_prices: dict[str, float] | None = None) -> int:
    """Rebuild `state.price_drops` from price_change rows; return #drops found.

    Args:
        rows: one row per product in the PRICE_CHANGES_SQL shape (`product_id`,
            `latest_price`, `ref_price`). `price_eur` is accepted as an alias
            for the latest price and later rows overwrite earlier ones.
        baseline_prices: optional {pid: baseline} cache override used when a
            row carries no ref_price. A row with NEITHER reference proves
            nothing (one price point alone is not a discount) and is skipped.

    The map is replaced wholesale on every refresh, so a product whose price
    recovered (latest >= reference) automatically loses its boost.
    """
    ref: dict[str, float] = dict(baseline_prices) if baseline_prices else {}
    latest: dict[str, float] = {}
    for row in rows or []:
        pid = str(row.get("product_id") or "")
        if not pid:
            continue
        price = _to_float(row.get("latest_price"))
        if price is None:
            price = _to_float(row.get("price_eur"))
        if price is None:
            continue
        latest[pid] = price
        base = _to_float(row.get("ref_price"))
        if base is None and baseline_prices:
            base = _to_float(baseline_prices.get(pid))
        if base is not None and base > 0.0:
            ref[pid] = base

    drops: dict[str, float] = {}
    for pid, price in latest.items():
        baseline = ref.get(pid)
        if baseline is None or baseline <= 0.0:
            continue
        frac = (baseline - price) / baseline
        if frac >= PRICE_DROP_MIN_REL:
            drops[pid] = min(frac, PRICE_DROP_MAX_FRAC)

    with state.lock:
        state.price_drops = drops
    return len(drops)


# --------------------------------------------------------------------------- #
# Request path: pure dict lookups only (50 ms budget)                         #
# --------------------------------------------------------------------------- #
def apply_online_adjustments(scored_candidates: dict[str, float],
                             live_cache: Any) -> dict[str, float]:
    """Rescore candidates with the online bandit signal (M3 entry point).

        scores = get_cold_start_candidates(request, live_cache)   # M2
        scores = apply_online_adjustments(scores, live_cache)     # here
        # merge with P2's warm-user scores, keep top-10 distinct ACTIVE

    Multipliers per candidate pid (all from the precomputed state snapshot):
      * exploration  1 + UCB_EXPLORATION*sqrt(log(T+2)/(n+1)), capped so a
        never-shown product can at most double its score (classic UCB shape,
        where T = total slot-weighted exposure and n = the product's own);
      * exploitation (rate / global_rate) ** PERF_EXPONENT clamped to
        [PERF_MULT_MIN, PERF_MULT_MAX]; rate = (success + global_rate*PRIOR_N)
        / (exposure + PRIOR_N), success = clicks + PURCHASE_REWARD*purchases;
      * price drop   1 + PRICE_DROP_WEIGHT * drop_fraction (0 when none).

    Returns a NEW dict; the input is never mutated. Without a state, with no
    folded feedback yet, or with a price-only state, scores pass through as a
    copy (so early on we never uniformly inflate our candidates relative to
    Person 2's warm-user scores).
    """
    if not scored_candidates:
        return {}
    state = _bandit_state(live_cache)
    if state is None:
        return dict(scored_candidates)
    snap = state.snapshot()
    if snap.total_requests <= 0 and not snap.price_drops:
        return dict(scored_candidates)

    g_rate = ((snap.total_clicks + PURCHASE_REWARD * snap.total_purchases + 0.5)
              / (snap.total_exposure + PRIOR_N))
    log_term = math.log(snap.total_exposure + 2.0)
    out: dict[str, float] = {}
    for pid, base in scored_candidates.items():
        n = snap.impressions.get(pid, 0.0)
        explore = 1.0 + UCB_EXPLORATION * math.sqrt(log_term / (n + 1.0))
        if explore > EXPLORE_MAX:
            explore = EXPLORE_MAX
        exploit = 1.0
        if n > 0.0 and g_rate > 0.0:
            succ = (snap.clicks.get(pid, 0)
                    + PURCHASE_REWARD * snap.purchases.get(pid, 0))
            rate = (succ + g_rate * PRIOR_N) / (n + PRIOR_N)
            exploit = (rate / g_rate) ** PERF_EXPONENT
            exploit = min(max(exploit, PERF_MULT_MIN), PERF_MULT_MAX)
        drop = snap.price_drops.get(pid, 0.0)
        out[pid] = base * exploit * explore * (1.0 + PRICE_DROP_WEIGHT * drop)
    return out


def _bandit_state(live_cache: Any) -> BanditState | None:
    """Best-effort BanditState lookup on Person 1's cache (None = not there)."""
    for attr in ("bandit_state", "bandit", "online_state", "feedback_state"):
        obj = getattr(live_cache, attr, None)
        if isinstance(obj, BanditState):
            return obj
    return None


# --------------------------------------------------------------------------- #
# Background ingest loop                                                      #
# --------------------------------------------------------------------------- #
class FeedbackIngestor:
    """Background daemon that keeps a BanditState in sync with the live stream.

    Feedback is the low-latency signal (written seconds after each request's
    deadline) so it is polled every `poll_seconds`; price drops matter on the
    scale of minutes, so they are refreshed every `price_refresh_seconds`.
    Every error is recorded (`errors`, `last_error`) and swallowed: the loop
    must survive flaky API moments, shoppers keep arriving while it sleeps.
    The thread is a daemon, so an interrupted process always shuts down cleanly.
    """

    def __init__(self, client, state: BanditState | None = None, *,
                 poll_seconds: float = DEFAULT_POLL_SECONDS,
                 price_refresh_seconds: float = DEFAULT_PRICE_REFRESH_SECONDS,
                 include_fallback: bool = False,
                 feedback_window: int = FEEDBACK_WINDOW) -> None:
        self.client = client
        self.state = state if state is not None else BanditState()
        self.team = str(getattr(client, "team_id", "") or "team_12")
        self.poll_seconds = float(poll_seconds)
        self.price_refresh_seconds = float(price_refresh_seconds)
        self.include_fallback = bool(include_fallback)
        self.feedback_window = int(feedback_window)
        self.errors = 0
        self.last_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_price_refresh = -1e18  # force a first price refresh

    # ---- lifecycle -------------------------------------------------------- #
    def start(self) -> "FeedbackIngestor":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop,
                                            name="bandit-feedback-ingest",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> "FeedbackIngestor":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- one round of work (public so tests / scripts can drive it) ------- #
    def run_once(self, *, force_prices: bool = False) -> int:
        """Poll feedback once (+ price refresh when due); return folded rows."""
        folded = self._poll_feedback()
        now = time.monotonic()
        if force_prices or (now - self._last_price_refresh
                            >= self.price_refresh_seconds):
            self._last_price_refresh = now
            try:
                update_price_drops(self.state,
                                   self.client.query(PRICE_CHANGES_SQL))
            except Exception as exc:  # noqa: BLE001 -- keep the loop alive
                self._record_error(exc)
        return folded

    # ---- internals -------------------------------------------------------- #
    def _poll_feedback(self) -> int:
        since = self.state.watermark_sql()
        if since is not None:
            before = self.errors
            folded = self._fetch_and_ingest(feedback_since_sql(self.team, since))
            if self.errors == before:
                return folded
            # incremental SQL rejected (dialect/HTTP) -> recent-window fallback
        return self._fetch_and_ingest(
            feedback_recent_sql(self.team, self.feedback_window))

    def _fetch_and_ingest(self, sql: str) -> int:
        try:
            rows = self.client.query(sql)
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            return 0
        return self._ingest(rows)

    def _ingest(self, rows) -> int:
        try:
            return ingest_feedback_rows(self.state, rows,
                                        include_fallback=self.include_fallback)
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            return 0

    def _record_error(self, exc: Exception) -> None:
        self.errors += 1
        self.last_error = exc

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 -- never die
                self._record_error(exc)