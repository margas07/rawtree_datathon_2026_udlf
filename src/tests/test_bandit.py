"""Unit tests for src/bandit.py -- fully offline (synthetic feedback rows).

Run from the project root:  py -m pytest src/tests/test_bandit.py -v
"""
from __future__ import annotations

import sys
import time
from math import log, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bandit import (  # noqa: E402
    EXPLORE_MAX,
    PERF_MULT_MAX,
    PERF_MULT_MIN,
    PRICE_CHANGES_SQL,
    PRICE_DROP_MAX_FRAC,
    PRICE_DROP_WEIGHT,
    PURCHASE_REWARD,
    SLOT_DECAY,
    UCB_EXPLORATION,
    BanditState,
    FeedbackIngestor,
    apply_online_adjustments,
    feedback_recent_sql,
    feedback_since_sql,
    ingest_feedback_rows,
    update_price_drops,
)
from src.client import RawTreeError  # noqa: E402


# ---------------------------------------------------------------- helpers --- #
def _fb(rid: str, shown, clicked: str = "", purchased: str = "",
        served_by: str = "team", ts: str = "2026-11-14 10:00:00"):
    """One synthetic team_NN.feedback row."""
    return {"request_id": rid, "ts": ts, "shown": list(shown),
            "clicked": clicked, "purchased": purchased, "revenue_eur": 0.0,
            "latency_ms": 100.0, "served_by": served_by, "fallback_reason": ""}


def _req_rows(pid: str, n: int, clicked: int = 0, purchased: int = 0):
    """`n` single-slot (slot 0 -> weight exactly 1.0) feedback rows for one
    product, `clicked` / `purchased` of which carry the outcome."""
    return [_fb(f"r_{pid}_{i}", [pid],
                clicked=pid if i < clicked else "",
                purchased=pid if i < purchased else "")
            for i in range(n)]


def _adjusted(scores, state):
    return apply_online_adjustments(scores, SimpleNamespace(bandit_state=state))


# ---------------------------------------------------------------- ingest ---- #
def test_ingest_counts_and_slot_weights() -> None:
    state = BanditState()
    rows = [_fb("r1", ["p_a", "p_b", "p_c"], clicked="p_a"),
            _fb("r2", ["p_a"], purchased="p_a")]
    assert ingest_feedback_rows(state, rows) == 2
    assert state.impressions == {"p_a": 2.0,
                                 "p_b": pytest.approx(SLOT_DECAY),
                                 "p_c": pytest.approx(SLOT_DECAY ** 2)}
    assert state.clicks == {"p_a": 1}
    assert state.purchases == {"p_a": 1}
    assert state.total_requests == 2


def test_init_row_and_fallback_rows_are_ignored() -> None:
    rows = [_fb("__init__", ["p_x"], clicked="p_x"),
            _fb("r_fb", ["p_y"], clicked="p_y", served_by="fallback")]
    state = BanditState()
    assert ingest_feedback_rows(state, rows) == 0
    assert state.impressions == {}
    assert state.total_requests == 0
    # but their bookkeeping still advances (dedup + watermark progress)
    assert "__init__" not in state.seen_requests
    assert "r_fb" in state.seen_requests
    assert state.watermark_sql() == "2026-11-14 10:00:00.000"
    # opt-in folding treats the baseline carousel as another policy's arm
    state2 = BanditState()
    ingest_feedback_rows(state2, rows, include_fallback=True)
    assert state2.impressions.get("p_y") == pytest.approx(1.0)
    assert "p_x" not in state2.impressions


def test_duplicate_rows_ingested_once() -> None:
    rows = [_fb("r1", ["p_a"]), _fb("r2", ["p_b"], clicked="p_b")]
    state = BanditState()
    assert ingest_feedback_rows(state, rows) == 2
    before = dict(state.impressions)
    assert ingest_feedback_rows(state, rows) == 0  # overlapping re-poll
    assert state.impressions == before
    assert state.clicks == {"p_b": 1}
    assert state.total_requests == 2


def test_watermark_tracks_max_ts() -> None:
    state = BanditState()
    assert state.watermark_sql() is None
    ingest_feedback_rows(state, [_fb("r1", ["p_a"], ts="2026-11-14 10:00:00")])
    ingest_feedback_rows(state, [_fb("r2", ["p_a"], ts="2026-11-14 09:00:00")])
    assert state.watermark_sql() == "2026-11-14 10:00:00.000"


def test_shown_arriving_as_stringified_array_is_handled() -> None:
    state = BanditState()
    ingest_feedback_rows(state, [{"request_id": "r1", "ts": "2026-11-14 10:00:00",
                                  "shown": '["p_a","p_b"]', "clicked": "p_a",
                                  "purchased": "", "served_by": "team"}])
    assert state.impressions == {"p_a": 1.0, "p_b": pytest.approx(SLOT_DECAY)}


# ------------------------------------------------------- UCB adjustments ---- #
def test_exploration_promotes_under_shown_products() -> None:
    state = BanditState()
    ingest_feedback_rows(state, _req_rows("p_heavy", 200))  # shown, no clicks
    adj = _adjusted({"p_heavy": 1.0, "p_fresh": 1.0}, state)
    # exact UCB multipliers: p_fresh has no data (exploit neutral, max
    # exploration); p_heavy's zero successes shrink to rate = g*100/300
    explore_heavy = 1.0 + UCB_EXPLORATION * sqrt(log(202.0) / 201.0)
    exploit_heavy = sqrt(100.0 / 300.0)                 # Bayesian shrink of 0/200
    explore_fresh = 1.0 + UCB_EXPLORATION * sqrt(log(202.0) / 1.0)
    assert adj["p_heavy"] == pytest.approx(explore_heavy * exploit_heavy,
                                           abs=1e-9)
    assert adj["p_fresh"] == pytest.approx(explore_fresh, abs=1e-9)
    assert adj["p_fresh"] > adj["p_heavy"]
    assert explore_fresh <= EXPLORE_MAX


def test_exploitation_prefers_proven_products() -> None:
    state = BanditState()
    ingest_feedback_rows(state, _req_rows("p_good", 100, clicked=30)
                         + _req_rows("p_bad", 100))
    adj = _adjusted({"p_good": 1.0, "p_bad": 1.0}, state)
    assert adj["p_good"] > adj["p_bad"]
    assert adj["p_good"] > 1.0          # above-average product is boosted
    assert adj["p_bad"] < 1.0           # below-average product is damped


def test_purchase_rewards_more_than_click() -> None:
    state = BanditState()
    ingest_feedback_rows(state, _req_rows("p_click", 1, clicked=1)
                         + _req_rows("p_buy", 1, purchased=1))
    adj = _adjusted({"p_click": 1.0, "p_buy": 1.0}, state)
    assert adj["p_buy"] > adj["p_click"] > 1.0


def test_exploitation_multiplier_is_clamped() -> None:
    state = BanditState()
    rows = _req_rows("p_hot", 1, purchased=1) + _req_rows("p_dud", 400)
    for j in range(40):                     # soak up exposure with no successes
        rows += _req_rows(f"p_fill_{j}", 10)
    ingest_feedback_rows(state, rows)       # exposure 801, 3 click-equivalents
    adj = _adjusted({"p_hot": 1.0, "p_dud": 1.0}, state)
    # p_hot: raw sqrt-ratio far above 2 -> clamped to PERF_MULT_MAX
    explore_hot = 1.0 + UCB_EXPLORATION * sqrt(log(803.0) / 2.0)
    assert adj["p_hot"] / 1.0 == pytest.approx(PERF_MULT_MAX * explore_hot,
                                               abs=1e-9)
    # p_dud: 400 exposures, zero clicks -> ratio sqrt(0.2) below floor
    explore_dud = 1.0 + UCB_EXPLORATION * sqrt(log(803.0) / 401.0)
    assert adj["p_dud"] / 1.0 == pytest.approx(PERF_MULT_MIN * explore_dud,
                                               abs=1e-9)


# ----------------------------------------------------------- price drops ---- #
def test_price_drop_detection_and_clearing() -> None:
    state = BanditState()
    n = update_price_drops(state, [
        {"product_id": "p_d", "latest_price": 80.0, "ref_price": 100.0},
        {"product_id": "p_up", "latest_price": 110.0, "ref_price": 100.0},
        {"product_id": "p_tiny", "latest_price": 99.0, "ref_price": 100.0},
        {"product_id": "p_big", "latest_price": 10.0, "ref_price": 100.0},
        {"product_id": "p_noref", "latest_price": 5.0},   # no reference: skip
    ])
    assert n == 2
    assert state.price_drops == {"p_d": pytest.approx(0.2),
                                 "p_big": pytest.approx(PRICE_DROP_MAX_FRAC)}
    # recovery clears the boost
    assert update_price_drops(state, [
        {"product_id": "p_d", "latest_price": 100.0, "ref_price": 100.0}]) == 0
    assert "p_d" not in state.price_drops
    # baseline_prices override fills in for a missing ref_price
    state2 = BanditState()
    update_price_drops(state2, [{"product_id": "p_d", "latest_price": 90.0}],
                       baseline_prices={"p_d": 100.0})
    assert state2.price_drops["p_d"] == pytest.approx(0.1)


def test_price_drop_multiplier_breaks_ties() -> None:
    state = BanditState()
    ingest_feedback_rows(state, [_fb("r_d", ["p_d"]), _fb("r_up", ["p_up"]),
                                 _fb("r_ref", ["p_ref"]),
                                 _fb("r_big", ["p_big"])])   # equal exposure
    update_price_drops(state, [
        {"product_id": "p_d", "latest_price": 80.0, "ref_price": 100.0},
        {"product_id": "p_big", "latest_price": 10.0, "ref_price": 100.0}])
    adj = _adjusted({"p_d": 2.0, "p_up": 2.0, "p_ref": 2.0, "p_big": 2.0}, state)
    # equal base scores + equal exposure -> ratio is exactly the price factor
    assert adj["p_d"] / adj["p_up"] == pytest.approx(1.0 + PRICE_DROP_WEIGHT * 0.2)
    assert (adj["p_big"] / adj["p_up"]
            == pytest.approx(1.0 + PRICE_DROP_WEIGHT * PRICE_DROP_MAX_FRAC))
    assert adj["p_ref"] == pytest.approx(adj["p_up"])  # unknown pid: neutral


def test_price_only_state_still_adjusts_drops() -> None:
    state = BanditState()                     # zero feedback, one price drop
    update_price_drops(state, [{"product_id": "p_d", "latest_price": 50.0,
                                "ref_price": 100.0}])
    adj = _adjusted({"p_a": 1.0, "p_d": 1.0}, state)
    assert adj["p_d"] / adj["p_a"] == pytest.approx(
        1.0 + PRICE_DROP_WEIGHT * PRICE_DROP_MAX_FRAC)


# ------------------------------------------- passthrough + safety checks ---- #
def test_scores_pass_through_without_a_state() -> None:
    scores = {"p_a": 1.0, "p_b": 0.5}
    cache = SimpleNamespace(active={"p_a"})            # no bandit attribute
    out = apply_online_adjustments(scores, cache)
    assert out == scores and out is not scores         # copy, not the input
    assert apply_online_adjustments({}, cache) == {}


def test_zero_data_state_passes_through() -> None:
    scores = {"p_a": 1.0}
    out = apply_online_adjustments(scores,
                                   SimpleNamespace(bandit_state=BanditState()))
    assert out == scores


def test_input_is_never_mutated() -> None:
    state = BanditState()
    ingest_feedback_rows(state, _req_rows("p_a", 10, clicked=2))
    scores = {"p_a": 1.0, "p_z": 0.5}
    snapshot = dict(scores)
    out = _adjusted(scores, state)
    assert scores == snapshot
    assert out is not scores and out["p_a"] > out["p_z"]


def test_state_is_found_under_every_agreed_alias() -> None:
    state = BanditState()
    ingest_feedback_rows(state, _req_rows("p_a", 10))
    results = [apply_online_adjustments(
        {"p_a": 1.0}, SimpleNamespace(**{attr: state}))
        for attr in ("bandit_state", "bandit", "online_state", "feedback_state")]
    assert all(r == results[0] for r in results)
    assert results[0]["p_a"] > 1.0                     # actually adjusted


# ------------------------------------------------- FeedbackIngestor loops --- #
class FakeClient:
    """Scripted stand-in for RawTreeClient: returns queued feedback pages and
    a fixed price-drop page; can fail a number of first calls per table."""

    def __init__(self, feedback_pages, price_rows=(), *,
                 fail_feedback_first: int = 0, fail_prices: bool = False):
        self.team_id = "team_12"
        self.feedback_pages = list(feedback_pages)
        self.price_rows = list(price_rows)
        self.fail_feedback_first = int(fail_feedback_first)
        self.fail_prices = bool(fail_prices)
        self.feedback_calls = 0
        self.price_calls = 0
        self.sqls: list[str] = []
        self._page = 0

    def query(self, sql: str, **kwargs):
        self.sqls.append(sql)
        if "catalog_updates" in sql:
            self.price_calls += 1
            if self.fail_prices:
                raise RawTreeError("price probe boom")
            return [dict(r) for r in self.price_rows]
        self.feedback_calls += 1
        if self.fail_feedback_first > 0:
            self.fail_feedback_first -= 1
            raise RawTreeError("feedback probe boom")
        page = self.feedback_pages[min(self._page, len(self.feedback_pages) - 1)]
        self._page += 1
        return [dict(r) for r in page]


ROWS = [_fb("r1", ["p_a", "p_b", "p_c"], clicked="p_a"),
        _fb("r2", ["p_a"], purchased="p_a"),
        _fb("r3", ["p_b", "p_d"], served_by="fallback")]


def _ingestor(client, state, **kw) -> FeedbackIngestor:
    defaults = dict(poll_seconds=0.02, price_refresh_seconds=3600.0)
    defaults.update(kw)
    return FeedbackIngestor(client, state, **defaults)


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_run_once_manual_poll_folds_feedback_and_prices() -> None:
    client = FakeClient([ROWS],
                        price_rows=[{"product_id": "p_a", "latest_price": 9.0,
                                     "ref_price": 10.0}])
    state = BanditState()
    ing = _ingestor(client, state, poll_seconds=999.0)
    assert ing.run_once() == 2                          # fallback row ignored
    assert state.total_requests == 2
    assert state.impressions["p_a"] == pytest.approx(2.0)
    assert state.price_drops["p_a"] == pytest.approx(0.1)
    assert client.feedback_calls == 1 and client.price_calls == 1
    assert "ORDER BY ts DESC" in client.sqls[0]         # startup: recent window


def test_incremental_poll_uses_the_watermark() -> None:
    client = FakeClient([ROWS, ROWS])
    state = BanditState()
    ing = _ingestor(client, state, poll_seconds=999.0)
    assert ing.run_once() == 2
    assert ing.run_once() == 0                          # duplicate rows skipped
    assert state.total_requests == 2
    feedback_sqls = [s for s in client.sqls if "feedback" in s]
    assert len(feedback_sqls) == 2
    assert "ORDER BY ts DESC" in feedback_sqls[0]       # first poll: full window
    assert "toDateTime64('2026-11-14 10:00:00.000', 3)" in feedback_sqls[1]


def test_background_thread_ingests_then_stops_cleanly() -> None:
    client = FakeClient([ROWS, ROWS],
                        price_rows=[{"product_id": "p_a", "latest_price": 9.0,
                                     "ref_price": 10.0}])
    state = BanditState()
    ing = _ingestor(client, state).start()
    try:
        assert _wait_until(lambda: state.total_requests >= 2)
        assert state.impressions["p_a"] == pytest.approx(2.0)
        assert state.clicks == {"p_a": 1}
        assert state.purchases == {"p_a": 1}
        assert state.price_drops["p_a"] == pytest.approx(0.1)
        assert ing.errors == 0
    finally:
        ing.stop()
    assert not ing.is_running


def test_thread_survives_api_errors_and_still_ingests() -> None:
    client = FakeClient([ROWS], fail_feedback_first=3, fail_prices=True)
    state = BanditState()
    ing = _ingestor(client, state).start()
    try:
        assert _wait_until(lambda: state.total_requests >= 2)
        assert ing.errors >= 1                          # failures were recorded
        assert state.impressions["p_a"] == pytest.approx(2.0)  # r1 + r2
    finally:
        ing.stop()
    assert not ing.is_running


# ---------------------------------------------------------------- latency --- #
def test_request_path_latency_with_5k_product_state() -> None:
    """5,000 tracked products, 5,000 candidates, 300 calls < 50 ms each."""
    rng = np.random.default_rng(11)
    ids = [f"p_{i:05d}" for i in range(5000)]
    state = BanditState()
    for pid in ids:
        n = float(rng.integers(0, 60))
        if n:
            state.impressions[pid] = n
            clicks = int(n * 0.06)
            state.clicks[pid] = clicks
            if clicks and rng.random() < 0.3:
                state.purchases[pid] = int(clicks * 0.3)
    for pid in ids[:50]:
        state.price_drops[pid] = 0.1

    cache = SimpleNamespace(bandit_state=state)
    scores = {pid: float(rng.uniform(0.01, 2.0)) for pid in ids}
    timings = []
    for _ in range(300):
        t0 = time.perf_counter()
        out = apply_online_adjustments(scores, cache)
        timings.append((time.perf_counter() - t0) * 1000.0)
        assert out != scores                            # genuinely rescored
    timings = np.array(timings)
    p50, p95, p99 = np.percentile(timings, [50, 95, 99])
    print(f"\nbandit adjust ms: mean={timings.mean():.3f} p50={p50:.3f} "
          f"p95={p95:.3f} p99={p99:.3f} max={timings.max():.3f}")
    assert timings.max() < 50.0, \
        f"bandit path exceeded 50 ms budget: {timings.max():.2f}"


# ----------------------------------------------------------- SQL builders --- #
def test_sql_builders() -> None:
    recent = feedback_recent_sql("team_12")
    assert "team_12.feedback" in recent and "__init__" in recent
    assert "ORDER BY ts DESC" in recent and "LIMIT 5000" in recent
    since = feedback_since_sql("team_12", "2026-11-14 10:00:00.000")
    assert "toDateTime64('2026-11-14 10:00:00.000', 3)" in since
    assert "ts::DateTime64(3) >" in since
    assert "argMax(price_eur" in PRICE_CHANGES_SQL
    assert "change::String = 'price_change'" in PRICE_CHANGES_SQL