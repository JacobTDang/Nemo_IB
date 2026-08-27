"""Two more sweeps: what a rerun does, and what comes back in the dict.

**Rerunning changes nothing.** Every recorder here is called on a schedule and
will be called twice -- a cron that overlaps, an operator filling a gap, a
container restarted mid-run. A recorder that appends on the second pass doubles
a book or a bar series, and nothing about the result says it happened.

**The dict has the keys it promises.** These functions return a dict rather
than an object, so a renamed key is not a type error anywhere -- it is a
`.get()` returning None three modules away, which reads as missing data. The
scanner has already been bitten once by exactly this, reading `data["earnings"]`
from a payload whose key was `events` and recording nothing for its whole life
while running green.
"""
import pytest

from research import (daily_job, pit_store, scanner, scoring, seed_consensus,
                      spread, sue, sue_cs)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


DAY = "2026-03-02"
STAMP = f"{DAY}T21:00:00Z"


def _bar(day=DAY, close=100.0):
    return {"trade_date": day, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1_000_000}


def _counts(store):
    with pit_store.connect() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables}


# --- rerunning changes nothing ----------------------------------------------

RECORDERS = [
    ("record_bars",
     lambda: pit_store.record_bars("AAA", [_bar()], recorded_at=STAMP)),
    ("record_corporate_action",
     lambda: pit_store.record_corporate_action("AAA", DAY, "split", 2.0,
                                               recorded_at=STAMP)),
    ("record_universe",
     lambda: pit_store.record_universe(
         DAY, [{"ticker": "AAA", "cik": "1", "eligible": True}],
         recorded_at=STAMP)),
    ("record_announcement",
     lambda: pit_store.record_announcement("AAA", "2026Q1", DAY,
                                           recorded_at=STAMP)),
    ("record_consensus",
     lambda: pit_store.record_consensus(DAY, "AAA", "2026Q1",
                                        eps_estimate=1.0, eps_actual=1.2,
                                        recorded_at=STAMP)),
    ("record_paper_orders",
     lambda: pit_store.record_paper_orders(
         DAY, [{"ticker": "AAA", "fiscal_period": "2026Q1", "side": "long",
                "sue": 2.0, "target_dollars": 100.0,
                "intended_session": "2026-03-03"}], recorded_at=STAMP)),
    ("record_activist_filings",
     lambda: pit_store.record_activist_filings([{
         "accession": "0001-25-000001", "subject_ticker": "AAA",
         "filer": "Someone LP", "form": "SC 13D", "filing_date": DAY,
         "detected_at": STAMP}], recorded_at=STAMP)),
]


@pytest.mark.parametrize("name,call", RECORDERS)
def test_a_recorder_run_twice_writes_nothing_the_second_time(store, name, call):
    call()
    after_first = _counts(store)
    call()
    assert _counts(store) == after_first, f"{name} appended on a rerun"


@pytest.mark.parametrize("name,call", RECORDERS)
def test_a_recorder_reports_what_it_actually_wrote(store, name, call):
    """The count is used to decide whether a run did anything, so it has to
    fall to zero rather than repeat the first pass's number."""
    first = call()
    second = call()
    if first is None:
        pytest.skip(f"{name} returns nothing to check")
    assert first > 0 and second == 0, (
        f"{name} reported {second} written on a rerun that wrote nothing")


def test_a_rerun_of_the_whole_day_leaves_the_store_identical(store,
                                                             monkeypatch):
    import pandas as pd
    import yfinance

    def frame(*a, **k):
        rows = k["tickers"].split()
        data = {}
        for t in rows:
            for f in ("Open", "High", "Low", "Close"):
                data[(t, f)] = [100.0]
            data[(t, "Volume")] = [1e6]
            data[(t, "Stock Splits")] = [0.0]
            data[(t, "Dividends")] = [0.0]
        f_ = pd.DataFrame(data, index=pd.to_datetime([DAY]))
        f_.columns = pd.MultiIndex.from_tuples(f_.columns)
        return f_

    monkeypatch.setattr(yfinance, "download", frame)
    monkeypatch.setattr(daily_job, "_today", lambda: DAY)
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": "AAA", "cik": "1", "name": "A"}])
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})

    daily_job.run_all(as_of=DAY)
    before = _counts(store)
    before.pop("run_log")          # a second run is a second run; that is the point
    daily_job.run_all(as_of=DAY)
    after = _counts(store)
    after.pop("run_log")
    assert after == before
    assert pit_store.revisions("AAA") == []


def test_seeding_twice_writes_nothing_the_second_time(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises", lambda t: [
        {"year": 2026, "quarter": 1, "estimate_eps": 1.0, "actual_eps": 1.2}])
    monkeypatch.setattr(seed_consensus, "_filing_dates", lambda t, as_of=None: {
        "2026Q1": {"period_end": "2026-03-31", "known_at": "2026-04-30"}})

    assert seed_consensus.seed(["AAA"])["written"] == 2
    assert seed_consensus.seed(["AAA"])["written"] == 0


# --- the dict has the keys it promises --------------------------------------

SIGNAL_KEYS = {"ticker", "success", "error", "as_of", "fiscal_period", "sue"}


def test_a_time_series_surprise_has_the_signal_keys(store, monkeypatch):
    monkeypatch.setattr(sue, "_fetch_cik_map", lambda: {})
    assert SIGNAL_KEYS <= set(sue.sue_ts("NOPE", as_of=DAY))


def test_an_analyst_surprise_has_the_signal_keys_and_its_own(store, monkeypatch):
    monkeypatch.setattr(sue, "_fetch_cik_map", lambda: {})
    out = sue.sue_af("NOPE", as_of=DAY)
    assert SIGNAL_KEYS <= set(out)
    assert {"surprise", "consensus", "seeded_quarters",
            "recorded_quarters"} <= set(out)


def test_a_cross_sectional_rank_has_its_keys(store):
    out = sue_cs.surprise_rank("NOPE", as_of=DAY)
    assert {"ticker", "success", "error", "z", "robust_z", "percentile",
            "rank_on", "cohort_size", "cohort_tickers", "surprise",
            "scaled_surprise"} <= set(out)


def test_a_spread_estimate_has_its_keys(store):
    out = spread.estimate_spread("NOPE", DAY, window=60)
    assert {"ticker", "as_of", "window", "spread", "spread_upper",
            "standard_error", "resolved", "tick_floor", "reason"} <= set(out)


def test_a_cost_has_its_keys(store):
    out = spread.round_trip_cost("NOPE", DAY, 1000.0, window=60)
    assert {"cost", "spread_cost", "impact_cost", "spread_basis",
            "resolution", "position_dollars", "reason"} <= set(out)


def test_a_scan_has_its_keys(store):
    out = scanner.scan(as_of=DAY)
    assert {"as_of", "regime", "regime_scale", "gross_target", "screened",
            "considered", "narrowed_by", "narrowing_note", "candidates",
            "rejected", "undetermined", "costs_total", "costs_measured",
            "costs_floored", "assumptions"} <= set(out)


def test_a_score_has_its_keys(store):
    out = scoring.score_orders(as_of=DAY)
    assert {"as_of", "horizon_days", "scored", "pending", "unfilled",
            "sample", "hit_rate", "mean_net_bps", "median_net_bps",
            "t_stat", "drift_bps_per_sue", "calibrated",
            "calibration_note"} <= set(out)


def test_a_candidate_carries_everything_needed_to_score_it_later(store,
                                                                 monkeypatch):
    """A row that reaches paper_order without one of these cannot be scored,
    and the gap only shows up a month later when the horizon closes."""
    from datetime import date, timedelta

    days = []
    d = date(2026, 1, 1)
    while len(days) < 300:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    for t in ("AAA", spread.REFERENCE_TICKER):
        pit_store.record_bars(t, [_bar(x) for x in days],
                              recorded_at=f"{days[-1]}T21:00:00Z")
    pit_store.record_universe(days[-1],
                              [{"ticker": "AAA", "cik": "1", "eligible": True}],
                              recorded_at=f"{days[-1]}T21:00:00Z")
    pit_store.record_consensus(days[-1], "AAA", "2026Q1", eps_estimate=1.0,
                               eps_actual=1.2,
                               recorded_at=f"{days[-1]}T21:00:00Z")
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: {
        "ticker": t, "success": True, "error": None, "sue": 3.0,
        "fiscal_period": "2026Q1", "known_at": a, "sigma_quarters": 8,
        "sigma_periods": ["2026Q1"], "basis_changes": [], "variant": "ts"})
    monkeypatch.setattr(scanner, "_cost_for", lambda t, a, d_: {
        "cost": 0.0002, "cost_floor": 0.00002, "reason": None,
        "spread": 0.0001, "resolved": True, "resolution": "measured"})

    out = scanner.scan(as_of=days[-1])
    assert out["candidates"], out["rejected"]
    c = out["candidates"][0]
    for key in ("ticker", "side", "sue", "fiscal_period", "cost_bps",
                "target_dollars", "intended_session", "expected_edge_bps",
                "net_edge_bps", "rank", "variant"):
        assert key in c, f"a candidate reached the book without {key}"
