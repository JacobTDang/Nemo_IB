"""Tests for pre-earnings schema tables and state/preearnings.py helpers.

Layer 3 tests (schema + state module) — no MCP, no network, no runners.
Uses the same in-memory DB pattern as other schema tests.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.preearnings import (
    record_signal,
    signals_for_ticker,
    record_supplier_readthrough,
    recent_supplier_readthroughs,
    record_eval,
    get_eval,
    eval_accuracy_summary,
)


# Use a temp DB for all tests so they don't pollute the live session.db
_TMP_DB = os.path.join(tempfile.gettempdir(), "nemo_preearnings_test.db")


def _conn():
    return get_connection(_TMP_DB)


def setup_module(_):
    init_schema(_TMP_DB)


def teardown_module(_):
    if os.path.exists(_TMP_DB):
        os.remove(_TMP_DB)


def _clean():
    conn = _conn()
    try:
        conn.execute("DELETE FROM preearnings_signals WHERE ticker LIKE 'ZZ%'")
        conn.execute("DELETE FROM supplier_readthroughs WHERE downstream_ticker LIKE 'ZZ%'")
        conn.execute("DELETE FROM preearnings_evals WHERE ticker LIKE 'ZZ%'")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

def test_tables_created():
    conn = _conn()
    try:
        tables = {r[0] for r in
                  conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    finally:
        conn.close()
    assert "preearnings_signals" in tables, "preearnings_signals table missing"
    assert "supplier_readthroughs" in tables, "supplier_readthroughs table missing"
    assert "pricing_snapshots" in tables, "pricing_snapshots table missing"
    assert "preearnings_evals" in tables, "preearnings_evals table missing"


def test_preearnings_signals_columns():
    conn = _conn()
    try:
        cols = {r["name"] for r in
                conn.execute("PRAGMA table_info(preearnings_signals)").fetchall()}
    finally:
        conn.close()
    required = {"id", "ticker", "earnings_date", "signal_category", "signal_name",
                "direction", "magnitude", "raw_value", "source_url",
                "collected_at", "days_before_earnings"}
    assert required <= cols, f"missing columns: {required - cols}"


def test_preearnings_evals_unique_index():
    conn = _conn()
    try:
        indexes = {r[1] for r in
                   conn.execute("PRAGMA index_list(preearnings_evals)").fetchall()}
    finally:
        conn.close()
    assert "idx_preearnings_evals_ticker_date" in indexes


# ---------------------------------------------------------------------------
# record_signal / signals_for_ticker
# ---------------------------------------------------------------------------

def test_record_signal_roundtrip():
    row_id = record_signal(
        ticker="ZZTST",
        earnings_date="2026-08-20",
        signal_category="demand",
        signal_name="google_trends",
        direction="bullish",
        magnitude=0.72,
        raw_value='{"yoy_ratio": 1.14}',
        source_url="",
        days_before_earnings=-14,
        _db=_TMP_DB,
    )
    assert row_id > 0
    rows = signals_for_ticker("ZZTST", "2026-08-20", _db=_TMP_DB)
    assert len(rows) >= 1
    row = next(r for r in rows if r["signal_name"] == "google_trends")
    assert row["direction"] == "bullish"
    assert abs(row["magnitude"] - 0.72) < 0.001
    assert row["days_before_earnings"] == -14
    _clean()


def test_signals_ordered_by_collected_at():
    for name in ["google_trends", "finbert_sentiment", "job_postings"]:
        record_signal("ZZORD", "2026-09-01", "demand", name, "neutral", _db=_TMP_DB)
    rows = signals_for_ticker("ZZORD", "2026-09-01", _db=_TMP_DB)
    names = [r["signal_name"] for r in rows]
    assert names == ["google_trends", "finbert_sentiment", "job_postings"]
    _clean()


def test_signals_different_earnings_dates_isolated():
    record_signal("ZZISO", "2026-06-01", "demand", "google_trends", "bullish", _db=_TMP_DB)
    record_signal("ZZISO", "2026-09-01", "demand", "google_trends", "bearish", _db=_TMP_DB)
    q1 = signals_for_ticker("ZZISO", "2026-06-01", _db=_TMP_DB)
    q3 = signals_for_ticker("ZZISO", "2026-09-01", _db=_TMP_DB)
    assert all(r["direction"] == "bullish" for r in q1)
    assert all(r["direction"] == "bearish" for r in q3)
    _clean()


# ---------------------------------------------------------------------------
# record_supplier_readthrough / recent_supplier_readthroughs
# ---------------------------------------------------------------------------

def test_record_supplier_readthrough():
    row_id = record_supplier_readthrough(
        supplier_ticker="TSM",
        downstream_ticker="ZZAPL",
        supplier_report_date="2026-04-17",
        direction="bullish",
        key_findings="TSMC Q1 revenue +34% YoY; Apple accounts for ~25% of revenue",
        confidence=0.78,
        _db=_TMP_DB,
    )
    assert row_id > 0
    rows = recent_supplier_readthroughs("ZZAPL", days=90, _db=_TMP_DB)
    assert len(rows) >= 1
    assert rows[0]["supplier_ticker"] == "TSM"
    assert rows[0]["direction"] == "bullish"
    _clean()


def test_recent_supplier_readthroughs_cutoff():
    from datetime import datetime, timezone, timedelta
    from state.schema import get_connection
    # Insert an old readthrough (95 days ago) manually
    old_ts = (datetime.now(timezone.utc) - timedelta(days=95)).isoformat()
    conn = get_connection(_TMP_DB)
    try:
        conn.execute(
            """INSERT INTO supplier_readthroughs
               (supplier_ticker, downstream_ticker, supplier_report_date,
                direction, triggered_at)
               VALUES ('TSM', 'ZZCUTOFF', '2026-01-10', 'bearish', ?)""",
            (old_ts,),
        )
        conn.commit()
    finally:
        conn.close()
    rows = recent_supplier_readthroughs("ZZCUTOFF", days=90, _db=_TMP_DB)
    assert len(rows) == 0, "old readthrough should not appear in 90-day window"
    _clean()


# ---------------------------------------------------------------------------
# record_eval / get_eval / eval_accuracy_summary
# ---------------------------------------------------------------------------

def test_record_eval_insert():
    row_id = record_eval(
        ticker="ZZEVAL",
        earnings_date="2026-07-30",
        prediction="likely_beat",
        confidence=0.72,
        implied_move_pct=0.082,
        _db=_TMP_DB,
    )
    assert row_id > 0
    row = get_eval("ZZEVAL", "2026-07-30", _db=_TMP_DB)
    assert row is not None
    assert row["prediction"] == "likely_beat"
    assert abs(row["confidence"] - 0.72) < 0.001
    _clean()


def test_record_eval_upsert_outcome():
    record_eval("ZZUPS", "2026-07-30", "likely_beat", 0.65, _db=_TMP_DB)
    record_eval(
        "ZZUPS", "2026-07-30", "likely_beat", 0.65,
        actual_eps_surprise=0.05,
        outcome="beat",
        prediction_correct=1,
        _db=_TMP_DB,
    )
    row = get_eval("ZZUPS", "2026-07-30", _db=_TMP_DB)
    assert row["prediction_correct"] == 1
    assert row["outcome"] == "beat"
    # Should still be one row, not two
    conn = _conn()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM preearnings_evals WHERE ticker='ZZUPS'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1
    _clean()


def test_eval_accuracy_summary():
    record_eval("ZZACC1", "2026-07-01", "likely_beat", 0.80,
                outcome="beat", prediction_correct=1, _db=_TMP_DB)
    record_eval("ZZACC2", "2026-07-01", "likely_miss", 0.60,
                outcome="beat", prediction_correct=0, _db=_TMP_DB)
    record_eval("ZZACC3", "2026-07-01", "in_line", 0.55,
                outcome="in_line", prediction_correct=1, _db=_TMP_DB)
    summary = eval_accuracy_summary(_db=_TMP_DB)
    assert summary["total"] >= 3
    assert summary["correct"] >= 2
    assert 0 < summary["accuracy_pct"] <= 100
    _clean()
