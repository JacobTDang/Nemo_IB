"""State helpers for the pre-earnings research pipeline.

Mirrors the pattern of state/sentry_queue.py: thin wrappers over raw SQLite
that handle connection lifecycle and return plain dicts for easy consumption
by MCP tools and skills.

All functions accept an optional `_db` keyword argument so tests can point
at a temp DB without touching the live session.db.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from state.schema import get_connection, DB_PATH


def _db_conn(db: Optional[str]):
    return get_connection(db) if db else get_connection()


def record_signal(
    ticker: str,
    earnings_date: str,
    signal_category: str,
    signal_name: str,
    direction: str,
    magnitude: Optional[float] = None,
    raw_value: Optional[str] = None,
    source_url: Optional[str] = None,
    days_before_earnings: Optional[int] = None,
    *,
    _db: Optional[str] = None,
) -> int:
    """Insert a pre-earnings signal row. Returns the new row id."""
    conn = _db_conn(_db)
    try:
        cur = conn.execute(
            """INSERT INTO preearnings_signals
               (ticker, earnings_date, signal_category, signal_name,
                direction, magnitude, raw_value, source_url,
                collected_at, days_before_earnings)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker.upper(),
                earnings_date,
                signal_category,
                signal_name,
                direction,
                magnitude,
                raw_value,
                source_url,
                datetime.now(timezone.utc).isoformat(),
                days_before_earnings,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def signals_for_ticker(
    ticker: str, earnings_date: str, *, _db: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return all signals for a ticker + earnings date in collection order."""
    conn = _db_conn(_db)
    try:
        rows = conn.execute(
            """SELECT * FROM preearnings_signals
               WHERE ticker = ? AND earnings_date = ?
               ORDER BY collected_at ASC""",
            (ticker.upper(), earnings_date),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def record_supplier_readthrough(
    supplier_ticker: str,
    downstream_ticker: str,
    supplier_report_date: str,
    direction: str,
    key_findings: Optional[str] = None,
    confidence: Optional[float] = None,
    *,
    _db: Optional[str] = None,
) -> int:
    """Record that a supplier's earnings triggered a readthrough for a downstream ticker."""
    conn = _db_conn(_db)
    try:
        cur = conn.execute(
            """INSERT INTO supplier_readthroughs
               (supplier_ticker, downstream_ticker, supplier_report_date,
                direction, key_findings, confidence, triggered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                supplier_ticker.upper(),
                downstream_ticker.upper(),
                supplier_report_date,
                direction,
                key_findings,
                confidence,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def recent_supplier_readthroughs(
    downstream_ticker: str, days: int = 90, *, _db: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return supplier readthroughs for a downstream ticker within the last N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = _db_conn(_db)
    try:
        rows = conn.execute(
            """SELECT * FROM supplier_readthroughs
               WHERE downstream_ticker = ? AND triggered_at >= ?
               ORDER BY triggered_at DESC""",
            (downstream_ticker.upper(), cutoff),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def record_eval(
    ticker: str,
    earnings_date: str,
    prediction: str,
    confidence: float,
    implied_move_pct: Optional[float] = None,
    actual_eps_surprise: Optional[float] = None,
    actual_rev_surprise: Optional[float] = None,
    actual_price_move_1d: Optional[float] = None,
    outcome: Optional[str] = None,
    prediction_correct: Optional[int] = None,
    notes: Optional[str] = None,
    *,
    _db: Optional[str] = None,
) -> int:
    """Upsert a pre-earnings eval row. Returns the row id."""
    conn = _db_conn(_db)
    try:
        cur = conn.execute(
            """INSERT INTO preearnings_evals
               (ticker, earnings_date, prediction, confidence,
                implied_move_pct, actual_eps_surprise, actual_rev_surprise,
                actual_price_move_1d, outcome, prediction_correct, notes,
                evaluated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                 actual_eps_surprise  = excluded.actual_eps_surprise,
                 actual_rev_surprise  = excluded.actual_rev_surprise,
                 actual_price_move_1d = excluded.actual_price_move_1d,
                 outcome              = excluded.outcome,
                 prediction_correct   = excluded.prediction_correct,
                 notes                = excluded.notes,
                 evaluated_at         = excluded.evaluated_at""",
            (
                ticker.upper(),
                earnings_date,
                prediction,
                confidence,
                implied_move_pct,
                actual_eps_surprise,
                actual_rev_surprise,
                actual_price_move_1d,
                outcome,
                prediction_correct,
                notes,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_eval(
    ticker: str, earnings_date: str, *, _db: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the eval row for a ticker + earnings date, or None."""
    conn = _db_conn(_db)
    try:
        row = conn.execute(
            "SELECT * FROM preearnings_evals WHERE ticker = ? AND earnings_date = ?",
            (ticker.upper(), earnings_date),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def eval_accuracy_summary(*, _db: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate accuracy stats across all completed evals."""
    conn = _db_conn(_db)
    try:
        rows = conn.execute(
            "SELECT * FROM preearnings_evals WHERE prediction_correct IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return {"total": 0, "message": "no completed evals yet"}
    rows = [dict(r) for r in rows]
    total = len(rows)
    correct = sum(1 for r in rows if r["prediction_correct"] == 1)
    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": round(correct / total * 100, 1),
        "avg_confidence_correct": _avg([r["confidence"] for r in rows if r["prediction_correct"] == 1]),
        "avg_confidence_wrong":   _avg([r["confidence"] for r in rows if r["prediction_correct"] == 0]),
    }


def _avg(values: list) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


# ---------------------------------------------------------------------------
# Deep research layers (Phase A) — sub-agent + structured component outputs.
# Upserted on the natural key (ticker, earnings_date, component) so repeated
# pre-earnings runs (7d/3d/1d out) reuse fresh research instead of re-paying.
# ---------------------------------------------------------------------------

import json as _json


def record_layer(
    ticker: str,
    earnings_date: str,
    layer: int,
    component: str,
    direction: Optional[str] = None,
    magnitude: Optional[float] = None,
    confidence: Optional[float] = None,
    payload: Optional[Any] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    *,
    _db: Optional[str] = None,
) -> int:
    """Upsert a research-layer component. `payload` and `sources` are JSON-encoded.
    `sources` is a list of {claim, tool} dicts for the citation audit."""
    conn = _db_conn(_db)
    try:
        cur = conn.execute(
            """INSERT INTO preearnings_research_layers
               (ticker, earnings_date, layer, component, direction, magnitude,
                confidence, payload_json, sources_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, earnings_date, component) DO UPDATE SET
                 layer        = excluded.layer,
                 direction    = excluded.direction,
                 magnitude    = excluded.magnitude,
                 confidence   = excluded.confidence,
                 payload_json = excluded.payload_json,
                 sources_json = excluded.sources_json,
                 created_at   = excluded.created_at""",
            (
                ticker.upper(),
                earnings_date,
                int(layer),
                component,
                direction,
                magnitude,
                confidence,
                _json.dumps(payload if payload is not None else {}, default=str),
                _json.dumps(sources or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _decode_layer(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    try:
        d["payload"] = _json.loads(d.get("payload_json") or "{}")
    except (ValueError, TypeError):
        d["payload"] = {}
    try:
        d["sources"] = _json.loads(d.get("sources_json") or "[]")
    except (ValueError, TypeError):
        d["sources"] = []
    return d


def get_layers(
    ticker: str, earnings_date: str, *, _db: Optional[str] = None
) -> List[Dict[str, Any]]:
    """All research-layer components for a ticker + earnings date (payload/sources decoded)."""
    conn = _db_conn(_db)
    try:
        rows = conn.execute(
            """SELECT * FROM preearnings_research_layers
               WHERE ticker = ? AND earnings_date = ?
               ORDER BY layer ASC, component ASC""",
            (ticker.upper(), earnings_date),
        ).fetchall()
        return [_decode_layer(r) for r in rows]
    finally:
        conn.close()


def latest_component(
    ticker: str, earnings_date: str, component: str, *, _db: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the stored component for a ticker + earnings date, or None."""
    conn = _db_conn(_db)
    try:
        row = conn.execute(
            """SELECT * FROM preearnings_research_layers
               WHERE ticker = ? AND earnings_date = ? AND component = ?""",
            (ticker.upper(), earnings_date, component),
        ).fetchone()
        return _decode_layer(row) if row else None
    finally:
        conn.close()


def is_fresh(
    ticker: str,
    earnings_date: str,
    component: str,
    max_age_hours: float,
    *,
    _db: Optional[str] = None,
) -> bool:
    """True if the component exists and was written within max_age_hours.
    Lets a re-run skip re-computing a component that is still current."""
    row = latest_component(ticker, earnings_date, component, _db=_db)
    if not row or not row.get("created_at"):
        return False
    try:
        created = datetime.fromisoformat(row["created_at"])
    except (ValueError, TypeError):
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
    return age_h <= max_age_hours
