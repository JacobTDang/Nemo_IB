"""CRUD for the theses table.

A "thesis" is the persisted output of one full analysis run on a ticker:
recommendation, target/stop, confidence, key assumptions, and the full report
markdown. When the agent re-analyzes a ticker, the new thesis supersedes the
old (old.superseded_by = new.thesis_id), keeping history intact.
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from state.schema import get_connection


def insert_thesis(
  ticker: str,
  recommendation: str,
  signal: str,
  target_price: Optional[float],
  stop_loss: Optional[float],
  confidence: float,
  analysis_summary: str,
  key_assumptions: List[str],
  data_gaps: List[str],
  full_report_md: str,
  arbiter_verdict_id: Optional[int] = None,
) -> int:
  """Insert a new thesis row. Returns thesis_id."""
  conn = get_connection()
  try:
    cur = conn.execute("""
      INSERT INTO theses
        (ticker, thesis_date, recommendation, signal, target_price, stop_loss,
         confidence, analysis_summary, key_assumptions, data_gaps,
         full_report_md, arbiter_verdict_id, superseded_by)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)
    """, (ticker.upper(), datetime.now().isoformat(),
          recommendation, signal, target_price, stop_loss, confidence,
          analysis_summary, json.dumps(key_assumptions or []),
          json.dumps(data_gaps or []), full_report_md, arbiter_verdict_id))
    conn.commit()
    return cur.lastrowid
  finally:
    conn.close()


def latest_thesis(ticker: str) -> Optional[Dict[str, Any]]:
  """Active (non-superseded) thesis for a ticker, or None."""
  conn = get_connection()
  try:
    row = conn.execute("""
      SELECT * FROM theses
      WHERE ticker = ? AND superseded_by IS NULL
      ORDER BY thesis_date DESC LIMIT 1
    """, (ticker.upper(),)).fetchone()
    if not row:
      return None
    d = dict(row)
    d['key_assumptions'] = json.loads(d.get('key_assumptions') or '[]')
    d['data_gaps'] = json.loads(d.get('data_gaps') or '[]')
    return d
  finally:
    conn.close()


def supersede_thesis(old_id: int, new_id: int) -> None:
  """Mark old_id as superseded by new_id."""
  conn = get_connection()
  try:
    conn.execute("UPDATE theses SET superseded_by = ? WHERE thesis_id = ?",
                 (new_id, old_id))
    conn.commit()
  finally:
    conn.close()


def thesis_history(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
  """Full history of theses for a ticker, newest first."""
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT thesis_id, thesis_date, recommendation, signal,
             confidence, target_price, stop_loss, analysis_summary,
             superseded_by
      FROM theses WHERE ticker = ?
      ORDER BY thesis_date DESC LIMIT ?
    """, (ticker.upper(), limit)).fetchall()
    return [dict(r) for r in rows]
  finally:
    conn.close()


def get_thesis(thesis_id: int) -> Optional[Dict[str, Any]]:
  conn = get_connection()
  try:
    row = conn.execute("SELECT * FROM theses WHERE thesis_id = ?",
                       (thesis_id,)).fetchone()
    if not row:
      return None
    d = dict(row)
    d['key_assumptions'] = json.loads(d.get('key_assumptions') or '[]')
    d['data_gaps'] = json.loads(d.get('data_gaps') or '[]')
    return d
  finally:
    conn.close()


def active_theses(limit: int = 100) -> List[Dict[str, Any]]:
  """All currently-active theses across the watchlist."""
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT * FROM theses WHERE superseded_by IS NULL
      ORDER BY thesis_date DESC LIMIT ?
    """, (limit,)).fetchall()
    out = []
    for r in rows:
      d = dict(r)
      d['key_assumptions'] = json.loads(d.get('key_assumptions') or '[]')
      d['data_gaps'] = json.loads(d.get('data_gaps') or '[]')
      out.append(d)
    return out
  finally:
    conn.close()
