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
  falsifiers: Optional[List[str]] = None,
  variant_perception: Optional[str] = None,
) -> int:
  """Insert a new thesis row. Returns thesis_id.

  `falsifiers` is the Soros-discipline required field — what specific
  observable would prove this thesis wrong? Cannot be empty for analyst
  workflow integrity.

  `variant_perception` is the differentiated view vs consensus.
  """
  conn = get_connection()
  try:
    cur = conn.execute("""
      INSERT INTO theses
        (ticker, thesis_date, recommendation, signal, target_price, stop_loss,
         confidence, analysis_summary, key_assumptions, data_gaps,
         full_report_md, arbiter_verdict_id, superseded_by,
         falsifiers, variant_perception)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)
    """, (ticker.upper(), datetime.now().isoformat(),
          recommendation, signal, target_price, stop_loss, confidence,
          analysis_summary, json.dumps(key_assumptions or []),
          json.dumps(data_gaps or []), full_report_md, arbiter_verdict_id,
          json.dumps(falsifiers or []), variant_perception))
    conn.commit()
    return cur.lastrowid
  finally:
    conn.close()


def record_thesis_evolution(
  thesis_id: int,
  observation: str,
  conviction_delta: float,
  tag: Optional[str] = None,
) -> int:
  """Record a check-in against an existing thesis. Returns evolution_id.

  Use after new data arrives (earnings print, news, macro shift) to log
  how the analyst's conviction shifted and why. The cumulative evolution
  log is the Soros-style reflexivity trace.

  `observation`: free text — what happened, what the data showed
  `conviction_delta`: positive = conviction up, negative = down (in 0.0-1.0 units)
  `tag`: optional category — 'earnings', 'macro', 'insider', 'sector', etc.
  """
  conn = get_connection()
  try:
    # Resolve current conviction
    row = conn.execute("SELECT ticker, confidence FROM theses WHERE thesis_id = ?",
                       (thesis_id,)).fetchone()
    if not row:
      raise ValueError(f"thesis {thesis_id} not found")
    ticker = row['ticker']
    prev_conviction = float(row['confidence'] or 0)
    # Compute new conviction (clamped 0-1)
    new_conviction = max(0.0, min(1.0, prev_conviction + conviction_delta))

    cur = conn.execute("""
      INSERT INTO thesis_evolution
        (thesis_id, ticker, timestamp, observation, conviction_delta,
         new_conviction, tag)
      VALUES (?,?,?,?,?,?,?)
    """, (thesis_id, ticker, datetime.now().isoformat(),
          observation, conviction_delta, new_conviction, tag))
    # Update the thesis's current confidence
    conn.execute("UPDATE theses SET confidence = ? WHERE thesis_id = ?",
                 (new_conviction, thesis_id))
    conn.commit()
    return cur.lastrowid
  finally:
    conn.close()


def get_thesis_evolution(thesis_id: int) -> List[Dict[str, Any]]:
  """Full chronological evolution log for a thesis — the reflexivity trace."""
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT * FROM thesis_evolution
      WHERE thesis_id = ?
      ORDER BY timestamp ASC
    """, (thesis_id,)).fetchall()
    return [dict(r) for r in rows]
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
    if d.get('falsifiers'):
      try:
        d['falsifiers'] = json.loads(d['falsifiers'])
      except (TypeError, ValueError):
        d['falsifiers'] = []
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
    if d.get('falsifiers'):
      try:
        d['falsifiers'] = json.loads(d['falsifiers'])
      except (TypeError, ValueError):
        d['falsifiers'] = []
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
      if d.get('falsifiers'):
        try:
          d['falsifiers'] = json.loads(d['falsifiers'])
        except (TypeError, ValueError):
          d['falsifiers'] = []
      out.append(d)
    return out
  finally:
    conn.close()
