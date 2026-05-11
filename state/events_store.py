"""CRUD for the events table.

Dedup is via event_id = sha256(source + headline + published_at)[:16].
Same article from 3 sources -> 3 rows (different source); same article from
same source pushed twice -> 1 row (INSERT OR IGNORE).
"""
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

from state.schema import get_connection


def event_id(source: str, headline: str, published_at: str) -> str:
  raw = f"{source}|{headline}|{published_at}".encode()
  return hashlib.sha256(raw).hexdigest()[:16]


def store_event(
  source: str,
  ticker: str,
  headline: str,
  body: str,
  url: str,
  published_at,
  materiality: str,
  category: str,
  affected_tickers: list,
  primary_ticker: Optional[str] = None,
  directional_signal: str = 'neutral',
  urgency: str = 'days',
  classifier_reason: str = '',
) -> str:
  """Insert an event, dedup by content hash. Returns event_id."""
  eid = event_id(source, headline, str(published_at))
  conn = get_connection()
  try:
    conn.execute("""INSERT OR IGNORE INTO events
      (event_id, source, ticker, headline, body, url, published_at, ingested_at,
       materiality, category, affected_tickers, primary_ticker,
       directional_signal, urgency, classifier_reason, processed)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
      (eid, source, ticker, headline, body[:5000] if body else '', url,
       str(published_at), datetime.now().isoformat(),
       materiality, category, json.dumps(affected_tickers), primary_ticker,
       directional_signal, urgency, classifier_reason))
    conn.commit()
  finally:
    conn.close()
  return eid


def seen(source: str, headline: str, published_at: str) -> bool:
  """Cheap dedup check before running an expensive classifier call."""
  eid = event_id(source, headline, str(published_at))
  conn = get_connection()
  try:
    row = conn.execute(
      "SELECT 1 FROM events WHERE event_id = ?", (eid,)
    ).fetchone()
    return row is not None
  finally:
    conn.close()


_MATERIALITY_RANK = {'high': 3, 'medium': 2, 'low': 1, 'noise': 0}


def unprocessed_events(min_materiality: str = 'medium', limit: int = 100) -> List[Dict[str, Any]]:
  """Return events ranked at or above the materiality floor."""
  threshold = _MATERIALITY_RANK.get(min_materiality, 2)
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT * FROM events WHERE processed = 0
      ORDER BY ingested_at DESC LIMIT ?
    """, (limit,)).fetchall()
    out = []
    for r in rows:
      mat = r['materiality'] or 'noise'
      if _MATERIALITY_RANK.get(mat, 0) >= threshold:
        out.append(dict(r))
    return out
  finally:
    conn.close()


def mark_processed(event_ids: List[str]) -> None:
  if not event_ids:
    return
  conn = get_connection()
  try:
    placeholders = ','.join('?' * len(event_ids))
    conn.execute(
      f"UPDATE events SET processed = 1 WHERE event_id IN ({placeholders})",
      event_ids
    )
    conn.commit()
  finally:
    conn.close()


def recent_events_for_ticker(ticker: str, hours: int = 24) -> List[Dict[str, Any]]:
  """Events touching a specific ticker (in affected_tickers or as primary) within N hours."""
  from datetime import timedelta
  cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT * FROM events
      WHERE (primary_ticker = ? OR ticker = ? OR affected_tickers LIKE ?)
        AND ingested_at >= ?
      ORDER BY ingested_at DESC
    """, (ticker, ticker, f'%"{ticker}"%', cutoff)).fetchall()
    return [dict(r) for r in rows]
  finally:
    conn.close()
