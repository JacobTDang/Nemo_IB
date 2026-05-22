"""Candidate validation logic for the nemo_sentry MCP server.

The triage daemon's job is fast classification + scoring. It cannot do
deep semantic checks. So before /sentry-tick spends the deep-research
budget on a candidate, the skill calls `validate_candidate(queue_id)`
to catch structural defects:

  (a) primary_ticker IS NULL and queue.ticker NOT in affected_tickers
      -- the event's tickers don't corroborate the queued name (the
      AAPL-in-ECB-payments-story case from 2026-05-22).
  (b) event is older than STALE_AGE_DAYS -- candidate is stale.
  (c) ticker has a recent eval_log row with decision = 'skipped_misclassified'
      -- we already rejected this exact (ticker, event_type) before;
      don't re-spend the budget.

This module imports nothing MCP-specific so it can be exercised directly
in unit tests without spinning up a server.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

STALE_AGE_DAYS = 7


def _parse_affected_tickers(raw: Any) -> list:
  if not raw:
    return []
  if isinstance(raw, list):
    return raw
  if isinstance(raw, str):
    try:
      return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
      return []
  return []


def _is_stale(published_at: Any, ingested_at: Any,
              now_utc: Optional[datetime] = None) -> bool:
  """An event is stale when both its publish AND ingest timestamps are
  older than STALE_AGE_DAYS. (We use ingest as a fallback since some
  feeds carry malformed publish dates.)"""
  now = now_utc or datetime.now(timezone.utc)
  cutoff = now - timedelta(days=STALE_AGE_DAYS)
  for raw in (published_at, ingested_at):
    if not raw:
      continue
    try:
      ts = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
      if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
      if ts >= cutoff:
        return False
    except (ValueError, TypeError):
      continue
  # If we could not parse any timestamp, don't claim stale -- the row
  # is suspicious for other reasons.
  return True


def _recent_misclassified(conn: sqlite3.Connection, ticker: str,
                          event_id: Optional[str]) -> bool:
  """Has this (ticker, event_id) been rejected as misclassified before?

  Looks at the last 30 days of eval_log rows for ticker with
  decision='skipped_misclassified' AND matching trigger_event_id."""
  if not event_id:
    return False
  cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
  try:
    row = conn.execute(
      """SELECT 1 FROM sentry_evaluation_log
         WHERE ticker = ? AND decision = 'skipped_misclassified'
           AND trigger_event_id = ?
           AND evaluated_at >= ?
         LIMIT 1""",
      (ticker.upper(), event_id, cutoff),
    ).fetchone()
    return row is not None
  except sqlite3.OperationalError:
    return False


def validate_candidate(conn: sqlite3.Connection, queue_id: int,
                       now_utc: Optional[datetime] = None) -> Dict[str, Any]:
  """Structural pre-flight check before /sentry-tick spends budget.

  Returns a dict shaped:
    {ok: bool, queue_id, ticker, event_id, reason, suggested_action}

  suggested_action is one of:
    'proceed'                  - ok=True
    'drop_misclassified'       - ticker doesn't appear in affected_tickers
    'drop_stale'               - event is older than STALE_AGE_DAYS
    'drop_repeat_misclassify'  - already rejected this exact case before
    'drop_missing_event'       - queue row references a non-existent event
    'drop_no_event_link'       - queue row has no source_event_id
  """
  q = conn.execute(
    """SELECT queue_id, ticker, source_event_id, score, status
       FROM sentry_queue WHERE queue_id = ?""",
    (queue_id,),
  ).fetchone()
  if q is None:
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': None,
      'event_id': None, 'reason': f'queue_id {queue_id} not found',
      'suggested_action': 'drop_missing_event',
    }

  ticker = (q['ticker'] or '').upper()
  event_id = q['source_event_id']

  if not event_id:
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': ticker,
      'event_id': None,
      'reason': 'queue row has no source_event_id; cannot validate',
      'suggested_action': 'drop_no_event_link',
    }

  ev = conn.execute(
    """SELECT event_id, ticker AS raw_ticker, primary_ticker, affected_tickers,
              published_at, ingested_at, headline, category
       FROM events WHERE event_id = ?""",
    (event_id,),
  ).fetchone()
  if ev is None:
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': ticker,
      'event_id': event_id,
      'reason': f'source event {event_id} not found in events table',
      'suggested_action': 'drop_missing_event',
    }

  affected = _parse_affected_tickers(ev['affected_tickers'])
  affected_upper = {t.upper() for t in affected if t}
  primary = (ev['primary_ticker'] or '').upper() or None

  # Misclassification check: when the classifier had no clear primary,
  # the raw event.ticker is just a news-provider tag. Queue.ticker must
  # appear in affected_tickers to be considered corroborated.
  if not primary and ticker not in affected_upper:
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': ticker,
      'event_id': event_id,
      'reason': (
        f"ticker '{ticker}' not in event.affected_tickers={list(affected_upper)} "
        f"and event has no primary_ticker; this is a misclassified candidate"
      ),
      'suggested_action': 'drop_misclassified',
    }

  # If primary IS set, sanity check ticker == primary OR ticker in affected
  if primary and ticker != primary and ticker not in affected_upper:
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': ticker,
      'event_id': event_id,
      'reason': (
        f"ticker '{ticker}' does not match event.primary_ticker='{primary}' "
        f"and is not in affected_tickers={list(affected_upper)}"
      ),
      'suggested_action': 'drop_misclassified',
    }

  # Stale-event check
  if _is_stale(ev['published_at'], ev['ingested_at'], now_utc=now_utc):
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': ticker,
      'event_id': event_id,
      'reason': (
        f"event {event_id} is older than {STALE_AGE_DAYS} days "
        f"(published={ev['published_at']}, ingested={ev['ingested_at']})"
      ),
      'suggested_action': 'drop_stale',
    }

  # Repeat-misclassify check (cheap pre-filter for the same event_id)
  if _recent_misclassified(conn, ticker, event_id):
    return {
      'ok': False, 'queue_id': queue_id, 'ticker': ticker,
      'event_id': event_id,
      'reason': (
        f"event {event_id} was previously rejected as misclassified "
        f"for ticker {ticker}; don't re-spend budget"
      ),
      'suggested_action': 'drop_repeat_misclassify',
    }

  return {
    'ok': True, 'queue_id': queue_id, 'ticker': ticker,
    'event_id': event_id, 'reason': None,
    'suggested_action': 'proceed',
  }
