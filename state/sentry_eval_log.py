"""Sentry's evaluation log — the system's memory of every ticker it considers.

Before queueing a candidate, the triage daemon checks this log to avoid
re-researching the same name during a cooldown window. The /sentry-tick skill
also consults `recent_evals_for_ticker` to remind Claude of prior verdicts.

Cooldown rules (computed at record time, stored in next_review_at):
  acted          → 7 days   (acted last week, monitor for falsifier triggers)
  buy            → 7 days   (researched and would buy; re-check thesis intact)
  watchlist      → 14 days  (interesting but not actionable; longer cooldown)
  avoid          → 30 days  (researched and skipped; don't churn back fast)
  no_position    → 30 days  (no clear edge; long cooldown)
  None (skipped) → no cooldown set (will be re-eligible immediately)

Falsifier triggers bypass the cooldown — if falsifier_watcher fires on an
acted thesis, Sentry should be allowed to re-evaluate immediately regardless
of the 7-day window.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from state.schema import get_connection


_COOLDOWN_DAYS = {
  'acted':       7,
  'buy':         7,
  'watchlist':   14,
  'avoid':       30,
  'no_position': 30,
}


def _compute_next_review(verdict: Optional[str], decision: str) -> Optional[str]:
  """Return ISO-format next_review_at timestamp, or None if no cooldown applies."""
  # Use verdict if researched; fall back to decision (e.g., 'acted')
  key = verdict or decision
  days = _COOLDOWN_DAYS.get(key)
  if days is None:
    return None
  return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def record_eval(
    ticker: str,
    decision: str,
    *,
    triggered_by: str,
    trigger_event_id: Optional[str] = None,
    verdict: Optional[str] = None,
    confidence: Optional[float] = None,
    sizing: Optional[str] = None,
    factor_buckets: Optional[List[str]] = None,
    skip_reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
  """Insert an eval row. Computes next_review_at from cooldown rules.

  Args:
    ticker: stock symbol (uppercased)
    decision: one of {researched, skipped_cooldown, skipped_recent_verdict,
              acted, skipped_budget}
    triggered_by: which discovery channel queued it (event_score, theme_flow,
                  insider_cluster, catalyst_calendar, manual)
    verdict: required if decision='researched' or 'acted' — one of
             {buy, watchlist, avoid, no_position}
    confidence: 0.0-1.0 from the synthesis (required when verdict is set)
    sizing: aggressive / normal / cautious / no_position
    factor_buckets: list of factor tags from factor-exposure-check
    skip_reason: free text if decision starts with 'skipped_'
    notes: any additional context worth recording

  Returns the eval_id of the new row.
  """
  next_review_at = _compute_next_review(verdict, decision)
  conn = get_connection()
  try:
    cur = conn.execute(
      """INSERT INTO sentry_evaluation_log
         (ticker, evaluated_at, triggered_by, trigger_event_id, decision,
          verdict, confidence, sizing, factor_buckets, skip_reason,
          next_review_at, notes)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        ticker.upper(),
        datetime.now(timezone.utc).isoformat(),
        triggered_by,
        trigger_event_id,
        decision,
        verdict,
        confidence,
        sizing,
        json.dumps(factor_buckets) if factor_buckets else None,
        skip_reason,
        next_review_at,
        notes,
      ),
    )
    conn.commit()
    return cur.lastrowid
  finally:
    conn.close()


def is_in_cooldown(ticker: str) -> Tuple[bool, Optional[str]]:
  """Check the most-recent eval row for `ticker`.

  Returns (in_cooldown, reason). If next_review_at is in the future, the
  ticker is in cooldown — the reason is a human-readable string. If the
  most recent eval has no next_review_at (e.g., decision='skipped_budget'),
  the ticker is NOT in cooldown.
  """
  conn = get_connection()
  try:
    cur = conn.execute(
      """SELECT verdict, decision, next_review_at, evaluated_at
         FROM sentry_evaluation_log
         WHERE ticker = ?
         ORDER BY evaluated_at DESC
         LIMIT 1""",
      (ticker.upper(),),
    )
    row = cur.fetchone()
  finally:
    conn.close()

  if row is None:
    return (False, None)

  next_review = row['next_review_at']
  if next_review is None:
    return (False, None)

  now_iso = datetime.now(timezone.utc).isoformat()
  if next_review > now_iso:
    reason = f"last eval ({row['evaluated_at'][:10]}) verdict={row['verdict']}, next_review_at={next_review[:10]}"
    return (True, reason)

  return (False, None)


def recent_evals_for_ticker(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
  """Return all eval rows for `ticker` within the last N days, newest first.

  Used by /sentry-tick to remind Claude of prior verdicts before
  initiating new research — informs whether to trust this round or fade it.
  """
  cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
  conn = get_connection()
  try:
    cur = conn.execute(
      """SELECT eval_id, ticker, evaluated_at, triggered_by, decision,
                verdict, confidence, sizing, factor_buckets, skip_reason,
                next_review_at, notes
         FROM sentry_evaluation_log
         WHERE ticker = ? AND evaluated_at >= ?
         ORDER BY evaluated_at DESC""",
      (ticker.upper(), cutoff),
    )
    rows = [dict(r) for r in cur.fetchall()]
  finally:
    conn.close()
  # Parse factor_buckets JSON for caller convenience
  for r in rows:
    if r.get('factor_buckets'):
      try:
        r['factor_buckets'] = json.loads(r['factor_buckets'])
      except (json.JSONDecodeError, TypeError):
        pass
  return rows


def should_skip(
    ticker: str,
    event: Optional[Dict[str, Any]] = None,
    *,
    bypass_for_falsifier: bool = False,
) -> Tuple[bool, Optional[str]]:
  """Triage's gate before queueing a candidate.

  Returns (skip, skip_reason). Combines:
    - cooldown check (most recent eval)
    - falsifier bypass (if a falsifier_alerts row exists for this ticker in
      the last 24h, the cooldown is overridden)

  Future extensions:
    - budget check (sentry_budget — Phase 2)
    - watchlist priority (high-priority watchlist items skip cooldown)
  """
  if bypass_for_falsifier:
    return (False, None)

  # Falsifier bypass — if a falsifier fired in the last 24h on a thesis for
  # this ticker, allow re-evaluation regardless of cooldown.
  if _has_recent_falsifier_alert(ticker, hours=24):
    return (False, None)

  in_cooldown, reason = is_in_cooldown(ticker)
  if in_cooldown:
    return (True, f"cooldown — {reason}")

  return (False, None)


def _has_recent_falsifier_alert(ticker: str, hours: int = 24) -> bool:
  """Check whether falsifier_watcher has fired on this ticker recently.

  The falsifier_alerts table is created on demand by daemons/falsifier_watcher.py
  (_ensure_alerts_table). If it doesn't exist yet, return False without crashing.
  """
  cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
  conn = get_connection()
  try:
    # The alerts table tracks (thesis_id, falsifier_hash, evidence_id). Join
    # against theses to filter by ticker.
    cur = conn.execute(
      """SELECT 1
         FROM falsifier_alerts a
         JOIN theses t ON t.thesis_id = a.thesis_id
         WHERE t.ticker = ? AND a.fired_at >= ?
         LIMIT 1""",
      (ticker.upper(), cutoff),
    )
    row = cur.fetchone()
    return row is not None
  except Exception:
    # falsifier_alerts table may not exist yet — that's OK
    return False
  finally:
    conn.close()
