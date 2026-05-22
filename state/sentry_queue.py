"""Sentry's triage queue — pending candidates awaiting Claude review.

The triage daemon (`daemons/sentry_triage.py`) inserts rows. The `/sentry-tick`
skill reads top-N by score and processes them. The unique partial index on
(ticker) WHERE status='pending' (defined in schema.py) enforces deduplication
at the DB layer: same ticker can't be queued twice simultaneously even if
different discovery channels both flag it.

Statuses:
  pending     → newly queued, not yet picked up by /sentry-tick
  processing  → /sentry-tick has claimed it, work in progress
  completed   → processed successfully (eval row recorded in sentry_evaluation_log)
  dropped     → dropped from queue (queue cap overflow, or stale)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from state.schema import get_connection


def enqueue(
    ticker: str,
    score: float,
    *,
    triggered_by: str,
    source_event_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[int]:
  """Add a candidate to the queue. Idempotent: if a pending row for this
  ticker already exists, returns the existing queue_id without inserting.

  Returns the queue_id of the new (or pre-existing) pending row, or None
  if the insert failed for a reason other than the unique-pending constraint.
  """
  conn = get_connection()
  try:
    cur = conn.execute(
      """INSERT INTO sentry_queue
         (ticker, source_event_id, triggered_by, score, queued_at, status, notes)
         VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
      (
        ticker.upper(),
        source_event_id,
        triggered_by,
        float(score),
        datetime.now(timezone.utc).isoformat(),
        notes,
      ),
    )
    conn.commit()
    return cur.lastrowid
  except sqlite3.IntegrityError:
    # Unique index violation: a pending row already exists for this ticker.
    # Fetch its queue_id and return — caller can choose to update score if
    # the new candidate scores higher.
    cur = conn.execute(
      "SELECT queue_id, score FROM sentry_queue WHERE ticker = ? AND status = 'pending'",
      (ticker.upper(),),
    )
    row = cur.fetchone()
    if row and score > row['score']:
      # Upgrade the existing pending row to the higher score
      conn.execute(
        """UPDATE sentry_queue
           SET score = ?, triggered_by = ?, source_event_id = ?, queued_at = ?, notes = ?
           WHERE queue_id = ?""",
        (
          float(score), triggered_by, source_event_id,
          datetime.now(timezone.utc).isoformat(), notes, row['queue_id'],
        ),
      )
      conn.commit()
    return row['queue_id'] if row else None
  finally:
    conn.close()


def dequeue_top(n: int = 3) -> List[Dict[str, Any]]:
  """Return the top-N pending candidates by score, newest-queued as tiebreaker.

  Does NOT change status. Caller should call mark_status(queue_id, 'processing')
  before starting work, then 'completed' or 'dropped' when done.
  """
  conn = get_connection()
  try:
    cur = conn.execute(
      """SELECT queue_id, ticker, source_event_id, triggered_by, score,
                queued_at, status, notes
         FROM sentry_queue
         WHERE status = 'pending'
         ORDER BY score DESC, queued_at DESC
         LIMIT ?""",
      (int(n),),
    )
    return [dict(r) for r in cur.fetchall()]
  finally:
    conn.close()


def mark_status(queue_id: int, status: str, *, notes_append: Optional[str] = None) -> None:
  """Update queue row status. Sets processed_at when status leaves 'pending'."""
  if status not in ('pending', 'processing', 'completed', 'dropped'):
    raise ValueError(f"invalid status: {status}")

  conn = get_connection()
  try:
    if status == 'pending':
      conn.execute(
        "UPDATE sentry_queue SET status = ?, processed_at = NULL WHERE queue_id = ?",
        (status, queue_id),
      )
    else:
      processed_at = datetime.now(timezone.utc).isoformat()
      if notes_append:
        conn.execute(
          """UPDATE sentry_queue
             SET status = ?, processed_at = ?,
                 notes = COALESCE(notes || ' | ', '') || ?
             WHERE queue_id = ?""",
          (status, processed_at, notes_append, queue_id),
        )
      else:
        conn.execute(
          "UPDATE sentry_queue SET status = ?, processed_at = ? WHERE queue_id = ?",
          (status, processed_at, queue_id),
        )
    conn.commit()
  finally:
    conn.close()


def pending_count() -> int:
  """Number of pending rows currently in the queue."""
  conn = get_connection()
  try:
    cur = conn.execute("SELECT COUNT(*) AS n FROM sentry_queue WHERE status = 'pending'")
    return int(cur.fetchone()['n'])
  finally:
    conn.close()


def drop_below_score(min_score: float) -> int:
  """Queue cap enforcement — drop pending rows below `min_score`.

  Returns the number of rows dropped. Called by the triage daemon when the
  queue grows beyond `budget-max-queue` to keep only the highest-scored
  candidates.
  """
  conn = get_connection()
  try:
    cur = conn.execute(
      """UPDATE sentry_queue
         SET status = 'dropped', processed_at = ?
         WHERE status = 'pending' AND score < ?""",
      (datetime.now(timezone.utc).isoformat(), float(min_score)),
    )
    conn.commit()
    return cur.rowcount
  finally:
    conn.close()


def trim_to_cap(max_pending: int) -> int:
  """Drop the lowest-scored pending rows until pending_count <= max_pending.

  Returns the number of rows dropped. More precise than drop_below_score
  when you want an exact cap rather than a score threshold.
  """
  current = pending_count()
  if current <= max_pending:
    return 0

  to_drop = current - max_pending
  conn = get_connection()
  try:
    # Find the lowest-scored pending rows, drop them
    cur = conn.execute(
      """UPDATE sentry_queue
         SET status = 'dropped', processed_at = ?
         WHERE queue_id IN (
             SELECT queue_id FROM sentry_queue
             WHERE status = 'pending'
             ORDER BY score ASC, queued_at ASC
             LIMIT ?
         )""",
      (datetime.now(timezone.utc).isoformat(), to_drop),
    )
    conn.commit()
    return cur.rowcount
  finally:
    conn.close()


def get_by_id(queue_id: int) -> Optional[Dict[str, Any]]:
  """Fetch a single queue row by id."""
  conn = get_connection()
  try:
    cur = conn.execute(
      """SELECT queue_id, ticker, source_event_id, triggered_by, score,
                queued_at, status, processed_at, notes
         FROM sentry_queue WHERE queue_id = ?""",
      (queue_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None
  finally:
    conn.close()
