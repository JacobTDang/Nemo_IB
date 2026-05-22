"""Sentry triage daemon — the master loop that decides which events deserve Claude.

Every TICK_INTERVAL_S seconds:
  1. Read unprocessed events from events_store
  2. For each event:
     a. Resolve to primary ticker (use event.primary_ticker, fallback to ticker)
     b. Check sentry_eval_log.should_skip — bypass if cooldown or recent verdict
     c. Score via agent.event_scorer.score_event
     d. If overall_score >= ENQUEUE_THRESHOLD: enqueue via sentry_queue.enqueue
        (the unique partial index dedupes same-ticker pending rows automatically)
  3. Mark all processed events via events_store.mark_processed
  4. Trim queue to MAX_QUEUE_PENDING

This daemon never invokes Claude — it just populates the queue. Claude
reasoning happens in the /sentry-tick skill (Phase 2).

Run:
  python -m daemons.sentry_triage                    # default 5min interval
  python -m daemons.sentry_triage --once             # single pass, useful for tests
  python -m daemons.sentry_triage --interval 60      # custom interval (seconds)
  python -m daemons.sentry_triage --max-queue 30     # custom queue cap

Stop via Ctrl+C — signal handler shuts down cleanly.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.events_store import unprocessed_events, mark_processed
from state.sentry_eval_log import record_eval, should_skip
from state import sentry_queue
from agent.event_scorer import score_event


# -- Tuning constants (Phase 5 will adjust based on observation) -------------
DEFAULT_INTERVAL_S = 300       # 5 min between ticks
DEFAULT_MAX_QUEUE = 20          # max pending candidates at any time
ENQUEUE_THRESHOLD = 0.50        # below this, log only
HIGH_SIGNAL_THRESHOLD = 0.70    # at this and above, force-enqueue even past cap
EVENTS_PER_TICK = 100           # max events to process per tick

_ET_OFFSET_HOURS = -5          # ET = UTC-5 (approximation; see agent/sentry_budget.py)

_running = True


def _today_et() -> str:
  """Today's date in ET as YYYY-MM-DD."""
  from datetime import timedelta as _td
  return (datetime.now(timezone.utc) + _td(hours=_ET_OFFSET_HOURS)).strftime('%Y-%m-%d')


def _discovery_ran_today() -> bool:
  """Check whether discovery channels have already run today."""
  conn = get_connection()
  try:
    row = conn.execute(
      "SELECT 1 FROM sentry_discovery_runs WHERE day = ?",
      (_today_et(),),
    ).fetchone()
    return row is not None
  finally:
    conn.close()


def _record_discovery_run(results: Dict[str, Dict[str, int]]) -> None:
  """Insert today's discovery run row with per-channel enqueue counts."""
  catalyst = results.get('catalyst_calendar', {}).get('enqueued', 0)
  insider = results.get('insider_cluster', {}).get('insider_clusters', 0)
  activist = results.get('insider_cluster', {}).get('activist_enqueued', 0)
  theme = results.get('theme_flow', {}).get('enqueued', 0)
  total = catalyst + insider + activist + theme

  conn = get_connection()
  try:
    conn.execute(
      """INSERT OR REPLACE INTO sentry_discovery_runs
         (day, ran_at, catalyst_enqueued, insider_enqueued, activist_enqueued,
          theme_flow_enqueued, total_enqueued, errors)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        _today_et(), datetime.now(timezone.utc).isoformat(),
        catalyst, insider, activist, theme, total, None,
      ),
    )
    conn.commit()
  finally:
    conn.close()


def _maybe_run_daily_discovery() -> bool:
  """If discovery hasn't run today, run all 3 channels and record the result.
  Returns True if discovery ran this call, False if it was already done today."""
  if _discovery_ran_today():
    return False

  print(f"[sentry_triage] running daily discovery scan (day={_today_et()})...",
        file=sys.stderr, flush=True)
  try:
    # Import here to avoid heavy imports when triage is event-only
    from daemons import sentry_discovery
    results = sentry_discovery.run_all()
    _record_discovery_run(results)
    return True
  except Exception as exc:
    import traceback
    print(f"[sentry_triage] discovery scan crashed: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    # Still record a row so we don't retry on every tick; mark errors
    conn = get_connection()
    try:
      conn.execute(
        """INSERT OR REPLACE INTO sentry_discovery_runs
           (day, ran_at, total_enqueued, errors)
           VALUES (?, ?, 0, ?)""",
        (_today_et(), datetime.now(timezone.utc).isoformat(),
         f"{type(exc).__name__}: {exc}"),
      )
      conn.commit()
    finally:
      conn.close()
    return False


def _install_signal_handlers() -> None:
  def _stop(*_):
    global _running
    _running = False
    print("[sentry_triage] shutdown signal received", file=sys.stderr, flush=True)

  signal.signal(signal.SIGINT, _stop)
  if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _stop)


def _resolve_ticker(event: Dict[str, Any]) -> str | None:
  """Return the primary ticker for an event, preferring primary_ticker then ticker."""
  return event.get('primary_ticker') or event.get('ticker')


def tick(max_queue: int = DEFAULT_MAX_QUEUE,
         skip_discovery: bool = False) -> Dict[str, int]:
  """Run one triage pass. Returns counts dict for logging/testing.

  On the first tick of each ET day, runs the 3 discovery channels
  (catalyst_calendar, insider_cluster, theme_flow) before processing
  events. Subsequent ticks the same day skip discovery (already done).

  Args:
    max_queue: trim pending queue to at most this size after the tick
    skip_discovery: set True in tests to bypass the daily discovery run
  """
  counts = {
    'fetched':           0,
    'no_ticker':         0,
    'skipped':           0,
    'scored':            0,
    'enqueued':          0,
    'below_thr':         0,
    'processed':         0,
    'discovery_ran':     False,
  }

  # 1. Day rollover: run discovery channels once per day
  if not skip_discovery:
    counts['discovery_ran'] = _maybe_run_daily_discovery()

  events = unprocessed_events(min_materiality='low', limit=EVENTS_PER_TICK)
  counts['fetched'] = len(events)
  if not events:
    return counts

  processed_event_ids: List[str] = []

  for ev in events:
    eid = ev.get('event_id')
    ticker = _resolve_ticker(ev)

    # 1. Resolve ticker — skip events with no ticker (e.g., pure macro events)
    if not ticker:
      counts['no_ticker'] += 1
      processed_event_ids.append(eid)
      continue

    ticker = ticker.upper()

    # 2. Cooldown / recent-verdict check
    skip, skip_reason = should_skip(ticker, ev)
    if skip:
      counts['skipped'] += 1
      record_eval(
        ticker, decision='skipped_cooldown', triggered_by='event_score',
        trigger_event_id=eid, skip_reason=skip_reason,
      )
      processed_event_ids.append(eid)
      continue

    # 3. Score
    score = score_event(ev)
    counts['scored'] += 1
    overall = score['overall_score']

    if overall < ENQUEUE_THRESHOLD:
      counts['below_thr'] += 1
      # No record_eval here — we don't want to flood the eval log with every
      # low-score event. Triage daemon only logs evals for events it acted on
      # (skipped via cooldown, or enqueued).
      processed_event_ids.append(eid)
      continue

    # 4. Enqueue
    qid = sentry_queue.enqueue(
      ticker, score=overall, triggered_by='event_score',
      source_event_id=eid,
      notes=f"mag={score['magnitude']} nov={score['novelty']} rel={score['reliability']} aware={score['market_awareness']} thrlv={score['thesis_relevance']}",
    )
    if qid:
      counts['enqueued'] += 1
    processed_event_ids.append(eid)

  # 5. Mark all processed
  if processed_event_ids:
    mark_processed(processed_event_ids)
    counts['processed'] = len(processed_event_ids)

  # 6. Trim queue to cap (low-score pending rows get dropped)
  trimmed = sentry_queue.trim_to_cap(max_queue)
  if trimmed:
    counts['trimmed'] = trimmed

  return counts


def _format_counts(counts: Dict[str, int]) -> str:
  return (
    f"fetched={counts['fetched']} "
    f"enqueued={counts['enqueued']} "
    f"skipped={counts['skipped']} "
    f"below_thr={counts['below_thr']} "
    f"no_ticker={counts['no_ticker']}"
    + (f" trimmed={counts['trimmed']}" if counts.get('trimmed') else '')
  )


def main() -> None:
  parser = argparse.ArgumentParser(description='Sentry triage daemon')
  parser.add_argument('--once', action='store_true', help='Run one tick and exit')
  parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_S,
                      help=f'Seconds between ticks (default {DEFAULT_INTERVAL_S})')
  parser.add_argument('--max-queue', type=int, default=DEFAULT_MAX_QUEUE,
                      help=f'Max pending queue size (default {DEFAULT_MAX_QUEUE})')
  args = parser.parse_args()

  init_schema()
  _install_signal_handlers()

  print(
    f"[sentry_triage] starting | interval={args.interval}s | max_queue={args.max_queue} | "
    f"enqueue_thr={ENQUEUE_THRESHOLD}",
    file=sys.stderr, flush=True,
  )

  while _running:
    started = time.time()
    try:
      counts = tick(max_queue=args.max_queue)
      elapsed = time.time() - started
      print(
        f"[sentry_triage] tick {datetime.now(timezone.utc).isoformat()} "
        f"({elapsed:.2f}s) {_format_counts(counts)}",
        file=sys.stderr, flush=True,
      )
    except Exception as exc:
      import traceback
      print(f"[sentry_triage] tick crashed: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True)
      traceback.print_exc(file=sys.stderr)

    if args.once:
      break

    # Sleep in small increments so Ctrl+C is responsive
    slept = 0.0
    while _running and slept < args.interval:
      time.sleep(min(2.0, args.interval - slept))
      slept += 2.0

  print("[sentry_triage] exited cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
  main()
