"""Always-on watcher that monitors active theses for falsifier triggers.

Loop, every TICK_INTERVAL seconds:
  1. Load all active theses from SQLite
  2. For each thesis:
     a. Pull recent events for the ticker (events_store)
     b. Pull current macro snapshot (cached)
     c. For each falsifier:
        - evaluate against (events + macro)
        - if triggered AND not already recorded as triggered:
          - record to thesis_evolution with negative conviction delta
          - log alert
          - mark thesis_evolution.tag = 'falsifier_triggered'
  3. Sleep TICK_INTERVAL

Idempotency: the watcher tracks which (thesis_id, falsifier_text, evidence_id)
triples it has already fired on via a small table `falsifier_alerts`. Same
trigger never fires twice.

Run via:
  python -m daemons.falsifier_watcher
  python -m daemons.falsifier_watcher --once    (single pass, useful for tests)
  python -m daemons.falsifier_watcher --interval 300   (seconds)

Stop via Ctrl+C — handler cleans up gracefully.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
import sys
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.theses import (
    active_theses, record_thesis_evolution, get_thesis_evolution,
)
from state.events_store import recent_events_for_ticker
from agent.falsifier_evaluator import evaluate_falsifier


DEFAULT_INTERVAL_S = 900     # 15 minutes between ticks
DEFAULT_LOOKBACK_HOURS = 48  # events from last 48h
DEFAULT_CONVICTION_DELTA = -0.10  # default penalty when a falsifier fires
TRIGGER_THRESHOLD = 0.35

_running = True


def _ensure_alerts_table():
  """Idempotency table: (thesis_id, falsifier_hash, evidence_id) -> already_fired."""
  conn = get_connection()
  try:
    conn.execute("""
      CREATE TABLE IF NOT EXISTS falsifier_alerts(
        alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        thesis_id       INTEGER NOT NULL,
        ticker          TEXT,
        falsifier_hash  TEXT,
        falsifier_text  TEXT,
        evidence_id     TEXT,
        score           REAL,
        reason          TEXT,
        fired_at        TIMESTAMP,
        UNIQUE(thesis_id, falsifier_hash, evidence_id)
      )
    """)
    conn.execute(
      "CREATE INDEX IF NOT EXISTS idx_falsifier_alerts_thesis "
      "ON falsifier_alerts(thesis_id, fired_at)"
    )
    conn.commit()
  finally:
    conn.close()


def _falsifier_hash(falsifier_text: str) -> str:
  return hashlib.sha256(falsifier_text.strip().encode()).hexdigest()[:16]


def _already_alerted(thesis_id: int, falsifier_hash: str,
                     evidence_id: Optional[str]) -> bool:
  conn = get_connection()
  try:
    row = conn.execute("""
      SELECT 1 FROM falsifier_alerts
      WHERE thesis_id=? AND falsifier_hash=? AND COALESCE(evidence_id,'')=COALESCE(?, '')
    """, (thesis_id, falsifier_hash, evidence_id)).fetchone()
    return row is not None
  finally:
    conn.close()


def _record_alert(thesis_id: int, ticker: str, falsifier_text: str,
                  evidence_id: Optional[str], score: float, reason: str) -> int:
  conn = get_connection()
  try:
    cur = conn.execute("""
      INSERT OR IGNORE INTO falsifier_alerts
        (thesis_id, ticker, falsifier_hash, falsifier_text,
         evidence_id, score, reason, fired_at)
      VALUES (?,?,?,?,?,?,?,?)
    """, (thesis_id, ticker, _falsifier_hash(falsifier_text), falsifier_text,
          evidence_id, float(score), reason,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()
    return cur.lastrowid or 0
  finally:
    conn.close()


_MACRO_CACHE_PATH = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  'db_cache', 'macro_snapshot.json',
)


def fetch_macro_observed() -> Dict[str, float]:
  """Read macro observations from the on-disk cache written by the
  separate macro-snapshot writer (or any other process).

  The watcher itself should NOT hit FRED every tick — that would leak
  aiohttp connections and burn rate-limit budget. A separate process
  (e.g. a periodic job or the news_watcher) writes the cache; we just
  read it.

  Cache format:
    {"DGS10": {"label": "10Y Treasury Yield", "current": 4.67}, ...}
  Returns flat dict keyed by both raw key and label.
  """
  observed: Dict[str, float] = {}
  try:
    if os.path.exists(_MACRO_CACHE_PATH):
      with open(_MACRO_CACHE_PATH, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
      for key, info in (data or {}).items():
        if isinstance(info, dict) and 'current' in info:
          try:
            v = float(info['current'])
            observed[key] = v
            if info.get('label'):
              observed[info['label']] = v
          except (TypeError, ValueError):
            continue
        elif isinstance(info, (int, float)):
          observed[key] = float(info)
  except Exception:
    pass
  return observed


def write_macro_snapshot(snapshot: Dict[str, Any]) -> None:
  """Helper for the macro-cache writer (separate job). Atomically writes the
  snapshot dict to disk so the watcher reads a consistent view."""
  os.makedirs(os.path.dirname(_MACRO_CACHE_PATH), exist_ok=True)
  tmp = _MACRO_CACHE_PATH + '.tmp'
  with open(tmp, 'w', encoding='utf-8') as fh:
    json.dump(snapshot, fh)
  os.replace(tmp, _MACRO_CACHE_PATH)


def evaluate_thesis(thesis: Dict[str, Any],
                    observed: Optional[Dict[str, float]] = None,
                    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
                    log_fn=print) -> Dict[str, Any]:
  """Evaluate one thesis's falsifiers against recent events + macro.
  Returns a summary dict; side-effects: records alerts + evolution rows."""
  thesis_id = thesis['thesis_id']
  ticker = thesis['ticker']
  falsifiers = thesis.get('falsifiers') or []
  if not falsifiers:
    return {'thesis_id': thesis_id, 'ticker': ticker,
            'skipped_reason': 'no falsifiers defined', 'triggers': []}

  # Pull recent events for this ticker
  events = recent_events_for_ticker(ticker, hours=lookback_hours) or []
  # Each event becomes one evidence item. Concatenate headline + body.
  evidence_pool = []
  for ev in events:
    evidence_pool.append({
      'text':         (ev.get('headline') or '') + ' ' + (ev.get('body') or ''),
      'headline':     ev.get('headline'),
      'body':         ev.get('body'),
      'event_id':     ev.get('event_id'),
      'source':       ev.get('source'),
      'published_at': ev.get('published_at'),
    })

  observed = observed if observed is not None else fetch_macro_observed()

  triggers = []
  for falsifier_text in falsifiers:
    if not isinstance(falsifier_text, str) or not falsifier_text.strip():
      continue
    r = evaluate_falsifier(
      falsifier_text, evidence_pool,
      observed_values=observed,
      threshold=TRIGGER_THRESHOLD,
    )
    if not r.triggered:
      continue

    fhash = _falsifier_hash(falsifier_text)
    evidence_id = (r.best_evidence or {}).get('event_id')
    if _already_alerted(thesis_id, fhash, evidence_id):
      triggers.append({
        'falsifier':   falsifier_text,
        'duplicate':   True,
        'evidence_id': evidence_id,
        'score':       r.score,
      })
      continue

    _record_alert(thesis_id, ticker, falsifier_text, evidence_id,
                  r.score, r.reason)

    # Record to thesis_evolution so the analyst's evolution log captures it
    try:
      record_thesis_evolution(
        thesis_id=thesis_id,
        observation=(f"FALSIFIER TRIGGERED: {falsifier_text} | "
                     f"evidence: {(r.best_evidence or {}).get('headline') or ''} | "
                     f"reason: {r.reason}"),
        conviction_delta=DEFAULT_CONVICTION_DELTA,
        tag='falsifier_triggered',
      )
    except Exception as exc:
      log_fn(f"  [warn] could not record evolution for thesis {thesis_id}: {exc}")

    log_fn(f"  ALERT  thesis={thesis_id} {ticker} | "
           f"score={r.score:.3f} reason={r.reason} | "
           f"falsifier={falsifier_text[:80]}")

    triggers.append({
      'falsifier':         falsifier_text,
      'duplicate':         False,
      'evidence_id':       evidence_id,
      'score':             r.score,
      'reason':            r.reason,
      'matched_tokens':    r.matched_tokens,
      'numeric_conditions': r.numeric_conditions,
    })

  return {
    'thesis_id': thesis_id,
    'ticker':    ticker,
    'falsifier_count': len(falsifiers),
    'event_count':     len(events),
    'triggers':        triggers,
    'new_triggers':    [t for t in triggers if not t.get('duplicate')],
  }


def tick(log_fn=print) -> Dict[str, Any]:
  """One pass over all active theses. Returns aggregate summary."""
  t0 = time.time()
  theses = active_theses()
  observed = fetch_macro_observed()
  log_fn(f"[tick] {len(theses)} active theses, "
         f"observed macro keys: {len(observed)}")

  total_triggers = 0
  new_triggers = 0
  per_thesis = []
  for th in theses:
    summary = evaluate_thesis(th, observed=observed, log_fn=log_fn)
    per_thesis.append(summary)
    total_triggers += len(summary.get('triggers', []))
    new_triggers += len(summary.get('new_triggers', []))

  elapsed = time.time() - t0
  log_fn(f"[tick done] {len(theses)} theses scanned in {elapsed:.2f}s | "
         f"{total_triggers} triggers ({new_triggers} new)")
  return {
    'theses_scanned': len(theses),
    'total_triggers': total_triggers,
    'new_triggers':   new_triggers,
    'elapsed_s':      elapsed,
    'per_thesis':     per_thesis,
  }


def _install_signal_handlers():
  def _stop(*_):
    global _running
    _running = False
    print("[falsifier_watcher] shutdown signal received", file=sys.stderr,
          flush=True)
  signal.signal(signal.SIGINT, _stop)
  if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _stop)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--once', action='store_true',
                      help='Run one tick and exit')
  parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_S,
                      help='Seconds between ticks')
  args = parser.parse_args()

  init_schema()
  _ensure_alerts_table()
  _install_signal_handlers()

  print(f"[falsifier_watcher] starting | interval={args.interval}s",
        file=sys.stderr, flush=True)

  while _running:
    try:
      tick()
    except Exception as exc:
      print(f"[falsifier_watcher] tick crashed: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True)
    if args.once:
      break
    # Sleep in small increments so Ctrl+C is responsive
    slept = 0
    while _running and slept < args.interval:
      time.sleep(min(2.0, args.interval - slept))
      slept += 2.0

  print("[falsifier_watcher] exited cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
  main()
