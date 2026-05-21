"""End-to-end stress tests for the Falsifier Watcher daemon.

Each test seeds the DB with a thesis + events, runs evaluate_thesis() or
tick(), then inspects the DB state. Tests clean up after themselves.

Categories:
  1. Single-thesis trigger flow (event → alert → evolution row)
  2. Idempotency — same trigger must not fire twice
  3. Multi-thesis concurrent tick
  4. Negative cases — non-matching events shouldn't fire
  5. Macro numeric thresholds — observed value crosses threshold
  6. Performance — 50 theses × 5 falsifiers × 100 events
  7. Adversarial — malformed falsifiers, missing fields

Run via:
  ./.venv/Scripts/python.exe testing/test_falsifier_watcher_e2e.py
"""
from __future__ import annotations

import sys
import os
import time
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.theses import (
    insert_thesis, get_thesis, get_thesis_evolution, latest_thesis,
)
from state.events_store import store_event
from daemons.falsifier_watcher import (
    evaluate_thesis, tick, _ensure_alerts_table, _already_alerted,
    _falsifier_hash,
)


# ===========================================================================
# Test infrastructure
# ===========================================================================

_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition, hint: str = ''):
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _section(title: str):
  print(f"\n=== {title} ===")


def _cleanup_test_theses(prefix: str = "FWTEST_"):
  """Remove all theses + evolution + alerts created by tests with this ticker prefix."""
  conn = get_connection()
  try:
    # Get the thesis IDs we'll delete
    rows = conn.execute(
      f"SELECT thesis_id FROM theses WHERE ticker LIKE '{prefix}%'"
    ).fetchall()
    ids = [r['thesis_id'] for r in rows]
    if ids:
      placeholders = ','.join('?' * len(ids))
      conn.execute(f"DELETE FROM thesis_evolution WHERE thesis_id IN ({placeholders})", ids)
      conn.execute(f"DELETE FROM falsifier_alerts WHERE thesis_id IN ({placeholders})", ids)
    conn.execute(f"DELETE FROM theses WHERE ticker LIKE '{prefix}%'")
    # Clean test events (those with ticker prefix in primary_ticker)
    conn.execute(
      f"DELETE FROM events WHERE ticker LIKE '{prefix}%' OR primary_ticker LIKE '{prefix}%'"
    )
    conn.commit()
  finally:
    conn.close()


def _seed_thesis(ticker: str, falsifiers: list, confidence: float = 0.70) -> int:
  return insert_thesis(
    ticker=ticker,
    recommendation='BUY',
    signal='entry',
    target_price=100.0,
    stop_loss=80.0,
    confidence=confidence,
    analysis_summary=f'Test thesis for {ticker}',
    key_assumptions=['k1', 'k2'],
    data_gaps=[],
    full_report_md='test',
    falsifiers=falsifiers,
    variant_perception='test',
  )


def _seed_event(ticker: str, headline: str, body: str = '',
                materiality: str = 'high', source: str = 'test') -> str:
  return store_event(
    source=source,
    ticker=ticker,
    headline=headline,
    body=body,
    url=f'https://example.com/{int(time.time()*1000)}',
    published_at=datetime.now().isoformat(),
    materiality=materiality,
    category='financial_results',
    affected_tickers=[ticker],
    primary_ticker=ticker,
    directional_signal='neutral',
    urgency='days',
    classifier_reason='test seed',
  )


# ===========================================================================
# 1. Single-thesis trigger flow
# ===========================================================================

def test_single_thesis_trigger():
  _section("1. Single-thesis trigger flow")

  ticker = 'FWTEST_T1'
  thesis_id = _seed_thesis(ticker, falsifiers=[
    'Amy Hood replaced as CFO',
    '10Y treasury > 5.25%',
    'Major customer contract terminated',
  ], confidence=0.72)
  _seed_event(
    ticker,
    'Microsoft announces Amy Hood will be replaced as CFO effective Q4',
    body='In a regulatory filing today, Microsoft Corporation announced that Amy Hood will be replaced as Chief Financial Officer.',
  )

  summary = evaluate_thesis(get_thesis(thesis_id), observed={}, log_fn=lambda *a: None)
  triggers = summary['triggers']
  _check("  produced 1 trigger from matching event",
         len(triggers) == 1 and not triggers[0]['duplicate'],
         f"triggers={[t['falsifier'][:40] for t in triggers]}")
  _check("  recorded thesis_evolution row",
         len(get_thesis_evolution(thesis_id)) == 1)
  th = get_thesis(thesis_id)
  _check("  thesis confidence dropped from 0.72 to ~0.62",
         abs(th['confidence'] - 0.62) < 1e-6,
         f"got {th['confidence']}")


# ===========================================================================
# 2. Idempotency
# ===========================================================================

def test_idempotency():
  _section("2. Idempotency — same trigger should not fire twice")

  ticker = 'FWTEST_IDEMP'
  thesis_id = _seed_thesis(ticker, ['Major restructuring announced'])
  _seed_event(ticker, 'Company X announces Major Restructuring across all divisions')

  # First evaluation
  s1 = evaluate_thesis(get_thesis(thesis_id), observed={}, log_fn=lambda *a: None)
  new1 = [t for t in s1['triggers'] if not t['duplicate']]
  _check("  first pass fires exactly 1 trigger", len(new1) == 1,
         f"got {len(new1)}")

  # Second evaluation — same event, same falsifier
  s2 = evaluate_thesis(get_thesis(thesis_id), observed={}, log_fn=lambda *a: None)
  new2 = [t for t in s2['triggers'] if not t['duplicate']]
  dup2 = [t for t in s2['triggers'] if t['duplicate']]
  _check("  second pass: 0 new triggers", len(new2) == 0)
  _check("  second pass: marked as duplicate", len(dup2) >= 1,
         f"got {len(dup2)} duplicates")
  _check("  thesis_evolution has only 1 entry (no double-record)",
         len(get_thesis_evolution(thesis_id)) == 1)


# ===========================================================================
# 3. Multi-thesis concurrent tick
# ===========================================================================

def test_multi_thesis_tick():
  _section("3. Multi-thesis tick")

  # Create 5 theses with different falsifiers
  ids = []
  for i in range(5):
    tk = f'FWTEST_M{i}'
    tid = _seed_thesis(tk, falsifiers=[f'Falsifier specific to {tk} happened'])
    _seed_event(tk, f'Falsifier specific to {tk} happened today — major news')
    ids.append(tid)

  # Run tick
  summary = tick(log_fn=lambda *a: None)
  _check("  tick scanned at least 5 test theses (may include other active)",
         summary['theses_scanned'] >= 5)
  _check("  at least 5 new triggers fired (one per test thesis)",
         summary['new_triggers'] >= 5,
         f"got {summary['new_triggers']}")

  # Confirm each thesis has an evolution row
  for tid in ids:
    ev = get_thesis_evolution(tid)
    _check(f"  thesis {tid} has 1 evolution row", len(ev) == 1)


# ===========================================================================
# 4. Negative cases — non-matching events should NOT fire
# ===========================================================================

def test_no_false_positives():
  _section("4. False-positive guard — irrelevant events")

  ticker = 'FWTEST_NEG'
  thesis_id = _seed_thesis(ticker, falsifiers=[
    'CFO Amy Hood replaced',
    'Azure revenue growth drops below 18% YoY',
  ])
  # Seed events that are TOTALLY UNRELATED
  _seed_event(ticker, 'Company opens new office in Dublin Ireland')
  _seed_event(ticker, 'Annual employee satisfaction survey reports positive results')
  _seed_event(ticker, 'Quarterly dividend declared at $0.83 per share')

  summary = evaluate_thesis(get_thesis(thesis_id), observed={}, log_fn=lambda *a: None)
  new_triggers = [t for t in summary['triggers'] if not t['duplicate']]
  _check("  no false-positive triggers on unrelated events",
         len(new_triggers) == 0, f"got {new_triggers}")


# ===========================================================================
# 5. Macro numeric threshold crossing
# ===========================================================================

def test_macro_numeric_trigger():
  _section("5. Macro numeric trigger (no event, observed value crosses)")

  ticker = 'FWTEST_MACRO'
  thesis_id = _seed_thesis(ticker, ['10Y treasury > 5.25%'])
  # Seed no events — observed macro alone should fire it

  # Below threshold — should NOT trigger
  s1 = evaluate_thesis(get_thesis(thesis_id),
                       observed={'10Y treasury': 4.67},
                       log_fn=lambda *a: None)
  _check("  10Y at 4.67% does not trigger (>5.25 falsifier)",
         len([t for t in s1['triggers'] if not t['duplicate']]) == 0)

  # Above threshold — should trigger
  s2 = evaluate_thesis(get_thesis(thesis_id),
                       observed={'10Y treasury': 5.45},
                       log_fn=lambda *a: None)
  triggers = [t for t in s2['triggers'] if not t['duplicate']]
  _check("  10Y at 5.45% triggers (>5.25 falsifier)",
         len(triggers) == 1,
         f"got {len(triggers)} triggers")


# ===========================================================================
# 6. Performance — 50 theses × 5 falsifiers × 100 events each
# ===========================================================================

def test_performance_scale():
  _section("6. Performance — 50 theses × 5 falsifiers")

  ids = []
  for i in range(50):
    tk = f'FWTEST_P{i:03d}'
    tid = _seed_thesis(tk, falsifiers=[
      'Major customer contract loss',
      f'Revenue growth drops below {15 + (i % 5)}% YoY',
      'CFO replaced',
      'Regulatory investigation opened',
      'Margin compression vs prior year',
    ])
    # Seed 3 events per ticker (varying noise)
    _seed_event(tk, f'Routine quarterly update for {tk} — no major changes')
    _seed_event(tk, f'Industry conference attended by company representatives')
    if i % 7 == 0:
      _seed_event(tk, f'Major customer contract loss reported in 10-Q for {tk}')
    ids.append(tid)

  t0 = time.time()
  summary = tick(log_fn=lambda *a: None)
  elapsed = time.time() - t0

  _check("  50-thesis tick under 20 seconds",
         elapsed < 20.0, f"elapsed={elapsed:.2f}s")
  # We expect ~50/7 = ~8 triggers from the seeded "Major customer contract loss" events
  expected_min_triggers = 5
  _check(f"  at least {expected_min_triggers} triggers (one per 7th thesis)",
         summary['new_triggers'] >= expected_min_triggers,
         f"got {summary['new_triggers']}")
  print(f"  -- {summary['theses_scanned']} theses, {summary['total_triggers']} triggers, {elapsed:.2f}s")


# ===========================================================================
# 7. Adversarial — malformed inputs
# ===========================================================================

def test_adversarial_inputs():
  _section("7. Adversarial / malformed inputs")

  ticker = 'FWTEST_ADV'
  # Empty falsifier list
  tid_empty = _seed_thesis(ticker + '_EMPTY', falsifiers=[])
  s = evaluate_thesis(get_thesis(tid_empty), observed={}, log_fn=lambda *a: None)
  _check("  empty falsifier list -> skipped gracefully",
         s.get('skipped_reason') is not None)

  # Falsifier with empty string mixed in
  tid_mix = _seed_thesis(ticker + '_MIX', falsifiers=['', 'CFO replaced', None])
  _seed_event(ticker + '_MIX', 'No relevant news today')
  s = evaluate_thesis(get_thesis(tid_mix), observed={}, log_fn=lambda *a: None)
  _check("  empty/None falsifiers skipped without crash",
         isinstance(s.get('triggers'), list))

  # Falsifier that's just whitespace
  tid_ws = _seed_thesis(ticker + '_WS', falsifiers=['   ', '\n\t'])
  s = evaluate_thesis(get_thesis(tid_ws), observed={}, log_fn=lambda *a: None)
  _check("  whitespace-only falsifiers -> 0 triggers",
         len(s.get('triggers', [])) == 0)


# ===========================================================================
# 8. Persistent alert table verification
# ===========================================================================

def test_alert_table_persistence():
  _section("8. Alert table persistence")

  ticker = 'FWTEST_ALERTS'
  thesis_id = _seed_thesis(ticker, ['Specific contract loss event'])
  _seed_event(ticker, 'Specific contract loss event reported in 10-Q')

  s = evaluate_thesis(get_thesis(thesis_id), observed={}, log_fn=lambda *a: None)
  new = [t for t in s['triggers'] if not t['duplicate']]
  _check("  alert fired", len(new) == 1)

  conn = get_connection()
  try:
    rows = conn.execute(
      "SELECT * FROM falsifier_alerts WHERE thesis_id = ?",
      (thesis_id,)
    ).fetchall()
    _check("  alert row persisted to falsifier_alerts", len(rows) == 1,
           f"got {len(rows)} rows")
    if rows:
      r = rows[0]
      _check("  alert row has score > 0", r['score'] > 0)
      _check("  alert row has correct ticker", r['ticker'] == ticker)
      _check("  alert row has falsifier_hash", bool(r['falsifier_hash']))
  finally:
    conn.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
  print("\nFalsifier Watcher — end-to-end stress tests\n")
  init_schema()
  _ensure_alerts_table()

  # Always clean any leftover test data BEFORE running so prior failed
  # runs don't pollute results
  _cleanup_test_theses()

  try:
    test_single_thesis_trigger()
    test_idempotency()
    test_multi_thesis_tick()
    test_no_false_positives()
    test_macro_numeric_trigger()
    test_performance_scale()
    test_adversarial_inputs()
    test_alert_table_persistence()
  finally:
    _cleanup_test_theses()

  print(f"\n=== Summary ===")
  print(f"  PASS: {_results['pass']}")
  print(f"  FAIL: {_results['fail']}")
  if _results['failures']:
    print("\nFailures:")
    for name, hint in _results['failures']:
      print(f"  - {name}: {hint}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
