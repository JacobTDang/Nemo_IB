"""Integration tests for the Sentry triage layer.

Covers:
  - End-to-end flow: synthetic events injected → tick() → expected queue state
  - Cooldown skip path: ticker with recent eval → skipped, no queue row
  - Dedup: same ticker referenced in multiple events → exactly 1 pending row
  - Idempotency: rerunning tick() with no new events → no new enqueues
  - No-ticker events handled gracefully

Run:
  .venv\\Scripts\\python.exe testing\\test_sentry_triage.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import get_connection, init_schema
from state import sentry_queue, sentry_eval_log
from daemons.sentry_triage import tick


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


_TEST_PREFIX_EVENT = 'INTEGTEST_'
_TEST_PREFIX_TICKER = 'INTEG_'


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE event_id LIKE ?", (f"{_TEST_PREFIX_EVENT}%",))
    conn.execute("DELETE FROM sentry_queue WHERE ticker LIKE ?", (f"{_TEST_PREFIX_TICKER}%",))
    conn.execute("DELETE FROM sentry_evaluation_log WHERE ticker LIKE ?", (f"{_TEST_PREFIX_TICKER}%",))
    conn.commit()
  finally:
    conn.close()


def _inject_event(event_id, ticker, source='sec_edgar', category='8-K',
                  materiality='high', urgency='breaking',
                  published_at=None):
  if published_at is None:
    published_at = datetime.now(timezone.utc).isoformat()
  conn = get_connection()
  try:
    conn.execute(
      """INSERT INTO events
         (event_id, source, ticker, headline, body, url, published_at, ingested_at,
          materiality, category, affected_tickers, primary_ticker,
          directional_signal, urgency, classifier_reason, processed)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
      (event_id, source, ticker, f"{ticker} headline", "body", "http://x",
       published_at, published_at, materiality, category, None, ticker,
       'positive', urgency, 'test'),
    )
    conn.commit()
  finally:
    conn.close()


def test_end_to_end_basic():
  print("\n== end-to-end: high signal enqueues, low signal filtered ==")
  _cleanup()

  _inject_event(f'{_TEST_PREFIX_EVENT}1', f'{_TEST_PREFIX_TICKER}A')   # high
  _inject_event(f'{_TEST_PREFIX_EVENT}2', f'{_TEST_PREFIX_TICKER}B',
                source='finnhub', category='guidance_cut', materiality='high', urgency='high')
  _inject_event(f'{_TEST_PREFIX_EVENT}3', f'{_TEST_PREFIX_TICKER}C',
                source='finnhub', category='general_news', materiality='low', urgency='low')

  counts = tick(skip_discovery=True)
  _check("enqueued >= 2 (TST_A and TST_B)", counts['enqueued'] >= 2,
         f"got {counts['enqueued']}")
  _check("below_threshold >= 1 (TST_C)", counts['below_thr'] >= 1,
         f"got {counts['below_thr']}")

  pending = sentry_queue.dequeue_top(10)
  tickers = {r['ticker'] for r in pending}
  _check("TST_A in queue", f'{_TEST_PREFIX_TICKER}A' in tickers, str(tickers))
  _check("TST_B in queue", f'{_TEST_PREFIX_TICKER}B' in tickers, str(tickers))
  _check("TST_C NOT in queue", f'{_TEST_PREFIX_TICKER}C' not in tickers, str(tickers))


def test_dedup_same_ticker():
  print("\n== dedup: same ticker referenced twice gets 1 pending row ==")
  _cleanup()

  _inject_event(f'{_TEST_PREFIX_EVENT}4', f'{_TEST_PREFIX_TICKER}D')
  _inject_event(f'{_TEST_PREFIX_EVENT}5', f'{_TEST_PREFIX_TICKER}D',
                source='finnhub', category='guidance_cut', materiality='high', urgency='high')

  tick(skip_discovery=True)
  pending = sentry_queue.dequeue_top(10)
  d_rows = [r for r in pending if r['ticker'] == f'{_TEST_PREFIX_TICKER}D']
  _check("TST_D appears exactly once (dedup)", len(d_rows) == 1, f"got {len(d_rows)}")


def test_cooldown_skip():
  print("\n== cooldown skip: ticker with recent eval is not re-queued ==")
  _cleanup()

  # Pre-record a recent 'acted' eval for TST_E (sets 7-day cooldown)
  sentry_eval_log.record_eval(
    f'{_TEST_PREFIX_TICKER}E', decision='acted', triggered_by='manual',
    verdict='buy', confidence=0.75, sizing='cautious',
  )

  # Now inject a high-signal event for TST_E
  _inject_event(f'{_TEST_PREFIX_EVENT}6', f'{_TEST_PREFIX_TICKER}E')

  counts = tick(skip_discovery=True)
  _check("event was skipped due to cooldown", counts['skipped'] >= 1,
         f"got {counts['skipped']}")

  pending = sentry_queue.dequeue_top(10)
  e_rows = [r for r in pending if r['ticker'] == f'{_TEST_PREFIX_TICKER}E']
  _check("TST_E NOT in queue (cooldown active)", len(e_rows) == 0, f"got {len(e_rows)}")


def test_idempotency():
  print("\n== idempotency: rerun tick without new events ==")
  _cleanup()
  _inject_event(f'{_TEST_PREFIX_EVENT}7', f'{_TEST_PREFIX_TICKER}F')
  counts1 = tick(skip_discovery=True)
  counts2 = tick(skip_discovery=True)
  _check("first tick enqueued >= 1",
         counts1['enqueued'] >= 1, f"got {counts1['enqueued']}")
  _check("second tick enqueues 0 (all events already processed)",
         counts2['enqueued'] == 0, f"got {counts2['enqueued']}")


def test_no_ticker_event():
  print("\n== no-ticker events don't crash ==")
  _cleanup()

  # Pure macro event with no ticker
  conn = get_connection()
  try:
    conn.execute(
      """INSERT INTO events
         (event_id, source, ticker, headline, body, url, published_at, ingested_at,
          materiality, category, affected_tickers, primary_ticker,
          directional_signal, urgency, classifier_reason, processed)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
      (f'{_TEST_PREFIX_EVENT}8', 'fred', None, 'CPI surprise', 'macro body',
       'http://x', datetime.now(timezone.utc).isoformat(),
       datetime.now(timezone.utc).isoformat(),
       'high', 'macro_shock', None, None, 'positive', 'high', 'test'),
    )
    conn.commit()
  finally:
    conn.close()

  counts = tick(skip_discovery=True)
  _check("no_ticker count >= 1", counts['no_ticker'] >= 1, f"got {counts['no_ticker']}")
  _check("tick didn't crash", True)


def test_schema_idempotent():
  print("\n== schema migration is idempotent ==")
  init_schema()
  init_schema()
  _check("init_schema called twice without error", True)


def main():
  print("\nSentry triage integration tests\n")
  test_schema_idempotent()
  test_end_to_end_basic()
  test_dedup_same_ticker()
  test_cooldown_skip()
  test_idempotency()
  test_no_ticker_event()
  _cleanup()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
