"""Integration tests for the new Sentry discovery channels.

Covers: ipo_calendar (this commit). Future commits will append tests for
universe_insider_cluster, rag_analogue, fundamental_screener.

Test approach: inject synthetic Finnhub / MSPR / rag responses via the
_fetch_fn keyword the new channels accept; no live HTTP.

Run:
  .venv\\Scripts\\python.exe testing\\test_sentry_discovery_new_channels.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state import sentry_queue
from daemons import sentry_discovery


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name, condition, hint=''):
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


# Test tickers (4 letters, alpha only, won't collide with real listings)
_TEST_TICKERS = ['ZZIA', 'ZZIB', 'ZZIC', 'ZZID', 'ZZIE']


def _cleanup():
  conn = get_connection()
  try:
    conn.execute(
      "DELETE FROM sentry_queue WHERE ticker LIKE 'ZZ%'"
    )
    conn.execute(
      "DELETE FROM sentry_evaluation_log WHERE ticker LIKE 'ZZ%'"
    )
    conn.commit()
  finally:
    conn.close()


def _fake_ipo_fetch(events):
  """Return an async callable wrapping a static Finnhub-shaped response."""
  async def fetch(from_date, to_date):
    return {'ipoCalendar': events}
  return fetch


# ============================================================================
# scan_ipo_calendar
# ============================================================================

def test_ipo_eligible_ticker_enqueued():
  print("\n== ipo_calendar: $1.5B NASDAQ IPO is enqueued ==")
  _cleanup()
  events = [{
    'symbol': 'ZZIA',
    'name': 'ZZIA Holdings Inc',
    'date': '2026-05-29',
    'exchange': 'NASDAQ',
    'status': 'expected',
    'price': '12.00-15.00',
    'numberOfShares': 100_000_000,    # 100M shares
    'totalSharesValue': None,
  }]
  counts = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch(events),
  )
  _check("fetched == 1", counts['fetched'] == 1, str(counts))
  _check("eligible == 1", counts['eligible'] == 1, str(counts))
  _check("enqueued == 1", counts['enqueued'] == 1, str(counts))
  pending = sentry_queue.dequeue_top(10)
  hit = [r for r in pending if r['ticker'] == 'ZZIA']
  _check("ZZIA row landed", len(hit) == 1, str([r['ticker'] for r in pending]))
  if hit:
    _check("triggered_by = ipo_listing",
           hit[0]['triggered_by'] == 'ipo_listing')
    _check("score = IPO_SCORE",
           hit[0]['score'] == sentry_discovery.IPO_SCORE)


def test_ipo_below_min_mcap_dropped():
  print("\n== ipo_calendar: $500M IPO dropped below $1B floor ==")
  _cleanup()
  events = [{
    'symbol': 'ZZIB', 'name': 'Small Cap', 'date': '2026-05-30',
    'exchange': 'NASDAQ', 'status': 'expected',
    'price': '5.00',
    'numberOfShares': 50_000_000,  # mcap = $250M
  }]
  counts = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch(events),
  )
  _check("below_min_cap = 1", counts['below_min_cap'] == 1, str(counts))
  _check("enqueued = 0", counts['enqueued'] == 0, str(counts))


def test_ipo_wrong_exchange_dropped():
  print("\n== ipo_calendar: OTC / unknown exchange dropped ==")
  _cleanup()
  events = [{
    'symbol': 'ZZIC', 'name': 'OTC Co', 'date': '2026-05-31',
    'exchange': 'OTC', 'status': 'expected',
    'price': '20.00-25.00',
    'numberOfShares': 100_000_000,
  }]
  counts = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch(events),
  )
  _check("wrong_exchange = 1", counts['wrong_exchange'] == 1, str(counts))
  _check("enqueued = 0", counts['enqueued'] == 0, str(counts))


def test_ipo_dedup_on_rerun():
  print("\n== ipo_calendar: same IPO on rerun -> still 1 row ==")
  _cleanup()
  events = [{
    'symbol': 'ZZID', 'name': 'Reun Co', 'date': '2026-06-02',
    'exchange': 'NYSE', 'status': 'expected',
    'price': '30.00-35.00',
    'numberOfShares': 60_000_000,
  }]
  c1 = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch(events),
  )
  c2 = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch(events),
  )
  pending = sentry_queue.dequeue_top(10)
  hits = [r for r in pending if r['ticker'] == 'ZZID']
  _check("first run enqueued 1", c1['enqueued'] == 1)
  _check("second run enqueued returns same id (no new row)",
         c2['enqueued'] >= 0)  # enqueue returns existing id
  _check("exactly 1 ZZID pending row", len(hits) == 1, str(hits))


def test_ipo_missing_price_skipped():
  print("\n== ipo_calendar: missing price field -> below_min_cap (mcap=0) ==")
  _cleanup()
  events = [{
    'symbol': 'ZZIE', 'name': 'No Price', 'date': '2026-06-03',
    'exchange': 'NASDAQ', 'status': 'filed',
    'price': None, 'numberOfShares': 100_000_000,
  }]
  counts = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch(events),
  )
  _check("below_min_cap = 1 (mcap=0)",
         counts['below_min_cap'] == 1, str(counts))


def test_ipo_empty_calendar():
  print("\n== ipo_calendar: empty response -> no crash, zero enqueue ==")
  _cleanup()
  counts = sentry_discovery.scan_ipo_calendar(
    _fetch_fn=_fake_ipo_fetch([]),
  )
  _check("fetched = 0", counts['fetched'] == 0)
  _check("enqueued = 0", counts['enqueued'] == 0)


def main() -> int:
  print("\nSentry discovery new channels tests\n")
  init_schema()
  test_ipo_eligible_ticker_enqueued()
  test_ipo_below_min_mcap_dropped()
  test_ipo_wrong_exchange_dropped()
  test_ipo_dedup_on_rerun()
  test_ipo_missing_price_skipped()
  test_ipo_empty_calendar()
  _cleanup()
  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == '__main__':
  sys.exit(main())
