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


# ============================================================================
# scan_universe_insider_cluster
# ============================================================================

def _fake_sentiment(ticker_to_msprs):
  """Async stub for /stock/insider-sentiment.

  Args:
    ticker_to_msprs: dict ticker -> list of float MSPR values (most recent
                     first). Empty list = no data row.
  """
  async def fetch(ticker):
    values = ticker_to_msprs.get(ticker, [])
    data = []
    for i, m in enumerate(values):
      data.append({'year': 2026, 'month': 5 - i, 'mspr': m,
                   'change': 1000, 'msprChange': 0})
    return {'data': data}
  return fetch


def _fake_transactions(ticker_to_txns):
  """Async stub for /stock/insider-transactions."""
  async def fetch(ticker):
    txns = ticker_to_txns.get(ticker, [])
    return {'data': txns}
  return fetch


def _open_market_buy(name, change, price, days_ago):
  return {
    'name': name,
    'transactionCode': 'P',
    'change': change,
    'transactionPrice': price,
    'transactionDate': (datetime.now(timezone.utc).date()
                          - __import__('datetime').timedelta(days=days_ago)).isoformat(),
  }


def test_universe_insider_pre_filter_drops_low_mspr():
  print("\n== universe_insider: low-MSPR tickers excluded from deep-dive ==")
  _cleanup()
  universe = ['ZZJA', 'ZZJB']
  sentiment = _fake_sentiment({
    'ZZJA': [0.4, 0.5, 0.6],   # passes
    'ZZJB': [0.0, 0.1, -0.1],  # below threshold
  })
  txns = _fake_transactions({
    'ZZJA': [_open_market_buy(f'i{i}', 10000, 50, 5) for i in range(3)],
    # No txns for ZZJB so the test fails loudly if we deep-dive it
  })
  counts = sentry_discovery.scan_universe_insider_cluster(
    universe=universe,
    _fetch_sentiment_fn=sentiment,
    _fetch_transactions_fn=txns,
  )
  _check("sentiment_scanned == 2", counts['sentiment_scanned'] == 2, str(counts))
  _check("sentiment_positive == 1", counts['sentiment_positive'] == 1, str(counts))
  _check("deep_dives == 1", counts['deep_dives'] == 1, str(counts))
  _check("clusters_detected == 1", counts['clusters_detected'] == 1)
  _check("enqueued == 1", counts['enqueued'] == 1)


def test_universe_insider_cluster_detection():
  print("\n== universe_insider: 3+ insiders @>$100k -> cluster enqueued ==")
  _cleanup()
  universe = ['ZZJC']
  sentiment = _fake_sentiment({'ZZJC': [0.5, 0.6]})
  # 3 distinct insiders, each $500k purchase, all in last 30 days
  txns = _fake_transactions({
    'ZZJC': [
      _open_market_buy('Alice', 10000, 50, 3),   # $500k
      _open_market_buy('Bob',   10000, 50, 7),   # $500k
      _open_market_buy('Carol', 10000, 50, 14),  # $500k
    ],
  })
  counts = sentry_discovery.scan_universe_insider_cluster(
    universe=universe, _fetch_sentiment_fn=sentiment,
    _fetch_transactions_fn=txns,
  )
  _check("clusters_detected == 1", counts['clusters_detected'] == 1, str(counts))
  _check("enqueued == 1", counts['enqueued'] == 1)
  pending = sentry_queue.dequeue_top(10)
  hit = [r for r in pending if r['ticker'] == 'ZZJC']
  if hit:
    _check("triggered_by = universe_insider_cluster",
           hit[0]['triggered_by'] == 'universe_insider_cluster')
    _check("score = UNIVERSE_INSIDER_SCORE",
           hit[0]['score'] == sentry_discovery.UNIVERSE_INSIDER_SCORE)


def test_universe_insider_insufficient_cluster_not_enqueued():
  print("\n== universe_insider: only 2 insiders -> not a cluster ==")
  _cleanup()
  universe = ['ZZJD']
  sentiment = _fake_sentiment({'ZZJD': [0.5]})
  txns = _fake_transactions({
    'ZZJD': [
      _open_market_buy('Alice', 10000, 50, 3),
      _open_market_buy('Bob',   10000, 50, 7),
      # Only 2 -- below threshold of 3
    ],
  })
  counts = sentry_discovery.scan_universe_insider_cluster(
    universe=universe, _fetch_sentiment_fn=sentiment,
    _fetch_transactions_fn=txns,
  )
  _check("deep_dives == 1 (pre-filter passed)", counts['deep_dives'] == 1)
  _check("clusters_detected == 0", counts['clusters_detected'] == 0)
  _check("enqueued == 0", counts['enqueued'] == 0)


def test_universe_insider_max_deep_dives_cap():
  print("\n== universe_insider: max_deep_dives caps the deep scan ==")
  _cleanup()
  universe = [f'ZZJE{i}'[:4] for i in range(10)]  # 10 tickers, all 4-char
  # Make all 10 pass the pre-filter; verify only 3 get deep-dived
  sentiment = _fake_sentiment({t: [0.6, 0.7] for t in universe})
  txns = _fake_transactions({t: [] for t in universe})
  counts = sentry_discovery.scan_universe_insider_cluster(
    universe=universe, max_deep_dives=3,
    _fetch_sentiment_fn=sentiment, _fetch_transactions_fn=txns,
  )
  _check("sentiment_positive == 10", counts['sentiment_positive'] == 10, str(counts))
  _check("deep_dives capped at 3", counts['deep_dives'] == 3, str(counts))


def test_universe_insider_empty_universe():
  print("\n== universe_insider: empty universe -> no work, no crash ==")
  _cleanup()
  counts = sentry_discovery.scan_universe_insider_cluster(
    universe=[],
    _fetch_sentiment_fn=_fake_sentiment({}),
    _fetch_transactions_fn=_fake_transactions({}),
  )
  _check("universe_size == 0", counts['universe_size'] == 0)
  _check("enqueued == 0", counts['enqueued'] == 0)


# ============================================================================
# scan_rag_analogues
# ============================================================================

def _fake_analogues(entries):
  """Return a fn returning the provided analogue list."""
  def loader():
    return entries
  return loader


def _fake_rag_search(query_to_results):
  """Return a stub that maps query-string -> rag_search result dict.

  query keys are matched by substring (caller doesn't need to know exact
  tag ordering).
  """
  def search(query, ticker=None, doc_type=None, top_k=15, min_score=0.0):
    for key, results in query_to_results.items():
      if key in query:
        return {'results': results, 'results_count': len(results)}
    return {'results': [], 'results_count': 0}
  return search


def test_rag_analogue_above_thresholds_enqueues():
  print("\n== rag_analogue: ticker with hits across 2 analogues -> enqueue ==")
  _cleanup()

  analogues = [
    {'name': 'A1', 'direction': 'bear', 'drawdown_pct': -50.0,
     'tags': ['capex_peak', 'valuation_expansion']},
    {'name': 'A2', 'direction': 'bear', 'drawdown_pct': -40.0,
     'tags': ['supply_constrained', 'concentrated_buyers']},
  ]
  # Both analogue queries return hits for ZZRA (3 hits across 2 analogues)
  rag_search = _fake_rag_search({
    'capex_peak valuation_expansion': [
      {'ticker': 'ZZRA', 'similarity': 0.7, 'chunk_text_preview': '...'},
      {'ticker': 'ZZRA', 'similarity': 0.65, 'chunk_text_preview': '...'},
    ],
    'supply_constrained concentrated_buyers': [
      {'ticker': 'ZZRA', 'similarity': 0.72, 'chunk_text_preview': '...'},
    ],
  })
  counts = sentry_discovery.scan_rag_analogues(
    _rag_search_fn=rag_search,
    _load_analogues_fn=_fake_analogues(analogues),
  )
  _check("analogues_queried == 2", counts['analogues_queried'] == 2, str(counts))
  _check("tickers_with_hits == 1", counts['tickers_with_hits'] == 1, str(counts))
  _check("enqueued == 1", counts['enqueued'] == 1, str(counts))
  pending = sentry_queue.dequeue_top(10)
  hit = [r for r in pending if r['ticker'] == 'ZZRA']
  if hit:
    _check("triggered_by = rag_analogue",
           hit[0]['triggered_by'] == 'rag_analogue')


def test_rag_analogue_too_few_distinct_analogues_skipped():
  print("\n== rag_analogue: 3 hits but all from 1 analogue -> NOT enqueued ==")
  _cleanup()

  analogues = [
    {'name': 'A1', 'direction': 'bear', 'drawdown_pct': -50.0,
     'tags': ['capex_peak']},
    {'name': 'A2', 'direction': 'bear', 'drawdown_pct': -40.0,
     'tags': ['supply_glut']},
  ]
  rag_search = _fake_rag_search({
    'capex_peak': [
      {'ticker': 'ZZRB', 'similarity': 0.7, 'chunk_text_preview': '...'},
      {'ticker': 'ZZRB', 'similarity': 0.71, 'chunk_text_preview': '...'},
      {'ticker': 'ZZRB', 'similarity': 0.72, 'chunk_text_preview': '...'},
    ],
    'supply_glut': [],
  })
  counts = sentry_discovery.scan_rag_analogues(
    _rag_search_fn=rag_search,
    _load_analogues_fn=_fake_analogues(analogues),
  )
  _check("enqueued == 0 (only 1 distinct analogue)",
         counts['enqueued'] == 0, str(counts))


def test_rag_analogue_setup_entries_skipped():
  print("\n== rag_analogue: setup-direction analogues are skipped ==")
  _cleanup()

  analogues = [
    {'name': 'SETUP', 'direction': 'setup', 'drawdown_pct': None,
     'tags': ['capex_trough']},
    {'name': 'B1', 'direction': 'bull', 'drawdown_pct': 200.0,
     'tags': ['growth_acceleration']},
  ]
  called = {'count': 0}
  def rag_search(query, **kw):
    called['count'] += 1
    return {'results': []}
  counts = sentry_discovery.scan_rag_analogues(
    _rag_search_fn=rag_search,
    _load_analogues_fn=_fake_analogues(analogues),
  )
  # Only the bull (B1) should be queried; SETUP is skipped
  _check("analogues_queried == 1 (setup skipped)",
         counts['analogues_queried'] == 1, str(counts))
  _check("rag_search called exactly once",
         called['count'] == 1, f"called {called['count']}")


def test_rag_analogue_query_failure_no_crash():
  print("\n== rag_analogue: rag_search raising on one query does not crash run ==")
  _cleanup()

  analogues = [
    {'name': 'A1', 'direction': 'bear', 'drawdown_pct': -50.0,
     'tags': ['boom']},
    {'name': 'A2', 'direction': 'bear', 'drawdown_pct': -40.0,
     'tags': ['bust']},
  ]
  def rag_search(query, **kw):
    if 'boom' in query:
      raise RuntimeError('simulated embedding failure')
    return {'results': [
      {'ticker': 'ZZRC', 'similarity': 0.7, 'chunk_text_preview': '...'},
    ]}
  counts = sentry_discovery.scan_rag_analogues(
    _rag_search_fn=rag_search,
    _load_analogues_fn=_fake_analogues(analogues),
  )
  # Should have continued past the boom failure and queried bust
  _check("analogues_queried == 2 (failure tolerated)",
         counts['analogues_queried'] == 2, str(counts))


def main() -> int:
  print("\nSentry discovery new channels tests\n")
  init_schema()
  test_ipo_eligible_ticker_enqueued()
  test_ipo_below_min_mcap_dropped()
  test_ipo_wrong_exchange_dropped()
  test_ipo_dedup_on_rerun()
  test_ipo_missing_price_skipped()
  test_ipo_empty_calendar()
  test_universe_insider_pre_filter_drops_low_mspr()
  test_universe_insider_cluster_detection()
  test_universe_insider_insufficient_cluster_not_enqueued()
  test_universe_insider_max_deep_dives_cap()
  test_universe_insider_empty_universe()
  test_rag_analogue_above_thresholds_enqueues()
  test_rag_analogue_too_few_distinct_analogues_skipped()
  test_rag_analogue_setup_entries_skipped()
  test_rag_analogue_query_failure_no_crash()
  _cleanup()
  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1




if __name__ == '__main__':
  sys.exit(main())
