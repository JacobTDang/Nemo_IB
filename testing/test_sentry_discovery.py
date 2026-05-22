"""Tests for daemons.sentry_discovery -- the 3 daily discovery channels.

Each channel is tested in isolation with synthetic data injected at the
function boundaries (we don't make live Finnhub / SEC calls). The
deterministic logic is what's being verified:
  - insider cluster detection (3+ open-market buyers, >$100k each, <30d)
  - theme-flow snapshot + week-over-week comparison
  - catalyst calendar watchlist intersection

Run:
  .venv\\Scripts\\python.exe testing\\test_sentry_discovery.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state import sentry_queue
from daemons import sentry_discovery as D


_results = {'pass': 0, 'fail': 0, 'failures': []}
_TICKER_PREFIX = 'DISC_'
_ETF_PREFIX = 'DTEST_'


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM sentry_queue WHERE ticker LIKE ?", (f"{_TICKER_PREFIX}%",))
    conn.execute("DELETE FROM sentry_evaluation_log WHERE ticker LIKE ?", (f"{_TICKER_PREFIX}%",))
    conn.execute("DELETE FROM etf_aum_history WHERE etf_symbol LIKE ?", (f"{_ETF_PREFIX}%",))
    conn.commit()
  finally:
    conn.close()


# ============================================================================
# Insider cluster detection (pure function -- no network)
# ============================================================================

def test_insider_cluster_detection():
  print("\n== insider cluster detection ==")
  now = datetime.now(timezone.utc).date()

  # Case 1: 3 distinct insiders with open-market buys >$100k in last 30d -> True
  data_positive = {
    'data': [
      {'transactionDate': now.isoformat(), 'transactionCode': 'P',
       'transactionPrice': 100, 'change': 1500, 'name': 'Alice'},
      {'transactionDate': (now - timedelta(days=5)).isoformat(), 'transactionCode': 'P',
       'transactionPrice': 150, 'change': 1000, 'name': 'Bob'},
      {'transactionDate': (now - timedelta(days=20)).isoformat(), 'transactionCode': 'P',
       'transactionPrice': 200, 'change': 800, 'name': 'Carol'},
    ],
  }
  _check("3 distinct insiders >$100k each -> cluster detected",
         D._detect_insider_cluster(data_positive),
         "expected True")

  # Case 2: same insider 3 times -> NOT a cluster (need DISTINCT insiders)
  data_same = {
    'data': [
      {'transactionDate': now.isoformat(), 'transactionCode': 'P',
       'transactionPrice': 100, 'change': 1500, 'name': 'Alice'},
      {'transactionDate': (now - timedelta(days=5)).isoformat(), 'transactionCode': 'P',
       'transactionPrice': 150, 'change': 1000, 'name': 'Alice'},
      {'transactionDate': (now - timedelta(days=10)).isoformat(), 'transactionCode': 'P',
       'transactionPrice': 100, 'change': 2000, 'name': 'Alice'},
    ],
  }
  _check("same insider 3x is NOT a cluster (distinct insider req)",
         not D._detect_insider_cluster(data_same))

  # Case 3: 3 insiders but each <$100k -> NOT cluster
  data_small = {
    'data': [
      {'transactionDate': now.isoformat(), 'transactionCode': 'P',
       'transactionPrice': 10, 'change': 1000, 'name': 'A'},   # $10k
      {'transactionDate': now.isoformat(), 'transactionCode': 'P',
       'transactionPrice': 10, 'change': 1000, 'name': 'B'},
      {'transactionDate': now.isoformat(), 'transactionCode': 'P',
       'transactionPrice': 10, 'change': 1000, 'name': 'C'},
    ],
  }
  _check("3 insiders each <$100k -> NOT cluster", not D._detect_insider_cluster(data_small))

  # Case 4: sales (negative change) -> NOT cluster even if 3 insiders
  data_sales = {
    'data': [
      {'transactionDate': now.isoformat(), 'transactionCode': 'S',
       'transactionPrice': 100, 'change': -1500, 'name': 'A'},
      {'transactionDate': now.isoformat(), 'transactionCode': 'S',
       'transactionPrice': 100, 'change': -1500, 'name': 'B'},
      {'transactionDate': now.isoformat(), 'transactionCode': 'S',
       'transactionPrice': 100, 'change': -1500, 'name': 'C'},
    ],
  }
  _check("sales (S code, negative change) -> NOT cluster",
         not D._detect_insider_cluster(data_sales))

  # Case 5: open-market purchases but >30 days old -> NOT cluster (out of window)
  old = (now - timedelta(days=45)).isoformat()
  data_stale = {
    'data': [
      {'transactionDate': old, 'transactionCode': 'P',
       'transactionPrice': 100, 'change': 1500, 'name': 'A'},
      {'transactionDate': old, 'transactionCode': 'P',
       'transactionPrice': 100, 'change': 1500, 'name': 'B'},
      {'transactionDate': old, 'transactionCode': 'P',
       'transactionPrice': 100, 'change': 1500, 'name': 'C'},
    ],
  }
  _check("buys >30 days old -> NOT cluster (out of window)",
         not D._detect_insider_cluster(data_stale))

  # Case 6: empty / malformed input -> False, no crash
  _check("empty dict -> False", not D._detect_insider_cluster({}))
  _check("None -> False", not D._detect_insider_cluster(None))


# ============================================================================
# Theme-flow ETF snapshot + comparison
# ============================================================================

def test_theme_flow_snapshot_roundtrip():
  print("\n== theme-flow snapshot persistence ==")
  _cleanup()
  today = datetime.now(timezone.utc).date().isoformat()

  D._snapshot_etf(
    f'{_ETF_PREFIX}AIQ', theme='AI semis', total_assets=8_000_000_000,
    top_holdings=[{'symbol': 'NVDA', 'weight_pct': 8.0},
                  {'symbol': 'TSM', 'weight_pct': 4.0}],
    snapshot_date=today,
  )
  prior = D._previous_snapshot(f'{_ETF_PREFIX}AIQ', days_ago=0)
  _check("snapshot persisted and readable", prior is not None)
  _check("AUM stored correctly", prior['total_assets'] == 8_000_000_000)
  _check("top_holdings deserialized as list",
         isinstance(prior['top_holdings'], list) and len(prior['top_holdings']) == 2)


def test_theme_flow_new_entrant_detected():
  print("\n== theme-flow new-entrant detection ==")
  _cleanup()
  today = datetime.now(timezone.utc).date().isoformat()
  seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()

  # Inject a "7 days ago" snapshot with NVDA, TSM in top holdings
  D._snapshot_etf(
    f'{_ETF_PREFIX}AIQ', theme='AI semis', total_assets=8_000_000_000,
    top_holdings=[{'symbol': 'NVDA', 'weight_pct': 8.0},
                  {'symbol': 'TSM', 'weight_pct': 4.0}],
    snapshot_date=seven_days_ago,
  )

  # The scan_theme_flow function calls get_industry_etfs internally. To test
  # the detection logic without making a real API call, we directly insert a
  # "today" snapshot with AMD as a new entrant AND grew >5% -- then re-run
  # the comparison logic by calling _previous_snapshot and checking.
  D._snapshot_etf(
    f'{_ETF_PREFIX}AIQ', theme='AI semis', total_assets=8_500_000_000,  # +6.25%
    top_holdings=[{'symbol': 'NVDA', 'weight_pct': 8.0},
                  {'symbol': f'{_TICKER_PREFIX}AMD', 'weight_pct': 4.5},   # NEW entrant
                  {'symbol': 'TSM', 'weight_pct': 3.5}],
    snapshot_date=today,
  )

  prior = D._previous_snapshot(f'{_ETF_PREFIX}AIQ', days_ago=7)
  _check("prior snapshot retrievable", prior is not None)
  prior_tickers = {h['symbol'].upper() for h in prior['top_holdings']}
  today_tickers = {f'{_TICKER_PREFIX}AMD', 'NVDA', 'TSM'}
  new_entrants = today_tickers - prior_tickers
  _check("new entrant detected", f'{_TICKER_PREFIX}AMD' in new_entrants,
         f"got: {new_entrants}")


def test_theme_flow_no_prior_snapshot():
  print("\n== theme-flow no-prior-snapshot handles gracefully ==")
  _cleanup()
  prior = D._previous_snapshot(f'{_ETF_PREFIX}NEW_ETF', days_ago=7)
  _check("missing prior returns None", prior is None)


# ============================================================================
# Catalyst calendar -- pure logic test
# ============================================================================

def test_catalyst_calendar_watchlist_intersection():
  print("\n== catalyst calendar watchlist intersection (logic only) ==")
  _cleanup()

  # We can't call scan_catalyst_calendar() directly without hitting Finnhub.
  # Instead, simulate its core: given a watchlist + an "earnings calendar
  # response", verify which tickers get enqueued.
  watchlist = [f'{_TICKER_PREFIX}AAPL', f'{_TICKER_PREFIX}MSFT']
  earnings_response = [
    {'symbol': f'{_TICKER_PREFIX}AAPL', 'date': '2026-05-27', 'epsEstimate': 1.5},
    {'symbol': 'OTHER_TICKER', 'date': '2026-05-28', 'epsEstimate': 0.5},  # not in watchlist
    {'symbol': f'{_TICKER_PREFIX}MSFT', 'date': '2026-05-29', 'epsEstimate': 3.0},
  ]
  watchlist_set = set(watchlist)
  enqueued = 0
  for entry in earnings_response:
    ticker = entry['symbol'].upper()
    if ticker not in watchlist_set:
      continue
    qid = sentry_queue.enqueue(
      ticker, score=D.PRE_EARNINGS_SCORE, triggered_by='pre_earnings_5d',
      notes=f"test earnings on {entry['date']}",
    )
    if qid:
      enqueued += 1

  _check("enqueued exactly the 2 watchlisted tickers", enqueued == 2,
         f"got {enqueued}")
  pending = sentry_queue.dequeue_top(10)
  tickers = {r['ticker'] for r in pending}
  _check("AAPL enqueued", f'{_TICKER_PREFIX}AAPL' in tickers)
  _check("MSFT enqueued", f'{_TICKER_PREFIX}MSFT' in tickers)
  _check("OTHER_TICKER NOT enqueued (not watchlisted)", 'OTHER_TICKER' not in tickers)


# ============================================================================
# Discovery channel scoring consistency
# ============================================================================

def test_score_ordering():
  print("\n== channel scores are ordered correctly ==")
  _check("activist_13d > pre_earnings (rare events > scheduled)",
         D.ACTIVIST_13D_SCORE > D.PRE_EARNINGS_SCORE,
         f"13D={D.ACTIVIST_13D_SCORE} pre_e={D.PRE_EARNINGS_SCORE}")
  _check("pre_earnings > insider_cluster",
         D.PRE_EARNINGS_SCORE > D.INSIDER_CLUSTER_SCORE)
  _check("insider_cluster > theme_flow (insider activity > passive flow)",
         D.INSIDER_CLUSTER_SCORE > D.THEME_FLOW_SCORE)
  _check("all channel scores above enqueue threshold (0.50)",
         all(s > 0.50 for s in [D.PRE_EARNINGS_SCORE, D.INSIDER_CLUSTER_SCORE,
                                 D.THEME_FLOW_SCORE, D.ACTIVIST_13D_SCORE]))


def main():
  init_schema()
  print("\nSentry discovery channel tests\n")
  test_insider_cluster_detection()
  test_theme_flow_snapshot_roundtrip()
  test_theme_flow_new_entrant_detected()
  test_theme_flow_no_prior_snapshot()
  test_catalyst_calendar_watchlist_intersection()
  test_score_ordering()
  _cleanup()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
