"""Sentry discovery channels — proactive candidate generation.

Three channels run once per day (06:00 ET) and write candidates to
sentry_queue alongside the event-reactive ones from sentry_triage:

  1. catalyst_calendar — earnings within 5 days for watched tickers.
     Highest signal of the three. Triggered_by='pre_earnings_5d',
     score 0.75 (forces queue priority).

  2. insider_cluster — 3+ distinct insiders open-market buying $100k+
     in last 30 days. Triggered_by='insider_cluster'.

  3. theme_flow — ETF AUM growth >5% week-over-week with top-holding
     shifts. New top holdings get triggered_by='theme_flow'.

This module never invokes Claude — it just populates the queue.
Reasoning happens in /sentry-tick.

Run:
  python -m daemons.sentry_discovery --once               # run all 3 channels, exit
  python -m daemons.sentry_discovery --once --channel catalyst_calendar
  python -m daemons.sentry_discovery --once --channel insider_cluster
  python -m daemons.sentry_discovery --once --channel theme_flow
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection, get_watchlist
from state import sentry_queue
from state.sentry_eval_log import should_skip


# -- Tuning constants -------------------------------------------------------
PRE_EARNINGS_LOOKAHEAD_DAYS = 5
PRE_EARNINGS_SCORE = 0.75              # high — forces queue priority for watched names
INSIDER_CLUSTER_DAYS = 30
INSIDER_CLUSTER_MIN_INSIDERS = 3
INSIDER_CLUSTER_MIN_TX_USD = 100_000   # per-insider open-market buy minimum
INSIDER_CLUSTER_SCORE = 0.72           # strong but slightly below pre-earnings
ACTIVIST_13D_SCORE = 0.78              # activist filings are very high signal
THEME_FLOW_LOOKBACK_DAYS = 7
THEME_FLOW_MIN_AUM_GROWTH_PCT = 5.0
THEME_FLOW_SCORE = 0.65                # mid signal — themes move slowly
DEFAULT_THEMES = [
  'AI semis', 'cloud', 'cybersecurity', 'clean energy',
  'biotech', 'fintech', 'defense', 'energy', 'banks',
]

# IPO calendar tuning
IPO_LOOKAHEAD_DAYS = 10
IPO_MIN_MKT_CAP_USD = 1_000_000_000    # $1B floor to filter micro-cap noise
IPO_SCORE = 0.70                       # high — new listings are decay-fresh signal
IPO_VALID_EXCHANGES = {'NASDAQ', 'NYSE', 'NYSE ARCA', 'NYSE AMERICAN'}

# Universe insider cluster tuning
UNIVERSE_INSIDER_SCORE = 0.68          # below watchlist version (less context)
UNIVERSE_INSIDER_MSPR_MIN = 0.3        # MSPR threshold for the pre-filter
UNIVERSE_INSIDER_MAX_DEEP_DIVES = 50   # hard cap on transactions calls per run
UNIVERSE_INSIDER_PACE_S = 1.0          # sleep between Finnhub calls (60 RPM ceiling)


# ============================================================================
# Channel 1: catalyst calendar (pre-earnings deep-context trigger)
# ============================================================================

def scan_catalyst_calendar(watchlist: Optional[List[str]] = None) -> Dict[str, int]:
  """Enqueue watched tickers with earnings in the next 5 trading days.

  /sentry-tick recognizes triggered_by='pre_earnings_5d' and runs the
  pre-earnings deep-context bundle (extract_8k_events + insider_transactions
  + 13D + revisions + options + cross-company-readthrough).

  Returns counts dict for logging.
  """
  from tools.news_agregator.finnhub_utils import FinnhubClient

  counts = {'fetched': 0, 'in_watchlist': 0, 'enqueued': 0, 'skipped': 0}

  if watchlist is None:
    # get_watchlist() returns plain ticker strings, not dicts
    watchlist = [t.upper() for t in get_watchlist()]
  watchlist_set = set(watchlist)

  if not watchlist_set:
    return counts

  today = datetime.now(timezone.utc).date()
  from_date = today.isoformat()
  to_date = (today + timedelta(days=PRE_EARNINGS_LOOKAHEAD_DAYS)).isoformat()

  async def _fetch():
    client = FinnhubClient()
    try:
      return await client.get('/calendar/earnings', {'from': from_date, 'to': to_date})
    finally:
      await client.close()

  try:
    raw = asyncio.run(_fetch())
  except Exception as exc:
    print(f"[discovery.catalyst_calendar] fetch error: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)
    return counts

  entries = raw.get('earningsCalendar', []) if isinstance(raw, dict) else []
  counts['fetched'] = len(entries)

  for entry in entries:
    ticker = (entry.get('symbol') or '').upper()
    if not ticker or ticker not in watchlist_set:
      continue
    counts['in_watchlist'] += 1

    # Cooldown — even pre-earnings respects recent verdicts
    skip, _ = should_skip(ticker)
    if skip:
      counts['skipped'] += 1
      continue

    earnings_date = entry.get('date', '?')
    qid = sentry_queue.enqueue(
      ticker, score=PRE_EARNINGS_SCORE, triggered_by='pre_earnings_5d',
      notes=f"earnings on {earnings_date}, EPS est {entry.get('epsEstimate', '?')}",
    )
    if qid:
      counts['enqueued'] += 1

  return counts


# ============================================================================
# Channel 2: insider + 13D cluster
# ============================================================================

def _detect_insider_cluster(insider_data: Dict[str, Any]) -> bool:
  """Return True if 3+ distinct insiders made open-market buys >$100k in
  the last 30 days. `insider_data` is Finnhub's condensed shape from
  /stock/insider-transactions."""
  if not isinstance(insider_data, dict):
    return False

  txns = insider_data.get('data') or insider_data.get('transactions') or []
  if not txns:
    return False

  cutoff = (datetime.now(timezone.utc) - timedelta(days=INSIDER_CLUSTER_DAYS)).date()
  open_market_buyers: set[str] = set()

  for tx in txns:
    if not isinstance(tx, dict):
      continue
    # Finnhub schema: transactionDate, transactionCode (P=open-market purchase),
    # transactionPrice, change (shares — negative = sale), name (insider)
    code = (tx.get('transactionCode') or '').upper()
    if code != 'P':                       # P = open-market purchase
      continue
    change = tx.get('change') or 0
    price = tx.get('transactionPrice') or 0
    if change <= 0 or price <= 0:         # sales or zero-value rows
      continue
    usd = float(change) * float(price)
    if usd < INSIDER_CLUSTER_MIN_TX_USD:
      continue
    tx_date_str = tx.get('transactionDate') or ''
    try:
      tx_date = datetime.fromisoformat(tx_date_str[:10]).date()
    except ValueError:
      continue
    if tx_date < cutoff:
      continue
    insider_name = (tx.get('name') or '').strip()
    if insider_name:
      open_market_buyers.add(insider_name)

  return len(open_market_buyers) >= INSIDER_CLUSTER_MIN_INSIDERS


def scan_insider_cluster(watchlist: Optional[List[str]] = None) -> Dict[str, int]:
  """For each watched ticker, detect insider cluster buying. Separately,
  scan recent SC 13D filings for activist activity — any 13D on a
  non-watched ticker is a discovery candidate."""
  from tools.news_agregator.finnhub_utils import FinnhubClient

  counts = {
    'watchlist_scanned': 0, 'insider_clusters': 0,
    'activist_13d_scanned': 0, 'activist_enqueued': 0,
    'skipped': 0,
  }

  if watchlist is None:
    # get_watchlist() returns plain ticker strings, not dicts
    watchlist = [t.upper() for t in get_watchlist()]

  # -- 2a. Insider cluster scan on watchlist --
  async def _fetch_insider(ticker: str):
    client = FinnhubClient()
    try:
      raw = await client.get('/stock/insider-transactions', {'symbol': ticker})
      return raw
    finally:
      await client.close()

  for ticker in watchlist:
    counts['watchlist_scanned'] += 1
    try:
      raw = asyncio.run(_fetch_insider(ticker))
    except Exception as exc:
      print(f"[discovery.insider_cluster] {ticker} fetch error: {exc}",
            file=sys.stderr, flush=True)
      continue

    if not _detect_insider_cluster(raw):
      continue

    counts['insider_clusters'] += 1
    skip, _ = should_skip(ticker)
    if skip:
      counts['skipped'] += 1
      continue

    qid = sentry_queue.enqueue(
      ticker, score=INSIDER_CLUSTER_SCORE, triggered_by='insider_cluster',
      notes=f"3+ insider open-market buys >${INSIDER_CLUSTER_MIN_TX_USD:,} in {INSIDER_CLUSTER_DAYS}d",
    )

  # -- 2b. Activist 13D scan (broader — discovery beyond watchlist) --
  # We scan recent 13D filings via the SEC sec_utils helper. The function
  # takes a ticker, but we want to find ANY recent 13D activity. So we
  # scan a watchlist of common activist-targeted names (small/mid caps,
  # underperformers) plus the user's existing watchlist. For v1, just
  # use the user's watchlist.
  from tools.web_search_server.sec_utils import get_schedule_13d_filings

  for ticker in watchlist:
    counts['activist_13d_scanned'] += 1
    try:
      result = get_schedule_13d_filings(ticker, limit=10, include_passive=False)
    except Exception as exc:
      print(f"[discovery.insider_cluster] {ticker} 13D fetch error: {exc}",
            file=sys.stderr, flush=True)
      continue

    if not (isinstance(result, dict) and result.get('success')):
      continue
    filings = result.get('filings') or []
    # Only count activist filings filed in the last 30 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date()
    recent_activist = []
    for f in filings:
      if not f.get('is_activist'):
        continue
      try:
        fd = datetime.fromisoformat(str(f.get('filing_date'))[:10]).date()
      except (ValueError, TypeError):
        continue
      if fd >= cutoff:
        recent_activist.append(f)

    if not recent_activist:
      continue

    skip, _ = should_skip(ticker)
    if skip:
      counts['skipped'] += 1
      continue

    most_recent = recent_activist[0]
    filer = most_recent.get('filer_name', 'unknown')
    qid = sentry_queue.enqueue(
      ticker, score=ACTIVIST_13D_SCORE, triggered_by='activist_13d',
      notes=f"activist 13D from {filer} on {most_recent.get('filing_date')}",
    )
    if qid:
      counts['activist_enqueued'] += 1

  return counts


# ============================================================================
# Channel 3: theme-flow (ETF AUM rotation detection)
# ============================================================================

def _snapshot_etf(etf_symbol: str, theme: str, total_assets: float,
                  top_holdings: List[Dict[str, Any]], snapshot_date: str) -> None:
  """Insert today's ETF snapshot. Idempotent via the unique (etf, date) index."""
  conn = get_connection()
  try:
    conn.execute(
      """INSERT OR REPLACE INTO etf_aum_history
         (etf_symbol, snapshot_date, theme, total_assets, top_holdings, captured_at)
         VALUES (?, ?, ?, ?, ?, ?)""",
      (
        etf_symbol.upper(), snapshot_date, theme,
        float(total_assets) if total_assets else 0.0,
        json.dumps(top_holdings),
        datetime.now(timezone.utc).isoformat(),
      ),
    )
    conn.commit()
  finally:
    conn.close()


def _previous_snapshot(etf_symbol: str, days_ago: int) -> Optional[Dict[str, Any]]:
  """Return the snapshot from N days ago for this ETF, or None if missing."""
  target_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).date().isoformat()
  conn = get_connection()
  try:
    cur = conn.execute(
      """SELECT etf_symbol, snapshot_date, theme, total_assets, top_holdings
         FROM etf_aum_history
         WHERE etf_symbol = ? AND snapshot_date <= ?
         ORDER BY snapshot_date DESC LIMIT 1""",
      (etf_symbol.upper(), target_date),
    )
    row = cur.fetchone()
    if not row:
      return None
    d = dict(row)
    if d.get('top_holdings'):
      try:
        d['top_holdings'] = json.loads(d['top_holdings'])
      except (json.JSONDecodeError, TypeError):
        d['top_holdings'] = []
    return d
  finally:
    conn.close()


def scan_theme_flow(themes: Optional[List[str]] = None) -> Dict[str, int]:
  """Daily snapshot each theme ETF, compare to 7-day-ago snapshot. If AUM
  grew >5% AND a new ticker appeared in top holdings (vs the prior
  week), enqueue the new ticker as a theme_flow candidate."""
  from tools.financial_modeling_engine.utils import get_industry_etfs

  if themes is None:
    themes = DEFAULT_THEMES

  counts = {
    'themes_scanned': 0, 'etfs_snapshotted': 0,
    'etfs_with_growth': 0, 'enqueued': 0, 'skipped': 0,
  }

  today = datetime.now(timezone.utc).date().isoformat()

  for theme in themes:
    counts['themes_scanned'] += 1
    try:
      result = get_industry_etfs(theme, top_holdings_per_etf=10)
    except Exception as exc:
      print(f"[discovery.theme_flow] {theme} fetch error: {exc}",
            file=sys.stderr, flush=True)
      continue

    if not (isinstance(result, dict) and result.get('success')):
      continue
    etfs = result.get('etfs') or []

    for etf in etfs:
      etf_sym = etf.get('symbol')
      if not etf_sym:
        continue
      total_assets = etf.get('total_assets', 0)
      top_holdings = etf.get('top_holdings') or []

      # Always snapshot today's state (idempotent)
      _snapshot_etf(etf_sym, theme, total_assets, top_holdings, today)
      counts['etfs_snapshotted'] += 1

      # Compare to 7-day-ago snapshot
      prior = _previous_snapshot(etf_sym, THEME_FLOW_LOOKBACK_DAYS)
      if prior is None or not prior.get('total_assets'):
        # No prior data — first time we've seen this ETF. Skip flow analysis.
        continue

      prior_aum = float(prior['total_assets'])
      if prior_aum <= 0:
        continue
      growth_pct = (float(total_assets) - prior_aum) / prior_aum * 100.0

      if growth_pct < THEME_FLOW_MIN_AUM_GROWTH_PCT:
        continue
      counts['etfs_with_growth'] += 1

      # Top-holding shift: any ticker in today's top 10 that wasn't in
      # last week's top 10 is a "fresh entrant" — surface it.
      prior_tickers = {h.get('symbol', '').upper() for h in (prior.get('top_holdings') or [])}
      today_tickers = {h.get('symbol', '').upper() for h in top_holdings}
      new_entrants = today_tickers - prior_tickers - {''}

      for ticker in new_entrants:
        # Skip non-US tickers (contain '.' like '6861.T') — they're not in
        # our scope for paper trading via Alpaca
        if '.' in ticker:
          continue

        skip, _ = should_skip(ticker)
        if skip:
          counts['skipped'] += 1
          continue

        qid = sentry_queue.enqueue(
          ticker, score=THEME_FLOW_SCORE, triggered_by='theme_flow',
          notes=f"new top-holding in {etf_sym} ({theme}), ETF AUM +{growth_pct:.1f}% over {THEME_FLOW_LOOKBACK_DAYS}d",
        )
        if qid:
          counts['enqueued'] += 1

  return counts


# ============================================================================
# Channel 4: universe-wide insider cluster
# ============================================================================

def scan_universe_insider_cluster(
  universe: Optional[List[str]] = None,
  max_deep_dives: int = UNIVERSE_INSIDER_MAX_DEEP_DIVES,
  _fetch_sentiment_fn=None,
  _fetch_transactions_fn=None,
) -> Dict[str, int]:
  """Universe-wide insider cluster discovery.

  Two-stage filter to fit inside the Finnhub free-tier rate budget:
    1. **Fast MSPR pre-filter**: for each ticker in the universe, call
       /stock/insider-sentiment (cheap, one row per month). Keep
       tickers where avg MSPR over recent months exceeds the threshold.
    2. **Bounded deep-dive**: for the top N MSPR-positive tickers
       (default 50), call /stock/insider-transactions and apply the
       same 3+ insiders / >$100k / 30d cluster heuristic as the
       watchlist version. Enqueue clusters with score
       UNIVERSE_INSIDER_SCORE (0.68 vs 0.72 watchlist).

  Both Finnhub call sites pace at UNIVERSE_INSIDER_PACE_S (1.0s) to
  stay under the 60 RPM ceiling. With ~2000 universe tickers the
  pre-filter takes ~33 minutes; the deep-dive adds ~1 minute. Runs
  async in the daemon, does not block triage ticks.

  Args:
    universe: ticker list (default: read from sentry_universe table).
    max_deep_dives: cap on transactions-call count after pre-filter.
    _fetch_sentiment_fn / _fetch_transactions_fn: test-only injection.
  """
  import time as _t
  from tools.news_agregator.finnhub_utils import FinnhubClient

  counts = {
    'universe_size': 0,
    'sentiment_scanned': 0,
    'sentiment_positive': 0,
    'sentiment_errors': 0,
    'deep_dives': 0,
    'clusters_detected': 0,
    'enqueued': 0,
    'skipped': 0,
  }

  if universe is None:
    # Defer the import so we don't form a circular reference at module load
    try:
      from daemons.sentry_universe import get_universe
      universe = get_universe()
    except Exception as exc:
      print(f"[discovery.universe_insider] universe load failed: {exc}",
            file=sys.stderr, flush=True)
      return counts

  counts['universe_size'] = len(universe)
  if not universe:
    return counts

  # -- Stage 1: MSPR pre-filter ----------------------------------------
  async def _default_sentiment(ticker: str):
    client = FinnhubClient()
    try:
      return await client.get('/stock/insider-sentiment', {'symbol': ticker})
    finally:
      await client.close()

  fetch_sent = _fetch_sentiment_fn or _default_sentiment

  candidates = []  # tickers passing the MSPR threshold, ordered by avg_mspr desc
  for ticker in universe:
    counts['sentiment_scanned'] += 1
    try:
      raw = asyncio.run(fetch_sent(ticker))
    except Exception:
      counts['sentiment_errors'] += 1
      continue
    if not isinstance(raw, dict):
      continue
    data = raw.get('data') or []
    if not data:
      continue
    # Compute avg MSPR over last 6 months (mirror the condense helper)
    sorted_data = sorted(data, key=lambda x: (x.get('year', 0), x.get('month', 0)),
                          reverse=True)[:6]
    msprs = [r.get('mspr') for r in sorted_data if r.get('mspr') is not None]
    if not msprs:
      continue
    avg_mspr = sum(msprs) / len(msprs)
    if avg_mspr >= UNIVERSE_INSIDER_MSPR_MIN:
      counts['sentiment_positive'] += 1
      candidates.append((ticker, avg_mspr))
    if _fetch_sentiment_fn is None:
      _t.sleep(UNIVERSE_INSIDER_PACE_S)

  candidates.sort(key=lambda x: x[1], reverse=True)
  top_candidates = candidates[:max_deep_dives]

  # -- Stage 2: bounded deep-dive --------------------------------------
  async def _default_txns(ticker: str):
    client = FinnhubClient()
    try:
      return await client.get('/stock/insider-transactions', {'symbol': ticker})
    finally:
      await client.close()

  fetch_txns = _fetch_transactions_fn or _default_txns

  for ticker, avg_mspr in top_candidates:
    counts['deep_dives'] += 1
    try:
      raw = asyncio.run(fetch_txns(ticker))
    except Exception:
      continue
    if not _detect_insider_cluster(raw):
      continue
    counts['clusters_detected'] += 1

    skip, _ = should_skip(ticker)
    if skip:
      counts['skipped'] += 1
      continue

    qid = sentry_queue.enqueue(
      ticker, score=UNIVERSE_INSIDER_SCORE,
      triggered_by='universe_insider_cluster',
      notes=f"avg_mspr={avg_mspr:.3f} over 6mo; >=3 insider buys >$100k/30d",
    )
    if qid:
      counts['enqueued'] += 1
    if _fetch_transactions_fn is None:
      _t.sleep(UNIVERSE_INSIDER_PACE_S)

  return counts


# ============================================================================
# Channel 5: IPO / new listing calendar
# ============================================================================

def scan_ipo_calendar(lookahead_days: int = IPO_LOOKAHEAD_DAYS,
                      min_mkt_cap_usd: float = IPO_MIN_MKT_CAP_USD,
                      _fetch_fn=None) -> Dict[str, int]:
  """Enqueue upcoming IPOs above a market-cap floor.

  IPO calendar is fetched from Finnhub /calendar/ipo via FinnhubClient.
  Filters: US exchange, expected market cap >= min_mkt_cap_usd. Same IPO
  surfaces every day until its date passes — relies on sentry_queue
  unique partial index to keep one pending row per ticker.

  Args:
    lookahead_days: how many days forward to scan (default 10)
    min_mkt_cap_usd: skip listings below this expected market cap
    _fetch_fn: test-only injection; async callable taking (from_date,
               to_date) and returning the Finnhub-shaped dict
  """
  from tools.news_agregator.finnhub_utils import FinnhubClient

  counts = {
    'fetched': 0, 'eligible': 0, 'below_min_cap': 0,
    'wrong_exchange': 0, 'enqueued': 0, 'skipped': 0,
  }

  today = datetime.now(timezone.utc).date()
  from_date = today.isoformat()
  to_date = (today + timedelta(days=lookahead_days)).isoformat()

  async def _fetch():
    if _fetch_fn is not None:
      return await _fetch_fn(from_date, to_date)
    client = FinnhubClient()
    try:
      return await client.get('/calendar/ipo', {'from': from_date, 'to': to_date})
    finally:
      await client.close()

  try:
    raw = asyncio.run(_fetch())
  except Exception as exc:
    print(f"[discovery.ipo_calendar] fetch error: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)
    return counts

  entries = raw.get('ipoCalendar', []) if isinstance(raw, dict) else []
  counts['fetched'] = len(entries)

  def _parse_price_mid(p):
    if not p:
      return None
    s = str(p)
    try:
      if '-' in s:
        lo, hi = s.split('-', 1)
        return (float(lo) + float(hi)) / 2.0
      return float(s)
    except (ValueError, TypeError):
      return None

  for entry in entries:
    ticker = (entry.get('symbol') or '').upper()
    if not ticker:
      continue

    exchange = (entry.get('exchange') or '').upper()
    if exchange not in IPO_VALID_EXCHANGES:
      counts['wrong_exchange'] += 1
      continue

    shares = entry.get('numberOfShares') or 0
    price_mid = _parse_price_mid(entry.get('price'))
    expected_mcap = (price_mid * shares) if (price_mid and shares) else 0
    if expected_mcap < min_mkt_cap_usd:
      counts['below_min_cap'] += 1
      continue

    counts['eligible'] += 1

    skip, _ = should_skip(ticker)
    if skip:
      counts['skipped'] += 1
      continue

    ipo_date = entry.get('date', '?')
    qid = sentry_queue.enqueue(
      ticker, score=IPO_SCORE, triggered_by='ipo_listing',
      notes=(
        f"IPO {ipo_date} on {exchange}; "
        f"price ~${price_mid:.2f}; shares {shares:,}; "
        f"expected mcap ${expected_mcap/1e9:.2f}B"
      ),
    )
    if qid:
      counts['enqueued'] += 1

  return counts


# ============================================================================
# Orchestration
# ============================================================================

def run_all() -> Dict[str, Dict[str, int]]:
  """Run all daily channels in sequence and return per-channel counts."""
  print(f"[discovery] starting full scan at {datetime.now(timezone.utc).isoformat()}",
        file=sys.stderr, flush=True)
  results = {}

  print("[discovery] channel 1/5: catalyst_calendar", file=sys.stderr, flush=True)
  results['catalyst_calendar'] = scan_catalyst_calendar()
  print(f"  -> {results['catalyst_calendar']}", file=sys.stderr, flush=True)

  print("[discovery] channel 2/5: insider_cluster", file=sys.stderr, flush=True)
  results['insider_cluster'] = scan_insider_cluster()
  print(f"  -> {results['insider_cluster']}", file=sys.stderr, flush=True)

  print("[discovery] channel 3/5: theme_flow", file=sys.stderr, flush=True)
  results['theme_flow'] = scan_theme_flow()
  print(f"  -> {results['theme_flow']}", file=sys.stderr, flush=True)

  print("[discovery] channel 4/5: ipo_calendar", file=sys.stderr, flush=True)
  results['ipo_calendar'] = scan_ipo_calendar()
  print(f"  -> {results['ipo_calendar']}", file=sys.stderr, flush=True)

  print("[discovery] channel 5/5: universe_insider_cluster", file=sys.stderr, flush=True)
  results['universe_insider_cluster'] = scan_universe_insider_cluster()
  print(f"  -> {results['universe_insider_cluster']}", file=sys.stderr, flush=True)

  print(f"[discovery] full scan complete at {datetime.now(timezone.utc).isoformat()}",
        file=sys.stderr, flush=True)
  return results


def main():
  parser = argparse.ArgumentParser(description='Sentry discovery channels')
  parser.add_argument('--once', action='store_true', required=True,
                      help='Run once and exit (no continuous mode for discovery)')
  parser.add_argument('--channel', choices=['catalyst_calendar', 'insider_cluster',
                                             'theme_flow', 'ipo_calendar',
                                             'universe_insider_cluster', 'all'],
                      default='all', help='Run a specific channel or all')
  args = parser.parse_args()

  init_schema()

  if args.channel == 'all':
    run_all()
  elif args.channel == 'catalyst_calendar':
    print(scan_catalyst_calendar(), file=sys.stderr, flush=True)
  elif args.channel == 'insider_cluster':
    print(scan_insider_cluster(), file=sys.stderr, flush=True)
  elif args.channel == 'theme_flow':
    print(scan_theme_flow(), file=sys.stderr, flush=True)
  elif args.channel == 'ipo_calendar':
    print(scan_ipo_calendar(), file=sys.stderr, flush=True)
  elif args.channel == 'universe_insider_cluster':
    print(scan_universe_insider_cluster(), file=sys.stderr, flush=True)


if __name__ == '__main__':
  main()
