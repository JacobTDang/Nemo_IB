"""Sentry universe builder -- maintains the `sentry_universe` table as the
union of top holdings across all curated theme ETFs.

Downstream discovery channels (universe_insider_cluster, fundamental
screener) read from this table when they need a wider ticker set than
the watchlist. The universe is refreshed periodically (default 7 days)
since ETF compositions shift quarterly.

Usage:
  python -m daemons.sentry_universe                 # refresh if stale
  python -m daemons.sentry_universe --force         # refresh regardless of age
  python -m daemons.sentry_universe --stale-days 3  # tighter staleness gate

This is a one-shot script, NOT a long-running daemon. Discovery channels
that need the universe call `refresh_universe()` themselves before
reading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from tools.financial_modeling_engine.utils import (
    _THEME_TO_ETFS,
    get_industry_etfs,
)


DEFAULT_STALE_DAYS = 7
TOP_HOLDINGS_PER_ETF = 25
# Small pacing sleep between get_industry_etfs calls to avoid hammering
# the underlying yfinance backend.
PER_THEME_SLEEP_S = 0.5


def _is_us_ticker(symbol: str) -> bool:
  """Filter out non-US tickers. yfinance returns things like 'TSM' (ADR),
  'BABA' (ADR), and '7203.T' (foreign-listed). The dot-suffix names are
  the obvious foreign ones we want to drop."""
  if not symbol:
    return False
  if '.' in symbol or '-' in symbol:
    return False  # foreign-listed or weird share class
  if not symbol.isupper() or not symbol.isalpha():
    return False
  if len(symbol) > 5:
    return False
  return True


def _most_recent_refresh(conn) -> Optional[datetime]:
  """Most recent `refreshed_at` across the table, or None if empty."""
  row = conn.execute(
    "SELECT MAX(refreshed_at) AS ts FROM sentry_universe"
  ).fetchone()
  ts = row['ts'] if row else None
  if not ts:
    return None
  try:
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
  except (ValueError, AttributeError):
    return None


def _gather_universe(theme_keys: Optional[List[str]] = None) -> Dict[str, Set[str]]:
  """Iterate themes, collect {ticker -> set of theme keys it appeared in}.

  Returns the mapping. yfinance failures on a single theme don't abort
  the whole refresh -- they're logged and the theme is skipped.
  """
  themes = theme_keys or list(_THEME_TO_ETFS.keys())
  ticker_themes: Dict[str, Set[str]] = {}
  for i, theme in enumerate(themes):
    try:
      result = get_industry_etfs(theme, top_holdings_per_etf=TOP_HOLDINGS_PER_ETF)
    except Exception as exc:
      print(f"[sentry_universe] theme '{theme}' failed: {exc}",
            file=sys.stderr, flush=True)
      continue
    if not result.get('success'):
      continue
    for etf in result.get('etfs', []) or []:
      for holding in etf.get('top_holdings', []) or []:
        sym = (holding.get('symbol') or '').upper()
        if not _is_us_ticker(sym):
          continue
        ticker_themes.setdefault(sym, set()).add(theme)
    if i + 1 < len(themes):
      time.sleep(PER_THEME_SLEEP_S)
  return ticker_themes


def refresh_universe(stale_days: int = DEFAULT_STALE_DAYS,
                    force: bool = False) -> Dict[str, int]:
  """Refresh the sentry_universe table from the theme ETF holdings.

  If the most-recent refreshed_at is younger than stale_days AND force is
  False, returns immediately with counts={'tickers_total': N,
  'tickers_inserted': 0, 'refreshed': False}.

  Otherwise iterates every theme key, gathers holdings, upserts each
  ticker (updates last_seen_in_themes + refreshed_at; preserves
  first_seen_at + excluded flag).
  """
  init_schema()
  conn = get_connection()
  counts = {'tickers_total': 0, 'tickers_inserted': 0,
            'tickers_updated': 0, 'refreshed': False}
  try:
    last = _most_recent_refresh(conn)
    if last and not force:
      age = datetime.now(timezone.utc) - last
      if age < timedelta(days=stale_days):
        cur = conn.execute("SELECT COUNT(*) AS c FROM sentry_universe").fetchone()
        counts['tickers_total'] = cur['c']
        print(f"[sentry_universe] last refresh {age.days}d ago; under stale_days={stale_days}; skipping",
              file=sys.stderr, flush=True)
        return counts
  finally:
    conn.close()

  print(f"[sentry_universe] refreshing from {len(_THEME_TO_ETFS)} themes...",
        file=sys.stderr, flush=True)
  t0 = time.time()
  ticker_themes = _gather_universe()
  elapsed = time.time() - t0
  print(f"[sentry_universe] gathered {len(ticker_themes)} unique tickers in {elapsed:.1f}s",
        file=sys.stderr, flush=True)

  now = datetime.now(timezone.utc).isoformat()
  conn = get_connection()
  try:
    for ticker, themes in ticker_themes.items():
      themes_json = json.dumps(sorted(themes))
      # Upsert: if row exists, update last_seen_in_themes + refreshed_at
      # (preserve first_seen_at + excluded). Otherwise insert new row.
      cur = conn.execute(
        "SELECT 1 FROM sentry_universe WHERE ticker = ?", (ticker,)
      ).fetchone()
      if cur:
        conn.execute(
          """UPDATE sentry_universe
             SET last_seen_in_themes = ?, refreshed_at = ?
             WHERE ticker = ?""",
          (themes_json, now, ticker),
        )
        counts['tickers_updated'] += 1
      else:
        conn.execute(
          """INSERT INTO sentry_universe
             (ticker, last_seen_in_themes, first_seen_at, refreshed_at)
             VALUES (?, ?, ?, ?)""",
          (ticker, themes_json, now, now),
        )
        counts['tickers_inserted'] += 1
    conn.commit()
    cur = conn.execute("SELECT COUNT(*) AS c FROM sentry_universe").fetchone()
    counts['tickers_total'] = cur['c']
    counts['refreshed'] = True
  finally:
    conn.close()
  return counts


def get_universe(include_excluded: bool = False) -> List[str]:
  """Return the current universe as a list of ticker strings."""
  init_schema()
  conn = get_connection()
  try:
    if include_excluded:
      rows = conn.execute("SELECT ticker FROM sentry_universe").fetchall()
    else:
      rows = conn.execute(
        "SELECT ticker FROM sentry_universe WHERE excluded = 0"
      ).fetchall()
    return [r['ticker'] for r in rows]
  finally:
    conn.close()


def main() -> None:
  parser = argparse.ArgumentParser(description='Sentry universe refresh')
  parser.add_argument('--stale-days', type=int, default=DEFAULT_STALE_DAYS)
  parser.add_argument('--force', action='store_true',
                      help='Refresh regardless of age')
  args = parser.parse_args()
  counts = refresh_universe(stale_days=args.stale_days, force=args.force)
  print(f"[sentry_universe] result: {counts}", file=sys.stderr, flush=True)


if __name__ == '__main__':
  main()
