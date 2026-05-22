"""Unit tests for daemons/sentry_universe.py.

Uses monkey-patched get_industry_etfs to inject synthetic ETF holdings;
no live yfinance calls. Test tickers are 4-letter codes starting with
ZZ so they pass the _is_us_ticker filter AND don't collide with real
US listings.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from daemons import sentry_universe


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name, condition, hint=''):
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


# Test tickers all start with ZZ — short, pure alpha, won't collide with
# real US listings. Cleanup targets WHERE ticker LIKE 'ZZ%'.
_TEST_TICKERS = ['ZZAA', 'ZZAB', 'ZZAC', 'ZZAD', 'ZZAE', 'ZZAF', 'ZZAG']


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM sentry_universe WHERE ticker LIKE 'ZZ%'")
    conn.commit()
  finally:
    conn.close()


def _stub_get_industry_etfs(themes_to_holdings):
  """Return a stub that maps theme query -> synthetic ETF result."""
  def stub(theme, top_holdings_per_etf=25):
    holdings = themes_to_holdings.get(theme, [])
    if not holdings:
      return {'success': False, 'etfs': []}
    return {
      'success': True,
      'etfs': [{
        'symbol': 'XXX',
        'top_holdings': [{'symbol': h, 'weight_pct': 5.0} for h in holdings],
      }]
    }
  return stub


def _with_stubs(theme_data):
  """Context manager swap of theme map + get_industry_etfs."""
  saved_map = sentry_universe._THEME_TO_ETFS
  saved_fn = sentry_universe.get_industry_etfs
  saved_sleep = sentry_universe.PER_THEME_SLEEP_S
  sentry_universe._THEME_TO_ETFS = {k: ['STUB'] for k in theme_data}
  sentry_universe.get_industry_etfs = _stub_get_industry_etfs(theme_data)
  sentry_universe.PER_THEME_SLEEP_S = 0
  return saved_map, saved_fn, saved_sleep


def _restore_stubs(saved):
  saved_map, saved_fn, saved_sleep = saved
  sentry_universe._THEME_TO_ETFS = saved_map
  sentry_universe.get_industry_etfs = saved_fn
  sentry_universe.PER_THEME_SLEEP_S = saved_sleep


def test_refresh_inserts_universe_rows():
  print("\n== refresh from synthetic ETFs -> universe rows inserted ==")
  _cleanup()

  themes_data = {
    'semis':  ['ZZAA', 'ZZAB', 'ZZAC'],
    'cloud':  ['ZZAD', 'ZZAE'],
    'energy': ['ZZAF'],
  }
  saved = _with_stubs(themes_data)
  try:
    counts = sentry_universe.refresh_universe(force=True)
    _check("refreshed = True", counts['refreshed'] is True, str(counts))
    _check("tickers_inserted == 6",
           counts['tickers_inserted'] == 6, str(counts))
    conn = get_connection()
    rows = conn.execute(
      "SELECT ticker FROM sentry_universe WHERE ticker LIKE 'ZZ%'"
    ).fetchall()
    conn.close()
    _check("6 universe rows landed", len(rows) == 6,
           f"got {len(rows)}: {[r['ticker'] for r in rows]}")
  finally:
    _restore_stubs(saved)


def test_staleness_gate_skips_recent_refresh():
  print("\n== refresh within stale_days -> skipped ==")
  _cleanup()

  # Insert a recent-refreshed row so MAX(refreshed_at) is fresh
  conn = get_connection()
  try:
    conn.execute(
      """INSERT INTO sentry_universe
         (ticker, last_seen_in_themes, first_seen_at, refreshed_at)
         VALUES ('ZZAG', '[]', ?, ?)""",
      (datetime.now(timezone.utc).isoformat(),
       datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
  finally:
    conn.close()

  saved_fn = sentry_universe.get_industry_etfs
  called = {'count': 0}
  def fail_if_called(theme, top_holdings_per_etf=25):
    called['count'] += 1
    return {'success': False, 'etfs': []}
  sentry_universe.get_industry_etfs = fail_if_called
  try:
    counts = sentry_universe.refresh_universe(stale_days=7, force=False)
    _check("refreshed = False (staleness gate)",
           counts['refreshed'] is False, str(counts))
    _check("get_industry_etfs was NOT called",
           called['count'] == 0, f"called {called['count']} times")
  finally:
    sentry_universe.get_industry_etfs = saved_fn


def test_force_bypasses_staleness():
  print("\n== --force bypasses staleness gate ==")
  _cleanup()

  themes_data = {'tech': ['ZZAA']}
  saved = _with_stubs(themes_data)
  try:
    sentry_universe.refresh_universe(force=True)  # first run
    counts = sentry_universe.refresh_universe(stale_days=30, force=True)
    _check("refreshed = True under force",
           counts['refreshed'] is True, str(counts))
  finally:
    _restore_stubs(saved)


def test_excluded_rows_preserved_on_refresh():
  print("\n== excluded=1 rows preserved through refresh ==")
  _cleanup()

  conn = get_connection()
  try:
    conn.execute(
      """INSERT INTO sentry_universe
         (ticker, last_seen_in_themes, first_seen_at, refreshed_at, excluded, excluded_reason)
         VALUES ('ZZAB', '[]', ?, ?, 1, 'manual exclude for test')""",
      (datetime.now(timezone.utc).isoformat(),
       datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
  finally:
    conn.close()

  themes_data = {'banks': ['ZZAB']}
  saved = _with_stubs(themes_data)
  try:
    sentry_universe.refresh_universe(force=True)
    conn = get_connection()
    row = conn.execute(
      "SELECT excluded, excluded_reason FROM sentry_universe WHERE ticker = 'ZZAB'"
    ).fetchone()
    conn.close()
    _check("excluded flag preserved",
           row is not None and row['excluded'] == 1,
           f"got {dict(row) if row else None}")
    _check("excluded_reason preserved",
           row['excluded_reason'] == 'manual exclude for test')
  finally:
    _restore_stubs(saved)


def test_non_us_tickers_filtered():
  print("\n== non-US ticker symbols filtered ==")
  _cleanup()

  themes_data = {
    'global': ['ZZAC', '7203.T', 'BRK-B', 'TOOLONG', 'lower'],
  }
  saved = _with_stubs(themes_data)
  try:
    sentry_universe.refresh_universe(force=True)
    conn = get_connection()
    rows = [r['ticker'] for r in conn.execute(
      "SELECT ticker FROM sentry_universe WHERE ticker IN ('ZZAC', '7203.T', 'BRK-B', 'TOOLONG', 'lower')"
    ).fetchall()]
    conn.close()
    _check("ZZAC landed (valid US)", 'ZZAC' in rows, str(rows))
    _check("7203.T filtered (dot suffix)", '7203.T' not in rows)
    _check("BRK-B filtered (dash)", 'BRK-B' not in rows)
    _check("TOOLONG (7 chars) filtered", 'TOOLONG' not in rows)
    _check("lower filtered (lowercase)", 'lower' not in rows)
  finally:
    _restore_stubs(saved)


def test_get_universe_excludes_flagged():
  print("\n== get_universe() filters excluded rows by default ==")
  _cleanup()

  conn = get_connection()
  try:
    conn.execute(
      """INSERT INTO sentry_universe (ticker, last_seen_in_themes, refreshed_at, excluded)
         VALUES ('ZZAD', '[]', ?, 0), ('ZZAE', '[]', ?, 1)""",
      (datetime.now(timezone.utc).isoformat(),
       datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
  finally:
    conn.close()

  active = sentry_universe.get_universe()
  _check("ZZAD in active", 'ZZAD' in active)
  _check("ZZAE NOT in active", 'ZZAE' not in active)
  full = sentry_universe.get_universe(include_excluded=True)
  _check("ZZAE IS in include_excluded=True", 'ZZAE' in full)


def main() -> int:
  print("\nSentry universe builder tests\n")
  init_schema()
  test_refresh_inserts_universe_rows()
  test_staleness_gate_skips_recent_refresh()
  test_force_bypasses_staleness()
  test_excluded_rows_preserved_on_refresh()
  test_non_us_tickers_filtered()
  test_get_universe_excludes_flagged()
  _cleanup()
  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == '__main__':
  sys.exit(main())
