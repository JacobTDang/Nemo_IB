"""Unit tests for daemons/gdelt_poller.py.

Uses synthetic GDELT JSON (no live HTTP) by injecting `fetch_fn` into
`tick()`. Tests the dedup path, flood cap, classifier None handling, and
graceful HTTP-failure behavior.

Run:
  .venv\\Scripts\\python.exe testing\\test_gdelt_poller.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import get_connection, init_schema
from daemons import gdelt_poller


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


_TEST_HEADLINE_PREFIX = 'GDELTTEST_'


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE headline LIKE ?",
                 (f"{_TEST_HEADLINE_PREFIX}%",))
    conn.commit()
  finally:
    conn.close()
  gdelt_poller._reset_flood_state()


def _gdelt_art(title: str, url: str = '', seendate: str = '20260522120000') -> Dict[str, Any]:
  """Mimic the shape GDELT 2.0 ArtList returns."""
  return {
    'title': title,
    'url': url or f'http://example.com/{title.replace(" ", "_")}',
    'seendate': seendate,
    'socialimage': '',
    'domain': 'example.com',
    'language': 'English',
  }


class _Classifier:
  """Returns a fixed MaterialityResult-shaped object."""
  def __init__(self, is_material=True, urgency='hours',
               primary_ticker='AAPL', category='earnings'):
    self.is_material = is_material
    self.urgency = urgency
    self.primary_ticker = primary_ticker
    self.category = category

  def classify(self, headline, summary, source):
    class R: pass
    R.is_material = self.is_material
    R.urgency = self.urgency
    R.primary_ticker = self.primary_ticker
    R.category = self.category
    R.affected_tickers = [self.primary_ticker] if self.primary_ticker else []
    R.directional_signal = 'neutral'
    R.one_line_reason = 'test'
    return R


class _NullClassifier:
  def classify(self, *a, **kw):
    return None


def _make_fetch_fn(articles_per_ticker):
  async def _fn(ticker, timespan):
    return articles_per_ticker.get(ticker, [])
  return _fn


def _row_count_headline_like(prefix: str) -> int:
  conn = get_connection()
  try:
    cur = conn.execute("SELECT COUNT(*) FROM events WHERE headline LIKE ?",
                       (f"{prefix}%",))
    return cur.fetchone()[0]
  finally:
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_article_stored():
  print("\n== synthetic article -> event stored ==")
  _cleanup()
  fetch_fn = _make_fetch_fn({
    'AAPL': [_gdelt_art(f'{_TEST_HEADLINE_PREFIX}AAPL earnings beat')],
  })
  counts = asyncio.run(gdelt_poller.tick(
    classifier=_Classifier(),
    watchlist_override=['AAPL'],
    fetch_fn=fetch_fn,
  ))
  _check("fetched == 1", counts['fetched'] == 1, str(counts))
  _check("new == 1", counts['new'] == 1, str(counts))
  _check("row in events", _row_count_headline_like(_TEST_HEADLINE_PREFIX) == 1)


def test_dedup_on_rerun():
  print("\n== dedup: rerun same fetch fn yields zero new ==")
  _cleanup()
  art = _gdelt_art(f'{_TEST_HEADLINE_PREFIX}dedup AAPL filing')
  fetch_fn = _make_fetch_fn({'AAPL': [art]})

  c1 = asyncio.run(gdelt_poller.tick(
    classifier=_Classifier(), watchlist_override=['AAPL'],
    fetch_fn=fetch_fn,
  ))
  c2 = asyncio.run(gdelt_poller.tick(
    classifier=_Classifier(), watchlist_override=['AAPL'],
    fetch_fn=fetch_fn,
  ))
  _check("first tick new == 1", c1['new'] == 1, str(c1))
  _check("second tick new == 0", c2['new'] == 0, str(c2))
  _check("second tick dropped_dup >= 1", c2['dropped_dup'] >= 1, str(c2))


def test_flood_cap_engages():
  print("\n== hourly cap blocks further inserts once tripped ==")
  _cleanup()
  # Pre-fill the deque past the cap
  import time as _t
  now = _t.monotonic()
  for _ in range(gdelt_poller.HOURLY_INSERT_CAP + 5):
    gdelt_poller._recent_inserts.append(now)

  fetch_fn = _make_fetch_fn({
    'AAPL': [
      _gdelt_art(f'{_TEST_HEADLINE_PREFIX}flood {i}', seendate='20260522180000')
      for i in range(10)
    ],
  })
  counts = asyncio.run(gdelt_poller.tick(
    classifier=_Classifier(), watchlist_override=['AAPL'],
    fetch_fn=fetch_fn,
  ))
  _check("flood_capped > 0 (cap engaged)", counts['flood_capped'] > 0, str(counts))
  _check("new == 0 (all gated)", counts['new'] == 0, str(counts))


def test_classifier_none_dropped():
  print("\n== classifier returns None -> entry skipped, no crash ==")
  _cleanup()
  fetch_fn = _make_fetch_fn({
    'AAPL': [_gdelt_art(f'{_TEST_HEADLINE_PREFIX}null classifier path')],
  })
  counts = asyncio.run(gdelt_poller.tick(
    classifier=_NullClassifier(), watchlist_override=['AAPL'],
    fetch_fn=fetch_fn,
  ))
  _check("dropped_classifier_none == 1",
         counts['dropped_classifier_none'] == 1, str(counts))
  _check("new == 0", counts['new'] == 0, str(counts))
  _check("errors == 0", counts['errors'] == 0, str(counts))


def test_fetch_failure_no_crash():
  print("\n== fetch_fn raises -> no crash, other tickers continue ==")
  _cleanup()
  async def broken_fetch(ticker, timespan):
    if ticker == 'BROKEN':
      raise RuntimeError('simulated GDELT failure')
    return [_gdelt_art(f'{_TEST_HEADLINE_PREFIX}{ticker} headline')]

  counts = asyncio.run(gdelt_poller.tick(
    classifier=_Classifier(),
    watchlist_override=['BROKEN', 'AAPL'],
    fetch_fn=broken_fetch,
  ))
  # AAPL still processed despite BROKEN failing
  _check("new >= 1 (AAPL got through)", counts['new'] >= 1, str(counts))


def test_no_headline_dropped():
  print("\n== empty headline article is dropped ==")
  _cleanup()
  fetch_fn = _make_fetch_fn({
    'AAPL': [_gdelt_art(title='')],
  })
  counts = asyncio.run(gdelt_poller.tick(
    classifier=_Classifier(), watchlist_override=['AAPL'],
    fetch_fn=fetch_fn,
  ))
  _check("dropped_no_headline == 1",
         counts['dropped_no_headline'] == 1, str(counts))
  _check("new == 0", counts['new'] == 0, str(counts))


def test_date_parsing():
  print("\n== _parse_gdelt_date handles GDELT YYYYMMDDHHMMSS format ==")
  out = gdelt_poller._parse_gdelt_date('20260522143000')
  _check("parsed has 2026 prefix", '2026-05-22' in out, out)
  out2 = gdelt_poller._parse_gdelt_date('')
  _check("empty falls back to now", out2.startswith('2026-') or out2.startswith('20'),
         out2)


def test_schema_idempotent():
  print("\n== schema migration is idempotent ==")
  init_schema()
  init_schema()
  _check("init_schema called twice without error", True)


def main() -> int:
  print("\nGDELT poller unit tests\n")
  test_schema_idempotent()
  test_article_stored()
  test_dedup_on_rerun()
  test_flood_cap_engages()
  test_classifier_none_dropped()
  test_fetch_failure_no_crash()
  test_no_headline_dropped()
  test_date_parsing()
  _cleanup()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
