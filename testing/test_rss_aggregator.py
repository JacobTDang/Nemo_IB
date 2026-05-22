"""Unit tests for daemons/rss_aggregator.py.

Uses synthetic feedparser-shaped input (no live HTTP). The `_parse_feed`
hook is injectable via `tick(_feed_parser=...)`, so we substitute a
fake parser that returns canned entries.

Run:
  .venv\\Scripts\\python.exe testing\\test_rss_aggregator.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import get_connection, init_schema
from daemons import rss_aggregator


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


_TEST_PREFIX_SOURCE = 'rss:rsstest_'


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE source LIKE ?",
                 (f"{_TEST_PREFIX_SOURCE}%",))
    conn.commit()
  finally:
    conn.close()


class _FakeEntry:
  """Mimic the attribute-style access feedparser uses."""
  def __init__(self, title='', summary='', link='', published=''):
    self.title = title
    self.summary = summary
    self.link = link
    self.published = published


class _FakeFeed:
  def __init__(self, entries):
    self.entries = entries


def _make_parser(entries_per_feed):
  """Build a parser callable that returns canned entries keyed by feed name."""
  def parser(feed_cfg):
    return _FakeFeed(entries_per_feed.get(feed_cfg['name'], []))
  return parser


def _row_count_for_source_like(prefix: str) -> int:
  conn = get_connection()
  try:
    cur = conn.execute("SELECT COUNT(*) FROM events WHERE source LIKE ?",
                       (f"{prefix}%",))
    return cur.fetchone()[0]
  finally:
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hardcoded_feed_stores_entry():
  print("\n== hardcoded-materiality feed: entry stored without classifier ==")
  _cleanup()

  feed = {
    'name': 'RSSTEST hardcoded',
    'url': 'http://example.com/rss',
    'materiality': 'high',
    'category': 'corporate_event',
    'default_ticker': 'AAPL',
    'watchlist_only': False,
  }
  entry = _FakeEntry(
    title='RSSTEST AAPL files 8-K',
    summary='Test body',
    link='http://example.com/aapl-8k',
    published='2026-05-22T10:00:00Z',
  )
  parser = _make_parser({feed['name']: [entry]})

  counts = rss_aggregator.tick(
    feeds=[feed], classifier=None,
    watchlist_override=['AAPL', 'NVDA'],
    _feed_parser=parser,
  )
  _check("fetched == 1", counts['fetched'] == 1, str(counts))
  _check("new == 1 (entry stored)", counts['new'] == 1, str(counts))
  _check("dropped_dup == 0 first run", counts['dropped_dup'] == 0)
  _check("row appears in events for this source",
         _row_count_for_source_like('rss:rsstest_hardcoded') == 1)


def test_dedup_on_rerun():
  print("\n== dedup: re-running with same entry produces no new rows ==")
  _cleanup()

  feed = {
    'name': 'RSSTEST dedup',
    'url': 'http://example.com/rss',
    'materiality': 'high',
    'category': 'corporate_event',
    'default_ticker': 'AAPL',
  }
  entry = _FakeEntry(
    title='RSSTEST dedup AAPL filing',
    summary='Test body',
    link='http://example.com/x',
    published='2026-05-22T11:00:00Z',
  )
  parser = _make_parser({feed['name']: [entry]})

  c1 = rss_aggregator.tick(feeds=[feed], classifier=None,
                           watchlist_override=['AAPL'], _feed_parser=parser)
  c2 = rss_aggregator.tick(feeds=[feed], classifier=None,
                           watchlist_override=['AAPL'], _feed_parser=parser)

  _check("first run new == 1", c1['new'] == 1, str(c1))
  _check("second run new == 0", c2['new'] == 0, str(c2))
  _check("second run dropped_dup >= 1", c2['dropped_dup'] >= 1, str(c2))


def test_watchlist_filter_drops_unrelated():
  print("\n== watchlist_only: classifier sets ticker outside watchlist -> dropped ==")
  _cleanup()

  feed = {
    'name': 'RSSTEST watchlist',
    'url': 'http://example.com/rss',
    'materiality': 'classify',
    'watchlist_only': True,
  }
  entry = _FakeEntry(
    title='RSSTEST off-watchlist news',
    summary='ZZZZ does something',
    published='2026-05-22T12:00:00Z',
  )
  parser = _make_parser({feed['name']: [entry]})

  class _FakeClassifier:
    def classify(self, headline, summary, source):
      class R:
        is_material = True
        category = 'sector'
        affected_tickers = ['ZZZZ']
        primary_ticker = 'ZZZZ'
        directional_signal = 'neutral'
        urgency = 'days'
        one_line_reason = 'unrelated'
      return R()

  counts = rss_aggregator.tick(
    feeds=[feed], classifier=_FakeClassifier(),
    watchlist_override=['AAPL', 'NVDA'],  # ZZZZ not in watchlist
    _feed_parser=parser,
  )
  _check("new == 0 (off-watchlist dropped)", counts['new'] == 0, str(counts))
  _check("dropped_unrelated == 1", counts['dropped_unrelated'] == 1, str(counts))


def test_classifier_none_does_not_store():
  print("\n== classifier returns None: entry skipped, no crash ==")
  _cleanup()

  feed = {
    'name': 'RSSTEST classify None',
    'url': 'http://example.com/rss',
    'materiality': 'classify',
    'watchlist_only': False,
  }
  entry = _FakeEntry(
    title='RSSTEST classifier None',
    summary='body',
    published='2026-05-22T13:00:00Z',
  )
  parser = _make_parser({feed['name']: [entry]})

  class _NullClassifier:
    def classify(self, *a, **kw):
      return None  # parse failure path

  counts = rss_aggregator.tick(
    feeds=[feed], classifier=_NullClassifier(),
    watchlist_override=['AAPL'],
    _feed_parser=parser,
  )
  _check("new == 0", counts['new'] == 0, str(counts))
  _check("dropped_classifier_none == 1",
         counts['dropped_classifier_none'] == 1, str(counts))
  _check("errors == 0 (None is graceful path)",
         counts['errors'] == 0, str(counts))


def test_no_headline_dropped():
  print("\n== entries with no headline are dropped gracefully ==")
  _cleanup()

  feed = {
    'name': 'RSSTEST no headline',
    'url': 'http://example.com/rss',
    'materiality': 'high',
    'category': 'macro',
  }
  entry = _FakeEntry(title='', summary='no title', published='2026-05-22T14Z')
  parser = _make_parser({feed['name']: [entry]})

  counts = rss_aggregator.tick(
    feeds=[feed], classifier=None,
    watchlist_override=['AAPL'],
    _feed_parser=parser,
  )
  _check("dropped_no_headline == 1", counts['dropped_no_headline'] == 1,
         str(counts))
  _check("new == 0", counts['new'] == 0, str(counts))


def test_feed_fetch_exception_does_not_crash():
  print("\n== feed-fetch exception: counted, daemon continues ==")
  _cleanup()

  feed = {
    'name': 'RSSTEST broken',
    'url': 'http://broken.example.com/rss',
    'materiality': 'high',
    'category': 'macro',
  }
  def boom(feed_cfg):
    raise RuntimeError('simulated network failure')

  counts = rss_aggregator.tick(
    feeds=[feed], classifier=None,
    watchlist_override=['AAPL'],
    _feed_parser=boom,
  )
  _check("feeds_failed == 1", counts['feeds_failed'] == 1, str(counts))
  _check("feeds_processed == 0", counts['feeds_processed'] == 0, str(counts))


def test_schema_idempotent():
  print("\n== schema migration is idempotent ==")
  init_schema()
  init_schema()
  _check("init_schema called twice without error", True)


def main() -> int:
  print("\nRSS aggregator unit tests\n")
  test_schema_idempotent()
  test_hardcoded_feed_stores_entry()
  test_dedup_on_rerun()
  test_watchlist_filter_drops_unrelated()
  test_classifier_none_does_not_store()
  test_no_headline_dropped()
  test_feed_fetch_exception_does_not_crash()
  _cleanup()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
