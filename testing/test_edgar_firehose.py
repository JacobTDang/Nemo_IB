"""Tests for the EDGAR firehose daemon — feed parsing, filtering, event store
integration."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daemons.edgar_firehose import (
    _parse_feed, filter_relevant, store_filing_as_event,
    FORM_CATEGORIES, tick,
)
from state.schema import init_schema, get_connection


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name, cond, hint=''):
  if cond:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _section(t): print(f"\n=== {t} ===")


SYNTHETIC_ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <title>Latest Filings</title>
  <updated>2026-05-21T14:00:00-04:00</updated>
  <entry>
    <title>8-K - Microsoft Corporation (0000789019) (Filer)</title>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019&type=8-K"/>
    <updated>2026-05-21T13:59:00-04:00</updated>
    <id>urn:tag:msft-8k-001</id>
    <summary>Item 5.02 disclosure</summary>
  </entry>
  <entry>
    <title>SC 13D - Berkshire Hathaway Inc (0001067983) (Filer)</title>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001067983&type=SC13D"/>
    <updated>2026-05-21T13:45:00-04:00</updated>
    <id>urn:tag:brk-13d-001</id>
    <summary>Schedule 13D filing</summary>
  </entry>
  <entry>
    <title>4 - SOME RANDOM NAME INC (0001234567) (Filer)</title>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001234567&type=4"/>
    <updated>2026-05-21T13:30:00-04:00</updated>
    <id>urn:tag:random-4-001</id>
    <summary>Form 4 insider transaction</summary>
  </entry>
  <entry>
    <title>EFFECT - APPLE INC (0000320193) (Filer)</title>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=EFFECT"/>
    <updated>2026-05-21T13:00:00-04:00</updated>
    <id>urn:tag:aapl-effect-001</id>
    <summary>SEC Form Effectiveness</summary>
  </entry>
</feed>
"""


def test_parse_feed():
  _section("1. Feed parsing")
  entries = _parse_feed(SYNTHETIC_ATOM)
  _check(f"  parsed 4 entries", len(entries) == 4, f"got {len(entries)}")
  msft = entries[0]
  _check("  MSFT entry: correct form_type",
         msft['form_type'] == '8-K', f"got {msft['form_type']}")
  _check("  MSFT entry: correct CIK",
         msft['cik'] == '0000789019', f"got {msft['cik']}")
  brk = entries[1]
  _check("  BRK entry: form 'SC 13D'",
         brk['form_type'] == 'SC 13D', f"got {brk['form_type']}")
  _check("  BRK entry: CIK matches Berkshire",
         brk['cik'] == '0001067983')


def test_filter_relevant():
  _section("2. Filter logic")
  entries = _parse_feed(SYNTHETIC_ATOM)

  # Empty watchlist -> no entries
  out = filter_relevant(entries, cik_to_ticker={})
  _check("  empty watchlist -> 0 relevant",
         len(out) == 0, f"got {len(out)}")

  # MSFT-only watchlist
  out = filter_relevant(entries, cik_to_ticker={'0000789019': 'MSFT'})
  _check("  MSFT in watchlist -> 1 relevant",
         len(out) == 1, f"got {len(out)}")
  _check("  ticker tagged correctly", out[0]['ticker'] == 'MSFT')

  # MSFT + AAPL — but AAPL filing is 'EFFECT' which isn't in our form filter
  out = filter_relevant(entries, cik_to_ticker={
    '0000789019': 'MSFT', '0000320193': 'AAPL',
  })
  _check("  MSFT 8-K matches; AAPL EFFECT does not",
         len(out) == 1 and out[0]['ticker'] == 'MSFT',
         f"got {[(o['ticker'], o['form_type']) for o in out]}")

  # All 3 watchlist tickers
  out = filter_relevant(entries, cik_to_ticker={
    '0000789019': 'MSFT', '0001067983': 'BRK', '0001234567': 'RANDOM',
  })
  forms = sorted([o['form_type'] for o in out])
  _check("  3 watchlist matches with relevant forms",
         len(out) == 3,
         f"got {len(out)}: {[(o['ticker'], o['form_type']) for o in out]}")
  _check("  forms include 8-K, SC 13D, 4",
         '8-K' in forms and 'SC 13D' in forms and '4' in forms,
         f"got {forms}")


def test_store_filing_dedupe():
  _section("3. Event store dedupe")
  init_schema()
  entries = _parse_feed(SYNTHETIC_ATOM)
  filtered = filter_relevant(entries, cik_to_ticker={'0000789019': 'MSFT'})
  _check("  fixture: 1 filtered entry", len(filtered) == 1)

  # First store
  eid1 = store_filing_as_event(filtered[0])
  _check("  first store returns event_id", eid1 is not None)

  # Second store of same entry
  eid2 = store_filing_as_event(filtered[0])
  _check("  duplicate store returns None (dedupe)", eid2 is None)

  # Cleanup
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE event_id = ?", (eid1,))
    conn.commit()
  finally:
    conn.close()


def test_tick_with_injected_feed():
  _section("4. Tick() with synthetic feed override")
  entries = _parse_feed(SYNTHETIC_ATOM)
  # tick() builds the CIK map from watchlist; for the test we need MSFT
  from state.schema import add_to_watchlist, remove_from_watchlist
  add_to_watchlist('MSFT', priority=5, notes='test')
  try:
    summary = tick(log_fn=lambda *a: None, _entries_override=entries)
    _check("  feed_size = 4", summary['feed_size'] == 4,
           f"got {summary['feed_size']}")
    _check("  at least 1 relevant entry for MSFT",
           summary['relevant_count'] >= 1,
           f"got {summary['relevant_count']}")
  finally:
    # Cleanup: drop the test event
    conn = get_connection()
    try:
      conn.execute("DELETE FROM events WHERE source LIKE 'sec_edgar_atom:%' AND ticker = 'MSFT'")
      conn.commit()
    finally:
      conn.close()
    # Don't remove MSFT from watchlist if it was already there for real;
    # skip that — V1 just makes it idempotent enough


def main():
  print("\nEDGAR Firehose — tests\n")
  init_schema()
  test_parse_feed()
  test_filter_relevant()
  test_store_filing_dedupe()
  test_tick_with_injected_feed()
  print(f"\n=== Summary ===\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  if _results['failures']:
    for n, h in _results['failures']:
      print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
