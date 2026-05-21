"""SEC EDGAR firehose — sub-minute alert on new 8-K, 13D, 13G filings for
watchlist tickers.

Strategy: poll EDGAR's "current filings" Atom feed every 30-60 seconds,
filter by form type + watchlist CIK match, and store new filings to the
events table (which the falsifier_watcher will then pick up on its next
tick).

The Atom feed returns the last ~40 filings across the whole market. We
poll every 45 seconds and dedupe by accession number. For high-activity
watchlists, this is fast enough to catch a 13D within 1-2 minutes of
publication.

Run via:
  python -m daemons.edgar_firehose
  python -m daemons.edgar_firehose --once

Stop via Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import signal
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import feedparser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_watchlist, get_connection
from state.events_store import store_event, seen


DEFAULT_INTERVAL_S = 45
# Forms we care about (mapped to category tags)
FORM_CATEGORIES = {
  '8-K':       'corporate_event',
  '8-K/A':     'corporate_event',
  'SC 13D':    'activist',
  'SC 13D/A':  'activist',
  'SC 13G':    'institutional',
  'SC 13G/A':  'institutional',
  '4':         'insider_form4',
  '4/A':       'insider_form4',
  '10-K':      'annual_report',
  '10-K/A':    'annual_report',
  '10-Q':      'quarterly_report',
}

# Atom feed: latest filings, no auth required, SEC rate-limits to ~10 req/s
ATOM_URL = (
  "https://www.sec.gov/cgi-bin/browse-edgar?"
  "action=getcurrent&type=&owner=include&count=40&output=atom"
)
HEADERS = {
  'User-Agent': f"{os.getenv('NAME', 'analyst')} {os.getenv('SEC_EMAIL', 'analyst@example.com')}",
}


_running = True


def _build_cik_to_ticker_map() -> Dict[str, str]:
  """Resolve watchlist tickers to CIKs so we can match Atom feed entries.

  Uses edgartools' built-in ticker resolver. Falls back to skipping
  unresolvable tickers."""
  try:
    from edgar import Company, set_identity
    set_identity(HEADERS['User-Agent'])
  except ImportError:
    return {}

  cik_map: Dict[str, str] = {}
  for ticker in get_watchlist():
    try:
      cik = str(Company(ticker).cik).lstrip('0').zfill(10)
      cik_map[cik] = ticker
    except Exception:
      continue
  return cik_map


def _parse_feed(text: Optional[str] = None) -> List[Dict[str, Any]]:
  """Pull + parse the current-filings Atom feed. Returns list of normalized
  entries. If text is provided (for testing), uses that instead of HTTP."""
  if text is None:
    feed = feedparser.parse(ATOM_URL, request_headers=HEADERS)
  else:
    feed = feedparser.parse(text)
  entries = []
  for e in feed.entries:
    title = getattr(e, 'title', '')
    # Title format: "<FORM_TYPE> - <COMPANY_NAME> (<CIK>) (Filer)"
    # We split on ' - ' first then extract CIK and form
    form_type = None
    cik = None
    company = None
    try:
      head, rest = title.split(' - ', 1)
      form_type = head.strip()
      # Extract CIK with regex
      import re
      m = re.search(r'\((\d{10}|\d{4,10})\)', rest)
      if m:
        cik = m.group(1).zfill(10)
      # Company name: everything before the first '('
      company = rest.split('(')[0].strip()
    except Exception:
      pass

    entries.append({
      'form_type':  form_type,
      'cik':        cik,
      'company':    company,
      'title':      title,
      'link':       getattr(e, 'link', ''),
      'updated':    getattr(e, 'updated', ''),
      'summary':    getattr(e, 'summary', ''),
      'id':         getattr(e, 'id', ''),
    })
  return entries


def filter_relevant(
  entries: List[Dict[str, Any]],
  cik_to_ticker: Dict[str, str],
  form_filter: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
  """Return only entries where CIK is in watchlist AND form_type is one we
  care about."""
  forms_we_care_about = form_filter or set(FORM_CATEGORIES.keys())
  out = []
  for e in entries:
    if not e.get('cik'):
      continue
    if e['cik'] not in cik_to_ticker:
      continue
    if e.get('form_type') not in forms_we_care_about:
      continue
    e['ticker'] = cik_to_ticker[e['cik']]
    e['category'] = FORM_CATEGORIES.get(e['form_type'], 'other')
    out.append(e)
  return out


def store_filing_as_event(entry: Dict[str, Any]) -> Optional[str]:
  """Store an EDGAR Atom entry into the events table. Returns event_id
  (or None if already seen)."""
  ticker = entry['ticker']
  form = entry['form_type']
  source = f"sec_edgar_atom:{form}"
  headline = entry['title']
  published_at = entry.get('updated', datetime.now(timezone.utc).isoformat())
  if seen(source, headline, published_at):
    return None
  eid = store_event(
    source=source,
    ticker=ticker,
    headline=headline,
    body=entry.get('summary', ''),
    url=entry.get('link', ''),
    published_at=published_at,
    materiality='high',  # any new 8-K/13D for a watchlist name is material
    category=entry['category'],
    affected_tickers=[ticker],
    primary_ticker=ticker,
    directional_signal='neutral',
    urgency='hours',
    classifier_reason=f'EDGAR firehose: {form}',
  )
  return eid


def tick(log_fn=print, _entries_override=None) -> Dict[str, Any]:
  """One poll cycle. Returns summary."""
  cik_to_ticker = _build_cik_to_ticker_map()
  entries = _entries_override if _entries_override is not None else _parse_feed()
  relevant = filter_relevant(entries, cik_to_ticker)

  new_count = 0
  details = []
  for e in relevant:
    eid = store_filing_as_event(e)
    if eid:
      new_count += 1
      details.append({
        'event_id':  eid,
        'ticker':    e['ticker'],
        'form_type': e['form_type'],
        'title':     e['title'],
        'url':       e['link'],
        'category':  e['category'],
      })
      log_fn(f"  EDGAR  {e['ticker']:6s} {e['form_type']:9s} | {e['title'][:80]}")

  return {
    'feed_size':       len(entries),
    'watchlist_size':  len(cik_to_ticker),
    'relevant_count':  len(relevant),
    'new_event_count': new_count,
    'new_events':      details,
  }


def _install_signal_handlers():
  def _stop(*_):
    global _running
    _running = False
    print("[edgar_firehose] shutdown signal received", file=sys.stderr, flush=True)
  signal.signal(signal.SIGINT, _stop)
  if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _stop)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--once', action='store_true')
  parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_S)
  args = parser.parse_args()

  init_schema()
  _install_signal_handlers()
  print(f"[edgar_firehose] starting | interval={args.interval}s",
        file=sys.stderr, flush=True)

  while _running:
    try:
      summary = tick()
      if summary['new_event_count'] == 0:
        print(f"[tick] {summary['feed_size']} feed items, "
              f"{summary['relevant_count']} watchlist-relevant, 0 new",
              file=sys.stderr, flush=True)
      else:
        print(f"[tick] {summary['new_event_count']} NEW filings for watchlist",
              file=sys.stderr, flush=True)
    except Exception as e:
      print(f"[edgar_firehose] tick crashed: {type(e).__name__}: {e}",
            file=sys.stderr, flush=True)
    if args.once:
      break
    slept = 0
    while _running and slept < args.interval:
      time.sleep(min(2.0, args.interval - slept))
      slept += 2.0
  print("[edgar_firehose] exited cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
  main()
