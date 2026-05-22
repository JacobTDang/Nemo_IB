"""RSS aggregator -- polls a curated list of feeds and stores material items
into events_store.

Two flavors of feed per `rss_feeds.yaml`:

  1. Typed feeds (SEC EDGAR Atom, FDA, Fed press, BLS, BEA, Federal Register)
     where the source itself signals materiality. These have a hardcoded
     `materiality` and `category` in the YAML config -- no classifier call.

  2. General news feeds (Reuters, Bloomberg, WSJ, etc.) where each entry
     must be classified for relevance. These run through Materiality_Classifier
     and self-pace at ~24 RPM via time.sleep to share the Groq budget with
     news_watcher and gdelt_poller.

Stop via Ctrl+C.

Run:
  python -m daemons.rss_aggregator              # default 30-min interval
  python -m daemons.rss_aggregator --once       # single pass
  python -m daemons.rss_aggregator --feeds <path>
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import feedparser
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_watchlist
from state.events_store import store_event, seen
from agent.Materiality_Classifier import Materiality_Classifier


DEFAULT_INTERVAL_S = 1800       # 30 min between full poll cycles
DEFAULT_FEEDS_PATH = Path(__file__).parent / 'rss_feeds.yaml'

# Self-pace classifier calls to share the 25-RPM Groq budget with the other
# classifier-using daemons (news_watcher, gdelt_poller). 2.5s = ~24 RPM ceiling
# from this one daemon; aggregate across all three may still exceed limit, in
# which case we rely on Groq's 429 retry + None-return-on-fail behavior.
CLASSIFY_SLEEP_S = 2.5

# Per-feed safety caps to bound work per tick
MAX_ITEMS_PER_FEED = 30
MAX_BODY_CHARS = 4000

# SEC requires a UA header that identifies the requester
SEC_UA = f"{os.getenv('NAME', 'analyst')} {os.getenv('SEC_EMAIL', 'analyst@example.com')} (Nemo Sentry RSS)"


_running = True


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_feeds(path: Path = DEFAULT_FEEDS_PATH) -> List[Dict[str, Any]]:
  """Load rss_feeds.yaml. Returns the `feeds` list, or empty list on missing.
  Schema per entry:
    name        (str)  -- human label
    url         (str)  -- feed endpoint
    materiality (str)  -- 'high' | 'medium' | 'low' | 'classify'
    category    (str)  -- when hardcoded; else classifier sets it
    default_ticker  (str, optional)  -- single-issuer feeds (an IR feed)
    watchlist_only  (bool, optional) -- drop entries whose tickers don't match
    is_sec      (bool, optional)     -- sets SEC User-Agent header
  """
  if not path.exists():
    print(f"[rss] feeds file not found: {path}", file=sys.stderr, flush=True)
    return []
  with open(path, encoding='utf-8') as f:
    doc = yaml.safe_load(f) or {}
  return doc.get('feeds') or []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
  """Stable identifier from feed name. Used in event source field."""
  s = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')
  return s or 'unnamed'


def _entry_published(entry: Any) -> str:
  """Best-effort published timestamp from a feedparser entry."""
  for k in ('published', 'updated', 'pubDate'):
    v = getattr(entry, k, None) or (entry.get(k) if isinstance(entry, dict) else None)
    if v:
      return str(v)
  return datetime.now(timezone.utc).isoformat()


def _entry_text(entry: Any, attr: str) -> str:
  v = getattr(entry, attr, None)
  if v is None and isinstance(entry, dict):
    v = entry.get(attr)
  return str(v) if v else ''


def _parse_feed(feed_cfg: Dict[str, Any]) -> Any:
  """Wrap feedparser.parse with optional SEC User-Agent. Returns the parsed
  feed object (with .entries) regardless of network outcome -- feedparser
  exposes errors via .bozo."""
  headers = None
  if feed_cfg.get('is_sec'):
    headers = {'User-Agent': SEC_UA}
  return feedparser.parse(feed_cfg['url'], request_headers=headers)


def _map_classifier_materiality(result) -> str:
  """Mirror news_watcher.py:160-161 mapping."""
  if result.is_material and result.urgency == 'immediate':
    return 'high'
  return 'medium' if result.is_material else 'low'


# ---------------------------------------------------------------------------
# Core tick
# ---------------------------------------------------------------------------

def process_entry(
  entry: Any,
  feed_cfg: Dict[str, Any],
  classifier: Optional[Materiality_Classifier],
  watchlist_set: Set[str],
  counts: Dict[str, int],
) -> None:
  """Process one feed entry: dedup, classify if needed, store."""
  headline = _entry_text(entry, 'title')
  if not headline:
    counts['dropped_no_headline'] += 1
    return

  summary = _entry_text(entry, 'summary')[:MAX_BODY_CHARS]
  link = _entry_text(entry, 'link')
  published_at = _entry_published(entry)
  source = f"rss:{_slug(feed_cfg['name'])}"

  if seen(source, headline, published_at):
    counts['dropped_dup'] += 1
    return

  # Decide materiality + category + ticker
  materiality_cfg = feed_cfg.get('materiality', 'classify')
  category = feed_cfg.get('category', 'other')
  default_ticker = feed_cfg.get('default_ticker')
  watchlist_only = bool(feed_cfg.get('watchlist_only', False))

  if materiality_cfg == 'classify':
    if classifier is None:
      counts['errors'] += 1
      return
    # Self-pace before each classify call
    time.sleep(CLASSIFY_SLEEP_S)
    try:
      result = classifier.classify(headline, summary, source)
    except Exception as exc:
      counts['errors'] += 1
      print(f"[rss] classify failed for '{headline[:60]}': {exc}",
            file=sys.stderr, flush=True)
      return
    if result is None:
      counts['dropped_classifier_none'] += 1
      return
    materiality = _map_classifier_materiality(result)
    category = result.category or category
    affected = result.affected_tickers or []
    primary = result.primary_ticker or default_ticker
    directional = result.directional_signal or 'neutral'
    urgency = result.urgency or 'days'
    classifier_reason = result.one_line_reason or ''
  else:
    materiality = materiality_cfg
    affected = [default_ticker] if default_ticker else []
    primary = default_ticker
    directional = 'neutral'
    urgency = 'hours' if materiality == 'high' else 'days'
    classifier_reason = f'RSS hardcoded: {feed_cfg["name"]}'

  # Watchlist filter -- if requested, drop when no overlap
  if watchlist_only:
    affected_set = {t.upper() for t in affected if t}
    if primary:
      affected_set.add(primary.upper())
    if not (affected_set & watchlist_set):
      counts['dropped_unrelated'] += 1
      return

  store_event(
    source=source,
    ticker=primary or '',
    headline=headline,
    body=summary,
    url=link,
    published_at=published_at,
    materiality=materiality,
    category=category,
    affected_tickers=affected,
    primary_ticker=primary,
    directional_signal=directional,
    urgency=urgency,
    classifier_reason=classifier_reason,
  )
  counts['new'] += 1


def tick(
  feeds: Optional[List[Dict[str, Any]]] = None,
  classifier: Optional[Materiality_Classifier] = None,
  watchlist_override: Optional[List[str]] = None,
  _feed_parser=_parse_feed,   # injectable for tests
) -> Dict[str, int]:
  """One pass over all feeds. Returns counters dict."""
  if feeds is None:
    feeds = load_feeds()
  if classifier is None:
    # Lazy-instantiate so test paths that supply pre-classified feeds don't
    # need Groq credentials
    has_classify_feed = any(f.get('materiality', 'classify') == 'classify' for f in feeds)
    if has_classify_feed:
      classifier = Materiality_Classifier()

  watchlist = watchlist_override if watchlist_override is not None else get_watchlist()
  watchlist_set = {t.upper() for t in watchlist}

  counts = {
    'fetched': 0, 'new': 0, 'dropped_dup': 0, 'dropped_unrelated': 0,
    'dropped_no_headline': 0, 'dropped_classifier_none': 0,
    'errors': 0, 'feeds_processed': 0, 'feeds_failed': 0,
  }

  for feed_cfg in feeds:
    if not _running:
      break
    name = feed_cfg.get('name', '?')
    try:
      parsed = _feed_parser(feed_cfg)
    except Exception as exc:
      counts['feeds_failed'] += 1
      print(f"[rss] {name} fetch failed: {exc}", file=sys.stderr, flush=True)
      continue

    entries = list(getattr(parsed, 'entries', []) or [])
    counts['feeds_processed'] += 1

    for entry in entries[:MAX_ITEMS_PER_FEED]:
      if not _running:
        break
      counts['fetched'] += 1
      try:
        process_entry(entry, feed_cfg, classifier, watchlist_set, counts)
      except Exception as exc:
        counts['errors'] += 1
        print(f"[rss] {name} entry failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)

  return counts


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _install_signal_handlers() -> None:
  def _stop(*_):
    global _running
    _running = False
    print("[rss_aggregator] shutdown signal received",
          file=sys.stderr, flush=True)
  signal.signal(signal.SIGINT, _stop)
  if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _stop)


def _format_counts(counts: Dict[str, int]) -> str:
  return (
    f"feeds={counts['feeds_processed']}/{counts['feeds_processed'] + counts['feeds_failed']} "
    f"fetched={counts['fetched']} new={counts['new']} "
    f"dup={counts['dropped_dup']} unrel={counts['dropped_unrelated']} "
    f"err={counts['errors']}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description='RSS aggregator daemon')
  parser.add_argument('--once', action='store_true', help='Run one tick and exit')
  parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_S,
                      help=f'Seconds between ticks (default {DEFAULT_INTERVAL_S})')
  parser.add_argument('--feeds', type=str, default=str(DEFAULT_FEEDS_PATH),
                      help='Path to rss_feeds.yaml')
  args = parser.parse_args()

  init_schema()
  _install_signal_handlers()

  feeds_path = Path(args.feeds)
  feeds = load_feeds(feeds_path)
  if not feeds:
    print(f"[rss_aggregator] no feeds in {feeds_path}; nothing to do",
          file=sys.stderr, flush=True)
    sys.exit(1)

  print(f"[rss_aggregator] starting | feeds={len(feeds)} | interval={args.interval}s",
        file=sys.stderr, flush=True)

  while _running:
    started = time.time()
    try:
      counts = tick(feeds=feeds)
      elapsed = time.time() - started
      print(
        f"[rss_aggregator] tick {datetime.now(timezone.utc).isoformat()} "
        f"({elapsed:.1f}s) {_format_counts(counts)}",
        file=sys.stderr, flush=True,
      )
    except Exception as exc:
      import traceback
      print(f"[rss_aggregator] tick crashed: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True)
      traceback.print_exc(file=sys.stderr)

    if args.once:
      break

    slept = 0.0
    while _running and slept < args.interval:
      time.sleep(min(2.0, args.interval - slept))
      slept += 2.0

  print("[rss_aggregator] exited cleanly", file=sys.stderr, flush=True)


if __name__ == "__main__":
  main()
