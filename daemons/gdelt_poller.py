"""GDELT 2.0 poller -- pulls global media coverage of watchlist tickers
into events_store every 15 minutes.

Architecture:
  - Async (httpx) so per-ticker GDELT queries fan out via asyncio.gather
  - Bounded concurrency via Semaphore (4 simultaneous GDELT calls max)
  - Three flood guards:
      (1) per-tick maxrecords=50 per ticker query
      (2) rolling hourly insert cap (default 500) -- in-memory deque
      (3) Materiality_Classifier filters most as is_material=False ->
          materiality='low', which sits below sentry_triage's enqueue
          threshold (0.50) and never reaches the queue
  - Classifier calls rate-limited via the existing MaterialityRateLimiter
    pattern (sliding-window token bucket; 25 RPM ceiling shared with
    news_watcher and rss_aggregator by the same Groq API key)

GDELT 2.0 DOC API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
  Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  Free, no auth. Fair-use rate limits.

Run:
  python -m daemons.gdelt_poller                # 15-min loop
  python -m daemons.gdelt_poller --once         # single pass, useful for tests
  python -m daemons.gdelt_poller --interval 60  # custom poll cadence
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_watchlist
from state.events_store import store_event, seen
from agent.Materiality_Classifier import Materiality_Classifier


# -- Tuning -----------------------------------------------------------------
DEFAULT_INTERVAL_S = 900           # 15 min between full polls
GDELT_ENDPOINT = 'https://api.gdeltproject.org/api/v2/doc/doc'
# GDELT's public API is documented as slow (~10-20s typical). Generous timeout
# is more important than fast failure here; the daemon polls every 15 min.
GDELT_TIMEOUT_S = 30.0
GDELT_MAXRECORDS = 50              # per-ticker query cap
HOURLY_INSERT_CAP = 500            # circuit breaker
CONCURRENT_QUERIES = 4             # semaphore-bounded concurrency
MATERIALITY_RPM = 25               # share-of-budget ceiling from this daemon
USER_AGENT = (
  f"Nemo Sentry GDELT poller "
  f"({os.getenv('SEC_EMAIL', 'analyst@example.com')})"
)

_running = True

# Rolling hourly window of insert timestamps for flood-cap accounting
_recent_inserts: Deque[float] = deque()


# ---------------------------------------------------------------------------
# Rate limiter (mirrors agent/MaterialityRateLimiter pattern in news_watcher;
# inlined here so this daemon has no cross-daemon import.)
# ---------------------------------------------------------------------------

class _RateLimiter:
  """Sliding-window pacing limiter. Caller awaits acquire() before each call."""

  def __init__(self, max_calls: int = MATERIALITY_RPM, window: float = 60.0):
    self.max_calls = max_calls
    self.window = window
    self._timestamps: List[float] = []
    self._lock = asyncio.Lock()

  async def acquire(self) -> None:
    async with self._lock:
      now = time.monotonic()
      min_interval = self.window / self.max_calls
      if self._timestamps:
        since_last = now - self._timestamps[-1]
        sleep_for = max(0.0, min_interval - since_last)
      else:
        sleep_for = 0.0
      scheduled = now + sleep_for
      self._timestamps.append(scheduled)
      cutoff = scheduled - self.window
      self._timestamps = [t for t in self._timestamps if t > cutoff]
    if sleep_for > 0:
      await asyncio.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Flood cap
# ---------------------------------------------------------------------------

def _hourly_count(now: Optional[float] = None) -> int:
  """Trim the deque and return current count within the 1-hour window."""
  current = now if now is not None else time.monotonic()
  cutoff = current - 3600.0
  while _recent_inserts and _recent_inserts[0] < cutoff:
    _recent_inserts.popleft()
  return len(_recent_inserts)


def _note_insert() -> None:
  _recent_inserts.append(time.monotonic())


def _reset_flood_state() -> None:
  """Test helper: clear the rolling deque between cases."""
  _recent_inserts.clear()


# ---------------------------------------------------------------------------
# GDELT fetch
# ---------------------------------------------------------------------------

async def fetch_gdelt_articles(
  client: httpx.AsyncClient,
  query: str,
  timespan: str = '15min',
  maxrecords: int = GDELT_MAXRECORDS,
) -> List[Dict[str, Any]]:
  """Single GDELT DOC API call. Returns the `articles` list (or [] on
  failure)."""
  params = {
    'query': query,
    'mode': 'ArtList',
    'format': 'json',
    'timespan': timespan,
    'maxrecords': maxrecords,
    'sort': 'DateDesc',
  }
  try:
    resp = await client.get(GDELT_ENDPOINT, params=params,
                            timeout=GDELT_TIMEOUT_S)
    if resp.status_code == 429:
      # one retry after backoff
      await asyncio.sleep(2.0)
      resp = await client.get(GDELT_ENDPOINT, params=params,
                              timeout=GDELT_TIMEOUT_S)
    if resp.status_code != 200:
      print(f"[gdelt] {query}: HTTP {resp.status_code}",
            file=sys.stderr, flush=True)
      return []
    data = resp.json()
    return data.get('articles', []) or []
  except Exception as exc:
    print(f"[gdelt] {query} fetch failed: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)
    return []


def _parse_gdelt_date(s: str) -> str:
  """GDELT 'seendate' is YYYYMMDDTHHMMSSZ (or YYYYMMDDHHMMSS in older
  payloads). Strip non-digits and parse the first 14. Return ISO 8601."""
  if not s:
    return datetime.now(timezone.utc).isoformat()
  digits = ''.join(c for c in s if c.isdigit())[:14]
  if len(digits) < 14:
    return datetime.now(timezone.utc).isoformat()
  try:
    return datetime.strptime(digits, '%Y%m%d%H%M%S').replace(
      tzinfo=timezone.utc).isoformat()
  except Exception:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-ticker processing
# ---------------------------------------------------------------------------

async def process_articles(
  ticker: str,
  articles: List[Dict[str, Any]],
  classifier: Optional[Materiality_Classifier],
  limiter: _RateLimiter,
  counts: Dict[str, int],
) -> None:
  """Classify and store each article. Honors the hourly insert cap."""
  for art in articles:
    if not _running:
      return

    if _hourly_count() >= HOURLY_INSERT_CAP:
      counts['flood_capped'] = counts.get('flood_capped', 0) + 1
      # Log once per ticker per tick rather than once per article
      break

    headline = art.get('title') or ''
    if not headline:
      counts['dropped_no_headline'] += 1
      continue
    seendate = art.get('seendate', '')
    published_at = _parse_gdelt_date(seendate)
    source = 'gdelt'
    if seen(source, headline, published_at):
      counts['dropped_dup'] += 1
      continue

    if classifier is None:
      counts['errors'] += 1
      continue

    await limiter.acquire()
    try:
      result = await asyncio.to_thread(
        classifier.classify, headline,
        art.get('socialimage', '') + ' ' + art.get('url', ''),
        source,
      )
    except Exception as exc:
      counts['errors'] += 1
      print(f"[gdelt] classify failed for '{headline[:60]}': {exc}",
            file=sys.stderr, flush=True)
      continue
    if result is None:
      counts['dropped_classifier_none'] += 1
      continue

    materiality = (
      'high' if result.is_material and result.urgency == 'immediate'
      else 'medium' if result.is_material
      else 'low'
    )
    primary = result.primary_ticker or ticker

    store_event(
      source=source,
      ticker=primary,
      headline=headline,
      body=art.get('url', ''),  # GDELT doesn't return body text -- URL is the link
      url=art.get('url', ''),
      published_at=published_at,
      materiality=materiality,
      category=result.category,
      affected_tickers=result.affected_tickers or [],
      primary_ticker=primary,
      directional_signal=result.directional_signal,
      urgency=result.urgency,
      classifier_reason=result.one_line_reason or '',
    )
    _note_insert()
    counts['new'] += 1


async def watch_ticker(
  ticker: str,
  classifier: Optional[Materiality_Classifier],
  limiter: _RateLimiter,
  semaphore: asyncio.Semaphore,
  client: httpx.AsyncClient,
  counts: Dict[str, int],
  timespan: str = '15min',
  fetch_fn: Optional[Callable[..., Awaitable[List[Dict[str, Any]]]]] = None,
) -> None:
  """Query GDELT for one ticker then process the results."""
  async with semaphore:
    if fetch_fn is None:
      articles = await fetch_gdelt_articles(client, ticker, timespan=timespan)
    else:
      articles = await fetch_fn(ticker, timespan)
  counts['fetched'] += len(articles)
  await process_articles(ticker, articles, classifier, limiter, counts)


# ---------------------------------------------------------------------------
# Core tick
# ---------------------------------------------------------------------------

async def tick(
  classifier: Optional[Materiality_Classifier] = None,
  watchlist_override: Optional[List[str]] = None,
  timespan: str = '15min',
  fetch_fn: Optional[Callable[..., Awaitable[List[Dict[str, Any]]]]] = None,
  client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, int]:
  """One full poll cycle across the watchlist. Returns counters dict."""
  watchlist = watchlist_override if watchlist_override is not None else get_watchlist()
  counts = {
    'fetched': 0, 'new': 0, 'dropped_dup': 0, 'dropped_unrelated': 0,
    'dropped_no_headline': 0, 'dropped_classifier_none': 0,
    'errors': 0, 'flood_capped': 0, 'tickers_polled': 0,
  }
  if not watchlist:
    return counts

  if classifier is None:
    classifier = Materiality_Classifier()

  limiter = _RateLimiter()
  semaphore = asyncio.Semaphore(CONCURRENT_QUERIES)

  close_client = False
  if client is None:
    client = httpx.AsyncClient(headers={'User-Agent': USER_AGENT})
    close_client = True
  try:
    await asyncio.gather(
      *(watch_ticker(t, classifier, limiter, semaphore, client, counts,
                     timespan=timespan, fetch_fn=fetch_fn)
        for t in watchlist),
      return_exceptions=True,
    )
    counts['tickers_polled'] = len(watchlist)
  finally:
    if close_client:
      await client.aclose()

  return counts


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _install_signal_handlers() -> None:
  def _stop(*_):
    global _running
    _running = False
    print("[gdelt_poller] shutdown signal received", file=sys.stderr, flush=True)
  signal.signal(signal.SIGINT, _stop)
  if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _stop)


def _format_counts(counts: Dict[str, int]) -> str:
  return (
    f"tickers={counts['tickers_polled']} fetched={counts['fetched']} "
    f"new={counts['new']} dup={counts['dropped_dup']} "
    f"err={counts['errors']} flood_cap={counts['flood_capped']}"
  )


async def _main_loop(args) -> None:
  init_schema()
  _install_signal_handlers()
  print(f"[gdelt_poller] starting | interval={args.interval}s | "
        f"hourly_cap={HOURLY_INSERT_CAP}",
        file=sys.stderr, flush=True)
  while _running:
    started = time.time()
    try:
      counts = await tick()
      elapsed = time.time() - started
      print(
        f"[gdelt_poller] tick {datetime.now(timezone.utc).isoformat()} "
        f"({elapsed:.1f}s) {_format_counts(counts)}",
        file=sys.stderr, flush=True,
      )
    except Exception as exc:
      import traceback
      print(f"[gdelt_poller] tick crashed: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True)
      traceback.print_exc(file=sys.stderr)
    if args.once:
      break
    slept = 0.0
    while _running and slept < args.interval:
      await asyncio.sleep(min(2.0, args.interval - slept))
      slept += 2.0
  print("[gdelt_poller] exited cleanly", file=sys.stderr, flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description='GDELT 2.0 poller daemon')
  parser.add_argument('--once', action='store_true', help='Single pass and exit')
  parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_S,
                      help=f'Seconds between ticks (default {DEFAULT_INTERVAL_S})')
  args = parser.parse_args()
  asyncio.run(_main_loop(args))


if __name__ == "__main__":
  main()
