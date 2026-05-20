"""Always-on daemon that polls news feeds, classifies materiality, and writes
material events to the events table.

Sources:
  - Finnhub /company-news per watchlist ticker (every 60s)
  - SEC EDGAR latest filings Atom feed (every 120s)
  - Finnhub /general-news category=general (every 180s, broad macro)

Each ingest path:
  1. Pull recent items
  2. Skip if already in events table (dedup by content hash)
  3. Classify via Materiality_Classifier (Groq Llama 8B)
  4. Store in events table

Run via:  python -m daemons.news_watcher
Stop via: Ctrl+C (handler cleans up gracefully)
"""
import asyncio
import sys
import os
import signal
import time
from datetime import datetime, timedelta
from typing import List, Optional, Callable
import json

import feedparser

# Ensure project root is on path when run as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_watchlist
from state.events_store import store_event, seen
from agent.Materiality_Classifier import Materiality_Classifier
from tools.news_agregator.finnhub_utils import FinnhubClient


SEC_FEED_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&owner=include&count=40&output=atom"
POLL_FINNHUB = 60
POLL_SEC = 120
POLL_MACRO = 180

# Throttle classifier calls below Groq free-tier limits (30 RPM). 25 leaves
# headroom for retries. Cap articles per source to keep one tick from
# saturating the bucket — recent items are the materially-moving ones anyway.
MATERIALITY_RPM = 25
MAX_PER_TICKER = 8
MAX_SEC_FILINGS = 15
MAX_GENERAL_NEWS = 12

_running = True


class MaterialityRateLimiter:
  """Sliding-window token bucket. Throttles to max_calls per window seconds.

  Awaitable: `await limiter.acquire()` blocks (via asyncio.sleep) until a slot
  frees. `clock` is injectable for testing — defaults to time.monotonic.
  """
  def __init__(self, max_calls: int = MATERIALITY_RPM, window: float = 60.0,
               clock: Optional[Callable[[], float]] = None):
    self.max_calls = max_calls
    self.window = window
    self._clock = clock or time.monotonic
    self._timestamps: List[float] = []
    self._lock = asyncio.Lock()

  async def acquire(self) -> None:
    """Even-pacing acquire: enforce a minimum interval of `window / max_calls`
    between consecutive acquires. The lock is held only long enough to
    record the timestamp, so contending waiters serialize cleanly without
    a thundering herd.
    """
    async with self._lock:
      now = self._clock()
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


# Module-level singleton so all three watch tasks share the bucket.
_LIMITER: Optional[MaterialityRateLimiter] = None


def _get_limiter() -> MaterialityRateLimiter:
  global _LIMITER
  if _LIMITER is None:
    _LIMITER = MaterialityRateLimiter()
  return _LIMITER


async def _classify(classifier, headline: str, summary: str, source: str):
  """Rate-limited classifier wrapper. Runs the sync classify in a thread so
  the Groq HTTP call doesn't block the event loop."""
  await _get_limiter().acquire()
  return await asyncio.to_thread(classifier.classify, headline, summary, source)


def _stop(*_):
  global _running
  _running = False
  print("\n[news_watcher] Shutdown requested...", file=sys.stderr, flush=True)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


async def watch_finnhub_company_news(classifier: Materiality_Classifier):
  """Poll Finnhub company-news for each watchlist ticker."""
  client = FinnhubClient()
  global _running
  while _running:
    watchlist = get_watchlist()
    end = datetime.now()
    start = end - timedelta(minutes=POLL_FINNHUB // 60 + 5)
    for ticker in watchlist:
      try:
        url = "https://finnhub.io/api/v1/company-news"
        params = {
          "symbol": ticker,
          "from": start.strftime("%Y-%m-%d"),
          "to": end.strftime("%Y-%m-%d"),
          "token": client._api_key,
        }
        import aiohttp
        async with aiohttp.ClientSession() as session:
          async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
              print(f"[news_watcher][finnhub] {ticker}: HTTP {resp.status}", file=sys.stderr, flush=True)
              continue
            articles = await resp.json()
      except Exception as e:
        print(f"[news_watcher][finnhub] {ticker} failed: {e}", file=sys.stderr, flush=True)
        continue

      for art in (articles or [])[:MAX_PER_TICKER]:
        headline = art.get('headline', '')
        summary = art.get('summary', '')
        published = datetime.fromtimestamp(art.get('datetime', 0)).isoformat()
        source = f"finnhub:{art.get('source', 'unknown')}"
        if seen(source, headline, published):
          continue
        # Classify (rate-limited + threaded)
        result = await _classify(classifier, headline, summary, source)
        if result is None:
          continue  # parse failed; skip
        store_event(
          source=source, ticker=ticker, headline=headline, body=summary,
          url=art.get('url', ''), published_at=published,
          materiality='high' if result.is_material and result.urgency == 'immediate' else
                      'medium' if result.is_material else 'low',
          category=result.category,
          affected_tickers=result.affected_tickers,
          primary_ticker=result.primary_ticker,
          directional_signal=result.directional_signal,
          urgency=result.urgency,
          classifier_reason=result.one_line_reason,
        )
        if result.is_material:
          print(f"[news_watcher][material] {ticker}: {headline[:80]}", file=sys.stderr, flush=True)
    # Sleep until next round; check _running every second so Ctrl+C is fast
    for _ in range(POLL_FINNHUB):
      if not _running:
        return
      await asyncio.sleep(1)


async def watch_sec_edgar(classifier: Materiality_Classifier):
  """Poll SEC EDGAR Atom feed for new filings across the whole market.
  Filter to watchlist tickers only when ingesting (most filings won't match)."""
  global _running
  while _running:
    try:
      # feedparser is sync; run in thread to not block event loop
      feed = await asyncio.to_thread(feedparser.parse, SEC_FEED_URL)
    except Exception as e:
      print(f"[news_watcher][sec] feed fetch failed: {e}", file=sys.stderr, flush=True)
      await asyncio.sleep(POLL_SEC)
      continue

    watchlist = set(get_watchlist())
    for entry in feed.entries[:MAX_SEC_FILINGS]:
      title = entry.get('title', '')
      # Title format: "FORM_TYPE - COMPANY NAME (CIK)"
      # We only ingest when the company matches a watched ticker; this requires
      # a CIK->ticker mapping. For MVP, classify any new filing and let the
      # classifier identify affected tickers.
      summary = entry.get('summary', '')[:1500]
      link = entry.get('link', '')
      published = entry.get('published', '')
      source = 'sec:edgar'
      if seen(source, title, published):
        continue
      result = await _classify(classifier, title, summary, source)
      if result is None:
        continue
      # Only store if relevant to our watchlist OR classifier flagged material
      relevant = any(t in watchlist for t in result.affected_tickers)
      if not (result.is_material or relevant):
        continue
      store_event(
        source=source,
        ticker=result.primary_ticker or '',
        headline=title, body=summary, url=link,
        published_at=published,
        materiality='high' if result.is_material else 'low',
        category=result.category,
        affected_tickers=result.affected_tickers,
        primary_ticker=result.primary_ticker,
        directional_signal=result.directional_signal,
        urgency=result.urgency,
        classifier_reason=result.one_line_reason,
      )
      if result.is_material:
        print(f"[news_watcher][sec material] {title[:80]}", file=sys.stderr, flush=True)

    for _ in range(POLL_SEC):
      if not _running:
        return
      await asyncio.sleep(1)


async def watch_finnhub_general(classifier: Materiality_Classifier):
  """Broad market news (Fed, macro, sector). Light cadence."""
  client = FinnhubClient()
  global _running
  while _running:
    try:
      url = "https://finnhub.io/api/v1/news"
      params = {"category": "general", "token": client._api_key}
      import aiohttp
      async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
          if resp.status == 200:
            articles = await resp.json()
          else:
            articles = []
    except Exception as e:
      print(f"[news_watcher][general] failed: {e}", file=sys.stderr, flush=True)
      articles = []

    for art in (articles or [])[:MAX_GENERAL_NEWS]:
      headline = art.get('headline', '')
      summary = art.get('summary', '')
      published = datetime.fromtimestamp(art.get('datetime', 0)).isoformat()
      source = f"finnhub-general:{art.get('source', 'unknown')}"
      if seen(source, headline, published):
        continue
      result = await _classify(classifier, headline, summary, source)
      if result is None or not result.is_material:
        continue
      store_event(
        source=source, ticker=result.primary_ticker or '',
        headline=headline, body=summary, url=art.get('url', ''),
        published_at=published,
        materiality='high' if result.urgency in ('immediate', 'hours') else 'medium',
        category=result.category,
        affected_tickers=result.affected_tickers,
        primary_ticker=result.primary_ticker,
        directional_signal=result.directional_signal,
        urgency=result.urgency,
        classifier_reason=result.one_line_reason,
      )
      print(f"[news_watcher][macro] {headline[:80]}", file=sys.stderr, flush=True)

    for _ in range(POLL_MACRO):
      if not _running:
        return
      await asyncio.sleep(1)


async def main():
  init_schema()
  watchlist = get_watchlist()
  print(f"[news_watcher] Starting. Watchlist: {watchlist}", file=sys.stderr, flush=True)
  print(f"[news_watcher] Polling: finnhub-company={POLL_FINNHUB}s, "
        f"sec={POLL_SEC}s, macro={POLL_MACRO}s", file=sys.stderr, flush=True)

  classifier = Materiality_Classifier()

  await asyncio.gather(
    watch_finnhub_company_news(classifier),
    watch_sec_edgar(classifier),
    watch_finnhub_general(classifier),
    return_exceptions=True,
  )
  print("[news_watcher] Stopped.", file=sys.stderr, flush=True)


if __name__ == "__main__":
  asyncio.run(main())
