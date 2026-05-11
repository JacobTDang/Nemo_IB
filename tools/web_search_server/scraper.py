"""Article scraper. Trafilatura primary (HTTP+HTML); Crawl4AI fallback (Playwright).

Response shape matches what agent/workflows/execution_engine.py _process_scrape expects:
    {success, url, title, content, error?, word_count, char_count, extraction_method, timestamp}
"""
import asyncio
import sys
from typing import Dict, Any, Optional
from datetime import datetime

import httpx
import trafilatura

try:
  from crawl4ai import AsyncWebCrawler
  _CRAWL4AI_AVAILABLE = True
except ImportError:
  _CRAWL4AI_AVAILABLE = False

_DEFAULT_TIMEOUT = 20.0

_HTTP_HEADERS = {
  "User-Agent": "Mozilla/5.0 (compatible; NemoFinancialAgent/1.0; +https://github.com/JacobTDang/Nemo_IB)",
  "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.9",
}
_SEC_HEADERS = {
  **_HTTP_HEADERS,
  # SEC requires a contact email in the User-Agent for programmatic access
  "User-Agent": "NemoFinancialAgent/1.0 (usosempai@gmail.com)",
}


def _headers_for(url: str) -> Dict[str, str]:
  return _SEC_HEADERS if "sec.gov" in url else _HTTP_HEADERS


async def _fetch_with_trafilatura(url: str) -> Optional[Dict[str, Any]]:
  """Fast path: HTTP GET + Trafilatura extraction. Returns None on extraction failure."""
  try:
    async with httpx.AsyncClient(follow_redirects=True, timeout=_DEFAULT_TIMEOUT) as client:
      response = await client.get(url, headers=_headers_for(url))
      response.raise_for_status()
      html = response.text
  except Exception as e:
    print(f"  [Trafilatura] HTTP fetch failed for {url}: {e}", file=sys.stderr, flush=True)
    return None

  extracted = trafilatura.extract(
    html,
    output_format='markdown',
    include_comments=False,
    include_tables=True,
    favor_recall=True,
    with_metadata=False,
  )
  if not extracted or len(extracted) < 100:
    return None

  try:
    meta = trafilatura.extract_metadata(html)
    title = meta.title if (meta and meta.title) else url
  except Exception:
    title = url

  return {
    'success': True,
    'url': url,
    'title': title,
    'content': extracted,
    'word_count': len(extracted.split()),
    'char_count': len(extracted),
    'extraction_method': 'trafilatura',
    'timestamp': datetime.now().isoformat(),
  }


async def _fetch_with_crawl4ai(url: str) -> Optional[Dict[str, Any]]:
  """Fallback: Crawl4AI with Playwright for JS-rendered content."""
  if not _CRAWL4AI_AVAILABLE:
    return None
  try:
    async with AsyncWebCrawler(verbose=False) as crawler:
      result = await crawler.arun(url=url, timeout=30000)
      if not (result and result.success and result.markdown):
        return None
      md = result.markdown
      if isinstance(md, str) and len(md) < 100:
        return None
      title = url
      if result.metadata and isinstance(result.metadata, dict):
        title = result.metadata.get('title', url) or url
      return {
        'success': True,
        'url': url,
        'title': title,
        'content': md,
        'word_count': len(md.split()) if isinstance(md, str) else 0,
        'char_count': len(md) if isinstance(md, str) else 0,
        'extraction_method': 'crawl4ai',
        'timestamp': datetime.now().isoformat(),
      }
  except Exception as e:
    print(f"  [Crawl4AI] Fallback failed for {url}: {e}", file=sys.stderr, flush=True)
    return None


async def scrape_url(url: str, cache=None) -> Dict[str, Any]:
  """Scrape one URL. Tries cache, then Trafilatura, then Crawl4AI."""
  # Cache hit
  if cache is not None:
    cached = cache.get_scrape(url)
    if cached:
      print(f"  [Scrape Cache HIT] {url[:80]}", file=sys.stderr, flush=True)
      cached['from_cache'] = True
      return cached

  # Validate URL
  if not url or not (url.startswith('http://') or url.startswith('https://')):
    return {'success': False, 'url': url, 'error': f'Invalid URL: {url}'}

  # Primary: Trafilatura
  result = await _fetch_with_trafilatura(url)
  if result:
    print(f"  [Trafilatura OK] {url[:60]} ({result['word_count']} words)",
          file=sys.stderr, flush=True)
    if cache is not None:
      cache.put_scrape(url, result)
    return result

  # Fallback: Crawl4AI
  print(f"  [Trafilatura empty] Falling back to Crawl4AI for {url[:60]}",
        file=sys.stderr, flush=True)
  result = await _fetch_with_crawl4ai(url)
  if result:
    print(f"  [Crawl4AI OK] {url[:60]} ({result['word_count']} words)",
          file=sys.stderr, flush=True)
    if cache is not None:
      cache.put_scrape(url, result)
    return result

  return {
    'success': False,
    'url': url,
    'error': 'Both Trafilatura and Crawl4AI returned empty/insufficient content',
  }


async def scrape_urls(urls: list, cache=None) -> list:
  """Scrape a list of URLs concurrently. Bounded concurrency to be polite to hosts."""
  sem = asyncio.Semaphore(4)

  async def _bounded(u):
    async with sem:
      return await scrape_url(u, cache=cache)

  return await asyncio.gather(*[_bounded(u) for u in urls], return_exceptions=False)
