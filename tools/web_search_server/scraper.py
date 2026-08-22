"""Article scraper. Trafilatura over HTTP+HTML.

Response shape matches what agent/workflows/execution_engine.py _process_scrape expects:
    {success, url, title, content, error?, word_count, char_count, extraction_method, timestamp}
"""
import asyncio
import sys
from typing import Dict, Any, Optional
from datetime import datetime

import httpx
import trafilatura

# Crawl4AI was the JS-rendering fallback here. It requires lxml>=5.3,<6 and
# this project pins lxml==6.0.2, so it has never been installable alongside the
# rest of the dependency set -- the guarded import always took the False branch
# and every scrape went through trafilatura. Removed rather than left as a
# branch that cannot execute.

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



async def scrape_url(url: str, cache=None) -> Dict[str, Any]:
  """Scrape one URL. Tries the cache, then Trafilatura."""
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


  return {
    'success': False,
    'url': url,
    'error': 'Trafilatura returned empty or insufficient content',
  }


async def scrape_urls(urls: list, cache=None) -> list:
  """Scrape a list of URLs concurrently. Bounded concurrency to be polite to hosts."""
  sem = asyncio.Semaphore(4)

  async def _bounded(u):
    async with sem:
      return await scrape_url(u, cache=cache)

  return await asyncio.gather(*[_bounded(u) for u in urls], return_exceptions=False)
