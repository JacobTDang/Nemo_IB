"""Gate 4: URL-keyed scrape cache round-trip + failure-not-cached behavior."""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.cache import Session_Cache
from tools.web_search_server.scraper import scrape_url


async def test_cache_miss_then_hit():
  cache = Session_Cache()
  cache.clear()

  url = "https://www.sec.gov/edgar/aboutedgar.htm"

  r1 = await scrape_url(url, cache=cache)
  assert r1['success'], f"first call failed: {r1.get('error')}"
  assert not r1.get('from_cache'), "First call should NOT be cached"

  r2 = await scrape_url(url, cache=cache)
  assert r2['success']
  assert r2.get('from_cache') is True, "Second call SHOULD be from cache"
  assert r2['content'] == r1['content']

  print("PASS: cache miss -> hit roundtrip works")


def test_cache_doesnt_store_failures():
  cache = Session_Cache()
  cache.clear()
  failed = {'success': False, 'url': 'http://x/y', 'error': 'whatever'}
  cache.put_scrape('http://x/y', failed)
  assert cache.get_scrape('http://x/y') is None, "Failures must not be cached"
  print("PASS: failures are never cached")


def test_table_exists():
  cache = Session_Cache()
  cache.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scrape_cache'")
  row = cache.cursor.fetchone()
  assert row is not None, "scrape_cache table not created"
  print("PASS: scrape_cache table exists")


if __name__ == "__main__":
  test_table_exists()
  asyncio.run(test_cache_miss_then_hit())
  test_cache_doesnt_store_failures()
  print("\nAll tests passed.")
