"""Gate 2: SearXNG client unit test. Requires `docker compose up -d searxng` first."""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.web_search_server.searxng_client import searxng_search


async def test_basic_query():
  results = await searxng_search("apple earnings 2025", max_results=5)
  assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
  for r in results:
    assert 'title' in r and 'link' in r and 'snippet' in r
    assert r['link'].startswith('http')
  print(f"PASS: searxng returned {len(results)} results")


async def test_zero_result_query():
  results = await searxng_search("zzzqqxxnonexistentquery98765", max_results=5)
  assert isinstance(results, list)
  print(f"PASS: empty/edge query handled, got {len(results)} results")


async def test_container_down_handled():
  import tools.web_search_server.searxng_client as mod
  original = mod.SEARXNG_URL
  mod.SEARXNG_URL = "http://localhost:1"  # will refuse connection
  try:
    results = await searxng_search("anything")
    assert results == [], "Expected empty list on connection failure"
    print("PASS: graceful failure when SearXNG is down")
  finally:
    mod.SEARXNG_URL = original


if __name__ == "__main__":
  asyncio.run(test_basic_query())
  asyncio.run(test_zero_result_query())
  asyncio.run(test_container_down_handled())
  print("\nAll tests passed.")
