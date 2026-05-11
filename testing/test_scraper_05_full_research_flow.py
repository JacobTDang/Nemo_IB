"""Gate 6 (scraper-only): full research flow.

Exercise the new search + scrape stack against a real research query that
structured tools (Finnhub/SEC/FRED/yfinance) cannot answer. Proves SearXNG
returns URLs, Trafilatura extracts content, cache populates, and the response
shapes match the execution_engine contract end-to-end.

Test query: "What did Tim Cook say about Apple's AI strategy in 2025?" — a
qualitative question only available via web articles, not via structured APIs.
"""
import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.web_search_server.web_search import WebSearchServer
from agent.cache import Session_Cache


async def test_full_research_flow():
  server = WebSearchServer()
  server.cache.clear()  # fresh state

  # Step 1: search
  search_result = await server.search("AAPL", {"q1": "Tim Cook AI strategy 2025"})
  body = json.loads(search_result[0].text)
  urls = [r['link'] for r in body['search_result'][:3]]
  assert len(urls) >= 1, "search returned no URLs"
  print(f"PASS: searxng search returned {len(body['search_result'])} URLs")
  print(f"      First 3 URLs to scrape:")
  for u in urls:
    print(f"        {u}")

  # Step 2: scrape (cold cache)
  scrape_result = await server.get_urls_content(urls)
  body = json.loads(scrape_result[0].text)
  successes = [r for r in body['results'] if r.get('success')]
  assert len(successes) >= 1, f"scrape: 0/{len(urls)} succeeded"
  print(f"PASS: scrape {len(successes)}/{len(urls)} URLs successful (cold cache)")
  for r in successes:
    print(f"      {r['extraction_method']:11} {r['word_count']:5}w  {r['url'][:70]}")

  # Step 3: scrape same URLs again (warm cache — should be instant)
  import time
  t0 = time.time()
  scrape_result2 = await server.get_urls_content(urls)
  elapsed = time.time() - t0
  body2 = json.loads(scrape_result2[0].text)
  cache_hits = sum(1 for r in body2['results'] if r.get('from_cache'))
  successes2 = [r for r in body2['results'] if r.get('success')]
  assert cache_hits == len(successes), \
    f"expected {len(successes)} cache hits, got {cache_hits}"
  assert elapsed < 2.0, f"warm cache should be <2s; took {elapsed:.2f}s"
  print(f"PASS: warm cache: {cache_hits}/{len(urls)} hits, took {elapsed*1000:.0f}ms (vs cold ~3-10s)")

  # Step 4: confirm content is real (not boilerplate)
  total_chars = sum(r.get('char_count', 0) for r in successes)
  assert total_chars > 1000, f"total content too small: {total_chars} chars"
  print(f"PASS: total content extracted: {total_chars:,} chars across {len(successes)} pages")


if __name__ == "__main__":
  asyncio.run(test_full_research_flow())
  print("\nAll tests passed.")
