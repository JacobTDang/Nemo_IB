"""Gate 3: Trafilatura + Crawl4AI scraper unit test. Hits real URLs."""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.web_search_server.scraper import scrape_url, scrape_urls


async def test_trafilatura_static_page():
  url = "https://www.sec.gov/edgar/aboutedgar.htm"
  result = await scrape_url(url)
  assert result['success'], f"Failed: {result.get('error')}"
  assert result['word_count'] > 50
  assert result['extraction_method'] == 'trafilatura'
  print(f"PASS: SEC static page scraped via trafilatura, {result['word_count']} words")


async def test_invalid_url():
  result = await scrape_url("not-a-url")
  assert not result['success']
  assert 'Invalid URL' in result.get('error', '')
  print("PASS: invalid URL rejected")


async def test_concurrent_scrape():
  urls = [
    "https://www.sec.gov/edgar/aboutedgar.htm",
    "https://en.wikipedia.org/wiki/Discounted_cash_flow",
  ]
  results = await scrape_urls(urls)
  assert len(results) == 2
  succeeded = sum(1 for r in results if r.get('success'))
  assert succeeded == 2, f"Expected both to succeed; {succeeded}/2 did"
  print(f"PASS: 2-URL concurrent scrape, both succeeded")


if __name__ == "__main__":
  asyncio.run(test_trafilatura_static_page())
  asyncio.run(test_invalid_url())
  asyncio.run(test_concurrent_scrape())
  print("\nAll tests passed.")
