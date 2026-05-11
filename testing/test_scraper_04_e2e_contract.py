"""Gate 5: end-to-end response shape contract for execution_engine.

The execution engine (_process_search and _process_scrape in
agent/workflows/execution_engine.py) expects specific keys. This test verifies
the new WebSearchServer.search() and .get_urls_content() preserve that contract.
"""
import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.web_search_server.web_search import WebSearchServer


async def test_search_response_shape():
  server = WebSearchServer()
  result = await server.search("AAPL", {"q1": "earnings 2025"})
  body = json.loads(result[0].text)
  assert 'ticker' in body and body['ticker'] == 'AAPL'
  assert 'search_result' in body and isinstance(body['search_result'], list)
  if body['search_result']:
    item = body['search_result'][0]
    for key in ('title', 'link', 'snippet'):
      assert key in item, f"missing '{key}' in search_result item"
  print(f"PASS: search() shape matches execution_engine contract "
        f"({len(body['search_result'])} results)")


async def test_scrape_response_shape():
  server = WebSearchServer()
  result = await server.get_urls_content(["https://www.sec.gov/edgar/aboutedgar.htm"])
  body = json.loads(result[0].text)
  assert 'results' in body and isinstance(body['results'], list)
  assert len(body['results']) == 1
  item = body['results'][0]
  assert 'success' in item and 'url' in item
  if item['success']:
    assert 'content' in item and 'title' in item
  print(f"PASS: get_urls_content() shape matches execution_engine contract "
        f"(success={item.get('success')})")


if __name__ == "__main__":
  asyncio.run(test_search_response_shape())
  asyncio.run(test_scrape_response_shape())
  print("\nAll tests passed.")
