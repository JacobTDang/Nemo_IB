from typing import Any, Dict, List
import asyncio
import json
import os

from utils import search_duckduckgo, _session_manager, batch_scrape, web_scrape
from mcp.server import Server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

class WebSearchServer:
  def __init__(self):
    self.server = Server("Web_Search")
    self._setup_handlers()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="search",
          description="search the internet for information",
          inputSchema={
            "type": "object",
            'properties': {
              "ticker": {
                "type": "string",
                "description": "Ticker symbol for company to search"
              },
              "query": {
                'type': 'object',
                'description': "Search queries as key-value pairs",
                'additionalProperties': {
                  "type": "string"
                }
              }
            },
            "required": ["ticker", "query"]
          }
        ),
        Tool(
          name="get_urls_content",
          description="get content from list of urls",
          inputSchema={
            "type": "object",
            'properties': {
              'urls': {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "list of urls to gather information from"
              }
            },
            "required": ["urls"]
          }
        )]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == 'search':
          return await parent.search(args['ticker'], args['query'])
        else:
          return [TextContent(
            type ='text',
            text ='Failed to find tool'
          )]
      except Exception as e:
        return [TextContent(
          type = 'text',
          text = f'Failed to search: {str(e)}'
        )]


  async def search(self, ticker: str, query: Dict) -> List[TextContent]:
    # flexible, so we can search for multiple queries at a time
    queries = list(query.values())

    # create corotines that will run sync functions in threads
    tasks = [asyncio.to_thread(search_duckduckgo, q, 5) for q in queries]
    result_list = await asyncio.gather(*tasks)
    search_results = []

    # flatten from [[dicts, ...]] to [dicts ...]
    for result in result_list:
      search_results.extend(result)

    return [TextContent(
      type = 'text',
      text = json.dumps({
        'ticker': ticker,
        'search_result' : search_results
      })
    )]


  async def get_urls_content(self, urls: List[str]) -> List[TextContent]:
    # turn each url scrap into a corontine
    tasks = [asyncio.to_thread(web_scrape, url, 3, 1) for url in urls]
    results = await asyncio.gather(*tasks)

    _session_manager.close_all # close session for good practice
    return [TextContent(
      type="text",
      text=json.dumps({
        "results": results
      })
    )]

if __name__ == "__main__":
  w= WebSearchServer()
  res = asyncio.run(w.get_urls_content(["https://finance.yahoo.com/news/jpmorgan-backs-citigroup-c-russia-000607541.html", "https://finance.yahoo.com/news/td-synnex-corporation-snx-earnings-203223578.html"]))
  print(res)
