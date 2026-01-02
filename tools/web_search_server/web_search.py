from typing import Any, Dict, List
import asyncio
import json
import os

from mcp.server import Server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

class WebSearchServer:
  def __init__(self):
    self.server = Server("Web_Search")
    self._setup_handers()

  def _setup_handers(self):
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
                "type": "String",
                "description": "Ticker symbol for company to search"
              }
            }
          }
        )]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == 'search':
          return await parent.search(args['ticker'])

      except Exception as e:
        return [TextContent(
          type = 'text',
          text = f'Failed to search: {str(e)}'
        )]


  async def search(self, ticker: str) -> List[TextContent]:

    return [TextContent(
      type = 'text',
      text = 'TODO'
    )]
