from typing import Any, Dict, List
import asyncio
import json
import os
from datetime import date, datetime
import os
# NOTE: ClientSession, StdioServerParameters, stdio_client are imported only in client mode below
from tools.web_search_server.webscraper_utils import search_duckduckgo, _session_manager, batch_scrape, web_scrape
from tools.web_search_server.sec_utils import (
    get_revenue_base, get_ebitda_margin, get_capex_pct_revenue,
    get_tax_rate, get_depreciation, get_disclosures_names,
    extract_disclosure_data, get_latest_filing
)
import importlib.util
import sys

# import module with number in name
try:
    spec = importlib.util.spec_from_file_location(
        "filing_parser",
        "tools/web_search_server/8K_and_DEF14A_utils.py"
    )
    if spec and spec.loader:
        filing_parser = importlib.util.module_from_spec(spec)
        sys.modules["filing_parser"] = filing_parser
        spec.loader.exec_module(filing_parser)
    else:
        raise ImportError("Could not load filing parser module")
except Exception:
    # fallback - set filing_parser to None and handle in methods
    filing_parser = None

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

def json_serializer(obj):
  """JSON serializer for objects not serializable by default json code"""
  if isinstance(obj, (date, datetime)):
    return obj.isoformat()
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def safe_json_dumps(obj):
  """Safely serialize objects to JSON, handling dates and other non-serializable types"""
  return json.dumps(obj, default=json_serializer)

class WebSearchServer:
  def __init__(self):
    self.server = Server("web_client")
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
        ),
        # SEC XBRL Tools
        Tool(
          name="get_revenue_base",
          description="Get company's primary revenue from SEC filings (10-K/10-Q)",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_ebitda_margin",
          description="Calculate EBITDA margin from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_capex_pct_revenue",
          description="Get capital expenditures as percentage of revenue",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_tax_rate",
          description="Get effective tax rate from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_depreciation",
          description="Get depreciation & amortization as percentage of revenue",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_disclosures_names",
          description="Get list of available disclosure names from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_disclosure_data",
          description="Extract specific disclosure data from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "disclosure_name": {"type": "string", "description": "Name of disclosure to extract"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker", "disclosure_name"]
          }
        ),
        Tool(
          name="get_latest_filing",
          description="Get metadata and raw access to latest SEC filing",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        # SEC Filing Parser Tools (8-K and Proxy Analysis)
        Tool(
          name="extract_8k_events",
          description="Extract material corporate events from 8-K filings for due diligence",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "limit": {"type": "integer", "description": "Max filings to process", "default": 10},
              "debug": {"type": "boolean", "description": "Print debug output", "default": False}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_proxy_compensation",
          description="Analyze executive compensation from proxy filings (DEF 14A)",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "debug": {"type": "boolean", "description": "Print debug output", "default": False}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_governance_data",
          description="Extract board composition and independence from proxy filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "debug": {"type": "boolean", "description": "Print debug output", "default": False}
            },
            "required": ["ticker"]
          }
        )]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == 'search':
          result = await parent.search(args['ticker'], args['query'])
          return result
        elif name == 'get_urls_content':
          return await parent.get_urls_content(args['urls'])

        # SEC XBRL Tools
        elif name == 'get_revenue_base':
          result = await parent.get_revenue_base(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_ebitda_margin':
          result = await parent.get_ebitda_margin(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_capex_pct_revenue':
          result = await parent.get_capex_pct_revenue(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_tax_rate':
          result = await parent.get_tax_rate(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_depreciation':
          return await parent.get_depreciation(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'get_disclosures_names':
          return await parent.get_disclosures_names(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'extract_disclosure_data':
          return await parent.extract_disclosure_data(args['ticker'], args['disclosure_name'], args.get('form_type', '10-K'))
        elif name == 'get_latest_filing':
          result = await parent.get_latest_filing(args['ticker'], args.get('form_type', '10-K'))
          return result

        # SEC Filing Parser Tools
        elif name == 'extract_8k_events':
          return await parent.extract_8k_events(args['ticker'], args.get('limit', 10), args.get('debug', False))
        elif name == 'extract_proxy_compensation':
          return await parent.extract_proxy_compensation(args['ticker'], args.get('debug', False))
        elif name == 'extract_governance_data':
          return await parent.extract_governance_data(args['ticker'], args.get('debug', False))
        else:
          return [TextContent(
            type='text',
            text=f'Unknown tool: {name}'
          )]
      except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return [TextContent(
          type='text',
          text=f'Failed to execute {name}: {str(e)}'
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
      text = safe_json_dumps({
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
      text=safe_json_dumps({
        "results": results
      })
    )]

  # SEC XBRL Tools
  async def get_revenue_base(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_revenue_base, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_ebitda_margin(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_ebitda_margin, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_capex_pct_revenue(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_capex_pct_revenue, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_tax_rate(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_tax_rate, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_depreciation(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_depreciation, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_disclosures_names(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_disclosures_names, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_disclosure_data(self, ticker: str, disclosure_name: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(extract_disclosure_data, ticker, disclosure_name, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_latest_filing(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    # Note: This returns filing metadata, not the full filing object (which isn't JSON serializable)
    result = await asyncio.to_thread(get_latest_filing, ticker, form_type)
    if result:
      # Convert to JSON-serializable format
      json_result = {
        'ticker': ticker,
        'form_type': form_type,
        'filing_date': str(result.get('filing_date')),
        'url': result.get('url'),
        'accession_number': result.get('accession_number'),
        'has_xbrl_data': result.get('xbrl_data') is not None,
        'success': True
      }
    else:
      json_result = {
        'ticker': ticker,
        'form_type': form_type,
        'error': 'No filing found',
        'success': False
      }
    return [TextContent(type="text", text=safe_json_dumps(json_result))]

  # SEC Filing Parser Tools
  async def extract_8k_events(self, ticker: str, limit: int = 10, debug: bool = False) -> List[TextContent]:
    if filing_parser is None:
      return [TextContent(type="text", text=safe_json_dumps({"error": "Filing parser not available", "success": False}))]
    result = await asyncio.to_thread(filing_parser.extract_8k_events, ticker, limit, debug)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_proxy_compensation(self, ticker: str, debug: bool = False) -> List[TextContent]:
    if filing_parser is None:
      return [TextContent(type="text", text=safe_json_dumps({"error": "Filing parser not available", "success": False}))]
    result = await asyncio.to_thread(filing_parser.extract_proxy_compensation, ticker, debug)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_governance_data(self, ticker: str, debug: bool = False) -> List[TextContent]:
    if filing_parser is None:
      return [TextContent(type="text", text=safe_json_dumps({"error": "Filing parser not available", "success": False}))]
    result = await asyncio.to_thread(filing_parser.extract_governance_data, ticker, debug)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(read_stream, write_stream, InitializationOptions(
          server_name="web_client",
          server_version='1.0.0',
          capabilities=ServerCapabilities()
        ))
        print("Successfully created web_client process", file=sys.stderr, flush=True)
    except Exception as e:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.web_search_server.web_search server", file=sys.stderr)
    sys.exit(1)

  system_args = sys.argv[1]

  if system_args == "server":
    print("Starting web_client process", file=sys.stderr, flush=True)
    try:
      server = WebSearchServer()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {system_args}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.web_search_server.web_search server", file=sys.stderr)
    sys.exit(1)
