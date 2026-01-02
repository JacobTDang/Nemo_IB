from typing import Any, Dict, List
import asyncio
import json
import os

from utils import get_data, calculate_percentiles
from mcp.server import Server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

class Financial_Analysis:
  def __init__(self, args= Dict):
    self.server = Server("Financial_Analysis")
    self._setup_handlers() # execute decorators

  def _setup_handlers(self):
    # Store reference to self for use in nested functions
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name= "comparable_company_analysis",
          description="method for valuing a company by looking at the prices and valuation mutiples of similar, publicly traded companies",
          inputSchema={
            "type": "object",
            "properties": {
              "companies": {
                "type": "array",
                "description": "List of comparable companies"
              }
            }
          }
        )]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == "comparable_company_analysis":
          return await parent.comparable_company_analysis(args)
      except Exception as e:
        return [TextContent(
          type="text",
          text=json.dumps({
            "success": False,
            "error": f"Failed to call tool: {str(e)}"
          })
        )]
      return [TextContent(
          type="text",
          text=json.dumps({
            "success": False,
            "error": f"Failed to call tool:"
          })
        )]

  async def comparable_company_analysis(self, args: Dict[str, Any]) -> List[TextContent]:
    # get the select universe of companies
    comparables = args["companies"]
    data = []

    # create a list of coroutine tasks to run
    tasks = [get_data(ticker) for ticker in comparables]

    # use asyncio gather to run all of the task concurrently
    # '*' unpacks the list of tasks into arguments for gather to run
    data = await asyncio.gather(*tasks)

    # build list and sort before calculating percentiles
    pe_data = await calculate_percentiles(data, 'pe_ratio')
    pb_data = await calculate_percentiles(data, 'pb_ratio')
    ev_revenue_data = await calculate_percentiles(data, 'ev_revenue')
    ev_ebitda_data = await calculate_percentiles(data, 'ev_ebitda')
    ev_ebit_data = await calculate_percentiles(data, 'ev_ebit')


    return [TextContent(
      type="text",
      text=json.dumps({
        'pe_ratio': pe_data,
        'pb_data' : pb_data,
        'ev_revenue_data' : ev_revenue_data,
        'ev_ebitda_data' : ev_ebitda_data,
        'ev_ebit_data' : ev_ebit_data
      })
    )]

if __name__ == "__main__":
  comparables = {'companies': ['AAPL', 'MSFT', 'GOOGL']}
  f = Financial_Analysis()
  res = asyncio.run(f.comparable_company_analysis(comparables))
  print(res)
