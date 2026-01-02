from typing import Any, Dict, List
import asyncio
import json
import os

from agent import FinancialAnalysisAgent

from .utils import get_data, calculate_percentiles
from mcp.server import Server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

dcf_tool_description = """Calculates the intrinsic value of a company based on the present value of its future cash flows (Discounted Cash Flow analysis). This is a detailed, fundamental valuation method.
Should use: When you have access to multi-year financial forecasts and want to determine a company's value based on its core cash-generating ability, independent of current market sentiment.
Should NOT use: For companies with highly unpredictable cash flows (e.g., pre-revenue startups) or when detailed forecasts are unavailable. The valuation is very sensitive to assumptions about future growth and risk."""

comps_tool_description = """Estimates a company's value by calculating valuation multiples (e.g., EV/EBITDA, P/E) for a set of comparable public companies and applying the median multiple to the target company.
Should use: To get a quick valuation range based on current market sentiment and to see how a company is valued relative to its direct peers.
Should NOT use: When there are no truly comparable public companies, or if the entire market sector is believed to be in a bubble or a crash. This method provides a relative value, not a fundamental intrinsic value."""

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
          description=comps_tool_description,
          inputSchema={
            "type": "object",
            "properties": {
              "companies": {
                "type": "List",
                "description": "List of comparable companies"
              }
            }
          }
        ),
        Tool(
          name= 'discounted_cash_flow_analysis',
          description=dcf_tool_description,
          inputSchema={
            "type": "object",
            "properties": {

            }
          }
        )]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == "comparable_company_analysis":
          return await parent.comparable_company_analysis(args['companies'])
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

  async def comparable_company_analysis(self, args: List[str]) -> List[TextContent]:
    # get the select universe of companies
    comparables = args
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
        'comparables': comparables,
        'pe_ratio': pe_data,
        'pb_data' : pb_data,
        'ev_revenue_data' : ev_revenue_data,
        'ev_ebitda_data' : ev_ebitda_data,
        'ev_ebit_data' : ev_ebit_data
      })
    )]

  async def discount_cash_flow(self, ticker: str) -> List[TextContent]:

    # create assumptions with agent
    dcf_analyst_agent = FinancialAnalysisAgent()
    data = await get_data(ticker)
    response = await dcf_analyst_agent.generate_assumptions(ticker,data)
    print(response)

    # forcast future free cash flow

    # estimate the terminal value

    # discount everything


    # note to self dont forget to add function to tool
    return [TextContent(
      type='text',
      text='TODO: Implement method'
    )]


if __name__ == "__main__":
  f = Financial_Analysis()
  res = asyncio.run(f.discount_cash_flow('MSFT'))
  print(res)
