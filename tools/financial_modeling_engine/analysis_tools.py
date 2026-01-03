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

    # parse response from model
    revenue_base = response['revenue_base']
    ebitda_margin = response['ebitda_margin']
    capex_pct_revenue = response['capex_pct_revenue']
    tax_rate = response['tax_rate']
    revenue_growth = response['revenue_growth']
    depreciation = response['depreciation']
    terminal_growth = response['terminal_growth']
    terminal_multiple = response['terminal_multiple']
    wacc = response['wacc']
    revenue_year5 = response['revenue_year5']

    # forcast future free cash flow
    fcf_projections = []
    for year, growth in enumerate(revenue_growth):
      revenue = revenue_base * (1 + growth) ** (year + 1)
      ebitda = revenue * ebitda_margin
      ebit = ebitda - (revenue * depreciation) # depends on the depreciation, for now 0.02 is temp val
      taxes = ebit * tax_rate
      nopat = ebit - taxes
      capex = revenue * capex_pct_revenue
      fcf = nopat - capex
      fcf_projections.append(fcf)

    # estimate the terminal value
    # perpetuity growth
    final_year_fcf = fcf_projections[-1]
    terminal_fcf = final_year_fcf * (1 + terminal_growth)
    terminal_value_growth = terminal_fcf / (wacc - terminal_growth)

    # exit multiple
    final_year_ebitda = revenue_year5 * ebitda_margin
    terminal_value_multiple = final_year_ebitda * terminal_multiple
    terminal_value = min(terminal_value_growth, terminal_value_multiple)

    # present value calculation
    pv_fcfs = []
    for year, fcf in enumerate(fcf_projections):
      pv = fcf / (1 + wacc) ** (year + 1)
      pv_fcfs.append(pv)
    pv_terminal = terminal_value / (1 + wacc) ** 5

    enterprise_value = sum(pv_fcfs) + pv_terminal

    # equity value bridge
    cash = data['cash'] / 1_000_000
    debt = data['totalDebt'] / 1_000_000
    shares_outstanding = data['sharesOutstanding']

    equity_value = enterprise_value + cash - debt
    price_per_share = equity_value / shares_outstanding

    # note to self dont forget to add function to tool
    return [TextContent(
      type='text',
      text=json.dumps({
          'ticker': ticker,
          'enterprise_value': enterprise_value,
          'equity_value': equity_value,
          'price_per_share': price_per_share,
          'fcf_projections': fcf_projections,
          'terminal_value': terminal_value,
          'assumptions': response
      })
    )]


if __name__ == "__main__":
  f = Financial_Analysis()
  res = asyncio.run(f.discount_cash_flow('MSFT'))
  print(res)
