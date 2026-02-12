from typing import Any, Dict, List
import asyncio
import json
import sys
import traceback

from .utils import get_data, calculate_percentiles
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

market_data_description = """Retrieves real-time market data for a company from Yahoo Finance including market cap, enterprise value, revenue, EBITDA, cash, debt, shares outstanding, beta, interest expense, and valuation multiples (P/E, P/B, EV/Revenue, EV/EBITDA, EV/EBIT).
Should use: When you need current financial data for a company to perform valuation analysis, equity bridge calculations, or to get inputs for WACC calculation (beta, market cap, debt, interest expense).
Should NOT use: When you need historical time-series data or SEC filing data (use SEC tools instead)."""

wacc_tool_description = """Calculates the Weighted Average Cost of Capital (WACC) using CAPM for cost of equity. WACC = (E/V) * Cost of Equity + (D/V) * Cost of Debt * (1 - Tax Rate). Cost of Equity = Risk-Free Rate + Beta * Equity Risk Premium. This is a deterministic calculation -- you must provide all inputs.
Should use: After gathering beta and capital structure from get_market_data, tax rate from get_tax_rate, and risk-free rate + equity risk premium from web search. The WACC output feeds into calculate_dcf.
Should NOT use: If you don't have the required inputs yet. Gather beta, market cap, debt, tax rate, risk-free rate, and equity risk premium first."""

dcf_tool_description = """Calculates the intrinsic value of a company using a Discounted Cash Flow model. Takes all financial inputs as arguments and returns enterprise value, equity value, and price per share. This is a deterministic calculation -- you must provide the assumptions.
Should use: After gathering revenue base, EBITDA margin, capex %, tax rate, depreciation %, growth rates, WACC, terminal growth, terminal multiple, cash, debt, and shares outstanding from other tools.
Should NOT use: If you don't have the required financial inputs yet. Gather data first with get_market_data and SEC tools, then call this."""

comps_tool_description = """Estimates a company's value by calculating valuation multiples (e.g., EV/EBITDA, P/E) for a set of comparable public companies and applying the median multiple to the target company.
Should use: To get a quick valuation range based on current market sentiment and to see how a company is valued relative to its direct peers.
Should NOT use: When there are no truly comparable public companies, or if the entire market sector is believed to be in a bubble or a crash. This method provides a relative value, not a fundamental intrinsic value."""

class Financial_Analysis:
  def __init__(self, args=None):
    self.server = Server("Financial_Analysis")
    self._setup_handlers()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="get_market_data",
          description=market_data_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL, MSFT)"
              }
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="comparable_company_analysis",
          description=comps_tool_description,
          inputSchema={
            "type": "object",
            "properties": {
              "companies": {
                "type": "array",
                "description": "List of comparable company ticker symbols",
                "items": {"type": "string"}
              }
            },
            "required": ["companies"]
          }
        ),
        Tool(
          name="calculate_dcf",
          description=dcf_tool_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Company ticker symbol"},
              "revenue_base": {"type": "number", "description": "Current annual revenue in raw dollars (e.g. 416161000000 for $416B)"},
              "ebitda_margin": {"type": "number", "description": "EBITDA margin as decimal (e.g. 0.30 for 30%)"},
              "capex_pct_revenue": {"type": "number", "description": "Capital expenditure as % of revenue (e.g. 0.05 for 5%)"},
              "tax_rate": {"type": "number", "description": "Effective tax rate as decimal (e.g. 0.21 for 21%)"},
              "depreciation": {"type": "number", "description": "Depreciation & amortization as % of revenue (e.g. 0.02 for 2%)"},
              "revenue_growth": {"type": "array", "description": "List of 5 annual revenue growth rates as decimals (e.g. [0.10, 0.08, 0.06, 0.05, 0.04])", "items": {"type": "number"}},
              "wacc": {"type": "number", "description": "Weighted average cost of capital as decimal (e.g. 0.10 for 10%)"},
              "terminal_growth": {"type": "number", "description": "Terminal perpetuity growth rate as decimal (e.g. 0.025 for 2.5%)"},
              "terminal_multiple": {"type": "number", "description": "Terminal EV/EBITDA exit multiple (e.g. 12.0)"},
              "cash": {"type": "number", "description": "Total cash and equivalents in raw dollars (from get_market_data)"},
              "debt": {"type": "number", "description": "Total debt in raw dollars (from get_market_data)"},
              "shares_outstanding": {"type": "number", "description": "Total shares outstanding (raw number, from get_market_data)"}
            },
            "required": ["ticker", "revenue_base", "ebitda_margin", "capex_pct_revenue", "tax_rate", "depreciation", "revenue_growth", "wacc", "terminal_growth", "terminal_multiple", "cash", "debt", "shares_outstanding"]
          }
        ),
        Tool(
          name="calculate_wacc",
          description=wacc_tool_description,
          inputSchema={
            "type": "object",
            "properties": {
              "beta": {"type": "number", "description": "Company beta (from get_market_data)"},
              "risk_free_rate": {"type": "number", "description": "Risk-free rate as decimal, typically 10-year Treasury yield (e.g. 0.04 for 4%)"},
              "equity_risk_premium": {"type": "number", "description": "Equity risk premium as decimal (e.g. 0.055 for 5.5%)"},
              "cost_of_debt": {"type": "number", "description": "Pre-tax cost of debt as decimal. Calculate as interest_expense / total_debt from get_market_data (e.g. 0.05 for 5%)"},
              "tax_rate": {"type": "number", "description": "Effective tax rate as decimal (e.g. 0.21 for 21%)"},
              "market_cap": {"type": "number", "description": "Market capitalization in raw dollars (from get_market_data)"},
              "total_debt": {"type": "number", "description": "Total debt in raw dollars (from get_market_data)"}
            },
            "required": ["beta", "risk_free_rate", "equity_risk_premium", "cost_of_debt", "tax_rate", "market_cap", "total_debt"]
          }
        )
      ]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == "get_market_data":
          return await parent.get_market_data(args['ticker'])
        elif name == "comparable_company_analysis":
          return await parent.comparable_company_analysis(args['companies'])
        elif name == "calculate_dcf":
          return await parent.calculate_dcf(args)
        elif name == "calculate_wacc":
          return await parent.calculate_wacc(args)
      except Exception as e:
        return [TextContent(
          type="text",
          text=json.dumps({
            "success": False,
            "error": f"Failed to call tool '{name}': {str(e)}"
          })
        )]
      return [TextContent(
        type="text",
        text=json.dumps({
          "success": False,
          "error": f"Unknown tool: {name}"
        })
      )]

  async def get_market_data(self, ticker: str) -> List[TextContent]:
    data = await get_data(ticker)

    # Convert any numpy/pandas types to native Python for JSON serialization
    clean_data = {}
    for key, value in data.items():
      if value is None or value == 'N/A':
        clean_data[key] = None
      elif hasattr(value, 'item'):
        clean_data[key] = value.item()
      else:
        clean_data[key] = value

    return [TextContent(
      type="text",
      text=json.dumps(clean_data)
    )]

  async def comparable_company_analysis(self, comparables: List[str]) -> List[TextContent]:
    tasks = [get_data(ticker) for ticker in comparables]
    data = await asyncio.gather(*tasks)

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
        'pb_data': pb_data,
        'ev_revenue_data': ev_revenue_data,
        'ev_ebitda_data': ev_ebitda_data,
        'ev_ebit_data': ev_ebit_data
      })
    )]

  async def calculate_dcf(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = args['ticker']
    revenue_base = args['revenue_base']
    ebitda_margin = args['ebitda_margin']
    capex_pct_revenue = args['capex_pct_revenue']
    tax_rate = args['tax_rate']
    depreciation = args['depreciation']
    revenue_growth = args['revenue_growth']
    wacc = args['wacc']
    terminal_growth = args['terminal_growth']
    terminal_multiple = args['terminal_multiple']
    cash = args['cash']
    debt = args['debt']
    shares_outstanding = args['shares_outstanding']

    # Forecast future free cash flows
    fcf_projections = []
    yearly_details = []
    current_revenue = revenue_base

    for year, growth in enumerate(revenue_growth):
      current_revenue = current_revenue * (1 + growth)
      ebitda = current_revenue * ebitda_margin
      da = current_revenue * depreciation
      ebit = ebitda - da
      taxes = ebit * tax_rate
      nopat = ebit - taxes
      capex = current_revenue * capex_pct_revenue
      fcf = nopat + da - capex

      fcf_projections.append(fcf)
      yearly_details.append({
        'year': year + 1,
        'revenue': round(current_revenue, 2),
        'ebitda': round(ebitda, 2),
        'ebit': round(ebit, 2),
        'nopat': round(nopat, 2),
        'capex': round(capex, 2),
        'fcf': round(fcf, 2)
      })

    # Terminal value - perpetuity growth method
    final_year_fcf = fcf_projections[-1]
    terminal_fcf = final_year_fcf * (1 + terminal_growth)
    terminal_value_growth = terminal_fcf / (wacc - terminal_growth)

    # Terminal value - exit multiple method
    final_year_ebitda = current_revenue * ebitda_margin
    terminal_value_multiple = final_year_ebitda * terminal_multiple

    # Use the lower of the two (conservative)
    terminal_value = min(terminal_value_growth, terminal_value_multiple)

    # Present value calculation
    pv_fcfs = []
    for year, fcf in enumerate(fcf_projections):
      pv = fcf / (1 + wacc) ** (year + 1)
      pv_fcfs.append(round(pv, 2))
    pv_terminal = terminal_value / (1 + wacc) ** len(revenue_growth)

    enterprise_value = sum(pv_fcfs) + pv_terminal

    # Equity value bridge (all values in raw dollars)
    equity_value = enterprise_value + cash - debt
    price_per_share = equity_value / shares_outstanding

    return [TextContent(
      type='text',
      text=json.dumps({
        'ticker': ticker,
        'enterprise_value': round(enterprise_value, 2),
        'equity_value': round(equity_value, 2),
        'price_per_share': round(price_per_share, 2),
        'fcf_projections': yearly_details,
        'pv_fcfs': pv_fcfs,
        'pv_terminal_value': round(pv_terminal, 2),
        'terminal_value_perpetuity': round(terminal_value_growth, 2),
        'terminal_value_exit_multiple': round(terminal_value_multiple, 2),
        'terminal_value_used': round(terminal_value, 2),
        'assumptions': {
          'revenue_base': revenue_base,
          'ebitda_margin': ebitda_margin,
          'capex_pct_revenue': capex_pct_revenue,
          'tax_rate': tax_rate,
          'depreciation': depreciation,
          'revenue_growth': revenue_growth,
          'wacc': wacc,
          'terminal_growth': terminal_growth,
          'terminal_multiple': terminal_multiple,
          'cash': cash,
          'debt': debt,
          'shares_outstanding': shares_outstanding
        }
      })
    )]

  async def calculate_wacc(self, args: Dict[str, Any]) -> List[TextContent]:
    beta = args['beta']
    risk_free_rate = args['risk_free_rate']
    equity_risk_premium = args['equity_risk_premium']
    cost_of_debt = args['cost_of_debt']
    tax_rate = args['tax_rate']
    market_cap = args['market_cap']
    total_debt = args['total_debt']

    # CAPM: Cost of Equity = Risk-Free Rate + Beta * Equity Risk Premium
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Capital structure weights
    total_value = market_cap + total_debt
    equity_weight = market_cap / total_value
    debt_weight = total_debt / total_value

    # After-tax cost of debt
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

    # WACC = (E/V) * Ke + (D/V) * Kd * (1 - T)
    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)

    return [TextContent(
      type='text',
      text=json.dumps({
        'wacc': round(wacc, 6),
        'wacc_pct': f"{round(wacc * 100, 2)}%",
        'cost_of_equity': round(cost_of_equity, 6),
        'cost_of_equity_pct': f"{round(cost_of_equity * 100, 2)}%",
        'after_tax_cost_of_debt': round(after_tax_cost_of_debt, 6),
        'equity_weight': round(equity_weight, 4),
        'debt_weight': round(debt_weight, 4),
        'inputs': {
          'beta': beta,
          'risk_free_rate': risk_free_rate,
          'equity_risk_premium': equity_risk_premium,
          'cost_of_debt': cost_of_debt,
          'tax_rate': tax_rate,
          'market_cap': market_cap,
          'total_debt': total_debt
        }
      })
    )]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(read_stream, write_stream, InitializationOptions(
          server_name="financial_analysis",
          server_version='1.0.0',
          capabilities=ServerCapabilities()
        ))
    except Exception as e:
      print(f"Financial Analysis Server error: {e}", file=sys.stderr, flush=True)
      traceback.print_exc(file=sys.stderr)
      raise


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.financial_modeling_engine.analysis_tools server", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "server":
    print("Starting Financial Analysis Server", file=sys.stderr, flush=True)
    try:
      server = Financial_Analysis()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.financial_modeling_engine.analysis_tools server", file=sys.stderr)
