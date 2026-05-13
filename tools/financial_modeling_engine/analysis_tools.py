from typing import Any, Dict, List, Optional
import asyncio
import json
import sys
import traceback

from .utils import get_data, calculate_percentiles
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions


# ---------------------------------------------------------------------------
# Tool descriptions
# These are the single source of truth for what each tool needs.
# Both the Probing Agent (to surface data requirements) and the
# Financial Modeling Agent (to decide what to run) read these at runtime.
# ---------------------------------------------------------------------------

market_data_description = """Retrieves real-time market data for a company from Yahoo Finance including market cap, enterprise value, revenue, EBITDA, cash, debt, shares outstanding, beta, interest expense, and valuation multiples (P/E, P/B, EV/Revenue, EV/EBITDA, EV/EBIT).
Should use: When you need current financial data for a company to perform valuation analysis, equity bridge calculations, or to get inputs for WACC calculation (beta, market cap, debt, interest expense).
Should NOT use: When you need historical time-series data or SEC filing data (use SEC tools instead)."""

wacc_tool_description = """Calculates Weighted Average Cost of Capital (WACC) using CAPM. Formula: WACC = (E/V)*Ke + (D/V)*Kd*(1-T).

PASS NO ARGUMENTS. The execution engine automatically resolves all inputs (beta, risk_free_rate, cost_of_debt, tax_rate, market_cap, total_debt) from the variable store populated by prior tools. Plan it as: {"tool": "calculate_wacc", "arguments": {}}.

Required prior tools (must have already run):
- get_market_data(ticker)   -> beta, market_cap, total_debt, cost_of_debt
- get_tax_rate(ticker)      -> tax_rate
- get_macro_snapshot()      -> risk_free_rate (10Y Treasury)

Should NOT use: If get_market_data, get_tax_rate, or get_macro_snapshot have not yet run."""

dcf_tool_description = """Calculates intrinsic equity value using a 5-year free cash flow DCF model. Returns enterprise value, equity value, and price per share.

PASS ONLY TICKER. The execution engine automatically resolves ALL other inputs from the variable store. Plan it as: {"tool": "calculate_dcf", "arguments": {"ticker": "AAPL"}}. Never pass any numeric arguments -- they are populated automatically from prior tool results.

THIS IS THE LAST TOOL TO RUN. All of the following must have already executed before planning calculate_dcf:
1. get_revenue_base(ticker)        -> revenue_base
2. get_ebitda_margin(ticker)       -> ebitda_margin
3. get_capex_pct_revenue(ticker)   -> capex_pct_revenue
4. get_depreciation(ticker)        -> depreciation
5. get_tax_rate(ticker)            -> tax_rate
6. get_market_data(ticker)         -> cash, debt, shares_outstanding
7. get_basic_financials(ticker)    -> revenue_growth, terminal_multiple
8. get_macro_snapshot()            -> risk_free_rate, terminal_growth
9. calculate_wacc()                -> wacc"""

comps_tool_description = """Estimates a company's value by calculating valuation multiples (e.g., EV/EBITDA, P/E) for a set of comparable public companies and applying the median multiple to the target company.
Should use: To get a quick valuation range based on current market sentiment and to see how a company is valued relative to its direct peers.
Should NOT use: When there are no truly comparable public companies, or if the entire market sector is believed to be in a bubble or a crash. This method provides a relative value, not a fundamental intrinsic value."""

scenario_dcf_description = """Runs the DCF model under three scenarios (Bear / Base / Bull) using different revenue growth rates and EBITDA margin assumptions. Returns a price-per-share range across all three cases.

MODELING PHASE ONLY -- called by the Financial Modeling Agent, not the execution engine.

Data that must be in the variable store before modeling:
- revenue_base          <- get_revenue_base
- ebitda_margin         <- get_ebitda_margin
- capex_pct_revenue     <- get_capex_pct_revenue
- depreciation          <- get_depreciation
- tax_rate              <- get_tax_rate
- cash, debt, shares_outstanding, beta  <- get_market_data
- risk_free_rate        <- get_macro_snapshot
- revenueGrowthTTMYoy, evEbitdaTTM      <- get_basic_financials
- forward estimate low/high revenue     <- get_forward_estimates (for bear/bull anchors)
- historical income statement           <- get_financial_statements ic annual (for margin trend)"""

lbo_description = """Leveraged buyout model. Computes IRR and MOIC to equity given entry EV, debt structure (leverage turns, interest rate), and exit multiple over a hold period.

MODELING PHASE ONLY -- called by the Financial Modeling Agent, not the execution engine.
ONLY run when the user query involves M&A, private equity, buyout potential, or takeout analysis.

Data that must be in the variable store before modeling:
- market_cap, totalDebt, totalCash, shares_outstanding  <- get_market_data
- revenue_base          <- get_revenue_base
- ebitda_margin         <- get_ebitda_margin
- capex_pct_revenue     <- get_capex_pct_revenue
- depreciation          <- get_depreciation
- tax_rate              <- get_tax_rate
- revenueGrowthTTMYoy   <- get_basic_financials (hold period growth)
- risk_free_rate        <- get_macro_snapshot
- HY credit spread      <- get_credit_spreads (for debt pricing)
- evEbitdaTTM           <- get_basic_financials (exit multiple anchor)"""

credit_profile_description = """Computes key credit metrics: Net Debt/EBITDA, Interest Coverage (EBIT/Interest), leverage label (Investment Grade through Distressed), and FCF yield.

MODELING PHASE ONLY -- called by the Financial Modeling Agent, not the execution engine.

Data that must be in the variable store before modeling:
- totalDebt, totalCash, interest_expense, market_cap  <- get_market_data
- revenue_base          <- get_revenue_base
- ebitda_margin         <- get_ebitda_margin
- depreciation          <- get_depreciation
- capex_pct_revenue     <- get_capex_pct_revenue
- tax_rate              <- get_tax_rate"""

capital_returns_description = """Calculates shareholder return profile: FCF yield, dividend yield, buyback yield, total shareholder yield, and payout ratio sustainability.

MODELING PHASE ONLY -- called by the Financial Modeling Agent, not the execution engine.

Data that must be in the variable store before modeling:
- market_cap, shares_outstanding          <- get_market_data
- revenue_base, ebitda_margin             <- get_revenue_base, get_ebitda_margin
- capex_pct_revenue, depreciation         <- get_capex_pct_revenue, get_depreciation
- tax_rate                                <- get_tax_rate
- dividendsPaid, repurchaseOfCapitalStock <- get_financial_statements cf annual"""


# ---------------------------------------------------------------------------
# Pure math functions
# Module-level so Financial_Modeling_Agent can import them directly without
# going through the MCP layer. MCP handlers call these same functions.
# ---------------------------------------------------------------------------

def _to_native(obj):
  """Recursively convert numpy/pandas types to Python natives for JSON serialization."""
  if isinstance(obj, dict):
    return {k: _to_native(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_to_native(v) for v in obj]
  if hasattr(obj, 'item'):
    return obj.item()
  return obj


def _dcf_math(revenue_base: float, ebitda_margin: float, capex_pct_revenue: float,
              tax_rate: float, depreciation: float, revenue_growth: list,
              wacc: float, terminal_growth: float, terminal_multiple: float,
              cash: float, debt: float, shares_outstanding: float,
              ticker: str = '') -> dict:
  """5-year FCF DCF model. All margin/rate inputs as decimals."""
  # Normalize percent-form inputs to decimal (defensive guard)
  if ebitda_margin > 1:
    ebitda_margin /= 100
  if capex_pct_revenue > 1:
    capex_pct_revenue /= 100
  if tax_rate > 1:
    tax_rate /= 100
  if depreciation > 1:
    depreciation /= 100

  if terminal_growth == 0:
    terminal_growth = 0.025  # GDP-match default

  # FCF projections
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

  # Terminal value -- perpetuity growth method
  final_year_fcf = fcf_projections[-1]
  terminal_fcf = final_year_fcf * (1 + terminal_growth)
  perpetuity_spread = wacc - terminal_growth
  terminal_value_warning = None
  if perpetuity_spread <= 0.01:
    terminal_value_warning = (
      f"WARNING: wacc ({wacc:.4f}) is too close to terminal_growth ({terminal_growth:.4f}). "
      f"Spread clamped to 1% to prevent formula instability."
    )
    perpetuity_spread = 0.01
  terminal_value_growth = terminal_fcf / perpetuity_spread

  # Terminal value -- exit multiple method
  final_year_ebitda = current_revenue * ebitda_margin
  terminal_value_multiple = final_year_ebitda * terminal_multiple

  # Conservative convention: use lower of the two when both are available
  if terminal_multiple > 0:
    terminal_value = min(terminal_value_growth, terminal_value_multiple)
  else:
    terminal_value = terminal_value_growth

  # Present values
  pv_fcfs = [round(fcf / (1 + wacc) ** (i + 1), 2) for i, fcf in enumerate(fcf_projections)]
  pv_terminal = terminal_value / (1 + wacc) ** len(revenue_growth)

  enterprise_value = sum(pv_fcfs) + pv_terminal
  equity_value = enterprise_value + cash - debt
  price_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

  output = {
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
    'wacc_minus_tg_spread': round(perpetuity_spread, 4),
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
  }
  if terminal_value_warning:
    output['warning'] = terminal_value_warning
  return output


def _wacc_math(beta: float, risk_free_rate: float, equity_risk_premium: float = 0.06,
               cost_of_debt: float = 0, tax_rate: float = 0,
               market_cap: float = 0, total_debt: float = 0) -> dict:
  """WACC via CAPM. All rate inputs as decimals."""
  if equity_risk_premium > 1:
    equity_risk_premium /= 100
  if risk_free_rate > 1:
    risk_free_rate /= 100
  if cost_of_debt > 1:
    cost_of_debt /= 100
  if tax_rate > 1:
    tax_rate /= 100

  cost_of_equity = risk_free_rate + beta * equity_risk_premium

  total_value = market_cap + total_debt
  if total_value == 0:
    return {'error': 'market_cap + total_debt is zero, cannot compute WACC'}

  equity_weight = market_cap / total_value
  debt_weight = total_debt / total_value
  after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
  wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)

  return {
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
  }


def _lbo_math(entry_ev: float, revenue_base: float, ebitda_margin: float,
              capex_pct_revenue: float, depreciation: float, tax_rate: float,
              revenue_growth: list, debt_interest_rate: float,
              leverage_turns: float, exit_multiple: float,
              hold_years: int = 5) -> dict:
  """
  Simplified LBO model with cash sweep debt paydown.

  entry_ev: total acquisition EV (market cap + net debt + entry premium)
  leverage_turns: acquisition debt as multiple of entry EBITDA (e.g. 4.5)
  debt_interest_rate: all-in interest rate on acquisition debt (decimal, e.g. 0.08)
  exit_multiple: EV/EBITDA at exit
  hold_years: investment horizon (default 5)

  Returns IRR and MOIC assuming all FCF sweeps debt and no interim equity distributions.
  """
  # Normalize
  if ebitda_margin > 1:
    ebitda_margin /= 100
  if capex_pct_revenue > 1:
    capex_pct_revenue /= 100
  if depreciation > 1:
    depreciation /= 100
  if tax_rate > 1:
    tax_rate /= 100
  if debt_interest_rate > 1:
    debt_interest_rate /= 100

  entry_ebitda = revenue_base * ebitda_margin
  debt_amount = entry_ebitda * leverage_turns
  equity_invested = max(entry_ev - debt_amount, entry_ev * 0.10)  # floor: 10% equity

  current_revenue = revenue_base
  current_debt = debt_amount
  year_by_year = []

  for yr in range(hold_years):
    growth = revenue_growth[yr] if yr < len(revenue_growth) else (revenue_growth[-1] if revenue_growth else 0.03)
    current_revenue *= (1 + growth)
    ebitda = current_revenue * ebitda_margin
    da = current_revenue * depreciation
    ebit = ebitda - da
    interest = current_debt * debt_interest_rate
    taxable = max(0.0, ebit - interest)
    taxes = taxable * tax_rate
    capex = current_revenue * capex_pct_revenue
    # Cash available after interest, taxes, capex -- sweeps to debt
    fcf_after_service = ebitda - capex - taxes - interest
    debt_paydown = max(0.0, fcf_after_service)
    current_debt = max(0.0, current_debt - debt_paydown)

    year_by_year.append({
      'year': yr + 1,
      'revenue': round(current_revenue, 2),
      'ebitda': round(ebitda, 2),
      'interest': round(interest, 2),
      'taxes': round(taxes, 2),
      'fcf_after_service': round(fcf_after_service, 2),
      'debt_paydown': round(debt_paydown, 2),
      'debt_remaining': round(current_debt, 2),
    })

  exit_ebitda = current_revenue * ebitda_margin
  exit_ev = exit_ebitda * exit_multiple
  equity_proceeds = max(0.0, exit_ev - current_debt)

  moic = equity_proceeds / equity_invested if equity_invested > 0 else 0
  # IRR: single cash-on-cash (no interim distributions) -- MOIC^(1/N) - 1
  irr = (moic ** (1.0 / hold_years) - 1) if moic > 0 else -1.0

  return {
    'entry_ev': round(entry_ev, 2),
    'entry_ebitda': round(entry_ebitda, 2),
    'entry_multiple': round(entry_ev / entry_ebitda, 2) if entry_ebitda > 0 else 0,
    'debt_amount': round(debt_amount, 2),
    'equity_invested': round(equity_invested, 2),
    'leverage_turns_entry': round(leverage_turns, 2),
    'exit_ebitda': round(exit_ebitda, 2),
    'exit_ev': round(exit_ev, 2),
    'exit_multiple': exit_multiple,
    'debt_at_exit': round(current_debt, 2),
    'equity_proceeds': round(equity_proceeds, 2),
    'moic': round(moic, 2),
    'irr_pct': round(irr * 100, 2),
    'achieves_20pct_irr': irr >= 0.20,
    'hold_years': hold_years,
    'assumptions': {
      'debt_interest_rate': debt_interest_rate,
      'exit_multiple': exit_multiple,
      'leverage_turns': leverage_turns,
    },
    'year_by_year': year_by_year,
  }


def _credit_profile_math(total_debt: float, cash: float, ebitda: float,
                          interest_expense: float, depreciation_abs: float,
                          capex_abs: float, tax_rate: float,
                          market_cap: float = 0) -> dict:
  """
  Key credit metrics from capital structure and income data.

  depreciation_abs: D&A in raw dollars (revenue * depreciation_pct)
  capex_abs: CapEx in raw dollars (revenue * capex_pct_revenue)
  """
  # Fail-fast: a missing fundamental (ebitda) silently produces garbage like
  # `Net Debt / 1.0 = $16B (labeled as "16B x leverage")`. Refuse to compute
  # rather than fabricate a denominator.
  if ebitda <= 0:
    return {
      'error': f'_credit_profile_math: ebitda must be positive, got {ebitda}',
      'success': False,
    }

  if tax_rate > 1:
    tax_rate /= 100

  net_debt = total_debt - cash
  ebit = ebitda - depreciation_abs

  # Interest expense is acceptable as zero (debt-free companies). Treat as
  # large-but-finite so interest_coverage produces a meaningful "infinite" signal.
  safe_interest = interest_expense if interest_expense > 0 else 1.0

  net_debt_ebitda = net_debt / ebitda
  total_debt_ebitda = total_debt / ebitda
  interest_coverage = ebit / safe_interest

  # FCF approximation: NOPAT + D&A - CapEx
  nopat = max(0.0, ebit) * (1 - tax_rate)
  fcf_estimate = nopat + depreciation_abs - capex_abs

  if net_debt_ebitda <= 0:
    credit_label = "Net Cash"
  elif net_debt_ebitda <= 1:
    credit_label = "Investment Grade (Minimal Leverage)"
  elif net_debt_ebitda <= 2:
    credit_label = "Investment Grade"
  elif net_debt_ebitda <= 3:
    credit_label = "Investment Grade / High Yield Crossover"
  elif net_debt_ebitda <= 4:
    credit_label = "High Yield"
  elif net_debt_ebitda <= 5:
    credit_label = "Highly Leveraged"
  else:
    credit_label = "Distressed / Over-leveraged"

  result = {
    'net_debt': round(net_debt, 2),
    'net_debt_ebitda': round(net_debt_ebitda, 2),
    'total_debt_ebitda': round(total_debt_ebitda, 2),
    'interest_coverage': round(interest_coverage, 2),
    'ebit': round(ebit, 2),
    'fcf_estimate': round(fcf_estimate, 2),
    'credit_label': credit_label,
    'inputs': {
      'total_debt': total_debt,
      'cash': cash,
      'ebitda': ebitda,
      'interest_expense': interest_expense,
      'depreciation_abs': depreciation_abs,
      'capex_abs': capex_abs,
      'tax_rate': tax_rate,
    }
  }
  if market_cap > 0:
    result['fcf_yield_pct'] = round(fcf_estimate / market_cap * 100, 2)
  return result


def _scenario_dcf_math(base_inputs: dict,
                        bear_growth: list, base_growth: list, bull_growth: list,
                        bear_margin: float, base_margin: float, bull_margin: float) -> dict:
  """
  Run DCF for three scenarios and return price range.

  base_inputs: dict of all DCF inputs except revenue_growth and ebitda_margin.
  bear/base/bull_growth: 5-element list of annual growth rates per scenario.
  bear/base/bull_margin: EBITDA margin assumption per scenario (decimal).
  """
  results = {}
  for case_name, growth, margin in (
    ('bear', bear_growth, bear_margin),
    ('base', base_growth, base_margin),
    ('bull', bull_growth, bull_margin),
  ):
    inputs = dict(base_inputs)
    inputs['revenue_growth'] = growth
    inputs['ebitda_margin'] = margin
    r = _dcf_math(**inputs)
    results[case_name] = {
      'price_per_share': r['price_per_share'],
      'enterprise_value': r['enterprise_value'],
      'equity_value': r['equity_value'],
      'pv_terminal_value': r['pv_terminal_value'],
      'revenue_growth_y1_pct': round(growth[0] * 100, 2) if growth else 0,
      'ebitda_margin_pct': round(margin * 100, 2),
    }

  prices = [results[c]['price_per_share'] for c in ('bear', 'base', 'bull')]
  return {
    'bear': results['bear'],
    'base': results['base'],
    'bull': results['bull'],
    'price_range': {
      'low': round(min(prices), 2),
      'mid': round(results['base']['price_per_share'], 2),
      'high': round(max(prices), 2),
    }
  }


def _capital_returns_math(market_cap: float, ebitda: float, capex_abs: float,
                           tax_rate: float, depreciation_abs: float,
                           dividends_paid: float = 0, shares_repurchased: float = 0,
                           shares_outstanding: float = 0) -> dict:
  """
  Shareholder return profile from cash flow and market data.

  dividends_paid / shares_repurchased: raw dollar amounts (negative = outflow from CF stmt).
  """
  # Fail-fast: market_cap is the denominator for every yield; without it
  # the entire model fabricates garbage percentages.
  if market_cap <= 0:
    return {
      'error': f'_capital_returns_math: market_cap must be positive, got {market_cap}',
      'success': False,
    }
  if ebitda <= 0:
    return {
      'error': f'_capital_returns_math: ebitda must be positive, got {ebitda}',
      'success': False,
    }

  if tax_rate > 1:
    tax_rate /= 100

  ebit = ebitda - depreciation_abs
  nopat = max(0.0, ebit) * (1 - tax_rate)
  fcf_estimate = nopat + depreciation_abs - capex_abs

  safe_mktcap = market_cap  # validated above

  # CF statement reports outflows as negative -- take abs
  div_abs = abs(dividends_paid)
  buyback_abs = abs(shares_repurchased)
  total_returned = div_abs + buyback_abs

  fcf_yield = fcf_estimate / safe_mktcap
  div_yield = div_abs / safe_mktcap
  buyback_yield = buyback_abs / safe_mktcap

  payout_ratio = total_returned / fcf_estimate if fcf_estimate > 0 else None

  if payout_ratio is None:
    sustainability = "FCF Negative"
  elif payout_ratio <= 0.80:
    sustainability = "Sustainable"
  elif payout_ratio <= 1.0:
    sustainability = "Elevated"
  else:
    sustainability = "Unsustainable"

  result = {
    'fcf_estimate': round(fcf_estimate, 2),
    'fcf_yield_pct': round(fcf_yield * 100, 2),
    'dividends_paid': round(div_abs, 2),
    'shares_repurchased': round(buyback_abs, 2),
    'total_capital_returned': round(total_returned, 2),
    'dividend_yield_pct': round(div_yield * 100, 2),
    'buyback_yield_pct': round(buyback_yield * 100, 2),
    'total_shareholder_yield_pct': round((div_yield + buyback_yield) * 100, 2),
    'sustainability': sustainability,
  }
  if payout_ratio is not None:
    result['payout_ratio_pct'] = round(payout_ratio * 100, 2)
  if shares_outstanding > 0:
    result['dividend_per_share'] = round(div_abs / shares_outstanding, 4)
    result['buyback_per_share'] = round(buyback_abs / shares_outstanding, 4)
  return result


def _ddm_math(current_dps: float, cost_of_equity: float, terminal_growth: float,
              high_growth_rate: float = None, high_growth_years: int = 0) -> dict:
  """Dividend Discount Model. Gordon Growth or two-stage.

  Gordon: P = D_1 / (Ke - g) where D_1 = current_dps * (1 + g).
  Two-stage: PV of dividends grown at high_growth_rate for high_growth_years,
             then perpetuity at terminal_growth.
  """
  if cost_of_equity <= terminal_growth + 0.005:
    return {'error': 'Cost of equity must exceed terminal growth by >0.5%', 'success': False}

  if high_growth_years == 0:
    d1 = current_dps * (1 + terminal_growth)
    price = d1 / (cost_of_equity - terminal_growth)
    return {
      'method': 'gordon_growth',
      'intrinsic_value_per_share': round(price, 2),
      'd1': round(d1, 4),
      'cost_of_equity': cost_of_equity,
      'terminal_growth': terminal_growth,
      'success': True,
    }

  # Two-stage: explicit high-growth phase then perpetuity
  pv = 0
  d = current_dps
  for year in range(1, high_growth_years + 1):
    d *= (1 + high_growth_rate)
    pv += d / (1 + cost_of_equity) ** year

  d_terminal = d * (1 + terminal_growth)
  terminal_pv = (d_terminal / (cost_of_equity - terminal_growth)) / (1 + cost_of_equity) ** high_growth_years
  total = pv + terminal_pv
  return {
    'method': 'two_stage',
    'intrinsic_value_per_share': round(total, 2),
    'pv_high_growth_phase': round(pv, 2),
    'pv_terminal': round(terminal_pv, 2),
    'high_growth_years': high_growth_years,
    'high_growth_rate': high_growth_rate,
    'terminal_growth': terminal_growth,
    'cost_of_equity': cost_of_equity,
    'success': True,
  }


def _sensitivity_table_math(base_inputs: dict,
                            wacc_range: list = None,
                            tg_range: list = None) -> dict:
  """2D sensitivity grid: (WACC, terminal_growth) -> price per share.

  base_inputs: same dict as _dcf_math, with wacc/terminal_growth overridden per cell.
  wacc_range: list of WACC decimals. Default: base WACC +/-2% in 1% steps.
  tg_range:   list of terminal_growth decimals. Default: 1.5%, 2.0%, 2.5%, 3.0%, 3.5%.

  Cells where WACC - terminal_growth <= 0.005 return None (perpetuity formula unstable).
  """
  base_wacc = base_inputs.get('wacc', 0.10)
  if wacc_range is None:
    wacc_range = [round(base_wacc + d, 4) for d in (-0.02, -0.01, 0, 0.01, 0.02)]
  if tg_range is None:
    tg_range = [0.015, 0.02, 0.025, 0.03, 0.035]

  table = {}
  prices = []
  for w in wacc_range:
    table[f"{w:.4f}"] = {}
    for tg in tg_range:
      if w - tg <= 0.005:
        table[f"{w:.4f}"][f"{tg:.4f}"] = None
        continue
      inputs = {**base_inputs, 'wacc': w, 'terminal_growth': tg}
      r = _dcf_math(**inputs)
      px = r.get('price_per_share', 0)
      table[f"{w:.4f}"][f"{tg:.4f}"] = round(px, 2)
      prices.append(px)

  mid_price = None
  if f"{base_wacc:.4f}" in table and f"{0.025:.4f}" in table[f"{base_wacc:.4f}"]:
    mid_price = table[f"{base_wacc:.4f}"][f"{0.025:.4f}"]

  return {
    'table': table,
    'wacc_range': wacc_range,
    'tg_range': tg_range,
    'min_price': round(min(prices), 2) if prices else None,
    'max_price': round(max(prices), 2) if prices else None,
    'mid_price': mid_price,
    'cells_filled': len(prices),
  }


# ---------------------------------------------------------------------------
# Quantitative depth: reverse DCF, Monte Carlo, Piotroski, Altman Z
# ---------------------------------------------------------------------------

def _reverse_dcf_math(current_price: float, base_inputs: dict,
                      growth_lower: float = -0.20,
                      growth_upper: float = 0.50,
                      tol: float = 0.0005,
                      max_iter: int = 60) -> dict:
  """Solve for the uniform revenue growth rate that justifies the current price.

  Strategy: bisection over a monotonic-in-growth DCF. Higher growth -> higher
  price_per_share. We search for growth such that |dcf_price - current_price|
  is below tol fraction.

  Returns implied growth (decimal) and comparison vs the base case.
  """
  inputs = dict(base_inputs)
  # Strip the revenue_growth list — we replace it
  inputs.pop('revenue_growth', None)
  base_horizon = len(base_inputs.get('revenue_growth', [0]*5))

  def price_at(g):
    try:
      r = _dcf_math(revenue_growth=[g] * base_horizon, **inputs)
      return r.get('price_per_share', 0)
    except Exception:
      return float('nan')

  p_lo = price_at(growth_lower)
  p_hi = price_at(growth_upper)
  if not (p_lo < current_price < p_hi or p_lo > current_price > p_hi):
    return {
      'error': 'no_solution_in_range',
      'current_price': current_price,
      'price_at_lower_growth': round(p_lo, 2) if p_lo == p_lo else None,
      'price_at_upper_growth': round(p_hi, 2) if p_hi == p_hi else None,
      'growth_range_tested': [growth_lower, growth_upper],
    }

  lo, hi = growth_lower, growth_upper
  for _ in range(max_iter):
    mid = (lo + hi) / 2
    p_mid = price_at(mid)
    if abs(p_mid - current_price) / current_price < tol:
      break
    if (p_mid < current_price) == (p_lo < current_price):
      lo, p_lo = mid, p_mid
    else:
      hi = mid

  implied_g = mid
  # Base case growth (avg of input revenue_growth)
  base_growth = base_inputs.get('revenue_growth', [0])
  base_avg = sum(base_growth) / len(base_growth) if base_growth else 0
  spread = implied_g - base_avg

  return {
    'implied_growth_pct': round(implied_g * 100, 2),
    'implied_growth_decimal': round(implied_g, 4),
    'base_case_growth_pct': round(base_avg * 100, 2),
    'spread_pct': round(spread * 100, 2),
    'verdict': ('rich' if spread > 0.02
                else 'cheap' if spread < -0.02
                else 'fairly_priced'),
    'current_price': current_price,
    'method': 'bisection_on_uniform_revenue_growth',
    'horizon_years': base_horizon,
  }


def _monte_carlo_dcf_math(base_inputs: dict, n_iter: int = 5000,
                          wacc_std: float = 0.0075,
                          margin_std: float = 0.015,
                          growth_std: float = 0.025,
                          seed: int = 42) -> dict:
  """Run n_iter DCFs with WACC/margin/growth perturbed by gaussian noise.

  Returns price distribution stats. Use to gauge fair-value uncertainty.
  Default stds are conservative: ~75bps WACC, 150bps margin, 250bps growth.
  """
  import numpy as np
  rng = np.random.default_rng(seed)
  prices = []
  base_growth = base_inputs.get('revenue_growth', [0.05] * 5)
  base_g_mean = float(sum(base_growth) / len(base_growth))
  horizon = len(base_growth)

  for _ in range(n_iter):
    w = float(rng.normal(base_inputs['wacc'], wacc_std))
    if w < 0.02:
      w = 0.02  # floor
    m = float(rng.normal(base_inputs['ebitda_margin'], margin_std))
    if m < 0.0:
      m = 0.0
    g = float(rng.normal(base_g_mean, growth_std))
    g = max(g, -0.30)  # cap downside revenue growth
    inputs = {**base_inputs, 'wacc': w, 'ebitda_margin': m,
              'revenue_growth': [g] * horizon}
    try:
      r = _dcf_math(**inputs)
      p = r.get('price_per_share', 0)
      if p > 0 and p < 1e6:
        prices.append(p)
    except Exception:
      continue

  if not prices:
    return {'error': 'no_valid_iterations', 'n_attempted': n_iter}

  prices_arr = np.array(prices)
  return {
    'n_iter_attempted': n_iter,
    'n_iter_valid': len(prices),
    'mean': round(float(prices_arr.mean()), 2),
    'median': round(float(np.median(prices_arr)), 2),
    'std': round(float(prices_arr.std()), 2),
    'p5': round(float(np.percentile(prices_arr, 5)), 2),
    'p10': round(float(np.percentile(prices_arr, 10)), 2),
    'p25': round(float(np.percentile(prices_arr, 25)), 2),
    'p75': round(float(np.percentile(prices_arr, 75)), 2),
    'p90': round(float(np.percentile(prices_arr, 90)), 2),
    'p95': round(float(np.percentile(prices_arr, 95)), 2),
    'coefficient_of_variation': round(float(prices_arr.std() / prices_arr.mean()), 4),
    'assumptions': {
      'wacc_std': wacc_std, 'margin_std': margin_std,
      'growth_std': growth_std, 'seed': seed,
    },
  }


def _piotroski_f_score_math(financials: dict) -> dict:
  """Piotroski F-score: 9 binary tests on financial strength. Range 0-9.

  Each test requires specific inputs; when ANY required input is missing
  (the key is absent OR its value is None), the test is SKIPPED — neither
  passed nor failed. Score is the count of passes; `max_score_evaluated`
  is the count of tests that actually ran. `skipped_tests` lists the names
  of skipped tests so callers can see how complete the evaluation was.

  Pre-fix, defaults like `float('inf')` and `1` caused tests to evaluate
  spuriously True/False when inputs were missing.
  """
  def g(key):
    """Return the value or None — distinguishes 'missing' from 'zero'."""
    v = financials.get(key)
    return v if v is not None else None

  def _gt(a, b):
    return None if a is None or b is None else a > b
  def _lt(a, b):
    return None if a is None or b is None else a < b
  def _lte(a, b):
    return None if a is None or b is None else a <= b

  ni = g('net_income')
  ocf = g('op_cash_flow')
  ta_now = g('total_assets')
  ta_prev = g('total_assets_prior')
  ni_prev = g('net_income_prior')

  # ROA needs NI and TA for the period; either missing -> ROA is None
  def _roa(ni_val, ta_val):
    if ni_val is None or ta_val is None or ta_val == 0:
      return None
    return ni_val / ta_val
  roa_now = _roa(ni, ta_now)
  roa_prev = _roa(ni_prev, ta_prev)

  raw_tests = {
    'positive_net_income':      (ni > 0)  if ni  is not None else None,
    'positive_op_cash_flow':    (ocf > 0) if ocf is not None else None,
    'roa_improving':            _gt(roa_now, roa_prev),
    'cfo_exceeds_ni':           _gt(ocf, ni),
    'lt_debt_decreasing':       _lt(g('long_term_debt'), g('long_term_debt_prior')),
    'current_ratio_improving':  _gt(g('current_ratio'), g('current_ratio_prior')),
    'no_dilution':              _lte(g('shares_outstanding'),
                                      g('shares_outstanding_prior')),
    'gross_margin_improving':   _gt(g('gross_margin'), g('gross_margin_prior')),
    'asset_turnover_improving': _gt(g('asset_turnover'), g('asset_turnover_prior')),
  }

  skipped_tests = [k for k, v in raw_tests.items() if v is None]
  evaluated = {k: bool(v) for k, v in raw_tests.items() if v is not None}
  score = sum(1 for v in evaluated.values() if v)
  max_score_evaluated = len(evaluated)

  if max_score_evaluated == 0:
    rating = 'insufficient_data'
  else:
    ratio = score / max_score_evaluated
    if ratio >= 7 / 9:
      rating = 'strong'
    elif ratio <= 3 / 9:
      rating = 'weak'
    else:
      rating = 'mixed'

  return {
    'score': score,
    'max_score': 9,
    'max_score_evaluated': max_score_evaluated,
    'rating': rating,
    'tests': evaluated,
    'skipped_tests': skipped_tests,
    'method': 'Piotroski (2000) F-score',
  }


def _altman_z_score_math(financials: dict) -> dict:
  """Altman Z-score (original 1968 manufacturing form).

  Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = EBIT / total_assets
    X4 = market_cap / total_liabilities
    X5 = revenue / total_assets

  Zones:
    Z > 2.99 -> safe
    1.81 <= Z <= 2.99 -> grey
    Z < 1.81 -> distress
  """
  ta = financials.get('total_assets', 0) or 1
  wc = financials.get('working_capital', 0)
  re_ = financials.get('retained_earnings', 0)
  ebit = financials.get('ebit', 0)
  mc = financials.get('market_cap', 0)
  tl = financials.get('total_liabilities', 0) or 1
  rev = financials.get('revenue', 0)

  x1 = wc / ta
  x2 = re_ / ta
  x3 = ebit / ta
  x4 = mc / tl
  x5 = rev / ta

  z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

  if z > 2.99:
    zone = 'safe'
  elif z >= 1.81:
    zone = 'grey'
  else:
    zone = 'distress'

  return {
    'z_score': round(z, 2),
    'zone': zone,
    'components': {
      'X1_wc_ta': round(x1, 4),
      'X2_re_ta': round(x2, 4),
      'X3_ebit_ta': round(x3, 4),
      'X4_mc_tl': round(x4, 4),
      'X5_rev_ta': round(x5, 4),
    },
    'method': 'Altman (1968) Z-score (manufacturing form)',
  }


def _detect_insider_clusters(transactions: list, lookback_days: int = 30) -> dict:
  """Cluster analysis on insider transactions.

  transactions: list of dicts with keys 'date' (ISO str), 'shares' (signed),
                'insider_name', 'transaction_value' (optional).
  Returns directional cluster signal when 3+ distinct insiders trade the same
  direction within lookback_days.
  """
  from datetime import datetime, timedelta
  if not transactions:
    return {'signal': None, 'reason': 'no_transactions'}
  cutoff = datetime.now() - timedelta(days=lookback_days)
  recent = []
  for t in transactions:
    try:
      d = datetime.fromisoformat(str(t.get('date', '')).replace('Z', ''))
    except Exception:
      continue
    if d >= cutoff:
      recent.append(t)
  if not recent:
    return {'signal': None, 'reason': 'no_recent_transactions'}

  buyers = {t.get('insider_name', 'unknown') for t in recent if t.get('shares', 0) > 0}
  sellers = {t.get('insider_name', 'unknown') for t in recent if t.get('shares', 0) < 0}
  buyer_dollars = sum(abs(t.get('transaction_value', 0))
                      for t in recent if t.get('shares', 0) > 0)
  seller_dollars = sum(abs(t.get('transaction_value', 0))
                       for t in recent if t.get('shares', 0) < 0)

  if len(buyers) >= 3 and not sellers:
    signal = 'strong_cluster_buy'
  elif len(buyers) >= 3 and len(buyers) >= 2 * len(sellers):
    signal = 'cluster_buy'
  elif len(sellers) >= 3 and not buyers:
    signal = 'strong_cluster_sell'
  elif len(sellers) >= 3 and len(sellers) >= 2 * len(buyers):
    signal = 'cluster_sell'
  else:
    signal = None

  return {
    'signal': signal,
    'distinct_buyers': len(buyers),
    'distinct_sellers': len(sellers),
    'buyer_dollar_volume': buyer_dollars,
    'seller_dollar_volume': seller_dollars,
    'lookback_days': lookback_days,
    'transactions_in_window': len(recent),
  }


def _revisions_momentum_math(trends: list) -> dict:
  """Given Finnhub-shaped recommendation_trends list (newest first), compute
  the change in net buy-side consensus over 30/90 day windows.

  Each trend dict has keys: strongBuy, buy, hold, sell, strongSell, period.
  """
  if not trends or len(trends) < 2:
    return {'error': 'insufficient_history', 'periods_available': len(trends or [])}

  def net(t):
    return (t.get('strongBuy', 0) + t.get('buy', 0)
            - t.get('sell', 0) - t.get('strongSell', 0))

  curr = trends[0]
  m1 = trends[1] if len(trends) >= 2 else curr
  m3 = trends[3] if len(trends) >= 4 else m1

  delta_30 = net(curr) - net(m1)
  delta_90 = net(curr) - net(m3)
  composite = delta_30 + 0.5 * delta_90

  if composite > 2:
    direction = 'strong_rising'
  elif composite > 0:
    direction = 'rising'
  elif composite < -2:
    direction = 'strong_falling'
  elif composite < 0:
    direction = 'falling'
  else:
    direction = 'flat'

  return {
    'net_current': net(curr),
    'net_30d_ago': net(m1),
    'net_90d_ago': net(m3),
    'delta_30d': delta_30,
    'delta_90d': delta_90,
    'composite_score': round(composite, 2),
    'direction': direction,
    'current_period': curr.get('period'),
  }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

# Tools only the execution engine should plan -- used to filter the tool list
# shown to the orchestrator when building the execution plan.
MODELING_PHASE_TOOLS = {
  'calculate_scenario_dcf',
  'calculate_lbo',
  'calculate_credit_profile',
  'calculate_capital_returns',
}


class Financial_Analysis:
  def __init__(self, args=None):
    self.server = Server("Financial_Analysis")
    self._setup_handlers()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        # ---- Data tools (execution phase) ----
        Tool(
          name="get_market_data",
          description=market_data_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, MSFT)"}
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
              "ticker": {"type": "string", "description": "Company ticker symbol (e.g. 'NVDA')"},
              "revenue_base": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_revenue_base."},
              "ebitda_margin": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_ebitda_margin."},
              "capex_pct_revenue": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_capex_pct_revenue."},
              "tax_rate": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_tax_rate."},
              "depreciation": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_depreciation."},
              "revenue_growth": {"type": "array", "description": "SET TO [0,0,0,0,0] -- auto-resolved from get_basic_financials.", "items": {"type": "number"}},
              "wacc": {"type": "number", "description": "SET TO 0 -- auto-resolved from calculate_wacc."},
              "terminal_growth": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_macro_snapshot GDP."},
              "terminal_multiple": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_basic_financials evEbitdaTTM."},
              "cash": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."},
              "debt": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."},
              "shares_outstanding": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="calculate_wacc",
          description=wacc_tool_description,
          inputSchema={
            "type": "object",
            "properties": {
              "beta": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."},
              "risk_free_rate": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_macro_snapshot."},
              "equity_risk_premium": {"type": "number", "description": "ALWAYS pass 0.06 (standard 6% ERP). Do NOT pass 6."},
              "cost_of_debt": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."},
              "tax_rate": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_tax_rate."},
              "market_cap": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."},
              "total_debt": {"type": "number", "description": "SET TO 0 -- auto-resolved from get_market_data."}
            },
            "required": []
          }
        ),
        # ---- Modeling phase tools ----
        Tool(
          name="calculate_scenario_dcf",
          description=scenario_dcf_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Company ticker"},
              "bear_growth": {"type": "array", "items": {"type": "number"}, "description": "5 annual growth rates for bear case (decimals)"},
              "base_growth": {"type": "array", "items": {"type": "number"}, "description": "5 annual growth rates for base case (decimals)"},
              "bull_growth": {"type": "array", "items": {"type": "number"}, "description": "5 annual growth rates for bull case (decimals)"},
              "bear_margin": {"type": "number", "description": "EBITDA margin for bear case (decimal)"},
              "base_margin": {"type": "number", "description": "EBITDA margin for base case (decimal)"},
              "bull_margin": {"type": "number", "description": "EBITDA margin for bull case (decimal)"},
              "revenue_base": {"type": "number"},
              "capex_pct_revenue": {"type": "number"},
              "tax_rate": {"type": "number"},
              "depreciation": {"type": "number"},
              "wacc": {"type": "number"},
              "terminal_growth": {"type": "number"},
              "terminal_multiple": {"type": "number"},
              "cash": {"type": "number"},
              "debt": {"type": "number"},
              "shares_outstanding": {"type": "number"}
            },
            "required": ["ticker", "bear_growth", "base_growth", "bull_growth",
                         "bear_margin", "base_margin", "bull_margin"]
          }
        ),
        Tool(
          name="calculate_lbo",
          description=lbo_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "entry_ev": {"type": "number", "description": "Total acquisition EV in dollars"},
              "revenue_base": {"type": "number", "description": "Current annual revenue"},
              "ebitda_margin": {"type": "number", "description": "EBITDA margin (decimal)"},
              "capex_pct_revenue": {"type": "number", "description": "CapEx as % of revenue (decimal)"},
              "depreciation": {"type": "number", "description": "D&A as % of revenue (decimal)"},
              "tax_rate": {"type": "number", "description": "Effective tax rate (decimal)"},
              "revenue_growth": {"type": "array", "items": {"type": "number"}, "description": "Annual growth rates for hold period"},
              "debt_interest_rate": {"type": "number", "description": "All-in interest rate on acquisition debt (decimal, e.g. 0.08)"},
              "leverage_turns": {"type": "number", "description": "Acquisition debt as multiple of entry EBITDA (e.g. 4.5)"},
              "exit_multiple": {"type": "number", "description": "EV/EBITDA at exit"},
              "hold_years": {"type": "integer", "description": "Hold period in years (default 5)"}
            },
            "required": ["ticker", "entry_ev", "revenue_base", "ebitda_margin",
                         "capex_pct_revenue", "depreciation", "tax_rate",
                         "revenue_growth", "debt_interest_rate", "leverage_turns", "exit_multiple"]
          }
        ),
        Tool(
          name="calculate_credit_profile",
          description=credit_profile_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "total_debt": {"type": "number", "description": "Total debt in dollars"},
              "cash": {"type": "number", "description": "Cash and equivalents in dollars"},
              "ebitda": {"type": "number", "description": "Annual EBITDA in dollars"},
              "interest_expense": {"type": "number", "description": "Annual interest expense in dollars"},
              "depreciation_abs": {"type": "number", "description": "Annual D&A in dollars"},
              "capex_abs": {"type": "number", "description": "Annual CapEx in dollars"},
              "tax_rate": {"type": "number", "description": "Effective tax rate (decimal)"},
              "market_cap": {"type": "number", "description": "Market cap in dollars (for FCF yield)"}
            },
            "required": ["ticker", "total_debt", "cash", "ebitda",
                         "interest_expense", "depreciation_abs", "capex_abs", "tax_rate"]
          }
        ),
        Tool(
          name="calculate_capital_returns",
          description=capital_returns_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "market_cap": {"type": "number", "description": "Market cap in dollars"},
              "ebitda": {"type": "number", "description": "Annual EBITDA in dollars"},
              "capex_abs": {"type": "number", "description": "Annual CapEx in dollars"},
              "tax_rate": {"type": "number", "description": "Effective tax rate (decimal)"},
              "depreciation_abs": {"type": "number", "description": "Annual D&A in dollars"},
              "dividends_paid": {"type": "number", "description": "Dividends paid from CF statement (negative = outflow)"},
              "shares_repurchased": {"type": "number", "description": "Share repurchases from CF statement (negative = outflow)"},
              "shares_outstanding": {"type": "number", "description": "Total shares outstanding"}
            },
            "required": ["ticker", "market_cap", "ebitda", "capex_abs", "tax_rate", "depreciation_abs"]
          }
        ),
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
        elif name == "calculate_scenario_dcf":
          return await parent.calculate_scenario_dcf(args)
        elif name == "calculate_lbo":
          return await parent.calculate_lbo(args)
        elif name == "calculate_credit_profile":
          return await parent.calculate_credit_profile(args)
        elif name == "calculate_capital_returns":
          return await parent.calculate_capital_returns(args)
      except Exception as e:
        return [TextContent(
          type="text",
          text=json.dumps({"success": False, "error": f"Failed to call tool '{name}': {str(e)}"})
        )]
      return [TextContent(
        type="text",
        text=json.dumps({"success": False, "error": f"Unknown tool: {name}"})
      )]

  # ---- Tool implementations ----

  async def get_market_data(self, ticker: str) -> List[TextContent]:
    data = await asyncio.to_thread(get_data, ticker)
    clean_data = {}
    for key, value in data.items():
      if value is None or value == 'N/A':
        clean_data[key] = None
      elif hasattr(value, 'item'):
        clean_data[key] = value.item()
      else:
        clean_data[key] = value
    return [TextContent(type="text", text=json.dumps(clean_data))]

  async def comparable_company_analysis(self, comparables: List[str]) -> List[TextContent]:
    tasks = [asyncio.to_thread(get_data, ticker) for ticker in comparables]
    data = await asyncio.gather(*tasks)
    result = {
      'comparables': comparables,
      'pe_ratio': _to_native(calculate_percentiles(data, 'pe_ratio')),
      'pb_data': _to_native(calculate_percentiles(data, 'pb_ratio')),
      'ev_revenue_data': _to_native(calculate_percentiles(data, 'ev_revenue')),
      'ev_ebitda_data': _to_native(calculate_percentiles(data, 'ev_ebitda')),
      'ev_ebit_data': _to_native(calculate_percentiles(data, 'ev_ebit')),
    }
    return [TextContent(type="text", text=json.dumps(result))]

  async def calculate_dcf(self, args: Dict[str, Any]) -> List[TextContent]:
    result = _dcf_math(
      revenue_base=args['revenue_base'],
      ebitda_margin=args['ebitda_margin'],
      capex_pct_revenue=args['capex_pct_revenue'],
      tax_rate=args['tax_rate'],
      depreciation=args['depreciation'],
      revenue_growth=args['revenue_growth'],
      wacc=args['wacc'],
      terminal_growth=args['terminal_growth'],
      terminal_multiple=args['terminal_multiple'],
      cash=args['cash'],
      debt=args['debt'],
      shares_outstanding=args['shares_outstanding'],
      ticker=args.get('ticker', ''),
    )
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_wacc(self, args: Dict[str, Any]) -> List[TextContent]:
    result = _wacc_math(
      beta=args['beta'],
      risk_free_rate=args['risk_free_rate'],
      equity_risk_premium=args.get('equity_risk_premium', 0.06),
      cost_of_debt=args.get('cost_of_debt', 0),
      tax_rate=args.get('tax_rate', 0),
      market_cap=args.get('market_cap', 0),
      total_debt=args.get('total_debt', 0),
    )
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_scenario_dcf(self, args: Dict[str, Any]) -> List[TextContent]:
    base_inputs = {
      'revenue_base': args.get('revenue_base', 0),
      'capex_pct_revenue': args.get('capex_pct_revenue', 0),
      'tax_rate': args.get('tax_rate', 0),
      'depreciation': args.get('depreciation', 0),
      'wacc': args.get('wacc', 0),
      'terminal_growth': args.get('terminal_growth', 0),
      'terminal_multiple': args.get('terminal_multiple', 0),
      'cash': args.get('cash', 0),
      'debt': args.get('debt', 0),
      'shares_outstanding': args.get('shares_outstanding', 0),
      'ticker': args.get('ticker', ''),
      # ebitda_margin and revenue_growth are supplied per-scenario
      'ebitda_margin': args.get('base_margin', 0),
      'revenue_growth': args.get('base_growth', [0, 0, 0, 0, 0]),
    }
    result = _scenario_dcf_math(
      base_inputs=base_inputs,
      bear_growth=args['bear_growth'],
      base_growth=args['base_growth'],
      bull_growth=args['bull_growth'],
      bear_margin=args['bear_margin'],
      base_margin=args['base_margin'],
      bull_margin=args['bull_margin'],
    )
    result['ticker'] = args.get('ticker', '')
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_lbo(self, args: Dict[str, Any]) -> List[TextContent]:
    result = _lbo_math(
      entry_ev=args['entry_ev'],
      revenue_base=args['revenue_base'],
      ebitda_margin=args['ebitda_margin'],
      capex_pct_revenue=args['capex_pct_revenue'],
      depreciation=args['depreciation'],
      tax_rate=args['tax_rate'],
      revenue_growth=args['revenue_growth'],
      debt_interest_rate=args['debt_interest_rate'],
      leverage_turns=args['leverage_turns'],
      exit_multiple=args['exit_multiple'],
      hold_years=args.get('hold_years', 5),
    )
    result['ticker'] = args.get('ticker', '')
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_credit_profile(self, args: Dict[str, Any]) -> List[TextContent]:
    result = _credit_profile_math(
      total_debt=args['total_debt'],
      cash=args['cash'],
      ebitda=args['ebitda'],
      interest_expense=args['interest_expense'],
      depreciation_abs=args['depreciation_abs'],
      capex_abs=args['capex_abs'],
      tax_rate=args['tax_rate'],
      market_cap=args.get('market_cap', 0),
    )
    result['ticker'] = args.get('ticker', '')
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_capital_returns(self, args: Dict[str, Any]) -> List[TextContent]:
    result = _capital_returns_math(
      market_cap=args['market_cap'],
      ebitda=args['ebitda'],
      capex_abs=args['capex_abs'],
      tax_rate=args['tax_rate'],
      depreciation_abs=args['depreciation_abs'],
      dividends_paid=args.get('dividends_paid', 0),
      shares_repurchased=args.get('shares_repurchased', 0),
      shares_outstanding=args.get('shares_outstanding', 0),
    )
    result['ticker'] = args.get('ticker', '')
    return [TextContent(type='text', text=json.dumps(result))]

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
