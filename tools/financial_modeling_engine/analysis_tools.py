from tools.response_meta import annotating, warning
from typing import Any, Dict, List, Optional
import asyncio
from tools.financial_modeling_engine.corporate_actions import get_corporate_actions
from tools.financial_modeling_engine.trading_metrics import get_trading_metrics
import json
import sys
import traceback

from .utils import get_data, calculate_percentiles, get_institutional_holdings, get_options_metrics, get_short_interest, get_price_history, get_industry_etfs, get_historical_analogue
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ---------------------------------------------------------------------------
# Tool descriptions
# These are the single source of truth for what each tool needs.
# Both the Probing Agent (to surface data requirements) and the
# Financial Modeling Agent (to decide what to run) read these at runtime.
# ---------------------------------------------------------------------------

market_data_description = """Retrieves real-time market data for a company from Yahoo Finance including market cap, enterprise value, revenue, EBITDA, cash, debt, shares outstanding, beta, interest expense, and valuation multiples (P/E, P/B, EV/Revenue, EV/EBITDA, EV/EBIT).
Should use: When you need current financial data for a company to perform valuation analysis, equity bridge calculations, or to get inputs for WACC calculation (beta, market cap, debt, interest expense).
Should NOT use: When you need historical time-series data or SEC filing data (use SEC tools instead).
Share basis: marketCap is stated across every share class and the provider's sharesOutstanding is stated for one of them, so the two do not multiply together for a multi-class filer -- GOOGL is 2.0845x apart and BRK-B 1.5204x. shares_outstanding_basis says which case this response is; use shares_outstanding_all_classes for anything divided by a share count (value per share, book per share, ownership percentages) and read the share_count_basis_mismatch warning when it is present. pe_ratio and pb_ratio divide marketCap and are unaffected."""

extract_13f_holdings_description = """Returns the top institutional holders and mutual fund holders of a stock, plus aggregate institutional ownership statistics. Data sourced from Yahoo's aggregation of SEC 13F-HR filings (institutional holders, filed quarterly) and NPORT-P filings (mutual funds). Each holder row includes shares held, market value in USD, percent of shares outstanding, and quarter-over-quarter percent change in position size.
Should use: When researching who owns the stock, to detect institutional buying/selling pressure (pct_change_qoq), to size up the 'smart money' bull/bear setup, or to find catalysts in institutional positioning. Critical for the Layer-5 capital-allocation read in the IB analyst playbook.
Should NOT use: For company insider transactions (use get_insider_transactions instead — that covers officer/director Form 4 filings). For real-time positioning (this lags 45-90 days due to 13F reporting cadence)."""

options_metrics_description = """Options-market signals derived from yfinance option chains. Returns: (1) ATM implied volatility term structure at ~7d/30d/60d/90d expirations, (2) 30d put/call skew (0.9*spot put IV minus 1.1*spot call IV; positive = downside fear premium), (3) nearest-expiry open interest and volume aggregates with put/call ratios.
Should use: For forward-looking risk read (rising IV into earnings = market pricing surprise), sentiment positioning (volume ratio <0.5 = call-heavy/bullish, >1.0 = put-heavy/bearish), and skew (elevated put skew = institutional hedging).
Should NOT use: As a primary valuation input — options are sentiment, not fundamentals. Returns a data_quality field — yfinance occasionally returns sentinel IV values (powers of 2) for stale snapshots; trust the volume ratios over IV in those cases."""

short_interest_description = """Short interest snapshot: shares short, days to cover (short ratio), percent of float, and month-over-month change. Source: yfinance (underlying FINRA biweekly disclosures). Includes a signal classifier (low/moderate/elevated/crowded short).
Should use: For positioning context — crowded shorts on a stock with bull momentum signal squeeze risk; rising shorts on a name with deteriorating fundamentals confirm the bear case. Pair with get_insider_transactions to triangulate sentiment.
Should NOT use: For intraday positioning — short interest reports lag 2 weeks. Days-to-cover is an estimate based on average volume, not a hard ceiling.
Share basis: float_shares and shares_outstanding come from the provider on different share bases for a multi-class filer -- GOOGL's float exceeds its shares outstanding, so 1 - float/shares gives -85% insider ownership. shares_outstanding_basis says which case this is, shares_outstanding_all_classes is the count the float is comparable with, and float_exceeds_shares_outstanding / short_interest_exceeds_float appear in warnings when the two cannot be combined."""

price_history_description = """Historical price summary: returns over 1M/3M/6M/YTD/1Y/3Y windows, realized volatility (annualized) over 30d/90d/180d/1Y, 52-week high/low with dates, max drawdown from trailing-12-month peak, and a configurable number of recent OHLCV bars. Source: yfinance auto-adjusted daily history.
Should use: For drawdown framing ('stock down N% from peak'), volatility regime read (vol spike before earnings = expected surprise), and return decomposition over multiple windows.
Should NOT use: For minute-level technical analysis (this returns daily bars). For intraday spot price use get_market_data.
Price basis: every close is back-adjusted onto the newest session's basis for splits AND dividends, which the response now states in price_basis and quantifies in price_adjustment. A close from before a split is stated in post-split shares, so it pairs with get_share_count_series.total_split_adjusted and NOT with total -- NVDA's 2024-05-24 bar against the as-filed count is out by exactly 10.0x. prices_split_adjusted and prices_dividend_adjusted appear in warnings when the window contains either."""

trading_metrics_description = """Execution and position-sizing inputs from daily bars: relative volume (RVOL, the latest session's volume against its own trailing 20-session average), average dollar volume (ADV over 20 and 60 sessions, in USD and in shares), and average true range (ATR, Wilder-smoothed over 14 sessions, plus ATR as a percent of price). Source: the same yfinance daily bars as get_price_history.
Should use: When deciding HOW to trade a name you have already researched -- how large a position the tape can absorb, how many days it takes to exit, how wide a stop has to be to survive ordinary daily noise, and whether today's volume is unusual enough to signal that something happened.
Should NOT use: For any question about the business. These are tape mechanics and say nothing about revenue, margins, competitive position, or valuation. Do not reach for this during fundamental research -- it will not help and it is not evidence about the company.
Notes: ADV is dollars because share counts do not tell you whether a position can be exited. ATR uses true range, so overnight gaps are included -- ATR will exceed the average high-minus-low on gappy names. Windows that the available history cannot fill are returned as null with a note, never as a shorter average. If the newest bar is the session currently in progress, latest_bar_is_partial_session is true: RVOL is then a live intraday read against full-session baselines and will understate, while ADV and ATR exclude the partial bar entirely."""

industry_etfs_description = """Map a research theme (e.g. 'AI semis', 'energy', 'cloud', 'uranium', 'biotech') to relevant ETFs and return their top holdings + weights. Acts as the bridge from top-down thematic conviction to bottom-up ticker selection — e.g. an AI/memory thesis surfaces SK Hynix (HBM) as AIQ's top holding, not just NVDA. Returns ETF AUM, top 10 holdings, sector weightings. Covers ~50 themes across tech, energy, financials, healthcare, geographic, and macro categories.
Should use: As the FIRST step in thematic research — convert 'I think AI capex will accelerate' into a list of pure-play tickers via the ETF holdings.
Should NOT use: For specific ticker lookup (use the ticker directly). For comp basket construction within a sector (use get_company_peers)."""

historical_analogue_description = """Match a current investment thesis against a curated catalog of historical market setups (1999 dot-com, 2007 housing, 2014 oil collapse, 2018 memory bust, 2020 cloud acceleration, 2024 AI capex cycle, etc.). Returns top N matches by structural-tag overlap plus each analogue's lessons. Use structural keywords in the thesis description (capex_cycle, capex_peak, capex_trough, valuation_expansion, valuation_compression, margin_compression, supply_constrained, supply_glut, concentrated_buyers, insider_selling, retail_frenzy, tech, energy, commodities, etc.) for cleaner matches.
Should use: At the synthesis stage of deep research — once you have the data, ask 'what does this remind me of?' to surface historical lessons and avoid repeating prior cycle mistakes.
Should NOT use: For ticker-specific predictions; this returns pattern-level analogues, not stock-level forecasts."""

record_thesis_evolution_description = """Record an evolution event against an existing thesis. Use after a meaningful new data point arrives (earnings print, news, macro shift, insider activity) to log how analyst conviction shifted and why. Implements the Soros reflexivity discipline — write your assumptions down, then re-read them when new data arrives. conviction_delta is in 0.0-1.0 units (e.g. +0.05 for a beat that slightly improves conviction, -0.10 for a meaningful negative surprise). Automatically updates the thesis's current confidence, clamped to [0,1]."""

get_thesis_evolution_description = """Return the chronological evolution log for a thesis — the reflexivity trace. Each row shows what observation triggered a conviction change, the magnitude of the change, the new conviction level, and an optional tag (earnings, macro, insider, sector, sentiment). Reading the full evolution log forces the analyst to confront whether the original thesis is still intact or has drifted."""

backtest_signal_description = """Backtest a signal definition on historical price data. Given a rule (e.g. "buy when RSI<30 hold 30 days") and a ticker, returns historical hit rate, mean return, sharpe, sample trades. Use to validate that a research pattern actually worked before betting on it. Supports threshold rules ({metric, op, value}), AND/OR composites, and named built-in signals: oversold_rsi, overbought_rsi, big_drawdown. Available metrics: rsi_14, sma_20, sma_50, sma_200, drawdown_pct, close. Returns n_trades, hit_rate %, mean_return %, sharpe_simple, sample trades with entry/exit dates."""

analyze_exposures_description = """Decompose all active theses into latent factor buckets (ai_capex_long, rate_sensitive_long_duration, energy_long, china_exposure, commodity_supply_constrained, crypto_beta, biotech_speculative, etc.). Surfaces hidden concentration — the diversification illusion that a portfolio of NVDA/AMD/TSM/MSFT is really one AI-capex bet. Returns top concentrations with conviction-weighted sums and explicit warning flags when any factor exceeds 3 theses or 2.0 cumulative conviction. Run before sizing up any position."""

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
6. get_market_data(ticker)         -> cash, debt, shares_outstanding_all_classes
   (per-share output divides the share count, so it needs the all-class figure;
   the provider's sharesOutstanding covers one class and is 2.0845x short for
   GOOGL, 1.5204x for BRK-B. Single-class filers are the same number.)
7. get_basic_financials(ticker)    -> revenue_growth, terminal_multiple
8. get_macro_snapshot()            -> risk_free_rate, terminal_growth
9. calculate_wacc()                -> wacc"""

comps_tool_description = """Estimates a company's value by calculating valuation multiples (e.g., EV/EBITDA, P/E) for a set of comparable public companies and applying the median multiple to the target company.
Should use: To get a quick valuation range based on current market sentiment and to see how a company is valued relative to its direct peers.
Should NOT use: When there are no truly comparable public companies, or if the entire market sector is believed to be in a bubble or a crash. This method provides a relative value, not a fundamental intrinsic value."""

scenario_dcf_description = """Runs the DCF model under three scenarios (Bear / Base / Bull) using different revenue growth rates and EBITDA margin assumptions. Returns a price-per-share range across all three cases.

Every price in the payload -- the headline bear/base/bull price and every cell of terminal_sensitivity -- uses the same conservative min(perpetuity, exit multiple) terminal value, so the grid cell at terminal_sensitivity_base_multiple equals that scenario's headline price. terminal_value_method on each scenario names which of the two is binding; where the perpetuity binds across the whole sweep the row is flat and terminal_sensitivity_floor_note says so.

MODELING PHASE ONLY -- called by the Financial Modeling Agent, not the execution engine.

Data that must be in the variable store before modeling:
- revenue_base          <- get_revenue_base
- ebitda_margin         <- get_ebitda_margin
- capex_pct_revenue     <- get_capex_pct_revenue
- depreciation          <- get_depreciation
- tax_rate              <- get_tax_rate
- cash, debt, shares_outstanding_all_classes, beta  <- get_market_data
  (all-class, not the provider's single-class sharesOutstanding -- every price
   in this model is a per-share figure)
- risk_free_rate        <- get_macro_snapshot
- revenueGrowthTTMYoy, evEbitdaTTM      <- get_basic_financials
- forward estimate low/high revenue     <- get_forward_estimates (for bear/bull anchors)
- historical income statement           <- get_financial_statements ic annual (for margin trend)"""

lbo_description = """Leveraged buyout model. Computes IRR and MOIC to equity given entry EV, debt structure (leverage turns, interest rate), and exit multiple over a hold period.

Refuses rather than modelling an impossible structure: entry EBITDA at or below zero (nothing to lever, to service the debt, or to price the exit off), an exit_multiple at or below zero (a non-positive exit enterprise value), a hold_years below 1, and acquisition debt meeting or exceeding entry_ev. Each refusal names the input to change.

MODELING PHASE ONLY -- called by the Financial Modeling Agent, not the execution engine.
ONLY run when the user query involves M&A, private equity, buyout potential, or takeout analysis.

Data that must be in the variable store before modeling:
- market_cap, totalDebt, totalCash, shares_outstanding_all_classes  <- get_market_data
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
- market_cap, shares_outstanding_all_classes  <- get_market_data
  (dividend and buyback per share divide the share count; the provider's
   sharesOutstanding is one class where market_cap is all of them)
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


def peer_distribution(values, *, lower=0.0, upper=1000.0,
                      tickers=None, reasons=None) -> dict:
  """Summary statistics over comparable multiples only.

  A multiple built on a denominator approaching zero is arithmetically correct
  and analytically meaningless. INTC's ev_ebit of -20,768 dragged a four-name
  peer mean to -5,113 against a median of 53, with success: true and no
  warning -- and a mean and median on opposite sides of zero is the signature
  of exactly one outlier doing all the work.

  Negative multiples are excluded for the same reason: a company losing money
  has no meaningful P/E, and averaging one in states something about the peer
  group that is not true of any member of it.

  Exclusions are counted and explained. A distribution quietly computed over a
  different set than the caller asked for is worse than one that says so.

  `tickers` and `reasons` carry the cause of each absence down from the
  caller. Without them this function knows a value is missing and nothing
  about why, and it used to fill that gap with one fixed sentence -- "a
  foreign issuer whose multiples are suppressed across currencies, or a filer
  tagging nothing" -- printed over every exclusion regardless of cause. Run
  against MU, BRK-B and ZZZZNOTREAL the real causes were a null provider
  market cap, a negative enterprise value, and a symbol that is not a company;
  the counts were right and the stated reason was wrong for all three. An
  analyst told "foreign issuer" goes looking for a currency problem that is
  not there and never learns one of their comparables does not exist.
  """
  supplied = list(values or [])
  labels = list(tickers or [])
  labels += [f"peer {i + 1}" for i in range(len(labels), len(supplied))]
  reasons = dict(reasons or {})

  def _is_number(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and abs(v) != float('inf'))

  # A peer with no comparable multiple at all was filtered out before the
  # count, so a four-name comp set reported "included 2, excluded 0" and
  # published a median of the two that remained. Suppressing the multiple is
  # right; making the peer disappear is not.
  kept, excluded, absent, implausible = [], [], 0, 0
  for label, value in zip(labels, supplied):
    if not _is_number(value):
      absent += 1
      excluded.append({
          'ticker': label,
          'value': None,
          'reason': reasons.get(label) or (
              f"{label} reported no comparable multiple and no cause was "
              f"supplied with the value"),
      })
      continue
    number = float(value)
    if lower < number <= upper:
      kept.append(number)
      continue
    implausible += 1
    excluded.append({
        'ticker': label,
        'value': number,
        'reason': (
            f"{label}'s multiple of {number:,.2f} falls outside "
            f"({lower}, {upper}]: such a multiple comes from a denominator at "
            f"or near zero, or from negative earnings, and is not comparable"),
    })
  dropped = absent + implausible

  stats = {
      'included_count': len(kept),
      'excluded_count': dropped,
      'excluded_reason': None,
      'mean': None, 'median': None, 'q1': None, 'q3': None,
      'low': None, 'high': None,
  }
  if dropped:
    if tickers:
      stats['excluded_reason'] = (
          "; ".join(f"{e['ticker']}: {e['reason']}" for e in excluded) +
          ". A distribution computed over a different set than the caller "
          "asked for must say so.")
    else:
      # Called with bare values, so there is no cause to report. Saying so is
      # the honest form; asserting one is how the misattribution started.
      parts = []
      if absent:
        parts.append(
            f"{absent} peer(s) reported no comparable multiple, with no cause "
            f"supplied alongside the values")
      if implausible:
        parts.append(
            f"{implausible} peer(s) fell outside ({lower}, {upper}]: such a "
            f"multiple comes from a denominator at or near zero, or from "
            f"negative earnings, and is not comparable")
      stats['excluded_reason'] = (
          "; ".join(parts) +
          ". A distribution computed over a different set than the caller "
          "asked for must say so.")
  stats['excluded_absent'] = absent
  stats['excluded_implausible'] = implausible
  stats['excluded_peers'] = excluded
  if not kept:
    return stats

  import statistics
  ordered = sorted(kept)
  stats['mean'] = statistics.fmean(ordered)
  stats['median'] = statistics.median(ordered)
  stats['low'] = ordered[0]
  stats['high'] = ordered[-1]
  if len(ordered) >= 2:
    import numpy as _np
    stats['q1'] = float(_np.percentile(ordered, 25))
    stats['q3'] = float(_np.percentile(ordered, 75))
  return stats


def as_rate(name: str, value, *, allow_negative: bool = False) -> float:
  """Interpret a rate given as a decimal or a percentage, or refuse.

  The calculators previously did `if x > 1: x /= 100`, which is safe for a
  value that really is a rate and catastrophic for one that is not: it turns
  an implausible number into a differently implausible number instead of an
  error. `calculate_dcf` documents `depreciation` as coming from
  `get_depreciation`, whose `d&a` is an absolute figure -- NVDA's 2,843,000,000
  became 28,430,000 and then a multiplier on revenue, and the model returned
  $242,204,233 per share with success: true.

  0 to 1 is read as a decimal, above 1 to 100 as a percentage. Anything else
  is refused, naming the parameter, because there is no reading of 2.8 billion
  that is a rate.
  """
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(
        f"{name} must be a number given as a decimal (0.25) or a percentage "
        f"(25), received {type(value).__name__}")
  number = float(value)
  if number != number:                                  # NaN
    raise ValueError(f"{name} is NaN, which is neither a decimal nor a percentage")
  if number < 0:
    if not allow_negative:
      raise ValueError(f"{name} cannot be negative, received {number}")
    return number if abs(number) <= 1 else number / 100
  if number <= 1:
    return number
  if number <= 100:
    return number / 100
  raise ValueError(
      f"{name}={number:,.0f} is neither a decimal rate (0.25) nor a "
      f"percentage (25). If this is an absolute amount, pass the rate "
      f"instead -- get_depreciation returns `d&a` in dollars and `d&a_pct` as "
      f"the percentage this expects.")


def _dcf_math(revenue_base: float, ebitda_margin: float, capex_pct_revenue: float,
              tax_rate: float, depreciation: float, revenue_growth: list,
              wacc: float, terminal_growth: float, terminal_multiple: float,
              cash: float, debt: float, shares_outstanding: float,
              ticker: str = '') -> dict:
  """5-year FCF DCF model. All margin/rate inputs as decimals."""
  # Every rate parameter, interpreted the same way. Previously wacc and
  # terminal_growth were left out of the normalisation the others got, so
  # `wacc: 10` -- the percentage convention the rest accept -- became a 1000%
  # discount rate and a price per share of 1,104,824,357.
  ebitda_margin = as_rate('ebitda_margin', ebitda_margin)
  capex_pct_revenue = as_rate('capex_pct_revenue', capex_pct_revenue)
  tax_rate = as_rate('tax_rate', tax_rate)
  depreciation = as_rate('depreciation', depreciation)
  wacc = as_rate('wacc', wacc)
  terminal_growth = as_rate('terminal_growth', terminal_growth, allow_negative=True)
  # Zero is not a discount rate; it is the absence of one. Accepted, it left
  # pv_fcfs identical to the undiscounted series and produced a $5.98tn
  # enterprise value for a bank with a $948bn market cap -- success: true,
  # price_per_share 0, no warnings.
  if wacc <= 0:
    raise ValueError(
        f"wacc must be greater than zero, received {wacc}. A zero discount "
        f"rate does not discount: the present value equals the undiscounted "
        f"cash flow and the terminal value is unbounded.")

  # A company does not have negative revenue. Accepted, the model ran happily
  # and returned a negative enterprise value -- an answer no caller asked for
  # and no filing supports.
  if revenue_base is not None and revenue_base <= 0:
    raise ValueError(
        f"revenue_base must be greater than zero, received {revenue_base}. "
        f"A DCF built on it returns a negative enterprise value, which "
        f"describes no company.")
  if terminal_growth >= wacc:
    raise ValueError(
        f"terminal_growth ({terminal_growth}) must be below wacc ({wacc}). "
        f"The perpetuity is undefined at or above it and returns an "
        f"arbitrarily large number rather than a valuation.")
  revenue_growth = [as_rate('revenue_growth', g, allow_negative=True)
                    for g in (revenue_growth or [])]

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
  # None, not zero. A $3.79bn equity divided by a share count nobody supplied
  # was reported as "$0.00 per share" -- not an error and not a null, but the
  # most plausible wrong answer available, reading as "this equity is
  # worthless". The company is still worth what it is worth, so only the
  # per-share figure goes.
  price_per_share = (equity_value / shares_outstanding
                     if shares_outstanding and shares_outstanding > 0
                     else None)

  output = {
    'ticker': ticker,
    'enterprise_value': round(enterprise_value, 2),
    'equity_value': round(equity_value, 2),
    'price_per_share': (round(price_per_share, 2)
                        if price_per_share is not None else None),
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
  if price_per_share is None:
    # A null with no reason is only marginally better than the zero it
    # replaced: the caller still cannot tell a refusal from a gap.
    output['price_per_share_note'] = (
        f"price_per_share is not reported because shares_outstanding was "
        f"{shares_outstanding!r}. The enterprise and equity values above "
        f"stand; only the per-share figure needs a share count.")
  return output


def _wacc_math(beta: float, risk_free_rate: float, equity_risk_premium: float = 0.06,
               cost_of_debt: float = 0, tax_rate: float = 0,
               market_cap: float = 0, total_debt: float = 0) -> dict:
  """WACC via CAPM. All rate inputs as decimals."""
  equity_risk_premium = as_rate('equity_risk_premium', equity_risk_premium)
  risk_free_rate = as_rate('risk_free_rate', risk_free_rate)
  cost_of_debt = as_rate('cost_of_debt', cost_of_debt)
  tax_rate = as_rate('tax_rate', tax_rate)

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
  ebitda_margin = as_rate('ebitda_margin', ebitda_margin)
  capex_pct_revenue = as_rate('capex_pct_revenue', capex_pct_revenue)
  depreciation = as_rate('depreciation', depreciation)
  tax_rate = as_rate('tax_rate', tax_rate)
  debt_interest_rate = as_rate('debt_interest_rate', debt_interest_rate)
  revenue_growth = [as_rate('revenue_growth', g, allow_negative=True)
                    for g in (revenue_growth or [])]

  # An exit multiple at or below zero is not a structure. At -5x the model
  # produced exit_ev -241,576,500,000 -- a manufactured negative enterprise
  # value, the class get_market_data already refuses to build multiples on --
  # and then floored equity_proceeds to 0.0, so the payload read as an
  # ordinary wiped-out deal: moic 0.0, irr_pct -100.0, success true. A client
  # charting MOIC across exit multiples gets a clean zero where they should
  # get an error.
  if exit_multiple is None or exit_multiple <= 0:
    raise ValueError(
      f"exit_multiple must be greater than zero, received {exit_multiple}. "
      f"At or below zero the exit enterprise value is not positive, which is "
      f"not a low valuation but no valuation at all -- and the equity "
      f"proceeds floor at zero, making an invalid input indistinguishable "
      f"from a deal that went to zero.")

  # MOIC ** (1 / hold_years) divides by the hold period, so a zero one left
  # the tool answering "Failed to call tool 'calculate_lbo': float division by
  # zero" -- a stack-trace fragment that names neither the input nor the fix.
  if hold_years is None or hold_years < 1:
    raise ValueError(
      f"hold_years must be at least 1, received {hold_years}. There is no "
      f"entry and exit inside a zero-length hold, so there is no return to "
      f"annualise. Raise hold_years.")

  entry_ebitda = revenue_base * ebitda_margin
  # Every figure below is sized off entry EBITDA: the debt, the exit
  # enterprise value, and the cash that services the debt in between. With
  # none of it the model reported entry_multiple 0 on a 5,115,451,801,600
  # purchase -- a claim that a $5.1tn company was bought at 0x EBITDA -- next
  # to debt_amount 0.0 and leverage_turns_entry 5, which cannot both be true,
  # and moic 0.0 with success true.
  #
  # calculate_dcf meets the same division and answers None with a note,
  # because there its enterprise value survives a missing share count and
  # only the per-share figure is unanswerable. Here nothing survives, so the
  # refusal is the whole model -- the same shape as the unfundable-structure
  # guard below.
  if entry_ebitda <= 0:
    raise ValueError(
      f"entry EBITDA is {entry_ebitda:,.0f} (revenue_base "
      f"{revenue_base:,.0f} x ebitda_margin {ebitda_margin}), so there is "
      f"nothing to lever, nothing to service the debt, and nothing to price "
      f"the exit off. An LBO is financed against EBITDA; at zero the entry "
      f"multiple is not 0x, it is undefined. Correct revenue_base or "
      f"ebitda_margin.")

  debt_amount = entry_ebitda * leverage_turns
  if debt_amount >= entry_ev:
    # Debt is sized off EBITDA and equity off EV, so an entry multiple below
    # the leverage turns puts more debt in the deal than the whole purchase
    # price. The 10% floor below would invent an equity cheque and the model
    # would report a fictitious MOIC and IRR with achieves_20pct_irr True.
    raise ValueError(
      f"acquisition debt {debt_amount:,.0f} ({leverage_turns}x entry EBITDA "
      f"{entry_ebitda:,.0f}) meets or exceeds the entry_ev {entry_ev:,.0f}. "
      f"That is an entry multiple of {entry_ev / entry_ebitda:.2f}x against "
      f"{leverage_turns}x of leverage -- the structure cannot be funded. "
      "Lower leverage_turns or raise entry_ev."
    )
  equity_invested = max(entry_ev - debt_amount, entry_ev * 0.10)  # floor: 10% equity

  current_revenue = revenue_base
  current_debt = debt_amount
  cash_accumulated = 0.0
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
    # Sweep only what there is debt to repay. Paying the full free cash flow
    # against a smaller balance destroyed the surplus -- $191.0bn swept against
    # a $93.4bn balance in one run -- and the excess never reached the equity
    # holder, understating MOIC and IRR. Immaterial where equity is large;
    # material in the thin-equity structure this model exists to evaluate.
    debt_beginning = current_debt
    debt_paydown = max(0.0, min(fcf_after_service, current_debt))
    current_debt = max(0.0, current_debt - debt_paydown)
    cash_accumulated += max(0.0, fcf_after_service - debt_paydown)

    year_by_year.append({
      'year': yr + 1,
      'revenue': round(current_revenue, 2),
      'ebitda': round(ebitda, 2),
      'interest': round(interest, 2),
      'taxes': round(taxes, 2),
      'fcf_after_service': round(fcf_after_service, 2),
      'debt_beginning': round(debt_beginning, 2),
      'debt_paydown': round(debt_paydown, 2),
      'debt_remaining': round(current_debt, 2),
    })

  exit_ebitda = current_revenue * ebitda_margin
  exit_ev = exit_ebitda * exit_multiple
  # Cash swept past a cleared balance belongs to the equity holder.
  equity_proceeds = max(0.0, exit_ev - current_debt) + cash_accumulated

  moic = equity_proceeds / equity_invested if equity_invested > 0 else 0
  # IRR: single cash-on-cash (no interim distributions) -- MOIC^(1/N) - 1
  irr = (moic ** (1.0 / hold_years) - 1) if moic > 0 else -1.0

  return {
    'entry_ev': round(entry_ev, 2),
    'entry_ebitda': round(entry_ebitda, 2),
    'entry_multiple': round(entry_ev / entry_ebitda, 2),
    'debt_amount': round(debt_amount, 2),
    'equity_invested': round(equity_invested, 2),
    'leverage_turns_entry': round(leverage_turns, 2),
    'exit_ebitda': round(exit_ebitda, 2),
    'exit_ev': round(exit_ev, 2),
    'exit_multiple': exit_multiple,
    'debt_at_exit': round(current_debt, 2),
    'cash_accumulated': round(cash_accumulated, 2),
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

  Every price in this payload is built the same way, on the headline's
  `min(perpetuity, exit_multiple)` terminal value. That is why:

  The terminal_sensitivity grid used to strip the perpetuity floor and price
  each multiple on the pure exit-multiple terminal value, so the payload
  carried two price targets from two different methods with nothing to
  separate them. For NVDA the base case reported price_per_share 151.96 at
  terminal_multiple 25 while terminal_sensitivity.base["25x"] read 303.54 --
  the same scenario, the same multiple, twice the price, because the
  perpetuity (4.373tn) beat the exit-multiple terminal value (10.286tn) in the
  headline's min() and the grid ignored it. The bear row showed the same gap,
  60.80 against 119.79. Both numbers were arithmetically correct; a grid keyed
  by the headline's own multiple reads as a sensitivity around the headline,
  and the higher number is the one that ends up in a deck.

  Applying the same rule per cell makes the grid a genuine sensitivity: the
  cell at the base multiple IS the headline price, which is checkable, where a
  label would only have been readable.

  Nothing is lost by it. The grid existed to show how load-bearing the
  terminal multiple is, and it now answers that honestly: where the perpetuity
  binds across the sweep the row goes flat, and a flat row is the true answer
  -- the exit multiple is doing no work at all. The old grid hid exactly that
  behind a 255-to-352 range the model would never have produced.
  terminal_value_method on each scenario names which of the two bound.

  The headline's min() convention is deliberate and is not touched.
  """
  cases = (
    ('bear', bear_growth, bear_margin),
    ('base', base_growth, base_margin),
    ('bull', bull_growth, bull_margin),
  )

  results = {}
  models = {}
  for case_name, growth, margin in cases:
    inputs = dict(base_inputs)
    inputs['revenue_growth'] = growth
    inputs['ebitda_margin'] = margin
    r = _dcf_math(**inputs)
    models[case_name] = r
    results[case_name] = {
      'price_per_share': r['price_per_share'],
      'enterprise_value': r['enterprise_value'],
      'equity_value': r['equity_value'],
      'pv_terminal_value': r['pv_terminal_value'],
      'terminal_value_perpetuity': r['terminal_value_perpetuity'],
      'terminal_value_exit_multiple': r['terminal_value_exit_multiple'],
      'terminal_value_used': r['terminal_value_used'],
      # Which of the two the min() picked. This single fact explains the
      # whole valuation and was never reported.
      'terminal_value_method': (
          'perpetuity'
          if r['terminal_value_used'] == r['terminal_value_perpetuity']
          else 'exit_multiple'),
      'revenue_growth_y1_pct': round(growth[0] * 100, 2) if growth else 0,
      'ebitda_margin_pct': round(margin * 100, 2),
    }
    if r.get('price_per_share_note'):
      results[case_name]['price_per_share_note'] = r['price_per_share_note']

  prices = [results[c]['price_per_share'] for c in ('bear', 'base', 'bull')
            if results[c]['price_per_share'] is not None]

  # Terminal-multiple sensitivity: re-price each scenario at five terminal
  # multiples spaced around the base assumption, under the headline's own
  # min(perpetuity, exit_multiple) rule.
  base_multiple = base_inputs.get('terminal_multiple', 0)
  sensitivity = None
  floor_bound_cases = []
  if base_multiple and base_multiple > 0:
    raw_multiples = [base_multiple - 4, base_multiple - 2,
                     base_multiple, base_multiple + 2, base_multiple + 4]
    # Clamp to positive; a 0x or negative exit multiple is meaningless
    multiples = [m for m in raw_multiples if m > 0]
    sensitivity = {}
    for case_name, growth, margin in cases:
      r = models[case_name]
      pv_fcfs_sum = sum(r['pv_fcfs'])
      original_multiple = r['assumptions']['terminal_multiple']
      # Terminal-year EBITDA: the exit-multiple terminal value at 1x.
      unit_terminal = (r['terminal_value_exit_multiple'] / original_multiple
                       if original_multiple > 0 else 0)
      perpetuity = r['terminal_value_perpetuity']
      wacc = r['assumptions']['wacc']
      n_years = len(growth)
      cash_v = r['assumptions']['cash']
      debt_v = r['assumptions']['debt']
      shares_v = r['assumptions']['shares_outstanding']

      row = {}
      for m in multiples:
        terminal_value_m = min(perpetuity, unit_terminal * m)
        pv_terminal_m = terminal_value_m / ((1 + wacc) ** n_years)
        ev = pv_fcfs_sum + pv_terminal_m
        eq = ev + cash_v - debt_v
        # None, not zero, for the same reason _dcf_math reports None: $0.00
        # per share reads as a worthless equity rather than a question nobody
        # supplied a share count to answer.
        px = eq / shares_v if shares_v and shares_v > 0 else None
        row[f"{m}x"] = round(px, 2) if px is not None else None
      sensitivity[case_name] = row
      if unit_terminal * min(multiples) >= perpetuity:
        floor_bound_cases.append(case_name)

  output = {
    'bear': results['bear'],
    'base': results['base'],
    'bull': results['bull'],
    'price_range': {
      'low': round(min(prices), 2) if prices else None,
      'mid': (round(results['base']['price_per_share'], 2)
              if results['base']['price_per_share'] is not None else None),
      'high': round(max(prices), 2) if prices else None,
    }
  }
  if sensitivity is not None:
    output['terminal_sensitivity'] = sensitivity
    output['terminal_sensitivity_base_multiple'] = base_multiple
    output['terminal_sensitivity_method'] = (
      "Each cell is priced on min(perpetuity, exit multiple), the same "
      "terminal-value rule as the headline bear/base/bull price. The cell at "
      f"{base_multiple}x therefore equals the headline price for its "
      "scenario. Read terminal_value_method on each scenario for which of the "
      "two is binding.")
    if floor_bound_cases:
      output['terminal_sensitivity_floor_note'] = (
        f"The perpetuity is below the exit-multiple terminal value at every "
        f"multiple in this sweep for: {', '.join(floor_bound_cases)}. Those "
        f"rows are flat because the exit multiple is not setting the terminal "
        f"value -- the perpetuity is. Moving the multiple changes nothing "
        f"until it falls below the perpetuity.")
  return output


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
      hi, p_hi = mid, p_mid

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

  # Refuse to issue a strong/weak verdict off fewer than this many evaluable
  # tests — a 9-test framework loses its discriminating power if most tests
  # are skipped due to missing inputs.
  _PIOTROSKI_MIN_EVALUATED = 6

  if max_score_evaluated < _PIOTROSKI_MIN_EVALUATED:
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
  if not _altman_inputs_ok(financials):
    # Substituting a $1 balance sheet to dodge a division makes every ratio
    # enormous and the score meaningless rather than absent.
    return {'score': None, 'zone': None,
            'error': 'total_assets is missing or zero; the Z-score is undefined '
                     'without it and a substituted denominator would produce a '
                     'number rather than an answer'}
  ta = financials['total_assets']
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


def _altman_inputs_ok(financials) -> bool:
    """Whether a Z-score can be computed at all. Total assets is the
    denominator of four of the five ratios, so absent or zero means undefined,
    not zero."""
    ta = (financials or {}).get('total_assets')
    return isinstance(ta, (int, float)) and not isinstance(ta, bool) and ta > 0


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
          name="get_options_metrics",
          description=options_metrics_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_short_interest",
          description=short_interest_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="analyze_exposures",
          description=analyze_exposures_description,
          inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
          name="backtest_signal",
          description=backtest_signal_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "signal": {"type": ["object", "string"], "description": "Rule dict like {'metric':'rsi_14','op':'<','value':30} OR a named signal: oversold_rsi, overbought_rsi, big_drawdown"},
              "hold_days": {"type": "integer", "description": "Trading days to hold after signal fires", "default": 30},
              "cooldown_days": {"type": "integer", "description": "Bars to wait between entries to prevent overlap", "default": 0},
              "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD (optional)"},
              "end_date": {"type": "string", "description": "ISO date YYYY-MM-DD (optional)"}
            },
            "required": ["ticker", "signal"]
          }
        ),
        Tool(
          name="record_thesis_evolution",
          description=record_thesis_evolution_description,
          inputSchema={
            "type": "object",
            "properties": {
              "thesis_id": {"type": "integer", "description": "ID of the thesis being updated"},
              "observation": {"type": "string", "description": "What happened — the new data point that triggered a conviction shift"},
              "conviction_delta": {"type": "number", "description": "Change in conviction (e.g. +0.05, -0.03). Will be added to current confidence and clamped to [0,1]"},
              "tag": {"type": "string", "description": "Optional category: earnings | macro | insider | sector | sentiment | governance | other"}
            },
            "required": ["thesis_id", "observation", "conviction_delta"]
          }
        ),
        Tool(
          name="get_thesis_evolution",
          description=get_thesis_evolution_description,
          inputSchema={
            "type": "object",
            "properties": {
              "thesis_id": {"type": "integer", "description": "ID of the thesis to inspect"}
            },
            "required": ["thesis_id"]
          }
        ),
        Tool(
          name="get_historical_analogue",
          description=historical_analogue_description,
          inputSchema={
            "type": "object",
            "properties": {
              "thesis_description": {"type": "string", "description": "Free-text description of the current setup. Include structural keywords (capex_peak, supply_constrained, valuation_expansion, tech, energy, etc.) for best matches."},
              "top_n": {"type": "integer", "description": "Number of top matches to return", "default": 3}
            },
            "required": ["thesis_description"]
          }
        ),
        Tool(
          name="get_industry_etfs",
          description=industry_etfs_description,
          inputSchema={
            "type": "object",
            "properties": {
              "theme": {"type": "string", "description": "Research theme (e.g. 'AI semis', 'cloud', 'uranium', 'biotech', 'fintech', 'EV')"},
              "top_holdings_per_etf": {"type": "integer", "description": "Number of top holdings to return per ETF", "default": 10}
            },
            "required": ["theme"]
          }
        ),
        Tool(
          name="get_trading_metrics",
          description=trading_metrics_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "period": {"type": "string", "description": "yfinance period spec; only has to cover the widest window requested", "default": "1y"},
              "rvol_lookback": {"type": "integer", "description": "Sessions in the RVOL baseline, which excludes the session being measured", "default": 20},
              "atr_period": {"type": "integer", "description": "Sessions in the Wilder ATR", "default": 14}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_price_history",
          description=price_history_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "period": {"type": "string", "description": "yfinance period spec (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)", "default": "2y"},
              "include_recent_bars": {"type": "integer", "description": "Number of most recent daily OHLCV bars to include", "default": 20}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_13f_holdings",
          description=extract_13f_holdings_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol"},
              "top_n": {"type": "integer", "description": "Number of top holders to return per category (default 10)", "default": 10}
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
            "required": ["ticker"],
            # Every parameter here is a term in an arithmetic
            # expression, so one dropped silently changes the
            # number without changing its shape. `net_debt` was
            # accepted, never read, and the valuation came back
            # confident and wrong.
            "additionalProperties": False
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
            "required": [],
            # Every parameter here is a term in an arithmetic
            # expression, so one dropped silently changes the
            # number without changing its shape. `net_debt` was
            # accepted, never read, and the valuation came back
            # confident and wrong.
            "additionalProperties": False
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
                         "bear_margin", "base_margin", "bull_margin"],
                         # Every parameter here is a term in an arithmetic
                         # expression, so one dropped silently changes the
                         # number without changing its shape. `net_debt` was
                         # accepted, never read, and the valuation came back
                         # confident and wrong.
                         "additionalProperties": False
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
                         "revenue_growth", "debt_interest_rate", "leverage_turns", "exit_multiple"],
                         # Every parameter here is a term in an arithmetic
                         # expression, so one dropped silently changes the
                         # number without changing its shape. `net_debt` was
                         # accepted, never read, and the valuation came back
                         # confident and wrong.
                         "additionalProperties": False
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
                         "interest_expense", "depreciation_abs", "capex_abs", "tax_rate"],
                         # Every parameter here is a term in an arithmetic
                         # expression, so one dropped silently changes the
                         # number without changing its shape. `net_debt` was
                         # accepted, never read, and the valuation came back
                         # confident and wrong.
                         "additionalProperties": False
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
            "required": ["ticker", "market_cap", "ebitda", "capex_abs", "tax_rate", "depreciation_abs"],
            # Every parameter here is a term in an arithmetic
            # expression, so one dropped silently changes the
            # number without changing its shape. `net_debt` was
            # accepted, never read, and the valuation came back
            # confident and wrong.
            "additionalProperties": False
          }
        ),
        Tool(
          name="get_corporate_actions",
          description=(
            "Dividend and split history, with the trailing-twelve-month dividend "
            "and the most recent split ratio.\n\n"
            "Check this before ANY historical per-share comparison. NVDA split "
            "10:1 in June 2024; comparing its FY2023 EPS to FY2025 without "
            "adjusting produces an answer wrong by an order of magnitude, and "
            "nothing in the raw data signals the error.\n\n"
            "'latest_split_ratio' is null when no split falls in the window -- "
            "never 1.0, which would imply a split occurred that changed nothing.\n\n"
            "Dividend basis: the provider restates every historical dividend into "
            "today's share units, so AAPL's 2020-08-07 payment reads 0.205 against "
            "an as-declared $0.82. 'amount' is that restated figure and pairs with "
            "a split-adjusted share count; 'amount_as_declared' is what the company "
            "declared and pairs with the as-filed cover-page count from the same "
            "quarter. Mixing them is out by the split ratio, and in the opposite "
            "direction to get_price_history, whose closes are back-adjusted the "
            "other way."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "years":  {"type": "integer", "description": "Lookback window in years", "default": 10}
            },
            "required": ["ticker"]
          }
        ),
      ]

    @self.server.call_tool()
    @annotating(
      "Yahoo Finance (yfinance)",
      per_tool={
        # These compute from inputs the caller supplies rather than reading an
        # upstream, so naming a data provider would misattribute the number.
        "calculate_dcf": "Nemo (computed)",
        "calculate_scenario_dcf": "Nemo (computed)",
        "calculate_wacc": "Nemo (computed)",
        "calculate_lbo": "Nemo (computed)",
        "calculate_credit_profile": "Nemo (computed)",
        "calculate_capital_returns": "Nemo (computed)",
        "comparable_company_analysis": "Nemo (computed)",
        "backtest_signal": "Nemo (computed)",
        "get_historical_analogue": "Nemo (curated)",
        # Both said "SEC EDGAR" and both read yfinance. corporate_actions.py
        # calls yf.Ticker(symbol); get_institutional_holdings reads
        # yf.Ticker(ticker).major_holders. The response shapes agree
        # independently -- tz-aware timestamps and split-adjusted dividends,
        # neither of which EDGAR publishes.
        #
        # Stated explicitly rather than left to the module default so
        # re-adding EDGAR is a deliberate act. The harm was never only
        # attribution: get_share_count_series.split_adjustment.source says
        # "yfinance", so a caller cross-checking a split ratio between the two
        # believed two independent providers agreed. They had one source read
        # twice, and the EDGAR label is what sold it.
        "extract_13f_holdings": "Yahoo Finance (yfinance)",
        "get_corporate_actions": "Yahoo Finance (yfinance)",
        "analyze_exposures": "Nemo book state",
        "record_thesis_evolution": "Nemo book state",
        "get_thesis_evolution": "Nemo book state",
      },
warnings_per_tool={
        # Sourced from the repository's own documentation. See the sec server
        # for why nothing unsourced is added here.
        "get_short_interest": [
          warning("stale_by_design",
                  "Exchange short interest is published on a lag and is "
                  "normally 2-3 weeks old. There is no live alternative "
                  "configured."),
        ],
        "get_options_metrics": [
          warning("stale_after_hours",
                  "Options quotes can be stale outside market hours, and "
                  "illiquid strikes can yield invalid implied-volatility "
                  "values."),
        ],
        "get_corporate_actions": [
          warning("not_an_independent_source",
                  "Dividends and splits here are yfinance's, not EDGAR's. "
                  "get_share_count_series.split_adjustment reads the same "
                  "upstream -- its own `source` field says \"yfinance\" -- so "
                  "a split ratio agreeing across the two is one source read "
                  "twice, not two providers corroborating. For independent "
                  "confirmation read the filing (get_latest_filing, "
                  "extract_8k_events)."),
        ],
        "extract_13f_holdings": [
          warning("aggregator_not_the_filing",
                  "Holdings come from Yahoo's aggregation of SEC 13F-HR and "
                  "NPORT-P filings, not from EDGAR. Yahoo decides the "
                  "as-of quarter, which managers it carries and how it "
                  "reconciles amendments, and none of that is stated in the "
                  "response. For the filings themselves use the SEC tools "
                  "(get_fund_holdings, compare_fund_holdings)."),
        ],
        "get_market_data": [
          warning("not_execution_grade",
                  "yfinance is convenient for research and is not a "
                  "consolidated execution-grade market-data feed. Do not use "
                  "it to price a trade."),
        ],
        "get_price_history": [
          warning("not_execution_grade",
                  "yfinance is convenient for research and is not a "
                  "consolidated execution-grade market-data feed."),
        ],
      })
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == "get_corporate_actions":
          return await parent.get_corporate_actions(args['ticker'], args.get('years', 10))
        elif name == "get_market_data":
          return await parent.get_market_data(args['ticker'])
        elif name == "extract_13f_holdings":
          return await parent.extract_13f_holdings(args['ticker'], args.get('top_n', 10))
        elif name == "get_options_metrics":
          return await parent.get_options_metrics(args['ticker'])
        elif name == "get_short_interest":
          return await parent.get_short_interest(args['ticker'])
        elif name == "get_price_history":
          return await parent.get_price_history(args['ticker'], args.get('period', '2y'), args.get('include_recent_bars', 20))
        elif name == "get_trading_metrics":
          return await parent.get_trading_metrics(args['ticker'], args.get('period', '1y'), args.get('rvol_lookback', 20), args.get('atr_period', 14))
        elif name == "get_industry_etfs":
          return await parent.get_industry_etfs(args['theme'], args.get('top_holdings_per_etf', 10))
        elif name == "get_historical_analogue":
          return await parent.get_historical_analogue(args['thesis_description'], args.get('top_n', 3))
        elif name == "backtest_signal":
          return await parent.backtest_signal_tool(args)
        elif name == "analyze_exposures":
          return await parent.analyze_exposures_tool()
        elif name == "record_thesis_evolution":
          return await parent.record_thesis_evolution(args['thesis_id'], args['observation'], args['conviction_delta'], args.get('tag'))
        elif name == "get_thesis_evolution":
          return await parent.get_thesis_evolution(args['thesis_id'])
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

  async def get_corporate_actions(self, ticker: str,
                                  years: int = 10) -> List[TextContent]:
    result = await asyncio.to_thread(get_corporate_actions, ticker, years)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

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

  async def extract_13f_holdings(self, ticker: str, top_n: int = 10) -> List[TextContent]:
    result = await asyncio.to_thread(get_institutional_holdings, ticker, top_n)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_options_metrics(self, ticker: str) -> List[TextContent]:
    result = await asyncio.to_thread(get_options_metrics, ticker)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_short_interest(self, ticker: str) -> List[TextContent]:
    result = await asyncio.to_thread(get_short_interest, ticker)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_price_history(self, ticker: str, period: str = '2y', include_recent_bars: int = 20) -> List[TextContent]:
    result = await asyncio.to_thread(get_price_history, ticker, period, include_recent_bars)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_trading_metrics(self, ticker: str, period: str = '1y',
                                rvol_lookback: int = 20,
                                atr_period: int = 14) -> List[TextContent]:
    result = await asyncio.to_thread(get_trading_metrics, ticker, period,
                                     rvol_lookback, atr_period)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_industry_etfs(self, theme: str, top_holdings_per_etf: int = 10) -> List[TextContent]:
    result = await asyncio.to_thread(get_industry_etfs, theme, top_holdings_per_etf)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_historical_analogue(self, thesis_description: str, top_n: int = 3) -> List[TextContent]:
    result = await asyncio.to_thread(get_historical_analogue, thesis_description, top_n)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def analyze_exposures_tool(self) -> List[TextContent]:
    from agent.exposure_analyzer import analyze_exposures
    from state.theses import active_theses
    def _run():
      theses = active_theses()
      return analyze_exposures(theses)
    result = await asyncio.to_thread(_run)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def backtest_signal_tool(self, args: Dict[str, Any]) -> List[TextContent]:
    from agent.backtest_engine import backtest_signal, NAMED_SIGNALS
    signal = args.get('signal')
    if isinstance(signal, str):
      signal = NAMED_SIGNALS.get(signal)
      if signal is None:
        return [TextContent(type="text", text=json.dumps({
          "success": False,
          "error": f"Unknown named signal {args.get('signal')!r}. Available: {list(NAMED_SIGNALS.keys())}",
        }))]

    def _run():
      r = backtest_signal(
        ticker=args['ticker'],
        signal=signal,
        hold_days=int(args.get('hold_days', 30)),
        cooldown_days=int(args.get('cooldown_days', 0)),
        start_date=args.get('start_date'),
        end_date=args.get('end_date'),
        signal_name=args.get('signal') if isinstance(args.get('signal'), str) else 'custom',
      )
      return {
        'ticker': r.ticker, 'signal_name': r.signal_name,
        # Both halves, side by side: date_range alone gave a reader nothing to
        # check the window against, so a clamped one read as the one requested.
        'requested_range': r.requested_range,
        'date_range': r.date_range,
        'n_trades': r.n_trades, 'hit_rate_pct': r.hit_rate,
        'mean_return_pct': r.mean_return, 'median_return_pct': r.median_return,
        'best_trade_pct': r.best_trade, 'worst_trade_pct': r.worst_trade,
        'mean_hold_days': r.mean_hold_days,
        'max_drawdown_pct_in_any_trade': r.max_drawdown_pct,
        'sharpe_simple_annualized': r.sharpe_simple,
        # `warning` means the backtest did not run, and `success` is derived
        # from it. `caveats` means it ran and the numbers need reading in
        # context -- a short sample, or a window the data could not cover.
        'warning': r.warning,
        'caveats': r.caveats,
        'sample_trades': [{
          'entry_date': t.entry_date, 'exit_date': t.exit_date,
          'entry_price': t.entry_price, 'exit_price': t.exit_price,
          'return_pct': t.return_pct, 'hold_days': t.hold_days,
          'max_dd_in_trade_pct': t.max_dd_in_trade,
        } for t in r.trades[:10]],
        'success': r.warning is None,
      }
    result = await asyncio.to_thread(_run)
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def record_thesis_evolution(self, thesis_id: int, observation: str, conviction_delta: float, tag = None) -> List[TextContent]:
    from state.theses import record_thesis_evolution as _rec, get_thesis
    try:
      eid = await asyncio.to_thread(_rec, thesis_id, observation, float(conviction_delta), tag)
      th = await asyncio.to_thread(get_thesis, thesis_id)
      result = {
        "success": True,
        "evolution_id": eid,
        "thesis_id": thesis_id,
        "new_conviction": th['confidence'] if th else None,
      }
    except Exception as e:
      result = {"success": False, "error": f"{type(e).__name__}: {e}"}
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def get_thesis_evolution(self, thesis_id: int) -> List[TextContent]:
    from state.theses import get_thesis_evolution as _get, get_thesis
    try:
      log = await asyncio.to_thread(_get, thesis_id)
      th = await asyncio.to_thread(get_thesis, thesis_id)
      result = {
        "success": True,
        "thesis_id": thesis_id,
        "ticker": th['ticker'] if th else None,
        "current_conviction": th['confidence'] if th else None,
        "falsifiers": th.get('falsifiers') if th else None,
        "variant_perception": th.get('variant_perception') if th else None,
        "evolution_count": len(log),
        "evolution": log,
      }
    except Exception as e:
      result = {"success": False, "error": f"{type(e).__name__}: {e}"}
    return [TextContent(type="text", text=json.dumps(_to_native(result), default=str))]

  async def comparable_company_analysis(self, comparables: List[str]) -> List[TextContent]:
    tasks = [asyncio.to_thread(get_data, ticker) for ticker in comparables]
    data = await asyncio.gather(*tasks)

    def _reason(peer: dict, metric: str) -> Optional[str]:
      """Why this peer contributes no `metric`, in its own terms.

      `get_market_data` now refuses a symbol it could not resolve and records
      the input behind every suppressed multiple, so the cause of each
      exclusion is available here. It used to be discarded at this boundary
      and replaced downstream by one fixed sentence covering all of them --
      a symbol that is not a company was folded into `excluded_absent` with
      no mention that the lookup had failed at all.
      """
      if peer.get('success') is False:
        return peer.get('error')
      detail = peer.get('multiples_suppressed_detail') or {}
      return detail.get(metric) or peer.get('multiples_suppressed_reason')

    def _block(metric: str) -> dict:
      return _to_native(peer_distribution(
          [d.get(metric) for d in data],
          tickers=[d.get('ticker') or t for d, t in zip(data, comparables)],
          reasons={(d.get('ticker') or t): _reason(d, metric)
                   for d, t in zip(data, comparables)},
      ))

    result = {
      'comparables': comparables,
      'pe_ratio': _block('pe_ratio'),
      'pb_data': _block('pb_ratio'),
      'ev_revenue_data': _block('ev_revenue'),
      'ev_ebitda_data': _block('ev_ebitda'),
      'ev_ebit_data': _block('ev_ebit'),
    }
    return [TextContent(type="text", text=json.dumps(result))]

  async def calculate_dcf(self, args: Dict[str, Any]) -> List[TextContent]:
    # A DCF with no revenue base produces enterprise_value 0 and
    # price_per_share 0, which reads as a real valuation rather than a missing
    # input and is exactly the kind of number that ends up in a thesis.
    # calculate_wacc already refuses this way; match it.
    if not args.get('revenue_base'):
      return [TextContent(type='text', text=json.dumps({
        'error': ("revenue_base is zero or absent, cannot compute a DCF. "
                  "Resolve it with get_revenue_base first, then pass it in."),
        'ticker': args.get('ticker', ''),
      }))]
    result = _dcf_math(
      revenue_base=args.get('revenue_base', 0),
      ebitda_margin=args.get('ebitda_margin', 0),
      capex_pct_revenue=args.get('capex_pct_revenue', 0),
      tax_rate=args.get('tax_rate', 0),
      depreciation=args.get('depreciation', 0),
      # Declared as an array in the schema ("SET TO [0,0,0,0,0]"), so its
      # absent default has to be a sequence rather than a scalar zero.
      revenue_growth=args.get('revenue_growth') or [0, 0, 0, 0, 0],
      wacc=args.get('wacc', 0),
      terminal_growth=args.get('terminal_growth', 0),
      terminal_multiple=args.get('terminal_multiple', 0),
      cash=args.get('cash', 0),
      debt=args.get('debt', 0),
      shares_outstanding=args.get('shares_outstanding', 0),
      ticker=args.get('ticker', ''),
    )
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_wacc(self, args: Dict[str, Any]) -> List[TextContent]:
    result = _wacc_math(
      beta=args.get('beta', 0),
      risk_free_rate=args.get('risk_free_rate', 0),
      equity_risk_premium=args.get('equity_risk_premium', 0.06),
      cost_of_debt=args.get('cost_of_debt', 0),
      tax_rate=args.get('tax_rate', 0),
      market_cap=args.get('market_cap', 0),
      total_debt=args.get('total_debt', 0),
    )
    return [TextContent(type='text', text=json.dumps(result))]

  async def calculate_scenario_dcf(self, args: Dict[str, Any]) -> List[TextContent]:
    # Same refusal as calculate_dcf: this runs _dcf_math three times, so a
    # missing revenue_base produces three price targets of 0 instead of one.
    if not args.get('revenue_base'):
      return [TextContent(type='text', text=json.dumps({
        'error': ("revenue_base is zero or absent, cannot compute a scenario DCF. "
                  "Resolve it with get_revenue_base first, then pass it in."),
        'ticker': args.get('ticker', ''),
      }))]
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
    try:
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
    except ValueError as e:
      # Unfundable capital structure -- surface the refusal rather than a model.
      return [TextContent(type='text', text=json.dumps({
        'error': str(e), 'ticker': args.get('ticker', ''),
      }))]
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
        await self.server.run(
          read_stream,
          write_stream,
          self.server.create_initialization_options(),
        )
    except Exception as e:
      print(f"Financial Analysis Server error: {e}", file=sys.stderr, flush=True)
      traceback.print_exc(file=sys.stderr)
      raise


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.financial_modeling_engine.analysis_tools [server|http]", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "http":
    # Streamable HTTP, for a host a client connects to rather than one
    # that spawns it. stdio stays the default for local use.
    from tools.mcp_http import run_http
    print("Starting financial engine over streamable HTTP", file=sys.stderr, flush=True)
    run_http(Financial_Analysis().server)

  elif sys.argv[1] == "server":
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
