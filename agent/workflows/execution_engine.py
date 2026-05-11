"""
Tool execution engine -- runs MCP tools and processes results by type.
Extracted from analysis_workflow.py to keep the graph definition clean.
"""
from typing import Dict, Any, List
from ..MCP_manager import MCPConnectionManager
from ..Search_Summarizer_Agent import Search_Summarizer_Agent
from ..News_Processing_Agent import News_Processing_Agent
from ..cache import Session_Cache
import asyncio
import json
import sys


# SEC tools return structured XBRL data -- pass through without summarization
SEC_TOOLS = {
  'get_revenue_base', 'get_ebitda_margin', 'get_capex_pct_revenue',
  'get_depreciation', 'get_tax_rate', 'extract_8k_events',
  'get_disclosures_names', 'extract_disclosure_data', 'get_latest_filing',
  'extract_proxy_compensation', 'extract_governance_data',
  'get_margin_breakdown', 'get_historical_fcf', 'get_working_capital',
}

# Financial modeling tools -- deterministic calculations, pass through
FINANCIAL_TOOLS = {
  'get_market_data', 'calculate_dcf', 'calculate_wacc', 'comparable_company_analysis'
}

# Market intelligence tools -- Finnhub forward-looking data, pass through
MARKET_INTEL_TOOLS = {
  'get_company_news', 'get_market_news', 'get_insider_transactions',
  'get_earnings_calendar', 'get_analyst_recommendations', 'get_company_peers',
  'get_basic_financials', 'get_earnings_surprises', 'get_forward_estimates',
  'get_financial_statements', 'get_company_profile', 'get_insider_sentiment',
  # Removed (Finnhub premium / unavailable on free tier):
  # 'get_upgrade_downgrades', 'get_price_target', 'get_sector_metrics'
}

# Macro tools -- FRED economic data, pass through
MACRO_TOOLS = {
  'get_macro_snapshot', 'get_fred_series', 'get_treasury_yields', 'search_fred',
  'get_credit_spreads',
}

# News tools -- unstructured text that needs LLM analysis
NEWS_TOOLS = {'get_company_news', 'get_market_news'}

# Tools safe to cache within a session.
CACHEABLE_TOOLS = SEC_TOOLS | FINANCIAL_TOOLS | MARKET_INTEL_TOOLS | MACRO_TOOLS

# Tools whose results get flattened into the shared variable store
FLATTENABLE_TOOLS = SEC_TOOLS | FINANCIAL_TOOLS

# Metadata keys to skip when flattening tool results into variables
SKIP_KEYS = {
  'success', 'error', 'concept_used', 'concept', 'period_end', 'filing_date',
  'tax_concept_used', 'pretax_concept_used', 'operating_income_concept_used',
  'd&a_concept_used', 'capex_concept_used', 'data_type', 'data_shape',
  'columns', 'sample_data'
}

# Variable keys that are percentages (e.g. 15.61 for 15.61%) and need /100 for tool args
PERCENT_FIELDS = {'effective_tax_rate', 'ebitda_margin_percent', 'capex_pct_revenue', 'd&a_pct'}

# Canonical argument templates for calculation tools.
_DCF_ARG_TEMPLATE: Dict[str, Any] = {
  'revenue_base':       0,
  'ebitda_margin':      0,
  'capex_pct_revenue':  0,
  'tax_rate':           0,
  'depreciation':       0,
  'revenue_growth':     [0, 0, 0, 0, 0],
  'wacc':               0,
  'terminal_growth':    0,
  'terminal_multiple':  0,
  'cash':               0,
  'debt':               0,
  'shares_outstanding': 0,
}

_WACC_ARG_TEMPLATE: Dict[str, Any] = {
  'beta':                 0,
  'risk_free_rate':       0,
  'equity_risk_premium':  0.06,   # Damodaran 6% ERP -- always hardcoded
  'cost_of_debt':         0,
  'tax_rate':             0,
  'market_cap':           0,
  'total_debt':           0,
}

# Cap concurrent MCP API calls. Finnhub free tier 429s above ~5 req/sec.
_MCP_SEMAPHORE = asyncio.Semaphore(4)

# Maps tool argument names to variable store keys when they differ
ARGUMENT_ALIASES = {
  'ebitda_margin': 'ebitda_margin_percent',
  'depreciation': 'd&a_pct',
  'tax_rate': 'effective_tax_rate',
  'market_cap': 'marketCap',
  'total_debt': 'totalDebt',
  'cash': 'totalCash',
  'debt': 'totalDebt',
  'shares_outstanding': 'sharesOutstanding',
}


def _flatten_market_intel(variables: Dict[str, Any], tool_name: str, data: Dict[str, Any]):
  """Flatten market intelligence results into the shared variable store."""
  if not isinstance(data, dict) and not isinstance(data, list):
    return

  if tool_name == 'get_insider_transactions':
    for key in ('total_bought', 'total_sold', 'net_shares', 'buy_count', 'sell_count', 'signal'):
      if key in data:
        variables[f"insider.{key}"] = data[key]
        variables[f"insider_{key}"] = data[key]
    top = data.get('top_insiders', [])
    if top:
      variables['insider.top_insiders'] = top
      variables['insider_top_insiders'] = top
    for bucket in ('recent_30d', 'recent_90d'):
      val = data.get(bucket, {})
      if val and isinstance(val, dict) and val.get('net', 0) != 0:
        variables[f"insider.{bucket}_net"] = val['net']
        variables[f"insider_{bucket}_net"] = val['net']

  elif tool_name == 'get_analyst_recommendations':
    for key in ('consensus', 'trend', 'total_analysts'):
      if key in data:
        variables[f"analyst.{key}"] = data[key]
        variables[f"analyst_{key}"] = data[key]
    latest = data.get('latest', {})
    if latest:
      variables['analyst.latest'] = latest
      variables['analyst_latest'] = latest

  elif tool_name == 'get_basic_financials':
    metrics = data.get('metric', {})
    for key, value in metrics.items():
      if value is not None and not (isinstance(value, float) and value != value):
        variables[f"financials.{key}"] = value

  elif tool_name == 'get_earnings_calendar':
    total = data.get('total_companies', 0)
    if total:
      variables['earnings.total_companies'] = total
      variables['earnings_total_companies'] = total
    by_date = data.get('by_date', [])
    if by_date:
      variables['earnings.by_date'] = by_date

  elif tool_name == 'get_company_peers':
    if isinstance(data, list):
      variables['company_peers'] = data

  elif tool_name == 'get_price_target':
    for key in ('targetMean', 'targetMedian', 'targetHigh', 'targetLow', 'numberOfAnalysts'):
      if key in data and data[key] is not None:
        variables[f"price_target.{key}"] = data[key]
        variables[f"price_target_{key}"] = data[key]

  elif tool_name == 'get_earnings_surprises':
    for key in ('beat_count', 'miss_count', 'avg_surprise_pct', 'beat_rate_pct', 'total_periods'):
      if key in data and data[key] is not None:
        variables[f"earnings_quality.{key}"] = data[key]

  elif tool_name == 'get_forward_estimates':
    for label in ('eps', 'revenue_B', 'ebitda_B'):
      periods = data.get(label, {}).get('periods', [])
      if periods and isinstance(periods[0], dict):
        avg = periods[0].get('avg')
        if avg is not None:
          variables[f"estimate.{label}_next"] = avg

  elif tool_name == 'get_company_profile':
    for key in ('finnhubIndustry', 'gics', 'gicsSubIndustry', 'country', 'employeeTotal', 'marketCapitalization'):
      if key in data and data[key] is not None:
        variables[f"profile.{key}"] = data[key]

  elif tool_name == 'get_insider_sentiment':
    for key in ('signal', 'avg_mspr'):
      if key in data and data[key] is not None:
        variables[f"insider_sentiment.{key}"] = data[key]
        variables[f"insider_mspr_{key}"] = data[key]

  elif tool_name == 'get_financial_statements':
    # data is already unwrapped by _process_market_intel (envelope stripped)
    # Shape: {"statement": "cf"|"ic"|"bs", "freq": "annual", "periods": [{...}, ...]}
    stmt = data.get('statement', '')
    periods = data.get('periods', [])
    if not (periods and isinstance(periods, list)):
      return
    latest = periods[0]  # most recent annual period

    if stmt == 'cf':
      for field in ('dividendsPaid', 'repurchaseOfCapitalStock', 'operatingCashFlow',
                    'capitalExpenditures', 'freeCashFlow'):
        if field in latest and latest[field] is not None:
          variables[f"cf.{field}"] = latest[field]
          variables[field] = latest[field]

    elif stmt == 'ic':
      for field in ('revenue', 'costOfRevenue', 'grossProfit', 'operatingExpense',
                    'operatingIncome', 'ebitda', 'ebit', 'netIncome', 'eps', 'epsDiluted'):
        if field in latest and latest[field] is not None:
          variables[f"ic.{field}"] = latest[field]
          variables[field] = latest[field]
      rev = latest.get('revenue')
      if rev and rev > 0:
        gp = latest.get('grossProfit')
        if gp is not None:
          variables['ic.grossMargin'] = gp / rev
        ni = latest.get('netIncome')
        if ni is not None:
          variables['ic.netMargin'] = ni / rev

    elif stmt == 'bs':
      for field in ('totalAssets', 'totalCurrentAssets', 'cashAndEquivalents',
                    'totalLiabilities', 'totalCurrentLiabilities', 'longTermDebt',
                    'shortTermDebt', 'totalDebt', 'totalEquity', 'stockholdersEquity',
                    'goodwill', 'intangibleAssets'):
        if field in latest and latest[field] is not None:
          variables[f"bs.{field}"] = latest[field]
          # Don't overwrite flat keys produced by get_market_data
          if field not in ('totalDebt', 'cashAndEquivalents'):
            variables[field] = latest[field]

    extracted = [k for k in variables if k.startswith(f"{stmt}.")]
    print(f"[Validate Flatten] get_financial_statements ({stmt}) wrote {len(extracted)} keys",
          file=sys.stderr, flush=True)


def _flatten_news_analysis(variables: Dict[str, Any], tool_name: str, analysis: Dict[str, Any]):
  """Flatten news analysis results into the shared variable store."""
  prefix = 'company_news' if tool_name == 'get_company_news' else 'market_news'

  for key in ('overall_sentiment', 'sentiment_score', 'articles_analyzed', 'articles_relevant'):
    if key in analysis:
      variables[f"{prefix}.{key}"] = analysis[key]
      variables[f"{prefix}_{key}"] = analysis[key]

  themes = analysis.get('key_themes', [])
  if themes:
    variables[f"{prefix}.key_themes"] = themes


def _flatten_macro(variables: Dict[str, Any], tool_name: str, data: Dict[str, Any]):
  """Flatten macro data results into the shared variable store.

  Key output: risk_free_rate (decimal) for calculate_wacc auto-resolution.
  """
  if not isinstance(data, dict):
    return

  if tool_name == 'get_macro_snapshot':
    dgs10 = data.get('DGS10', {})
    if isinstance(dgs10, dict) and dgs10.get('current') is not None:
      variables['risk_free_rate'] = round(dgs10['current'] / 100, 4)
      variables['macro.treasury_10y'] = dgs10['current']

    ff = data.get('FEDFUNDS', {})
    if isinstance(ff, dict) and ff.get('current') is not None:
      variables['macro.fed_funds'] = ff['current']

    hy = data.get('BAMLH0A0HYM2', {})
    if isinstance(hy, dict) and hy.get('current') is not None:
      variables['macro.hy_spread'] = hy['current']
      variables['hy_spread'] = hy['current']

    cpi = data.get('CPIAUCSL', {})
    if isinstance(cpi, dict) and cpi.get('yoy_pct') is not None:
      variables['macro.cpi_yoy'] = cpi['yoy_pct']

    pce = data.get('PCEPILFE', {})
    if isinstance(pce, dict) and pce.get('yoy_pct') is not None:
      variables['macro.core_pce_yoy'] = pce['yoy_pct']

    unrate = data.get('UNRATE', {})
    if isinstance(unrate, dict) and unrate.get('current') is not None:
      variables['macro.unemployment_rate'] = unrate['current']

    gdp = data.get('A191RL1Q225SBEA', {})
    if isinstance(gdp, dict) and gdp.get('current') is not None:
      variables['macro.real_gdp_growth'] = gdp['current']

    spread = data.get('T10Y2Y', {})
    if isinstance(spread, dict) and spread.get('current') is not None:
      variables['macro.spread_10y_2y'] = spread['current']

    nfci = data.get('NFCI', {})
    if isinstance(nfci, dict) and nfci.get('current') is not None:
      variables['macro.NFCI'] = nfci['current']
      variables['NFCI'] = nfci['current']

    umich = data.get('UMCSENT', {})
    if isinstance(umich, dict) and umich.get('current') is not None:
      variables['macro.consumer_sentiment'] = umich['current']

  elif tool_name == 'get_treasury_yields':
    curve = data.get('curve', {})
    if '10Y' in curve:
      variables['risk_free_rate'] = round(curve['10Y'] / 100, 4)
    for maturity, rate in curve.items():
      variables[f'treasury.{maturity}'] = rate

    spreads = data.get('spreads', {})
    if '10Y_2Y' in spreads:
      variables['treasury.spread_10Y_2Y'] = spreads['10Y_2Y']
    if '10Y_3M' in spreads:
      variables['treasury.spread_10Y_3M'] = spreads['10Y_3M']

    shape = data.get('shape', 'unknown')
    variables['yield_curve_shape'] = shape

  elif tool_name == 'get_credit_spreads':
    for sid, entry in data.items():
      if not isinstance(entry, dict) or 'error' in entry:
        continue
      bps = entry.get('current_bps')
      if bps is not None:
        variables[f"credit_spread.{sid}"] = bps
    hy = data.get('BAMLH0A0HYM2', {})
    if isinstance(hy, dict) and hy.get('current_bps') is not None:
      variables['credit_spread_hy'] = hy['current_bps']
    bbb = data.get('BAMLC0A4CBBB', {})
    if isinstance(bbb, dict) and bbb.get('current_bps') is not None:
      variables['credit_spread_bbb'] = bbb['current_bps']


def _flatten_result(variables: Dict[str, Any], tool_name: str, result: Dict[str, Any]):
  """Flatten a tool result into the shared variable store."""
  for key, value in result.items():
    if key in SKIP_KEYS or key.endswith('_concept_used'):
      continue
    if isinstance(value, (dict, list)):
      continue
    if value is None:
      continue
    if isinstance(value, float) and value != value:
      continue
    variables[f"{tool_name}.{key}"] = value
    variables[key] = value


def _has_value(v) -> bool:
  """Check if a value is meaningful (not a placeholder)."""
  if v is None:
    return False
  if v == 0 or v == 0.0:
    return False
  if v == [] or v == [0]:
    return False
  if isinstance(v, str):
    if v.startswith("FROM_"):
      return False
    # Angle-bracket placeholders from Nemotron: <beta_from_get_market_data>, <value>, etc.
    if v.startswith("<") and v.endswith(">"):
      return False
    try:
      if float(v) == 0:
        return False
    except ValueError:
      pass
  if isinstance(v, list) and len(v) > 0 and all(x == 0 for x in v):
    return False
  return True


def _resolve_args(arguments: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
  """Resolve placeholder/zero arguments from the shared variable store.

  For each argument with a falsy value:
    1. Direct key match in variables
    2. Alias match via ARGUMENT_ALIASES
    3. Derived values (e.g. cost_of_debt from interestExpense / totalDebt)
  Applies /100 conversion for PERCENT_FIELDS.
  """
  if not variables:
    return arguments

  args = dict(arguments)

  for arg_name, arg_value in arguments.items():
    if _has_value(arg_value):
      continue

    resolved_key = None
    resolved_value = None

    # 1. Direct match
    if arg_name in variables:
      resolved_key = arg_name
      resolved_value = variables[arg_name]

    # 2. Alias match
    if resolved_value is None and arg_name in ARGUMENT_ALIASES:
      alias = ARGUMENT_ALIASES[arg_name]
      if alias in variables:
        resolved_key = alias
        resolved_value = variables[alias]

    # 3. Special case: cost_of_debt derived from interest expense / total debt
    if resolved_value is None and arg_name == 'cost_of_debt':
      interest = variables.get('interestExpense')
      debt = variables.get('totalDebt')
      if interest and debt and debt > 0:
        resolved_value = abs(interest) / debt

    # 4. Special case: revenue_growth from get_basic_financials revenueGrowthTTMYoy
    if resolved_value is None and arg_name == 'revenue_growth':
      ttm_pct = variables.get('financials.revenueGrowthTTMYoy')
      five_y_pct = variables.get('financials.revenueGrowth5Y')
      if ttm_pct is not None and isinstance(ttm_pct, (int, float)):
        yr1 = ttm_pct / 100 if abs(ttm_pct) > 1 else float(ttm_pct)
        if five_y_pct is not None and isinstance(five_y_pct, (int, float)):
          long_run = five_y_pct / 100 if abs(five_y_pct) > 1 else float(five_y_pct)
        else:
          long_run = yr1 * 0.5
        step = (yr1 - long_run) / 4
        resolved_value = [round(yr1 - step * i, 4) for i in range(5)]

    # 5. Special case: terminal_growth from GDP growth, default 2.5%
    if resolved_value is None and arg_name == 'terminal_growth':
      gdp_growth = variables.get('macro.real_gdp_growth')
      if gdp_growth is not None and isinstance(gdp_growth, (int, float)):
        resolved_value = max(round(gdp_growth / 100, 4), 0.0)
      else:
        resolved_value = 0.025

    # 6. Special case: terminal_multiple from sector EV/EBITDA
    if resolved_value is None and arg_name == 'terminal_multiple':
      ev_ebitda = variables.get('financials.evEbitdaTTM')
      if ev_ebitda is not None and isinstance(ev_ebitda, (int, float)) and ev_ebitda > 0:
        resolved_value = round(float(ev_ebitda), 1)

    if resolved_value is not None:
      if resolved_key and resolved_key in PERCENT_FIELDS:
        resolved_value = resolved_value / 100
      args[arg_name] = resolved_value

  # Post-process: normalize percent-form DCF args that LLM passed explicitly
  _DCF_PERCENT_ARGS = {'ebitda_margin', 'capex_pct_revenue', 'tax_rate', 'depreciation'}
  for _field in _DCF_PERCENT_ARGS:
    if _field in args and isinstance(args[_field], (int, float)) and args[_field] > 1:
      args[_field] = round(args[_field] / 100, 6)

  return args


# Tools that MUST run after others because they read the variable store
SEQUENTIAL_TOOLS = {
  'calculate_wacc',
  'calculate_dcf',
  'get_urls_content',
  'extract_disclosure_data',
}


def _needs_sequential(tool: Dict[str, Any]) -> bool:
  """Return True if this tool must run after earlier tools complete."""
  if tool['tool'] in SEQUENTIAL_TOOLS:
    return True
  args_str = json.dumps(tool.get('arguments', {}))
  return 'FROM_SEARCH' in args_str or 'FROM_PREVIOUS' in args_str


async def _call_single_tool(
  tool: Dict[str, Any],
  label: str,
  mcp: MCPConnectionManager,
  cache: Session_Cache,
  variables: Dict[str, Any],
  semaphore: asyncio.Semaphore = None,
) -> tuple:
  """Execute one tool and return (tool_name, arguments, tool_result)."""
  tool_name = tool['tool']
  original_args = tool['arguments']

  if tool_name == 'calculate_dcf':
    merged = {**_DCF_ARG_TEMPLATE, **original_args}
  elif tool_name == 'calculate_wacc':
    merged = {**_WACC_ARG_TEMPLATE, **original_args}
  else:
    merged = original_args

  arguments = _resolve_args(merged, variables)
  resolved = {k for k in arguments if arguments[k] != merged.get(k)}
  suffix = f" (resolved: {resolved})" if resolved else ""
  print(f"{label} Executing: {tool_name}{suffix}", flush=True)
  print(f"  Arguments: {json.dumps(arguments, indent=2, default=str)}", flush=True)

  if cache and tool_name in CACHEABLE_TOOLS:
    cached = cache.get(tool_name, arguments)
    if cached is not None:
      _cached_inner = cached.get('data', cached) if isinstance(cached, dict) else cached
      _cached_errored = isinstance(cached, dict) and (
        'error' in cached or (isinstance(_cached_inner, dict) and 'error' in _cached_inner)
      )
      if not _cached_errored:
        print(f"  [CACHE HIT] {tool_name}", flush=True)
        return tool_name, arguments, cached
      else:
        print(f"  [CACHE SKIP] {tool_name}: cached result was an error, retrying fresh...", flush=True)

  _sem = semaphore or _MCP_SEMAPHORE
  async with _sem:
    try:
      tool_result = await mcp.call_tool(tool_name, arguments)
    except RuntimeError as e:
      print(f"  [SKIP] {e}", flush=True)
      return tool_name, arguments, {"error": str(e)}

  _result_inner = tool_result.get('data', tool_result) if isinstance(tool_result, dict) else tool_result
  _result_errored = isinstance(tool_result, dict) and (
    'error' in tool_result or (isinstance(_result_inner, dict) and 'error' in _result_inner)
  )
  if cache and tool_name in CACHEABLE_TOOLS and not _result_errored:
    cache.put(tool_name, arguments, tool_result)

  print(f"  Result preview: {str(tool_result)[:300]}...\n", flush=True)
  return tool_name, arguments, tool_result


def _integrate_result(
  tool_name: str,
  arguments: Dict[str, Any],
  tool_result: Any,
  ticker: str,
  results: List[Dict[str, Any]],
  variables: Dict[str, Any],
  news_agent,
  cache: Session_Cache,
):
  """Integrate one tool result into results list and variable store."""
  if isinstance(tool_result, dict) and 'error' in tool_result and tool_name not in SEC_TOOLS:
    results.append({"type": "other", "tool": tool_name, "data": tool_result})
    return

  if tool_name in FLATTENABLE_TOOLS:
    _flatten_result(variables, tool_name, tool_result)

  if tool_name in SEC_TOOLS:
    results.append(_process_sec(tool_name, tool_result))
  elif tool_name in FINANCIAL_TOOLS:
    results.append(_process_financial(tool_name, tool_result))
  elif tool_name in MACRO_TOOLS:
    processed = _process_macro(tool_name, tool_result)
    results.append(processed)
    _flatten_macro(variables, tool_name, processed['data'])
  elif tool_name in MARKET_INTEL_TOOLS:
    if tool_name in NEWS_TOOLS:
      _process_news(tool_name, tool_result, ticker, news_agent, results, variables, cache, arguments)
    else:
      processed = _process_market_intel(tool_name, tool_result)
      results.append(processed)
      _flatten_market_intel(variables, tool_name, processed['data'])
  else:
    results.append(_process_other(tool_name, tool_result))


async def run_tools(
  tool_sequence: List[Dict[str, Any]],
  ticker: str,
  mcp: MCPConnectionManager,
  summarizer: Search_Summarizer_Agent,
  cache: Session_Cache = None,
  variables: Dict[str, Any] = None,
  news_agent: News_Processing_Agent = None
) -> tuple:
  """Execute a tool sequence with parallel I/O where safe.

  Independent data-fetch tools (SEC, macro, market intel, search) all fire
  concurrently via asyncio.gather(). Calculation tools that depend on the
  variable store (WACC, DCF, get_urls_content) run sequentially afterward.

  Returns:
    Tuple of (results list, updated variables dict, exec_stats dict)
  """
  results: List[Dict[str, Any]] = []
  variables = dict(variables) if variables else {}

  for idx, tool in enumerate(tool_sequence, 1):
    if 'tool' not in tool or 'arguments' not in tool:
      raise KeyError(f"Unable to find tool and arguments in step {idx}: {tool}")

  parallel_batch = [t for t in tool_sequence if not _needs_sequential(t)]
  sequential_batch = [t for t in tool_sequence if _needs_sequential(t)]

  total = len(tool_sequence)
  print(f"\n{'='*60}", flush=True)
  print(f"EXECUTION PHASE - {total} tools ({len(parallel_batch)} parallel, {len(sequential_batch)} sequential)", flush=True)
  print(f"{'='*60}\n", flush=True)

  # --- Parallel batch ---
  if parallel_batch:
    print(f"[Parallel] Launching {len(parallel_batch)} tools concurrently...", flush=True)
    coros = [
      _call_single_tool(tool, f"[P{i+1}/{len(parallel_batch)}]", mcp, cache, variables,
                        semaphore=_MCP_SEMAPHORE)
      for i, tool in enumerate(parallel_batch)
    ]
    parallel_results = await asyncio.gather(*coros, return_exceptions=True)

    for tool, outcome in zip(parallel_batch, parallel_results):
      tool_name = tool['tool']
      if isinstance(outcome, Exception):
        print(f"  [ERROR] {tool_name}: {outcome}", flush=True)
        results.append({"type": "other", "tool": tool_name, "data": {"error": str(outcome)}})
        continue
      rname, rargs, rresult = outcome
      _integrate_result(rname, rargs, rresult, ticker, results, variables, news_agent, cache)

  # --- Sequential batch ---
  for idx, tool in enumerate(sequential_batch, 1):
    tool_name = tool['tool']
    label = f"[S{idx}/{len(sequential_batch)}]"

    outcome = await _call_single_tool(tool, label, mcp, cache, variables)
    _, rargs, rresult = outcome

    if isinstance(rresult, dict) and rresult.get('error') and tool_name not in SEQUENTIAL_TOOLS:
      results.append({"type": "other", "tool": tool_name, "data": rresult})
      continue

    if tool_name == 'search':
      await _process_search(rresult, rargs, ticker, mcp, summarizer, results, variables)
    elif 'get_urls_content' in tool_name:
      await _process_scrape(rresult, ticker, summarizer, results, variables)
    else:
      _integrate_result(tool_name, rargs, rresult, ticker, results, variables, news_agent, cache)

  def _is_error(r):
    data = r.get('data')
    if isinstance(data, dict) and data.get('error'):
      return True
    return False

  error_count = sum(1 for r in results if _is_error(r))
  success_count = len(results) - error_count
  exec_stats = {"errors": error_count, "successes": success_count, "total": len(results)}

  print(f"\n{'='*60}", flush=True)
  print(f"EXECUTION COMPLETE - {len(results)} items retained ({success_count} ok, {error_count} errors)", flush=True)
  print(f"VARIABLES: {len(variables)} keys accumulated", flush=True)
  print(f"{'='*60}\n", flush=True)

  return results, variables, exec_stats


def _process_sec(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Pass-through] {tool_name}: structured SEC data", flush=True)
  return {"type": "sec_data", "tool": tool_name, "data": tool_result}


async def _process_search(
  tool_result: Dict,
  arguments: Dict,
  ticker: str,
  mcp: MCPConnectionManager,
  summarizer: Search_Summarizer_Agent,
  results: List[Dict[str, Any]],
  variables: Dict[str, Any] = None
):
  search_results = tool_result.get('search_result', [])
  query_info = arguments.get('query', {})

  if not search_results:
    print(f"  [Search] WARNING: 0 results returned -- DuckDuckGo may be rate-limited or blocked.", flush=True)
    results.append({"type": "other", "tool": "search", "data": {
      "error": "Search returned 0 results. DuckDuckGo may be rate-limited. Structured tool data (SEC/Finnhub) is still available."
    }})
    return

  snippets = []
  for item in search_results[:5]:
    title = item.get('title', '')
    snippet = item.get('snippet', '')
    if title or snippet:
      snippets.append({"title": title, "snippet": snippet, "link": item.get('link', '')})

  if snippets:
    print(f"  [Search] Kept {len(snippets)}/{len(search_results)} snippets", flush=True)
    results.append({
      "type": "search_snippets",
      "ticker": ticker,
      "queries": query_info,
      "results": snippets
    })

  urls = [item['link'] for item in search_results[:3] if item.get('link')]
  if urls:
    print(f"  [Auto-inject] Scraping {len(urls)} URLs from search results...", flush=True)
    try:
      scrape_result = await mcp.call_tool('get_urls_content', {'urls': urls})
      scrape_items = scrape_result.get('results', [])
      search_intent = ", ".join(query_info.values()) if isinstance(query_info, dict) else str(query_info)
      print(f"  [Auto-inject] Summarizing {len(scrape_items)} scraped pages...", flush=True)
      _summarize_scraped_items(scrape_items, ticker, summarizer, results, search_intent, variables)
    except Exception as e:
      print(f"  [Auto-inject] Failed to scrape URLs: {e}\n", flush=True)


async def _process_scrape(
  tool_result: Dict,
  ticker: str,
  summarizer: Search_Summarizer_Agent,
  results: List[Dict[str, Any]],
  variables: Dict[str, Any] = None
):
  scrape_items = tool_result.get('results', [])
  print(f"  [Scrape] Summarizing {len(scrape_items)} pages...", flush=True)
  _summarize_scraped_items(scrape_items, ticker, summarizer, results, "financial data, metrics, analyst opinions", variables)


def _process_financial(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Pass-through] {tool_name}: financial modeling data", flush=True)
  return {"type": "financial_data", "tool": tool_name, "data": tool_result}


def _process_macro(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  """Process FRED macro data -- unwrap envelope and pass through."""
  print(f"  [Pass-through] {tool_name}: macro economic data", flush=True)
  inner = tool_result.get('data', tool_result) if 'domain' in tool_result else tool_result
  return {"type": "macro", "tool": tool_name, "data": inner}


def _process_market_intel(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Pass-through] {tool_name}: market intelligence data", flush=True)
  inner = tool_result.get('data', tool_result) if 'domain' in tool_result else tool_result
  return {"type": "market_intel", "tool": tool_name, "data": inner}


def _process_news(
  tool_name: str,
  tool_result: Dict,
  ticker: str,
  news_agent: News_Processing_Agent,
  results: List[Dict[str, Any]],
  variables: Dict[str, Any] = None,
  cache: Session_Cache = None,
  arguments: Dict[str, Any] = None
):
  """Route news tool results through per-article sentiment analysis."""
  articles = tool_result.get("data", [])
  if not articles or not isinstance(articles, list):
    print(f"  [Pass-through] {tool_name}: no articles to analyze", flush=True)
    results.append({"type": "market_intel", "tool": tool_name, "data": tool_result})
    return

  if cache is not None:
    cached = cache.get_news(tool_name, articles)
    if cached is not None:
      print(f"  [News Cache HIT] {tool_name} ({len(articles)} articles) -- skipping LLM analysis", flush=True)
      results.append({"type": "news_analysis", "tool": tool_name, **cached})
      if variables is not None:
        _flatten_news_analysis(variables, tool_name, cached)
      return

  print(f"  [News] Analyzing {len(articles)} articles via News Agent...", flush=True)
  analysis = news_agent.analyze_news(articles, ticker, tool_name)
  results.append({
    "type": "news_analysis",
    "tool": tool_name,
    **analysis
  })
  if variables is not None:
    _flatten_news_analysis(variables, tool_name, analysis)
  if cache is not None:
    cache.put_news(tool_name, articles, analysis)


def _process_other(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Unknown] {tool_name}: passing through", flush=True)
  return {"type": "other", "tool": tool_name, "data": tool_result}


def _flatten_search_summary(variables: Dict[str, Any], summary: dict, idx: int):
  """Flatten a search summary into the variable store."""
  if variables is None:
    return

  title = summary.get('source_title', 'unknown')
  variables[f"search.{idx}.title"] = title

  data_points = summary.get('data_points') or []
  for dp in data_points:
    metric = dp.get('metric', '')
    value = dp.get('value', '')
    if metric and value:
      clean_key = metric.lower().replace(' ', '_').replace('/', '_')
      variables[f"search.{idx}.{clean_key}"] = value

  key_facts = summary.get('key_facts') or []
  if key_facts:
    variables[f"search.{idx}.key_facts"] = key_facts

  sentiment = summary.get('sentiment')
  if sentiment:
    variables[f"search.{idx}.sentiment"] = sentiment


def _summarize_scraped_items(
  items: List[Dict],
  ticker: str,
  summarizer: Search_Summarizer_Agent,
  results: List[Dict[str, Any]],
  search_intent: str = "financial data, metrics, analyst opinions",
  variables: Dict[str, Any] = None
):
  existing_search = sum(1 for k in (variables or {}) if k.startswith('search.') and k.endswith('.title'))

  for item in items:
    if not (item.get('success') and item.get('content')):
      print(f"    [X] Failed: {item.get('error', 'unknown error')[:50]}", flush=True)
      continue

    summary = summarizer.summarize_single(item, ticker, search_intent)
    if not summary:
      print(f"    [X] Summarizer returned None for: {item.get('title', 'unknown')[:50]}", flush=True)
      continue
    if summary.get('relevant', False):
      print(f"    [OK] Relevant: {item.get('title', 'unknown')[:50]}", flush=True)
      results.append({"type": "web_content", **summary})
      _flatten_search_summary(variables, summary, existing_search)
      existing_search += 1
    else:
      reason = summary.get('reason', summary.get('error', 'not relevant'))
      print(f"    [X] Filtered: {reason[:50]}", flush=True)
