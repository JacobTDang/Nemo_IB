"""
Tool execution engine -- runs MCP tools and processes results by type.
Extracted from analysis_workflow.py to keep the graph definition clean.
"""
from typing import Dict, Any, List
from ..MCP_manager import MCPConnectionManager
from ..Search_Summarizer_Agent import Search_Summarizer_Agent
from ..cache import Session_Cache
import json


# SEC tools return structured XBRL data -- pass through without summarization
SEC_TOOLS = {
  'get_revenue_base', 'get_ebitda_margin', 'get_capex_pct_revenue',
  'get_depreciation', 'get_tax_rate', 'extract_8k_events',
  'get_disclosures_names', 'extract_disclosure_data', 'get_latest_filing'
}

# Financial modeling tools -- deterministic calculations, pass through
FINANCIAL_TOOLS = {
  'get_market_data', 'calculate_dcf', 'calculate_wacc', 'comparable_company_analysis'
}


# Tools safe to cache (deterministic, don't change within a session)
CACHEABLE_TOOLS = SEC_TOOLS | FINANCIAL_TOOLS

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

# Fallback values when no data source exists
DEFAULTS = {
  'revenue_growth': [0.08, 0.07, 0.06, 0.05, 0.04],
  'terminal_growth': 0.025,
  'terminal_multiple': 15.0,
  'risk_free_rate': 0.04,
  'equity_risk_premium': 0.055,
}


def _flatten_result(variables: Dict[str, Any], tool_name: str, result: Dict[str, Any]):
  """Flatten a tool result into the shared variable store.

  Stores each scalar value under two keys:
    - 'tool_name.key' (namespaced, no collisions)
    - 'key' (flat, last-writer-wins for easy resolution)

  Skips metadata keys and nested dicts/lists.
  """
  for key, value in result.items():
    if key in SKIP_KEYS or key.endswith('_concept_used'):
      continue
    if isinstance(value, (dict, list)):
      continue
    if value is None:
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
  # LLM placeholder strings like "FROM_SEARCH", "FROM_PREVIOUS", "FROM_MARKET_DATA"
  if isinstance(v, str) and v.startswith("FROM_"):
    return False
  return True


def _resolve_args(arguments: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
  """Resolve placeholder/zero arguments from the shared variable store.

  For each argument with a falsy value:
    1. Direct key match in variables
    2. Alias match via ARGUMENT_ALIASES
    3. Default from DEFAULTS

  Applies /100 conversion for PERCENT_FIELDS.
  Special-cases cost_of_debt as derived from interestExpense / totalDebt.
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

    # 3. Default
    if resolved_value is None and arg_name in DEFAULTS:
      resolved_value = DEFAULTS[arg_name]

    # 4. Special case: cost_of_debt derived from interest expense / total debt
    if resolved_value is None and arg_name == 'cost_of_debt':
      interest = variables.get('interestExpense')
      debt = variables.get('totalDebt')
      if interest and debt and debt > 0:
        resolved_value = abs(interest) / debt

    if resolved_value is not None:
      # Apply percent conversion if needed
      if resolved_key and resolved_key in PERCENT_FIELDS:
        resolved_value = resolved_value / 100
      args[arg_name] = resolved_value

  return args


async def run_tools(
  tool_sequence: List[Dict[str, Any]],
  ticker: str,
  mcp: MCPConnectionManager,
  summarizer: Search_Summarizer_Agent,
  cache: Session_Cache = None,
  variables: Dict[str, Any] = None
) -> tuple:
  """
  Execute a sequence of MCP tools and process each result into a
  normalized format suitable for the analysis agent.

  Args:
    tool_sequence: List of {"tool": str, "arguments": dict} dicts
    ticker: Ticker symbol for context in summarization
    mcp: MCP connection manager for tool calls
    summarizer: Search summarizer agent for web content extraction
    cache: Optional session cache for skipping duplicate tool calls
    variables: Shared variable store - accumulated key-value pairs from tool results

  Returns:
    Tuple of (results list, updated variables dict)
  """
  results: List[Dict[str, Any]] = []
  variables = dict(variables) if variables else {}

  print(f"\n{'='*60}", flush=True)
  print(f"EXECUTION PHASE - Running {len(tool_sequence)} tools", flush=True)
  print(f"{'='*60}\n", flush=True)

  for idx, tool in enumerate(tool_sequence, 1):
    if not (tool.get('tool') and tool.get('arguments')):
      raise KeyError(f"Unable to find tool and arguments in step {idx}: {tool}")

    tool_name = tool['tool']
    arguments = tool['arguments']

    # Resolve placeholder/zero arguments from the variable store
    original_args = dict(arguments)
    arguments = _resolve_args(arguments, variables)
    resolved = {k for k in arguments if arguments[k] != original_args.get(k)}
    if resolved:
      print(f"[Step {idx}/{len(tool_sequence)}] Executing: {tool_name} (resolved: {resolved})", flush=True)
    else:
      print(f"[Step {idx}/{len(tool_sequence)}] Executing: {tool_name}", flush=True)
    print(f"  Arguments: {json.dumps(arguments, indent=2, default=str)}", flush=True)

    # Check cache for deterministic tools
    cached = None
    if cache and tool_name in CACHEABLE_TOOLS:
      cached = cache.get(tool_name, arguments)

    if cached:
      tool_result = cached
      print(f"  [CACHE HIT] Skipped MCP call", flush=True)
    else:
      tool_result = await mcp.call_tool(tool_name, arguments)
      # Store in cache if cacheable
      if cache and tool_name in CACHEABLE_TOOLS:
        cache.put(tool_name, arguments, tool_result)
      print(f"  Result preview: {str(tool_result)[:500]}...\n", flush=True)

    # Flatten result into shared variable store
    if tool_name in FLATTENABLE_TOOLS:
      _flatten_result(variables, tool_name, tool_result)

    if tool_name in SEC_TOOLS:
      results.append(_process_sec(tool_name, tool_result))
    elif tool_name in FINANCIAL_TOOLS:
      results.append(_process_financial(tool_name, tool_result))
    elif tool_name == 'search':
      await _process_search(tool_result, arguments, ticker, mcp, summarizer, results)
    elif 'get_urls_content' in tool_name:
      await _process_scrape(tool_result, ticker, summarizer, results)
    else:
      results.append(_process_other(tool_name, tool_result))

  print(f"\n{'='*60}", flush=True)
  print(f"EXECUTION COMPLETE - {len(results)} items retained", flush=True)
  print(f"VARIABLES: {len(variables)} keys accumulated", flush=True)
  print(f"{'='*60}\n", flush=True)

  return results, variables



def _process_sec(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Pass-through] {tool_name}: structured SEC data", flush=True)
  return {"type": "sec_data", "tool": tool_name, "data": tool_result}


async def _process_search(
  tool_result: Dict,
  arguments: Dict,
  ticker: str,
  mcp: MCPConnectionManager,
  summarizer: Search_Summarizer_Agent,
  results: List[Dict[str, Any]]
):
  search_results = tool_result.get('search_result', [])
  query_info = arguments.get('query', {})

  # Keep top 5 snippets
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

  # Auto-scrape top 3 URLs
  urls = [item['link'] for item in search_results[:3] if item.get('link')]
  if urls:
    print(f"  [Auto-inject] Scraping {len(urls)} URLs from search results...", flush=True)
    try:
      scrape_result = await mcp.call_tool('get_urls_content', {'urls': urls})
      scrape_items = scrape_result.get('results', [])
      print(f"  [Auto-inject] Summarizing {len(scrape_items)} scraped pages...", flush=True)
      _summarize_scraped_items(scrape_items, ticker, summarizer, results)
    except Exception as e:
      print(f"  [Auto-inject] Failed to scrape URLs: {e}\n", flush=True)


async def _process_scrape(
  tool_result: Dict,
  ticker: str,
  summarizer: Search_Summarizer_Agent,
  results: List[Dict[str, Any]]
):
  scrape_items = tool_result.get('results', [])
  print(f"  [Scrape] Summarizing {len(scrape_items)} pages...", flush=True)
  _summarize_scraped_items(scrape_items, ticker, summarizer, results)


def _process_financial(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Pass-through] {tool_name}: financial modeling data", flush=True)
  return {"type": "financial_data", "tool": tool_name, "data": tool_result}


def _process_other(tool_name: str, tool_result: Dict) -> Dict[str, Any]:
  print(f"  [Unknown] {tool_name}: passing through", flush=True)
  return {"type": "other", "tool": tool_name, "data": tool_result}



def _summarize_scraped_items(
  items: List[Dict],
  ticker: str,
  summarizer: Search_Summarizer_Agent,
  results: List[Dict[str, Any]]
):
  for item in items:
    if not (item.get('success') and item.get('content')):
      print(f"    [X] Failed: {item.get('error', 'unknown error')[:50]}", flush=True)
      continue

    summary = summarizer.summarize_single(item, ticker, "financial data, metrics, analyst opinions")
    if summary.get('relevant', False):
      print(f"    [OK] Relevant: {item.get('title', 'unknown')[:50]}", flush=True)
      results.append({"type": "web_content", **summary})
    else:
      reason = summary.get('reason', summary.get('error', 'not relevant'))
      print(f"    [X] Filtered: {reason[:50]}", flush=True)
