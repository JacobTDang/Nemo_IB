"""
Hub-and-spoke workflow for financial analysis.
Master orchestrator is the hub; probe, plan, execute, plan_verify, model, analyze are spokes.
Every spoke returns to master. Master reads state and routes to the next spoke.
"""
from langgraph.graph import StateGraph, END
from .agent_state import AgentState
from ..Financial_Analysis_Agent import Financial_Analysis_Agent
from ..Financial_Modeling_Agent import Financial_Modeling_Agent
from ..Orchestrator_Agent import Orchestrator_Agent
from ..Probing_Agent import Probing_Agent
from ..Plan_Verifier_Agent import Plan_Verifier_Agent
from ..Search_Summarizer_Agent import Search_Summarizer_Agent
from ..Master_Orchestrator import Master_Orchestrator
from ..News_Processing_Agent import News_Processing_Agent
from ..Verification_Agent import Verification_Agent
from ..MCP_manager import MCPConnectionManager
from .execution_engine import run_tools
from .constants import MAX_ITERATIONS, MAX_EXECUTIONS, MAX_ANALYSES
from tools.financial_modeling_engine.analysis_tools import MODELING_PHASE_TOOLS
from typing import cast
import asyncio
import json, sys


class WorkFlow:
  def __init__(self, mcp: MCPConnectionManager):
    self.master_orchestrator = Master_Orchestrator()
    self.prober = Probing_Agent()
    self.orchestrator = Orchestrator_Agent()
    self.search_summarizer = Search_Summarizer_Agent()
    self.financial_analyst = Financial_Analysis_Agent()
    self.modeling_agent = Financial_Modeling_Agent()
    self.plan_verifier = Plan_Verifier_Agent()
    self.news_agent = News_Processing_Agent()
    self.verification_agent = Verification_Agent()
    self.workflow = StateGraph(AgentState)
    self.mcp = mcp

    self.app = self.setup_graph()

  def master_node(self, state: AgentState):
    print(f"\n[Routing] Phase: {state.get('current_phase', 'init')} -> deciding...", flush=True)
    global_iteration = state.get('global_iteration', 0) + 1
    phase_history = list(state.get('phase_history', []))

    # Hard stop guardrail
    if global_iteration > MAX_ITERATIONS:
      print(f"\nWARNING: Hit MAX_ITERATIONS ({MAX_ITERATIONS}). Returning current results.", file=sys.stderr, flush=True)
      return {
        'global_iteration': global_iteration,
        'phase_history': phase_history,
        'next_action': 'done',
        'master_reasoning': f'Hard stop: exceeded {MAX_ITERATIONS} iterations.'
      }

    # Execution count guardrail
    execution_count = state.get('execution_count', 0)
    if execution_count >= MAX_EXECUTIONS and state.get('current_phase') == 'planned':
      print(f"\nWARNING: Hit MAX_EXECUTIONS ({MAX_EXECUTIONS}). Proceeding to model.", file=sys.stderr, flush=True)
      phase_history.append(f"master->model (exec limit)")
      return {
        'global_iteration': global_iteration,
        'phase_history': phase_history,
        'next_action': 'model',
        'master_reasoning': f'Execution limit reached ({MAX_EXECUTIONS}). Proceeding to modeling with available data.',
        'query_complexity': state.get('query_complexity', 'standard')
      }

    # Analysis count guardrail
    analysis_count = state.get('analysis_count', 0)
    if analysis_count >= MAX_ANALYSES and state.get('current_phase') == 'analyzed':
      print(f"\nWARNING: Hit MAX_ANALYSES ({MAX_ANALYSES}). Returning best analysis.", file=sys.stderr, flush=True)
      phase_history.append(f"master->done (analysis limit)")
      return {
        'global_iteration': global_iteration,
        'phase_history': phase_history,
        'next_action': 'done',
        'master_reasoning': f'Analysis limit reached ({MAX_ANALYSES}). Returning best analysis so far.',
        'query_complexity': state.get('query_complexity', 'standard')
      }

    # Call the master orchestrator LLM to decide
    decision = self.master_orchestrator.decide(state)

    next_action = decision['next_action']
    query_complexity = decision.get('query_complexity', state.get('query_complexity', 'standard'))
    revision_context = decision.get('revision_context')

    master_notes = list(state.get('master_notes', []))

    # Guardrail: must execute after planning
    if state.get('current_phase') == 'planned' and next_action != 'execute':
      print(f"Guardrail: Phase is 'planned' but LLM chose '{next_action}'. Overriding to execute.", file=sys.stderr, flush=True)
      next_action = 'execute'

    # Guardrail: after execution, check data completeness before allowing analysis.
    if state.get('current_phase') == 'executed':
      stats = state.get('execution_stats', {})
      errors = stats.get('errors', 0)
      variables = state.get('variables', {})

      # Check 1: mass tool failure (>50%) on first execution -> re-plan to fix
      total = stats.get('total', 0)
      error_rate = errors / total if total > 0 else 0
      if error_rate > 0.5 and execution_count == 1:
        print(f"Guardrail: {errors}/{total} tools failed ({error_rate:.0%}). Re-planning.", file=sys.stderr, flush=True)
        next_action = 'plan'
        revision_context = {
          'type': 'missing_data',
          'feedback': f'{errors} of {total} tools failed during execution. Review the results, identify which tools errored (likely wrong ticker or bad arguments), and create a corrected plan.'
        }

      # Check 2: plan_verifier flagged critical gaps -> re-plan with specific tool guidance
      elif execution_count < MAX_EXECUTIONS:
        plan_verification = state.get('plan_verification', {})
        if not plan_verification.get('complete', True):
          gaps = plan_verification.get('gaps', [])
          critical = [g for g in gaps if g.get('priority') == 'critical']
          if critical:
            print(f"Guardrail: plan_verify found {len(critical)} critical gaps. Re-planning.", file=sys.stderr, flush=True)
            next_action = 'plan'
            gap_descriptions = [f"{g['description']} -> {g['recommended_tool']}" for g in critical]
            revision_context = {
              'type': 'missing_data',
              'feedback': (
                f'Pre-analysis verification found {len(critical)} critical data gaps: '
                f'{gap_descriptions}. '
                'Plan ONLY the specific tools listed to fill these gaps. '
                'Do NOT re-fetch data already in the variable store.'
              ),
              'suggested_tools': [
                {'tool': g['recommended_tool'], 'arguments': g['suggested_arguments']}
                for g in critical
              ]
            }

      # Check 3: no issues or budget exhausted -> force model
      if next_action not in ('plan', 'model'):
        print(f"Guardrail: Phase is 'executed' but LLM chose '{next_action}'. Overriding to model.", file=sys.stderr, flush=True)
        next_action = 'model'

    # Guardrail: after modeling, always analyze next
    if state.get('current_phase') == 'modeled' and next_action != 'analyze':
      print(f"Guardrail: Phase is 'modeled' but LLM chose '{next_action}'. Overriding to analyze.", file=sys.stderr, flush=True)
      next_action = 'analyze'

    # Guardrail: if execution budget exhausted and analysis done -> force done
    if state.get('current_phase') == 'analyzed' and execution_count >= MAX_EXECUTIONS:
      if next_action not in ('done',):
        print(f"Guardrail: Exec limit hit and analysis done. LLM chose '{next_action}'. Overriding to done.", file=sys.stderr, flush=True)
        next_action = 'done'

    phase_history.append(f"master->{next_action}")

    return {
      'global_iteration': global_iteration,
      'phase_history': phase_history,
      'next_action': next_action,
      'master_reasoning': decision.get('reasoning', ''),
      'query_complexity': query_complexity,
      'revision_context': revision_context,
      'master_notes': master_notes
    }

  async def _build_modeling_context(self) -> str:
    """Build a concise description of modeling-phase tools from MCP schemas."""
    try:
      tool_list = await self.mcp.list_tools()
      lines = []
      for name in sorted(MODELING_PHASE_TOOLS):
        schema = tool_list.get(name)
        if schema:
          desc = schema.get('description', '(no description)')
          sentences = [s.strip() for s in desc.split('.') if s.strip()]
          short_desc = '. '.join(sentences[:2]) + '.'
          lines.append(f"- {name}: {short_desc}")
      return "\n".join(lines) if lines else "(no modeling tools registered)"
    except Exception as e:
      print(f"[Workflow] Could not build modeling context: {e}", file=sys.stderr, flush=True)
      return "(modeling tools context unavailable)"

  async def probe_node(self, state: AgentState):
    print("\n[1/6] Probing: analyzing query and identifying data requirements...", flush=True)
    user_query = state['user_query']
    ticker = state.get("ticker")

    modeling_tools_context = await self._build_modeling_context()

    result = self.prober.probe(
      user_query=user_query,
      ticker=ticker,
      modeling_tools_context=modeling_tools_context
    )

    if not result:
      raise RuntimeError(f"Probing result is None: {result}")

    detected_ticker = result.get('ticker')

    return {
      "ticker": state.get('ticker') or detected_ticker,
      "data_requirements": result.get('data_requirements', []),
      "analytical_considerations": result.get('analytical_considerations', []),
      "recommended_approach": result.get('recommended_approach', ''),
      "current_phase": "probed"
    }

  async def plan_node(self, state: AgentState):
    print("\n[2/6] Planning: selecting tools and building execution plan...", flush=True)
    user_query = state['user_query']
    # Filter out modeling-phase tools so the orchestrator never plans them.
    tool_list = {k: v for k, v in (await self.mcp.list_tools()).items()
                 if k not in MODELING_PHASE_TOOLS}
    variables = state.get('variables', {})
    execution_count = state.get('execution_count', 0)

    gathered = {k: v for k, v in variables.items() if '.' not in k} if variables else {}

    revision_context = state.get('revision_context')

    if revision_context and revision_context.get('type') == 'missing_data':
      data_gaps = variables.get('analysis.data_gaps', [])
      feedback = {
        "previous_plan": state.get('execution_plan'),
        "issues": [revision_context.get('feedback', '')],
        "missing_data": data_gaps,
        "recommendations": [f"Fetch: {gap}" for gap in data_gaps] if data_gaps else [],
        "reasoning": revision_context.get('feedback', ''),
        "already_gathered": gathered
      }
      plan = self.orchestrator.create_plan(
        user_query=user_query,
        tool_list=tool_list,
        revision_feedback=feedback,
        gathered_variables=gathered,
        execution_count=execution_count
      )
    else:
      data_requirements = state.get('data_requirements', [])
      plan = self.orchestrator.create_plan(
        user_query=user_query,
        tool_list=tool_list,
        data_requirements=data_requirements,
        execution_count=execution_count
      )

    if not plan:
      print(f"  [Plan] First attempt failed, retrying...", file=sys.stderr, flush=True)
      data_requirements = state.get('data_requirements', [])
      plan = self.orchestrator.create_plan(
        user_query=user_query,
        tool_list=tool_list,
        data_requirements=data_requirements,
        gathered_variables=gathered if revision_context else None,
        execution_count=execution_count
      )
    if not plan:
      raise RuntimeError(f"Plan result is None after 2 attempts. LLM may be overloaded or prompt too large.")

    # Post-planning validation: remove hallucinated tools and rename known variants
    _TOOL_RENAME = {
      'get_depreciation_pct_revenue': 'get_depreciation',
      'get_depreciation_percentage': 'get_depreciation',
      'get_d_and_a': 'get_depreciation',
      'get_d_and_a_pct': 'get_depreciation',
    }
    _MARKET_DATA_SUBTOOL_HALLUCINATIONS = {
      'get_beta', 'get_market_cap', 'get_total_debt', 'get_cash', 'get_shares_outstanding',
      'get_interest_expense', 'get_stock_price', 'get_equity_value',
    }
    known_tools = set(tool_list.keys())
    sanitized_steps = []
    removed_tools = []
    for step in plan.get('tools_sequence', []):
      tool_name = step['tool']
      if tool_name in _TOOL_RENAME:
        new_name = _TOOL_RENAME[tool_name]
        print(f"  [Plan Sanitize] Renamed '{tool_name}' -> '{new_name}'", file=sys.stderr, flush=True)
        step = {**step, 'tool': new_name}
        tool_name = new_name
      if tool_name in _MARKET_DATA_SUBTOOL_HALLUCINATIONS:
        removed_tools.append(f"{tool_name} (covered by get_market_data)")
        continue
      if tool_name not in known_tools:
        removed_tools.append(tool_name)
        continue
      sanitized_steps.append(step)
    if removed_tools:
      print(f"  [Plan Sanitize] Removed hallucinated tools: {removed_tools}", file=sys.stderr, flush=True)
    plan['tools_sequence'] = sanitized_steps

    # Post-planning: inject tools the LLM missed
    already_run = {r.get('tool') for r in (state.get('summarized_output') or []) if r.get('tool')}
    plan = self._inject_missing_intent_tools(
      user_query, plan, state.get('ticker', 'UNKNOWN'), already_run, variables
    )

    return {
      'execution_plan': plan,
      'plan_reasoning': plan['reasoning'],
      'current_phase': 'planned',
      'revision_context': None,
      'plan_verification': None,
      'analysis_report': None
    }

  @staticmethod
  def _inject_missing_intent_tools(
    user_query: str,
    plan: dict,
    ticker: str,
    already_run: set = None,
    variables: dict = None,
  ) -> dict:
    """Post-planning validation: inject tools the LLM missed and fix bad arguments."""
    already_run = already_run or set()
    from datetime import datetime, timedelta
    query_lower = user_query.lower()
    planned_tools = {t['tool'] for t in plan.get('tools_sequence', [])}

    today = datetime.now().strftime("%Y-%m-%d")
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Fix 1: Correct bad dates on news tools
    for step in plan.get('tools_sequence', []):
      if step['tool'] in ('get_company_news', 'get_market_news'):
        args = step['arguments']
        needs_fix = False
        try:
          from_dt = datetime.strptime(args.get('from_date', ''), '%Y-%m-%d')
          if (datetime.now() - from_dt).days > 90:
            needs_fix = True
        except (ValueError, TypeError):
          needs_fix = True
        if needs_fix:
          args['from_date'] = month_ago
          args['to_date'] = today
          print(f"  [Plan Fix] Corrected {step['tool']} dates to {month_ago} -> {today}", file=sys.stderr, flush=True)

    # Fix 2: Keyword -> tool injection for explicit user intent
    intent_map = [
      {
        'keywords': ['news', 'sentiment', 'headlines'],
        'tool': 'get_company_news',
        'check': lambda tools: 'get_company_news' not in tools and 'get_market_news' not in tools,
        'args': {'ticker': ticker, 'from_date': month_ago, 'to_date': today}
      },
      {
        'keywords': ['insider', 'insider trading', 'insider buying'],
        'tool': 'get_insider_transactions',
        'check': lambda tools: 'get_insider_transactions' not in tools,
        'args': {'ticker': ticker}
      },
      {
        'keywords': ['analyst', 'rating', 'recommendation', 'consensus'],
        'tool': 'get_analyst_recommendations',
        'check': lambda tools: 'get_analyst_recommendations' not in tools,
        'args': {'ticker': ticker}
      },
      {
        'keywords': ['macro', 'interest rate', 'inflation', 'gdp', 'economy'],
        'tool': 'get_macro_snapshot',
        'check': lambda tools: 'get_macro_snapshot' not in tools and 'get_treasury_yields' not in tools,
        'args': {}
      },
    ]

    injected = []
    for mapping in intent_map:
      if any(kw in query_lower for kw in mapping['keywords']):
        if mapping['check'](planned_tools) and mapping['tool'] not in already_run:
          plan['tools_sequence'].append({
            'tool': mapping['tool'],
            'arguments': mapping['args']
          })
          planned_tools.add(mapping['tool'])
          injected.append(mapping['tool'])

    # Fix 3: Implicit intent for advanced analysis
    advanced_keywords = ['advanced', 'advance', 'comprehensive', 'deep dive', 'thorough',
                         'good time to buy', 'should i buy', 'buy or sell', 'investment']
    if any(kw in query_lower for kw in advanced_keywords):
      implicit_tools = [
        ('get_analyst_recommendations', {'ticker': ticker}),
        ('get_insider_transactions', {'ticker': ticker}),
        ('get_basic_financials', {'ticker': ticker}),
        ('get_company_news', {'ticker': ticker, 'from_date': month_ago, 'to_date': today}),
      ]
      for tool_name, args in implicit_tools:
        if tool_name not in planned_tools and tool_name not in already_run:
          plan['tools_sequence'].append({'tool': tool_name, 'arguments': args})
          planned_tools.add(tool_name)
          injected.append(tool_name)

    # Fix 4: DCF prerequisites
    if 'calculate_dcf' in planned_tools or 'calculate_dcf' in already_run:
      dcf_prereqs = [
        ('get_basic_financials', {'ticker': ticker}),
        ('get_depreciation',     {'ticker': ticker, 'form_type': '10-K'}),
      ]
      for tool_name, args in dcf_prereqs:
        if tool_name not in planned_tools and tool_name not in already_run:
          plan['tools_sequence'].insert(0, {'tool': tool_name, 'arguments': args})
          planned_tools.add(tool_name)
          injected.append(tool_name)

    # Fix 5: Search fallback for missing DCF inputs
    variables = variables or {}
    dcf_relevant = 'calculate_dcf' in planned_tools or 'calculate_dcf' in already_run
    basic_financials_ran = 'get_basic_financials' in already_run

    if dcf_relevant and basic_financials_ran and 'search' not in already_run:
      search_queries = []
      if variables.get('financials.revenueGrowthTTMYoy') is None:
        search_queries.append(
          f"{ticker} analyst revenue growth estimates 2025 2026 2027 site:finance.yahoo.com OR site:seekingalpha.com"
        )
      if variables.get('financials.evEbitdaTTM') is None:
        search_queries.append(
          f"{ticker} EV EBITDA multiple current valuation site:finance.yahoo.com OR site:macrotrends.net"
        )
      for query in search_queries:
        if 'search' not in planned_tools:
          plan['tools_sequence'].insert(0, {'tool': 'search', 'arguments': {'query': query}})
          planned_tools.add('search')
          injected.append(f"search (fallback for missing DCF inputs)")
          break

    # Fix 6: Inject get_financial_statements(cf) when capital returns data is needed
    cf_keywords = ['capital return', 'dividend', 'buyback', 'shareholder yield',
                   'comprehensive', 'advanced', 'deep dive', 'investment',
                   'good time to buy', 'should i buy']
    cf_tools_planned = (
      'get_financial_statements' in planned_tools or
      'get_financial_statements' in already_run
    )
    cf_data_present = (
      variables.get('cf.dividendsPaid') is not None or
      variables.get('dividendsPaid') is not None or
      variables.get('cf.repurchaseOfCapitalStock') is not None
    )
    if any(kw in query_lower for kw in cf_keywords) and not cf_tools_planned and not cf_data_present:
      plan['tools_sequence'].append({
        'tool': 'get_financial_statements',
        'arguments': {'ticker': ticker, 'statement': 'cf', 'freq': 'annual'}
      })
      planned_tools.add('get_financial_statements')
      injected.append('get_financial_statements (cf, for capital returns)')

    if injected:
      print(f"  [Plan Inject] Added missing tools: {injected}", file=sys.stderr, flush=True)

    return plan

  @staticmethod
  def _extract_analysis_metadata(report: str) -> dict:
    """Parse the analysis report and extract structured metadata."""
    import re
    meta = {}

    if not report:
      meta['empty'] = True
      return meta

    report_upper = report.upper()

    gaps = []
    gap_match = re.search(
      r'(?:DATA\s*GAPS?|MISSING\s*DATA)[:\s]*\n(.*?)(?=\n\s*(?:##|\*\*[A-Z]|---|\Z))',
      report, re.IGNORECASE | re.DOTALL
    )
    if gap_match:
      for line in gap_match.group(1).split('\n'):
        stripped = line.strip()
        if stripped and re.match(r'^[-*\d]', stripped):
          cleaned = stripped.lstrip('-*0123456789. ')
          if cleaned and len(cleaned) > 5:
            gaps.append(cleaned)

    not_provided = re.findall(r'([\w\s]+?):\s*NOT PROVIDED', report, re.IGNORECASE)
    for item in not_provided:
      cleaned = item.strip()
      if cleaned and cleaned not in gaps:
        gaps.append(cleaned)

    if gaps:
      meta['data_gaps'] = gaps
      meta['has_gaps'] = True
    else:
      meta['has_gaps'] = False

    conclusion_match = re.search(
      r'(?:CONCLUSION|RECOMMENDATION|VERDICT)[:\s]*\n?(.*?)$',
      report, re.IGNORECASE | re.DOTALL
    )
    if conclusion_match:
      conclusion = conclusion_match.group(1).strip()
      meta['conclusion'] = conclusion[:300] if len(conclusion) > 300 else conclusion

    if any(kw in report_upper for kw in ['STRONG BUY', 'RECOMMEND BUY', 'BULLISH', 'BUY RECOMMENDATION']):
      meta['signal'] = 'bullish'
    elif any(kw in report_upper for kw in ['STRONG SELL', 'BEARISH', 'OVERVALUED', 'SELL RECOMMENDATION']):
      meta['signal'] = 'bearish'
    elif any(kw in report_upper for kw in ['HOLD', 'NEUTRAL', 'MIXED']):
      meta['signal'] = 'neutral'

    return meta

  async def execution_node(self, state: AgentState):
    tool_count = len((state.get('execution_plan') or {}).get('tools_sequence', []))
    print(f"\n[3/6] Executing: running {tool_count} tools to gather financial data...", flush=True)
    execution_plan = state['execution_plan']
    tool_sequence = list(execution_plan['tools_sequence'])
    ticker = state.get('ticker', 'UNKNOWN')

    additional_tools = state.get('additional_tools', [])
    if additional_tools:
      print(f"  [Execute] Merging {len(additional_tools)} additional tools from revision", file=sys.stderr, flush=True)
      tool_sequence.extend(additional_tools)

    existing = list(state.get('summarized_output', []) or [])
    existing_vars = dict(state.get('variables', {}) or {})
    new_results, updated_vars, exec_stats = await run_tools(
      tool_sequence, ticker, self.mcp, self.search_summarizer, self.mcp.cache,
      variables=existing_vars,
      news_agent=self.news_agent
    )
    results = existing + new_results

    execution_count = state.get('execution_count', 0) + 1

    return {
      'summarized_output': results,
      'variables': updated_vars,
      'current_phase': 'executed',
      'execution_count': execution_count,
      'execution_stats': exec_stats
    }

  async def plan_verify_node(self, state: AgentState) -> dict:
    print("\n[4/6] Verifying: checking data completeness...", flush=True)
    """LLM-based pre-analysis data completeness check.

    Runs automatically after every execution_node, before master routes to
    analyze. Fails open on LLM parse failure to avoid blocking analysis indefinitely.
    """
    result = await asyncio.to_thread(
      self.plan_verifier.verify,
      state['user_query'],
      state.get('variables', {}),
      state.get('summarized_output', []),
      state.get('execution_plan', {}),
      state.get('ticker', ''),
      state.get('execution_count', 0)
    )
    pv = result.model_dump() if result else {'complete': True, 'summary': 'parse failure', 'gaps': []}

    # If all critical-gap tools already ran (even with errors), force complete to avoid infinite loops
    if not pv.get('complete', True):
      critical_gaps = [g for g in pv.get('gaps', []) if g.get('priority') == 'critical']
      if critical_gaps:
        ran_tools = {item.get('tool', '') for item in state.get('summarized_output', [])}
        critical_tools = {g['recommended_tool'] for g in critical_gaps}
        if critical_tools.issubset(ran_tools):
          print(
            f"[Plan Verify] All critical gap tools already attempted {critical_tools}. "
            "Forcing complete=True -- analysis will make explicit assumptions for missing data.",
            file=sys.stderr, flush=True
          )
          pv['complete'] = True
          pv['summary'] = (
            pv.get('summary', '') +
            " (All critical tools attempted; proceeding -- analysis will note assumptions.)"
          )

    print(f"\n[Plan Verify] Complete: {pv['complete']}", file=sys.stderr, flush=True)
    if not pv['complete']:
      for gap in pv.get('gaps', []):
        print(f"  [{gap['priority']}] {gap['description']} -> {gap['recommended_tool']}", file=sys.stderr, flush=True)

    return {'plan_verification': pv}

  async def model_node(self, state: AgentState) -> dict:
    print("\n[5/6] Modeling: running financial calculations (scenario DCF, credit profile, etc.)...", flush=True)
    """Financial modeling spoke.

    Runs after plan_verify confirms data is complete. Uses Financial_Modeling_Agent
    to decide which models are appropriate and execute them via pure-Python math.
    """
    user_query = state['user_query']
    variables = state.get('variables', {})

    modeling_tools_context = await self._build_modeling_context()

    print(f"\n[Model Node] Financial Modeling Agent starting...", file=sys.stderr, flush=True)

    outputs = await asyncio.to_thread(
      self.modeling_agent.model,
      user_query,
      variables,
      modeling_tools_context
    )

    models_run = outputs.get('models_run', [])
    print(f"[Model Node] Complete. Models run: {models_run}", file=sys.stderr, flush=True)

    return {
      'model_outputs': outputs,
      'current_phase': 'modeled'
    }

  async def analyze_node(self, state: AgentState):
    print("\n[6/6] Analyzing: generating investment analysis...", flush=True)
    user_query = state['user_query']
    summarized_output = state['summarized_output']
    execution_plan = state['execution_plan']
    analytical_considerations = state.get('analytical_considerations', [])

    variables = state.get('variables', {})
    flat_vars = {k: v for k, v in variables.items() if '.' not in k}
    print(f"\n[Analyze] Variables accumulated ({len(flat_vars)} flat keys): {sorted(flat_vars)}", file=sys.stderr, flush=True)

    # Safety valve: if almost no data was gathered, re-route to plan
    if len(flat_vars) < 5:
      print(f"[Analyze] Aborting: only {len(flat_vars)} variables gathered. Re-routing to plan.", file=sys.stderr, flush=True)
      return {
        'current_phase': 'planned',
        'execution_plan': None,
        'revision_context': {
          'type': 'missing_data',
          'feedback': (
            f'Analysis aborted: only {len(flat_vars)} variables in store. '
            'Fetch foundational data (revenue, margins, market data) before analysis.'
          )
        }
      }

    revision_context = state.get('revision_context')
    previous_analysis = None
    revision_feedback = None
    if revision_context and revision_context.get('type') == 'poor_analysis':
      previous_analysis = variables.get('analysis.conclusion')
      revision_feedback = revision_context.get('feedback', '')

    plan_verification = state.get('plan_verification', {})
    data_gaps = plan_verification.get('gaps', []) if plan_verification else []

    result = self.financial_analyst.analyze(
      user_query=user_query,
      execution_plan=execution_plan,
      tools_results=summarized_output,
      analytical_considerations=analytical_considerations,
      variables=variables,
      previous_analysis=previous_analysis,
      revision_feedback=revision_feedback,
      data_gaps=data_gaps if data_gaps else None,
      model_outputs=state.get('model_outputs')
    )

    # Non-blocking QC: append note to report if verifier flags a critical issue
    try:
      verification = await asyncio.to_thread(
        self.verification_agent.verify,
        user_query,
        result,
        execution_plan
      )
      action = verification.get('action', 'approve') if isinstance(verification, dict) else 'approve'
      feedback = verification.get('feedback', '') if isinstance(verification, dict) else ''
      if action == 'revise' and feedback:
        result = result + f"\n\n---\nQC NOTE: {feedback}"
        print(f"[QC] Appended verifier note: {feedback[:100]}", file=sys.stderr, flush=True)
      else:
        score = verification.get('quality_score', 0) if isinstance(verification, dict) else 0
        print(f"[QC] Verification: {action} (score={score})", file=sys.stderr, flush=True)
    except Exception as e:
      print(f"[QC] Verification failed (non-critical): {e}", file=sys.stderr, flush=True)

    analysis_count = state.get('analysis_count', 0) + 1

    updated_vars = dict(variables)
    stale_keys = [k for k in updated_vars if k.startswith('analysis.')]
    for k in stale_keys:
      del updated_vars[k]
    analysis_meta = self._extract_analysis_metadata(result)
    for k, v in analysis_meta.items():
      updated_vars[f"analysis.{k}"] = v

    if analysis_meta.get('data_gaps'):
      print(f"  [Analysis] Data gaps detected: {analysis_meta['data_gaps']}", file=sys.stderr, flush=True)

    return {
      'analysis_report': result,
      'current_phase': 'analyzed',
      'analysis_count': analysis_count,
      'variables': updated_vars,
      'revision_context': None
    }

  async def run(self, user_query: str, ticker: str = '') -> str:
    """
    Run the workflow for a single user query.

    Args:
        user_query: The user's question or analysis request.
        ticker:     Optional ticker symbol. Probing agent can detect it from the query.

    Returns:
        The final analysis report as a string.
    """
    initial_state = cast(AgentState, {
      'user_query': user_query,
      'ticker': ticker,
    })
    result = await self.app.ainvoke(initial_state)
    return result.get('analysis_report', '')

  def setup_graph(self):
    # Register nodes
    self.workflow.add_node('master', self.master_node)
    self.workflow.add_node('probe', self.probe_node)
    self.workflow.add_node('plan', self.plan_node)
    self.workflow.add_node('execute', self.execution_node)
    self.workflow.add_node('plan_verify', self.plan_verify_node)
    self.workflow.add_node('model', self.model_node)
    self.workflow.add_node('analyze', self.analyze_node)

    # Entry point is always master
    self.workflow.set_entry_point('master')

    # Master routes via conditional edge
    def route_from_master(state: AgentState):
      return state.get('next_action', 'done')

    self.workflow.add_conditional_edges('master', route_from_master, {
      'probe': 'probe',
      'plan': 'plan',
      'execute': 'execute',
      'model': 'model',
      'analyze': 'analyze',
      'done': END
    })

    # execute always feeds plan_verify before returning to master
    self.workflow.add_edge('execute', 'plan_verify')
    self.workflow.add_edge('plan_verify', 'master')

    # All other spokes return directly to master
    for node in ['probe', 'plan', 'model', 'analyze']:
      self.workflow.add_edge(node, 'master')
    return self.workflow.compile()


if __name__ == "__main__":
  async def main():
    user_query = input("You: ")
    async with MCPConnectionManager() as mcp:
      w = WorkFlow(mcp=mcp)

      report = await w.run(
        user_query=user_query,
        ticker=''
      )

      print("\n" + "="*80)
      print("FINAL ANALYSIS")
      print("="*80)
      print(report or "No analysis generated")

  asyncio.run(main())
