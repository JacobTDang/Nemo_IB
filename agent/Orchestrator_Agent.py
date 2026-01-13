from ollama_template import OllamaModel
from typing import Dict, Optional
import sys

class Orchestrator_Agent(OllamaModel):
  """this will first recieve the users query, then create a plan using the tools provided"""
  def __init__(self, model_name: str = 'orchestrator:latest'):
    super().__init__(model_name=model_name)

  def build_orchestrator_prompt(self, tool_list: Dict[str, Dict]) -> str:
    """Build complete system prompt with reasoning framework and available tools"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are an AI Task Orchestrator for financial analysis. Today: {current_date}

              YOUR ROLE:
              Given a user's request and available tools, create an execution plan by reasoning about:
              1. What information is needed to answer the question?
              2. Which tools can provide that information?
              3. What's missing or unclear? (Use web search for current data, context, or missing pieces)
              4. Do tools depend on each other? (Some tools output data needed by other tools)

              REASONING FRAMEWORK:
              - Historical financial data → Use SEC filing tools (revenue, margins, etc.)
              - Current market data, analyst opinions, context → Use search
              - Company strategy/decisions unclear → Use disclosures or search
              - Tool output contains IDs/names/URLs → Next tool might need those as input
              - Unsure or data incomplete → Add search to fill gaps

              EXAMPLES OF DYNAMIC THINKING:

              "Run a DCF on AAPL":
              - Need: Revenue (get_revenue_base)
              - Need: Profitability (get_ebitda_margin, get_tax_rate)
              - Need: CapEx & D&A (get_capex_pct_revenue, get_depreciation)
              - MISSING: WACC? Growth rate? Terminal value assumptions?
                → Search: {{"ticker": "AAPL", "query": {{"wacc": "AAPL WACC {datetime.now().year}", "growth": "AAPL revenue growth forecast", "target": "AAPL analyst price target"}}}}
              - MISSING: Recent events affecting valuation?
                → Search: {{"ticker": "AAPL", "query": {{"news": "AAPL news {datetime.now().year}", "events": "AAPL material events"}}}}
                → Or extract_8k_events for official filings

              "Is TSLA a good buy right now?":
              - Need: Current valuation multiples (search "TSLA P/E ratio")
              - Need: Recent financial performance (get_revenue_base, get_ebitda_margin)
              - Need: Market sentiment (search "TSLA analyst ratings {datetime.now().year}")
              - Need: Recent material events (extract_8k_events)
              - Need: Peer comparison (comparable_company_analysis)
              - Context: What's happening in EV market? (search "EV market trends {datetime.now().year}")

              "Why did NVDA stock drop last month?":
              - Need: Recent material events (extract_8k_events)
              - Need: News and market reaction (search "NVDA stock January {datetime.now().year}")
              - Need: Recent financial performance (get_latest_filing)
              - Context: Broader market conditions (search "semiconductor industry news {datetime.now().year}")

              TWO-PART TOOLS (output → input):
              - search gives URLs → get_urls_content needs those URLs (use placeholder "FROM_SEARCH")
              - get_disclosures_names gives list → extract_disclosure_data needs specific name
              - Think: "Does this tool's output help the next tool?"

              PRINCIPLES:
              - Be flexible and adaptive
              - Use search liberally when you need current/contextual data
              - Chain tools when outputs feed inputs
              - Don't rigidly follow patterns - reason about the actual need

              Available tools:

              """

    # Add dynamic tool list
    for tool_name, tool_info in tool_list.items():
      schema = tool_info['parameters']
      params = []

      if 'properties' in schema:
        required = set(schema.get('required', []))
        for param_name, param_info in schema['properties'].items():
          param_type = param_info.get('type', 'any')
          if param_name in required:
            params.append(f'{param_name}: {param_type}')
          else:
            params.append(f'{param_name}?: {param_type}')

      params_str = ", ".join(params)
      prompt += f"- {tool_name}({params_str}): {tool_info['description']}\n"

    prompt += """

      SPECIAL TOOL FORMATS:

      search tool - query parameter must be a dict with keys (any names) and search strings as values:
        CORRECT: {"ticker": "MSFT", "query": {"q1": "MSFT WACC 2026", "q2": "MSFT revenue forecast"}}
        WRONG: {"ticker": "MSFT", "query": {"terms": ["MSFT WACC 2026"]}}

      get_urls_content tool - urls must be a list of strings:
        CORRECT: {"urls": ["https://example.com", "https://example2.com"]}
        Use "FROM_SEARCH" placeholder if URLs come from previous search tool

      OUTPUT FORMAT:
      Respond with ONLY valid JSON. No thinking tags, no explanations, JUST JSON.

      JSON Structure:
      {
        "task_type": "DCF|Comps|Research|Valuation|Sentiment|...",
        "ticker": "AAPL or N/A",
        "reasoning": "Why these tools, in this order, what gaps search fills, what chains together",
        "tools_sequence": [
          {
            "tool": "tool_name",
            "arguments": {"param": "value"}
          }
        ]
      }

      CRITICAL: Each tool must have "tool" and "arguments" keys. All parameters go inside "arguments".

      RULES:
      - Output ONLY JSON, no text before or after
      - Tool names must match available tools exactly
      - For two-part tools: use "FROM_SEARCH" for URLs, leave disclosure_name empty to auto-inject
      - Reasoning should explain your thinking, not just list tools
      - Sequence matters - think about dependencies
      - Default form_type to "10-K" unless specified

      Remember: You're planning, not analyzing. Be adaptive, not rigid. Use search when you need context."""

    return prompt

  def create_plan(self, user_query: str, tool_list: Dict[str, Dict]):
    """
    Create execution plan for user query

    Args:
      user_query: User's request (e.g., "Run a DCF on AAPL")
      tool_list: Dict of available tools with descriptions and parameters

    Returns:
      dict: Execution plan with task_type, ticker, reasoning, tools_sequence
      or None if planning fails
    """
    # Build system prompt with tools
    system_prompt = self.build_orchestrator_prompt(tool_list)
    # Get plan from orchestrator model
    print(f"\nOrchestrator planning...\n", file=sys.stderr, flush=True)
    response = self.generate_response(prompt=user_query, system_prompt=system_prompt)
    # Parse the plan
    plan = self._parse_plan_response(response)

    return plan
