from .gemini_template import GeminiModel
from typing import Dict, Any, List, Optional
import json
import sys
from datetime import datetime


class Financial_Analysis_Agent(GeminiModel):
    """
    Senior Investment Banker AI that performs institutional-grade financial analysis.

    This agent does NOT call tools or orchestrate execution. It receives data from
    tool execution and generates comprehensive analysis considering multiple factors:
    - Company-specific factors (business model, financials, management)
    - Market factors (sector trends, competitive landscape, valuation)
    - Macro factors (interest rates, economic growth, regulation)
    - Risk factors (business, financial, market risks)
    """

    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        super().__init__(model_name)

    def analyze(self,
                user_query: str,
                execution_plan: Dict[str, Any],
                tools_results: List[Dict[str, Any]],
                analytical_considerations: Optional[List[Dict[str, str]]] = None,
                variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive financial analysis based on gathered data.

        Args:
            user_query: The user's original question
            execution_plan: The plan that was executed (provides context for what data was gathered)
            tools_results: Results from all tool executions
            analytical_considerations: Strategic guidance from probing phase (topics + guidance)
            variables: Shared variable store - flat key-value pairs from all tool results

        Returns:
            Comprehensive analysis as a string
        """
        # Clear conversation history for fresh analysis
        self.conversatoin_history = []

        # Build comprehensive analysis prompt
        system_prompt = self._build_system_prompt()
        analysis_prompt = self._build_analysis_prompt(
            user_query,
            execution_plan,
            tools_results,
            analytical_considerations,
            variables or {}
        )

        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"ANALYSIS PHASE - DeepSeek-R1 Analyzing...", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)

        # Generate analysis
        analysis = self.generate_response(
            prompt=analysis_prompt,
            system_prompt=system_prompt
        )

        return analysis

    def _build_system_prompt(self) -> str:
        """Build specialized system prompt for financial analysis"""
        current_date = datetime.now().strftime("%B %d, %Y")

        prompt = f"""You are a Managing Director at Goldman Sachs with 20+ years of experience in investment banking and equity research.

TODAY'S DATE: {current_date}

YOUR ROLE:
You receive data gathered from various sources (SEC filings, web searches, financial databases) and synthesize it into institutional-grade analysis that answers the client's question.

CRITICAL ANALYSIS FRAMEWORK:

1. UNDERSTAND THE QUESTION
   - What is the client really asking?
   - What decision does this analysis support?
   - What level of precision is required?

2. ASSESS THE DATA
   - What data was gathered and from what sources?
   - What is the quality and recency of the data?
   - What's missing or incomplete?

3. MULTI-FACTOR ANALYSIS
   Consider ALL relevant factors:

   COMPANY-SPECIFIC FACTORS:
   - Business model and competitive position
   - Revenue drivers and growth trajectory
   - Profitability and margin trends
   - Capital allocation and efficiency
   - Management quality and strategy
   - Recent events and material changes

   MARKET FACTORS:
   - Current market conditions and sentiment
   - Sector/industry trends and dynamics
   - Competitive landscape and peer positioning
   - Valuation relative to peers and historical levels

   MACRO FACTORS:
   - Interest rate environment and impact on WACC/multiples
   - Economic growth expectations
   - Inflation and its impact on margins/growth
   - Regulatory environment
   - Geopolitical considerations

   VALUATION FACTORS:
   - Intrinsic value vs market price
   - Multiple methodologies (DCF, comps, precedents)
   - Key assumptions and their sensitivities
   - Risk-adjusted returns

   RISK FACTORS:
   - Business risks (execution, competition, disruption)
   - Financial risks (leverage, liquidity, FCF)
   - Market risks (volatility, correlation)
   - Regulatory/legal risks

4. SYNTHESIS & INSIGHT
   - Connect the dots across all data points
   - Identify what's most material to the question
   - Provide actionable insights, not just data summary
   - Support every conclusion with specific evidence
   - Acknowledge uncertainties and limitations

5. INSTITUTIONAL STANDARDS
   - Professional tone and language
   - Precise with numbers, dates, sources
   - Provide ranges, not point estimates
   - Include caveats and risk disclosures
   - Structure clearly with headers/sections
   - Be balanced and objective (not promotional)

OUTPUT REQUIREMENTS:
- Start with executive summary (2-3 sentences)
- Use clear section headers
- Support claims with specific data points from the gathered information
- Include quantitative analysis where appropriate (calculations, ratios, percentages)
- Provide actionable conclusions
- Acknowledge data gaps and their impact on confidence
- Be concise but thorough - quality over quantity

CRITICAL RULES - VIOLATIONS ARE UNACCEPTABLE:
1. NEVER invent or hallucinate numbers - use ONLY data provided in the prompt
2. If a metric is not in the data, say "NOT PROVIDED IN DATA" or "ASSUMPTION: [your assumption]"
3. You're advising sophisticated institutional investors who will verify your numbers
4. Your analysis must be defensible - every number must trace to a data source
5. Today's date is {current_date} - use this year for all context
6. If critical data is missing, clearly state it's missing and what assumption you're using
7. Cite the source for each key number (e.g., "Revenue of $X per SEC filing")

COMMON MISTAKES TO AVOID:
- Using a different number than what's in the provided data
- Making up metrics that weren't provided (invent nothing!)
- Referencing outdated years when data shows a different fiscal period
- Providing valuations without showing the calculation steps
- Ignoring the actual data and generating generic analysis
"""

        return prompt

    def _format_variable(self, key: str, value: Any) -> str:
        """Format a single variable for display. Handles numbers, strings, dicts, lists."""
        if isinstance(value, dict):
            # Nested result (e.g. DCF yearly projections) - compact JSON
            return f"{key}: {json.dumps(value, separators=(',', ':'), default=str)}"
        if isinstance(value, list):
            return f"{key}: {json.dumps(value, separators=(',', ':'), default=str)}"
        if isinstance(value, float):
            if abs(value) >= 1e9:
                return f"{key}: {value/1e9:.2f}B"
            if abs(value) >= 1e6:
                return f"{key}: {value/1e6:.2f}M"
            return f"{key}: {value:,.4f}" if abs(value) < 1 else f"{key}: {value:,.2f}"
        return f"{key}: {value}"

    def _build_analysis_prompt(self,
                               user_query: str,
                               execution_plan: Dict[str, Any],
                               tools_results: List[Dict[str, Any]],
                               analytical_considerations: Optional[List[Dict[str, str]]] = None,
                               variables: Dict[str, Any] = None) -> str:
        """Build the analysis prompt with all context.

        Uses the shared variable store as the primary data source.
        Variables contain all flat key-value pairs accumulated from tool results.
        Supplementary data shows unstructured content (search results, web scrapes).
        """
        current_date = datetime.now().strftime("%B %d, %Y")
        variables = variables or {}

        # Split variables: flat keys (primary data) vs namespaced keys (dupes, skip)
        flat_vars = {k: v for k, v in variables.items() if '.' not in k}

        # Build the gathered data display from variables
        if flat_vars:
            var_lines = [self._format_variable(k, v) for k, v in flat_vars.items()]
            gathered_display = "\n".join(var_lines)
        else:
            gathered_display = "(No structured data gathered)"

        prompt = f"""
################################################################################
#                           CRITICAL INSTRUCTIONS                               #
################################################################################

TODAY'S DATE: {current_date}
CURRENT YEAR: {datetime.now().year}
TICKER: {execution_plan.get('ticker', flat_vars.get('ticker', 'N/A'))}

>>> THE CURRENT YEAR IS {datetime.now().year}. ALL PROJECTIONS START FROM {datetime.now().year}. <<<
>>> MANDATORY: USE ONLY THE DATA PROVIDED BELOW <<<
>>> DO NOT MAKE UP OR HALLUCINATE ANY NUMBERS <<<
>>> IF DATA IS MISSING, SAY "ASSUMPTION: [value]" <<<

################################################################################
#                         GATHERED DATA                                        #
################################################################################

{gathered_display}

################################################################################
#                           CLIENT REQUEST                                      #
################################################################################

{user_query}

"""

        # Add analytical guidance from probing phase
        if analytical_considerations:
            prompt += f"""
################################################################################
#                    ANALYTICAL GUIDANCE                                        #
################################################################################
"""
            for idx, item in enumerate(analytical_considerations, 1):
                if isinstance(item, dict):
                    prompt += f"{idx}. [{item.get('topic', 'General')}] {item.get('guidance', '')}\n"
                else:
                    prompt += f"{idx}. {item}\n"
            prompt += "\n"

        # Add execution context
        prompt += f"""
################################################################################
#                         ANALYSIS CONTEXT                                      #
################################################################################
Task Type: {execution_plan.get('task_type', 'N/A')}
Current Date: {current_date}
"""

        if 'reasoning' in execution_plan:
            prompt += f"Data Gathering Strategy: {execution_plan['reasoning']}\n"

        # Add tool results that contain nested data (lists, dicts) that
        # _flatten_result skips. These are full outputs like DCF projections,
        # comp tables, disclosure data, etc.
        detailed_parts = []
        for result in tools_results:
            result_type = result.get('type', '')
            tool_name = result.get('tool', '')
            data = result.get('data', {})

            if result_type in ('sec_data', 'financial_data') and isinstance(data, dict):
                # Check if there's nested data that variables missed
                nested = {k: v for k, v in data.items() if isinstance(v, (dict, list)) and v}
                if nested:
                    detailed_parts.append(f"[{tool_name}] {json.dumps(nested, indent=1, default=str)}\n")

        if detailed_parts:
            prompt += f"""
################################################################################
#                    DETAILED TOOL OUTPUT                                       #
################################################################################

{"".join(detailed_parts)}
"""

        # Add unstructured data (search results, web content) as supplementary
        # The search summarizer already filters irrelevant content, so show everything
        supp_parts = []

        for result in tools_results:
            result_type = result.get('type', '')

            if result_type == 'search_snippets':
                lines = [f"  {s.get('title','')}: {s.get('snippet','')}" for s in result.get('results', [])]
                part = f"[SEARCH] {result.get('queries', {})}\n" + "\n".join(lines) + "\n"
                supp_parts.append(part)
            elif result_type == 'web_content':
                dp_lines = [f"  {dp.get('metric','')}: {dp.get('value','')}" for dp in (result.get('data_points') or [])]
                fact_lines = [f"  {f}" for f in (result.get('key_facts') or [])]
                part = f"[WEB: {result.get('source_title', 'Unknown')}]\n" + "\n".join(dp_lines + fact_lines) + "\n"
                supp_parts.append(part)

        if supp_parts:
            prompt += f"""
################################################################################
#                    SUPPLEMENTARY RESEARCH ({len(supp_parts)} sources)          #
################################################################################

{"".join(supp_parts)}
"""

        # Final instructions
        prompt += f"""
################################################################################
#                              YOUR TASK                                        #
################################################################################

Analyze the data above to answer: "{user_query}"

RULES:
1. Use ONLY the data provided above. Do not invent numbers.
2. If data is missing, write "ASSUMPTION: [value]" and justify it.
3. Show calculations step-by-step where applicable.
4. Do not confuse millions, billions, and trillions.

STRUCTURE:
1. Executive Summary (2-3 sentences)
2. Key Data Points (cite exact figures from the data)
3. Analysis (show your work)
4. Risks & Considerations
5. Conclusion

Begin your analysis:
"""

        return prompt
