from .ollama_template import OllamaModel
from typing import Dict, Any, List, Optional
import json
import sys
from datetime import datetime

class Financial_Analysis_Agent(OllamaModel):
    """
    Senior Investment Banker AI that performs institutional-grade financial analysis.

    This agent does NOT call tools or orchestrate execution. It receives data from
    tool execution and generates comprehensive analysis considering multiple factors:
    - Company-specific factors (business model, financials, management)
    - Market factors (sector trends, competitive landscape, valuation)
    - Macro factors (interest rates, economic growth, regulation)
    - Risk factors (business, financial, market risks)
    """

    def __init__(self, model_name: str = 'DeepSeek-R1-Distill-Llama-8B:latest'):
        super().__init__(model_name)

    def analyze(self,
                user_query: str,
                execution_plan: Dict[str, Any],
                tools_results: List[Dict[str, Any]],
                research_questions: Optional[List[str]] = None) -> str:
        """
        Generate comprehensive financial analysis based on gathered data.

        Args:
            user_query: The user's original question
            execution_plan: The plan that was executed (provides context for what data was gathered)
            tools_results: Results from all tool executions
            research_questions: Optional strategic questions from probing phase

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
            research_questions
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

ANALYSIS TYPES (adapt based on question):

DCF VALUATION:
- Calculate Free Cash Flow (Revenue - OpEx - CapEx - Taxes + D&A - Change in NWC)
- Apply WACC to discount FCFs
- Calculate terminal value (perpetuity growth + exit multiple methods)
- Derive equity value and price per share
- Provide valuation range with sensitivities
- Compare to current price and analyst consensus

COMPARABLE COMPANY ANALYSIS:
- Identify relevant peer group and rationale
- Calculate key multiples (EV/Revenue, EV/EBITDA, P/E, P/B)
- Analyze premium/discount vs peers
- Consider quality differences (growth, margins, ROIC)
- Derive implied valuation range

INVESTMENT DECISION:
- Synthesize valuation, momentum, sentiment, risks
- Provide clear recommendation (Buy/Hold/Sell or equivalent)
- State price target and timeframe
- Identify key catalysts and risks
- Define what would change the thesis

SECTOR/COMPANY RESEARCH:
- Provide comprehensive overview
- Highlight key trends and dynamics
- Identify winners and losers
- Assess opportunities and threats

OUTPUT REQUIREMENTS:
- Start with executive summary (2-3 sentences)
- Use clear section headers
- Support claims with specific data points from the gathered information
- Include quantitative analysis where appropriate (calculations, ratios, percentages)
- Provide actionable conclusions
- Acknowledge data gaps and their impact on confidence
- Be concise but thorough - quality over quantity

REMEMBER:
- You're advising sophisticated institutional investors
- Your analysis must be defensible and data-driven
- Consider the current date ({current_date}) for all market context
- Never make up data - only use what was provided
- If critical data is missing, state what else would be needed
"""

        return prompt

    def _build_analysis_prompt(self,
                               user_query: str,
                               execution_plan: Dict[str, Any],
                               tools_results: List[Dict[str, Any]],
                               research_questions: Optional[List[str]] = None) -> str:
        """Build the analysis prompt with all context"""
        current_date = datetime.now().strftime("%B %d, %Y")

        prompt = f"""CURRENT DATE: {current_date}

{'='*80}
CLIENT REQUEST
{'='*80}
{user_query}

"""

        # Add strategic context from probing phase
        if research_questions:
            prompt += f"""
{'='*80}
STRATEGIC QUESTIONS TO ADDRESS
{'='*80}
"""
            for idx, question in enumerate(research_questions[:5], 1):  # Top 5
                if isinstance(question, dict):
                    prompt += f"{idx}. {question.get('question', question)}\n"
                else:
                    prompt += f"{idx}. {question}\n"
            prompt += "\n"

        # Add execution context
        prompt += f"""
{'='*80}
ANALYSIS CONTEXT
{'='*80}
Task Type: {execution_plan.get('task_type', 'N/A')}
Ticker: {execution_plan.get('ticker', 'N/A')}
"""

        if 'reasoning' in execution_plan:
            prompt += f"Data Gathering Strategy: {execution_plan['reasoning']}\n"

        prompt += f"\n"

        # Add all tool results
        prompt += f"""
{'='*80}
DATA GATHERED ({len(tools_results)} data sources)
{'='*80}

"""

        for idx, result in enumerate(tools_results, 1):
            if result.get('success', True):
                tool_name = result.get('tool', f'Tool {idx}')
                tool_data = result.get('result', {})

                prompt += f"""
--- DATA SOURCE {idx}: {tool_name} ---
{json.dumps(tool_data, indent=2)}

"""
            else:
                tool_name = result.get('tool', f'Tool {idx}')
                error = result.get('error', 'Unknown error')
                prompt += f"""
--- DATA SOURCE {idx}: {tool_name} (FAILED) ---
Error: {error}
Note: Proceed with analysis using available data, acknowledge this limitation

"""

        # Final instructions
        prompt += f"""
{'='*80}
YOUR TASK
{'='*80}

Analyze the data gathered above to answer the client's question: "{user_query}"

Apply your full analytical framework:
1. Assess what data was gathered and its relevance
2. Consider company-specific, market, macro, and valuation factors
3. Perform quantitative analysis where appropriate (calculate metrics, ratios, valuations)
4. Synthesize insights across all data sources
5. Provide clear, actionable conclusions
6. Acknowledge any limitations due to data availability

Structure your analysis professionally:
- Executive Summary (key findings in 2-3 sentences)
- Main Analysis (with clear sections and headers)
- Quantitative Analysis (calculations, if applicable)
- Risks & Considerations
- Conclusion & Recommendations

Remember:
- Today is {current_date} - use this for all market context
- Support every claim with specific data from the sources above
- Be precise with numbers and sources
- Provide ranges where appropriate
- Be balanced and objective
- Acknowledge uncertainties

Begin your analysis:
"""

        return prompt
