from .openrouter_template import OpenRouterModel
from typing import Dict, Any, List, Optional
import json
import sys
from datetime import datetime


class Financial_Analysis_Agent(OpenRouterModel):
    """
    Senior Investment Banker AI that performs institutional-grade financial analysis.

    This agent does NOT call tools or orchestrate execution. It receives data from
    tool execution and generates comprehensive analysis considering multiple factors:
    - Company-specific factors (business model, financials, management)
    - Market factors (sector trends, competitive landscape, valuation)
    - Macro factors (interest rates, economic growth, regulation)
    - Risk factors (business, financial, market risks)
    """
    MAX_OUTPUT_TOKENS = 4096  # R1 needs room for thinking + actual analysis output

    def __init__(self, model_name: str = 'deepseek/deepseek-r1-0528:free'):
        super().__init__(model_name=model_name)

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
        model_label = self.model_name.split('/')[-1].split(':')[0]
        print(f"ANALYSIS PHASE - {model_label} Analyzing...", file=sys.stderr, flush=True)
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
- CONCISE: 400-600 words max. Density over length. No filler prose.
- Use labeled sections with clear headers (e.g., "VALUATION:", "RISKS:")
- Use bullet points and "Key: Value" pairs, NOT paragraphs of prose
- Show calculations inline (e.g., "WACC = 4.09% + 1.11*5.0% = 9.64%")
- Every number must cite its source (e.g., "Revenue: $416.2B (SEC filing)")
- State assumptions explicitly with label "ASSUMPTION:" and justification
- List data gaps and how they affect your confidence
- End with a clear, actionable conclusion

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

    def _format_market_intel(self, tool_name: str, data: Any) -> str:
        """Format a market intelligence result into a readable section for the analysis prompt."""
        if tool_name == 'get_insider_transactions':
            if not isinstance(data, dict):
                return f"[INSIDER ACTIVITY] {data}\n\n"
            signal = data.get('signal', 'unknown').replace('_', ' ').upper()
            lines = [
                f"[INSIDER ACTIVITY] Signal: {signal}",
                f"  Net shares: {data.get('net_shares', 0):,} (bought: {data.get('total_bought', 0):,}, sold: {data.get('total_sold', 0):,})",
                f"  Transactions: {data.get('buy_count', 0)} buys, {data.get('sell_count', 0)} sells",
            ]
            r30 = data.get('recent_30d', {})
            r90 = data.get('recent_90d', {})
            if r30.get('net', 0) != 0 or r90.get('net', 0) != 0:
                lines.append(f"  Last 30d net: {r30.get('net', 0):,} | Last 90d net: {r90.get('net', 0):,}")
            for insider in data.get('top_insiders', []):
                lines.append(f"  - {insider['name']}: {insider['net_shares']:,} shares ({insider['transaction_count']} txns)")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_analyst_recommendations':
            if not isinstance(data, dict):
                return f"[ANALYST CONSENSUS] {data}\n\n"
            consensus = data.get('consensus', 'unknown').replace('_', ' ').upper()
            trend = data.get('trend', 'unknown').upper()
            lines = [
                f"[ANALYST CONSENSUS] {consensus} (trend: {trend}, {data.get('total_analysts', 0)} analysts)",
            ]
            latest = data.get('latest', {})
            if latest:
                lines.append(f"  Current: Strong Buy={latest.get('strong_buy',0)}, Buy={latest.get('buy',0)}, Hold={latest.get('hold',0)}, Sell={latest.get('sell',0)}, Strong Sell={latest.get('strong_sell',0)}")
            prior = data.get('prior', {})
            if prior:
                lines.append(f"  Prior ({prior.get('period','')}): SB={prior.get('strong_buy',0)}, B={prior.get('buy',0)}, H={prior.get('hold',0)}, S={prior.get('sell',0)}, SS={prior.get('strong_sell',0)}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_company_peers':
            if isinstance(data, list):
                return f"[PEER COMPANIES] {', '.join(data)}\n\n"
            return f"[PEER COMPANIES] {data}\n\n"

        elif tool_name == 'get_earnings_calendar':
            if not isinstance(data, dict):
                return f"[EARNINGS CALENDAR] {data}\n\n"
            lines = [f"[EARNINGS CALENDAR] {data.get('total_companies', 0)} companies reporting"]
            for d in data.get('by_date', []):
                lines.append(f"  {d['date']}: {d['count']} companies")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_basic_financials':
            if not isinstance(data, dict):
                return f"[KEY FINANCIALS] {data}\n\n"
            metrics = data.get('metric') or {}
            lines = [f"[KEY FINANCIALS] {data.get('metric_count', len(metrics))} metrics"]
            # Group into readable categories
            val = []
            if metrics.get('peTTM') is not None: val.append(f"P/E={metrics['peTTM']:.1f}")
            if metrics.get('forwardPE') is not None: val.append(f"Fwd P/E={metrics['forwardPE']:.1f}")
            if metrics.get('evEbitdaTTM') is not None: val.append(f"EV/EBITDA={metrics['evEbitdaTTM']:.1f}")
            if metrics.get('pegTTM') is not None: val.append(f"PEG={metrics['pegTTM']:.2f}")
            if metrics.get('psTTM') is not None: val.append(f"P/S={metrics['psTTM']:.1f}")
            if metrics.get('pbQuarterly') is not None: val.append(f"P/B={metrics['pbQuarterly']:.1f}")
            if val: lines.append(f"  Valuation: {', '.join(val)}")

            margins = []
            if metrics.get('grossMarginTTM') is not None: margins.append(f"Gross={metrics['grossMarginTTM']:.1f}%")
            if metrics.get('operatingMarginTTM') is not None: margins.append(f"Op={metrics['operatingMarginTTM']:.1f}%")
            if metrics.get('netProfitMarginTTM') is not None: margins.append(f"Net={metrics['netProfitMarginTTM']:.1f}%")
            if margins: lines.append(f"  Margins: {', '.join(margins)}")

            growth = []
            if metrics.get('epsGrowthTTMYoy') is not None: growth.append(f"EPS YoY={metrics['epsGrowthTTMYoy']:.1f}%")
            if metrics.get('revenueGrowthTTMYoy') is not None: growth.append(f"Rev YoY={metrics['revenueGrowthTTMYoy']:.1f}%")
            if metrics.get('ebitdaCagr5Y') is not None: growth.append(f"EBITDA 5Y CAGR={metrics['ebitdaCagr5Y']:.1f}%")
            if growth: lines.append(f"  Growth: {', '.join(growth)}")

            returns = []
            if metrics.get('roeTTM') is not None: returns.append(f"ROE={metrics['roeTTM']:.1f}%")
            if metrics.get('roaTTM') is not None: returns.append(f"ROA={metrics['roaTTM']:.1f}%")
            if returns: lines.append(f"  Returns: {', '.join(returns)}")

            leverage = []
            if metrics.get('totalDebt/totalEquityQuarterly') is not None: leverage.append(f"D/E={metrics['totalDebt/totalEquityQuarterly']:.2f}")
            if metrics.get('currentRatioQuarterly') is not None: leverage.append(f"Current={metrics['currentRatioQuarterly']:.2f}")
            if metrics.get('beta') is not None: leverage.append(f"Beta={metrics['beta']:.2f}")
            if leverage: lines.append(f"  Leverage/Risk: {', '.join(leverage)}")

            size = []
            if 'marketCapitalization' in metrics:
                mc = metrics['marketCapitalization']
                size.append(f"MktCap={'${:.1f}T'.format(mc/1e6) if mc >= 1e6 else '${:.1f}B'.format(mc/1e3)}")
            if 'enterpriseValue' in metrics:
                ev = metrics['enterpriseValue']
                size.append(f"EV={'${:.1f}T'.format(ev/1e6) if ev >= 1e6 else '${:.1f}B'.format(ev/1e3)}")
            if size: lines.append(f"  Size: {', '.join(size)}")

            price = []
            if metrics.get('52WeekHigh') is not None: price.append(f"52wk High=${metrics['52WeekHigh']:.2f}")
            if metrics.get('52WeekLow') is not None: price.append(f"52wk Low=${metrics['52WeekLow']:.2f}")
            if price: lines.append(f"  Price: {', '.join(price)}")

            return "\n".join(lines) + "\n\n"

        else:
            return f"[{tool_name.upper()}] {json.dumps(data, indent=1, default=str)[:500]}\n\n"

    def _format_macro(self, tool_name: str, data: Any) -> str:
        """Format a macro data result into a readable section for the analysis prompt."""
        if tool_name == 'get_macro_snapshot':
            if not isinstance(data, dict):
                return f"[MACRO SNAPSHOT] {data}\n\n"
            sections = {"Rates": [], "Inflation": [], "Employment": [], "Growth": []}
            rate_ids = {"DGS10", "DGS2", "FEDFUNDS", "T10Y2Y", "BAMLH0A0HYM2"}
            inflation_ids = {"CPIAUCSL", "PCEPILFE"}
            employment_ids = {"UNRATE", "PAYEMS", "ICSA"}
            growth_ids = {"A191RL1Q225SBEA"}

            for sid, entry in data.items():
                if not isinstance(entry, dict) or 'error' in entry:
                    continue
                label = entry.get('label', sid)

                if sid in inflation_ids:
                    yoy = entry.get('yoy_pct')
                    if yoy is not None:
                        sections["Inflation"].append(f"  {label}: {yoy:.1f}% YoY")
                else:
                    current = entry.get('current')
                    if current is None:
                        continue
                    line = f"  {label}: {current}"
                    chg_3m = entry.get('3m_change_bps')
                    chg_1y = entry.get('1y_change_bps')
                    if chg_3m is not None or chg_1y is not None:
                        parts = []
                        if chg_3m is not None:
                            parts.append(f"3M: {chg_3m:+.0f}bps")
                        if chg_1y is not None:
                            parts.append(f"1Y: {chg_1y:+.0f}bps")
                        line += f" ({', '.join(parts)})"

                    if sid in rate_ids:
                        sections["Rates"].append(line)
                    elif sid in employment_ids:
                        sections["Employment"].append(line)
                    elif sid in growth_ids:
                        sections["Growth"].append(line)

            lines = ["[MACRO SNAPSHOT]"]
            for section_name, items in sections.items():
                if items:
                    lines.append(f"  --- {section_name} ---")
                    lines.extend(items)
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_treasury_yields':
            if not isinstance(data, dict):
                return f"[YIELD CURVE] {data}\n\n"
            curve = data.get('curve', {})
            shape = data.get('shape', 'unknown')
            spreads = data.get('spreads', {})

            lines = [f"[YIELD CURVE] Shape: {shape.upper()}"]
            # Order maturities
            order = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
            row = []
            for m in order:
                if m in curve:
                    row.append(f"{m}={curve[m]:.2f}%")
            lines.append(f"  {' | '.join(row)}")
            if spreads:
                spread_parts = [f"{k}: {v:+.3f}%" for k, v in spreads.items()]
                lines.append(f"  Spreads: {', '.join(spread_parts)}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_fred_series':
            if not isinstance(data, dict):
                return f"[FRED SERIES] {data}\n\n"
            sid = data.get('series_id', 'unknown')
            obs = data.get('observations', [])
            lines = [f"[FRED SERIES: {sid}] {len(obs)} observations"]
            # Show last 5 values
            for o in obs[-5:]:
                lines.append(f"  {o.get('date', '?')}: {o.get('value', '?')}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'search_fred':
            if not isinstance(data, dict):
                return f"[FRED SEARCH] {data}\n\n"
            results = data.get('results', [])
            lines = [f"[FRED SEARCH] {data.get('total_matches', 0)} matches, showing {len(results)}"]
            for r in results[:10]:
                lines.append(f"  {r.get('series_id','')}: {r.get('title','')} ({r.get('frequency','')}, {r.get('units','')})")
            return "\n".join(lines) + "\n\n"

        else:
            return f"[{tool_name.upper()}] {json.dumps(data, indent=1, default=str)[:500]}\n\n"

    def _format_news_summary(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format a news result into a readable section for the analysis prompt."""
        source_label = "Company News" if tool_name == 'get_company_news' else "Market News"
        result_type = result.get('type', '')

        if result_type == 'news_analysis':
            score = result.get('sentiment_score', 0)
            overall = result.get('overall_sentiment', 'unknown')
            analyzed = result.get('articles_analyzed', 0)
            relevant = result.get('articles_relevant', 0)

            lines = [
                f"[{source_label.upper()}] {analyzed} articles | "
                f"{relevant} relevant | Sentiment: {overall.upper()} (score: {score:+.2f})"
            ]

            themes = result.get('key_themes', [])
            if themes:
                lines.append("  Key themes:")
                for theme in themes:
                    lines.append(f"  - {theme}")

            # Show high-impact articles for traceability
            assessments = result.get('article_assessments', [])
            high_impact = [a for a in assessments if a.get('relevant') and a.get('impact') == 'high']
            if high_impact:
                lines.append("  High-impact articles:")
                for a in high_impact[:5]:
                    direction = "+" if a['sentiment'] == 'bullish' else "-" if a['sentiment'] == 'bearish' else "~"
                    headline = a.get('headline', '')[:80]
                    reason = a.get('reason', '')[:100]
                    lines.append(f"  [{direction}] {headline}: {reason}")

            return "\n".join(lines) + "\n\n"

        else:
            return f"[{source_label.upper()}] Unrecognized news format\n\n"

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

        # Add market intelligence data (Finnhub: insider, analyst, peers, earnings, financials)
        intel_parts = []
        macro_parts = []
        news_parts = []

        for result in tools_results:
            result_type = result.get('type', '')
            tool_name = result.get('tool', '')
            data = result.get('data', {})

            if result_type == 'market_intel':
                intel_parts.append(self._format_market_intel(tool_name, data))
            elif result_type == 'macro':
                macro_parts.append(self._format_macro(tool_name, data))
            elif result_type == 'news_analysis':
                news_parts.append(self._format_news_summary(tool_name, result))

        if intel_parts:
            prompt += f"""
################################################################################
#                    MARKET INTELLIGENCE                                       #
################################################################################

{"".join(intel_parts)}
"""

        if macro_parts:
            prompt += f"""
################################################################################
#                    MACRO ENVIRONMENT                                        #
################################################################################

{"".join(macro_parts)}
"""

        if news_parts:
            prompt += f"""
################################################################################
#                    NEWS & SENTIMENT                                          #
################################################################################

{"".join(news_parts)}
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

FORMAT GUIDELINES:
- Start with a 2-3 sentence executive summary including your recommendation if applicable
- Use section headers relevant to the query (e.g., VALUATION, MACRO CONTEXT, SENTIMENT, RISKS)
- Present data as "Label: Value (source)" pairs and bullet points, not prose paragraphs
- Show all calculations step-by-step on a single line (e.g., "FCF = EBITDA - Capex - Tax = $144.8B - $12.7B - $20.7B = $111.4B")
- End with ASSUMPTIONS (what you assumed), DATA GAPS (what was missing), and your CONCLUSION
- Keep it tight -- the output goes directly to a quality verification agent

Begin your analysis:
"""

        return prompt
