from .openrouter_template import OpenRouterModel
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional
import json
import sys
from datetime import datetime


class AnalysisReport(BaseModel):
    """Structured output for the analysis phase. Replaces free-form markdown.

    Downstream code reads `data_gaps`, `conclusion`, `signal`, `recommendation`,
    and `confidence` directly from this model rather than regex-parsing markdown.
    """
    executive_summary: str = Field(description="2-3 sentence top-line answer to the user query.")
    recommendation: str = Field(description="One of: BUY, HOLD, SELL, NEUTRAL, INFO. Use INFO when the query is factual not directional.")
    signal: str = Field(description="Directional read: 'bullish', 'bearish', 'neutral', or 'n/a'")
    valuation: str = Field(description="Free-text valuation section with calculations. Use 'Label: Value (source)' format.")
    financial_performance: str = Field(description="Financial metrics, margins, growth, returns on capital.")
    macro_context: Optional[str] = Field(default=None, description="Rates, inflation, sector context. Omit if irrelevant.")
    sentiment: Optional[str] = Field(default=None, description="News, analyst consensus, insider activity.")
    risks: List[str] = Field(default_factory=list, description="Bulleted risk factors.")
    assumptions: List[str] = Field(default_factory=list, description="Each as 'NAME: value because reason'.")
    data_gaps: List[str] = Field(default_factory=list, description="Specific data points that were missing and how it affected the analysis.")
    confidence: float = Field(description="0.0 to 1.0 — confidence in the recommendation given data quality.")
    conclusion: str = Field(description="Final 1-2 sentence verdict with the recommendation rationale.")

    def render_markdown(self) -> str:
        """Render the structured report as markdown for the user-visible output."""
        out = [f"## EXECUTIVE SUMMARY\n{self.executive_summary}\n"]
        out.append(f"## RECOMMENDATION: {self.recommendation}\n")
        out.append(f"## VALUATION\n{self.valuation}\n")
        out.append(f"## FINANCIAL PERFORMANCE\n{self.financial_performance}\n")
        if self.macro_context:
            out.append(f"## MACRO CONTEXT\n{self.macro_context}\n")
        if self.sentiment:
            out.append(f"## SENTIMENT\n{self.sentiment}\n")
        if self.risks:
            out.append("## RISKS\n" + "\n".join(f"- {r}" for r in self.risks) + "\n")
        if self.assumptions:
            out.append("## ASSUMPTIONS\n" + "\n".join(f"- {a}" for a in self.assumptions) + "\n")
        out.append("## DATA GAPS\n" + ("\n".join(f"- {g}" for g in self.data_gaps) or "- None") + "\n")
        out.append(f"## CONFIDENCE: {self.confidence:.2f}\n")
        out.append(f"## CONCLUSION\n{self.conclusion}")
        return "\n".join(out)


class Financial_Analysis_Agent(OpenRouterModel):
    """
    Senior Investment Banker AI that performs institutional-grade financial analysis.

    This agent does NOT call tools or orchestrate execution. It receives data from
    tool execution and generates comprehensive analysis considering multiple factors:
    - Company-specific factors (business model, financials, management)
    - Market factors (sector trends, competitive landscape, valuation)
    - Macro factors (interest rates, economic growth, regulation)
    - Risk factors (business, financial, market risks)

    Outputs an AnalysisReport (Pydantic model) — see render_markdown() for the
    user-visible string form.
    """
    response_schema = AnalysisReport
    MAX_OUTPUT_TOKENS = 8192  # R1 thinking + full analysis; also reduces stream drops by finishing faster

    def __init__(self, model_name: str = None):
        # None -> base class picks PRIMARY_REASONING_MODEL (verified-alive at import)
        super().__init__(model_name=model_name)

    def analyze(self,
                user_query: str,
                execution_plan: Dict[str, Any],
                tools_results: List[Dict[str, Any]],
                analytical_considerations: Optional[List[Dict[str, str]]] = None,
                variables: Optional[Dict[str, Any]] = None,
                previous_analysis: Optional[str] = None,
                revision_feedback: Optional[str] = None,
                data_gaps: Optional[List[Dict[str, Any]]] = None,
                model_outputs: Optional[Dict[str, Any]] = None) -> str:
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
            variables or {},
            previous_analysis=previous_analysis,
            revision_feedback=revision_feedback,
            data_gaps=data_gaps,
            model_outputs=model_outputs
        )

        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        model_label = self.model_name.split('/')[-1].split(':')[0]
        print(f"ANALYSIS PHASE - {model_label} Analyzing...", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)

        # Generate analysis
        raw_response = self.generate_response(
            prompt=analysis_prompt,
            system_prompt=system_prompt
        )

        # Try structured parse first. On failure, fall back to a minimal report
        # built from the raw text so the workflow never breaks.
        try:
          report = self.parse_response(raw_response)
          print(f"[Analyze] Structured output parsed: {report.recommendation}, "
                f"confidence={report.confidence:.2f}, gaps={len(report.data_gaps)}",
                file=sys.stderr, flush=True)
        except (ValidationError, Exception) as e:
          print(f"[Analyze] Pydantic parse failed ({type(e).__name__}); "
                "falling back to raw-text report.", file=sys.stderr, flush=True)
          report = self._fallback_report_from_raw(raw_response, user_query)

        markdown = self._strip_unicode_artifacts(report.render_markdown())
        return markdown, report

    @staticmethod
    def _fallback_report_from_raw(raw: str, user_query: str) -> 'AnalysisReport':
        """When Pydantic parsing fails, salvage an AnalysisReport from raw markdown.

        Uses lightweight regex extraction to populate the most-used fields.
        Everything else gets a sensible default so downstream code keeps working.
        """
        import re
        # Pull the first sensible conclusion-shaped section
        concl_match = re.search(
            r'(?:CONCLUSION|RECOMMENDATION|VERDICT)[:\s]*\n?(.*?)$',
            raw, re.IGNORECASE | re.DOTALL,
        )
        conclusion = (concl_match.group(1).strip()[:500] if concl_match
                      else "(unstructured output - see valuation section)")

        # Pull data gaps as bulleted lines under DATA GAPS heading
        gaps: List[str] = []
        gap_match = re.search(
            r'(?:DATA\s*GAPS?|MISSING\s*DATA)[:\s]*\n(.*?)(?=\n\s*(?:##|---|\Z))',
            raw, re.IGNORECASE | re.DOTALL,
        )
        if gap_match:
          for line in gap_match.group(1).split('\n'):
            stripped = line.strip().lstrip('-*0123456789. ')
            if len(stripped) > 5:
              gaps.append(stripped)

        # Recommendation heuristic. Pre-fix the keyword set was too narrow —
        # any directional vocabulary outside ('STRONG BUY', 'BULLISH', etc.)
        # fell through to INFO and suppressed the Bull/Bear debate downstream.
        # Expanded list covers analyst lexicon commonly used when the
        # structured parse fails for reasons unrelated to clarity of direction.
        raw_upper = raw.upper()
        bullish_terms = (
          'STRONG BUY', 'RECOMMEND BUY', 'BULLISH',
          'OUTPERFORM', 'OVERWEIGHT',
          'APPRECIATE', 'ATTRACTIVE', 'UNDERVALUED',
        )
        bearish_terms = (
          'STRONG SELL', 'RECOMMEND SELL', 'BEARISH',
          'UNDERPERFORM', 'UNDERWEIGHT',
          'DEPRECIATE', 'UNFAVORABLE', 'OVERVALUED',
        )
        if any(k in raw_upper for k in bullish_terms):
          rec, signal = 'BUY', 'bullish'
        elif any(k in raw_upper for k in bearish_terms):
          rec, signal = 'SELL', 'bearish'
        elif 'HOLD' in raw_upper or 'NEUTRAL' in raw_upper:
          rec, signal = 'HOLD', 'neutral'
        else:
          rec, signal = 'INFO', 'n/a'

        return AnalysisReport(
          executive_summary=f"(Fallback) Analysis for: {user_query}",
          recommendation=rec,
          signal=signal,
          valuation=raw[:4000],  # dump raw text as valuation section
          financial_performance="(see valuation)",
          risks=[],
          assumptions=[],
          data_gaps=gaps,
          confidence=0.4,  # low confidence since structure was missing
          conclusion=conclusion,
        )

    @staticmethod
    def _strip_unicode_artifacts(text: str) -> str:
        """Remove non-ASCII characters that DeepSeek and other multilingual models
        occasionally bleed into English output (CJK, Cyrillic, Arabic, etc.).

        Examples of artifacts this fixes:
          "9цами.64%"  ->  "9.64%"
          "للحالة"     ->  "" (whole garbled section header removed)
          "ンダ発売"    ->  ""
        """
        import re
        # Drop every non-ASCII character
        cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
        # Collapse multiple spaces left behind by removed runs
        cleaned = re.sub(r'  +', ' ', cleaned)
        return cleaned

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

   VALUATION FACTORS — METHODOLOGY HIERARCHY:
   PRIMARY SIGNALS (drive the conclusion):
   - Relative valuation: P/E vs peers and sector, EV/EBITDA vs peers, PEG ratio
   - Analyst consensus: direction, conviction, and recent trend changes
   - Price momentum: 52-week range positioning, recent trend
   - Insider activity: net buying/selling, insider sentiment MSPR score
   - Forward estimates: consensus EPS/Revenue growth expectations

   SECONDARY SIGNALS (context and confirmation):
   - Macro environment: rate environment impact on multiples, sector tailwinds/headwinds
   - News sentiment: recent catalysts, management guidance, material events
   - Quality indicators: ROIC vs WACC, FCF conversion, balance sheet health

   DCF — STRESS TEST ONLY (never the primary conclusion driver):
   - DCF shows what growth rate the CURRENT PRICE implies, not what the price "should be"
   - A large DCF-to-market gap on a high-quality business (strong ROIC, low PEG, analyst buy) means
     REVISIT TERMINAL ASSUMPTIONS first, not call overvaluation
   - DCF is one data point among many — do not anchor your conclusion on it alone
   - Always contextualize: "DCF implies X% growth embedded in price vs consensus of Y%"
   - If DCF is significantly below market but: analysts are bullish, earnings beats are consistent,
     and ROIC > WACC — the market is pricing in growth the DCF assumptions don't capture

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
- For DCF: always frame as "DCF implies X% embedded growth vs Y% consensus" NOT as a target price verdict
- Conclusion must weigh ALL signals: relative valuation, analyst consensus, momentum, quality, then DCF context

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
- Treating DCF as the primary or dominant valuation signal — it is a STRESS TEST
- Recommending SHORT on a quality business solely because DCF < market price
  (market prices in growth that DCF conservative assumptions miss; that is not overvaluation)
- Ignoring analyst consensus, PEG, and earnings beat rate when DCF disagrees with market
- Giving a definitive BUY/SELL on the basis of DCF alone without referencing relative valuation
"""

        return prompt

    def _format_variable(self, key: str, value: Any) -> str:
        """Format a single variable for display. Handles numbers, strings, dicts, lists."""
        if isinstance(value, bool):
            # Must check bool before int/float — bool is a subclass of int in Python
            return f"{key}: {value}"
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

        elif tool_name == 'get_earnings_surprises':
            if not isinstance(data, dict):
                return f"[EARNINGS SURPRISES] {data}\n\n"
            lines = [
                f"[EARNINGS SURPRISES] {data.get('total_periods', 0)} quarters | "
                f"Beat rate: {data.get('beat_rate_pct', 'N/A')}% | "
                f"Avg surprise: {data.get('avg_surprise_pct', 'N/A')}%"
            ]
            for q in data.get('quarters', [])[:8]:
                result = q.get('result', '?').upper()
                surprise = f"{q.get('surprise_pct', 0):+.1f}%" if 'surprise_pct' in q else ''
                lines.append(f"  {q.get('period', '')} Q{q.get('quarter', '')}: "
                             f"Actual={q.get('actual_eps', 'N/A')} vs Est={q.get('estimate_eps', 'N/A')} "
                             f"{surprise} [{result}]")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_forward_estimates':
            if not isinstance(data, dict):
                return f"[FORWARD ESTIMATES] {data}\n\n"
            lines = ["[FORWARD CONSENSUS ESTIMATES]"]
            for label, unit_note in [('eps', 'USD/share'), ('revenue_B', 'USD B'), ('ebitda_B', 'USD B')]:
                section = data.get(label, {})
                if 'error' in section:
                    continue
                lines.append(f"  {label.replace('_B', '').upper()} ({unit_note}):")
                for p in section.get('periods', [])[:4]:
                    avg = p.get('avg', 'N/A')
                    high = p.get('high', '')
                    low = p.get('low', '')
                    n = p.get('analysts', '')
                    range_str = f" (range: {low}-{high})" if low and high else ''
                    analysts_str = f" [{n} analysts]" if n else ''
                    lines.append(f"    {p.get('period', '')}: {avg}{range_str}{analysts_str}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_price_target':
            if not isinstance(data, dict):
                return f"[PRICE TARGET] {data}\n\n"
            mean = data.get('targetMean', 'N/A')
            median = data.get('targetMedian', 'N/A')
            high = data.get('targetHigh', 'N/A')
            low = data.get('targetLow', 'N/A')
            n = data.get('numberOfAnalysts', 'N/A')
            updated = data.get('lastUpdated', '')
            lines = [
                f"[PRICE TARGET] Mean: ${mean} | Median: ${median} | "
                f"Range: ${low}-${high} | {n} analysts | Updated: {updated}"
            ]
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_financial_statements':
            if not isinstance(data, dict):
                return f"[FINANCIAL STATEMENTS] {data}\n\n"
            stmt = data.get('statement', '?').upper()
            freq = data.get('freq', '')
            periods = data.get('periods', [])
            lines = [f"[FINANCIAL STATEMENTS - {stmt} {freq.upper()}] {len(periods)} periods"]
            for p in periods[:5]:
                period_label = p.get('period', p.get('endDate', '?'))
                fields = {k: v for k, v in p.items() if k != 'period' and v is not None}
                # Format large numbers in billions
                formatted = []
                for k, v in list(fields.items())[:10]:
                    if isinstance(v, (int, float)) and abs(v) >= 1e6:
                        formatted.append(f"{k}: ${v/1e9:.2f}B")
                    else:
                        formatted.append(f"{k}: {v}")
                lines.append(f"  {period_label}: {' | '.join(formatted)}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_company_profile':
            if not isinstance(data, dict):
                return f"[COMPANY PROFILE] {data}\n\n"
            name = data.get('name', 'N/A')
            industry = data.get('finnhubIndustry', data.get('gics', 'N/A'))
            country = data.get('country', 'N/A')
            employees = data.get('employeeTotal', 'N/A')
            ipo = data.get('ipo', 'N/A')
            lines = [
                f"[COMPANY PROFILE] {name} | Industry: {industry} | Country: {country}",
                f"  Employees: {employees} | IPO: {ipo}",
            ]
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_upgrade_downgrades':
            if not isinstance(data, dict):
                return f"[ANALYST RATING CHANGES] {data}\n\n"
            upgrades = data.get('upgrade_count', 0)
            downgrades = data.get('downgrade_count', 0)
            initiations = data.get('initiation_count', 0)
            events = data.get('events', [])
            lines = [
                f"[ANALYST RATING CHANGES] Upgrades: {upgrades} | Downgrades: {downgrades} | Initiations: {initiations}"
            ]
            for ev in events[:10]:
                date = ev.get('date', '')
                firm = ev.get('firm', 'Unknown')
                action = ev.get('action', '').title()
                from_g = ev.get('from', '')
                to_g = ev.get('to', '')
                if from_g and to_g:
                    lines.append(f"  {date} | {firm}: {action} ({from_g} -> {to_g})")
                else:
                    lines.append(f"  {date} | {firm}: {action}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_insider_sentiment':
            if not isinstance(data, dict):
                return f"[INSIDER SENTIMENT (MSPR)] {data}\n\n"
            signal = data.get('signal', 'neutral').upper()
            avg_mspr = data.get('avg_mspr')
            months = data.get('months', [])
            lines = [f"[INSIDER SENTIMENT (MSPR)] Signal: {signal} | Avg MSPR (6M): {avg_mspr}"]
            if months:
                month_parts = []
                for m in months[:6]:
                    mspr = m.get('mspr')
                    if mspr is not None:
                        month_parts.append(f"{m.get('year')}-{m.get('month', 0):02d}:{mspr:+.3f}")
                if month_parts:
                    lines.append(f"  Monthly: {' | '.join(month_parts)}")
            return "\n".join(lines) + "\n\n"

        elif tool_name == 'get_sector_metrics':
            # data may be a dict or list after flattening
            entry = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
            if not entry:
                return f"[SECTOR METRICS] No data\n\n"
            sector = entry.get('sector', 'Sector')
            parts = []
            for key, label in [('pe', 'P/E'), ('pb', 'P/B'), ('pc', 'P/C'),
                                ('pfcf', 'P/FCF'), ('dividendYield', 'Div Yield')]:
                val = entry.get(key)
                if val is not None:
                    parts.append(f"{label}: {val:.2f}")
            return f"[SECTOR METRICS: {sector}] {' | '.join(parts)}\n\n"

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

        elif tool_name == 'get_credit_spreads':
            if not isinstance(data, dict):
                return f"[CREDIT SPREADS] {data}\n\n"
            lines = ["[CREDIT SPREADS (OAS in bps)]"]
            for sid, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                label = entry.get('label', sid)
                bps = entry.get('current_bps', 'N/A')
                chg_3m = entry.get('3m_change_bps')
                chg_1y = entry.get('1y_change_bps')
                parts = [f"  {label}: {bps}bps"]
                if chg_3m is not None:
                    parts.append(f"3M: {chg_3m:+.1f}bps")
                if chg_1y is not None:
                    parts.append(f"1Y: {chg_1y:+.1f}bps")
                lines.append(' | '.join(parts))
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

    def _format_model_outputs(self, model_outputs: Dict[str, Any]) -> str:
        """Format Financial_Modeling_Agent results as a labeled prompt section."""
        models_run = model_outputs.get('models_run', [])
        if not models_run:
            return ""

        lines = ["--- FINANCIAL MODELING OUTPUTS (Python math, verified calculations) ---\n"]

        if 'scenario_dcf' in model_outputs:
            s = model_outputs['scenario_dcf']
            pr = s.get('price_range', {})
            assumptions = s.get('scenario_assumptions', {})
            lines.append("SCENARIO DCF (5-year FCF, three-case):")
            lines.append(f"  Bear (low growth / margin compression): ${pr.get('low', 0):.2f}/share")
            lines.append(f"  Base (consensus anchored):              ${pr.get('mid', 0):.2f}/share")
            lines.append(f"  Bull (strong growth / margin expansion): ${pr.get('high', 0):.2f}/share")
            lines.append(f"  Price range: ${pr.get('low', 0):.2f} - ${pr.get('high', 0):.2f}")
            if assumptions:
                bg1 = assumptions.get('base_growth_y1', 0) * 100
                bgl = assumptions.get('base_growth_long_run', 0) * 100
                bma = assumptions.get('bear_margin_adj', 0) * 100
                bua = assumptions.get('bull_margin_adj', 0) * 100
                lines.append(f"  Base growth Y1={bg1:.1f}% -> long-run={bgl:.1f}% | "
                              f"Bear margin adj={bma:.1f}pp | Bull margin adj={bua:+.1f}pp")
            # Regime-weighted expected price
            rw = s.get('regime_weighted')
            if rw:
                w = rw.get('weights', {})
                lines.append(f"  Regime: {rw.get('regime', 'N/A')} | "
                              f"Weights: bear={w.get('bear',0):.0%} "
                              f"base={w.get('base',0):.0%} "
                              f"bull={w.get('bull',0):.0%}")
                lines.append(f"  Probability-weighted expected price: ${rw.get('expected_price', 0):.2f}")
            lines.append("")

        if 'sensitivity_table' in model_outputs:
            st = model_outputs['sensitivity_table']
            lines.append("SENSITIVITY TABLE (price per share, WACC x terminal growth):")
            lines.append(f"  Price range: ${st.get('min_price', 0):.2f} - ${st.get('max_price', 0):.2f} | "
                          f"Mid: ${st.get('mid_price', 0):.2f} | Cells: {st.get('cells_filled', 0)}")
            tg_range = st.get('tg_range', [])
            header = "  WACC \\ g% |" + " | ".join(f"{tg*100:5.2f}%" for tg in tg_range)
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for w_key, row in st.get('table', {}).items():
                w_pct = float(w_key) * 100
                cells = []
                for tg in tg_range:
                    v = row.get(f"{tg:.4f}")
                    cells.append("  N/A " if v is None else f"${v:6.2f}")
                lines.append(f"  {w_pct:6.2f}%  | " + " | ".join(cells))
            lines.append("")

        if 'ddm' in model_outputs:
            d = model_outputs['ddm']
            if d.get('success'):
                lines.append("DIVIDEND DISCOUNT MODEL (DDM):")
                lines.append(f"  Method: {d.get('method', 'gordon_growth')}")
                lines.append(f"  Intrinsic value per share: ${d.get('intrinsic_value_per_share', 0):.2f}")
                lines.append(f"  Inputs: Ke={d.get('cost_of_equity', 0)*100:.2f}% | "
                              f"g={d.get('terminal_growth', 0)*100:.2f}%")
                if d.get('method') == 'two_stage':
                    lines.append(f"  High-growth phase: {d.get('high_growth_years')}y at "
                                  f"{d.get('high_growth_rate', 0)*100:.1f}%")
                lines.append("")

        if 'credit_profile' in model_outputs:
            c = model_outputs['credit_profile']
            lines.append("CREDIT PROFILE:")
            lines.append(f"  Credit label: {c.get('credit_label', 'N/A')}")
            lines.append(f"  Net Debt/EBITDA: {c.get('net_debt_ebitda', 0):.1f}x")
            lines.append(f"  Interest Coverage (EBITDA/Interest): {c.get('interest_coverage', 0):.1f}x")
            lines.append(f"  FCF Yield: {c.get('fcf_yield_pct', 0):.1f}%")
            lines.append(f"  Net Debt: ${c.get('net_debt', 0)/1e9:.2f}B")
            lines.append("")

        if 'capital_returns' in model_outputs:
            r = model_outputs['capital_returns']
            lines.append("CAPITAL RETURNS (Shareholder Yield):")
            lines.append(f"  FCF Yield: {r.get('fcf_yield_pct', 0):.1f}%")
            lines.append(f"  Dividend Yield: {r.get('dividend_yield_pct', 0):.1f}%")
            lines.append(f"  Buyback Yield: {r.get('buyback_yield_pct', 0):.1f}%")
            lines.append(f"  Total Shareholder Yield: {r.get('total_shareholder_yield_pct', 0):.1f}%")
            lines.append(f"  Payout Sustainability: {r.get('sustainability', 'N/A')}")
            lines.append("")

        if 'lbo' in model_outputs:
            l = model_outputs['lbo']
            lines.append("LBO ANALYSIS:")
            lines.append(f"  Entry EV: ${l.get('entry_ev', 0)/1e9:.2f}B | Entry EBITDA multiple: {l.get('entry_ebitda_multiple', 0):.1f}x")
            lines.append(f"  IRR: {l.get('irr_pct', 0):.1f}% | MOIC: {l.get('moic', 0):.2f}x")
            lines.append(f"  Achieves 20%+ IRR: {l.get('achieves_20pct_irr', False)}")
            lines.append(f"  Exit equity: ${l.get('exit_equity', 0)/1e9:.2f}B over {l.get('hold_years', 5)} years")
            lines.append("")

        lines.append("")
        return "\n".join(lines)

    def _build_analysis_prompt(self,
                               user_query: str,
                               execution_plan: Dict[str, Any],
                               tools_results: List[Dict[str, Any]],
                               analytical_considerations: Optional[List[Dict[str, str]]] = None,
                               variables: Dict[str, Any] = None,
                               previous_analysis: Optional[str] = None,
                               revision_feedback: Optional[str] = None,
                               data_gaps: Optional[List[Dict[str, Any]]] = None,
                               model_outputs: Optional[Dict[str, Any]] = None) -> str:
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

        prompt = f"""TODAY'S DATE: {current_date}
TICKER: {execution_plan.get('ticker', flat_vars.get('ticker', 'N/A'))}
ALL PROJECTIONS START FROM {datetime.now().year}. USE ONLY DATA PROVIDED BELOW. DO NOT HALLUCINATE NUMBERS.

KEY-PREFERENCE RULES when citing revenue, EBITDA, or net income:
  - Prefer `revenue_ttm`, `ebitda_ttm`, `net_income_ttm` (trailing-twelve-month, most current).
  - `revenue_base` and `ebitda_amount` are LATEST FISCAL YEAR values from SEC filings — use these only for DCF starting points or when explicitly comparing to a prior FY. They are NOT current-period revenue.
  - When both exist, narrative claims like "Apple did $X in revenue" must use the `_ttm` variant.

--- GATHERED DATA ---

{gathered_display}

"""

        # Add financial modeling outputs (scenario DCF, credit profile, capital returns, LBO)
        # These are Python-computed, verified calculations -- treat as primary quantitative inputs.
        if model_outputs and model_outputs.get('models_run'):
            prompt += self._format_model_outputs(model_outputs)

        prompt += f"""--- CLIENT REQUEST ---

{user_query}

"""

        # Add analytical guidance from probing phase
        if analytical_considerations:
            prompt += "--- ANALYTICAL GUIDANCE ---\n\n"
            for idx, item in enumerate(analytical_considerations, 1):
                if isinstance(item, dict):
                    prompt += f"{idx}. [{item.get('topic', 'General')}] {item.get('guidance', '')}\n"
                else:
                    prompt += f"{idx}. {item}\n"
            prompt += "\n"

        # Add execution context
        prompt += f"--- ANALYSIS CONTEXT ---\n\nTask Type: {execution_plan.get('task_type', 'N/A')}\n"
        if 'reasoning' in execution_plan:
            prompt += f"Strategy: {execution_plan['reasoning']}\n"
        prompt += "\n"

        # Add tool results that contain nested data (lists, dicts) that
        # _flatten_result skips. These are full outputs like DCF projections,
        # comp tables, disclosure data, etc.
        detailed_parts = []
        for result in tools_results:
            result_type = result.get('type', '')
            tool_name = result.get('tool', '')
            data = result.get('data', {})

            if result_type in ('sec_data', 'financial_data') and isinstance(data, dict):
                nested = {k: v for k, v in data.items() if isinstance(v, (dict, list)) and v}
                if nested:
                    detailed_parts.append(f"[{tool_name}] {json.dumps(nested, indent=1, default=str)}\n")

        if detailed_parts:
            prompt += f"--- DETAILED TOOL OUTPUT ---\n\n{''.join(detailed_parts)}\n"

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
            prompt += f"--- MARKET INTELLIGENCE ---\n\n{''.join(intel_parts)}\n"

        if macro_parts:
            prompt += f"--- MACRO ENVIRONMENT ---\n\n{''.join(macro_parts)}\n"

        if news_parts:
            prompt += f"--- NEWS & SENTIMENT ---\n\n{''.join(news_parts)}\n"

        # Add unstructured data (search results, web content) as supplementary
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
            prompt += f"--- SUPPLEMENTARY RESEARCH ({len(supp_parts)} sources) ---\n\n{''.join(supp_parts)}\n"

        # Surface data gaps from Plan Verifier so the model makes explicit labeled assumptions
        # rather than silently ignoring missing inputs or hallucinating substitutes.
        if data_gaps:
            gap_lines = []
            for gap in data_gaps:
                priority = gap.get('priority', 'helpful').upper()
                description = gap.get('description', '')
                tool = gap.get('recommended_tool', '')
                gap_lines.append(f"  [{priority}] {description} (tool: {tool})")
            prompt += (
                "--- DATA GAPS (fetching these failed -- make explicit ASSUMPTION: labels) ---\n\n"
                + "\n".join(gap_lines)
                + "\n\nFor each gap above: state the assumption you are using and why, "
                "label it ASSUMPTION:, and note how it affects your confidence.\n\n"
            )

        # If this is a revision, anchor on the prior conclusion + reviewer feedback.
        # previous_analysis is just the conclusion section -- not the full report.
        if revision_feedback:
            if previous_analysis:
                prompt += f"--- PRIOR CONCLUSION ---\n\n{previous_analysis}\n\n"
            prompt += f"""--- REVISION INSTRUCTIONS ---

The quality reviewer found the following issue with the previous analysis:

{revision_feedback}

--- YOUR TASK ---

Write a COMPLETE fresh analysis answering: "{user_query}"
"""
            if previous_analysis:
                prompt += "The prior conclusion above shows where the analysis landed -- remain consistent with it unless the reviewer's feedback directly contradicts it.\n\n"
            prompt += """OUTPUT FORMAT: Return ONLY valid JSON matching this schema, no markdown wrapper:
{
  "executive_summary": "2-3 sentences with recommendation",
  "recommendation": "BUY | HOLD | SELL | NEUTRAL | INFO",
  "signal": "bullish | bearish | neutral | n/a",
  "valuation": "Free-text section. Use 'Label: Value (source)' pairs. Show calculations inline e.g. 'FCF = EBITDA - Capex - Tax = $144.8B - $12.7B - $20.7B = $111.4B'",
  "financial_performance": "Margins, growth, returns on capital",
  "macro_context": "Rates, inflation, sector context (or null if irrelevant)",
  "sentiment": "News, analyst consensus, insider activity (or null)",
  "risks": ["risk 1", "risk 2", ...],
  "assumptions": ["NAME: value because reason", ...],
  "data_gaps": ["specific missing data point", ...],
  "confidence": 0.75,
  "conclusion": "1-2 sentence final verdict with recommendation rationale"
}
"""
        else:
            # Split the interpolated part from the literal JSON schema so the
            # braces in the schema example don't get parsed as f-string format
            # specifiers (would raise ValueError: Invalid format specifier).
            prompt += f"""--- YOUR TASK ---

Analyze the data above to answer: "{user_query}"

Rules:
1. Use ONLY the data provided. Do not invent numbers.
2. If data is missing, write "ASSUMPTION: [value]" and justify it.
3. Show calculations step-by-step where applicable.
4. Do not confuse millions, billions, and trillions.

OUTPUT FORMAT: Return ONLY valid JSON matching this schema, no markdown wrapper:
"""
            prompt += """{
  "executive_summary": "2-3 sentences with recommendation",
  "recommendation": "BUY | HOLD | SELL | NEUTRAL | INFO",
  "signal": "bullish | bearish | neutral | n/a",
  "valuation": "Free-text section. Use 'Label: Value (source)' pairs. Show calculations inline e.g. 'FCF = EBITDA - Capex - Tax = $144.8B - $12.7B - $20.7B = $111.4B'",
  "financial_performance": "Margins, growth, returns on capital",
  "macro_context": "Rates, inflation, sector context (or null if irrelevant)",
  "sentiment": "News, analyst consensus, insider activity (or null)",
  "risks": ["risk 1", "risk 2", ...],
  "assumptions": ["NAME: value because reason", ...],
  "data_gaps": ["specific missing data point", ...],
  "confidence": 0.75,
  "conclusion": "1-2 sentence final verdict with recommendation rationale"
}
"""

        return prompt
