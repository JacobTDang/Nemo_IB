from typing import Any, Dict, List
import asyncio
import json
import os
from datetime import date, datetime
from tools.web_search_server.searxng_client import searxng_search
from tools.web_search_server.scraper import scrape_urls
from agent.cache import Session_Cache
from tools.web_search_server.sec_utils import (
    get_revenue_base, get_ebitda_margin, get_capex_pct_revenue,
    get_tax_rate, get_depreciation, get_disclosures_names,
    extract_disclosure_data, get_latest_filing,
    get_margin_breakdown, get_historical_fcf, get_working_capital,
    get_buyback_history, get_segment_financials, extract_risk_factors,
    extract_mda, get_earnings_releases, get_patent_filings,
    get_company_filings_history, get_supply_chain, diff_10k,
    get_schedule_13d_filings, track_segment_growth, extract_call_sentiment,
    extract_forward_signals,
)
from tools.web_search_server.hf_letters import (
    compare_fund_holdings, list_known_funds, get_fund_holdings,
)
import importlib.util
import sys

# import module with number in name
try:
    spec = importlib.util.spec_from_file_location(
        "filing_parser",
        "tools/web_search_server/8K_and_DEF14A_utils.py"
    )
    if spec and spec.loader:
        filing_parser = importlib.util.module_from_spec(spec)
        sys.modules["filing_parser"] = filing_parser
        spec.loader.exec_module(filing_parser)
    else:
        raise ImportError("Could not load filing parser module")
except Exception:
    # fallback - set filing_parser to None and handle in methods
    filing_parser = None

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

def json_serializer(obj):
  """JSON serializer for objects not serializable by default json code"""
  if isinstance(obj, (date, datetime)):
    return obj.isoformat()
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def safe_json_dumps(obj):
  """Safely serialize objects to JSON, handling dates and other non-serializable types"""
  return json.dumps(obj, default=json_serializer)

def _warm_embedder_session() -> None:
  """Pre-load the sentence-transformers model in a daemon thread so the
  first rag_search call doesn't pay the ~5-10s cold-start cost. The
  embedder is a module-level singleton, so once loaded it stays hot for
  the lifetime of this MCP server process. Failures are silent — RAG
  tools will fail loudly at query time if the model truly can't load,
  which is the correct behavior."""
  try:
    from agent.rag.embedder import embed
    _ = embed("warmup")
  except Exception:
    pass


class WebSearchServer:
  def __init__(self):
    self.server = Server("web_client")
    self.cache = Session_Cache()
    self._setup_handlers()
    # Warm the embedding model in the background so first rag_search
    # arrives to a hot model. Same pattern as the yfinance warmup in
    # finnhub_server.py and alpaca/server.py.
    import threading
    threading.Thread(target=_warm_embedder_session, daemon=True).start()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="search",
          description="search the internet for information",
          inputSchema={
            "type": "object",
            'properties': {
              "ticker": {
                "type": "string",
                "description": "Ticker symbol for company to search"
              },
              "query": {
                'type': 'object',
                'description': "Search queries as key-value pairs",
                'additionalProperties': {
                  "type": "string"
                }
              }
            },
            "required": ["ticker", "query"]
          }
        ),
        Tool(
          name="get_urls_content",
          description="get content from list of urls",
          inputSchema={
            "type": "object",
            'properties': {
              'urls': {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "list of urls to gather information from"
              }
            },
            "required": ["urls"]
          }
        ),
        # SEC XBRL Tools
        Tool(
          name="get_revenue_base",
          description="Get company's primary revenue from SEC filings (10-K/10-Q)",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_ebitda_margin",
          description="Calculate EBITDA margin from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_capex_pct_revenue",
          description="Get capital expenditures as percentage of revenue",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_tax_rate",
          description="Get effective tax rate from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_depreciation",
          description="Get depreciation & amortization as percentage of revenue",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_margin_breakdown",
          description="Extract gross margin, SG&A pct revenue, and R&D pct revenue from latest SEC filing. Critical for scenario DCF (separates pricing power from cost discipline) and for benchmarking R&D intensity vs peers. Banks/financials typically lack gross_profit XBRL concept; absence is expected.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_historical_fcf",
          description="Extract operating cash flow, capex, and computed free cash flow from latest SEC filing. More authoritative than Finnhub's derived FCF since it comes directly from XBRL CF statement. Returns FCF margin percentage.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_working_capital",
          description="Extract current assets, current liabilities, AR/inventory/AP, compute net working capital and NWC as percent of revenue. Negative NWC indicates supplier-financed operations (capital-efficient); positive NWC indicates cash trapped in operations.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="compare_fund_holdings",
          description="Compare the latest 13F-HR filing of a known hedge fund manager against the prior quarter. Returns position deltas: new positions opened, positions added to, positions trimmed, positions exited entirely. Use to learn what the smart money is doing — Berkshire/Ackman/Loeb/Burry/Druckenmiller etc. Knowing Berkshire just built a massive Alphabet position is decisive context. Accepts fund name (e.g. 'berkshire', 'ackman', 'loeb') or 10-digit CIK.",
          inputSchema={
            "type": "object",
            "properties": {
              "fund": {"type": "string", "description": "Fund name (berkshire, ackman, loeb, pabrai, burry, druckenmiller, einhorn, tepper, etc.) or 10-digit CIK"}
            },
            "required": ["fund"]
          }
        ),
        Tool(
          name="list_known_funds",
          description="List the known hedge fund managers and their CIKs that compare_fund_holdings can query.",
          inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
          name="get_fund_holdings",
          description="Pull the last N 13F-HR filings for a fund and return all parsed holdings (issuer, ticker, shares, market value). Use when you need the full position table, not just deltas. compare_fund_holdings is usually more useful — it surfaces what changed.",
          inputSchema={
            "type": "object",
            "properties": {
              "fund": {"type": "string", "description": "Fund name or CIK"},
              "n_filings": {"type": "integer", "description": "Number of recent 13F-HR filings to return", "default": 2}
            },
            "required": ["fund"]
          }
        ),
        Tool(
          name="extract_call_sentiment",
          description="Score sentiment over the last N quarterly earnings releases. Counts confident terms (record, strong, momentum) vs hedging terms (uncertainty, softness, headwinds), normalized per 1000 words. Computes net_score per quarter and YoY tonal shift. Signal classifier: tone_improving / stable / tone_deteriorating / tone_deteriorating_strong. CFO language shifts often precede price moves — the 1999 dot-com and 2008 housing collapses were visible in management tone 6+ months before consensus.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "quarters": {"type": "integer", "description": "Number of quarterly releases to score", "default": 4}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="track_segment_growth",
          description="Time-series analysis of per-segment revenue + operating income over the full 10-K history (typically 3 years). Computes YoY growth series, multi-year CAGR, op-margin trajectory, acceleration signal (latest YoY vs CAGR), and operating-leverage signal (op-income growth vs revenue growth). Lets the analyst see at a glance which segments are accelerating vs decelerating and which have margin compression — the cleanest read on a multi-segment business's underlying trends.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_schedule_13d_filings",
          description="Return SC 13D (activist) and SC 13G (passive) filings naming the target ticker as subject. 13D = institutional holder with >5% stake AND intent to influence (activist); 13G = passive (index funds, long-only). Returns filer name, CIK, stake percentage (where parseable), filing date, and URL. Knowing Ackman/Loeb/Icahn has built a position is decisive context; rising 13D activity = activist setup brewing.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "limit": {"type": "integer", "description": "Max filings to return", "default": 15},
              "include_passive": {"type": "boolean", "description": "Include SC 13G (passive) filings; if false only returns 13D activists", "default": True}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="diff_10k",
          description="Diff Item 1A (risk factors) or Item 7 (MD&A) across two years of 10-K filings. Returns added/removed/changed paragraphs. Use to detect regime shifts before consensus catches them — e.g. a company adding 'AI safety' or 'supply chain disruption' risks YoY, or removing risks management considers resolved. Defaults to latest 10-K vs prior; specify current_year and prior_year for non-adjacent comparisons.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "item": {"type": "string", "description": "10-K item to diff: '1A' (Risk Factors) or '7' (MD&A)", "default": "1A"},
              "current_year": {"type": "integer", "description": "Filing year of the 'current' 10-K (defaults to latest)"},
              "prior_year": {"type": "integer", "description": "Filing year of the 'prior' 10-K (defaults to prior)"},
              "max_changes": {"type": "integer", "description": "Max paragraphs to return per category", "default": 20}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_supply_chain",
          description="Extract supply-chain and competitor mentions from the company's 10-K Item 1 (Business). Returns: (1) related_companies — known mega-caps matched by name with mention counts and sample context, mapped to their tickers; (2) trigger_sentences — sentences containing 'compete with', 'rely on', 'customers include', 'partner with' phrases for category-style descriptions. Best for hardware/semis/auto names that disclose specific suppliers and customers (e.g. NVDA → TSM/Samsung/Intel/AMD/MSFT). Software/services names often use generic competitor categories — those surface in trigger_sentences.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_company_filings_history",
          description="Return the last N filings of a given form type for a company. Generalizes get_latest_filing to support YoY 10-K diffs, multi-quarter 10-Q comparisons, or tracking 8-K cadence. Returns metadata only (date, accession, URL, has_xbrl); use other extractors with specific accession numbers for content.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type (10-K, 10-Q, 8-K, DEF 14A, etc.)", "default": "10-K"},
              "n": {"type": "integer", "description": "Number of most recent filings to return", "default": 5}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_patent_filings",
          description="Patent filing counts and recent samples from Google Patents (aggregates USPTO + EPO + WIPO + national patents). Returns total assignee patent count, year-by-year publication counts for the last N years (R&D output proxy), and a small sample of recent patents with titles and snippets. Useful for tech/biotech research to validate the R&D-intensity narrative and detect patent-cliff risk. Note: patents publish ~18 months after filing; the most recent year always undercounts.",
          inputSchema={
            "type": "object",
            "properties": {
              "company_name": {"type": "string", "description": "Company name (e.g. 'Microsoft', 'Apple'). Match is by assignee field on the patent."},
              "years_back": {"type": "integer", "description": "Years of historical year-counts to return", "default": 5},
              "sample_count": {"type": "integer", "description": "Number of recent patent samples", "default": 5}
            },
            "required": ["company_name"]
          }
        ),
        Tool(
          name="get_earnings_transcripts",
          description="Pull the last N quarterly earnings releases as filed with the SEC (8-K Item 2.02 with EX-99.1 attachment). Each release contains the company-written prepared remarks, key financial metrics table, segment commentary, and CEO/CFO quotes — the SEC-authoritative equivalent of a paid transcript service's prepared remarks section. Note: analyst Q&A is NOT in 8-K filings; for Q&A use a paid transcript provider. Returns up to N quarterly releases newest-first with full text and metadata.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "max_quarters": {"type": "integer", "description": "Max quarterly releases to return", "default": 4},
              "max_chars_per_release": {"type": "integer", "description": "Truncate each release text at N chars", "default": 50000}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_mda",
          description="Extract 10-K Item 7 (Management's Discussion and Analysis) with sub-section heading detection. Covers Executive Summary, Results of Operations, Segment Results, Liquidity & Capital Resources, Critical Accounting Estimates. Use to understand management's own framing of business performance — how they explain segment trends, margin movements, and forward outlook. Pair with extract_risk_factors to capture the full qualitative 10-K context.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"},
              "max_chars": {"type": "integer", "description": "Truncate output text at N chars", "default": 80000}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_forward_signals",
          description="Scans the last N quarterly earnings releases + the latest 10-K MD&A for forward-looking language (guidance, capacity adds, capex plans, multi-year commitments, backlog, product roadmap). Returns structured excerpts ranked by category. Use to capture management's explicit forward statements before they show up in financial models. Each excerpt is also ingested into RAG for later semantic retrieval.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "lookback_quarters": {"type": "integer", "description": "Number of recent quarterly earnings releases to scan", "default": 4}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_risk_factors",
          description="Extract 10-K Item 1A (Risk Factors) full text and detect uppercase sub-section headings (e.g. 'CYBERSECURITY, DATA PRIVACY, AND PLATFORM ABUSE RISKS'). The company's own framing of bear-case risks — pre-formatted and ranked. Use to populate the bear case in IB analyst playbook §3.2 with risks the company itself has disclosed. Returns full text bounded to 80k chars, plus heading list with character offsets so consumers can navigate.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"},
              "max_chars": {"type": "integer", "description": "Truncate output text at N chars", "default": 80000}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_segment_financials",
          description="Extract per-segment revenue and operating income from the latest 10-K XBRL using the us-gaap:StatementBusinessSegmentsAxis. Returns up to 5 years of history per segment, plus the most recent YoY growth and operating margin. Critical for resolving variant-perception questions on multi-segment companies (e.g. Azure inside MSFT's Intelligent Cloud segment). Segments are SEC-defined by the company itself, so trust is highest.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_buyback_history",
          description="Extract share-repurchase (buyback) history from the latest 10-K XBRL. Returns ttm_repurchase (the most recent fiscal year's repurchases in raw USD) and annual_repurchases (up to 5 years of historical annual values). Use as the SEC-tier input to calculate_capital_returns when Finnhub /stock/financials returns empty (free-tier limitation). Concept priority: PaymentsForRepurchaseOfCommonStock, StockRepurchasedAndRetiredDuringPeriodValue, TreasuryStockAcquiredCostOfSharesAcquired, PaymentsForRepurchaseOfEquity.",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"},
              "max_years": {"type": "integer", "description": "Max years of history to return", "default": 5}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_disclosures_names",
          description="Get list of available disclosure names from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_disclosure_data",
          description="Extract specific disclosure data from SEC filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "disclosure_name": {"type": "string", "description": "Name of disclosure to extract"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker", "disclosure_name"]
          }
        ),
        Tool(
          name="get_latest_filing",
          description="Get metadata and raw access to latest SEC filing",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "form_type": {"type": "string", "description": "SEC form type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        # SEC Filing Parser Tools (8-K and Proxy Analysis)
        Tool(
          name="extract_8k_events",
          description="Extract material corporate events from 8-K filings for due diligence",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "limit": {"type": "integer", "description": "Max filings to process", "default": 10},
              "debug": {"type": "boolean", "description": "Print debug output", "default": False}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_proxy_compensation",
          description="Analyze executive compensation from proxy filings (DEF 14A)",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "debug": {"type": "boolean", "description": "Print debug output", "default": False}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_governance_data",
          description="Extract board composition and independence from proxy filings",
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock symbol"},
              "debug": {"type": "boolean", "description": "Print debug output", "default": False}
            },
            "required": ["ticker"]
          }
        ),
        # RAG: retrieval + ingest over the local vector store
        Tool(
          name="rag_search",
          description=(
            "Semantic search over the RAG corpus (historical analogues, analyst writeups, "
            "ingested 10-K sections, primers, scraped letters). Use for fuzzy questions like "
            "'find prior bubbles with similar capex profiles' or 'what did Pershing Square write "
            "about MSFT'. NOT for precise numerical lookups -- for those use the structured "
            "extractors (get_revenue_base, get_margin_breakdown, get_segment_financials, etc.). "
            "Returns top_k chunks ranked by cosine similarity, each with full chunk_text, "
            "a 300-char preview, source metadata, and a similarity score in [0, 1]. "
            "Filter by ticker or doc_type to narrow noisy queries."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "query":     {"type": "string", "description": "Natural-language query"},
              "ticker":    {"type": "string", "description": "Optional filter by ticker"},
              "doc_type":  {"type": "string", "description": "Optional filter: analogue, analyst_writeup, rule, 10K_risk_factors, 10K_mda, earnings_release, supply_chain_signals, forward_signal"},
              "top_k":     {"type": "integer", "default": 10},
              "min_score": {"type": "number", "default": 0.0}
            },
            "required": ["query"]
          }
        ),
        Tool(
          name="rag_ingest",
          description=(
            "Push a document into the RAG store so future rag_search calls can retrieve it. "
            "Chunks the text, embeds each chunk, writes to rag_chunks + rag_chunk_embeddings. "
            "Use when the analyst pastes in an external primer, scraped letter, research note, "
            "or any qualitative writeup that should be available to future semantic search. "
            "NOT a replacement for the structured ingest daemons (filings, news firehose) which "
            "ingest on their own schedule. doc_id is auto-generated by sha256 of text+metadata "
            "when omitted -- supply your own only if you need stable IDs across re-ingests."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "text":            {"type": "string"},
              "ticker":          {"type": "string"},
              "source_tool":     {"type": "string"},
              "doc_type":        {"type": "string"},
              "filing_date":     {"type": "string"},
              "section_heading": {"type": "string"},
              "doc_id":          {"type": "string", "description": "Optional explicit doc_id; auto-generated by sha256 if omitted"}
            },
            "required": ["text"]
          }
        )]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == 'search':
          result = await parent.search(args['ticker'], args['query'])
          return result
        elif name == 'get_urls_content':
          return await parent.get_urls_content(args['urls'])

        # SEC XBRL Tools
        elif name == 'get_revenue_base':
          result = await parent.get_revenue_base(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_ebitda_margin':
          result = await parent.get_ebitda_margin(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_capex_pct_revenue':
          result = await parent.get_capex_pct_revenue(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_tax_rate':
          result = await parent.get_tax_rate(args['ticker'], args.get('form_type', '10-K'))
          return result
        elif name == 'get_depreciation':
          return await parent.get_depreciation(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'get_margin_breakdown':
          return await parent.get_margin_breakdown(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'get_historical_fcf':
          return await parent.get_historical_fcf(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'get_working_capital':
          return await parent.get_working_capital(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'get_buyback_history':
          return await parent.get_buyback_history(args['ticker'], args.get('form_type', '10-K'), args.get('max_years', 5))
        elif name == 'get_segment_financials':
          return await parent.get_segment_financials(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'extract_risk_factors':
          return await parent.extract_risk_factors(args['ticker'], args.get('form_type', '10-K'), args.get('max_chars', 80000))
        elif name == 'extract_mda':
          return await parent.extract_mda(args['ticker'], args.get('form_type', '10-K'), args.get('max_chars', 80000))
        elif name == 'extract_forward_signals':
          return await parent.extract_forward_signals(args['ticker'], args.get('lookback_quarters', 4))
        elif name == 'get_earnings_transcripts':
          return await parent.get_earnings_transcripts(args['ticker'], args.get('max_quarters', 4), args.get('max_chars_per_release', 50000))
        elif name == 'get_patent_filings':
          return await parent.get_patent_filings(args['company_name'], args.get('years_back', 5), args.get('sample_count', 5))
        elif name == 'get_company_filings_history':
          return await parent.get_company_filings_history(args['ticker'], args.get('form_type', '10-K'), args.get('n', 5))
        elif name == 'get_supply_chain':
          return await parent.get_supply_chain(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'diff_10k':
          return await parent.diff_10k(args['ticker'], args.get('item', '1A'), args.get('current_year'), args.get('prior_year'), args.get('max_changes', 20))
        elif name == 'get_schedule_13d_filings':
          return await parent.get_schedule_13d_filings(args['ticker'], args.get('limit', 15), args.get('include_passive', True))
        elif name == 'track_segment_growth':
          return await parent.track_segment_growth(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'extract_call_sentiment':
          return await parent.extract_call_sentiment(args['ticker'], args.get('quarters', 4))
        elif name == 'compare_fund_holdings':
          return await parent.compare_fund_holdings(args['fund'])
        elif name == 'list_known_funds':
          return await parent.list_known_funds()
        elif name == 'get_fund_holdings':
          return await parent.get_fund_holdings(args['fund'], args.get('n_filings', 2))
        elif name == 'get_disclosures_names':
          return await parent.get_disclosures_names(args['ticker'], args.get('form_type', '10-K'))
        elif name == 'extract_disclosure_data':
          return await parent.extract_disclosure_data(args['ticker'], args['disclosure_name'], args.get('form_type', '10-K'))
        elif name == 'get_latest_filing':
          result = await parent.get_latest_filing(args['ticker'], args.get('form_type', '10-K'))
          return result

        # SEC Filing Parser Tools
        elif name == 'extract_8k_events':
          return await parent.extract_8k_events(args['ticker'], args.get('limit', 10), args.get('debug', False))
        elif name == 'extract_proxy_compensation':
          return await parent.extract_proxy_compensation(args['ticker'], args.get('debug', False))
        elif name == 'extract_governance_data':
          return await parent.extract_governance_data(args['ticker'], args.get('debug', False))

        # RAG Tools
        elif name == 'rag_search':
          return await parent.rag_search_tool(args)
        elif name == 'rag_ingest':
          return await parent.rag_ingest_tool(args)
        else:
          return [TextContent(
            type='text',
            text=f'Unknown tool: {name}'
          )]
      except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return [TextContent(
          type='text',
          text=f'Failed to execute {name}: {str(e)}'
        )]


  async def search(self, ticker: str, query: Dict) -> List[TextContent]:
    """Run multiple queries concurrently against the local SearXNG instance."""
    queries = list(query.values())
    tasks = [searxng_search(f"{ticker} {q}", max_results=5) for q in queries]
    result_list = await asyncio.gather(*tasks)
    search_results: List[Dict] = []
    for sub in result_list:
      search_results.extend(sub)

    print(f"  [Validate Search] {len(queries)} queries -> {len(search_results)} total URLs",
          file=sys.stderr, flush=True)

    return [TextContent(
      type='text',
      text=safe_json_dumps({
        'ticker': ticker,
        'search_result': search_results,
      })
    )]


  async def get_urls_content(self, urls: List[str]) -> List[TextContent]:
    """Scrape URLs concurrently via Trafilatura (Crawl4AI fallback), cached by URL."""
    results = await scrape_urls(urls, cache=self.cache)

    successes = sum(1 for r in results if r.get('success'))
    print(f"  [Validate Scrape] {successes}/{len(urls)} URLs scraped successfully",
          file=sys.stderr, flush=True)

    return [TextContent(
      type="text",
      text=safe_json_dumps({
        "results": results
      })
    )]

  # SEC XBRL Tools
  async def get_revenue_base(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_revenue_base, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_ebitda_margin(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_ebitda_margin, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_capex_pct_revenue(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_capex_pct_revenue, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_tax_rate(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_tax_rate, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_depreciation(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_depreciation, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_margin_breakdown(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_margin_breakdown, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_historical_fcf(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_historical_fcf, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_working_capital(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_working_capital, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_buyback_history(self, ticker: str, form_type: str = '10-K', max_years: int = 5) -> List[TextContent]:
    result = await asyncio.to_thread(get_buyback_history, ticker, form_type, max_years)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_segment_financials(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_segment_financials, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_risk_factors(self, ticker: str, form_type: str = '10-K', max_chars: int = 80000) -> List[TextContent]:
    result = await asyncio.to_thread(extract_risk_factors, ticker, form_type, max_chars)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_mda(self, ticker: str, form_type: str = '10-K', max_chars: int = 80000) -> List[TextContent]:
    result = await asyncio.to_thread(extract_mda, ticker, form_type, max_chars)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_forward_signals(self, ticker: str, lookback_quarters: int = 4) -> List[TextContent]:
    result = await asyncio.to_thread(extract_forward_signals, ticker, lookback_quarters)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_earnings_transcripts(self, ticker: str, max_quarters: int = 4, max_chars_per_release: int = 50000) -> List[TextContent]:
    result = await asyncio.to_thread(get_earnings_releases, ticker, max_quarters, max_chars_per_release)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_patent_filings(self, company_name: str, years_back: int = 5, sample_count: int = 5) -> List[TextContent]:
    result = await asyncio.to_thread(get_patent_filings, company_name, years_back, sample_count)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_company_filings_history(self, ticker: str, form_type: str = '10-K', n: int = 5) -> List[TextContent]:
    result = await asyncio.to_thread(get_company_filings_history, ticker, form_type, n)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_supply_chain(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_supply_chain, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def diff_10k(self, ticker: str, item: str = '1A', current_year=None, prior_year=None, max_changes: int = 20) -> List[TextContent]:
    result = await asyncio.to_thread(diff_10k, ticker, item, current_year, prior_year, max_changes)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_schedule_13d_filings(self, ticker: str, limit: int = 15, include_passive: bool = True) -> List[TextContent]:
    result = await asyncio.to_thread(get_schedule_13d_filings, ticker, limit, include_passive)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def track_segment_growth(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(track_segment_growth, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_call_sentiment(self, ticker: str, quarters: int = 4) -> List[TextContent]:
    result = await asyncio.to_thread(extract_call_sentiment, ticker, quarters)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def compare_fund_holdings(self, fund: str) -> List[TextContent]:
    result = await asyncio.to_thread(compare_fund_holdings, fund)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def list_known_funds(self) -> List[TextContent]:
    result = await asyncio.to_thread(list_known_funds)
    return [TextContent(type="text", text=safe_json_dumps({'funds': result, 'count': len(result)}))]

  async def get_fund_holdings(self, fund: str, n_filings: int = 2) -> List[TextContent]:
    result = await asyncio.to_thread(get_fund_holdings, fund, n_filings)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_disclosures_names(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_disclosures_names, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_disclosure_data(self, ticker: str, disclosure_name: str, form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(extract_disclosure_data, ticker, disclosure_name, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_latest_filing(self, ticker: str, form_type: str = '10-K') -> List[TextContent]:
    # Note: This returns filing metadata, not the full filing object (which isn't JSON serializable)
    result = await asyncio.to_thread(get_latest_filing, ticker, form_type)
    if result:
      # Convert to JSON-serializable format
      json_result = {
        'ticker': ticker,
        'form_type': form_type,
        'filing_date': str(result.get('filing_date')),
        'url': result.get('url'),
        'accession_number': result.get('accession_number'),
        'has_xbrl_data': result.get('xbrl_data') is not None,
        'success': True
      }
    else:
      json_result = {
        'ticker': ticker,
        'form_type': form_type,
        'error': 'No filing found',
        'success': False
      }
    return [TextContent(type="text", text=safe_json_dumps(json_result))]

  # SEC Filing Parser Tools
  async def extract_8k_events(self, ticker: str, limit: int = 10, debug: bool = False) -> List[TextContent]:
    if filing_parser is None:
      return [TextContent(type="text", text=safe_json_dumps({"error": "Filing parser not available", "success": False}))]
    result = await asyncio.to_thread(filing_parser.extract_8k_events, ticker, limit, debug)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_proxy_compensation(self, ticker: str, debug: bool = False) -> List[TextContent]:
    if filing_parser is None:
      return [TextContent(type="text", text=safe_json_dumps({"error": "Filing parser not available", "success": False}))]
    # SECFilingParser.extract_proxy_compensation takes only `ticker`. The
    # `debug` arg is still on the MCP schema for symmetry with the other
    # parser tools, but it's a no-op here until the underlying method
    # learns to honor it.
    result = await asyncio.to_thread(filing_parser.extract_proxy_compensation, ticker)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_governance_data(self, ticker: str, debug: bool = False) -> List[TextContent]:
    if filing_parser is None:
      return [TextContent(type="text", text=safe_json_dumps({"error": "Filing parser not available", "success": False}))]
    result = await asyncio.to_thread(filing_parser.extract_governance_data, ticker, debug)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  # RAG Tools — lazy imports keep the web_search server boot cheap when
  # sentence-transformers / sqlite-vec aren't actually needed this session.
  async def rag_search_tool(self, args: Dict[str, Any]) -> List[TextContent]:
    from agent.rag.search import rag_search
    result = await asyncio.to_thread(
      rag_search,
      args['query'],
      args.get('ticker'),
      args.get('doc_type'),
      args.get('top_k', 10),
      args.get('min_score', 0.0),
    )
    return [TextContent(type='text', text=safe_json_dumps(result))]

  async def rag_ingest_tool(self, args: Dict[str, Any]) -> List[TextContent]:
    from agent.rag.ingest import ingest_document
    metadata = {
      k: args.get(k)
      for k in ('ticker', 'source_tool', 'doc_type', 'filing_date', 'section_heading')
      if args.get(k)
    }
    result = await asyncio.to_thread(
      ingest_document,
      args['text'],
      metadata,
      args.get('doc_id'),
    )
    return [TextContent(type='text', text=safe_json_dumps(result))]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(
          read_stream,
          write_stream,
          self.server.create_initialization_options(),
        )
        print("Successfully created web_client process", file=sys.stderr, flush=True)
    except Exception as e:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.web_search_server.web_search server", file=sys.stderr)
    sys.exit(1)

  system_args = sys.argv[1]

  if system_args == "server":
    print("Starting web_client process", file=sys.stderr, flush=True)
    try:
      server = WebSearchServer()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {system_args}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.web_search_server.web_search server", file=sys.stderr)
    sys.exit(1)
