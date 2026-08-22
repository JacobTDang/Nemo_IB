from typing import Any, Dict, List
import asyncio
import json
import os
from datetime import date, datetime
from tools.web_search_server.searxng_client import searxng_search
from tools.web_search_server.scraper import scrape_urls
from agent.cache import Session_Cache
from tools.web_search_server.forward_metrics import (
  get_contracted_revenue,
  get_geographic_revenue,
  get_public_float,
)
from tools.web_search_server.earnings_quality import (
  get_accruals_quality,
  get_operating_leases,
  get_working_capital_trends,
)
from tools.web_search_server.peers import find_peers_by_sic, get_sic_code
from tools.web_search_server.debt_maturity import get_debt_maturity_schedule
from tools.web_search_server.foreign_issuer import (
  get_annual_revenue,
  get_foreign_filer_profile,
)
from tools.web_search_server.sbc import get_sbc_series
from tools.web_search_server.dilution import (
  get_share_count_series,
  get_shelf_activity,
)
from tools.web_search_server.sec_utils import (
  extract_litigation,
  extract_customer_concentration,
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
from tools.web_search_server.guidance import extract_guidance
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


def _build_all_tools() -> List[Tool]:
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
          description=(
            "Company's primary revenue from SEC filings.\n\n"
            "Reports 'currency' -- do not assume dollars. Pass form_type='20-F' "
            "('40-F' for Canada) for a foreign private issuer such as TSM, "
            "ASML, SAP, NVO or BABA; those route to the guarded reader that "
            "takes the consolidated undimensioned fact, because an IFRS filer "
            "also tags constant-currency and pro-forma variants that look like "
            "revenue. With the default 10-K an ADR gets an explanation of the "
            "form mismatch rather than an empty result. Prefer "
            "get_annual_revenue when you do not know which form applies."
          ),
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
          description="Patent filing counts and recent samples from Google Patents (aggregates USPTO + EPO + WIPO + national patents). Returns total assignee patent count, year-by-year publication counts for the last N years (R&D output proxy), and a small sample of recent patents with titles and snippets. Useful for tech/biotech research to validate the R&D-intensity narrative and detect patent-cliff risk. Note: patents publish ~18 months after filing; the most recent year always undercounts. Google throttles this endpoint, so check `failed_years` before reading a trend — any year listed there is missing from `year_counts`, not zero.",
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
        ),
        Tool(
          name="get_share_count_series",
          description=(
            "Shares outstanding across recent filings, with per-class breakdown and "
            "the percentage change over the window. Sourced from the cover-page tag "
            "dei:EntityCommonStockSharesOutstanding in each 10-Q or 10-K.\n\n"
            "Use this before trusting any per-share metric. A company can grow EPS "
            "purely by shrinking the denominator, or erode it by growing the "
            "denominator, and neither shows up in the income statement.\n\n"
            "Multi-class companies (GOOGL, BRK, META) report each class separately; "
            "'total' sums them and 'by_class' shows the split. Check 'classes_found' "
            "-- if a company you know has multiple classes reports only one, the "
            "total is understated. 'direction' is dilution / buyback / flat / "
            "insufficient_history."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer", "description": "How many filings to walk back", "default": 8},
              "form":   {"type": "string", "description": "10-Q for quarterly granularity, 10-K for annual", "default": "10-Q"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_shelf_activity",
          description=(
            "Shelf registrations (S-3) and takedowns (424B5) filed in the lookback "
            "window.\n\n"
            "This is the mechanism behind dilution, where get_share_count_series is "
            "the effect. An effective S-3 means the company is authorised to sell "
            "shares; each 424B5 is an actual sale off that shelf. A rising share "
            "count plus repeated 424B5 filings means dilution is ongoing rather "
            "than finished -- the distinction matters for whether you size into it.\n\n"
            "Cash-generative megacaps typically show nothing here. Serial issuers "
            "show many takedowns."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker":        {"type": "string", "description": "Ticker symbol"},
              "lookback_days": {"type": "integer", "description": "Window in days", "default": 730}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_sbc_series",
          description=(
            "Stock-based compensation over recent annual filings, with its share "
            "of revenue and of operating cash flow.\n\n"
            "SBC is the largest single line between GAAP earnings and the adjusted "
            "figures companies prefer to be judged on, and it is the other engine "
            "of dilution alongside shelf issuance. A company whose SBC is a large "
            "and rising share of revenue is paying employees with your ownership.\n\n"
            "Returns the consolidated figure per filing, not a sum of award-type "
            "breakdowns -- filers tag dozens of component facts and adding them "
            "would report several times the real expense. 'concept_used' names "
            "which XBRL tag the figure came from."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer", "description": "How many filings to walk back", "default": 5},
              "form":   {"type": "string", "description": "10-K for annual, 10-Q for quarterly", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_debt_maturity_schedule",
          description=(
            "Principal coming due by year from the long-term debt footnote, plus "
            "the share due within twelve months.\n\n"
            "Leverage alone does not tell you much. 3x turns maturing next year is "
            "a refinancing problem; the same 3x maturing in 2031 is not. Use this "
            "alongside calculate_credit_profile before drawing any conclusion about "
            "balance-sheet risk.\n\n"
            "IMPORTANT: coverage is genuinely partial across filers. Check the "
            "'coverage' field -- 'full' means all six buckets were tagged, "
            "'partial' means some, and 'not_covered' means this filer does not tag "
            "the split in XBRL. not_covered does NOT mean no debt matures: Ford "
            "carries enormous debt and tags none of these concepts. Never read an "
            "absent schedule as an absent obligation."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "form":   {"type": "string", "description": "Filing type", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_litigation",
          description=(
            "Item 3, Legal Proceedings, from the latest annual or quarterly filing.\n\n"
            "Material legal exposure had no coverage at all. Use when sizing a "
            "position in a company facing regulatory action, patent disputes, or "
            "class actions.\n\n"
            "NOTE: most large filers cross-reference a contingencies note rather "
            "than restating detail in Item 3, so a short result is normal. The "
            "'cross_referenced_only' flag is true in that case, and the substance "
            "lives in the notes to the financial statements -- follow up there "
            "rather than concluding there is no litigation."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker":    {"type": "string", "description": "Ticker symbol"},
              "form_type": {"type": "string", "description": "10-K or 10-Q", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_customer_concentration",
          description=(
            "Major-customer disclosure: who accounts for a material share of "
            "revenue, and how much.\n\n"
            "Often the entire thesis. A supplier deriving 22% of revenue from one "
            "buyer has a different risk profile from a diversified one, and it "
            "does not appear anywhere in the financial statements.\n\n"
            "Read the two flags together. 'has_concentration' true with entries in "
            "'named_customers' means real concentration was disclosed. "
            "'explicitly_none' true means the filer stated that no customer crosses "
            "the threshold, which is a genuine finding of LOW concentration -- not "
            "missing data. Both false means nothing was found either way. Most "
            "filers describe the customer without naming it, so a null name with a "
            "percentage is the common case."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker":    {"type": "string", "description": "Ticker symbol"},
              "form_type": {"type": "string", "description": "10-K or 10-Q", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="find_peers_by_sic",
          description=(
            "Listed companies sharing this filer's SIC classification, for use as "
            "a starting comp set.\n\n"
            "comparable_company_analysis requires you to supply peers, which makes "
            "the comps depend on already knowing the answer. This derives them from "
            "the filings instead.\n\n"
            "Two caveats. SIC groups filers by declared classification rather than "
            "competitive overlap, so treat the output as a candidate list to prune, "
            "not a finished comp table. And coverage is partial: an SIC query "
            "returns deregistered and private filers with no listed ticker, so "
            "compare 'peer_count' against 'filers_matched' -- 'unresolved_count' is "
            "the gap."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer", "description": "Max filers to request from EDGAR", "default": 20}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_sic_code",
          description=(
            "The filer's SIC classification code and industry description from "
            "EDGAR. Useful on its own to confirm how a company classifies itself, "
            "which is occasionally not how the market thinks of it."
          ),
          inputSchema={
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol"}},
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_contracted_revenue",
          description=(
            "Revenue already under contract but not yet recognised: remaining "
            "performance obligation (RPO) and deferred revenue.\n\n"
            "RPO is the strongest forward number an enterprise filer publishes "
            "-- signed business that has not hit the income statement yet. It "
            "leads reported revenue by quarters, so RPO growth stalling while "
            "revenue still looks healthy is an early warning that the income "
            "statement cannot show you.\n\n"
            "Reports the consolidated total, not a sum of the customer-type and "
            "timing breakdowns filers publish alongside it. An empty 'rpo' with "
            "deferred revenue present is normal and not a failure -- most "
            "non-subscription businesses never disclose RPO."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer", "description": "Annual filings to walk back", "default": 3}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_geographic_revenue",
          description=(
            "Revenue by geography with each region's share of the disclosed "
            "total, and several years of history from a single filing.\n\n"
            "Business segments tell you what a company sells; this tells you "
            "where, which is where China exposure, tariff risk and FX "
            "translation actually live. NVDA books roughly 20% of revenue in "
            "Taiwan and 9% in China including Hong Kong -- neither is visible "
            "in segment reporting.\n\n"
            "Region names come from whatever the filer tagged, mixing standard "
            "country codes with company-specific groupings, so check "
            "'regions_found'. Percentages are of the disclosed geographic total, "
            "which can differ from consolidated revenue when part is grouped "
            "under 'other'.\n\n"
            "'members_overlap' true means the filer tagged nested regions on "
            "one axis -- SAP tags EMEA, EMEA-excluding-Germany and Germany "
            "together -- so percentages are of consolidated revenue and do NOT "
            "sum to 100. Read regions individually and never add them.\n\n"
            "For an ADR pass form='20-F' ('40-F' for Canada); with the default "
            "10-K the result explains the mismatch instead of reporting no "
            "geographic disclosure. IFRS filers are covered -- TSM splits 74% "
            "United States, 9% China, 8% Taiwan."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "form":   {"type": "string",
                         "description": "10-K, or 20-F/40-F for a foreign private issuer",
                         "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_public_float",
          description=(
            "Market value of shares held by non-affiliates, from the 10-K cover "
            "page.\n\n"
            "Different from shares outstanding, and the difference is the point: "
            "float excludes insider and affiliate holdings, so it is what "
            "actually trades. For founder-controlled companies the gap is large, "
            "and float is what governs volatility, liquidity and squeeze risk.\n\n"
            "Pair with get_share_count_series -- a wide gap between float and "
            "total shares means less stock changes hands than the share count "
            "implies. Measured at the filer's second-quarter close, so it lags; "
            "'as_of' gives the measurement date rather than leaving you to "
            "assume it is current.\n\n"
            "Foreign private issuers file 20-F; pass form='20-F' for one. With "
            "the default the result says so rather than implying no float was "
            "disclosed."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "form":   {"type": "string",
                         "description": "10-K, or 20-F/40-F for a foreign private issuer",
                         "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_accruals_quality",
          description=(
            "Net income against operating cash flow, the accrual ratio, and "
            "the multi-period trend. The single best earnings-quality "
            "screen.\n\n"
            "Earnings are an opinion; cash is a fact. When net income rises "
            "while operating cash flow does not, the gap is being filled by "
            "accruals -- revenue booked before the cash arrives, costs "
            "deferred, reserves released. That divergence is the classic "
            "pre-blowup signature and it usually appears several quarters "
            "before the restatement or the guidance cut. Reach for this "
            "before sizing any long on a name whose earnings have been "
            "beating.\n\n"
            "'accrual_ratio_pct' is (net income - operating cash flow) / total "
            "assets. Negative means cash flow more than covers earnings, which "
            "is what a healthy filer looks like; above 5% is high. "
            "'divergence' is true only for the specific shape -- earnings up "
            "while operating cash flow falls.\n\n"
            "A null 'accrual_ratio_pct' means the filer did not tag total "
            "assets, and success:false means net income or operating cash flow "
            "is untagged. Neither means accruals were zero. Assets are the "
            "period-end balance rather than an average, so a large mid-year "
            "acquisition flatters the ratio."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer",
                         "description": "Filings to walk; each carries 2-3 comparative years",
                         "default": 2},
              "form":   {"type": "string", "description": "10-K or 10-Q", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_working_capital_trends",
          description=(
            "Days sales outstanding, days inventory, days payable and the cash "
            "conversion cycle, per period, with growth gaps against "
            "revenue.\n\n"
            "Where an earnings-quality problem shows up on the balance sheet. "
            "Receivables growing faster than revenue means the company is "
            "stuffing the channel or not getting paid: "
            "'receivables_vs_revenue_gap_pct' above roughly 10 points is the "
            "flag. Inventory building faster than sales is demand "
            "deteriorating ahead of the reported numbers, which is the "
            "earliest read available on a consumer or hardware name.\n\n"
            "DSO is receivables per revenue-day; DIO and DPO are inventory and "
            "payables per cost-of-revenue-day, computed on each period's "
            "actual span so 10-Q figures and 52/53-week years stay comparable. "
            "A negative cash conversion cycle means suppliers finance the "
            "business -- COST and WMT both run near zero.\n\n"
            "A null DIO means the filer tags no inventory at all, which is the "
            "correct answer for software and services and is not the same as "
            "zero days of stock. Balances are period-end, so a seasonal "
            "year-end distorts the level and the growth gaps are the more "
            "reliable signal."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer",
                         "description": "Filings to walk; each carries 2-3 comparative years",
                         "default": 2},
              "form":   {"type": "string", "description": "10-K or 10-Q", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_operating_leases",
          description=(
            "Operating lease obligations and the year-by-year payment "
            "ladder.\n\n"
            "ASC 842 put the liability on the balance sheet but left the "
            "maturity profile in the footnote, so a lease book larger than the "
            "bond stack stays invisible beside get_debt_maturity_schedule. For "
            "retailers, restaurants and anything asset-light this is the "
            "bigger fixed obligation and it does not appear in leverage ratios "
            "built from debt alone. Call it whenever the debt schedule looks "
            "reassuringly light.\n\n"
            "'lease_liability' is the discounted balance-sheet figure. "
            "'maturity_schedule' and 'undiscounted_payments_total' are the "
            "contractual cash payments, and the difference between them is "
            "'imputed_interest'.\n\n"
            "A null bucket means the filer did not tag that year; 0.0 means it "
            "disclosed nothing due, and the two are never merged. A missing "
            "current portion stays null rather than being backed out of the "
            "total, because a derived figure is not a disclosure. Coverage is "
            "'full' only when the liability, the right-of-use asset and all "
            "six buckets are tagged -- check 'buckets_found'."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "form":   {"type": "string", "description": "10-K or 10-Q", "default": "10-K"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="extract_guidance",
          description=(
            "Company-issued forward guidance, verbatim, from 8-K Item 2.02 "
            "earnings-release exhibits, with the filing each statement came "
            "from.\n\n"
            "This is what management SAID it would do -- the raw material for "
            "judging management credibility, which nothing else here "
            "captured. extract_forward_signals matches forward-looking "
            "language but truncates at the first period, so '$1.5 billion' "
            "becomes '$1'.\n\n"
            "IT DOES NOT SAY WHETHER GUIDANCE WAS MET, and will not be made "
            "to. Grading a guide needs the actual on the same basis for the "
            "same fiscal period; GAAP and non-GAAP EPS can differ several-"
            "fold, and the available actuals are labelled by calendar "
            "quarter-end. Read the statements and judge them yourself.\n\n"
            "Guidance rendered only in tables is REFUSED rather than parsed. "
            "Flattening a filing's HTML interleaves columns: Salesforce's EPS "
            "table renders as '$1.74 - $7.93', which is the quarterly low "
            "bolted to the full-year high, and Intel's 'Gross margin 41.0% "
            "42.0%' is the GAAP and non-GAAP columns rather than a range.\n\n"
            "'no_guidance_reason' separates the three ways of finding "
            "nothing, which are not the same finding: "
            "'no_earnings_releases_found' (could not look), "
            "'release_text_unavailable' (the exhibit would not extract), and "
            "'no_guidance_language_found' (looked, found none). When the last "
            "is paired with 'guidance_may_be_table_only': true, the company "
            "does guide -- in a table this tool declined to read.\n\n"
            "Measured at these defaults on 24 large caps: 3 had no usable "
            "source, 21 had readable text, and 17 of those yielded prose "
            "guidance (145 statements). The 4 empties were Apple, Costco and "
            "Microsoft, which genuinely do not guide in the release, and "
            "Walmart, which does but only in tables and is flagged. A hand "
            "audit of all 145 found 1 that is not guidance and 2 with a wrong "
            "period label.\n\n"
            "'quarters' changes what you find, and not gently: Coca-Cola's "
            "most recent release is truncated before its outlook section, so "
            "quarters=1 returns nothing for a company that guides every "
            "quarter while quarters=4 returns 25 statements. Never read an "
            "empty result -- least of all a narrow one -- as 'management gave "
            "no guidance'.\n\n"
            "Each statement carries 'caveats': 'period_inherited_from_section"
            "_lead_in' (the period came from an outlook heading, not the "
            "sentence), 'period_may_not_be_the_guided_period', "
            "'no_period_identified', 'contains_past_tense_reporting'. The "
            "verbatim 'text' is always there to check them against."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker":   {"type": "string", "description": "Ticker symbol"},
              "quarters": {"type": "integer", "description": "How many recent earnings releases to scan", "default": 4}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_foreign_filer_profile",
          description=(
            "Which SEC forms a company actually files, under which accounting "
            "standard, in which currency.\n\n"
            "CALL THIS FIRST for any ADR or non-US company -- TSM, ASML, BABA, "
            "SAP, NVO, BCE, TM, SHOP. Foreign private issuers file 20-F (or "
            "40-F for Canada) instead of 10-K, and every other SEC tool here "
            "defaults to 10-K. Without this you cannot tell \"foreign issuer, "
            "use its form\" from \"nothing to report\", and the second reading "
            "is how an ADR gets a clean bill of health it never earned.\n\n"
            "'is_foreign_private_issuer' is decided by the MOST RECENT annual "
            "form, not by history: Shopify filed 40-F through 2024 and files "
            "10-K now, so it reads false. null means no annual filing was found "
            "at all, which is a lookup failure rather than an answer.\n\n"
            "'taxonomy' is read from the filing, not guessed from the form, "
            "because the two disagree. TSM, SAP and NVO file 20-F under IFRS "
            "(ifrs-full concepts); ASML and BABA file the same form under US "
            "GAAP. Picking concepts from the form would miss two of those "
            "five.\n\n"
            "'interim_xbrl' false is the field to act on. A foreign issuer "
            "reports interim results on 6-K, and 6-K carries NO XBRL at all -- "
            "verified across TSM, ASML and BABA. No quarterly tagged figure "
            "exists for these filers anywhere, so any 10-Q-based tool "
            "(get_share_count_series, quarterly accruals) cannot be served for "
            "them at all. Use the annual form.\n\n"
            "'reporting_currency' is never assumed to be USD: TSM reports TWD, "
            "SAP and ASML EUR, NVO DKK, BABA CNY. When "
            "'usd_convenience_translation' is true the filer also tagged a "
            "dollar figure for the latest year at its own rate -- their number, "
            "not a live conversion. A null currency means the filing's units "
            "were untagged or opaque, not that it reports in dollars."
          ),
          inputSchema={
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol"}},
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_annual_revenue",
          description=(
            "Consolidated annual revenue in the currency it was actually "
            "reported in, for domestic and foreign filers alike.\n\n"
            "Use this instead of get_revenue_base whenever the company might "
            "not be American, or when you do not know. It resolves the annual "
            "form itself (10-K, 20-F or 40-F) and tries both the IFRS and US "
            "GAAP concept chains, so an ADR does not cost a second call to "
            "discover which form to ask for.\n\n"
            "ALWAYS read 'currency' before using 'latest_revenue'. TSM's FY2025 "
            "revenue is 3,809,054,300,000 -- New Taiwan dollars, about $121bn. "
            "Read as dollars it overstates the company roughly 31x, and nothing "
            "about the number looks wrong. 'latest_revenue_usd' is populated "
            "only when the filer itself tagged a dollar convenience translation "
            "(TSM and BABA do) or reports in dollars already; "
            "'usd_is_filer_translation' says which. It is never computed here, "
            "and null does NOT mean the company is small.\n\n"
            "A concept only answers if it answers in the newest annual filing. "
            "TSM stopped tagging ifrs-full:Revenue undimensioned in 2026, and "
            "accepting the older filings' rows reported FY2024 as current -- a "
            "year stale and 24% low. success:false with 'concepts_tried' means "
            "this filer tags revenue under an element in neither chain, or only "
            "with dimensions."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Ticker symbol"},
              "limit":  {"type": "integer",
                         "description": "Annual filings to walk for history",
                         "default": 3}
            },
            "required": ["ticker"]
          }
        )]


_ALL_TOOLS = _build_all_tools()


# Capability gating: never advertise a tool this deployment cannot perform.
# `search` needs SearXNG and the rag_* pair needs the RAG stack. rag_search at
# least raises when it is missing; `search` returns an empty result list with
# no error, so an absent SearXNG is indistinguishable from "nothing matched".

def _searxng_reachable() -> bool:
  import socket
  from urllib.parse import urlparse
  url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
  parsed = urlparse(url)
  host = parsed.hostname or "localhost"
  port = parsed.port or (443 if parsed.scheme == "https" else 80)
  try:
    with socket.create_connection((host, port), timeout=1.5):
      return True
  except OSError:
    return False


def _rag_available() -> bool:
  import importlib.util
  return importlib.util.find_spec("agent.rag") is not None


_GATED_TOOLS = {
  "search": lambda: _searxng_reachable(),
  "rag_search": lambda: _rag_available(),
  "rag_ingest": lambda: _rag_available(),
}


def _tool_is_available(name: str) -> bool:
  check = _GATED_TOOLS.get(name)
  return True if check is None else check()


def available_tool_names():
  """Names this deployment can actually serve. Used by list_tools and tests."""
  return [t.name for t in _ALL_TOOLS if _tool_is_available(t.name)]


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
      return [t for t in _ALL_TOOLS if _tool_is_available(t.name)]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
      try:
        if name == 'search':
          result = await parent.search(args['ticker'], args['query'])
          return result
        elif name == 'get_urls_content':
          return await parent.get_urls_content(args['urls'])

        elif name == 'get_contracted_revenue':
          return await parent.get_contracted_revenue(args['ticker'], args.get('limit', 3))
        elif name == 'get_geographic_revenue':
          return await parent.get_geographic_revenue(
            args['ticker'], args.get('form', '10-K'))
        elif name == 'get_public_float':
          return await parent.get_public_float(
            args['ticker'], args.get('form', '10-K'))
        elif name == 'get_foreign_filer_profile':
          return await parent.get_foreign_filer_profile(args['ticker'])
        elif name == 'get_annual_revenue':
          return await parent.get_annual_revenue(
            args['ticker'], args.get('limit', 3))
        elif name == 'get_accruals_quality':
          return await parent.get_accruals_quality(
            args['ticker'], args.get('limit', 2), args.get('form', '10-K'))
        elif name == 'get_working_capital_trends':
          return await parent.get_working_capital_trends(
            args['ticker'], args.get('limit', 2), args.get('form', '10-K'))
        elif name == 'get_operating_leases':
          return await parent.get_operating_leases(
            args['ticker'], args.get('form', '10-K'))
        elif name == 'find_peers_by_sic':
          return await parent.find_peers_by_sic(args['ticker'], args.get('limit', 20))
        elif name == 'get_sic_code':
          return await parent.get_sic_code(args['ticker'])
        elif name == 'extract_litigation':
          return await parent.extract_litigation(
            args['ticker'], args.get('form_type', '10-K'))
        elif name == 'extract_customer_concentration':
          return await parent.extract_customer_concentration(
            args['ticker'], args.get('form_type', '10-K'))
        elif name == 'extract_guidance':
          return await parent.extract_guidance(
            args['ticker'], args.get('quarters', 4))
        elif name == 'get_debt_maturity_schedule':
          return await parent.get_debt_maturity_schedule(
            args['ticker'], args.get('form', '10-K'))
        elif name == 'get_sbc_series':
          return await parent.get_sbc_series(
            args['ticker'], args.get('limit', 5), args.get('form', '10-K'))

        # Dilution / share count
        elif name == 'get_share_count_series':
          return await parent.get_share_count_series(
            args['ticker'], args.get('limit', 8), args.get('form', '10-Q'))
        elif name == 'get_shelf_activity':
          return await parent.get_shelf_activity(
            args['ticker'], args.get('lookback_days', 730))

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
  async def get_contracted_revenue(self, ticker: str, limit: int = 3) -> List[TextContent]:
    result = await asyncio.to_thread(get_contracted_revenue, ticker, limit)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_geographic_revenue(self, ticker: str,
                                   form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_geographic_revenue, ticker, 1, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_public_float(self, ticker: str,
                             form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_public_float, ticker, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_foreign_filer_profile(self, ticker: str) -> List[TextContent]:
    result = await asyncio.to_thread(get_foreign_filer_profile, ticker)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_annual_revenue(self, ticker: str,
                               limit: int = 3) -> List[TextContent]:
    result = await asyncio.to_thread(get_annual_revenue, ticker, limit)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_accruals_quality(self, ticker: str, limit: int = 2,
                                 form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_accruals_quality, ticker, limit, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_working_capital_trends(self, ticker: str, limit: int = 2,
                                       form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(
      get_working_capital_trends, ticker, limit, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_operating_leases(self, ticker: str,
                                 form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_operating_leases, ticker, 1, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def find_peers_by_sic(self, ticker: str, limit: int = 20) -> List[TextContent]:
    result = await asyncio.to_thread(find_peers_by_sic, ticker, limit)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_sic_code(self, ticker: str) -> List[TextContent]:
    result = await asyncio.to_thread(get_sic_code, ticker)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_litigation(self, ticker: str,
                               form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(extract_litigation, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def extract_customer_concentration(self, ticker: str,
                                           form_type: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(extract_customer_concentration, ticker, form_type)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_debt_maturity_schedule(self, ticker: str,
                                       form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_debt_maturity_schedule, ticker, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_sbc_series(self, ticker: str, limit: int = 5,
                          form: str = '10-K') -> List[TextContent]:
    result = await asyncio.to_thread(get_sbc_series, ticker, limit, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_share_count_series(self, ticker: str, limit: int = 8,
                                   form: str = '10-Q') -> List[TextContent]:
    result = await asyncio.to_thread(get_share_count_series, ticker, limit, form)
    return [TextContent(type="text", text=safe_json_dumps(result))]

  async def get_shelf_activity(self, ticker: str,
                               lookback_days: int = 730) -> List[TextContent]:
    result = await asyncio.to_thread(get_shelf_activity, ticker, lookback_days)
    return [TextContent(type="text", text=safe_json_dumps(result))]

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

  async def extract_guidance(self, ticker: str, quarters: int = 4) -> List[TextContent]:
    result = await asyncio.to_thread(extract_guidance, ticker, quarters)
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
    # Block briefly for the embedder warmup so the first rag_search call
    # doesn't pay the ~10s sentence-transformers cold-start cost. The
    # daemon thread started in __init__ has been loading torch + the model
    # while the MCP handshake was being prepared, so this wait usually
    # finishes inside a few seconds. Cap at 15s — if the load is still
    # not done, fall through and let the first rag_search caller take
    # the remainder of the hit.
    try:
      from agent.rag.embedder import await_loaded
      await asyncio.to_thread(await_loaded, 15.0)
    except Exception:
      pass

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
    print("Usage: python -m tools.web_search_server.web_search [server|http]", file=sys.stderr)
    sys.exit(1)

  system_args = sys.argv[1]

  if system_args == "http":
    # Streamable HTTP, for a host a client connects to rather than one
    # that spawns it. stdio stays the default for local use.
    from tools.mcp_http import run_http
    print("Starting web_client over streamable HTTP", file=sys.stderr, flush=True)
    run_http(WebSearchServer().server)

  elif system_args == "server":
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
