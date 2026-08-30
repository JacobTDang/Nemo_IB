"""Call every shipped tool once and hold it to its own contract.

`test_tool_health.py` was the only harness that did this, and it is deleted
in favour of this file. It listed tools by hand -- 40 of the 96 the image
serves, with the whole `altdata` server absent -- and defined no test
function, so nothing ever ran it. Both failures come from one root: a list
written once drifts from the registry, and a script nobody invokes cannot
tell you it has drifted. Three of the six failures it did report were its own
arguments rather than the tools'.

So the roster here comes from each server's own registry. A tool added
tomorrow is covered tomorrow, and `test_the_roster_matches_the_registry` fails
if this file starts describing a set that no longer exists.

What is asserted is the envelope, not the answer. A tool that refuses is
passing, provided it says why: this suite has spent a long time on refusals
that dropped their reason, or carried a live number beside one, and that is
the property worth pinning here. The numbers themselves belong in the tool's
own tests, against fixtures, not against whatever the vendor says today.
"""
from __future__ import annotations

import asyncio
import importlib
import json

import mcp.types as types
import pytest

from testing._gates import requires_sec

SERVERS = {
    "sec": ("tools.web_search_server.web_search", "WebSearchServer"),
    "financial": ("tools.financial_modeling_engine.analysis_tools",
                  "Financial_Analysis"),
    "finnhub": ("tools.news_agregator.finnhub_server", "FinnhubServer"),
    "fred": ("tools.news_agregator.fred_server", "FredServer"),
    "altdata": ("tools.altdata_server.server", "AltDataServer"),
}

TICKER = "MSFT"

# Only what a bare ticker cannot satisfy, keyed off each schema's `required`.
# Values are plausible rather than meaningful: this asserts the shape of the
# answer, so the arithmetic behind it is someone else's test.
ARGUMENTS = {
    "get_fred_series": {"series_id": "DGS10", "limit": 3},
    "search_fred": {"search_text": "unemployment"},
    "get_urls_content": {"urls": ["https://www.sec.gov/"]},
    "get_patent_filings": {"company_name": "Microsoft"},
    "get_fund_holdings": {"fund": "BERKSHIRE HATHAWAY INC"},
    "compare_fund_holdings": {"fund": "BERKSHIRE HATHAWAY INC",
                              "other_fund": "BRIDGEWATER ASSOCIATES, LP"},
    "extract_disclosure_data": {"ticker": TICKER, "disclosure_name": "Leases"},
    "rag_search": {"query": "revenue growth"},
    "rag_ingest": {"text": "A short document for the smoke check.",
                   "doc_id": "SMOKE_TEST_1", "doc_type": "note"},
    "get_company_news": {"ticker": TICKER, "from_date": "2026-08-01",
                         "to_date": "2026-08-20"},
    "get_market_news": {"category": "general"},
    "get_earnings_calendar": {"from_date": "2026-08-01", "to_date": "2026-08-20"},
    "get_ipo_calendar": {"from_date": "2026-08-01", "to_date": "2026-08-20"},
    "get_financial_statements": {"ticker": TICKER, "statement": "ic",
                                 "freq": "annual"},
    "get_taiwan_monthly_revenue": {"company_codes": ["2330"]},
    "get_job_postings_count": {"company_slug": "microsoft"},
    "get_congress_trades": {"ticker": TICKER, "days": 90},
    "get_industry_etfs": {"theme": "semiconductors"},
    "get_historical_analogue": {"thesis_description": "a semiconductor downturn"},
    "get_thesis_evolution": {"thesis_id": 1},
    "comparable_company_analysis": {"companies": [TICKER, "AAPL", "GOOGL"]},
    "backtest_signal": {"signal": "golden_cross", "start": "2024-01-01",
                        "end": "2024-06-30"},
    # The pure calculators take no ticker-shaped shortcut: every input is
    # required because the tool refuses to invent one. Figures are MSFT-scale
    # and internally consistent, so a refusal here means the calculator
    # rejected the structure rather than the arguments being nonsense.
    "calculate_wacc": {"beta": 1.09, "risk_free_rate": 0.042,
                       "equity_risk_premium": 0.06, "cost_of_debt": 0.019,
                       "tax_rate": 0.176, "market_cap": 3_130_000_000_000,
                       "total_debt": 125_400_000_000},
    "calculate_scenario_dcf": {"ticker": TICKER,
                               "bear_growth": [0.02, 0.02, 0.02, 0.02, 0.02],
                               "base_growth": [0.06, 0.06, 0.05, 0.05, 0.04],
                               "bull_growth": [0.10, 0.10, 0.09, 0.08, 0.07],
                               "bear_margin": 0.32, "base_margin": 0.42,
                               "bull_margin": 0.48},
    "calculate_lbo": {"ticker": TICKER, "entry_ev": 30_000_000_000,
                      "revenue_base": 10_000_000_000, "ebitda_margin": 0.25,
                      "capex_pct_revenue": 0.04,
                      "depreciation": 400_000_000, "tax_rate": 0.21,
                      "revenue_growth": [0.05, 0.05, 0.04, 0.04, 0.03],
                      "debt_interest_rate": 0.08, "leverage_turns": 5.0,
                      "exit_multiple": 11.0},
    "calculate_credit_profile": {"ticker": TICKER,
                                 "total_debt": 125_400_000_000,
                                 "cash": 75_000_000_000,
                                 "ebitda": 137_000_000_000,
                                 "interest_expense": 2_900_000_000,
                                 "depreciation_abs": 22_000_000_000,
                                 "capex_abs": 44_000_000_000,
                                 "tax_rate": 0.176},
    "calculate_capital_returns": {"ticker": TICKER,
                                  "market_cap": 3_130_000_000_000,
                                  "ebitda": 137_000_000_000,
                                  "capex_abs": 44_000_000_000,
                                  "tax_rate": 0.176,
                                  "depreciation_abs": 22_000_000_000},
}

# Tools that write. A smoke check must not leave rows behind, and a tool whose
# whole job is to mutate cannot be exercised without doing so.
MUTATES = {"record_thesis_evolution", "rag_ingest"}


def _server(name):
    module, cls = SERVERS[name]
    return getattr(importlib.import_module(module), cls)().server


def _tool_names(name):
    srv = _server(name)
    listed = asyncio.run(srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")))
    return [t.name for t in listed.root.tools]


def _roster():
    return [(s, t) for s in SERVERS for t in _tool_names(s) if t not in MUTATES]


ROSTER = _roster()


def _call(server_name, tool):
    srv = _server(server_name)
    listed = asyncio.run(srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")))
    schema = next((t.inputSchema for t in listed.root.tools if t.name == tool), {})
    args = dict(ARGUMENTS.get(tool, {"ticker": TICKER}))
    if "ticker" in (schema or {}).get("required", []) and "ticker" not in args:
        args["ticker"] = TICKER
    handler = srv.request_handlers[types.CallToolRequest]
    return asyncio.run(handler(types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=tool, arguments=args))))


def test_the_roster_matches_the_registry():
    """A count kept by hand falls out of step; this one is read each run.

    The number is not asserted -- it moves with capability gating, which is
    why the previous harness's hand-written list could drift unnoticed. What
    is asserted is that every server contributed and nothing came back empty.
    """
    assert ROSTER, "no tools discovered; the registry walk is broken"
    for name in SERVERS:
        assert any(s == name for s, _ in ROSTER), f"{name} contributed no tools"


@requires_sec
@pytest.mark.network
@pytest.mark.parametrize("server_name,tool", ROSTER,
                         ids=[f"{s}:{t}" for s, t in ROSTER])
def test_a_tool_answers_or_says_why(server_name, tool):
    """Either a parseable answer, or a refusal that names its reason.

    `isError` is checked first because MCP returns a plain-text message for a
    schema mismatch and sets that flag -- correct protocol, and the previous
    harness parsed it as JSON regardless, reporting its own bad arguments as
    the tool failing.
    """
    result = _call(server_name, tool)

    if getattr(result.root, "isError", False):
        text = result.root.content[0].text if result.root.content else ""
        pytest.fail(f"{tool} rejected the call: {text[:200]}")

    assert result.root.content, f"{tool} returned no content"
    text = result.root.content[0].text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        pytest.fail(f"{tool} returned text that is not JSON: {text[:200]}")

    if isinstance(payload, dict) and payload.get("success") is False:
        assert payload.get("error"), (
            f"{tool} refused without saying why; a caller reading `error` "
            f"gets nothing and cannot tell this from a malformed response")
