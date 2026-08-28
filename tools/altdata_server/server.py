"""nemo_altdata MCP server — alternative data tools for pre-earnings research.

6 tools:
  get_taiwan_monthly_revenue -- FinMind API for TSMC/Foxconn/MediaTek/ASE revenue
  get_job_postings_count     -- Multi-ATS job listing count (Greenhouse/Lever/Workday
                                auto-discovery — no hardcoded company list)
  get_government_contracts   -- Federal contract awards via USASpending.gov (no auth)
  get_policy_signals         -- Legislative climate via GovTrack (+ Congress.gov if key set)
  get_capex_announcements    -- Capital investment announcements via DuckDuckGo news

Failure vocabulary, shared with tools/web_search_server/debt_maturity.py:

  success: false  -- the lookup could not be performed at all (no provider
                     answered, unknown entity, upstream 404, missing credential)
  coverage        -- "full" | "partial" | "not_covered", describing the data
                     actually obtained
  reason          -- machine-readable cause on a failure (no_provider,
                     unknown_ticker, provider_unavailable, ...)
  degraded        -- named degradations that narrowed the answer without
                     failing it (e.g. an unset API key)

A lookup that succeeded and found genuinely nothing returns success: true,
coverage "full" and a zero count. That is a different finding from a lookup
that could not be performed, and the two are never conflated.

Heavy tools run in isolated subprocesses to avoid asyncio
conflicts on Windows.
Light tools (FinMind, job postings, gov contracts, policy, capex) run
directly in async handlers via asyncio.to_thread.

Taiwan revenue uses FinMind (api.finmindtrade.com).
Job postings: auto-discovers Greenhouse → Lever → Workday in parallel (no curated list).

Register:
  claude mcp add -s user nemo_altdata -e PYTHONPATH=<repo> -- \\
    "<repo>/.venv/Scripts/python.exe" -m tools.altdata_server.server server

Optional env vars:
  FINMIND_TOKEN      -- FinMind API token (free tier: 600 req/day)
  CONGRESS_API_KEY   -- congress.gov API key (free; enhances get_policy_signals)
"""
from __future__ import annotations
from tools.response_meta import annotating, warning

import asyncio
import calendar
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.altdata_server.text_utils import (
    text_contains, count_matches, extract_dollar_amounts, parse_news_date,
    # The same compiled pattern and units that produce the VALUES, borrowed so
    # the classifier can find where each figure SITS in the text. A second copy
    # here would be free to drift from the one doing the extraction, and a
    # figure the two disagreed about would be summed without a category.
    _AMOUNT_PATTERN as _AMOUNT_RE,
    _UNIT_MULTIPLIERS as _AMOUNT_UNITS,
)

# ---------------------------------------------------------------------------
# Subprocess runner paths + Python executable
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(_VENV_PYTHON):
    _VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "bin", "python")
if not os.path.isfile(_VENV_PYTHON):
    _VENV_PYTHON = sys.executable


_SUBPROCESS_TIMEOUT_S = 40.0

# Greenhouse is tried twice -- ?content=true for departments, then the plain
# listing -- and both run inside the stage-1 pool. The pool must outwait their
# sum, or a board that is merely slow is reported as no provider at all.
_GREENHOUSE_CONTENT_TIMEOUT_S = 10
_GREENHOUSE_LISTING_TIMEOUT_S = 8
_ATS_POOL_TIMEOUT_S = 20


# ---------------------------------------------------------------------------
# Subprocess dispatch helper
# ---------------------------------------------------------------------------

def _run_subprocess(runner_path: str, tool_name: str, kwargs: dict,
                    sub_timeout: float) -> dict:
    try:
        proc = subprocess.run(
            [_VENV_PYTHON, runner_path, tool_name, json.dumps(kwargs)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=sub_timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False,
                "error": f"subprocess timed out after {sub_timeout}s"}

    stdout = (proc.stdout or "").strip()
    if not stdout:
        stderr_snippet = (proc.stderr or "").strip()[-300:]
        return {"success": False,
                "error": f"no output (exit {proc.returncode}): {stderr_snippet}"}
    # Runners emit exactly one JSON line LAST — parse the last non-empty line so
    # a stray library print to stdout (yfinance deprecation notice)
    # can't break the protocol.
    last_line = stdout.splitlines()[-1].strip()
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {"success": False,
                "error": f"output not valid JSON: {last_line[:300]}"}


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

def _envelope(data: Any, tool: str, ticker: str = "",
               errors: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "domain": "alt_data",
        "ticker": ticker,
        "tool": tool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": not bool(errors),
        "data": data,
        "metadata": {"errors": errors or []},
    }


def _ok(tool: str, data: Any, ticker: str = "") -> List[TextContent]:
    return [TextContent(type="text",
                        text=json.dumps(_envelope(data, tool, ticker), default=str))]


def _err(tool: str, msg: str, ticker: str = "") -> List[TextContent]:
    return [TextContent(type="text",
                        text=json.dumps(_envelope(None, tool, ticker, errors=[msg]),
                                        default=str))]


def _fail(tool: str, data: Any, msg: str, ticker: str = "") -> List[TextContent]:
    """A lookup that could not be performed, with its diagnostic payload kept.

    `_err` drops the data, which throws away what was tried and why it failed.
    Every fetcher on this server returns that detail (`coverage`, `reason`,
    the providers/queries attempted), so the failure envelope carries it.
    """
    return [TextContent(type="text",
                        text=json.dumps(_envelope(data, tool, ticker, errors=[msg]),
                                        default=str))]


def _dispatch(tool: str, result: Dict[str, Any], ticker: str = "") -> List[TextContent]:
    """Route a fetcher result to the success or failure envelope.

    `success: false` in the fetcher payload means the lookup could not be
    performed -- no provider answered, the entity is unknown, a credential is
    missing. It is deliberately distinct from a successful lookup whose answer
    is zero, which returns success with a coverage of "full".
    """
    if result.get("success") is False:
        return _fail(tool, result,
                     result.get("error") or f"{tool} lookup failed", ticker)
    return _ok(tool, result, ticker)


# ---------------------------------------------------------------------------
# Capex announcement helpers
# ---------------------------------------------------------------------------

# Sector-agnostic direction verbs. Classification keys on these universal
# expansion/contraction verbs (and a few facility phrases), NOT on sector nouns
# (factory / refinery / store / mine / data center), so the signal generalizes
# across every sector. Conjugations are listed explicitly because word-boundary
# matching is exact (avoids "invest" matching "investigation", "cut" matching
# "circuit") and English stemming via regex is error-prone.
_CAPEX_BULLISH = frozenset([
    "invest", "invests", "investing", "investment", "investments",
    "build", "builds", "building",
    "construct", "constructs", "constructing", "construction",
    "expand", "expands", "expanding", "expansion",
    "new factory", "new plant", "new facility", "new site", "new mill",
    "new refinery", "new mine", "new store", "new warehouse", "data center",
    "add capacity", "additional capacity", "new capacity", "increase capacity",
    "expand capacity", "boost production", "increase production",
    "ramp up", "ramping", "scale up", "scaling up",
    "upgrade", "upgrades", "upgrading",
    "modernize", "modernizing", "modernization",
    "groundbreaking", "break ground", "broke ground", "breaks ground",
    "gigafactory", "megafactory",
    "commission", "commissions", "commissioning",
])
_CAPEX_BEARISH = frozenset([
    "cancel", "cancels", "canceled", "cancelled", "cancelling", "cancellation",
    "delay", "delays", "delayed", "delaying",
    "halt", "halts", "halted", "halting",
    "suspend", "suspends", "suspended", "suspension",
    "pause", "pauses", "paused", "pausing",
    "scrap", "scraps", "scrapped",
    "shelve", "shelved", "shelving",
    "mothball", "mothballed", "mothballing",
    "idle", "idled", "idling",
    "shutdown", "shut down", "shutting down",
    "close", "closes", "closing", "closure",
    "wind down", "winding down",
    "cut capex", "cut spending", "capex cut", "spending cut", "cutting",
    "scale back", "scaling back", "scaled back",
    "write off", "write-off", "writeoff", "write down", "writedown", "write-down",
    "impairment", "impairments",
    "divest", "divests", "divesting", "divestiture", "divestment",
    "layoff", "layoffs", "lay off", "laying off",
    "restructure", "restructures", "restructuring",
    "reduce capacity", "reducing capacity", "reduced capacity",
])

# Backward-compatible alias — dollar extraction now lives in text_utils.
_extract_dollar_amounts = extract_dollar_amounts


def _classify_capex_text(text: str) -> str:
    """Direction of a capex headline via word-boundary verb matching."""
    bull = count_matches(text, _CAPEX_BULLISH)
    bear = count_matches(text, _CAPEX_BEARISH)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


# Generic corporate-name words too common to identify an article as on-topic.
_GENERIC_NAME_WORDS = frozenset([
    "inc", "corp", "corporation", "company", "co", "group", "holdings", "ltd",
    "plc", "the", "and", "international", "industries", "enterprises", "energy",
    "motors", "systems", "technologies", "communications", "financial", "services",
    "global", "worldwide", "partners", "trust", "incorporated", "limited",
])


def _company_name_tokens(company_name: str) -> List[str]:
    """Distinctive lowercased tokens from a company name, for relevance filtering.
    Drops generic corp words so 'NextEra Energy' -> ['nextera'], not ['energy']."""
    tokens = []
    for w in company_name.split():
        w = w.strip(".,'\"()").lower()
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) >= 4 and w not in _GENERIC_NAME_WORDS:
            tokens.append(w)
    return tokens


def _article_is_relevant(text: str, name_tokens: List[str]) -> bool:
    """True if any distinctive company token appears in the article text.
    If no distinctive tokens exist (all generic), keep the article (no filter)."""
    if not name_tokens:
        return True
    tl = text.lower()
    return any(tok in tl for tok in name_tokens)


def _capex_signal(announcements: List[Dict]) -> str:
    """Aggregate capex signal. A >=$1B item counts toward its OWN direction only
    (a cancelled $2B plant is bearish, never bullish) — the old code force-bulled
    on any large dollar figure regardless of direction."""
    bullish_n = sum(1 for a in announcements if a["direction"] == "bullish")
    bearish_n = sum(1 for a in announcements if a["direction"] == "bearish")
    major_bull = any(a["max_amount_usd"] >= 1_000_000_000 and a["direction"] == "bullish"
                     for a in announcements)
    major_bear = any(a["max_amount_usd"] >= 1_000_000_000 and a["direction"] == "bearish"
                     for a in announcements)
    if major_bull and not major_bear:
        return "bullish"
    if major_bear and not major_bull:
        return "bearish"
    if bullish_n > bearish_n * 2:
        return "bullish"
    if bearish_n > bullish_n * 2:
        return "bearish"
    return "neutral"



# ---------------------------------------------------------------------------
# What a dollar figure in a headline actually refers to
# ---------------------------------------------------------------------------
#
# A headline number is not capital expenditure because it is large and sits
# next to the word "expand". `get_capex_announcements("NVDA")` returned
# $1.62 TRILLION against single-digit-billion actual capex, built from $750B of
# customer deals, $500B of third-party capital raised by Apollo/BlackRock/KKR,
# a $250B lease guarantee, a $105B financing backstop and a $10B joint venture.
# Not one dollar of it was NVIDIA spending on property, plant or equipment.
#
# The error was structural: the old code took the LARGEST figure anywhere in an
# article and scanned the WHOLE article for direction verbs, so any expansion
# language in the piece attached itself to any number in the piece. Broadcom's
# "expand U.S. facility after $30 billion chip deal with Apple" is the pure
# case -- real expansion language, and a figure that is Apple paying Broadcom.
#
# So each figure is read in the clause that carries it, and a figure is capital
# expenditure only on positive evidence: the company named, a spending (or
# stopping) verb, and a physical asset, with nothing in the clause marking the
# money as financed, guaranteed, mobilised from third parties, invested in
# another company's equity, paid in by a customer, or restated as a running
# programme total. Anything else is unplaced, and unplaced money is never
# summed into a total labelled capex.

# Clause boundaries. Sentence ends need the abbreviation guard: without it
# "brings Taiwan Semiconductor Manufacturing Co.'s U.S. investment to $265
# billion" splits at "Co." and the containment evidence is severed from the
# figure it governs.
_CLAUSE_SEPARATOR = re.compile(r"\s+[–—|;:]\s+|\s+-\s+")
_SENTENCE_END = re.compile(r"[.!?]\s+(?=[A-Z“‘\"'])")
_ABBREVIATIONS = frozenset([
    "inc.", "corp.", "co.", "ltd.", "llc.", "plc.", "no.", "vs.", "etc.",
    "mr.", "mrs.", "ms.", "dr.", "jr.", "sr.", "st.", "gen.", "sen.", "rep.",
    "gov.", "prof.", "dept.", "est.", "approx.", "fig.",
])
_INITIALS = re.compile(r"(?:[A-Za-z]\.)+")


def _is_abbreviation(token: str) -> bool:
    t = token.strip("(\"'“”‘’,")
    return t.lower() in _ABBREVIATIONS or bool(_INITIALS.fullmatch(t))


def _clause_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each clause in `text`, split on sentence and clause marks."""
    cuts = {0, len(text)}
    for m in _CLAUSE_SEPARATOR.finditer(text):
        cuts.add(m.start())
        cuts.add(m.end())
    for m in _SENTENCE_END.finditer(text):
        head = text[:m.start() + 1].split()
        if head and _is_abbreviation(head[-1]):
            continue
        cuts.add(m.end())
    ordered = sorted(c for c in cuts if 0 <= c <= len(text))
    return [(a, b) for a, b in zip(ordered, ordered[1:])
            if b > a and text[a:b].strip()]


# Evidence that the money is NOT the company spending on its own assets,
# in the order it is tested. Financing precedes cumulative because "expected to
# total roughly $45 billion" is a tranche of a debt raise, not a programme
# total; third-party capital precedes deals because "$500 billion private
# capital deal" is both and is decided by the capital, not by the deal.
_FIGURE_EXCLUSIONS: List[Tuple[str, frozenset]] = [
    ("third_party_capital", frozenset([
        "third-party capital", "third party capital", "private capital",
        "outside capital", "external capital", "mobilize", "mobilise",
        "mobilized", "mobilised", "mobilizing", "asset manager",
        "asset management", "private credit", "sovereign wealth",
        "co-invest", "co-investment", "limited partners",
    ])),
    ("financing", frozenset([
        # "finance" and "notes" are deliberately not bare terms: "Yahoo
        # Finance" is in the boilerplate of half this corpus and "the analyst
        # notes" is a verb. Both would have taken real capex out of the total.
        "financing", "to finance", "will finance", "financed", "refinance",
        "refinancing", "raise", "raises", "raising", "raised", "borrow",
        "borrows", "borrowing", "debt", "bond", "bonds", "senior notes",
        "notes offering", "loan", "loans", "tranche", "tranches",
        "credit facility", "revolver", "underwrite", "underwriting",
        "securitization", "securitisation", "convertible", "leveraged",
        "repayment",
    ])),
    ("guarantee", frozenset([
        "guarantee", "guarantees", "guaranteed", "guaranty", "backstop",
        "backstops", "backstopped", "backstopping", "contingent",
        "lease obligation", "lease obligations", "indemnity",
    ])),
    ("equity_or_ma", frozenset([
        "acquire", "acquires", "acquired", "acquiring", "acquisition",
        "acquisitions", "merger", "merge", "merges", "takeover", "buyout",
        "tender offer", "stake", "stakes", "equity investment",
        "minority investment", "buyback", "buybacks", "repurchase",
        "repurchases", "dividend", "dividends",
    ])),
    ("customer_or_partner_deal", frozenset([
        "deal", "deals", "agreement", "agreements", "contract", "contracts",
        "order", "orders", "bookings", "backlog", "supply", "supplies",
        "purchase", "purchases", "customer", "customers", "signed", "signing",
        "lands", "landed", "awarded", "revenue", "revenues", "sales",
        "partnership", "partnerships", "joint venture", "subscription",
    ])),
    ("cumulative_total", frozenset([
        "brings", "bringing", "brought", "total", "totals", "totaling",
        "totalling", "to date", "so far", "cumulative", "combined",
        "overall", "including an additional", "includes an additional",
        "which includes", "up from", "on top of", "running total",
    ])),
]

# "invest $1 billion in NAVER Corp" buys shares; "invest $20 billion in a new
# Ohio factory" buys a factory. The difference is the object, and only the
# first has a corporate suffix on it.
_INVESTED_IN_ENTITY = re.compile(
    r"\bin\s+(?:[A-Z][\w&.’'-]*\s+){0,3}"
    r"(?:Corp|Corporation|Inc|Incorporated|Ltd|Limited|LLC|PLC|plc|AG|SA|NV|"
    r"Holdings|Group|Technologies|Industries|Motors|Partners)\b"
)

# Universal spend verbs. A cancelled fab is still a capital-expenditure
# announcement, so the stop verbs qualify a figure exactly as the spend verbs
# do -- the direction of the news is a separate axis from what the money is.
_CAPEX_SPEND_VERBS = frozenset([
    "invest", "invests", "investing", "invested", "investment", "investments",
    "spend", "spends", "spending", "spent", "outlay", "outlays",
    "commit", "commits", "committing", "committed", "commitment",
    "pledge", "pledges", "pledging", "pledged", "pour", "pours", "pouring",
    "plow", "plowing", "plough", "ploughing", "earmark", "earmarks",
    "earmarked", "build", "builds", "building", "built", "construct",
    "constructs", "constructing", "construction", "expand", "expands",
    "expanding", "expansion", "upgrade", "upgrades", "upgrading",
    "modernize", "modernise", "modernizing", "retool", "retooling",
    "break ground", "broke ground", "capex", "capital expenditure",
    "capital spending", "capital investment",
])
_CAPEX_STOP_VERBS = frozenset([
    "cancel", "cancels", "canceled", "cancelled", "cancelling", "cancellation",
    "halt", "halts", "halted", "halting", "scrap", "scraps", "scrapped",
    "shelve", "shelves", "shelved", "shelving", "mothball", "mothballed",
    "suspend", "suspends", "suspended", "suspension", "delay", "delays",
    "delayed", "abandon", "abandons", "abandoned", "scale back", "scaled back",
    "wind down", "winding down", "write down", "write-down", "writedown",
    "impairment",
])
# What the money has to be spent ON. Sector-agnostic on purpose: a refinery, a
# distribution centre and a fab are the same kind of claim.
_CAPEX_ASSET_NOUNS = frozenset([
    "plant", "plants", "factory", "factories", "fab", "fabs", "foundry",
    "foundries", "facility", "facilities", "data center", "data centre",
    "datacenter", "datacentre", "campus", "site", "sites", "mill", "mills",
    "refinery", "refineries", "mine", "mines", "smelter", "warehouse",
    "warehouses", "distribution center", "distribution centre", "store",
    "stores", "capacity", "manufacturing", "production line", "assembly line",
    "gigafactory", "megafactory", "infrastructure", "equipment", "machinery",
    "ai factory", "chipmaking", "fabrication", "network", "pipeline",
    "terminal", "rail", "fleet",
])
# A figure has to belong to the company being asked about. The relevance filter
# only asks whether the ARTICLE mentions it, which is how "Samsung and SK Hynix
# to build four new chip plants as South Korea unveils $520 billion" survives a
# TSMC query -- the body says "semiconductor" and the figure is Samsung's.
_SELF_REFERENCE = frozenset([
    "the company", "the firm", "the group", "the chipmaker", "the maker",
    "the manufacturer", "the automaker", "the retailer", "the miner",
    "the producer", "the operator",
])
# Naming the company is not being the company. "Nvidia supplier King Yuan
# Electronics to invest up to $1.4 billion in US facility" is a live headline
# that put a supplier's plant in NVIDIA's capex: the token, the verb and the
# asset were all in the clause and all belonged to someone else. These words
# turn the name in front of them into a modifier of the real spender.
_THIRD_PARTY_ROLES = (
    "supplier", "suppliers", "customer", "customers", "partner", "partners",
    "rival", "rivals", "competitor", "competitors", "client", "clients",
    "vendor", "vendors", "contractor", "contractors", "investor", "investors",
    "backer", "backers", "spinoff", "backed", "owned", "funded", "led",
    "linked", "adjacent", "related",
)
_ROLE_SUFFIX = re.compile(
    r"(?:’s|'s)?\s*[- ]\s*(?:" + "|".join(_THIRD_PARTY_ROLES) + r")\b",
    re.IGNORECASE)
# The same story with the relationship on the other side: "the supplier to
# chipmaker Nvidia said on Friday". The linking preposition is what makes the
# name the OBJECT of the role -- "Apple partner Broadcom" has no "to", and
# there Broadcom really is the one spending.
_ROLE_PREFIX = re.compile(
    r"\b(?:" + "|".join(_THIRD_PARTY_ROLES) + r")\s+(?:to|of|for)\s+"
    r"(?:[a-z][\w-]*\s+){0,2}$",
    re.IGNORECASE)
# A capex announcement is announced. "Nvidia Weighs $3 Billion SB Energy
# Investment" is a figure under consideration, and a total built from options
# a company is thinking about is not a total of anything it has committed.
_UNCOMMITTED = frozenset([
    "weighs", "weighing", "weighed", "considers", "considering", "considered",
    "mulls", "mulling", "mulled", "explores", "exploring", "eyes", "eyeing",
    "in talks", "may invest", "could invest", "might invest", "may spend",
    "could spend", "would invest", "would spend", "reportedly", "rumored",
    "rumoured", "is said to", "potential", "potentially", "proposed",
    "proposal", "seeks", "seeking", "weighing up", "studying",
])


def _names_a_third_party(clause: str, token: str) -> bool:
    """True when the company is named as somebody else's supplier/partner/rival
    rather than as the party doing the spending."""
    at = clause.lower().find(token)
    if at < 0:
        return False
    return bool(_ROLE_SUFFIX.match(clause, at + len(token))
                or _ROLE_PREFIX.search(clause[:at]))


def _classify_figure(clause: str, after: str,
                     name_tokens: List[str]) -> Tuple[str, str]:
    """What one dollar figure refers to, and the words that say so.

    Returns (category, evidence). Never guesses: a figure with no positive
    capex evidence comes back "unclassified" with the reason it failed, which
    keeps it out of the total instead of into it.
    """
    for category, terms in _FIGURE_EXCLUSIONS:
        hits = sorted(t for t in terms if text_contains(clause, t))
        if hits:
            return category, "; ".join(hits[:3])
        # Equity has a second test no keyword covers: what the money went INTO.
        if category == "equity_or_ma":
            m = _INVESTED_IN_ENTITY.search(after[:90])
            if m:
                return category, f"invested {m.group(0)}"

    named = [t for t in name_tokens if text_contains(clause, t)]
    modifiers = [t for t in named if _names_a_third_party(clause, t)]
    attribution = [t for t in named if t not in modifiers]
    attribution += [p for p in _SELF_REFERENCE if text_contains(clause, p)]
    hedges = sorted(t for t in _UNCOMMITTED if text_contains(clause, t))
    verbs = sorted(t for t in (_CAPEX_SPEND_VERBS | _CAPEX_STOP_VERBS)
                   if text_contains(clause, t))
    assets = sorted(t for t in _CAPEX_ASSET_NOUNS if text_contains(clause, t))

    if not attribution:
        if modifiers:
            return "unclassified", (
                f"{modifiers[0]} appears only as a modifier of the party "
                f"actually spending")
        return "unclassified", (
            "the clause carrying the figure does not name "
            + (" / ".join(name_tokens) if name_tokens else "the company"))
    if hedges:
        return "unclassified", (
            f"the spend is being weighed, not announced ({hedges[0]})")
    if not verbs:
        return "unclassified", "no spending or cancellation verb beside the figure"
    if not assets:
        return "unclassified", "nothing physical named for the money to be spent on"
    return "capital_expenditure", (
        f"{attribution[0]} + {verbs[0]} + {assets[0]}")


def _figures_in_article(text: str, name_tokens: List[str]) -> List[Dict[str, Any]]:
    """Every dollar figure in one article, each read in its own clause."""
    spans = _clause_spans(text)
    out: List[Dict[str, Any]] = []
    for m in _AMOUNT_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        mult = _AMOUNT_UNITS.get(m.group(2).lower())
        if not mult:
            continue
        clause = next((text[a:b] for a, b in spans if a <= m.start() < b), text)
        after = text[m.end():min(len(text), m.end() + 120)]
        category, evidence = _classify_figure(clause, after, name_tokens)
        out.append({
            "amount_usd": value * mult,
            "category": category,
            "evidence": evidence,
            "context": clause.strip()[:220],
        })
    return out


# The order a tie between two readings of the same figure is broken in. Capex
# comes last on purpose: it is the claim that has to be earned, so a figure one
# article calls a deal and another calls a plant is not capex.
_CATEGORY_PRIORITY = [c for c, _ in _FIGURE_EXCLUSIONS] + ["capital_expenditure"]


def _resolve_figure(readings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One verdict for a figure several articles each described differently.

    "unclassified" is an absence of evidence, not evidence of absence, so it
    never outvotes an article that did say what the money was.
    """
    definite = [r for r in readings if r["category"] != "unclassified"]
    if not definite:
        return dict(readings[0])
    counts: Dict[str, int] = {}
    for r in definite:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    best = max(counts.values())
    tied = sorted((c for c, n in counts.items() if n == best),
                  key=_CATEGORY_PRIORITY.index)
    return dict(next(r for r in definite if r["category"] == tied[0]))


def _usd_short(amount: float) -> str:
    """$265,000,000,000 -> '$265B'. Used in prose, never in a numeric field."""
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(amount) >= cutoff:
            return f"${amount / cutoff:.6g}{suffix}"
    return f"${amount:,.0f}"


# ---------------------------------------------------------------------------
# Company name resolution (ticker → display name)
# ---------------------------------------------------------------------------

class LookupFailure(Exception):
    """A lookup that could not be performed. Never an empty result."""
    reason = "lookup_failed"


class UnknownTicker(LookupFailure):
    """The quote provider has no such symbol."""
    reason = "unknown_ticker"


class ResolverUnavailable(LookupFailure):
    """The quote provider could not be reached, so the symbol is unjudged."""
    reason = "resolver_unavailable"


def _resolve_ticker(ticker: str) -> Dict[str, str]:
    """Company name and sector for a ticker, or raise.

    The old helpers swallowed every failure and echoed the symbol back as the
    company name, so `capex_announcements(ticker="ZZZZ")` searched the news for
    "ZZZZ", found nothing and reported success. yfinance answers an unknown
    symbol with a logged 404 and a near-empty dict rather than an exception, so
    a missing name is the signal that the symbol does not exist.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        raise ResolverUnavailable(
            f"could not resolve ticker '{ticker}': yfinance raised "
            f"{type(exc).__name__}: {str(exc)[:150]}") from exc

    name = info.get("longName") or info.get("shortName")
    if not name:
        raise UnknownTicker(
            f"'{ticker}' is not a symbol yfinance can resolve (no longName or "
            f"shortName in its quote response). No lookup was attempted.")
    # `industry` rides along because the sector alone is too coarse for some
    # questions. yfinance has no "Defense" sector -- every prime is
    # "Industrials" -- and the legislation that matters to them is named at
    # the industry level.
    return {"name": name, "sector": info.get("sector") or "",
            "industry": info.get("industry") or ""}


def _lookup_failed(exc: LookupFailure, **payload: Any) -> Dict[str, Any]:
    """Failure envelope body for a lookup that could not be performed."""
    return {"success": False, "coverage": "not_covered",
            "reason": exc.reason, "error": str(exc), **payload}


# ---------------------------------------------------------------------------
# Job postings — multi-ATS with generic Workday discovery
# ---------------------------------------------------------------------------

# ATS fingerprint patterns (regex, ats_type)
_ATS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"boards\.greenhouse\.io/([^/\"'\s&?]+)"), "greenhouse"),
    (re.compile(r"jobs\.lever\.co/([^/\"'\s&?]+)"), "lever"),
    (re.compile(r"([a-z0-9-]+)\.(wd\d+)\.myworkdayjobs\.com"), "workday"),
    (re.compile(r"([a-z0-9-]+)\.icims\.com"), "icims"),
    (re.compile(r"([a-z0-9-]+)\.taleo\.net"), "taleo"),
    (re.compile(r"successfactors\.com"), "successfactors"),
    (re.compile(r"smartrecruiters\.com/([^/\"'\s&?]+)"), "smartrecruiters"),
    (re.compile(r"([a-z0-9-]+)\.bamboohr\.com"), "bamboohr"),
    (re.compile(r"([a-z0-9-]+)\.jobvite\.com"), "jobvite"),
]

# Named in every failure so the caller knows exactly what was attempted.
_JOB_PROVIDERS_TRIED = (
    "greenhouse:boards-api.greenhouse.io",
    "lever:api.lever.co",
    "workday:tenant auto-discovery",
    "careers-page ATS fingerprint",
)

_ATS_UNSUPPORTED_MESSAGES: Dict[str, str] = {
    "taleo": "Oracle Taleo requires authentication — no public job count API",
    "successfactors": "SAP SuccessFactors requires authentication",
    "icims": "iCIMS does not expose a public job count API",
    "bamboohr": "BambooHR is internal HR software, not a public job board",
    "smartrecruiters": "SmartRecruiters has restricted public API access",
    "jobvite": "Jobvite does not expose a public job count API",
}


def _job_department(job: dict, source: str) -> Optional[str]:
    """The posting's department, or None when the provider did not disclose one.

    None is deliberately not the string "Unknown". Greenhouse's plain job
    listing carries no `departments` key at all, so bucketing it as "Unknown"
    reported a 100%-Unknown breakdown as if it were a real answer.
    """
    if source == "greenhouse":
        depts = job.get("departments") or []
        name = depts[0].get("name") if depts else None
    else:
        name = (job.get("categories") or {}).get("department")
    name = (name or "").strip()
    return name or None


def _job_postings_result(*, slug: str, source: str, source_url: str,
                          total: int, total_all: int, by_dept: Dict[str, int],
                          undisclosed: int,
                          dept_filter: Optional[str]) -> Dict[str, Any]:
    """Shared result shape for every ATS.

    `department_coverage` is reported separately from the count because a
    provider can return a reliable total with no breakdown whatsoever, and a
    silently degraded breakdown reads as a real one.
    """
    if total_all == 0 or undisclosed == 0:
        dept_coverage: str = "full"
        dept_reason: Optional[str] = None
    elif not by_dept:
        dept_coverage = "not_covered"
        dept_reason = (f"{source} returned no department for any of the "
                       f"{total_all} postings at {source_url}")
    else:
        dept_coverage = "partial"
        dept_reason = (f"{undisclosed} of {total_all} postings carry no "
                       f"department in the {source} response")

    if dept_filter and dept_coverage == "not_covered":
        # The unfiltered total was returned here, which reads as "every one of
        # these roles is in the requested department".
        return {
            "success": False,
            "coverage": "not_covered",
            "reason": "departments_unavailable",
            "error": (f"department_filter={dept_filter!r} cannot be applied: "
                      f"{dept_reason}"),
            "slug": slug, "ats": source, "source": source,
            "source_url": source_url,
            "total_postings": None,
            "total_all_depts": total_all,
            "by_department": None,
            "department_coverage": dept_coverage,
            "department_coverage_reason": dept_reason,
            "dept_filter_applied": dept_filter,
        }

    return {
        "success": True,
        "coverage": "full" if dept_coverage == "full" else "partial",
        "slug": slug, "ats": source, "source": source, "source_url": source_url,
        "total_postings": total,
        # `total` / `total_all_depts` predate `total_postings` and are kept so
        # existing callers keep working.
        "total": total,
        "total_all_depts": total_all,
        "by_department": (dict(sorted(by_dept.items(), key=lambda x: -x[1])[:15])
                          if by_dept else None),
        "departments_found": len(by_dept),
        "department_coverage": dept_coverage,
        "department_coverage_reason": dept_reason,
        "dept_filter_applied": dept_filter,
    }


def _normalize_ats_jobs(jobs: list, source: str, source_url: str,
                         slug: str, dept_filter: Optional[str]) -> Dict[str, Any]:
    by_dept: Dict[str, int] = {}
    undisclosed = 0
    for job in jobs:
        dept = _job_department(job, source)
        if dept is None:
            undisclosed += 1
        else:
            by_dept[dept] = by_dept.get(dept, 0) + 1

    total = len(jobs)
    if dept_filter:
        total = sum(1 for j in jobs
                    if dept_filter.lower() in (_job_department(j, source) or "").lower())

    return _job_postings_result(
        slug=slug, source=source, source_url=source_url,
        total=total, total_all=len(jobs), by_dept=by_dept,
        undisclosed=undisclosed, dept_filter=dept_filter)


def _try_greenhouse_norm(slug: str, dept_filter: Optional[str]) -> Optional[Dict]:
    """None means "not a Greenhouse board" -- try the next provider."""
    import requests
    base = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    # content=true is the only variant of this endpoint that carries per-job
    # departments; the plain listing omits the key entirely. The plain listing
    # is the fallback so a slow/oversized content response still yields a count
    # (with department_coverage saying the breakdown is missing).
    for url, timeout in ((f"{base}?content=true", _GREENHOUSE_CONTENT_TIMEOUT_S),
                         (base, _GREENHOUSE_LISTING_TIMEOUT_S)):
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return _normalize_ats_jobs(resp.json().get("jobs", []),
                                       "greenhouse", url, slug, dept_filter)
        except Exception:
            continue
    return None


def _try_lever_norm(slug: str, dept_filter: Optional[str]) -> Optional[Dict]:
    import requests
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _normalize_ats_jobs(resp.json(), "lever", url, slug, dept_filter)
    except Exception:
        return None


def _workday_probe(tenant: str, wd_n: int, path: str) -> Optional[Dict]:
    """Single Workday endpoint probe with limit=1. Returns discovery metadata or None."""
    import requests
    url = (f"https://{tenant}.wd{wd_n}.myworkdayjobs.com"
           f"/wday/cxs/{tenant}/{path}/jobs")
    try:
        resp = requests.post(
            url,
            json={"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""},
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        if resp.status_code not in (200, 201):
            return None
        data = resp.json()
        if "total" not in data:
            return None
        return {"tenant": tenant, "wd_n": wd_n, "path": path,
                "url": url, "_total": data["total"]}
    except Exception:
        return None


# Workday rejects a page size above 20 with HTTP 400. The full fetch asked for
# 50, so every discovered tenant 400'd and read as "no Workday board".
_WORKDAY_PAGE_LIMIT = 20


def _workday_fetch_full(tenant: str, wd_n: int, path: str,
                         dept_filter: Optional[str]) -> Optional[Dict]:
    """Full Workday job fetch with facet department breakdown."""
    import requests
    url = (f"https://{tenant}.wd{wd_n}.myworkdayjobs.com"
           f"/wday/cxs/{tenant}/{path}/jobs")
    try:
        resp = requests.post(
            url,
            json={"appliedFacets": {}, "limit": _WORKDAY_PAGE_LIMIT,
                  "offset": 0, "searchText": ""},
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        if resp.status_code not in (200, 201):
            return None
        data = resp.json()
        if "total" not in data:
            return None

        total = data["total"]
        by_dept: Dict[str, int] = {}
        for facet in data.get("facets", []):
            param = facet.get("facetParameter", "").lower()
            if any(kw in param for kw in ("department", "jobfamily", "workertype",
                                           "organization", "function")):
                for entry in facet.get("facetValues", []):
                    name = (entry.get("value") or "").strip()
                    if name:
                        by_dept[name] = entry.get("count", 0)
                if by_dept:
                    break

        filtered_total = total
        if dept_filter and by_dept:
            filtered_total = sum(v for k, v in by_dept.items()
                                 if dept_filter.lower() in k.lower())

        # Facet counts can cover fewer postings than the board reports; the
        # remainder is undisclosed rather than absent.
        faceted = sum(by_dept.values())
        undisclosed = max(total - faceted, 0) if by_dept else total

        return _job_postings_result(
            slug=tenant, source="workday", source_url=url,
            total=filtered_total if dept_filter else total,
            total_all=total, by_dept=by_dept, undisclosed=undisclosed,
            dept_filter=dept_filter)
    except Exception:
        return None


def _discover_workday_tenant(slug: str) -> Optional[Dict]:
    """Generic parallel Workday tenant discovery — probes only (limit=1), no full
    fetch. Returns discovery metadata {tenant, wd_n, path, url, _total} or None.

    Kept separate from the full fetch so the heavy fetch never runs inside a
    racing outer pool (which could orphan it past cancel_futures).
    No hardcoded company list — works for any Workday customer.
    """
    clean = slug.replace("-", "").replace("_", "")
    tenants = list(dict.fromkeys([slug, clean, slug.replace("-", "_")]))
    wd_nums = [1, 3, 5, 12, 2, 7]
    paths = ["External_Career_Site", "External", f"{clean}_Careers", "Careers"]

    candidates = [(t, n, p) for t in tenants for n in wd_nums for p in paths]

    found_meta = None
    pool = ThreadPoolExecutor(max_workers=16)
    try:
        futures = {pool.submit(_workday_probe, t, n, p): (t, n, p)
                   for t, n, p in candidates}
        for future in as_completed(futures, timeout=8):
            try:
                result = future.result()
                if result is not None and found_meta is None:
                    found_meta = result
                    break
            except Exception:
                pass
    except Exception:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return found_meta


def _try_workday_discovery(slug: str, dept_filter: Optional[str]) -> Optional[Dict]:
    """Discover the Workday tenant, then fetch the full job breakdown. Returns the
    normalized result or None. Run this in its OWN scope (not inside a racing
    pool), so the heavy fetch cannot be orphaned."""
    meta = _discover_workday_tenant(slug)
    if meta is None:
        return None
    return _workday_fetch_full(meta["tenant"], meta["wd_n"],
                                meta["path"], dept_filter)


def _detect_ats_from_website(slug: str) -> Optional[Tuple[str, str]]:
    """Follow the company careers page URL to fingerprint which ATS they use."""
    import requests
    base = slug.replace("-", "").replace("_", "")
    careers_urls = [
        f"https://www.{base}.com/careers",
        f"https://careers.{base}.com",
        f"https://www.{slug}.com/careers",
        f"https://jobs.{base}.com",
    ]
    for url in careers_urls:
        try:
            resp = requests.get(
                url, timeout=6, allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
                stream=True,
            )
            for pattern, ats in _ATS_PATTERNS:
                m = pattern.search(resp.url)
                if m:
                    extracted = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                    return (ats, extracted)
            chunk = b""
            for block in resp.iter_content(8192):
                chunk += block
                break
            content = chunk.decode("utf-8", errors="ignore")
            for pattern, ats in _ATS_PATTERNS:
                m = pattern.search(content)
                if m:
                    extracted = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                    return (ats, extracted)
        except Exception:
            continue
    return None


def _fetch_job_postings(slug: str, ats: str,
                         dept_filter: Optional[str]) -> Dict[str, Any]:
    """
    Multi-ATS job listing fetch with fully generic discovery.
    Stage 1: Greenhouse + Lever in parallel (cheap single GETs).
    Stage 2: Workday auto-discovery (own scope — the heavy nested-pool op never
             races inside the Stage-1 pool, so it can't be orphaned/leak threads).
    Stage 3: ATS fingerprint via company careers URL redirect.
    Stage 4: Explicit failure -- `success: false`, never a count of zero.

    A provider returning None means "not this provider, try the next one". Only
    every provider returning None is a failed lookup, and that is reported as
    one: a reachable board with no open roles returns success with
    total_postings 0, which is a different finding entirely.
    """
    # Direct workday bypass (explicit request)
    if ats == "workday":
        result = _try_workday_discovery(slug, dept_filter)
        if result:
            return result
        return {
            "success": False,
            "coverage": "not_covered",
            "reason": "no_provider",
            "error": f"Workday not found for '{slug}' — auto-discovery tried "
                     f"all common tenant variants.",
            "ats_detected": "workday_not_found", "slug": slug,
            "total_postings": None,
            "providers_tried": ["workday"],
        }

    # Stage 1: greenhouse + lever in parallel (both are single fast GETs).
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        gh_f = pool.submit(_try_greenhouse_norm, slug, dept_filter)
        lv_f = pool.submit(_try_lever_norm, slug, dept_filter)
        results: Dict[str, Dict] = {}
        try:
            # Bounds both providers, including Greenhouse's content=true
            # attempt and its plain-listing fallback (10s + 8s worst case).
            # A tighter bound here would report "no provider answered" for a
            # board that was merely slow.
            for future in as_completed([gh_f, lv_f],
                                       timeout=_ATS_POOL_TIMEOUT_S):
                try:
                    result = future.result()
                    if result:
                        results[result.get("ats", "")] = result
                except Exception:
                    pass
        except Exception:
            pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # Prefer the explicitly requested ATS, otherwise any hit. A provider that
    # answered is returned even when its own result is a failure (e.g. a
    # department filter it cannot apply) -- that is a real finding about a real
    # board, not a reason to keep hunting.
    chosen = results.get(ats) or (next(iter(results.values())) if results else None)
    if chosen:
        return chosen

    # Stage 2: Workday auto-discovery in its own scope (no nested racing pool).
    wd = _try_workday_discovery(slug, dept_filter)
    if wd:
        return wd

    # Stage 3: ATS fingerprinting via website
    detected = _detect_ats_from_website(slug)
    if detected:
        detected_ats, detected_slug = detected
        if detected_ats == "greenhouse" and detected_slug:
            r = _try_greenhouse_norm(detected_slug, dept_filter)
            if r:
                return r
        elif detected_ats == "lever" and detected_slug:
            r = _try_lever_norm(detected_slug, dept_filter)
            if r:
                return r
        elif detected_ats == "workday":
            r = _try_workday_discovery(detected_slug or slug, dept_filter)
            if r:
                return r

        # Detected but not queryable
        msg = _ATS_UNSUPPORTED_MESSAGES.get(
            detected_ats,
            f"'{detected_ats}' ATS is not publicly queryable",
        )
        return {
            "success": False,
            "coverage": "not_covered",
            "reason": "ats_not_queryable",
            "error": f"{msg} (detected for '{slug}')",
            "ats_detected": detected_ats,
            "detected_slug": detected_slug, "slug": slug,
            "total_postings": None,
            "providers_tried": list(_JOB_PROVIDERS_TRIED),
        }

    # Stage 4: every provider declined. This is a failed lookup, not zero jobs.
    return {
        "success": False,
        "coverage": "not_covered",
        "reason": "no_provider",
        "error": (
            f"No public job board answered for '{slug}'. Tried Greenhouse "
            "(boards-api.greenhouse.io), Lever (api.lever.co), Workday tenant "
            "auto-discovery, and ATS fingerprinting via the company careers "
            "page. This is a failed lookup, not a count of zero: the company "
            "may use a proprietary portal (Microsoft, Apple, Google, Meta) or "
            "may not exist."
        ),
        "ats_detected": "unknown",
        "slug": slug,
        "total_postings": None,
        "providers_tried": list(_JOB_PROVIDERS_TRIED),
    }


# ---------------------------------------------------------------------------
# Taiwan monthly revenue via FinMind
# ---------------------------------------------------------------------------

def _fetch_taiwan_revenue_finmind(company_codes: List[str], months: int) -> Dict[str, Any]:
    import requests
    lookback_days = (months + 14) * 31
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    token = os.environ.get("FINMIND_TOKEN", "")
    token_hint = ("" if token else
                  " FINMIND_TOKEN is unset, so this ran on FinMind's anonymous "
                  "tier (shared per-IP daily quota).")
    degraded: List[str] = [] if token else [
        "FINMIND_TOKEN unset — requests use FinMind's anonymous tier, which "
        "shares a daily quota per IP."]
    results: Dict[str, Any] = {}

    for code in company_codes:
        params: Dict[str, Any] = {
            "dataset": "TaiwanStockMonthRevenue",
            "data_id": code,
            "start_date": start_date,
        }
        if token:
            params["token"] = token
        try:
            resp = requests.get(
                "https://api.finmindtrade.com/api/v4/data",
                params=params, timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            results[code] = {
                "error": (f"FinMind request for code {code} failed: "
                          f"{type(exc).__name__}: {str(exc)[:150]}")
            }
            continue

        if payload.get("status") != 200:
            results[code] = {
                "error": (f"FinMind status {payload.get('status')}: "
                          f"{payload.get('msg', 'no message')}"
                          f" (code {code}).{token_hint}")
            }
            continue

        if not payload.get("data"):
            # FinMind answers an unknown code with status 200 / msg 'success'
            # and an empty list, which folded into the branch above as the
            # self-contradicting error "FinMind status 200: success".
            results[code] = {
                "error": (f"FinMind returned no TaiwanStockMonthRevenue rows for "
                          f"code {code} since {start_date}. Check the code is a "
                          "Taiwan-listed ticker (TSMC=2330, Foxconn=2317, "
                          "MediaTek=2454, ASE Group=3711).")
            }
            continue

        raw = payload["data"]
        raw.sort(key=lambda r: r["date"])
        rev_lookup: Dict[tuple, float] = {
            (r["revenue_year"], r["revenue_month"]): r["revenue"] for r in raw
        }
        recent = raw[-months:]
        parsed = []
        for r in recent:
            yr, mo = r["revenue_year"], r["revenue_month"]
            rev_raw = r.get("revenue") or 0
            rev_ntd_m = round(rev_raw / 1_000_000, 1) if rev_raw else None
            prior_rev = rev_lookup.get((yr - 1, mo))
            yoy_pct = None
            if prior_rev and prior_rev != 0:
                yoy_pct = round((rev_raw - prior_rev) / abs(prior_rev) * 100, 2)
            # FinMind's `date` is the month the figure was ANNOUNCED, one ahead
            # of the month it describes: TSMC's June 2026 revenue is stamped
            # 2026-07-01. Passing that through as `date` put it next to
            # year=2026 / month=6 disagreeing with both, and anything charted
            # on it attributed every month's revenue to the following month.
            # `date` is now the first day of the period; the announcement
            # stamp is kept under a name that says what it is.
            parsed.append({"year": yr, "month": mo,
                           "date": f"{yr:04d}-{mo:02d}-01",
                           "announced_date": r["date"],
                           "revenue_ntd_m": rev_ntd_m, "yoy_pct": yoy_pct})
        results[code] = {"company_code": code, "months_returned": len(parsed),
                         "months": parsed, "source": "finmind"}

    failed = [c for c in company_codes if "error" in results.get(c, {})]

    if failed and len(failed) == len(company_codes):
        # Every code failed. The per-code errors were there all along, nested
        # inside a payload the envelope still called a success.
        return {
            "success": False,
            "coverage": "not_covered",
            "reason": "no_data",
            "error": ("FinMind returned no monthly revenue for any requested "
                      f"code ({', '.join(company_codes)}): "
                      + "; ".join(results[c]["error"] for c in failed)),
            "companies": results,
            "codes_requested": company_codes,
            "codes_failed": failed,
            "degraded": degraded,
        }

    return {
        "success": True,
        "coverage": "partial" if failed else "full",
        "companies": results,
        "codes_requested": company_codes,
        "codes_failed": failed,
        "degraded": degraded,
    }


# ---------------------------------------------------------------------------
# Government contracts via USASpending.gov
# ---------------------------------------------------------------------------

_USASPENDING_BASE = "https://api.usaspending.gov/api/v2/search"
_SOT_URL = _USASPENDING_BASE + "/spending_over_time/"
_AGENCY_CATEGORY_URL = _USASPENDING_BASE + "/spending_by_category/awarding_agency/"
_AWARD_COUNT_URL = _USASPENDING_BASE + "/spending_by_award_count/"
_USA_TIMEOUT = 30
_CONTRACT_AWARD_TYPES = ["A", "B", "C", "D"]
# Grants/cooperative agreements (NIH, NSF) — relevant for biotech with include_grants.
_GRANT_AWARD_TYPES = ["02", "03", "04", "05"]
_USA_HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}


def _usa_filters(company_name: str, start: str, end: str,
                  award_types: List[str]) -> Dict[str, Any]:
    return {
        "recipient_search_text": [company_name],
        "time_period": [{"start_date": start, "end_date": end}],
        "award_type_codes": award_types,
    }


def _fiscal_bucket_month(fiscal_year: Any, month: Any) -> str:
    """Calendar month for one spending_over_time bucket, as YYYY-MM-01.

    The endpoint labels its buckets by FEDERAL FISCAL month, where month 1 is
    October. `{"fiscal_year": "2026", "month": "9"}` is June 2026, not
    September. Reading the label as a calendar month slides the whole series
    three months and would put the data horizon in the wrong place.
    """
    fy, fm = int(fiscal_year), int(month)
    return f"{fy - 1 if fm <= 3 else fy:04d}-{((fm + 8) % 12) + 1:02d}-01"


def _usa_month_series(company_name: Optional[str], start: str, end: str,
                       award_types: List[str]) -> List[Dict[str, Any]]:
    """Obligation FLOW per calendar month over a window, oldest first.

    Unlike spending_by_award (which returns multi-year contract ceiling
    values), spending_over_time returns the obligated amount per period. The
    time_period filter bounds the window, so the buckets are the true period
    flow. `company_name` of None asks the same question of the whole federal
    government, which is how data freshness is measured.
    """
    import requests
    filters = _usa_filters(company_name or "", start, end, award_types)
    if not company_name:
        filters.pop("recipient_search_text", None)
    resp = requests.post(_SOT_URL, json={"group": "month", "filters": filters},
                         timeout=_USA_TIMEOUT, headers=_USA_HEADERS)
    resp.raise_for_status()
    series = []
    for row in resp.json().get("results", []) or []:
        bucket = row.get("time_period") or {}
        if "fiscal_year" not in bucket or "month" not in bucket:
            continue
        series.append({
            "month": _fiscal_bucket_month(bucket["fiscal_year"], bucket["month"]),
            "obligations_usd": float(row.get("aggregated_amount") or 0),
        })
    series.sort(key=lambda r: r["month"])
    return series


def _usa_obligations_total(company_name: str, start: str, end: str,
                            award_types: List[str]) -> float:
    """Total obligation flow over a window."""
    return float(sum(r["obligations_usd"] for r in
                     _usa_month_series(company_name, start, end, award_types)))


def _usa_top_agencies(company_name: str, start: str, end: str,
                       award_types: List[str]) -> List[Dict]:
    import requests
    payload = {"category": "awarding_agency",
               "filters": _usa_filters(company_name, start, end, award_types),
               "limit": 5}
    resp = requests.post(_AGENCY_CATEGORY_URL, json=payload, timeout=_USA_TIMEOUT,
                         headers=_USA_HEADERS)
    resp.raise_for_status()
    return resp.json().get("results", [])


def _usa_award_count(company_name: str, start: str, end: str,
                      award_types: List[str]) -> int:
    import requests
    payload = {"filters": _usa_filters(company_name, start, end, award_types)}
    resp = requests.post(_AWARD_COUNT_URL, json=payload, timeout=_USA_TIMEOUT,
                         headers=_USA_HEADERS)
    resp.raise_for_status()
    counts = resp.json().get("results", {}) or {}
    # NOTE: summing all categories is safe ONLY because award_type_codes filters
    # the response server-side (contract codes -> grant buckets are zero). If
    # award types are ever widened, restrict this sum to the requested kinds.
    return sum(int(counts.get(k, 0) or 0)
               for k in ("contracts", "grants", "idvs", "direct_payments", "other"))


def _gov_contracts_signal(trailing_total: float, prior_total: Optional[float],
                           yoy_pct: Optional[float]) -> str:
    """Pure signal logic. Driven by real period-flow YoY only — no absolute-size
    override in EITHER direction. prior_total=None means the prior-period fetch
    FAILED (unknown), which must never read as 'newly winning business'."""
    if prior_total is None:
        # prior unknown (fetch error): cannot judge trend; never bullish.
        return "not_applicable" if trailing_total < 10_000_000 else "neutral"
    if trailing_total < 10_000_000 and prior_total < 10_000_000:
        return "not_applicable"  # essentially no federal business either period
    if yoy_pct is not None:
        if yoy_pct > 15:
            return "bullish"
        if yoy_pct < -15:
            return "bearish"   # incl. total collapse from a real prior base
        return "neutral"
    if prior_total == 0 and trailing_total >= 10_000_000:
        return "bullish"  # newly winning federal business where there was none
    return "neutral"


# ---------------------------------------------------------------- data horizon

# A published month runs at roughly the level of its neighbours. Below this
# share of the median month it is not a quiet month, it is a month the
# agencies have not filed yet. 0.8 separates the two cleanly on the measured
# series: October 2025 (a genuinely thin month at 32% of median) sits inside
# the published run and does not move the horizon, because only an unbroken
# run of thin months at the END of the series counts as unpublished.
_HORIZON_COMPLETENESS = 0.8
_USA_HORIZON_LOOKBACK_MONTHS = 18
_USA_HORIZON_TTL_SECONDS = 6 * 3600
# Above this share of the window sitting past the horizon, the window is
# mostly unpublished and cannot carry a signal in either direction.
_LAG_MOSTLY = 0.5

_usa_horizon_cache: Dict[Tuple[str, ...], Tuple[float, Dict[str, Any]]] = {}


def _shift_months(day, count: int) -> date:
    """Calendar-exact month arithmetic, clamped to the length of the month.

    timedelta(days=months * 365/12) drifts: three months before 2026-08-25 came
    out as 2026-05-26. And a year before 2024-02-29 has to be 2023-02-28,
    because 2023-02-29 does not exist.
    """
    anchor = day.date() if isinstance(day, datetime) else day
    index = anchor.month - 1 + count
    year, month = anchor.year + index // 12, index % 12 + 1
    return date(year, month, min(anchor.day, calendar.monthrange(year, month)[1]))


def _data_horizon_from_series(series: List[Dict[str, Any]],
                               threshold: float = _HORIZON_COMPLETENESS
                               ) -> Dict[str, Any]:
    """The last date USASpending has actually published, from its own series.

    Federal contract actions reach USASpending months in arrears -- the
    Department of Defense above all, whose actions are withheld from public
    FPDS release for 90 days. Measured government-wide on 2026-08-25, monthly
    contract obligations held $50-100bn through 2026-05 and then fell to
    $33.1bn, $39.9bn and $15.5bn. Lockheed's own months went from a $4.1bn
    median to $147m, $27m and $16m over the same three.

    Those months are unpublished, not empty, and the difference decides
    whether a trailing total is a finding or a gap. So the horizon is measured
    from the provider each time rather than assumed from a constant that would
    silently go stale.
    """
    if not series:
        raise ValueError("cannot measure a data horizon from an empty series")

    amounts = sorted(row["obligations_usd"] for row in series)
    middle = len(amounts) // 2
    baseline = (amounts[middle] if len(amounts) % 2
                else (amounts[middle - 1] + amounts[middle]) / 2)
    floor = baseline * threshold

    incomplete: List[str] = []
    for row in reversed(series):
        if row["obligations_usd"] >= floor:
            break
        incomplete.append(row["month"][:7])
    incomplete.reverse()

    last_published = (incomplete[0] if incomplete else None)
    if last_published:
        year, month = int(last_published[:4]), int(last_published[5:7])
        published_through = _shift_months(date(year, month, 1), -1)
    else:
        last = series[-1]["month"]
        published_through = date(int(last[:4]), int(last[5:7]), 1)
    end_of_month = calendar.monthrange(published_through.year,
                                       published_through.month)[1]
    return {
        "horizon": published_through.replace(day=end_of_month).strftime("%Y-%m-%d"),
        "incomplete_months": incomplete,
        "baseline_usd": baseline,
        "measured_from": "spending_over_time, government-wide, by month",
    }


def _usa_data_horizon(award_types: List[str]) -> Dict[str, Any]:
    """Measure how current USASpending's published data is, and cache it.

    Government-wide rather than per company: one company's quiet quarter is
    ambiguous, the whole federal government's is not. Keyed by award type
    because the feeds differ -- measured 2026-08-25, grants (02/03/04/05) ran
    current at $240bn for 2026-07 while contracts (A/B/C/D) stopped at
    2026-05 -- so the probe measures exactly the universe being summed.

    Cached for six hours: the answer changes daily at most, and the probe
    would otherwise repeat on every call.
    """
    key = tuple(award_types)
    cached = _usa_horizon_cache.get(key)
    now = time.monotonic()
    if cached and now - cached[0] < _USA_HORIZON_TTL_SECONDS:
        return cached[1]

    today = date.today()
    series = _usa_month_series(
        None, _shift_months(today, -_USA_HORIZON_LOOKBACK_MONTHS).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"), award_types)
    horizon = _data_horizon_from_series(series)
    _usa_horizon_cache[key] = (now, horizon)
    return horizon


def _window_lag_fraction(start: str, end: str, horizon: str) -> float:
    """How much of [start, end] lies past the provider's published data."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    edge = date.fromisoformat(horizon)
    span = (last - first).days
    if span <= 0:
        return 0.0
    return max(0.0, (last - max(first, edge)).days / span)


def _gov_windows(today, months: int):
    """Pure: (trailing_start, trailing_end, compare_start, compare_end).

    The comparison window is the SAME months one year earlier, which is what
    `yoy_change_pct` claims to measure. It used to be the months immediately
    before the trailing window: at months=3 the "prior" figure came out equal
    to the months=6 trailing figure, which is how the substitution was caught.

    A sequential comparison cannot be labelled year-over-year on this series.
    Federal obligations peak enormously at the 30 September fiscal year end --
    government-wide, 2025-09 was $148bn against a $60bn median month -- so a
    window-against-previous-window change measures where in the fiscal year
    the window happens to sit.

    USASpending's time_period is inclusive on both ends, so the comparison
    window stops the day before the trailing window's own start date shifted
    back a year: at months=12 the two windows are adjacent, and the shared
    boundary day would otherwise be counted in both.
    """
    anchor = today.date() if isinstance(today, datetime) else today
    trailing_start = _shift_months(anchor, -months)
    return (trailing_start.strftime("%Y-%m-%d"),
            anchor.strftime("%Y-%m-%d"),
            _shift_months(trailing_start, -12).strftime("%Y-%m-%d"),
            (_shift_months(anchor, -12) - timedelta(days=1)).strftime("%Y-%m-%d"))


def _fetch_government_contracts(ticker: str, company_name: str,
                                  months: int, include_grants: bool) -> Dict[str, Any]:
    base = {"ticker": ticker, "company_name": company_name or None,
            "period_months": months, "trailing_awards_usd": None,
            "prior_period_awards_usd": None, "yoy_change_pct": None,
            "data_horizon": None, "window_lag_fraction": None,
            "signal": None, "top_agencies": [], "source": "usaspending.gov"}

    # An explicit company_name is the caller asserting the entity; only derive
    # it when they did not, and refuse to search for a symbol that does not
    # resolve (USASpending answers "$0 of federal business" for any string).
    if not company_name:
        try:
            company_name = _resolve_ticker(ticker)["name"]
        except LookupFailure as exc:
            return _lookup_failed(exc, **base)
        base["company_name"] = company_name

    trailing_start, end, prior_start, prior_end = _gov_windows(datetime.now(), months)

    award_types = _CONTRACT_AWARD_TYPES + (_GRANT_AWARD_TYPES if include_grants else [])

    # Run the independent USASpending queries concurrently.
    results: Dict[str, Any] = {}
    errors: List[str] = []
    degraded: List[str] = []
    pool = ThreadPoolExecutor(max_workers=5)
    try:
        futures = {
            pool.submit(_usa_obligations_total, company_name, trailing_start, end, award_types): "trailing",
            pool.submit(_usa_obligations_total, company_name, prior_start, prior_end, award_types): "prior",
            pool.submit(_usa_top_agencies, company_name, trailing_start, end, award_types): "agencies",
            pool.submit(_usa_award_count, company_name, trailing_start, end, award_types): "count",
            pool.submit(_usa_data_horizon, award_types): "horizon",
        }
        for future in as_completed(futures, timeout=_USA_TIMEOUT + 5):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                detail = f"{key}: {type(exc).__name__}: {str(exc)[:120]}"
                if key == "horizon":
                    # A failed freshness probe narrows what can be said without
                    # invalidating the totals, so it is named as a degradation
                    # rather than counted against coverage -- but it is never
                    # read as "no lag", which would be the silent guess this
                    # measurement exists to remove.
                    degraded.append(
                        f"USASpending data-horizon probe failed ({detail}); "
                        "whether this window falls inside the provider's "
                        "reporting lag is unknown.")
                else:
                    errors.append(detail)
    except Exception as exc:
        errors.append(f"timeout waiting for USASpending: {type(exc).__name__}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # Trailing obligations drive the core YoY signal; without it we cannot proceed.
    if "trailing" not in results:
        return {
            **base, "company_name": company_name,
            "success": False, "coverage": "not_covered",
            "reason": "provider_unavailable",
            "error": (f"USASpending.gov returned no trailing obligations for "
                      f"{company_name!r}: {'; '.join(errors) or 'no data'}"),
            "partial_errors": errors,
        }

    trailing_total = results.get("trailing", 0) or 0
    # MISSING is not ZERO: a failed prior fetch must not read as "no prior
    # business" (which would manufacture a bullish new-entrant signal).
    prior_total = results["prior"] if "prior" in results else None
    agencies = results.get("agencies", []) or []
    award_count = results.get("count", 0) or 0

    yoy_pct = None
    if prior_total is not None and prior_total > 0:
        yoy_pct = round((trailing_total - prior_total) / abs(prior_total) * 100, 1)

    top_agencies = [
        {"agency": a.get("name") or "Unknown",
         "amount_usd": a.get("amount") or 0,
         "pct_of_total": round((a.get("amount") or 0) / trailing_total * 100, 1) if trailing_total else 0}
        for a in agencies[:5]
    ]

    # How much of the trailing window the provider has not published yet. An
    # unpublished month reads exactly like a month with no awards, and the two
    # are opposite findings.
    measured = results.get("horizon")
    horizon = measured["horizon"] if measured else None
    lag_fraction = (round(_window_lag_fraction(trailing_start, end, horizon), 3)
                    if horizon else None)
    mostly_unpublished = lag_fraction is not None and lag_fraction >= _LAG_MOSTLY

    warnings: List[Dict[str, Any]] = []
    if lag_fraction:
        share = f"{lag_fraction * 100:.0f}%"
        if mostly_unpublished:
            warnings.append(warning(
                "reporting_lag",
                f"USASpending has published contract obligations only through "
                f"{horizon}, and {share} of the {months}-month window "
                f"({trailing_start} to {end}) falls after that date. The "
                f"trailing total measures what has been published, not what "
                f"was awarded, so no signal is reported: this is a gap in the "
                f"record rather than a change in the company's federal "
                f"business.",
                data_horizon=horizon,
                window_lag_fraction=lag_fraction,
                window=f"{trailing_start}..{end}"))
        else:
            warnings.append(warning(
                "partial_reporting_lag",
                f"USASpending has published contract obligations only through "
                f"{horizon}. {share} of the trailing window falls after that "
                f"date while the year-ago comparison window "
                f"({prior_start} to {prior_end}) is fully published, so "
                f"yoy_change_pct is biased downward by roughly that share of "
                f"the period.",
                data_horizon=horizon,
                window_lag_fraction=lag_fraction,
                window=f"{trailing_start}..{end}"))

    signal = (None if mostly_unpublished
              else _gov_contracts_signal(trailing_total, prior_total, yoy_pct))

    # prior/agencies/count are supporting detail: missing any of them narrows
    # the answer without invalidating the trailing total.
    return {
        "success": True,
        "coverage": "partial" if (errors or mostly_unpublished) else "full",
        "company_name": company_name,
        "ticker": ticker,
        "period_months": months,
        "trailing_window": {"start": trailing_start, "end": end},
        "comparison_window": {"start": prior_start, "end": prior_end},
        "trailing_awards_usd": round(trailing_total, 2),
        "trailing_award_count": award_count,
        "prior_period_awards_usd": round(prior_total, 2) if prior_total is not None else None,
        "yoy_change_pct": yoy_pct,
        "data_horizon": horizon,
        "window_lag_fraction": lag_fraction,
        "signal": signal,
        "top_agencies": top_agencies,
        "source": "usaspending.gov",
        "basis": ("Contract obligations (period flow) via spending_over_time — "
                  "not multi-year contract ceiling values. yoy_change_pct "
                  "compares the trailing window against the same calendar "
                  "months one year earlier, never against the window "
                  "immediately before it: federal obligations peak at the "
                  "30 September fiscal year end, so a sequential change "
                  "measures the calendar."),
        "warnings": warnings,
        "degraded": degraded,
        "partial_errors": errors,
    }


# ---------------------------------------------------------------------------
# Policy / legislative signals via GovTrack (+ Congress.gov if key set)
# ---------------------------------------------------------------------------

SECTOR_BILL_KEYWORDS: Dict[str, List[str]] = {
    "Technology":             ["artificial intelligence", "semiconductor", "chip export", "data privacy"],
    "Basic Materials":        ["critical minerals", "mining regulation", "carbon border"],
    "Communication Services": ["broadband", "spectrum", "social media regulation"],
    "Consumer Cyclical":      ["electric vehicle", "consumer protection", "trade tariff"],
    "Consumer Defensive":     ["food safety", "agriculture subsidy", "trade"],
    "Energy":                 ["clean energy", "LNG export", "offshore wind", "nuclear", "carbon tax"],
    "Financial Services":     ["banking regulation", "cryptocurrency", "stablecoin", "SEC"],
    "Healthcare":             ["drug pricing", "Medicare negotiation", "biosimilar", "FDA"],
    "Industrials":            ["infrastructure", "reshoring", "defense procurement", "NDAA"],
    "Real Estate":            ["housing policy", "interest rate", "zoning"],
    "Utilities":              ["grid infrastructure", "nuclear energy", "clean power"],
    "Defense":                ["NDAA", "defense authorization", "military procurement"],
}

# yfinance industry -> a keyword sector that its GICS sector cannot reach.
#
# "Defense" has always been in SECTOR_BILL_KEYWORDS and has never been
# reachable: yfinance files every prime under sector "Industrials", so LMT was
# researched on ["infrastructure", "reshoring", "defense procurement"] and came
# back with the Restoring the Death Penalty in DC Act and the Native American
# Housing Assistance Modernization Act. The NDAA -- the one bill that moves a
# defense prime's revenue line -- was never searched for.
#
# Deliberately narrow. An override only belongs here when the provider's
# taxonomy has no way to express the distinction, never as a second opinion
# about a sector the provider named correctly.
INDUSTRY_SECTOR_OVERRIDES: Dict[str, str] = {
    "aerospace & defense": "Defense",
}

# See the comment on the request itself: GovTrack's cold path is ~18s.
_GOVTRACK_TIMEOUT_S = 25
_POLICY_SIGNALS_TIMEOUT_S = 60.0

# Bills fetched per keyword, and rows carried in `bills`. `bill_count`
# describes the whole matched set, `rows_returned` describes this page, and
# `truncated` says when they differ -- the rule
# testing/test_counts_survive_paging.py holds the SEC tools to, and which
# get_congress_trades on this server already follows.
_BILL_ROW_CAP = 10

# Pro-industry vs anti-industry bill language. Complete words/conjugations only
# (word-boundary matched) — partial stems like "fund"/"invest"/"ban" caused
# false friends ("refund", "investigation", "banking"). Multi-word phrases are
# matched as substrings. Ambiguous terms (tax/tariff/regulation/mandate) are
# deliberately omitted: their polarity depends on the company's position.
_BILL_BULLISH = frozenset([
    "funding", "appropriation", "appropriations",
    "authorize", "authorized", "authorization", "reauthorization",
    "investment", "investments",
    "incentive", "incentives", "tax credit", "credit", "credits",
    "subsidy", "subsidies", "subsidize",
    "grant", "grants",
    "research", "development",
    "support", "supports",
    "modernize", "modernization",
    "promote", "promotion",
    "expand", "expansion",
    "innovation", "innovate",
    "manufacturing", "reshoring", "domestic manufacturing",
    "rebuild", "strengthen", "strengthening",
    "deregulation", "streamline", "streamlining",
    "exemption", "exempt",
])
_BILL_BEARISH = frozenset([
    "ban", "bans", "banned",
    "restrict", "restriction", "restrictions",
    "prohibit", "prohibits", "prohibition",
    "penalty", "penalties",
    "investigation", "investigations",
    "antitrust",
    "price cap", "price control", "price controls",
    "sanction", "sanctions",
    "moratorium",
    "export ban", "export control", "export controls", "export restriction",
    "windfall",
    "breakup",
])

# GovTrack/Congress.gov status -> probability weight (likelihood of taking effect).
_GOVTRACK_STATUSES = {
    "enacted_signed": 1.0, "enacted_veto_override": 1.0, "enacted_tendayrule": 1.0,
    "passed_bill": 0.70,
    "passed_house": 0.50, "passed_senate": 0.50,
    "pass_over_house": 0.50, "pass_over_senate": 0.50,
    "passed_resolution": 0.45, "passed_simpleres": 0.40, "passed_concurrentres": 0.40,
    "reported": 0.25, "referred": 0.10, "introduced": 0.08,
}


def _score_bill_title(title: str, status: str) -> float:
    """Net polarity of a bill (word-boundary matched) weighted by status."""
    pos = count_matches(title, _BILL_BULLISH)
    neg = count_matches(title, _BILL_BEARISH)
    weight = _GOVTRACK_STATUSES.get((status or "").lower(), 0.08)
    return (pos - neg) * weight


def _fetch_keywords(fetch_one, keywords: List[str]
                    ) -> Tuple[List[Dict], List[str], List[str]]:
    """Run `fetch_one(keyword)` over every keyword, concurrently.

    (bills, errors, keywords_queried). Concurrent because the alternative was
    a cap: both fetchers used `keywords[:3]` while the response reported the
    whole mapping in `keywords_searched`, so for Industrials the response said
    it had looked for "NDAA" and it never had. Four sequential 10s requests
    across two congresses do not fit the handler's 30s budget; four concurrent
    ones do.

    A keyword whose query failed is named in `errors`. Swallowing the
    exception made "GovTrack is down" and "no bill mentions semiconductors"
    the same answer.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not keywords:
        return [], [], []
    with ThreadPoolExecutor(max_workers=min(len(keywords), 6)) as pool:
        results = list(pool.map(fetch_one, keywords))

    bills: List[Dict] = []
    errors: List[str] = []
    seen: set = set()
    # Merged in keyword order rather than completion order so the same inputs
    # produce the same list.
    for rows, errs in results:
        errors += errs
        for row in rows:
            key = row.get("link") or row.get("title")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            bills.append(row)
    return bills, errors, list(keywords)


def _govtrack_fetch_bills(keywords: List[str], congress: int,
                           limit: int = 8) -> Tuple[List[Dict], List[str], List[str]]:
    """(bills, errors, keywords_queried) from GovTrack, one query per keyword."""
    import requests

    def one(kw: str) -> Tuple[List[Dict], List[str]]:
        rows: List[Dict] = []
        try:
            resp = requests.get(
                "https://www.govtrack.us/api/v2/bill/",
                params={"q": kw, "congress": congress,
                        "order_by": "-introduced_date", "limit": limit},
                headers={"User-Agent": "Mozilla/5.0"},
                # Measured 2026-08-26: a cold GovTrack query answers in ~18s
                # and the same query 0.6s once warm. At the old 10s every
                # first call of a session timed out, and the tool reported
                # `provider_unavailable` about a provider that was up. The
                # queries run concurrently, so this bounds one round rather
                # than the sum of them.
                timeout=_GOVTRACK_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            for obj in data.get("objects", []):
                # GovTrack bill objects have no "id" field; dedup on the unique
                # link (fall back to title). The old "id" key was always None,
                # which silently dropped every bill.
                rows.append({
                    "title": obj.get("title", ""),
                    "short_title": obj.get("short_title", ""),
                    "status": obj.get("current_status", "introduced"),
                    "introduced_date": obj.get("introduced_date", ""),
                    # current_status_date = latest activity; best recency signal
                    "activity_date": obj.get("current_status_date")
                                     or obj.get("introduced_date", ""),
                    "link": obj.get("link", ""),
                    "congress": congress,
                    "source": "govtrack",
                    # Which query surfaced this bill. GovTrack's `q` searches
                    # full text, so a bill can arrive on a keyword its title
                    # never mentions -- naming the keyword is what lets a
                    # reader discount the Native American Housing Assistance
                    # Modernization Act arriving on "infrastructure".
                    "matched_keyword": kw,
                })
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            return rows, [f"govtrack {kw!r} (congress {congress}): "
                          f"{type(exc).__name__}: {str(exc)[:120]}"]
        return rows, []

    return _fetch_keywords(one, keywords)


def _congress_api_fetch_bills(keywords: List[str], congress: int,
                               api_key: str,
                               limit: int = 10) -> Tuple[List[Dict], List[str], List[str]]:
    """(bills, errors, keywords_queried) — same contract as _govtrack_fetch_bills."""
    import requests

    def one(kw: str) -> Tuple[List[Dict], List[str]]:
        rows: List[Dict] = []
        try:
            resp = requests.get(
                "https://api.congress.gov/v3/bill",
                params={"q": kw, "congress": congress, "format": "json",
                        "api_key": api_key, "limit": limit},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12,
            )
            resp.raise_for_status()
            data = resp.json()
            for b in data.get("bills", []):
                title = b.get("title", "")
                latest = b.get("latestAction", {})
                action_text = (latest.get("text") or "").lower()
                if "became public law" in action_text:
                    status = "enacted_signed"
                elif "passed" in action_text:
                    status = "passed_bill"
                elif "referred" in action_text:
                    status = "referred"
                else:
                    status = "introduced"
                rows.append({
                    "title": title,
                    "short_title": "",
                    "status": status,
                    "introduced_date": latest.get("actionDate", ""),
                    "activity_date": latest.get("actionDate", ""),
                    "link": "",
                    "congress": congress,
                    "source": "congress.gov",
                    "matched_keyword": kw,
                })
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            return rows, [f"congress.gov {kw!r} (congress {congress}): "
                          f"{type(exc).__name__}: {str(exc)[:120]}"]
        return rows, []

    return _fetch_keywords(one, keywords)


def _fetch_policy_signals(ticker: str, sector: str,
                           lookback_days: int) -> Dict[str, Any]:
    """Legislative climate for a sector.

    Every failure mode here used to look like "no relevant bills found": an
    unknown ticker, a sector with no keyword mapping (which silently answered
    with the technology/semiconductor/defense default), and GovTrack being
    unreachable. Each is now named.

    Three labels were wrong on top of that, all found on LMT. The sector was
    auto-detected as "Industrials" because that is what yfinance calls a
    defense prime, so the Defense keyword set was unreachable. The response
    reported four keywords in `keywords_searched` while the fetcher used three,
    dropping "NDAA". And `bill_count: 21` shipped beside ten bills with no
    truncation flag, so the `total_score` behind the verdict was summed over
    rows the caller could not see.
    """
    sector_source = "caller"
    sector_reported = sector or None
    base = {"ticker": ticker, "sector": sector or None,
            "sector_reported_by_provider": sector_reported,
            "sector_source": sector_source,
            "lookback_days": lookback_days, "bill_count": None,
            "rows_returned": None, "truncated": None,
            "bills": [], "signal": None}

    if not sector:
        try:
            resolved = _resolve_ticker(ticker)
        except LookupFailure as exc:
            return _lookup_failed(exc, **base)
        sector = resolved["sector"]
        sector_reported = sector or None
        industry = resolved.get("industry") or ""
        if not sector:
            return {
                **base, "success": False, "coverage": "not_covered",
                "reason": "sector_unresolved",
                "error": (f"no sector for '{ticker}' in the quote response "
                          f"(ETFs and indices carry none). Pass an explicit "
                          f"`sector` to search legislation."),
                "sectors_supported": sorted(SECTOR_BILL_KEYWORDS),
            }
        override = INDUSTRY_SECTOR_OVERRIDES.get(industry.strip().lower())
        if override:
            sector_source = (f"industry {industry!r} overrides provider sector "
                             f"{sector!r}")
            sector = override
        else:
            sector_source = f"provider sector for {ticker}"
        base.update({"sector": sector,
                     "sector_reported_by_provider": sector_reported,
                     "sector_source": sector_source})

    keywords = SECTOR_BILL_KEYWORDS.get(sector)
    if not keywords:
        # The old default keyword set answered about semiconductors and
        # defense for any sector it did not recognise.
        return {
            **base, "success": False, "coverage": "not_covered",
            "reason": "sector_not_covered",
            "error": (f"no bill keywords are mapped for sector "
                      f"{sector!r}; refusing to answer from another sector's "
                      f"keywords."),
            "sectors_supported": sorted(SECTOR_BILL_KEYWORDS),
        }

    # Determine current and prior congress
    year = datetime.now().year
    start_year = year if year % 2 == 1 else year - 1
    current_congress = (start_year - 1789) // 2 + 1
    prior_congress = current_congress - 1

    api_key = os.environ.get("CONGRESS_API_KEY", "")
    source = "congress.gov" if api_key else "govtrack.us"
    degraded: List[str] = []
    if not api_key:
        degraded.append(
            "CONGRESS_API_KEY unset — Congress.gov was not queried; this "
            "answer is from GovTrack alone.")

    fetch_errors: List[str] = []
    # Only keywords actually put to the provider reach `keywords_searched`.
    # Echoing the mapping there was the lie: it claimed a search for the NDAA
    # that never happened.
    queried: List[str] = []
    if api_key:
        bills, errs, kws = _congress_api_fetch_bills(keywords, current_congress, api_key)
        fetch_errors += errs
        queried += kws
        if not bills:
            bills, errs, kws = _congress_api_fetch_bills(keywords, prior_congress, api_key)
            fetch_errors += errs
            queried += kws
    else:
        bills, errs, kws = _govtrack_fetch_bills(keywords, current_congress)
        fetch_errors += errs
        queried += kws
        if len(bills) < 3:
            more, errs, kws = _govtrack_fetch_bills(keywords, prior_congress)
            bills += more
            fetch_errors += errs
            queried += kws
    keywords_searched = list(dict.fromkeys(queried))
    base["sector"] = sector

    if not bills and fetch_errors:
        # Every query the provider was asked errored. An empty bill list here
        # means the provider is unreachable, not that Congress is idle.
        return {
            **base, "success": False,
            "coverage": "not_covered", "reason": "provider_unavailable",
            "error": (f"{source} returned no usable response for any of "
                      f"{keywords_searched}: " + "; ".join(fetch_errors)),
            "keywords_searched": keywords_searched,
            "keywords_mapped": keywords, "source": source,
            "degraded": degraded,
        }

    # Apply lookback_days: keep bills with recent legislative activity. Bills with
    # an unparseable/missing date are kept (don't over-filter on bad metadata).
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    def _recent(b: Dict) -> bool:
        d = parse_news_date(b.get("activity_date", "") or b.get("introduced_date", ""))
        return d is None or d >= cutoff
    bills = [b for b in bills if _recent(b)]

    coverage = "partial" if (degraded or fetch_errors) else "full"

    if not bills:
        # The provider answered; nothing it returned matched. A genuine empty.
        return {
            "success": True,
            "coverage": coverage,
            "ticker": ticker, "sector": sector,
            "sector_reported_by_provider": sector_reported,
            "sector_source": sector_source,
            "lookback_days": lookback_days,
            "keywords_searched": keywords_searched,
            "keywords_mapped": keywords,
            "bill_count": 0,
            "rows_returned": 0,
            "truncated": False,
            "bills": [],
            "signal": "neutral",
            "signal_basis": (f"{source} returned no bill matching "
                             f"{keywords_searched} with activity in the last "
                             f"{lookback_days} days"),
            "source": source,
            "degraded": degraded,
            "partial_errors": fetch_errors,
        }

    # Score each bill
    for b in bills:
        b["score"] = _score_bill_title(b["title"], b["status"])

    total_score = sum(b["score"] for b in bills)

    if total_score > 0.5:
        signal = "bullish"
        direction = "net positive"
    elif total_score < -0.5:
        signal = "bearish"
        direction = "net negative (restrictions/controls)"
    else:
        signal = "neutral"
        direction = "mixed or low probability"

    # Sort by abs score, then truncate -- counted first, so `bill_count`
    # describes the matched set and not the page.
    bills_out = sorted(bills, key=lambda b: -abs(b["score"]))[:_BILL_ROW_CAP]

    basis = (
        f"{direction}: title-word polarity weighted by enactment probability, "
        f"summed over all {len(bills)} bills matching {keywords_searched} "
        f"with activity in the last {lookback_days} days, total_score "
        f"{round(total_score, 3)}. Each bill names the keyword that surfaced "
        f"it in `matched_keyword`; the provider searches full text, so a bill "
        f"can match on a keyword its title never uses.")

    return {
        "success": True,
        "coverage": coverage,
        "ticker": ticker, "sector": sector,
        "sector_reported_by_provider": sector_reported,
        "sector_source": sector_source,
        "lookback_days": lookback_days,
        "keywords_searched": keywords_searched,
        "keywords_mapped": keywords,
        "bill_count": len(bills),
        "rows_returned": len(bills_out),
        "truncated": len(bills_out) < len(bills),
        "total_score": round(total_score, 3),
        "signal": signal,
        "signal_basis": basis,
        "bills": bills_out,
        "source": source,
        "degraded": degraded,
        "partial_errors": fetch_errors,
    }


# ---------------------------------------------------------------------------
# Capex announcements via DuckDuckGo news
# ---------------------------------------------------------------------------

# Rows carried in the payload. `announcement_count` describes the whole set,
# `rows_returned` describes this page, and `truncated` says when they differ --
# the rule test_counts_survive_paging.py holds the SEC tools to.
_CAPEX_ROW_CAP = 8
# Distinct figures carried in `figures`. Kept above the row cap because the
# figures are what the total is built from and what a caller audits it with.
_CAPEX_FIGURE_CAP = 20


def _fetch_capex_announcements(ticker: str, company_name: str,
                                 lookback_days: int) -> Dict[str, Any]:
    try:
        from ddgs import DDGS
    except ImportError:
        return {"success": False, "coverage": "not_covered",
                "reason": "provider_unavailable",
                "error": "ddgs not installed — pip install ddgs",
                "ticker": ticker}

    # An explicit company_name is the caller asserting the entity; only derive
    # it when they did not, and fail loudly when the symbol does not resolve.
    if not company_name:
        try:
            company_name = _resolve_ticker(ticker)["name"]
        except LookupFailure as exc:
            return _lookup_failed(exc, ticker=ticker, company_name=None,
                                  lookback_days=lookback_days,
                                  announcement_count=None,
                                  capex_total_usd=None,
                                  signal="data_gap", announcements=[])

    # Sector-agnostic queries: the first two capture capex events in ANY sector
    # (retail distribution, energy refinery, pharma manufacturing, REIT
    # development) via universal frames; the third keeps industrial/tech specificity.
    queries = [
        f"{company_name} capital investment expansion announcement",
        f"{company_name} new facility construction billion",
        f"{company_name} factory plant data center capacity expand",
    ]

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    name_tokens = _company_name_tokens(company_name)
    all_articles: List[Dict] = []
    seen_titles: set = set()
    # A query that errored is not a query that found nothing; the two were
    # indistinguishable while every failure was swallowed by `continue`.
    query_errors: List[str] = []

    try:
        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = list(ddgs.news(query, max_results=10, timelimit=None))
                    for r in results:
                        title = r.get("title", "")
                        if not title or title in seen_titles:
                            continue
                        # Relevance: drop macro headlines that don't name the company.
                        if not _article_is_relevant(title + " " + r.get("body", ""), name_tokens):
                            continue
                        pub_date = parse_news_date(r.get("date", ""))
                        if pub_date is None or pub_date >= cutoff:
                            seen_titles.add(title)
                            all_articles.append(r)
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    query_errors.append(
                        f"{query!r}: {type(exc).__name__}: {str(exc)[:120]}")
                    continue
    except Exception as exc:
        return {
            "success": False, "coverage": "not_covered",
            "reason": "provider_unavailable",
            "error": (f"DuckDuckGo news search failed: {type(exc).__name__}: "
                      f"{str(exc)[:200]}"),
            "ticker": ticker, "company_name": company_name,
            "queries_tried": queries, "announcement_count": None,
            "signal": "data_gap", "announcements": [],
        }

    if query_errors and len(query_errors) == len(queries):
        return {
            "success": False, "coverage": "not_covered",
            "reason": "provider_unavailable",
            "error": ("every DuckDuckGo news query failed: "
                      + "; ".join(query_errors)),
            "ticker": ticker, "company_name": company_name,
            "queries_tried": queries, "announcement_count": None,
            "signal": "data_gap", "announcements": [],
        }

    if not all_articles:
        # A news corpus cannot assert absence: no matching article is not
        # evidence that no capex was announced, so this is an uncovered
        # lookup rather than a count of zero.
        return {
            "success": False, "coverage": "not_covered",
            "reason": "no_results",
            "error": (f"no news article naming {company_name} matched any of "
                      f"{len(queries)} capex queries in the last "
                      f"{lookback_days} days. Absence of news is not evidence "
                      f"that no capital investment was announced."),
            "ticker": ticker, "company_name": company_name,
            "lookback_days": lookback_days, "announcement_count": 0,
            "rows_returned": 0, "truncated": False,
            # Withheld, not zero, for the reason the error above gives.
            "capex_total_usd": None,
            "capex_total_basis": ("no total: no article was returned, and an "
                                  "empty news search is not a finding that no "
                                  "capital expenditure was announced."),
            "capex_figure_count": 0, "largest_capex_usd": None,
            "containment_detected": False, "cumulative_program_usd": None,
            "figure_count": 0, "figures": [], "amounts_by_category": {},
            "signal": "data_gap",
            "signal_basis": "no article to read a spending direction from",
            "queries_tried": queries,
            "partial_errors": query_errors,
            "announcements": [],
        }

    # Read every dollar figure in its own clause. One row per article (the
    # count/page contract of test_counts_survive_paging.py), one entry per
    # DISTINCT figure in `figures` -- a figure restated by five outlets is one
    # announcement, and it is now one announcement of a stated KIND.
    readings_by_amount: Dict[float, List[Dict[str, Any]]] = {}
    articles_by_amount: Dict[float, int] = {}
    per_article: List[Tuple[Dict, str, List[Dict[str, Any]]]] = []

    for r in all_articles:
        title = r.get("title", "")
        body = r.get("body", "")
        # The pipe is a clause boundary: without it a headline runs into the
        # first sentence of the body and the two get read as one claim.
        combined = f"{title} | {body}"
        figures = _figures_in_article(combined, name_tokens)
        per_article.append((r, combined, figures))
        for amt in {f["amount_usd"] for f in figures}:
            articles_by_amount[amt] = articles_by_amount.get(amt, 0) + 1
        for f in figures:
            readings_by_amount.setdefault(f["amount_usd"], []).append(
                dict(f, title=title[:120], url=r.get("url", ""),
                     date=r.get("date", "")))

    resolved: Dict[float, Dict[str, Any]] = {
        amt: dict(_resolve_figure(readings), mentions=articles_by_amount[amt])
        for amt, readings in readings_by_amount.items()
    }

    announcements = []
    for r, combined, figures in per_article:
        amounts = {f["amount_usd"] for f in figures}
        capex_here = [a for a in amounts
                      if resolved[a]["category"] == "capital_expenditure"]
        largest = max(amounts) if amounts else None
        body = r.get("body", "")
        announcements.append({
            "title": r.get("title", "")[:120],
            "date": r.get("date", ""),
            "url": r.get("url", ""),
            "largest_figure_usd": largest,
            "figure_category": resolved[largest]["category"] if amounts else None,
            "capex_amount_usd": max(capex_here) if capex_here else None,
            "mentions": articles_by_amount[largest] if amounts else None,
            "direction": _classify_capex_text(combined),
            "snippet": body[:200] if body else "",
        })

    # A total is only checkable if the rows it was built from survive the page
    # cap. Sorting purely by size buried Broadcom's $1.5B plant upgrade -- the
    # one figure in its whole corpus that WAS capex -- under six larger
    # headlines about bookings and debt.
    announcements.sort(key=lambda a: (a["capex_amount_usd"] is None,
                                      -(a["capex_amount_usd"] or 0),
                                      -(a["largest_figure_usd"] or 0)))

    by_category: Dict[str, Dict[str, Any]] = {}
    for amt, fig in sorted(resolved.items(), key=lambda kv: -kv[0]):
        slot = by_category.setdefault(
            fig["category"], {"figure_count": 0, "total_usd": 0.0, "amounts_usd": []})
        slot["figure_count"] += 1
        slot["total_usd"] += amt
        slot["amounts_usd"].append(amt)

    capex_amounts = sorted(
        (a for a, f in resolved.items() if f["category"] == "capital_expenditure"),
        reverse=True)
    cumulative_amounts = sorted(
        (a for a, f in resolved.items() if f["category"] == "cumulative_total"),
        reverse=True)
    cumulative_program = cumulative_amounts[0] if cumulative_amounts else None
    # "brings TSMC's U.S. investment to $265 billion" is the programme the
    # $100B announced that week sits INSIDE. Adding the two reported $365B for
    # $100B of news.
    containment = bool(cumulative_program) and any(
        a < cumulative_program for a in capex_amounts)

    other_slots = [(cat, slot) for cat, slot in by_category.items()
                   if cat != "capital_expenditure"]
    other = ", ".join(
        f"{cat} {_usd_short(slot['total_usd'])} across {slot['figure_count']} figure(s)"
        for cat, slot in other_slots)

    if capex_amounts:
        capex_total: Optional[float] = sum(capex_amounts)
        capex_basis = (
            f"sum of {len(capex_amounts)} distinct figure(s) the text attributes to "
            f"{company_name} spending on physical assets, drawn from "
            f"{len(all_articles)} article(s)."
            + (f" Figures in other categories are excluded, not missed: {other}."
               if other_slots else
               " Every dollar figure found was capital expenditure.")
        )
        if containment:
            capex_basis += (
                f" A cumulative programme figure of {_usd_short(cumulative_program)} "
                f"appears in the same corpus and is NOT added, because it already "
                f"contains the announcement(s) counted above."
            )
    else:
        capex_total = None
        capex_basis = (
            f"no total. Not one of {len(resolved)} dollar figure(s) across "
            f"{len(all_articles)} article(s) is attributable to {company_name} "
            f"spending on physical assets"
            + (f": {other}." if other_slots else ".")
            + " A news corpus cannot show that no capital expenditure was "
              "announced, so this is a withheld total and never a zero."
        )

    if capex_amounts:
        signal = _capex_signal([
            {"direction": _classify_capex_text(resolved[a]["context"]),
             "max_amount_usd": a}
            for a in capex_amounts])
        signal_basis = (
            f"derived from {len(capex_amounts)} capital-expenditure figure(s) only "
            f"({', '.join(_usd_short(a) for a in capex_amounts)})"
            + (f"; {other} did not contribute."
               if other_slots else "; no figure of another kind was found.")
        )
    else:
        signal = "data_gap"
        signal_basis = (
            f"withheld. {len(resolved)} dollar figure(s) were found and none is "
            f"capital expenditure"
            + (f" ({other})" if other_slots else "")
            + ", so there is nothing to read a spending direction from. A "
              "verdict drawn from customer deals or financing would be a "
              "verdict about the opposite of capex."
        )

    rows = announcements[:_CAPEX_ROW_CAP]
    figures_out = [
        dict(resolved[a], amount_usd=a)
        for a in sorted(resolved, reverse=True)[:_CAPEX_FIGURE_CAP]
    ]

    return {
        "success": True,
        "coverage": "full" if not query_errors else "partial",
        "ticker": ticker, "company_name": company_name,
        "lookback_days": lookback_days,
        "announcement_count": len(announcements),
        "rows_returned": len(rows),
        "truncated": len(rows) < len(announcements),
        "capex_total_usd": capex_total,
        "capex_total_basis": capex_basis,
        "capex_figure_count": len(capex_amounts),
        "largest_capex_usd": capex_amounts[0] if capex_amounts else None,
        "containment_detected": containment,
        "cumulative_program_usd": cumulative_program,
        "figure_count": len(resolved),
        "figures": figures_out,
        "amounts_by_category": by_category,
        "signal": signal,
        "signal_basis": signal_basis,
        "queries_tried": queries,
        "partial_errors": query_errors,
        "announcements": rows,
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class AltDataServer:
    def __init__(self):
        self.server = Server("altdata")
        self._setup_handlers()

    def _setup_handlers(self):
        parent = self

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_taiwan_monthly_revenue",
                    description=(
                        "Monthly revenue for Taiwan-listed companies via FinMind "
                        "(TWSE feed). Key codes: TSMC=2330, Foxconn=2317, "
                        "MediaTek=2454, ASE Group=3711. "
                        "Returns NTD millions per month + YoY%. `date` is the first "
                        "day of the month the revenue DESCRIBES and always agrees "
                        "with `year`/`month`; FinMind's announcement stamp (a month "
                        "later) is kept separately as `announced_date`. "
                        "Every requested code "
                        "failing returns success=false, coverage='not_covered'; some "
                        "failing returns coverage='partial' with codes_failed listed. "
                        "An unset FINMIND_TOKEN is named in 'degraded' (anonymous tier, "
                        "shared per-IP daily quota)."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["company_codes"],
                        "properties": {
                            "company_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Taiwan stock codes (e.g. ['2330', '2317']). Max 5.",
                            },
                            "months": {
                                "type": "integer",
                                "default": 6,
                                "description": "Number of recent months to return.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_job_postings_count",
                    description=(
                        "Count open job postings via Greenhouse, Lever, or Workday — "
                        "auto-discovered in parallel for any company (no hardcoded list). "
                        "For Workday companies (Oracle, Salesforce, ServiceNow, etc.), "
                        "tenant and URL are discovered automatically. "
                        "Fallback: ATS fingerprinting via the company's careers page. "
                        "A reachable board with no open roles returns success with "
                        "total_postings 0. When no provider answers (proprietary portal, "
                        "or the company does not exist) the tool returns success=false, "
                        "coverage='not_covered' and names every provider tried — that is "
                        "a failed lookup, never a count of zero. 'department_coverage' "
                        "is reported separately: 'not_covered' means the board exposes "
                        "no department breakdown, not that the roles are uncategorised."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["company_slug"],
                        "properties": {
                            "company_slug": {
                                "type": "string",
                                "description": "Lowercase company identifier (e.g. 'nvidia', 'oracle', 'palo-alto-networks').",
                            },
                            "ats": {
                                "type": "string",
                                "enum": ["greenhouse", "lever", "workday"],
                                "default": "greenhouse",
                                "description": "Preferred ATS. Auto-falls back through greenhouse→lever→workday.",
                            },
                            "department_filter": {
                                "type": "string",
                                "description": "Optional: filter to departments containing this string.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_government_contracts",
                    description=(
                        "Federal contract (and optional grant) obligations to a company "
                        "via USASpending.gov — free, no auth required. Uses spending_over_time "
                        "for true period FLOW (not multi-year contract ceiling values). "
                        "Returns trailing-period obligations, the SAME calendar months "
                        "one year earlier, the YoY change between them, top awarding "
                        "agencies, and award count. The comparison is never the window "
                        "immediately before the trailing one: federal obligations peak "
                        "at the 30 September fiscal year end, so a sequential change "
                        "would measure the calendar. "
                        "Signal: YoY > +15% = bullish; YoY < -15% = bearish; "
                        "< $10M total = not_applicable (consumer/B2C company). "
                        "Agencies publish contract actions months in arrears (DoD "
                        "actions are withheld from public FPDS for 90 days), so every "
                        "response carries 'data_horizon' — the last date USASpending "
                        "has actually published, measured government-wide at call "
                        "time — and 'window_lag_fraction'. When most of the window "
                        "falls past that horizon no signal is returned at all: it is "
                        "coverage='partial' plus a 'reporting_lag' warning, because "
                        "an unpublished quarter is a gap in the record rather than a "
                        "collapse in the company's federal business. "
                        "A ticker the quote provider cannot resolve returns success=false "
                        "(USASpending answers '$0 of federal business' for any string, so "
                        "an unknown company would otherwise read as a real zero). If the "
                        "prior-period, agency or count query fails the trailing total is "
                        "still returned with coverage='partial' and the failures named. "
                        "Most relevant for: defense (LMT, RTX, NOC), cloud (AMZN, MSFT), "
                        "IT services, biotech (with include_grants=true)."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["ticker"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "company_name": {
                                "type": "string",
                                "description": "Optional override for USASpending search name "
                                               "(auto-derived from yfinance if omitted).",
                            },
                            "months": {
                                "type": "integer",
                                "default": 12,
                                "description": "Trailing period in months.",
                            },
                            "include_grants": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include federal grants (NIH, NSF, etc.) — useful for biotech.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_policy_signals",
                    description=(
                        "Legislative climate for a company's sector via GovTrack "
                        "(free, no auth) or Congress.gov (set CONGRESS_API_KEY). "
                        "Finds recent bills matching sector-specific keywords, scores "
                        "them by title sentiment and status probability, returns "
                        "bullish / bearish / neutral legislative signal. "
                        "Sector is auto-detected from yfinance if not provided; an "
                        "unresolvable ticker, a ticker with no sector, or a sector with "
                        "no keyword mapping returns success=false rather than answering "
                        "from another sector's keywords. yfinance has no Defense sector, "
                        "so an 'Aerospace & Defense' industry is mapped to it and "
                        "'sector_source' says so alongside "
                        "'sector_reported_by_provider'. Without CONGRESS_API_KEY the "
                        "answer is GovTrack-only: coverage='partial' and the missing key "
                        "is named in 'degraded'. "
                        "Bills are found by full-text keyword search, so a bill can match "
                        "on a keyword its title never uses -- every row names its "
                        "'matched_keyword' and 'signal_basis' states what was summed. "
                        "'bill_count' counts the whole matched set, 'rows_returned' the "
                        "page returned, and 'truncated' says when they differ. "
                        "Most relevant for: semiconductors (CHIPS Act), defense (NDAA), "
                        "pharma (drug pricing), energy (IRA credits), fintech (crypto regs)."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["ticker"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "sector": {
                                "type": "string",
                                "description": "Optional sector override "
                                               "(Technology, Healthcare, Defense, Energy, "
                                               "Financial Services, Industrials, Consumer Cyclical, etc.).",
                            },
                            "lookback_days": {
                                "type": "integer",
                                "default": 180,
                                "description": "Days of legislative history.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_capex_announcements",
                    description=(
                        "Search recent news for capital investment announcements "
                        "(factories, data centers, R&D facilities, major equipment). "
                        "EVERY dollar figure is classified before it is used: "
                        "capital_expenditure / customer_or_partner_deal / financing / "
                        "guarantee / third_party_capital / equity_or_ma / "
                        "cumulative_total / unclassified, each in `figures` with the "
                        "words that placed it. A customer deal is REVENUE and a "
                        "financing is DEBT — opposite signals to capex about the same "
                        "company — so read the category, never the raw number. "
                        "capex_total_usd sums ONLY distinct capital_expenditure "
                        "figures; it is null (never 0) when no figure can be "
                        "attributed to the company spending on physical assets, which "
                        "is the common case for a company whose news is about demand. "
                        "capex_total_basis names every figure excluded and why. "
                        "containment_detected / cumulative_program_usd flag a running "
                        "programme total ('brings its US investment to $265bn') that "
                        "already contains the announcement, so the two are not added. "
                        "signal (bullish / bearish / neutral / data_gap) is derived "
                        "from capital_expenditure figures alone and is data_gap when "
                        "there are none. "
                        "Returns success=false with reason='unknown_ticker' for a symbol "
                        "the quote provider cannot resolve, and reason='no_results' when "
                        "no article matched — a news corpus cannot assert that no capex "
                        "was announced, so that is never reported as a zero. "
                        "announcement_count counts every matching article; rows_returned "
                        "and truncated describe the announcements list. "
                        "Uses DuckDuckGo news (ddgs). Best for: semiconductors, industrials, "
                        "energy, cloud hyperscalers."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["ticker"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "company_name": {
                                "type": "string",
                                "description": "Optional: override company name for news search.",
                            },
                            "lookback_days": {
                                "type": "integer",
                                "default": 90,
                                "description": "Days of news history to search.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_congress_trades",
                    description=(
                        "Congressional stock trades from official STOCK Act "
                        "disclosures (House Clerk PTRs + Senate eFD), served from "
                        "a local store. These are TRANSACTIONS, NOT HOLDINGS: "
                        "Congress does not publish current positions. Amounts are "
                        "BRACKETS — every row carries amount_min and amount_max and "
                        "there is no midpoint. Members file up to 45 days after "
                        "trading, so transaction_date and filed_date differ. ALWAYS "
                        "read 'coverage': when coverage.complete is false the answer "
                        "is drawn from an incomplete record and an absent ticker does "
                        "NOT mean it was not traded. Populate the store with "
                        "`python -m tools.altdata_server.congress_sync --house 2026 "
                        "--senate`."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string",
                                       "description": "Filter to one ticker. Bonds and funds have no ticker."},
                            "member": {"type": "string",
                                       "description": "Filter by member name, matched loosely. The members matched are returned."},
                            "chamber": {"type": "string", "enum": ["house", "senate"]},
                            "since": {"type": "string",
                                      "description": "Earliest transaction date, YYYY-MM-DD."},
                            "until": {"type": "string",
                                      "description": "Latest transaction date, YYYY-MM-DD."},
                            "limit": {"type": "integer", "default": 200},
                        },
                    },
                ),
                Tool(
                    name="get_congress_leaderboard",
                    description=(
                        "Aggregate congressional trading activity from the local "
                        "store: most-traded tickers or most-active members. Totals "
                        "are bracketed sums (sum of lower bounds, sum of upper "
                        "bounds), never midpoints. Rows with no ticker are excluded "
                        "from the ticker leaderboard rather than grouped, since "
                        "bonds and private funds have no symbol. Read 'coverage'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["tickers", "members"],
                                     "default": "tickers"},
                            "since": {"type": "string",
                                      "description": "Earliest transaction date, YYYY-MM-DD."},
                            "chamber": {"type": "string", "enum": ["house", "senate"]},
                            "limit": {"type": "integer", "default": 25},
                        },
                    },
                ),
                Tool(
                    name="get_congress_holdings",
                    description=(
                        "Congressional ASSET HOLDINGS from annual financial "
                        "disclosures — the closest thing to positions that exists, "
                        "and the only place holdings are published at all. READ THIS "
                        "BEFORE USING: an annual report covers assets held at some "
                        "point DURING the calendar year it names, valued in brackets, "
                        "and is filed months after that year ends. A row is NOT a "
                        "current position — the member may have exited before filing, "
                        "and trades disclosed since are not reflected in it. Values "
                        "are brackets with no midpoint. Holdings the filer could not "
                        "price (state pensions, family trusts) carry no bounds and are "
                        "counted in unpriced_count rather than summed as zero. Roughly "
                        "a third of rows are Excepted Investment Funds whose underlying "
                        "holdings are legally not itemised. Senate only at present; "
                        "House annual reports are PDFs and are not yet ingested."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string",
                                       "description": "Who disclosed holding this ticker."},
                            "member": {"type": "string",
                                       "description": "One member's disclosed holdings, matched loosely."},
                            "limit": {"type": "integer", "default": 200},
                        },
                    },
                ),
                Tool(
                    name="get_congress_coverage",
                    description=(
                        "What the congressional disclosure store actually holds: "
                        "filings ingested, how many parsed, and how many could not "
                        "be read because they were filed on paper and scanned. Call "
                        "this before treating any empty congressional result as an "
                        "absence rather than a gap."
                    ),
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        @annotating(
            "altdata",
            per_tool={
                "get_taiwan_monthly_revenue": "FinMind (TWSE)",
                "get_job_postings_count": "Greenhouse / Lever / Workday boards",
                "get_government_contracts": "USAspending.gov",
                "get_policy_signals": "GovTrack / Congress.gov",
                "get_capex_announcements": "company newsroom / press releases",
                "get_congress_trades": "US House Clerk / Senate eFD",
                "get_congress_holdings": "US Senate eFD",
                "get_congress_leaderboard": "US House Clerk / Senate eFD",
                "get_congress_coverage": "Nemo congressional store",
            },
warnings_per_tool={
                "get_congress_holdings": [
                    warning("not_current_positions",
                            "Annual disclosures cover assets held at some "
                            "point DURING the year they name and are filed "
                            "months after it ends. A row is not a current "
                            "position."),
                    warning("senate_only",
                            "Holdings are Senate-only. House annual reports "
                            "are PDFs that are not yet ingested, so absence "
                            "here is not absence of a holding."),
                ],
                "get_congress_trades": [
                    warning("disclosure_lag",
                            "Members file up to 45 days after trading, so "
                            "transaction_date and filed_date differ."),
                ],
            })
        async def call_tool(name: str, args: Dict[str, Any]):
            if name == "get_taiwan_monthly_revenue":
                return await parent.taiwan_monthly_revenue(args)
            if name == "get_job_postings_count":
                return await parent.job_postings_count(args)
            if name == "get_government_contracts":
                return await parent.government_contracts(args)
            if name == "get_policy_signals":
                return await parent.policy_signals(args)
            if name == "get_capex_announcements":
                return await parent.capex_announcements(args)
            if name == "get_congress_trades":
                return await parent.congress_trades(args)
            if name == "get_congress_leaderboard":
                return await parent.congress_leaderboard(args)
            if name == "get_congress_holdings":
                return await parent.congress_holdings(args)
            if name == "get_congress_coverage":
                return await parent.congress_coverage(args)
            return _err(name, f"unknown tool: {name}")

    # -----------------------------------------------------------------------
    # Existing tool handlers
    # -----------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Congressional disclosures, served from the local store.
    #
    # These read rather than fetch. Parsing a House PTR is an HTTP round trip
    # plus a PDF parse, which on demand bought about twenty filings per call
    # and made every answer partial. Ingestion belongs to congress_sync.
    # ------------------------------------------------------------------

    _EMPTY_STORE = (
        "The congressional disclosure store is empty, so this is not a "
        "statement about what was traded. Populate it with: "
        "python -m tools.altdata_server.congress_sync --house 2026 --senate"
    )

    @staticmethod
    def _mark_empty(result: Dict[str, Any]) -> Dict[str, Any]:
        """An empty store must never read as an empty record."""
        if not result.get("coverage", {}).get("total"):
            result["store_empty"] = True
            result["note"] = f"{AltDataServer._EMPTY_STORE} {result.get('note', '')}"
        return result

    async def congress_trades(self, args: Dict[str, Any]) -> List[TextContent]:
        from . import congress_queries as queries

        try:
            if args.get("member"):
                # The ticker narrows the query, not the page. Filtering the
                # returned rows afterwards left `totals` and `per_member`
                # describing every trade the member made in anything, beside a
                # `transaction_count` describing one symbol.
                result = await asyncio.to_thread(
                    queries.member_activity, args["member"],
                    since=args.get("since"), limit=int(args.get("limit", 200)),
                    ticker=args.get("ticker"))
            else:
                result = await asyncio.to_thread(
                    queries.ticker_activity, args.get("ticker", ""),
                    since=args.get("since"), until=args.get("until"),
                    chamber=args.get("chamber"), limit=int(args.get("limit", 200)))
        except Exception as exc:  # noqa: BLE001 - surfaced, never masked
            return _err("get_congress_trades",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _dispatch("get_congress_trades", self._mark_empty(result))

    async def congress_leaderboard(self, args: Dict[str, Any]) -> List[TextContent]:
        from . import congress_queries as queries

        kind = args.get("kind", "tickers")
        try:
            if kind == "members":
                result = await asyncio.to_thread(
                    queries.most_active_members, since=args.get("since"),
                    limit=int(args.get("limit", 25)))
            else:
                result = await asyncio.to_thread(
                    queries.most_traded_tickers, since=args.get("since"),
                    chamber=args.get("chamber"), limit=int(args.get("limit", 25)))
        except Exception as exc:  # noqa: BLE001 - surfaced, never masked
            return _err("get_congress_leaderboard",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _dispatch("get_congress_leaderboard", self._mark_empty(result))

    async def congress_holdings(self, args: Dict[str, Any]) -> List[TextContent]:
        from . import congress_queries as queries

        if not args.get("ticker") and not args.get("member"):
            return _err("get_congress_holdings",
                        "pass a ticker or a member; holdings are only meaningful "
                        "scoped to one or the other")
        try:
            if args.get("member"):
                result = await asyncio.to_thread(
                    queries.member_holdings, args["member"],
                    limit=int(args.get("limit", 200)))
            else:
                result = await asyncio.to_thread(
                    queries.ticker_holdings, args["ticker"],
                    limit=int(args.get("limit", 200)))
        except Exception as exc:  # noqa: BLE001 - surfaced, never masked
            return _err("get_congress_holdings",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _dispatch("get_congress_holdings", self._mark_empty(result))

    async def congress_coverage(self, args: Dict[str, Any]) -> List[TextContent]:
        from . import congress_store as cstore

        try:
            overall = await asyncio.to_thread(cstore.coverage)
            per_chamber = {c: await asyncio.to_thread(cstore.coverage, c)
                           for c in ("house", "senate")}
        except Exception as exc:  # noqa: BLE001 - surfaced, never masked
            return _err("get_congress_coverage",
                        f"{type(exc).__name__}: {str(exc)[:200]}")

        note = ("filings_parsed against total is the only basis for reading an "
                "empty congressional result as an absence rather than a gap. "
                "'scanned' filings were filed on paper and carry no extractable "
                "text; they will not become readable on a retry.")
        if not overall["total"]:
            note = f"{self._EMPTY_STORE} {note}"
        overall["by_chamber"] = per_chamber
        overall["note"] = note
        overall["database"] = cstore.current_db_path()
        return _dispatch("get_congress_coverage", overall)

    async def taiwan_monthly_revenue(self, args: Dict[str, Any]) -> List[TextContent]:
        codes = args.get("company_codes", [])
        if not codes:
            return _err("get_taiwan_monthly_revenue", "company_codes is required")
        months = int(args.get("months", 6))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_fetch_taiwan_revenue_finmind, codes[:5], months),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return _err("get_taiwan_monthly_revenue", "FinMind request timed out after 30s")
        except Exception as exc:
            return _err("get_taiwan_monthly_revenue",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _dispatch("get_taiwan_monthly_revenue", result)

    async def job_postings_count(self, args: Dict[str, Any]) -> List[TextContent]:
        slug = str(args.get("company_slug", "")).strip().lower()
        if not slug:
            return _err("get_job_postings_count", "company_slug is required")
        ats = str(args.get("ats", "greenhouse"))
        dept_filter = args.get("department_filter")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_fetch_job_postings, slug, ats, dept_filter),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return _err("get_job_postings_count", "job postings request timed out after 30s")
        except Exception as exc:
            return _err("get_job_postings_count",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _dispatch("get_job_postings_count", result, slug.upper())

    # -----------------------------------------------------------------------
    # New tool handlers
    # -----------------------------------------------------------------------

    async def government_contracts(self, args: Dict[str, Any]) -> List[TextContent]:
        ticker = str(args.get("ticker", "")).upper()
        if not ticker:
            return _err("get_government_contracts", "ticker is required")
        company_name = str(args.get("company_name", ""))
        months = int(args.get("months", 12))
        include_grants = bool(args.get("include_grants", False))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _fetch_government_contracts, ticker, company_name,
                    months, include_grants,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            return _err("get_government_contracts", "USASpending.gov timed out after 45s", ticker)
        except Exception as exc:
            return _err("get_government_contracts",
                        f"{type(exc).__name__}: {str(exc)[:200]}", ticker)
        return _dispatch("get_government_contracts", result, ticker)

    async def policy_signals(self, args: Dict[str, Any]) -> List[TextContent]:
        ticker = str(args.get("ticker", "")).upper()
        if not ticker:
            return _err("get_policy_signals", "ticker is required")
        sector = str(args.get("sector", ""))
        lookback_days = int(args.get("lookback_days", 180))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _fetch_policy_signals, ticker, sector, lookback_days,
                ),
                # Two rounds at most -- the current congress, then the prior
                # one if the current returned almost nothing -- and each round
                # queries every keyword concurrently, so the budget is two
                # cold GovTrack requests plus parsing rather than one per
                # keyword.
                timeout=_POLICY_SIGNALS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _err("get_policy_signals",
                        f"GovTrack/Congress.gov timed out after "
                        f"{_POLICY_SIGNALS_TIMEOUT_S:.0f}s", ticker)
        except Exception as exc:
            return _err("get_policy_signals",
                        f"{type(exc).__name__}: {str(exc)[:200]}", ticker)
        return _dispatch("get_policy_signals", result, ticker)

    async def capex_announcements(self, args: Dict[str, Any]) -> List[TextContent]:
        ticker = str(args.get("ticker", "")).upper()
        if not ticker:
            return _err("get_capex_announcements", "ticker is required")
        company_name = str(args.get("company_name", ""))
        lookback_days = int(args.get("lookback_days", 90))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _fetch_capex_announcements, ticker, company_name, lookback_days,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            return _err("get_capex_announcements", "news search timed out after 45s", ticker)
        except Exception as exc:
            return _err("get_capex_announcements",
                        f"{type(exc).__name__}: {str(exc)[:200]}", ticker)
        return _dispatch("get_capex_announcements", result, ticker)

    async def run_server(self):
        async with stdio_server() as (read, write):
            await self.server.run(read, write,
                                  self.server.create_initialization_options())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("server", "http"):
        print("Usage: python -m tools.altdata_server.server [server|http]", file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == "http":
        # Streamable HTTP, for a host a client connects to rather than one
        # that spawns it. stdio stays the default for local use.
        from tools.mcp_http import run_http
        print("[altdata] starting over streamable HTTP", file=sys.stderr, flush=True)
        # Explicitly none. CONGRESS_API_KEY and FINMIND_TOKEN are real keys
        # this server reads, but their absence is a documented degradation
        # the tools report in their own output -- declaring them here would
        # mark a container that is working as designed unhealthy forever.
        run_http(AltDataServer().server, required_env=())
    else:
        print("[altdata] starting", file=sys.stderr, flush=True)
        srv = AltDataServer()
        asyncio.run(srv.run_server())
