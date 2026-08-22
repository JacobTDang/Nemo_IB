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

import asyncio
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.altdata_server.text_utils import (
    text_contains, count_matches, extract_dollar_amounts, parse_news_date,
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
    return {"name": name, "sector": info.get("sector") or ""}


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
    for url, timeout in ((f"{base}?content=true", 10), (base, 8)):
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
            for future in as_completed([gh_f, lv_f], timeout=20):
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
            parsed.append({"year": yr, "month": mo, "date": r["date"],
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


def _usa_obligations_total(company_name: str, start: str, end: str,
                            award_types: List[str]) -> float:
    """Sum real obligation FLOWS over a window via spending_over_time.

    Unlike spending_by_award (which returns multi-year contract ceiling
    values), spending_over_time returns the obligated amount per period.
    The time_period filter bounds the window, so summing the buckets gives
    the true period flow.
    """
    import requests
    payload = {"group": "month",
               "filters": _usa_filters(company_name, start, end, award_types)}
    resp = requests.post(_SOT_URL, json=payload, timeout=_USA_TIMEOUT, headers=_USA_HEADERS)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return float(sum((r.get("aggregated_amount") or 0) for r in results))


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


def _gov_windows(today, months: int):
    """Pure: (trailing_start, trailing_end, prior_start, prior_end) date strings.
    Calendar-accurate spans (365/12 days per month, not 30) so '12 months' is a
    true year for the YoY. USASpending time_period is inclusive on both ends,
    so the prior window ends one day before the trailing window starts (no
    boundary-day double count)."""
    span = timedelta(days=round(months * 365 / 12))
    t_start = today - span
    return (t_start.strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
            (t_start - span).strftime("%Y-%m-%d"),
            (t_start - timedelta(days=1)).strftime("%Y-%m-%d"))


def _fetch_government_contracts(ticker: str, company_name: str,
                                  months: int, include_grants: bool) -> Dict[str, Any]:
    base = {"ticker": ticker, "company_name": company_name or None,
            "period_months": months, "trailing_awards_usd": None,
            "prior_period_awards_usd": None, "yoy_change_pct": None,
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

    # Run the four independent USASpending queries concurrently.
    results: Dict[str, Any] = {}
    errors: List[str] = []
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        futures = {
            pool.submit(_usa_obligations_total, company_name, trailing_start, end, award_types): "trailing",
            pool.submit(_usa_obligations_total, company_name, prior_start, prior_end, award_types): "prior",
            pool.submit(_usa_top_agencies, company_name, trailing_start, end, award_types): "agencies",
            pool.submit(_usa_award_count, company_name, trailing_start, end, award_types): "count",
        }
        for future in as_completed(futures, timeout=_USA_TIMEOUT + 5):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                errors.append(f"{key}: {type(exc).__name__}: {str(exc)[:120]}")
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

    signal = _gov_contracts_signal(trailing_total, prior_total, yoy_pct)

    # prior/agencies/count are supporting detail: missing any of them narrows
    # the answer without invalidating the trailing total.
    return {
        "success": True,
        "coverage": "partial" if errors else "full",
        "company_name": company_name,
        "ticker": ticker,
        "period_months": months,
        "trailing_awards_usd": round(trailing_total, 2),
        "trailing_award_count": award_count,
        "prior_period_awards_usd": round(prior_total, 2) if prior_total is not None else None,
        "yoy_change_pct": yoy_pct,
        "signal": signal,
        "top_agencies": top_agencies,
        "source": "usaspending.gov",
        "basis": ("Contract obligations (period flow) via spending_over_time — "
                  "not multi-year contract ceiling values."),
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


def _govtrack_fetch_bills(keywords: List[str], congress: int,
                           limit: int = 8) -> Tuple[List[Dict], List[str]]:
    """(bills, errors). A keyword whose query failed is named in `errors`.

    Swallowing the exception made "GovTrack is down" and "no bill mentions
    semiconductors" the same answer.
    """
    import requests
    bills = []
    errors: List[str] = []
    seen_ids: set = set()

    for kw in keywords[:3]:
        try:
            resp = requests.get(
                "https://www.govtrack.us/api/v2/bill/",
                params={"q": kw, "congress": congress,
                        "order_by": "-introduced_date", "limit": limit},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            for obj in data.get("objects", []):
                # GovTrack bill objects have no "id" field; dedup on the unique
                # link (fall back to title). The old "id" key was always None,
                # which silently dropped every bill.
                bid = obj.get("link") or obj.get("title")
                if bid and bid not in seen_ids:
                    seen_ids.add(bid)
                    bills.append({
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
                    })
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            errors.append(f"govtrack {kw!r} (congress {congress}): "
                          f"{type(exc).__name__}: {str(exc)[:120]}")
            continue
    return bills, errors


def _congress_api_fetch_bills(keywords: List[str], congress: int,
                               api_key: str,
                               limit: int = 10) -> Tuple[List[Dict], List[str]]:
    """(bills, errors) — same contract as _govtrack_fetch_bills."""
    import requests
    bills = []
    errors: List[str] = []
    seen_titles: set = set()

    for kw in keywords[:3]:
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
                if title in seen_titles:
                    continue
                seen_titles.add(title)
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
                bills.append({
                    "title": title,
                    "short_title": "",
                    "status": status,
                    "introduced_date": latest.get("actionDate", ""),
                    "activity_date": latest.get("actionDate", ""),
                    "link": "",
                    "congress": congress,
                    "source": "congress.gov",
                })
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            errors.append(f"congress.gov {kw!r} (congress {congress}): "
                          f"{type(exc).__name__}: {str(exc)[:120]}")
            continue
    return bills, errors


def _fetch_policy_signals(ticker: str, sector: str,
                           lookback_days: int) -> Dict[str, Any]:
    """Legislative climate for a sector.

    Every failure mode here used to look like "no relevant bills found": an
    unknown ticker, a sector with no keyword mapping (which silently answered
    with the technology/semiconductor/defense default), and GovTrack being
    unreachable. Each is now named.
    """
    base = {"ticker": ticker, "sector": sector or None,
            "lookback_days": lookback_days, "bill_count": None,
            "bills": [], "signal": None}

    if not sector:
        try:
            sector = _resolve_ticker(ticker)["sector"]
        except LookupFailure as exc:
            return _lookup_failed(exc, **base)
        if not sector:
            return {
                **base, "success": False, "coverage": "not_covered",
                "reason": "sector_unresolved",
                "error": (f"no sector for '{ticker}' in the quote response "
                          f"(ETFs and indices carry none). Pass an explicit "
                          f"`sector` to search legislation."),
                "sectors_supported": sorted(SECTOR_BILL_KEYWORDS),
            }
        base["sector"] = sector

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
    if api_key:
        bills, errs = _congress_api_fetch_bills(keywords, current_congress, api_key)
        fetch_errors += errs
        if not bills:
            bills, errs = _congress_api_fetch_bills(keywords, prior_congress, api_key)
            fetch_errors += errs
    else:
        bills, errs = _govtrack_fetch_bills(keywords, current_congress)
        fetch_errors += errs
        if len(bills) < 3:
            more, errs = _govtrack_fetch_bills(keywords, prior_congress)
            bills += more
            fetch_errors += errs

    if not bills and fetch_errors:
        # Every query the provider was asked errored. An empty bill list here
        # means the provider is unreachable, not that Congress is idle.
        return {
            **base, "sector": sector, "success": False,
            "coverage": "not_covered", "reason": "provider_unavailable",
            "error": (f"{source} returned no usable response for any of "
                      f"{keywords[:3]}: " + "; ".join(fetch_errors)),
            "keywords_searched": keywords, "source": source,
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
            "keywords_searched": keywords,
            "bill_count": 0,
            "bills": [],
            "signal": "neutral",
            "signal_basis": (f"{source} returned no bill matching "
                             f"{keywords[:3]} with activity in the last "
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
        basis = "net positive legislative activity in sector"
    elif total_score < -0.5:
        signal = "bearish"
        basis = "net negative legislative activity (restrictions/controls) in sector"
    else:
        signal = "neutral"
        basis = "legislative activity is mixed or low probability"

    # Sort by abs score
    bills_out = sorted(bills, key=lambda b: -abs(b["score"]))[:10]

    return {
        "success": True,
        "coverage": coverage,
        "ticker": ticker, "sector": sector,
        "keywords_searched": keywords,
        "bill_count": len(bills),
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
                                  total_announced_usd=None,
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
            "total_announced_usd": 0, "signal": "data_gap",
            "queries_tried": queries,
            "partial_errors": query_errors,
            "announcements": [],
        }

    announcements = []
    for r in all_articles:
        title = r.get("title", "")
        body = r.get("body", "")
        combined = title + " " + body
        amounts = _extract_dollar_amounts(combined)
        max_amount = max(amounts) if amounts else 0
        announcements.append({
            "title": title[:120],
            "date": r.get("date", ""),
            "url": r.get("url", ""),
            "max_amount_usd": max_amount,
            "direction": _classify_capex_text(combined),
            "snippet": body[:200] if body else "",
        })

    announcements.sort(key=lambda x: -x["max_amount_usd"])

    total_usd = sum(a["max_amount_usd"] for a in announcements)
    signal = _capex_signal(announcements)

    return {
        "success": True,
        "coverage": "full" if not query_errors else "partial",
        "ticker": ticker, "company_name": company_name,
        "lookback_days": lookback_days,
        "announcement_count": len(announcements),
        "total_announced_usd": total_usd,
        "signal": signal,
        "queries_tried": queries,
        "partial_errors": query_errors,
        "announcements": announcements[:8],
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
                        "Returns NTD millions per month + YoY%. Every requested code "
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
                        "Returns trailing-period obligations, prior-period obligations, YoY change, "
                        "top awarding agencies, and award count. "
                        "Signal: YoY > +15% = bullish; YoY < -15% = bearish; "
                        "< $10M total = not_applicable (consumer/B2C company). "
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
                        "from another sector's keywords. Without CONGRESS_API_KEY the "
                        "answer is GovTrack-only: coverage='partial' and the missing key "
                        "is named in 'degraded'. "
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
                        "Extracts dollar amounts, classifies direction, and returns "
                        "bullish / bearish / neutral / data_gap signal. "
                        "bullish: new investment announced; bearish: cancellation/delay/cut. "
                        "Any announcement >= $1B → strong bullish signal. "
                        "Returns success=false with reason='unknown_ticker' for a symbol "
                        "the quote provider cannot resolve, and reason='no_results' when "
                        "no article matched — a news corpus cannot assert that no capex "
                        "was announced, so that is never reported as a zero. "
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
            ]

        @self.server.call_tool()
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
            return _err(name, f"unknown tool: {name}")

    # -----------------------------------------------------------------------
    # Existing tool handlers
    # -----------------------------------------------------------------------


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
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return _err("get_policy_signals", "GovTrack/Congress.gov timed out after 30s", ticker)
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
        run_http(AltDataServer().server)
    else:
        print("[altdata] starting", file=sys.stderr, flush=True)
        srv = AltDataServer()
        asyncio.run(srv.run_server())
