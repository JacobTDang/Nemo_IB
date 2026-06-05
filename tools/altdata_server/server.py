"""nemo_altdata MCP server — alternative data tools for pre-earnings research.

8 tools:
  get_google_trends          -- pytrends wrapper; YoY demand signal
  get_finbert_sentiment      -- FinBERT financial sentiment on news headlines
  get_taiwan_monthly_revenue -- FinMind API for TSMC/Foxconn/MediaTek/ASE revenue
  get_job_postings_count     -- Multi-ATS job listing count (Greenhouse/Lever/Workday
                                auto-discovery — no hardcoded company list)
  get_options_implied_move   -- ATM straddle implied move + put/call skew
                                (yfinance auto-fetch; no obb_options_chain required)
  get_government_contracts   -- Federal contract awards via USASpending.gov (no auth)
  get_policy_signals         -- Legislative climate via GovTrack (+ Congress.gov if key set)
  get_capex_announcements    -- Capital investment announcements via DuckDuckGo news

Heavy tools (pytrends, FinBERT) run in isolated subprocesses via the same
pattern as tools/openbb_server/server.py to avoid asyncio conflicts on Windows.
Light tools (FinMind, job postings, options, gov contracts, policy, capex) run
directly in async handlers via asyncio.to_thread.

Taiwan revenue uses FinMind (api.finmindtrade.com).
Job postings: auto-discovers Greenhouse → Lever → Workday in parallel (no curated list).
Options: calls yfinance directly when ATM strikes are missing from supplied rows.

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

_TRENDS_RUNNER = os.path.join(_HERE, "trends_runner.py")
_FINBERT_RUNNER = os.path.join(_HERE, "finbert_runner.py")

# Extended timeouts: trends_runner now retries on 429 (adds up to 30s)
_TRENDS_TIMEOUT_S = 45.0
_FINBERT_TIMEOUT_S = 120.0
_SUBPROCESS_TIMEOUT_S = 40.0
_FINBERT_SUB_TIMEOUT_S = 115.0


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
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"success": False,
                "error": f"output not valid JSON: {stdout[:300]}"}


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


# ---------------------------------------------------------------------------
# Options implied move — pure math
# ---------------------------------------------------------------------------

def compute_implied_move(spot: float, atm_call_ask: float,
                         atm_put_ask: float) -> Dict[str, Any]:
    straddle = atm_call_ask + atm_put_ask
    implied_move_pct = straddle / spot if spot > 0 else 0.0
    return {"implied_move_pct": round(implied_move_pct, 4), "straddle_cost": straddle}


_ATM_GAP_THRESHOLD = 0.08  # nearest strike >8% from spot → treat as ATM missing


def _find_atm_options(rows: List[Dict], spot: float,
                      target_expiry: Optional[str] = None):
    """Find the nearest ATM call and put. Returns (None, None, None) if ATM gap > 8%."""
    if not rows:
        return None, None, None

    expiries = sorted({r.get("expiration") or r.get("expiration_date", "") for r in rows
                       if r.get("expiration") or r.get("expiration_date")})
    if not expiries:
        return None, None, None

    today = datetime.now(timezone.utc).date().isoformat()
    future = [e for e in expiries if e > today]
    chosen_expiry = future[0] if future else expiries[-1]
    if target_expiry:
        chosen_expiry = target_expiry

    chain = [r for r in rows
             if (r.get("expiration") or r.get("expiration_date", "")) == chosen_expiry]
    calls = [r for r in chain if (r.get("option_type") or r.get("optionType", "")).lower() == "call"]
    puts  = [r for r in chain if (r.get("option_type") or r.get("optionType", "")).lower() == "put"]

    def nearest_atm(options):
        if not options:
            return None
        best = min(options, key=lambda r: abs(float(r.get("strike", 0)) - spot))
        # Guard: reject if gap is too large (truncated chain)
        if spot > 0 and abs(float(best.get("strike", 0)) - spot) / spot > _ATM_GAP_THRESHOLD:
            return None
        return best

    atm_call = nearest_atm(calls)
    atm_put  = nearest_atm(puts)
    return atm_call, atm_put, chosen_expiry


def _fetch_options_yfinance(ticker: str, spot: float,
                             near_days: int = 60) -> List[Dict]:
    """Fetch a complete options chain via yfinance. No row-count cap."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    exps = t.options  # tuple of 'YYYY-MM-DD' strings, sorted
    if not exps:
        return []

    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=near_days)
    target_exps = [
        e for e in exps
        if today < datetime.strptime(e, "%Y-%m-%d").date() <= cutoff
    ][:4]
    if not target_exps:
        target_exps = [exps[0]]  # nearest available expiry

    rows: List[Dict] = []
    for exp in target_exps:
        try:
            chain = t.option_chain(exp)
            for _, row in chain.calls.iterrows():
                rows.append({
                    "expiration": exp, "option_type": "call",
                    "strike": float(row.get("strike") or 0),
                    "ask": float(row.get("ask") or 0),
                    "implied_volatility": float(row.get("impliedVolatility") or 0),
                })
            for _, row in chain.puts.iterrows():
                rows.append({
                    "expiration": exp, "option_type": "put",
                    "strike": float(row.get("strike") or 0),
                    "ask": float(row.get("ask") or 0),
                    "implied_volatility": float(row.get("impliedVolatility") or 0),
                })
        except Exception:
            continue
    return rows


# ---------------------------------------------------------------------------
# Capex announcement helpers
# ---------------------------------------------------------------------------

_AMOUNT_PATTERN = re.compile(
    r'\$\s*(\d+(?:\.\d+)?)\s*(billion|trillion|million|B|M|T)\b',
    re.IGNORECASE,
)
_CAPEX_BULLISH = frozenset([
    "invest", "new factory", "new plant", "expand", "expansion", "construction",
    "groundbreaking", "build", "opening", "data center", "new facility",
    "ramp up", "increase capex", "megafactory", "gigafactory", "announce",
])
_CAPEX_BEARISH = frozenset([
    "cancel", "delay", "cut", "reduce", "shutdown", "close", "idle",
    "lay off", "restructure", "write off", "impairment", "pause",
])


def _extract_dollar_amounts(text: str) -> List[float]:
    amounts = []
    for m in _AMOUNT_PATTERN.finditer(text):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("billion", "b"):
            amounts.append(val * 1_000_000_000)
        elif unit in ("million", "m"):
            amounts.append(val * 1_000_000)
        elif unit in ("trillion", "t"):
            amounts.append(val * 1_000_000_000_000)
    return amounts


def _classify_capex_text(text: str) -> str:
    t = text.lower()
    bull = sum(1 for w in _CAPEX_BULLISH if w in t)
    bear = sum(1 for w in _CAPEX_BEARISH if w in t)
    if bull > bear:
        return "bullish"
    elif bear > bull:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Company name resolution (ticker → display name)
# ---------------------------------------------------------------------------

def _ticker_to_company_name(ticker: str) -> str:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def _ticker_to_sector(ticker: str) -> str:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("sector", "")
    except Exception:
        return ""


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

_ATS_UNSUPPORTED_MESSAGES: Dict[str, str] = {
    "taleo": "Oracle Taleo requires authentication — no public job count API",
    "successfactors": "SAP SuccessFactors requires authentication",
    "icims": "iCIMS does not expose a public job count API",
    "bamboohr": "BambooHR is internal HR software, not a public job board",
    "smartrecruiters": "SmartRecruiters has restricted public API access",
    "jobvite": "Jobvite does not expose a public job count API",
}


def _normalize_ats_jobs(jobs: list, source: str, source_url: str,
                         slug: str, dept_filter: Optional[str]) -> Dict[str, Any]:
    by_dept: Dict[str, int] = {}
    for job in jobs:
        if source == "greenhouse":
            dept = (job.get("departments") or [{}])[0].get("name", "Unknown")
        else:
            dept = job.get("categories", {}).get("department", "Unknown")
        by_dept[dept] = by_dept.get(dept, 0) + 1

    filtered = jobs
    if dept_filter:
        filtered = [
            j for j in jobs
            if dept_filter.lower() in str(
                (j.get("departments") or [{}])[0].get("name", "") or
                j.get("categories", {}).get("department", "")
            ).lower()
        ]

    return {
        "slug": slug, "ats": source, "source_url": source_url,
        "total": len(filtered),
        "total_all_depts": len(jobs),
        "by_department": dict(sorted(by_dept.items(), key=lambda x: -x[1])[:15]),
        "dept_filter_applied": dept_filter,
    }


def _try_greenhouse_norm(slug: str, dept_filter: Optional[str]) -> Optional[Dict]:
    import requests
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _normalize_ats_jobs(resp.json().get("jobs", []),
                                   "greenhouse", url, slug, dept_filter)
    except Exception:
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


def _workday_fetch_full(tenant: str, wd_n: int, path: str,
                         dept_filter: Optional[str]) -> Optional[Dict]:
    """Full Workday job fetch (limit=50) with facet department breakdown."""
    import requests
    url = (f"https://{tenant}.wd{wd_n}.myworkdayjobs.com"
           f"/wday/cxs/{tenant}/{path}/jobs")
    try:
        resp = requests.post(
            url,
            json={"appliedFacets": {}, "limit": 50, "offset": 0, "searchText": ""},
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
                    by_dept[entry.get("value", "Unknown")] = entry.get("count", 0)
                if by_dept:
                    break

        filtered_total = total
        if dept_filter and by_dept:
            filtered_total = sum(v for k, v in by_dept.items()
                                 if dept_filter.lower() in k.lower())

        return {
            "slug": tenant, "ats": "workday", "source_url": url,
            "total": filtered_total if dept_filter else total,
            "total_all_depts": total,
            "by_department": dict(sorted(by_dept.items(), key=lambda x: -x[1])[:15]),
            "dept_filter_applied": dept_filter,
        }
    except Exception:
        return None


def _try_workday_discovery(slug: str, dept_filter: Optional[str]) -> Optional[Dict]:
    """
    Generic parallel Workday discovery. Derives tenant name variants from slug
    and probes all wd_num × path combinations simultaneously.
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

    if found_meta is None:
        return None
    return _workday_fetch_full(found_meta["tenant"], found_meta["wd_n"],
                                found_meta["path"], dept_filter)


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
    Stage 1: Greenhouse + Lever + Workday auto-discovery in parallel.
    Stage 2: ATS fingerprint via company careers URL redirect.
    Stage 3: Structured error with detected ATS info.
    """
    # Direct workday bypass (explicit request)
    if ats == "workday":
        result = _try_workday_discovery(slug, dept_filter)
        if result:
            return result
        return {
            "error": f"Workday not found for '{slug}' — auto-discovery tried "
                     f"all common tenant variants.",
            "ats_detected": "workday_not_found", "slug": slug,
        }

    # Stage 1: parallel discovery
    pool = ThreadPoolExecutor(max_workers=3)
    gh_f = pool.submit(_try_greenhouse_norm, slug, dept_filter)
    lv_f = pool.submit(_try_lever_norm, slug, dept_filter)
    wd_f = pool.submit(_try_workday_discovery, slug, dept_filter)

    first_success: Optional[Dict] = None
    preferred_success: Optional[Dict] = None

    try:
        for future in as_completed([gh_f, lv_f, wd_f], timeout=12):
            try:
                result = future.result()
                if result and "error" not in result:
                    src = result.get("ats", "")
                    if first_success is None:
                        first_success = result
                    if src == ats and preferred_success is None:
                        preferred_success = result
            except Exception:
                pass
    except Exception:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    chosen = preferred_success or first_success
    if chosen:
        return chosen

    # Stage 2: ATS fingerprinting via website
    detected = _detect_ats_from_website(slug)
    if detected:
        detected_ats, detected_slug = detected
        if detected_ats == "greenhouse" and detected_slug:
            r = _try_greenhouse_norm(detected_slug, dept_filter)
            if r and "error" not in r:
                return r
        elif detected_ats == "lever" and detected_slug:
            r = _try_lever_norm(detected_slug, dept_filter)
            if r and "error" not in r:
                return r
        elif detected_ats == "workday":
            r = _try_workday_discovery(detected_slug or slug, dept_filter)
            if r and "error" not in r:
                return r

        # Detected but not queryable
        msg = _ATS_UNSUPPORTED_MESSAGES.get(
            detected_ats,
            f"'{detected_ats}' ATS is not publicly queryable",
        )
        return {"error": msg, "ats_detected": detected_ats,
                "detected_slug": detected_slug, "slug": slug}

    # Stage 3: clean failure
    return {
        "error": (
            f"No public job data found for '{slug}'. "
            "Tried Greenhouse, Lever, and Workday auto-discovery. "
            "This company likely uses a proprietary careers portal "
            "(common for Microsoft, Apple, Google, Meta)."
        ),
        "ats_detected": "unknown",
        "slug": slug,
    }


# ---------------------------------------------------------------------------
# Taiwan monthly revenue via FinMind
# ---------------------------------------------------------------------------

def _fetch_taiwan_revenue_finmind(company_codes: List[str], months: int) -> Dict[str, Any]:
    import requests
    lookback_days = (months + 14) * 31
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    token = os.environ.get("FINMIND_TOKEN", "")
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
            results[code] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
            continue

        if payload.get("status") != 200 or not payload.get("data"):
            results[code] = {
                "error": (f"FinMind status {payload.get('status')}: "
                          f"{payload.get('msg', 'no data returned')}")
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

    return {"companies": results, "codes_requested": company_codes}


# ---------------------------------------------------------------------------
# Government contracts via USASpending.gov
# ---------------------------------------------------------------------------

_USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
_CONTRACT_AWARD_TYPES = ["A", "B", "C", "D"]
_GRANT_AWARD_TYPES = ["04", "05", "06", "07", "08"]


def _usaspending_query(company_name: str, start: str, end: str,
                        award_types: List[str], limit: int = 100) -> Dict:
    import requests
    payload = {
        "filters": {
            "recipient_search_text": [company_name],
            "time_period": [{"start_date": start, "end_date": end}],
            "award_type_codes": award_types,
        },
        "fields": ["Award ID", "Recipient Name", "Award Amount",
                   "Action Date", "Awarding Agency Name", "Description"],
        "sort": "Award Amount",
        "order": "desc",
        "limit": limit,
        "page": 1,
    }
    resp = requests.post(
        _USASPENDING_URL, json=payload, timeout=35,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_government_contracts(ticker: str, company_name: str,
                                  months: int, include_grants: bool) -> Dict[str, Any]:
    if not company_name:
        company_name = _ticker_to_company_name(ticker)

    today = datetime.now()
    end = today.strftime("%Y-%m-%d")
    start = (today - timedelta(days=months * 31)).strftime("%Y-%m-%d")
    prior_start = (today - timedelta(days=months * 31 * 2)).strftime("%Y-%m-%d")
    prior_end = start

    award_types = _CONTRACT_AWARD_TYPES + (_GRANT_AWARD_TYPES if include_grants else [])

    try:
        current_data = _usaspending_query(company_name, start, end, award_types)
        prior_data = _usaspending_query(company_name, prior_start, prior_end, award_types)
    except Exception as exc:
        return {"error": f"USASpending.gov request failed: {type(exc).__name__}: {str(exc)[:200]}"}

    current_results = current_data.get("results", [])
    prior_results = prior_data.get("results", [])
    current_count = current_data.get("page_metadata", {}).get("total", 0)

    current_total = sum((r.get("Award Amount") or 0) for r in current_results)
    prior_total = sum((r.get("Award Amount") or 0) for r in prior_results)

    yoy_pct = None
    if prior_total > 0:
        yoy_pct = round((current_total - prior_total) / abs(prior_total) * 100, 1)

    # Agency breakdown
    agency_totals: Dict[str, float] = {}
    for r in current_results:
        agency = r.get("Awarding Agency Name", "Unknown")
        agency_totals[agency] = agency_totals.get(agency, 0) + (r.get("Award Amount") or 0)

    top_agencies = sorted(
        [{"agency": k, "amount_usd": v,
          "pct_of_total": round(v / current_total * 100, 1) if current_total else 0}
         for k, v in agency_totals.items()],
        key=lambda x: -x["amount_usd"],
    )[:5]

    # Major recent awards (>$100M)
    major = [
        {
            "description": (r.get("Description") or "")[:100],
            "amount_usd": r.get("Award Amount") or 0,
            "date": r.get("Action Date") or "",
            "agency": r.get("Awarding Agency Name") or "",
        }
        for r in current_results
        if (r.get("Award Amount") or 0) >= 100_000_000
    ][:5]

    # Signal
    signal = "neutral"
    if current_total < 10_000_000:
        signal = "not_applicable"  # company has essentially no federal contracts
    elif yoy_pct is not None:
        if yoy_pct > 10 or any(a["amount_usd"] >= 1_000_000_000 for a in major):
            signal = "bullish"
        elif yoy_pct < -10:
            signal = "bearish"

    return {
        "company_name": company_name,
        "ticker": ticker,
        "period_months": months,
        "trailing_awards_usd": current_total,
        "trailing_award_count": current_count,
        "prior_period_awards_usd": prior_total,
        "yoy_change_pct": yoy_pct,
        "signal": signal,
        "top_agencies": top_agencies,
        "major_recent_awards": major,
        "source": "usaspending.gov",
        "note": (
            "Dollar totals are based on the top 100 awards by amount "
            "(captures majority of contract value for most companies)."
        ),
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

_BILL_BULLISH = frozenset([
    "fund", "authoriz", "invest", "incentiv", "subsidi", "credit", "grant",
    "research", "develop", "support", "modern", "promot", "commerc", "expand",
    "innovat", "manufactur", "rebuild", "strengthen",
])
_BILL_BEARISH = frozenset([
    "ban", "restrict", "prohibit", "penalt", "investigat", "antitrust",
    "price cap", "price control", "sanction", "tariff increase", "moratorium",
    "export ban", "import restrict", "windfall",
])

_GOVTRACK_STATUSES = {
    "enacted_signed": 1.0, "enacted_veto_override": 1.0,
    "passed_bill": 0.70, "passed_resolution": 0.50,
    "reported": 0.25, "referred": 0.10, "introduced": 0.08,
}


def _score_bill_title(title: str, status: str) -> float:
    t = title.lower()
    pos = sum(1 for kw in _BILL_BULLISH if kw in t)
    neg = sum(1 for kw in _BILL_BEARISH if kw in t)
    net = pos - neg
    weight = _GOVTRACK_STATUSES.get(status, 0.08)
    return net * weight


def _govtrack_fetch_bills(keywords: List[str], congress: int,
                           limit: int = 8) -> List[Dict]:
    import requests
    bills = []
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
                bid = obj.get("id")
                if bid and bid not in seen_ids:
                    seen_ids.add(bid)
                    bills.append({
                        "title": obj.get("title", ""),
                        "short_title": obj.get("short_title", ""),
                        "status": obj.get("current_status", "introduced"),
                        "introduced_date": obj.get("introduced_date", ""),
                        "link": obj.get("link", ""),
                        "congress": congress,
                        "source": "govtrack",
                    })
        except Exception:
            continue
    return bills


def _congress_api_fetch_bills(keywords: List[str], congress: int,
                               api_key: str, limit: int = 10) -> List[Dict]:
    import requests
    bills = []
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
                    "link": "",
                    "congress": congress,
                    "source": "congress.gov",
                })
        except Exception:
            continue
    return bills


def _fetch_policy_signals(ticker: str, sector: str,
                           lookback_days: int) -> Dict[str, Any]:
    if not sector:
        sector = _ticker_to_sector(ticker)

    # Determine current and prior congress
    year = datetime.now().year
    start_year = year if year % 2 == 1 else year - 1
    current_congress = (start_year - 1789) // 2 + 1
    prior_congress = current_congress - 1

    keywords = SECTOR_BILL_KEYWORDS.get(sector, ["technology", "semiconductor", "defense"])

    api_key = os.environ.get("CONGRESS_API_KEY", "")
    if api_key:
        bills = _congress_api_fetch_bills(keywords, current_congress, api_key)
        if not bills:
            bills = _congress_api_fetch_bills(keywords, prior_congress, api_key)
    else:
        bills = _govtrack_fetch_bills(keywords, current_congress)
        if len(bills) < 3:
            bills += _govtrack_fetch_bills(keywords, prior_congress)

    if not bills:
        return {
            "ticker": ticker, "sector": sector,
            "keywords_searched": keywords,
            "bill_count": 0,
            "bills": [],
            "signal": "neutral",
            "signal_basis": "no relevant bills found",
            "note": "Set CONGRESS_API_KEY env var for more comprehensive legislative data.",
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
        "ticker": ticker, "sector": sector,
        "keywords_searched": keywords,
        "bill_count": len(bills),
        "total_score": round(total_score, 3),
        "signal": signal,
        "signal_basis": basis,
        "bills": bills_out,
        "source": "congress.gov" if api_key else "govtrack.us",
        "note": (None if api_key
                 else "Set CONGRESS_API_KEY env var for Congress.gov access."),
    }


# ---------------------------------------------------------------------------
# Capex announcements via DuckDuckGo news
# ---------------------------------------------------------------------------

def _parse_article_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def _fetch_capex_announcements(ticker: str, company_name: str,
                                 lookback_days: int) -> Dict[str, Any]:
    try:
        from ddgs import DDGS
    except ImportError:
        return {"error": "ddgs not installed — pip install ddgs",
                "ticker": ticker}

    if not company_name:
        company_name = _ticker_to_company_name(ticker)

    queries = [
        f"{company_name} factory plant construction investment announce",
        f"{company_name} data center capital expenditure billion expand",
    ]

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    all_articles: List[Dict] = []
    seen_titles: set = set()

    try:
        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = list(ddgs.news(query, max_results=10, timelimit=None))
                    for r in results:
                        title = r.get("title", "")
                        if title and title not in seen_titles:
                            pub_date = _parse_article_date(r.get("date", ""))
                            if pub_date is None or pub_date >= cutoff:
                                seen_titles.add(title)
                                all_articles.append(r)
                except Exception:
                    continue
    except Exception as exc:
        return {"error": f"DuckDuckGo search failed: {type(exc).__name__}: {str(exc)[:200]}",
                "ticker": ticker}

    if not all_articles:
        return {
            "ticker": ticker, "company_name": company_name,
            "lookback_days": lookback_days, "announcement_count": 0,
            "total_announced_usd": 0, "signal": "data_gap",
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
    has_major = any(a["max_amount_usd"] >= 1_000_000_000 for a in announcements)
    bullish_n = sum(1 for a in announcements if a["direction"] == "bullish")
    bearish_n = sum(1 for a in announcements if a["direction"] == "bearish")

    if has_major or bullish_n > bearish_n * 2:
        signal = "bullish"
    elif bearish_n > bullish_n * 2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "ticker": ticker, "company_name": company_name,
        "lookback_days": lookback_days,
        "announcement_count": len(announcements),
        "total_announced_usd": total_usd,
        "signal": signal,
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
                    name="get_google_trends",
                    description=(
                        "Google Trends interest-over-time for up to 5 search keywords. "
                        "Returns weekly index values (0-100) and a YoY ratio comparing "
                        "the last 13 weeks vs the same period 1 year prior. "
                        "Use to gauge consumer demand before a company's earnings: "
                        "ratio > 1.10 = bullish, < 0.90 = bearish. "
                        "Best for consumer-facing companies (tech, automotive, retail). "
                        "Results are cached for 12 hours to avoid rate limiting."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["keywords"],
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Search terms, max 5",
                            },
                            "timeframe": {
                                "type": "string",
                                "default": "today 12-m",
                                "description": "pytrends timeframe string. Use 12-m for YoY ratio.",
                            },
                            "geo": {
                                "type": "string",
                                "default": "US",
                                "description": "Country code or '' for worldwide.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_finbert_sentiment",
                    description=(
                        "Score news headlines using FinBERT (ProsusAI/finbert). "
                        "Returns per-article labels and aggregated net_score "
                        "(-1.0 all negative → +1.0 all positive). "
                        "Signal: net_score > 0.15 = bullish, < -0.15 = bearish."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["texts", "ticker"],
                        "properties": {
                            "texts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "News headlines or summaries (max 50).",
                            },
                            "ticker": {"type": "string"},
                        },
                    },
                ),
                Tool(
                    name="get_taiwan_monthly_revenue",
                    description=(
                        "Monthly revenue for Taiwan-listed companies via FinMind "
                        "(TWSE feed). Key codes: TSMC=2330, Foxconn=2317, "
                        "MediaTek=2454, ASE Group=3711. "
                        "Returns NTD millions per month + YoY%."
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
                        "Large companies with proprietary portals (Microsoft, Apple, Google) "
                        "return a structured error with the detected ATS type."
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
                    name="get_options_implied_move",
                    description=(
                        "Compute the options-implied earnings move from the ATM straddle. "
                        "When options_chain_rows is omitted or lacks ATM coverage, "
                        "fetches the full chain via yfinance automatically — no "
                        "obb_options_chain pre-call required. "
                        "Returns: implied_move_pct, put_call_skew, front_expiry, source. "
                        "Rule: implied_move_pct > 0.15 = high binary risk."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["ticker", "spot_price"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "spot_price": {
                                "type": "number",
                                "description": "Current stock price.",
                            },
                            "options_chain_rows": {
                                "type": "array",
                                "description": "Optional: pre-fetched rows from obb_options_chain. "
                                               "When omitted or ATM is missing, yfinance is used.",
                            },
                            "target_expiry": {
                                "type": "string",
                                "description": "Optional YYYY-MM-DD: use this expiry instead of front-month.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_government_contracts",
                    description=(
                        "Federal contract (and optional grant) awards to a company "
                        "via USASpending.gov — free, no auth required. "
                        "Returns trailing-period total, YoY change, top awarding agencies, "
                        "and major awards (>$100M). "
                        "Signal: YoY > +10% or any >$1B award = bullish; YoY < -10% = bearish; "
                        "< $10M total = not_applicable (consumer/B2C company). "
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
                        "Sector is auto-detected from yfinance if not provided. "
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
            if name == "get_google_trends":
                return await parent.google_trends(args)
            if name == "get_finbert_sentiment":
                return await parent.finbert_sentiment(args)
            if name == "get_taiwan_monthly_revenue":
                return await parent.taiwan_monthly_revenue(args)
            if name == "get_job_postings_count":
                return await parent.job_postings_count(args)
            if name == "get_options_implied_move":
                return await parent.options_implied_move(args)
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

    async def google_trends(self, args: Dict[str, Any]) -> List[TextContent]:
        keywords = args.get("keywords")
        if not keywords:
            return _err("get_google_trends", "keywords is required")
        kwargs = {
            "keywords": keywords,
            "timeframe": args.get("timeframe", "today 12-m"),
            "geo": args.get("geo", "US"),
        }
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_subprocess, _TRENDS_RUNNER, "get_google_trends",
                    kwargs, _SUBPROCESS_TIMEOUT_S,
                ),
                timeout=_TRENDS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _err("get_google_trends",
                        f"timeout after {_TRENDS_TIMEOUT_S}s")
        if not result.get("success"):
            return _err("get_google_trends", result.get("error", "unknown error"))
        return _ok("get_google_trends", result["data"])

    async def finbert_sentiment(self, args: Dict[str, Any]) -> List[TextContent]:
        texts = args.get("texts")
        ticker = str(args.get("ticker", "")).upper()
        if not texts:
            return _err("get_finbert_sentiment", "texts is required", ticker)
        kwargs = {"texts": texts, "ticker": ticker}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_subprocess, _FINBERT_RUNNER, "get_finbert_sentiment",
                    kwargs, _FINBERT_SUB_TIMEOUT_S,
                ),
                timeout=_FINBERT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _err("get_finbert_sentiment",
                        f"timeout after {_FINBERT_TIMEOUT_S}s", ticker)
        if not result.get("success"):
            return _err("get_finbert_sentiment", result.get("error", "unknown error"), ticker)
        return _ok("get_finbert_sentiment", result["data"], ticker)

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
        return _ok("get_taiwan_monthly_revenue", result)

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
        return _ok("get_job_postings_count", result, slug.upper())

    async def options_implied_move(self, args: Dict[str, Any]) -> List[TextContent]:
        ticker = str(args.get("ticker", "")).upper()
        spot = float(args.get("spot_price", 0))
        rows = args.get("options_chain_rows") or []
        target_expiry = args.get("target_expiry")

        if spot <= 0:
            return _err("get_options_implied_move", "spot_price must be > 0", ticker)

        source = "supplied"
        try:
            atm_call, atm_put, expiry = _find_atm_options(rows, spot, target_expiry)

            # Fallback to yfinance when rows are empty or ATM gap too large
            if (not atm_call or not atm_put) and ticker:
                yf_rows = await asyncio.to_thread(_fetch_options_yfinance, ticker, spot)
                if yf_rows:
                    atm_call, atm_put, expiry = _find_atm_options(yf_rows, spot, target_expiry)
                    source = "yfinance"

            if not atm_call or not atm_put:
                msg = "could not find ATM options"
                if rows and not source == "yfinance":
                    all_strikes = [float(r.get("strike", 0)) for r in rows]
                    if all_strikes and spot > 0:
                        nearest = min(all_strikes, key=lambda s: abs(s - spot))
                        gap_pct = abs(nearest - spot) / spot * 100
                        msg += (f" (nearest strike {nearest:.1f} is "
                                f"{gap_pct:.1f}% from spot {spot} — "
                                f"supplied chain lacks ATM coverage)")
                return _err("get_options_implied_move", msg, ticker)

            call_ask = float(atm_call.get("ask") or atm_call.get("ask_price") or 0)
            put_ask  = float(atm_put.get("ask")  or atm_put.get("ask_price")  or 0)
            call_iv  = float(atm_call.get("implied_volatility") or atm_call.get("impliedVolatility") or 0)
            put_iv   = float(atm_put.get("implied_volatility")  or atm_put.get("impliedVolatility")  or 0)
            strike   = float(atm_call.get("strike", 0))

            move = compute_implied_move(spot, call_ask, put_ask)
            skew_diff = put_iv - call_iv
            skew_label = (
                "put_heavy" if skew_diff > 0.03
                else "call_heavy" if skew_diff < -0.03
                else "balanced"
            )

            data = {
                "ticker": ticker,
                "spot_price": spot,
                "front_expiry": expiry,
                "atm_strike": strike,
                "atm_call_ask": call_ask,
                "atm_put_ask": put_ask,
                "implied_move_pct": move["implied_move_pct"],
                "straddle_cost": move["straddle_cost"],
                "call_iv": round(call_iv, 4),
                "put_iv": round(put_iv, 4),
                "put_call_skew": round(skew_diff, 4),
                "skew_label": skew_label,
                "source": source,
                "risk_flag": (
                    "HIGH VOLATILITY WARNING: implied move >15%"
                    if move["implied_move_pct"] > 0.15 else None
                ),
            }
        except Exception as exc:
            return _err("get_options_implied_move",
                        f"{type(exc).__name__}: {str(exc)[:200]}", ticker)
        return _ok("get_options_implied_move", data, ticker)

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
        if "error" in result:
            return _err("get_government_contracts", result["error"], ticker)
        return _ok("get_government_contracts", result, ticker)

    async def policy_signals(self, args: Dict[str, Any]) -> List[TextContent]:
        ticker = str(args.get("ticker", "")).upper()
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
        return _ok("get_policy_signals", result, ticker)

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
        if "error" in result:
            return _err("get_capex_announcements", result["error"], ticker)
        return _ok("get_capex_announcements", result, ticker)

    async def run_server(self):
        async with stdio_server() as (read, write):
            await self.server.run(read, write,
                                  self.server.create_initialization_options())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "server":
        print("Usage: python -m tools.altdata_server.server server", file=sys.stderr)
        sys.exit(1)
    print("[altdata] starting", file=sys.stderr, flush=True)
    srv = AltDataServer()
    asyncio.run(srv.run_server())
