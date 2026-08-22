"""Peer discovery by SIC code.

`comparable_company_analysis` required the caller to supply the peer set, which
makes the comps depend on already knowing the answer. Every SEC filer carries
an SIC classification, so the peer set can come from the filings instead.

Two quirks of the SEC endpoints shape this module:

1. The browse-edgar atom feed's `title` and `name` attributes come back as
   leaked Perl array references — literally `ARRAY(0x5648a8af6810)` — so
   company names must come from the ticker map, never from the feed.
2. An SIC query returns every filer in the class, including deregistered and
   private ones with no listed ticker. For SIC 3674, 13 of 40 resolved. The
   shortfall is reported rather than hidden, because 13 peers presented as the
   whole universe is a different claim from 13 of 40.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from .sec_series import _require_identity

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_BROWSE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
_TIMEOUT_S = 30

_ticker_map_cache: Optional[Dict[int, Tuple[str, str]]] = None


def _headers() -> Dict[str, str]:
    return {"User-Agent": _require_identity()}


def _parse_ciks(xml: str) -> List[int]:
    """CIKs from the atom feed, in order, de-duplicated."""
    seen = set()
    out: List[int] = []
    for match in re.findall(r"<cik>(\d+)</cik>", xml):
        cik = int(match)
        if cik not in seen:
            seen.add(cik)
            out.append(cik)
    return out


def _ticker_map() -> Dict[int, Tuple[str, str]]:
    """CIK -> (ticker, company name) for exchange-listed filers.

    Cached for the process. The file is ~10k entries and changes slowly.
    """
    global _ticker_map_cache
    if _ticker_map_cache is not None:
        return _ticker_map_cache
    response = requests.get(_TICKER_MAP_URL, headers=_headers(), timeout=_TIMEOUT_S)
    response.raise_for_status()
    _ticker_map_cache = {
        int(row["cik_str"]): (row["ticker"], row["title"])
        for row in response.json().values()
    }
    return _ticker_map_cache


def _company_sic(ticker: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    from edgar import Company

    _require_identity()
    company = Company(ticker)
    return (getattr(company, "sic", None),
            getattr(company, "industry", None),
            getattr(company, "cik", None))


def _fetch_sic_ciks(sic: str, limit: int) -> List[int]:
    params = {
        "action": "getcompany", "SIC": sic, "type": "10-K",
        "dateb": "", "owner": "include", "count": str(limit), "output": "atom",
    }
    response = requests.get(_BROWSE_URL, params=params, headers=_headers(),
                            timeout=_TIMEOUT_S)
    response.raise_for_status()
    return _parse_ciks(response.text)


def get_sic_code(ticker: str) -> Dict[str, Any]:
    """The filer's SIC classification and industry description."""
    try:
        sic, industry, cik = _company_sic(ticker)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"ticker": ticker, "success": False,
                "error": f"{type(exc).__name__}: {exc}"}

    if not sic:
        return {"ticker": ticker, "success": False,
                "error": f"{ticker} has no SIC classification in EDGAR"}

    return {"ticker": ticker, "success": True, "sic": sic,
            "industry": industry, "cik": cik}


def find_peers_by_sic(ticker: str, limit: int = 20) -> Dict[str, Any]:
    """Listed companies sharing this filer's SIC classification.

    `filers_matched` counts every filer EDGAR returned; `peers` holds only those
    resolving to a listed ticker, and `unresolved_count` is the difference. SIC
    is a coarse classification — it groups by what a company files as, not by
    what it competes with — so treat the result as a starting set rather than a
    finished comp table.
    """
    try:
        sic, industry, cik = _company_sic(ticker)
        if not sic:
            return {"ticker": ticker, "success": False, "peers": [],
                    "error": f"{ticker} has no SIC classification in EDGAR"}

        ciks = _fetch_sic_ciks(sic, limit)
        mapping = _ticker_map()
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"ticker": ticker, "success": False, "peers": [],
                "error": f"{type(exc).__name__}: {exc}"}

    found: List[Dict[str, Any]] = []
    for peer_cik in ciks:
        if cik is not None and peer_cik == cik:
            continue
        entry = mapping.get(peer_cik)
        if entry is None:
            continue
        found.append({"ticker": entry[0], "name": entry[1], "cik": peer_cik})

    self_matched = 1 if (cik is not None and cik in ciks) else 0
    unresolved = len(ciks) - len(found) - self_matched

    return {
        "ticker": ticker,
        "success": True,
        "sic": sic,
        "industry": industry,
        "peers": found,
        "peer_count": len(found),
        "filers_matched": len(ciks),
        "unresolved_count": max(unresolved, 0),
        "note": ("SIC groups filers by declared classification, not by "
                 "competitive overlap. Unresolved filers are deregistered or "
                 "private and have no listed ticker."),
    }
