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
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import registrant
from .sec_series import _require_identity, _throttle

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

    SEC lists one row per security. The first row for a CIK is the one kept.
    On the live file that row is the common share for BAC, C, WFC, USB, COF
    and GOOGL; the later rows are preferred shares or other classes. A dict
    comprehension would keep the last row, and a peer named BAC-PS is the
    preferred's price, not the company's. Alphabet has no unsuffixed ticker,
    so "drop the class suffix" does not choose between GOOGL and GOOG.

    Cached for the process. The file is ~10k entries and changes slowly.
    """
    global _ticker_map_cache
    if _ticker_map_cache is not None:
        return _ticker_map_cache
    response = requests.get(_TICKER_MAP_URL, headers=_headers(), timeout=_TIMEOUT_S)
    response.raise_for_status()
    mapping: Dict[int, Tuple[str, str]] = {}
    for row in response.json().values():
        cik = int(row["cik_str"])
        if cik not in mapping:
            mapping[cik] = (row["ticker"], row["title"])
    _ticker_map_cache = mapping
    return _ticker_map_cache


def _company_sic(ticker: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    _require_identity()
    company = registrant.company(ticker)
    return (getattr(company, "sic", None),
            getattr(company, "industry", None),
            getattr(company, "cik", None))


# EDGAR's browse endpoint caps `count` at 100 and lists an SIC group in name
# order. Reading one page made the peer set the start of the alphabet: SIC 3674
# runs to 506 filers, so NVDA's peers were AMD, Amkor and Allegro, and MU and
# TXN were not in the set. Every page is read, whatever `limit` the caller asked
# for, so the counts describe the whole group.
_SIC_PAGE_SIZE = 100
# The longest group measured is SIC 6770, blank checks, at 34 pages. A group
# past this bound is refused, not ranked from the pages that were read.
_SIC_MAX_PAGES = 60


def _fetch_sic_ciks(sic: str) -> List[int]:
    """Every filer EDGAR lists under this SIC code, across all pages.

    A failed page raises. Ranking the pages that did arrive would present part
    of the group as all of it, and the filers on the missing page would read as
    companies that do not exist.
    """
    ciks: List[int] = []
    for page in range(_SIC_MAX_PAGES):
        start = page * _SIC_PAGE_SIZE
        params = {
            "action": "getcompany", "SIC": sic, "type": "10-K",
            "dateb": "", "owner": "include", "count": str(_SIC_PAGE_SIZE),
            "start": str(start), "output": "atom",
        }
        _throttle()
        try:
            response = requests.get(_BROWSE_URL, params=params, headers=_headers(),
                                    timeout=_TIMEOUT_S)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"EDGAR's SIC {sic} list failed at start={start}: {exc}") from exc
        # A body that is not the feed parses to no CIKs and no next link, and
        # the list would end there looking complete.
        if "<feed" not in response.text:
            raise RuntimeError(
                f"EDGAR's SIC {sic} list at start={start} is not an atom feed")
        ciks.extend(_parse_ciks(response.text))
        if 'rel="next"' not in response.text:
            return list(dict.fromkeys(ciks))
    raise RuntimeError(
        f"EDGAR's SIC {sic} list runs past {_SIC_MAX_PAGES} pages of "
        f"{_SIC_PAGE_SIZE}; refused rather than ranking part of the group")


# Size is total assets, `us-gaap:Assets`, read from SEC's XBRL frames: one
# request returns the concept for every filer. Two alternatives were measured
# and rejected. Public float (`dei:EntityPublicFloat`) is a cover-page fact that
# no audit checks, and filers mis-scale it: Trustmark's FY2025 10-K states
# $961.6bn against $1.0bn the year before, which ranked it first among JPM's
# peers, and Universal Display states $6.8tn. Revenue is split across concepts
# -- NVDA tags Revenues, while AMD, AVGO and MU tag
# RevenueFromContractWithCustomerExcludingAssessedTax -- so a ranking by it
# would rank one concept against another. Assets is one concept, it is a
# balance-sheet line rather than a cover-page entry, and every 10-Q reports it.
_FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{period}.json"
# Four quarters hold every filer's balance sheet at least once, since a filer
# that files no 10-Q still files a 10-K each year.
_SIZE_QUARTERS = 4


def _size_periods(today: date) -> List[str]:
    """The frames to read, newest first.

    SEC builds a frame only once a fact is dated in it, and answers 404 until
    then. The quarter in progress and the one just closed are skipped, so every
    frame read is one that exists and a 404 is a real failure.
    """
    year, quarter = today.year, (today.month - 1) // 3 + 1
    periods: List[str] = []
    for _ in range(_SIZE_QUARTERS + 2):
        periods.append(f"CY{year}Q{quarter}I")
        year, quarter = (year - 1, 4) if quarter == 1 else (year, quarter - 1)
    return periods[2:]


def _total_assets(periods: List[str]) -> Dict[int, Tuple[float, str]]:
    """CIK -> (total assets in USD, the balance-sheet date), latest date wins.

    A failed frame raises. The filers in it would otherwise sink to the unranked
    tail and read as filers that report no assets.
    """
    assets: Dict[int, Tuple[float, str]] = {}
    for period in periods:
        _throttle()
        try:
            response = requests.get(_FRAMES_URL.format(period=period),
                                    headers=_headers(), timeout=_TIMEOUT_S)
            response.raise_for_status()
            rows = response.json()["data"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"SEC frame us-gaap:Assets {period} failed: {exc}") from exc
        for row in rows:
            cik = int(row["cik"])
            held = assets.get(cik)
            if held is None or row["end"] > held[1]:
                assets[cik] = (row["val"], row["end"])
    return assets


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
    """Listed companies sharing this filer's SIC classification, largest first.

    `filers_matched` counts every filer EDGAR lists for the code; `peers` holds
    only those resolving to a listed ticker, and `unresolved_count` is the
    difference. Peers are ordered by total assets before `limit` applies, and
    those with no assets in the frames read follow in EDGAR's name order. SIC
    is a coarse classification — it groups by what a company files as, not by
    what it competes with — so treat the result as a starting set rather than a
    finished comp table.
    """
    try:
        sic, industry, cik = _company_sic(ticker)
        if not sic:
            return {"ticker": ticker, "success": False, "peers": [],
                    "error": f"{ticker} has no SIC classification in EDGAR"}

        ciks = _fetch_sic_ciks(sic)
        mapping = _ticker_map()
        periods = _size_periods(datetime.now(timezone.utc).date())
        assets = _total_assets(periods)
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
        size, as_of = assets.get(peer_cik, (None, None))
        found.append({"ticker": entry[0], "name": entry[1], "cik": peer_cik,
                      "assets_usd": size, "assets_as_of": as_of})

    self_matched = 1 if (cik is not None and cik in ciks) else 0
    unresolved = len(ciks) - len(found) - self_matched

    # Largest first. A peer with no assets in the frames is still in the
    # industry, so it is kept after the ranked ones, in EDGAR's name order.
    ranked = sorted((p for p in found if p["assets_usd"] is not None),
                    key=lambda p: p["assets_usd"], reverse=True)
    unranked = [p for p in found if p["assets_usd"] is None]
    found = ranked + unranked

    # Count the resolved set, then take the caller's page.
    peer_count = len(found)
    page = found[:limit]

    note = ("SIC groups filers by declared classification, not by "
            "competitive overlap. Unresolved filers are deregistered or "
            "private and have no listed ticker. Peers are ordered by total "
            "assets from each filer's latest balance sheet in the frames read "
            "(assets_as_of). That measures balance-sheet size, not market "
            "value: an asset-heavy filer ranks above an asset-light one of the "
            "same market cap.")
    if unranked:
        note += (f" {len(unranked)} peers have no us-gaap:Assets value in USD "
                 f"in {periods[-1]}..{periods[0]} and follow in name order. An "
                 f"IFRS filer tags ifrs-full instead.")

    return {
        "ticker": ticker,
        "success": True,
        "sic": sic,
        "industry": industry,
        "peers": page,
        "peer_count": peer_count,
        "peers_returned": len(page),
        "truncated": peer_count > len(page),
        "filers_matched": len(ciks),
        "unresolved_count": max(unresolved, 0),
        "order": ("total assets, largest first; peers with no assets follow "
                  "in EDGAR's name order"),
        "size_measure": {
            "concept": "us-gaap:Assets",
            "unit": "USD",
            "periods": periods,
            "basis": "each filer's latest value in these quarterly XBRL frames",
        },
        "unranked_count": len(unranked),
        "note": note,
    }
