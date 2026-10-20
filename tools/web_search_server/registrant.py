"""Which registrant a tool reads: SEC's ticker file, then the CIK.

edgartools resolves a ticker from a parquet file bundled with the package, and
a ticker missing from that file maps to nothing. `Company(ticker)` does not
raise for one. It returns a company whose CIK is -999999999 and whose filing
list is empty. On 2026-10-19 that was 103 of the 1,810 eligible names, and
every one of them is in SEC's own `company_tickers.json` (#136). Over MCP,
`get_schedule_13d_filings("DMC")` answered `success: true, count: 0` while
DMC's folder held 14 SC 13D filings. Nothing in that answer separates an
unmapped ticker from a quiet company.

So every tool in this package looks the ticker up in SEC's live file and asks
EDGAR by the CIK. The bundled map never enters into it. A miss fetches the
file again when the copy is over an hour old, so a name listed after the
server started still resolves. A ticker the file still does not map is
refused by name, which the tools report as a failure rather than as a company
that files nothing. `research/registrant.py` is the jobs' half of the same fix
(#130).
"""
from __future__ import annotations

import threading
from time import monotonic
from typing import Dict, Optional

import requests

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_TIMEOUT_S = 30

# How old the map may be before a miss fetches the file again. The servers run
# for weeks between restarts, and a map kept for the life of the process
# refused every name listed after it started. Inside this age a miss is
# refused without a request, so a junk ticker cannot make every call fetch
# SEC's whole file.
_MAP_MAX_AGE_S = 3600

_cik_map_cache: Optional[Dict[str, int]] = None
_cik_map_fetched_at = 0.0
_cik_map_lock = threading.Lock()


def _normalise(ticker: str) -> str:
    """SEC's file writes a share class with a hyphen: BRK-B, not BRK.B.

    edgartools accepted the dotted form, so callers send it.
    """
    return str(ticker).strip().upper().replace(".", "-")


def _fetch_cik_map() -> Dict[str, int]:
    """Ticker -> CIK, from SEC's live file.

    Not `peers._ticker_map`, which is keyed the other way. A CIK with more than
    one listed class keeps one ticker there (GOOGL and GOOG are both 1652044),
    and the other class would be refused here.
    """
    from .sec_series import _require_identity

    response = requests.get(_TICKER_MAP_URL,
                            headers={"User-Agent": _require_identity()},
                            timeout=_TIMEOUT_S)
    response.raise_for_status()
    return {_normalise(row["ticker"]): int(row["cik_str"])
            for row in response.json().values()}


def _load_cik_map() -> Dict[str, int]:
    """Fetch the file and cache it. The caller holds the lock.

    A failed fetch raises, names the file, and leaves the cache as it was, so
    the next call asks SEC again. Refused from the old map, a name SEC lists
    would read as unmapped because of our own outage.
    """
    global _cik_map_cache, _cik_map_fetched_at
    try:
        fetched = _fetch_cik_map()
    except Exception as exc:  # noqa: BLE001 - re-raised, naming the file
        raise RuntimeError(
            f"Could not fetch SEC's company_tickers.json: "
            f"{type(exc).__name__}: {exc}") from exc
    _cik_map_cache, _cik_map_fetched_at = fetched, monotonic()
    return fetched


def _cik_map() -> Dict[str, int]:
    """The cached map, fetched on first use. The file is ~10k rows.

    The lock keeps concurrent tool calls to one fetch.
    """
    with _cik_map_lock:
        if _cik_map_cache is None:
            return _load_cik_map()
        return _cik_map_cache


def _cik_map_after_a_miss() -> Dict[str, int]:
    """The map again, fetched once more if it is older than `_MAP_MAX_AGE_S`.

    Read under the lock, so a miss that waited on another thread's refresh
    looks again in that map rather than fetching a second time.
    """
    with _cik_map_lock:
        if monotonic() - _cik_map_fetched_at > _MAP_MAX_AGE_S:
            return _load_cik_map()
        return _cik_map_cache


def cik_for(ticker: str) -> int:
    """The CIK SEC's ticker file gives `ticker`, or a refusal that names it.

    An all-digit key is already a CIK. edgartools read it as one, so a caller
    that had a CIK could pass it, and it still can.
    """
    key = _normalise(ticker)
    if key.isdigit():
        return int(key)
    cik = _cik_map().get(key)
    if cik is None:
        cik = _cik_map_after_a_miss().get(key)
    if cik is None:
        raise LookupError(
            f"SEC's company_tickers.json does not map {ticker} to a "
            f"registrant. Refused rather than read as a company with no "
            f"filings.")
    return cik


def company(ticker: str):
    """`edgar.Company` for the registrant behind `ticker`, built from its CIK.

    The CIK route has its own empty answer: edgartools turns SEC's 404 into a
    company flagged `not_found` with no filings. That is refused too. The flag
    reads the same submissions index `get_filings` reads, so it costs no
    request of its own.
    """
    from edgar import Company

    cik = cik_for(ticker)
    found = Company(cik)
    if found.not_found:
        raise LookupError(
            f"SEC has no submissions for {ticker} under CIK {cik}. Refused "
            f"rather than read as a company with no filings.")
    return found
