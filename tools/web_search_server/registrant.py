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
from typing import Any, Dict, List, Optional, Tuple

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


def _accession_cik(accession: str) -> Optional[int]:
    """The CIK written into an accession number, or None.

    An accession is `CCCCCCCCCC-YY-SSSSSS`. The first ten digits are the CIK
    of the filer that submitted it. A filing agent uses a number that is not
    a CIK; that prefix is not a registrant.
    """
    head = str(accession).split("-", 1)[0]
    if len(head) != 10 or not head.isdigit():
        return None
    return int(head)


def _company_by_cik(cik: int):
    """The registrant for `cik`, or None when SEC has no such submissions."""
    from edgar import Company

    found = Company(int(cik))
    if getattr(found, "not_found", False):
        return None
    if int(getattr(found, "cik", -1)) in (-1, -999999999):
        return None
    return found


# Reports a registrant files about itself. A co-registrant link is one of
# these listed under both CIKs. Anything else in a company's list can be
# another party's filing about it: a SCHEDULE 13G sits under the subject's CIK
# with the holder's CIK in its accession, and a holder such as BlackRock files
# 10-Ks of its own.
_OWN_REPORTS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A"})
_PREDECESSOR_CACHE: Dict[Tuple[int, str], Any] = {}


def _accession_of(filing) -> str:
    return (getattr(filing, "accession_number", None)
            or getattr(filing, "accession_no", None) or "")


def predecessor_with_form(company, form_type: str):
    """A co-registrant that has filed `form_type`, or None.

    SEC lists a co-registrant filing under both CIKs. The accession begins
    with the CIK that submitted it. Only a report in `_OWN_REPORTS` that SEC
    lists under both CIKs links them. XOM's 2026-08-03 10-Q is
    0000034088-26-000093 under both ExxonMobil Holdings Corp (2115436) and
    EXXON MOBIL CORP (34088), and the filing names both as filers. The
    holding company's 8-K12B names only itself, so it does not supply the
    predecessor. A prefix that is not a registrant is a filing agent, not a
    predecessor. More than one co-registrant with the form is a refusal:
    choosing one would be a guess.
    """
    own = int(company.cik)
    cache_key = (own, form_type)
    if cache_key in _PREDECESSOR_CACHE:
        return _PREDECESSOR_CACHE[cache_key]

    listed = company.get_filings(trigger_full_load=False)
    shared: Dict[int, set] = {}
    for filing in listed:
        if getattr(filing, "form", None) not in _OWN_REPORTS:
            continue
        accession = _accession_of(filing)
        cik = _accession_cik(accession)
        if cik is None or cik == own:
            continue
        shared.setdefault(cik, set()).add(accession)

    found = []
    for cik, accessions in shared.items():
        other = _company_by_cik(cik)
        if other is None:
            continue
        # The prefix says only who submitted the filing. SEC listing the same
        # accession under the other CIK is what makes it a co-registrant.
        theirs = {_accession_of(f)
                  for f in other.get_filings(trigger_full_load=False)}
        if not accessions & theirs:
            continue
        if other.get_filings(form=form_type, amendments=False):
            found.append(other)
    if len(found) > 1:
        names = ", ".join(
            f"{item.name} (CIK {int(item.cik)})" for item in found)
        raise RuntimeError(
            f"CIK {own} lists filings submitted by more than one registrant "
            f"that has a {form_type}: {names}. Refused rather than choosing.")
    res = found[0] if found else None
    _PREDECESSOR_CACHE[cache_key] = res
    return res


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


def filings_history(
    comp,
    form: str,
    limit: Optional[int] = None,
    min_count: Optional[int] = None,
    since: Optional[str] = None,
    amendments: bool = True,
) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
    """A form's filing history across `comp` and its co-registrant.

    Returns `(filings, read_from)`.
    Own filings come first, then the co-registrant's, ordered newest first by
    filing_date.
    The co-registrant is only queried when the registrant's own history is
    short for the purpose:
      - if not own: True
      - elif since is provided: oldest own filing date > since
      - elif min_count is provided: len(own) < min_count
      - else: False
    When filings from the co-registrant are added, `read_from` is
    `{'name': predecessor.name, 'cik': int(predecessor.cik)}`; otherwise None.
    """
    if amendments is False:
        raw_own = comp.get_filings(form=form, amendments=False)
    else:
        raw_own = comp.get_filings(form=form)
    if hasattr(raw_own, "head"):
        own = list(raw_own.head(limit) if limit is not None else (raw_own if hasattr(raw_own, "__iter__") else raw_own.head(100)))
    else:
        own = list(raw_own)

    if not own:
        is_short = True
    elif since is not None:
        oldest_date = min(str(getattr(f, "filing_date", "")) for f in own)
        is_short = oldest_date > str(since)
    elif min_count is not None:
        is_short = len(own) < min_count
    else:
        is_short = False

    if not is_short:
        if limit is not None:
            own = own[:limit]
        return own, None

    predecessor = predecessor_with_form(comp, form)
    if predecessor is None:
        if limit is not None:
            own = own[:limit]
        return own, None

    co_info = {"name": predecessor.name, "cik": int(predecessor.cik)}
    if amendments is False:
        raw_theirs = predecessor.get_filings(form=form, amendments=False)
    else:
        raw_theirs = predecessor.get_filings(form=form)
    if hasattr(raw_theirs, "head"):
        theirs = list(raw_theirs.head(limit) if limit is not None else (raw_theirs if hasattr(raw_theirs, "__iter__") else raw_theirs.head(100)))
    else:
        theirs = list(raw_theirs)

    seen_accessions = {_accession_of(f) for f in own if _accession_of(f)}
    added_from_theirs = []
    for f in theirs:
        acc = _accession_of(f)
        if acc and acc in seen_accessions:
            continue
        f.read_from = co_info
        added_from_theirs.append(f)

    if not added_from_theirs:
        if limit is not None:
            own = own[:limit]
        return own, None

    combined = own + added_from_theirs
    combined.sort(key=lambda f: str(getattr(f, "filing_date", "")), reverse=True)
    if limit is not None:
        combined = combined[:limit]

    used_theirs = any(f in added_from_theirs for f in combined)
    return combined, (co_info if used_theirs else None)

