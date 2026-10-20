"""Which registrant a job reads, by CIK where it has one.

edgartools resolves a ticker from a parquet file bundled with the package
before it asks SEC, and a ticker missing from that file maps to nothing.
`Company(ticker)` does not raise for one. It returns a company whose CIK is
-999999999 and whose filing list is empty. Against the 2026-10-19 universe
that was 103 of 1,810 eligible names, every one a registrant that SEC's own
ticker file maps (#130). DMC was one: the watcher refused it on every pass
while its folder held 14 SC 13D filings, and the announcement job recorded it
as filing no earnings releases when its latest 40 8-Ks held 19.

The universe carries each name's CIK from SEC's file, and `sue.eps_series`
returns one from the same file, so a job that has either asks by CIK and the
bundled map never enters into it. A caller with only a ticker still asks by
ticker, and an answer from an unresolvable ticker is refused here rather than
read as a company that files nothing. Empty is the same shape a quiet company
returns, and a record that cannot tell the two apart is the failure every job
in this package is built to avoid.
"""
from __future__ import annotations

from typing import Any, Optional


def _resolved_cik(ticker: str, cik: Any) -> str:
    """The registrant's CIK, or a refusal. Never a sentinel.

    `Company("PARA")` does not raise for a ticker EDGAR no longer maps. It
    returns a company whose CIK is -999999999 and whose filing list is empty --
    observed live. Without this a watchlist could rot one delisted name at a
    time, each reporting "no filings" forever and counting as coverage while it
    did.
    """
    try:
        value = int(cik)
    except (TypeError, ValueError):
        value = -1
    if value <= 0:
        raise LookupError(
            f"EDGAR does not map {ticker} to a registrant (CIK {cik!r}). "
            f"Refused rather than treated as a company with no filings, which "
            f"is what an empty result from an unresolvable ticker looks like.")
    return str(value)


def company(ticker: str, cik: Optional[str] = None):
    """`edgar.Company` for one name, by its CIK when the caller has one.

    `cik` is SEC's, zero-padded or not. It goes straight to the submissions
    index, so a name the bundled ticker map lacks is read like any other. With
    no CIK the ticker is asked for, and the sentinel refused.

    The CIK route has its own empty answer: edgartools turns SEC's 404 into a
    company flagged `not_found` with no filings. That is refused too. The flag
    reads the same submissions index `get_filings` reads, so it costs no
    request of its own.
    """
    from edgar import Company

    found = Company(ticker) if cik is None else Company(int(cik))
    _resolved_cik(ticker, found.cik)
    if found.not_found:
        raise LookupError(
            f"SEC has no submissions for {ticker} under CIK {found.cik}. "
            f"Refused rather than treated as a company with no filings.")
    return found
