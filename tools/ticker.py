"""One spelling of a security, for joining sources that disagree about it.

Congressional filings write class shares with a dot because that is what the
filer typed -- `BRK.B`, `BF.B`, `HEI.A`. Market-data providers use a dash:
`BRK-B`. 108 rows in the congressional store use the dotted form, and a
workflow joining that flow to market data on one spelling got rows from one
side and silence from the other.

Stored text is never rewritten. A filing said `BRK.B` and changing it would
put words in the filer's mouth; the filing is the record. Only lookups
normalise, and responses report which form was resolved so a caller can see
that two sources were joined on the same security.
"""
from __future__ import annotations

from typing import List, Optional


def normalize_ticker(ticker: Optional[str]) -> Optional[str]:
    """A single comparable form, or None when there is nothing to compare.

    None rather than "" because an empty ticker is the absence of a filter,
    and a caller that cannot tell those apart filters on nothing by accident.
    """
    if not ticker:
        return None
    cleaned = str(ticker).strip().upper().replace(".", "-")
    return cleaned or None


def ticker_variants(ticker: Optional[str]) -> List[str]:
    """Every spelling worth trying against a provider, best guess first.

    Yahoo uses the dash, so that leads. The dotted form follows because some
    sources keep it and a lookup that tries only one is a lookup that fails on
    half the class shares in existence.
    """
    canonical = normalize_ticker(ticker)
    if canonical is None:
        return []
    variants = [canonical]
    dotted = canonical.replace("-", ".")
    if dotted != canonical:
        variants.append(dotted)
    return variants
