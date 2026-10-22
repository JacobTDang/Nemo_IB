"""Reading one filing: its 8-K items from the index, and errors that name it.

EDGAR's submissions index lists the items each 8-K reports next to its
accession number -- "2.02,7.01,8.01,9.01" -- and edgartools carries that list
on every filing `get_filings` returns, so reading it costs no request.

Both 8-K readers in this package used to parse the document for the items
instead, and both swallowed the failure. When SEC began to answer edgartools
5.8.0's filing index URL with 403 (#138), every parse failed. WINA reports
Item 2.02 on 17 of its last 30 8-Ks, and `get_earnings_releases` answered that
it had none. A read that fails must say which filing it could not read.
"""
from __future__ import annotations

from typing import Any, List


def filing_label(filing: Any) -> str:
    """One filing as an error names it: form, accession number, filing date."""
    form = getattr(filing, 'form', None) or 'filing'
    accession = (getattr(filing, 'accession_no', None)
                 or getattr(filing, 'accession_number', None)
                 or 'with no accession number')
    filed = getattr(filing, 'filing_date', None)
    return f'{form} {accession}' + (f' filed {filed}' if filed else '')


def index_items(filing: Any) -> List[str]:
    """The item codes the filing index lists for one 8-K, in index order.

    Raises ValueError when the index lists none. Every 8-K reports at least
    one item, so an empty entry is a gap in what EDGAR returned, never an 8-K
    about nothing -- and read as "no Item 2.02" it becomes a claim about the
    company. A list is accepted as well as the comma-joined string, because a
    shape change upstream would otherwise match nothing.
    """
    raw = getattr(filing, 'items', None)
    codes = raw if isinstance(raw, (list, tuple)) else str(raw or '').split(',')
    codes = [str(c).strip() for c in codes if str(c).strip()]
    if not codes:
        raise ValueError(
            f'Could not read the 8-K items of {filing_label(filing)}: the '
            f'filing index lists none (items={raw!r}). Looked for the items '
            f'column of the EDGAR submissions index.')
    return codes
