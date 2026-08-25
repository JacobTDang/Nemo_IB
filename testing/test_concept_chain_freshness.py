"""A chain must pick the freshest element, not the first one that answers.

`earnings_quality._series` already states why: "Ford's FY2025 10-K tags
`us-gaap:ProfitLoss` and not `us-gaap:NetIncomeLoss`. Stopping at the first
hit found NetIncomeLoss in the FY2024 filing and reported Ford's latest year
as +5.9bn of net income when the year it had just reported was an 8.2bn
loss."

`sbc._consolidated_by_filing` and `forward_metrics._series_for` both stopped
at the first concept that returned anything. NVDA is the live case: it tags
`us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` only in its
FY2022 10-K and `us-gaap:Revenues` since. The chain lists the ASC 606
element first, so get_sbc_series resolved revenue for 2022 alone and
reported `pct_of_revenue: None` for all four recent years -- the column the
tool exists to produce, blank, with no reason given.
"""
from typing import Dict, List

import pytest

from tools.web_search_server.sec_series import ConceptFact, FilingPoint
import tools.web_search_server.sbc as sbc
import tools.web_search_server.forward_metrics as fm

# Filing date -> value, for each element. The first element of each chain is
# the abandoned one, covering only the oldest filing.
ABANDONED = {"2022-03-18": 26_914_000_000.0}
CURRENT = {"2026-02-25": 130_497_000_000.0, "2025-02-26": 60_922_000_000.0,
           "2024-02-21": 26_974_000_000.0, "2023-02-24": 26_974_000_000.0,
           "2022-03-18": 26_914_000_000.0}


def _points(by_filing: Dict[str, float], concept: str) -> List[FilingPoint]:
    out = []
    for filing_date, value in sorted(by_filing.items(), reverse=True):
        period = f"{int(filing_date[:4]) - 1}-12-31"
        out.append(FilingPoint(
            filing_date=filing_date, form="10-K", accession=f"a-{filing_date}",
            facts=[ConceptFact(value=value, period=period,
                               context_ref=f"c-{filing_date}", concept=concept,
                               unit="USD")]))
    return out


@pytest.fixture
def _nvidia(monkeypatch):
    """The element switch, applied to whichever module is under test."""
    def install(module, chain):
        def fetch(ticker, concept, form="10-K", limit=8):
            if concept == chain[0]:
                return _points(ABANDONED, concept)
            if concept == chain[1]:
                return _points(CURRENT, concept)
            return []
        monkeypatch.setattr(module, "fetch_concept_series", fetch)
    return install


def test_sbc_revenue_chain_prefers_the_element_covering_the_latest_filing(_nvidia):
    _nvidia(sbc, sbc.REVENUE_CONCEPTS)
    values, concept = sbc._consolidated_by_filing(
        "NVDA", sbc.REVENUE_CONCEPTS, "10-K", 5)

    assert concept == sbc.REVENUE_CONCEPTS[1], (
        f"chose {concept}, which NVDA abandoned after FY2022")
    assert max(values) == "2026-02-25", (
        f"newest filing resolved is {max(values)}; the ratio column is blank "
        f"for every year after it")
    assert len(values) == 5


def test_forward_metrics_chain_prefers_the_element_covering_the_latest_filing(
        _nvidia):
    _nvidia(fm, fm.REVENUE_CONCEPTS)
    rows, concept = fm._series_for("NVDA", fm.REVENUE_CONCEPTS, "10-K", 5)

    assert concept == fm.REVENUE_CONCEPTS[1]
    assert rows[0]["filing_date"] == "2026-02-25"
    assert len(rows) == 5


def test_chain_order_still_breaks_a_tie(_nvidia):
    """Freshness decides; equal freshness keeps the chain's preference."""
    def fetch(ticker, concept, form="10-K", limit=8):
        return _points(CURRENT, concept)  # every element covers everything

    import tools.web_search_server.sbc as mod
    mod_fetch = mod.fetch_concept_series
    mod.fetch_concept_series = fetch
    try:
        _, concept = mod._consolidated_by_filing(
            "NVDA", mod.REVENUE_CONCEPTS, "10-K", 5)
    finally:
        mod.fetch_concept_series = mod_fetch

    assert concept == mod.REVENUE_CONCEPTS[0], (
        "with equal coverage the chain's first element must win, which is "
        "what keeps the more specific ASC 606 element preferred")
