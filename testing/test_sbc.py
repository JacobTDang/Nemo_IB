"""Stock-based compensation.

The largest single bridge between GAAP and "adjusted" earnings, and the other
engine of dilution alongside shelf issuance. Previously had zero coverage.

The hazard here is not a missing number, it is a wrong one. NVDA reports 51
SBC facts in a single 10-K, nearly all broken out by award type. Those are
components of the consolidated figure, so anything that sums facts reports
several times the real expense.
"""
import os

import pytest

from tools.web_search_server import sbc
from tools.web_search_server.sec_series import ConceptFact, FilingPoint, NotCovered

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")


def _point(filing_date, facts):
    return FilingPoint(filing_date, "10-K", "acc", facts=facts)


def _fact(value, period="duration_2025-01-27_2026-01-25", dims=None):
    return ConceptFact(value, period, dims or {}, "c-1")


def test_consolidated_value_is_used_not_the_sum(monkeypatch):
    """The regression that matters: award-type breakdowns must not be added."""
    monkeypatch.setattr(sbc, "fetch_concept_series", lambda t, c, **k: [
        _point("2026-02-25", [
            _fact(6386000000.0),
            _fact(70000000.0, dims={"us-gaap:AwardTypeAxis": "nvda:RSUMember"}),
            _fact(89000000.0, dims={"us-gaap:AwardTypeAxis": "nvda:PSUMember"}),
        ])])
    result = sbc.get_sbc_series("NVDA", limit=1)
    assert result["success"] is True
    assert result["series"][0]["sbc"] == 6386000000.0


def test_series_is_newest_first(monkeypatch):
    monkeypatch.setattr(sbc, "fetch_concept_series", lambda t, c, **k: [
        _point("2026-02-25", [_fact(6386000000.0, "duration_2025-01-27_2026-01-25")]),
        _point("2025-02-26", [_fact(4737000000.0, "duration_2024-01-29_2025-01-26")]),
    ])
    result = sbc.get_sbc_series("NVDA", limit=2)
    assert [row["sbc"] for row in result["series"]] == [6386000000.0, 4737000000.0]


def test_concept_chain_falls_through_to_the_next_tag(monkeypatch):
    """Filers tag SBC under several concepts. The first that yields data wins."""
    calls = []

    def fake(ticker, concept, **kwargs):
        calls.append(concept)
        if concept != "us-gaap:AllocatedShareBasedCompensationExpense":
            raise NotCovered(concept)
        return [_point("2026-02-25", [_fact(500.0)])]

    monkeypatch.setattr(sbc, "fetch_concept_series", fake)
    result = sbc.get_sbc_series("ODDTAG", limit=1)
    assert result["success"] is True
    assert result["concept_used"] == "us-gaap:AllocatedShareBasedCompensationExpense"
    assert len(calls) >= 2


def test_uncovered_returns_explicit_failure_not_zero(monkeypatch):
    def fake(ticker, concept, **kwargs):
        raise NotCovered(concept)

    monkeypatch.setattr(sbc, "fetch_concept_series", fake)
    result = sbc.get_sbc_series("NOSBC", limit=1)
    assert result["success"] is False
    assert result["series"] == []
    assert "not covered" in result["error"].lower()


def test_ratios_are_none_when_the_denominator_is_missing(monkeypatch):
    """A missing revenue figure must not silently become a zero percentage."""
    monkeypatch.setattr(sbc, "fetch_concept_series", lambda t, c, **k: (
        [_point("2026-02-25", [_fact(6386000000.0)])]
        if "ShareBased" in c else (_ for _ in ()).throw(NotCovered(c))))
    result = sbc.get_sbc_series("NOREV", limit=1)
    assert result["series"][0]["pct_of_revenue"] is None
    assert result["series"][0]["pct_of_ocf"] is None


def test_ratio_is_computed_when_revenue_is_available(monkeypatch):
    def fake(ticker, concept, **kwargs):
        if "ShareBased" in concept:
            return [_point("2026-02-25", [_fact(1000.0)])]
        if "Revenue" in concept:
            return [_point("2026-02-25", [_fact(10000.0)])]
        raise NotCovered(concept)

    monkeypatch.setattr(sbc, "fetch_concept_series", fake)
    result = sbc.get_sbc_series("HASREV", limit=1)
    assert result["series"][0]["pct_of_revenue"] == pytest.approx(10.0)


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_nvda_sbc_matches_the_filing():
    """NVDA FY2026 SBC is 6.386bn. A sum of the award-type breakdowns would be
    far larger, and one quarter would be far smaller."""
    result = sbc.get_sbc_series("NVDA", limit=2, form="10-K")
    assert result["success"] is True
    assert result["series"][0]["sbc"] == pytest.approx(6_386_000_000.0, rel=0.001)


@network
def test_msft_sbc_matches_the_filing():
    """MSFT FY2026 SBC is 12.405bn against a June fiscal year end."""
    result = sbc.get_sbc_series("MSFT", limit=2, form="10-K")
    assert result["success"] is True
    assert result["series"][0]["sbc"] == pytest.approx(12_405_000_000.0, rel=0.001)


@network
def test_sbc_as_a_share_of_revenue_is_plausible():
    """A sanity bound that a summed-breakdown bug would fail outright: SBC is a
    meaningful slice of revenue but never a majority of it for these filers."""
    result = sbc.get_sbc_series("NVDA", limit=1, form="10-K")
    pct = result["series"][0]["pct_of_revenue"]
    assert pct is not None, "revenue not resolved, ratio unavailable"
    assert 0 < pct < 40, f"SBC at {pct:.1f}% of revenue is implausible"
