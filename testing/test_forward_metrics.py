"""Contracted revenue, geographic exposure, and public float.

Three gaps that make a researcher confidently wrong rather than merely
uninformed:

- **Contracted revenue.** RPO is the best forward indicator an enterprise
  filer publishes -- signed revenue not yet recognised. It was only ever
  scraped from MD&A prose here, never read as a number.
- **Geographic revenue.** Business segments were covered; geography was not,
  so China and FX exposure were invisible.
- **Public float.** Share count was covered but float was not, and they differ
  enormously for founder-controlled names. Float is what actually trades.

The dimension handling repeats a lesson learned on share classes: members mix
standard tags (country:US) with company-specific ones
(nvda:ChinaIncludingHongKongMember), so a whitelist silently drops geographies.
"""
import os

import pytest

from tools.web_search_server import forward_metrics as fm
from tools.web_search_server.sec_series import ConceptFact, FilingPoint, NotCovered

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")

GEO_AXIS = "srt:StatementGeographicalAxis"


def _point(facts, filing_date="2026-02-25"):
    return FilingPoint(filing_date, "10-K", "acc", facts=facts)


def _fact(value, dims=None, period="duration_2025-01-27_2026-01-25", ref="c-1"):
    return ConceptFact(value, period, dims or {}, ref)


# ------------------------------------------------------------ contracted revenue

def test_rpo_uses_the_consolidated_total_not_a_breakdown(monkeypatch):
    """MSFT reports RPO of 684bn alongside a 678bn commercial-customer
    breakdown and two zero-valued timing placeholders. Summing gives nonsense;
    the undimensioned fact is the total."""
    monkeypatch.setattr(fm, "fetch_concept_series", lambda t, c, **k: [_point([
        _fact(684_000_000_000.0),
        _fact(678_000_000_000.0, {"srt:MajorCustomersAxis": "msft:CommercialCustomersMember"}),
        _fact(0.0, {"us-gaap:RevenueRemainingPerformanceObligationExpectedTimingOfSatisfactionStartDateAxis": "2026-07-01"}),
    ])])
    result = fm.get_contracted_revenue("MSFT")
    assert result["success"] is True
    assert result["rpo"][0]["value"] == 684_000_000_000.0


def test_deferred_revenue_falls_through_the_concept_chain(monkeypatch):
    """ASC 606 renamed deferred revenue to contract liability, and filers use
    either. MSFT tags ContractWithCustomerLiabilityCurrent and not
    DeferredRevenueCurrent."""
    seen = []

    def fake(ticker, concept, **kwargs):
        seen.append(concept)
        if "RemainingPerformanceObligation" in concept:
            raise NotCovered(concept)
        if concept == "us-gaap:ContractWithCustomerLiabilityCurrent":
            return [_point([_fact(72_965_000_000.0)])]
        raise NotCovered(concept)

    monkeypatch.setattr(fm, "fetch_concept_series", fake)
    result = fm.get_contracted_revenue("MSFT")
    assert result["deferred_revenue"][0]["value"] == 72_965_000_000.0
    assert result["deferred_concept_used"] == "us-gaap:ContractWithCustomerLiabilityCurrent"


def test_neither_concept_covered_is_an_explicit_failure(monkeypatch):
    def fake(ticker, concept, **kwargs):
        raise NotCovered(concept)
    monkeypatch.setattr(fm, "fetch_concept_series", fake)
    result = fm.get_contracted_revenue("NOTAGS")
    assert result["success"] is False
    assert result["rpo"] == [] and result["deferred_revenue"] == []
    assert "not covered" in result["error"].lower()


def test_a_filer_with_only_deferred_revenue_still_succeeds(monkeypatch):
    """Most non-SaaS filers report deferred revenue but no RPO. That is a
    partial answer, not a failure."""
    def fake(ticker, concept, **kwargs):
        if "RemainingPerformanceObligation" in concept:
            raise NotCovered(concept)
        if "ContractWithCustomerLiability" in concept:
            return [_point([_fact(500.0)])]
        raise NotCovered(concept)
    monkeypatch.setattr(fm, "fetch_concept_series", fake)
    result = fm.get_contracted_revenue("RETAILER")
    assert result["success"] is True
    assert result["rpo"] == []
    assert result["deferred_revenue"][0]["value"] == 500.0


# --------------------------------------------------------------- geographic mix

def test_geographic_revenue_groups_by_member(monkeypatch):
    monkeypatch.setattr(fm, "fetch_concept_series", lambda t, c, **k: [_point([
        _fact(149_617_000_000.0, {GEO_AXIS: "country:US"}),
        _fact(42_345_000_000.0, {GEO_AXIS: "country:TW"}),
        _fact(19_677_000_000.0, {GEO_AXIS: "nvda:ChinaIncludingHongKongMember"}),
        _fact(999.0),  # consolidated total, must not appear as a geography
    ])])
    result = fm.get_geographic_revenue("NVDA")
    assert result["success"] is True
    by = {g["region"]: g["periods"][0]["value"] for g in result["by_region"]}
    assert by["United States"] == 149_617_000_000.0
    assert by["Taiwan"] == 42_345_000_000.0


def test_company_specific_region_is_kept_not_dropped(monkeypatch):
    """A whitelist of country: tags would silently drop NVDA's China figure."""
    monkeypatch.setattr(fm, "fetch_concept_series", lambda t, c, **k: [_point([
        _fact(19_677_000_000.0, {GEO_AXIS: "nvda:ChinaIncludingHongKongMember"}),
    ])])
    result = fm.get_geographic_revenue("NVDA")
    regions = [g["region"] for g in result["by_region"]]
    assert any("China" in r for r in regions), regions


def test_undimensioned_total_is_excluded_from_regions(monkeypatch):
    monkeypatch.setattr(fm, "fetch_concept_series", lambda t, c, **k: [_point([
        _fact(500.0), _fact(300.0, {GEO_AXIS: "country:US"}),
    ])])
    result = fm.get_geographic_revenue("X")
    assert [g["region"] for g in result["by_region"]] == ["United States"]


def test_concentration_pct_is_reported(monkeypatch):
    """The reason to call this: how much revenue sits in one country."""
    monkeypatch.setattr(fm, "fetch_concept_series", lambda t, c, **k: [_point([
        _fact(750.0, {GEO_AXIS: "country:US"}),
        _fact(250.0, {GEO_AXIS: "country:CN"}),
    ])])
    result = fm.get_geographic_revenue("X")
    top = result["by_region"][0]
    assert top["region"] == "United States"
    assert top["pct_of_total"] == pytest.approx(75.0)


# ---------------------------------------------------------------- public float

def test_float_is_reported_with_its_filing_date(monkeypatch):
    monkeypatch.setattr(fm, "fetch_concept_series",
                        lambda t, c, **k: [_point([_fact(3_600_000_000_000.0)],
                                                  filing_date="2026-07-29")])
    result = fm.get_public_float("MSFT")
    assert result["success"] is True
    assert result["public_float"] == 3_600_000_000_000.0
    assert result["filing_date"] == "2026-07-29"


def test_missing_float_is_explicit(monkeypatch):
    def fake(ticker, concept, **kwargs):
        raise NotCovered(concept)
    monkeypatch.setattr(fm, "fetch_concept_series", fake)
    result = fm.get_public_float("NOFLOAT")
    assert result["success"] is False
    assert result["public_float"] is None


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_msft_contracted_revenue_matches_the_filing():
    result = fm.get_contracted_revenue("MSFT")
    assert result["success"] is True
    assert result["rpo"], "MSFT discloses RPO"
    assert result["rpo"][0]["value"] > 500e9


@network
def test_nvda_geographic_mix_shows_taiwan_and_china():
    result = fm.get_geographic_revenue("NVDA")
    assert result["success"] is True
    regions = " ".join(g["region"] for g in result["by_region"])
    assert "Taiwan" in regions
    assert "China" in regions


@network
def test_msft_public_float_is_plausible():
    result = fm.get_public_float("MSFT")
    assert result["success"] is True
    assert result["public_float"] > 1e12
