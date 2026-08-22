"""Multi-filing XBRL access.

Nothing else in the SEC layer walks more than one filing. get_segment_financials
gets five years from a single filing's comparative periods, and get_historical_fcf
reads only the latest despite its name. Share count lives on the cover page as one
instant per filing, so a dilution series is impossible without this module.

The subtle part is dimension resolution. A fact dataframe carries no dimension
column, and by_dimension() returns empty for cover-page concepts, so class identity
has to come from xbrl.contexts[context_ref].dimensions.
"""
import os

import pytest

from tools.web_search_server.sec_series import (
    ConceptFact,
    FilingPoint,
    NotCovered,
    fetch_concept_series,
    resolve_dimensions,
)

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")


class _Ctx:
    def __init__(self, dimensions):
        self.dimensions = dimensions


class _FakeXBRL:
    def __init__(self, contexts):
        self.contexts = contexts


def test_undimensioned_context_resolves_to_empty():
    """Single-class filers emit one fact with no dimensions at all."""
    xbrl = _FakeXBRL({"c-1": _Ctx({})})
    assert resolve_dimensions(xbrl, "c-1") == {}


def test_dimensioned_context_returns_its_members():
    xbrl = _FakeXBRL({"c-28": _Ctx(
        {"us-gaap:StatementClassOfStockAxis": "us-gaap:CommonClassAMember"})})
    assert resolve_dimensions(xbrl, "c-28") == {
        "us-gaap:StatementClassOfStockAxis": "us-gaap:CommonClassAMember"}


def test_missing_context_resolves_to_empty_rather_than_raising():
    """A fact referencing an absent context is a malformed filing, not a crash."""
    assert resolve_dimensions(_FakeXBRL({}), "c-99") == {}


def test_context_without_dimensions_attribute_resolves_to_empty():
    xbrl = _FakeXBRL({"c-1": object()})
    assert resolve_dimensions(xbrl, "c-1") == {}


def test_concept_fact_carries_its_dimensions():
    fact = ConceptFact(value=835000000.0, period="2026-07-15",
                       dimensions={"us-gaap:StatementClassOfStockAxis":
                                   "us-gaap:CommonClassBMember"},
                       context_ref="c-29")
    assert fact.value == 835000000.0
    assert fact.dimension_member("us-gaap:StatementClassOfStockAxis") == \
        "us-gaap:CommonClassBMember"


def test_concept_fact_missing_axis_returns_none():
    fact = ConceptFact(value=1.0, period="2026-01-01", dimensions={}, context_ref="c-1")
    assert fact.dimension_member("us-gaap:StatementClassOfStockAxis") is None


def test_filing_point_totals_across_facts():
    """The whole point: a multi-class filer's total is the sum of its classes."""
    point = FilingPoint(
        filing_date="2026-07-23", form="10-Q", accession="x",
        facts=[
            ConceptFact(5868000000.0, "2026-07-15", {"a": "ClassA"}, "c-28"),
            ConceptFact(835000000.0, "2026-07-15", {"a": "ClassB"}, "c-29"),
            ConceptFact(5527000000.0, "2026-07-15", {"a": "CapitalClassC"}, "c-30"),
        ])
    assert point.total() == 12230000000.0
    assert len(point.facts) == 3


def test_filing_point_total_of_single_fact():
    point = FilingPoint(filing_date="2026-04-29", form="10-Q", accession="y",
                        facts=[ConceptFact(7428434704.0, "2026-04-15", {}, "c-1")])
    assert point.total() == 7428434704.0


def test_empty_filing_point_total_is_none_not_zero():
    """Zero would read as 'no shares outstanding'. Absence must be distinguishable."""
    point = FilingPoint(filing_date="2026-04-29", form="10-Q", accession="y", facts=[])
    assert point.total() is None


def test_not_covered_is_an_exception():
    assert issubclass(NotCovered, Exception)


# --------------------------------------------------------------------------
# Live EDGAR golden tests.
#
# These pin the two code paths that matter. Mocking them would only assert that
# the mock matches my belief about EDGAR, which is exactly the belief that needs
# checking -- the multi-class shape was not obvious and cost a live probe to find.
# --------------------------------------------------------------------------

SHARES_CONCEPT = "dei:EntityCommonStockSharesOutstanding"
CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_single_class_filer_returns_one_undimensioned_fact():
    """MSFT has one class of common stock, so each filing yields exactly one
    fact carrying no dimensions."""
    points = fetch_concept_series("MSFT", SHARES_CONCEPT, form="10-Q", limit=3)
    assert points, "no filings returned for MSFT"
    for point in points:
        assert len(point.facts) == 1, (
            f"MSFT {point.filing_date} returned {len(point.facts)} facts; "
            f"expected 1 for a single-class filer")
        assert point.facts[0].dimensions == {}
        assert point.total() > 7_000_000_000


@network
def test_multi_class_filer_returns_every_class():
    """GOOGL reports Class A, B, and C separately. This is the regression that
    matters: taking the first fact understates share count by 52%."""
    points = fetch_concept_series("GOOGL", SHARES_CONCEPT, form="10-Q", limit=2)
    assert points, "no filings returned for GOOGL"

    latest = points[0]
    assert len(latest.facts) == 3, (
        f"GOOGL returned {len(latest.facts)} share-count facts; expected 3 "
        f"(Class A, B, C). A missing class silently understates dilution.")

    members = set(latest.by_axis(CLASS_AXIS).keys())
    assert "us-gaap:CommonClassAMember" in members
    assert "us-gaap:CommonClassBMember" in members
    assert any("ClassC" in m for m in members), (
        f"no Class C member among {members}. Alphabet tags Class C as a "
        f"company-specific member (goog:CapitalClassCMember), so a whitelist "
        f"of us-gaap members drops it.")

    # Alphabet's real share count is ~12.2bn. The first fact alone is ~5.87bn.
    assert latest.total() > 11_000_000_000, (
        f"GOOGL total {latest.total():,.0f} is implausibly low -- a share class "
        f"was dropped")


@network
def test_absent_concept_raises_not_covered_rather_than_returning_empty():
    with pytest.raises(NotCovered):
        fetch_concept_series("MSFT", "us-gaap:ThisConceptDoesNotExist",
                             form="10-Q", limit=2)


# --------------------------------------------------------------------------
# Regressions found by probing duration concepts (SBC). The share-count path
# never exposed these because instants are always populated and cleanly typed.
# --------------------------------------------------------------------------

def test_nan_is_not_treated_as_a_present_value():
    """pandas returns float('nan') for a missing cell, and nan is truthy, so
    `row.get(a) or row.get(b)` silently keeps the nan instead of falling back."""
    from tools.web_search_server.sec_series import _clean_number
    assert _clean_number(float("nan")) is None
    assert _clean_number(None) is None
    assert _clean_number("") is None
    assert _clean_number(3.5) == 3.5
    assert _clean_number("1234") == 1234.0


def test_period_falls_back_to_key_when_instant_is_nan():
    """Duration concepts (SBC, revenue) have no period_instant at all."""
    from tools.web_search_server.sec_series import _clean_period
    row = {"period_instant": float("nan"),
           "period_key": "duration_2025-01-27_2026-01-25"}
    assert _clean_period(row) == "duration_2025-01-27_2026-01-25"


def test_period_prefers_instant_when_present():
    from tools.web_search_server.sec_series import _clean_period
    row = {"period_instant": "2026-07-15", "period_key": "instant_2026-07-15"}
    assert _clean_period(row) == "2026-07-15"


def test_undimensioned_selects_only_consolidated_facts():
    """NVDA reports 59+ SBC facts per filing, nearly all broken out by award
    type. Those are components of the total, not additions to it -- summing
    them would multiply the real figure several times over."""
    point = FilingPoint("2026-02-25", "10-K", "acc", facts=[
        ConceptFact(6386000000.0, "duration_2025_2026", {}, "c-1"),
        ConceptFact(70000000.0, "duration_2025_2026",
                    {"us-gaap:AwardTypeAxis": "nvda:RSUsPSUsMember"}, "c-2"),
        ConceptFact(89000000.0, "duration_2025_2026",
                    {"us-gaap:AwardTypeAxis": "nvda:MarketbasedPSUMember"}, "c-3"),
    ])
    assert [f.value for f in point.undimensioned()] == [6386000000.0]


def test_latest_undimensioned_picks_the_most_recent_period():
    """A 10-K carries three comparative years for a duration concept. The
    filing's own year is the one we want, not the sum and not an arbitrary row."""
    point = FilingPoint("2026-02-25", "10-K", "acc", facts=[
        ConceptFact(3549000000.0, "duration_2023-01-30_2024-01-28", {}, "c-3"),
        ConceptFact(6386000000.0, "duration_2025-01-27_2026-01-25", {}, "c-1"),
        ConceptFact(4737000000.0, "duration_2024-01-29_2025-01-26", {}, "c-2"),
    ])
    assert point.latest_undimensioned().value == 6386000000.0


def test_latest_undimensioned_is_none_when_every_fact_is_dimensioned():
    point = FilingPoint("2026-02-25", "10-K", "acc", facts=[
        ConceptFact(70000000.0, "d", {"us-gaap:AwardTypeAxis": "x"}, "c-2"),
    ])
    assert point.latest_undimensioned() is None


def test_annual_fact_beats_a_quarterly_one_ending_the_same_day():
    """Lexicographic max on the period string is wrong.

    'duration_2025-10-27_2026-01-25' (Q4) sorts ABOVE
    'duration_2025-01-27_2026-01-25' (FY) because '10' > '01' at the month
    position. Selecting by string would silently return one quarter's expense
    as the year's -- roughly a 4x understatement that looks like a real number.
    """
    point = FilingPoint("2026-02-25", "10-K", "acc", facts=[
        ConceptFact(6386000000.0, "duration_2025-01-27_2026-01-25", {}, "fy"),
        ConceptFact(1800000000.0, "duration_2025-10-27_2026-01-25", {}, "q4"),
    ])
    assert point.latest_undimensioned().value == 6386000000.0


def test_most_recent_year_wins_over_an_older_one():
    point = FilingPoint("2026-02-25", "10-K", "acc", facts=[
        ConceptFact(4737000000.0, "duration_2024-01-29_2025-01-26", {}, "prior"),
        ConceptFact(6386000000.0, "duration_2025-01-27_2026-01-25", {}, "current"),
    ])
    assert point.latest_undimensioned().value == 6386000000.0


def test_instant_facts_still_select_by_latest_date():
    point = FilingPoint("2026-07-23", "10-Q", "acc", facts=[
        ConceptFact(100.0, "2026-01-15", {}, "old"),
        ConceptFact(200.0, "2026-07-15", {}, "new"),
    ])
    assert point.latest_undimensioned().value == 200.0
