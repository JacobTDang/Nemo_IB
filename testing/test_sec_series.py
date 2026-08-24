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


def network(func):
  """Apply the real `network` marker plus the offline skip.

  This name used to be bound to a bare pytest.mark.skipif. A skipif is not
  a registered marker, so `-m network` and `-m "not network"` collected
  nothing here -- the tests were selectable only by file path.
  """
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


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


def test_duplicate_facts_sharing_a_context_are_collapsed():
    """BIIB emits the same share-count fact twice -- identical value, period,
    and context_ref. In XBRL a context plus a concept defines exactly one fact,
    so two rows sharing a context are duplicates rather than share classes.

    Summing them doubled Biogen's share count to 295.5M against a real 147.75M,
    which the market-cap reconciliation caught at 100% off.
    """
    point = FilingPoint("2026-07-29", "10-Q", "acc", facts=[
        ConceptFact(147753998.0, "2026-07-27", {}, "c-2"),
        ConceptFact(147753998.0, "2026-07-27", {}, "c-2"),
    ])
    assert point.total() == 147753998.0
    assert len(point.deduplicated()) == 1


def test_distinct_share_classes_are_not_collapsed():
    """The dedup must not undo the multi-class handling: GOOGL's three classes
    have distinct contexts and must all survive."""
    point = FilingPoint("2026-07-23", "10-Q", "acc", facts=[
        ConceptFact(5868000000.0, "2026-07-15", {"a": "A"}, "c-28"),
        ConceptFact(835000000.0, "2026-07-15", {"a": "B"}, "c-29"),
        ConceptFact(5527000000.0, "2026-07-15", {"a": "C"}, "c-30"),
    ])
    assert point.total() == 12230000000.0
    assert len(point.deduplicated()) == 3


def test_same_context_but_different_values_are_both_kept():
    """Defensive: only an exact duplicate is dropped, never a differing value."""
    point = FilingPoint("2026-07-29", "10-Q", "acc", facts=[
        ConceptFact(100.0, "2026-07-27", {}, ""),
        ConceptFact(200.0, "2026-07-27", {}, ""),
    ])
    assert point.total() == 300.0


# --------------------------------------------------------------------------
# by_concept() is a PREFIX match, not an exact one.
#
# Probed live against MSFT's FY2026 10-K: querying "us-gaap:Assets" returns
# four rows, two of them us-gaap:AssetsCurrent. latest_undimensioned() then
# returned 207.7bn -- total *current* assets -- as MSFT's total assets against
# a real 758.4bn. Nothing about the value looks wrong, which is what makes it
# dangerous: it is the denominator of the accrual ratio.
#
# The same trap sits under us-gaap:OperatingLeaseLiability (matches
# ...LiabilityCurrent, ...LiabilityNoncurrent and the whole
# LesseeOperatingLeaseLiabilityPaymentsDue* family) and us-gaap:NetIncomeLoss
# (matches ...AttributableToNoncontrollingInterest).
# --------------------------------------------------------------------------

class _FakeQuery:
    """Mimics edgartools' prefix-matching by_concept."""

    def __init__(self, frame):
        self._frame = frame

    def by_concept(self, concept):
        if "concept" not in self._frame.columns:
            return _FakeQuery(self._frame)
        mask = self._frame["concept"].map(lambda c: str(c).startswith(concept))
        return _FakeQuery(self._frame[mask])

    def to_dataframe(self):
        return self._frame


class _FakeFacts:
    def __init__(self, frame):
        self._frame = frame

    def query(self):
        return _FakeQuery(self._frame)


class _FakeFiling:
    def __init__(self, frame, contexts, filing_date="2026-07-29"):
        self._xbrl = type("X", (), {"facts": _FakeFacts(frame),
                                    "contexts": contexts})()
        self.filing_date = filing_date
        self.form = "10-K"
        self.accession_no = "0000-00-000000"

    def xbrl(self):
        return self._xbrl


def _fake_edgar(monkeypatch, frame, contexts=None):
    import tools.web_search_server.sec_series as ss

    monkeypatch.setattr(ss, "_require_identity", lambda: "test")
    filings = [_FakeFiling(frame, contexts or {})]
    monkeypatch.setattr(ss, "Company", lambda ticker: type("C", (), {
        "get_filings": lambda self, form, amendments=True: type("F", (), {
            "head": lambda self, limit: filings})()})())


def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_prefix_matched_concepts_are_dropped(monkeypatch):
    """The MSFT regression: AssetsCurrent must not answer for Assets."""
    _fake_edgar(monkeypatch, _frame([
        {"concept": "us-gaap:AssetsCurrent", "numeric_value": 207_710_000_000.0,
         "period_instant": "2026-06-30", "period_key": "instant_2026-06-30",
         "context_ref": "c-1"},
        {"concept": "us-gaap:Assets", "numeric_value": 758_376_000_000.0,
         "period_instant": "2026-06-30", "period_key": "instant_2026-06-30",
         "context_ref": "c-1"},
    ]))
    points = fetch_concept_series("MSFT", "us-gaap:Assets", form="10-K", limit=1)
    assert [f.value for f in points[0].facts] == [758_376_000_000.0]
    assert points[0].latest_undimensioned().value == 758_376_000_000.0


def test_only_prefix_matches_present_raises_not_covered(monkeypatch):
    """A filer tagging OperatingLeaseLiabilityCurrent but not the total does
    not have a total. Returning the current portion as the whole obligation
    understates it by whatever the noncurrent piece is."""
    _fake_edgar(monkeypatch, _frame([
        {"concept": "us-gaap:OperatingLeaseLiabilityCurrent",
         "numeric_value": 1_631_000_000.0, "period_instant": "2026-01-31",
         "period_key": "instant_2026-01-31", "context_ref": "c-1"},
    ]))
    with pytest.raises(NotCovered):
        fetch_concept_series("WMT", "us-gaap:OperatingLeaseLiability",
                             form="10-K", limit=1)


def test_facts_record_the_concept_they_came_from(monkeypatch):
    _fake_edgar(monkeypatch, _frame([
        {"concept": "us-gaap:Assets", "numeric_value": 1.0,
         "period_instant": "2026-06-30", "period_key": "instant_2026-06-30",
         "context_ref": "c-1"},
    ]))
    points = fetch_concept_series("MSFT", "us-gaap:Assets", form="10-K", limit=1)
    assert points[0].facts[0].concept == "us-gaap:Assets"


def test_a_frame_without_a_concept_column_is_not_silently_emptied(monkeypatch):
    """Defensive: an edgartools version that stops emitting the column must
    degrade to today's behaviour rather than reporting every filer uncovered."""
    _fake_edgar(monkeypatch, _frame([
        {"numeric_value": 5.0, "period_instant": "2026-06-30",
         "period_key": "instant_2026-06-30", "context_ref": "c-1"},
    ]))
    points = fetch_concept_series("MSFT", "us-gaap:Assets", form="10-K", limit=1)
    assert points[0].facts[0].value == 5.0


@network
def test_msft_total_assets_is_not_total_current_assets():
    """Live pin on the regression. MSFT FY2026: 758.4bn total assets, 207.7bn
    current. The prefix match returned the latter."""
    points = fetch_concept_series("MSFT", "us-gaap:Assets", form="10-K", limit=1)
    assets = points[0].latest_undimensioned()
    assert assets is not None
    assert assets.value > 500e9, (
        f"MSFT total assets {assets.value:,.0f} -- current assets leaked in")


# --------------------------------------------------------------------------
# Amendments displace the filing that carries the data.
#
# TSLA's most recent 10-K "filing" is a 10-K/A from 2026-04-30 carrying 37
# fact rows -- the Part III proxy information, no financial statements. The
# real FY2025 10-K sits behind it. Any single-filing lookup therefore reported
# Tesla as tagging no operating leases, no revenue and no assets at all.
# --------------------------------------------------------------------------

def test_amendments_are_excluded_from_the_filing_walk(monkeypatch):
    import tools.web_search_server.sec_series as ss

    seen = {}
    frame = _frame([{"concept": "us-gaap:Assets", "numeric_value": 1.0,
                     "period_instant": "2026-06-30",
                     "period_key": "instant_2026-06-30", "context_ref": "c-1"}])

    def get_filings(self, form, amendments=True):
        seen["form"] = form
        seen["amendments"] = amendments
        # An amendment carries almost no XBRL; returning it here would stand
        # in for the real 10-K and answer "not covered" for every concept.
        filings = [_FakeFiling(frame, {})]
        return type("F", (), {"head": lambda self, limit: filings})()

    monkeypatch.setattr(ss, "_require_identity", lambda: "test")
    monkeypatch.setattr(ss, "Company", lambda ticker: type(
        "C", (), {"get_filings": get_filings})())

    fetch_concept_series("TSLA", "us-gaap:Assets", form="10-K", limit=1)
    assert seen["amendments"] is False, (
        "a 10-K/A with no financial statements must not consume a slot in the "
        "filing walk")


# --------------------------------------------------------------------------
# Units, and the convenience translation.
#
# Every fact in a US filer's 10-K is denominated in usd, so nothing here ever
# needed a unit. A foreign private issuer's 20-F is not: TSM reports in TWD,
# SAP and ASML in EUR, NVO in DKK, BABA in CNY. Reporting 3,809,054,300,000
# with no unit attached reads as $3.8 trillion of revenue.
#
# Worse, SEC rules let a foreign filer add a US-dollar convenience translation
# of the most recent year. TSM and BABA both do, and they tag it with the same
# concept, the same period and the *same context* as the reporting-currency
# fact. It is undimensioned, so it survives every dimension filter:
# latest_undimensioned() had two maximal candidates and returned whichever
# pandas happened to yield first, and total() summed TWD and USD together.
# --------------------------------------------------------------------------

from tools.web_search_server.sec_series import currency_of  # noqa: E402


@pytest.mark.parametrize("unit_ref,expected", [
    ("twd", "TWD"),                                    # TSM
    ("usd", "USD"),                                    # AAPL
    ("dkk", "DKK"),                                    # NVO
    ("eur", "EUR"),                                    # ASML
    ("cad", "CAD"),                                    # BCE, 40-F
    ("U_CNY", "CNY"),                                  # BABA
    ("U_USD", "USD"),
    ("Unit_Standard_EUR_DzninQPyI02xgP6HIeCj9A", "EUR"),   # SAP
    ("iso4217:JPY", "JPY"),
])
def test_currency_is_read_from_the_unit_reference(unit_ref, expected):
    """Filers spell the same unit four different ways; all five basket members
    were checked live."""
    assert currency_of(unit_ref) == expected


@pytest.mark.parametrize("unit_ref", [
    "shares", "number", "pure", "employee", "vote", "site", "rate",
    "Unit_Standard_pure_h56Qul0IO0iTmrP0VNUG2Q",
    "Unit_Standard_shares_ZhX2cvyn2kmnAWTzNhwVQA",
    "U_pure",
    "twdPerShare", "usdPerShare", "eurPerShare", "cadPerShare",
    "dkkPerUSD",                                       # an FX rate, not money
    "Unit_Divide_EUR_shares_rFrgpyX0UUqjguloF55e-A",   # earnings per share
    "Unit_Divide_CHF_EUR_oPh-6DUpEUyhcKcDVmZoKg",      # an FX rate
    "U_UnitedStatesOfAmericaDollarsShare",
    "Unit12",                                          # ENB tags units opaquely
    "", None,
])
def test_non_monetary_units_have_no_currency(unit_ref):
    """None means "not a plain amount of money". Guessing a currency for a
    per-share or ratio unit would put a price where a total belongs."""
    assert currency_of(unit_ref) is None


def test_facts_carry_the_unit_they_were_tagged_with(monkeypatch):
    _fake_edgar(monkeypatch, _frame([
        {"concept": "ifrs-full:Revenue", "numeric_value": 309_064_000_000.0,
         "period_key": "duration_2025-01-01_2025-12-31",
         "context_ref": "c-1", "unit_ref": "dkk"},
    ]))
    points = fetch_concept_series("NVO", "ifrs-full:Revenue", form="20-F", limit=1)
    fact = points[0].facts[0]
    assert fact.unit == "dkk"
    assert fact.currency == "DKK"


def test_a_frame_without_a_unit_column_leaves_the_unit_blank(monkeypatch):
    """Defensive: an edgartools version that drops unit_ref must keep working."""
    _fake_edgar(monkeypatch, _frame([
        {"concept": "us-gaap:Assets", "numeric_value": 5.0,
         "period_instant": "2026-06-30", "period_key": "instant_2026-06-30",
         "context_ref": "c-1"},
    ]))
    fact = fetch_concept_series("MSFT", "us-gaap:Assets", form="10-K", limit=1)[0].facts[0]
    assert fact.unit == ""
    assert fact.currency is None


def _tsm_point():
    """TSM's FY2025 revenue as actually tagged: three years in TWD plus a
    single USD convenience translation sharing context c-1 with FY2025."""
    return FilingPoint("2026-04-16", "20-F", "acc", facts=[
        ConceptFact(3_809_054_300_000.0, "duration_2025-01-01_2025-12-31",
                    {}, "c-1", "ifrs-full:RevenueFromContractsWithCustomers", "twd"),
        ConceptFact(121_423_500_000.0, "duration_2025-01-01_2025-12-31",
                    {}, "c-1", "ifrs-full:RevenueFromContractsWithCustomers", "usd"),
        ConceptFact(2_894_307_700_000.0, "duration_2024-01-01_2024-12-31",
                    {}, "c-6", "ifrs-full:RevenueFromContractsWithCustomers", "twd"),
        ConceptFact(2_161_735_800_000.0, "duration_2023-01-01_2023-12-31",
                    {}, "c-5", "ifrs-full:RevenueFromContractsWithCustomers", "twd"),
    ])


def test_the_reporting_currency_wins_over_the_convenience_translation():
    """Both facts are undimensioned, share a context and share a period, so
    period ranking alone cannot separate them. The reporting currency is the
    one that covers every comparative year; a convenience translation is
    permitted only for the latest."""
    fact = _tsm_point().latest_undimensioned()
    assert fact.value == 3_809_054_300_000.0
    assert fact.currency == "TWD"


def test_a_caller_can_ask_for_the_convenience_translation_explicitly():
    fact = _tsm_point().latest_undimensioned(currency="USD")
    assert fact.value == 121_423_500_000.0


def test_asking_for_an_absent_currency_returns_none_rather_than_the_wrong_one():
    assert _tsm_point().latest_undimensioned(currency="JPY") is None


def test_currencies_present_reports_both():
    assert _tsm_point().currencies() == {"TWD": 3, "USD": 1}


def test_reporting_currency_is_the_one_covering_the_most_periods():
    assert _tsm_point().reporting_currency() == "TWD"


def test_total_does_not_add_a_convenience_translation_to_the_reporting_figure():
    """TWD 3.81tn + USD 121bn = 3.93tn of nothing.

    One period only, because total() deliberately adds across every distinct
    fact -- it exists for share classes, where the parts really do sum.
    """
    point = FilingPoint("2026-04-16", "20-F", "acc", facts=[
        ConceptFact(3_809_054_300_000.0, "duration_2025-01-01_2025-12-31",
                    {}, "c-1", "ifrs-full:RevenueFromContractsWithCustomers", "twd"),
        ConceptFact(121_423_500_000.0, "duration_2025-01-01_2025-12-31",
                    {}, "c-1", "ifrs-full:RevenueFromContractsWithCustomers", "usd"),
    ])
    assert point.total() == 3_809_054_300_000.0


def test_a_single_year_tie_still_prefers_the_non_usd_reporting_currency():
    """With limit=1 on a concept tagged for one period only, both currencies
    appear once. A foreign filer's convenience translation is always the USD
    one -- if it reported in USD there would be no second currency."""
    point = FilingPoint("2026-05-20", "20-F", "acc", facts=[
        ConceptFact(1_023_670_000_000.0, "duration_2025-04-01_2026-03-31",
                    {}, "C_bdc", "us-gaap:Revenues", "U_CNY"),
        ConceptFact(148_401_000_000.0, "duration_2025-04-01_2026-03-31",
                    {}, "C_bdc", "us-gaap:Revenues", "U_USD"),
    ])
    assert point.latest_undimensioned().value == 1_023_670_000_000.0
    assert point.reporting_currency() == "CNY"


def test_a_us_filer_is_unaffected():
    """Every existing caller runs through this path. One currency means the
    selection rule is a no-op."""
    point = FilingPoint("2026-07-29", "10-K", "acc", facts=[
        ConceptFact(281_724_000_000.0, "duration_2025-07-01_2026-06-30",
                    {}, "c-1", "us-gaap:Revenues", "usd"),
        ConceptFact(245_122_000_000.0, "duration_2024-07-01_2025-06-30",
                    {}, "c-2", "us-gaap:Revenues", "usd"),
    ])
    assert point.latest_undimensioned().value == 281_724_000_000.0
    assert point.total() == 526_846_000_000.0
    assert point.reporting_currency() == "USD"


def test_share_classes_still_sum_when_the_unit_is_not_money():
    """NVO's cover page carries A and B share counts, both in `shares`. The
    currency rule must not touch them -- total() is what dilution depends on."""
    point = FilingPoint("2026-02-04", "20-F", "acc", facts=[
        ConceptFact(1_074_872_000.0, "2026-02-04", {}, "c-1",
                    "dei:EntityCommonStockSharesOutstanding", "shares"),
        ConceptFact(3_390_128_000.0, "2026-02-04", {}, "c-2",
                    "dei:EntityCommonStockSharesOutstanding", "shares"),
    ])
    assert point.total() == 4_465_000_000.0


def test_facts_with_no_unit_at_all_still_select():
    """Filings predating unit_ref in the frame, and the ENB case where units
    are opaque tokens carrying no currency."""
    point = FilingPoint("2017-02-17", "40-F", "acc", facts=[
        ConceptFact(100.0, "duration_2016-01-01_2016-12-31", {}, "c-1", "x", "Unit12"),
        ConceptFact(90.0, "duration_2015-01-01_2015-12-31", {}, "c-2", "x", "Unit12"),
    ])
    assert point.latest_undimensioned().value == 100.0
    assert point.reporting_currency() is None


@network
def test_tsm_revenue_is_reported_in_twd_not_dollars():
    """The live pin. TSM FY2025: NT$3,809,054,300,000, with a USD convenience
    translation of $121,423,500,000 sharing the same context."""
    points = fetch_concept_series(
        "TSM", "ifrs-full:RevenueFromContractsWithCustomers",
        form="20-F", limit=1)
    fact = points[0].latest_undimensioned()
    assert fact.currency == "TWD"
    assert fact.value == pytest.approx(3.8090543e12, rel=1e-6)
    assert points[0].reporting_currency() == "TWD"
    assert "USD" in points[0].currencies()


# --------------------------------------------------------------------------
# Precision tie-break.
#
# A filer may tag the same concept twice in one context at different
# precisions. Amazon's FY2024 income tax appears as both 9,265,000,000 and a
# rounded 9,300,000,000 under us-gaap:IncomeTaxExpenseBenefit in context c-4;
# FY2023 likewise as 7,120,000,000 and 7,100,000,000.
#
# Ties fell to document order, which happened to yield the precise figure in
# both observed cases. That is luck, not a rule, and the wrong side of it is a
# 0.4% error that nothing would flag. XBRL's `decimals` attribute states the
# precision outright: a higher value means accurate to a smaller unit, so the
# larger `decimals` wins.
# --------------------------------------------------------------------------

def test_concept_fact_carries_decimals():
    fact = ConceptFact(9_265_000_000.0, "duration_2024", {}, "c-4", decimals=-6)
    assert fact.decimals == -6


def test_the_more_precise_fact_wins_a_tie():
    """decimals=-6 is accurate to millions; -8 only to hundred millions."""
    point = FilingPoint("2026-02-06", "10-K", "acc", facts=[
        ConceptFact(9_300_000_000.0, "duration_2024-01-01_2024-12-31", {}, "c-4",
                    decimals=-8),
        ConceptFact(9_265_000_000.0, "duration_2024-01-01_2024-12-31", {}, "c-4",
                    decimals=-6),
    ])
    assert point.latest_undimensioned().value == 9_265_000_000.0


def test_precision_does_not_override_a_later_period():
    """A newer period beats a more precise older one -- freshness first."""
    point = FilingPoint("2026-02-06", "10-K", "acc", facts=[
        ConceptFact(7_120_000_000.0, "duration_2023-01-01_2023-12-31", {}, "c-8",
                    decimals=-6),
        ConceptFact(9_300_000_000.0, "duration_2024-01-01_2024-12-31", {}, "c-4",
                    decimals=-8),
    ])
    assert point.latest_undimensioned().value == 9_300_000_000.0


def test_absent_decimals_does_not_lose_to_a_present_one_by_default():
    """An untagged precision is unknown, not worst. Falling back to document
    order there preserves the behaviour that was already correct."""
    point = FilingPoint("2026-02-06", "10-K", "acc", facts=[
        ConceptFact(100.0, "duration_2024", {}, "c-1", decimals=None),
        ConceptFact(200.0, "duration_2024", {}, "c-1", decimals=None),
    ])
    assert point.latest_undimensioned().value == 100.0
