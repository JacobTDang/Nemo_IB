"""Which fact `sec_utils` picks out of a filing, and which element it names.

Two audits (`test_accounting_identities.py`, `test_cross_source_reconciliation.py`)
traced most of this project's wrong numbers to one function.
`sec_utils.filter_annual_data` is the read path for roughly ten tools and it
had two compounding faults:

1. It passed edgartools' `by_concept()` **prefix match** straight through, so
   `us-gaap:Revenues` was answered by `us-gaap:RevenuesNetOfInterestExpense`
   and `us-gaap:NetCashProvidedByUsedInOperatingActivities` by the
   `...ContinuingOperations` element.
2. It broke ties with `idxmax` -- the largest fact for the period -- on the
   stated assumption that "the consolidated total is always the largest
   positive value". It is not. A segment aggregate is struck before
   intersegment eliminations, an unconsolidated joint venture's revenue is not
   the filer's revenue at all, and for a bank whose consolidated operating cash
   flow is negative the largest fact is the parent-company-only Schedule I
   figure -- so the sign flips.

Fixing either one alone makes things worse for some filers: with the concept
order corrected but the selection still `idxmax`, XOM's `us-gaap:Revenues`
returns 452.209bn against a consolidated 332.238bn and GE's 48.024bn against
45.855bn. Both faults are checked here together.

The offline half builds frames in the shape edgartools actually produces --
`period_key`, `context_ref`, `unit_ref`, and contexts carrying dimensions --
because the previous mocks carried none of those, which is precisely how a
selection bug hid under passing tests. The network half asserts the figure each
filing actually reports, established by reading the raw XBRL facts before any
code was changed.
"""
import os
from typing import Dict, List, Optional

import pandas as pd
import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


# ------------------------------------------------------------------ fake XBRL


class _Context:
    def __init__(self, dimensions: Dict[str, str]):
        self.dimensions = dict(dimensions)


class _Query:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def by_concept(self, concept: str) -> "_Query":
        """edgartools matches by PREFIX. Reproduced deliberately: a fake that
        matched exactly would make the production filter untestable."""
        prefix = str(concept).replace("_", ":")
        keep = self._frame["concept"].map(
            lambda c: str(c).replace("_", ":").startswith(prefix))
        return _Query(self._frame[keep])

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


class _Facts:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def query(self) -> _Query:
        return _Query(self._frame)


class _XBRL:
    """The two attributes `concept_point` reads: `facts` and `contexts`."""

    def __init__(self, rows: List[dict]):
        frame = pd.DataFrame([{
            "concept": row["concept"],
            "numeric_value": row["value"],
            "period_key": row.get("period_key", ""),
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
            "context_ref": row["context"],
            "unit_ref": row.get("unit", "usd"),
        } for row in rows])
        self.facts = _Facts(frame)
        self.contexts = {row["context"]: _Context(row.get("dimensions", {}))
                         for row in rows}


def _duration(start: str, end: str) -> dict:
    return {"period_key": f"duration_{start}_{end}",
            "period_start": start, "period_end": end}


def _instant(when: str) -> dict:
    return {"period_key": f"instant_{when}", "period_start": None,
            "period_end": when}


def _fact(concept: str, value: float, context: str, period: dict,
          dimensions: Optional[Dict[str, str]] = None) -> dict:
    row = {"concept": concept, "value": value, "context": context,
           "dimensions": dimensions or {}}
    row.update(period)
    return row


FY2025 = _duration("2025-01-01", "2025-12-31")
FY2024 = _duration("2024-01-01", "2024-12-31")
Q4_2025 = _duration("2025-10-01", "2025-12-31")
YTD_2025 = _duration("2025-01-01", "2025-09-30")
Q3_2025 = _duration("2025-07-01", "2025-09-30")


# ------------------------------------------------------ fault 1: prefix match


def test_a_longer_element_name_never_answers_for_a_shorter_one():
    """The MSFT case the audits open with, in miniature.

    `by_concept('us-gaap:Assets')` returns `us-gaap:AssetsCurrent` too, and
    current assets share the balance-sheet context, so they survive every
    dimension filter. MSFT's total assets read 207.7bn against a real 758.4bn.
    """
    from tools.web_search_server.sec_utils import filter_instant_data

    xbrl = _XBRL([
        _fact("us-gaap:AssetsCurrent", 207_700_000_000, "c-1",
              _instant("2025-12-31")),
        _fact("us-gaap:Assets", 758_400_000_000, "c-1", _instant("2025-12-31")),
    ])
    result = filter_instant_data(xbrl, "us-gaap:Assets")
    assert result is not None
    assert result["value"] == 758_400_000_000
    assert result["concept_used"] == "us-gaap:Assets"


def test_operating_cash_flow_is_not_answered_by_the_continuing_operations_element():
    """GE: 8,543,000,000 is `...ContinuingOperations` reached by prefix; the
    concept asked for reads 8,537,000,000 in the same filing."""
    from tools.web_search_server.sec_utils import filter_annual_data

    concept = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
    xbrl = _XBRL([
        _fact(concept + "ContinuingOperations", 8_543_000_000, "c-1", FY2025),
        _fact(concept, 8_537_000_000, "c-1", FY2025),
    ])
    result = filter_annual_data(xbrl, concept)
    assert result is not None
    assert result["value"] == 8_537_000_000
    assert result["concept_used"] == concept


def test_provenance_names_the_element_the_value_actually_came_from():
    """GS returns the right number under a concept GS does not tag.

    58,283,000,000 is Goldman's total net revenues and is correct; it is
    `us-gaap:RevenuesNetOfInterestExpense`, reached from a query for
    `us-gaap:Revenues`, which GS does not tag at all. A caller reconciling
    against the filing would find nothing under the named element.
    """
    from tools.web_search_server.sec_utils import filter_annual_data

    xbrl = _XBRL([
        _fact("us-gaap:RevenuesNetOfInterestExpense", 58_283_000_000, "c-1",
              FY2025),
    ])
    assert filter_annual_data(xbrl, "us-gaap:Revenues") is None

    result = filter_annual_data(xbrl, "us-gaap:RevenuesNetOfInterestExpense")
    assert result is not None
    assert result["concept_used"] == "us-gaap:RevenuesNetOfInterestExpense"


# -------------------------------------------------------- fault 2: "largest"


def test_a_dimensioned_fact_never_wins_over_the_consolidated_one():
    """CVX: 231,370,000,000 sits on
    `srt:ConsolidationItemsAxis=us-gaap:OperatingSegmentsMember` -- segment
    revenue struck before intersegment eliminations -- against a consolidated
    189,031,000,000."""
    from tools.web_search_server.sec_utils import filter_annual_data

    xbrl = _XBRL([
        _fact("us-gaap:Revenues", 231_370_000_000, "c-9", FY2025,
              {"srt:ConsolidationItemsAxis": "us-gaap:OperatingSegmentsMember"}),
        _fact("us-gaap:Revenues", 189_031_000_000, "c-1", FY2025),
    ])
    result = filter_annual_data(xbrl, "us-gaap:Revenues")
    assert result is not None
    assert result["value"] == 189_031_000_000


def test_a_negative_consolidated_figure_survives_a_larger_parent_only_one():
    """JPM: `idxmax` picked the parent-company-only Schedule I cash flow off
    `srt:ConsolidatedEntitiesAxis / srt:ParentCompanyMember`, +44,468,000,000,
    against a consolidated -147,782,000,000. The sign flips."""
    from tools.web_search_server.sec_utils import filter_annual_data

    concept = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
    xbrl = _XBRL([
        _fact(concept, 44_468_000_000, "c-88", FY2025,
              {"srt:ConsolidatedEntitiesAxis": "srt:ParentCompanyMember"}),
        _fact(concept, -147_782_000_000, "c-1", FY2025),
    ])
    result = filter_annual_data(xbrl, concept)
    assert result is not None
    assert result["value"] == -147_782_000_000


def test_a_concept_tagged_only_on_dimensions_is_not_reported_at_all():
    """GOOGL tags every ASC 606 fact on the segment note and none on the income
    statement. There is no consolidated figure to return, so the answer is "not
    here" -- which lets the caller's chain move to the element that is."""
    from tools.web_search_server.sec_utils import filter_annual_data

    concept = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    xbrl = _XBRL([
        _fact(concept, 342_721_000_000, "c-40", FY2025,
              {"us-gaap:StatementBusinessSegmentsAxis":
               "goog:GoogleServicesMember"}),
        _fact(concept, 60_115_000_000, "c-41", FY2025,
              {"us-gaap:StatementBusinessSegmentsAxis":
               "goog:GoogleCloudMember"}),
    ])
    assert filter_annual_data(xbrl, concept) is None


def test_the_consolidated_fact_wins_even_when_it_is_the_smallest():
    """SPG returns its unconsolidated joint ventures' revenue, 12,461,291,000,
    against its own 6,364,505,000. Size carries no information about which
    fact is the filer's own."""
    from tools.web_search_server.sec_utils import filter_annual_data

    xbrl = _XBRL([
        _fact("us-gaap:Revenues", 12_461_291_000, "c-77", FY2025,
              {"us-gaap:EquityMethodInvestmentNonconsolidatedInvesteeAxis":
               "spg:PlatformInvestmentsExcludingTrgAndKlepierre"}),
        _fact("us-gaap:Revenues", 6_364_505_000, "c-1", FY2025),
    ])
    result = filter_annual_data(xbrl, "us-gaap:Revenues")
    assert result is not None
    assert result["value"] == 6_364_505_000


# ------------------------------------------------------------ period windows


def test_the_annual_window_prefers_the_year_over_a_quarter_ending_the_same_day():
    from tools.web_search_server.sec_utils import filter_annual_data

    xbrl = _XBRL([
        _fact("us-gaap:Revenues", 25_000_000_000, "c-4", Q4_2025),
        _fact("us-gaap:Revenues", 100_000_000_000, "c-1", FY2025),
        _fact("us-gaap:Revenues", 90_000_000_000, "c-2", FY2024),
    ])
    result = filter_annual_data(xbrl, "us-gaap:Revenues")
    assert result is not None
    assert result["value"] == 100_000_000_000
    assert result["period_end"] == "2025-12-31"
    assert result["duration_days"] >= 350


def test_the_quarterly_window_excludes_the_year_to_date_duration():
    """A 10-Q carries the year-to-date period alongside the quarter and both
    end on the same day, so period ranking alone returns nine months where the
    caller asked for three."""
    from tools.web_search_server.sec_utils import filter_annual_data

    xbrl = _XBRL([
        _fact("us-gaap:Revenues", 75_000_000_000, "c-2", YTD_2025),
        _fact("us-gaap:Revenues", 26_000_000_000, "c-1", Q3_2025),
    ])
    result = filter_annual_data(xbrl, "us-gaap:Revenues", "10-Q")
    assert result is not None
    assert result["value"] == 26_000_000_000


def test_a_missing_concept_is_not_an_exception():
    from tools.web_search_server.sec_utils import filter_annual_data

    xbrl = _XBRL([_fact("us-gaap:Revenues", 1.0, "c-1", FY2025)])
    assert filter_annual_data(xbrl, "us-gaap:NoSuchConcept") is None
    assert filter_annual_data(None, "us-gaap:Revenues") is None


# ================================================================ live filings
#
# Every figure below was read out of the filing's raw XBRL -- concept, context
# and dimensions -- before any code was changed. They are the filings' own
# numbers, not a vendor's.

# ticker -> (consolidated revenue, the element the filing tags it under)
REVENUE = {
    "GOOGL": (402_836_000_000, "us-gaap:Revenues"),
    "AMT": (10_644_600_000, "us-gaap:Revenues"),
    "WFC": (83_699_000_000, "us-gaap:RevenuesNetOfInterestExpense"),
    "GS": (58_283_000_000, "us-gaap:RevenuesNetOfInterestExpense"),
    "SPG": (6_364_505_000, "us-gaap:Revenues"),
    "CVX": (189_031_000_000, "us-gaap:Revenues"),
    "BA": (89_463_000_000, "us-gaap:Revenues"),
    "XOM": (332_238_000_000, "us-gaap:Revenues"),
    "GE": (45_855_000_000, "us-gaap:Revenues"),
    "CAT": (67_589_000_000, "us-gaap:Revenues"),
    "RIOT": (647_435_000, "us-gaap:Revenues"),
    "WMT": (713_163_000_000, "us-gaap:Revenues"),
}

# ticker -> consolidated operating cash flow off the cash flow statement
OPERATING_CASH_FLOW = {
    "JPM": -147_782_000_000,
    "GS": -45_154_000_000,
    "WFC": -19_001_000_000,
    "BAC": 12_613_000_000,
    "GE": 8_537_000_000,
}

# ticker -> (operating cash flow, capex the filing tags, free cash flow)
FREE_CASH_FLOW = {
    "AMZN": (139_514_000_000, 131_819_000_000, 7_695_000_000),
    "T": (40_284_000_000, 20_842_000_000, 19_442_000_000),
    "NVDA": (102_718_000_000, 6_042_000_000, 96_676_000_000),
    "CVX": (33_939_000_000, 17_347_000_000, 16_592_000_000),
    "HD": (16_325_000_000, 3_679_000_000, 12_646_000_000),
    "SPG": (4_136_551_000, 934_346_000, 3_202_205_000),
    "REGN": (4_978_900_000, 898_400_000, 4_080_500_000),
    "PLD": (5_008_434_000, 2_781_260_000, 2_227_174_000),
}


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@pytest.fixture(scope="module", autouse=True)
def _gentle_request_rate():
    """Hold the SEC request rate below edgartools' default, as the two audit
    sweeps do. Its default of 8/s earns a 429 that blocks this host for about
    nine minutes, and every tool here reports that as "No filing found"."""
    import edgar.httprequests as httprequests

    previous = httprequests.max_requests_per_second
    httprequests.max_requests_per_second = 3
    yield
    httprequests.max_requests_per_second = previous


@network
@pytest.mark.parametrize("ticker", sorted(REVENUE))
def test_reported_revenue_is_the_consolidated_total(ticker):
    from tools.web_search_server.sec_utils import get_revenue_base

    expected, concept = REVENUE[ticker]
    result = get_revenue_base(ticker)
    assert result["success"], result.get("error")
    assert result["revenue_base"] == pytest.approx(expected, rel=1e-9), (
        f"{ticker}: reported {result['revenue_base']:,.0f} against "
        f"{expected:,.0f} tagged undimensioned in the filing")
    assert result["concept_used"] == concept, (
        f"{ticker}: provenance names {result['concept_used']}, but the value "
        f"is tagged {concept}")


@network
@pytest.mark.parametrize("ticker", sorted(OPERATING_CASH_FLOW))
def test_operating_cash_flow_is_the_consolidated_statement_figure(ticker):
    """Three of these flip sign: the parent-company-only Schedule I cash flow
    is positive where the consolidated figure is negative."""
    from tools.web_search_server.sec_utils import get_historical_fcf

    expected = OPERATING_CASH_FLOW[ticker]
    result = get_historical_fcf(ticker)
    assert result["operating_cash_flow"] == pytest.approx(expected, rel=1e-9), (
        f"{ticker}: read {result['operating_cash_flow']:,.0f} against "
        f"{expected:,.0f} on the consolidated statement of cash flows")


@network
@pytest.mark.parametrize("ticker", sorted(FREE_CASH_FLOW))
def test_free_cash_flow_subtracts_the_capex_the_filing_tags(ticker):
    """`fcf = ocf - (capex or 0)` turned a capex element the two-concept chain
    did not try into zero. Amazon's free cash flow read 139,514m against a real
    7,695m -- an 18x overstatement."""
    from tools.web_search_server.sec_utils import get_historical_fcf

    ocf, capex, fcf = FREE_CASH_FLOW[ticker]
    result = get_historical_fcf(ticker)
    assert result["success"], result.get("error")
    assert result["operating_cash_flow"] == pytest.approx(ocf, rel=1e-9)
    assert result["capex"] == pytest.approx(capex, rel=1e-9)
    assert result["free_cash_flow"] == pytest.approx(fcf, rel=1e-9)


@network
@pytest.mark.parametrize("ticker", ["JPM", "BAC", "WFC"])
def test_a_filer_that_tags_no_capex_is_not_given_a_free_cash_flow(ticker):
    """A bank tags no capital expenditure element at all. Substituting zero
    reports operating cash flow as free cash flow; the honest answer is that
    the input is missing."""
    from tools.web_search_server.sec_utils import get_historical_fcf

    result = get_historical_fcf(ticker)
    assert result["success"] is False
    assert result["capex"] is None
    assert result["free_cash_flow"] is None
    assert "capex" in result["error"].lower()
    # The half that was read is still handed back rather than thrown away.
    assert result["operating_cash_flow"] == pytest.approx(
        OPERATING_CASH_FLOW[ticker], rel=1e-9)
