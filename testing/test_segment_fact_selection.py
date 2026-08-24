"""Which fact answers for a segment.

`get_segment_financials` queried `by_dimension(axis, member)` and took the
first row of the result. Facts carrying the segment axis *plus another* axis --
`srt:ConsolidationItemsAxis`, `srt:ProductOrServiceAxis`,
`srt:StatementGeographicalAxis` -- are in that result and could win it. GE's
largest segment reported **-62,000,000** of revenue, the intersegment-
elimination context, against 33,252,000,000 tagged on the segment-only context
in the same filing; both GE segments came back negative and the tool reported
total segment revenue of -1,748,000,000. A negative number for a company's
largest segment is the visible symptom; the general fault is that a fact
qualified by a second dimension is not the segment's figure.

The selection rule the fix implements, and what each half of it is for:

* a fact whose only dimension is the segment axis is the segment's figure;
* where a filer tags none, the fact additionally qualified by
  `srt:ConsolidationItemsAxis = us-gaap:OperatingSegmentsMember` is -- that is
  the segment column of the reconciliation table, not a breakdown of it. AAPL,
  BA, HON, JPM, NVDA and WFC tag their segments only that way, so a rule that
  demanded the segment axis alone would report those filers as untagged;
* anything else -- a product, a geography, an intersegment elimination, a
  corporate reconciling item -- is a piece of a segment or an adjustment to it,
  never the segment;
* the choice between the two is made **once per filing**, by which resolves
  more members, so every segment of one filer is on one basis. Per-member
  preference mixes them: AMT tags five members' non-lease revenue on the
  segment axis alone and all seven members' total revenue on the operating-
  segments column, and picking each member's most specific fact reports
  935,900,000 of revenue beside 10,305,000,000.

The offline half builds frames in the shape edgartools produces. The network
half asserts figures read out of each filing's raw XBRL -- concept, context and
dimensions -- before any code was changed.
"""
import os
from typing import Dict, List, Optional

import pandas as pd
import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


SEGMENT_AXIS = "us-gaap:StatementBusinessSegmentsAxis"
CONSOLIDATION_AXIS = "srt:ConsolidationItemsAxis"
OPERATING_SEGMENTS = "us-gaap:OperatingSegmentsMember"
INTERSEGMENT = "us-gaap:IntersegmentEliminationMember"
PRODUCT_AXIS = "srt:ProductOrServiceAxis"
GEOGRAPHY_AXIS = "srt:StatementGeographicalAxis"

REVENUES = "us-gaap:Revenues"
ASC606 = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
BANK_REVENUE = "us-gaap:RevenuesNetOfInterestExpense"


# ------------------------------------------------------------------ fake XBRL


class _Context:
    def __init__(self, dimensions: Dict[str, str]):
        self.dimensions = dict(dimensions)


class _Query:
    def __init__(self, frame: pd.DataFrame, contexts: Dict[str, _Context]):
        self._frame = frame
        self._contexts = contexts

    def by_concept(self, concept: str) -> "_Query":
        """edgartools matches by PREFIX. Reproduced deliberately -- a fake that
        matched exactly would make the production filter untestable, and the
        prefix is load-bearing here: `us-gaap:Revenues` also returns
        `us-gaap:RevenuesNetOfInterestExpense`, which is the element a bank
        tags its segments under."""
        prefix = str(concept).replace("_", ":")
        keep = self._frame["concept"].map(
            lambda c: str(c).replace("_", ":").startswith(prefix))
        return _Query(self._frame[keep], self._contexts)

    def by_dimension(self, axis: str, member: str) -> "_Query":
        """Every fact whose context carries that axis at that member --
        including the ones carrying a second axis as well. That is the whole of
        defect 1: the query is not wrong, taking the first row of it is."""
        def _matches(context_ref: str) -> bool:
            context = self._contexts.get(str(context_ref))
            if context is None:
                return False
            return context.dimensions.get(str(axis)) == member

        return _Query(self._frame[self._frame["context_ref"].map(_matches)],
                      self._contexts)

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


class _Facts:
    def __init__(self, frame: pd.DataFrame, contexts: Dict[str, _Context]):
        self._frame = frame
        self._contexts = contexts

    def query(self) -> _Query:
        return _Query(self._frame, self._contexts)

    def get_unique_dimensions(self) -> Dict[str, set]:
        """Axis -> members, keyed the way edgartools keys it: ':' becomes '_'."""
        out: Dict[str, set] = {}
        for context in self._contexts.values():
            for axis, member in context.dimensions.items():
                out.setdefault(axis.replace(":", "_", 1), set()).add(member)
        return out


class _XBRL:
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
        self.contexts = {row["context"]: _Context(row.get("dimensions", {}))
                         for row in rows}
        self.facts = _Facts(frame, self.contexts)


def _duration(start: str, end: str) -> dict:
    return {"period_key": f"duration_{start}_{end}",
            "period_start": start, "period_end": end}


FY2025 = _duration("2025-01-01", "2025-12-31")
FY2024 = _duration("2024-01-01", "2024-12-31")


def _fact(concept: str, value: float, context: str,
          dimensions: Optional[Dict[str, str]] = None,
          period: Optional[dict] = None) -> dict:
    row = {"concept": concept, "value": value, "context": context,
           "dimensions": dimensions or {}}
    row.update(period or FY2025)
    return row


def _segment(member: str, extra: Optional[Dict[str, str]] = None) -> dict:
    dimensions = {SEGMENT_AXIS: member}
    dimensions.update(extra or {})
    return dimensions


@pytest.fixture
def run_segments(monkeypatch):
    """Call `get_segment_financials` against a fake filing."""
    from tools.web_search_server import sec_utils

    def _run(rows: List[dict], ticker: str = "TEST"):
        xbrl = _XBRL(rows)
        monkeypatch.setattr(
            sec_utils, "get_latest_filing",
            lambda t, form_type="10-K": {"xbrl_data": xbrl,
                                         "filing_date": "2026-01-29",
                                         "url": None})
        return sec_utils.get_segment_financials(ticker)

    return _run


def _by_member(result: dict) -> Dict[str, Optional[float]]:
    out = {}
    for segment in result.get("segments") or []:
        revenue = segment.get("revenue") or []
        out[segment["segment_member"]] = revenue[0]["value"] if revenue else None
    return out


# ============================================ defect 1: a second axis answered

CES = "ge:CommercialEnginesAndServicesReportableSegmentMember"
DPT = "ge:DefenseAndPropulsionTechnologiesReportableSegmentMember"

# GE's FY2025 10-K, the facts that carry the segment axis for us-gaap:Revenues.
# Read out of the filing before anything was changed.
GE_ROWS = [
    _fact(REVENUES, -62_000_000, "c-602",
          _segment(CES, {CONSOLIDATION_AXIS: INTERSEGMENT})),
    _fact(REVENUES, 33_252_000_000, "c-605", _segment(CES)),
    _fact(REVENUES, 33_314_000_000, "c-132",
          _segment(CES, {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
    _fact(REVENUES, 8_304_000_000, "c-622",
          _segment(CES, {CONSOLIDATION_AXIS: OPERATING_SEGMENTS,
                         PRODUCT_AXIS: "us-gaap:ProductMember"})),
    _fact(REVENUES, 25_010_000_000, "c-623",
          _segment(CES, {CONSOLIDATION_AXIS: OPERATING_SEGMENTS,
                         PRODUCT_AXIS: "us-gaap:ServiceMember"})),
    _fact(REVENUES, -1_686_000_000, "c-608",
          _segment(DPT, {CONSOLIDATION_AXIS: INTERSEGMENT})),
    _fact(REVENUES, 8_868_000_000, "c-601", _segment(DPT)),
    _fact(REVENUES, 10_554_000_000, "c-135",
          _segment(DPT, {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
    _fact(REVENUES, 45_855_000_000, "c-1"),
]


def test_an_intersegment_elimination_never_answers_for_the_segment(run_segments):
    """GE's largest segment read -62,000,000 against 33,252,000,000 tagged on
    the segment-only context in the same filing."""
    result = run_segments(GE_ROWS, "GE")
    assert result["success"], result.get("error")
    assert _by_member(result) == {CES: 33_252_000_000, DPT: 8_868_000_000}
    assert result["total_latest_segment_revenue"] == 42_120_000_000


def test_no_segment_comes_back_negative_when_the_filing_tags_none(run_segments):
    """The visible symptom. A negative largest segment is arithmetically
    possible for operating income and not for revenue."""
    result = run_segments(GE_ROWS, "GE")
    values = [s["revenue"][0]["value"] for s in result["segments"]
              if s["revenue"]]
    assert values and all(v > 0 for v in values), values


def test_a_product_breakdown_is_not_the_segments_revenue(run_segments):
    """GE tags Commercial Engines & Services product revenue at 8,304,000,000
    and service revenue at 25,010,000,000 on the same segment axis. Either
    could win a query for the segment; neither is the segment."""
    result = run_segments(GE_ROWS, "GE")
    assert _by_member(result)[CES] == 33_252_000_000


def test_the_operating_segments_column_answers_when_nothing_else_does(
        run_segments):
    """AAPL, BA, HON, JPM, NVDA and WFC tag every segment revenue fact with
    `srt:ConsolidationItemsAxis = us-gaap:OperatingSegmentsMember`. Demanding
    the segment axis alone would report all six as tagging nothing."""
    rows = [
        _fact(ASC606, 170_000_000_000, "c-150",
              _segment("aapl:AmericasSegmentMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(ASC606, 110_000_000_000, "c-151",
              _segment("aapl:EuropeSegmentMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(ASC606, 280_000_000_000, "c-1"),
    ]
    result = run_segments(rows, "AAPL")
    assert result["success"], result.get("error")
    assert _by_member(result) == {"aapl:AmericasSegmentMember": 170_000_000_000,
                                 "aapl:EuropeSegmentMember": 110_000_000_000}


def test_the_basis_is_chosen_once_for_the_filing(run_segments):
    """CAT's shape: two members carry a segment-only fact and all six carry the
    operating-segments column. Preferring the most specific fact per member
    would put Power & Energy's 27,143m external revenue beside Construction's
    25,060m of total segment revenue -- two different measures in one total."""
    rows = [
        _fact(REVENUES, 25_060_000_000, "c-878",
              _segment("cat:ConstructionIndustriesMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(REVENUES, 32_201_000_000, "c-892",
              _segment("cat:PowerEnergyMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(REVENUES, 27_143_000_000, "c-271",
              _segment("cat:PowerEnergyMember")),
        _fact(REVENUES, 57_000_000_000, "c-1"),
    ]
    result = run_segments(rows, "CAT")
    assert result["success"], result.get("error")
    assert _by_member(result) == {
        "cat:ConstructionIndustriesMember": 25_060_000_000,
        "cat:PowerEnergyMember": 32_201_000_000}


def test_a_bank_tags_its_segments_under_the_element_it_reports_revenue_on(
        run_segments):
    """`us-gaap:Revenues` prefix-matches `us-gaap:RevenuesNetOfInterestExpense`,
    so the bank element used to be reached by accident. Matched exactly it has
    to be named, or JPM, GS and WFC lose their segments entirely."""
    rows = [
        _fact(BANK_REVENUE, 100_000_000_000, "c-10",
              _segment("jpm:ConsumerBankingMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(BANK_REVENUE, 78_000_000_000, "c-11",
              _segment("jpm:CorporateAndInvestmentBankMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(BANK_REVENUE, 182_000_000_000, "c-1"),
    ]
    result = run_segments(rows, "JPM")
    assert result["success"], result.get("error")
    assert result["revenue_concept_used"] == BANK_REVENUE
    assert result["total_latest_segment_revenue"] == 178_000_000_000


def test_the_broadest_revenue_element_answers_where_a_filer_tags_two(
        run_segments):
    """AMT's shape. Its ASC 606 element is non-lease revenue -- 8.8% of the
    total, because tower rent is lease income under ASC 842 -- and it is the
    one tagged on the segment axis alone. Trying the ASC 606 element first
    reports 935,900,000 of segment revenue against 10,644,600,000 of revenue."""
    rows = [
        _fact(ASC606, 632_400_000, "c-104",
              _segment("amt:PropertyUSAndCanadaMember")),
        _fact(ASC606, 117_600_000, "c-107",
              _segment("amt:PropertyLatinAmericaMember")),
        _fact(REVENUES, 5_248_700_000, "c-579",
              _segment("amt:PropertyUSAndCanadaMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(REVENUES, 1_642_600_000, "c-582",
              _segment("amt:PropertyLatinAmericaMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS})),
        _fact(REVENUES, 6_891_300_000, "c-1"),
    ]
    result = run_segments(rows, "AMT")
    assert result["success"], result.get("error")
    assert result["revenue_concept_used"] == REVENUES
    assert _by_member(result) == {
        "amt:PropertyUSAndCanadaMember": 5_248_700_000,
        "amt:PropertyLatinAmericaMember": 1_642_600_000}


def test_a_segment_the_filing_qualifies_by_geography_is_not_resolved(
        run_segments):
    """XOM tags every segment revenue fact in combination with
    `srt:StatementGeographicalAxis`; no segment-only fact exists. Summing
    across the geography axis is not something any tool here does, so the
    honest answer is that it is not extractable -- not a plausible fraction of
    revenue."""
    rows = [
        _fact(REVENUES, 120_000_000_000, "c-40",
              _segment("xom:UpstreamMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS,
                        GEOGRAPHY_AXIS: "country:US"})),
        _fact(REVENUES, 80_000_000_000, "c-41",
              _segment("xom:UpstreamMember",
                       {CONSOLIDATION_AXIS: OPERATING_SEGMENTS,
                        GEOGRAPHY_AXIS: "us-gaap:NonUsMember"})),
        _fact(REVENUES, 332_238_000_000, "c-1"),
    ]
    result = run_segments(rows, "XOM")
    assert not result["success"]
    assert "xom:UpstreamMember" in str(result.get("error"))
    assert result.get("total_latest_segment_revenue") in (None, 0)


def test_an_earlier_year_keeps_its_own_context(run_segments):
    """The series is per period, and the comparative year carries the same
    contexts. A fix that only looked at the latest period would leave the
    history reading off eliminations."""
    rows = GE_ROWS + [
        _fact(REVENUES, -55_000_000, "c-602-2024",
              _segment(CES, {CONSOLIDATION_AXIS: INTERSEGMENT}), FY2024),
        _fact(REVENUES, 30_000_000_000, "c-605-2024", _segment(CES), FY2024),
    ]
    result = run_segments(rows, "GE")
    series = {s["segment_member"]: s["revenue"] for s in result["segments"]}
    assert [row["value"] for row in series[CES]] == [33_252_000_000,
                                                    30_000_000_000]


# ================================================================ live filings
#
# Every figure below was read out of the filing's raw XBRL -- concept, context
# and dimensions -- before any code was changed.

# ticker -> {member: latest annual revenue}
SEGMENT_REVENUE = {
    "GE": {CES: 33_252_000_000, DPT: 8_868_000_000},
    "CAT": {
        "cat:ConstructionIndustriesMember": 25_060_000_000,
        "cat:ResourceIndustriesMember": 12_474_000_000,
        "cat:PowerEnergyMember": 32_201_000_000,
        "cat:FinancialProductsSegmentMember": 4_220_000_000,
        "us-gaap:AllOtherSegmentsMember": 327_000_000,
        "us-gaap:ReportableSegmentAggregationBeforeOtherOperatingSegmentMember":
            73_955_000_000,
    },
    "HD": {"hd:PrimarySegmentMember": 151_966_000_000,
           "us-gaap:AllOtherSegmentsMember": 12_717_000_000},
}

# Filers whose segment revenue is not extractable as tagged, with the reason.
NOT_EXTRACTABLE = {
    "XOM": "every segment revenue fact also carries a geography axis",
    "BAC": "no revenue element is tagged on the segment axis at all",
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
@pytest.mark.parametrize("ticker", sorted(SEGMENT_REVENUE))
def test_each_segment_reports_the_figure_the_filing_tags(ticker):
    from tools.web_search_server.sec_utils import get_segment_financials

    result = get_segment_financials(ticker)
    assert result["success"], result.get("error")
    actual = {s["segment_member"]: s["revenue"][0]["value"]
              for s in result["segments"] if s["revenue"]}
    assert actual == pytest.approx(SEGMENT_REVENUE[ticker])


@network
@pytest.mark.parametrize("ticker", sorted(NOT_EXTRACTABLE))
def test_a_filer_whose_segments_cannot_be_resolved_says_so(ticker):
    """Not a number. XOM's segment revenue is a real absence, and a tool that
    answered it with the 41.4% the geography-qualified facts happen to sum to
    would be inventing a figure the filing does not report."""
    from tools.web_search_server.sec_utils import get_segment_financials

    result = get_segment_financials(ticker)
    assert not result["success"], (
        f"{ticker}: {NOT_EXTRACTABLE[ticker]}, but the tool returned "
        f"{result.get('total_latest_segment_revenue')}")
    assert result.get("total_latest_segment_revenue") in (None, 0)
