"""One filing walk per call, shared by every concept the call reads.

`earnings_quality` fixed this once and `test_slow_tool_bounds.py` guards it
there. Four sibling tools still walked EDGAR from scratch per concept, and
`fetch_concept_series` re-parses every filing each time it is called. The
parse is not the expensive half: edgartools enriches a filing's whole fact
table on first query and memoises it *on the XBRL object*, so a fresh object
per concept rebuilds it per concept.

Measured live against GS before the change:

    get_sbc_series               40 parses of  5 filings   66.7s
    get_debt_maturity_schedule   12 parses of  1 filing    65.9s
    get_annual_revenue           21 parses of  3 filings   46.4s
    get_contracted_revenue       18 parses of  3 filings   40.2s
    get_geographic_revenue        5 parses of  1 filing    25.8s

Parses are what these tests assert on, never wall time. A warm HTTP cache
hides the difference in seconds entirely while the fact table is still being
rebuilt once per concept, so a timing assertion would pass on a machine where
the bug is fully present.

Two things the speedup must not have bought its speed with, and both have a
test here:

* **Every concept in the chain is still evaluated.** That is the Ford rule --
  Ford's FY2025 10-K abandoned `NetIncomeLoss`, and stopping at the first hit
  reported an 8.2bn loss as +5.9bn of income. `_bucket_value` in
  `debt_maturity` is the deliberate exception: it stops at the first concept
  that answers, and it still must.
* **`fetch_concept_series` is still the seam.** Roughly thirty tests across
  this directory replace that name on the module under test. A walk that
  ignored the replacement would send every one of them to live EDGAR.
"""
import threading

import pytest

from tools.web_search_server import debt_maturity as dm
from tools.web_search_server import foreign_issuer as fi
from tools.web_search_server import forward_metrics as fm
from tools.web_search_server import sbc
from tools.web_search_server import sec_series
from tools.web_search_server import shared_filings
from tools.web_search_server.sec_series import ConceptFact, FilingPoint

# Newest first, as EDGAR returns them.
FILINGS = [("acc-2026", "2026-02-25"), ("acc-2025", "2025-02-26"),
           ("acc-2024", "2024-02-21"), ("acc-2023", "2023-02-24"),
           ("acc-2022", "2022-03-18")]
NEWEST = FILINGS[0][1]
OLDEST = FILINGS[-1][1]


class _FakeXBRL:
    def __init__(self, accession):
        self.accession = accession


class _FakeFilings(list):
    def head(self, n):
        return _FakeFilings(self[:n])


class _FakeFiling:
    def __init__(self, accession, filing_date, counters):
        self.accession_no = accession
        self.filing_date = filing_date
        self.form = "10-K"
        self._counters = counters

    def xbrl(self):
        self._counters["parses"] += 1
        return _FakeXBRL(self.accession_no)


@pytest.fixture
def edgar(monkeypatch):
    """A five-filing EDGAR that counts parses, filing lists and concept reads.

    Deliberately owns the object that gets parsed rather than stubbing
    `fetch_concept_series`: the whole question is how many times a filing is
    parsed, and a stub at the series level cannot answer it.
    """
    counters = {"parses": 0, "companies": 0, "filing_lists": 0,
                "reads": 0, "concepts": []}
    filings = _FakeFilings(_FakeFiling(acc, filed, counters)
                           for acc, filed in FILINGS)

    class _FakeCompany:
        def __init__(self, ticker):
            counters["companies"] += 1
            self.ticker = ticker

        def get_filings(self, form=None, amendments=True, **kwargs):
            counters["filing_lists"] += 1
            counters["amendments"] = amendments
            return filings

    monkeypatch.setattr(sec_series, "Company", _FakeCompany)
    monkeypatch.setattr(sec_series, "_require_identity",
                        lambda: "test@example.invalid")
    monkeypatch.setattr(sec_series, "_throttle", lambda: None)
    return counters


def _reader(monkeypatch, counters, tagged, dimensions=None, oldest_only=()):
    """Install a concept_point that tags `tagged` and records every read.

    `oldest_only` names concepts that appear in the oldest filing alone -- the
    NVDA case, where an element the filer abandoned still answers from an old
    filing and must not win the latest period's label.
    """
    def read(xbrl, concept, filing_date, form, accession=""):
        counters["reads"] += 1
        counters["concepts"].append((concept, accession))
        if concept not in tagged:
            return None
        if concept in oldest_only and accession != FILINGS[-1][0]:
            return None
        period = f"duration_{int(filing_date[:4]) - 1}-01-01_{filing_date}"
        return FilingPoint(filing_date, form, accession, facts=[
            ConceptFact(tagged[concept], period,
                        (dimensions or {}).get(concept, {}), f"c-{accession}")])

    monkeypatch.setattr(sec_series, "concept_point", read)
    return read


# ======================================== one parse per filing, not per concept

def test_sbc_parses_each_filing_once_not_once_per_concept(edgar, monkeypatch):
    """Eight concepts over five filings cost five parses, not forty.

    `get_sbc_series` reads three SBC elements, three revenue elements and two
    operating-cash-flow elements. Live against GS that was 40 parses of the
    same 5 filings and 66.7 seconds.
    """
    _reader(monkeypatch, edgar, {
        "us-gaap:ShareBasedCompensation": 6.4e9,
        "us-gaap:Revenues": 130.0e9,
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": 60.0e9,
    })

    result = sbc.get_sbc_series("FAKE", limit=5)

    assert result["success"] is True, result.get("error")
    assert edgar["parses"] == 5, (
        f"{edgar['parses']} parses for 5 filings -- the per-concept walk is "
        f"back")
    # 3 SBC + 3 revenue + 2 OCF concepts, each asked of each of the 5 filings.
    assert edgar["reads"] == 8 * 5, (
        "a concept was skipped; the shared walk must not have bought its "
        "speed by shortening the chains")


def test_debt_maturity_parses_the_one_filing_once_for_twelve_concepts(
        edgar, monkeypatch):
    """Six buckets, two concept families, one filing -- one parse.

    Live against GS this was 12 parses of a single 10-K and 65.9 seconds, all
    of it edgartools rebuilding the same fact table twelve times.
    """
    first = {bucket: options[0] for bucket, options in dm.MATURITY_CONCEPTS.items()}
    _reader(monkeypatch, edgar, {first["year_1"]: 9.0e9, first["year_2"]: 0.0})

    result = dm.get_debt_maturity_schedule("FAKE")

    assert result["coverage"] == "partial"
    assert result["by_year"]["year_1"] == 9.0e9
    assert result["by_year"]["year_2"] == 0.0, "a tagged zero was lost"
    assert edgar["parses"] == 1, (
        f"{edgar['parses']} parses of one filing; each bucket re-walked EDGAR")
    assert edgar["filing_lists"] == 1, (
        "the filing list was re-fetched per concept, which is a request to "
        "EDGAR per concept even when the parse is cached")


def test_contracted_revenue_parses_each_filing_once(edgar, monkeypatch):
    """RPO's two elements plus deferred revenue's four, over three filings."""
    _reader(monkeypatch, edgar, {
        "us-gaap:RevenueRemainingPerformanceObligation": 684.0e9,
        "us-gaap:ContractWithCustomerLiabilityCurrent": 73.0e9,
    })

    result = fm.get_contracted_revenue("FAKE")

    assert result["success"] is True, result.get("error")
    assert result["rpo"][0]["value"] == 684.0e9
    assert edgar["parses"] == 3, f"{edgar['parses']} parses for 3 filings"
    assert edgar["reads"] == 6 * 3


def test_geographic_revenue_parses_the_one_filing_once(edgar, monkeypatch):
    """The chain falls through four elements before it finds the tagged one.

    A filer on `us-gaap:SalesRevenueNet` cost four parses of the same 10-K
    reaching it, and the fifth element is only reached when none answers.
    """
    axis = "srt:StatementGeographicalAxis"
    _reader(monkeypatch, edgar, {"us-gaap:SalesRevenueNet": 150.0e9},
            dimensions={"us-gaap:SalesRevenueNet": {axis: "country:US"}})

    result = fm.get_geographic_revenue("FAKE")

    assert result["success"] is True, result.get("error")
    assert result["by_region"][0]["region"] == "United States"
    assert edgar["parses"] == 1, (
        f"{edgar['parses']} parses of one filing for a chain of "
        f"{len(fm.REVENUE_CONCEPTS)} elements")


def test_annual_revenue_parses_each_filing_once(edgar, monkeypatch):
    """Seven elements across two taxonomies, over three filings.

    `get_annual_revenue` tries the whole US-GAAP chain and the whole IFRS one,
    because a 20-F filer may be on either basis. Live against GS that was 21
    parses of 3 filings and 46.4 seconds.
    """
    monkeypatch.setattr(fi, "_annual_filing_index", lambda t: {"10-K": NEWEST})
    _reader(monkeypatch, edgar, {"ifrs-full:RevenueFromSaleOfGoods": 3.8e12})

    result = fi.get_annual_revenue("FAKE")

    assert result["success"] is True, result.get("error")
    assert result["concept_used"] == "ifrs-full:RevenueFromSaleOfGoods"
    assert edgar["parses"] == 3, f"{edgar['parses']} parses for 3 filings"


# ============================================== the chain rules are unchanged

def test_every_concept_is_still_evaluated_so_freshness_can_decide(
        edgar, monkeypatch):
    """The Ford rule, through the shared walk.

    NVDA tags `RevenueFromContractWithCustomerExcludingAssessedTax` only in
    its FY2022 10-K and `Revenues` since. The chain lists the ASC 606 element
    first, so stopping at the first hit resolved revenue for 2022 alone and
    left `pct_of_revenue` blank for every year after it -- the column the tool
    exists to produce, empty, with no reason given.
    """
    abandoned = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    _reader(monkeypatch, edgar, {
        "us-gaap:ShareBasedCompensation": 6.4e9,
        abandoned: 26.9e9,
        "us-gaap:Revenues": 130.0e9,
    }, oldest_only=(abandoned,))

    result = sbc.get_sbc_series("FAKE", limit=5)

    assert result["success"] is True, result.get("error")
    assert result["series"][0]["pct_of_revenue"] is not None, (
        "the chain stopped at the element NVDA abandoned, so the latest "
        "year's ratio has no denominator")
    assert result["series"][0]["pct_of_revenue"] == pytest.approx(
        6.4e9 / 130.0e9 * 100.0)


def test_a_maturity_bucket_still_stops_at_the_first_concept_that_answers(
        edgar, monkeypatch):
    """`_bucket_value` is the deliberate exception to the every-concept rule.

    A bucket is one number from one filing, so there is no stale-element
    hazard to guard against and the rolling-year alternative is only worth
    asking for when the fixed-year one is silent. Sharing the walk must not
    quietly turn that into "read both".
    """
    fixed, rolling = dm.MATURITY_CONCEPTS["year_1"]
    _reader(monkeypatch, edgar, {fixed: 9.0e9, rolling: 7.0e9})

    result = dm.get_debt_maturity_schedule("FAKE")

    assert result["by_year"]["year_1"] == 9.0e9, (
        "the rolling-year alternative overwrote the fixed-year answer")
    assert (rolling, "acc-2026") not in edgar["concepts"], (
        "the fallback element was read even though the primary answered")


def test_the_walk_still_excludes_amendments(edgar, monkeypatch):
    """A 10-K/A carrying only Part III must not take a slot in the walk.

    TSLA's most recent "10-K" is one: 37 fact rows and no financial
    statements. Left in, every concept in the real 10-K behind it reads as
    untagged.
    """
    _reader(monkeypatch, edgar, {"us-gaap:ShareBasedCompensation": 1.0})

    sbc.get_sbc_series("FAKE", limit=5)

    assert edgar["amendments"] is False


# ============================================ fetch_concept_series is the seam

SEAM_CASES = [
    (sbc, "get_sbc_series"),
    (dm, "get_debt_maturity_schedule"),
    (fm, "get_contracted_revenue"),
    (fm, "get_geographic_revenue"),
    (fm, "get_public_float"),
    (fi, "get_annual_revenue"),
]


@pytest.mark.parametrize("module,tool", SEAM_CASES,
                         ids=[f"{m.__name__.split('.')[-1]}.{t}"
                              for m, t in SEAM_CASES])
def test_a_replaced_seam_stands_the_shared_walk_down(module, tool, monkeypatch):
    """Roughly thirty tests here replace `fetch_concept_series` on a module.

    If the walk ran anyway they would all silently start talking to live
    EDGAR: slow, rate-limited, and asserting against whatever the filer
    happens to disclose today rather than against the canned facts they set
    up. `Company` is made to explode so "the walk ran" cannot pass quietly.
    """
    monkeypatch.setattr(fi, "_annual_filing_index", lambda t: {"10-K": NEWEST})
    calls = []

    def stub(ticker, concept, form="10-K", limit=3):
        calls.append(concept)
        raise sec_series.NotCovered(concept)

    monkeypatch.setattr(module, "fetch_concept_series", stub)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the shared walk ran despite a replaced seam")

    monkeypatch.setattr(sec_series, "Company", explode)

    result = getattr(module, tool)("FAKE")

    assert result["success"] is False
    assert calls, "the replaced seam was never called"


def test_a_walk_left_by_one_module_does_not_serve_another(edgar, monkeypatch):
    """The walk is keyed to one ticker, and it is torn down on the way out.

    The MCP server hands each tool call to a pooled worker thread, so a walk
    left behind on that thread would serve one ticker's filings to the next
    ticker's call -- and one module's to the next module's.
    """
    _reader(monkeypatch, edgar, {
        "us-gaap:ShareBasedCompensation": 6.4e9,
        "us-gaap:RevenueRemainingPerformanceObligation": 684.0e9,
    })

    sbc.get_sbc_series("FIRST", limit=5)
    assert getattr(shared_filings.ACTIVE, "walk", None) is None, (
        "the walk outlived the call that opened it")

    fm.get_contracted_revenue("SECOND")
    assert edgar["companies"] == 2, (
        "the second call was served the first call's filings")


def test_two_threads_do_not_share_one_walk(edgar, monkeypatch):
    """`ACTIVE` is a threading.local for exactly this reason.

    Tool calls run concurrently on the server's worker pool. A module-level
    walk would let a call for one ticker read another ticker's parsed
    filings, which is the worst failure this package can produce: a real
    number, from the wrong company.
    """
    _reader(monkeypatch, edgar, {"us-gaap:ShareBasedCompensation": 6.4e9})
    seen = {}

    def run(ticker):
        seen[ticker] = sbc.get_sbc_series(ticker, limit=1)

    threads = [threading.Thread(target=run, args=(t,))
               for t in ("ALPHA", "BETA")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert set(seen) == {"ALPHA", "BETA"}
    assert edgar["companies"] == 2, (
        "one thread's walk was reused by the other")


# ==================================================================== the clock

BUDGET_CASES = [
    (sbc, "get_sbc_series"),
    (dm, "get_debt_maturity_schedule"),
    (fm, "get_contracted_revenue"),
    (fm, "get_geographic_revenue"),
    (fi, "get_annual_revenue"),
]


@pytest.mark.parametrize("module,tool", BUDGET_CASES,
                         ids=[f"{m.__name__.split('.')[-1]}.{t}"
                              for m, t in BUDGET_CASES])
def test_a_timeout_is_never_reported_as_a_finding_about_the_filer(
        module, tool, edgar, monkeypatch):
    """The rule `test_outage_is_not_a_finding` enforces, applied to the clock.

    A chain abandoned to the budget was not evaluated. "FAKE does not tag
    long-term debt maturities" would be an affirmative claim about a filer's
    disclosure, made on the strength of a stopwatch, and phrased for an agent
    to repeat.
    """
    import time

    monkeypatch.setattr(fi, "_annual_filing_index", lambda t: {"10-K": NEWEST})
    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0.05")

    def slow(xbrl, concept, filing_date, form, accession=""):
        time.sleep(0.02)
        return None

    monkeypatch.setattr(sec_series, "concept_point", slow)

    result = getattr(module, tool)("FAKE")

    assert result["success"] is False
    assert result.get("timed_out") is True, (
        f"{tool} did not distinguish the clock from a coverage gap")
    assert result.get("coverage") != "not_covered"
    message = (result.get("error") or "").lower()
    for claim in ("does not tag", "does not disaggregate", "does not report",
                  "not covered", "does not disclose"):
        assert claim not in message, (
            f"{tool} reported a timeout as a fact about the filer: {message}")
    assert "budget" in message and f"{tool}(fake)" in message, (
        f"{tool} does not say what ran long: {message}")


def test_a_timed_out_call_returns_no_partial_answer(edgar, monkeypatch):
    """A half-walked chain looks exactly like a filer that tags nothing.

    Returning the buckets it reached before the clock ran out would put
    `total` beside a schedule missing whichever years came last in the loop,
    and nothing in the response would say so.
    """
    import time

    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0.05")
    first = {bucket: options[0] for bucket, options in dm.MATURITY_CONCEPTS.items()}

    def slow(xbrl, concept, filing_date, form, accession=""):
        time.sleep(0.02)
        if concept != first["year_1"]:
            return None
        return FilingPoint(filing_date, form, accession, facts=[
            ConceptFact(9.0e9, "2026-06-30", {}, "c-1")])

    monkeypatch.setattr(sec_series, "concept_point", slow)

    result = dm.get_debt_maturity_schedule("FAKE")

    assert result["timed_out"] is True
    assert result["by_year"] == {}
    assert result["total"] is None
    assert result["buckets_found"] == 0


def test_an_unbounded_budget_does_not_time_out(edgar, monkeypatch):
    """`0` is an operator saying they will wait; a typo is not.

    `budget_seconds` falls back to the default for anything unparseable, so
    only an explicit zero removes the bound.
    """
    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0")
    _reader(monkeypatch, edgar, {"us-gaap:ShareBasedCompensation": 6.4e9})

    result = sbc.get_sbc_series("FAKE", limit=5)

    assert result["success"] is True, result.get("error")
    assert "timed_out" not in result
