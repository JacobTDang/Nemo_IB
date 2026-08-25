"""What `get_ebitda_margin` does for a filer that tags no operating income.

`filter_annual_data` now returns the undimensioned fact or nothing, and that
turned a silent wrong answer into a visible gap: 11 of the 32 filers in the
audit basket tag no `us-gaap:OperatingIncomeLoss` at all -- JPM, BAC, WFC, GS,
O, BIIB, XOM, CVX, GE, FOXA and LEN. Before it, the chain trailed off into a
bare `OperatingIncomeLoss` and a bare `IncomeLossFromContinuingOperations`, and
the prefix match answered both with whatever longer element a filer happened to
tag: pre-tax income for JPM (72,595,000,000) and XOM (45,969,000,000), and for
GS `gs:GeographicReportingInformationPercentageOfOperatingIncomeLoss` -- four
facts of 0.69, 0.23, 0.08 and 1.00, of which `idxmax` took the 1.00 and
reported it as Goldman's operating income in dollars.

None of the eleven can be served, and this file pins the reasons so a later
change cannot quietly re-open the hole:

* **A bank has no operating income line and EBITDA means nothing for one.**
  Interest is the raw material, not a financing cost. Goldman's revenue is
  struck as `us-gaap:RevenuesNetOfInterestExpense` -- already net of the
  66,814,000,000 that EBITDA exists to add back.
* **The others present no operating income subtotal.** The only subtotal they
  tag is `...BeforeIncomeTaxes...`, which is pre-tax income, and revenue less
  `us-gaap:CostsAndExpenses` is not operating income either: GE's expense total
  carries `ge:InterestAndOtherFinancialCharges` (843,000,000) and a
  non-operating pension credit (-788,000,000), O tags its financing interest as
  `us-gaap:InterestExpenseOperating` (1,134,879,000) inside that total, and
  BIIB nests `us-gaap:NonoperatingIncomeExpense` (-305,600,000) in it. GE is
  the case that kills the general rule: revenue less costs plus other income
  reconciles to GE's tagged pre-tax income exactly, and the difference is still
  not operating income.

Measured against the live filings on 2026-08-24 before any code was changed.
"""
import os

import pytest

from testing.test_consolidated_fact_selection import FY2025, _XBRL, _fact

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    """SEC fair access needs a real contact address, and without it every live
    assertion below passes for the wrong reason: the fetch fails, nothing is
    tagged, and "not covered" is indistinguishable from a filer that genuinely
    tags nothing. The same is true of an SEC 429, which is why the live tests
    assert on D&A -- a figure only a filing that was actually read can carry."""
    from dotenv import load_dotenv
    load_dotenv()


# The eleven, by the reason each one is refused.
BANKS = ["JPM", "BAC", "WFC", "GS"]
NO_OPERATING_INCOME_SUBTOTAL = ["O", "BIIB", "XOM", "CVX", "GE", "FOXA", "LEN"]
UNCOVERED = BANKS + NO_OPERATING_INCOME_SUBTOTAL

# Filers that do tag operating income, one per structure, to prove the refusal
# did not widen into the covered set.
COVERED = ["MSFT", "CAT", "PLD", "WMT"]


# --------------------------------------------------------------- fake filings


def _filing(rows):
    return {"filing_date": "2026-02-20", "url": None,
            "accession_number": "0000000000-26-000001",
            "filing_object": None, "xbrl_data": _XBRL(rows)}


def _use(monkeypatch, rows):
    from tools.web_search_server import sec_utils
    monkeypatch.setattr(sec_utils, "get_latest_filing",
                        lambda ticker, form_type='10-K': _filing(rows))


DA = _fact("us-gaap:DepreciationDepletionAndAmortization", 2_524_200_000,
           "c-1", FY2025)
REVENUE = _fact("us-gaap:Revenues", 5_749_377_000, "c-1", FY2025)
# The element the old chain reached by prefix and reported as operating income.
PRETAX = _fact("us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
               "ExtraordinaryItemsNoncontrollingInterest", 1_155_129_000,
               "c-1", FY2025)


# ------------------------------------------------------------------- offline


def test_a_filer_that_tags_operating_income_is_still_served(monkeypatch):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    _use(monkeypatch, [
        _fact("us-gaap:OperatingIncomeLoss", 963_395_000, "c-1", FY2025),
        DA, REVENUE, PRETAX,
    ])
    result = get_ebitda_margin("FAKE")
    assert result["success"] is True
    assert result["coverage"] == "full"
    assert result["operating_income"] == 963_395_000
    assert result["ebitda_amount"] == 963_395_000 + 2_524_200_000
    assert result["operating_income_concept_used"] == "us-gaap:OperatingIncomeLoss"


def test_pre_tax_income_is_never_substituted_for_operating_income(monkeypatch):
    """The defect itself. Everything EBITDA needs is here except the one
    element that is actually operating income, and pre-tax income sits beside
    it looking usable."""
    from tools.web_search_server.sec_utils import get_ebitda_margin

    _use(monkeypatch, [DA, REVENUE, PRETAX])
    result = get_ebitda_margin("FAKE")
    assert result["success"] is False
    assert result["coverage"] == "not_covered"
    assert result["operating_income"] is None
    assert result["ebitda_amount"] is None
    assert result["ebitda_margin_percent"] is None
    assert 1_155_129_000 not in [v for v in result.values()
                                 if isinstance(v, (int, float))]


def test_the_refusal_names_the_element_it_looked_for(monkeypatch):
    """`get_debt_maturity_schedule` reports which buckets it tried. A refusal
    that does not name the element cannot be checked against the filing."""
    from tools.web_search_server.sec_utils import (OPERATING_INCOME_CONCEPTS,
                                                   get_ebitda_margin)

    _use(monkeypatch, [DA, REVENUE, PRETAX])
    result = get_ebitda_margin("FAKE")
    assert result["concepts_tried"] == list(OPERATING_INCOME_CONCEPTS)
    for concept in OPERATING_INCOME_CONCEPTS:
        assert concept in result["error"]
    assert "pre-tax" in result["error"].lower()


def test_the_halves_that_were_read_are_still_handed_back(monkeypatch):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    _use(monkeypatch, [DA, REVENUE, PRETAX])
    result = get_ebitda_margin("FAKE")
    assert result["d&a"] == 2_524_200_000
    assert result["revenue"] == 5_749_377_000


def test_a_bank_is_refused_for_being_a_bank_not_for_a_missing_element(
        monkeypatch):
    """A bank's income statement has no operating income line and never will.
    Reporting that as "element not found" invites someone to go find another
    element; the reason is that the measure does not apply."""
    from tools.web_search_server.sec_utils import get_ebitda_margin

    _use(monkeypatch, [
        _fact("us-gaap:RevenuesNetOfInterestExpense", 58_283_000_000, "c-1",
              FY2025),
        _fact("us-gaap:InterestIncomeExpenseNet", 13_559_000_000, "c-1",
              FY2025),
        _fact("us-gaap:NoninterestExpense", 37_544_000_000, "c-1", FY2025),
        _fact("us-gaap:DepreciationAndAmortization", 2_182_000_000, "c-1",
              FY2025),
        PRETAX,
    ])
    result = get_ebitda_margin("FAKE")
    assert result["success"] is False
    assert result["coverage"] == "not_covered"
    assert result["ebitda_margin_percent"] is None
    error = result["error"].lower()
    assert "bank" in error
    assert "interest" in error


def test_a_missing_d_and_a_is_refused_with_its_own_reason(monkeypatch):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    _use(monkeypatch, [
        _fact("us-gaap:OperatingIncomeLoss", 963_395_000, "c-1", FY2025),
        REVENUE,
    ])
    result = get_ebitda_margin("FAKE")
    assert result["success"] is False
    assert result["coverage"] == "not_covered"
    assert "d&a" in result["error"].lower() or \
           "depreciation" in result["error"].lower()
    assert result["operating_income"] == 963_395_000


def test_success_and_refusal_carry_the_same_keys(monkeypatch):
    """A caller reading result['coverage'] must not have to branch on whether
    the key exists."""
    from tools.web_search_server.sec_utils import get_ebitda_margin

    _use(monkeypatch, [
        _fact("us-gaap:OperatingIncomeLoss", 963_395_000, "c-1", FY2025),
        DA, REVENUE,
    ])
    served = get_ebitda_margin("FAKE")
    _use(monkeypatch, [DA, REVENUE, PRETAX])
    refused = get_ebitda_margin("FAKE")
    for key in ("ticker", "success", "coverage", "error", "concepts_tried",
                "ebitda_margin_percent", "ebitda_amount", "operating_income",
                "d&a", "revenue"):
        assert key in served, key
        assert key in refused, key


def test_an_unreadable_filing_names_the_failure(monkeypatch):
    """`except Exception: return 'Failed to get latest file'` reported a bug in
    this function as a fact about the filer."""
    from tools.web_search_server import sec_utils

    def boom(ticker, form_type='10-K'):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(sec_utils, "get_latest_filing", boom)
    result = sec_utils.get_ebitda_margin("FAKE")
    assert result["success"] is False
    assert "connection reset by peer" in result["error"]


# ---------------------------------------------------------------------- live


@network
@pytest.mark.parametrize("ticker", UNCOVERED)
def test_an_uncovered_filer_refuses_rather_than_returning_a_number(ticker):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    result = get_ebitda_margin(ticker, "10-K")
    assert result["success"] is False, (
        f"{ticker} returned {result.get('ebitda_margin_percent')}% -- it tags "
        f"no us-gaap:OperatingIncomeLoss, so that came from somewhere else")
    assert result["coverage"] == "not_covered"
    assert result["ebitda_margin_percent"] is None
    assert result["operating_income"] is None
    # Proof the filing was read rather than throttled away. All eleven tag a
    # combined D&A total; a 429 or a missing SEC_EMAIL yields None here and
    # would otherwise pass this test as a false negative.
    assert result["d&a"] is not None, (
        f"{ticker}: no D&A either, so the filing was not read -- {result['error']}")
    assert "us-gaap:OperatingIncomeLoss" in result["error"]


@network
@pytest.mark.parametrize("ticker", BANKS)
def test_a_bank_is_told_that_ebitda_does_not_apply_to_it(ticker):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    error = get_ebitda_margin(ticker, "10-K")["error"].lower()
    assert "bank" in error and "interest" in error, error


@network
@pytest.mark.parametrize("ticker", NO_OPERATING_INCOME_SUBTOTAL)
def test_a_filer_with_no_operating_subtotal_is_told_which_element_is_missing(
        ticker):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    result = get_ebitda_margin(ticker, "10-K")
    assert "us-gaap:OperatingIncomeLoss" in result["error"]
    assert "bank" not in result["error"].lower()


@network
@pytest.mark.parametrize("ticker", COVERED)
def test_a_covered_filer_is_unaffected(ticker):
    from tools.web_search_server.sec_utils import get_ebitda_margin

    result = get_ebitda_margin(ticker, "10-K")
    assert result["success"] is True, result.get("error")
    assert result["coverage"] == "full"
    assert result["operating_income_concept_used"] == "us-gaap:OperatingIncomeLoss"
    assert result["ebitda_margin_percent"] == pytest.approx(
        (result["operating_income"] + result["d&a"]) / result["revenue"] * 100)
