"""Tier 3: accounting identities, checked against live filings.

Every serious extraction defect in this project passed its own tests. MSFT's
total assets read 207.7bn against a real 758.4bn because `by_concept()` prefix-
matched `AssetsCurrent`. The number was plausible, the shape was right, and
nothing in the suite compared it to anything else. One identity would have
caught it on the first run: current assets cannot exceed total assets.

That is what this file is. No golden values, no prediction of which bug happens
next -- only relationships arithmetic forces to hold for every filer in every
year. A corrupted extraction fails them without anyone having anticipated the
corruption.

Three rules govern the file:

* **A violation is a finding to adjudicate against the filing, never a
  tolerance to widen.** Every tolerance below names what it absorbs and why
  that thing is not a defect. Legitimate exceptions -- minority interest,
  mezzanine equity, MSFT's debt-exchange premium -- are encoded by name, with
  the filing's own figures, never by loosening the check for everyone.
* **Adjudicated tool defects go in KNOWN_DEFECTS, not into a tolerance.** The
  check stays live for every other filer, each entry carries the evidence, and
  `test_known_defect_register_is_not_stale` fails the moment one is fixed, so
  the register cannot quietly outlive the bug.
* **Not-checkable is reported, never silently passed.** Six filers in this
  basket do not tag `us-gaap:Liabilities` at all. That is a coverage gap the
  caller needs to see, not a pass.

Network-gated and rate-limited. Never runs in the offline suite.
"""
import os
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
pytestmark = pytest.mark.skipif(SKIP_NETWORK, reason="live SEC identity sweep")


# Structural variety rather than familiarity. Banks carry no classified balance
# sheet, REITs earn most revenue under ASC 842 rather than ASC 606, CHTR and T
# carry large minority interests, SPG carries mezzanine equity, and FOXA, LEN,
# GOOGL and META are multi-class. Each group breaks an assumption the megacaps
# never test.
BASKET = {
    "megacap":     ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA"],
    "bank":        ["JPM", "BAC", "WFC", "GS"],
    "reit":        ["O", "SPG", "PLD", "AMT"],
    "retailer":    ["WMT", "COST", "TGT", "HD"],
    "biotech":     ["MRNA", "BIIB", "VRTX", "REGN"],
    "industrial":  ["CAT", "GE", "BA", "HON"],
    "energy":      ["XOM", "CVX"],
    "nci_heavy":   ["CHTR", "T"],
    "multi_class": ["FOXA", "LEN"],
}
ALL_TICKERS = sorted({t for group in BASKET.values() for t in group})

# ------------------------------------------------------------------ concepts

ASSETS = "us-gaap:Assets"
LIABILITIES = "us-gaap:Liabilities"
LIABILITIES_AND_EQUITY = "us-gaap:LiabilitiesAndStockholdersEquity"
EQUITY_PARENT = "us-gaap:StockholdersEquity"
EQUITY_TOTAL = ("us-gaap:StockholdersEquityIncludingPortionAttributable"
                "ToNoncontrollingInterest")
MINORITY_INTEREST = "us-gaap:MinorityInterest"

# Mezzanine. Redeemable stock and redeemable noncontrolling interests sit
# between liabilities and equity on the face of the balance sheet and belong to
# neither total, so a filer carrying them fails A = L + E by exactly their
# amount. SPG carries 233,306,000 of it.
TEMP_EQUITY_TOTAL = ("us-gaap:TemporaryEquityCarryingAmountIncludingPortion"
                     "AttributableToNoncontrollingInterests")
TEMP_EQUITY_PARENT = "us-gaap:TemporaryEquityCarryingAmountAttributableToParent"
TEMP_EQUITY_NCI = ("us-gaap:RedeemableNoncontrollingInterestEquity"
                   "CarryingAmount")

REVENUES = "us-gaap:Revenues"
REVENUE_ASC606 = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
# The total-revenue line on a bank's income statement, and the only revenue
# element GS and WFC tag undimensioned -- GS does not tag us-gaap:Revenues at
# all. Without it here, identity 2b can only report that a bank's revenue came
# off an element the check does not know about, which is a gap in the check
# rather than a finding about the filer.
REVENUES_NET_OF_INTEREST = "us-gaap:RevenuesNetOfInterestExpense"
REVENUE_ELEMENTS = (REVENUES, REVENUES_NET_OF_INTEREST, REVENUE_ASC606,
                    "us-gaap:SalesRevenueNet")
OPERATING_INCOME = "us-gaap:OperatingIncomeLoss"
OCF = "us-gaap:NetCashProvidedByUsedInOperatingActivities"

# `get_historical_fcf` looks only at the first two of these. The rest are here
# so the test can prove a filer DOES tag capex before calling a missing capex a
# defect -- AMZN, T and NVDA all use PaymentsToAcquireProductiveAssets, and PLD
# builds warehouses under PaymentsToDevelopRealEstateAssets.
#
# Acquisition elements (PaymentsToAcquireRealEstate,
# PaymentsToAcquireCommercialRealEstate) are deliberately absent. Buying a
# finished building is closer to an acquisition than to capital expenditure, and
# a tool that leaves it out of free cash flow is making a defensible choice
# rather than a mistake. Only spend that is unambiguously capex is grounds for
# calling the omission a defect.
CAPEX_ELEMENTS = (
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsForCapitalImprovements",
    "us-gaap:PaymentsToAcquireProductiveAssets",
    "us-gaap:PaymentsToAcquireOtherProductiveAssets",
    "us-gaap:PaymentsToDevelopRealEstateAssets",
)

LONG_TERM_DEBT_NONCURRENT = "us-gaap:LongTermDebtNoncurrent"
LONG_TERM_DEBT_CURRENT = "us-gaap:LongTermDebtCurrent"
LONG_TERM_DEBT = "us-gaap:LongTermDebt"
SHARES_OUTSTANDING = "dei:EntityCommonStockSharesOutstanding"

CONCEPTS = [
    ASSETS, LIABILITIES, LIABILITIES_AND_EQUITY, EQUITY_PARENT, EQUITY_TOTAL,
    MINORITY_INTEREST, TEMP_EQUITY_TOTAL, TEMP_EQUITY_PARENT, TEMP_EQUITY_NCI,
    "us-gaap:AssetsCurrent", "us-gaap:AssetsNoncurrent",
    "us-gaap:LiabilitiesCurrent", "us-gaap:LiabilitiesNoncurrent",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:InventoryNet", "us-gaap:AccountsReceivableNetCurrent",
    "us-gaap:AccountsPayableCurrent", "us-gaap:PropertyPlantAndEquipmentNet",
    LONG_TERM_DEBT_NONCURRENT, LONG_TERM_DEBT_CURRENT, LONG_TERM_DEBT,
    "us-gaap:OperatingLeaseLiability", "us-gaap:OperatingLeaseLiabilityCurrent",
    "us-gaap:OperatingLeaseLiabilityNoncurrent",
    "us-gaap:GrossProfit", OPERATING_INCOME, "us-gaap:NetIncomeLoss", OCF,
    SHARES_OUTSTANDING,
] + list(REVENUE_ELEMENTS) + list(CAPEX_ELEMENTS)

# ---------------------------------------------------------------- tolerances

# A balance sheet foots exactly. XBRL facts are the filer's own tagged decimals
# off one statement, so the residual is not "small", it is zero -- measured at
# exactly 0 for all 31 filers in this basket that tag the components. The
# allowance exists only so a float64 sum at bank scale (JPM: 4.4e12) cannot fail
# on representation: 1e-9 of that is $4,425, nine orders of magnitude below any
# error worth naming and thirteen below the MSFT defect that motivated this file.
_FOOTING_REL = 1e-9
_FOOTING_ABS = 1.0

# Two figures read from the same filing for the same concept and period are the
# same number. Anything above float64 noise means one of the two read paths
# picked a different fact.
_SAME_FACT_REL = 1e-9


def _footing_tolerance(scale: float) -> float:
    return max(_FOOTING_ABS, abs(scale) * _FOOTING_REL)


# --------------------------------------------------------- known tool defects

# Violations adjudicated against the filing and traced to the tool rather than
# to the identity. Listed here so the check stays live for the other filers,
# each with the filing figure that contradicts the tool. Written up in
# docs/known_issues.md under "Accounting-identity sweep (2026-08-22)".
#
# NOT a tolerance. Each entry is a specific filer failing a specific identity,
# and test_known_defect_register_is_not_stale fails if one starts holding.
KNOWN_DEFECTS: Dict[tuple, str] = {
    # get_segment_financials sums an aggregation member alongside the segments
    # it aggregates. CAT's us-gaap:ReportableSegmentAggregationBeforeOther
    # OperatingSegmentMember is 37,106m, exactly Construction 14,064 +
    # Financial Products 2,841 + Power & Energy 15,558 + Resource 4,643.
    ("3 segment revenue vs consolidated", "CAT"):
        "aggregation member double-counted with the segments it aggregates",
    ("3 segment revenue vs consolidated", "CVX"):
        "aggregation member double-counted with Upstream and Downstream",
    ("3 segment revenue vs consolidated", "LEN"):
        "len:LennarHomebuildingEastCentralWestHoustonandOtherMember is the sum "
        "of the five Homebuilding region members and is added to them",
    ("3 segment revenue vs consolidated", "AMT"):
        "amt:PropertyMember is the parent of the five Property region members "
        "and is added to them",
    ("3 segment revenue vs consolidated", "GE"):
        "segment revenue read off the intersegment-elimination context: "
        "Commercial Engines & Services reads -62,000,000 against "
        "33,252,000,000 tagged on the segment-only context in the same filing",
}


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@pytest.fixture(scope="module", autouse=True)
def _gentle_request_rate():
    """Hold the SEC request rate below edgartools' default for this sweep.

    `sec_series._throttle` already floors the gap between requests at
    `_MIN_REQUEST_GAP_S`, and it is deliberately not the lever used here: it
    guards the `fetch_concept_series` path only. `get_revenue_base`,
    `get_segment_financials` and `get_historical_fcf` reach EDGAR through
    `sec_utils.get_latest_filing`, which never passes through it, and each of
    those issues several requests per call. What earns a 429 is the rate summed
    across every path, and edgartools' own limiter is the one chokepoint that
    sees all of them -- so this lowers that rather than adding a third throttle
    beside the two already in play.

    Its default of 8/s is inside SEC fair access for a single call and outside
    it for a few thousand back to back: measured on this basket, the sweep earns
    a 429 that then blocks every request from this host for roughly nine
    minutes. The value is restored afterwards rather than left changed for the
    process.
    """
    import edgar.httprequests as httprequests

    previous = httprequests.max_requests_per_second
    httprequests.max_requests_per_second = 3
    yield
    httprequests.max_requests_per_second = previous


@pytest.fixture(scope="module")
def facts():
    """{ticker: {concept: FilingPoint | None}} from one filing fetch per filer.

    `fetch_concept_series` re-fetches the filing for every concept asked of it.
    Thirty-odd concepts across thirty filers is a thousand filing downloads and
    a guaranteed 429, so the filing is fetched once and every concept read out
    of the parsed XBRL through `concept_point` -- the same function the series
    walk calls, so this exercises the production prefix filter, dimension
    resolution and period selection rather than a copy of them.

    A filer whose latest 10-K cannot be fetched or parsed is recorded as a
    fetch failure and asserted on, never quietly dropped.
    """
    from edgar import Company
    from tools.web_search_server.sec_series import (
        _require_identity, _throttle, concept_point)

    _require_identity()

    out: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    for ticker in ALL_TICKERS:
        _throttle()
        try:
            filing = Company(ticker).get_filings(
                form="10-K", amendments=False).head(1)[0]
            xbrl = filing.xbrl()
            if xbrl is None:
                raise ValueError("filing carries no XBRL")
        except Exception as exc:  # noqa: BLE001 - recorded, then asserted on
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            continue

        row: Dict[str, Any] = {
            "_filing_date": str(filing.filing_date),
            "_accession": str(getattr(filing, "accession_no", "")),
        }
        for concept in CONCEPTS:
            row[concept] = concept_point(
                xbrl, concept, filing_date=str(filing.filing_date),
                form=str(filing.form),
                accession=str(getattr(filing, "accession_no", "")))
        out[ticker] = row

    return {"points": out, "failures": failures}


# Roughly seventy SEC requests per filer across the seven tools, because each
# one re-fetches the filing for every concept it tries. Two thousand requests
# back to back earns a 429 that blocks this host for about nine minutes, and the
# tools report that as "No filing found" or "not covered" -- a rate limit
# wearing the face of a filer that does not disclose something. Detected and
# retried rather than believed.
_RATE_LIMIT_MARKERS = ("Too Many Requests", "429", "No filing found",
                       "No 10-K filing", "No XBRL data available")
_RATE_LIMIT_BACKOFF_S = (60, 150, 300)


def _looks_rate_limited(result: Optional[dict]) -> bool:
    """True when a result is the shape a throttled request leaves behind.

    Every ticker in the basket has a parseable 10-K -- `facts` proves it before
    this fixture runs -- so "no filing" from a tool cannot be about the filer.
    """
    if result is None:
        return True
    text = f"{result.get('error', '')}"
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


@pytest.fixture(scope="module")
def tool_results(facts):
    """One call per tool per filer, shared across every identity below.

    Depends on `facts` for ordering as much as for data: the filing fetch runs
    first, so by the time a tool says "no filing" the filing is known to exist
    and the message can only be a throttled request.
    """
    import time

    from tools.web_search_server.debt_maturity import get_debt_maturity_schedule
    from tools.web_search_server.earnings_quality import get_accruals_quality
    from tools.web_search_server.forward_metrics import (get_geographic_revenue,
                                                         get_public_float)
    from tools.web_search_server.sec_utils import (get_historical_fcf,
                                                   get_revenue_base,
                                                   get_segment_financials)

    tools = {
        "revenue_base": lambda t: get_revenue_base(t),
        "segment": lambda t: get_segment_financials(t),
        "geographic": lambda t: get_geographic_revenue(t),
        "debt_maturity": lambda t: get_debt_maturity_schedule(t),
        "accruals": lambda t: get_accruals_quality(t, limit=1),
        "fcf": lambda t: get_historical_fcf(t),
        "public_float": lambda t: get_public_float(t),
    }

    results: Dict[str, Dict[str, Any]] = defaultdict(dict)
    raised: Dict[str, list] = defaultdict(list)
    throttled: list = []

    for ticker in ALL_TICKERS:
        if ticker not in facts["points"]:
            for name in tools:
                results[name][ticker] = None
            continue
        pending = dict(tools)
        for attempt in range(len(_RATE_LIMIT_BACKOFF_S) + 1):
            retry = {}
            for name, run in pending.items():
                try:
                    result = run(ticker)
                except Exception as exc:  # noqa: BLE001 - a raise IS a finding
                    raised[name].append((ticker, f"{type(exc).__name__}: {exc}"))
                    result = None
                results[name][ticker] = result
                if _looks_rate_limited(result):
                    retry[name] = run
            if not retry:
                break
            if attempt == len(_RATE_LIMIT_BACKOFF_S):
                throttled.extend(f"{name}/{ticker}" for name in retry)
                break
            time.sleep(_RATE_LIMIT_BACKOFF_S[attempt])
            pending = retry

    return {"results": results, "raised": raised, "throttled": throttled}


# ----------------------------------------------------------------- reporting

# Populated by each identity test, printed by the last one. Verdicts are the
# three the brief asks for and nothing else: a filer is either checked and
# holding, checked and violating, or not checkable.
LEDGER: Dict[str, Dict[str, list]] = defaultdict(
    lambda: {"holds": [], "violated": [], "not_checkable": []})


def _record(identity: str, verdict: str, ticker: str, detail: str = "") -> None:
    LEDGER[identity][verdict].append(f"{ticker}{' | ' + detail if detail else ''}")


def _fail(identity: str, ticker: str, messages: list, violations: list) -> None:
    """Record a violation and add it to the assertion unless it is registered.

    A registered defect still shows as violated in the table -- it is a
    violation -- it just does not re-break the build for a bug already written
    up. Anything unregistered fails here.
    """
    _record(identity, "violated", ticker, "; ".join(messages)[:200])
    if (identity, ticker) not in KNOWN_DEFECTS:
        violations.extend(f"{ticker}: {m}" for m in messages)


def _value(row: Optional[dict], concept: str) -> Optional[float]:
    """Consolidated value for a concept in one filing, or None if untagged."""
    if not row:
        return None
    point = row.get(concept)
    if point is None:
        return None
    fact = point.latest_undimensioned()
    return None if fact is None else fact.value


def _total(row: Optional[dict], concept: str) -> Optional[float]:
    """Sum across every distinct fact -- for share counts, where the classes
    are additive rather than a breakdown of a total."""
    if not row:
        return None
    point = row.get(concept)
    return None if point is None else point.total()


def _period(row: Optional[dict], concept: str) -> Optional[str]:
    """Period end for the fact `_value` selected, so two concepts are only ever
    compared when they describe the same date."""
    from tools.web_search_server.sec_series import _period_rank

    if not row:
        return None
    point = row.get(concept)
    if point is None:
        return None
    fact = point.latest_undimensioned()
    return None if fact is None else _period_rank(fact.period)[0]


def _consolidated_revenue(row: Optional[dict]) -> Optional[float]:
    """The largest revenue element the filer tags undimensioned.

    Largest rather than first: `Revenues` is the whole of it and the ASC 606
    element is the part earned under customer contracts, which for a REIT is a
    twelfth of the total. Where only one is tagged there is nothing to choose.
    """
    values = [v for v in (_value(row, c) for c in REVENUE_ELEMENTS)
              if v is not None]
    return max(values) if values else None


# ==================================================================== fetching

def test_every_filer_in_the_basket_yields_a_parsed_filing(facts):
    """A fetch failure invalidates every identity below it for that filer.

    Reported first and separately so a network problem is never mistaken for an
    accounting one.
    """
    failures = facts["failures"]
    assert not failures, (
        "latest 10-K could not be fetched or parsed:\n" +
        "\n".join(f"  {t}: {e}" for t, e in failures.items()))
    assert len(facts["points"]) >= 25, (
        f"only {len(facts['points'])} filers resolved; the sweep is specified "
        f"over at least 25")


def test_no_tool_result_is_a_throttled_request(tool_results):
    """A rate-limited sweep must fail, not report the filers it never reached.

    Every tool here answers a miss with "not covered", so a 429 arrives looking
    exactly like a filer that does not disclose something. Left unchecked it
    turns a degraded run into a clean report with holes in it.
    """
    throttled = tool_results["throttled"]
    assert not throttled, (
        "these tool calls still looked rate-limited after "
        f"{len(_RATE_LIMIT_BACKOFF_S)} backoffs; the sweep is incomplete and "
        "its results cannot be read as coverage:\n" +
        "\n".join(f"  {t}" for t in throttled))


# ============================================================== identity 1a/1b

_ID_1A = "1a assets == liabilities+equity total"
_ID_1B = "1b assets == liabilities + equity + mezzanine"


def test_balance_sheet_totals_agree(facts):
    """`Assets` == `LiabilitiesAndStockholdersEquity`.

    The strongest form of "the balance sheet balances", and the one with the
    widest coverage: a filer that does not tag `us-gaap:Liabilities` still tags
    the footing total, so this reaches every filer in the basket including the
    six that identity 1b cannot check.

    Tolerance is `_footing_tolerance` -- effectively exact. Measured residual is
    0 for every filer here.
    """
    violations = []
    for ticker, row in facts["points"].items():
        assets = _value(row, ASSETS)
        footing = _value(row, LIABILITIES_AND_EQUITY)
        if assets is None or footing is None:
            _record(_ID_1A, "not_checkable", ticker,
                    f"Assets={'tagged' if assets else 'untagged'}, "
                    f"LiabilitiesAndStockholdersEquity="
                    f"{'tagged' if footing else 'untagged'}")
            continue
        if _period(row, ASSETS) != _period(row, LIABILITIES_AND_EQUITY):
            _record(_ID_1A, "not_checkable", ticker,
                    "the two totals report different period ends")
            continue
        residual = assets - footing
        if abs(residual) > _footing_tolerance(assets):
            _fail(_ID_1A, ticker, [
                f"Assets {assets:,.0f} vs LiabilitiesAndStockholdersEquity "
                f"{footing:,.0f}, off by {residual:,.0f} "
                f"({residual / assets:+.4%})"], violations)
        else:
            _record(_ID_1A, "holds", ticker)

    assert not violations, (
        "the balance sheet does not foot -- one of the two totals is being "
        "read from the wrong fact:\n" + "\n".join(f"  {v}" for v in violations))


def test_assets_equal_liabilities_plus_equity(facts):
    """`Assets` == `Liabilities` + total equity + mezzanine.

    The decomposition, which is where a naive check produces false alarms:

    * **Minority interest.** `us-gaap:StockholdersEquity` is the parent's share
      only. Total equity is `StockholdersEquityIncludingPortionAttributableTo
      NoncontrollingInterest` where the filer tags it, otherwise the parent
      figure plus `MinorityInterest`. Checking against the parent figure alone
      reports every filer with a subsidiary as broken -- CHTR's NCI is 4.5bn.
    * **Mezzanine equity.** Redeemable stock and redeemable noncontrolling
      interests sit between the two totals and belong to neither, so they are
      added explicitly rather than absorbed by a wider tolerance. SPG carries
      233,306,000 of it and this identity fails by exactly that without the term.

    Not checkable for a filer that does not tag `us-gaap:Liabilities` at all --
    AMZN, CHTR, HON, T, TGT and WMT in this basket. Identity 1a covers them.
    """
    violations = []
    for ticker, row in facts["points"].items():
        assets = _value(row, ASSETS)
        liabilities = _value(row, LIABILITIES)

        total_equity = _value(row, EQUITY_TOTAL)
        equity_source = "StockholdersEquityIncludingNCI"
        if total_equity is None:
            parent = _value(row, EQUITY_PARENT)
            nci = _value(row, MINORITY_INTEREST)
            if parent is not None:
                total_equity = parent + (nci or 0.0)
                equity_source = ("StockholdersEquity+MinorityInterest"
                                 if nci is not None else "StockholdersEquity")

        mezzanine = _value(row, TEMP_EQUITY_TOTAL)
        if mezzanine is None:
            parent_temp = _value(row, TEMP_EQUITY_PARENT)
            nci_temp = _value(row, TEMP_EQUITY_NCI)
            if parent_temp is not None or nci_temp is not None:
                mezzanine = (parent_temp or 0.0) + (nci_temp or 0.0)

        if assets is None or liabilities is None or total_equity is None:
            missing = [name for name, v in (("Assets", assets),
                                            ("Liabilities", liabilities),
                                            ("equity", total_equity))
                       if v is None]
            _record(_ID_1B, "not_checkable", ticker,
                    f"untagged: {', '.join(missing)}")
            continue

        residual = assets - (liabilities + total_equity + (mezzanine or 0.0))
        if abs(residual) > _footing_tolerance(assets):
            _fail(_ID_1B, ticker, [
                f"Assets {assets:,.0f} - Liabilities {liabilities:,.0f} - "
                f"equity {total_equity:,.0f} ({equity_source}) - mezzanine "
                f"{mezzanine or 0:,.0f} = {residual:,.0f} "
                f"({residual / assets:+.4%})"], violations)
        else:
            detail = equity_source
            if mezzanine:
                detail += f", mezzanine {mezzanine:,.0f}"
            _record(_ID_1B, "holds", ticker, detail)

    assert not violations, (
        "assets do not equal liabilities plus equity:\n" +
        "\n".join(f"  {v}" for v in violations))


# ================================================================= identity 2

# Every pair here is a total and something the taxonomy defines as part of it.
# The relationship is containment, not equality: a filer may hold assets outside
# any of these components. Only the direction is asserted.
CONTAINMENT_PAIRS = [
    (ASSETS, "us-gaap:AssetsCurrent"),
    (ASSETS, "us-gaap:AssetsNoncurrent"),
    (ASSETS, "us-gaap:PropertyPlantAndEquipmentNet"),
    (ASSETS, "us-gaap:CashAndCashEquivalentsAtCarryingValue"),
    (ASSETS, "us-gaap:InventoryNet"),
    (ASSETS, "us-gaap:AccountsReceivableNetCurrent"),
    (LIABILITIES, "us-gaap:LiabilitiesCurrent"),
    (LIABILITIES, "us-gaap:LiabilitiesNoncurrent"),
    (LIABILITIES, LONG_TERM_DEBT_NONCURRENT),
    (LIABILITIES, "us-gaap:AccountsPayableCurrent"),
    (LIABILITIES, "us-gaap:OperatingLeaseLiability"),
    ("us-gaap:AssetsCurrent", "us-gaap:CashAndCashEquivalentsAtCarryingValue"),
    ("us-gaap:AssetsCurrent", "us-gaap:InventoryNet"),
    ("us-gaap:AssetsCurrent", "us-gaap:AccountsReceivableNetCurrent"),
    ("us-gaap:LiabilitiesCurrent", "us-gaap:AccountsPayableCurrent"),
    ("us-gaap:LiabilitiesCurrent", LONG_TERM_DEBT_CURRENT),
    ("us-gaap:LiabilitiesCurrent", "us-gaap:OperatingLeaseLiabilityCurrent"),
    (LONG_TERM_DEBT, LONG_TERM_DEBT_NONCURRENT),
    (LONG_TERM_DEBT, LONG_TERM_DEBT_CURRENT),
    ("us-gaap:OperatingLeaseLiability", "us-gaap:OperatingLeaseLiabilityCurrent"),
    ("us-gaap:OperatingLeaseLiability",
     "us-gaap:OperatingLeaseLiabilityNoncurrent"),
    (REVENUES, "us-gaap:GrossProfit"),
    (REVENUES, OPERATING_INCOME),
    (REVENUES, REVENUE_ASC606),
]

_ID_2 = "2 component <= parent"


def test_no_component_exceeds_its_parent(facts):
    """The literal check that would have caught the prefix-match bug.

    `us-gaap:Assets` prefix-matches `us-gaap:AssetsCurrent`, and the two share
    the balance-sheet context, so the current-assets fact survived every
    dimension filter and won the selection. Current assets cannot exceed total
    assets -- one line, no golden value, and MSFT's 207.7bn fails it instantly.

    Zero tolerance. Both sides are the filer's own tagged decimals from one
    statement at one instant, and only pairs reporting the same period end are
    compared, so there is nothing for a tolerance to absorb.

    `Revenues` over `RevenueFromContractWithCustomerExcludingAssessedTax` is in
    the list for the same reason: ASC 606 contract revenue is a *component* of
    revenue whenever a filer earns anything outside a customer contract, which
    for AMT is eleven twelfths of the total.
    """
    violations = []
    for ticker, row in facts["points"].items():
        checked = 0
        here = []
        for parent, child in CONTAINMENT_PAIRS:
            parent_value = _value(row, parent)
            child_value = _value(row, child)
            if parent_value is None or child_value is None:
                continue
            if _period(row, parent) != _period(row, child):
                continue
            checked += 1
            if child_value > parent_value:
                here.append(
                    f"{child.split(':')[1]} {child_value:,.0f} exceeds "
                    f"{parent.split(':')[1]} {parent_value:,.0f} by "
                    f"{(child_value - parent_value) / abs(parent_value):+.1%}")
        if here:
            _fail(_ID_2, ticker, here, violations)
        elif checked:
            _record(_ID_2, "holds", ticker, f"{checked} pairs")
        else:
            _record(_ID_2, "not_checkable", ticker,
                    "no parent/child pair tagged for a common period")

    assert not violations, (
        "a component exceeds the total it is part of -- the shape of the "
        "prefix-match defect:\n" + "\n".join(f"  {v}" for v in violations))


_ID_2B = "2b reported revenue is the consolidated total"


def test_reported_revenue_is_the_filers_consolidated_total(facts, tool_results):
    """`get_revenue_base` returns the consolidated total, not a piece of it.

    Two ways it can fail, and both are checked because they fail in opposite
    directions:

    * **The concept it names is not tagged undimensioned in that filing.** Then
      whatever it returned came off a dimension -- a segment or a geography --
      and is not consolidated anything. GOOGL does not tag the ASC 606 element
      undimensioned, and the tool returns 342,721,000,000, which is the Google
      Services segment, against 402,836,000,000 of revenue.
    * **It picked a smaller element while a larger one was tagged.** Revenue
      elements nest: `Revenues` is the whole and the ASC 606 element is the part
      earned under customer contracts. A chain that tries ASC 606 first returns
      AMT's 935,900,000 rather than its 10,644,600,000.

    Zero tolerance either way. Both comparisons are against the same filing, the
    same period, and in the first case the same concept.
    """
    violations = []
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["revenue_base"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or not result.get("success") or row is None:
            _record(_ID_2B, "not_checkable", ticker,
                    str((result or {}).get("error", "no result"))[:70])
            continue

        reported = float(result["revenue_base"])
        concept_used = str(result.get("concept_used") or "")
        here = []

        same_concept = _value(row, concept_used)
        if same_concept is None:
            here.append(
                f"reported {reported:,.0f} from {concept_used}, which has no "
                f"undimensioned fact in the filing -- the value came off a "
                f"dimension")
        elif abs(reported - same_concept) > _footing_tolerance(same_concept):
            here.append(
                f"reported {reported:,.0f} but {concept_used} reads "
                f"{same_concept:,.0f} undimensioned in the same filing")

        largest = _consolidated_revenue(row)
        if largest is not None and reported < largest * (1 - _SAME_FACT_REL):
            here.append(
                f"reported {reported:,.0f}, but the filing tags {largest:,.0f} "
                f"on a broader revenue element ({reported / largest:.1%} of it)")

        if here:
            _fail(_ID_2B, ticker, here, violations)
        elif largest is None:
            _record(_ID_2B, "not_checkable", ticker,
                    "no revenue element tagged undimensioned")
        else:
            _record(_ID_2B, "holds", ticker, concept_used.split(":")[-1])

    assert not violations, (
        "reported revenue is not the filer's consolidated total:\n" +
        "\n".join(f"  {v}" for v in violations))


# ================================================================= identity 3

# Reportable segments reconcile to the consolidated total through unallocated
# corporate costs, intersegment eliminations and an "all other" bucket that a
# filer may or may not tag on the segment axis, so a shortfall is a coverage
# number rather than a failure. An overshoot is not: it means a member is a
# parent of the others, or is counted twice. The 2% allowance covers a filer
# that tags an intersegment-revenue member on the same axis as its operating
# segments -- that member is genuinely additive to the parts and eliminated from
# the total. Measured on this basket, the largest legitimate overshoot is
# GOOGL's 0.03%; the registered defects run from 5.6% to 88%.
_SEGMENT_OVERSHOOT = 0.02

_ID_3 = "3 segment revenue vs consolidated"


def test_segment_revenue_reconciles_to_consolidated(facts, tool_results):
    """Segments sum to no more than consolidated revenue, and none is negative.

    The negative check is not redundant with the sum. GE's Commercial Engines &
    Services reads -62,000,000 -- the intersegment-elimination context -- while
    the segment-only context in the same filing carries 33,252,000,000. Netted
    into a two-segment total it still looked like a shortfall rather than a
    defect, which is exactly how it survived.
    """
    violations = []
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["segment"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or not result.get("success"):
            _record(_ID_3, "not_checkable", ticker,
                    str((result or {}).get("error", "no result"))[:70])
            continue

        here = []
        for segment in result.get("segments") or []:
            if not segment.get("revenue"):
                continue
            value = segment["revenue"][0]["value"]
            if value < 0:
                here.append(f"segment {segment['segment_member']} reports "
                            f"negative revenue {value:,.0f}")

        segment_total = result.get("total_latest_segment_revenue")
        consolidated = _consolidated_revenue(row)
        if not segment_total or not consolidated:
            if here:
                _fail(_ID_3, ticker, here, violations)
            else:
                _record(_ID_3, "not_checkable", ticker,
                        "no segment revenue or no consolidated revenue element")
            continue

        ratio = segment_total / consolidated
        if ratio > 1 + _SEGMENT_OVERSHOOT:
            here.append(
                f"segments sum to {segment_total:,.0f} against consolidated "
                f"revenue {consolidated:,.0f} ({ratio:.1%}); members "
                f"{[s['segment_member'] for s in result['segments']]}")

        if here:
            _fail(_ID_3, ticker, here, violations)
        else:
            _record(_ID_3, "holds", ticker, f"{ratio:.1%} of consolidated")

    assert not violations, (
        "segment revenue does not reconcile:\n" +
        "\n".join(f"  {v}" for v in violations))


# ================================================================= identity 4

_ID_4 = "4 geographic revenue vs consolidated"


def test_geographic_revenue_reconciles_to_consolidated(facts, tool_results):
    """Geographic revenue reconciles to the filing it was read from.

    The tool already detects nested members and sets `members_overlap` -- SAP
    tags EMEA alongside EMEA-excluding-Germany -- so an overlap is respected
    rather than reported as a defect. Comparing the disclosed sum against
    consolidated revenue would also be circular, because that comparison is how
    the flag is computed. What is checked is everything the flag does not imply:

    * the consolidated figure the tool reconciled against equals the concept it
      says it used, read independently from the same filing;
    * no single region exceeds consolidated revenue where members do not
      overlap;
    * no region is negative;
    * the period selected for each region is that region's latest and longest,
      not a quarter ending the same day. Region rows are sorted on the raw
      period string, and "duration_2025-10-27_2026-01-25" sorts above
      "duration_2025-01-27_2026-01-25" because "10" > "01" -- the four-fold
      understatement `sec_series._period_rank` exists to prevent.
    """
    from tools.web_search_server.sec_series import _period_rank

    violations = []
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["geographic"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or not result.get("success"):
            _record(_ID_4, "not_checkable", ticker,
                    str((result or {}).get("error", "no result"))[:70])
            continue

        consolidated = result.get("consolidated_revenue")
        regions = result.get("by_region") or []
        here = []

        reference = _value(row, str(result.get("concept_used") or ""))
        if consolidated and reference is not None:
            if abs(consolidated - reference) > _footing_tolerance(reference):
                here.append(
                    f"reconciled against consolidated revenue "
                    f"{consolidated:,.0f}, but {result['concept_used']} reads "
                    f"{reference:,.0f} in the same filing")

        for region in regions:
            latest = region["periods"][0]
            if latest["value"] < 0:
                here.append(f"region {region['region']} reports negative "
                            f"revenue {latest['value']:,.0f}")
            if (consolidated and not result.get("members_overlap")
                    and latest["value"] > consolidated):
                here.append(
                    f"region {region['region']} {latest['value']:,.0f} exceeds "
                    f"consolidated revenue {consolidated:,.0f} with "
                    f"members_overlap false")
            ranks = [_period_rank(p["period"]) for p in region["periods"]]
            if _period_rank(latest["period"]) != max(ranks):
                here.append(
                    f"region {region['region']} selected period "
                    f"{latest['period']} while {max(ranks)} is later or longer "
                    f"-- raw string sort picked a quarter over the year")

        if here:
            _fail(_ID_4, ticker, here, violations)
        elif not regions:
            _record(_ID_4, "not_checkable", ticker, "success with no regions")
        else:
            share = (f"{result['disclosed_total'] / consolidated:.1%} of "
                     f"consolidated" if consolidated else "no consolidated fact")
            _record(_ID_4, "holds", ticker,
                    f"{len(regions)} regions, {share}"
                    + (", members_overlap"
                       if result.get("members_overlap") else ""))

    assert not violations, (
        "geographic revenue does not reconcile:\n" +
        "\n".join(f"  {v}" for v in violations))


# ================================================================= identity 5

# The ladder is disclosed at face principal; the balance sheet carries long-term
# debt net of unamortised discount, premium and issuance costs. That wedge is a
# real difference between two correctly extracted numbers. Measured across this
# basket the drift is under 1% for eleven filers and 7.4% at the worst (T), so
# 10% separates the wedge from a dropped bucket, which is worth at least one
# sixth of a ladder.
_DEBT_LADDER_TOLERANCE = 0.10

# Adjudicated against the filing rather than absorbed into the tolerance above.
# MSFT's FY2026 10-K reconciles its own ladder explicitly:
#     Total face value                          46,136
#     Unamortized discount and issuance costs   (1,081)
#     Hedge fair value adjustments                 (11)
#     Premium on debt exchange                  (4,750)
#     Total debt                                40,294
# The 4,750 premium from the 2020 debt exchange is the whole of the 14.5% gap
# and is not tagged in any concept that would let the wedge be computed, so this
# one filer is allowed a wider band and no one else is.
_DEBT_LADDER_EXCEPTIONS = {
    "MSFT": (0.16, "face value 46,136 vs carrying 40,294 per the filing's own "
                   "reconciliation: 1,081 discount + 11 hedge + 4,750 premium "
                   "on debt exchange"),
}

_ID_5 = "5 debt buckets vs long-term debt"


def test_debt_maturity_buckets_sum_to_long_term_debt(facts, tool_results):
    """Bucket sum reconciles to long-term debt on the balance sheet.

    The reference is `LongTermDebtNoncurrent` plus the current portion, falling
    back to `LongTermDebt` where a filer tags only the combined figure -- eleven
    of this basket do, and without the fallback they read as not-checkable
    despite tagging a full ladder.

    Only where `coverage` is "full". A partial ladder is short by whatever the
    filer did not tag, so reconciling it would be asserting on a number the tool
    already says is incomplete.
    """
    violations = []
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["debt_maturity"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or result.get("coverage") != "full":
            _record(_ID_5, "not_checkable", ticker,
                    f"coverage={(result or {}).get('coverage', 'no result')}, "
                    f"buckets={(result or {}).get('buckets_found')}")
            continue

        noncurrent = _value(row, LONG_TERM_DEBT_NONCURRENT)
        current = _value(row, LONG_TERM_DEBT_CURRENT)
        if noncurrent is not None:
            balance_sheet = noncurrent + (current or 0.0)
            source = "LongTermDebtNoncurrent+Current"
        else:
            balance_sheet = _value(row, LONG_TERM_DEBT)
            source = "LongTermDebt"
        if not balance_sheet:
            _record(_ID_5, "not_checkable", ticker,
                    "no long-term debt concept tagged on the balance sheet")
            continue

        ladder = result["total"]
        drift = abs(ladder - balance_sheet) / balance_sheet
        allowed, why = _DEBT_LADDER_EXCEPTIONS.get(
            ticker, (_DEBT_LADDER_TOLERANCE, ""))
        if drift > allowed:
            _fail(_ID_5, ticker, [
                f"ladder sums to {ladder:,.0f} against {source} "
                f"{balance_sheet:,.0f}, {drift:.1%} apart; buckets "
                f"{result['by_year']}"], violations)
        else:
            note = f"{drift:.1%} apart ({source})"
            if why:
                note += f" -- allowed to {allowed:.0%}: {why}"
            _record(_ID_5, "holds", ticker, note)

    assert not violations, (
        "the maturity ladder does not reconcile to long-term debt:\n" +
        "\n".join(f"  {v}" for v in violations))


# ================================================================= identity 6

_ID_6A = "6a fcf == ocf - capex"
_ID_6B = "6b accruals == net income - ocf"


def test_free_cash_flow_is_operating_cash_flow_less_capex(facts, tool_results):
    """FCF == OCF - capex, with both inputs checked against the filing.

    The arithmetic alone would pass on two wrong inputs, so four things are
    checked:

    * the tool's own arithmetic, to the cent;
    * capex reported positive, since it is tagged negative on the cash flow
      statement and the tool takes its absolute value;
    * capex is not silently absent. `fcf = ocf - (capex or 0)` reports operating
      cash flow as free cash flow when the filer tags capex under an element the
      two-concept chain does not try, which for AMZN is the difference between
      139.5bn and 7.7bn;
    * the OCF the tool used matches the same filing read through `sec_series`.
      `get_historical_fcf` reads through `sec_utils.filter_annual_data`, which
      does not filter `by_concept`'s prefix match and breaks ties by taking the
      largest fact -- including dimensioned ones. For a bank, whose consolidated
      operating cash flow is negative, the largest fact is the parent-company-
      only Schedule I figure. Pitting the two extraction paths against each
      other is what makes this a real check rather than a restatement of
      `a - b == c`.
    """
    violations = []
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["fcf"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or not result.get("success"):
            _record(_ID_6A, "not_checkable", ticker,
                    str((result or {}).get("error", "no result"))[:70])
            continue

        ocf = result["operating_cash_flow"]
        capex = result["capex"]
        fcf = result["free_cash_flow"]
        here = []

        if abs(fcf - (ocf - (capex or 0.0))) > 0.01:
            here.append(f"free_cash_flow {fcf:,.0f} != operating_cash_flow "
                        f"{ocf:,.0f} - capex {capex or 0:,.0f}")
        if capex is not None and capex < 0:
            here.append(f"capex reported negative ({capex:,.0f}); free cash "
                        f"flow is being computed as OCF plus capex")
        if capex is None:
            tagged = [(c, _value(row, c)) for c in CAPEX_ELEMENTS]
            tagged = [(c, v) for c, v in tagged if v]
            if tagged:
                concept, value = max(tagged, key=lambda cv: abs(cv[1]))
                here.append(
                    f"capex untagged so free cash flow was returned as "
                    f"operating cash flow {fcf:,.0f}, but the filing tags "
                    f"{concept} at {abs(value):,.0f}")

        reference_ocf = _value(row, OCF)
        if reference_ocf is not None and ocf:
            drift = abs(ocf - reference_ocf) / abs(reference_ocf)
            if drift > _SAME_FACT_REL:
                here.append(
                    f"operating cash flow read as {ocf:,.0f}; the same filing "
                    f"through sec_series reads {reference_ocf:,.0f}")

        if here:
            _fail(_ID_6A, ticker, here, violations)
        else:
            _record(_ID_6A, "holds", ticker)

    assert not violations, (
        "free cash flow does not reconcile:\n" +
        "\n".join(f"  {v}" for v in violations))


def test_accruals_equal_net_income_less_operating_cash_flow(facts, tool_results):
    """accruals == net income - OCF, and the ratio's denominator is total assets.

    Same shape as the FCF check: the internal arithmetic to the cent, then both
    inputs against the filing read independently. The accrual ratio's
    denominator is where the MSFT defect actually landed -- 207.7bn instead of
    758.4bn makes the ratio 3.65x too large -- so total assets is checked too.
    """
    violations = []
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["accruals"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or not result.get("success") or not result.get("latest"):
            _record(_ID_6B, "not_checkable", ticker,
                    str((result or {}).get("error", "no result"))[:70])
            continue

        latest = result["latest"]
        net_income = latest["net_income"]
        ocf = latest["operating_cash_flow"]
        accruals = latest["accruals"]
        assets = latest["total_assets"]
        here = []

        if abs(accruals - (net_income - ocf)) > 0.01:
            here.append(f"accruals {accruals:,.0f} != net income "
                        f"{net_income:,.0f} - operating cash flow {ocf:,.0f}")
        ratio = latest["accrual_ratio_pct"]
        if ratio is not None and assets:
            if abs(ratio - accruals / assets * 100.0) > 1e-6:
                here.append(f"accrual_ratio_pct {ratio} does not equal "
                            f"accruals over assets {assets:,.0f}")

        reference_assets = _value(row, ASSETS)
        if assets is not None and reference_assets is not None:
            drift = abs(assets - reference_assets) / abs(reference_assets)
            if drift > _SAME_FACT_REL:
                here.append(f"accrual ratio uses total assets {assets:,.0f}; "
                            f"the same filing reads {reference_assets:,.0f}")
        reference_ocf = _value(row, OCF)
        if ocf and reference_ocf is not None:
            drift = abs(ocf - reference_ocf) / abs(reference_ocf)
            if drift > _SAME_FACT_REL:
                here.append(f"accruals use operating cash flow {ocf:,.0f}; the "
                            f"same filing reads {reference_ocf:,.0f}")

        if here:
            _fail(_ID_6B, ticker, here, violations)
        else:
            _record(_ID_6B, "holds", ticker,
                    f"accrual ratio {ratio:.2f}%" if ratio is not None
                    else "total assets untagged, ratio null")

    assert not violations, (
        "accruals do not reconcile:\n" + "\n".join(f"  {v}" for v in violations))


# ================================================================= identity 7

# Float is measured at the filer's second-quarter close; the cover-page share
# count is as of a date near the filing. Both come off the same cover page, so
# they describe the same capital structure -- which matters: HON's share count
# halved between its 10-K and its next 10-Q on a corporate separation, and
# comparing that 10-Q count against the 10-K's float reads as a 180% violation
# of an identity that is not violated at all.
#
# The price is the highest close in the year ending at the filing date, because
# the exact measurement date is not reliably recoverable -- CAT tags
# EntityPublicFloat on its fiscal-year-end context, not on the June measurement
# date. Taking the most favourable price in the window removes the date question
# entirely and the check still bites: a dropped share class halves the market
# value and no price in the window recovers it. Measured, the tightest filer is
# T at 99.8%, so 5% is slack and not cover.
_FLOAT_TOLERANCE = 0.05
_PRICE_WINDOW_DAYS = 400

_ID_7 = "7 float <= shares x price"


def test_public_float_does_not_exceed_the_value_of_all_shares(facts,
                                                              tool_results):
    """Float <= cover-page shares x the best close in the year to the filing."""
    import yfinance as yf

    violations = []
    checked = 0
    for ticker in ALL_TICKERS:
        result = tool_results["results"]["public_float"].get(ticker)
        row = facts["points"].get(ticker)
        if not result or not result.get("success") or row is None:
            _record(_ID_7, "not_checkable", ticker,
                    str((result or {}).get("error", "float not covered"))[:70])
            continue
        shares = _total(row, SHARES_OUTSTANDING)
        if not shares:
            _record(_ID_7, "not_checkable", ticker,
                    "cover-page share count untagged in the same filing")
            continue

        filed = date.fromisoformat(row["_filing_date"][:10])
        try:
            history = yf.Ticker(ticker).history(
                start=(filed - timedelta(days=_PRICE_WINDOW_DAYS)).isoformat(),
                end=filed.isoformat(), auto_adjust=False)
            high = float(history["Close"].max()) if len(history) else None
        except Exception:  # noqa: BLE001 - a price outage is not a finding here
            high = None
        if not high:
            _record(_ID_7, "not_checkable", ticker,
                    f"no price history in the year to {filed}")
            continue

        market_value = shares * high
        checked += 1
        ratio = result["public_float"] / market_value
        if ratio > 1 + _FLOAT_TOLERANCE:
            _fail(_ID_7, ticker, [
                f"public float {result['public_float'] / 1e9:,.1f}B exceeds "
                f"{shares:,.0f} cover-page shares x the year's best close "
                f"${high:,.2f} = {market_value / 1e9:,.1f}B ({ratio:.0%})"],
                violations)
        else:
            _record(_ID_7, "holds", ticker,
                    f"float is {ratio:.0%} of the year's peak market value")

    assert checked >= 10, f"only {checked} filers could be reconciled"
    assert not violations, (
        "public float exceeds the market value of every share outstanding, "
        "which is arithmetically impossible -- one of the two is wrong:\n" +
        "\n".join(f"  {v}" for v in violations))


# ===================================================================== report

def test_known_defect_register_is_not_stale(tool_results):
    """Every registered defect must still be violating.

    Without this the register outlives the bug: a fixed tool keeps its excuse
    and the next regression lands on a check that has already been told to
    ignore that filer.

    A throttled sweep cannot judge this -- a filer the run never reached looks
    exactly like a filer whose defect was fixed -- so it skips and lets
    test_no_tool_result_is_a_throttled_request carry the failure.
    """
    if tool_results["throttled"]:
        pytest.skip("sweep was rate-limited; staleness cannot be judged")
    still_violating = {
        (identity, entry.split(" | ")[0])
        for identity, counts in LEDGER.items()
        for entry in counts["violated"]
    }
    fixed = [f"{identity} / {ticker}: {reason}"
             for (identity, ticker), reason in KNOWN_DEFECTS.items()
             if (identity, ticker) not in still_violating]
    assert not fixed, (
        "KNOWN_DEFECTS lists violations that no longer occur. Delete these "
        "entries and the matching notes in docs/known_issues.md:\n" +
        "\n".join(f"  {f}" for f in fixed))


def test_identity_results_are_recorded(capsys, tool_results):
    """Print the holds / violated / not-checkable table for every identity.

    Runs last so it sees every ledger entry. Not a pass/fail gate on coverage:
    an identity that only reaches half the basket is still worth having, and
    hiding that number is how a half-covered check gets mistaken for a whole one.
    """
    lines = ["", f"accounting identities over {len(ALL_TICKERS)} filers:", ""]
    header = f"  {'identity':46s} {'holds':>6s} {'viol':>5s} {'n/a':>5s}"
    lines += [header, "  " + "-" * (len(header) - 2)]
    for identity in sorted(LEDGER):
        counts = LEDGER[identity]
        lines.append(f"  {identity:46s} {len(counts['holds']):>6d} "
                     f"{len(counts['violated']):>5d} "
                     f"{len(counts['not_checkable']):>5d}")

    for identity in sorted(LEDGER):
        counts = LEDGER[identity]
        if counts["violated"]:
            lines += ["", f"  {identity} VIOLATED by:"]
            for entry in counts["violated"]:
                registered = KNOWN_DEFECTS.get(
                    (identity, entry.split(" | ")[0]))
                lines.append(f"      {entry}")
                if registered:
                    lines.append(f"        adjudicated: {registered}")
        if counts["not_checkable"]:
            lines += ["", f"  {identity} not checkable for:"]
            for entry in counts["not_checkable"]:
                lines.append(f"      {entry}")

    raised = tool_results["raised"]
    if any(raised.values()):
        lines += ["", "  tools that raised instead of returning a result:"]
        for tool, items in raised.items():
            for entry in items:
                lines.append(f"      {tool}: {entry}")

    with capsys.disabled():
        print("\n".join(lines))

    assert LEDGER, "no identity recorded a result; the sweep did not run"
