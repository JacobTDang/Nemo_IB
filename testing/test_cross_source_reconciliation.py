"""Tier 2: cross-source reconciliation. The same fact, from two feeds.

Every serious defect this project has shipped passed its own tests. MSFT's
total assets read 207.7bn against a real 758.4bn. Ford reported +5.9bn of net
income in a year it lost 8.2bn. Biogen's share count doubled. None crashed and
none looked wrong, because a test written beside the code inherits the code's
blind spots.

Reconciliation does not inherit them. Nobody has to anticipate the bug: the
same number is available from SEC XBRL, from Yahoo and from Finnhub, and where
they disagree materially at least one of them is wrong. That is how every
finding below was caught -- none of them was being looked for.

Three rules the checks are built on:

1. **Period-match before comparing.** A fiscal year and a calendar year are
   different periods, not a disagreement. Every statement comparison joins on
   the actual period end date and reports `not_comparable` when the vendor has
   no period within a week of the filing's.
2. **Currency-match before comparing.** TSM reports TWD and ASML EUR. A TWD
   figure against a USD figure is not a disagreement either.
3. **A disagreement is adjudicated, never tolerated.** Every entry in
   ADJUDICATED was resolved by reading the filing's raw XBRL facts, and the
   entry records which source is right. Widening a tolerance to make a check
   pass deletes the check, so no tolerance here is wider than the measured
   spread of the cases that genuinely agree.

ADJUDICATED is a baseline, not an exemption: a disagreement outside it fails,
and an entry inside it that starts agreeing ALSO fails, so the entry is deleted
when the bug is fixed rather than outliving it.

Network-gated. Never runs in the offline suite. Budget ~12 minutes: SEC is
rate-limited to a few filings a second and Finnhub to 60 calls a minute.
"""
import os
import time
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    SKIP_NETWORK, reason="live SEC / Yahoo / Finnhub reconciliation")


# Structural variety, not familiarity. Each group reaches a code path the
# megacaps never touch: a lessor tags revenue under ASC 842, a bank under fee
# income, an ADR reports ordinary shares against an ADS price, an UPREIT
# carries operating-partnership units outside its share count.
BASKET = {
    "megacap":     ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA"],
    "mid":         ["WSM", "DKS", "EXPD"],
    "small":       ["PLUG", "RIOT", "SAVA"],
    "reit":        ["O", "SPG", "PLD", "AMT"],
    "bank":        ["JPM", "BAC", "WFC", "GS"],
    "biotech":     ["BIIB", "MRNA", "REGN"],
    "retailer":    ["WMT", "COST", "TGT", "HD"],
    "recent_ipo":  ["ARM", "RDDT", "CART"],
    "multi_class": ["GOOGL", "META", "RDDT", "DKS"],
    "industrial":  ["F", "CAT", "GE"],
    "energy":      ["XOM", "CVX"],
    "foreign_adr": ["TSM", "ASML"],
}
ALL_TICKERS = sorted({t for group in BASKET.values() for t in group})

# Foreign private issuers file a 20-F, not a 10-K, and report interim results
# on 6-K, which carries no XBRL. Reading them on the domestic form finds
# nothing and reports it as a filer that does not disclose.
ANNUAL_FORM = {"TSM": "20-F", "ASML": "20-F", "ARM": "20-F"}

FINANCIALS = set(BASKET["bank"])

# ------------------------------------------------------------------ tolerances
#
# Each band sits above the widest drift measured among the pairs that genuinely
# agree and below the narrowest genuine disagreement, so there is daylight on
# both sides. Measured 2026-08-22 over this basket; the gap is quoted per band.

# Same period end, same line item, two feeds. Widest agreement 0.24%.
TOL_STATEMENT = 0.01
# Revenue carries restatement and rounding noise the others do not: widest
# agreement 1.14% (MRNA), narrowest disagreement 3.6% (RIOT).
TOL_REVENUE = 0.02
# Cover-page share count is as of a date inside the last quarter; the widest
# agreement was 1.82%, and a dropped share class is 15% or more.
TOL_SHARES = 0.02
# Share count is a filing-date snapshot against a live quote. Widest agreement
# 4.01% (PLD), narrowest disagreement 13.4%.
TOL_MARKET_CAP = 0.05
# market cap + debt - cash omits minority interest and preferred, which a
# proper enterprise value includes. That incompleteness, not vendor drift, is
# what puts AMT at 5.2%; the rest of the basket sits inside 3.2%. The band is
# 10% to cover the formula's own gap, and the failure it exists to catch is
# three orders of magnitude away. SPG straddles this band as its price moves
# and is adjudicated rather than accommodated -- widening to fit it would have
# cost the AMT-sized signal for nothing.
TOL_ENTERPRISE_VALUE = 0.10
# Both sides trailing twelve months. Widest agreement 1.55%.
TOL_TTM_REVENUE = 0.03
# Two revenue figures derivable from one Finnhub payload. Widest agreement
# 8.1% (RIOT); the bank contradictions are 115% and up.
TOL_VENDOR_INTERNAL = 0.10
# Our P/E against Yahoo's own trailing P/E. Widest agreement 16.5% (SPG, whose
# net income to common differs from the consolidated figure).
TOL_PE = 0.20

# A vendor period this far from the filing's period end is a different period.
PERIOD_MATCH_DAYS = 7


# =========================================================== adjudicated cases
#
# Every entry was resolved by reading the raw XBRL facts out of the filing --
# concept, label, statement and context dimensions -- not by preferring
# whichever source looked nicer. `wrong` names the source that is wrong.

ADJUDICATED: Dict[tuple, str] = {

    # ---- get_revenue_base: the ASC 606 element is tried before us-gaap:Revenues,
    # and for a lessor, a bank or a segment-reporting filer that element is a
    # fragment of revenue rather than revenue.
    ("GOOGL", "revenue_annual"): (
        "wrong=get_revenue_base. Every RevenueFromContractWithCustomer fact in "
        "Alphabet's FY2025 10-K sits on the segment disclosure; the largest is "
        "Google Services at 342.721bn. Consolidated total revenues are tagged "
        "us-gaap:Revenues = 402.836bn on CONSOLIDATEDSTATEMENTSOFINCOME. "
        "Understated by 60.1bn."),
    ("AMT", "revenue_annual"): (
        "wrong=get_revenue_base. AMT labels the ASC 606 element 'Total non-lease "
        "revenue' = 0.9359bn. Tower rents are lease income under ASC 842. Total "
        "operating revenues (us-gaap:Revenues) = 10.6446bn. The tool returns 8.8% "
        "of AMT's revenue."),
    ("WFC", "revenue_annual"): (
        "wrong=get_revenue_base. WFC labels the ASC 606 element 'Fee income' = "
        "10.498bn on ConsolidatedStatementofIncome. 'Total revenue' "
        "(us-gaap:Revenues) on the same statement = 83.699bn. Understated by "
        "73.2bn."),
    ("GE", "revenue_annual"): (
        "wrong=get_revenue_base. The ASC 606 element is labelled 'Sales' = "
        "30.163bn; 'Total revenue' = 45.855bn in the undimensioned context. "
        "Understated by 15.7bn."),
    ("XOM", "revenue_annual"): (
        "wrong=get_revenue_base. XOM's only ASC 606 fact is on the segment "
        "disclosure under ProductOrServiceAxis = 226.909bn. Consolidated "
        "us-gaap:Revenues (undimensioned) = 332.238bn; Yahoo reports the "
        "323.905bn sales-only line. The tool returns one segment."),

    # ---- filter_annual_data takes the largest fact for the period. Its comment
    # asserts the consolidated total is always the largest positive value. It is
    # not: an operating-segment aggregate is struck before intersegment
    # eliminations, and a joint-venture disclosure is not the filer's revenue.
    ("CVX", "revenue_annual"): (
        "wrong=get_revenue_base. The undimensioned FY2025 fact is 184.432bn, "
        "which Yahoo matches. The tool returns 231.370bn from the context "
        "srt:ConsolidationItemsAxis=OperatingSegmentsMember -- segment revenue "
        "before eliminations. Overstated by 46.9bn."),
    ("CAT", "revenue_annual"): (
        "wrong=get_revenue_base. Consolidated 'Total sales and revenues' = "
        "67.589bn (context c-1, no dimensions). The tool returns 73.955bn from a "
        "dimensioned context. Overstated by 6.4bn."),
    ("SPG", "revenue_annual"): (
        "wrong=get_revenue_base. The tool returns 12.461bn from the context "
        "EquityMethodInvestmentNonconsolidatedInvesteeAxis="
        "spg:PlatformInvestmentsExcludingTrgAndKlepierre -- the revenue of "
        "Simon's unconsolidated joint ventures. Simon's own total revenue is "
        "6.3645bn. Overstated by 96%."),
    ("RIOT", "revenue_annual"): (
        "wrong=get_revenue_base. Consolidated total revenue 647.435m; the tool "
        "returns 670.718m from ConsolidationItemsAxis=OperatingSegmentsMember "
        "combined with ReportableSegmentAggregationBeforeOtherOperatingSegment."),

    # ---- get_accruals_quality
    ("AMT", "net_income"): (
        "wrong=get_accruals_quality. AMT tags us-gaap:NetIncomeLoss twice in the "
        "same undimensioned context c-1: 2.6285bn labelled 'Net income (loss)' on "
        "CONSOLIDATEDSTATEMENTSOFEQUITY and 2.5295bn labelled 'NET INCOME "
        "ATTRIBUTABLE TO AMERICAN TOWER CORPORATION COMMON STOCKHOLDERS' on the "
        "income statement. _by_period keeps whichever appears first in document "
        "order and returns the equity-statement figure. Yahoo and the income "
        "statement both say 2.5295bn."),
    ("SPG", "net_income"): (
        "definitional, both defensible, neither flagged. SPG does not tag "
        "us-gaap:NetIncomeLoss at all -- only NetIncomeLossAvailableToCommon"
        "StockholdersBasic and friends -- so the chain falls to us-gaap:ProfitLoss "
        "= 5.36412bn 'Consolidated Net Income', which includes the operating "
        "partnership's noncontrolling interests. Yahoo reports 4.6276bn to common. "
        "The 15.9% gap is SPG's NCI. concepts_used names ProfitLoss, but nothing "
        "warns that this is not the per-share numerator."),

    # ---- ADR unit mismatch
    ("TSM", "shares"): (
        "wrong=nobody; the unit is unstated. SEC dei reports 25,932,524,521 "
        "ordinary shares, Yahoo 5,186,474,013 ADS, and the ratio is 5.00003 -- "
        "TSM's 5:1 ADR ratio. Finnhub's profile also reports ordinary shares, so "
        "two of three sources agree and get_share_count_series is right in its "
        "own unit. It carries no note that the quote is per ADS."),
    ("TSM", "market_cap"): (
        "wrong=the caller, unavoidably. Ordinary shares times the ADS price gives "
        "10.86tn against a real 2.17tn -- the same factor of five. Nothing in the "
        "share-count result says the two cannot be multiplied."),
    ("TSM", "pe_ratio"): (
        "wrong=get_market_data. marketCap and currentPrice come back in USD while "
        "revenue, EBITDA, netIncomeToCommon, debt and cash come back in TWD, and "
        "pe_ratio divides one by the other: 0.98 against Yahoo's own trailingPE of "
        "31.24, off by the TWD/USD rate. The response carries no currency field. "
        "ASML shows the same defect at 63.68 against 59.97, inflated by EUR/USD."),
    ("TSM", "enterprise_value_internal"): (
        "wrong=get_market_data, same currency mix. enterpriseValue is TWD while "
        "marketCap is USD, so marketCap + debt - cash comes out negative."),
    ("ASML", "enterprise_value_internal"): (
        "wrong=yfinance, passed through unchecked. ASML's enterpriseValue field "
        "reads 37,631bn against a 677bn market cap and ~2bn of net debt. "
        "get_market_data divides it straight into revenue and publishes "
        "ev_revenue=1065x and ev_ebit=3330x with no plausibility check."),

    # ---- yfinance share count is the quoted class, not the company
    ("GOOGL", "shares"): (
        "wrong=yfinance sharesOutstanding. SEC dei totals Class A + B + C = "
        "12.23bn; Yahoo reports 5.867bn, the Class A count. Yahoo's own marketCap "
        "(4.217tn) reconciles with the SEC total at the quoted price, not with "
        "its own share count."),
    ("META", "shares"): (
        "wrong=yfinance sharesOutstanding. SEC dei Class A + B = 2.5475bn; Yahoo "
        "reports 2.2051bn."),
    ("RDDT", "shares"): (
        "wrong=yfinance sharesOutstanding. SEC dei Class A + B + C = 192.40m; "
        "Yahoo reports 146.10m."),
    ("DKS", "shares"): (
        "wrong=yfinance sharesOutstanding. SEC dei Class A + B = 89.50m; Yahoo "
        "reports 65.93m."),
    ("GOOGL", "market_cap_internal"): (
        "wrong=yfinance, internally. Its own sharesOutstanding times its own "
        "price is 2.02tn against its own marketCap of 4.22tn. Both fields ship in "
        "one get_market_data response and cannot both be right."),
    ("META", "market_cap_internal"): "wrong=yfinance, internally: 1.21tn vs 1.40tn.",
    ("RDDT", "market_cap_internal"): "wrong=yfinance, internally: 22.4bn vs 29.5bn.",
    ("DKS", "market_cap_internal"): "wrong=yfinance, internally: 12.1bn vs 16.4bn.",

    # ---- UPREIT: operating-partnership units
    ("SPG", "market_cap"): (
        "wrong=yfinance marketCap. SEC dei says 323,559,515 shares and Yahoo's own "
        "sharesOutstanding says 323,551,515 -- they agree to 0.002%. Yahoo's "
        "marketCap divided by its own price implies 379.6m shares, 56.1m more "
        "than either count: Simon Property Group LP units. A defensible "
        "fully-exchanged figure, but it contradicts the share count in the same "
        "response, and Finnhub's profile sides with the share count."),
    ("SPG", "market_cap_internal"): (
        "wrong=yfinance, internally: its own share count times its own price is "
        "70.63bn against its own marketCap of 82.87bn. Same OP-unit gap."),
    ("SPG", "enterprise_value_internal"): (
        "consequence of the SPG marketCap entry above, and the proof of it. "
        "Yahoo's enterpriseValue reconciles to SEC-shares x price + debt - cash "
        "within 0.78% (99.73bn against a reported 100.50bn) but sits 10.3% away "
        "from its own marketCap + debt - cash. So Yahoo builds EV on the "
        "common-only market cap while publishing an OP-unit-inclusive marketCap "
        "in the same payload -- a third source, internal to Yahoo, agreeing that "
        "the marketCap field is the outlier. Deliberately not a 'wrong=' entry: "
        "the gap here moves with the share price (measured at 9.7% and 10.3% a "
        "day apart), so the persistence guard belongs on market_cap and "
        "market_cap_internal, where the same defect shows a stable 14.8%."),

    # ---- Finnhub /stock/metric contradicts itself for banks
    ("JPM", "vendor_internal_revenue"): (
        "wrong=get_basic_financials. One payload yields two TTM revenues: "
        "marketCapitalization/psTTM = 332.6bn and revenuePerShareTTM x shares = "
        "138.6bn, a factor of 2.4. Yahoo says 186.3bn. Neither Finnhub figure is "
        "labelled gross or net of interest expense, and neither matches."),
    ("BAC", "vendor_internal_revenue"): (
        "wrong=get_basic_financials: 192.1bn vs 87.4bn from one payload; Yahoo "
        "says 113.9bn."),
    ("WFC", "vendor_internal_revenue"): (
        "wrong=get_basic_financials: 141.1bn vs 65.7bn from one payload; Yahoo "
        "says 83.0bn."),
    ("JPM", "revenue_ttm"): (
        "definitional and unlabelled. Finnhub's psTTM implies gross revenue "
        "including total interest income; Yahoo's totalRevenue is revenue net of "
        "interest expense. Both are real measures of a bank; nothing says which."),
    ("BAC", "revenue_ttm"): "definitional: gross vs net of interest expense, unlabelled.",
    ("WFC", "revenue_ttm"): "definitional: gross vs net of interest expense, unlabelled.",
    ("GS", "revenue_ttm"): "definitional: gross vs net of interest expense, unlabelled.",

    # ---- Yahoo's TTM revenue does not reconcile to Yahoo's own quarters.
    # Settled by summing the four most recent reported quarters: Finnhub's
    # implied TTM matches that sum to the dollar for all three, and Yahoo's
    # info.totalRevenue -- which get_market_data returns as revenue_ttm and
    # divides into enterprise value for ev_revenue -- does not match any
    # four-quarter window of Yahoo's own quarterly income statement.
    ("PLD", "revenue_ttm"): (
        "wrong=get_market_data (yahoo info.totalRevenue). Q2'26+Q1'26+Q4'25+Q3'25 "
        "= 9.1898bn, exactly Finnhub's implied TTM. Yahoo's info says 9.657bn, "
        "which is 467m more than its own quarterly statements add up to."),
    ("SPG", "revenue_ttm"): (
        "wrong=get_market_data (yahoo info.totalRevenue). Last four reported "
        "quarters = 6.6486bn, exactly Finnhub's implied TTM; Yahoo's info says "
        "6.941bn."),
    ("RIOT", "revenue_ttm"): (
        "wrong=get_market_data (yahoo info.totalRevenue). Last four reported "
        "quarters = 653.3m, exactly Finnhub's implied TTM; Yahoo's info says "
        "674.5m."),
}

# Entries claiming a specific source is wrong must keep disagreeing until that
# source is fixed. The rest -- definitional splits, unit mismatches nobody got
# wrong -- are recorded but not required to persist, because a vendor can
# change its mind about a definition without anything having been repaired.
_DEFECT_PREFIX = "wrong="


# ================================================================== harness

def _parse_date(value: Any) -> Optional[date]:
    """A period end, from either shape get_revenue_base returns.

    The two branches of that one function disagree on the format: the 10-K path
    returns '2025-12-31' and the 20-F path hands back the raw XBRL period key,
    'duration_2025-01-01_2025-12-31'. Recorded in docs/known_issues.md. Without
    this, every foreign private issuer reports not-comparable and three exact
    agreements -- ASML, TSM and ARM all match the vendor to the dollar -- go
    unmeasured, which is the reverse of what a coverage floor is for.
    """
    text = str(value)
    if text.startswith(("duration_", "instant_")):
        text = text.rsplit("_", 1)[-1]
    try:
        return date.fromisoformat(text[:10])
    except (TypeError, ValueError):
        return None


def _vendor_periods(envelope: Any) -> tuple:
    """(periods, source tag) out of a get_financial_statements envelope."""
    if not isinstance(envelope, dict):
        return [], None
    data = envelope.get("data", envelope)
    if not isinstance(data, dict):
        return [], None
    return data.get("periods") or [], data.get("_source")


def _at_period(periods: Sequence[dict], target: Any, field: str):
    """The vendor's value for `field` at the filing's period end, or (None, None).

    Matched on the date rather than on position. A fiscal year ending 2026-01-31
    and a calendar year ending 2025-12-31 are different periods; comparing them
    would manufacture a disagreement out of a calendar.
    """
    wanted = _parse_date(target)
    if wanted is None:
        return None, None
    best, best_gap = None, None
    for row in periods:
        end = _parse_date(row.get("period"))
        if end is None or row.get(field) is None:
            continue
        gap = abs((end - wanted).days)
        if gap <= PERIOD_MATCH_DAYS and (best_gap is None or gap < best_gap):
            best, best_gap = row, gap
    if best is None:
        return None, None
    return best.get(field), best.get("period")


def _sec_call(fn, ticker: str, form: str):
    """Run a SEC-backed tool, retrying through the rate limiter.

    `get_latest_filing` catches every exception, caches the None, and reports it
    to the caller as "No XBRL data available" -- so one HTTP 429 turns into a
    permanent, mislabelled "this filer has no XBRL" for the rest of the process
    (recorded in docs/known_issues.md). Retrying alone cannot recover from that;
    the poisoned entry has to be dropped first, which is why this reaches into
    the cache. Remove the eviction once the tool distinguishes a rate limit from
    an absent filing.
    """
    from tools.web_search_server import sec_utils

    result = None
    for attempt in range(4):
        result = fn()
        if not isinstance(result, dict):
            return result
        if result.get("success"):
            return result
        error = str(result.get("error", ""))
        transient = ("Too Many Requests" in error or "429" in error
                     or "No XBRL data available" in error)
        if not transient:
            return result
        sec_utils._filing_cache_lru.pop((ticker.upper(), form), None)
        # Exponential, because SEC's block is measured in tens of seconds
        # rather than one. A linear backoff spent 30s and still came back
        # rate-limited; this spends 75s at worst and only when already failing.
        time.sleep(5 * 2 ** attempt)
    return result


def _fetch_one(ticker: str, finnhub, loop) -> Dict[str, Any]:
    """Every source for one ticker. Exceptions are recorded, never swallowed.

    `loop` is shared across the whole basket on purpose. FinnhubClient opens its
    aiohttp session lazily and binds it to whatever loop is running at the time,
    so a per-ticker `asyncio.run` gets one working ticker and then "Event loop is
    closed" for the other thirty-six.
    """
    import json

    import yfinance as yf

    from tools.financial_modeling_engine.utils import get_data
    from tools.web_search_server.dilution import get_share_count_series
    from tools.web_search_server.earnings_quality import get_accruals_quality
    from tools.web_search_server.sec_utils import get_revenue_base

    form = ANNUAL_FORM.get(ticker, "10-K")
    interim = "20-F" if form == "20-F" else "10-Q"
    record: Dict[str, Any] = {"form": form, "errors": []}

    def _run(name, fn):
        try:
            record[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a raise IS the finding
            record[name] = None
            record["errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    _run("sec_revenue", lambda: _sec_call(
        lambda: get_revenue_base(ticker, form), ticker, form))
    _run("sec_accruals", lambda: _sec_call(
        lambda: get_accruals_quality(ticker, limit=2, form=form), ticker, form))
    _run("sec_shares", lambda: _sec_call(
        lambda: get_share_count_series(ticker, limit=3, form=interim),
        ticker, interim))
    _run("market_data", lambda: get_data(ticker))
    _run("yf_info", lambda: yf.Ticker(ticker).info)

    async def _finnhub():
        out = {}
        for name, coro in (
            ("fh_metric", finnhub.get_basic_financials(ticker)),
            ("fh_profile", finnhub.get_company_profile(ticker)),
            ("fh_income", finnhub.get_financial_statements(ticker, "ic", "annual")),
            ("fh_balance", finnhub.get_financial_statements(ticker, "bs", "annual")),
            ("fh_cashflow", finnhub.get_financial_statements(ticker, "cf", "annual")),
        ):
            out[name] = json.loads((await coro)[0].text)
        return out

    try:
        record.update(loop.run_until_complete(_finnhub()))
    except Exception as exc:  # noqa: BLE001
        record["errors"].append(f"finnhub: {type(exc).__name__}: {exc}")
    return record


# =================================================================== the checks

def _check(rows: List[dict], ticker: str, name: str, left_name: str, left,
           right_name: str, right, tolerance: float, note: str = "") -> None:
    if left is None or right is None:
        rows.append(dict(ticker=ticker, check=name, status="absent",
                         left_name=left_name, left=left, right_name=right_name,
                         right=right, drift=None,
                         note=note or "one source absent"))
        return
    if right == 0:
        rows.append(dict(ticker=ticker, check=name, status="not_comparable",
                         left_name=left_name, left=left, right_name=right_name,
                         right=right, drift=None, note="zero denominator"))
        return
    drift = (left - right) / abs(right)
    rows.append(dict(ticker=ticker, check=name,
                     status="agree" if abs(drift) <= tolerance else "disagree",
                     left_name=left_name, left=left, right_name=right_name,
                     right=right, drift=drift, note=note))


def _not_comparable(rows: List[dict], ticker: str, name: str, reason: str,
                    left=None, right=None) -> None:
    rows.append(dict(ticker=ticker, check=name, status="not_comparable",
                     left_name="", left=left, right_name="", right=right,
                     drift=None, note=reason))


def _reconcile(ticker: str, record: Dict[str, Any]) -> List[dict]:
    rows: List[dict] = []
    revenue = record.get("sec_revenue") or {}
    accruals = record.get("sec_accruals") or {}
    shares = record.get("sec_shares") or {}
    market = record.get("market_data") or {}
    info = record.get("yf_info") or {}
    metric = ((record.get("fh_metric") or {}).get("data") or {}).get("metric") or {}
    profile = (record.get("fh_profile") or {}).get("data") or {}
    income, _ = _vendor_periods(record.get("fh_income"))
    balance, _ = _vendor_periods(record.get("fh_balance"))
    cashflow, _ = _vendor_periods(record.get("fh_cashflow"))

    sec_currency = revenue.get("currency") if revenue.get("success") else None
    vendor_currency = info.get("financialCurrency")
    quote_currency = info.get("currency")
    currency_split = bool(sec_currency and vendor_currency
                          and sec_currency != vendor_currency)
    foreign_books = bool(vendor_currency and quote_currency
                         and vendor_currency != quote_currency)

    price = market.get("currentPrice")
    market_cap = market.get("marketCap")

    # --- 1. annual revenue: SEC XBRL against the vendor income statement
    if not revenue.get("success"):
        _not_comparable(rows, ticker, "revenue_annual",
                        f"sec: {str(revenue.get('error'))[:100]}")
    elif currency_split:
        _not_comparable(rows, ticker, "revenue_annual",
                        f"currency {sec_currency} vs {vendor_currency}")
    else:
        value, matched = _at_period(income, revenue.get("period_end"), "revenue")
        if value is None:
            _not_comparable(
                rows, ticker, "revenue_annual",
                f"no vendor period within {PERIOD_MATCH_DAYS}d of "
                f"{revenue.get('period_end')}")
        else:
            _check(rows, ticker, "revenue_annual", "sec_xbrl",
                   revenue["revenue_base"], "vendor_income_stmt", value,
                   TOL_REVENUE,
                   f"{revenue.get('period_end')} vs {matched}; "
                   f"{revenue.get('concept_used')}")

    # --- 2. trailing-twelve-month revenue: Yahoo against Finnhub
    # Both sides are TTM, so this one is period-safe by construction.
    finnhub_cap = metric.get("marketCapitalization")
    price_to_sales = metric.get("psTTM")
    finnhub_revenue = (finnhub_cap * 1e6 / price_to_sales
                       if finnhub_cap and price_to_sales else None)
    if foreign_books:
        _not_comparable(rows, ticker, "revenue_ttm",
                        f"yahoo books in {vendor_currency}, quote in "
                        f"{quote_currency}")
    else:
        _check(rows, ticker, "revenue_ttm", "market_data",
               market.get("revenue_ttm"), "finnhub_metric", finnhub_revenue,
               TOL_TTM_REVENUE, "marketCap/psTTM")

    # --- 3. Finnhub against itself: two TTM revenues out of one payload
    per_share = metric.get("revenuePerShareTTM")
    profile_shares = profile.get("shareOutstanding")
    _check(rows, ticker, "vendor_internal_revenue", "cap/psTTM", finnhub_revenue,
           "revPerShare*shares",
           per_share * profile_shares * 1e6 if per_share and profile_shares else None,
           TOL_VENDOR_INTERNAL, "same Finnhub response, two derivations")

    # --- 4/5. share count: SEC cover page against each vendor
    sec_shares = shares.get("latest_total") if shares.get("success") else None
    classes = shares.get("classes_found") or []
    _check(rows, ticker, "shares", "sec_dei", sec_shares,
           "yfinance", market.get("sharesOutstanding"), TOL_SHARES,
           f"classes={classes}")
    _check(rows, ticker, "shares_finnhub", "sec_dei", sec_shares,
           "finnhub_profile",
           profile_shares * 1e6 if profile_shares else None, TOL_SHARES,
           f"classes={classes}")

    # --- 6/7. market cap. Price times shares is the check that caught Biogen:
    # a shape check passes on a total missing an entire share class, this does
    # not. `market_cap_internal` asks the narrower question of whether one
    # get_market_data response agrees with itself.
    _check(rows, ticker, "market_cap", "sec_shares*price",
           sec_shares * price if sec_shares and price else None,
           "yfinance_market_cap", market_cap, TOL_MARKET_CAP, "")
    _check(rows, ticker, "market_cap_internal", "yf_shares*price",
           market.get("sharesOutstanding") * price
           if market.get("sharesOutstanding") and price else None,
           "yfinance_market_cap", market_cap, TOL_MARKET_CAP, "")

    # --- 8. enterprise value against its own components. Excluded for banks:
    # Yahoo's totalDebt and totalCash are not a bank's net debt in any sense
    # that makes EV reconcile.
    if ticker in FINANCIALS:
        _not_comparable(rows, ticker, "enterprise_value_internal",
                        "bank: totalDebt/totalCash are not net debt")
    else:
        debt, cash = market.get("totalDebt"), market.get("cash")
        _check(rows, ticker, "enterprise_value_internal", "yf_enterprise_value",
               market.get("enterpriseValue"), "cap+debt-cash",
               market_cap + (debt or 0) - (cash or 0) if market_cap else None,
               TOL_ENTERPRISE_VALUE, "")

    # --- 9/10/11. the accruals inputs, period-joined
    latest = (accruals.get("latest") or {}) if accruals.get("success") else {}
    period_end = latest.get("period_end")
    for name, sec_field, vendor_rows, vendor_field in (
        ("net_income", "net_income", income, "netIncome"),
        ("operating_cash_flow", "operating_cash_flow", cashflow, "operatingCashFlow"),
        ("total_assets", "total_assets", balance, "totalAssets"),
    ):
        sec_value = latest.get(sec_field)
        if sec_value is None:
            _not_comparable(
                rows, ticker, name,
                f"sec: {str(accruals.get('error'))[:100]}"
                if not accruals.get("success") else "concept not tagged")
            continue
        if currency_split:
            _not_comparable(rows, ticker, name,
                            f"currency {sec_currency} vs {vendor_currency}")
            continue
        value, matched = _at_period(vendor_rows, period_end, vendor_field)
        if value is None:
            _not_comparable(rows, ticker, name,
                            f"no vendor period within {PERIOD_MATCH_DAYS}d of "
                            f"{period_end}")
            continue
        _check(rows, ticker, name, "sec_xbrl", sec_value, "vendor", value,
               TOL_STATEMENT, f"{period_end} vs {matched}")

    # --- 12. our P/E against Yahoo's own. Same feed, same instant, so any gap
    # is arithmetic we did -- which is how the currency mix surfaces.
    ours, theirs = market.get("pe_ratio"), info.get("trailingPE")
    _check(rows, ticker, "pe_ratio", "market_data", ours,
           "yfinance_trailing_pe", theirs, TOL_PE,
           f"books {vendor_currency} / quote {quote_currency}"
           if ours is not None and theirs is not None
           else ("no market cap or no earnings, so no P/E on our side"
                 if ours is None else "yahoo publishes no trailing P/E"))

    return rows


# ================================================================== fixtures

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@pytest.fixture(scope="module")
def reconciliation():
    """Fetch the basket from every source once and reconcile it."""
    import asyncio

    from tools.news_agregator.finnhub_server import FinnhubServer

    finnhub = FinnhubServer()
    loop = asyncio.new_event_loop()
    rows: List[dict] = []
    errors: Dict[str, List[str]] = {}
    records: Dict[str, dict] = {}
    try:
        for ticker in ALL_TICKERS:
            record = _fetch_one(ticker, finnhub, loop)
            records[ticker] = record
            if record["errors"]:
                errors[ticker] = record["errors"]
            rows.extend(_reconcile(ticker, record))
            # SEC fair access is a few requests a second and each tool call
            # costs several. Without this the walk earns an HTTP 429 around the
            # twentieth ticker and the rest of the basket reads as uncovered.
            time.sleep(1.0)
    finally:
        loop.run_until_complete(finnhub.client.close())
        loop.close()
    return {"rows": rows, "errors": errors, "records": records}


def _by_check(rows: List[dict], name: str) -> List[dict]:
    return [row for row in rows if row["check"] == name]


def _unexpected(rows: List[dict], name: str) -> List[str]:
    """Disagreements in this check that ADJUDICATED does not already account for."""
    out = []
    for row in _by_check(rows, name):
        if row["status"] != "disagree":
            continue
        if (row["ticker"], name) in ADJUDICATED:
            continue
        out.append(
            f"{row['ticker']}: {row['left_name']}={row['left']:,.0f} vs "
            f"{row['right_name']}={row['right']:,.0f} "
            f"({row['drift'] * 100:+.1f}%) {row['note']}")
    return out


def _assert_reconciles(rows: List[dict], name: str, minimum: int) -> None:
    compared = [r for r in _by_check(rows, name)
                if r["status"] in ("agree", "disagree")]
    assert len(compared) >= minimum, (
        f"{name}: only {len(compared)} of {len(ALL_TICKERS)} tickers could be "
        f"compared, below the floor of {minimum}. A check that silently stops "
        f"comparing anything passes forever.")
    failures = _unexpected(rows, name)
    assert not failures, (
        f"{name}: sources disagree beyond tolerance and the disagreement is not "
        f"in ADJUDICATED. Read the filing, decide which source is right, then "
        f"add an entry -- do not widen the tolerance:\n"
        + "\n".join(f"  {f}" for f in failures))


# ==================================================================== tests

def test_no_source_raises_on_any_ticker(reconciliation):
    """A tool must return a not-covered result, never propagate an exception.

    An exception is the one outcome the caller cannot reason about, and
    get_revenue_base has a live instance of it: the 20-F branch calls
    _foreign_revenue_base outside its own try block.
    """
    errors = reconciliation["errors"]
    assert not errors, ("sources raised instead of returning a result:\n" +
                        "\n".join(f"  {t}: {e}" for t, e in errors.items()))


def test_total_assets_agrees_with_the_vendor(reconciliation):
    """The standing guard on the prefix-match bug.

    `by_concept` matches by prefix, so querying us-gaap:Assets also returned
    us-gaap:AssetsCurrent, which shares the balance-sheet context and survived
    every dimension filter. MSFT's total assets read 207.7bn against 758.4bn --
    a 27% figure, plausible enough to go unnoticed, and the denominator of the
    accrual ratio. No test of the extractor could see it; a second feed can.
    """
    _assert_reconciles(reconciliation["rows"], "total_assets", minimum=25)


def test_operating_cash_flow_agrees_with_the_vendor(reconciliation):
    _assert_reconciles(reconciliation["rows"], "operating_cash_flow", minimum=25)


def test_net_income_agrees_with_the_vendor(reconciliation):
    """The standing guard on the Ford sign error.

    Ford's FY2025 10-K tags us-gaap:ProfitLoss and not us-gaap:NetIncomeLoss.
    Stopping at the first concept that answered found NetIncomeLoss in the
    FY2024 filing and reported +5.9bn for a year Ford lost 8.2bn.
    """
    _assert_reconciles(reconciliation["rows"], "net_income", minimum=25)


def test_annual_revenue_agrees_with_the_vendor(reconciliation):
    """Nine of these are open defects in get_revenue_base -- see ADJUDICATED.

    Two independent causes, both in sec_utils: the ASC 606 element is tried
    ahead of us-gaap:Revenues even where it covers a fragment of revenue, and
    filter_annual_data then takes the largest fact for the period on the stated
    assumption that the consolidated total is always the largest positive value.
    """
    _assert_reconciles(reconciliation["rows"], "revenue_annual", minimum=20)


def test_ttm_revenue_agrees_between_vendors(reconciliation):
    _assert_reconciles(reconciliation["rows"], "revenue_ttm", minimum=20)


def test_finnhub_agrees_with_itself_on_revenue(reconciliation):
    """One payload must not yield two TTM revenues 2.4x apart.

    marketCapitalization/psTTM and revenuePerShareTTM x shareOutstanding are the
    same quantity by construction. Where they diverge, at least one of the
    metrics in that response is unusable -- and nothing in the response says so.
    """
    _assert_reconciles(reconciliation["rows"], "vendor_internal_revenue",
                       minimum=25)


def test_share_count_agrees_with_yfinance(reconciliation):
    """Where the two differ it is Yahoo counting one class, not us dropping one.

    The direction matters: SEC dei sums every class the filer tags, Yahoo
    reports the quoted class. Every entry in ADJUDICATED for this check is
    Yahoo's error, confirmed by Yahoo's own market cap reconciling with the SEC
    total rather than with its own share count.
    """
    _assert_reconciles(reconciliation["rows"], "shares", minimum=25)


def test_share_count_agrees_with_finnhub(reconciliation):
    """The tie-break. Finnhub's profile is a third, independent count.

    It agrees with SEC dei everywhere in the basket including the multi-class
    filers, which is what settles those cases against yfinance.
    """
    _assert_reconciles(reconciliation["rows"], "shares_finnhub", minimum=25)


def test_market_cap_reconciles_with_the_share_count(reconciliation):
    """Price times shares against reported market cap -- the Biogen check."""
    _assert_reconciles(reconciliation["rows"], "market_cap", minimum=25)


def test_market_data_agrees_with_itself_on_market_cap(reconciliation):
    """marketCap and sharesOutstanding ship in one response and must agree.

    They do not for any multi-class filer, so a caller doing per-share work on
    that response gets an answer that is wrong by the size of the classes Yahoo
    left out.
    """
    _assert_reconciles(reconciliation["rows"], "market_cap_internal", minimum=25)


def test_enterprise_value_reconciles_with_its_components(reconciliation):
    """EV against market cap plus debt minus cash, from the same response."""
    _assert_reconciles(reconciliation["rows"], "enterprise_value_internal",
                       minimum=20)


def test_pe_ratio_agrees_with_yahoos_own(reconciliation):
    """Our P/E against Yahoo's trailingPE, from the same fetch.

    Any gap is arithmetic we did on top of Yahoo's fields. It is how the
    currency mix surfaces: for a foreign filer marketCap is in the quote
    currency and netIncomeToCommon in the reporting currency, and get_market_data
    divides them without converting or labelling either.
    """
    _assert_reconciles(reconciliation["rows"], "pe_ratio", minimum=25)


def test_adjudicated_entries_still_disagree(reconciliation):
    """An entry that starts agreeing has been fixed and must be deleted.

    Without this the file rots into a list of exemptions that outlive the bugs
    they describe, and a real regression hides behind a stale entry.
    """
    status = {(r["ticker"], r["check"]): r["status"]
              for r in reconciliation["rows"]}
    resolved = [f"{t}/{c}: now {status[(t, c)]}"
                for (t, c), verdict in ADJUDICATED.items()
                if verdict.startswith(_DEFECT_PREFIX)
                and status.get((t, c)) == "agree"]
    assert not resolved, (
        "these sources now agree, so the ADJUDICATED entry is stale and should "
        "be removed along with any fix note in docs/known_issues.md:\n"
        + "\n".join(f"  {r}" for r in resolved))


def test_vendor_statements_declare_their_true_source(reconciliation):
    """Finnhub's /stock/financials is paywalled, so 'Finnhub' here is Yahoo.

    get_financial_statements falls back to yfinance on a 403. That is a real
    fallback, but it collapses two of the three feeds into one, and every
    statement comparison in this file is therefore SEC XBRL against Yahoo rather
    than against Finnhub. The fallback tags itself; this asserts the tag is
    there so nobody reads the table as a three-way agreement it is not.
    """
    untagged = []
    for ticker, record in reconciliation["records"].items():
        for key in ("fh_income", "fh_balance", "fh_cashflow"):
            periods, source = _vendor_periods(record.get(key))
            if periods and source is None:
                untagged.append(f"{ticker}/{key}")
    assert not untagged or len(untagged) == 3 * len(ALL_TICKERS), (
        "some statement responses are tagged with their source and some are "
        "not, so a caller cannot tell which feed answered:\n"
        f"  untagged: {untagged[:12]}")


def test_reconciliation_table_is_recorded(reconciliation, capsys):
    """Print the table. A number nobody prints is a number nobody checks."""
    rows = reconciliation["rows"]
    tally: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        tally[row["check"]][row["status"]] += 1

    lines = ["", "cross-source reconciliation "
                 f"({len(ALL_TICKERS)} tickers, {len(rows)} comparisons)", ""]
    lines.append(f"  {'check':26}{'agree':>7}{'disagree':>10}"
                 f"{'not_cmp':>9}{'absent':>8}")
    for name, counts in tally.items():
        lines.append(f"  {name:26}{counts['agree']:7d}{counts['disagree']:10d}"
                     f"{counts['not_comparable']:9d}{counts['absent']:8d}")

    disagreements = sorted((r for r in rows if r["status"] == "disagree"),
                           key=lambda r: -abs(r["drift"]))
    lines.append("")
    lines.append(f"  {len(disagreements)} material disagreements:")
    for row in disagreements:
        verdict = ADJUDICATED.get((row["ticker"], row["check"]), "UNADJUDICATED")
        lines.append(f"    {row['ticker']:6}{row['check']:26}"
                     f"{row['drift'] * 100:+9.1f}%  {verdict.split('.')[0]}")

    lines.append("")
    lines.append("  not comparable / absent:")
    for row in rows:
        if row["status"] in ("not_comparable", "absent"):
            lines.append(f"    {row['ticker']:6}{row['check']:26}{row['note']}")

    with capsys.disabled():
        print("\n".join(lines))

    for name in ("total_assets", "operating_cash_flow", "net_income"):
        compared = sum(tally[name][s] for s in ("agree", "disagree"))
        assert compared / len(ALL_TICKERS) > 0.60, (
            f"{name} could be reconciled for only {compared}/{len(ALL_TICKERS)} "
            f"tickers; below that the check stops being evidence of anything")
