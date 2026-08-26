"""Stock-based compensation.

The largest single line between GAAP earnings and the "adjusted" figures
companies prefer to be judged on, and the other engine of dilution alongside
shelf issuance. It had no coverage at all.

The hazard is a wrong number rather than a missing one. NVDA tags 51 separate
SBC facts in one 10-K, nearly all split by award type, and those are components
of the consolidated expense rather than additions to it. Anything that sums
facts reports several times the real figure. Selection therefore goes through
`FilingPoint.latest_undimensioned()`, which takes the consolidated value for
the longest period ending latest.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .foreign_issuer import form_mismatch_note, not_covered_reason
from .sec_series import NotCovered, fetch_concept_series
from .shared_filings import (
    Deadline,
    ToolTimeout,
    budget_seconds,
    concept_series,
    shared_filings,
)

# Filers tag SBC under several concepts. Ordered by how commonly each one
# carries the consolidated figure.
SBC_CONCEPTS = (
    "us-gaap:ShareBasedCompensation",
    "us-gaap:AllocatedShareBasedCompensationExpense",
    "us-gaap:ShareBasedCompensationArrangementByShareBasedPaymentAwardCompensationCost",
)

REVENUE_CONCEPTS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
)

OCF_CONCEPTS = (
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)


def _shared_filings(ticker: str, deadline: Deadline):
    """`shared_filings`, bound to this module's `fetch_concept_series`.

    Read here rather than passed in from the call site so the name stays a
    module global a test can replace -- which is what the walk checks before
    it stands down.
    """
    return shared_filings(ticker, deadline, fetch_concept_series)


def _concept_series(ticker: str, concept: str, form: str, limit: int) -> list:
    """One concept's series, reusing filings already parsed for this call.

    With no shared walk open this is `fetch_concept_series` verbatim, which is
    what keeps `_consolidated_by_filing` callable on its own -- several tests
    call it directly -- and keeps that name the seam they replace.
    """
    return concept_series(ticker, concept, form, limit, fetch_concept_series)


def _consolidated_by_filing(ticker: str, concepts: tuple, form: str,
                            limit: int) -> tuple:
    """The chain element reaching the most recent filing, as {filing_date: value}.

    Returns ({}, None) when no concept in the chain is covered. The caller
    decides whether that is fatal — it is for SBC itself, and merely means "no
    ratio" for a denominator.

    Every concept is evaluated rather than stopping at the first that returns
    anything, because filers change elements between filings and the abandoned
    element still answers from the older filing. NVDA tags
    `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` only in its
    FY2022 10-K and `us-gaap:Revenues` since; stopping at the first hit
    resolved revenue for 2022 alone and left `pct_of_revenue` None for every
    year after it. Freshness decides, and ties keep chain order so the more
    specific element still wins when a filer tags both.
    """
    best_values: Dict[str, float] = {}
    best_concept: Optional[str] = None
    best_end = ""
    for concept in concepts:
        try:
            points = _concept_series(ticker, concept, form, limit)
        # Only NotCovered is swallowed, and only to try the next concept.
        # A network failure or an unknown ticker propagates: reporting an
        # outage as "this filer does not disclose it" is the one answer
        # worse than an error.
        except NotCovered:
            continue

        values: Dict[str, float] = {}
        for point in points:
            fact = point.latest_undimensioned()
            if fact is not None:
                values[point.filing_date] = fact.value
        if not values:
            continue
        newest = max(values)
        if newest > best_end:
            best_values, best_concept, best_end = values, concept, newest
    return best_values, best_concept


def get_sbc_series(ticker: str, limit: int = 5,
                   form: str = "10-K") -> Dict[str, Any]:
    """Stock-based compensation over recent filings, newest first.

    Each row carries the raw expense plus its share of revenue and of operating
    cash flow. Ratios are None when the denominator could not be resolved —
    never zero, which would read as "no SBC burden".

    All three chains — eight concepts — are read inside one filing walk. They
    are the same `limit` 10-Ks, and parsing them once per concept cost GS 40
    parses of 5 filings and 66.7 seconds.
    """
    deadline = Deadline(budget_seconds(), f"get_sbc_series({ticker})")
    try:
        with _shared_filings(ticker, deadline):
            try:
                sbc_values, concept_used = _consolidated_by_filing(
                    ticker, SBC_CONCEPTS, form, limit)
            # The clock is not a fetch failure and must not be described as
            # one: it is handled below, where it can say so in its own words.
            except ToolTimeout:
                raise
            except Exception as exc:  # noqa: BLE001 - surface it, never mask it
                return {
                    "ticker": ticker,
                    "success": False,
                    "wrong_form": False,
                    "error": (f"fetching stock-based compensation concepts "
                              f"failed: {exc}"),
                    "series": [],
                    "concept_used": None,
                }

            if not sbc_values:
                mismatch = form_mismatch_note(ticker, form)
                return {
                    "ticker": ticker,
                    "success": False,
                    "wrong_form": bool(mismatch),
                    "error": not_covered_reason(
                        ticker, form,
                        f"stock-based compensation not covered: none of "
                        f"{list(SBC_CONCEPTS)} found in the last {limit} "
                        f"{form} filings."),
                    "series": [],
                    "concept_used": None,
                }

            # Inside the same walk: the denominators come from the filings
            # already parsed above, so they cost concept reads and no fetches.
            revenue_values, _ = _consolidated_by_filing(
                ticker, REVENUE_CONCEPTS, form, limit)
            ocf_values, _ = _consolidated_by_filing(
                ticker, OCF_CONCEPTS, form, limit)
    except ToolTimeout as exc:
        return {
            "ticker": ticker,
            "success": False,
            "timed_out": True,
            "wrong_form": False,
            "error": str(exc),
            "series": [],
            "concept_used": None,
        }
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        # The SBC chain above is guarded, but the revenue and operating-cash-
        # flow reads that build pct_of_revenue were not, so a failure there
        # left this function by raising. Every other tool here answers with a
        # dict naming the ticker and what went wrong; a bare exception reaches
        # the caller as a framework message with none of that, and cannot be
        # told apart from a filer that tags no stock compensation.
        return {
            "ticker": ticker,
            "success": False,
            "timed_out": False,
            "wrong_form": False,
            "error": (f"reading stock-compensation denominators for {ticker} "
                      f"failed: {type(exc).__name__}: {exc}"),
            "series": [],
            "concept_used": None,
        }

    series: List[Dict[str, Any]] = []
    for filing_date, value in sbc_values.items():
        revenue = revenue_values.get(filing_date)
        ocf = ocf_values.get(filing_date)
        series.append({
            "filing_date": filing_date,
            "sbc": value,
            "revenue": revenue,
            "operating_cash_flow": ocf,
            "pct_of_revenue": (value / revenue * 100.0) if revenue else None,
            "pct_of_ocf": (value / ocf * 100.0) if ocf else None,
        })

    series.sort(key=lambda row: row["filing_date"], reverse=True)

    latest = series[0]["sbc"] if series else None
    oldest = series[-1]["sbc"] if len(series) > 1 else None
    change_pct: Optional[float] = None
    if oldest:
        change_pct = (latest - oldest) / oldest * 100.0

    return {
        "ticker": ticker,
        "success": True,
        "concept_used": concept_used,
        "series": series,
        "latest_sbc": latest,
        "change_pct": change_pct,
        "periods_examined": len(series),
    }
