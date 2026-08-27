"""Contracted revenue, geographic exposure, and public float.

Three gaps that let a researcher be confidently wrong:

**Contracted revenue.** Remaining performance obligation is the strongest
forward number an enterprise filer publishes -- signed revenue not yet
recognised. It was only ever scraped from MD&A prose here, never read as a
figure. Deferred revenue rides along because ASC 606 split the same idea across
two concept families and filers use either.

**Geographic revenue.** Business segments were covered; geography was not, so
China and FX exposure did not appear anywhere.

**Public float.** Share count was covered but float was not, and the two differ
enormously for founder-controlled names. Float is what actually trades, and it
drives volatility and squeeze dynamics in a way total shares do not.

Dimension handling repeats the lesson from share classes: members mix standard
tags (`country:US`) with company-specific ones
(`nvda:ChinaIncludingHongKongMember`), so a whitelist silently drops
geographies. Whatever the filer used is what gets reported.
"""
from __future__ import annotations

import re
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

RPO_CONCEPTS = (
    "us-gaap:RevenueRemainingPerformanceObligation",
    "us-gaap:RevenueFromRemainingPerformanceObligation",
)

# ASC 606 renamed deferred revenue to contract liability. Filers use either,
# and MSFT tags only the newer form.
DEFERRED_CONCEPTS = (
    "us-gaap:ContractWithCustomerLiabilityCurrent",
    "us-gaap:ContractWithCustomerLiability",
    "us-gaap:DeferredRevenueCurrent",
    "us-gaap:DeferredRevenue",
)

# The IFRS members matter only for foreign private issuers, and they are
# tried last so nothing changes for a domestic filer. A 20-F filer splits
# revenue on `ifrs-full:GeographicalAreasAxis`, which the "Geograph" hint
# below already matches -- the concept chain was the only thing missing.
REVENUE_CONCEPTS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "ifrs-full:Revenue",
    "ifrs-full:RevenueFromContractsWithCustomers",
)

FLOAT_CONCEPT = "dei:EntityPublicFloat"

GEO_AXIS_HINTS = ("Geograph", "Country")

# ISO country codes the SEC uses in `country:XX` members. Only the ones likely
# to matter for an equity thesis are named; anything else falls back to the raw
# code, which is still readable.
_COUNTRY_NAMES = {
    "US": "United States", "CN": "China", "TW": "Taiwan", "JP": "Japan",
    "KR": "South Korea", "DE": "Germany", "GB": "United Kingdom",
    "FR": "France", "IN": "India", "CA": "Canada", "SG": "Singapore",
    "IE": "Ireland", "NL": "Netherlands", "MX": "Mexico", "BR": "Brazil",
    "AU": "Australia", "CH": "Switzerland", "IL": "Israel", "VN": "Vietnam",
    "MY": "Malaysia", "HK": "Hong Kong",
}


def _region_label(member: str) -> str:
    """Human label for a geographic member tag.

    `country:TW` becomes Taiwan. `nvda:ChinaIncludingHongKongMember` becomes
    "China Including Hong Kong" -- derived from the tag rather than looked up,
    because filers invent their own regions and a lookup table drops the ones
    it has not seen.
    """
    if member.startswith("country:"):
        code = member.split(":", 1)[1]
        return _COUNTRY_NAMES.get(code, code)
    local = member.split(":", 1)[-1]
    if local.endswith("Member"):
        local = local[: -len("Member")]
    words = re.findall(r"[A-Z][a-z0-9]*|[A-Z]+(?![a-z])", local) or [local]
    return " ".join(words)


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
    what keeps `_series_for` callable on its own -- the freshness tests call
    it directly -- and keeps that name the seam they replace.
    """
    return concept_series(ticker, concept, form, limit, fetch_concept_series)


def _series_for(ticker: str, concepts: tuple, form: str, limit: int):
    """The chain element reaching the most recent filing, newest first.

    Every concept in the chain is evaluated rather than stopping at the first
    that returns anything, because filers change elements between filings and
    the abandoned element still answers from the older filing. NVDA tags
    `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` only in its
    FY2022 10-K and `us-gaap:Revenues` since; stopping at the first hit
    resolved revenue for 2022 alone. Freshness decides, and ties keep chain
    order so the more specific element still wins when a filer tags both.
    """
    best_rows: List[Dict[str, Any]] = []
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
        rows = []
        for point in points:
            fact = point.latest_undimensioned()
            if fact is not None:
                rows.append({"filing_date": point.filing_date,
                             "period": fact.period, "value": fact.value})
        if not rows:
            continue
        newest = max(r["filing_date"] for r in rows)
        if newest > best_end:
            best_rows, best_concept, best_end = rows, concept, newest
    return best_rows, best_concept


def get_contracted_revenue(ticker: str, limit: int = 3,
                           form: str = "10-K") -> Dict[str, Any]:
    """Revenue already contracted but not yet recognised.

    `rpo` is the total remaining performance obligation -- the consolidated
    figure, not a sum of the customer-type and timing breakdowns filers publish
    alongside it. `deferred_revenue` is the balance-sheet counterpart.

    A filer reporting only deferred revenue is a partial answer, not a failure;
    most non-enterprise filers never disclose RPO.

    Both chains — six concepts — are read inside one filing walk. They are the
    same `limit` 10-Ks, and parsing them once per concept cost GS 18 parses of
    3 filings and 40.2 seconds.
    """
    deadline = Deadline(budget_seconds(), f"get_contracted_revenue({ticker})")
    try:
        with _shared_filings(ticker, deadline):
            rpo, rpo_concept = _series_for(ticker, RPO_CONCEPTS, form, limit)
            deferred, deferred_concept = _series_for(
                ticker, DEFERRED_CONCEPTS, form, limit)
    except ToolTimeout as exc:
        return {
            "ticker": ticker, "success": False, "timed_out": True,
            "wrong_form": False, "error": str(exc),
            "rpo": [], "deferred_revenue": [],
            "rpo_concept_used": None, "deferred_concept_used": None,
        }
    except Exception as exc:  # noqa: BLE001 - surface the failure, never mask it
        return {
            "ticker": ticker, "success": False, "wrong_form": False,
            "error": f"fetching contracted-revenue concepts failed: {exc}",
            "rpo": [], "deferred_revenue": [],
            "rpo_concept_used": None, "deferred_concept_used": None,
        }

    if not rpo and not deferred:
        mismatch = form_mismatch_note(ticker, form)
        return {
            "ticker": ticker, "success": False,
            "wrong_form": bool(mismatch),
            "error": not_covered_reason(
                ticker, form,
                f"contracted revenue not covered: neither {RPO_CONCEPTS[0]} "
                f"nor any deferred-revenue concept appears in the last "
                f"{limit} {form} filings."),
            "rpo": [], "deferred_revenue": [],
            "rpo_concept_used": None, "deferred_concept_used": None,
        }

    return {
        "ticker": ticker, "success": True,
        "rpo": rpo,
        "deferred_revenue": deferred,
        "rpo_concept_used": rpo_concept,
        "deferred_concept_used": deferred_concept,
        "note": ("RPO is contracted revenue not yet recognised, the strongest "
                 "forward figure a filer publishes. Absence of RPO is normal "
                 "outside enterprise and subscription businesses."),
    }


def get_geographic_revenue(ticker: str, limit: int = 1,
                           form: str = "10-K") -> Dict[str, Any]:
    """Revenue by geography, with each region's share of the disclosed total.

    A single 10-K carries several years of comparative figures per region, so
    one filing is usually enough for a trend.

    The chain is read inside one filing walk. Every element after the first
    was re-parsing the same 10-K to ask it a second question, which cost GS 5
    parses of one filing and 25.8 seconds.
    """
    deadline = Deadline(budget_seconds(), f"get_geographic_revenue({ticker})")
    try:
        with _shared_filings(ticker, deadline):
            answer = _geographic_chain(ticker, limit, form)
    except ToolTimeout as exc:
        return {
            "ticker": ticker, "success": False, "timed_out": True,
            "wrong_form": False, "error": str(exc),
            "by_region": [], "regions_found": [],
        }
    if answer is not None:
        return answer

    # "TSM does not disaggregate revenue by geography in its 10-K" was the
    # answer here. TSMC has no 10-K, and its 20-F splits revenue four ways.
    mismatch = form_mismatch_note(ticker, form)
    return {
        "ticker": ticker, "success": False,
        "wrong_form": bool(mismatch),
        "error": not_covered_reason(
            ticker, form,
            f"{ticker} does not disaggregate revenue by geography in its "
            f"{form}."),
        "by_region": [], "regions_found": [],
    }


def _geographic_chain(ticker: str, limit: int,
                      form: str) -> Optional[Dict[str, Any]]:
    """The first chain element that yields a geographic split, or None.

    Split out so the walk above wraps the fetching and nothing else. None
    means no element produced a breakdown; phrasing that as an answer needs
    the filing index, and that lookup has no business inside the walk.
    """
    for concept in REVENUE_CONCEPTS:
        try:
            points = _concept_series(ticker, concept, form, limit)
        # Only NotCovered is swallowed, and only to try the next concept.
        # A network failure or an unknown ticker propagates: reporting an
        # outage as "this filer does not disclose it" is the one answer
        # worse than an error.
        except NotCovered:
            continue
        # The clock says nothing about the filer, so it is not turned into a
        # message about one here. The caller reports it in its own words.
        except ToolTimeout:
            raise
        except Exception as exc:  # noqa: BLE001 - surface it, never mask it
            return {
                "ticker": ticker, "success": False, "wrong_form": False,
                "error": f"fetching geographic-revenue concepts failed: {exc}",
                "by_region": [], "regions_found": [],
            }

        by_region: Dict[str, List[Dict[str, Any]]] = {}
        consolidated: Optional[float] = None
        for point in points:
            total_fact = point.latest_undimensioned()
            if total_fact is not None and consolidated is None:
                consolidated = total_fact.value
            for fact in point.deduplicated():
                member = None
                for axis, value in fact.dimensions.items():
                    if any(hint in axis for hint in GEO_AXIS_HINTS):
                        member = value
                        break
                if member is None:
                    continue  # consolidated total, not a geography
                by_region.setdefault(_region_label(member), []).append(
                    {"filing_date": point.filing_date,
                     "period": fact.period, "value": fact.value})

        if not by_region:
            continue

        for rows in by_region.values():
            rows.sort(key=lambda r: r["period"], reverse=True)

        latest_total = sum(rows[0]["value"] for rows in by_region.values())

        # Members can nest, and IFRS filers nest aggressively: SAP tags
        # "EMEA", "EMEA excluding Germany" and "country of domicile" as three
        # separate members of the same axis, so their parent is counted twice
        # and every percentage comes out low. Detected by comparing the sum
        # against the consolidated fact already in hand rather than by trying
        # to recognise parent labels, which cannot be done from a tag name.
        nested = bool(consolidated) and latest_total > consolidated * 1.02
        denominator = consolidated if nested else latest_total

        regions = [
            {"region": region, "periods": rows,
             "pct_of_total": (rows[0]["value"] / denominator * 100.0)
                             if denominator else None}
            for region, rows in by_region.items()
        ]
        regions.sort(key=lambda r: r["periods"][0]["value"], reverse=True)

        note = ("Percentages are of the disclosed geographic total, which "
                "may differ from consolidated revenue when a filer groups "
                "part of it under an 'other' region.")
        if nested:
            note = (
                "This filer's geographic members OVERLAP: they sum to "
                f"{latest_total:,.0f} against consolidated revenue of "
                f"{consolidated:,.0f}, so at least one region is a parent of "
                "another (SAP tags EMEA, EMEA-excluding-Germany and Germany "
                "on the same axis). Percentages are of consolidated revenue "
                "and therefore do NOT sum to 100. Read individual regions, "
                "not the ranking, and do not add them together.")
        return {
            "ticker": ticker, "success": True, "concept_used": concept,
            "by_region": regions,
            "regions_found": [r["region"] for r in regions],
            "disclosed_total": latest_total,
            "consolidated_revenue": consolidated,
            "members_overlap": nested,
            "note": note,
        }
    return None


def get_public_float(ticker: str, form: str = "10-K") -> Dict[str, Any]:
    """Aggregate market value of shares held by non-affiliates.

    Distinct from shares outstanding, and the difference is the point: float
    excludes insider and affiliate holdings, so it is what actually trades.
    Reported on the cover page as of the filer's second-quarter close, so it
    lags -- the filing date is returned alongside it rather than implied.
    """
    try:
        rows, _ = _series_for(ticker, (FLOAT_CONCEPT,), form, 1)
    except Exception as exc:  # noqa: BLE001 - surface the failure, never mask it
        return {
            "ticker": ticker, "success": False, "wrong_form": False,
            "error": f"fetching {FLOAT_CONCEPT} failed: {exc}",
            "public_float": None, "filing_date": None,
        }
    if not rows:
        mismatch = form_mismatch_note(ticker, form)
        return {
            "ticker": ticker, "success": False,
            "wrong_form": bool(mismatch),
            "error": not_covered_reason(
                ticker, form,
                f"{ticker} does not tag {FLOAT_CONCEPT} in its {form}. "
                f"Smaller reporting companies sometimes omit it, and foreign "
                f"private issuers generally do not report float on a 20-F "
                f"cover page at all."),
            "public_float": None, "filing_date": None,
        }
    latest = rows[0]
    return {
        "ticker": ticker, "success": True,
        "public_float": latest["value"],
        "filing_date": latest["filing_date"],
        "as_of": latest["period"],
        "note": ("Market value of shares held by non-affiliates, measured at "
                 "the filer's second-quarter close. Compare against "
                 "get_share_count_series: a large gap means insiders hold much "
                 "of the register and less stock trades than the share count "
                 "suggests."),
    }
