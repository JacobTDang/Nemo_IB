"""Debt maturity schedule.

`calculate_credit_profile` could report that leverage was 3x but not when the
principal came due, and those describe different companies. A wall inside twelve
months is a refinancing problem; the same leverage maturing in 2031 is not.

Coverage is genuinely partial and this module says so. Measured live against
EDGAR: MSFT, T, and AAPL tag all six buckets, while Ford and PLUG tag none —
and Ford is among the largest debt issuers in the market, so its silence is a
tagging choice rather than an absence of debt. Returning an empty schedule for
those filers would read as "nothing comes due", which is the most dangerous
possible answer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .foreign_issuer import form_mismatch_note
from .sec_series import NotCovered, fetch_concept_series

_BASE = "us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipal"

# Filers use either the fixed-year family or the rolling-year family. Each
# bucket lists its alternatives in the order they are tried.
MATURITY_CONCEPTS: Dict[str, tuple] = {
    "year_1": (f"{_BASE}InNextTwelveMonths", f"{_BASE}InRollingYearOne"),
    "year_2": (f"{_BASE}InYearTwo", f"{_BASE}InRollingYearTwo"),
    "year_3": (f"{_BASE}InYearThree", f"{_BASE}InRollingYearThree"),
    "year_4": (f"{_BASE}InYearFour", f"{_BASE}InRollingYearFour"),
    "year_5": (f"{_BASE}InYearFive", f"{_BASE}InRollingYearFive"),
    "after_year_5": (f"{_BASE}AfterYearFive",
                     f"{_BASE}DueAfterRollingYearFive"),
}


def _bucket_value(ticker: str, concepts: tuple, form: str) -> Optional[float]:
    """First covered concept's consolidated value, or None if none is tagged.

    None means "not tagged". It is deliberately distinct from 0.0, which means
    the filer disclosed that nothing matures in that window — MSFT genuinely
    reports zero in years two and four.
    """
    for concept in concepts:
        try:
            points = fetch_concept_series(ticker, concept, form=form, limit=1)
        except NotCovered:
            continue
        except Exception:  # noqa: BLE001 - try the next concept
            continue
        for point in points:
            fact = point.latest_undimensioned()
            if fact is not None:
                return fact.value
    return None


def get_debt_maturity_schedule(ticker: str,
                               form: str = "10-K") -> Dict[str, Any]:
    """Principal coming due by year, from the long-term debt footnote.

    `coverage` is "full" when all six buckets are tagged, "partial" when some
    are, and "not_covered" when none are. A partial schedule is still useful;
    a silently partial one is not, so the count is always reported.

    Never synthesises a schedule from total debt. If the filer did not disclose
    the split in tagged form, that is the answer.
    """
    by_year: Dict[str, Optional[float]] = {}
    tried: List[str] = []

    for bucket, concepts in MATURITY_CONCEPTS.items():
        tried.extend(concepts)
        by_year[bucket] = _bucket_value(ticker, concepts, form)

    found = [v for v in by_year.values() if v is not None]
    buckets_found = len(found)

    if buckets_found == 0:
        # "TSM does not tag long-term debt maturities in its 10-K" was the
        # answer here before this guard. TSMC has no 10-K, and its 20-F
        # carries a full maturity ladder.
        mismatch = form_mismatch_note(ticker, form)
        return {
            "ticker": ticker,
            "success": False,
            "wrong_form": bool(mismatch),
            "coverage": "not_covered",
            "error": mismatch or (
                      f"{ticker} does not tag long-term debt maturities in "
                      f"its {form}. This means the split was not disclosed in "
                      f"XBRL, not that no debt matures."),
            "by_year": by_year,
            "total": None,
            "buckets_found": 0,
            "concepts_tried": tried,
            "pct_due_within_one_year": None,
        }

    total = float(sum(found))
    near_term = by_year.get("year_1")
    pct_near = (near_term / total * 100.0) if (near_term is not None and total) else None

    return {
        "ticker": ticker,
        "success": True,
        "coverage": "full" if buckets_found == len(MATURITY_CONCEPTS) else "partial",
        "by_year": by_year,
        "total": total,
        "buckets_found": buckets_found,
        "buckets_expected": len(MATURITY_CONCEPTS),
        "pct_due_within_one_year": pct_near,
        "concepts_tried": tried,
    }
