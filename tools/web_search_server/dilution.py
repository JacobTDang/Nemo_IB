"""Share count, dilution, and shelf activity.

Dilution is the blind spot this module closes. Share count previously existed
only as a point-in-time snapshot from yfinance, so a company could grow its
denominator steadily and every per-share metric would drift in the flattering
direction with nothing to show it.

Two things make this harder than reading one number:

1. **Multi-class filers.** GOOGL tags Class A, B, and C as separate facts with
   identical labels. Taking the first yields 5.868bn against a true 12.23bn.
   Every result therefore reports its per-class breakdown alongside the total,
   so a caller can see which classes were found rather than trusting a bare sum.

2. **Company-specific member tags.** Alphabet's Class C is
   `goog:CapitalClassCMember`, not `us-gaap:CommonClassCMember`. Labels are
   derived from whatever the filer used; a whitelist of standard members would
   silently drop the class.

Shelf activity is the other half. A share count that rose is history; an
effective S-3 with 424B5 takedowns is the mechanism, and it tells you the
dilution is ongoing rather than finished.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from edgar import Company

from .foreign_issuer import form_mismatch_note
from .sec_series import NotCovered, _require_identity, fetch_concept_series

SHARES_CONCEPT = "dei:EntityCommonStockSharesOutstanding"
CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"

# Below this, a change is rounding and share-count noise from option exercises
# rather than a signal worth naming.
_FLAT_THRESHOLD_PCT = 0.1


def _class_label(member: Optional[str]) -> str:
    """Turn an XBRL member tag into something a human reads.

    `us-gaap:CommonClassAMember` becomes "Class A" and
    `goog:CapitalClassCMember` becomes "Capital Class C". Derived from the tag
    rather than looked up, because filers invent their own members and a lookup
    table would drop the ones it had not seen.
    """
    if not member:
        return "Common"
    local = member.split(":", 1)[-1]
    if local.endswith("Member"):
        local = local[: -len("Member")]
    words = re.findall(r"[A-Z][a-z0-9.]*|[A-Z]+(?![a-z])", local) or [local]
    # "CommonClassA" reads better as "Class A" than "Common Class A".
    if len(words) > 1 and words[0] == "Common" and words[1] == "Class":
        words = words[1:]
    return " ".join(words)


def _failure(ticker: str, message: str,
             form: Optional[str] = None) -> Dict[str, Any]:
    """Every miss in this module funnels here, so the form guard sits here too.

    A foreign private issuer reports on 6-K, which carries no XBRL, so a share
    count walked over 10-Q filings finds nothing. "Not covered" would read as
    a filer that does not disclose its share count.
    """
    mismatch = form_mismatch_note(ticker, form) if form else None
    return {
        "ticker": ticker,
        "success": False,
        "wrong_form": bool(mismatch),
        "error": mismatch or message,
        "latest_total": None,
        "by_class": {},
        "classes_found": [],
        "total_series": [],
        "change_pct": None,
        "direction": "not_covered",
    }


def get_share_count_series(ticker: str, limit: int = 8,
                           form: str = "10-Q") -> Dict[str, Any]:
    """Shares outstanding across the last `limit` filings, newest first.

    Returns the per-class breakdown, the summed total per filing, and the change
    from oldest to newest in the window. A filer that does not tag the concept
    gets an explicit failure rather than a zero, because zero shares outstanding
    is a meaningful and very different claim.
    """
    try:
        points = fetch_concept_series(ticker, SHARES_CONCEPT, form=form, limit=limit)
    except NotCovered as exc:
        return _failure(ticker, f"share count not covered: {exc}", form)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return _failure(ticker, f"{type(exc).__name__}: {exc}", form)

    by_class: Dict[str, List[Dict[str, Any]]] = {}
    total_series: List[Dict[str, Any]] = []

    for point in points:
        for fact in point.facts:
            label = _class_label(fact.dimension_member(CLASS_AXIS))
            by_class.setdefault(label, []).append({
                "filing_date": point.filing_date,
                "period": fact.period,
                "shares": fact.value,
            })
        total_series.append({
            "filing_date": point.filing_date,
            "form": point.form,
            "total": point.total(),
        })

    latest_total = total_series[0]["total"] if total_series else None

    change_pct: Optional[float] = None
    direction = "insufficient_history"
    if len(total_series) > 1:
        oldest_total = total_series[-1]["total"]
        if oldest_total:
            change_pct = (latest_total - oldest_total) / oldest_total * 100.0
            if change_pct > _FLAT_THRESHOLD_PCT:
                direction = "dilution"
            elif change_pct < -_FLAT_THRESHOLD_PCT:
                direction = "buyback"
            else:
                direction = "flat"

    return {
        "ticker": ticker,
        "success": True,
        "latest_total": latest_total,
        "by_class": by_class,
        "classes_found": sorted(by_class.keys()),
        "total_series": total_series,
        "change_pct": change_pct,
        "direction": direction,
        "periods_examined": len(total_series),
    }


def get_shelf_activity(ticker: str, lookback_days: int = 730) -> Dict[str, Any]:
    """Shelf registrations and takedowns in the window.

    An S-3 is the authorisation to sell shares; a 424B5 is an actual sale off
    that shelf. Neither form had any coverage in this codebase, which meant the
    mechanism of dilution was invisible even when its effect was not.
    """
    try:
        _require_identity()
        company = Company(ticker)
        cutoff = date.today() - timedelta(days=lookback_days)

        def _recent(form_name: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            try:
                filings = company.get_filings(form=form_name)
            except Exception:
                return out
            for filing in filings:
                filed = getattr(filing, "filing_date", None)
                if filed is None:
                    continue
                filed_date = filed if isinstance(filed, date) else None
                if filed_date is None:
                    try:
                        filed_date = date.fromisoformat(str(filed))
                    except ValueError:
                        continue
                if filed_date < cutoff:
                    break  # EDGAR returns newest first
                out.append({
                    "form": form_name,
                    "filing_date": str(filed_date),
                    "accession": str(getattr(filing, "accession_no", "")),
                })
            return out

        registrations = _recent("S-3")
        takedowns = _recent("424B5")
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {
            "ticker": ticker,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "s3_registrations": [],
            "b5_takedowns": [],
        }

    return {
        "ticker": ticker,
        "success": True,
        "lookback_days": lookback_days,
        "s3_registrations": registrations,
        "b5_takedowns": takedowns,
        "has_active_shelf": bool(registrations),
        "takedown_count": len(takedowns),
    }
