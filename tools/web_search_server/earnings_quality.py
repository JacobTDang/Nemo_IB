"""Are these earnings real? Net income against operating cash flow.

A question the SEC layer could not answer:

**Accruals.** Net income rising while operating cash flow does not is the
classic pre-blowup signature -- earnings arriving as promises rather than cash.
Both numbers were extractable one at a time and nothing compared them.

Everything is period-joined on the fiscal period end date, so a balance-sheet
instant is matched to the income-statement duration that ends the same day.
Day counts come from the actual period span rather than a hardcoded 365, which
is what makes the ratios comparable when the filing is a 10-Q or the filer runs
a 52/53-week year.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sec_series import NotCovered, _period_rank, fetch_concept_series

NET_INCOME_CONCEPTS = (
    "us-gaap:NetIncomeLoss",
    "us-gaap:ProfitLoss",
)

# Filers with discontinued operations tag only the ContinuingOperations form.
OCF_CONCEPTS = (
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)

ASSETS_CONCEPTS = ("us-gaap:Assets",)

# Accrual ratio bands, applied to the latest period. Sloan's original work
# sorted the market into deciles; these are the round numbers that separate
# them well enough to act on, and the raw ratio is always returned alongside.
_MODERATE_ACCRUAL_PCT = 5.0


def _period_bounds(period: str) -> Tuple[str, int]:
    """(period end date, span in days) for a fact's period label.

    Instants report a span of 0. An unparseable label reports an empty end
    date, which callers drop rather than guess at.
    """
    end, days = _period_rank(period)
    return end, int(days)


def _by_period(points: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    """Consolidated value per period end, across every filing walked.

    Two selection rules, both learned the hard way:

    * Only undimensioned facts. A 10-K tags net income by segment and by share
      class alongside the consolidated figure, and those are components of the
      total rather than additions to it.
    * Within one filing, the longest span wins for a given end date. A 10-K
      carries both the fourth quarter and the full year ending the same day,
      and the quarter understates the year roughly fourfold.

    Filings arrive newest first, so the first value seen for a period end is
    the most recently filed -- the restated figure rather than the original.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for point in points:
        best: Dict[str, Tuple[int, float]] = {}
        for fact in point.undimensioned():
            end, days = _period_bounds(fact.period)
            if not end:
                continue
            current = best.get(end)
            if current is None or days > current[0]:
                best[end] = (days, fact.value)
        for end, (days, value) in best.items():
            if end not in out:
                out[end] = {"period_end": end, "period_days": days,
                            "value": value, "filing_date": point.filing_date}
    return out


def _series(ticker: str, concepts: Sequence[str], form: str,
            limit: int) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """The chain concept reaching the most recent period, as {period end: row}.

    Every concept in the chain is evaluated rather than stopping at the first
    one that returns anything, because filers change elements between filings
    and the abandoned element still answers from the older filing:

    * Ford's FY2025 10-K tags `us-gaap:ProfitLoss` and not
      `us-gaap:NetIncomeLoss`. Stopping at the first hit found NetIncomeLoss in
      the FY2024 filing and reported Ford's latest year as +5.9bn of net income
      when the year it had just reported was an 8.2bn loss.
    * Alphabet's FY2025 10-K tags `us-gaap:Revenues`; the ASC 606 element only
      covers 2024 and earlier, so DSO was computed against a year-old revenue.

    A stale figure carrying the latest year's label is worse than no figure, so
    freshness decides. Ties keep chain order, which preserves the preference
    for the more specific element when a filer tags both.

    Only NotCovered is swallowed, and only to try the next concept. A network
    failure or an unknown ticker propagates: reporting an outage as "this filer
    does not disclose it" is the one answer worse than an error.
    """
    best_rows: Dict[str, Dict[str, Any]] = {}
    best_concept: Optional[str] = None
    best_end = ""
    for concept in concepts:
        try:
            points = fetch_concept_series(ticker, concept, form=form, limit=limit)
        except NotCovered:
            continue
        rows = _by_period(points)
        if not rows:
            continue
        newest = max(rows)
        if newest > best_end:
            best_rows, best_concept, best_end = rows, concept, newest
    return best_rows, best_concept


def _pct_change(latest: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Percentage change, or None when the base cannot carry one.

    A prior value of zero has no percentage change, and a negative base flips
    the sign of the answer, so both return None rather than a number that
    reads as growth when it is the opposite.
    """
    if latest is None or prior is None or prior <= 0:
        return None
    return (latest - prior) / prior * 100.0


def _ratio(numerator: Optional[float], denominator: Optional[float],
           scale: float = 1.0) -> Optional[float]:
    """numerator / denominator * scale, or None when it cannot be computed.

    `is not None` throughout: a tagged zero numerator is a real disclosure and
    must survive into the answer, while a zero denominator is undefined.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator * scale


def _flat(chains: Sequence[Sequence[str]]) -> List[str]:
    return [concept for chain in chains for concept in chain]


# ==================================================================== accruals

def get_accruals_quality(ticker: str, limit: int = 2,
                         form: str = "10-K") -> Dict[str, Any]:
    """Net income against operating cash flow, and the accrual ratio.

    The accrual ratio is (net income - operating cash flow) / total assets,
    expressed as a percent of assets. Negative means cash flow exceeds reported
    earnings, which is what a healthy filer looks like. Persistently positive
    and rising means earnings are being recognised faster than cash arrives.

    `limit` is the number of annual filings walked. Each 10-K carries two or
    three comparative years, so the default of two filings yields roughly four
    periods of history.
    """
    tried = _flat((NET_INCOME_CONCEPTS, OCF_CONCEPTS, ASSETS_CONCEPTS))
    try:
        ni_rows, ni_concept = _series(ticker, NET_INCOME_CONCEPTS, form, limit)
        ocf_rows, ocf_concept = _series(ticker, OCF_CONCEPTS, form, limit)
        asset_rows, asset_concept = _series(ticker, ASSETS_CONCEPTS, form, limit)
    except Exception as exc:  # noqa: BLE001 - surface the failure, never mask it
        return {"ticker": ticker, "success": False,
                "error": f"fetching earnings-quality concepts failed: {exc}",
                "periods": [], "latest": None, "concepts_tried": tried}

    missing = [name for name, rows in (("net income", ni_rows),
                                       ("operating cash flow", ocf_rows))
               if not rows]
    if missing:
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "error": (f"{ticker} does not tag {' and '.join(missing)} in the "
                      f"last {limit} {form} filings, so accruals cannot be "
                      f"computed. This is a tagging gap, not zero accruals."),
            "periods": [], "latest": None, "trend": None, "flag": None,
            "concepts_tried": tried,
        }

    periods: List[Dict[str, Any]] = []
    for end in sorted(set(ni_rows) & set(ocf_rows), reverse=True):
        net_income = ni_rows[end]["value"]
        ocf = ocf_rows[end]["value"]
        assets = asset_rows[end]["value"] if end in asset_rows else None
        accruals = net_income - ocf
        periods.append({
            "period_end": end,
            "period_days": ni_rows[end]["period_days"],
            "filing_date": ni_rows[end]["filing_date"],
            "net_income": net_income,
            "operating_cash_flow": ocf,
            "total_assets": assets,
            "accruals": accruals,
            "accrual_ratio_pct": _ratio(accruals, assets, 100.0),
            # Only meaningful against positive earnings; a loss-making filer
            # would produce a negative multiple that reads as a warning when
            # it is arithmetic.
            "ocf_to_net_income": (_ratio(ocf, net_income)
                                  if net_income > 0 else None),
        })

    if not periods:
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "error": (f"{ticker} tags net income and operating cash flow but "
                      f"for no common period end, so the two cannot be paired"),
            "periods": [], "latest": None, "trend": None, "flag": None,
            "concepts_tried": tried,
        }

    latest = periods[0]
    prior = periods[1] if len(periods) > 1 else None
    trend = {
        "periods_compared": len(periods),
        "net_income_change_pct": _pct_change(
            latest["net_income"], prior["net_income"] if prior else None),
        "operating_cash_flow_change_pct": _pct_change(
            latest["operating_cash_flow"],
            prior["operating_cash_flow"] if prior else None),
        # Raw comparison rather than percentages so it survives a loss year.
        "divergence": bool(
            prior is not None
            and latest["net_income"] > prior["net_income"]
            and latest["operating_cash_flow"] < prior["operating_cash_flow"]),
        "accrual_ratio_pct_series": [p["accrual_ratio_pct"] for p in periods],
    }

    ratio = latest["accrual_ratio_pct"]
    if ratio is None:
        flag = None
    elif ratio <= 0:
        flag = "cash_backed"
    elif ratio <= _MODERATE_ACCRUAL_PCT:
        flag = "moderate_accruals"
    else:
        flag = "high_accruals"

    return {
        "ticker": ticker, "success": True,
        "coverage": "full" if asset_rows else "partial",
        "periods": periods,
        "latest": latest,
        "trend": trend,
        "flag": flag,
        "concepts_used": {"net_income": ni_concept,
                          "operating_cash_flow": ocf_concept,
                          "total_assets": asset_concept},
        "concepts_tried": tried,
        "note": ("accrual_ratio_pct is (net income - operating cash flow) / "
                 "total assets. Negative means cash flow covers earnings. "
                 "Above 5% is high and above 10% is the range Sloan's accrual "
                 "anomaly work associates with subsequent underperformance. "
                 "'divergence' is the specific pre-blowup shape: earnings up "
                 "while operating cash flow falls. Assets are the period-end "
                 "balance, not an average, so a large acquisition mid-year "
                 "flatters the ratio. A null accrual_ratio_pct means the filer "
                 "did not tag total assets, not that accruals were zero."),
    }
