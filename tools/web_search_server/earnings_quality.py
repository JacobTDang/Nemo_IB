"""Are these earnings real? Accruals, working capital, and operating leases.

Three questions the SEC layer could not answer:

**Accruals.** Net income rising while operating cash flow does not is the
classic pre-blowup signature -- earnings arriving as promises rather than cash.
Both numbers were extractable one at a time and nothing compared them.

**Working capital.** `sec_utils.get_working_capital` pulls receivables,
inventory and payables and stops at net working capital. It never divides them
by revenue, so receivables growing faster than sales (channel stuffing, or
customers who have stopped paying) and inventory building ahead of demand were
both invisible.

**Operating leases.** ASC 842 put them on the balance sheet in 2019. There was
zero coverage here, which left a hole beside `get_debt_maturity_schedule`: for
a retailer the lease book is the larger fixed obligation, and it comes due on a
schedule nobody was reading.

Everything is period-joined on the fiscal period end date, so a balance-sheet
instant is matched to the income-statement duration that ends the same day.
Day counts come from the actual period span rather than a hardcoded 365, which
is what makes the ratios comparable when the filing is a 10-Q or the filer runs
a 52/53-week year.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .foreign_issuer import form_mismatch_note
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

# WMT and COST tag ReceivablesNetCurrent; MSFT, AAPL and CRM tag
# AccountsReceivableNetCurrent. Neither alone covers the market.
RECEIVABLES_CONCEPTS = (
    "us-gaap:AccountsReceivableNetCurrent",
    "us-gaap:ReceivablesNetCurrent",
    "us-gaap:AccountsAndOtherReceivablesNetCurrent",
    "us-gaap:AccountsNotesAndLoansReceivableNetCurrent",
)

# InventoryNet covers most filers, but two large sectors do not use it at all:
# aerospace and defence net customer advances off the balance
# (BA carries 84.7bn there), and the majors tag energy inventory separately.
INVENTORY_CONCEPTS = (
    "us-gaap:InventoryNet",
    "us-gaap:InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings",
    "us-gaap:EnergyRelatedInventory",
    "us-gaap:RetailRelatedInventory",
    "us-gaap:InventoryFinishedGoodsNetOfReserves",
)

PAYABLES_CONCEPTS = (
    "us-gaap:AccountsPayableCurrent",
    "us-gaap:AccountsPayableTradeCurrent",
    "us-gaap:AccountsPayableAndAccruedLiabilitiesCurrent",
)

REVENUE_CONCEPTS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
)

# WMT tags CostOfRevenue, MSFT/AAPL/COST tag CostOfGoodsAndServicesSold.
COST_OF_REVENUE_CONCEPTS = (
    "us-gaap:CostOfRevenue",
    "us-gaap:CostOfGoodsAndServicesSold",
    "us-gaap:CostOfGoodsSold",
    "us-gaap:CostOfServices",
)

LEASE_LIABILITY_CONCEPTS = ("us-gaap:OperatingLeaseLiability",)
LEASE_LIABILITY_CURRENT_CONCEPTS = ("us-gaap:OperatingLeaseLiabilityCurrent",)
LEASE_LIABILITY_NONCURRENT_CONCEPTS = ("us-gaap:OperatingLeaseLiabilityNoncurrent",)
ROU_ASSET_CONCEPTS = ("us-gaap:OperatingLeaseRightOfUseAsset",)

_LEASE_DUE = "us-gaap:LesseeOperatingLeaseLiabilityPaymentsDue"

# Same two families as the debt schedule: fixed years or rolling years. The
# rolling names are not formed the same way as the fixed ones -- the taxonomy
# spells them "...PaymentsDueInRollingYearTwo", with an "In" the fixed variant
# does not have. Guessing the pattern gave PFE two buckets out of six.
LEASE_MATURITY_CONCEPTS: Dict[str, tuple] = {
    "year_1": (f"{_LEASE_DUE}NextTwelveMonths", f"{_LEASE_DUE}NextRollingTwelveMonths"),
    "year_2": (f"{_LEASE_DUE}YearTwo", f"{_LEASE_DUE}InRollingYearTwo"),
    "year_3": (f"{_LEASE_DUE}YearThree", f"{_LEASE_DUE}InRollingYearThree"),
    "year_4": (f"{_LEASE_DUE}YearFour", f"{_LEASE_DUE}InRollingYearFour"),
    "year_5": (f"{_LEASE_DUE}YearFive", f"{_LEASE_DUE}InRollingYearFive"),
    "after_year_5": (f"{_LEASE_DUE}AfterYearFive", f"{_LEASE_DUE}AfterRollingYearFive"),
}

LEASE_PAYMENTS_TOTAL_CONCEPTS = (_LEASE_DUE,)
LEASE_IMPUTED_INTEREST_CONCEPTS = (
    "us-gaap:LesseeOperatingLeaseLiabilityUndiscountedExcessAmount",
)

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
        mismatch = form_mismatch_note(ticker, form)
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "wrong_form": bool(mismatch),
            "error": mismatch or (
                      f"{ticker} does not tag {' and '.join(missing)} in the "
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


# ============================================================ working capital

def get_working_capital_trends(ticker: str, limit: int = 2,
                               form: str = "10-K") -> Dict[str, Any]:
    """Days sales outstanding, days inventory, days payable, and the cycle.

    DSO is receivables divided by revenue per day; DIO and DPO divide
    inventory and payables by cost of revenue per day. Days come from each
    period's actual span, so quarterly filings and 52/53-week years stay
    comparable rather than being scaled against a nominal 365.
    """
    tried = _flat((REVENUE_CONCEPTS, COST_OF_REVENUE_CONCEPTS,
                   RECEIVABLES_CONCEPTS, INVENTORY_CONCEPTS, PAYABLES_CONCEPTS))
    try:
        rev_rows, rev_concept = _series(ticker, REVENUE_CONCEPTS, form, limit)
        cogs_rows, cogs_concept = _series(
            ticker, COST_OF_REVENUE_CONCEPTS, form, limit)
        ar_rows, ar_concept = _series(ticker, RECEIVABLES_CONCEPTS, form, limit)
        inv_rows, inv_concept = _series(ticker, INVENTORY_CONCEPTS, form, limit)
        ap_rows, ap_concept = _series(ticker, PAYABLES_CONCEPTS, form, limit)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "success": False,
                "error": f"fetching working-capital concepts failed: {exc}",
                "periods": [], "latest": None, "concepts_tried": tried}

    if not rev_rows:
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "error": form_mismatch_note(ticker, form) or (
                      f"{ticker} does not tag revenue in the last {limit} "
                      f"{form} filings. Every ratio here is per revenue-day, "
                      f"so none can be computed."),
            "periods": [], "latest": None, "concepts_tried": tried,
        }
    if not (ar_rows or inv_rows or ap_rows):
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "error": (f"{ticker} tags none of receivables, inventory or "
                      f"payables in its {form}, so there is no working-capital "
                      f"cycle to report"),
            "periods": [], "latest": None, "concepts_tried": tried,
        }

    periods: List[Dict[str, Any]] = []
    for end in sorted(rev_rows, reverse=True):
        days = rev_rows[end]["period_days"]
        if days <= 0:
            continue  # a revenue fact with no usable span cannot carry a rate
        revenue = rev_rows[end]["value"]
        cost = cogs_rows[end]["value"] if end in cogs_rows else None
        receivables = ar_rows[end]["value"] if end in ar_rows else None
        inventory = inv_rows[end]["value"] if end in inv_rows else None
        payables = ap_rows[end]["value"] if end in ap_rows else None
        dso = _ratio(receivables, revenue, days)
        dio = _ratio(inventory, cost, days)
        dpo = _ratio(payables, cost, days)
        periods.append({
            "period_end": end,
            "period_days": days,
            "filing_date": rev_rows[end]["filing_date"],
            "revenue": revenue,
            "cost_of_revenue": cost,
            "accounts_receivable": receivables,
            "inventory": inventory,
            "accounts_payable": payables,
            "dso": dso,
            "dio": dio,
            "dpo": dpo,
            "cash_conversion_cycle": (dso + dio - dpo
                                      if None not in (dso, dio, dpo) else None),
        })

    if not periods:
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "error": (f"{ticker} tags revenue but no period carries a usable "
                      f"duration, so no per-day ratio can be computed"),
            "periods": [], "latest": None, "concepts_tried": tried,
        }

    # Growth is measured against the next-older period in the series, which is
    # the comparison that matters: receivables outrunning revenue.
    for index, row in enumerate(periods):
        older = periods[index + 1] if index + 1 < len(periods) else None
        rev_growth = _pct_change(row["revenue"],
                                 older["revenue"] if older else None)
        ar_growth = _pct_change(row["accounts_receivable"],
                                older["accounts_receivable"] if older else None)
        inv_growth = _pct_change(row["inventory"],
                                 older["inventory"] if older else None)
        row["revenue_growth_pct"] = rev_growth
        row["receivables_growth_pct"] = ar_growth
        row["inventory_growth_pct"] = inv_growth
        row["receivables_vs_revenue_gap_pct"] = (
            ar_growth - rev_growth
            if ar_growth is not None and rev_growth is not None else None)
        row["inventory_vs_revenue_gap_pct"] = (
            inv_growth - rev_growth
            if inv_growth is not None and rev_growth is not None else None)

    covered = all((cogs_rows, ar_rows, inv_rows, ap_rows))
    return {
        "ticker": ticker, "success": True,
        "coverage": "full" if covered else "partial",
        "periods": periods,
        "latest": periods[0],
        "concepts_used": {"revenue": rev_concept,
                          "cost_of_revenue": cogs_concept,
                          "receivables": ar_concept,
                          "inventory": inv_concept,
                          "payables": ap_concept},
        "concepts_tried": tried,
        "note": ("DSO is receivables per revenue-day; DIO and DPO are "
                 "inventory and payables per cost-of-revenue-day. The cash "
                 "conversion cycle is DSO + DIO - DPO, and a negative cycle "
                 "means suppliers finance the business. A null DIO means the "
                 "filer tags no inventory at all, which is the correct answer "
                 "for software and most services businesses -- it is not zero "
                 "days of stock. Balances are period-end, not averages, so a "
                 "seasonal year-end distorts the level; the growth gaps are "
                 "the more reliable signal. receivables_vs_revenue_gap_pct "
                 "above roughly 10 points means receivables are outrunning "
                 "sales, which is channel stuffing or a collection problem."),
    }


# =========================================================== operating leases

def _latest_value(rows: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Most recent period's value, or None when the concept was not tagged."""
    if not rows:
        return None
    return rows[max(rows)]["value"]


def get_operating_leases(ticker: str, limit: int = 1,
                         form: str = "10-K") -> Dict[str, Any]:
    """Operating lease obligations and when the payments come due.

    ASC 842 put these on the balance sheet, but the maturity ladder stayed in
    the footnote. For an asset-light retailer or restaurant chain the lease
    book is the larger fixed obligation and it belongs next to the debt
    schedule, not in a separate mental bucket.
    """
    balance_chains = (LEASE_LIABILITY_CONCEPTS, LEASE_LIABILITY_CURRENT_CONCEPTS,
                      LEASE_LIABILITY_NONCURRENT_CONCEPTS, ROU_ASSET_CONCEPTS,
                      LEASE_PAYMENTS_TOTAL_CONCEPTS,
                      LEASE_IMPUTED_INTEREST_CONCEPTS)
    tried = _flat(tuple(balance_chains) + tuple(LEASE_MATURITY_CONCEPTS.values()))
    try:
        total_rows, _ = _series(ticker, LEASE_LIABILITY_CONCEPTS, form, limit)
        current_rows, _ = _series(
            ticker, LEASE_LIABILITY_CURRENT_CONCEPTS, form, limit)
        noncurrent_rows, _ = _series(
            ticker, LEASE_LIABILITY_NONCURRENT_CONCEPTS, form, limit)
        rou_rows, _ = _series(ticker, ROU_ASSET_CONCEPTS, form, limit)
        payments_rows, _ = _series(
            ticker, LEASE_PAYMENTS_TOTAL_CONCEPTS, form, limit)
        interest_rows, _ = _series(
            ticker, LEASE_IMPUTED_INTEREST_CONCEPTS, form, limit)
        schedule: Dict[str, Optional[float]] = {}
        for bucket, concepts in LEASE_MATURITY_CONCEPTS.items():
            rows, _ = _series(ticker, concepts, form, 1)
            # None means untagged; 0.0 means the filer disclosed nothing due.
            schedule[bucket] = _latest_value(rows)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "success": False,
                "error": f"fetching operating-lease concepts failed: {exc}",
                "lease_liability": None, "maturity_schedule": {},
                "periods": [], "concepts_tried": tried}

    liability = _latest_value(total_rows)
    rou_asset = _latest_value(rou_rows)
    buckets_found = sum(1 for v in schedule.values() if v is not None)

    if liability is None and rou_asset is None and buckets_found == 0:
        return {
            "ticker": ticker, "success": False, "coverage": "not_covered",
            "error": form_mismatch_note(ticker, form) or (
                      f"{ticker} tags no operating-lease concepts in its "
                      f"{form}. Either the filer has no material operating "
                      f"leases or it did not tag them; this is not a zero "
                      f"obligation."),
            "lease_liability": None, "lease_liability_current": None,
            "lease_liability_noncurrent": None, "right_of_use_asset": None,
            "maturity_schedule": schedule, "buckets_found": 0,
            "periods": [], "concepts_tried": tried,
        }

    tagged_payments = _latest_value(payments_rows)
    bucket_values = [v for v in schedule.values() if v is not None]
    if tagged_payments is not None:
        payments_total: Optional[float] = tagged_payments
        payments_source = "tagged"
    elif bucket_values:
        payments_total = float(sum(bucket_values))
        payments_source = "sum_of_buckets"
    else:
        payments_total = None
        payments_source = None

    # One 10-K carries two balance-sheet dates, which is the trend.
    period_ends = sorted(set(total_rows) | set(rou_rows) | set(current_rows)
                         | set(noncurrent_rows), reverse=True)
    periods = [{
        "period_end": end,
        "filing_date": (total_rows.get(end) or rou_rows.get(end)
                        or current_rows.get(end)
                        or noncurrent_rows.get(end))["filing_date"],
        "lease_liability": total_rows[end]["value"] if end in total_rows else None,
        "lease_liability_current": (current_rows[end]["value"]
                                    if end in current_rows else None),
        "lease_liability_noncurrent": (noncurrent_rows[end]["value"]
                                       if end in noncurrent_rows else None),
        "right_of_use_asset": rou_rows[end]["value"] if end in rou_rows else None,
    } for end in period_ends]

    covered = (liability is not None and rou_asset is not None
               and buckets_found == len(LEASE_MATURITY_CONCEPTS))
    return {
        "ticker": ticker, "success": True,
        "coverage": "full" if covered else "partial",
        "as_of": period_ends[0] if period_ends else None,
        "lease_liability": liability,
        "lease_liability_current": _latest_value(current_rows),
        "lease_liability_noncurrent": _latest_value(noncurrent_rows),
        "right_of_use_asset": rou_asset,
        "maturity_schedule": schedule,
        "buckets_found": buckets_found,
        "buckets_expected": len(LEASE_MATURITY_CONCEPTS),
        "undiscounted_payments_total": payments_total,
        "undiscounted_payments_source": payments_source,
        "imputed_interest": _latest_value(interest_rows),
        "pct_due_within_one_year": _ratio(
            schedule.get("year_1"), payments_total, 100.0),
        "periods": periods,
        "concepts_tried": tried,
        "note": ("lease_liability is the discounted present value on the "
                 "balance sheet; the maturity schedule and "
                 "undiscounted_payments_total are the contractual cash "
                 "payments, and the difference between them is "
                 "imputed_interest. A null bucket means the filer did not tag "
                 "that year, while 0.0 means it disclosed nothing due -- the "
                 "two are never merged. A missing current portion is left null "
                 "rather than backed out of the total, because a derived "
                 "figure is not a disclosure. Read alongside "
                 "get_debt_maturity_schedule: for retailers and restaurants "
                 "the lease ladder is the larger of the two."),
    }
