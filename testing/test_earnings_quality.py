"""Earnings quality: accruals and working-capital dynamics.

Two gaps that let a company look healthy right up to the point it does not:

- **Accruals.** Net income rising while operating cash flow does not is the
  classic pre-blowup signature. Nothing here compared the two.
- **Working capital.** get_working_capital pulls receivables and inventory and
  stops at net working capital. Receivables growing faster than revenue is
  channel stuffing or a collection problem; inventory building is demand
  deterioration. Neither was computable.

The live figures below were read off the filings before the code was written,
so a passing test means the arithmetic matches EDGAR rather than matching my
belief about EDGAR.
"""
import os

import pytest

from tools.web_search_server import earnings_quality as eq
from tools.web_search_server.sec_series import ConceptFact, FilingPoint, NotCovered

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Apply the real `network` marker plus the offline skip.

  This name used to be bound to a bare pytest.mark.skipif. A skipif is not
  a registered marker, so `-m network` and `-m "not network"` collected
  nothing here -- the tests were selectable only by file path.
  """
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)

SEGMENT_AXIS = "us-gaap:StatementBusinessSegmentsAxis"


def _point(facts, filing_date="2026-07-29"):
    return FilingPoint(filing_date, "10-K", "acc", facts=facts)


def _year(value, end="2026-06-30", start="2025-07-01", dims=None, ref="c-1"):
    return ConceptFact(value, f"duration_{start}_{end}", dims or {}, ref)


def _instant(value, end="2026-06-30", dims=None, ref="c-1"):
    return ConceptFact(value, end, dims or {}, ref)


def _chain(mapping):
    """fetch_concept_series stub: concept -> [FilingPoint], else NotCovered."""
    def fake(ticker, concept, form="10-K", limit=3):
        if concept in mapping:
            return mapping[concept]
        raise NotCovered(concept)
    return fake


# ================================================================== accruals

NI = "us-gaap:NetIncomeLoss"
OCF = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
OCF_CONT = "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
ASSETS = "us-gaap:Assets"


def test_accruals_use_the_consolidated_figure_not_a_segment_split(monkeypatch):
    """A 10-K carries segment and per-class net income alongside the total.
    Summing them reports several times the real earnings."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([
            _year(133_749_000_000.0),
            _year(90_000_000_000.0, dims={SEGMENT_AXIS: "msft:ProductivityMember"}),
            _year(60_000_000_000.0, dims={SEGMENT_AXIS: "msft:CloudMember"}),
        ])],
        OCF: [_point([_year(182_935_000_000.0)])],
        ASSETS: [_point([_instant(758_376_000_000.0)])],
    }))
    result = eq.get_accruals_quality("MSFT")
    assert result["success"] is True
    assert result["latest"]["net_income"] == 133_749_000_000.0


def test_operating_cash_flow_falls_through_the_concept_chain(monkeypatch):
    """Filers with discontinued operations tag only the ContinuingOperations
    variant. A single-concept lookup reports them as having no cash flow."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(100.0)])],
        OCF_CONT: [_point([_year(60.0)])],
        ASSETS: [_point([_instant(1000.0)])],
    }))
    result = eq.get_accruals_quality("X")
    assert result["success"] is True
    assert result["concepts_used"]["operating_cash_flow"] == OCF_CONT
    assert result["latest"]["operating_cash_flow"] == 60.0


def test_accrual_ratio_arithmetic(monkeypatch):
    """(net income - operating cash flow) / total assets, as a percentage."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(100.0)])],
        OCF: [_point([_year(60.0)])],
        ASSETS: [_point([_instant(1000.0)])],
    }))
    latest = eq.get_accruals_quality("X")["latest"]
    assert latest["accruals"] == 40.0
    assert latest["accrual_ratio_pct"] == pytest.approx(4.0)
    assert latest["ocf_to_net_income"] == pytest.approx(0.6)


def test_rising_earnings_against_falling_cash_flow_is_flagged(monkeypatch):
    """The signature the tool exists to catch."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([
            _year(120.0, end="2026-06-30", start="2025-07-01"),
            _year(100.0, end="2025-06-30", start="2024-07-01", ref="c-2"),
        ])],
        OCF: [_point([
            _year(70.0, end="2026-06-30", start="2025-07-01"),
            _year(90.0, end="2025-06-30", start="2024-07-01", ref="c-2"),
        ])],
        ASSETS: [_point([
            _instant(1000.0, end="2026-06-30"),
            _instant(900.0, end="2025-06-30", ref="c-2"),
        ])],
    }))
    result = eq.get_accruals_quality("X")
    assert result["trend"]["divergence"] is True
    assert result["trend"]["net_income_change_pct"] == pytest.approx(20.0)
    assert result["trend"]["operating_cash_flow_change_pct"] == pytest.approx(-100 / 4.5)
    assert len(result["periods"]) == 2


def test_cash_backed_earnings_are_not_flagged(monkeypatch):
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([
            _year(120.0), _year(100.0, end="2025-06-30", start="2024-07-01", ref="c-2")])],
        OCF: [_point([
            _year(180.0), _year(140.0, end="2025-06-30", start="2024-07-01", ref="c-2")])],
        ASSETS: [_point([
            _instant(1000.0), _instant(900.0, end="2025-06-30", ref="c-2")])],
    }))
    result = eq.get_accruals_quality("X")
    assert result["trend"]["divergence"] is False
    assert result["latest"]["accrual_ratio_pct"] < 0
    assert result["flag"] == "cash_backed"


def test_a_tagged_zero_cash_flow_is_kept(monkeypatch):
    """Zero is falsy. A filer reporting exactly zero operating cash flow has
    made a disclosure, and reading it as 'missing' hides the worst case the
    tool can encounter."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(100.0)])],
        OCF: [_point([_year(0.0)])],
        ASSETS: [_point([_instant(1000.0)])],
    }))
    latest = eq.get_accruals_quality("X")["latest"]
    assert latest["operating_cash_flow"] == 0.0
    assert latest["accruals"] == 100.0
    assert latest["accrual_ratio_pct"] == pytest.approx(10.0)


def test_missing_cash_flow_is_an_explicit_failure(monkeypatch):
    """Never zero, never an empty list, and it names what it looked for."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(100.0)])],
    }))
    result = eq.get_accruals_quality("NOTAGS")
    assert result["success"] is False
    assert result["periods"] == []
    assert result["latest"] is None
    assert OCF in " ".join(result["concepts_tried"])


def test_missing_total_assets_leaves_the_ratio_null(monkeypatch):
    """Partial is honest; a guessed denominator is not."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(100.0)])],
        OCF: [_point([_year(60.0)])],
    }))
    result = eq.get_accruals_quality("X")
    assert result["success"] is True
    assert result["coverage"] == "partial"
    assert result["latest"]["accruals"] == 40.0
    assert result["latest"]["accrual_ratio_pct"] is None
    assert result["latest"]["total_assets"] is None


def test_a_quarter_never_answers_for_the_year(monkeypatch):
    """A 10-K tags Q4 and the full year with the same end date. Sorting on the
    period string picks the quarter, understating earnings roughly fourfold."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([
            _year(133_749_000_000.0, start="2025-07-01"),
            _year(30_000_000_000.0, start="2026-04-01", ref="q4"),
        ])],
        OCF: [_point([_year(182_935_000_000.0)])],
        ASSETS: [_point([_instant(758_376_000_000.0)])],
    }))
    result = eq.get_accruals_quality("MSFT")
    assert result["latest"]["net_income"] == 133_749_000_000.0
    assert result["latest"]["period_days"] > 300


def test_a_stale_concept_does_not_answer_for_the_latest_year(monkeypatch):
    """Ford's FY2025 10-K tags us-gaap:ProfitLoss and not us-gaap:NetIncomeLoss.
    Stopping at the first covered concept found NetIncomeLoss in the FY2024
    filing and reported Ford's latest year as +5.879bn of net income, when the
    year it actually just reported was a 8.162bn loss. A stale figure wearing
    the latest year's label is worse than no figure."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(5_879_000_000.0, end="2024-12-31", start="2024-01-01")],
                    filing_date="2025-02-06")],
        "us-gaap:ProfitLoss": [_point([
            _year(-8_162_000_000.0, end="2025-12-31", start="2025-01-01"),
            _year(5_894_000_000.0, end="2024-12-31", start="2024-01-01", ref="c-2"),
        ], filing_date="2026-02-11")],
        OCF: [_point([
            _year(21_282_000_000.0, end="2025-12-31", start="2025-01-01"),
            _year(15_423_000_000.0, end="2024-12-31", start="2024-01-01", ref="c-2")])],
        ASSETS: [_point([
            _instant(289_160_000_000.0, end="2025-12-31"),
            _instant(285_196_000_000.0, end="2024-12-31", ref="c-2")])],
    }))
    result = eq.get_accruals_quality("F")
    assert result["latest"]["period_end"] == "2025-12-31"
    assert result["latest"]["net_income"] == -8_162_000_000.0
    assert result["concepts_used"]["net_income"] == "us-gaap:ProfitLoss"


def test_chain_order_still_decides_when_both_concepts_reach_the_same_period(
        monkeypatch):
    """Preferring the fresher concept must not reorder the chain for filers
    that tag both. NetIncomeLoss is the parent-only figure and stays first."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [_point([_year(100.0)])],
        "us-gaap:ProfitLoss": [_point([_year(105.0)])],
        OCF: [_point([_year(60.0)])],
        ASSETS: [_point([_instant(1000.0)])],
    }))
    result = eq.get_accruals_quality("X")
    assert result["latest"]["net_income"] == 100.0
    assert result["concepts_used"]["net_income"] == NI


def test_the_newest_filing_wins_for_a_restated_period(monkeypatch):
    """Two filings both carry FY2025. The later one restated it, and the
    restatement is the number the company stands behind."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        NI: [
            _point([_year(88.0, end="2025-06-30", start="2024-07-01")],
                   filing_date="2026-07-29"),
            _point([_year(100.0, end="2025-06-30", start="2024-07-01")],
                   filing_date="2025-07-30"),
        ],
        OCF: [_point([_year(60.0, end="2025-06-30", start="2024-07-01")])],
        ASSETS: [_point([_instant(1000.0, end="2025-06-30")])],
    }))
    assert eq.get_accruals_quality("X")["latest"]["net_income"] == 88.0


# =========================================================== working capital

AR = "us-gaap:AccountsReceivableNetCurrent"
AR_ALT = "us-gaap:ReceivablesNetCurrent"
INV = "us-gaap:InventoryNet"
AP = "us-gaap:AccountsPayableCurrent"
REV = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
COGS = "us-gaap:CostOfRevenue"
COGS_ALT = "us-gaap:CostOfGoodsAndServicesSold"
INV_AEROSPACE = "us-gaap:InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings"


def _costco_year(end="2025-08-31", start="2024-09-02"):
    """COST FY2025, straight off the 10-K."""
    return {
        REV: [_point([_year(275_235_000_000.0, end=end, start=start)])],
        COGS_ALT: [_point([_year(239_886_000_000.0, end=end, start=start)])],
        AR_ALT: [_point([_instant(3_203_000_000.0, end=end)])],
        INV: [_point([_instant(18_116_000_000.0, end=end)])],
        AP: [_point([_instant(19_783_000_000.0, end=end)])],
    }


def test_dso_dio_dpo_and_the_cash_conversion_cycle(monkeypatch):
    monkeypatch.setattr(eq, "fetch_concept_series", _chain(_costco_year()))
    latest = eq.get_working_capital_trends("COST")["latest"]
    # 3,203 / 275,235 * 364 days
    assert latest["dso"] == pytest.approx(4.24, abs=0.05)
    assert latest["dio"] == pytest.approx(27.5, abs=0.15)
    assert latest["dpo"] == pytest.approx(30.0, abs=0.15)
    assert latest["cash_conversion_cycle"] == pytest.approx(
        latest["dso"] + latest["dio"] - latest["dpo"])


def test_cost_of_revenue_falls_through_the_concept_chain(monkeypatch):
    """WMT tags CostOfRevenue, MSFT and COST tag CostOfGoodsAndServicesSold.
    Either one alone reports half the market as having no cost of sales."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain(_costco_year()))
    result = eq.get_working_capital_trends("COST")
    assert result["concepts_used"]["cost_of_revenue"] == COGS_ALT


def test_receivables_fall_through_the_concept_chain(monkeypatch):
    """WMT and COST tag ReceivablesNetCurrent, not AccountsReceivableNetCurrent."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain(_costco_year()))
    result = eq.get_working_capital_trends("COST")
    assert result["concepts_used"]["receivables"] == AR_ALT
    assert result["latest"]["accounts_receivable"] == 3_203_000_000.0


def test_a_filer_with_no_inventory_reports_null_not_zero(monkeypatch):
    """CRM tags no InventoryNet at all. DIO of zero would read as 'sells
    through instantly', which is a different claim from 'holds no inventory'."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        REV: [_point([_year(41_525_000_000.0, end="2026-01-31", start="2025-02-01")])],
        COGS_ALT: [_point([_year(9_270_000_000.0, end="2026-01-31", start="2025-02-01")])],
        AR: [_point([_instant(14_339_000_000.0, end="2026-01-31")])],
    }))
    result = eq.get_working_capital_trends("CRM")
    assert result["success"] is True
    assert result["coverage"] == "partial"
    assert result["latest"]["inventory"] is None
    assert result["latest"]["dio"] is None
    assert result["latest"]["cash_conversion_cycle"] is None
    assert result["latest"]["dso"] == pytest.approx(126.0, abs=0.5)
    assert INV in " ".join(result["concepts_tried"])


def test_aerospace_inventory_is_found_under_its_own_concept(monkeypatch):
    """Boeing nets customer advances off inventory and tags it as
    InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings. A chain of
    InventoryNet alone reported the largest inventory position in the
    industrials index as no inventory at all."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        REV: [_point([_year(1000.0)])],
        COGS: [_point([_year(500.0)])],
        INV_AEROSPACE: [_point([_instant(250.0)])],
    }))
    result = eq.get_working_capital_trends("BA")
    assert result["latest"]["inventory"] == 250.0
    assert result["concepts_used"]["inventory"] == INV_AEROSPACE


def test_receivables_outgrowing_revenue_is_surfaced(monkeypatch):
    """Revenue +10%, receivables +50%. The gap is the tool's reason to exist."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        REV: [_point([
            _year(110.0, end="2026-06-30", start="2025-07-01"),
            _year(100.0, end="2025-06-30", start="2024-07-01", ref="c-2")])],
        COGS: [_point([
            _year(55.0, end="2026-06-30", start="2025-07-01"),
            _year(50.0, end="2025-06-30", start="2024-07-01", ref="c-2")])],
        AR: [_point([
            _instant(15.0, end="2026-06-30"),
            _instant(10.0, end="2025-06-30", ref="c-2")])],
    }))
    latest = eq.get_working_capital_trends("X")["latest"]
    assert latest["revenue_growth_pct"] == pytest.approx(10.0)
    assert latest["receivables_growth_pct"] == pytest.approx(50.0)
    assert latest["receivables_vs_revenue_gap_pct"] == pytest.approx(40.0)


def test_days_come_from_the_period_span_not_a_hardcoded_year(monkeypatch):
    """A 10-Q's revenue covers a quarter. Dividing a quarter's revenue by 365
    days quadruples DSO and turns a healthy filer into a fraud."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        REV: [_point([_year(100.0, end="2026-06-30", start="2026-04-01")])],
        COGS: [_point([_year(50.0, end="2026-06-30", start="2026-04-01")])],
        AR: [_point([_instant(50.0, end="2026-06-30")])],
    }))
    latest = eq.get_working_capital_trends("X", form="10-Q")["latest"]
    assert latest["period_days"] == 90
    assert latest["dso"] == pytest.approx(45.0)


def test_segment_receivables_do_not_displace_the_consolidated_balance(monkeypatch):
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        REV: [_point([_year(1000.0)])],
        COGS: [_point([_year(500.0)])],
        AR: [_point([
            _instant(100.0),
            _instant(400.0, dims={SEGMENT_AXIS: "x:BigSegmentMember"}, ref="c-9")])],
    }))
    assert eq.get_working_capital_trends("X")["latest"]["accounts_receivable"] == 100.0


def test_no_revenue_is_an_explicit_failure(monkeypatch):
    """Every ratio here divides by revenue or cost of revenue. Without one
    there is nothing to report, and reporting nothing must not look like zero
    days outstanding."""
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        AR: [_point([_instant(100.0)])],
    }))
    result = eq.get_working_capital_trends("NOTAGS")
    assert result["success"] is False
    assert result["periods"] == []
    assert result["latest"] is None
    assert "revenue" in result["error"].lower()


def test_a_tagged_zero_receivable_is_kept(monkeypatch):
    monkeypatch.setattr(eq, "fetch_concept_series", _chain({
        REV: [_point([_year(1000.0)])],
        COGS: [_point([_year(500.0)])],
        AR: [_point([_instant(0.0)])],
    }))
    latest = eq.get_working_capital_trends("X")["latest"]
    assert latest["accounts_receivable"] == 0.0
    assert latest["dso"] == 0.0


# ============================================================ live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


def _period(result, end):
    for row in result["periods"]:
        if row["period_end"] == end:
            return row
    raise AssertionError(
        f"period {end} missing from {[r['period_end'] for r in result['periods']]}")


@network
def test_msft_accruals_match_the_fy2026_10k():
    """MSFT FY2026: net income 133.749bn, operating cash flow 182.935bn,
    total assets 758.376bn. Cash flow well above earnings -- the clean case."""
    result = eq.get_accruals_quality("MSFT")
    assert result["success"] is True
    row = _period(result, "2026-06-30")
    assert row["net_income"] == 133_749_000_000.0
    assert row["operating_cash_flow"] == 182_935_000_000.0
    assert row["total_assets"] == 758_376_000_000.0
    assert row["accrual_ratio_pct"] == pytest.approx(-6.49, abs=0.05)
    assert result["flag"] == "cash_backed"


@network
def test_aapl_accruals_match_the_fy2025_10k():
    """AAPL FY2025 (ends 2025-09-27): net income 112.010bn, operating cash
    flow 111.482bn, total assets 359.241bn -- earnings just above cash for the
    first time in years.

    This is the live divergence case, and it is also why the periods are
    joined on end date rather than by position. Reading the three comparative
    cash-flow facts in filing order gives 118.254bn for FY2025; that is
    FY2024's figure, and it would have turned a positive accrual ratio
    negative.
    """
    result = eq.get_accruals_quality("AAPL")
    assert result["success"] is True
    row = _period(result, "2025-09-27")
    assert row["net_income"] == 112_010_000_000.0
    assert row["operating_cash_flow"] == 111_482_000_000.0
    assert row["total_assets"] == 359_241_000_000.0
    assert row["accruals"] == 528_000_000.0
    assert row["accrual_ratio_pct"] == pytest.approx(0.147, abs=0.005)

    prior = _period(result, "2024-09-28")
    assert prior["net_income"] == 93_736_000_000.0
    assert prior["operating_cash_flow"] == 118_254_000_000.0
    # Earnings up, cash flow down. FY2024 net income carried the EU State Aid
    # charge, so this particular divergence has a benign explanation -- the
    # flag is a prompt to look, not a verdict.
    assert result["trend"]["divergence"] is True
    assert result["flag"] == "moderate_accruals"


@network
def test_msft_total_assets_are_not_current_assets():
    """The denominator regression. MSFT tags AssetsCurrent at 207.7bn, and a
    prefix-matched concept query returned it as total assets -- which would
    have turned a -6.5% accrual ratio into -23.7%."""
    row = _period(eq.get_accruals_quality("MSFT"), "2026-06-30")
    assert row["total_assets"] > 500e9


@network
def test_cost_working_capital_matches_the_fy2025_10k():
    """COST FY2025: revenue 275.235bn, cost of goods 239.886bn, inventory
    18.116bn, receivables 3.203bn, payables 19.783bn. A warehouse retailer
    turns inventory in under a month and is paid before it pays."""
    result = eq.get_working_capital_trends("COST")
    assert result["success"] is True
    row = _period(result, "2025-08-31")
    assert row["inventory"] == 18_116_000_000.0
    assert row["dio"] == pytest.approx(27.5, abs=0.6)
    assert row["dso"] == pytest.approx(4.2, abs=0.6)
    assert row["cash_conversion_cycle"] < 10, "COST is paid before it pays"


@network
def test_wmt_working_capital_matches_the_fy2026_10k():
    """WMT FY2026 (ends 2026-01-31): inventory 58.851bn against 535.395bn cost
    of revenue -- about forty days of stock. WMT tags ReceivablesNetCurrent
    rather than AccountsReceivableNetCurrent, so the chain is load-bearing."""
    result = eq.get_working_capital_trends("WMT")
    assert result["success"] is True
    row = _period(result, "2026-01-31")
    assert row["inventory"] == 58_851_000_000.0
    assert row["dio"] == pytest.approx(40.0, abs=1.0)
    assert row["accounts_receivable"] == 11_172_000_000.0


@network
def test_crm_has_no_inventory_and_that_is_the_right_answer():
    """CRM tags no InventoryNet and no AccountsPayableCurrent. DIO must be
    null rather than zero, and the call still succeeds on DSO."""
    result = eq.get_working_capital_trends("CRM")
    assert result["success"] is True
    row = _period(result, "2026-01-31")
    assert row["inventory"] is None
    assert row["dio"] is None
    assert row["dso"] == pytest.approx(126.0, abs=1.0)


@network
def test_ford_reports_the_year_it_actually_just_filed():
    """Live pin on the stale-concept regression. Ford's FY2025 10-K tags
    us-gaap:ProfitLoss and drops us-gaap:NetIncomeLoss, so a chain that stops
    at the first covered concept reported FY2024's +5.879bn as Ford's latest
    net income instead of FY2025's 8.162bn loss."""
    result = eq.get_accruals_quality("F")
    assert result["success"] is True
    assert result["latest"]["period_end"] == "2025-12-31"
    assert result["latest"]["net_income"] == -8_162_000_000.0
    assert result["latest"]["operating_cash_flow"] == 21_282_000_000.0
    assert result["latest"]["total_assets"] == 289_160_000_000.0
    assert result["concepts_used"]["net_income"] == "us-gaap:ProfitLoss"


@network
def test_googl_working_capital_uses_the_concept_its_latest_10k_tags():
    """Alphabet's FY2025 10-K tags us-gaap:Revenues at 402.836bn. The ASC 606
    element covers 2024 and earlier only, and taking it computed DSO against a
    year-old revenue."""
    result = eq.get_working_capital_trends("GOOGL")
    assert result["success"] is True
    assert result["latest"]["period_end"] == "2025-12-31"
    assert result["latest"]["revenue"] == 402_836_000_000.0
    assert result["concepts_used"]["revenue"] == "us-gaap:Revenues"
    assert result["latest"]["dio"] is None, "Alphabet tags no inventory"


@network
def test_ba_inventory_is_the_aerospace_concept():
    """BA FY2025: 84.679bn of inventory net of customer advances against
    85.174bn of cost of revenue -- around a year of stock, which is what
    building aircraft to order looks like and is the reason DIO belongs on the
    same screen as accruals."""
    result = eq.get_working_capital_trends("BA")
    assert result["success"] is True
    row = _period(result, "2025-12-31")
    assert row["inventory"] == 84_679_000_000.0
    assert row["dio"] == pytest.approx(362.0, abs=3.0)