"""Three defects found by re-running QA against the rebuilt stack.

One is a regression I introduced, one is a discount rate of zero, and one is a
lookback that counts rows instead of time.
"""
import pytest


class TestSuppressedPeersAreNotInvisible:
    """A regression from the cross-currency fix.

    Suppressing a foreign issuer's multiples made them null. `peer_distribution`
    filters non-numeric values before counting exclusions, so the nulls
    vanished silently: a four-name comp set reported `included 2, excluded 0`
    and published a median that was really just the two US names. The
    suppression was correct; making it invisible was not.
    """

    def test_a_null_multiple_is_counted_as_excluded(self):
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([25.0, 30.0, None, None])

        assert stats["included_count"] == 2
        assert stats["excluded_count"] == 2, (
            "two peers had no comparable multiple and the count says none "
            "were dropped")
        assert stats["excluded_reason"]

    def test_the_reason_distinguishes_absent_from_implausible(self):
        """They are dropped for different reasons and a reader needs both."""
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        reason = peer_distribution([25.0, None, -20767.0])["excluded_reason"]
        assert "no comparable" in reason.lower() or "not reported" in reason.lower()
        assert "denominator" in reason.lower() or "negative" in reason.lower()

    def test_an_all_null_set_is_not_an_empty_set(self):
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([None, None, None])
        assert stats["mean"] is None
        assert stats["excluded_count"] == 3
        assert stats["excluded_reason"]


class TestDiscountRateMustDiscount:
    """`wacc: 0` was accepted and discounted nothing.

    pv_fcfs came back identical to the undiscounted series and the enterprise
    value reached $2.6 trillion on modest inputs -- for JPM the QA run saw
    $5.98tn against a market cap of $948bn, with price_per_share 0,
    success: true and no warnings. Zero is not a discount rate; it is the
    absence of one.
    """

    @pytest.mark.parametrize("bad", [0, 0.0])
    def test_a_zero_discount_rate_is_refused(self, bad):
        from tools.financial_modeling_engine.analysis_tools import _dcf_math
        with pytest.raises(ValueError) as exc:
            _dcf_math(revenue_base=100e9, ebitda_margin=0.30,
                      capex_pct_revenue=0.05, tax_rate=0.21, depreciation=0.03,
                      revenue_growth=[0.05] * 5, wacc=bad, terminal_growth=0.02,
                      terminal_multiple=0, cash=0, debt=0,
                      shares_outstanding=1e9)
        assert "wacc" in str(exc.value).lower()

    def test_terminal_growth_at_or_above_wacc_is_refused(self):
        """The perpetuity is undefined there and returns a vast number."""
        from tools.financial_modeling_engine.analysis_tools import _dcf_math
        with pytest.raises(ValueError):
            _dcf_math(revenue_base=100e9, ebitda_margin=0.30,
                      capex_pct_revenue=0.05, tax_rate=0.21, depreciation=0.03,
                      revenue_growth=[0.05] * 5, wacc=0.05, terminal_growth=0.05,
                      terminal_multiple=0, cash=0, debt=0,
                      shares_outstanding=1e9)

    def test_a_real_discount_rate_actually_discounts(self):
        from tools.financial_modeling_engine.analysis_tools import _dcf_math
        result = _dcf_math(revenue_base=100e9, ebitda_margin=0.30,
                           capex_pct_revenue=0.05, tax_rate=0.21,
                           depreciation=0.03, revenue_growth=[0.05] * 5,
                           wacc=0.10, terminal_growth=0.02, terminal_multiple=0,
                           cash=0, debt=0, shares_outstanding=1e9)
        fcf = [year["fcf"] for year in result["fcf_projections"]]
        pv = result["pv_fcfs"]
        assert all(p < f for p, f in zip(pv, fcf)), (
            f"every discounted cash flow must be smaller than the flow "
            f"itself; got pv={pv[:3]} against fcf={fcf[:3]}")


@pytest.mark.network
class TestLookbacksMeasureTime:
    """`valid[-13]` steps back thirteen OBSERVATIONS, not thirteen months.

    The comment beside it says "monthly data". DGS10 is daily, so its
    "1y_ago" was twelve business days ago: the ten-year reported a one-year
    change of +1bp when FRED's own series gives roughly +42bp. UNRATE is
    monthly, so it looked correct, which is why the row offset survived.

    The units fix earlier got the arithmetic right on top of a wrong baseline.
    """

    def test_a_daily_series_looks_back_an_actual_year(self):
        import asyncio, json
        from datetime import date
        from tools.news_agregator.fred_server import FredServer

        server = FredServer()
        try:
            body = json.loads(asyncio.run(server.get_macro_snapshot())[0].text)
        finally:
            asyncio.run(server.client.close())

        entry = body["data"]["DGS10"]
        as_of = date.fromisoformat(entry["as_of"])
        baseline = date.fromisoformat(entry["1y_ago_date"])
        gap_days = (as_of - baseline).days
        assert 330 <= gap_days <= 400, (
            f"1y_ago is {gap_days} days back, not a year. A daily series has "
            f"about 250 observations a year, so a row offset of 13 lands two "
            f"weeks ago.")

    def test_a_monthly_series_is_unaffected(self):
        import asyncio, json
        from datetime import date
        from tools.news_agregator.fred_server import FredServer

        server = FredServer()
        try:
            body = json.loads(asyncio.run(server.get_macro_snapshot())[0].text)
        finally:
            asyncio.run(server.client.close())

        entry = body["data"]["UNRATE"]
        gap = (date.fromisoformat(entry["as_of"])
               - date.fromisoformat(entry["1y_ago_date"])).days
        assert 330 <= gap <= 400
