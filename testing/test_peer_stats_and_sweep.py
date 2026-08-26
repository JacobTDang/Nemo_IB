"""A distribution is not a summary if one member destroys it.

`comparable_company_analysis(["NVDA","AMD","AVGO","INTC"])` returned:

    ev_ebit_data: {mean: -5113.10, median: 53.03, q1: -5162.53,
                   low: -20767.83, high: 209.37}

INTC's ev_ebit of -20,768 comes from an EBIT near zero. Dividing by a
denominator approaching zero produces a number that is arithmetically correct
and analytically meaningless, and feeding it into a mean makes the mean
meaningless too. `success: true`, no warning. Any analyst reading "peer mean
EV/EBIT" gets nonsense, and the median sitting at 53 beside a mean of -5,113
is the tell nobody is required to notice.

Negative multiples are the same problem: a company losing money has no
meaningful P/E, and including it drags the average below the median with no
statement that it happened.

Separately, the LBO's cash sweep pays down more debt than exists:

    year 4: debt_remaining      93,390,556,525.85
    year 5: fcf_after_service  191,004,475,751.96
            debt_paydown       191,004,475,751.96   <- against a 93bn balance
            debt_remaining     0.0

The surplus ~$97.6bn is destroyed rather than accruing to equity, so MOIC and
IRR are understated. Immaterial where equity is huge; material in the
thin-equity structure an LBO model exists to evaluate.
"""
import pytest


class TestPeerStatistics:
    def test_a_negative_multiple_is_excluded_from_the_distribution(self):
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([25.0, 30.0, 35.0, -20767.83])

        assert stats["mean"] == pytest.approx(30.0)
        assert stats["excluded_count"] == 1
        assert stats["included_count"] == 3

    def test_the_exclusion_is_reported_not_silent(self):
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([25.0, -5.0])
        assert stats["excluded_reason"], (
            "a peer dropped from the distribution without a word is a "
            "distribution over a different set than the caller asked for")

    def test_an_all_negative_set_yields_no_statistics_rather_than_nonsense(self):
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([-5.0, -12.0])
        assert stats["mean"] is None
        assert stats["included_count"] == 0
        assert stats["excluded_reason"]

    def test_an_absurd_positive_multiple_is_also_excluded(self):
        """A denominator near zero produces a huge positive just as easily."""
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([25.0, 30.0, 900_000.0])
        assert stats["included_count"] == 2
        assert stats["mean"] == pytest.approx(27.5)

    def test_a_healthy_set_is_untouched(self):
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([20.0, 25.0, 30.0, 35.0])
        assert stats["excluded_count"] == 0
        assert stats["mean"] == pytest.approx(27.5)
        assert stats["median"] == pytest.approx(27.5)

    def test_the_mean_and_median_cannot_end_up_on_opposite_sides_of_zero(self):
        """The signature of the original defect."""
        from tools.financial_modeling_engine.analysis_tools import peer_distribution
        stats = peer_distribution([53.0, 48.0, 60.0, -20767.83])
        assert stats["mean"] is not None and stats["median"] is not None
        assert (stats["mean"] > 0) == (stats["median"] > 0)


class TestCashSweep:
    def test_the_sweep_never_pays_more_debt_than_exists(self):
        from tools.financial_modeling_engine.analysis_tools import _lbo_math
        result = _lbo_math(entry_ev=30e9, revenue_base=10e9, ebitda_margin=0.30,
                           capex_pct_revenue=0.05, depreciation=0.04,
                           tax_rate=0.21, revenue_growth=[0.05] * 5,
                           debt_interest_rate=0.08, leverage_turns=2.0,
                           exit_multiple=10.0, hold_years=5)
        for year in result["year_by_year"]:
            assert year["debt_paydown"] <= year["debt_beginning"] + 1, (
                f"paid {year['debt_paydown']:,.0f} against a balance of "
                f"{year['debt_beginning']:,.0f}")
            assert year["debt_remaining"] >= -1, "debt went negative"

    def test_surplus_cash_reaches_equity_rather_than_vanishing(self):
        from tools.financial_modeling_engine.analysis_tools import _lbo_math
        result = _lbo_math(entry_ev=30e9, revenue_base=10e9, ebitda_margin=0.30,
                           capex_pct_revenue=0.05, depreciation=0.04,
                           tax_rate=0.21, revenue_growth=[0.05] * 5,
                           debt_interest_rate=0.08, leverage_turns=2.0,
                           exit_multiple=10.0, hold_years=5)
        assert result.get("cash_accumulated", 0) > 0, (
            "debt cleared before exit and the remaining free cash flow was "
            "destroyed rather than accruing to the equity holder")
        assert result["equity_proceeds"] > result["exit_ev"] - result["debt_at_exit"]
