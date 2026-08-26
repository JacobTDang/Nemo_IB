"""A rate is a rate, or the caller is told. It is never silently coerced.

The calculators accepted a rate as either a decimal or a percentage by
halving the problem: `if x > 1: x /= 100`. That is safe for a value that is
genuinely a rate and catastrophic for one that is not, because dividing by 100
turns an implausible number into a differently implausible number instead of
an error.

`calculate_dcf` documents `depreciation` as coming from `get_depreciation`,
which returns an absolute figure -- NVDA's is 2,843,000,000. Passed in, it was
read as a rate, divided by 100 to 28,430,000, and used as
`EBIT = EBITDA - depreciation x revenue`:

    year 1 ebit: -7.67e18
    price_per_share: 242,204,233.29     success: true     warnings: []

The heuristic was also applied inconsistently. `ebitda_margin`,
`capex_pct_revenue` and `tax_rate` were normalised; `wacc`, `terminal_growth`
and `revenue_growth` were not. Passing `wacc: 10` -- the percentage convention
the other three accept -- became a 1000% discount rate and a price per share
of 1,104,824,357.

So the rule is: interpret a decimal, interpret a percentage, and refuse
anything that cannot be either. Refusing names the parameter and the unit;
guessing produced a nine-figure share price with `success: true`.
"""
import pytest

from tools.financial_modeling_engine.analysis_tools import as_rate


class TestInterpretation:
    @pytest.mark.parametrize("given,expected", [
        (0.25, 0.25), (0.08, 0.08), (0.0, 0.0), (1.0, 1.0),
    ])
    def test_a_decimal_is_taken_as_written(self, given, expected):
        assert as_rate("tax_rate", given) == pytest.approx(expected)

    @pytest.mark.parametrize("given,expected", [
        (25, 0.25), (8, 0.08), (10, 0.10), (100, 1.0), (2.5, 0.025),
    ])
    def test_a_percentage_is_converted(self, given, expected):
        assert as_rate("tax_rate", given) == pytest.approx(expected)

    def test_the_boundary_is_documented_not_accidental(self):
        """1.0 is a decimal 100%; anything above it is read as a percentage."""
        assert as_rate("x", 1.0) == pytest.approx(1.0)
        assert as_rate("x", 1.5) == pytest.approx(0.015)


class TestRefusal:
    @pytest.mark.parametrize("absurd", [2_843_000_000, 1e12, 500, 101])
    def test_a_figure_that_is_no_kind_of_rate_is_refused(self, absurd):
        with pytest.raises(ValueError) as exc:
            as_rate("depreciation", absurd)
        assert "depreciation" in str(exc.value), "the refusal must name the parameter"

    def test_the_refusal_says_what_was_expected(self):
        with pytest.raises(ValueError) as exc:
            as_rate("depreciation", 2_843_000_000)
        message = str(exc.value).lower()
        assert "rate" in message or "decimal" in message or "percent" in message

    def test_an_absolute_amount_is_recognised_as_such(self):
        """The specific trap: a dollar figure passed where a rate belongs.

        get_depreciation returns `d&a` in dollars and `d&a_pct` as the rate.
        Passing the wrong one is the documented path into a nine-figure share
        price, so the message should point at the right field.
        """
        with pytest.raises(ValueError) as exc:
            as_rate("depreciation", 2_843_000_000)
        assert "d&a_pct" in str(exc.value) or "percent" in str(exc.value).lower()

    @pytest.mark.parametrize("bad", [None, "0.25", float("nan")])
    def test_a_non_number_is_refused_rather_than_coerced(self, bad):
        with pytest.raises((ValueError, TypeError)):
            as_rate("tax_rate", bad)

    def test_a_negative_rate_is_refused_unless_allowed(self):
        with pytest.raises(ValueError):
            as_rate("tax_rate", -0.2)
        # growth can legitimately be negative, so callers may opt in
        assert as_rate("terminal_growth", -0.02, allow_negative=True) == pytest.approx(-0.02)


class TestTheCalculators:
    """The defects exactly as reported, through the public functions."""

    def test_dcf_refuses_an_absolute_depreciation(self):
        from tools.financial_modeling_engine.analysis_tools import _dcf_math
        with pytest.raises(ValueError) as exc:
            _dcf_math(revenue_base=130_497_000_000, ebitda_margin=0.62,
                      capex_pct_revenue=0.05, depreciation=2_843_000_000,
                      tax_rate=0.15, wacc=0.10, terminal_growth=0.03,
                      revenue_growth=[0.5, 0.3, 0.2, 0.15, 0.1],
                      terminal_multiple=0, cash=0, debt=0,
                      shares_outstanding=24_400_000_000)
        assert "depreciation" in str(exc.value)

    def test_dcf_refuses_a_percentage_wacc_it_cannot_distinguish(self):
        """`wacc: 10` was silently a 1000% discount rate."""
        from tools.financial_modeling_engine.analysis_tools import _dcf_math
        result = _dcf_math(revenue_base=130_497_000_000, ebitda_margin=0.62,
                           capex_pct_revenue=0.05, depreciation=0.013,
                           tax_rate=0.15, wacc=10, terminal_growth=0.03,
                           revenue_growth=[0.5, 0.3, 0.2, 0.15, 0.1],
                           terminal_multiple=0, cash=0, debt=0,
                           shares_outstanding=24_400_000_000)
        assert result["assumptions"]["wacc"] == pytest.approx(0.10), (
            "wacc=10 must be read as 10%, the same convention the other rate "
            "parameters already accept")

    def test_a_sane_dcf_still_produces_a_sane_price(self):
        from tools.financial_modeling_engine.analysis_tools import _dcf_math
        result = _dcf_math(revenue_base=130_497_000_000, ebitda_margin=0.62,
                           capex_pct_revenue=0.05, depreciation=0.013,
                           tax_rate=0.15, wacc=0.10, terminal_growth=0.03,
                           revenue_growth=[0.5, 0.3, 0.2, 0.15, 0.1],
                           terminal_multiple=0, cash=0, debt=0,
                           shares_outstanding=24_400_000_000)
        price = result["price_per_share"]
        assert 0 < price < 10_000, f"implausible price per share: {price}"
