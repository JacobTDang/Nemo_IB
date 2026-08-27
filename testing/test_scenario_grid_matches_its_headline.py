"""Two price targets from two different methods must not sit side by side.

`calculate_scenario_dcf` returned, in one payload:

    base.price_per_share                    151.96
    assumptions.terminal_multiple           25
    terminal_sensitivity_base_multiple      25
    terminal_sensitivity.base["25x"]        303.54

Both figures are arithmetically correct and they disagree by 2x at the very
multiple the headline claims to use. The headline takes
`min(perpetuity, exit_multiple)` as its terminal value -- for NVDA the
perpetuity was 4.373tn against an exit-multiple 10.286tn, so the perpetuity
won -- while the grid always used the pure exit-multiple terminal value. The
bear row showed the same gap: 60.80 against 119.79.

Nothing in the payload said the two used different terminal methods, and the
grid is keyed by the headline's own multiple. Every reader takes it as a
sensitivity around the headline. It was not one, and the higher number is the
one that ends up in a deck.

The fix makes the grid a real sensitivity of the headline: every cell applies
the same `min(perpetuity, exit_multiple)` rule the headline applies, so the
cell at the base multiple IS the headline price. That is checkable, which a
label is not.

What the old grid was for -- showing how load-bearing the terminal multiple
assumption is -- survives and gets more honest. Where the perpetuity floor
binds at every multiple the grid goes flat, and a flat grid is the true
answer: the exit multiple is doing no work at all, which is precisely what
the old grid hid behind a 255-to-352 range that the model never produced.

The headline's own `min()` rule is deliberate and is not touched.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.analysis_tools import _scenario_dcf_math


def _base_inputs(**overrides):
    inputs = dict(
        ticker="NVDA",
        revenue_base=130_497_000_000.0,
        ebitda_margin=0.62,
        capex_pct_revenue=0.03,
        tax_rate=0.15,
        depreciation=0.023,
        revenue_growth=[0.55, 0.35, 0.25, 0.18, 0.12],
        wacc=0.10,
        terminal_growth=0.03,
        terminal_multiple=25.0,
        cash=43_210_000_000.0,
        debt=10_270_000_000.0,
        shares_outstanding=24_400_000_000.0,
    )
    inputs.update(overrides)
    return inputs


def _run(**overrides):
    return _scenario_dcf_math(
        _base_inputs(**overrides),
        bear_growth=[0.30, 0.18, 0.12, 0.08, 0.05],
        base_growth=[0.55, 0.35, 0.25, 0.18, 0.12],
        bull_growth=[0.70, 0.50, 0.35, 0.25, 0.18],
        bear_margin=0.55, base_margin=0.62, bull_margin=0.68,
    )


CASES = ("bear", "base", "bull")


class TestTheGridBelongsToTheHeadline:
    @pytest.mark.parametrize("case", CASES)
    def test_the_cell_at_the_headline_multiple_is_the_headline_price(self, case):
        """The one cell whose answer is already known must agree with it.

        This is the whole defect in one assertion. If the grid is keyed by the
        headline's multiple, the cell at that multiple has to reproduce the
        headline; anything else is a second valuation wearing the first one's
        label.
        """
        result = _run()
        headline = result[case]["price_per_share"]
        multiple = result["terminal_sensitivity_base_multiple"]
        cell = result["terminal_sensitivity"][case][f"{multiple}x"]

        assert cell == pytest.approx(headline, abs=0.02), (
            f"{case}: the grid says {cell} at {multiple}x while the headline "
            f"says {headline} at the same multiple")

    @pytest.mark.parametrize("case", CASES)
    def test_no_cell_doubles_the_headline_price(self, case):
        """The observed shape of the defect: an exactly-2x disagreement.

        A grid whose cells run to twice the price target is not a sensitivity,
        it is a different model, and the reader has no way to tell.
        """
        result = _run()
        headline = result[case]["price_per_share"]
        cells = list(result["terminal_sensitivity"][case].values())
        assert max(cells) < headline * 1.9, (
            f"{case}: headline {headline}, grid reaches {max(cells)}")


class TestThePayloadSaysWhichMethodItUsed:
    @pytest.mark.parametrize("case", CASES)
    def test_each_scenario_names_the_terminal_method_that_bound(self, case):
        """Which of the two terminal values won the min() is the single fact
        that explains the whole valuation, and it was never reported."""
        result = _run()
        method = result[case].get("terminal_value_method")
        assert method in ("perpetuity", "exit_multiple"), (
            f"{case}: terminal_value_method is {method!r}; the payload does "
            f"not say which terminal value the price was built on")

    @pytest.mark.parametrize("case", CASES)
    def test_each_scenario_shows_both_terminal_values(self, case):
        """The reader must be able to see the min() being taken, not trust it."""
        result = _run()
        perpetuity = result[case].get("terminal_value_perpetuity")
        exit_multiple = result[case].get("terminal_value_exit_multiple")
        used = result[case].get("terminal_value_used")
        assert perpetuity and exit_multiple and used
        assert used == pytest.approx(min(perpetuity, exit_multiple), rel=1e-9)

    def test_the_grid_states_the_rule_it_applies(self):
        note = _run().get("terminal_sensitivity_method")
        assert note, "the grid is unlabelled"
        assert "min" in note.lower() or "lower of" in note.lower(), (
            f"the grid does not say it applies the headline's rule: {note}")


class TestAFlatGridIsAnAnswer:
    def test_a_binding_perpetuity_floor_flattens_the_grid_and_says_so(self):
        """NVDA's own case: the perpetuity is below the exit-multiple terminal
        value at every multiple in the sweep, so the multiple changes nothing.

        That is the true answer and it is worth reporting. The old grid hid it
        behind a 255-to-352 range the model would never have produced.
        """
        result = _run()
        row = result["terminal_sensitivity"]["base"]
        spread = max(row.values()) - min(row.values())
        assert spread == pytest.approx(0.0, abs=0.02), (
            f"expected the floor to bind at every multiple, spread={spread}")
        assert result.get("terminal_sensitivity_floor_note"), (
            "a flat row with no explanation reads as a broken table")

    def test_a_multiple_below_the_floor_still_moves_the_price(self):
        """The sweep is not inert -- where the exit multiple is the binding
        constraint it drives the price, exactly as before."""
        result = _run(terminal_multiple=8.0)
        row = result["terminal_sensitivity"]["base"]
        assert max(row.values()) > min(row.values()), (
            "the grid did not move where the exit multiple binds")


class TestTheOrdinaryShapeSurvives:
    def test_the_headline_prices_are_unchanged(self):
        """The min() rule is deliberate and this change must not touch it."""
        result = _run()
        assert result["base"]["price_per_share"] == pytest.approx(110.93, abs=0.01)
        assert result["bear"]["price_per_share"] == pytest.approx(57.99, abs=0.01)
        assert result["bull"]["price_per_share"] == pytest.approx(174.54, abs=0.01)

    def test_the_grid_is_still_three_rows_of_five_multiples(self):
        grid = _run()["terminal_sensitivity"]
        assert set(grid) == {"bear", "base", "bull"}
        for case in CASES:
            assert set(grid[case]) == {"21.0x", "23.0x", "25.0x", "27.0x", "29.0x"}

    def test_perpetuity_only_mode_still_skips_the_grid(self):
        result = _run(terminal_multiple=0)
        assert "terminal_sensitivity" not in result
        assert "terminal_sensitivity_base_multiple" not in result

    def test_a_missing_share_count_leaves_no_zero_price_in_the_grid(self):
        """`px = eq / shares_v if shares_v > 0 else 0` is the same fallback
        `_dcf_math` stopped making: $0.00 reads as a worthless equity rather
        than an unanswerable question."""
        result = _run(shares_outstanding=0)
        for case in CASES:
            for multiple, price in result["terminal_sensitivity"][case].items():
                assert price is None, (
                    f"{case} {multiple}: reported {price} per share with no "
                    f"share count")
