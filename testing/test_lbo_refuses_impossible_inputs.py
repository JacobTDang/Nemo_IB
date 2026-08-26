"""An LBO that cannot exist must refuse, not report a clean total loss.

Three inputs `_lbo_math` accepted without complaint:

    ebitda_margin = 0    -> entry_ebitda 0.0, entry_multiple 0,
                            debt_amount 0.0, leverage_turns_entry 5,
                            moic 0.0, success true
    exit_multiple = -5   -> exit_ev -241,576,500,000, equity_proceeds 0.0,
                            moic 0.0, irr_pct -100.0, success true
    hold_years = 0       -> "Failed to call tool 'calculate_lbo':
                            float division by zero"

`entry_multiple: 0` claims a $5.1 trillion company was bought at 0x EBITDA.
It is the same fallback `calculate_dcf` stopped making when it divided a
$3.79bn equity by a share count nobody supplied and answered "$0.00 per
share": zero is the most plausible wrong answer a guard on a division can
produce. There the enterprise value survived and only the per-share figure
went. Here nothing survives -- debt is sized off EBITDA, the exit is priced
off EBITDA, and the sweep services debt out of EBITDA -- so the refusal has
to be the whole model, matching the guard that already refuses a structure
whose debt exceeds its purchase price.

A negative exit multiple manufactures a negative enterprise value, which is
the class `get_market_data` already suppresses: it refuses to build EV
multiples on BRK-B's -233.9bn because a multiple over a non-positive
numerator has no ordering. Here the tool invents one from an input it should
have rejected, then floors `equity_proceeds` to 0.0 so the output reads as an
ordinary wiped-out deal. A client charting MOIC across exit multiples gets a
clean 0.0 where they should get an error.

A zero hold period is as un-modelable as an unfundable structure. Leaking
`float division by zero` from `moic ** (1.0 / hold_years)` tells the caller
nothing about which input was wrong.

The over-leverage guard already in place is the template for all three:
refuse, name the input, say what to change.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.financial_modeling_engine.analysis_tools as at


def _args(**overrides):
    args = {
        "ticker": "TEST",
        "entry_ev": 200_000_000_000.0,
        "revenue_base": 50_000_000_000.0,
        "ebitda_margin": 0.35,
        "capex_pct_revenue": 0.05,
        "depreciation": 0.04,
        "tax_rate": 0.21,
        "revenue_growth": [0.08, 0.07, 0.06, 0.05, 0.04],
        "debt_interest_rate": 0.09,
        "leverage_turns": 5.0,
        "exit_multiple": 11.0,
        "hold_years": 5,
    }
    args.update(overrides)
    return args


def _math(**overrides):
    args = _args(**overrides)
    args.pop("ticker")
    return at._lbo_math(**args)


def _handler(**overrides):
    blocks = asyncio.run(at.Financial_Analysis().calculate_lbo(_args(**overrides)))
    assert blocks, "handler returned no content"
    return json.loads(blocks[0].text)


class TestAnEbitdaOfZeroIsNotAZeroTimesPurchase:
    def test_the_math_refuses_rather_than_paying_zero_turns(self):
        with pytest.raises(ValueError) as exc:
            _math(ebitda_margin=0)
        message = str(exc.value).lower()
        assert "ebitda" in message, message
        assert "ebitda_margin" in message or "revenue_base" in message, (
            f"the refusal does not name the input to change: {message}")

    def test_the_handler_returns_a_refusal_not_a_zero_entry_multiple(self):
        data = _handler(entry_ev=5_115_451_801_600.0, revenue_base=1e11,
                        ebitda_margin=0)
        assert data.get("entry_multiple") != 0, (
            "a $5.1tn purchase was reported at 0x EBITDA")
        assert "error" in data, f"expected a refusal, got a model: {data}"
        assert data.get("ticker") == "TEST"

    def test_no_moic_or_irr_is_published_for_a_deal_that_cannot_exist(self):
        data = _handler(ebitda_margin=0)
        assert "moic" not in data and "irr_pct" not in data
        assert "achieves_20pct_irr" not in data

    def test_leverage_turns_can_never_stand_beside_zero_debt(self):
        """`leverage_turns_entry: 5` and `debt_amount: 0.0` in one object are
        two claims that cannot both be true."""
        data = _handler(ebitda_margin=0)
        if "debt_amount" in data and data["debt_amount"] == 0:
            assert not data.get("leverage_turns_entry"), (
                f"{data.get('leverage_turns_entry')}x of leverage on "
                f"{data['debt_amount']} of debt")


class TestAnExitMultipleMustBePositive:
    @pytest.mark.parametrize("bad", [-5.0, -0.5, 0.0])
    def test_the_math_refuses_it(self, bad):
        with pytest.raises(ValueError) as exc:
            _math(exit_multiple=bad)
        assert "exit_multiple" in str(exc.value)

    def test_no_negative_enterprise_value_is_manufactured(self):
        data = _handler(exit_multiple=-5.0)
        assert "error" in data, f"expected a refusal, got a model: {data}"
        assert data.get("exit_ev") is None or data.get("exit_ev", 0) >= 0

    def test_the_loss_is_not_dressed_up_as_an_ordinary_total_loss(self):
        """`equity_proceeds` floored to 0.0 makes an invalid input look like a
        deal that merely went to zero -- indistinguishable on a MOIC chart."""
        data = _handler(exit_multiple=-5.0)
        assert data.get("moic") != 0.0
        assert data.get("irr_pct") != -100.0

    def test_a_positive_multiple_below_the_entry_is_still_a_real_answer(self):
        """Buying at 11x and exiting at 6x is a bad deal, not an impossible
        one, and the model must still price it."""
        result = _math(exit_multiple=6.0)
        assert result["exit_ev"] > 0
        assert result["moic"] < 1.0


class TestAHoldPeriodMustHaveLength:
    @pytest.mark.parametrize("bad", [0, -1])
    def test_the_math_refuses_instead_of_dividing_by_zero(self, bad):
        with pytest.raises(ValueError) as exc:
            _math(hold_years=bad)
        assert "hold_years" in str(exc.value)

    def test_the_handler_names_the_input_not_the_stack_trace(self):
        data = _handler(hold_years=0)
        assert "error" in data, f"expected a refusal, got: {data}"
        assert "hold_years" in data["error"]
        assert "division by zero" not in data["error"].lower(), (
            f"a stack-trace fragment leaked to the caller: {data['error']}")

    def test_a_one_year_hold_is_still_modelled(self):
        result = _math(hold_years=1)
        assert result["hold_years"] == 1
        assert result["moic"] > 0


class TestTheOrdinaryDealIsUnchanged:
    def test_a_normal_lbo_produces_the_same_numbers(self):
        result = _math()
        assert result["entry_ebitda"] == pytest.approx(17_500_000_000.0)
        assert result["entry_multiple"] == pytest.approx(11.43)
        assert result["debt_amount"] == pytest.approx(87_500_000_000.0)
        assert result["equity_invested"] == pytest.approx(112_500_000_000.0)
        assert result["exit_ev"] == pytest.approx(257_493_796_560.0)
        assert result["equity_proceeds"] == pytest.approx(215_908_207_314.43)
        assert result["moic"] == pytest.approx(1.92)
        assert result["irr_pct"] == pytest.approx(13.93)

    def test_the_unfundable_structure_guard_still_fires(self):
        with pytest.raises(ValueError) as exc:
            _math(entry_ev=500_000_000.0, revenue_base=1_000_000_000.0,
                  ebitda_margin=0.20, leverage_turns=4.5)
        assert "cannot be funded" in str(exc.value)
