"""An LBO whose debt alone exceeds the purchase price is not a deal.

_lbo_math sizes acquisition debt off EBITDA (leverage_turns x entry EBITDA) but
sizes the equity cheque off EV, with a 10% floor:

    equity_invested = max(entry_ev - debt_amount, entry_ev * 0.10)

When the entry multiple is below the leverage turns, debt_amount exceeds
entry_ev outright and the floor silently invents an equity cheque. The model
then reports a fictitious MOIC and IRR with achieves_20pct_irr True -- the same
class of answer calculate_dcf already refuses to produce.

The floor itself is a real modelling convention for thin-equity deals and stays
in place; only the arithmetically impossible structure is refused.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.financial_modeling_engine.analysis_tools as at


def _run(coro):
    return asyncio.run(coro)


def _payload(result):
    assert result, "handler returned no content"
    return json.loads(result[0].text)


def _args(entry_ev, leverage_turns=4.5):
    return {
        "ticker": "TEST",
        "entry_ev": entry_ev,
        "revenue_base": 1_000_000_000.0,
        "ebitda_margin": 0.20,          # entry EBITDA = 200m
        "capex_pct_revenue": 0.05,
        "depreciation": 0.03,
        "tax_rate": 0.21,
        "revenue_growth": [0.06, 0.05, 0.04, 0.04, 0.03],
        "debt_interest_rate": 0.08,
        "leverage_turns": leverage_turns,
        "exit_multiple": 10.0,
        "hold_years": 5,
    }


def test_math_refuses_a_structure_whose_debt_exceeds_the_entry_ev():
    # 200m EBITDA at 4.5 turns = 900m of debt against a 500m purchase price.
    a = _args(entry_ev=500_000_000.0)
    with pytest.raises(ValueError) as exc:
        at._lbo_math(
            entry_ev=a["entry_ev"], revenue_base=a["revenue_base"],
            ebitda_margin=a["ebitda_margin"],
            capex_pct_revenue=a["capex_pct_revenue"],
            depreciation=a["depreciation"], tax_rate=a["tax_rate"],
            revenue_growth=a["revenue_growth"],
            debt_interest_rate=a["debt_interest_rate"],
            leverage_turns=a["leverage_turns"],
            exit_multiple=a["exit_multiple"], hold_years=a["hold_years"],
        )
    message = str(exc.value).lower()
    assert "debt" in message and "entry_ev" in message


def test_handler_returns_an_explicit_refusal_not_a_fictitious_irr():
    data = _payload(_run(at.Financial_Analysis().calculate_lbo(
        _args(entry_ev=500_000_000.0))))
    assert "error" in data, f"expected a refusal, got a model: {data}"
    assert "irr_pct" not in data
    assert "moic" not in data
    assert data.get("ticker") == "TEST"


def test_a_fundable_structure_still_models_normally():
    # 200m EBITDA at 10x = 2bn EV, 900m debt, 1.1bn equity.
    data = _payload(_run(at.Financial_Analysis().calculate_lbo(
        _args(entry_ev=2_000_000_000.0))))
    assert "error" not in data, f"fundable deal refused: {data}"
    assert data["debt_amount"] == pytest.approx(900_000_000.0)
    assert data["equity_invested"] == pytest.approx(1_100_000_000.0)
    assert data["moic"] > 0


def test_thin_equity_floor_is_preserved_below_the_impossible_threshold():
    """equity < 10% of EV is a convention, not an impossibility -- keep it."""
    # 200m EBITDA at 4.5 turns = 900m debt against a 950m EV -> 50m of real
    # equity, floored up to 95m.
    result = at._lbo_math(
        entry_ev=950_000_000.0, revenue_base=1_000_000_000.0,
        ebitda_margin=0.20, capex_pct_revenue=0.05, depreciation=0.03,
        tax_rate=0.21, revenue_growth=[0.06, 0.05, 0.04, 0.04, 0.03],
        debt_interest_rate=0.08, leverage_turns=4.5, exit_multiple=10.0,
        hold_years=5,
    )
    assert result["debt_amount"] == pytest.approx(900_000_000.0)
    assert result["equity_invested"] == pytest.approx(95_000_000.0)
