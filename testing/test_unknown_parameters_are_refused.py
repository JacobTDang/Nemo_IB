"""A parameter name we do not know must not be silently ignored.

`calculate_dcf` subtracts debt and adds cash through parameters named `debt`
and `cash`. Called with `net_debt=9e14` -- a natural name, and the one a
caller reaching for "enterprise value to equity value" would try first -- the
key was accepted, never read, and the tool returned:

    enterprise_value  3,892,969,858
    equity_value      3,792,969,858     <- computed with debt defaulted to 0
    success           true

The answer is confident, well-formed, and wrong by whatever the caller thought
they were passing. The schema already rejects a wrong TYPE ("0.08 is not of
type 'array'"), so the machinery is there; it just permitted extra keys.

This matters most for the calculators, where every parameter is a term in an
arithmetic expression and a dropped one changes the number without changing
its shape. A tool that reads a filing and ignores a stray argument returns the
same filing; a DCF that ignores `net_debt` returns a different valuation.
"""
import asyncio
import importlib
import inspect

import pytest

from tools.manifest import SERVERS, _handler_for

CALCULATORS = ["calculate_dcf", "calculate_wacc", "calculate_lbo",
               "calculate_credit_profile", "calculate_capital_returns",
               "calculate_scenario_dcf"]


def _server():
    spec = [s for s in SERVERS if s.name == "financial"][0]
    return getattr(importlib.import_module(spec.module), spec.cls)()


def _schemas():
    inst = _server()

    async def go():
        listing = _handler_for(inst, "ListTools")(None)
        if inspect.isawaitable(listing):
            listing = await listing
        return {t.name: (t.inputSchema or {}) for t in listing.root.tools}

    return asyncio.run(go())


@pytest.mark.parametrize("tool", CALCULATORS)
def test_a_calculator_declares_its_parameters_closed(tool):
    schema = _schemas().get(tool)
    assert schema is not None, f"{tool} is not declared"
    assert schema.get("additionalProperties") is False, (
        f"{tool} accepts parameters it does not read, so a caller passing "
        f"`net_debt` gets a valuation computed without it")


def test_an_unknown_parameter_is_rejected_rather_than_dropped():
    inst = _server()
    call = _handler_for(inst, "CallTool")

    async def go():
        args = dict(ticker="TEST", revenue_base=1e9, revenue_growth=[0.08] * 5,
                    ebitda_margin=0.30, tax_rate=0.21, capex_pct_revenue=0.05,
                    depreciation=0.04, wacc=0.09, terminal_growth=0.025,
                    cash=5e7, debt=1.5e8, shares_outstanding=1e8,
                    net_debt=9e14)
        req = type("R", (), {"params": type("P", (), {
            "name": "calculate_dcf", "arguments": args})()})()
        out = call(req)
        if inspect.isawaitable(out):
            out = await out
        blocks = getattr(getattr(out, "root", out), "content", out)
        return getattr(blocks[0], "text", "")

    text = asyncio.run(go())
    assert "net_debt" in text, (
        f"an unknown parameter was accepted and dropped: {text[:200]}")


def test_the_correct_parameters_still_work():
    inst = _server()
    call = _handler_for(inst, "CallTool")

    async def go():
        args = dict(ticker="TEST", revenue_base=1e9, revenue_growth=[0.08] * 5,
                    ebitda_margin=0.30, tax_rate=0.21, capex_pct_revenue=0.05,
                    depreciation=0.04, wacc=0.09, terminal_growth=0.025,
                    cash=5e7, debt=1.5e8, shares_outstanding=1e8)
        req = type("R", (), {"params": type("P", (), {
            "name": "calculate_dcf", "arguments": args})()})()
        out = call(req)
        if inspect.isawaitable(out):
            out = await out
        blocks = getattr(getattr(out, "root", out), "content", out)
        return getattr(blocks[0], "text", "")

    import json
    result = json.loads(asyncio.run(go()))
    assert result.get("price_per_share") == pytest.approx(37.93, rel=1e-3)
