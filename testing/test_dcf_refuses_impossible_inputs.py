"""A valuation built on an impossible input is worse than no valuation.

Three inputs the DCF accepted without complaint:

    shares_outstanding = 0    -> price_per_share: 0, success: true
    shares_outstanding = -1e8 -> price_per_share: 0, success: true
    revenue_base = -1e9       -> enterprise_value: -3,892,969,858, success: true

The share-count one is the dangerous one. Equity value was $3.79bn and the
answer was "$0.00 per share" -- not an error, not a null, a number that reads
as "this equity is worthless". Zero is the most plausible wrong answer a
share-count guard can produce, which is why dividing by a share count nobody
supplied has to refuse instead.

Negative revenue is not a company. The model runs on it happily and returns a
negative enterprise value, which no caller asked for and no filing supports.

`debt` far above enterprise value is left alone: an over-levered company
genuinely has negative equity value, and saying so is the right answer.
"""
import asyncio
import importlib
import inspect
import json

import pytest

from tools.manifest import SERVERS, _handler_for

BASE = dict(ticker="TEST", revenue_base=1e9, revenue_growth=[0.08] * 5,
            ebitda_margin=0.30, tax_rate=0.21, capex_pct_revenue=0.05,
            depreciation=0.04, wacc=0.09, terminal_growth=0.025,
            cash=5e7, debt=1.5e8, shares_outstanding=1e8)


def _call(tool, args):
    spec = [s for s in SERVERS if s.name == "financial"][0]
    inst = getattr(importlib.import_module(spec.module), spec.cls)()
    handler = _handler_for(inst, "CallTool")

    async def go():
        req = type("R", (), {"params": type("P", (), {
            "name": tool, "arguments": args})()})()
        out = handler(req)
        if inspect.isawaitable(out):
            out = await out
        blocks = getattr(getattr(out, "root", out), "content", out)
        return getattr(blocks[0], "text", "")

    text = asyncio.run(go())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


def test_the_baseline_still_values_the_company():
    result = _call("calculate_dcf", BASE)
    assert result.get("price_per_share") == pytest.approx(37.93, rel=1e-3)


@pytest.mark.parametrize("shares", [0, -1e8])
def test_a_share_count_that_cannot_divide_refuses(shares):
    result = _call("calculate_dcf", {**BASE, "shares_outstanding": shares})

    assert result.get("price_per_share") != 0, (
        "a $3.79bn equity was reported at $0.00 per share")
    assert result.get("price_per_share") is None or result.get("success") is False


def test_the_refusal_says_what_was_wrong():
    result = _call("calculate_dcf", {**BASE, "shares_outstanding": 0})
    message = str(result.get("error") or result.get("note")
                   or result.get("price_per_share_note") or "")
    assert "share" in message.lower(), (
        f"the refusal does not name the share count: {message[:160]}")


def test_the_enterprise_value_survives_a_missing_share_count():
    """The company is still worth what it is worth. Only the per-share figure
    is unanswerable, so only it should go."""
    result = _call("calculate_dcf", {**BASE, "shares_outstanding": 0})
    assert result.get("enterprise_value"), (
        "a missing share count threw away the whole valuation")


def test_negative_revenue_is_not_a_company():
    result = _call("calculate_dcf", {**BASE, "revenue_base": -1e9})
    assert result.get("success") is False, (
        "a company with -$1bn of revenue was valued without complaint")
    assert "revenue" in str(result.get("error") or "").lower()


def test_debt_above_enterprise_value_is_still_answered():
    """An over-levered company genuinely has negative equity value."""
    result = _call("calculate_dcf", {**BASE, "debt": 1e15})
    assert result.get("equity_value") < 0
    assert result.get("success") is not False
