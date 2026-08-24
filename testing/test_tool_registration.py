"""Every registered tool must have a working handler.

The unit tests for the research tools exercised the underlying functions and
never the MCP handlers wrapping them, so `get_corporate_actions` shipped calling
a serialiser that does not exist in its module. It passed every test and failed
the moment a real MCP client called it.

These tests call each handler with the underlying function stubbed, so they run
offline and catch NameErrors, bad signatures, and unserialisable results.
"""
import asyncio
import json

import pytest


def _run(coro):
    return asyncio.run(coro)


def _payload(result):
    assert result, "handler returned no content"
    text = result[0].text
    return json.loads(text)


def test_web_search_research_handlers_serialise(monkeypatch):
    import tools.web_search_server.web_search as ws

    server = ws.WebSearchServer()
    cases = {
        "get_share_count_series": (
            "get_share_count_series", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_share_count_series("X", 2, "10-Q")),
        "get_shelf_activity": (
            "get_shelf_activity", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_shelf_activity("X", 730)),
        "get_sbc_series": (
            "get_sbc_series", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_sbc_series("X", 2, "10-K")),
        "get_debt_maturity_schedule": (
            "get_debt_maturity_schedule", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_debt_maturity_schedule("X", "10-K")),
        "extract_litigation": (
            "extract_litigation", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.extract_litigation("X", "10-K")),
        "extract_customer_concentration": (
            "extract_customer_concentration", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.extract_customer_concentration("X", "10-K")),
        "find_peers_by_sic": (
            "find_peers_by_sic", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.find_peers_by_sic("X", 20)),
        "get_sic_code": (
            "get_sic_code", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_sic_code("X")),
        "get_accruals_quality": (
            "get_accruals_quality", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_accruals_quality("X", 2, "10-K")),
        "get_working_capital_trends": (
            "get_working_capital_trends", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_working_capital_trends("X", 2, "10-K")),
        "extract_guidance": (
            "extract_guidance", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.extract_guidance("X", 4)),
        "get_operating_leases": (
            "get_operating_leases", lambda *a, **k: {"ticker": "X", "success": True},
            lambda: server.get_operating_leases("X", "10-K")),
    }
    for label, (attr, stub, call) in cases.items():
        monkeypatch.setattr(ws, attr, stub)
        data = _payload(_run(call()))
        assert data["success"] is True, f"{label} handler did not round-trip"


def test_modeling_corporate_actions_handler_serialises(monkeypatch):
    """The regression: this handler called safe_json_dumps, which does not
    exist in analysis_tools. Every unit test passed; the MCP call did not."""
    import tools.financial_modeling_engine.analysis_tools as at

    monkeypatch.setattr(at, "get_corporate_actions",
                        lambda *a, **k: {"ticker": "NVDA", "success": True,
                                         "splits": [], "dividends": []})
    server = at.Financial_Analysis()
    data = _payload(_run(server.get_corporate_actions("NVDA", 6)))
    assert data["success"] is True


def test_modeling_trading_metrics_handler_serialises(monkeypatch):
    """RVOL/ADV/ATR go out over MCP as JSON.

    numpy floats and pandas Timestamps do not serialise on their own, and the
    handler is where that breaks -- never in the unit tests, which look at the
    dict.
    """
    import tools.financial_modeling_engine.analysis_tools as at

    monkeypatch.setattr(at, "get_trading_metrics",
                        lambda *a, **k: {"ticker": "NVDA", "success": True,
                                         "rvol": {}, "adv": {}, "atr": {}})
    server = at.Financial_Analysis()
    data = _payload(_run(server.get_trading_metrics("NVDA", "1y", 20, 14)))
    assert data["success"] is True


def test_every_registered_tool_name_has_a_dispatch_branch():
    """A Tool() entry with no dispatch branch returns 'Unknown tool' at runtime
    while looking perfectly registered in tools/list."""
    import pathlib
    import re

    for path in ("tools/web_search_server/web_search.py",
                 "tools/financial_modeling_engine/analysis_tools.py"):
        source = pathlib.Path(path).read_text()
        declared = set(re.findall(r'Tool\(\s*\n\s*name=["\']([a-z0-9_]+)["\']', source))
        dispatched = set(re.findall(r'name\s*==\s*["\']([a-z0-9_]+)["\']', source))
        missing = declared - dispatched
        assert not missing, f"{path}: declared but never dispatched: {sorted(missing)}"


# --------------------------------------------------------------------------
# Schema/implementation contract.
#
# A tool declaring "required": [] promises an MCP client that it can be called
# with no arguments. Claude reads that schema and does exactly that. If the
# implementation then indexes args[...] with a bracket, the client gets a
# KeyError for obeying the contract the tool published.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name", ["calculate_dcf", "calculate_wacc"])
def test_optional_argument_tools_survive_an_empty_call(tool_name):
    """Both declare "required": [] and describe their numeric inputs as
    'SET TO 0 -- auto-resolved'. Calling with none must not raise."""
    import tools.financial_modeling_engine.analysis_tools as at

    server = at.Financial_Analysis()
    result = _run(getattr(server, tool_name)({}))
    data = _payload(result)
    assert isinstance(data, dict)
    assert "KeyError" not in json.dumps(data)


def test_declared_optional_arguments_are_read_with_get():
    """Guards the whole family rather than the two known cases: a handler whose
    schema says required=[] must not use bracket access on its args dict."""
    import inspect
    import re

    import tools.financial_modeling_engine.analysis_tools as at

    source = inspect.getsource(at)
    offenders = []
    for tool in ("calculate_dcf", "calculate_wacc"):
        match = re.search(rf"async def {tool}\(self, args.*?\n(?=  async def |\nclass )",
                          source, re.DOTALL)
        if not match:
            continue
        for bracket in re.findall(r"args\['([a-z_]+)'\]", match.group(0)):
            offenders.append(f"{tool}: args['{bracket}']")
    assert not offenders, (
        "bracket access on optional args -- these raise KeyError when a client "
        "honours the published schema:\n  " + "\n  ".join(offenders))


def test_dcf_refuses_to_value_a_company_with_no_inputs():
    """Returning price_per_share: 0 is worse than raising.

    A KeyError is obviously broken. A DCF that quietly reports a zero
    enterprise value looks like a real valuation, and it is the kind of answer
    that gets copied into a thesis. calculate_wacc already refuses this way --
    "market_cap + total_debt is zero, cannot compute WACC" -- and the DCF
    should match it.
    """
    import tools.financial_modeling_engine.analysis_tools as at

    data = _payload(_run(at.Financial_Analysis().calculate_dcf({"ticker": "MSFT"})))
    assert "error" in data, f"expected a refusal, got a valuation: {data}"
    assert data.get("price_per_share") in (None, 0) or "error" in data
    assert "revenue_base" in data["error"].lower() or "input" in data["error"].lower()


def test_scenario_dcf_refuses_to_value_a_company_with_no_inputs():
    """calculate_scenario_dcf runs the same _dcf_math three times.

    calculate_dcf refuses a zero revenue_base because a zero price_per_share
    reads as a real valuation. The scenario variant had no such guard, so a
    call with only the three growth arrays and three margins -- every field the
    schema marks required -- returned bear/base/bull prices of 0 and an
    enterprise value of 0.0 with success implied. Three fake price targets are
    worse than one.
    """
    import tools.financial_modeling_engine.analysis_tools as at

    data = _payload(_run(at.Financial_Analysis().calculate_scenario_dcf({
        "ticker": "MSFT",
        "bear_growth": [0.05, 0.04, 0.03, 0.03, 0.02],
        "base_growth": [0.12, 0.11, 0.10, 0.09, 0.08],
        "bull_growth": [0.20, 0.18, 0.16, 0.14, 0.12],
        "bear_margin": 0.45, "base_margin": 0.52, "bull_margin": 0.58,
    })))
    assert "error" in data, f"expected a refusal, got a valuation: {data}"
    assert "revenue_base" in data["error"].lower()
    for case in ("bear", "base", "bull"):
        assert case not in data, (
            f"refusal still carried a {case} price target: {data.get(case)}")


def test_scenario_dcf_still_values_a_company_with_real_inputs():
    """The guard must not swallow a legitimate call."""
    import tools.financial_modeling_engine.analysis_tools as at

    data = _payload(_run(at.Financial_Analysis().calculate_scenario_dcf({
        "ticker": "MSFT",
        "bear_growth": [0.05, 0.04, 0.03, 0.03, 0.02],
        "base_growth": [0.12, 0.11, 0.10, 0.09, 0.08],
        "bull_growth": [0.20, 0.18, 0.16, 0.14, 0.12],
        "bear_margin": 0.45, "base_margin": 0.52, "bull_margin": 0.58,
        "revenue_base": 331_839_000_000,
        "capex_pct_revenue": 0.20, "tax_rate": 0.18, "depreciation": 0.08,
        "wacc": 0.085, "terminal_growth": 0.025, "terminal_multiple": 20.0,
        "cash": 94_000_000_000, "debt": 60_000_000_000,
        "shares_outstanding": 7_430_000_000,
    })))
    assert "error" not in data, f"legitimate call refused: {data}"
    assert data["bear"]["price_per_share"] > 0
    assert data["base"]["price_per_share"] > data["bear"]["price_per_share"]
    assert data["bull"]["price_per_share"] > data["base"]["price_per_share"]
