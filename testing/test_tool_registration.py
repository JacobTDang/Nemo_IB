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
