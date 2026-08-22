"""A server must not advertise a tool it cannot perform.

Three tools depend on things the homelab image deliberately omits: `search`
needs SearXNG, and `rag_search` / `rag_ingest` need the RAG stack. Listing them
anyway produces two bad outcomes.

`rag_search` at least fails loudly. `search` does not -- with SearXNG absent it
returns `{"search_result": []}`, no error and no success flag, which reads as
"nothing found" for a query that should have plenty. A caller cannot tell that
apart from a genuine empty result.

Advertising capability you do not have is the problem; the fix is to register
these only when their dependency is actually present.
"""
import importlib

import pytest


def _tool_names(monkeypatch, searxng_up, rag_present):
    import tools.web_search_server.web_search as ws
    importlib.reload(ws)
    monkeypatch.setattr(ws, "_searxng_reachable", lambda: searxng_up)
    monkeypatch.setattr(ws, "_rag_available", lambda: rag_present)
    return set(ws.available_tool_names())


def test_search_is_hidden_when_searxng_is_absent(monkeypatch):
    names = _tool_names(monkeypatch, searxng_up=False, rag_present=True)
    assert "search" not in names, (
        "search was advertised without SearXNG; it returns an empty result "
        "list rather than an error, so callers cannot tell it failed")


def test_search_is_offered_when_searxng_is_present(monkeypatch):
    names = _tool_names(monkeypatch, searxng_up=True, rag_present=True)
    assert "search" in names


def test_rag_tools_are_hidden_without_the_rag_stack(monkeypatch):
    names = _tool_names(monkeypatch, searxng_up=True, rag_present=False)
    assert "rag_search" not in names
    assert "rag_ingest" not in names


def test_rag_tools_are_offered_when_available(monkeypatch):
    names = _tool_names(monkeypatch, searxng_up=True, rag_present=True)
    assert {"rag_search", "rag_ingest"} <= names


def test_capability_independent_tools_are_always_offered(monkeypatch):
    """SEC extraction depends on none of this and must never be gated."""
    names = _tool_names(monkeypatch, searxng_up=False, rag_present=False)
    for always in ("get_revenue_base", "get_share_count_series",
                   "extract_litigation", "get_urls_content"):
        assert always in names, f"{always} was gated but has no such dependency"
