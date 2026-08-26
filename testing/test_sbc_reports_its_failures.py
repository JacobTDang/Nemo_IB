"""A failed denominator fetch is still an answer about the tool, not a crash.

`get_sbc_series` wraps its filing walk in a try that catches `ToolTimeout` and
nothing else. The stock-compensation chain itself is guarded, but the revenue
and operating-cash-flow reads that build `pct_of_revenue` are not: a network
failure there leaves the function by raising.

Every other tool in this package answers a failure with a dict that names the
ticker, what it tried, and what went wrong. The MCP layer does turn a raw
exception into an error, so nothing is silently swallowed -- but the caller
gets a framework message instead of the tool's own, without `concepts_tried`
or `wrong_form`, and cannot tell a broken fetch from a filer that tags no SBC.
"""
import pytest

from tools.web_search_server import sbc


def test_a_failed_denominator_fetch_is_reported_not_raised(monkeypatch):
    """The escape only opens when the SBC chain SUCCEEDS and the denominator
    read fails, so the first call has to return a real-shaped result without
    touching the network."""
    calls = {"n": 0}

    def flaky(ticker, concepts, form, limit):
        calls["n"] += 1
        if calls["n"] == 1:
            return ({"2025-01-26": 4_500_000_000.0},
                    "us-gaap:ShareBasedCompensation")
        raise ConnectionError("SEC returned 503")

    monkeypatch.setattr(sbc, "_consolidated_by_filing", flaky)

    result = sbc.get_sbc_series("NVDA")

    assert isinstance(result, dict), "the tool raised instead of answering"
    assert result["success"] is False
    assert "503" in str(result.get("error"))
    assert result.get("ticker") == "NVDA"


def test_the_failure_is_not_phrased_as_a_fact_about_the_filer(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("SEC returned 503")

    monkeypatch.setattr(sbc, "_consolidated_by_filing", boom)
    message = str(sbc.get_sbc_series("NVDA").get("error") or "").lower()

    for forbidden in ("does not tag", "does not disclose", "no stock-based"):
        assert forbidden not in message, (
            f"an outage was described as a non-disclosure: {message[:160]}")


def test_a_timeout_still_reports_itself_as_a_timeout(monkeypatch):
    """The existing ToolTimeout path must survive the broader catch."""
    def slow(*a, **k):
        raise sbc.ToolTimeout("budget of 120s exhausted while walking filings")

    monkeypatch.setattr(sbc, "_consolidated_by_filing", slow)
    result = sbc.get_sbc_series("NVDA")

    assert result["success"] is False
    assert result.get("timed_out") is True
