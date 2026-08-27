"""What a slow SEC tool is allowed to cost, and what it must never do.

A QA pass reported `get_working_capital_trends` taking 395 seconds for JPM to
return 1,144 characters, alongside two restarts of the nemo-sec container.

Measured in the running container before any change, JPM at the default
`limit=2`: 19 filing walks, 36 `filing.xbrl()` parses of the same **2**
filings, 189 seconds. Only 37 of those seconds were parsing. The rest was
edgartools rebuilding its enriched fact table, which it memoises per XBRL
object -- so a fresh object per concept rebuilds it once per concept.

The every-concept evaluation in `_series` is not the bug and is not touched
here: it exists because Ford's FY2025 10-K abandoned `NetIncomeLoss`, and
stopping at the first hit reported an 8.2bn loss as +5.9bn of income. What was
wrong is that each of those concepts re-walked EDGAR from scratch.

Two invariants, then:

* the concepts are all still evaluated, but the filings behind them are parsed
  once per filing rather than once per concept;
* a call that runs past its budget says so, names what it was doing, and
  returns nothing that could be mistaken for an answer.

The last test covers the restart hypothesis. `/health` must stay answerable
while a tool is grinding, or a container healthcheck reports a live server as
dead.
"""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from tools.web_search_server import earnings_quality as eq
from tools.web_search_server import sec_series
from tools.web_search_server.sec_series import ConceptFact, FilingPoint


# ------------------------------------------------------------------ fakes
#
# Deliberately not fixtures on the real EDGAR path: the point of these tests is
# to count parses, which means owning the object that gets parsed.


class _FakeXBRL:
    def __init__(self, accession):
        self.accession = accession


class _FakeFilings(list):
    def head(self, n):
        return _FakeFilings(self[:n])


class _FakeFiling:
    def __init__(self, accession, filing_date, counters):
        self.accession_no = accession
        self.filing_date = filing_date
        self.form = "10-K"
        self._counters = counters

    def xbrl(self):
        self._counters["parses"] += 1
        return _FakeXBRL(self.accession_no)


def _install_fake_edgar(monkeypatch, counters, accessions=("acc-2026", "acc-2025")):
    """A two-filing EDGAR that counts parses and concept reads."""
    filings = _FakeFilings(
        _FakeFiling(acc, f"2026-0{i + 1}-15", counters)
        for i, acc in enumerate(accessions))

    class _FakeCompany:
        def __init__(self, ticker):
            counters["companies"] += 1
            self.ticker = ticker

        def get_filings(self, form=None, amendments=True, **kwargs):
            counters["filing_lists"] += 1
            counters["amendments"] = amendments
            return filings

    monkeypatch.setattr(sec_series, "Company", _FakeCompany)
    monkeypatch.setattr(sec_series, "_require_identity", lambda: "test@example.invalid")
    monkeypatch.setattr(sec_series, "_throttle", lambda: None)
    return filings


def _tagged(values):
    """concept_point stub: `values` maps concept -> numeric value, else None."""
    def fake(xbrl, concept, filing_date, form, accession=""):
        fake.reads += 1
        if concept not in values:
            return None
        end = "2026-06-30" if accession.endswith("2026") else "2025-06-30"
        start = "2025-07-01" if accession.endswith("2026") else "2024-07-01"
        period = end if concept in _INSTANTS else f"duration_{start}_{end}"
        return FilingPoint(filing_date, form, accession,
                           facts=[ConceptFact(values[concept], period, {}, "c-1")])
    fake.reads = 0
    return fake


_INSTANTS = {
    "us-gaap:AccountsReceivableNetCurrent",
    "us-gaap:InventoryNet",
    "us-gaap:AccountsPayableCurrent",
    "us-gaap:Assets",
}

_WC_VALUES = {
    "us-gaap:Revenues": 400.0,
    "us-gaap:CostOfRevenue": 300.0,
    "us-gaap:AccountsReceivableNetCurrent": 100.0,
    "us-gaap:InventoryNet": 50.0,
    "us-gaap:AccountsPayableCurrent": 60.0,
}


# ============================================ one parse per filing, not per concept

def test_a_filing_is_parsed_once_however_many_concepts_are_read(monkeypatch):
    """The measured bug: 19 concepts caused 36 parses of the same 2 filings.

    `get_working_capital_trends` reads 19 concepts. Two filings must cost two
    parses, not thirty-eight.
    """
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    reader = _tagged(_WC_VALUES)
    monkeypatch.setattr(sec_series, "concept_point", reader)

    result = eq.get_working_capital_trends("FAKE", limit=2)

    assert result["success"] is True, result.get("error")
    assert counters["parses"] == 2, (
        f"expected one parse per filing, got {counters['parses']} for 2 "
        f"filings -- the per-concept walk is back")
    # Every concept is still evaluated: that is the Ford rule, and it is the
    # thing this speedup must not have bought its speed with.
    assert reader.reads == 2 * len(result["concepts_tried"])


def test_every_concept_in_the_chain_is_still_evaluated(monkeypatch):
    """Freshness, not first-hit, still decides -- see `_series`.

    An older filing tagging `Revenues` must not beat the newest filing's
    `RevenueFromContractWithCustomerExcludingAssessedTax` simply by sitting
    earlier in the chain, and vice versa.
    """
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)

    def reader(xbrl, concept, filing_date, form, accession=""):
        # The ASC 606 element covers only the older year; Revenues carries the
        # newest. Chain order prefers the 606 element, freshness must not.
        newest = accession.endswith("2026")
        if concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax":
            if newest:
                return None
            return FilingPoint(filing_date, form, accession, facts=[
                ConceptFact(111.0, "duration_2024-07-01_2025-06-30", {}, "c")])
        if concept == "us-gaap:Revenues":
            end, start = ("2026-06-30", "2025-07-01") if newest else (
                "2025-06-30", "2024-07-01")
            return FilingPoint(filing_date, form, accession, facts=[
                ConceptFact(999.0, f"duration_{start}_{end}", {}, "c")])
        if concept in _WC_VALUES:
            end = "2026-06-30" if newest else "2025-06-30"
            return FilingPoint(filing_date, form, accession,
                               facts=[ConceptFact(_WC_VALUES[concept], end, {}, "c")])
        return None

    monkeypatch.setattr(sec_series, "concept_point", reader)
    result = eq.get_working_capital_trends("FAKE", limit=2)

    assert result["success"] is True, result.get("error")
    assert result["concepts_used"]["revenue"] == "us-gaap:Revenues", (
        "the chain stopped at the first hit; the stale element won the latest "
        "period's label, which is the Ford failure")
    assert result["latest"]["revenue"] == 999.0


def test_the_newest_filing_is_parsed_once_across_mixed_limits(monkeypatch):
    """`get_operating_leases` asks for `limit` filings, then 1 per bucket.

    Keyed on accession rather than on (form, limit), so the newest filing is
    parsed once and not once more for every maturity bucket.
    """
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setattr(sec_series, "concept_point", _tagged({
        "us-gaap:OperatingLeaseLiability": 1000.0,
        "us-gaap:OperatingLeaseRightOfUseAsset": 900.0,
    }))

    result = eq.get_operating_leases("FAKE", limit=2)

    assert result["success"] is True, result.get("error")
    assert counters["parses"] == 2, (
        f"{counters['parses']} parses for 2 filings; the limit=1 bucket "
        f"lookups re-parsed the newest filing")


def test_accruals_shares_the_same_walk(monkeypatch):
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setattr(sec_series, "concept_point", _tagged({
        "us-gaap:NetIncomeLoss": 100.0,
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": 120.0,
        "us-gaap:Assets": 2000.0,
    }))

    result = eq.get_accruals_quality("FAKE", limit=2)

    assert result["success"] is True, result.get("error")
    assert counters["parses"] == 2


def test_the_walk_still_excludes_amendments(monkeypatch):
    """A 10-K/A carrying only Part III must not take a slot in the walk."""
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setattr(sec_series, "concept_point", _tagged(_WC_VALUES))

    eq.get_working_capital_trends("FAKE", limit=2)

    assert counters["amendments"] is False


# ============================================================ the budget

def test_a_call_past_its_budget_fails_loudly(monkeypatch):
    """No partial result, and the message names what it was doing."""
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0.05")

    def slow(xbrl, concept, filing_date, form, accession=""):
        time.sleep(0.02)
        return None

    monkeypatch.setattr(sec_series, "concept_point", slow)
    result = eq.get_working_capital_trends("FAKE", limit=2)

    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["periods"] == []
    assert result["latest"] is None
    message = result["error"]
    assert "budget" in message
    assert "get_working_capital_trends(FAKE)" in message
    assert "No partial result" in message


def test_a_timeout_is_never_reported_as_a_coverage_gap(monkeypatch):
    """The rule test_outage_is_not_a_finding enforces, applied to the clock.

    A chain abandoned to the budget was not evaluated. Saying "FAKE does not
    tag receivables" would be an affirmative claim about a filer, made on the
    strength of a stopwatch.
    """
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0.05")

    def slow(xbrl, concept, filing_date, form, accession=""):
        time.sleep(0.02)
        return None

    monkeypatch.setattr(sec_series, "concept_point", slow)

    for tool in (eq.get_working_capital_trends, eq.get_accruals_quality,
                 eq.get_operating_leases):
        result = tool("FAKE", limit=2)
        assert result["success"] is False, tool.__name__
        assert result.get("coverage") != "not_covered", tool.__name__
        message = (result.get("error") or "").lower()
        for claim in ("does not tag", "does not disclose", "not covered",
                      "does not report"):
            assert claim not in message, (
                f"{tool.__name__} reported a timeout as a fact about the "
                f"filer: {message}")


def test_a_budget_is_generous_by_default_and_tunable(monkeypatch):
    monkeypatch.delenv("NEMO_SEC_TOOL_BUDGET_S", raising=False)
    assert eq._budget_seconds() == eq.DEFAULT_BUDGET_SECONDS

    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "45")
    assert eq._budget_seconds() == 45.0

    # An explicit zero is "no bound"; a typo is not.
    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0")
    assert eq._budget_seconds() == 0.0
    for bad in ("", "   ", "abc", "-30"):
        monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", bad)
        assert eq._budget_seconds() == eq.DEFAULT_BUDGET_SECONDS, bad


def test_an_unbounded_budget_does_not_time_out(monkeypatch):
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setenv("NEMO_SEC_TOOL_BUDGET_S", "0")
    monkeypatch.setattr(sec_series, "concept_point", _tagged(_WC_VALUES))

    result = eq.get_working_capital_trends("FAKE", limit=2)
    assert result["success"] is True


# =================================================== the seam the tests use

def test_replacing_fetch_concept_series_still_works(monkeypatch):
    """29 tests in test_earnings_quality.py patch that name.

    The shared walk must stand down when it has been replaced, or every one of
    them would silently start talking to EDGAR.
    """
    calls = []

    def stub(ticker, concept, form="10-K", limit=3):
        calls.append(concept)
        raise sec_series.NotCovered(concept)

    monkeypatch.setattr(eq, "fetch_concept_series", stub)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the shared walk ran despite a replaced seam")

    monkeypatch.setattr(sec_series, "Company", explode)

    result = eq.get_working_capital_trends("FAKE")
    assert result["success"] is False
    assert calls, "the replaced seam was never called"


def test_a_pooled_thread_does_not_inherit_the_previous_walk(monkeypatch):
    """asyncio.to_thread reuses workers; a leaked walk would cross tickers."""
    counters = {"parses": 0, "companies": 0, "filing_lists": 0}
    _install_fake_edgar(monkeypatch, counters)
    monkeypatch.setattr(sec_series, "concept_point", _tagged(_WC_VALUES))

    eq.get_working_capital_trends("FIRST", limit=2)
    assert getattr(eq._ACTIVE, "walk", None) is None, (
        "the walk outlived the call that opened it")

    eq.get_working_capital_trends("SECOND", limit=2)
    assert counters["companies"] == 2, (
        "the second ticker was served the first ticker's filings")


# ================================================== /health under a slow call

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(blocking: bool):
    """A real mcp_http app whose one tool takes 3 seconds.

    Built through `build_app` rather than mocked, because the question is
    whether *that* app keeps answering /health -- which is what the container
    healthcheck polls, and the only thing standing between a slow tool and a
    restart loop.

    `blocking` picks which of the two shapes the handler has: the work on the
    event loop, or the work on a worker thread. Both are three seconds of the
    same sleep, so any difference at /health is the shape and nothing else.
    """
    import anyio
    import uvicorn
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    from tools.mcp_http import build_app

    started = threading.Event()
    server = Server("slow-tool-test")

    @server.list_tools()
    async def list_tools():
        return [Tool(name="grind", description="takes three seconds",
                     inputSchema={"type": "object", "properties": {}})]

    @server.call_tool()
    async def call_tool(name, args):
        started.set()
        if blocking:
            # The shape the QA report hypothesised: a synchronous call awaited
            # nowhere, holding the event loop for its whole duration.
            time.sleep(3.0)
        else:
            # The shape web_search.py actually uses for the SEC tools.
            await anyio.to_thread.run_sync(lambda: time.sleep(3.0))
        return [TextContent(type="text", text="done")]

    app = build_app(server, auth_token=None, json_response=True)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.05)
    else:  # pragma: no cover - the server never came up
        pytest.fail("the test server never bound its port")

    try:
        yield port, started
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def threaded_tool_server():
    yield from _serve(blocking=False)


@pytest.fixture
def blocking_tool_server():
    yield from _serve(blocking=True)


def _probe_health_while_calling(port, started, seconds):
    """Run the slow tool; poll /health the way the healthcheck does.

    Returns (latencies, timeouts, call errors). A probe that exceeds the
    healthcheck's 3s timeout is recorded rather than raised, because the
    control case is supposed to produce them.
    """
    errors = []

    def grind():
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "grind", "arguments": {}}}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/mcp/", data=body,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json, text/event-stream"})
        try:
            urllib.request.urlopen(request, timeout=30).read()
        except Exception as exc:  # pragma: no cover - reported, not swallowed
            errors.append(exc)

    worker = threading.Thread(target=grind, daemon=True)
    worker.start()
    assert started.wait(timeout=10), "the slow tool never started"

    # Poll /health exactly as the healthcheck does, for as long as the tool runs.
    latencies, timeouts = [], 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        began = time.monotonic()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=3) as response:
                assert json.loads(response.read())["status"] == "ok"
            latencies.append(time.monotonic() - began)
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            timeouts += 1
        time.sleep(0.1)

    worker.join(timeout=30)
    return latencies, timeouts, errors


def test_health_stays_answerable_while_a_tool_grinds(threaded_tool_server):
    """The restart hypothesis, made testable.

    The compose healthcheck gives /health a 3s timeout, every 30s, 3 retries.
    Every SEC tool in web_search.py hands its blocking work to
    `asyncio.to_thread`, so the event loop stays free to answer the probe.
    """
    port, started = threaded_tool_server
    latencies, timeouts, errors = _probe_health_while_calling(port, started, 2.5)

    assert not errors, errors
    assert latencies, "no health probe completed"
    assert timeouts == 0, f"{timeouts} health probes timed out"
    assert max(latencies) < 3.0, (
        f"/health took {max(latencies):.1f}s while a tool was running; the "
        f"container healthcheck allows 3s, so this would fail the probe")


def test_a_handler_on_the_event_loop_would_fail_the_healthcheck(
        blocking_tool_server):
    """The control, and the reason the threading above is load-bearing.

    Same three seconds of sleep, run on the event loop instead of a worker.
    /health stops answering for the duration -- which is what would have to be
    true for the QA report's restart mechanism to hold. Asserting it here
    means the passing test above is evidence rather than decoration.
    """
    port, started = blocking_tool_server
    latencies, timeouts, errors = _probe_health_while_calling(port, started, 2.5)

    assert not errors, errors
    stalled = timeouts > 0 or (latencies and max(latencies) > 1.0)
    assert stalled, (
        "a handler blocking the event loop did not delay /health at all; "
        "either uvicorn grew a second worker or this control no longer "
        "controls for anything")
