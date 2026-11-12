"""Quiver server end-to-end protocol smoke.

Boots tools/quiver_server as a real subprocess via stdio and exercises
every tool through the MCP JSON-RPC framing, against the live
quiverquant.com (public, no-auth data). Guards against the class of bugs
unit tests miss: TextContent serialization, the annotating dispatcher,
async ordering at the protocol layer, and envelope shape (success /
coverage / provider / warnings).

Network calls are kept small (limit=2) and every assertion tolerates the
site's own gaps (e.g. a ticker with no mentions) — the contract under
test is the framing and the honesty envelope, not Quiver's content.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _exercise_all_tools() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.quiver_server.server", "server"],
        env=env,
    )

    results = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            results["__tool_count__"] = len(tools.tools)
            results["__tool_names__"] = sorted(t.name for t in tools.tools)

            r = await session.call_tool("get_quiver_congress_trades", {"limit": 2})
            results["get_quiver_congress_trades"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_congress_trades", {
                "ticker": "NVDA", "limit": 2})
            results["get_quiver_congress_trades_filtered"] = json.loads(
                r.content[0].text)

            r = await session.call_tool("get_quiver_politician_trades", {
                "politician": "Nancy Pelosi", "limit": 2})
            results["get_quiver_politician_trades"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_wsb_sentiment", {
                "ticker": "NVDA", "include_history": True})
            results["get_quiver_wsb_sentiment"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_lobbying", {"limit": 2})
            results["get_quiver_lobbying"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_gov_contracts",
                                        {"limit": 2})
            results["get_quiver_gov_contracts"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_ticker_snapshot", {
                "ticker": "NVDA", "trade_limit": 2})
            results["get_quiver_ticker_snapshot"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_insider_trades", {"limit": 2})
            results["get_quiver_insider_trades"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_congress_net_worth",
                                        {"limit": 2})
            results["get_quiver_congress_net_worth"] = json.loads(
                r.content[0].text)

            r = await session.call_tool("get_quiver_bills", {
                "query": "semiconductor", "limit": 2})
            results["get_quiver_bills"] = json.loads(r.content[0].text)

            # failure vocabulary: unknown entity must be success=false
            # with a machine reason, not an exception over the wire.
            r = await session.call_tool("get_quiver_politician_trades", {
                "politician": "Zzqqxx Nobody"})
            results["unknown_politician"] = json.loads(r.content[0].text)

    return results


def test_quiver_server_e2e():
    results = asyncio.run(_exercise_all_tools())

    assert results["__tool_count__"] == 9, results["__tool_names__"]
    assert "get_quiver_congress_trades" in results["__tool_names__"]

    # Every response shares the envelope contract.
    for name in ["get_quiver_congress_trades", "get_quiver_congress_trades_filtered",
                 "get_quiver_politician_trades", "get_quiver_wsb_sentiment",
                 "get_quiver_lobbying", "get_quiver_gov_contracts",
                 "get_quiver_ticker_snapshot", "get_quiver_insider_trades",
                 "get_quiver_congress_net_worth", "get_quiver_bills"]:
        body = results[name]
        assert body["success"] is True, (name, body.get("error"))
        assert body["domain"] == "quiver", name
        assert body["provider"], name
        assert isinstance(body["warnings"], list), name

    # Real data actually parsed out of the pages.
    trades = results["get_quiver_congress_trades"]["data"]["trades"]
    assert len(trades) == 2
    assert trades[0]["ticker"], trades
    assert trades[0]["filed_date"], trades

    filtered = results["get_quiver_congress_trades_filtered"]["data"]
    assert all(t["ticker"] == "NVDA" for t in filtered["trades"]), filtered

    pelosi = results["get_quiver_politician_trades"]["data"]
    assert pelosi["total"] > 50, pelosi  # full history, not a page slice
    assert pelosi["trades"][0]["tx_type"], pelosi

    wsb = results["get_quiver_wsb_sentiment"]["data"]
    assert wsb["data_horizon"], wsb  # staleness is always dated
    assert wsb["ticker_row"]["ticker"] == "NVDA", wsb
    assert wsb["history"], wsb  # daily mention series came back

    lobbying = results["get_quiver_lobbying"]["data"]
    assert lobbying["filings"], lobbying
    assert lobbying["top_spenders"], lobbying
    # normalized dates, not "Nov. 9, 2026, 3:10 p.m."
    assert len(lobbying["filings"][0]["date"]) == 10, lobbying["filings"][0]

    awards = results["get_quiver_gov_contracts"]["data"]
    assert awards["awards"], awards
    assert awards["awards"][0]["agency"], awards["awards"][0]

    snap = results["get_quiver_ticker_snapshot"]["data"]
    assert snap["price"]["price_data"]["current_price"], snap
    assert snap["insiders"]["total_trades"] > 0, snap
    assert snap["institutions"], snap
    # bulls_bears is session-gated: present only with a cookie, and when
    # absent it must be named, never silently missing.
    if "bulls_bears" not in snap:
        assert "bulls_bears:no_auth" in snap.get("degraded_sections", []), snap

    # Insider trades: the union of the dashboard's two lists, newest first,
    # with the disclosure window stated rather than implied.
    insiders = results["get_quiver_insider_trades"]["data"]
    assert insiders["total_available"] > 1000, insiders["total_available"]
    assert insiders["returned"] == 2
    assert insiders["date_range"]["newest"], insiders
    assert insiders["trades"][0]["transaction_date"], insiders["trades"][0]
    assert insiders["trades"][0]["list"], insiders["trades"][0]

    # Congressional net worth: ranked, richest first, with the id a caller
    # needs to follow up.
    worth = results["get_quiver_congress_net_worth"]["data"]
    assert worth["total_members"] > 400, worth["total_members"]
    assert worth["members"][0]["rank"] == 1, worth["members"][0]
    assert worth["members"][0]["net_worth_usd"] > 0, worth["members"][0]
    assert worth["members"][0]["bioguide_id"], worth["members"][0]

    # Bills: a searchable index, with paging reported rather than assumed.
    bills = results["get_quiver_bills"]["data"]
    assert bills["total_indexed"] > 10000, bills["total_indexed"]
    assert bills["matching"] >= bills["returned"], bills
    assert all("semiconductor" in b["title"].lower() for b in bills["bills"])

    # Unknown entity -> success=false with a machine-readable reason.
    miss = results["unknown_politician"]
    assert miss["success"] is False
    assert miss["data"]["reason"] == "unknown_politician", miss


# ---------------------------------------------------------------------------
# Transport honesty. Offline: a scripted transport stands in for the site, so
# these run without a network and without depending on Quiver's rate limiter
# actually firing.
# ---------------------------------------------------------------------------

def _client_with(handler):
    import httpx
    from tools.quiver_server import client as qc
    c = qc.QuiverWebClient()
    c._client = httpx.Client(base_url=qc.BASE_URL,
                             transport=httpx.MockTransport(handler))
    return c


def test_a_rate_limit_is_retried_once_and_then_named():
    """One 429 is a burst, not an outage: retry it, then say what it was."""
    import httpx
    from tools.quiver_server.client import QuiverError

    attempts = []

    def once(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"},
                                  text="slow down")
        return httpx.Response(200, text='{"response": "ok"}')

    c = _client_with(once)
    assert json.loads(c._fetch("/get_current_price/NVDA")) == {"response": "ok"}
    assert len(attempts) == 2, attempts

    # A second 429 in the same call is the site refusing, and the caller is
    # told so instead of being held open for another attempt.
    attempts.clear()
    c = _client_with(lambda request: (
        attempts.append(request.url.path),
        httpx.Response(429, headers={"retry-after": "0"}, text="nope"),
    )[1])
    try:
        c._fetch("/get_current_price/NVDA")
    except QuiverError as exc:
        assert exc.reason == "rate_limited", exc.reason
    else:
        raise AssertionError("a refused request must not return a body")
    assert len(attempts) == 2, attempts


def test_a_rate_limit_is_not_reported_as_an_unknown_ticker():
    """The reason names what happened. A throttled request says nothing about
    whether the ticker exists, and `unknown_ticker` would be a wrong answer."""
    from tools.quiver_server.client import QuiverError

    c = _client_with(lambda request: None)

    def throttled(path, *args, **kwargs):
        raise QuiverError("rate_limited", f"{path} returned HTTP 429")

    c._json = throttled
    try:
        c.get_ticker_snapshot("NVDA")
    except QuiverError as exc:
        assert exc.reason == "rate_limited", exc.reason
    else:
        raise AssertionError("a snapshot with no answering section must raise")

    # A ticker Quiver does not cover is the one case that earns that name.
    def missing(path, *args, **kwargs):
        raise QuiverError("not_found", f"{path} returned HTTP 404")

    c._json = missing
    try:
        c.get_ticker_snapshot("ZZZZZ")
    except QuiverError as exc:
        assert exc.reason == "unknown_ticker", exc.reason
    else:
        raise AssertionError("a ticker with no endpoints must raise")


def test_a_ticker_with_no_data_is_not_returned_as_six_empty_sections():
    """Quiver answers 200 with "N/A" and empty arrays for a ticker it does not
    cover. Handing that back would read as a covered company with no insiders,
    no institutions and no analyst coverage."""
    from tools.quiver_server.client import QuiverError, _snapshot_is_hollow

    c = _client_with(lambda request: None)

    def hollow(path, *args, **kwargs):
        if "bulls_bears" in path:
            return {"response": "empty"}
        if "insiders" in path:
            return {"insider_trades": []}
        if "institutions" in path:
            return {"response": {}}
        if "forecast" in path:
            return {"data": [], "price_targets": {}}
        if "tv_lazy" in path:
            return {"etf_data": []}
        return {"price_data": {"current_price": "N/A"}}

    c._json = hollow
    try:
        c.get_ticker_snapshot("ZZZZZ")
    except QuiverError as exc:
        assert exc.reason == "unknown_ticker", exc.reason
    else:
        raise AssertionError("a hollow snapshot must be named, not returned")

    # A real quote is not hollow even with every holding section empty ...
    assert not _snapshot_is_hollow(
        {"price": {"price_data": {"current_price": "173.73"}}})
    # ... and a covered ticker reports how complete the answer is.
    def quoted(path, *args, **kwargs):
        if "insiders" in path:
            return {"insider_trades": [["X", "Sale", 1]]}
        if "institutions" in path:
            return {"response": {"heatmap_data": [{"Fund": "BlackRock"}]}}
        if "forecast" in path:
            return {"data": [1], "price_targets": {"high": 200}}
        if "tv_lazy" in path:
            return {"etf_data": [{"ticker": "QQQ"}]}
        if "bulls_bears" in path:
            return {"response": "no-auth"}
        return {"price_data": {"current_price": "173.73"}}

    c._json = quoted
    snap = c.get_ticker_snapshot("NVDA")
    assert snap["coverage"] == "partial", snap.get("coverage")
    assert "bulls_bears:no_auth" in snap["degraded_sections"], snap


# ---------------------------------------------------------------------------
# The dashboard arrays. Offline: the page is a string this file writes, so the
# union, dedup and date-sanity rules are tested without asking Quiver for them.
# ---------------------------------------------------------------------------

def _page_with(**arrays) -> str:
    """A page-shaped string carrying the named JS array literals."""
    body = "<html><script>"
    for name, rows in arrays.items():
        body += f"\nlet {name} = {json.dumps(rows)};\n"
    return body + "</script></html>"


def _insider(owner, ticker, date, shares=1, code="Purchase", value=100.0):
    return {"rptOwnerName": owner, "officerTitle": "CEO",
            "issuerTradingSymbol": ticker, "transactionCode": code,
            "transactionShares": shares, "transactionPricePerShare": 10.0,
            "transactionDate": date, "fileDate": date,
            "rptOwnerCik": 1, "netWorthEstimate": "-",
            "transactionValue": value}


def test_inside_the_dashboard_union_duplicates_are_named_not_double_counted():
    """The two lists overlap. Adding them end to end would report the same
    trade twice and inflate the total a caller sizes a position against."""
    from tools.quiver_server import client as qc

    shared = _insider("ADA LOVELACE", "AAPL", "Nov 10, 2026")
    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: _page_with(
        recentInsiderTransactionsData=[
            shared,
            _insider("GRACE HOPPER", "NVDA", "Nov 09, 2026", code="Sale"),
        ],
        topMonthlyInsiderTransactionsData=[
            shared,
            _insider("ALAN TURING", "MSFT", "Oct 01, 2026"),
        ],
    )

    d = c.get_insider_trades()
    assert d["total_available"] == 3, d["total_available"]
    by_owner = {t["owner"]: t for t in d["trades"]}
    assert by_owner["ADA LOVELACE"]["list"] == "recent+top_monthly"
    assert by_owner["GRACE HOPPER"]["list"] == "recent"
    assert by_owner["ALAN TURING"]["list"] == "top_monthly"
    # "-" is the site's stand-in for "not disclosed", not a number.
    assert by_owner["ADA LOVELACE"]["net_worth_estimate"] is None


def test_a_bad_transaction_date_is_flagged_and_kept_out_of_the_range():
    """Two rows on the live page are dated years after they were filed. They
    must not become the newest thing in the response."""
    from tools.quiver_server import client as qc

    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: _page_with(
        recentInsiderTransactionsData=[
            _insider("TYPO CORP", "TMUS", "May 24, 2028"),
            _insider("ADA LOVELACE", "AAPL", "Nov 10, 2026"),
        ],
        topMonthlyInsiderTransactionsData=[],
    )

    d = c.get_insider_trades()
    assert d["anomalous_rows"] == 1, d
    assert d["date_range"]["newest"] == "2026-11-10", d["date_range"]
    # Kept rather than dropped -- but last, so it cannot lead the list.
    assert d["trades"][0]["ticker"] == "AAPL", d["trades"][0]
    assert d["trades"][-1]["date_anomaly"] == "transaction_date_after_today"


def test_bills_page_the_index_instead_of_dumping_it():
    """33,000 rows will not fit in a reply. The window has to be a window, and
    the caller has to be told whether more remains."""
    from tools.quiver_server import client as qc

    rows = [{"congress": "119", "number": i, "type": "hr",
             "clean_title": f"H.R. {i}: Semiconductor Act {i}"}
            for i in range(1, 6)]
    rows.append({"congress": "118", "number": 9, "type": "s",
                 "clean_title": "S. 9: Farm Bill"})

    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: _page_with(searchBills=rows)

    first = c.get_bills(query="semiconductor", limit=2)
    assert first["total_indexed"] == 6, first["total_indexed"]
    assert first["matching"] == 5, first["matching"]
    assert first["returned"] == 2 and first["more"] is True, first

    second = c.get_bills(query="semiconductor", limit=2, offset=4)
    assert second["returned"] == 1, second
    assert second["more"] is False, second

    # Filters compose, and the type filter is exact rather than a substring.
    assert c.get_bills(bill_type="s")["matching"] == 1
    assert c.get_bills(congress="118")["matching"] == 1


if __name__ == "__main__":
    test_quiver_server_e2e()
    test_a_rate_limit_is_retried_once_and_then_named()
    test_a_rate_limit_is_not_reported_as_an_unknown_ticker()
    test_a_ticker_with_no_data_is_not_returned_as_six_empty_sections()
    test_inside_the_dashboard_union_duplicates_are_named_not_double_counted()
    test_a_bad_transaction_date_is_flagged_and_kept_out_of_the_range()
    test_bills_page_the_index_instead_of_dumping_it()
    print("quiver server e2e: PASS")
