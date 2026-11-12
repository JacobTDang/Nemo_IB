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

One test here reaches the network: it boots the server and calls all fifteen
tools against the live site. Everything below it runs offline against
scripted pages. The live one is gated twice, because it depends on a
scraped site that owes us nothing: @pytest.mark.network keeps it out of a
SKIP_NETWORK_TESTS=1 run (the server is a subprocess, so conftest's socket
guard cannot see its fetches), and skip_if_provider_unavailable turns an
upstream that never answered into a skip rather than a red build — a CI
runner timing out on quiverquant.com is a fact about the provider, not
about this parser.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from testing._gates import skip_if_provider_unavailable

SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "0") == "1"


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

            r = await session.call_tool("get_quiver_election_contributions",
                                        {"limit": 2})
            results["get_quiver_election_contributions"] = json.loads(
                r.content[0].text)

            r = await session.call_tool("get_quiver_layoffs", {"limit": 2})
            results["get_quiver_layoffs"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_revolving_door",
                                        {"limit": 2})
            results["get_quiver_revolving_door"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_gov_spending",
                                        {"limit": 2})
            results["get_quiver_gov_spending"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_patent_search", {
                "query": "solid state battery", "limit": 2})
            results["get_quiver_patent_search"] = json.loads(r.content[0].text)

            # No fund named: the dashboard's filer list. Then resolve the
            # newest filer, so the test does not pin a name that rotates out
            # of the top 100 between runs.
            r = await session.call_tool("get_quiver_fund_holdings", {})
            results["get_quiver_fund_filers"] = json.loads(r.content[0].text)
            filers = results["get_quiver_fund_filers"]["data"].get("filers") or []
            if filers:
                r = await session.call_tool("get_quiver_fund_holdings", {
                    "fund": filers[0]["fund"], "limit": 2})
                results["get_quiver_fund_holdings"] = json.loads(r.content[0].text)
            else:
                # An empty list is a failed fetch, not a dashboard with no
                # funds. Carrying the failure forward lets the gate below see
                # it and skip, instead of an IndexError here reporting the
                # parser as broken.
                results["get_quiver_fund_holdings"] = \
                    results["get_quiver_fund_filers"]

            # failure vocabulary: unknown entity must be success=false
            # with a machine reason, not an exception over the wire.
            r = await session.call_tool("get_quiver_politician_trades", {
                "politician": "Zzqqxx Nobody"})
            results["unknown_politician"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_fund_holdings", {
                "fund": "Zzqqxx No Such Fund"})
            results["unknown_fund"] = json.loads(r.content[0].text)

    return results


@pytest.mark.network
@pytest.mark.skipif(SKIP_NETWORK, reason=(
    "SKIP_NETWORK_TESTS=1: this test boots the server as a subprocess and "
    "calls every tool against the live quiverquant.com. Unset it to run."))
def test_quiver_server_e2e():
    results = asyncio.run(_exercise_all_tools())

    assert results["__tool_count__"] == 15, results["__tool_names__"]
    assert "get_quiver_congress_trades" in results["__tool_names__"]

    # A page that never answered is not a parser verdict. Decided before any
    # content assertion, so a provider outage reads as "unanswerable here"
    # rather than as fifteen broken tools. The intentional misses below
    # (unknown_politician, unknown_fund) carry their own reasons and are not
    # caught by this -- the gate looks only at provider_unavailable.
    for name, body in results.items():
        if name.startswith("__"):
            continue
        skip_if_provider_unavailable(body.get("data") or {}, "QuiverQuant")

    # Every response shares the envelope contract.
    for name in ["get_quiver_congress_trades", "get_quiver_congress_trades_filtered",
                 "get_quiver_politician_trades", "get_quiver_wsb_sentiment",
                 "get_quiver_lobbying", "get_quiver_gov_contracts",
                 "get_quiver_ticker_snapshot", "get_quiver_insider_trades",
                 "get_quiver_congress_net_worth", "get_quiver_bills",
                 "get_quiver_election_contributions", "get_quiver_layoffs",
                 "get_quiver_revolving_door", "get_quiver_gov_spending",
                 "get_quiver_patent_search", "get_quiver_fund_filers",
                 "get_quiver_fund_holdings"]:
        body = results[name]
        assert body["success"] is True, (name, body.get("error"))
        assert body["domain"] == "quiver", name
        assert body["provider"], name
        assert isinstance(body["warnings"], list), name
        # Coverage is part of the envelope, not just the data, so a host reads
        # one field to know whether the answer is whole.
        assert body["coverage"] in ("full", "partial"), (name, body["coverage"])

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

    # Election contributions: two tables of separate money, each row naming
    # its source, with the party split as a number rather than a string.
    money = results["get_quiver_election_contributions"]["data"]
    assert money["total_embedded"] == 200, money["total_embedded"]
    assert money["by_source"] == {"pac": 100, "employee": 100}, money["by_source"]
    assert money["returned"] == 2
    top = money["contributions"][0]
    assert top["ticker"] and top["company"], top
    assert top["total_usd"] > 0, top
    # "45.20%" and "$2,006,000" are parsed, not passed through as text.
    assert isinstance(top["democrat_pct"], float), top
    assert isinstance(top["democrat_usd"], float), top

    # Layoffs: a 2,000-row window of state notices, with the ticker match
    # labelled rather than trusted.
    layoffs = results["get_quiver_layoffs"]["data"]
    assert layoffs["total_embedded"] > 500, layoffs["total_embedded"]
    assert layoffs["returned"] == 2
    assert layoffs["workers_matching"] > 0, layoffs
    assert layoffs["date_range"]["newest"], layoffs
    assert layoffs["notices"][0]["employer"], layoffs["notices"][0]
    assert layoffs["notices"][0]["ticker_match"] in (
        "exact", "fuzzy", "search", "unmatched"), layoffs["notices"][0]
    assert layoffs["notices"][0]["notice_date"], layoffs["notices"][0]
    # Counted over the matching set, not the returned slice: asking "how many
    # of these have a ticker" about the first two rows would be a guess.
    assert layoffs["tickers_mapped"] > layoffs["returned"], layoffs

    # Revolving door: filings, each naming the quarter it covers.
    revolving = results["get_quiver_revolving_door"]["data"]
    assert revolving["total_embedded"] > 500, revolving["total_embedded"]
    assert revolving["returned"] == 2
    assert revolving["registrations"][0]["lobbyist"], revolving["registrations"][0]
    assert revolving["registrations"][0]["filed_date"], revolving["registrations"][0]

    # Government spending: transaction level, each row naming its award.
    spending = results["get_quiver_gov_spending"]["data"]
    assert spending["total_embedded"] > 100, spending["total_embedded"]
    assert spending["amount_matching_usd"] > 0, spending
    assert spending["transactions"][0]["ticker"], spending["transactions"][0]
    assert spending["transactions"][0]["award_id"], spending["transactions"][0]
    assert spending["date_range"]["newest"], spending
    assert spending["tickers"] > spending["returned"], spending

    # Patents: a search over recent grants, by assignee ticker.
    patents = results["get_quiver_patent_search"]["data"]
    assert patents["patents"], patents
    assert patents["patents"][0]["ticker"], patents["patents"][0]
    assert len(patents["patents"][0]["grant_date"]) == 10, patents["patents"][0]
    assert patents["coverage"] in ("full", "partial"), patents["coverage"]

    # 13F fund holdings: positions, not the treemap's own branches.
    holdings = results["get_quiver_fund_holdings"]["data"]
    assert holdings["positions"], holdings
    assert holdings["invested_usd"] > 0, holdings
    assert holdings["returned"] == 2
    top_position = holdings["positions"][0]
    assert top_position["ticker"] and top_position["sector"], top_position
    assert 0 < top_position["portfolio_pct"] <= 100, top_position
    assert holdings["total_positions"] >= holdings["returned"], holdings

    # Unknown entity -> success=false with a machine-readable reason.
    miss = results["unknown_politician"]
    assert miss["success"] is False
    assert miss["data"]["reason"] == "unknown_politician", miss

    no_fund = results["unknown_fund"]
    assert no_fund["success"] is False
    assert no_fund["data"]["reason"] == "unknown_fund", no_fund


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


def test_the_two_donation_tables_are_told_apart_by_their_headings():
    """Both tables have identical headers, so their labels cannot come from
    the columns. They come from the heading above each table; a company's PAC
    and its employees are separate money and summing them would be wrong."""
    from tools.quiver_server import client as qc

    header = ("<table><tr><th>Ticker</th><th>Amount (Total)</th>"
              "<th>Amount (Democrat)</th><th>Amount (Republican)</th>"
              "<th>Cycle</th></tr>")
    pac = (header + "<tr><td>HON Honeywell International Inc</td>"
           "<td>$4,455,171</td><td>$2,006,000 (45.20%)</td>"
           "<td>$2,432,171 (54.80%)</td><td>2026</td></tr></table>")
    employee = (header + "<tr><td>HON Honeywell International Inc</td>"
                "<td>$120,000</td><td>$100,000 (83.33%)</td>"
                "<td>$20,000 (16.67%)</td><td>2026</td></tr></table>")
    page = ("<h2>Company PAC Donations</h2>" + pac
            + "<h2>Company Employee Donations</h2>" + employee)

    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: page

    d = c.get_election_contributions()
    assert d["by_source"] == {"pac": 1, "employee": 1}, d["by_source"]
    by_source = {r["source"]: r for r in d["contributions"]}
    assert by_source["pac"]["total_usd"] == 4455171.0, by_source["pac"]
    assert by_source["employee"]["total_usd"] == 120000.0, by_source["employee"]
    # The ticker cell carries the name too; both are kept.
    assert by_source["pac"]["ticker"] == "HON", by_source["pac"]
    assert by_source["pac"]["company"].startswith("Honeywell")

    # The party split is a number, so a lean is a comparison not a guess.
    assert by_source["pac"]["democrat_pct"] == 45.2, by_source["pac"]
    assert [r["source"] for r in
            c.get_election_contributions(leans="democrat")["contributions"]] \
        == ["employee"]
    assert len(c.get_election_contributions(leans="republican")["contributions"]) == 1
    assert c.get_election_contributions(ticker="HON", source="pac")["matching"] == 1


# ---------------------------------------------------------------------------
# The JSON-script dashboards. Offline: two pages ship their rows as a JSON
# block that the page's own table class renders, which neither of the older
# readers (JS literals, HTML tables) could see.
# ---------------------------------------------------------------------------

def _json_script_page(element_id: str, rows) -> str:
    return ("<html><body><table><tr><th>Employer</th></tr></table>"
            f'<script id="{element_id}" type="application/json">'
            f"{json.dumps(rows)}</script></body></html>")


def _layoff_row(employer, ticker="", match="unmatched", state="CA",
                kind="Layoff", workers=100, notice_date="2026-11-09",
                effective="2026-12-01", industry="Manufacturing"):
    row = [""] * 14
    row[0] = employer
    row[1] = ticker
    row[2] = match
    row[4] = state
    row[5] = kind
    row[7] = workers
    row[8] = notice_date
    row[10] = effective
    row[13] = industry
    return row


def test_layoffs_read_rows_out_of_the_json_script_block():
    """The rows are not in a JS literal or a table, so the parser has to read
    them where they are -- and a missing block is a layout change, not an
    empty set of layoffs."""
    import datetime as _dt
    from tools.quiver_server import client as qc
    from tools.quiver_server.client import QuiverError

    today = _dt.date.today()
    recent = today.isoformat()
    stale = (today - _dt.timedelta(days=200)).isoformat()

    rows = [
        _layoff_row("Exact Corp", "EXCT", "exact", workers=900,
                    notice_date=recent),
        _layoff_row("Fuzzy Holdings", "FZY", "fuzzy", state="TX",
                    kind="Closure", workers=50, notice_date=recent),
        _layoff_row("Nobody Ltd", "", "unmatched", state="TX",
                    notice_date=stale, workers=1500),
    ]

    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: _json_script_page("layoffsRowsData", rows)

    d = c.get_layoffs()
    assert d["total_embedded"] == 3, d
    assert d["matching"] == 3, d
    assert d["workers_matching"] == 2450, d
    assert d["date_range"] == {"newest": recent, "oldest": stale}, d
    # The heuristic match is carried, never smoothed into a bare ticker.
    by_employer = {n["employer"]: n for n in d["notices"]}
    assert by_employer["Fuzzy Holdings"]["ticker_match"] == "fuzzy", d
    assert by_employer["Nobody Ltd"]["ticker"] is None, d

    # A limit truncates the rows without truncating the counts that describe
    # the matching set.
    sliced = c.get_layoffs(limit=1)
    assert sliced["returned"] == 1, sliced
    assert sliced["matching"] == 3 and sliced["tickers_mapped"] == 2, sliced
    # Newest first, then largest: today's 900 beats today's 50.
    assert sliced["notices"][0]["employer"] == "Exact Corp", sliced

    # Filters compose on the fields the dashboard offers.
    assert c.get_layoffs(state="TX")["matching"] == 2
    assert c.get_layoffs(kind="Closure")["matching"] == 1
    assert c.get_layoffs(ticker="EXCT")["matching"] == 1
    assert c.get_layoffs(min_workers=1000)["matching"] == 1
    assert c.get_layoffs(days=30)["matching"] == 2  # the stale row falls out

    # Rows in another layout are skipped rather than indexed into.
    c._page = lambda path, **kw: _json_script_page(
        "layoffsRowsData", [["short", "row"]])
    assert c.get_layoffs()["total_embedded"] == 0

    # No block at all is a parse failure with a reason, not 0 rows.
    c._page = lambda path, **kw: "<html>no rows here</html>"
    try:
        c.get_layoffs()
    except QuiverError as exc:
        assert exc.reason == "parse_failed", exc.reason
    else:
        raise AssertionError("a page without the rows block must not look empty")


def _revolving_row(name, chamber="House", party="D", ticker="",
                   match="unmatched", client="ACME CORP",
                   firm="FIRM LLC", filed="2026-11-11"):
    row = [""] * 15
    row[0] = name
    row[1] = "Legislative Assistant"
    row[4] = "Member Name"
    row[5] = chamber
    row[6] = ticker
    row[7] = client
    row[8] = firm
    row[9] = filed
    row[11] = match
    row[13] = party
    row[14] = "Q3 2026"
    return row


def test_revolving_door_rows_keep_the_chamber_party_and_period():
    """A registration is several facts at once -- who, which office, which
    client, which quarter -- and the quarter is not the filed date."""
    import datetime as _dt
    from tools.quiver_server import client as qc

    today = _dt.date.today()
    rows = [
        _revolving_row("ADA LOVELACE", chamber="Senate", party="R",
                       ticker="ACME", match="exact",
                       filed=today.isoformat()),
        _revolving_row("GRACE HOPPER", chamber="House", party="D",
                       client="OTHER INC",
                       filed=(today - _dt.timedelta(days=60)).isoformat()),
    ]
    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: _json_script_page(
        "revolvingDoorRowsData", rows)

    d = c.get_revolving_door()
    assert d["total_embedded"] == 2 and d["matching"] == 2, d
    assert d["date_range"] == {
        "newest": today.isoformat(),
        "oldest": (today - _dt.timedelta(days=60)).isoformat()}, d
    first = d["registrations"][0]
    assert first["lobbyist"] == "ADA LOVELACE", first
    assert first["chamber"] == "Senate" and first["party"] == "R", first
    assert first["period"] == "Q3 2026", first
    assert first["ticker"] == "ACME" and first["ticker_match"] == "exact", first

    assert c.get_revolving_door(chamber="house")["matching"] == 1
    assert c.get_revolving_door(party="R")["matching"] == 1
    assert c.get_revolving_door(name="grace")["matching"] == 1
    assert c.get_revolving_door(client="other")["matching"] == 1
    assert c.get_revolving_door(ticker="ACME")["matching"] == 1
    assert c.get_revolving_door(days=30)["matching"] == 1


# ---------------------------------------------------------------------------
# Government spending: the one allowed JSON endpoint, in transaction shape.
# ---------------------------------------------------------------------------

def test_gov_spending_answers_per_transaction_not_as_a_totals_row():
    import datetime as _dt
    from tools.quiver_server import client as qc

    today = _dt.date.today()
    newest = today.isoformat()
    middle = (today - _dt.timedelta(days=5)).isoformat()
    old = (today - _dt.timedelta(days=40)).isoformat()
    transactions = [
        ["Department of Energy", "FLUOR CORP", newest, "FLR", 378824045.0,
         "CONT_AWD_1", "MANAGEMENT AND OPERATING CONTRACT"],
        ["Department of Veterans Affairs", "AT&T INC", old, "T",
         153001558.0, "CONT_AWD_2", "ENTERPRISE DATA NETWORK SERVICES"],
        ["Department of Homeland Security", "UNITED PARCEL SERVICE CO.",
         middle, "UPS", 45000.0, "CONT_AWD_3", "UPS SHIPPING"],
    ]
    c = qc.QuiverWebClient()
    c._json = lambda path, **kw: {"gov_transactions": transactions}

    d = c.get_gov_spending()
    assert d["total_embedded"] == 3 and d["matching"] == 3, d
    assert d["amount_matching_usd"] == 531870603.0, d
    assert d["tickers"] == 3, d
    assert d["date_range"] == {"newest": newest, "oldest": old}, d
    # Largest first, and each row is one award with an id to cite.
    assert [t["ticker"] for t in d["transactions"]] == ["FLR", "T", "UPS"], d
    assert d["transactions"][0]["award_id"] == "CONT_AWD_1", d
    assert d["transactions"][0]["agency"].startswith("Department of Energy")

    # The sum describes the matching set, not the returned slice.
    sliced = c.get_gov_spending(limit=1)
    assert sliced["returned"] == 1, sliced
    assert sliced["amount_matching_usd"] == 531870603.0, sliced
    assert sliced["tickers"] == 3, sliced

    assert c.get_gov_spending(agency="homeland")["matching"] == 1
    assert c.get_gov_spending(query="shipping")["matching"] == 1
    assert c.get_gov_spending(min_amount=1e8)["matching"] == 2
    assert c.get_gov_spending(ticker="T")["matching"] == 1
    assert c.get_gov_spending(days=30)["matching"] == 2  # the 40-day-old row


# ---------------------------------------------------------------------------
# Patent search: a POST form, submitted the way the page's own search box does.
# ---------------------------------------------------------------------------

def test_patent_search_posts_the_query_with_the_page_token():
    """There is no GET route for this search. The query goes in the body with
    the CSRF token the page handed out, or the site returns the empty form."""
    import httpx

    table = ("<table><tr><th>Ticker</th><th>Date</th>"
             "<th>Patent Title <span>*</span></th></tr>"
             "<tr><td>GLW</td><td>09/22/2026</td><td>solid electrolyte film</td></tr>"
             "<tr><td>HMC</td><td>09/22/2026</td><td>cathode slurry method</td></tr>"
             "</table>")
    seen = {}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=(
                '<form action=/patentsearch/ method="post">'
                '<input type="hidden" name="csrfmiddlewaretoken" value="TOK123">'
                "</form>"))
        body = request.content.decode()
        seen["body"] = body
        seen["referer"] = request.headers.get("referer")
        return httpx.Response(200, text=table)

    c = _client_with(handler)

    d = c.get_patent_search("solid state battery", limit=1)
    # The token travels with the term, and the query is only the term -- the
    # page's ticker field would add a second, conflicting filter.
    assert "csrfmiddlewaretoken=TOK123" in seen["body"], seen
    assert "searchterm=solid+state+battery" in seen["body"], seen
    assert "searchticker" not in seen["body"], seen
    assert seen["referer"].endswith("/patentsearch/"), seen

    assert d["query"] == "solid state battery", d
    assert d["total_matched"] == 2 and d["returned"] == 1, d
    assert d["patents"][0]["ticker"] == "GLW", d
    # 09/22/2026 is a date, not a string to sort on.
    assert d["patents"][0]["grant_date"] == "2026-09-22", d
    assert d["tickers"] == 2, d
    assert d["coverage"] == "full", d


def test_patent_results_report_the_cap_when_the_page_is_full():
    """100 rows is the page's ceiling, not the match count. Saying so is the
    difference between 'these are the patents' and 'these are the first 100'."""
    from tools.quiver_server import client as qc

    def _table(n):
        rows = "".join(f"<tr><td>TCK{i}</td><td>09/22/2026</td>"
                       f"<td>title {i}</td></tr>" for i in range(n))
        return ("<table><tr><th>Ticker</th><th>Date</th><th>Patent Title</th>"
                f"</tr>{rows}</table>")

    c = qc.QuiverWebClient()
    c._post = lambda path, data, **kw: _table(100)
    assert c.get_patent_search("battery")["coverage"] == "partial"

    c._post = lambda path, data, **kw: _table(4)
    assert c.get_patent_search("battery")["coverage"] == "full"


# ---------------------------------------------------------------------------
# 13F fund holdings: a Plotly treemap, whose branches are not positions.
# ---------------------------------------------------------------------------

def _plotly_body(labels, parents, values, customdata, *, escaped=True):
    traces = [{"type": "treemap", "labels": labels, "parents": parents,
               "values": values, "customdata": customdata,
               "branchvalues": "total"}]
    payload = json.dumps(traces)
    if escaped:
        payload = payload.replace('"', '\\"')
    return ('<div id="g" class="plotly-graph-div"></div>'
            "<script>Plotly.newPlot(\"g\", " + payload
            + ", {}, {});</script>")


def test_the_fund_treemap_yields_positions_not_tree_branches():
    """The treemap carries the fund, each sector, and then the leaves. Only
    the leaves are holdings; counting the rest would invent positions."""
    from tools.quiver_server import client as qc

    filers_page = ("<h1>13F Filings</h1><table>"
                   "<tr><th>Fund</th><th>Filed</th></tr>"
                   "<tr><td>Kailix Advisors LLC</td><td>1 day ago</td></tr>"
                   "<tr><td>Stephens Investment Management</td>"
                   "<td>2 days ago</td></tr></table>")
    body = _plotly_body(
        labels=["Kailix Advisors LLC", "Miscellaneous", "Energy",
                "DCH", "GPRE", "HONA"],
        parents=["", "Kailix Advisors LLC", "Kailix Advisors LLC",
                 "Kailix Advisors LLC/Miscellaneous",
                 "Kailix Advisors LLC/Energy",
                 "Kailix Advisors LLC/Miscellaneous"],
        values=[35000000.0, 26608500.0, 11528430.0,
                18281238.66, 11528430.0, 5125632.75],
        customdata=[[35000000.0, 0, 0, 0], [26608500.0, 0, 0, 0],
                    [11528430.0, 0, 0, 0],
                    [18281238.66, -674500.0, -3520890.0, -3520890.0],
                    [11528430.0, -219500.0, -3305670.0, -3305670.0],
                    # "(?)" is the site's stand-in for an unknown share change.
                    [5125632.75, "(?)", 5125632.75, 5125632.75]])

    c = qc.QuiverWebClient()
    c._page = lambda path, **kw: filers_page
    c._fetch = lambda path, **kw: body

    d = c.get_fund_holdings("Kailix Advisors LLC", limit=2)
    assert d["fund"] == "Kailix Advisors LLC", d
    assert d["total_positions"] == 3, d
    assert d["returned"] == 2, d
    # Only the leaves sum: the fund and sector branches would double it.
    assert d["invested_usd"] == 34935301.41, d["invested_usd"]
    assert d["sectors"] == 2, d["sectors"]

    top = d["positions"][0]
    assert top["ticker"] == "DCH" and top["sector"] == "Miscellaneous", top
    assert top["value_usd"] == 18281238.66, top
    assert top["shares_change"] == -674500.0, top
    assert top["est_bought_usd"] == -3520890.0, top
    # Share of the portfolio is computed over every position, not the slice.
    assert abs(top["portfolio_pct"] - 52.33) < 0.01, top

    # An unknown share change is None rather than the string the site sent.
    hona = {p["ticker"]: p for p in c.get_fund_holdings(
        "Kailix Advisors LLC")["positions"]}["HONA"]
    assert hona["shares_change"] is None, hona

    # Names resolve the way a reader types them.
    assert c.get_fund_holdings("kailix")["fund"] == "Kailix Advisors LLC"
    assert c.get_fund_holdings("Stephens")["fund"].startswith("Stephens")

    from tools.quiver_server.client import QuiverError
    try:
        c.get_fund_holdings("No Such Fund Zz")
    except QuiverError as exc:
        assert exc.reason == "unknown_fund", exc.reason
    else:
        raise AssertionError("an unmatched fund must be named, not guessed")

    # The filer list is what an empty request answers with.
    assert [f["fund"] for f in c.get_fund_filers()] == [
        "Kailix Advisors LLC", "Stephens Investment Management"]


if __name__ == "__main__":
    test_quiver_server_e2e()
    test_a_rate_limit_is_retried_once_and_then_named()
    test_a_rate_limit_is_not_reported_as_an_unknown_ticker()
    test_a_ticker_with_no_data_is_not_returned_as_six_empty_sections()
    test_inside_the_dashboard_union_duplicates_are_named_not_double_counted()
    test_a_bad_transaction_date_is_flagged_and_kept_out_of_the_range()
    test_bills_page_the_index_instead_of_dumping_it()
    test_the_two_donation_tables_are_told_apart_by_their_headings()
    test_layoffs_read_rows_out_of_the_json_script_block()
    test_revolving_door_rows_keep_the_chamber_party_and_period()
    test_gov_spending_answers_per_transaction_not_as_a_totals_row()
    test_patent_search_posts_the_query_with_the_page_token()
    test_patent_results_report_the_cap_when_the_page_is_full()
    test_the_fund_treemap_yields_positions_not_tree_branches()
    print("quiver server e2e: PASS")
