"""The MCP surface over the congressional store.

An MCP call has to answer quickly, so these tools read the store rather than
fetching and parsing PDFs. That makes the empty-store case the one to get
right: a server deployed before its first sync holds nothing, and "no trades"
from an empty store is the most misleading answer this pipeline could give.
It has to say that it is empty and how to fill it.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from tools.altdata_server import congress_store as store
from tools.altdata_server.server import AltDataServer


@pytest.fixture
def empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()


@pytest.fixture
def server():
    return AltDataServer()


def _payload(result):
    return json.loads(result[0].text)


async def test_an_empty_store_says_it_is_empty(server, empty_store):
    body = _payload(await server.congress_trades({"ticker": "NVDA"}))
    data = body.get("data", body)

    assert data["transaction_count"] == 0
    assert data.get("store_empty") is True
    assert "congress_sync" in data["note"], (
        "an empty store returned zero trades without saying it was empty or "
        "how to populate it")


async def test_a_populated_store_answers_from_it(server, empty_store):
    member = store.member_id("house", "Allen", "Richard", "GA")
    store.upsert_member({"member_id": member, "chamber": "house",
                         "first": "Richard", "last": "Allen",
                         "full_name": "Richard W. Allen", "state": "GA",
                         "district": "GA12", "office": None,
                         "first_seen": "2025-01-01", "last_seen": "2025-01-01"})
    store.upsert_filing({"filing_id": "house:1", "chamber": "house",
                         "doc_id": "1", "member_id": member, "filing_type": "ptr",
                         "filed_date": "2025-01-16", "year": 2025,
                         "parse_status": "parsed"})
    store.replace_transactions("house:1", member, [
        {"ticker": "ROL", "asset_name": "Rollins, Inc.", "owner": "spouse",
         "transaction_type": "purchase", "transaction_date": "2024-12-12",
         "amount_min": 15001, "amount_max": 50000}])

    body = _payload(await server.congress_trades({"ticker": "ROL"}))
    data = body.get("data", body)

    assert data["transaction_count"] == 1
    assert data.get("store_empty") is not True
    assert data["transactions"][0]["member"] == "Richard W. Allen"
    assert data["transactions"][0]["amount_min"] == 15001


async def test_the_leaderboard_tool_dispatches(server, empty_store):
    body = _payload(await server.congress_leaderboard({"kind": "tickers"}))
    data = body.get("data", body)
    assert "tickers" in data


async def test_the_coverage_tool_reports_the_store(server, empty_store):
    body = _payload(await server.congress_coverage({}))
    data = body.get("data", body)

    assert data["total"] == 0
    assert data["complete"] is False
    assert "congress_sync" in data["note"]


async def test_every_congress_tool_is_registered_and_dispatchable(server):
    """A tool advertised but not dispatched is worse than one that is absent."""
    tools = await server.server.request_handlers.__self__ if False else None
    # The registry is built inside _setup_handlers; reach it the same way the
    # existing registration test does, through the module-level list.
    import inspect
    source = inspect.getsource(type(server)._setup_handlers)
    for name in ("get_congress_trades", "get_congress_leaderboard",
                 "get_congress_coverage"):
        assert f'name="{name}"' in source, f"{name} is not advertised"
        assert f'if name == "{name}"' in source, f"{name} has no dispatch branch"


def _synced(source, when):
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO sync_state(source, last_synced_at, last_cursor,
                    filings_seen, filings_parsed, filings_failed)
               VALUES(?, ?, NULL, 1, 1, 0)""", (source, when))


def _hours_ago(hours):
    return (datetime.now(timezone.utc)
            - timedelta(hours=hours)).isoformat(timespec="seconds")


def _sync_warnings(body):
    """The envelope carries `warnings` only when there is one; the server's
    annotation adds the empty list on the way out."""
    return {w["chamber"]: w["code"] for w in body.get("warnings", [])
            if w["code"] in ("stale_chamber_sync", "no_successful_sync")}


async def test_coverage_reports_each_chambers_last_successful_sync(
        server, empty_store):
    _synced("house_ptr_2026", "2026-10-26T06:00:00+00:00")
    _synced("senate_ptr", "2026-10-23T06:00:00+00:00")

    data = _payload(await server.congress_coverage({}))["data"]

    assert data["by_chamber"]["house"]["last_successful_sync"] == (
        "2026-10-26T06:00:00+00:00")
    assert data["by_chamber"]["senate"]["last_successful_sync"] == (
        "2026-10-23T06:00:00+00:00")


async def test_a_chamber_past_its_sync_tolerance_is_named_in_every_answer(
        server, empty_store):
    """The daily sync's tolerance is three days: a Senate store four days
    behind is named, a House store two days behind is not."""
    _synced("house_ptr_2026", _hours_ago(48))
    _synced("senate_ptr", _hours_ago(96))

    both = _payload(await server.congress_trades({"ticker": "NVDA"}))
    house = _payload(await server.congress_trades({"ticker": "NVDA",
                                                   "chamber": "house"}))
    board = _payload(await server.congress_leaderboard({"kind": "tickers"}))
    held = _payload(await server.congress_holdings({"ticker": "MSFT"}))
    coverage = _payload(await server.congress_coverage({}))

    assert _sync_warnings(both) == {"senate": "stale_chamber_sync"}
    assert _sync_warnings(house) == {}
    assert _sync_warnings(board) == {"senate": "stale_chamber_sync"}
    assert _sync_warnings(held) == {"senate": "stale_chamber_sync"}
    assert _sync_warnings(coverage) == {"senate": "stale_chamber_sync"}


async def test_a_chamber_that_never_synced_is_named(server, empty_store):
    _synced("house_ptr_2026", _hours_ago(1))

    body = _payload(await server.congress_trades({"ticker": "NVDA"}))

    assert _sync_warnings(body) == {"senate": "no_successful_sync"}
