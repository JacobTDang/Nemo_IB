"""Short interest: a crowding count that must never price a short.

The feed is free and public, which makes it tempting to treat as a borrow
rate. It is not one, and the tests here are mostly about that line and about
the two ways a count can lie:

  * it can arrive before it was published, which is lookahead, and
  * it can arrive in part, which is a settlement that answers for half the
    universe and says nothing about the rest.

The live test at the bottom reaches FINRA. It is marked `network` and skips
when the run is offline, and it skips when FINRA itself does not answer --
that is a fact about the provider, not about this parser.
"""
import json
import os

import pytest

from research import borrow, pit_store, short_interest
from testing._gates import skip_if_provider_unavailable

SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "0") == "1"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


class _Resp:
    def __init__(self, status, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}


class _Finra:
    """FINRA's API as it behaves, including the parts that are traps.

    `rows_by_day` maps a settlement date to the rows FINRA holds for it. The
    POST filter is the only thing that narrows the result -- the real API
    accepts the GET form and ignores it -- so this double raises on anything
    but a POST body, which is what keeps the client from regressing to a GET
    that returns 2020 and looks fine.
    """

    def __init__(self, rows_by_day, fail=None):
        self.rows = rows_by_day
        self.calls = []
        self.fail = fail or (lambda day, offset: False)

    def post(self, url, content=None):
        assert content is not None, "FINRA is queried with a body, not a query"
        body = json.loads(content)
        flt = body["compareFilters"][0]
        day, compare = flt["fieldValue"], flt["compareType"]
        self.calls.append({"day": day, "offset": body["offset"],
                           "limit": body["limit"], "compare": compare,
                           "field": flt["fieldName"]})
        if self.fail(day, body["offset"]):
            import httpx
            raise httpx.ConnectError("connection reset by peer")
        if compare == "GREATER":
            later = sorted(d for d in self.rows if d > day)
            if not later:
                return _Resp(204)
            return _Resp(200, json.dumps([self.rows[later[0]][0]]))
        rows = self.rows.get(day, [])
        if not rows:
            return _Resp(204)
        return _Resp(200, json.dumps(
            rows[body["offset"]:body["offset"] + body["limit"]]))


def _finra_row(ticker, shares, *, prev=None, adv=None, dtc=None,
               split=None, day="2026-10-30"):
    return {
        "symbolCode": ticker,
        "currentShortPositionQuantity": shares,
        "previousShortPositionQuantity": prev,
        "averageDailyVolumeQuantity": adv,
        "daysToCoverQuantity": dtc,
        "changePercent": 1.76,
        "issueName": f"{ticker} Corporation",
        "marketClassCode": "NYSE",
        "stockSplitFlag": split,
        "settlementDate": day,
        "accountingYearMonthNumber": 20261030,
        "revisionFlag": None,
    }


DAY = "2026-10-30"
# FINRA published this settlement on 11 November, twelve days after it closed.
PUBLISHED = "2026-11-11"


# ------------------------------------------------------------------ the feed

def test_a_settlement_is_paged_in_full_before_anything_is_written(store):
    """Nineteen thousand rows arrive four pages at a time. A client that
    writes each page as it lands has written a settlement that is not one."""
    rows = [_finra_row(f"T{i:05d}", 1000 + i) for i in range(12_000)]
    api = _Finra({DAY: rows})

    got = short_interest.fetch_settlement(DAY, client=api)

    assert len(got) == 12_000, len(got)
    assert [c["offset"] for c in api.calls] == [0, 5000, 10_000], api.calls
    # The filter travels in the body, on the field the API documents.
    assert all(c["field"] == "settlementDate" for c in api.calls), api.calls
    assert all(c["compare"] == "EQUAL" for c in api.calls), api.calls
    assert got[0]["ticker"] == "T00000", got[0]
    assert got[0]["market"] == "NYSE", got[0]


def test_a_page_that_fails_leaves_no_settlement_behind(store, monkeypatch):
    """The failure this guards is not an exception: it is eleven thousand rows
    that look like a dataset and answer for 58% of the universe."""
    monkeypatch.setattr(short_interest, "RETRY_BACKOFF_SECONDS", 0.0)
    rows = [_finra_row(f"T{i:05d}", 1000) for i in range(12_000)]
    api = _Finra({DAY: rows}, fail=lambda day, offset: offset == 5000)

    # `record` reports the failure rather than raising it, so the nightly stage
    # can carry on and the status report says which stage did not land.
    result = short_interest.record("2026-11-12", settlement=DAY, client=api)
    assert result["status"] == "failed", result
    assert result["reason"] == "provider_unavailable", result
    assert result["written"] == 0, result

    # And the fetch itself raises rather than returning the pages it managed.
    with pytest.raises(short_interest.ShortInterestError) as caught:
        short_interest.fetch_settlement(DAY, client=api)
    assert caught.value.reason == "provider_unavailable", caught.value.reason
    assert store.short_interest_as_of("T00000", "2026-12-31") is None


def test_an_unpublished_settlement_is_empty_not_a_failure(store, monkeypatch):
    """204 is the documented way FINRA says "not yet". Reading it as a failure
    would make every night between settlements a red build."""
    monkeypatch.setattr(short_interest, "fetch_settlement",
                        lambda day, client=None: [])
    result = short_interest.record("2026-11-12", settlement=DAY)
    assert result["status"] == "ok", result
    assert result["written"] == 0, result


def test_the_next_settlement_is_found_rather_than_guessed(store):
    """A settlement is the 15th or the last business day, except when a
    holiday moves it. Guessing misses; asking the API cannot."""
    api = _Finra({"2026-10-15": [_finra_row("OLD", 1, day="2026-10-15")],
                  DAY: [_finra_row("NEW", 2)]})

    assert short_interest.next_settlement("2026-10-15", client=api) == DAY
    assert api.calls[-1]["compare"] == "GREATER", api.calls[-1]
    assert api.calls[-1]["limit"] == 1, api.calls[-1]
    assert short_interest.next_settlement(DAY, client=api) is None


def test_a_replay_run_does_not_reach_past_its_own_date(store):
    """A run replayed for last month asks for last month's settlements.

    The discovery walk stops at the run's date instead of spending requests on
    rows that run is going to discard.
    """
    api = _Finra({"2026-10-15": [_finra_row("OLD", 1, day="2026-10-15")],
                  DAY: [_finra_row("NEW", 2)]})

    found = short_interest.settlements_after("2026-10-01", until="2026-10-20",
                                             client=api)
    assert found == ["2026-10-15"], found
    assert api.calls[-1]["compare"] == "GREATER", api.calls[-1]


# ------------------------------------------------- point in time, both dates

def test_a_settlement_is_invisible_until_it_was_published(store):
    """The count is dated to the day the firms reported it and becomes public
    twelve days later. A scan inside that window must not see it."""
    store.record_short_interest(
        DAY, [{"ticker": "ACME", "shares_short": 1_000_000.0}],
        recorded_at=f"{PUBLISHED}T21:00:00Z", backfilled=True)

    assert store.short_interest_as_of("ACME", "2026-11-05") is None
    hit = store.short_interest_as_of("ACME", PUBLISHED)
    assert hit is not None and hit["shares_short"] == 1_000_000.0, hit
    assert hit["backfilled"] == 1, hit
    # And never for a date before the settlement itself.
    assert store.short_interest_as_of("ACME", "2026-10-01") is None


def test_a_live_ingest_is_stamped_the_night_it_was_read(store):
    """The ordinary path: no backdating, so the row is usable from today and
    from no earlier date."""
    store.record_short_interest(
        DAY, [{"ticker": "ACME", "shares_short": 500.0}],
        recorded_at="2026-11-12T22:40:00Z")

    assert store.short_interest_as_of("ACME", "2026-11-11") is None
    assert store.short_interest_as_of("ACME", "2026-11-12") is not None


def test_the_history_is_a_series_so_a_move_can_be_seen(store):
    """One settlement of a biweekly count is a level. The flow is the change,
    and it needs at least two."""
    for day, shares in (("2026-10-15", 100.0), (DAY, 300.0)):
        store.record_short_interest(
            day, [{"ticker": "ACME", "shares_short": shares}],
            recorded_at="2026-11-12T22:40:00Z")

    series = store.short_interest_history("ACME", "2026-12-01")
    assert [r["settlement_date"] for r in series] == [DAY, "2026-10-15"], series
    assert series[0]["shares_short"] == 300.0, series[0]


# ------------------------------------------------------------- revisions

def test_a_revised_count_is_kept_beside_the_original(store):
    """FINRA re-reports when a member corrects. The original is what a decision
    that day could have read; overwriting it destroys that record."""
    first = store.record_short_interest(
        DAY, [{"ticker": "ACME", "shares_short": 1_000.0}],
        recorded_at="2026-11-11T21:00:00Z")
    again = store.record_short_interest(
        DAY, [{"ticker": "ACME", "shares_short": 1_250.0}],
        recorded_at="2026-11-20T21:00:00Z")

    assert first == {"written": 1, "unchanged": 0, "revised": 0}, first
    assert again == {"written": 0, "unchanged": 0, "revised": 1}, again
    held = store.short_interest_as_of("ACME", "2026-12-01")
    assert held["shares_short"] == 1_000.0, held
    with store.connect() as conn:
        revisions = conn.execute(
            "SELECT * FROM short_interest_revision").fetchall()
    assert len(revisions) == 1, [dict(r) for r in revisions]
    assert revisions[0]["new_shares_short"] == 1_250.0, dict(revisions[0])

    # Re-recording the same file is a quiet no-op, not a revision.
    quiet = store.record_short_interest(
        DAY, [{"ticker": "ACME", "shares_short": 1_000.0}],
        recorded_at="2026-11-21T21:00:00Z")
    assert quiet == {"written": 0, "unchanged": 1, "revised": 0}, quiet


def test_a_row_that_will_not_parse_raises_rather_than_being_dropped(store):
    """A negative count is a sign error upstream. Skipping the row would leave
    a settlement that looks complete."""
    with pytest.raises(ValueError):
        store.record_short_interest(
            DAY, [{"ticker": "ACME", "shares_short": -5.0}])
    with pytest.raises(ValueError):
        store.record_short_interest(
            DAY, [{"ticker": "ACME", "shares_short": "not a number"}])
    with store.connect() as conn:
        assert conn.execute(
            "SELECT count(*) FROM short_interest").fetchone()[0] == 0


def test_a_row_with_no_symbol_is_dropped_not_stored_under_a_blank(store):
    """Rights, units and test symbols come back without a ticker. They cannot
    be joined to anything, so they are not joined to everything."""
    api = _Finra({DAY: [
        {"symbolCode": "", "currentShortPositionQuantity": 10},
        _finra_row("ACME", 20),
    ]})
    rows = short_interest.fetch_settlement(DAY, client=api)
    assert [r["ticker"] for r in rows] == ["ACME"], rows


# ------------------------------------------- and it still does not price it

def test_short_interest_on_file_still_does_not_price_a_short(store):
    """The claim the whole module rests on. A full year of crowding data is
    not a borrow rate, so the cost model keeps refusing -- and says why."""
    store.record_short_interest(
        DAY, [{"ticker": "ACME", "shares_short": 9_000_000.0,
               "days_to_cover": 12.5}],
        recorded_at=f"{PUBLISHED}T21:00:00Z")
    assert store.short_interest_as_of("ACME", "2026-11-20") is not None

    carry = borrow.carry_cost("ACME", "2026-11-20", "short")
    assert carry["cost"] is None, carry
    assert carry["annual_rate"] is None, carry
    assert "no borrow rate is on record" in carry["reason"], carry["reason"]


def test_backfill_refuses_to_invent_the_day_it_became_public(store):
    """A bulk load cannot carry one publication date, and inferring one from
    the settlement date is lookahead whenever the inferrer is wrong."""
    # The CLI says no before reaching the network.
    assert short_interest.main(["--backfill", PUBLISHED]) == 2

    # And so does the function, for a caller that never went near the CLI. The
    # empty double answers "no settlement later than the floor", so this
    # reaches the guard rather than the API.
    result = short_interest.record("2026-11-12", backfill=PUBLISHED,
                                   client=_Finra({}))
    assert result["status"] == "failed", result
    assert result["reason"] == "bad_request", result
    assert "needs the settlement named" in result["error"], result


def test_a_backfilled_settlement_is_stamped_the_day_it_became_public(
        store, monkeypatch):
    """What a backfill is for: loading history without claiming to have had it.

    The count closed on 30 October and FINRA published it on 11 November. The
    row is stamped the publication day and marked, so a study over October sees
    nothing and one over December sees a row that says it was loaded late.
    """
    monkeypatch.setattr(
        short_interest, "fetch_settlement",
        lambda day, client=None: [{"ticker": "ACME", "shares_short": 1234.0}])

    result = short_interest.record("2026-11-20", settlement=DAY,
                                   backfill=PUBLISHED)
    assert result["status"] == "ok", result
    assert result["written"] == 1, result

    row = store.short_interest_as_of("ACME", PUBLISHED)
    assert row is not None, row
    assert row["backfilled"] == 1, row
    assert row["settlement_date"] == DAY, row
    assert row["recorded_at"].startswith(PUBLISHED), row
    # Not the settlement date, and not the day it was loaded.
    assert store.short_interest_as_of("ACME", "2026-10-31") is None
    assert store.short_interest_as_of("ACME", "2026-11-10") is None
    # Readable from the publication day onward, not from the load day.
    assert store.short_interest_as_of("ACME", "2026-11-12") is not None


# ------------------------------------------------------------------- live

@pytest.mark.network
@pytest.mark.skipif(SKIP_NETWORK, reason=(
    "SKIP_NETWORK_TESTS=1: this test reads FINRA's short-interest API. "
    "Unset it to run."))
def test_finra_answers_with_a_settlement_this_client_can_read(store):
    """The shape, not the numbers: a settlement appears, it covers thousands of
    names, and the fields the store needs are present."""
    try:
        day = short_interest.newest_settlement()
        if day is None:
            pytest.skip("FINRA has no settlement in the last 60 days")
        rows = short_interest.fetch_settlement(day)
    except short_interest.ShortInterestError as exc:
        skip_if_provider_unavailable(
            {"reason": exc.reason, "error": exc.message}, "FINRA")
        raise

    assert day < short_interest._today(), day
    assert len(rows) > 1000, len(rows)
    sample = rows[0]
    assert sample["ticker"] and sample["ticker"].isupper(), sample
    assert sample["shares_short"] >= 0, sample
    assert isinstance(sample["issue_name"], str), sample
    # At least some names carry a days-to-cover figure; a settlement where none
    # does would mean the field was renamed.
    assert any(r["days_to_cover"] is not None for r in rows[:200]), sample
