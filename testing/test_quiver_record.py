"""The nightly QuiverQuant recorder: row keys, stages and the entry point."""
from datetime import datetime, timedelta, timezone

import pytest

from research import pit_store
from research import quiver_record as qr
from tools.quiver_server.client import QuiverError


def feed(name):
    return next(f for f in qr.FEEDS + qr.LISTS if f.dataset == name)


def test_the_feeds_are_the_seven_the_spec_names():
    assert [f.dataset for f in qr.FEEDS] == [
        "congress_trade", "insider_trade", "layoff", "contract_award",
        "gov_spending", "lobbying_filing", "revolving_door"]
    assert [f.dataset for f in qr.LISTS] == [
        "congress_net_worth", "election_contribution", "lobbying_top_spender"]


def test_every_feed_names_how_its_rows_are_identified():
    for f in qr.FEEDS + qr.LISTS:
        assert f.id_field or f.identity, f.dataset


def test_congress_rows_are_keyed_by_quivers_own_id():
    [r] = qr.keyed_rows(feed("congress_trade"), [
        {"tx_id": "Senate-C001047-750", "ticker": "aapl",
         "filed_date": "2026-11-04"}])
    assert r["row_key"] == "Senate-C001047-750"
    assert r["ticker"] == "AAPL"
    assert r["event_date"] == "2026-11-04"


def test_a_congress_row_without_an_id_is_refused():
    with pytest.raises(ValueError, match="tx_id"):
        qr.keyed_rows(feed("congress_trade"), [{"ticker": "AAPL"}])


def test_two_identical_lobbying_filings_are_two_rows():
    filing = {"ticker": "EVGO", "amount": "$30,000", "date": "2026-11-11"}
    keys = [r["row_key"] for r in
            qr.keyed_rows(feed("lobbying_filing"), [filing, dict(filing)])]
    assert len(set(keys)) == 2


def test_the_occurrence_index_is_stable_across_nights():
    a = {"ticker": "EVGO", "amount": "$30,000", "date": "2026-11-11"}
    b = {"ticker": "META", "amount": "$10,000", "date": "2026-11-11"}
    first = {r["row_key"] for r in
             qr.keyed_rows(feed("lobbying_filing"), [a, dict(a), b])}
    second = {r["row_key"] for r in
              qr.keyed_rows(feed("lobbying_filing"), [b, a, dict(a)])}
    assert first == second


def test_a_field_quiver_fills_in_later_does_not_change_the_key():
    base = {"employer": "Acme", "notice_date": "2026-11-09", "city": "Austin",
            "state": "TX", "workers": 120, "kind": "Layoff", "ticker": None,
            "ticker_match": None}
    later = {**base, "ticker": "ACME", "ticker_match": "fuzzy"}
    [first] = qr.keyed_rows(feed("layoff"), [base])
    [second] = qr.keyed_rows(feed("layoff"), [later])
    assert first["row_key"] == second["row_key"]
    assert first["ticker"] is None and second["ticker"] == "ACME"


def test_a_contract_award_carries_no_ticker():
    [r] = qr.keyed_rows(feed("contract_award"), [
        {"recipient": "Acme", "date_awarded": "2026-11-05", "amount": "$5",
         "agency": "DOD", "description": "parts"}])
    assert r["ticker"] is None
    assert r["event_date"] == "2026-11-05"


# A Thursday, ISO week 2026-W46.
NOW = datetime(2026, 11, 12, 6, 30, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


class Scripted:
    """Answers each client method from a dict; an exception value is raised."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.closed = False

    def __getattr__(self, name):
        if not name.startswith("get_"):
            raise AttributeError(name)

        def call(**kwargs):
            self.calls.append(name)
            answer = self.answers[name]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return call

    def close(self):
        self.closed = True


def answers(**overrides):
    base = {
        "get_congress_trades": {"trades": [
            {"tx_id": "T1", "ticker": "AAPL", "filed_date": "2026-11-10"}]},
        "get_insider_trades": {"trades": [
            {"owner": "Jane", "ticker": "MSFT", "transaction_date": "2026-11-01",
             "shares": 10, "tx_type": "Sale", "filed_date": "2026-11-03",
             "list": "all", "net_worth_estimate": 1}]},
        "get_layoffs": {"notices": [
            {"employer": "Acme", "notice_date": "2026-11-09", "city": "Austin",
             "state": "TX", "workers": 120, "kind": "Layoff", "ticker": None}]},
        "get_gov_contract_awards": {"awards": [
            {"recipient": "Acme", "date_awarded": "2026-11-05", "amount": "$5",
             "agency": "DOD", "description": "parts"}]},
        "get_gov_spending": {"transactions": [
            {"award_id": "A1", "date": "2026-10-01", "amount_usd": 5.0,
             "recipient": "Acme", "agency": "DOD", "ticker": "ACME"}]},
        "get_lobbying_filings": {
            "filings": [{"ticker": "EVGO", "amount": "$30,000",
                         "date": "2026-11-11"}],
            "top_spenders": [{"ticker": "META", "amount": "$5,880,000"}]},
        "get_revolving_door": {"registrations": [
            {"lobbyist": "D", "client": "C", "firm": "F",
             "filed_date": "2026-11-11", "filing_type": "Q3",
             "period": "Q3 2026", "is_new": False, "ticker": None}]},
        "get_congress_net_worth": {"members": [
            {"bioguide_id": "S001229", "name": "Jefferson Shreve",
             "net_worth_usd": 7.2e8}]},
        "get_election_contributions": {"contributions": [
            {"ticker": "SPCX", "company": "SpaceX", "source": "employee",
             "cycle": "2026", "total_usd": 1.0}]},
    }
    base.update(overrides)
    return base


def last_run(store, job):
    with store.connect() as conn:
        return dict(conn.execute(
            "SELECT * FROM run_log WHERE job = ? ORDER BY run_id DESC LIMIT 1",
            (job,)).fetchone())


def test_the_week_is_the_utc_iso_week():
    assert qr.iso_week(NOW) == "2026-W46"


def test_a_full_night_records_every_feed_and_list(store):
    result = qr.run(Scripted(answers()), NOW)
    assert result["status"] == "ok"
    assert result["as_of"] == "2026-11-12"
    assert set(result["quiver_feeds"]["feeds"]) == {f.dataset for f in qr.FEEDS}
    assert set(result["quiver_lists"]["lists"]) == {f.dataset for f in qr.LISTS}
    assert result["written"] == 7 + 3
    assert last_run(store, "quiver_feeds")["status"] == "ok"
    assert last_run(store, "quiver_lists")["status"] == "ok"
    [trade] = store.quiver_records_as_of("congress_trade", "2026-11-12")
    assert trade["first_seen_at"] == "2026-11-12T06:30:00Z"


def test_a_second_night_over_the_same_rows_records_nothing_new(store):
    qr.run(Scripted(answers()), NOW)
    second = qr.run(Scripted(answers()), NOW + timedelta(days=1))
    assert second["status"] == "ok"
    assert second["quiver_feeds"]["written"] == 0
    assert all(c["read"] == 1 and c["new"] == 0
               for c in second["quiver_feeds"]["feeds"].values())


def test_one_failed_feed_is_named_and_the_others_still_record(store):
    result = qr.run(Scripted(answers(
        get_layoffs=QuiverError("rate_limited", "HTTP 429 after one retry"))),
        NOW)
    assert result["status"] == "partial"
    assert result["quiver_feeds"]["failed"]["layoff"].startswith("rate_limited")
    assert "quiver_feeds/layoff" in result["failed"]
    assert store.quiver_records_as_of("congress_trade", "2026-11-12")
    run = last_run(store, "quiver_feeds")
    assert run["status"] == "partial"
    assert "layoff: rate_limited" in run["error"]


def test_an_empty_feed_is_a_failure_not_a_quiet_night(store):
    result = qr.run(Scripted(answers(get_gov_contract_awards={"awards": []})),
                    NOW)
    assert result["status"] == "partial"
    assert result["quiver_feeds"]["failed"]["contract_award"].startswith("empty")


def test_a_feed_with_no_row_list_is_a_parse_failure(store):
    result = qr.run(Scripted(answers(get_layoffs={"unexpected": 1})), NOW)
    assert result["quiver_feeds"]["failed"]["layoff"].startswith("parse_failed")


def test_every_feed_failing_fails_the_stage(store):
    down = QuiverError("provider_unavailable", "connect timeout")
    everything_down = {name: down for name in answers()}
    result = qr.run(Scripted(everything_down), NOW)
    assert result["status"] == "failed"
    assert last_run(store, "quiver_feeds")["status"] == "failed"
    assert last_run(store, "quiver_lists")["status"] == "failed"


def test_weekly_lists_are_written_once_per_week(store):
    first = qr.run(Scripted(answers()), NOW)
    second = qr.run(Scripted(answers()), NOW + timedelta(days=1))
    assert first["quiver_lists"]["written"] == 3
    assert second["quiver_lists"]["written"] == 0
    assert sorted(second["quiver_lists"]["already_held"]) == sorted(
        f.dataset for f in qr.LISTS)
    assert last_run(store, "quiver_lists")["status"] == "ok"


def test_a_failed_list_is_retried_the_next_night(store):
    qr.run(Scripted(answers(get_congress_net_worth=QuiverError(
        "provider_unavailable", "connect timeout"))), NOW)
    second = qr.run(Scripted(answers()), NOW + timedelta(days=1))
    assert second["quiver_lists"]["lists"]["congress_net_worth"]["written"] == 1


def test_a_new_week_is_copied_again(store):
    qr.run(Scripted(answers()), NOW)
    later = qr.run(Scripted(answers()), NOW + timedelta(days=7))
    assert later["quiver_lists"]["week"] == "2026-W47"
    assert later["quiver_lists"]["written"] == 3
