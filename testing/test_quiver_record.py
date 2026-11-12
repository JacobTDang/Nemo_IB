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
