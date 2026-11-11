"""What the Discord bot reads from the store for /orders and /ticker.

Each reader reads as of a date, through the store's own point-in-time
readers, so a reply says what Nemo knew on that date.
"""
from __future__ import annotations

import pytest

from research import lookup, pit_store

DECIDED = "2026-11-10"
SESSION = "2026-11-11"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return tmp_path


def _order(ticker, *, as_of=DECIDED, side="long", sue=1.4, dollars=3000.0,
           rank=1):
    return {"ticker": ticker, "side": side, "fiscal_period": "2026Q3",
            "sue": sue, "variant": "ts", "target_dollars": dollars,
            "net_edge_bps": 20.0, "rank": rank, "intended_session": SESSION}


def _file(as_of, accepted=(), rejected=()):
    pit_store.record_paper_orders(as_of, accepted, rejected,
                                  recorded_at=f"{as_of}T21:00:00Z")


def _leg(ticker, *, order_as_of=DECIDED, leg="entry", side="buy", qty=10,
         status="filled", filled=None, price=None, session=SESSION,
         table="paper_fill", attempt=1, reason=None):
    cid = f"{table}-{order_as_of}-{ticker}-{leg}-{attempt}"
    pit_store.record_fill_submission({
        "order_as_of": order_as_of, "ticker": ticker, "leg": leg,
        "attempt": attempt, "scheduled_session": session, "session": session,
        "side": side, "qty": qty, "client_order_id": cid,
        "status": "submitted" if status != "skipped" else "skipped",
        "reason": reason, "submitted_at": f"{session}T13:15:00Z"}, table=table)
    if status not in ("submitted", "skipped"):
        pit_store.update_paper_fill(
            cid, status, filled, price,
            f"{session}T14:30:00Z" if filled else None, table=table)


def _copy(txn, *, ticker="AAA", accepted=False, reason="no_ticker",
          as_of=DECIDED, member="Ada Lovelace", side=None, dollars=None,
          amount=(1001, 15000)):
    pit_store.record_copy_orders([{
        "txn_id": txn, "as_of_date": as_of,
        "decided_at": f"{as_of}T12:30:00+00:00",
        "learned_at": f"{as_of}T06:00:00+00:00", "filing_id": f"f-{txn}",
        "member_id": "m1", "member": member, "chamber": "senate",
        "owner": "self", "source_ticker": ticker, "ticker": ticker,
        "cik": None, "amount_min": amount[0], "amount_max": amount[1],
        "transaction_date": "2026-10-20", "filed_date": "2026-10-29",
        "accepted": int(accepted), "reason": None if accepted else reason,
        "side": side, "target_dollars": dollars,
        "intended_session": SESSION if accepted else None}])


# ------------------------------------------------------------- /orders

def test_orders_are_the_latest_decision_each_with_its_entry_fill(store):
    _file("2026-11-09", [_order("OLD")])
    _file(DECIDED, [_order("AAA", rank=1), _order("BBB", rank=2),
                    _order("CCC", rank=3)],
          [{"ticker": "ZZZ", "reason": "|SUE| 0.08 under the 1.0 threshold"}])
    _leg("AAA", qty=10, status="filled", filled=10, price=50.0)
    _leg("BBB", qty=40, status="expired", filled=0)

    book = lookup.orders(SESSION)["earnings"]

    assert (book["decided"], book["session"]) == (DECIDED, SESSION)
    assert [o["ticker"] for o in book["orders"]] == ["AAA", "BBB", "CCC"]
    entries = {o["ticker"]: o["entry"] for o in book["orders"]}
    assert entries["AAA"]["status"] == "filled"
    assert entries["AAA"]["filled_price"] == 50.0
    assert entries["BBB"]["status"] == "expired"
    assert entries["CCC"] is None


def test_orders_carry_the_exits_sent_for_the_same_session(store):
    _file(DECIDED, [_order("AAA")])
    _leg("AAA", qty=10, status="filled", filled=10, price=50.0)
    _leg("OLD", order_as_of="2026-11-02", leg="exit", side="sell", qty=5,
         status="filled", filled=5, price=12.0)
    _leg("OLDER", order_as_of="2026-10-26", leg="exit", side="sell", qty=5,
         status="filled", filled=5, price=12.0, session="2026-11-02")

    book = lookup.orders(SESSION)["earnings"]

    assert [e["ticker"] for e in book["exits"]] == ["OLD"]


def test_an_exit_sent_again_shows_only_its_latest_attempt(store):
    _file(DECIDED, [_order("AAA")])
    _leg("OLD", order_as_of="2026-11-02", leg="exit", side="sell", qty=5,
         status="expired", filled=2, price=12.0)
    _leg("OLD", order_as_of="2026-11-02", leg="exit", side="sell", qty=3,
         status="filled", filled=3, price=12.5, attempt=2)

    [exit_leg] = lookup.orders(SESSION)["earnings"]["exits"]

    assert exit_leg["attempt"] == 2


def test_a_store_with_no_decision_has_no_earnings_book(store):
    assert lookup.orders(SESSION)["earnings"] is None


def test_copy_orders_are_the_latest_decisions_and_their_refusals(store):
    _copy("t0", as_of="2026-11-09")
    _copy("t1", ticker="AAA", accepted=True, side="long", dollars=5000.0)
    _copy("t2", ticker=None)
    _copy("t3", ticker=None)
    _copy("t4", ticker="BBB", reason="stale")
    _leg("AAA", table="copy_fill", qty=20, status="filled", filled=20,
         price=25.0)

    copy = lookup.orders(SESSION)["copy"]

    assert copy["decided"] == DECIDED
    assert copy["seen"] == 4
    [accepted] = copy["orders"]
    assert accepted["ticker"] == "AAA"
    assert accepted["entry"]["filled_price"] == 25.0
    assert copy["refused"] == {"no_ticker": 2, "stale": 1}


def test_a_decision_recorded_after_the_date_is_not_read(store):
    pit_store.record_paper_orders(DECIDED, [_order("AAA")],
                                  recorded_at="2026-11-12T21:00:00Z")

    assert lookup.orders(SESSION)["earnings"] is None


# ------------------------------------------------------------- /ticker

@pytest.mark.parametrize("raw, symbol", [("tmo", "TMO"), (" brk.b ", "BRK.B"),
                                         ("BF-B", "BF-B")])
def test_a_symbol_is_read_in_capitals(raw, symbol):
    assert lookup.symbol(raw) == symbol


@pytest.mark.parametrize("raw", ["", "  ", "TMO; drop", "$TMO", "A" * 11])
def test_a_symbol_that_is_not_a_ticker_is_refused_by_name(raw):
    with pytest.raises(ValueError, match="not a ticker"):
        lookup.symbol(raw)


def test_a_ticker_has_its_latest_signal_even_when_it_was_not_traded(store):
    _file("2026-11-09", [_order("AAA", sue=1.6)])
    _file(DECIDED, rejected=[{"ticker": "AAA", "sue": 0.4, "reason":
                              "|SUE| 0.40 under the 1.0 threshold"},
                             {"ticker": "BBB", "sue": 0.1, "reason":
                              "|SUE| 0.10 under the 1.0 threshold"}])

    found = lookup.ticker("AAA", SESSION)

    assert found["signal"]["as_of_date"] == DECIDED
    assert found["signal"]["accepted"] is False
    assert found["signal"]["reason"].startswith("|SUE| 0.40")


def test_a_ticker_lists_its_orders_newest_first_with_their_legs(store):
    for day in ("2026-11-04", "2026-11-05", "2026-11-06", DECIDED):
        _file(day, [_order("AAA")])
    _file(DECIDED, [_order("BBB")])
    _leg("AAA", qty=10, status="filled", filled=10, price=50.0)
    _leg("AAA", order_as_of="2026-11-06", leg="entry", qty=10,
         status="filled", filled=10, price=40.0, session="2026-11-09")
    _leg("AAA", order_as_of="2026-11-06", leg="exit", side="sell", qty=10,
         status="filled", filled=10, price=44.0)

    orders = lookup.ticker("AAA", SESSION)["orders"]

    assert [o["as_of_date"] for o in orders] == [DECIDED, "2026-11-06",
                                                "2026-11-05"]
    assert orders[0]["entry"]["filled_price"] == 50.0
    assert orders[0]["exit"] is None
    assert orders[1]["exit"]["filled_price"] == 44.0


def test_a_ticker_has_its_releases_filings_and_congress_trades(store):
    pit_store.record_announcement("AAA", "2026Q2", "2026-07-22", "amc",
                                  recorded_at="2026-07-22T21:00:00Z")
    pit_store.record_announcement("AAA", "2026Q3", "2026-10-21", "bmo",
                                  recorded_at="2026-10-21T21:00:00Z")
    pit_store.record_activist_filings([{
        "accession": "0001-26-000001", "subject_ticker": "AAA",
        "filer_name": "Example LP", "form": "SC 13D",
        "filing_date": "2026-10-01", "url": "https://www.sec.gov/x.htm"}],
        recorded_at="2026-10-01T15:00:00Z")
    _copy("t1", ticker="AAA", reason="stale")
    _copy("t2", ticker="BBB", reason="stale")

    found = lookup.ticker("AAA", SESSION)

    assert [a["fiscal_period"] for a in found["releases"]] == ["2026Q3",
                                                               "2026Q2"]
    assert [f["form"] for f in found["filings"]] == ["SC 13D"]
    assert [c["txn_id"] for c in found["congress"]] == ["t1"]


def test_a_ticker_nemo_never_saw_has_empty_lists_not_an_error(store):
    found = lookup.ticker("AAA", SESSION)

    assert found["signal"] is None
    assert found["orders"] == found["releases"] == []
    assert found["filings"] == found["congress"] == []
