"""Paper fills: the scan's orders sent to a paper broker, and what it filled.

Issue #116. Every result so far is net of modeled costs -- a spread estimated
from daily bars plus an impact term -- and the release-timed arm's median trade
is about zero, so an error of a few tens of basis points in that model changes
its sign. The job sends each filed order to an Alpaca paper account for the
opening auction of the session the scan intended, records the fill, and the
report sets each fill against the open the scoring assumes.

It runs from its own image. The data image serves market data and holds no
positions; the one module that can place an order is not in it.

A fake broker stands in for Alpaca throughout. Paper fills are simulated from
quotes, so they measure the quoted spread at the open, not the price impact of
the order itself -- the report says so, and only small live orders measure
that.
"""
import asyncio
import re
from datetime import date, timedelta

import pytest
import yaml

from research import pit_store, scoring, spread
from tools.alpaca import fills
from tools.alpaca.async_broker import AsyncBrokerError

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _weekdays(start, n):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


SESSIONS = _weekdays("2026-06-01", 40)
HOLIDAY = SESSIONS[10]
CALENDAR = [d for d in SESSIONS if d != HOLIDAY]


class FakeBroker:
    """Alpaca's calendar, order, position and order-status calls, in memory.

    `positions` mirrors what `AsyncBroker.get_open_position` returns: a map of
    symbol to `{qty, side}`, or omit the symbol for a 404. `position_error`
    raises that exception instead, the way a broker outage does.
    """

    def __init__(self, calendar=CALENDAR, reject=(), positions=None,
                 position_error=None):
        self.calendar = list(calendar)
        self.reject = set(reject)
        self.positions = dict(positions or {})
        self.position_error = position_error
        self.sent = []
        self.orders = {}

    async def get_calendar(self, start, end):
        return [d for d in self.calendar if start <= d <= end]

    async def get_open_position(self, symbol):
        if self.position_error is not None:
            raise self.position_error
        pos = self.positions.get(symbol.upper())
        if pos is None:
            return None
        return {"symbol": symbol.upper(), "qty": float(pos["qty"]),
                "side": pos.get("side", "long")}

    async def submit_market_order(self, symbol, qty, side, *,
                                  client_order_id, time_in_force="day"):
        if symbol in self.reject:
            raise AsyncBrokerError("HTTP 422: asset is not tradable")
        self.sent.append({"symbol": symbol, "qty": qty, "side": side,
                          "client_order_id": client_order_id,
                          "time_in_force": time_in_force})
        order_id = f"b-{client_order_id}"
        self.orders[order_id] = {"id": order_id, "status": "accepted",
                                 "filled_qty": "0", "filled_avg_price": None,
                                 "filled_at": None}
        return {"id": order_id, "client_order_id": client_order_id,
                "status": "accepted", "symbol": symbol, "qty": qty,
                "side": side, "filled_at": None}

    async def get_order_by_id(self, order_id):
        return self.orders[order_id]

    def fill(self, client_order_id, price, qty, at):
        self.orders[f"b-{client_order_id}"].update({
            "status": "filled", "filled_qty": str(qty),
            "filled_avg_price": str(price), "filled_at": at})

    def expire(self, client_order_id, filled_qty=0, price=None, at=None):
        """An auction order that ended unfilled or part-filled, as paper does."""
        update = {"status": "expired", "filled_qty": str(filled_qty)}
        if filled_qty and price is not None:
            update["filled_avg_price"] = str(price)
            update["filled_at"] = at
        self.orders[f"b-{client_order_id}"].update(update)


def _bars(ticker, price_on, days=SESSIONS):
    """Each session's bar recorded that evening, as the recorder would."""
    for d in days:
        if d == HOLIDAY:
            continue
        pit_store.record_bars(ticker, [
            {"trade_date": d, "open": price_on(d), "high": price_on(d) * 1.01,
             "low": price_on(d) * 0.99, "close": price_on(d),
             "volume": 2_000_000}], recorded_at=f"{d}T21:00:00Z")


def _order(ticker, filed, session, target=5_000.0, side="long", cost=10.0):
    pit_store.record_paper_orders(
        filed,
        [{"ticker": ticker, "side": side, "sue": 3.0, "variant": "ts",
          "strength": 3.0, "fiscal_period": "2026Q2",
          "expected_edge_bps": 45.0, "cost_bps": cost, "net_edge_bps": 35.0,
          "target_dollars": target, "intended_session": session, "rank": 1}],
        recorded_at=f"{filed}T21:00:00Z")


def _submit(broker, today, horizon=5):
    return asyncio.run(fills.submit(broker, today=today, horizon_days=horizon))


def _collect(broker, today):
    return asyncio.run(fills.collect(broker, today=today))


def _cid(order_as_of, ticker, leg, attempt=1):
    return fills.client_order_id(order_as_of, ticker, leg, attempt=attempt)


def _enter_and_fill(ticker, qty=50, price=100.0, filed=None, session=None,
                    horizon=5):
    """File, send and fill an entry; return the broker holding that qty long."""
    filed = filed or SESSIONS[3]
    session = session or SESSIONS[4]
    _bars(ticker, lambda d: price)
    _order(ticker, filed=filed, session=session)
    broker = FakeBroker(positions={ticker: {"qty": qty, "side": "long"}})
    _submit(broker, today=session, horizon=horizon)
    broker.fill(_cid(filed, ticker, "entry"), price=price, qty=qty,
                at=f"{session}T13:30:01Z")
    _collect(broker, today=session)
    return broker, filed, session


# --- what gets sent, and when ------------------------------------------------

def test_an_order_due_today_goes_to_the_opening_auction_in_whole_shares(store):
    _bars("AAA", lambda d: 97.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()

    out = _submit(broker, today=SESSIONS[4])

    assert broker.sent == [{
        "symbol": "AAA", "qty": 51, "side": "buy",
        "client_order_id": _cid(SESSIONS[3], "AAA", "entry"),
        "time_in_force": "opg"}]
    assert out["submitted"] == 1


def test_a_short_enters_by_selling(store):
    _bars("BBB", lambda d: 100.0)
    _order("BBB", filed=SESSIONS[3], session=SESSIONS[4], side="short")
    broker = FakeBroker()

    _submit(broker, today=SESSIONS[4])

    assert broker.sent[0]["side"] == "sell"


def test_nothing_is_sent_on_a_day_the_market_is_shut(store):
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[9], session=HOLIDAY)
    broker = FakeBroker()

    out = _submit(broker, today=HOLIDAY)

    assert broker.sent == [] and out["status"] == "closed"


def test_an_order_for_a_holiday_enters_at_the_next_open(store):
    """The scan names the next weekday; scoring rolls a holiday to the next
    open, so the fill must be measured there too."""
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[9], session=HOLIDAY)
    broker = FakeBroker()

    _submit(broker, today=SESSIONS[11])

    assert [s["symbol"] for s in broker.sent] == ["AAA"]


def test_a_second_run_the_same_morning_sends_nothing(store):
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()

    _submit(broker, today=SESSIONS[4])
    _submit(broker, today=SESSIONS[4])

    assert len(broker.sent) == 1


def test_an_entry_the_job_missed_is_not_sent_late(store):
    """Entered a day late it is a different trade from the one the book
    scores, and measuring it would compare two trades."""
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()

    out = _submit(broker, today=SESSIONS[5])

    assert broker.sent == []
    assert out["missed"] == 1


def test_a_target_below_one_share_is_recorded_not_sent(store):
    _bars("PRICEY", lambda d: 9_000.0)
    _order("PRICEY", filed=SESSIONS[3], session=SESSIONS[4], target=5_000.0)
    broker = FakeBroker()

    _submit(broker, today=SESSIONS[4])

    row = pit_store.paper_fills_as_of(SESSIONS[4])[0]
    assert broker.sent == []
    assert row["status"] == "skipped" and "one share" in row["reason"]


def test_a_rejected_order_is_recorded_and_the_rest_still_go(store):
    """The refused name sorts first, so a job that stopped at the first
    refusal would send nothing."""
    _bars("BAD", lambda d: 100.0)
    _bars("GOOD", lambda d: 100.0)
    _order("BAD", filed=SESSIONS[3], session=SESSIONS[4])
    _order("GOOD", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker(reject={"BAD"})

    out = _submit(broker, today=SESSIONS[4])

    rows = {r["ticker"]: r for r in pit_store.paper_fills_as_of(SESSIONS[4])}
    assert [s["symbol"] for s in broker.sent] == ["GOOD"]
    assert rows["BAD"]["status"] == "rejected"
    assert "not tradable" in rows["BAD"]["reason"]
    assert out["status"] == "partial"


# --- what comes back ---------------------------------------------------------

def test_a_fill_is_recorded_and_a_later_look_does_not_change_it(store):
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    cid = _cid(SESSIONS[3], "AAA", "entry")
    broker.fill(cid, price=100.2, qty=50, at=f"{SESSIONS[4]}T13:30:01Z")

    _collect(broker, today=SESSIONS[4])
    broker.orders[f"b-{cid}"]["filled_avg_price"] = "999.0"
    _collect(broker, today=SESSIONS[4])

    row = pit_store.paper_fills_as_of(SESSIONS[4])[0]
    assert row["status"] == "filled"
    assert row["filled_price"] == pytest.approx(100.2)
    assert row["filled_qty"] == pytest.approx(50)


def test_a_fill_is_not_visible_before_it_happened(store):
    """Sent on one day and reported filled the next: on the first day the
    price did not exist yet."""
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    broker.fill(_cid(SESSIONS[3], "AAA", "entry"), price=100.0, qty=50,
                at=f"{SESSIONS[5]}T13:30:01Z")
    _collect(broker, today=SESSIONS[5])

    then = pit_store.paper_fills_as_of(SESSIONS[4])[0]
    later = pit_store.paper_fills_as_of(SESSIONS[5])[0]

    assert then["status"] == "pending" and then["filled_price"] is None
    assert later["status"] == "filled"


def test_the_exit_goes_out_horizon_sessions_after_the_entry(store):
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    broker.fill(_cid(SESSIONS[3], "AAA", "entry"), price=100.0, qty=50,
                at=f"{SESSIONS[4]}T13:30:01Z")
    _collect(broker, today=SESSIONS[4])

    for day in SESSIONS[5:9]:
        _submit(broker, today=day)
    assert len(broker.sent) == 1, "the exit went out before its session"

    _submit(broker, today=SESSIONS[9])

    assert broker.sent[-1] == {
        "symbol": "AAA", "qty": 50, "side": "sell",
        "client_order_id": _cid(SESSIONS[3], "AAA", "exit"),
        "time_in_force": "opg"}


def test_no_exit_is_sent_for_an_entry_that_never_filled(store):
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    broker.orders[f"b-{_cid(SESSIONS[3], 'AAA', 'entry')}"]["status"] = "canceled"
    _collect(broker, today=SESSIONS[4])

    _submit(broker, today=SESSIONS[9])

    assert len(broker.sent) == 1
    assert [r for r in pit_store.paper_fills_as_of(SESSIONS[9])
            if r["leg"] == "exit"] == [], "an exit was planned for no shares"


def test_a_part_filled_entry_is_exited_for_what_it_holds(store):
    """Twenty of fifty filled, the rest cancelled: twenty shares are held."""
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    broker.orders[f"b-{_cid(SESSIONS[3], 'AAA', 'entry')}"].update({
        "status": "canceled", "filled_qty": "20", "filled_avg_price": "100.1",
        "filled_at": f"{SESSIONS[4]}T13:30:01Z"})
    _collect(broker, today=SESSIONS[4])

    _submit(broker, today=SESSIONS[9])

    assert broker.sent[-1]["side"] == "sell" and broker.sent[-1]["qty"] == 20


def test_a_late_exit_flattens_the_position_and_is_not_measured(store):
    """The job was down on the exit day. The exit still goes, the next day it
    runs, and is left out: it is not the trade the book scores."""
    _bars("AAA", lambda d: 100.0)
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    broker.fill(_cid(SESSIONS[3], "AAA", "entry"), price=100.0, qty=50,
                at=f"{SESSIONS[4]}T13:30:01Z")
    _collect(broker, today=SESSIONS[4])

    _submit(broker, today=SESSIONS[11])
    broker.fill(_cid(SESSIONS[3], "AAA", "exit"), price=101.0, qty=50,
                at=f"{SESSIONS[11]}T13:30:01Z")
    _collect(broker, today=SESSIONS[11])

    exit_row = [r for r in pit_store.paper_fills_as_of(SESSIONS[11])
                if r["leg"] == "exit"][0]
    assert broker.sent[-1]["client_order_id"] == _cid(SESSIONS[3], "AAA",
                                                      "exit")
    assert "late" in exit_row["reason"]
    assert pit_store.measured_round_trips(SESSIONS[11]) == []


# --- exit retries when the auction leaves shares held (issue #189) -----------

def test_a_zero_fill_exit_is_resent_next_session_for_the_full_remainder(store):
    """An opening-auction exit that expires with nothing filled still holds
    the entry; the next session must send the same qty again."""
    broker, filed, _ = _enter_and_fill("AAA", qty=50)
    _submit(broker, today=SESSIONS[9])
    broker.expire(_cid(filed, "AAA", "exit"), filled_qty=0)
    _collect(broker, today=SESSIONS[9])

    _submit(broker, today=SESSIONS[11])

    assert broker.sent[-1] == {
        "symbol": "AAA", "qty": 50, "side": "sell",
        "client_order_id": _cid(filed, "AAA", "exit", attempt=2),
        "time_in_force": "opg"}


def test_a_part_filled_exit_is_resent_for_the_rest(store):
    """Paper injects partial fills on auction orders; the unfilled remainder
    expires and must go out again for only what is still held."""
    broker, filed, _ = _enter_and_fill("AAA", qty=50)
    _submit(broker, today=SESSIONS[9])
    broker.expire(_cid(filed, "AAA", "exit"), filled_qty=20, price=101.0,
                  at=f"{SESSIONS[9]}T13:30:01Z")
    _collect(broker, today=SESSIONS[9])
    broker.positions["AAA"] = {"qty": 30, "side": "long"}

    _submit(broker, today=SESSIONS[11])

    assert broker.sent[-1]["qty"] == 30
    assert broker.sent[-1]["client_order_id"] == _cid(
        filed, "AAA", "exit", attempt=2)


def test_no_exit_is_resent_once_the_position_is_flat(store):
    """When exit fills have summed to the entry, there is nothing left to
    send — a second morning must not invent another sell."""
    broker, filed, _ = _enter_and_fill("AAA", qty=50)
    _submit(broker, today=SESSIONS[9])
    broker.fill(_cid(filed, "AAA", "exit"), price=101.0, qty=50,
                at=f"{SESSIONS[9]}T13:30:01Z")
    _collect(broker, today=SESSIONS[9])
    before = len(broker.sent)

    _submit(broker, today=SESSIONS[11])

    assert len(broker.sent) == before


def test_exit_resends_are_marked_late_and_not_measured(store):
    """A remainder exit is a flatten, not the trade the book scores: late
    reason, and measured_round_trips ignores every attempt after the first."""
    broker, filed, _ = _enter_and_fill("AAA", qty=50)
    _submit(broker, today=SESSIONS[9])
    broker.expire(_cid(filed, "AAA", "exit"), filled_qty=0)
    _collect(broker, today=SESSIONS[9])

    _submit(broker, today=SESSIONS[11])
    broker.fill(_cid(filed, "AAA", "exit", attempt=2), price=101.0, qty=50,
                at=f"{SESSIONS[11]}T13:30:01Z")
    _collect(broker, today=SESSIONS[11])

    retry = [r for r in pit_store.paper_fills_as_of(SESSIONS[11])
             if r["leg"] == "exit" and r.get("attempt", 1) == 2][0]
    assert "late" in (retry["reason"] or "")
    assert pit_store.measured_round_trips(SESSIONS[11]) == []


def test_paper_fill_migration_keeps_existing_rows_and_is_safe_to_run_twice(store):
    """A store created before attempt existed must keep every row as attempt
    1, and a second migrate must not duplicate or drop them."""
    import sqlite3

    path = pit_store.db_path()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE IF EXISTS paper_fill")
        conn.execute("""
            CREATE TABLE paper_fill (
                order_as_of TEXT NOT NULL, ticker TEXT NOT NULL,
                leg TEXT NOT NULL, scheduled_session TEXT NOT NULL,
                session TEXT NOT NULL, side TEXT NOT NULL, qty INTEGER NOT NULL,
                client_order_id TEXT NOT NULL UNIQUE, broker_order_id TEXT,
                status TEXT NOT NULL, reason TEXT, filled_qty REAL,
                filled_price REAL, filled_at TEXT, submitted_at TEXT NOT NULL,
                PRIMARY KEY (order_as_of, ticker, leg)
            )""")
        conn.execute(
            """INSERT INTO paper_fill
               (order_as_of, ticker, leg, scheduled_session, session, side,
                qty, client_order_id, broker_order_id, status, reason,
                filled_qty, filled_price, filled_at, submitted_at)
               VALUES ('2026-06-04','AAA','entry','2026-06-05','2026-06-05',
                       'buy',50,'cid-old',NULL,'filled',NULL,50,100.0,
                       '2026-06-05T13:30:01Z','2026-06-05T13:15:00Z')""")

    pit_store.init_schema()
    pit_store.init_schema()

    rows = pit_store.paper_fills_as_of("2026-06-05")
    assert len(rows) == 1
    assert rows[0]["client_order_id"] == "cid-old"
    assert rows[0]["attempt"] == 1
    assert rows[0]["filled_qty"] == pytest.approx(50)


def test_a_remainder_larger_than_the_broker_position_is_skipped_naming_held_qty(
        store):
    """Never sell shares the account does not hold: a remainder of 50 against
    a broker long of 10 is skipped, and the reason names the 10."""
    broker, filed, _ = _enter_and_fill("AAA", qty=50)
    _submit(broker, today=SESSIONS[9])
    broker.expire(_cid(filed, "AAA", "exit"), filled_qty=0)
    _collect(broker, today=SESSIONS[9])
    broker.positions["AAA"] = {"qty": 10, "side": "long"}
    before = len(broker.sent)

    out = _submit(broker, today=SESSIONS[11])

    assert len(broker.sent) == before
    retry = [r for r in pit_store.paper_fills_as_of(SESSIONS[11])
             if r["leg"] == "exit" and r.get("attempt", 1) == 2][0]
    assert retry["status"] == "skipped"
    assert "10" in (retry["reason"] or "")
    assert out["skipped"] == 1


def test_a_broker_position_lookup_error_is_recorded_as_rejected(store):
    """A broker error on the position check is a refusal with that error as
    the reason — never a silent 'not held'."""
    broker, filed, _ = _enter_and_fill("AAA", qty=50)
    _submit(broker, today=SESSIONS[9])
    broker.expire(_cid(filed, "AAA", "exit"), filled_qty=0)
    _collect(broker, today=SESSIONS[9])
    broker.position_error = AsyncBrokerError("HTTP 503: broker unavailable")
    before = len(broker.sent)

    out = _submit(broker, today=SESSIONS[11])

    assert len(broker.sent) == before
    retry = [r for r in pit_store.paper_fills_as_of(SESSIONS[11])
             if r["leg"] == "exit" and r.get("attempt", 1) == 2][0]
    assert retry["status"] == "rejected"
    assert "503" in (retry["reason"] or "")
    assert "not held" not in (retry["reason"] or "").lower()
    assert out["rejected"] == 1


def test_the_store_will_not_restate_a_fill(store):
    """The job already skips settled legs; the store refuses as well, so a
    later caller cannot rewrite a price either."""
    pit_store.record_fill_submission({
        "order_as_of": SESSIONS[3], "ticker": "AAA", "leg": "entry",
        "scheduled_session": SESSIONS[4], "session": SESSIONS[4],
        "side": "buy", "qty": 50, "client_order_id": "cid-1",
        "broker_order_id": "b-1", "status": "accepted",
        "submitted_at": f"{SESSIONS[4]}T13:15:00Z"})
    pit_store.update_paper_fill("cid-1", "filled", 50, 100.2,
                                f"{SESSIONS[4]}T13:30:01Z")

    changed = pit_store.update_paper_fill("cid-1", "filled", 50, 999.0,
                                          f"{SESSIONS[4]}T13:31:00Z")

    assert changed == 0
    assert pit_store.paper_fills_as_of(SESSIONS[4])[0]["filled_price"] == 100.2


# --- what the fills say ------------------------------------------------------

def _round_trip(store, entry_fill, exit_fill, opens=(100.0, 110.0)):
    _bars("AAA", lambda d: opens[1] if d >= SESSIONS[9] else opens[0])
    _order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    _submit(broker, today=SESSIONS[4])
    broker.fill(_cid(SESSIONS[3], "AAA", "entry"), price=entry_fill, qty=50,
                at=f"{SESSIONS[4]}T13:30:01Z")
    _collect(broker, today=SESSIONS[4])
    _submit(broker, today=SESSIONS[9])
    broker.fill(_cid(SESSIONS[3], "AAA", "exit"), price=exit_fill, qty=50,
                at=f"{SESSIONS[9]}T13:30:01Z")
    _collect(broker, today=SESSIONS[9])


def test_a_completed_round_trip_is_measured_from_the_fills(store):
    _round_trip(store, entry_fill=100.0, exit_fill=110.0)

    trips = pit_store.measured_round_trips(SESSIONS[9])

    assert len(trips) == 1
    assert trips[0]["entry_price"] == 100.0 and trips[0]["exit_price"] == 110.0


def test_a_round_trip_is_not_measured_before_its_exit_filled(store):
    _round_trip(store, entry_fill=100.0, exit_fill=110.0)

    assert pit_store.measured_round_trips(SESSIONS[8]) == []


def test_the_report_sets_each_fill_against_the_open_the_scoring_assumes(store):
    """Bought 20bp above the open, sold 9.1bp below it: 29.1bp paid, against
    the 10bp the model charged."""
    _round_trip(store, entry_fill=100.2, exit_fill=109.9)

    report = fills.report(SESSIONS[9])

    assert report["round_trips"] == 1
    assert report["mean_measured_cost_bps"] == pytest.approx(
        20.0 + (110.0 - 109.9) / 110.0 * 10_000, abs=0.01)
    assert report["mean_modeled_cost_bps"] == pytest.approx(10.0)
    assert "impact" in report["note"]


# --- where it runs -----------------------------------------------------------

def _stages(dockerfile):
    stages, current = {}, None
    for line in dockerfile.splitlines():
        match = re.match(r"FROM\s+(\S+)\s+AS\s+(\S+)", line, re.I)
        if match:
            current = match.group(2)
            stages[current] = {"from": match.group(1), "lines": []}
        elif current:
            stages[current]["lines"].append(line)
    return stages


def test_the_default_image_cannot_place_an_order():
    """`docker build` with no target builds the last stage. Nothing it is
    built from may carry the broker."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    stages = _stages(dockerfile)
    last = list(stages)[-1]
    lineage, name = [], last
    while name in stages:
        lineage.append(name)
        name = stages[name]["from"]
    sources = set(lineage)
    for stage in lineage:
        sources |= set(re.findall(r"--from=(\S+)",
                                  "\n".join(stages[stage]["lines"])))

    assert last == "runtime"
    for stage in sources & set(stages):
        copies = [line for line in stages[stage]["lines"]
                  if line.startswith("COPY")]
        assert not [c for c in copies if "alpaca" in c], (
            f"the default image is built from `{stage}`, which copies the "
            f"broker")


def test_the_fill_image_is_its_own_target_and_carries_the_broker():
    stages = _stages((ROOT / "Dockerfile").read_text())

    assert "fills" in stages
    body = "\n".join(stages["fills"]["lines"]
                     + stages.get("fills-src", {"lines": []})["lines"])
    assert "tools/alpaca" in body


def test_only_the_fill_service_is_handed_a_broker_key():
    compose = yaml.safe_load(
        (ROOT / "deploy" / "docker-compose.yml").read_text())
    holders = sorted(name for name, service in compose["services"].items()
                     if any(k.startswith("ALPACA")
                            for k in (service.get("environment") or {})))

    assert holders == ["research-fills"]
    assert "nemo-fills" in compose["services"]["research-fills"]["image"]


# --- the gate reads them -----------------------------------------------------

def test_the_gate_measures_a_round_trip_from_its_fills(store):
    _round_trip(store, entry_fill=100.0, exit_fill=110.0)

    assert scoring.measured_net_bps(SESSIONS[9]) == [pytest.approx(1000.0)]


def test_the_weekly_score_counts_measured_trades_in_the_gate(store):
    _bars(spread.REFERENCE_TICKER, lambda d: 50.0)   # scoring's calendar
    _round_trip(store, entry_fill=100.0, exit_fill=110.0)

    scoring.score_orders(as_of=SESSIONS[9], horizon_days=5)

    gate = pit_store.latest_gate_check(SESSIONS[9])
    assert gate["measured_trades"] == 1
    assert "1 of 200" in gate["reason"]
