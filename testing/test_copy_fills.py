"""Copy-book fills: accepted congressional purchases sent to the paper broker.

Issue #190 Part B. The decision job records `copy_order` rows and sends no
order. The fills job submits those accepted rows to Alpaca PAPER beside the
earnings book, keeps their fills in `copy_fill`, and never lets them into the
earnings gate or risk book.
"""
import asyncio
from datetime import date, timedelta

import pytest
import yaml

from research import pit_store, status
from tools.alpaca import fills
from tools.alpaca.async_broker import AsyncBrokerError

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    monkeypatch.setenv("NEMO_COPY_TRADING", "on")
    monkeypatch.setenv("NEMO_COPY_HOLD_DAYS", "5")
    for name in ("MIN_AMOUNT", "DOLLARS", "MAX_OPEN", "MAX_TRADE_AGE_DAYS",
                 "MEMBERS"):
        monkeypatch.delenv("NEMO_COPY_" + name, raising=False)
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
    """Alpaca calendar, order, position and status calls, in memory."""

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


def _bars(ticker, price_on, days=SESSIONS):
    for d in days:
        if d == HOLIDAY:
            continue
        pit_store.record_bars(ticker, [
            {"trade_date": d, "open": price_on(d), "high": price_on(d) * 1.01,
             "low": price_on(d) * 0.99, "close": price_on(d),
             "volume": 2_000_000}], recorded_at=f"{d}T21:00:00Z")


def _copy_order(ticker, filed, session, target=5_000.0, txn_id=None):
    pit_store.record_copy_orders([{
        "txn_id": txn_id or f"txn-{ticker}-{filed}",
        "as_of_date": filed,
        "decided_at": f"{filed}T12:30:00+00:00",
        "learned_at": f"{filed}T06:00:00+00:00",
        "filing_id": f"f-{ticker}",
        "member_id": "m1",
        "member": "Jane Smith",
        "chamber": "house",
        "owner": "self",
        "source_ticker": ticker,
        "ticker": ticker,
        "cik": "1",
        "amount_min": 15001,
        "amount_max": 50000,
        "transaction_date": filed,
        "filed_date": filed,
        "accepted": 1,
        "reason": None,
        "side": "long",
        "target_dollars": target,
        "intended_session": session,
    }])


def _sue_order(ticker, filed, session, target=5_000.0):
    pit_store.record_paper_orders(
        filed,
        [{"ticker": ticker, "side": "long", "sue": 3.0, "variant": "ts",
          "strength": 3.0, "fiscal_period": "2026Q2",
          "expected_edge_bps": 45.0, "cost_bps": 10.0, "net_edge_bps": 35.0,
          "target_dollars": target, "intended_session": session, "rank": 1}],
        recorded_at=f"{filed}T21:00:00Z")


def _submit(broker, today, horizon=5):
    return asyncio.run(fills.submit(broker, today=today, horizon_days=horizon))


def _collect(broker, today):
    return asyncio.run(fills.collect(broker, today=today))


def _copy_cid(order_as_of, ticker, leg, attempt=1):
    return fills.client_order_id(order_as_of, ticker, leg, attempt=attempt,
                                 book="copy")


def test_copy_fills_are_written_to_copy_fill_not_paper_fill(store):
    """A copy leg must never land in paper_fill: the earnings gate reads that
    table, and a copy trade must not become a measured earnings round trip."""
    _bars("AAA", lambda d: 100.0)
    _copy_order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()

    _submit(broker, today=SESSIONS[4])

    assert pit_store.fills_as_of(SESSIONS[4], table="copy_fill")
    assert pit_store.paper_fills_as_of(SESSIONS[4]) == []
    assert pit_store.measured_round_trips(SESSIONS[4]) == []


def test_copy_legs_use_nemo_copy_client_order_ids(store):
    """Earnings ids are nemo-...; copy ids are nemo-copy-..., so the broker
    and the store can never confuse one book's leg for the other's."""
    _bars("AAA", lambda d: 100.0)
    _copy_order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()

    _submit(broker, today=SESSIONS[4])

    assert broker.sent[0]["client_order_id"] == _copy_cid(
        SESSIONS[3], "AAA", "entry")
    assert broker.sent[0]["client_order_id"].startswith("nemo-copy-")
    assert not broker.sent[0]["client_order_id"].startswith("nemo-2026")


def test_turning_the_flag_off_still_exits_held_copy_positions(store, monkeypatch):
    """Off means no new entries. Positions already held must still flatten;
    leaving them open would strand shares nobody exits."""
    _bars("AAA", lambda d: 100.0)
    _bars("BBB", lambda d: 100.0)
    _copy_order("AAA", filed=SESSIONS[3], session=SESSIONS[4], txn_id="held")
    broker = FakeBroker(positions={"AAA": {"qty": 50, "side": "long"}})
    _submit(broker, today=SESSIONS[4])
    broker.fill(_copy_cid(SESSIONS[3], "AAA", "entry"), price=100.0, qty=50,
                at=f"{SESSIONS[4]}T13:30:01Z")
    _collect(broker, today=SESSIONS[4])

    monkeypatch.setenv("NEMO_COPY_TRADING", "off")
    _copy_order("BBB", filed=SESSIONS[8], session=SESSIONS[9], txn_id="new")
    before = len(broker.sent)

    _submit(broker, today=SESSIONS[9])

    ids = [s["client_order_id"] for s in broker.sent[before:]]
    assert _copy_cid(SESSIONS[3], "AAA", "exit") in ids
    assert not any("BBB" in i for i in ids)


def test_a_copy_book_failure_does_not_stop_earnings_legs(store, monkeypatch):
    """One book's exception must leave the other book's legs free to send, and
    the run must name the failure rather than swallow it."""
    _bars("EARN", lambda d: 100.0)
    _bars("COPY", lambda d: 100.0)
    _sue_order("EARN", filed=SESSIONS[3], session=SESSIONS[4])
    _copy_order("COPY", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()
    real = pit_store.record_fill_submission

    def boom(row, table="paper_fill"):
        if table == "copy_fill":
            raise RuntimeError("copy store refused the write")
        return real(row, table=table)

    monkeypatch.setattr(pit_store, "record_fill_submission", boom)

    out = _submit(broker, today=SESSIONS[4])

    assert any(s["symbol"] == "EARN" for s in broker.sent)
    assert out["status"] == "partial"
    assert out["copy"]["status"] == "failed"
    assert "copy store refused" in out["copy"]["error"]
    assert out["submitted"] >= 1


def test_an_unreadable_copy_setting_does_not_stop_earnings_legs(store, monkeypatch):
    """A typo in a copy setting fails the copy book, by name, and nothing else.
    Read before either book's own try, it stopped the earnings orders too."""
    _bars("EARN", lambda d: 100.0)
    _bars("COPY", lambda d: 100.0)
    _sue_order("EARN", filed=SESSIONS[3], session=SESSIONS[4])
    _copy_order("COPY", filed=SESSIONS[3], session=SESSIONS[4])
    monkeypatch.setenv("NEMO_COPY_HOLD_DAYS", "twenty")
    broker = FakeBroker()

    out = _submit(broker, today=SESSIONS[4])

    assert [s["symbol"] for s in broker.sent] == ["EARN"]
    assert out["status"] == "partial"
    assert out["copy"]["status"] == "failed"
    assert "NEMO_COPY_HOLD_DAYS" in out["copy"]["error"]


def test_submit_and_collect_report_copy_counts_apart_from_earnings(store):
    """The result names each book's counts so a quiet copy book cannot hide
    inside the earnings totals."""
    _bars("EARN", lambda d: 100.0)
    _bars("COPY", lambda d: 100.0)
    _sue_order("EARN", filed=SESSIONS[3], session=SESSIONS[4])
    _copy_order("COPY", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker()

    submitted = _submit(broker, today=SESSIONS[4])
    broker.fill(fills.client_order_id(SESSIONS[3], "EARN", "entry"),
                price=100.0, qty=50, at=f"{SESSIONS[4]}T13:30:01Z")
    broker.fill(_copy_cid(SESSIONS[3], "COPY", "entry"),
                price=100.0, qty=50, at=f"{SESSIONS[4]}T13:30:01Z")
    collected = _collect(broker, today=SESSIONS[4])

    assert submitted["submitted"] == 1
    assert submitted["copy"]["submitted"] == 1
    assert collected["updated"] == 1
    assert collected["copy"]["updated"] == 1
    with pit_store.connect() as conn:
        note = conn.execute(
            "SELECT error FROM run_log WHERE job='fill_submit' "
            "ORDER BY run_id DESC LIMIT 1").fetchone()[0]
    assert note is not None and "copy" in note.lower()


def test_copy_exits_use_nemo_copy_hold_days_not_the_earnings_horizon(
        store, monkeypatch):
    """The copy book's hold is NEMO_COPY_HOLD_DAYS. Passing the earnings
    horizon into submit must not move the copy exit."""
    monkeypatch.setenv("NEMO_COPY_HOLD_DAYS", "3")
    _bars("AAA", lambda d: 100.0)
    _copy_order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker(positions={"AAA": {"qty": 50, "side": "long"}})
    _submit(broker, today=SESSIONS[4], horizon=5)
    broker.fill(_copy_cid(SESSIONS[3], "AAA", "entry"), price=100.0, qty=50,
                at=f"{SESSIONS[4]}T13:30:01Z")
    _collect(broker, today=SESSIONS[4])

    _submit(broker, today=SESSIONS[6], horizon=5)  # only 2 sessions later
    assert not any(s["client_order_id"].endswith("-exit") for s in broker.sent)

    _submit(broker, today=SESSIONS[7], horizon=5)  # 3 sessions later
    assert broker.sent[-1]["client_order_id"] == _copy_cid(
        SESSIONS[3], "AAA", "exit")


def test_report_and_status_show_the_copy_book_separately(store):
    """Open and closed copy positions, and the mean return from fill prices,
    sit beside GATE and RISK — never inside them."""
    _bars("AAA", lambda d: 100.0)
    _copy_order("AAA", filed=SESSIONS[3], session=SESSIONS[4])
    broker = FakeBroker(positions={"AAA": {"qty": 50, "side": "long"}})
    _submit(broker, today=SESSIONS[4])
    broker.fill(_copy_cid(SESSIONS[3], "AAA", "entry"), price=100.0, qty=50,
                at=f"{SESSIONS[4]}T13:30:01Z")
    _collect(broker, today=SESSIONS[4])

    open_report = fills.report(SESSIONS[4])
    assert open_report["copy"]["open"] == 1
    assert open_report["copy"]["closed"] == 0
    assert open_report["round_trips"] == 0

    _submit(broker, today=SESSIONS[9])
    broker.fill(_copy_cid(SESSIONS[3], "AAA", "exit"), price=110.0, qty=50,
                at=f"{SESSIONS[9]}T13:30:01Z")
    _collect(broker, today=SESSIONS[9])

    closed_report = fills.report(SESSIONS[9])
    assert closed_report["copy"]["open"] == 0
    assert closed_report["copy"]["closed"] == 1
    assert closed_report["copy"]["mean_return"] == pytest.approx(0.10)
    assert closed_report["round_trips"] == 0

    screen = status.collect(as_of=SESSIONS[9])
    assert screen["copy"]["open"] == 0
    assert screen["copy"]["closed"] == 1
    rendered = status._render(screen)
    assert "COPY" in rendered
    assert "GATE" in rendered and "RISK" in rendered


def test_research_fills_compose_passes_copy_trading_settings():
    compose = yaml.safe_load(
        (ROOT / "deploy" / "docker-compose.yml").read_text())
    env = compose["services"]["research-fills"]["environment"]

    assert "NEMO_COPY_TRADING" in env
    assert "NEMO_COPY_HOLD_DAYS" in env
