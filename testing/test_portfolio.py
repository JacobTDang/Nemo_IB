"""The paper portfolio card: the daily post to #signals and the /portfolio reply.

The account and the positions are the broker's. Each position is labelled
with the book whose fills hold it, and each book's closed trades come from
its own fill table, so the two books are shown side by side and never summed
into one return.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

import research.discord_post as post
from research import pit_store

WHEN = 1_794_000_000
SIGNALS = "https://hooks.example/signals"
OPS = "https://hooks.example/ops"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return tmp_path


def _leg(table, order_as_of, ticker, leg, side, qty, price, attempt=1):
    cid = f"{table}-{order_as_of}-{ticker}-{leg}-{attempt}"
    pit_store.record_fill_submission({
        "order_as_of": order_as_of, "ticker": ticker, "leg": leg,
        "attempt": attempt, "scheduled_session": order_as_of,
        "session": order_as_of, "side": side, "qty": qty,
        "client_order_id": cid, "status": "submitted",
        "submitted_at": f"{order_as_of}T13:15:00Z"}, table=table)
    pit_store.update_paper_fill(cid, "filled", qty, price,
                                f"{order_as_of}T13:30:01Z", table=table)


def _texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return "\n".join(found)


# ------------------------------------------------------------- the store

def test_a_closed_short_returns_its_entry_over_its_exit(store):
    _leg("paper_fill", "2026-11-02", "AAA", "entry", "sell", 10, 100.0)
    _leg("paper_fill", "2026-11-02", "AAA", "exit", "buy", 10, 90.0)

    book = pit_store.book_summary("2026-11-10", "paper_fill")

    assert book["closed"] == 1
    assert book["mean_return"] == pytest.approx(0.10)


def test_an_entry_with_no_exit_is_open_and_named(store):
    _leg("copy_fill", "2026-11-02", "BBB", "entry", "buy", 5, 50.0)

    book = pit_store.book_summary("2026-11-10", "copy_fill")

    assert book["open"] == 1
    assert book["open_tickers"] == ["BBB"]
    assert book["mean_return"] is None


# ------------------------------------------------------------- the broker

async def _with_broker(handler, call, monkeypatch):
    from tools.alpaca.async_broker import AsyncBroker

    monkeypatch.setenv("ALPACA_PAPER_KEY", "fake_key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET", "fake_secret")
    broker = AsyncBroker(paper=True)
    broker._client = httpx.AsyncClient(
        base_url=broker.base_url, transport=httpx.MockTransport(handler))
    try:
        return await call(broker)
    finally:
        await broker._client.aclose()


def test_the_account_carries_the_last_close_equity(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "equity": "100523.40", "last_equity": "100111.30",
            "cash": "61230.00", "buying_power": "200000",
            "portfolio_value": "100523.40", "status": "ACTIVE"})

    account = asyncio.run(_with_broker(
        handler, lambda b: b.get_account(), monkeypatch))

    assert account["last_equity"] == pytest.approx(100111.30)


def test_a_position_carries_its_unrealized_gain(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=[{
            "symbol": "nvda", "qty": "100", "side": "long",
            "market_value": "18402", "avg_entry_price": "176.74",
            "unrealized_pl": "728.00", "unrealized_plpc": "0.0412"}])

    positions = asyncio.run(_with_broker(
        handler, lambda b: b.get_all_positions(), monkeypatch))

    assert positions[0]["unrealized_pl"] == pytest.approx(728.0)
    assert positions[0]["unrealized_plpc"] == pytest.approx(0.0412)


# ------------------------------------------------------------- the snapshot

class FakeBroker:
    def __init__(self, positions, equity=100523.40, last_equity=100111.30):
        self.positions = positions
        self.account = {"paper": True, "equity": equity,
                        "last_equity": last_equity, "cash": 61230.0,
                        "buying_power": 0.0, "portfolio_value": equity,
                        "status": "ACTIVE"}

    async def get_account(self):
        return self.account

    async def get_all_positions(self):
        return self.positions


def _position(symbol, market_value, plpc, side="long"):
    return {"symbol": symbol, "qty": 10.0, "side": side,
            "market_value": market_value, "avg_entry_price": 1.0,
            "unrealized_pl": market_value * plpc, "unrealized_plpc": plpc}


def test_each_position_names_the_books_that_hold_it(store):
    from tools.alpaca import portfolio

    _leg("paper_fill", "2026-11-02", "EARN", "entry", "buy", 10, 10.0)
    _leg("copy_fill", "2026-11-02", "COPY", "entry", "buy", 10, 10.0)
    _leg("paper_fill", "2026-11-03", "BOTH", "entry", "buy", 10, 10.0)
    _leg("copy_fill", "2026-11-03", "BOTH", "entry", "buy", 10, 10.0)
    broker = FakeBroker([_position(t, 100.0, 0.01)
                         for t in ("EARN", "COPY", "BOTH", "HAND")])

    snap = asyncio.run(portfolio.snapshot(broker, as_of="2026-11-10"))

    books = {p["symbol"]: p["books"] for p in snap["positions"]}
    assert books == {"EARN": ["Earnings"], "COPY": ["Copy"],
                     "BOTH": ["Earnings", "Copy"], "HAND": []}


def test_positions_come_largest_first_whatever_their_side(store):
    from tools.alpaca import portfolio

    broker = FakeBroker([_position("SMALL", 100.0, 0.01),
                         _position("SHORT", -900.0, 0.02, side="short"),
                         _position("MID", 500.0, 0.01)])

    snap = asyncio.run(portfolio.snapshot(broker, as_of="2026-11-10"))

    assert [p["symbol"] for p in snap["positions"]] == ["SHORT", "MID", "SMALL"]


def test_the_snapshot_reports_each_books_closed_trades(store):
    from tools.alpaca import portfolio

    _leg("paper_fill", "2026-11-02", "AAA", "entry", "buy", 10, 100.0)
    _leg("paper_fill", "2026-11-02", "AAA", "exit", "sell", 10, 105.0)

    snap = asyncio.run(portfolio.snapshot(FakeBroker([]), as_of="2026-11-10"))

    assert snap["books"]["Earnings"]["closed"] == 1
    assert snap["books"]["Earnings"]["mean_return"] == pytest.approx(0.05)
    assert snap["books"]["Copy"]["closed"] == 0
    assert snap["equity"] == pytest.approx(100523.40)
    assert snap["last_equity"] == pytest.approx(100111.30)


# ------------------------------------------------------------- the card

def _result(positions=None, equity=100523.40, last_equity=100111.30,
            books=None):
    return {
        "status": "ok", "as_of": "2026-11-10", "equity": equity,
        "last_equity": last_equity, "cash": 61230.0,
        "positions": positions if positions is not None else [
            {**_position("NVDA", 18402.0, 0.0412), "books": ["Earnings"]},
            {**_position("AAPL", 12050.0, -0.0085), "books": ["Copy"]},
            {**_position("TSLA", -8840.0, 0.013, side="short"),
             "books": ["Earnings"]},
        ],
        "books": books or {
            "Earnings": {"open": 2, "closed": 14, "mean_return": 0.0062},
            "Copy": {"open": 1, "closed": 2, "mean_return": 0.031}},
    }


def test_the_card_leads_with_the_account_value_and_the_day():
    text = _texts(post.portfolio_message(_result(), when=WHEN))

    assert "## 📋 Paper portfolio · $100,523.40" in text
    assert "**Today** **+$412.10** (**+0.41%**)" in text
    assert "**Cash** **$61,230.00**" in text


def test_a_loss_day_is_signed():
    text = _texts(post.portfolio_message(
        _result(equity=99000.0, last_equity=100000.0), when=WHEN))

    assert "**Today** **-$1,000.00** (**-1.00%**)" in text


def test_an_account_with_no_prior_close_shows_no_day_percent():
    text = _texts(post.portfolio_message(
        _result(last_equity=0.0), when=WHEN))

    assert "**Today**" not in text


def test_the_card_lists_each_position_with_its_books_and_gain():
    text = _texts(post.portfolio_message(_result(), when=WHEN))

    assert "**3** open positions" in text
    assert "**NVDA** · Earnings · $18,402.00 · **+4.12%**" in text
    assert "**AAPL** · Copy · $12,050.00 · **-0.85%**" in text
    assert "**TSLA** short · Earnings · -$8,840.00 · **+1.30%**" in text


def test_a_position_no_book_holds_says_so():
    positions = [{**_position("HAND", 100.0, 0.01), "books": []},
                 {**_position("BOTH", 50.0, 0.02),
                  "books": ["Earnings", "Copy"]}]
    text = _texts(post.portfolio_message(_result(positions), when=WHEN))

    assert "**HAND** · no book · $100.00" in text
    assert "**BOTH** · Earnings · Copy · $50.00" in text


def test_more_positions_than_fit_end_in_a_count():
    positions = [{**_position(f"T{i}", 100.0 - i, 0.01), "books": []}
                 for i in range(post.WATCH_EVENTS_SHOWN + 2)]
    text = _texts(post.portfolio_message(_result(positions), when=WHEN))

    assert "and **2** more" in text
    assert f"**T{post.WATCH_EVENTS_SHOWN}**" not in text


def test_the_card_shows_each_books_closed_trades():
    text = _texts(post.portfolio_message(_result(), when=WHEN))

    assert ("Closed: Earnings **14** trades, mean **+0.62%** · "
            "Copy **2** trades, mean **+3.10%**") in text


def test_one_position_and_one_trade_are_singular():
    books = {"Earnings": {"open": 1, "closed": 1, "mean_return": 0.01},
             "Copy": {"open": 0, "closed": 0, "mean_return": None}}
    positions = [{**_position("NVDA", 100.0, 0.01), "books": ["Earnings"]}]
    text = _texts(post.portfolio_message(_result(positions, books=books),
                                         when=WHEN))

    assert "**1** open position\n" in text
    assert "Earnings **1** trade, mean **+1.00%** · Copy **0** trades" in text


def test_the_daily_card_goes_to_signals_without_a_ping():
    messages = post.messages_for("research-portfolio", 0, _result(), when=WHEN)

    assert [channel for channel, _ in messages] == ["signals"]
    payload = messages[0][1]
    assert payload["flags"] & post.SUPPRESS_NOTIFICATIONS
    assert "nemo research-portfolio" in _texts(payload)


def test_a_failed_read_posts_its_error_to_ops():
    messages = post.messages_for(
        "research-portfolio", 1,
        {"status": "failed", "error": "AsyncBrokerError: 401 unauthorized"},
        when=WHEN)

    assert [channel for channel, _ in messages] == ["ops"]
    text = _texts(messages[0][1])
    assert "❌" in text
    assert "401 unauthorized" in text


# ------------------------------------------------------------- the job

class Recorder:
    def __init__(self):
        self.posts = []

    def __call__(self, url, body):
        assert url.startswith((OPS, SIGNALS))
        self.posts.append(("ops" if url.startswith(OPS) else "signals", body))


def _broker_class(broker=None, error=None):
    class Broker:
        def __init__(self, **kwargs):
            assert kwargs == {"paper": True}

        async def __aenter__(self):
            if error:
                raise error
            return broker

        async def __aexit__(self, *exc):
            return False

    return Broker


def test_the_job_posts_the_card_and_exits_0(store, monkeypatch):
    from tools.alpaca import portfolio

    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    monkeypatch.setattr(portfolio, "AsyncBroker", _broker_class(
        FakeBroker([_position("NVDA", 100.0, 0.01)])))

    assert portfolio.main() == 0
    assert [channel for channel, _ in recorder.posts] == ["signals"]
    assert "**NVDA** · no book" in _texts(recorder.posts[0][1])


def test_a_broker_failure_posts_to_ops_and_exits_1(store, monkeypatch, capsys):
    from tools.alpaca import portfolio

    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    monkeypatch.setattr(portfolio, "AsyncBroker", _broker_class(
        error=RuntimeError("Missing Alpaca paper credentials")))

    assert portfolio.main() == 1
    assert [channel for channel, _ in recorder.posts] == ["ops"]
    assert "Missing Alpaca paper credentials" in _texts(recorder.posts[0][1])
    assert post.REPORTED in capsys.readouterr().err
