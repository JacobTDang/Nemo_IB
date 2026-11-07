"""What each scheduled job says in Discord, and when it says nothing.

The messages are built from result dictionaries. The paper-fill results here
come from the real ``submit()`` and ``collect()`` against an in-memory broker,
so a change to their shape fails a test instead of a Saturday morning.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

import research.discord_post as post
from research import pit_store
from tools.alpaca import fills
from tools.alpaca.async_broker import AsyncBrokerError

WHEN = 1_700_000_000


def texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return found


def body(payload):
    return "\n".join(texts(payload))


def rows(payload):
    return [node for box in payload["components"] for node in box["components"]
            if node["type"] == post.ACTION_ROW]


def buttons(payload):
    return [button for row in rows(payload) for button in row["components"]]


def channels(messages):
    return [channel for channel, _ in messages]


def message_to(messages, channel):
    return next(payload for name, payload in messages if name == channel)


def silent(payload):
    return bool(payload["flags"] & post.SUPPRESS_NOTIFICATIONS)


# ------------------------------------------------------- quiet runs say nothing

QUIET_RUNS = {
    "research-watch": {"status": "partial", "new_events": 0, "events": [],
                       "error": "watched 150 of 2000 names, from offset 0; "
                                "stopped by its time budget"},
    "research-copy-off": {"status": "off", "rows_written": 0},
    "research-copy-empty": {"status": "ok", "rows_written": 0, "accepted": 0,
                            "preview": []},
    "research-seed": {"written": 0, "failed": [], "tickers": 20},
    "research-scan": {"written": 0, "candidates": [], "rejected": [{"t": 1}]},
    "research-announce": {"written": 0, "failed": [], "tickers": 20},
    "research-score": {"sample": 0, "hit_rate": None,
                       "coefficient_usable": False,
                       "drift_bps_per_sue": None},
    "research-fills": {"status": "ok", "action": "submit", "submitted": 0,
                       "skipped": 0, "rejected": 0, "missed": 0,
                       "copy": {"status": "ok", "submitted": 0, "skipped": 0,
                                "rejected": 0, "missed": 0}},
    "congress-sync": {"parsed": 0, "failures": 0, "error": ""},
}


@pytest.mark.parametrize("name", sorted(QUIET_RUNS))
def test_a_run_that_found_nothing_and_did_not_fail_posts_nothing(name):
    job = name.rsplit("-off", 1)[0].rsplit("-empty", 1)[0]
    assert post.messages_for(job, 0, QUIET_RUNS[name], when=WHEN) == []


def test_a_quiet_fills_collect_posts_nothing():
    result = {"status": "ok", "action": "collect", "asked": 0, "updated": 0,
              "copy": {"status": "ok", "asked": 0, "updated": 0}}
    assert post.messages_for("research-fills", 0, result, when=WHEN) == []


def test_a_clean_status_screen_is_the_daily_heartbeat_and_does_not_notify():
    messages = post.messages_for(
        "research-status", 0, {"attention": []}, when=WHEN)
    assert channels(messages) == ["ops"]
    payload = messages[0][1]
    assert silent(payload)
    text = body(payload)
    assert "All clear" in text and "```" not in text
    assert payload["components"][0]["accent_color"] == post.GREEN


def test_a_status_screen_with_attention_items_notifies():
    messages = post.messages_for(
        "research-status", 1, {"attention": ["score failed on 2026-11-06"]},
        when=WHEN)
    payload = messages[0][1]
    assert not silent(payload)
    assert payload["components"][0]["accent_color"] == post.YELLOW
    assert "score failed on 2026-11-06" in body(payload)


def test_an_unknown_job_raises_instead_of_posting_a_guess():
    with pytest.raises(post.DiscordError):
        post.messages_for("research-nothing", 0, {}, when=WHEN)


def test_a_job_cannot_post_to_a_channel_it_is_not_configured_for(monkeypatch):
    monkeypatch.setitem(post.JOB_CHANNELS, "research-seed", ("signals",))
    with pytest.raises(post.DiscordError) as caught:
        post.messages_for("research-seed", 0, {"written": 3}, when=WHEN)
    assert "research-seed" in str(caught.value)


# ----------------------------------------- the code block shows only a failure

def test_a_good_watch_pass_has_no_code_block_even_with_a_coverage_note():
    result = {"status": "partial", "new_events": 1,
              "error": "watched 150 of 2000 names, from offset 0",
              "events": [{"subject_ticker": "INTC", "form": "SC 13D",
                          "filer_name": "Example LP",
                          "url": "https://www.sec.gov/a.htm"}]}
    messages = post.messages_for("research-watch", 0, result, when=WHEN)
    assert messages
    for _, payload in messages:
        assert "```" not in body(payload)


def test_a_failed_watch_pass_shows_the_error_in_a_code_block():
    result = {"status": "failed", "new_events": 0, "events": [],
              "error": "no eligible universe recorded"}
    messages = post.messages_for("research-watch", 1, result, when=WHEN)
    assert channels(messages) == ["ops"]
    text = body(messages[0][1])
    assert "```" in text and "no eligible universe recorded" in text
    assert not silent(messages[0][1])


# ------------------------------------------------------------ research-watch

def _events(count):
    return [{"subject_ticker": f"T{i}", "form": "SC 13D",
             "filer_name": f"Filer {i}",
             "url": f"https://www.sec.gov/Archives/edgar/data/{i}/x.htm"}
            for i in range(count)]


def test_each_watch_event_gets_its_own_link_button():
    result = {"status": "ok", "new_events": 3, "events": _events(3)}
    messages = post.messages_for("research-watch", 0, result, when=WHEN)
    payload = message_to(messages, "signals")
    found = buttons(payload)
    assert [b["url"] for b in found] == [e["url"] for e in _events(3)]
    assert all(b["style"] == 5 for b in found)
    assert len({b["label"] for b in found}) == 3


def test_no_action_row_holds_more_than_five_buttons():
    result = {"status": "ok", "new_events": 8, "events": _events(8)}
    payload = message_to(
        post.messages_for("research-watch", 0, result, when=WHEN), "signals")
    assert [len(row["components"]) for row in rows(payload)] == [5, 3]


def test_a_truncated_watch_list_says_how_many_more_there_are():
    result = {"status": "ok", "new_events": 11, "events": _events(11)}
    payload = message_to(
        post.messages_for("research-watch", 0, result, when=WHEN), "signals")
    assert "and **3** more" in body(payload)
    assert len(buttons(payload)) == 8


def test_a_watch_event_without_a_url_raises_instead_of_being_skipped():
    events = _events(2)
    events[1]["url"] = None
    result = {"status": "ok", "new_events": 2, "events": events}
    with pytest.raises(post.DiscordError) as caught:
        post.messages_for("research-watch", 0, result, when=WHEN)
    assert "T1" in str(caught.value)


def test_an_invalid_link_raises_instead_of_being_skipped():
    for bad in ("", "ftp://example.com/a", "not a url", None):
        with pytest.raises(post.DiscordError):
            post.signal_message("Filing", ["line"], links=[("Open", bad)],
                                when=WHEN)


# ------------------------------------------------- research-watch change of state

FAILED_PASS = {"status": "failed", "new_events": 0, "events": [],
               "error": "EDGAR returned 503"}
GOOD_PASS = {"status": "partial", "new_events": 0, "events": [],
             "error": "watched 150 of 2000 names"}


def test_the_first_failed_pass_notifies_and_starts_the_failing_state():
    messages, state = post.watch_messages(1, FAILED_PASS, {}, when=WHEN)
    assert channels(messages) == ["ops"]
    assert not silent(messages[0][1])
    assert state["failing_since"] == WHEN
    assert state["failed_passes"] == 1


def test_later_failed_passes_in_a_row_post_nothing():
    state = {"failing_since": WHEN - 1200, "failed_passes": 1}
    messages, new_state = post.watch_messages(1, FAILED_PASS, state, when=WHEN)
    assert messages == []
    assert new_state["failing_since"] == WHEN - 1200
    assert new_state["failed_passes"] == 2


def test_the_first_good_pass_after_a_failure_posts_one_silent_recovery():
    state = {"failing_since": WHEN - 3600, "failed_passes": 3}
    messages, new_state = post.watch_messages(0, GOOD_PASS, state, when=WHEN)
    assert channels(messages) == ["ops"]
    payload = messages[0][1]
    assert silent(payload)
    text = body(payload)
    assert "recovered" in text and "**3**" in text
    assert f"<t:{WHEN - 3600}:R>" in text
    assert new_state == {}


def test_a_good_pass_with_no_failure_before_it_posts_nothing():
    messages, state = post.watch_messages(0, GOOD_PASS, {}, when=WHEN)
    assert messages == [] and state == {}


def test_a_recovery_pass_that_also_found_filings_posts_both():
    result = {"status": "ok", "new_events": 1, "events": _events(1)}
    state = {"failing_since": WHEN - 600, "failed_passes": 1}
    messages, _ = post.watch_messages(0, result, state, when=WHEN)
    assert sorted(channels(messages)) == ["ops", "signals"]
    assert "recovered" in body(message_to(messages, "ops"))


def test_the_state_is_written_only_after_the_post_succeeds(tmp_path):
    path = tmp_path / "watch.json"

    def refuse(url, body):
        raise RuntimeError("down")

    code = post.finish("research-watch", 1, FAILED_PASS, transport=refuse,
                       state_path=path)
    assert code == 1
    assert not path.exists()

    sent = []
    code = post.finish("research-watch", 1, FAILED_PASS,
                       transport=lambda url, payload: sent.append(payload),
                       state_path=path)
    assert code == 1 and len(sent) == 1 and path.exists()

    sent.clear()
    post.finish("research-watch", 1, FAILED_PASS,
                transport=lambda url, payload: sent.append(payload),
                state_path=path)
    assert sent == []

    post.finish("research-watch", 0, GOOD_PASS,
                transport=lambda url, payload: sent.append(payload),
                state_path=path)
    assert len(sent) == 1 and "recovered" in body(sent[0])
    assert post.load_watch_state(path) == {}


def test_an_unreadable_state_file_fails_loud(tmp_path):
    path = tmp_path / "watch.json"
    path.write_text("{not json")
    with pytest.raises(post.DiscordError) as caught:
        post.load_watch_state(path)
    assert str(path) in str(caught.value)


def test_the_state_file_sits_beside_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    assert post.watch_state_path().parent == tmp_path


# ----------------------------------------------------------- paper fills

SESSIONS = []
_day = date.fromisoformat("2026-06-01")
while len(SESSIONS) < 12:
    if _day.weekday() < 5:
        SESSIONS.append(_day.isoformat())
    _day += timedelta(days=1)


class FakeBroker:
    def __init__(self, reject=()):
        self.reject = set(reject)
        self.orders = {}

    async def get_calendar(self, start, end):
        return [d for d in SESSIONS if start <= d <= end]

    async def get_open_position(self, symbol):
        return None

    async def submit_market_order(self, symbol, qty, side, *,
                                  client_order_id, time_in_force="day"):
        if symbol in self.reject:
            raise AsyncBrokerError("HTTP 422: asset is not tradable")
        order_id = f"b-{client_order_id}"
        self.orders[order_id] = {"id": order_id, "status": "accepted",
                                 "filled_qty": "0", "filled_avg_price": None,
                                 "filled_at": None}
        return {"id": order_id, "status": "accepted"}

    async def get_order_by_id(self, order_id):
        return self.orders[order_id]

    def fill(self, client_order_id, price, qty, at):
        self.orders[f"b-{client_order_id}"].update({
            "status": "filled", "filled_qty": str(qty),
            "filled_avg_price": str(price), "filled_at": at})


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    monkeypatch.setenv("NEMO_COPY_TRADING", "on")
    monkeypatch.setenv("NEMO_COPY_HOLD_DAYS", "5")
    pit_store.init_schema()
    for ticker in ("EARN", "EARN2", "COPY"):
        for session in SESSIONS:
            pit_store.record_bars(ticker, [{
                "trade_date": session, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.0, "volume": 2_000_000}],
                recorded_at=f"{session}T21:00:00Z")
    return pit_store


def _earn_order(ticker):
    pit_store.record_paper_orders(
        SESSIONS[3],
        [{"ticker": ticker, "side": "long", "sue": 3.0, "variant": "ts",
          "strength": 3.0, "fiscal_period": "2026Q2",
          "expected_edge_bps": 45.0, "cost_bps": 10.0, "net_edge_bps": 35.0,
          "target_dollars": 5000.0, "intended_session": SESSIONS[4], "rank": 1}],
        recorded_at=f"{SESSIONS[3]}T21:00:00Z")


def _copy_order(ticker):
    pit_store.record_copy_orders([{
        "txn_id": f"txn-{ticker}", "as_of_date": SESSIONS[3],
        "decided_at": f"{SESSIONS[3]}T12:30:00+00:00",
        "learned_at": f"{SESSIONS[3]}T06:00:00+00:00",
        "filing_id": f"f-{ticker}", "member_id": "m1", "member": "Jane Smith",
        "chamber": "house", "owner": "self", "source_ticker": ticker,
        "ticker": ticker, "cik": "1", "amount_min": 15001,
        "amount_max": 50000, "transaction_date": SESSIONS[3],
        "filed_date": SESSIONS[3], "accepted": 1, "reason": None,
        "side": "long", "target_dollars": 5000.0,
        "intended_session": SESSIONS[4]}])


def _submit(broker):
    return asyncio.run(fills.submit(broker, today=SESSIONS[4], horizon_days=5))


def _fills_message(result, action, code):
    return post.messages_for(
        "research-fills", code, {**result, "action": action}, when=WHEN)


def test_fills_sums_the_earnings_book_and_the_copy_book(book):
    _earn_order("EARN")
    _copy_order("COPY")
    result = _submit(FakeBroker())
    assert result["submitted"] == 1 and result["copy"]["submitted"] == 1

    messages = _fills_message(result, "submit", 0)
    text = body(message_to(messages, "signals"))
    assert "**2** orders submitted" in text
    assert "**2** orders submitted" in body(message_to(messages, "ops"))


def test_a_copy_only_run_is_not_reported_as_nothing(book):
    _copy_order("COPY")
    result = _submit(FakeBroker())
    assert result["submitted"] == 0 and result["copy"]["submitted"] == 1
    messages = _fills_message(result, "submit", 0)
    assert "**1** orders submitted" in body(message_to(messages, "signals"))


def test_a_rejection_is_a_yellow_warning_with_the_exit_code_kept(book):
    _earn_order("EARN")
    _earn_order("EARN2")
    _copy_order("COPY")
    result = _submit(FakeBroker(reject={"EARN2"}))
    assert result["status"] == "partial"
    assert "1 rejected" in result["notes"]

    messages = _fills_message(result, "submit", 1)
    for channel in ("ops", "signals"):
        payload = message_to(messages, channel)
        assert payload["components"][0]["accent_color"] == post.YELLOW
        assert "❌" not in body(payload)
        assert not silent(payload)
    text = body(message_to(messages, "ops"))
    assert "**2** orders submitted" in text
    assert "⚠️ **1** rejected" in text
    assert "1 rejected" in text.split("```", 1)[1]


def test_a_failed_copy_book_shows_its_error_text(book, monkeypatch):
    _earn_order("EARN")
    _copy_order("COPY")
    real = pit_store.record_fill_submission

    def boom(row, table="paper_fill"):
        if table == "copy_fill":
            raise RuntimeError("copy store refused the write")
        return real(row, table=table)

    monkeypatch.setattr(pit_store, "record_fill_submission", boom)
    result = _submit(FakeBroker())
    assert result["copy"]["status"] == "failed"

    messages = _fills_message(result, "submit", 1)
    text = body(message_to(messages, "ops"))
    assert "copy store refused the write" in text
    assert message_to(messages, "ops")["components"][0]["accent_color"] == post.YELLOW


def test_collect_counts_orders_updated_from_both_books(book):
    _earn_order("EARN")
    _copy_order("COPY")
    broker = FakeBroker()
    _submit(broker)
    broker.fill(fills.client_order_id(SESSIONS[3], "EARN", "entry"),
                price=100.0, qty=50, at=f"{SESSIONS[4]}T13:30:01Z")
    broker.fill(fills.client_order_id(SESSIONS[3], "COPY", "entry",
                                      book="copy"),
                price=100.0, qty=50, at=f"{SESSIONS[4]}T13:30:01Z")
    result = asyncio.run(fills.collect(broker, today=SESSIONS[4]))
    assert result["updated"] == 1 and result["copy"]["updated"] == 1

    messages = _fills_message(result, "collect", 0)
    text = body(message_to(messages, "signals"))
    assert "**2** orders updated" in text
    assert "collect" not in text


def test_a_failed_fills_run_is_red_and_carries_the_error():
    messages = post.messages_for(
        "research-fills", 1,
        {"action": "submit", "status": "failed", "error": "HTTP 401"},
        when=WHEN)
    payload = message_to(messages, "ops")
    assert payload["components"][0]["accent_color"] == post.RED
    assert "HTTP 401" in body(payload)


# -------------------------------------------------------------- research-scan

def _candidates(*tickers):
    return [{"ticker": t, "side": "long", "sue": 1.2345} for t in tickers]


def test_scan_counts_filed_orders_and_not_rejected_rows():
    result = {"written": 9, "candidates": _candidates("AAA", "BBB"),
              "rejected": [{"ticker": f"R{i}"} for i in range(7)]}
    messages = post.messages_for("research-scan", 0, result, when=WHEN)
    assert "**2** orders filed" in body(message_to(messages, "ops"))
    signals = body(message_to(messages, "signals"))
    assert "AAA" in signals and "BBB" in signals
    assert "SUE **1.23**" in signals


def test_a_superseded_book_is_not_reported_as_filed():
    result = {
        "written": 2, "candidates": _candidates("AAA", "BBB"), "rejected": [],
        "superseded": {"filed": ["AAA"], "proposed": ["AAA", "BBB"],
                       "dropped": ["BBB"],
                       "note": "2026-11-06 already holds a row for BBB, so "
                               "nothing was filed for it"}}
    messages = post.messages_for("research-scan", 0, result, when=WHEN)
    ops = message_to(messages, "ops")
    assert "**1** orders filed" in body(ops)
    assert "⚠️" in body(ops) and "already holds a row for BBB" in body(ops)
    assert ops["components"][0]["accent_color"] == post.YELLOW
    assert not silent(ops)
    signals = body(message_to(messages, "signals"))
    assert "AAA" in signals and "BBB" not in signals.split("⚠️")[0]


def test_a_scan_where_every_candidate_was_dropped_files_zero_and_warns():
    result = {
        "written": 1, "candidates": _candidates("AAA"), "rejected": [],
        "superseded": {"filed": [], "proposed": ["AAA"], "dropped": ["AAA"],
                       "note": "2026-11-06 already holds a row for AAA"}}
    messages = post.messages_for("research-scan", 0, result, when=WHEN)
    assert channels(messages) == ["ops"]
    text = body(messages[0][1])
    assert "**0** orders filed" in text and "⚠️" in text


def test_a_scan_with_no_candidates_and_no_supersession_posts_nothing():
    result = {"written": 5, "candidates": [], "rejected": [{"t": 1}] * 5}
    assert post.messages_for("research-scan", 0, result, when=WHEN) == []


# ------------------------------------------------------------ research-score

def test_score_prints_the_unit_it_used_per_sue():
    result = {"sample": 40, "hit_rate": 0.55, "coefficient_usable": True,
              "drift_bps_per_sue": 12.0, "drift_bps_per_tail": None}
    text = body(message_to(
        post.messages_for("research-score", 0, result, when=WHEN), "signals"))
    assert "**12.0 bp** per SUE" in text


def test_score_prints_the_unit_it_used_per_tail():
    result = {"sample": 40, "hit_rate": 0.55, "coefficient_usable": True,
              "drift_bps_per_sue": None, "drift_bps_per_tail": 9.0}
    text = body(message_to(
        post.messages_for("research-score", 0, result, when=WHEN), "signals"))
    assert "**9.0 bp** per tail" in text
    assert "per SUE" not in text


# ---------------------------------------------- research-seed and research-announce

def test_seed_with_some_failures_lists_them_and_warns():
    result = {"written": 40, "tickers": 3,
              "failed": ["AAA: HTTPError: 503", "BBB: KeyError: eps"]}
    messages = post.messages_for("research-seed", 0, result, when=WHEN)
    payload = message_to(messages, "ops")
    text = body(payload)
    assert "⚠️" in text and "AAA: HTTPError: 503" in text and "BBB" in text
    assert payload["components"][0]["accent_color"] == post.YELLOW
    assert not silent(payload)


def test_seed_that_wrote_everything_is_a_silent_green_line():
    messages = post.messages_for(
        "research-seed", 0, {"written": 40, "tickers": 3, "failed": []},
        when=WHEN)
    payload = message_to(messages, "ops")
    assert silent(payload)
    assert payload["components"][0]["accent_color"] == post.GREEN


def test_a_seed_that_failed_everywhere_shows_the_failures_as_the_error():
    messages = post.messages_for(
        "research-seed", 1,
        {"written": 0, "tickers": 2, "failed": ["AAA: HTTPError: 503"]},
        when=WHEN)
    payload = message_to(messages, "ops")
    assert payload["components"][0]["accent_color"] == post.RED
    assert "AAA: HTTPError: 503" in body(payload)


def test_announce_with_some_failures_lists_them_and_still_posts_the_dates():
    result = {"written": 3, "tickers": 5, "covered": 4,
              "failed": ["AAA: ReadTimeout: slow"], "outcomes": {}}
    messages = post.messages_for("research-announce", 0, result, when=WHEN)
    ops = message_to(messages, "ops")
    assert "⚠️" in body(ops) and "AAA: ReadTimeout: slow" in body(ops)
    assert ops["components"][0]["accent_color"] == post.YELLOW
    assert "dates recorded" in body(message_to(messages, "signals"))


def test_a_long_failure_list_is_cut_with_a_count():
    failed = [f"T{i}: Boom: x" for i in range(30)]
    messages = post.messages_for(
        "research-seed", 0, {"written": 1, "tickers": 31, "failed": failed},
        when=WHEN)
    text = body(message_to(messages, "ops"))
    assert "T0: Boom: x" in text
    assert "and 20 more" in text


# ------------------------------------------------------------- congress-sync

def test_congress_sync_reports_what_this_run_parsed():
    messages = post.messages_for(
        "congress-sync", 0, {"parsed": 12, "failures": 0, "error": ""},
        when=WHEN)
    assert "**12** filings parsed" in body(message_to(messages, "ops"))


# ------------------------------------------------------------------ the code block

def test_the_error_keeps_the_end_of_the_log_and_cuts_at_a_line():
    log = "\n".join(f"line {i:04d} " + "x" * 50 for i in range(200))
    log += "\nTraceback: the last line explains it"
    fenced = post._fit_error(log, 600)
    assert len(fenced) <= 600
    assert "the last line explains it" in fenced
    assert "line 0000" not in fenced
    inner = fenced.split("```\n", 1)[1].rsplit("\n```", 1)[0].splitlines()
    assert all(line.startswith("line ") or line.startswith("Traceback")
               or "cut" in line for line in inner)


def test_a_single_overlong_last_line_is_cut_from_its_start():
    fenced = post._fit_error("short\n" + "y" * 5000 + "END", 300)
    assert len(fenced) <= 300
    assert fenced.rstrip("`\n").endswith("END")


def test_three_backticks_in_an_error_cannot_close_the_code_block():
    fenced = post._fit_error("before ``` after ```` end", 500)
    inner = fenced[len("```\n"):-len("\n```")]
    assert "```" not in inner
    assert "before" in inner and "end" in inner


def test_a_brief_page_never_splits_a_fenced_block():
    code = "```python\n" + "\n".join(f"x = {i}" for i in range(150)) + "\n```"
    sections = [f"## Section {i}\n" + ("Words about the tape. " * 40)
                for i in range(6)]
    brief = "\n\n".join(sections[:3] + [code] + sections[3:])
    pages = post.brief_messages(brief, when=WHEN)
    assert len(pages) > 1
    for payload in pages:
        detail = "\n".join(texts(payload)[1:-1])
        assert detail.count("```") % 2 == 0, detail[:200]
        assert sum(len(item) for item in texts(payload)) <= post.TEXT_LIMIT


def test_a_fenced_block_longer_than_a_page_is_closed_and_reopened():
    code = "```python\n" + "\n".join(f"x = {i}" for i in range(900)) + "\n```"
    pages = post.brief_messages("## Code\n" + code, when=WHEN)
    assert len(pages) > 1
    for payload in pages:
        detail = "\n".join(texts(payload)[1:-1])
        assert detail.count("```") % 2 == 0
    assert "x = 899" in body(pages[-1])


def test_a_repeated_watch_failure_posts_nothing_and_still_marks_itself_reported(
        tmp_path, capsys):
    path = tmp_path / "watch.json"
    post.save_watch_state(path, {"failing_since": WHEN - 600, "failed_passes": 1})
    sent = []
    code = post.finish("research-watch", 1, FAILED_PASS,
                       transport=lambda url, payload: sent.append(payload),
                       state_path=path)
    assert code == 1 and sent == []
    assert post.REPORTED in capsys.readouterr().err
    assert post.load_watch_state(path)["failed_passes"] == 2


def test_a_run_that_exits_zero_does_not_print_the_reported_marker(capsys):
    sent = []
    code = post.finish("research-seed", 0, {"written": 3, "failed": []},
                       transport=lambda url, payload: sent.append(payload))
    assert code == 0 and len(sent) == 1
    assert post.REPORTED not in capsys.readouterr().err


def test_the_prune_message_never_blames_the_prune_for_a_post_failure():
    messages = post.messages_for("docker-prune", 0, {"ok": True}, when=WHEN)
    assert "container prune failed" not in body(messages[0][1])
    failed = post.messages_for("docker-prune", 1, {"error": "x"}, when=WHEN)
    assert "container prune failed" not in body(failed[0][1])
