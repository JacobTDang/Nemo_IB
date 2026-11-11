"""The cards the Discord bot answers /status, /orders and /ticker with.

Built from results shaped like `status.collect`, `lookup.orders` and
`lookup.ticker`, as a reader sees them on a phone. The reader is the owner,
not an engineer: every line is in plain words, and Nemo's own terms (SUE,
gate, drawdown, issuer, the job names) never reach the card.
"""
from __future__ import annotations

import re

import pytest

import research.discord_post as post

WHEN = 1_794_000_000
JARGON = re.compile(r"\b(SUE|gate|drawdown|issuers?|bmo|amc|book|partial|"
                    r"daily_bars|fill_submit|congress_copy|2026Q\d)\b")


def _texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return "\n".join(found)


def _color(payload):
    return payload["components"][0]["accent_color"]


def _buttons(payload):
    return [node for node in post._walk(payload["components"])
            if node.get("type") == post.BUTTON]


def _plain(text):
    """No internal term outside the footer, which names the command."""
    body = "\n".join(line for line in text.splitlines()
                     if not line.startswith("-# nemo /"))
    return JARGON.findall(body)


# ------------------------------------------------------------- dates

@pytest.mark.parametrize("iso, shown", [
    ("2026-11-10", "Nov 10"), ("2026-01-02", "Jan 2"),
    ("2025-12-16", "Dec 16, 2025")])
def test_a_date_is_a_short_month_and_day_and_a_year_only_when_not_this_one(
        iso, shown):
    assert post.day(iso, this_year=2026) == shown


# ------------------------------------------------------------- /status

def _job(name, state="ok", **extra):
    return {"job": name, "state": state, "partial_is_normal": False,
            "last_run": "2026-11-10", "every": "weeknight", "error": None,
            **extra}


def _report(jobs=None, attention=(), **extra):
    report = {
        "as_of": "2026-11-11",
        "jobs": jobs if jobs is not None else [_job("scan"), _job("universe")],
        "book": {"as_of": "2026-11-10", "accepted": 8, "issuers": 8,
                 "shorts": 0, "borrow_rates": 0},
        "risk": {"halted": False, "tripped": None,
                 "latest": {"equity_dollars": 120.0, "limit_dollars": 10000.0,
                            "closed_trades": 3, "open_trades": 18,
                            "unmarked_trades": 0},
                 "drawdown_dollars": 766.07},
        "gate": {"measured_trades": 0, "min_trades": 200, "passed": 0,
                 "reason": "0 of 200 trades"},
        "copy": {"open": 0, "closed": 0, "mean_return": None},
        "attention": list(attention)}
    report.update(extra)
    return report


def test_a_clean_status_says_everything_ran_in_plain_words():
    payload = post.status_card(_report(), when=WHEN)

    text = _texts(payload)
    assert "## ✅ All good" in text
    assert "✅ All **2** jobs ran on time" in text
    assert _color(payload) == post.GREEN
    assert _plain(text) == []


def test_the_status_says_what_nemo_picked_and_how_safe_and_proven_it_is():
    text = _texts(post.status_card(_report(), when=WHEN))

    assert "**Latest picks** **8** stocks to buy, chosen Nov 10" in text
    assert ("**Safety stop** not triggered · down **$766** from its best; "
            "it pauses new trades at **$10,000** down" in text)
    assert ("**Track record** **0** of the **200** finished trades needed "
            "to judge the strategy" in text)
    assert "**Congress copying** **0** open trades · **0** closed" in text


def test_shorts_are_said_as_bets_against():
    report = _report(book={"as_of": "2026-11-10", "accepted": 8,
                           "issuers": 8, "shorts": 2})

    text = _texts(post.status_card(report, when=WHEN))

    assert ("**Latest picks** **6** stocks to buy and **2** to bet against, "
            "chosen Nov 10" in text)


def test_a_failed_job_turns_the_card_yellow_and_says_what_failed():
    jobs = [_job("scan", "failed", error="HTTPError: 503 Service Unavailable"),
            _job("universe")]
    payload = post.status_card(_report(
        jobs, ["scan failed on 2026-11-10: HTTPError: 503"]), when=WHEN)

    text = _texts(payload)
    assert "## ⚠️ **1** thing needs a look" in text
    assert ("❌ Stock picking failed on Nov 10: HTTPError: 503 Service "
            "Unavailable" in text)
    assert "✅ **1** of **2** jobs ran on time" in text
    assert text.count("Stock picking failed") == 1
    assert _color(payload) == post.YELLOW
    assert _plain(text) == []


def test_a_strategy_at_its_best_says_so_rather_than_down_nothing():
    report = _report()
    report["risk"]["drawdown_dollars"] = 0.0

    text = _texts(post.status_card(report, when=WHEN))

    assert ("**Safety stop** not triggered · at its best so far; it pauses "
            "new trades at **$10,000** down" in text)


@pytest.mark.parametrize("state, said", [
    ("stale", "❌ Price update hasn't run since Nov 10"),
    ("crashed", "❌ Price update stopped partway through on Nov 10"),
    ("never", "❌ Price update has never run"),
])
def test_each_kind_of_trouble_is_said_plainly(state, said):
    jobs = [_job("daily_bars", state)]

    text = _texts(post.status_card(_report(jobs, ["daily_bars x"]), when=WHEN))

    assert said in text


def test_a_job_worth_a_glance_is_named_plainly_but_not_raised():
    jobs = [_job("scan"), _job("announce", "running"),
            _job("seed", "never"), _job("congress_copy", "off"),
            _job("activist_watch", "partial", partial_is_normal=True),
            _job("daily_bars", "partial")]

    text = _texts(post.status_card(_report(jobs), when=WHEN))

    assert "✅ **2** of **6** jobs ran on time" in text
    assert ("-# Earnings-date check is still running · Monthly estimate "
            "history hasn't run yet · Congress copying is turned off · Price "
            "update ran with gaps" in text)
    assert "## ✅ All good" in text
    assert _plain(text) == []


def test_a_tripped_safety_stop_is_said_once_and_counted():
    report = _report(
        risk={"halted": True, "tripped": {"as_of_date": "2026-11-09"},
              "latest": {"equity_dollars": -11000.0, "limit_dollars": 10000.0},
              "drawdown_dollars": 11000.0},
        attention=["risk switch tripped on 2026-11-09; the scan files no "
                   "orders until it is reset with a reason"])

    text = _texts(post.status_card(report, when=WHEN))

    assert "## ⚠️ **1** thing needs a look" in text
    assert ("**Safety stop** ❌ on since Nov 9: no new trades until it is "
            "reset" in text)
    assert "risk switch" not in text


def test_an_empty_start_is_said_plainly():
    report = _report(
        gate=None, risk={"halted": False, "latest": None},
        book={"as_of": None, "accepted": 0, "issuers": 0, "shorts": 0})

    text = _texts(post.status_card(report, when=WHEN))

    assert "**Latest picks** none yet" in text
    assert "**Safety stop** nothing traded yet" in text
    assert "**Track record** not checked yet; it is checked each Saturday" in text


def test_a_proven_strategy_says_so():
    report = _report(gate={"measured_trades": 214, "min_trades": 200,
                           "passed": 1, "reason": "t=2.4"})

    text = _texts(post.status_card(report, when=WHEN))

    assert ("**Track record** proven on **214** finished trades" in text)


# ------------------------------------------------------------- /orders

def _fill(status, qty, filled=None, price=None, reason=None, side="buy"):
    return {"status": status, "qty": qty, "filled_qty": filled,
            "filled_price": price, "reason": reason, "ticker": "X",
            "side": side, "leg": "entry"}


def _earnings(orders, exits=()):
    return {"decided": "2026-11-10", "session": "2026-11-11",
            "orders": orders, "exits": list(exits)}


def _order(ticker, entry, sue=1.42, side="long", dollars=3187.92):
    return {"ticker": ticker, "side": side, "sue": sue,
            "target_dollars": dollars, "entry": entry}


def test_the_orders_card_says_what_was_picked_and_what_was_bought():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings([
        _order("AAA", _fill("filled", 10, 10, 50.0)),
        _order("BBB", _fill("expired", 154, 113, 21.15)),
        _order("CCC", _fill("expired", 45, 0)),
        _order("DDD", _fill("expired", 20)),
        _order("EEE", _fill("accepted", 5)),
        _order("FFF", None)])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "## 📋 Orders for Nov 11" in text
    assert "**Earnings strategy** **6** stocks picked on Nov 10" in text
    assert ("**1** bought · **1** partly bought · **2** not bought · "
            "**1** waiting · **1** not sent yet" in text)
    assert "**AAA** buy **$3,188** · surprise **+1.42** · bought **10** shares at **$50.00**" in text
    assert ("**BBB** buy **$3,188** · surprise **+1.42** · bought **113** of "
            "**154** shares at **$21.15**; the rest expired" in text)
    assert "**CCC** buy **$3,188** · surprise **+1.42** · not bought: the order expired at the open" in text
    assert "**EEE** buy **$3,188** · surprise **+1.42** · sent, waiting for the market" in text
    assert "**FFF** buy **$3,188** · surprise **+1.42** · not sent yet" in text
    assert ("-# Surprise: how far earnings beat (+) or missed (−) what was "
            "expected. The bigger it is, the bigger the surprise." in text)
    assert _plain(text) == []


def test_a_bet_against_is_said_as_one():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings([
        _order("AAA", _fill("filled", 10, 10, 50.0, side="sell"),
               side="short", sue=-1.6)])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert ("**AAA** bet against **$3,188** · surprise **-1.60** · sold "
            "short **10** shares at **$50.00**" in text)
    assert "**1** done" in text


def test_a_refused_order_says_why():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings([
        _order("AAA", _fill("rejected", 10, reason="insufficient buying_power"))])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "the broker refused it: insufficient buying\\_power" in text
    assert "**1** refused" in text


def test_more_orders_than_fit_are_counted_not_dropped():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings(
        [_order(f"T{i}", None) for i in range(11)])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "**T7**" in text
    assert "**T8**" not in text
    assert "and **3** more" in text


def test_a_sale_closing_an_earlier_trade_is_listed_after_the_picks():
    exit_leg = {**_fill("filled", 5, 5, 12.0, side="sell"), "ticker": "OLD",
                "leg": "exit"}
    data = {"as_of": "2026-11-11", "copy": None,
            "earnings": _earnings([_order("AAA", None)], [exit_leg])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "**OLD** closing an earlier trade · sold **5** shares at **$12.00**" in text
    assert "**Earnings strategy** **1** stock picked on Nov 10 · **1** earlier trade closing" in text


def test_congress_copying_shows_what_it_copied_and_why_it_skipped_the_rest():
    data = {"as_of": "2026-11-11", "earnings": None, "copy": {
        "decided": "2026-11-10", "seen": 4,
        "orders": [{"ticker": "AAA", "member": "Ada Lovelace",
                    "target_dollars": 5000.0,
                    "entry": _fill("filled", 20, 20, 25.0)}],
        "refused": {"no_ticker": 2, "trade_too_old": 1}}}

    text = _texts(post.orders_card(data, when=WHEN))

    assert ("**Congress copying** **4** trades by members of Congress seen "
            "Nov 10 · **1** copied" in text)
    assert ("**AAA** copying Ada Lovelace · **$5,000** · bought **20** shares "
            "at **$25.00**" in text)
    assert "-# Not copied: no stock symbol **2** · reported too late **1**" in text
    assert _plain(text) == []


def test_congress_copying_that_copied_nothing_says_so():
    data = {"as_of": "2026-11-11", "earnings": None, "copy": {
        "decided": "2026-11-10", "seen": 1, "orders": [],
        "refused": {"no_ticker": 1}}}

    text = _texts(post.orders_card(data, when=WHEN))

    assert ("**Congress copying** **1** trade by a member of Congress seen "
            "Nov 10 · none copied" in text)


def test_no_orders_in_either_strategy_is_nothing_yet():
    payload = post.orders_card({"as_of": "2026-11-11", "earnings": None,
                                "copy": None}, when=WHEN)

    assert "## 💤 No orders yet" in _texts(payload)


# ------------------------------------------------------------- /ticker

def _ticker(**extra):
    found = {"symbol": "AAA", "as_of": "2026-11-11", "signal": None,
             "orders": [], "releases": [], "filings": [], "congress": [],
             "position": None, "position_error": None}
    found.update(extra)
    return found


def test_a_held_stock_leads_with_what_you_own():
    position = {"symbol": "AAA", "side": "long", "qty": 10.0,
                "market_value": 500.0, "unrealized_pl": 6.0,
                "unrealized_plpc": 0.012, "books": ["Earnings"]}

    text = _texts(post.ticker_card(_ticker(position=position), when=WHEN))

    assert "## 📋 AAA · you own **10** shares worth **$500.00**" in text
    assert ("**Gain/loss** **+$6.00** (**+1.20%**) · from the earnings "
            "strategy" in text)


def test_a_bet_against_shows_as_one():
    position = {"symbol": "AAA", "side": "short", "qty": -10.0,
                "market_value": -500.0, "unrealized_pl": -6.0,
                "unrealized_plpc": -0.012, "books": []}

    text = _texts(post.ticker_card(_ticker(position=position), when=WHEN))

    assert "## 📋 AAA · you're betting against **10** shares worth **$500.00**" in text
    assert "not from either strategy" in text


def test_a_stock_says_why_nemo_did_not_trade_it():
    signal = {"as_of_date": "2026-11-10", "accepted": False, "sue": 0.4,
              "side": None, "reason": "|SUE| 0.40 under the 1.0 threshold"}

    text = _texts(post.ticker_card(_ticker(signal=signal), when=WHEN))

    assert "## 📋 AAA · you don't own it" in text
    assert ("**Latest check** Nov 10 · surprise **+0.40** · not traded: the "
            "surprise was too small" in text)
    assert _plain(text) == []


@pytest.mark.parametrize("reason, said", [
    ("already acted on this issuer's 2026Q3; the signal stays fresh for 45 "
     "days and one print is one trade", "already traded this earnings report"),
    ("signal unavailable: ReadTimeout: slow", "its earnings could not be read"),
    ("expected 4.1bp does not clear 6.0bp of cost", "the expected gain was smaller than the cost"),
    ("ranked below the top 8 on net edge", "other picks ranked higher"),
    ("BBB is the same issuer as AAA", "another pick is the same company"),
    ("something new and unforeseen", "something new and unforeseen"),
])
def test_each_reason_not_to_trade_is_said_plainly(reason, said):
    signal = {"as_of_date": "2026-11-10", "accepted": False, "sue": None,
              "reason": reason}

    text = _texts(post.ticker_card(_ticker(signal=signal), when=WHEN))

    assert f"not traded: {said}" in text


def test_a_traded_stock_shows_its_orders_with_the_buy_and_the_sale():
    signal = {"as_of_date": "2026-11-10", "accepted": True, "sue": 1.42,
              "side": "long", "reason": None}
    orders = [{"as_of_date": "2026-11-10", "side": "long",
               "target_dollars": 3187.92,
               "entry": _fill("filled", 10, 10, 40.0),
               "exit": {**_fill("filled", 10, 10, 44.0, side="sell"),
                        "leg": "exit"}}]

    text = _texts(post.ticker_card(_ticker(signal=signal, orders=orders),
                                   when=WHEN))

    assert "**Latest check** Nov 10 · surprise **+1.42** · picked to buy" in text
    assert ("Nov 10 buy **$3,188** · bought **10** shares at **$40.00** · "
            "then sold **10** shares at **$44.00**" in text)


def test_a_stock_shows_its_earnings_reports_activists_and_congress_trades():
    releases = [{"fiscal_period": "2026Q3", "announced_date": "2026-10-21",
                 "timing": "bmo"},
                {"fiscal_period": "2026Q2", "announced_date": "2026-07-22",
                 "timing": "amc"}]
    filings = [{"form": "SC 13D/A", "filing_date": "2026-10-01",
                "filer_name": "Example LP",
                "url": "https://www.sec.gov/Archives/x.htm"}]
    congress = [{"transaction_date": "2026-10-20", "member": "Ada Lovelace",
                 "chamber": "senate", "amount_min": 1001,
                 "amount_max": 15000, "accepted": False,
                 "reason": "trade_too_old", "target_dollars": None}]

    payload = post.ticker_card(_ticker(releases=releases, filings=filings,
                                       congress=congress), when=WHEN)

    text = _texts(payload)
    assert ("**Last earnings report** Oct 21 (Q3 2026), before the market "
            "opened" in text)
    assert "-# Earlier reports: Jul 22" in text
    assert "**Activist investors**" in text
    assert "Oct 1 · Example LP · updated their stake" in text
    assert "**Congress**" in text
    assert ("Oct 20 · Ada Lovelace (Senate) · $1,001–$15,000 · not copied: "
            "reported too late" in text)
    [button] = _buttons(payload)
    assert button["label"] == "Filing Oct 1"
    assert _plain(text) == []


def test_an_account_that_cannot_be_read_is_said_not_taken_for_no_position():
    text = _texts(post.ticker_card(_ticker(
        position_error="AsyncBrokerError: 401 unauthorized"), when=WHEN))

    assert "## 📋 AAA · couldn't check your account" in text
    assert ("⚠️ The paper account could not be read: AsyncBrokerError: 401 "
            "unauthorized" in text)


def test_a_stock_with_nothing_on_record_says_so():
    payload = post.ticker_card(_ticker(), when=WHEN)

    text = _texts(payload)
    assert "## 💤 AAA · Nemo has nothing on it" in text
    assert ("No picks, orders, earnings reports, activist filings or "
            "congress trades for **AAA**." in text)


@pytest.mark.parametrize("builder, data", [
    (post.status_card, _report()),
    (post.orders_card, {"as_of": "2026-11-11", "earnings": None, "copy": None}),
    (post.ticker_card, _ticker()),
])
def test_every_card_is_one_components_v2_container_with_no_pings(builder, data):
    payload = builder(data, when=WHEN)

    assert payload["flags"] & post.IS_COMPONENTS_V2
    assert len(payload["components"]) == 1
    assert payload["allowed_mentions"] == {"parse": []}
