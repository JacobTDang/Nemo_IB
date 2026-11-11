"""The cards the Discord bot answers /status, /orders and /ticker with.

Built from results shaped like `status.collect`, `lookup.orders` and
`lookup.ticker`, as a reader sees them on a phone.
"""
from __future__ import annotations

import pytest

import research.discord_post as post

WHEN = 1_794_000_000


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


# ------------------------------------------------------------- /status

def _job(name, state="ok", **extra):
    return {"job": name, "state": state, "partial_is_normal": False,
            "last_run": "2026-11-10", "every": "weeknight", **extra}


def _report(jobs=None, attention=(), **extra):
    report = {
        "as_of": "2026-11-11",
        "jobs": jobs if jobs is not None else [_job("scan"), _job("universe")],
        "book": {"as_of": "2026-11-10", "accepted": 8, "issuers": 8,
                 "shorts": 0, "borrow_rates": 0},
        "risk": {"halted": False, "tripped": None,
                 "latest": {"equity_dollars": 120.0, "limit_dollars": 5000.0,
                            "closed_trades": 3, "open_trades": 18,
                            "unmarked_trades": 0},
                 "drawdown_dollars": 40.0},
        "gate": {"measured_trades": 4, "min_trades": 30, "passed": 0,
                 "reason": "4 of 30 trades"},
        "copy": {"open": 0, "closed": 0, "mean_return": None},
        "attention": list(attention)}
    report.update(extra)
    return report


def test_a_clean_status_says_all_clear_and_lists_the_jobs_on_one_line():
    payload = post.status_card(_report(), when=WHEN)

    text = _texts(payload)
    assert "## ✅ Nemo status · all clear" in text
    assert "✅ scan · universe" in text
    assert _color(payload) == post.GREEN
    assert "nemo /status" in text


def test_the_status_facts_are_the_book_the_risk_switch_and_the_gate():
    text = _texts(post.status_card(_report(), when=WHEN))

    assert "**Book** **8** orders for 2026-11-10 · **8** issuers · **0** short" in text
    assert "**Risk** switch armed · drawdown **$40.00** of **$5,000.00**" in text
    assert "**Gate** **4** of **30** trades measured · not passed" in text
    assert "**Copy book** **0** open · **0** closed" in text


def test_attention_turns_the_card_yellow_and_names_each_item_once():
    jobs = [_job("scan", "failed"), _job("universe")]
    payload = post.status_card(_report(
        jobs, ["scan failed on 2026-11-10: HTTP 503"]), when=WHEN)

    text = _texts(payload)
    assert "## ⚠️ Nemo status · **1** item needs attention" in text
    assert "❌ scan failed on 2026-11-10: HTTP 503" in text
    assert "✅ universe" in text
    assert "-#" not in text.split("❌")[0]
    assert _color(payload) == post.YELLOW


def test_a_job_worth_a_glance_is_named_with_its_state_but_not_raised():
    jobs = [_job("scan"), _job("announce", "running"),
            _job("seed", "never"), _job("congress_copy", "off"),
            _job("activist_watch", "partial", partial_is_normal=True)]

    text = _texts(post.status_card(_report(jobs), when=WHEN))

    assert "✅ scan · activist_watch" in text
    assert "-# announce running · seed not run yet · congress_copy off" in text
    assert "## ✅ Nemo status · all clear" in text


def test_a_tripped_switch_and_an_empty_store_are_said_plainly():
    report = _report(
        risk={"halted": True, "tripped": {"as_of_date": "2026-11-09"},
              "latest": {"equity_dollars": -6000.0, "limit_dollars": 5000.0},
              "drawdown_dollars": 6000.0},
        gate=None, book={"as_of": None, "accepted": 0, "issuers": 0,
                         "shorts": 0, "borrow_rates": 0})

    text = _texts(post.status_card(report, when=WHEN))

    assert "**Risk** switch **tripped** on 2026-11-09" in text
    assert "**Gate** not checked yet" in text
    assert "**Book** no orders yet" in text


# ------------------------------------------------------------- /orders

def _fill(status, qty, filled=None, price=None, reason=None):
    return {"status": status, "qty": qty, "filled_qty": filled,
            "filled_price": price, "reason": reason, "ticker": "X",
            "side": "buy", "leg": "entry"}


def _earnings(orders, exits=()):
    return {"decided": "2026-11-10", "session": "2026-11-11",
            "orders": orders, "exits": list(exits)}


def _order(ticker, entry, sue=1.42, side="long", dollars=3187.92):
    return {"ticker": ticker, "side": side, "sue": sue,
            "target_dollars": dollars, "entry": entry}


def test_the_orders_card_counts_each_outcome_and_shows_each_order():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings([
        _order("AAA", _fill("filled", 10, 10, 50.0)),
        _order("BBB", _fill("expired", 154, 113, 21.15)),
        _order("CCC", _fill("expired", 45, 0)),
        _order("DDD", _fill("expired", 20)),
        _order("EEE", _fill("accepted", 5)),
        _order("FFF", None)])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "## 📋 Orders for 2026-11-11" in text
    assert ("**Earnings** **6** orders decided 2026-11-10" in text)
    assert ("**1** filled · **1** part-filled · **2** expired unfilled · "
            "**1** waiting · **1** not sent" in text)
    assert "**AAA** long · SUE **1.42** · **$3,187.92** · filled **10** at **$50.00**" in text
    assert "**BBB** long · SUE **1.42** · **$3,187.92** · **113** of **154** filled at **$21.15**, then expired" in text
    assert "**CCC** long · SUE **1.42** · **$3,187.92** · expired, nothing filled" in text
    assert "**EEE** long · SUE **1.42** · **$3,187.92** · sent, waiting" in text
    assert "**FFF** long · SUE **1.42** · **$3,187.92** · not sent yet" in text


def test_a_refused_leg_says_why():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings([
        _order("AAA", _fill("rejected", 10, reason="insufficient buying_power"))])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "rejected: insufficient buying\\_power" in text
    assert "**1** rejected" in text


def test_more_orders_than_fit_are_counted_not_dropped():
    data = {"as_of": "2026-11-11", "copy": None, "earnings": _earnings(
        [_order(f"T{i}", None) for i in range(11)])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "**T7**" in text
    assert "**T8**" not in text
    assert "and **3** more" in text


def test_exits_for_the_session_are_listed_after_the_entries():
    exit_leg = {**_fill("filled", 5, 5, 12.0), "ticker": "OLD", "side": "sell",
                "leg": "exit"}
    data = {"as_of": "2026-11-11", "copy": None,
            "earnings": _earnings([_order("AAA", None)], [exit_leg])}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "**OLD** exit · filled **5** at **$12.00**" in text
    assert "**1** exit" in text


def test_the_copy_book_shows_what_it_copied_and_counts_its_refusals():
    data = {"as_of": "2026-11-11", "earnings": None, "copy": {
        "decided": "2026-11-10", "seen": 4,
        "orders": [{"ticker": "AAA", "member": "Ada Lovelace",
                    "target_dollars": 5000.0,
                    "entry": _fill("filled", 20, 20, 25.0)}],
        "refused": {"no_ticker": 2, "stale": 1}}}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "**Copy** **4** congressional trades seen 2026-11-10 · **1** copied" in text
    assert "**AAA** copy · Ada Lovelace · **$5,000.00** · filled **20** at **$25.00**" in text
    assert "-# Not copied: no ticker **2** · stale **1**" in text


def test_a_copy_book_that_copied_nothing_says_so():
    data = {"as_of": "2026-11-11", "earnings": None, "copy": {
        "decided": "2026-11-10", "seen": 1, "orders": [],
        "refused": {"no_ticker": 1}}}

    text = _texts(post.orders_card(data, when=WHEN))

    assert "**Copy** **1** congressional trade seen 2026-11-10 · none copied" in text


def test_no_decision_in_either_book_is_nothing_yet():
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


def test_a_held_ticker_leads_with_its_position_and_its_books():
    position = {"symbol": "AAA", "side": "long", "qty": 10.0,
                "market_value": 500.0, "unrealized_pl": 6.0,
                "unrealized_plpc": 0.012, "books": ["Earnings"]}

    text = _texts(post.ticker_card(_ticker(position=position), when=WHEN))

    assert "## 📋 AAA · long **10** shares · **$500.00**" in text
    assert "**Unrealized** **+$6.00** (**+1.20%**) · Earnings" in text
    assert "nemo /ticker" in text


def test_a_short_position_shows_its_value_as_negative():
    position = {"symbol": "AAA", "side": "short", "qty": -10.0,
                "market_value": -500.0, "unrealized_pl": -6.0,
                "unrealized_plpc": -0.012, "books": []}

    text = _texts(post.ticker_card(_ticker(position=position), when=WHEN))

    assert "## 📋 AAA · short **10** shares · **-$500.00**" in text
    assert "· no book" in text


def test_a_ticker_says_why_its_latest_signal_was_not_traded():
    signal = {"as_of_date": "2026-11-10", "accepted": False, "sue": 0.4,
              "side": None, "reason": "|SUE| 0.40 under the 1.0 threshold"}

    text = _texts(post.ticker_card(_ticker(signal=signal), when=WHEN))

    assert "## 📋 AAA · no position" in text
    assert ("**Signal** 2026-11-10 · SUE **0.40** · not traded: "
            "\\|SUE\\| 0.40 under the 1.0 threshold" in text)


def test_a_traded_signal_and_its_orders_with_both_legs():
    signal = {"as_of_date": "2026-11-10", "accepted": True, "sue": 1.42,
              "side": "long", "reason": None}
    orders = [{"as_of_date": "2026-11-10", "side": "long",
               "target_dollars": 3187.92,
               "entry": _fill("filled", 10, 10, 40.0),
               "exit": {**_fill("filled", 10, 10, 44.0), "leg": "exit"}}]

    text = _texts(post.ticker_card(_ticker(signal=signal, orders=orders),
                                   when=WHEN))

    assert "**Signal** 2026-11-10 · SUE **1.42** · **long**, ordered" in text
    assert ("2026-11-10 long **$3,187.92** · filled **10** at **$40.00** · "
            "exit filled **10** at **$44.00**" in text)


def test_a_ticker_shows_its_releases_filings_and_congress_trades():
    releases = [{"fiscal_period": "2026Q3", "announced_date": "2026-10-21",
                 "timing": "bmo"},
                {"fiscal_period": "2026Q2", "announced_date": "2026-07-22",
                 "timing": "amc"}]
    filings = [{"form": "SC 13D", "filing_date": "2026-10-01",
                "filer_name": "Example LP",
                "url": "https://www.sec.gov/Archives/x.htm"}]
    congress = [{"transaction_date": "2026-10-20", "member": "Ada Lovelace",
                 "chamber": "senate", "amount_min": 1001,
                 "amount_max": 15000, "accepted": False, "reason": "stale",
                 "target_dollars": None}]

    payload = post.ticker_card(_ticker(releases=releases, filings=filings,
                                       congress=congress), when=WHEN)

    text = _texts(payload)
    assert "**Last release** 2026-10-21 · 2026Q3 · before the open" in text
    assert "-# Earlier: 2026-07-22 (2026Q2)" in text
    assert "**SC 13D** 2026-10-01 · Example LP" in text
    assert ("2026-10-20 · Ada Lovelace (senate) · $1,001–$15,000 · "
            "not copied: stale" in text)
    [button] = _buttons(payload)
    assert button["label"] == "SC 13D 2026-10-01"


def test_an_account_that_cannot_be_read_is_said_not_taken_for_no_position():
    text = _texts(post.ticker_card(_ticker(
        position_error="AsyncBrokerError: 401 unauthorized"), when=WHEN))

    assert "## 📋 AAA · position unknown" in text
    assert "⚠️ The paper account could not be read: AsyncBrokerError: 401 unauthorized" in text


def test_a_ticker_with_nothing_on_record_says_so():
    payload = post.ticker_card(_ticker(), when=WHEN)

    text = _texts(payload)
    assert "## 💤 AAA · nothing on record" in text
    assert "no signal, order, release, 13D or congressional trade" in text


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
