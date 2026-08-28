"""A window the caller did not ask for must say so.

Three tools answered a narrower question than the one they were asked and
returned the number as though it were the answer:

    backtest_signal(NVDA, start_date="2015-01-01")
        -> date_range "2021-08-26 -> 2026-08-26", n_trades 4, warning null

    Eleven years requested, five returned. `start_date` alone never reached
    yfinance -- the fetch fell through to a hardcoded `period='5y'` because
    the code only honoured start and end *together*. `end_date` alone was
    worse: asking for history up to 2020 returned 2021-2026, a window that
    does not overlap the request at all. A hit rate over 4 trades is a
    different statistic from one over a decade of signals, and nothing in
    the response distinguished them.

    get_fred_series("UNRATE", start="2025-09-01", end="2025-12-01")
        -> total_observations 4, returned 3, dates Sep/Nov/Dec

    October 2025 is missing. FRED publishes that row with a '.' value: no
    household survey was conducted that month. The condenser filtered it and
    left two counts that do not reconcile. Worse, `returned` was overloaded --
    on DGS10 it means "tail-truncated from 1302", on UNRATE it means "one
    value was unavailable". A caller cannot tell a truncation from a hole,
    and a hole in a macro series is itself the news.

    get_company_news("NVDA", "2026-08-18", "2026-08-26")
        -> 20 articles, every one stamped 2026-08-26

    Finnhub returned 246. The seven earlier days vanished with no total, no
    returned count and no truncated flag, so the response reads as "here is
    the news for your window" when it is "here are the 20 most recent". For
    a catalyst workflow that is the difference between "nothing happened
    last week" and "we did not look".

The rule: honour the window, or name the one actually covered next to the
one requested. Echoing the covered range is not enough on its own -- a
reader has to notice the shortfall to learn there was one.
"""
from datetime import datetime, timedelta

import pytest

from testing._gates import requires_fred, requires_finnhub


# ---------------------------------------------------------------------------
# 1. backtest_signal
# ---------------------------------------------------------------------------

# A stand-in for NVDA: listed 1999, quoted through today. The engine has to
# slice this the way yfinance would, so a window it never asked for shows up
# as bars that are not there.
_LISTING_DATE = "1999-01-22"
_TODAY = "2026-08-26"


def _universe(listing_date: str = _LISTING_DATE):
    import pandas as pd
    index = pd.date_range(start=listing_date, end=_TODAY, freq="B")
    closes = [100.0 + (i % 37) - 18 for i in range(len(index))]
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": [1_000] * len(index),
    }, index=index)


@pytest.fixture
def recorded_fetch(monkeypatch):
    """Capture the kwargs the engine hands yfinance, and serve bars back.

    The clamp is invisible from the outside once a frame comes back, so the
    test watches the call itself: a `period` kwarg where the caller supplied
    dates *is* the defect. The frame is then sliced the way yfinance slices
    it, so the returned window reflects what was actually asked for.
    """
    import pandas as pd
    import yfinance

    class _Fetch(dict):
        """The kwargs seen, plus the listing date a test wants to pretend to."""
        listing_date = _LISTING_DATE

    captured = _Fetch()

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            captured.update(kwargs)
            frame = _universe(captured.listing_date)
            period = kwargs.get("period")
            if period:
                # Whatever a period means, it is not the caller's dates.
                years = int(period.rstrip("y"))
                floor = pd.Timestamp(_TODAY) - pd.DateOffset(years=years)
                return frame[frame.index >= floor]
            if kwargs.get("start"):
                frame = frame[frame.index >= kwargs["start"]]
            if kwargs.get("end"):
                frame = frame[frame.index < kwargs["end"]]
            return frame

    monkeypatch.setattr(yfinance, "Ticker", _Ticker)
    return captured


def _run(**kwargs):
    from agent.backtest_engine import backtest_signal
    return backtest_signal(
        ticker="NVDA",
        signal={"metric": "rsi_14", "op": "<", "value": 30},
        hold_days=30, cooldown_days=30, signal_name="oversold_rsi",
        **kwargs)


def test_a_start_date_alone_is_not_replaced_by_a_five_year_period(recorded_fetch):
    result = _run(start_date="2015-01-01")

    assert recorded_fetch.get("start") == "2015-01-01", (
        f"start_date was dropped on the way to the fetch: {recorded_fetch}")
    assert "period" not in recorded_fetch, (
        "the caller gave a start date and the fetch used a hardcoded period")
    assert result.date_range.startswith("2015-01"), (
        f"asked for history from 2015 and got {result.date_range}")


def test_an_end_date_alone_is_not_replaced_by_a_five_year_period(recorded_fetch):
    """The old fall-through returned 2021-2026 for a request ending in 2020 --
    a window with no overlap at all with the one asked for."""
    result = _run(end_date="2020-01-01")

    assert "period" not in recorded_fetch, (
        "the caller gave an end date and the fetch used a hardcoded period")
    assert result.date_range.endswith("2019-12-31"), (
        f"asked for history up to 2020-01-01 and got {result.date_range}")
    assert result.date_range.startswith("1999-01"), (
        "'everything up to 2020' must not silently become 'the month before'")


def test_both_dates_still_work(recorded_fetch):
    """The one case that already behaved must keep behaving."""
    _run(start_date="2023-01-01", end_date="2024-01-01")
    assert recorded_fetch.get("start") == "2023-01-01"
    assert recorded_fetch.get("end") == "2024-01-01"


def test_the_result_names_the_window_that_was_asked_for(recorded_fetch):
    """`date_range` alone cannot be checked against anything. A reader needs
    both halves in the response to see a shortfall without re-deriving it."""
    result = _run(start_date="2015-01-01")
    assert result.requested_range, "the response does not say what was requested"
    assert "2015-01-01" in result.requested_range


def test_the_default_window_says_it_is_a_default(recorded_fetch):
    """No dates given is still a 5-year answer to an unbounded question."""
    result = _run()
    assert result.requested_range, (
        "with no dates the engine picks a window and never says which")
    assert "5y" in result.requested_range or "default" in result.requested_range.lower()


def test_history_short_of_the_requested_start_is_flagged(recorded_fetch):
    """Honouring the request is not the same as covering it: a 2015 start on a
    1999-listed ticker is fine, on a 2021 listing it is a six-year hole."""
    recorded_fetch.listing_date = "2021-01-04"
    result = _run(start_date="2015-01-01")

    assert result.caveats, (
        f"requested from 2015, covered {result.date_range}, and nothing says so")
    assert any("2015-01-01" in c for c in result.caveats)


def _fires_on(indices):
    """A signal that triggers on exactly these bar indices."""
    return lambda ticker, idx, indicators: idx in indices


def test_a_four_trade_sample_is_labelled_as_one(recorded_fetch):
    """25.0% and a sharpe of 0.357 over four trades were reported with the
    same confidence as a hundred. The `warning` field existed and never fired."""
    from agent import backtest_engine

    result = backtest_engine.backtest_signal(
        ticker="NVDA", signal=_fires_on({300, 900, 1500, 2100}),
        hold_days=30, signal_name="rare", start_date="2015-01-01")

    assert result.n_trades == 4
    assert any("sample" in c.lower() or "trade" in c.lower() for c in result.caveats), (
        f"{result.n_trades} trades produced hit_rate {result.hit_rate} and "
        f"sharpe {result.sharpe_simple} with no caveat: {result.caveats}")


def test_a_caveat_is_not_a_failure(recorded_fetch):
    """`success` is derived from `warning`. A small-sample note must not make
    a working backtest look like a broken one."""
    from agent.backtest_engine import backtest_signal
    result = backtest_signal(
        ticker="NVDA", signal=_fires_on({300, 900, 1500, 2100}),
        hold_days=30, signal_name="rare", start_date="2015-01-01")
    assert result.warning is None
    assert result.caveats


def test_a_large_sample_carries_no_small_sample_caveat(recorded_fetch):
    """Removing the silence must not replace it with noise on every result."""
    from agent.backtest_engine import backtest_signal, MIN_TRADES_FOR_STATS
    result = backtest_signal(
        ticker="NVDA", signal=_fires_on(set(range(0, 2500, 50))),
        hold_days=30, signal_name="common", start_date="2015-01-01")
    assert result.n_trades >= MIN_TRADES_FOR_STATS
    assert not any("sample" in c.lower() for c in result.caveats), result.caveats


# ---------------------------------------------------------------------------
# 2. get_fred_series
# ---------------------------------------------------------------------------

def _fred_payload(rows, count=None):
    return {"series_id": "UNRATE", "count": count if count is not None else len(rows),
            "observations": [{"date": d, "value": v} for d, v in rows]}


def test_an_unavailable_observation_is_counted_not_dropped():
    """FRED carries October 2025 for UNRATE with value '.': the government
    shutdown meant no household survey. Filtering it left total 4 / returned 3
    and no way to see which month went missing."""
    from tools.news_agregator.fred_server import _condense_observations

    out = _condense_observations(_fred_payload([
        ("2025-09-01", "4.4"), ("2025-10-01", "."),
        ("2025-11-01", "4.6"), ("2025-12-01", "4.5"),
    ]))

    assert out["unavailable_observations"] == 1
    assert out["unavailable_dates"] == ["2025-10-01"], (
        "a missing UNRATE print is an event; the response must name the month")
    assert out["observations_with_values"] == 3
    assert out["total_observations"] == 4


def test_the_counts_reconcile():
    """total = published + unavailable. Two numbers that cannot be made to
    agree are how the hole hid in the first place."""
    from tools.news_agregator.fred_server import _condense_observations
    out = _condense_observations(_fred_payload([
        ("2025-09-01", "4.4"), ("2025-10-01", "."), ("2025-11-01", "4.6"),
    ]))
    assert (out["observations_with_values"] + out["unavailable_observations"]
            == out["total_observations"])


def test_a_hole_is_distinguishable_from_a_truncation():
    """`returned` meant 'tail-truncated from 1302' on DGS10 and 'one value was
    unavailable' on UNRATE. Those are different facts."""
    from tools.news_agregator.fred_server import _condense_observations

    hole = _condense_observations(_fred_payload([
        ("2025-09-01", "4.4"), ("2025-10-01", "."), ("2025-11-01", "4.6"),
    ]))
    assert hole["truncated"] is False, (
        "nothing was cut off; a missing value is not a truncation")

    long_run = _condense_observations(_fred_payload(
        [(f"2020-01-{d:02d}", "1.5") for d in range(1, 29)] * 4), cap=10)
    assert long_run["truncated"] is True
    assert long_run["unavailable_observations"] == 0


def test_a_clean_window_is_not_flagged():
    from tools.news_agregator.fred_server import _condense_observations
    out = _condense_observations(_fred_payload([
        ("2025-09-01", "4.4"), ("2025-10-01", "4.5"),
    ]))
    assert out["unavailable_observations"] == 0
    assert out["unavailable_dates"] == []
    assert out["truncated"] is False


@requires_fred
@pytest.mark.network
def test_unrate_reports_its_october_hole_live():
    import asyncio, json
    from tools.news_agregator.fred_server import FredServer

    async def go():
        server = FredServer()
        try:
            out = await server.get_fred_series(
                series_id="UNRATE", start="2025-09-01", end="2025-12-01")
        finally:
            await server.client.close()
        return json.loads(out[0].text)["data"]

    data = asyncio.run(go())
    assert data["unavailable_observations"] >= 1, data
    assert "2025-10-01" in data["unavailable_dates"], data


# ---------------------------------------------------------------------------
# 3. get_company_news
# ---------------------------------------------------------------------------

def _articles(n, day="2026-08-26"):
    stamp = int(datetime.fromisoformat(f"{day}T12:00:00+00:00").timestamp())
    return [{"headline": f"h{i}", "summary": "s", "source": "src",
             "datetime": stamp - i * 60, "url": f"u{i}", "id": i, "image": "x"}
            for i in range(n)]


def test_a_capped_news_window_reports_what_it_cut():
    """246 articles in, 20 out, and the response was a bare list."""
    from tools.news_agregator.finnhub_server import _news_page

    page = _news_page(_articles(246), "2026-08-18", "2026-08-26", cap=20)

    assert page["total_articles"] == 246
    assert page["returned"] == 20
    assert page["truncated"] is True
    assert len(page["articles"]) == 20


def test_the_window_actually_covered_is_named():
    """All twenty articles landed on the final day of an eight-day window.
    'Nothing happened last week' and 'we did not look' must not read alike."""
    from tools.news_agregator.finnhub_server import _news_page

    page = _news_page(_articles(246), "2026-08-18", "2026-08-26", cap=20)

    assert page["window_requested"] == "2026-08-18 -> 2026-08-26"
    assert page["window_returned"] == "2026-08-26 -> 2026-08-26", (
        "the returned articles span one day of the eight requested")


def test_an_uncapped_window_is_not_flagged():
    from tools.news_agregator.finnhub_server import _news_page
    page = _news_page(_articles(5), "2026-08-18", "2026-08-26", cap=20)
    assert page["truncated"] is False
    assert page["total_articles"] == page["returned"] == 5


def test_an_empty_window_is_still_a_window():
    from tools.news_agregator.finnhub_server import _news_page
    page = _news_page([], "2026-08-18", "2026-08-26", cap=20)
    assert page["total_articles"] == 0
    assert page["truncated"] is False
    assert page["window_returned"] is None
    assert page["articles"] == []


def test_articles_are_still_slimmed():
    """The bloat fields the old cap stripped must stay stripped."""
    from tools.news_agregator.finnhub_server import _news_page
    article = _news_page(_articles(1), "2026-08-18", "2026-08-26")["articles"][0]
    assert set(article) == {"headline", "summary", "source", "datetime", "url"}


def test_an_empty_window_still_earns_the_not_covered_label():
    """Two silences, and closing one must not open the other.

    A bare `[]` used to reach `build_envelope`, which labelled it
    `coverage: "not_covered"` with a warning that Finnhub answers the same
    empty body for an unknown symbol, an unentitled one, and a real company
    with a quiet week -- so the emptiness is not evidence about the company.
    Wrapping the list in a page put `window_requested` in the payload, which
    `_has_content` read as content, and the label stopped firing.

    The window echo and the label are both wanted: "we looked at these eight
    days and Finnhub had nothing for us" is the whole answer, and neither
    half says it alone.
    """
    from tools.news_agregator.finnhub_server import _news_page
    from tools.news_agregator.finnhub_utils import build_envelope

    page = _news_page([], "2026-08-18", "2026-08-26", cap=20)
    envelope = build_envelope(page, "ZZZZNOTREAL", "get_company_news")

    assert envelope["data"]["window_requested"] == "2026-08-18 -> 2026-08-26", (
        "the window a caller asked about is most worth echoing when the "
        "answer came back empty")
    assert envelope.get("coverage") == "not_covered"
    assert [w["code"] for w in envelope.get("warnings", [])] == [
        "finnhub_returned_nothing"]


def test_a_window_with_articles_is_not_labelled_uncovered():
    """The counts must not become a permanent 'we have content' -- that would
    trade a label that never fires for one that always does."""
    from tools.news_agregator.finnhub_server import _news_page
    from tools.news_agregator.finnhub_utils import build_envelope

    page = _news_page(_articles(3), "2026-08-18", "2026-08-26", cap=20)
    envelope = build_envelope(page, "NVDA", "get_company_news")

    assert envelope["data"]["window_requested"] == "2026-08-18 -> 2026-08-26"
    assert "coverage" not in envelope
    assert not envelope.get("warnings")


def test_a_lookup_echo_alone_is_not_content():
    """The rule under both cases above, stated where it lives: a restatement
    of the request is not an answer to it. `symbol` and `metricType` were
    already read this way."""
    from tools.news_agregator.finnhub_utils import _has_content

    assert _has_content({"window_requested": "a -> b", "total_articles": 0,
                         "returned": 0, "truncated": False,
                         "window_returned": None, "articles": []}) is False
    assert _has_content({"window_requested": "a -> b",
                         "articles": [{"headline": "x"}]}) is True


@requires_finnhub
@pytest.mark.network
def test_company_news_reports_its_cap_live():
    import asyncio, json
    from tools.news_agregator.finnhub_server import FinnhubServer

    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")

    async def go():
        server = FinnhubServer()
        try:
            out = await server.get_company_news("NVDA", from_date, to_date)
        finally:
            await server.client.close()
        return json.loads(out[0].text)["data"]

    data = asyncio.run(go())
    if not isinstance(data, dict):
        pytest.skip(f"Finnhub returned an error envelope: {data}")
    assert data["window_requested"] == f"{from_date} -> {to_date}"
    assert data["total_articles"] >= data["returned"]
    assert data["truncated"] is (data["total_articles"] > data["returned"])
