"""The insider tools must answer the window they were asked for, or name the one they answered.

Two defects, one rule.

1. `get_insider_sentiment` declares `from_date` and `to_date` and ignores them.

   Verified live 2026-08-26:

       get_insider_sentiment("AMAT")
       get_insider_sentiment("AMAT", from_date="2025-10-01", to_date="2026-07-01")

   returned byte-identical payloads -- the same six months (2026-02 .. 2026-07)
   and the same `avg_mspr: -66.6667`. Two causes, stacked:

     * Finnhub's `/stock/insider-sentiment` filters on the *year* of `from`
       and `to` and nothing finer. `from=2025-10-01&to=2026-07-01` and
       `from=2025-08-26&to=2026-08-26` both return the identical 16 rows
       (2025-01 .. 2026-07); `from=2024-01-01&to=2024-06-30` returns all
       twelve 2024 months. The upstream cannot honour a month-precise
       window, so the filtering has to happen here.

     * The condenser then kept the six most recent rows whatever the request,
       which erases any `from_date` older than six months and hides the
       window it actually answered. A caller reverse-engineers the coverage
       from the months array, or does not notice at all.

   This is what made the two insider tools irreconcilable.
   `get_insider_transactions("AMAT")` states `period_start 2025-10-01`,
   `period_end 2026-07-01` and `total_bought 0, buy_count 0`; sentiment
   reported 2026-03 at `mspr +100, change_shares +6669` -- a buy inside the
   window the other tool said had none -- and passing those exact dates to
   align them did nothing.

2. `recent_30d` reports zeros for a window the data does not reach.

   The buckets are measured from *today*, which is right. But Finnhub's
   transaction feed for AMAT ends 2026-07-01 and today is 2026-08-26, so the
   30-day window (2026-07-27 .. 2026-08-26) lies entirely outside the data.
   `recent_30d: {bought: 0, sold: 0, net: 0}` reads as "no insider bought or
   sold in the last month" when the truth is "we hold nothing for that
   month". `recent_90d` has the same problem in weaker form: it claims 90
   days and covers 34 of them.

The rule both violate: a window that cannot be honoured must be refused or
declared, never silently substituted. Zeros are a measurement; absence is not.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from testing._gates import requires_finnhub

from tools.news_agregator.finnhub_server import (
    _condense_insider_data,
    _condense_insider_sentiment,
)


def _month_rows(*specs):
    """Finnhub-shaped monthly MSPR rows: (year, month, mspr, change)."""
    return {"data": [{"year": y, "month": m, "mspr": s, "change": c}
                     for (y, m, s, c) in specs]}


# ---------------------------------------------------------------------------
# 1. get_insider_sentiment honours from_date / to_date
# ---------------------------------------------------------------------------

def test_months_outside_the_requested_window_are_dropped():
    """A row Finnhub returned because it shares a calendar year with the
    request is not a row inside the request. The year-granular upstream hands
    back all of 2025 and all of 2026 for any window touching them; a caller
    asking for 2026-02 onwards must not be given 2025-01."""
    raw = _month_rows(
        (2025, 1, -100.0, -5_000),
        (2025, 11, -50.0, -2_000),
        (2026, 3, 100.0, 6_669),
        (2026, 7, -100.0, -125),
    )
    out = _condense_insider_sentiment(raw, "2026-02-01", "2026-07-31")

    got = [(m["year"], m["month"]) for m in out["months"]]
    assert got == [(2026, 7), (2026, 3)], (
        f"months outside 2026-02..2026-07 survived the filter: {got}")


def test_two_different_windows_give_two_different_answers():
    """The defect in one assertion: the same rows, two requests, one answer.
    If narrowing the window cannot change the payload, the window argument is
    decoration."""
    raw = _month_rows(
        (2025, 10, -100.0, -1_000),
        (2025, 11, -100.0, -1_000),
        (2026, 3, 100.0, 6_669),
        (2026, 7, -100.0, -125),
    )
    wide = _condense_insider_sentiment(raw, "2025-01-01", "2026-12-31")
    narrow = _condense_insider_sentiment(raw, "2026-03-01", "2026-03-31")

    assert json.dumps(wide, sort_keys=True) != json.dumps(narrow, sort_keys=True), (
        "a twelve-month request and a one-month request returned the identical "
        "payload, which is how the ignored parameters went unnoticed")
    assert [(m["year"], m["month"]) for m in narrow["months"]] == [(2026, 3)]
    assert narrow["avg_mspr"] == 100.0, (
        f"avg must be taken over the requested window only: {narrow['avg_mspr']}")


def test_a_partly_covered_month_is_not_counted_as_a_whole_one():
    """MSPR is a monthly aggregate and cannot be pro-rated. A request ending
    2026-07-01 covers one day of July; reporting July's ratio inside it would
    answer with 30 days the caller excluded. Excluding it is the honest half;
    `window_returned` naming 2026-06 as the end is the other half."""
    raw = _month_rows((2026, 6, -100.0, -207_500), (2026, 7, -100.0, -125))
    out = _condense_insider_sentiment(raw, "2025-10-01", "2026-07-01")

    assert [(m["year"], m["month"]) for m in out["months"]] == [(2026, 6)], (
        f"July was only one day inside the request: {out['months']}")
    assert out["window_returned"] == "2026-06 -> 2026-06", out["window_returned"]


def test_the_window_actually_covered_is_stated():
    """`get_insider_transactions` states period_start/period_end. Sentiment
    stated nothing, so the only way to learn its coverage was to read the
    months array and infer. Both windows are named, the way `_news_page`
    names them, so a shortfall is visible without arithmetic."""
    raw = _month_rows((2026, 3, 100.0, 6_669), (2026, 5, -100.0, -19_251))
    out = _condense_insider_sentiment(raw, "2025-10-01", "2026-08-31")

    assert out["window_requested"] == "2025-10-01 -> 2026-08-31", out.get("window_requested")
    assert out["window_returned"] == "2026-03 -> 2026-05", out.get("window_returned")
    assert out["returned"] == 2 and out["total_months"] == 2
    assert out["truncated"] is False


def test_a_capped_window_says_it_was_capped():
    """Count before truncating. The cap is a payload bound, not an answer to
    a narrower question, so `total_months` describes the window and
    `returned` describes the page."""
    specs = []
    for i in range(30):
        year = 2024 + (i // 12)
        month = (i % 12) + 1
        specs.append((year, month, -100.0, -1_000))
    out = _condense_insider_sentiment(_month_rows(*specs), "2024-01-01", "2026-06-30")

    assert out["total_months"] == 30, out["total_months"]
    assert out["returned"] == len(out["months"]) < 30
    assert out["truncated"] is True
    assert out["window_returned"].endswith("2026-06"), out["window_returned"]


def test_an_empty_window_is_still_not_a_verdict():
    """Regression on the rule this file sits next to: with no months there is
    no average and no signal, and the payload must stay content-free so
    `build_envelope` can label the coverage. The added window fields are
    lookup echoes and must not resurrect it as an answer."""
    from tools.news_agregator.finnhub_utils import build_envelope

    out = _condense_insider_sentiment(_month_rows((2020, 1, -100.0, -1)),
                                      "2026-01-01", "2026-06-30")
    assert out["months"] == [] and out["signal"] is None and out["avg_mspr"] is None

    envelope = build_envelope(out, "AMAT", "get_insider_sentiment")
    assert envelope.get("coverage") == "not_covered", (
        "a window we could not fill was labelled as a real empty answer: "
        f"{envelope.get('coverage')!r}")


# ---------------------------------------------------------------------------
# 2. recent_30d must not report zeros for days the data does not reach
# ---------------------------------------------------------------------------

def _txn(days_ago: int, shares: int, code: str = "S"):
    date = (datetime.now(timezone.utc).date() - timedelta(days=days_ago)).isoformat()
    return {"name": "EXEC", "change": -shares if code in ("S", "F") else shares,
            "share": shares, "transactionCode": code, "transactionDate": date}


def test_a_recency_bucket_the_data_cannot_reach_is_null_not_zero():
    """AMAT, live: the feed ends 56 days ago, so nothing can fall in the last
    30 days. `{bought: 0, sold: 0, net: 0}` is a claim about insider
    behaviour; the truth is that the bucket has no coverage at all. A screen
    reading `recent_30d.net == 0` as "quiet month" was reading our own gap."""
    out = _condense_insider_data({"data": [_txn(56, 100_000), _txn(120, 50_000)]})

    assert out["recent_30d"] is None, (
        "the 30-day window ends 56 days after the last transaction we hold, "
        f"so it measures nothing: {out['recent_30d']}")
    assert out["recent_90d"] is not None, "the 90-day window does overlap the data"


def test_a_partly_covered_bucket_names_the_days_it_actually_holds():
    """The 90-day bucket above spans 90 days and holds 34 of them. Keeping the
    figure is right -- it is real -- but calling it "the last 90 days"
    without saying where the data stops overstates the window by a factor
    of nearly three."""
    out = _condense_insider_data({"data": [_txn(56, 100_000), _txn(120, 50_000)]})

    bucket = out["recent_90d"]
    assert "window" in bucket, f"no window on a partly covered bucket: {bucket}"
    assert bucket["window"].endswith(out["period_end"]), (
        f"the covered window must stop where the data stops: {bucket['window']} "
        f"vs period_end {out['period_end']}")


def test_the_lag_behind_today_is_stated_so_the_buckets_can_be_judged():
    """The buckets are anchored on today; the data is not. Naming the anchor
    and the gap is what lets a reader tell "no insider traded last month"
    from "our feed stops 56 days short of last month" -- and the two
    fixtures below have to be distinguishable, or the declaration is noise
    printed on every response."""
    stale = _condense_insider_data({"data": [_txn(56, 100_000), _txn(120, 50_000)]})
    fresh = _condense_insider_data({"data": [_txn(5, 10_000), _txn(200, 10_000)]})

    assert stale["as_of"] == datetime.now(timezone.utc).date().isoformat()
    assert stale["data_lag_days"] == 56, stale["data_lag_days"]
    assert stale["recent_30d"] is None

    assert fresh["data_lag_days"] == 5, fresh["data_lag_days"]
    assert fresh["recent_30d"] is not None
    assert fresh["recent_30d"]["window"].endswith(fresh["period_end"])


def test_the_existing_totals_are_untouched():
    """Regression: the coverage work must not move a single share count."""
    out = _condense_insider_data({"data": [_txn(56, 100_000), _txn(120, 50_000)]})
    assert out["total_sold"] == 150_000
    assert out["total_bought"] == 0
    assert out["net_shares"] == -150_000
    assert out["sell_count"] == 2


# ---------------------------------------------------------------------------
# 3. Live: the same rule against Finnhub itself
# ---------------------------------------------------------------------------

@requires_finnhub
async def test_live_narrowing_the_window_narrows_the_answer():
    """The reproduction, run against the real endpoint. Two windows, two
    answers, and each response naming what it covered."""
    from tools.news_agregator.finnhub_server import FinnhubServer

    server = FinnhubServer()
    try:
        wide = json.loads((await server.get_insider_sentiment(
            "AMAT", from_date="2025-01-01", to_date="2026-08-31"))[0].text)
        narrow = json.loads((await server.get_insider_sentiment(
            "AMAT", from_date="2026-03-01", to_date="2026-05-31"))[0].text)
    finally:
        await server.client.close()

    wide_months = wide["data"]["months"]
    narrow_months = narrow["data"]["months"]
    if not wide_months:
        pytest.skip("Finnhub returned no insider sentiment for AMAT at all")

    assert len(narrow_months) < len(wide_months), (
        f"a three-month request returned {len(narrow_months)} months and a "
        f"twenty-month request returned {len(wide_months)}")
    for month in narrow_months:
        assert (month["year"], month["month"]) >= (2026, 3)
        assert (month["year"], month["month"]) <= (2026, 5)
    assert narrow["data"]["window_requested"] == "2026-03-01 -> 2026-05-31"
    assert narrow["data"]["window_returned"] is not None
