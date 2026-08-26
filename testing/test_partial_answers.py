"""A partial answer must never be presented as a complete one.

Three live instances on nemo_altdata, one shape of defect: a number describing
a subset sat unlabelled beside a number describing the whole, or an empty
result caused by not looking was reported as a finding.

1. `get_congress_trades(ticker="NVDA", limit=10)` returned
   `transaction_count: 226` (the full matching set) next to `member_count: 7`
   and `totals.amount_min_total: 137,010` (the ten rows that fitted on the
   page). At `limit=300` the same query gave 42 members and $12,963,226 --
   the dollars were understated by roughly 98x, and nothing in the response
   said which of the adjacent numbers described what.

2. `get_government_contracts(ticker="LMT", months=3)` returned
   `trailing=$0.0B, yoy_change_pct=-99.9, signal="bearish", coverage="full"`.
   The window lay inside USAspending's reporting lag, so it was a gap in what
   has been published rather than a collapse in Lockheed's federal business --
   and `yoy_change_pct` was not a year-over-year figure at all, it compared
   the trailing months against the immediately preceding months of a series
   whose largest month of the year is September.

3. `get_taiwan_monthly_revenue` returned `{"year": 2026, "month": 6,
   "date": "2026-07-01"}`. `date` was FinMind's announcement month, one ahead
   of the period the revenue describes, so anything charted on `date`
   attributed every month's revenue to the following month.

Run:
  SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_partial_answers.py -q
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.altdata_server import congress_queries as q
from tools.altdata_server import congress_store as store
from tools.altdata_server import server as alt


# ===========================================================================
# 1. Congressional queries -- page-scoped totals beside full-set counts
# ===========================================================================

@pytest.fixture
def crowded(tmp_path, monkeypatch):
    """Two members, forty NVDA rows between them, one of them open-ended.

    Sized so that any limit below forty truncates, which is the condition the
    live bug needed: the page is a subset and the numbers beside it are not.
    """
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "p.db"))
    store.init_schema()

    whales = store.member_id("senate", "Whale", "Wanda", "FL")
    minnow = store.member_id("house", "Minnow", "Milton", "SC")
    for mid, chamber, last, first, full in (
            (whales, "senate", "Whale", "Wanda", "Wanda Whale"),
            (minnow, "house", "Minnow", "Milton", "Milton Minnow")):
        store.upsert_member({"member_id": mid, "chamber": chamber, "last": last,
                             "first": first, "full_name": full, "state": "XX",
                             "district": None, "office": None,
                             "first_seen": "2026-01-01", "last_seen": "2026-06-01"})
        store.upsert_filing({"filing_id": f"f:{mid}", "chamber": chamber,
                             "doc_id": mid, "member_id": mid,
                             "filing_type": "ptr", "filed_date": "2026-06-01",
                             "year": 2026, "parse_status": "parsed"})

    # Wanda: 30 NVDA purchases of 15,001-50,000, newest first so they win the
    # ORDER BY and crowd Milton off any short page.
    store.replace_transactions(f"f:{whales}", whales, [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": f"2026-06-{d:02d}",
         "amount_min": 15_001, "amount_max": 50_000}
        for d in range(1, 31)])
    # Milton: 9 NVDA sales plus one open-ended row, all dated earlier.
    store.replace_transactions(f"f:{minnow}", minnow, [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "spouse",
         "transaction_type": "sale_full", "transaction_date": f"2026-01-{d:02d}",
         "amount_min": 1_001, "amount_max": 15_000}
        for d in range(1, 10)] + [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "spouse",
         "transaction_type": "sale_partial", "transaction_date": "2026-01-10",
         "amount_min": 50_000_000, "amount_max": None}])

    store.replace_holdings(f"f:{whales}", whales, [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "value_min": 500_001, "value_max": 1_000_000, "as_of": "2025-12-31"}
        for _ in range(30)])
    store.replace_holdings(f"f:{minnow}", minnow, [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "spouse",
         "value_min": 1_001, "value_max": 15_000, "as_of": "2025-12-31"}
        for _ in range(9)] + [
        # Priced by nobody: a state pension carries no bracket at all and is
        # counted rather than summed as though it were worth zero.
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "spouse",
         "value_min": None, "value_max": None, "as_of": "2025-12-31"}])
    return {"whale": "Wanda Whale", "minnow": "Milton Minnow"}


def test_ticker_totals_do_not_move_when_the_row_limit_does(crowded):
    """The live defect, at the size that produced it.

    `transaction_count` counted every matching row while `totals` counted the
    page, and the two sat adjacent with nothing to tell them apart. Whatever
    `limit` a caller passes, the totals describe the same set of trades.
    """
    small = q.ticker_activity("NVDA", limit=5)
    full = q.ticker_activity("NVDA", limit=500)

    assert small["transaction_count"] == full["transaction_count"] == 40
    assert small["truncated"] is True
    assert small["totals"] == full["totals"], (
        "the totals shrank with the page: a subset reported as the whole")


def test_ticker_member_count_counts_members_not_the_ones_that_fitted(crowded):
    """7 of 42 members, printed as the member count, is a paging artefact.

    Wanda's thirty rows take the whole of a short page, so a page-scoped count
    sees one member where two traded.
    """
    small = q.ticker_activity("NVDA", limit=5)
    full = q.ticker_activity("NVDA", limit=500)

    assert small["member_count"] == full["member_count"] == 2


def test_ticker_totals_over_the_full_set_stay_bracketed(crowded):
    """The open-ended rule has to survive the move into SQL.

    "Over $50,000,000" has a floor and no ceiling. SUM(COALESCE(max, 0))
    closes it at zero, which is how a maximum ended up below its own minimum
    once before. The ceiling is dropped instead.
    """
    totals = q.ticker_activity("NVDA", limit=500)["totals"]

    assert totals["amount_min_total"] == 30 * 15_001 + 9 * 1_001 + 50_000_000
    assert totals["amount_max_total"] is None
    assert totals["open_ended_count"] == 1
    assert totals["purchase_count"] == 30
    assert totals["sale_count"] == 10


def test_ticker_totals_close_again_when_every_row_has_a_ceiling(crowded):
    """A bounded set still gets a real pair of numbers, not a blanket None."""
    totals = q.ticker_activity("NVDA", chamber="senate", limit=500)["totals"]

    assert totals["amount_min_total"] == 30 * 15_001
    assert totals["amount_max_total"] == 30 * 50_000
    assert totals["open_ended_count"] == 0


def test_member_trade_totals_do_not_move_when_the_row_limit_does(crowded):
    """`per_member` was already fixed to aggregate; the top-level `totals`
    beside it was still the page, so the two disagreed inside one response."""
    small = q.member_activity("Whale", limit=5)
    full = q.member_activity("Whale", limit=500)

    assert small["transaction_count"] == full["transaction_count"] == 30
    assert small["totals"] == full["totals"]
    assert small["totals"]["amount_min_total"] == 30 * 15_001


def test_member_trade_totals_agree_with_the_per_member_breakdown(crowded):
    """One response must not carry two different answers to one question."""
    result = q.member_activity("Whale", limit=3)
    per = result["per_member"][0]["totals"]

    assert result["totals"]["amount_min_total"] == per["amount_min_total"]


def test_ticker_holdings_totals_and_member_count_survive_the_limit(crowded):
    small = q.ticker_holdings("NVDA", limit=5)
    full = q.ticker_holdings("NVDA", limit=500)

    assert small["holding_count"] == full["holding_count"] == 40
    assert small["member_count"] == full["member_count"] == 2
    assert small["totals"] == full["totals"]


def test_holding_totals_over_the_full_set_keep_unpriced_rows_out(crowded):
    """A holding nobody could price is not a holding worth nothing.

    Summing it as zero would drag the floor down and hide it; it is counted in
    unpriced_count and left out of both bounds.
    """
    totals = q.ticker_holdings("NVDA", limit=500)["totals"]

    assert totals["value_min_total"] == 30 * 500_001 + 9 * 1_001
    assert totals["value_max_total"] == 30 * 1_000_000 + 9 * 15_000
    assert totals["priced_count"] == 39
    assert totals["unpriced_count"] == 1


def test_member_holdings_totals_survive_the_limit(crowded):
    small = q.member_holdings("Minnow", limit=3)
    full = q.member_holdings("Minnow", limit=500)

    assert small["holding_count"] == full["holding_count"] == 10
    assert small["totals"] == full["totals"]
    assert small["totals"]["unpriced_count"] == 1


def test_a_member_and_ticker_query_totals_the_rows_it_reports(crowded):
    """`get_congress_trades(member=..., ticker=...)` filtered the page after
    the fact and recounted it, leaving `totals` describing every trade the
    member made in anything. The ticker belongs in the query, not in a
    post-filter over whatever the page happened to hold."""
    result = q.member_activity("Minnow", ticker="NVDA", limit=500)

    assert result["transaction_count"] == 10
    assert result["totals"]["sale_count"] == 10
    assert result["totals"]["amount_min_total"] == 9 * 1_001 + 50_000_000
    assert result["totals"]["amount_max_total"] is None


# ===========================================================================
# 2. Government contracts -- a reporting-lag gap read as a collapse
# ===========================================================================

# Government-wide monthly contract obligations, USD, as measured against
# api.usaspending.gov on 2026-08-25. Data runs at a steady $50-100bn/month
# through 2026-05 and then falls off a cliff: agencies -- the Department of
# Defense above all -- publish contract actions months in arrears, so the
# most recent months are unpublished rather than empty.
_MEASURED_GOVERNMENT_WIDE = [
    {"month": "2025-06-01", "obligations_usd": 62_856_000_000.0},
    {"month": "2025-07-01", "obligations_usd": 71_848_000_000.0},
    {"month": "2025-08-01", "obligations_usd": 70_130_000_000.0},
    {"month": "2025-09-01", "obligations_usd": 148_504_000_000.0},
    {"month": "2025-10-01", "obligations_usd": 21_530_000_000.0},
    {"month": "2025-11-01", "obligations_usd": 50_652_000_000.0},
    {"month": "2025-12-01", "obligations_usd": 79_520_000_000.0},
    {"month": "2026-01-01", "obligations_usd": 55_709_000_000.0},
    {"month": "2026-02-01", "obligations_usd": 54_236_000_000.0},
    {"month": "2026-03-01", "obligations_usd": 102_381_000_000.0},
    {"month": "2026-04-01", "obligations_usd": 83_604_000_000.0},
    {"month": "2026-05-01", "obligations_usd": 59_145_000_000.0},
    {"month": "2026-06-01", "obligations_usd": 33_100_000_000.0},
    {"month": "2026-07-01", "obligations_usd": 39_920_000_000.0},
    {"month": "2026-08-01", "obligations_usd": 15_477_000_000.0},
]


def test_the_comparison_window_is_the_same_months_a_year_earlier():
    """`yoy_change_pct` compared the trailing months against the ones directly
    before them. At months=3 the "prior" figure was exactly the months=6
    trailing figure, which is how the substitution was caught.

    Federal awards peak hugely in September, the last month of the fiscal
    year: government-wide obligations for 2025-09 were $148bn against a $60bn
    median month. A sequential comparison over a series like that measures the
    calendar, not the company, and calling it year-over-year hides that.

    The comparison window stops a day short of a full year back because
    USASpending's time_period is inclusive at both ends; at months=12 the two
    windows touch, and the boundary day would otherwise be counted twice.
    """
    today = date(2026, 8, 25)
    t_start, t_end, c_start, c_end = alt._gov_windows(today, 3)

    assert (t_start, t_end) == ("2026-05-25", "2026-08-25")
    assert (c_start, c_end) == ("2025-05-25", "2025-08-24"), (
        "the comparison window was the preceding quarter, not the year-ago one")


def test_a_twelve_month_window_still_compares_against_the_year_before():
    today = date(2026, 8, 25)
    t_start, t_end, c_start, c_end = alt._gov_windows(today, 12)

    assert (t_start, t_end) == ("2025-08-25", "2026-08-25")
    assert (c_start, c_end) == ("2024-08-25", "2025-08-24")


def test_a_leap_day_window_does_not_fall_off_the_calendar():
    """29 February has no counterpart in the year before it, so a window
    anchored on one has to clamp rather than raise."""
    t_start, t_end, c_start, c_end = alt._gov_windows(date(2024, 2, 29), 12)

    assert t_end == "2024-02-29"
    assert t_start == "2023-02-28"
    assert c_start == "2022-02-28"


def test_the_data_horizon_is_measured_from_the_provider_not_assumed():
    """Where USAspending's published data actually stops, from its own series.

    Measured 2026-08-25: months at or above the completeness threshold run
    through 2026-05; 2026-06 ($33.1bn), 2026-07 ($39.9bn) and the part-month
    2026-08 ($15.5bn) sit far below a $59bn median. October 2025 is equally
    thin but is not at the end of the series, so it is a real lull rather than
    an unpublished month and does not move the horizon.
    """
    horizon = alt._data_horizon_from_series(_MEASURED_GOVERNMENT_WIDE)

    assert horizon["horizon"] == "2026-05-31"
    assert horizon["incomplete_months"] == ["2026-06", "2026-07", "2026-08"]


def test_a_fully_published_series_has_no_lag_at_all():
    """The detector must be capable of reporting "no gap", or it only ever
    manufactures caveats."""
    series = [{"month": f"2026-{m:02d}-01", "obligations_usd": 60e9}
              for m in range(1, 9)]

    horizon = alt._data_horizon_from_series(series)

    assert horizon["horizon"] == "2026-08-31"
    assert horizon["incomplete_months"] == []


def test_the_fraction_of_a_window_past_the_horizon_is_reported_as_a_number():
    """Nearly all of a 3-month window is unpublished; a quarter of a 12-month
    one is. The two deserve different treatment, so the overlap is measured
    rather than asserted."""
    assert alt._window_lag_fraction("2026-05-25", "2026-08-25", "2026-05-31") \
        == pytest.approx(0.93, abs=0.02)
    assert alt._window_lag_fraction("2025-08-25", "2026-08-25", "2026-05-31") \
        == pytest.approx(0.23, abs=0.02)
    assert alt._window_lag_fraction("2025-08-25", "2026-08-25", "2026-12-31") == 0.0


def _mock_usaspending(monkeypatch, *, trailing, comparison, count=0,
                      horizon="2026-05-31", horizon_error=None):
    """Patch the four USASpending queries and the freshness probe."""
    def obligations(company_name, start, end, award_types):
        return trailing if end == datetime.now().strftime("%Y-%m-%d") else comparison

    def horizon_probe(award_types):
        if horizon_error:
            raise requests.exceptions.HTTPError(horizon_error)
        return {"horizon": horizon, "incomplete_months": [],
                "baseline_usd": 59_145_000_000.0,
                "measured_from": "spending_over_time (government-wide)"}

    monkeypatch.setattr(alt, "_usa_obligations_total", obligations)
    monkeypatch.setattr(alt, "_usa_top_agencies", lambda *a: [])
    monkeypatch.setattr(alt, "_usa_award_count", lambda *a: count)
    monkeypatch.setattr(alt, "_usa_data_horizon", horizon_probe)


def _codes(result):
    return {w["code"] for w in result.get("warnings", [])}


def test_a_window_inside_the_lag_is_a_gap_and_not_a_bearish_signal(monkeypatch):
    """The live defect: LMT at months=3 read as a -99.9% collapse.

    Lockheed's federal business did not stop. The Department of Defense had
    not yet published the quarter. A window that is mostly unpublished cannot
    support a judgement in either direction, so it reports what is missing
    instead of what it appears to show.
    """
    _mock_usaspending(monkeypatch, trailing=40_000_000.0,
                      comparison=24_200_000_000.0, count=34)

    out = alt._fetch_government_contracts("LMT", "Lockheed Martin", 3, False)

    assert out["success"] is True
    assert out["coverage"] == "partial", "a gap in the record reported as full"
    assert out["signal"] is None, "judged a company on months nobody published"
    assert "reporting_lag" in _codes(out)
    assert out["data_horizon"] == "2026-05-31"
    assert out["window_lag_fraction"] > 0.5


def test_the_lag_warning_names_the_horizon_and_the_overlap(monkeypatch):
    """A caveat a caller cannot act on is decoration. The warning carries the
    date the data stops and how much of the window sits past it."""
    _mock_usaspending(monkeypatch, trailing=40_000_000.0,
                      comparison=24_200_000_000.0, count=34)

    out = alt._fetch_government_contracts("LMT", "Lockheed Martin", 3, False)
    lag = next(w for w in out["warnings"] if w["code"] == "reporting_lag")

    assert lag["data_horizon"] == "2026-05-31"
    assert lag["window_lag_fraction"] > 0.5
    assert "2026-05-31" in lag["message"]


def test_a_published_window_still_produces_a_confident_signal(monkeypatch):
    """The fix must not mute every answer. With the window inside published
    data, a real 99% fall is still a real finding."""
    _mock_usaspending(monkeypatch, trailing=40_000_000.0,
                      comparison=24_200_000_000.0, count=34,
                      horizon="2026-12-31")

    out = alt._fetch_government_contracts("LMT", "Lockheed Martin", 3, False)

    assert out["coverage"] == "full"
    assert out["signal"] == "bearish"
    assert out["window_lag_fraction"] == 0.0
    assert "reporting_lag" not in _codes(out)


def test_a_partly_unpublished_window_keeps_its_signal_but_says_what_is_missing(monkeypatch):
    """Three unpublished months out of twelve is not "mostly", so the trailing
    total is still worth reporting -- but it is compared against a year-ago
    window that IS complete, which biases the change downward. The caller is
    told, in the response, by how much."""
    _mock_usaspending(monkeypatch, trailing=45_000_000_000.0,
                      comparison=50_000_000_000.0, count=2696)

    out = alt._fetch_government_contracts("LMT", "Lockheed Martin", 12, False)

    assert out["coverage"] == "full"
    assert out["signal"] in {"bullish", "bearish", "neutral"}
    assert "partial_reporting_lag" in _codes(out)
    assert 0 < out["window_lag_fraction"] < 0.5


def test_an_unmeasurable_horizon_is_named_rather_than_assumed_absent(monkeypatch):
    """If the freshness probe fails we do not know whether the window is
    published. Reporting no lag would be the same silent guess this fix
    removes, so the unknown is stated and the horizon is null."""
    _mock_usaspending(monkeypatch, trailing=45_000_000_000.0,
                      comparison=50_000_000_000.0, count=2696,
                      horizon_error="503 Service Unavailable")

    out = alt._fetch_government_contracts("LMT", "Lockheed Martin", 12, False)

    assert out["success"] is True
    assert out["data_horizon"] is None
    assert out["window_lag_fraction"] is None
    assert any("horizon" in d or "freshness" in d for d in out["degraded"])


def test_a_company_with_no_federal_business_is_still_a_genuine_zero(monkeypatch):
    """Nothing about the lag fix may turn a real absence into a caveat: a
    burger chain has no federal contracts in either window, and that is a
    finding rather than a gap."""
    _mock_usaspending(monkeypatch, trailing=0.0, comparison=0.0, count=0)

    out = alt._fetch_government_contracts("MCD", "McDonald's Corporation", 12, False)

    assert out["success"] is True
    assert out["signal"] == "not_applicable"


# ===========================================================================
# 3. Taiwan monthly revenue -- a date one month ahead of its own figures
# ===========================================================================

def _finmind_rows():
    """FinMind's own shape: `date` is the month the figure was ANNOUNCED.

    TSMC announced June 2026 revenue on 2026-07-10 and FinMind stamps the row
    2026-07-01. NT$442.68bn against NT$263.7bn a year earlier is +67.9%, and
    NT$263.7bn was TSMC's June 2025 revenue -- which is what fixes the row to
    June, not July.
    """
    return [
        {"date": "2025-07-01", "revenue_year": 2025, "revenue_month": 6,
         "revenue": 263_700_000_000, "stock_id": "2330"},
        {"date": "2026-06-01", "revenue_year": 2026, "revenue_month": 5,
         "revenue": 320_520_000_000, "stock_id": "2330"},
        {"date": "2026-07-01", "revenue_year": 2026, "revenue_month": 6,
         "revenue": 442_680_000_000, "stock_id": "2330"},
    ]


@pytest.fixture
def finmind(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": 200, "msg": "success", "data": _finmind_rows()}

    monkeypatch.setattr(alt.requests if hasattr(alt, "requests") else requests,
                        "get", lambda *a, **k: _Resp(), raising=False)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    return None


def test_the_date_is_the_month_the_revenue_describes(finmind):
    """`date` sat between `year` and `month` disagreeing with both.

    A chart drawn on `date` put June's NT$442.68bn in July, every month, for
    every Taiwanese company on the tool -- and the error is invisible because
    the series still looks like a series.
    """
    out = alt._fetch_taiwan_revenue_finmind(["2330"], months=3)
    june = out["companies"]["2330"]["months"][-1]

    assert (june["year"], june["month"]) == (2026, 6)
    assert june["date"] == "2026-06-01", (
        "the announcement month was reported as the revenue month")
    assert june["revenue_ntd_m"] == 442_680.0


def test_the_announcement_month_is_kept_but_labelled(finmind):
    """FinMind's own stamp is real information -- when the figure became
    public. It stays, under a name that says what it is."""
    june = alt._fetch_taiwan_revenue_finmind(["2330"], months=3)["companies"]["2330"]["months"][-1]

    assert june["announced_date"] == "2026-07-01"


def test_no_row_disagrees_with_the_year_and_month_beside_it(finmind):
    months = alt._fetch_taiwan_revenue_finmind(["2330"], months=3)["companies"]["2330"]["months"]

    assert months, "fixture produced no rows"
    for row in months:
        assert row["date"] == f"{row['year']:04d}-{row['month']:02d}-01", row


def test_the_yoy_base_is_the_same_month_a_year_earlier(finmind):
    """The back-solve that identified the off-by-one in the first place."""
    june = alt._fetch_taiwan_revenue_finmind(["2330"], months=3)["companies"]["2330"]["months"][-1]

    assert june["yoy_pct"] == pytest.approx(67.87, abs=0.01)
