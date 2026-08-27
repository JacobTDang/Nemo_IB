"""Asking for every trade must not answer "none".

`get_congress_trades` routes to `ticker_activity`, which unconditionally adds
`WHERE t.ticker = ?`. Called with no ticker the dispatcher passes `""`, so the
clause becomes `t.ticker = ''` and matches nothing. Every unfiltered query --
`{}`, `{"limit": 200}`, `{"chamber": "senate"}`, `{"since": "2020-01-01"}` --
returned:

    success: true, truncated: false, transaction_count: 0, rows_returned: 0,
    member_count: 0, totals all zero

over a store holding 16,518 transactions. "What did the Senate trade this
quarter?" answered "nothing", and `truncated: false` asserted that the empty
set was the whole set.

It is contradicted by the store itself: `{"ticker": "NVDA"}` returns Senate
rows, and the leaderboard reports one member with 1,465 transactions. Only the
`ticker` and `member` paths ever worked.

A zero here is the shape this pipeline exists to prevent -- an absence
presented as a finding -- and it is the most natural question to ask of a
congressional trading tool.
"""
import pytest

from tools.altdata_server import congress_queries as q
from tools.altdata_server import congress_store as store


@pytest.fixture
def populated(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    for i, (chamber, ticker, date) in enumerate([
        ("house", "NVDA", "2026-01-15"),
        ("house", "AAPL", "2026-02-20"),
        ("senate", "MSFT", "2026-03-10"),
        ("senate", "NVDA", "2026-04-05"),
    ]):
        fid = f"f{i}"
        store.upsert_filing({"filing_id": fid, "chamber": chamber,
                             "doc_id": str(i), "member_id": f"{chamber}:m{i}",
                             "filing_type": "P", "raw_filing_type": "P",
                             "filed_date": "2026-05-01", "year": 2026,
                             "source_url": "http://example.invalid"})
        store.replace_transactions(fid, f"{chamber}:m{i}", [{
            "ticker": ticker, "transaction_type": "purchase",
            "transaction_date": date, "amount_min": 1001, "amount_max": 15000,
        }])
    return store


def test_an_unfiltered_query_returns_every_trade(populated):
    result = q.ticker_activity("")
    assert result["transaction_count"] == 4, (
        f"asked for every trade over a populated store, got "
        f"{result['transaction_count']}")
    assert result["rows_returned"] == 4


def test_a_chamber_filter_alone_works(populated):
    result = q.ticker_activity("", chamber="senate")
    assert result["transaction_count"] == 2
    assert all(t["chamber"] == "senate" for t in result["transactions"])


def test_a_date_filter_alone_works(populated):
    result = q.ticker_activity("", since="2026-03-01")
    assert result["transaction_count"] == 2


def test_a_ticker_filter_still_narrows(populated):
    result = q.ticker_activity("NVDA")
    assert result["transaction_count"] == 2
    assert {t["ticker"] for t in result["transactions"]} == {"NVDA"}


def test_the_unfiltered_totals_cover_every_row(populated):
    """The totals must describe the set, not the page -- the property the
    ticker path already guarantees."""
    result = q.ticker_activity("", limit=1)
    assert result["transaction_count"] == 4
    assert result["rows_returned"] == 1
    assert result["truncated"] is True
    assert result["totals"]["amount_min_total"] == 4 * 1001


def test_an_unfiltered_query_does_not_claim_a_ticker(populated):
    """`ticker: ""` in the echo reads as a filter that was applied."""
    assert q.ticker_activity("")["ticker"] is None
