"""Queries over the ingested disclosures.

Every answer carries the coverage it was computed from. An empty result from
a store holding four of five hundred filings is not a finding, and the only
thing that distinguishes it from a real absence is the coverage beside it.

Amounts stay brackets throughout. Aggregating them gives a bracketed total --
the sum of the lower bounds and the sum of the upper bounds -- and never a
midpoint, because a midpoint would turn "somewhere between $16k and $315k"
into a figure that reads as measured.
"""
import pytest

from tools.altdata_server import congress_store as store
from tools.altdata_server import congress_queries as q


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()

    pelosi = store.member_id("house", "Pelosi", "Nancy", "CA")
    tuberville = store.member_id("senate", "Tuberville", "Tommy")
    for mid, chamber, last, first in (
            (pelosi, "house", "Pelosi", "Nancy"),
            (tuberville, "senate", "Tuberville", "Tommy")):
        store.upsert_member({"member_id": mid, "chamber": chamber, "last": last,
                             "first": first, "full_name": f"{first} {last}",
                             "state": "CA", "district": None, "office": None,
                             "first_seen": "2026-01-01", "last_seen": "2026-06-01"})

    store.upsert_filing({"filing_id": "house:1", "chamber": "house", "doc_id": "1",
                         "member_id": pelosi, "filing_type": "ptr",
                         "filed_date": "2026-02-01", "year": 2026,
                         "parse_status": "parsed"})
    store.upsert_filing({"filing_id": "senate:2", "chamber": "senate",
                         "doc_id": "2", "member_id": tuberville,
                         "filing_type": "ptr", "filed_date": "2026-03-01",
                         "year": 2026, "parse_status": "parsed"})
    # One unreadable filing, so coverage is genuinely incomplete.
    store.upsert_filing({"filing_id": "house:3", "chamber": "house", "doc_id": "3",
                         "member_id": pelosi, "filing_type": "ptr",
                         "filed_date": "2026-04-01", "year": 2026,
                         "parse_status": "scanned",
                         "parse_error": "no extractable text"})

    store.replace_transactions("house:1", pelosi, [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "spouse",
         "transaction_type": "purchase", "transaction_date": "2026-01-15",
         "amount_min": 1_000_001, "amount_max": 5_000_000},
        {"ticker": "AAPL", "asset_name": "Apple Inc", "owner": "spouse",
         "transaction_type": "sale_full", "transaction_date": "2026-01-20",
         "amount_min": 15_001, "amount_max": 50_000},
    ])
    store.replace_transactions("senate:2", tuberville, [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "sale_partial", "transaction_date": "2026-02-10",
         "amount_min": 15_001, "amount_max": 50_000},
    ])
    return {"pelosi": pelosi, "tuberville": tuberville}


def test_a_ticker_query_finds_every_chamber(seeded):
    result = q.ticker_activity("NVDA")

    assert result["ticker"] == "NVDA"
    assert result["transaction_count"] == 2
    chambers = {t["chamber"] for t in result["transactions"]}
    assert chambers == {"house", "senate"}


def test_a_ticker_query_reports_its_coverage(seeded):
    """The scanned filing means this answer is not the whole record."""
    result = q.ticker_activity("NVDA")

    assert result["coverage"]["complete"] is False
    assert result["coverage"]["unparsed"] == 1
    assert "not the complete record" in result["note"]


def test_an_empty_result_is_not_presented_as_an_absence(seeded):
    result = q.ticker_activity("TSLA")

    assert result["transaction_count"] == 0
    assert result["coverage"]["complete"] is False
    assert "not the complete record" in result["note"], (
        "zero trades from an incomplete store reads as a finding about TSLA")


def test_totals_stay_bracketed(seeded):
    """Summing brackets gives a bracket, never a point."""
    result = q.ticker_activity("NVDA")
    totals = result["totals"]

    assert totals["amount_min_total"] == 1_015_002
    assert totals["amount_max_total"] == 5_050_000
    assert "amount" not in totals and "estimate" not in totals
    assert "midpoint" not in str(totals).lower()


def test_buy_and_sell_are_counted_separately(seeded):
    result = q.ticker_activity("NVDA")

    assert result["totals"]["purchase_count"] == 1
    assert result["totals"]["sale_count"] == 1


def test_a_member_query_returns_their_activity(seeded):
    result = q.member_activity("Pelosi")

    assert result["matched_members"], "no member matched 'Pelosi'"
    assert result["transaction_count"] == 2
    assert {t["ticker"] for t in result["transactions"]} == {"NVDA", "AAPL"}


def test_a_member_query_matches_loosely_but_reports_who_matched(seeded):
    """A loose match must say who it hit, or the numbers are unattributable."""
    result = q.member_activity("pelo")
    assert len(result["matched_members"]) == 1
    assert result["matched_members"][0]["full_name"] == "Nancy Pelosi"


def test_an_unmatched_member_says_so_rather_than_returning_nothing(seeded):
    result = q.member_activity("Nonexistent")

    assert result["matched_members"] == []
    assert result["transaction_count"] == 0
    assert "matched no member" in result["note"]


def test_the_leaderboard_ranks_by_activity(seeded):
    result = q.most_traded_tickers(limit=5)
    top = result["tickers"][0]

    assert top["ticker"] == "NVDA"
    assert top["transaction_count"] == 2
    assert top["member_count"] == 2


def test_the_leaderboard_excludes_rows_with_no_ticker(seeded):
    """Bonds and funds have no ticker; they must not become a 'None' row."""
    store.replace_transactions("house:1", seeded["pelosi"], [
        {"ticker": None, "asset_name": "US Treasury Bill", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-01-15",
         "amount_min": 1001, "amount_max": 15000}])

    result = q.most_traded_tickers(limit=10)
    assert all(t["ticker"] for t in result["tickers"])


def test_a_date_filter_bounds_the_window(seeded):
    result = q.ticker_activity("NVDA", since="2026-02-01")

    assert result["transaction_count"] == 1
    assert result["transactions"][0]["chamber"] == "senate"
