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


# ---------------------------------------------------------------- holdings

@pytest.fixture
def with_holdings(seeded):
    """Annual-report holdings alongside the trades already seeded."""
    store.upsert_filing({"filing_id": "senate:annual:u1", "chamber": "senate",
                         "doc_id": "u1", "member_id": seeded["tuberville"],
                         "filing_type": "annual", "filed_date": "2026-05-15",
                         "year": 2025, "parse_status": "parsed"})
    store.replace_holdings("senate:annual:u1", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "value_min": 15001, "value_max": 50000, "as_of": "2025-12-31"},
        {"ticker": "QQQ", "asset_name": "Invesco QQQ", "owner": "joint",
         "value_min": 100001, "value_max": 250000, "as_of": "2025-12-31"},
        # Unascertainable: bounds unknown, which is not a value of zero.
        {"ticker": None, "asset_name": "State Pension", "owner": "self",
         "value_min": None, "value_max": None, "as_of": "2025-12-31"},
    ])
    return seeded


def test_holdings_are_reported_with_their_snapshot_date(with_holdings):
    result = q.ticker_holdings("NVDA")

    assert result["holding_count"] == 1
    assert result["holdings"][0]["as_of"] == "2025-12-31"
    assert result["holdings"][0]["value_min"] == 15001


def test_holdings_say_they_are_not_current_positions(with_holdings):
    """The single most important caveat on this dataset."""
    result = q.ticker_holdings("NVDA")
    note = result["note"].lower()

    assert "not a current position" in note or "during" in note, (
        "an annual disclosure presented without its staleness reads as a "
        "live portfolio")


def test_a_holding_with_unknown_bounds_is_not_summed_as_zero(with_holdings):
    result = q.member_holdings("Tuberville")

    assert result["holding_count"] == 3
    assert result["totals"]["unpriced_count"] == 1, (
        "an unascertainable holding was folded into the totals as zero")
    assert result["totals"]["value_min_total"] == 115_002


def test_member_holdings_and_trades_are_kept_apart(with_holdings):
    """A disclosed holding and a disclosed trade are different claims."""
    holdings = q.member_holdings("Tuberville")
    trades = q.member_activity("Tuberville")

    assert holdings["holding_count"] == 3
    assert trades["transaction_count"] == 1
    assert "holdings" in holdings and "transactions" not in holdings
