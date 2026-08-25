"""A row limit must not be mistaken for the size of the data.

`get_congress_holdings(member="Scott")` reported Rick Scott with 199 holdings
and Tim Scott with 1. Tim Scott has 213. The query takes the top `limit` rows
across every matched member ordered by value, so Rick's larger positions
consumed 199 of the 200 slots and Tim was left with the remainder -- and the
per-member breakdown, computed from that truncated list, turned a paging
artefact into a statement about a senator's portfolio.

The counts and totals therefore come from an aggregate over every matching
row, independent of how many rows are returned, and a response that hit its
limit says so.
"""
import pytest

from tools.altdata_server import congress_store as store
from tools.altdata_server import congress_queries as q


@pytest.fixture
def two_members(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "t.db"))
    store.init_schema()
    big = store.member_id("senate", "Scott", "Rick", "FL")
    small = store.member_id("senate", "Scott", "Tim", "SC")
    for mid, first, full, n, value in ((big, "Rick", "Rick Scott", 40, 500_000),
                                       (small, "Tim", "Tim Scott", 10, 1_000)):
        store.upsert_member({"member_id": mid, "chamber": "senate", "last": "Scott",
                             "first": first, "full_name": full, "state": "XX",
                             "district": None, "office": None,
                             "first_seen": "2026-01-01", "last_seen": "2026-01-01"})
        store.upsert_filing({"filing_id": f"f:{mid}", "chamber": "senate",
                             "doc_id": mid, "member_id": mid,
                             "filing_type": "annual", "filed_date": "2026-05-15",
                             "year": 2025, "parse_status": "parsed"})
        store.replace_holdings(f"f:{mid}", mid, [
            {"ticker": f"T{i}", "asset_name": f"Asset {i}", "owner": "self",
             "value_min": value, "value_max": value * 2, "as_of": "2025-12-31"}
            for i in range(n)])
        store.replace_transactions(f"f:{mid}", mid, [
            {"ticker": f"T{i}", "asset_name": f"Asset {i}", "owner": "self",
             "transaction_type": "purchase", "transaction_date": "2025-06-01",
             "amount_min": value, "amount_max": value * 2} for i in range(n)])
    return {"big": "Rick Scott", "small": "Tim Scott"}


def test_per_member_counts_survive_a_row_limit(two_members):
    """The bug: the smaller holder was crowded out and reported as tiny."""
    result = q.member_holdings("Scott", limit=20)
    per = {p["member"]: p for p in result["per_member"]}

    assert per["Rick Scott"]["count"] == 40
    assert per["Tim Scott"]["count"] == 10, (
        f"reported {per['Tim Scott']['count']}; the limit took the rows, not "
        f"the senator")


def test_per_member_totals_are_not_computed_from_the_visible_rows(two_members):
    result = q.member_holdings("Scott", limit=5)
    per = {p["member"]: p for p in result["per_member"]}

    assert per["Tim Scott"]["totals"]["value_min_total"] == 10_000
    assert per["Rick Scott"]["totals"]["value_min_total"] == 20_000_000


def test_a_truncated_response_says_so(two_members):
    result = q.member_holdings("Scott", limit=20)

    assert result["truncated"] is True
    assert result["holding_count"] == 50, (
        "holding_count reported the rows returned rather than the rows that "
        "matched")
    assert len(result["holdings"]) == 20
    assert "truncated" in result["note"].lower()


def test_an_untruncated_response_is_not_flagged(two_members):
    result = q.member_holdings("Scott", limit=500)

    assert result["truncated"] is False
    assert result["holding_count"] == 50
    assert len(result["holdings"]) == 50


def test_trades_carry_the_same_guarantee(two_members):
    result = q.member_activity("Scott", limit=20)
    per = {p["member"]: p for p in result["per_member"]}

    assert result["truncated"] is True
    assert result["transaction_count"] == 50
    assert per["Tim Scott"]["count"] == 10
    assert per["Rick Scott"]["count"] == 40


def test_ticker_queries_report_truncation_too(two_members):
    result = q.ticker_activity("T1", limit=1)
    assert result["truncated"] is True
    assert result["transaction_count"] == 2, "counted only the row it returned"
