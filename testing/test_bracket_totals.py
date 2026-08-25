"""A bracketed total must stay a bracket.

Live, `get_congress_holdings(ticker="NVDA")` returned

    value_min_total: 1,600,002
    value_max_total: 1,250,000

a minimum above its own maximum. One of the three holdings was open-ended --
"Over $1,000,000 and held independently by spouse or dependent child" has a
floor and no ceiling -- so its lower bound was added to the minimum while
nothing was added to the maximum, and the total quietly inverted.

The disclosure has no upper bound once an open-ended row is in the sum, so
neither does the total. Reporting a smaller number instead understates the
largest positions in the record and, worse, produces a range that cannot be
read as a range at all.
"""
import pytest

from tools.altdata_server import congress_queries as q


def test_an_open_ended_holding_leaves_the_total_open_ended():
    holdings = [
        {"value_min": 100_001, "value_max": 250_000},
        {"value_min": 500_001, "value_max": 1_000_000},
        {"value_min": 1_000_000, "value_max": None},   # "Over $1,000,000"
    ]
    totals = q._holding_totals(holdings)

    assert totals["value_min_total"] == 1_600_002
    assert totals["value_max_total"] is None, (
        "the sum was closed at a figure below its own minimum")
    assert totals["open_ended_count"] == 1


def test_a_closed_set_still_sums_to_a_pair():
    holdings = [{"value_min": 1_001, "value_max": 15_000},
                {"value_min": 15_001, "value_max": 50_000}]
    totals = q._holding_totals(holdings)

    assert totals["value_min_total"] == 16_002
    assert totals["value_max_total"] == 65_000
    assert totals["open_ended_count"] == 0


def test_a_total_never_inverts():
    """The invariant the live bug broke."""
    for holdings in (
        [{"value_min": 50_000_000, "value_max": None}],
        [{"value_min": 1, "value_max": 2}, {"value_min": 9_000_000, "value_max": None}],
        [{"value_min": None, "value_max": None}],
    ):
        totals = q._holding_totals(holdings)
        low, high = totals["value_min_total"], totals["value_max_total"]
        assert high is None or high >= low, f"{low} > {high} for {holdings}"


def test_transactions_carry_the_same_guarantee():
    """"$50,000,000 +" is open-ended on the trade side too."""
    transactions = [
        {"amount_min": 1_001, "amount_max": 15_000, "transaction_type": "purchase"},
        {"amount_min": 50_000_000, "amount_max": None, "transaction_type": "sale"},
    ]
    totals = q._totals(transactions)

    assert totals["amount_min_total"] == 50_001_001
    assert totals["amount_max_total"] is None
    assert totals["open_ended_count"] == 1


def test_an_unpriced_holding_is_still_excluded_from_both_bounds():
    holdings = [{"value_min": 1_001, "value_max": 15_000},
                {"value_min": None, "value_max": None}]   # Unascertainable
    totals = q._holding_totals(holdings)

    assert totals["value_min_total"] == 1_001
    assert totals["value_max_total"] == 15_000
    assert totals["unpriced_count"] == 1


# ----------------------------------------------------------- SQL aggregates

@pytest.fixture
def store_with_open_ended(tmp_path, monkeypatch):
    from tools.altdata_server import congress_store as store
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "b.db"))
    store.init_schema()
    mid = store.member_id("senate", "Rich", "Ada", "XX")
    store.upsert_member({"member_id": mid, "chamber": "senate", "last": "Rich",
                         "first": "Ada", "full_name": "Ada Rich", "state": "XX",
                         "district": None, "office": None,
                         "first_seen": "2026-01-01", "last_seen": "2026-01-01"})
    store.upsert_filing({"filing_id": "f1", "chamber": "senate", "doc_id": "1",
                         "member_id": mid, "filing_type": "ptr",
                         "filed_date": "2026-01-01", "year": 2026,
                         "parse_status": "parsed"})
    store.replace_transactions("f1", mid, [
        {"ticker": "BRK.A", "asset_name": "Berkshire", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2025-06-01",
         "amount_min": 50_000_000, "amount_max": None},
        {"ticker": "BRK.A", "asset_name": "Berkshire", "owner": "self",
         "transaction_type": "sale", "transaction_date": "2025-07-01",
         "amount_min": 1_001, "amount_max": 15_000},
    ])
    return mid


def test_the_ticker_leaderboard_does_not_close_an_open_bracket(store_with_open_ended):
    row = q.most_traded_tickers()["tickers"][0]

    assert row["ticker"] == "BRK.A"
    assert row["amount_min_total"] == 50_001_001
    assert row["amount_max_total"] is None, (
        f"summed an open-ended trade as zero and reported "
        f"{row['amount_max_total']}, below its own minimum")
    assert row["open_ended_count"] == 1


def test_the_member_leaderboard_does_not_close_an_open_bracket(store_with_open_ended):
    row = q.most_active_members()["members"][0]

    assert row["amount_min_total"] == 50_001_001
    assert row["amount_max_total"] is None
