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


# ---------------------------------------------------- before the first sync

def test_a_query_against_a_never_synced_store_is_an_empty_answer(tmp_path,
                                                                 monkeypatch):
    """On a fresh volume the tables do not exist yet.

    The server surfaces the exception verbatim, so a new operator gets
    `OperationalError: no such table: filings` where the empty-store guidance
    was meant to be -- at precisely the moment it is needed.
    """
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "never-synced.db"))

    result = q.ticker_activity("NVDA")

    assert result["transaction_count"] == 0
    assert result["coverage"]["total"] == 0
    assert result["coverage"]["complete"] is False


def test_a_holdings_query_against_a_never_synced_store_is_empty_too(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "never-synced.db"))
    assert q.ticker_holdings("NVDA")["holding_count"] == 0
    assert q.most_traded_tickers()["tickers"] == []


# ------------------------------------------- a lost ceiling vs an absent one
#
# Two rows look identical in the store -- a floor with no ceiling -- and mean
# opposite things. "Over $50,000,000" and the EIGA spouse cap genuinely have
# no upper bound, so a total containing one has none either. Every other null
# ceiling is a parse failure: the filer disclosed a bracket and we lost its
# top. Those rows are deliberately produced by `_sane_transaction` and
# `repair_impossible_rows` when a filed ceiling sits below its own floor, and
# by `parse_amount_range` on a stray '+'.
#
# The narrow rule was written in `_open_ended` and then never reached: every
# live aggregate runs in SQL, where the test was simply "min not null and max
# null". So one repaired row erased `amount_max_total` for every result set it
# touched (issue #48).

def test_a_repaired_row_does_not_erase_the_ceiling_for_a_whole_result_set(seeded):
    """A mid-bracket floor with a lost ceiling is a parse failure, not a
    disclosure of an unbounded amount, and must not spend the whole set's
    ceiling."""
    store.replace_transactions("senate:2", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "sale_partial", "transaction_date": "2026-02-10",
         "amount_min": 15_001, "amount_max": 50_000},
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-11",
         "amount_min": 15_001, "amount_max": None},
    ])

    totals = q.ticker_activity("NVDA")["totals"]

    assert totals["amount_min_total"] == 1_030_003
    assert totals["amount_max_total"] == 5_050_000, (
        "one row whose ceiling was lost to a parse failure erased the ceiling "
        "of every other row in the set")
    assert totals["open_ended_count"] == 0, (
        "a mid-bracket floor was counted as an unbounded disclosure")
    assert totals["ceiling_unknown_count"] == 1


def test_a_genuinely_unbounded_bracket_still_removes_the_ceiling(seeded):
    """The other half of the rule: 'Over $50,000,000' has no top, so neither
    does any total containing it."""
    store.replace_transactions("senate:2", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-10",
         "amount_min": 50_000_000, "amount_max": None},
    ])

    totals = q.ticker_activity("NVDA")["totals"]

    assert totals["amount_max_total"] is None
    assert totals["open_ended_count"] == 1
    assert totals["ceiling_unknown_count"] == 0


def test_a_total_that_would_invert_still_reports_no_ceiling(seeded):
    """Counting a lost ceiling as zero can put a maximum below its own
    minimum. A range whose top is under its bottom is not a range."""
    store.replace_transactions("house:1", seeded["pelosi"], [])
    store.replace_transactions("senate:2", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-10",
         "amount_min": 15_001, "amount_max": None},
    ])

    totals = q.ticker_activity("NVDA")["totals"]

    assert totals["amount_min_total"] == 15_001
    assert totals["amount_max_total"] is None
    assert totals["ceiling_unknown_count"] == 1


def test_the_ticker_leaderboard_keeps_a_ceiling_a_repaired_row_did_not_remove(seeded):
    store.replace_transactions("senate:2", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-10",
         "amount_min": 15_001, "amount_max": None},
    ])

    nvda = next(t for t in q.most_traded_tickers(limit=10)["tickers"]
                if t["ticker"] == "NVDA")

    assert nvda["amount_max_total"] == 5_000_000
    assert nvda["open_ended_count"] == 0
    assert nvda["ceiling_unknown_count"] == 1


def test_the_member_leaderboard_keeps_a_ceiling_a_repaired_row_did_not_remove(seeded):
    store.replace_transactions("senate:2", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-10",
         "amount_min": 15_001, "amount_max": 50_000},
        {"ticker": "AAPL", "asset_name": "Apple Inc", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-11",
         "amount_min": 15_001, "amount_max": None},
    ])

    tommy = next(m for m in q.most_active_members(limit=10)["members"]
                 if m["member"] == "Tommy Tuberville")

    assert tommy["amount_max_total"] == 50_000
    assert tommy["ceiling_unknown_count"] == 1


def test_a_per_member_ceiling_survives_a_repaired_row(seeded):
    """The per-member breakout is computed from its own aggregate, so the rule
    has to hold there too."""
    store.replace_transactions("senate:2", seeded["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-10",
         "amount_min": 15_001, "amount_max": 50_000},
        {"ticker": "AAPL", "asset_name": "Apple Inc", "owner": "self",
         "transaction_type": "purchase", "transaction_date": "2026-02-11",
         "amount_min": 15_001, "amount_max": None},
    ])

    per_member = q.member_activity("Tuberville")["per_member"][0]["totals"]

    assert per_member["amount_max_total"] == 50_000
    assert per_member["ceiling_unknown_count"] == 1


def test_a_repaired_holding_does_not_erase_the_holdings_ceiling(with_holdings):
    """The same distinction on the holdings table, where the open-ended bands
    are the ones that actually occur."""
    store.replace_holdings("senate:annual:u1", with_holdings["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "self",
         "value_min": 100_001, "value_max": 250_000, "as_of": "2025-12-31"},
        {"ticker": "QQQ", "asset_name": "Invesco QQQ", "owner": "joint",
         "value_min": 15_001, "value_max": None, "as_of": "2025-12-31"},
    ])

    totals = q.member_holdings("Tuberville")["totals"]

    assert totals["value_min_total"] == 115_002
    assert totals["value_max_total"] == 250_000
    assert totals["open_ended_count"] == 0
    assert totals["ceiling_unknown_count"] == 1


def test_the_spouse_cap_is_still_an_unbounded_holding(with_holdings):
    """'Over $1,000,000' is a real band on the annual report and has no top."""
    store.replace_holdings("senate:annual:u1", with_holdings["tuberville"], [
        {"ticker": "NVDA", "asset_name": "Nvidia Corp", "owner": "spouse",
         "value_min": 1_000_000, "value_max": None, "as_of": "2025-12-31"},
    ])

    totals = q.member_holdings("Tuberville")["totals"]

    assert totals["value_max_total"] is None
    assert totals["open_ended_count"] == 1
    assert totals["ceiling_unknown_count"] == 0


def test_totals_are_shaped_the_same_whether_or_not_anything_matched(seeded):
    """The no-match default is built in Python and the real one in SQL. A
    caller reading `ceiling_unknown_count` must not find it missing simply
    because the name matched nobody."""
    matched = q.member_activity("Pelosi")["totals"]
    unmatched = q.member_activity("Nonexistent")["totals"]

    assert set(matched) == set(unmatched)


def test_holding_totals_are_shaped_the_same_whether_or_not_anything_matched(
        with_holdings):
    matched = q.member_holdings("Tuberville")["totals"]
    unmatched = q.member_holdings("Nonexistent")["totals"]

    assert set(matched) == set(unmatched)
