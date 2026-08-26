"""The store is the last place a false figure can be stopped.

Parsers get better and parsers get replaced. A row that cannot be true --
`amount_min` above `amount_max`, a trade dated after the filing that discloses
it -- should not depend on which parser produced it, because every parser
writes through here.

Both shapes were found in the live store:

    24 transactions with amount_min > amount_max, always amount_max = 200
     7 transactions dated after their own filing, one by ten months

The dates are not repaired, only refused. A trade on 2026-12-26 disclosed on
2026-02-09 is almost certainly December 2025, and "almost certainly" is how a
parsing bug becomes a fact in a database. An unreadable date stored as None
says what we know; a corrected one says more than we know.
"""
import pytest

from tools.altdata_server import congress_store as store


def test_an_inverted_amount_is_not_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    store.upsert_filing({"filing_id": "f1", "chamber": "house", "doc_id": "1",
                         "member_id": "house:x", "filing_type": "P",
                         "raw_filing_type": "P", "filed_date": "2026-02-09",
                         "year": 2026,
                         "source_url": "http://example.invalid"})

    store.replace_transactions("f1", "house:x", [{
        "ticker": "CSCO", "transaction_type": "sale",
        "transaction_date": "2024-08-06",
        "amount_min": 50001, "amount_max": 200,
    }])

    with store.connect() as conn:
        row = conn.execute(
            "SELECT amount_min, amount_max FROM transactions").fetchone()
    assert row[0] == 50001, "the floor was a real bracket bound and was dropped"
    assert row[1] is None, "an impossible ceiling was stored"


def test_a_trade_dated_after_its_own_filing_loses_the_date(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    store.upsert_filing({"filing_id": "f1", "chamber": "house", "doc_id": "1",
                         "member_id": "house:x", "filing_type": "P",
                         "raw_filing_type": "P", "filed_date": "2026-02-09",
                         "year": 2026,
                         "source_url": "http://example.invalid"})

    store.replace_transactions("f1", "house:x", [{
        "ticker": "SONY", "transaction_type": "purchase",
        "transaction_date": "2026-12-26", "notification_date": "2026-01-21",
        "amount_min": 1001, "amount_max": 15000,
    }])

    with store.connect() as conn:
        row = conn.execute(
            "SELECT transaction_date, ticker, amount_min FROM transactions"
    ).fetchone()
    assert row[0] is None, "a trade was stored as happening after it was filed"
    assert row[1] == "SONY", "the whole row was dropped instead of the date"
    assert row[2] == 1001, "the amount was discarded with the date"


def test_an_ordinary_row_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    store.upsert_filing({"filing_id": "f1", "chamber": "house", "doc_id": "1",
                         "member_id": "house:x", "filing_type": "P",
                         "raw_filing_type": "P", "filed_date": "2026-02-09",
                         "year": 2026,
                         "source_url": "http://example.invalid"})

    store.replace_transactions("f1", "house:x", [{
        "ticker": "AAPL", "transaction_type": "purchase",
        "transaction_date": "2026-01-15", "notification_date": "2026-01-20",
        "amount_min": 1001, "amount_max": 15000,
    }])

    with store.connect() as conn:
        row = conn.execute(
            "SELECT transaction_date, amount_min, amount_max FROM transactions"
    ).fetchone()
    assert row == ("2026-01-15", 1001, 15000)


def test_a_filing_with_no_date_does_not_reject_every_row(tmp_path, monkeypatch):
    """Absence of a filing date is not evidence the trade date is wrong."""
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    store.upsert_filing({"filing_id": "f1", "chamber": "house", "doc_id": "1",
                         "member_id": "house:x", "filing_type": "P",
                         "raw_filing_type": "P", "filed_date": None,
                         "year": 2026,
                         "source_url": "http://example.invalid"})

    store.replace_transactions("f1", "house:x", [{
        "ticker": "AAPL", "transaction_type": "purchase",
        "transaction_date": "2026-01-15", "amount_min": 1001,
        "amount_max": 15000,
    }])

    with store.connect() as conn:
        row = conn.execute(
            "SELECT transaction_date FROM transactions").fetchone()
    assert row[0] == "2026-01-15"


def test_the_repair_clears_rows_written_before_the_guard(tmp_path, monkeypatch):
    """The guard stops new rows; 47 were already stored without it."""
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    store.upsert_filing({"filing_id": "f1", "chamber": "house", "doc_id": "1",
                         "member_id": "house:x", "filing_type": "P",
                         "raw_filing_type": "P", "filed_date": "2026-02-09",
                         "year": 2026,
                         "source_url": "http://example.invalid"})

    # Written the way the old path wrote them, bypassing the guard.
    with store.connect() as conn:
        conn.executemany(
            """INSERT INTO transactions(txn_id, filing_id, member_id, row_index,
                    ticker, transaction_date, amount_min, amount_max)
               VALUES(?,?,?,?,?,?,?,?)""",
            [("f1#0", "f1", "house:x", 0, "CSCO", "2024-08-06", 50001, 200),
             ("f1#1", "f1", "house:x", 1, "SONY", "2026-12-26", 1001, 15000),
             ("f1#2", "f1", "house:x", 2, "AAPL", "2026-01-15", 1001, 15000)])

    report = store.repair_impossible_rows()
    assert report["amounts_cleared"] == 1
    assert report["dates_cleared"] == 1

    with store.connect() as conn:
        rows = dict((r[0], (r[1], r[2], r[3])) for r in conn.execute(
            "SELECT ticker, transaction_date, amount_min, amount_max "
            "FROM transactions"))
    assert rows["CSCO"] == ("2024-08-06", 50001, None)
    assert rows["SONY"] == (None, 1001, 15000)
    assert rows["AAPL"] == ("2026-01-15", 1001, 15000), "an ordinary row moved"


def test_the_repair_is_idempotent(tmp_path, monkeypatch):
    """Safe to run on every sync, so it does not need remembering."""
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()
    store.upsert_filing({"filing_id": "f1", "chamber": "house", "doc_id": "1",
                         "member_id": "house:x", "filing_type": "P",
                         "raw_filing_type": "P", "filed_date": "2026-02-09",
                         "year": 2026,
                         "source_url": "http://example.invalid"})
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO transactions(txn_id, filing_id, member_id, row_index,
                    ticker, transaction_date, amount_min, amount_max)
               VALUES('f1#0','f1','house:x',0,'CSCO','2024-08-06',50001,200)""")

    assert store.repair_impossible_rows()["amounts_cleared"] == 1
    assert store.repair_impossible_rows()["amounts_cleared"] == 0
