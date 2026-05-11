"""SQLite schema definitions for the autonomous system's persistent state.

All tables live in the same db (db_cache/session.db) as the existing
Session_Cache tool/news/scrape caches. Tables here are created idempotently
via CREATE TABLE IF NOT EXISTS so repeated imports are safe.
"""
import sqlite3
import os
from typing import Optional

DB_PATH = os.path.join("db_cache", "session.db")

CREATE_SCHEMA = [
    # --- Phase 0: watchlist (tickers actively monitored) ---
    """CREATE TABLE IF NOT EXISTS watchlist(
        ticker      TEXT PRIMARY KEY,
        added_at    TIMESTAMP,
        priority    INTEGER DEFAULT 1,
        notes       TEXT
    )""",

    # --- Phase 1: events (news + filings the daemon ingests) ---
    """CREATE TABLE IF NOT EXISTS events(
        event_id            TEXT PRIMARY KEY,
        source              TEXT,
        ticker              TEXT,
        headline            TEXT,
        body                TEXT,
        url                 TEXT,
        published_at        TIMESTAMP,
        ingested_at         TIMESTAMP,
        materiality         TEXT,
        category            TEXT,
        affected_tickers    TEXT,
        primary_ticker      TEXT,
        directional_signal  TEXT,
        urgency             TEXT,
        classifier_reason   TEXT,
        processed           BOOLEAN DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_events_ticker_published ON events(ticker, published_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_unprocessed ON events(processed, materiality)",

    # --- Phase 2: theses (agent recommendations persisted across runs) ---
    """CREATE TABLE IF NOT EXISTS theses(
        thesis_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT,
        thesis_date         TIMESTAMP,
        recommendation      TEXT,
        signal              TEXT,
        target_price        REAL,
        stop_loss           REAL,
        confidence          REAL,
        analysis_summary    TEXT,
        key_assumptions     TEXT,
        data_gaps           TEXT,
        full_report_md      TEXT,
        arbiter_verdict_id  INTEGER,
        superseded_by       INTEGER,
        FOREIGN KEY (superseded_by) REFERENCES theses(thesis_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_theses_ticker_active ON theses(ticker, superseded_by)",

    # --- Phase 6: positions (paper or live holdings) ---
    """CREATE TABLE IF NOT EXISTS positions(
        position_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT,
        side            TEXT,
        quantity        REAL,
        entry_price     REAL,
        entry_date      TIMESTAMP,
        current_price   REAL,
        unrealized_pnl  REAL,
        thesis_id       INTEGER,
        stop_loss       REAL,
        target_price    REAL,
        status          TEXT DEFAULT 'open',
        closed_at       TIMESTAMP,
        exit_price      REAL,
        realized_pnl    REAL,
        exit_reason     TEXT,
        paper           BOOLEAN DEFAULT 1
    )""",
    "CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(status, ticker)",

    # --- Phase 6: orders (broker order audit log) ---
    """CREATE TABLE IF NOT EXISTS orders(
        order_id            TEXT PRIMARY KEY,
        client_order_id     TEXT UNIQUE,
        ticker              TEXT,
        side                TEXT,
        order_type          TEXT,
        quantity            REAL,
        limit_price         REAL,
        status              TEXT,
        created_at          TIMESTAMP,
        filled_at           TIMESTAMP,
        thesis_id           INTEGER,
        arbiter_verdict_id  INTEGER,
        paper               BOOLEAN DEFAULT 1
    )""",
]


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with row factory set for dict-like access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(db_path: str = DB_PATH) -> None:
    """Create all autonomous-system tables if they don't exist."""
    conn = get_connection(db_path)
    try:
        for stmt in CREATE_SCHEMA:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def add_to_watchlist(ticker: str, priority: int = 1, notes: str = "") -> None:
    """Add a ticker to the watchlist. Idempotent."""
    from datetime import datetime
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, added_at, priority, notes) VALUES (?,?,?,?)",
            (ticker.upper(), datetime.now().isoformat(), priority, notes)
        )
        conn.commit()
    finally:
        conn.close()


def get_watchlist(min_priority: int = 1) -> list:
    """Return list of watched ticker symbols at or above the priority threshold."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ticker FROM watchlist WHERE priority >= ? ORDER BY priority DESC, ticker",
            (min_priority,)
        ).fetchall()
        return [r['ticker'] for r in rows]
    finally:
        conn.close()


def remove_from_watchlist(ticker: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # Initialize schema and seed watchlist from env if empty
    import os as _os
    init_schema()
    if not get_watchlist():
        default = _os.getenv("NEMO_WATCHLIST", "AAPL,MSFT,GOOGL,NVDA,TSLA,JPM,KO")
        for t in default.split(","):
            t = t.strip().upper()
            if t:
                add_to_watchlist(t)
        print(f"[Schema] Seeded watchlist with: {get_watchlist()}")
    else:
        print(f"[Schema] Tables initialized. Existing watchlist: {get_watchlist()}")
