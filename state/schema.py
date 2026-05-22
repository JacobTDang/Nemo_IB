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

    # --- Phase 11: thesis_evolution (Soros reflexivity log) ---
    # Tracks how a thesis's conviction evolves over time as new
    # observations arrive. Each row is a check-in against the originating
    # thesis_id with a conviction delta and a free-text observation.
    """CREATE TABLE IF NOT EXISTS thesis_evolution(
        evolution_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        thesis_id           INTEGER NOT NULL,
        ticker              TEXT,
        timestamp           TIMESTAMP,
        observation         TEXT,
        conviction_delta    REAL,
        new_conviction      REAL,
        tag                 TEXT,
        FOREIGN KEY (thesis_id) REFERENCES theses(thesis_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_thesis_evolution_thesis ON thesis_evolution(thesis_id, timestamp)",

    # --- RAG: chunked text documents for retrieval-augmented generation ---
    # Each row stores a chunk of source text (filing section, news body, etc).
    # The companion `rag_chunk_embeddings` vec0 virtual table holds a 384-dim
    # float embedding keyed by rowid == chunk_id (paired INSERT keeps them in
    # sync). Created in init_schema() because vec0 needs the loaded extension.
    """CREATE TABLE IF NOT EXISTS rag_chunks(
        chunk_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id          TEXT NOT NULL,
        ticker          TEXT,
        source_tool     TEXT,
        doc_type        TEXT,
        filing_date     TIMESTAMP,
        item_number     TEXT,
        section_heading TEXT,
        chunk_text      TEXT NOT NULL,
        chunk_offset    INTEGER,
        chunk_sequence  INTEGER,
        created_at      TIMESTAMP,
        UNIQUE(doc_id, chunk_sequence)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_rag_chunks_ticker ON rag_chunks(ticker, doc_type, filing_date)",
    "CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(doc_id)",

    # --- Sentry: evaluation log (every ticker the autonomous loop considers) ---
    # The eval log is the system's memory. Sentry checks it before queueing to
    # avoid re-researching the same name during a cooldown window. Each row
    # captures: which channel triggered the consideration, the decision, the
    # verdict if researched, and a computed next_review_at timestamp based on
    # cooldown rules (acted/buy=7d, watchlist=14d, avoid/no_position=30d).
    """CREATE TABLE IF NOT EXISTS sentry_evaluation_log(
        eval_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker            TEXT NOT NULL,
        evaluated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        triggered_by      TEXT NOT NULL,
        trigger_event_id  TEXT,
        decision          TEXT NOT NULL,
        verdict           TEXT,
        confidence        REAL,
        sizing            TEXT,
        factor_buckets    TEXT,
        skip_reason       TEXT,
        next_review_at    TIMESTAMP,
        notes             TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_eval_ticker_date ON sentry_evaluation_log(ticker, evaluated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_eval_next_review ON sentry_evaluation_log(next_review_at) WHERE next_review_at IS NOT NULL",

    # --- Sentry: triage queue (candidates awaiting Claude review) ---
    # The triage daemon writes pending rows; the /sentry-tick skill reads top-N
    # by score and processes them. The unique index on (ticker) WHERE status='pending'
    # enforces deduplication at the DB layer — same ticker can't be queued twice
    # simultaneously across discovery channels.
    """CREATE TABLE IF NOT EXISTS sentry_queue(
        queue_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker            TEXT NOT NULL,
        source_event_id   TEXT,
        triggered_by      TEXT NOT NULL,
        score             REAL NOT NULL,
        queued_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        status            TEXT NOT NULL DEFAULT 'pending',
        processed_at      TIMESTAMP,
        notes             TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_queue_status_score ON sentry_queue(status, score DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_ticker_pending ON sentry_queue(ticker) WHERE status = 'pending'",

    # --- Sentry: daily budget counters (resets at 00:00 ET each day) ---
    # One row per ET day; INSERT OR IGNORE on read+increment. The budget gate
    # in /sentry-tick consults these counters before allowing the next action.
    # `day` is YYYY-MM-DD in America/New_York timezone.
    """CREATE TABLE IF NOT EXISTS sentry_daily_actions(
        day                TEXT PRIMARY KEY,
        research_runs      INTEGER NOT NULL DEFAULT 0,
        slack_messages     INTEGER NOT NULL DEFAULT 0,
        paper_orders       INTEGER NOT NULL DEFAULT 0,
        new_positions      INTEGER NOT NULL DEFAULT 0,
        adds_or_trims      INTEGER NOT NULL DEFAULT 0,
        first_action_at    TIMESTAMP,
        last_action_at     TIMESTAMP
    )""",

    # --- Sentry: ETF AUM history (for theme-flow discovery channel) ---
    # Daily snapshot of AUM + top holdings per theme ETF. Theme-flow scan
    # compares today vs 7 days ago to detect rotations. top_holdings stored
    # as JSON list of {symbol, weight_pct} dicts.
    """CREATE TABLE IF NOT EXISTS etf_aum_history(
        snapshot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        etf_symbol       TEXT NOT NULL,
        snapshot_date    TEXT NOT NULL,
        theme            TEXT,
        total_assets     REAL,
        top_holdings     TEXT,
        captured_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_etf_aum_etf_date ON etf_aum_history(etf_symbol, snapshot_date)",
    "CREATE INDEX IF NOT EXISTS idx_etf_aum_date ON etf_aum_history(snapshot_date DESC)",

    # --- Sentry: discovery run tracker (one row per ET day) ---
    # sentry_triage checks this on every tick — if no row for today exists,
    # it triggers sentry_discovery.run_all() and inserts a row. Prevents
    # discovery from re-running every tick of the day.
    """CREATE TABLE IF NOT EXISTS sentry_discovery_runs(
        day                   TEXT PRIMARY KEY,
        ran_at                TIMESTAMP NOT NULL,
        catalyst_enqueued     INTEGER NOT NULL DEFAULT 0,
        insider_enqueued      INTEGER NOT NULL DEFAULT 0,
        activist_enqueued     INTEGER NOT NULL DEFAULT 0,
        theme_flow_enqueued   INTEGER NOT NULL DEFAULT 0,
        total_enqueued        INTEGER NOT NULL DEFAULT 0,
        errors                TEXT
    )""",

    # --- Sentry: universe of tickers from theme ETF holdings ---
    # Built by daemons/sentry_universe.py: refresh_universe() unions the top
    # holdings across all curated theme ETFs into this table. Discovery
    # channels that need a universe-wide ticker set (universe_insider_cluster,
    # fundamental_screener) read from here.
    """CREATE TABLE IF NOT EXISTS sentry_universe(
        ticker                TEXT PRIMARY KEY,
        last_seen_in_themes   TEXT,
        first_seen_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        refreshed_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        excluded              INTEGER NOT NULL DEFAULT 0,
        excluded_reason       TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sentry_universe_active ON sentry_universe(excluded, refreshed_at)",
]


# Soros reflexivity columns added to theses via ALTER TABLE (migration).
# Old DBs lacking these columns will get them on next init_schema() call.
_THESES_MIGRATIONS = [
    ("falsifiers",          "TEXT"),
    ("variant_perception",  "TEXT"),
]


# Discipline audit columns added to sentry_evaluation_log so the eval row
# itself records whether each pre-save discipline check was satisfied.
# Validator in state/sentry_eval_log.py:record_eval enforces presence
# conditional on (decision, verdict). Legacy rows stay NULL on these cols.
_EVAL_LOG_MIGRATIONS = [
    ("analogue_considered",        "TEXT"),    # name of analogue or 'none'
    ("terminal_sensitivity_ran",   "INTEGER"), # 0/1/NULL
    ("contradiction_check_passed", "INTEGER"), # 0/1/NULL
    ("provenance_filing_count",    "INTEGER"),
    ("provenance_press_count",     "INTEGER"),
]


# Per-channel counters added to sentry_discovery_runs as new discovery
# channels ship. Each is ADD COLUMN with a NULL/0 default so legacy rows
# stay valid. screener_last_ran is the only TEXT column — it stores the
# ET date of the most recent screener run so the weekly cadence gate can
# compare against today.
_DISCOVERY_RUNS_MIGRATIONS = [
    ("universe_insider_enqueued",  "INTEGER DEFAULT 0"),
    ("rag_analogue_enqueued",      "INTEGER DEFAULT 0"),
    ("ipo_enqueued",               "INTEGER DEFAULT 0"),
    ("screener_last_ran",          "TEXT"),
    ("screener_enqueued",          "INTEGER DEFAULT 0"),
]


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with row factory set for dict-like access.

    Also loads the sqlite-vec extension on every connection so the
    `rag_chunk_embeddings` virtual table is usable. If sqlite-vec is not
    installed (e.g. in a minimal environment) the import is skipped silently
    so non-RAG code paths continue to work.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (ImportError, AttributeError, sqlite3.OperationalError):
        # sqlite-vec not installed or platform doesn't support extension
        # loading; RAG features will fail loudly at query time which is
        # the correct behavior.
        pass
    return conn


def init_schema(db_path: str = DB_PATH) -> None:
    """Create all autonomous-system tables if they don't exist, and run
    additive ALTER TABLE migrations for new columns on existing rows.

    Also creates the sqlite-vec virtual table `rag_chunk_embeddings` so
    chunk_id rows in `rag_chunks` can be paired with a 384-dim float
    embedding (rowid == chunk_id by convention).
    """
    conn = get_connection(db_path)
    try:
        for stmt in CREATE_SCHEMA:
            conn.execute(stmt)

        # Additive migrations: ALTER TABLE ADD COLUMN is a no-op error if
        # the column already exists. SQLite raises an OperationalError we
        # catch and ignore.
        existing_cols = {row['name'] for row in
                         conn.execute("PRAGMA table_info(theses)").fetchall()}
        for col_name, col_type in _THESES_MIGRATIONS:
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE theses ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        # Same pattern for sentry_evaluation_log discipline audit columns.
        existing_cols = {row['name'] for row in
                         conn.execute("PRAGMA table_info(sentry_evaluation_log)").fetchall()}
        for col_name, col_type in _EVAL_LOG_MIGRATIONS:
            if col_name not in existing_cols:
                try:
                    conn.execute(
                        f"ALTER TABLE sentry_evaluation_log ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass

        # Same pattern for sentry_discovery_runs per-channel counters.
        existing_cols = {row['name'] for row in
                         conn.execute("PRAGMA table_info(sentry_discovery_runs)").fetchall()}
        for col_name, col_type in _DISCOVERY_RUNS_MIGRATIONS:
            if col_name not in existing_cols:
                try:
                    conn.execute(
                        f"ALTER TABLE sentry_discovery_runs ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass

        # vec0 virtual table for chunk embeddings. Requires the sqlite-vec
        # extension to be loaded on this connection (handled by
        # get_connection above). all-MiniLM-L6-v2 produces 384-dim vectors.
        try:
            import sqlite_vec  # noqa: F401  (presence check)
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunk_embeddings "
                "USING vec0(embedding float[384])"
            )
        except (ImportError, sqlite3.OperationalError) as exc:
            # If sqlite-vec is missing the RAG feature is unusable but the
            # rest of the schema is still valid. Print a single warning
            # rather than failing the whole init.
            print(f"[Schema] WARNING: rag_chunk_embeddings not created: {exc}")
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
