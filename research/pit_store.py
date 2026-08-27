"""An append-only record of what was known, and when it was known.

The strategy this supports needs a backtest, and a backtest needs history that
free vendors do not sell honestly: yfinance drops delisted tickers, so any
study of "every name that reported" quietly becomes "every name that reported
and survived"; its adjusted prices are recomputed as new splits and dividends
land, so the same run returns different numbers six months later; and nobody
retains what analysts expected on a past Tuesday.

History can be bought later. A forward record cannot be made later. That
asymmetry is the whole reason this module is the first thing built rather than
the last.

Three rules follow from it, and each has a test that fails if it is broken:

**Nothing is overwritten.** A row records a belief held at a moment. Update it
and the store stops being evidence of what we knew and becomes evidence of what
we know now -- which is precisely the thing a backtest must not have access to.
When a vendor hands back a different value for a session already recorded, the
original stands and the disagreement is filed as its own fact.

**Prices are stored raw.** An adjusted close is a derived number that changes
under you. Raw OHLC never moves; splits and dividends are recorded separately
with their own `recorded_at`, so the adjustment in force on any past date can
be rebuilt exactly.

**Reads are as-of, on `recorded_at` and not merely on the data's own date.**
This is the subtle one. A bar for 3 March back-filled on 10 May carries a March
date, and filtering on that date alone lets it leak into a simulation standing
on 4 March. Only the recording timestamp prevents it.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

_DEFAULT_DB = "db_cache/pit.db"

# Compared field by field when a session is re-recorded. Deliberately excludes
# nothing: a vendor changing any of these has changed history.
_BAR_FIELDS = ("open", "high", "low", "close", "volume")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_snapshot (
    as_of_date       TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    cik              TEXT,
    name             TEXT,
    eligible         INTEGER NOT NULL,
    exclusion_reason TEXT,
    recorded_at      TEXT NOT NULL,
    PRIMARY KEY (as_of_date, ticker)
);

-- Raw only. No adjusted column, by design: see the module docstring.
CREATE TABLE IF NOT EXISTS daily_bar (
    trade_date  TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (trade_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_bar_ticker ON daily_bar(ticker, trade_date);

CREATE TABLE IF NOT EXISTS corporate_action (
    ex_date     TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    action_type TEXT NOT NULL,
    value       REAL NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (ex_date, ticker, action_type)
);

-- A vendor rewriting a past session is itself a fact worth keeping.
CREATE TABLE IF NOT EXISTS bar_revision (
    trade_date TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    field      TEXT NOT NULL,
    old_value  REAL,
    new_value  REAL,
    noticed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS announcement (
    ticker         TEXT NOT NULL,
    fiscal_period  TEXT NOT NULL,
    announced_date TEXT NOT NULL,
    timing         TEXT NOT NULL DEFAULT 'unknown',
    source         TEXT,
    recorded_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, fiscal_period)
);

-- The series that cannot be reconstructed after the fact.
CREATE TABLE IF NOT EXISTS consensus_snapshot (
    as_of_date    TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    fiscal_period TEXT NOT NULL,
    eps_estimate  REAL,
    analyst_count INTEGER,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (as_of_date, ticker, fiscal_period)
);

CREATE TABLE IF NOT EXISTS run_log (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job          TEXT NOT NULL,
    as_of_date   TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    rows_written INTEGER,
    status       TEXT,
    error        TEXT
);
"""


def db_path() -> str:
    return os.environ.get("NEMO_PIT_DB", _DEFAULT_DB)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def connect() -> sqlite3.Connection:
    path = db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------- bars

def record_bars(ticker: str, rows: Iterable[Dict[str, Any]],
                recorded_at: Optional[str] = None) -> int:
    """Append sessions for one ticker. Returns the count newly written.

    A session already present is left exactly as it was. If the incoming values
    differ, the difference is filed in `bar_revision` -- the original is what we
    acted on, and the fact that the vendor now disagrees is separate
    information rather than a reason to lose the original.
    """
    stamp = recorded_at or _now()
    written = 0
    with connect() as conn:
        for row in rows:
            existing = conn.execute(
                "SELECT * FROM daily_bar WHERE trade_date = ? AND ticker = ?",
                (row["trade_date"], ticker)).fetchone()

            if existing is None:
                conn.execute(
                    """INSERT INTO daily_bar
                       (trade_date, ticker, open, high, low, close, volume,
                        recorded_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (row["trade_date"], ticker, row.get("open"),
                     row.get("high"), row.get("low"), row.get("close"),
                     row.get("volume"), stamp))
                written += 1
                continue

            for field in _BAR_FIELDS:
                before, after = existing[field], row.get(field)
                if after is None or before is None:
                    continue
                if abs(float(before) - float(after)) > 1e-9:
                    conn.execute(
                        """INSERT INTO bar_revision
                           (trade_date, ticker, field, old_value, new_value,
                            noticed_at)
                           VALUES (?,?,?,?,?,?)""",
                        (row["trade_date"], ticker, field, float(before),
                         float(after), stamp))
    return written


def bars_as_of(ticker: str, as_of: str) -> List[Dict[str, Any]]:
    """Sessions for `ticker` as they were known on `as_of`.

    Both clauses matter. `trade_date <= as_of` keeps out sessions that had not
    happened; `recorded_at <= as_of` keeps out sessions we had not yet learned.
    Dropping the second is the classic lookahead bug, and it is invisible in
    results -- it just makes them better.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT * FROM daily_bar
               WHERE ticker = ? AND trade_date <= ? AND date(recorded_at) <= ?
               ORDER BY trade_date""",
            (ticker, as_of, as_of)).fetchall()
    return [dict(r) for r in rows]


def revisions(ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM bar_revision"
    params: tuple = ()
    if ticker:
        sql += " WHERE ticker = ?"
        params = (ticker,)
    sql += " ORDER BY noticed_at"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------- corporate actions

def record_corporate_action(ticker: str, ex_date: str, action_type: str,
                            value: float,
                            recorded_at: Optional[str] = None) -> None:
    """A split ratio or a dividend per share, kept apart from the prices.

    Storing the action rather than a pre-adjusted price is what lets the
    adjustment in force on any past date be rebuilt. A vendor's adjusted close
    only tells you about today.
    """
    with connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO corporate_action
               (ex_date, ticker, action_type, value, recorded_at)
               VALUES (?,?,?,?,?)""",
            (ex_date, ticker, action_type, float(value),
             recorded_at or _now()))


def corporate_actions_as_of(ticker: str, as_of: str) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT * FROM corporate_action
               WHERE ticker = ? AND ex_date <= ? AND date(recorded_at) <= ?
               ORDER BY ex_date""",
            (ticker, as_of, as_of)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- universe

def record_universe(as_of_date: str,
                    entries: Iterable[Dict[str, Any]]) -> int:
    """What the universe was on one date, eligible names and rejected ones.

    Rejections are kept deliberately. A name excluded for liquidity today may
    qualify next quarter, and a study that only ever sees the survivors cannot
    tell the difference between a screen that worked and a screen that was
    never applied.
    """
    stamp = _now()
    written = 0
    with connect() as conn:
        for e in entries:
            cur = conn.execute(
                """INSERT OR IGNORE INTO universe_snapshot
                   (as_of_date, ticker, cik, name, eligible, exclusion_reason,
                    recorded_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (as_of_date, e["ticker"], e.get("cik"), e.get("name"),
                 1 if e.get("eligible") else 0, e.get("exclusion_reason"),
                 stamp))
            written += cur.rowcount
    return written


def universe_as_of(as_of_date: str) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM universe_snapshot WHERE as_of_date = ? ORDER BY ticker",
            (as_of_date,)).fetchall()
    return [{**dict(r), "eligible": bool(r["eligible"])} for r in rows]


# ------------------------------------------------------------ announcements

def record_announcement(ticker: str, fiscal_period: str, announced_date: str,
                        timing: str = "unknown",
                        source: Optional[str] = None) -> None:
    """Keyed on fiscal identity, never on a vendor's calendar bucket.

    get_earnings_surprises labels AMAT's 13 August print `2026-09-30` -- a
    calendar quarter end that can lead the fiscal close by weeks. Joining on it
    returns nothing.

    `timing` decides which session is the gap: AMAT reported after the close,
    so the move is on 14 August, and using the filing date instead measures
    -2.48% against a real -6.57%. It defaults to "unknown" rather than to a
    guess, because a guess here is a wrong number with no warning attached.
    """
    with connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO announcement
               (ticker, fiscal_period, announced_date, timing, source,
                recorded_at)
               VALUES (?,?,?,?,?,?)""",
            (ticker, fiscal_period, announced_date, timing or "unknown",
             source, _now()))


def announcements_as_of(ticker: str, as_of: str) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT * FROM announcement
               WHERE ticker = ? AND announced_date <= ?
                 AND date(recorded_at) <= ?
               ORDER BY announced_date""",
            (ticker, as_of, as_of)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- consensus

def record_consensus(as_of_date: str, ticker: str, fiscal_period: str,
                     eps_estimate: Optional[float] = None,
                     analyst_count: Optional[int] = None) -> None:
    """One day's view of what the street expects.

    This is the series with a clock on it. Finnhub returns four quarters at
    limit=12 and at limit=30 -- verified -- so the history simply does not
    exist to be fetched. It only accrues going forward, one snapshot per name
    per day, which is why the recorder starting today matters more than the
    backtest starting sooner.
    """
    with connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO consensus_snapshot
               (as_of_date, ticker, fiscal_period, eps_estimate, analyst_count,
                recorded_at)
               VALUES (?,?,?,?,?,?)""",
            (as_of_date, ticker, fiscal_period, eps_estimate, analyst_count,
             _now()))


def consensus_as_of(ticker: str, fiscal_period: str,
                    as_of: str) -> Optional[Dict[str, Any]]:
    """The most recent snapshot on or before `as_of`, or None.

    None, not zero and not the earliest available: before the first snapshot we
    did not know what the street expected, and any number here would be one we
    made up.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM consensus_snapshot
               WHERE ticker = ? AND fiscal_period = ? AND as_of_date <= ?
               ORDER BY as_of_date DESC LIMIT 1""",
            (ticker, fiscal_period, as_of)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- run log

_ACTIVE_RUN: Dict[str, Any] = {}


def start_run(job: str, as_of_date: Optional[str] = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO run_log (job, as_of_date, started_at) VALUES (?,?,?)",
            (job, as_of_date, _now()))
        run_id = cur.lastrowid
    _ACTIVE_RUN["run_id"] = run_id
    return run_id


def finish_run(rows_written: int = 0, status: str = "ok",
               error: Optional[str] = None,
               run_id: Optional[int] = None) -> None:
    rid = run_id or _ACTIVE_RUN.get("run_id")
    if rid is None:
        raise ValueError("finish_run called with no run in progress")
    with connect() as conn:
        conn.execute(
            """UPDATE run_log
               SET finished_at = ?, rows_written = ?, status = ?, error = ?
               WHERE run_id = ?""",
            (_now(), rows_written, status, error, rid))


def missing_days(job: str, start: str, end: str) -> List[str]:
    """Weekdays in the range with no successful run.

    A gap you cannot see becomes a conclusion; a gap you can see becomes a
    caveat. Only a finished run with status 'ok' counts -- a crashed process
    leaves a started row with no finish, and a failed fetch leaves a finish
    with no data, and neither is coverage.
    """
    from datetime import date, timedelta

    with connect() as conn:
        covered = {
            r["as_of_date"] for r in conn.execute(
                """SELECT DISTINCT as_of_date FROM run_log
                   WHERE job = ? AND status = 'ok' AND finished_at IS NOT NULL
                     AND as_of_date IS NOT NULL""", (job,)).fetchall()
        }

    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    gaps: List[str] = []
    day = first
    while day <= last:
        if day.weekday() < 5 and day.isoformat() not in covered:
            gaps.append(day.isoformat())
        day += timedelta(days=1)
    return gaps
