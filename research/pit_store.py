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
    -- The VENDOR's reported figure, not the filing's. A street estimate is
    -- non-GAAP and XBRL is GAAP -- Finnhub's MSFT 2026Q2 actual is 4.14
    -- against 5.16 in the 10-Q -- so an analyst surprise must be computed
    -- within one basis or it manufactures a gap that is not there.
    eps_actual    REAL,
    analyst_count INTEGER,
    -- 'recorded' if we were watching that day, 'seeded' if it was
    -- reconstructed afterwards from the vendor's own history. The two are not
    -- the same evidence and nothing downstream may treat them as though they
    -- were.
    source        TEXT NOT NULL DEFAULT 'recorded',
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (as_of_date, ticker, fiscal_period)
);

-- Schedule 13D events. `subject_ticker` is the company a stake was taken IN,
-- never one that took a stake in someone else: both sides live in the same
-- EDGAR folder and conflating them invents activist campaigns.
--
-- Four timestamps, none of them redundant. `filing_date` is the event's own
-- date; `accepted_at` is when SEC accepted it, which is when it became public;
-- `detected_at` is when we saw it, and the gap between those two is the only
-- thing this watcher is actually judged on; `recorded_at` is the store's
-- as-of discipline, so a filing learned years late stays invisible to a query
-- standing before we learned it.
CREATE TABLE IF NOT EXISTS activist_filing (
    accession        TEXT NOT NULL,
    subject_ticker   TEXT NOT NULL,
    subject_cik      TEXT,
    subject_name     TEXT,
    filer_name       TEXT,
    filer_cik        TEXT,
    form             TEXT NOT NULL,
    is_amendment     INTEGER NOT NULL,
    filing_date      TEXT NOT NULL,
    accepted_at      TEXT,
    detected_at      TEXT NOT NULL,
    subject_verified INTEGER NOT NULL,
    url              TEXT,
    recorded_at      TEXT NOT NULL,
    PRIMARY KEY (accession, subject_ticker)
);
CREATE INDEX IF NOT EXISTS idx_activist_ticker
    ON activist_filing(subject_ticker, filing_date);

CREATE TABLE IF NOT EXISTS paper_order (
    as_of_date       TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    accepted         INTEGER NOT NULL,
    -- Rejections are kept beside acceptances deliberately. A scan that quietly
    -- stops finding candidates is indistinguishable from a market with nothing
    -- in it, and the difference lives entirely in these reasons.
    reason           TEXT,
    side             TEXT,
    fiscal_period    TEXT,
    sue              REAL,
    expected_edge_bps REAL,
    cost_bps         REAL,
    net_edge_bps     REAL,
    target_dollars   REAL,
    participation    REAL,
    spread           REAL,
    spread_resolved  INTEGER,
    rank             INTEGER,
    regime           TEXT,
    gross_target     REAL,
    -- The session the order is FOR. No fill price: that session has not
    -- happened when the row is written, which is the entire point.
    intended_session TEXT,
    recorded_at      TEXT NOT NULL,
    PRIMARY KEY (as_of_date, ticker)
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


# Columns added after the first store existed. CREATE TABLE IF NOT EXISTS
# leaves an existing table untouched, so a new column needs saying explicitly
# or a store created yesterday silently lacks it.
_MIGRATIONS = (
    ("consensus_snapshot", "eps_actual", "REAL"),
    ("consensus_snapshot", "source", "TEXT NOT NULL DEFAULT 'recorded'"),
)


def init_schema() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)
        for table, column, decl in _MIGRATIONS:
            present = {r[1] for r in conn.execute(
                f"PRAGMA table_info({table})")}
            if column not in present:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


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


def adjusted_bars(ticker: str, as_of: str,
                  total_return: bool = False) -> List[Dict[str, Any]]:
    """`bars_as_of`, with the adjustment in force on `as_of` rebuilt onto it.

    This is the other half of storing raw. Keeping raw OHLC is what makes a
    result reproducible; rebuilding the adjustment here is what makes it
    usable. Without this, every consumer measuring a return over a window
    containing a split sees the split as the return -- NVDA's June 2024 10:1
    is a -90% session, and it sits inside any window longer than a year.

    The factor is built only from actions this reader could have known, so a
    split announced in June cannot reach back and change what a May reader
    computed. That drift is exactly why a vendor's adjusted close is not
    evidence of anything.

    Splits always apply: a discontinuity of that size is not a price move under
    any interpretation. Dividends are opt-in, because whether they belong
    depends on the question. Measuring what an investor earned, they do;
    measuring the price the exchange printed, they do not. Choosing silently
    means one caller or the other is quietly wrong.
    """
    bars = bars_as_of(ticker, as_of)
    if not bars:
        return []

    actions = corporate_actions_as_of(ticker, as_of)
    kinds = {"split"} if not total_return else {"split", "dividend"}
    actions = [a for a in actions
               if a["action_type"] in kinds and (a.get("value") or 0) > 0]
    if not actions:
        return [{**b, "adj_factor": 1.0} for b in bars]

    closes = {b["trade_date"]: b["close"] for b in bars}

    # Walk backwards from the most recent session, accumulating each action as
    # it is passed. A bar's factor is the product of every action with an
    # ex-date strictly after it: the ex-date session already trades at the new
    # price, so it belongs with what follows, not with what came before.
    by_date: Dict[str, float] = {}
    for action in actions:
        ex = action["ex_date"]
        if action["action_type"] == "split":
            factor = 1.0 / float(action["value"])
        else:
            # A dividend's effect is proportional to the price it came out of:
            # the last close before the ex-date, which is what the payment was
            # actually a claim against.
            prior = [d for d in closes if d < ex]
            if not prior or not closes[max(prior)]:
                continue
            ref = closes[max(prior)]
            factor = (ref - float(action["value"])) / ref
            if factor <= 0:
                # Real, and not adjustable this way. A fund distributing most
                # of its NAV pays more than the shares are worth the day
                # before, and the ratio method has no answer for it. Skipping
                # quietly would return a total-return series that silently
                # ignores the largest cash flow in it.
                raise ValueError(
                    f"{ticker}: dividend {action['value']} on {ex} is not "
                    f"smaller than the {ref} close before it, so a price-ratio "
                    f"adjustment cannot represent it. Read this series with "
                    f"total_return=False, or handle the distribution "
                    f"explicitly")
        by_date[ex] = by_date.get(ex, 1.0) * factor

    # Descending, so the backward walk consumes them in the order it meets
    # them. Keyed on "every action dated after this bar" rather than on an
    # action landing exactly on a bar's date: an ex-date can fall on a session
    # the store has no row for -- a holiday, a missed fetch, or a split
    # effective after the last one recorded -- and keying on an exact match
    # silently skips the adjustment in all three cases.
    pending = sorted(by_date.items(), reverse=True)

    out: List[Dict[str, Any]] = []
    running = 1.0
    cursor = 0
    for bar in reversed(bars):
        while cursor < len(pending) and pending[cursor][0] > bar["trade_date"]:
            running *= pending[cursor][1]
            cursor += 1
        row = {**bar, "adj_factor": running}
        if running != 1.0:
            for field in ("open", "high", "low", "close"):
                if row.get(field) is not None:
                    row[field] = row[field] * running
            if row.get("volume") is not None:
                # Volume moves the other way: the same economic quantity of
                # stock, denominated in more shares.
                row["volume"] = row["volume"] / running
        out.append(row)
    out.reverse()
    return out


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
                            recorded_at: Optional[str] = None) -> int:
    """A split ratio or a dividend per share, kept apart from the prices.

    Storing the action rather than a pre-adjusted price is what lets the
    adjustment in force on any past date be rebuilt. A vendor's adjusted close
    only tells you about today.
    """
    if not (value > 0):
        # A zero split ratio divides by zero at read time and a negative one
        # flips the sign of every price before it. A dividend of zero is not an
        # event. Neither can be applied, so neither is recorded.
        raise ValueError(
            f"{action_type} value must be positive; got {value!r} for "
            f"{ticker} on {ex_date}. An unapplicable action recorded now is a "
            f"read that fails or silently skips it every time after")

    with connect() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO corporate_action
               (ex_date, ticker, action_type, value, recorded_at)
               VALUES (?,?,?,?,?)""",
            (ex_date, ticker, action_type, float(value),
             recorded_at or _now()))
    # Reported like every other recorder here, so a caller can tell a
    # write from a rerun that found the row already present.
    return cur.rowcount or 0


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
                    entries: Iterable[Dict[str, Any]],
                    recorded_at: Optional[str] = None) -> int:
    """What the universe was on one date, eligible names and rejected ones.

    Rejections are kept deliberately. A name excluded for liquidity today may
    qualify next quarter, and a study that only ever sees the survivors cannot
    tell the difference between a screen that worked and a screen that was
    never applied.
    """
    stamp = recorded_at or _now()
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
    """Membership as it last stood on or before `as_of_date`.

    Not an exact-date match. The screen does not run on weekends, holidays or
    any day the job failed, and on those dates an exact match answers that the
    market had no members -- which is not a fact about the market. It also
    inverted the job's own ordering: bars are recorded before the screen runs,
    so asking for today's membership mid-run always returned nothing and the
    nightly ask silently widened to every registrant the SEC lists.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT MAX(as_of_date) FROM universe_snapshot
               WHERE as_of_date <= ? AND date(recorded_at) <= ?""",
            (as_of_date, as_of_date)).fetchone()
        latest = row[0] if row else None
        if latest is None:
            return []
        rows = conn.execute(
            """SELECT * FROM universe_snapshot
               WHERE as_of_date = ? AND date(recorded_at) <= ?
               ORDER BY ticker""", (latest, as_of_date)).fetchall()
    return [{**dict(r), "eligible": bool(r["eligible"])} for r in rows]


# ------------------------------------------------------------ announcements

def record_announcement(ticker: str, fiscal_period: str, announced_date: str,
                        timing: str = "unknown",
                        source: Optional[str] = None,
                        recorded_at: Optional[str] = None) -> int:
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
        cur = conn.execute(
            """INSERT OR IGNORE INTO announcement
               (ticker, fiscal_period, announced_date, timing, source,
                recorded_at)
               VALUES (?,?,?,?,?,?)""",
            (ticker, fiscal_period, announced_date, timing or "unknown",
             source, recorded_at or _now()))
    # Reported like every other recorder here, so a caller can tell a
    # write from a rerun that found the row already present.
    return cur.rowcount or 0


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
                     analyst_count: Optional[int] = None,
                     eps_actual: Optional[float] = None,
                     recorded_at: Optional[str] = None,
                     source: str = "recorded") -> int:
    """One day's view of what the street expects.

    This is the series with a clock on it. Finnhub returns four quarters at
    limit=12 and at limit=30 -- verified -- so the history simply does not
    exist to be fetched. It only accrues going forward, one snapshot per name
    per day, which is why the recorder starting today matters more than the
    backtest starting sooner.
    """
    with connect() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO consensus_snapshot
               (as_of_date, ticker, fiscal_period, eps_estimate, eps_actual,
                analyst_count, recorded_at, source)
               VALUES (?,?,?,?,?,?,?,?)""",
            (as_of_date, ticker, fiscal_period, eps_estimate, eps_actual,
             analyst_count, recorded_at or _now(), source))
    return cur.rowcount or 0


def consensus_as_of(ticker: str, fiscal_period: str,
                    as_of: str) -> Optional[Dict[str, Any]]:
    """The most recent snapshot on or before `as_of`, or None.

    None, not zero and not the earliest available: before the first snapshot we
    did not know what the street expected, and any number here would be one we
    made up.

    Filtered on `recorded_at` as well as on the date the row describes, like
    every other reader here. Without it a snapshot written today for a past
    date is visible to that past date -- the lookahead the column exists to
    stop, on the one field where it matters most, because a consensus revised
    after the print is the answer to the question being asked.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM consensus_snapshot
               WHERE ticker = ? AND fiscal_period = ? AND as_of_date <= ?
                 AND date(recorded_at) <= ?
               ORDER BY as_of_date DESC LIMIT 1""",
            (ticker, fiscal_period, as_of, as_of)).fetchone()
    return dict(row) if row else None


def actual_as_of(ticker: str, fiscal_period: str,
                 as_of: str) -> Optional[float]:
    """The vendor's own reported figure for a quarter, as known on `as_of`.

    Separate from `consensus_as_of` because the two legs of an analyst surprise
    live on different dates: the estimate is what stood before the print and the
    actual is what landed after it, so one lookup cannot serve both. Reading
    both from this table is what keeps them on one basis -- a street estimate
    is non-GAAP and the 10-Q is not, and Finnhub's MSFT 2026Q2 actual is 4.14
    against 5.16 in the filing.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT eps_actual FROM consensus_snapshot
               WHERE ticker = ? AND fiscal_period = ? AND as_of_date <= ?
                 AND date(recorded_at) <= ? AND eps_actual IS NOT NULL
               ORDER BY as_of_date DESC LIMIT 1""",
            (ticker, fiscal_period, as_of, as_of)).fetchone()
    return float(row["eps_actual"]) if row else None


# --------------------------------------------------------- 13D events

# Past this, a filing was history when we first looked at it rather than
# something we caught. The distinction protects the only number that says
# whether the watcher works: a folder read for the first time hands back every
# 13D ever filed on that company, and averaging a 2019 filing "detected" in
# 2026 into the latency would bury a real two-minute catch under five years of
# nothing. Generous on purpose -- a watcher down over a long weekend was still
# late rather than back-filling, and that lateness should count against it.
BACKFILL_AFTER_SECONDS = 7 * 24 * 3600


def iso_utc(value: Any) -> Optional[str]:
    """Canonical `...Z` UTC, or None. Anything else raises.

    EDGAR's `acceptanceDateTime` carries a `Z` that is genuinely UTC and not
    decorative -- checked against the acceptance-hour distribution of INTC's
    1000 most recent filings, which is empty from 03h to 09h UTC, exactly the
    complement of the 06:00-22:00 ET window in which EDGAR accepts filings.
    Read as Eastern instead, every latency in this table would be four hours
    wrong and still look plausible.

    A naive datetime is taken as UTC for the same reason: the only naive value
    that reaches here comes from that field, via a pyarrow timestamp column
    that drops the zone while keeping the instant. An unparseable string is
    refused rather than nulled, because a null here reads as "SEC never told
    us" and that is a different fact from "we could not read what it said".
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"unparseable timestamp {value!r}: {exc}. Refused rather than "
                f"stored as null, which would claim SEC gave no acceptance "
                f"time when in fact we failed to read one.") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def record_activist_filings(entries: Iterable[Dict[str, Any]],
                            detected_at: Optional[str] = None,
                            recorded_at: Optional[str] = None) -> int:
    """Append 13D events. Returns the count newly written.

    A filing already recorded is left exactly as it was, `detected_at`
    included. When we first saw something is the one field that cannot be
    re-derived afterwards, and a watcher on a timer re-reads the same folder
    every pass -- so the natural bug is for the tenth pass to overwrite the
    first pass's detection time with its own and quietly report a latency of
    zero forever.

    An entry whose header says the company is not the subject is refused
    outright. Whatever classifies filings today may be replaced tomorrow, and
    every classifier writes through here; a row asserting that Intel's stake in
    Vuzix is a stake in Intel cannot be true no matter what produced it.
    """
    detected = iso_utc(detected_at) or _now()
    stamp = iso_utc(recorded_at) or detected
    written = 0
    with connect() as conn:
        for e in entries:
            if e.get("is_subject") is False:
                raise ValueError(
                    f"refusing to record {e.get('accession')} as a stake in "
                    f"{e.get('subject_ticker')}: the filing header names "
                    f"{e.get('subject_name')!r} as the subject company. This "
                    f"is a filing {e.get('subject_ticker')} made about someone "
                    f"else, and recording it would invent an activist.")
            cur = conn.execute(
                """INSERT OR IGNORE INTO activist_filing
                   (accession, subject_ticker, subject_cik, subject_name,
                    filer_name, filer_cik, form, is_amendment, filing_date,
                    accepted_at, detected_at, subject_verified, url,
                    recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (e["accession"], e["subject_ticker"], e.get("subject_cik"),
                 e.get("subject_name"), e.get("filer_name"), e.get("filer_cik"),
                 e["form"], 1 if e.get("is_amendment") else 0,
                 e["filing_date"], iso_utc(e.get("accepted_at")),
                 iso_utc(e.get("detected_at")) or detected,
                 1 if e.get("subject_verified") else 0, e.get("url"), stamp))
            written += cur.rowcount
    return written


def _latency(row: sqlite3.Row) -> Optional[float]:
    if not row["accepted_at"] or not row["detected_at"]:
        return None
    accepted = datetime.fromisoformat(row["accepted_at"].replace("Z", "+00:00"))
    detected = datetime.fromisoformat(row["detected_at"].replace("Z", "+00:00"))
    return (detected - accepted).total_seconds()


def activist_filings_as_of(as_of: str,
                           ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """13D events as they were known on `as_of`, newest filing first.

    Both clauses again: `filing_date <= as_of` keeps out filings that had not
    happened, `recorded_at <= as_of` keeps out ones we had not yet seen. The
    second is what stops a study of 13D reactions from standing in 2019 holding
    a list we only assembled in 2026.

    `latency_seconds` and `is_backfill` are derived here rather than stored.
    They are arithmetic on two recorded facts, and a stored copy is a number
    that can end up disagreeing with the columns it came from. Both are None
    when the acceptance time is unknown -- an unknown latency is not a fast one.
    """
    sql = ("SELECT * FROM activist_filing "
           "WHERE filing_date <= ? AND date(recorded_at) <= ?")
    params: List[Any] = [as_of, as_of]
    if ticker:
        sql += " AND subject_ticker = ?"
        params.append(ticker)
    sql += " ORDER BY filing_date DESC, accession"

    out = []
    with connect() as conn:
        for row in conn.execute(sql, params).fetchall():
            latency = _latency(row)
            out.append({
                **dict(row),
                "is_amendment": bool(row["is_amendment"]),
                "subject_verified": bool(row["subject_verified"]),
                "latency_seconds": latency,
                "is_backfill": None if latency is None
                else latency > BACKFILL_AFTER_SECONDS,
            })
    return out


def known_activist_accessions(ticker: str) -> set:
    """Every accession already recorded for `ticker`, regardless of date.

    Deliberately not an as-of read. This answers "have we already reported
    this?", which is a question about our own bookkeeping and not about what a
    simulation standing on some past date was entitled to know. Filtering it by
    `recorded_at` would make the watcher re-announce old filings.
    """
    with connect() as conn:
        return {r["accession"] for r in conn.execute(
            "SELECT accession FROM activist_filing WHERE subject_ticker = ?",
            (ticker,)).fetchall()}


# ---------------------------------------------------------------- run log

_ACTIVE_RUN: Dict[str, Any] = {}


def reporters_since(since: str, as_of: str) -> List[Dict[str, Any]]:
    """Names whose vendor actual was recorded in a window, as known on `as_of`.

    The consensus recorder captures the vendor's own reported figure within
    days of each print, so this is the store's own answer to "who just
    reported" -- available without asking EDGAR about every company in the
    universe to find out that most of them did not.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT ticker, fiscal_period, MAX(as_of_date) AS as_of_date
               FROM consensus_snapshot
               WHERE eps_actual IS NOT NULL
                 AND as_of_date BETWEEN ? AND ?
                 AND date(recorded_at) <= ?
               GROUP BY ticker, fiscal_period
               ORDER BY ticker""", (since, as_of, as_of)).fetchall()
    return [dict(r) for r in rows]


def filed_periods(as_of: str) -> set:
    """Every (ticker, fiscal period) an order has already been filed for.

    One print is one trade. The signal stays fresh for weeks, so without this
    the same name is proposed every session until the window closes -- forty-
    five nights of the same position live, and one earnings event counted as
    forty-five independent trades in any study of it.

    Strictly before `as_of`, not up to it. A re-run of today is recomputing
    today's decision, and counting what today already filed would leave every
    re-scan empty -- and silently disable the supersede report, which exists to
    compare exactly those two answers.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT ticker, fiscal_period FROM paper_order
               WHERE accepted = 1 AND fiscal_period IS NOT NULL
                 AND as_of_date < ? AND date(recorded_at) <= ?""",
            (as_of, as_of)).fetchall()
    return {(r["ticker"], r["fiscal_period"]) for r in rows}


def has_consensus_history(as_of: str) -> bool:
    """Whether the consensus recorder had written anything by `as_of`.

    The difference between "nothing reported in the last six weeks" and "we
    have not been watching". The first is information and the scanner should
    act on it; the second is a young store, and treating it as information
    would report a quiet tape every night until the recorder catches up.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT 1 FROM consensus_snapshot
               WHERE date(recorded_at) <= ? LIMIT 1""", (as_of,)).fetchone()
    return row is not None


def record_paper_orders(as_of_date: str, candidates: Iterable[Dict[str, Any]],
                        rejected: Iterable[Dict[str, Any]] = (),
                        regime: Optional[str] = None,
                        gross_target: Optional[float] = None,
                        recorded_at: Optional[str] = None) -> int:
    """File one scan. Idempotent within a day: a rerun must not double a book.

    Accepted and rejected go into the same table because they answer the same
    question at different times -- "what did this scan see, and what did it do
    about it" -- and separating them is how a rejection quietly stops being
    part of the record.
    """
    stamp = recorded_at or _now()
    rows = [(c, 1) for c in candidates] + [(r, 0) for r in rejected]
    written = 0
    with connect() as conn:
        for row, accepted in rows:
            cur = conn.execute(
                """INSERT OR IGNORE INTO paper_order
                   (as_of_date, ticker, accepted, reason, side, fiscal_period,
                    sue, expected_edge_bps, cost_bps, net_edge_bps,
                    target_dollars, participation, spread, spread_resolved,
                    rank, regime, gross_target, intended_session, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (as_of_date, row["ticker"], accepted, row.get("reason"),
                 row.get("side"), row.get("fiscal_period"), row.get("sue"),
                 row.get("expected_edge_bps"), row.get("cost_bps"),
                 row.get("net_edge_bps"), row.get("target_dollars"),
                 row.get("participation"), row.get("spread"),
                 1 if row.get("spread_resolved") else 0, row.get("rank"),
                 regime, gross_target, row.get("intended_session"), stamp))
            written += cur.rowcount or 0
    return written


def paper_orders_as_of(as_of: str, accepted_only: bool = False
                       ) -> List[Dict[str, Any]]:
    """Orders decided on or before `as_of`, as they were decided.

    `recorded_at` is filtered like everywhere else: a scan run today is not
    something last week's reader knew.
    """
    sql = """SELECT * FROM paper_order
             WHERE as_of_date <= ? AND date(recorded_at) <= ?"""
    params: tuple = (as_of, as_of)
    if accepted_only:
        sql += " AND accepted = 1"
    sql += " ORDER BY as_of_date, rank IS NULL, rank, ticker"
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{**dict(r), "accepted": bool(r["accepted"]),
             "spread_resolved": bool(r["spread_resolved"])} for r in rows]


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
    caveat. Only a finished run counts -- a crashed process leaves a started
    row with no finish, and a failed fetch leaves a finish with no data, and
    neither is coverage.

    'closed' counts alongside 'ok'. The exchange shuts about ten weekdays a
    year, and a holiday listed as a permanent hole is ten false alarms
    annually, which is how the one real gap stops being noticed.
    """
    from datetime import date, timedelta

    with connect() as conn:
        covered = {
            r["as_of_date"] for r in conn.execute(
                """SELECT DISTINCT as_of_date FROM run_log
                   WHERE job = ? AND status IN ('ok', 'closed')
                     AND finished_at IS NOT NULL
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
