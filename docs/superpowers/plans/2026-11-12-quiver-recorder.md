# QuiverQuant Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A nightly batch job, `research-quiver`, that records QuiverQuant's seven published feeds into `pit.db` and three slow-moving lists once per ISO week, never overwriting a row.

**Architecture:** `research/pit_store.py` gains three tables and four functions (write, revise, as-of read, weekly snapshot). `research/quiver_record.py` holds the feed table, the row-key rules and the two stages, and calls `tools.quiver_server.client.QuiverWebClient` directly. Compose, cron, status and Discord wiring follow the pattern of `research-copy`.

**Tech Stack:** Python 3.12, sqlite3 (stdlib), pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-12-quiver-recorder-design.md`

## Global Constraints

- Every date is UTC: `datetime.now(timezone.utc)`, never `date.today()` or a local clock.
- Point in time: rows carry `recorded_at`; reads filter on `date(recorded_at) <= as_of`.
- Nothing is overwritten. A changed row is a `quiver_revision`; `first_seen_at` is written once.
- Fail loud: a failed or empty feed is named with its reason, the run is `partial`, the process exits non-zero. No default values, no swallowed exceptions.
- Mock data only in tests. No new dependency.
- A rule is pinned only when removing it fails a test by name (mutation check in Task 5).
- Commit messages say what changed and why. No attribution lines, no mention of Claude or AI.
- Offline suite command: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --ignore=testing/output -p no:cacheprovider`
- Lint: `uvx ruff@0.15.14 check .` (the venv has no ruff; CI pins this version).
- Work on branch `quiver-recorder` (already holds the spec commit).

## File Structure

| File | Responsibility |
| --- | --- |
| `research/pit_store.py` (modify) | Schema for `quiver_record`, `quiver_revision`, `quiver_snapshot`; `record_quiver_rows`, `quiver_records_as_of`, `record_quiver_snapshot`, `has_quiver_snapshot`; `missing_days` learns `daily`. |
| `research/quiver_record.py` (create) | `Feed`, `FEEDS`, `LISTS`, `keyed_rows`, `iso_week`, `record_feeds`, `record_lists`, `run`, `main`. |
| `research/status.py` (modify) | Two `JOBS` entries. |
| `research/discord_post.py` (modify) | `JOB_CHANNELS["research-quiver"]`, `_quiver` card. |
| `deploy/docker-compose.yml` (modify) | `research-quiver` service and its cron comment. |
| `deploy/README.md` (modify) | Cron line, env table row, a short section. |
| `README.md` (modify) | Job table row. |
| `testing/test_quiver_store.py` (create) | Store behaviour. |
| `testing/test_quiver_record.py` (create) | Keys, stages, entry point. |
| `testing/test_discord_wiring.py` (modify) | Scenario for `research-quiver`. |

---

### Task 1: Store tables, writes, as-of reads, `daily` schedule

**Files:**
- Modify: `research/pit_store.py` (imports at lines 34-38; `_SCHEMA` ends just after `ON xbrl_class_eps(accession);`; `missing_days` at ~line 1484; append new functions at the end of the file)
- Test: `testing/test_quiver_store.py` (create)

**Interfaces:**
- Produces:
  - `record_quiver_rows(dataset: str, rows: Iterable[dict], *, volatile: Sequence[str] = (), recorded_at: Optional[str] = None) -> Tuple[int, int]` returning `(new, revised)`. Each row dict has `row_key: str`, `ticker: Optional[str]`, `event_date: Optional[str]`, `payload: dict`.
  - `quiver_records_as_of(dataset: str, as_of: str, ticker: Optional[str] = None) -> List[dict]`; each dict has `dataset, row_key, ticker, event_date, payload (dict), first_seen_at, revised (bool)`.
  - `record_quiver_snapshot(dataset: str, as_of_week: str, rows: Iterable[dict], recorded_at: Optional[str] = None) -> int`; rows have `row_key, ticker, payload`.
  - `has_quiver_snapshot(dataset: str, as_of_week: str) -> bool`
  - `missing_days(..., every="daily")` counts every calendar day as due.

- [ ] **Step 1: Write the failing tests**

Create `testing/test_quiver_store.py`:

```python
"""The QuiverQuant tables in pit.db: first sight, revisions, as-of reads.

Quiver publishes rolling windows, so these tables are the only record that a
row was ever there and when we first saw it. Nothing here may overwrite.
"""
import pytest

from research import pit_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def row(key="k1", ticker="AAPL", event_date="2026-11-10", **payload):
    body = {"ticker": ticker, "amount": "$1,000", **payload}
    return {"row_key": key, "ticker": ticker, "event_date": event_date,
            "payload": body}


def stored(store, dataset="layoff"):
    with store.connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM quiver_record WHERE dataset = ? ORDER BY row_key",
            (dataset,))]


def revisions(store, dataset="layoff"):
    with store.connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM quiver_revision WHERE dataset = ? ORDER BY rowid",
            (dataset,))]


def test_first_sight_writes_every_row_with_its_first_seen_time(store):
    new, revised = store.record_quiver_rows(
        "layoff", [row("a"), row("b")], recorded_at="2026-11-12T06:30:00Z")
    assert (new, revised) == (2, 0)
    rows = stored(store)
    assert [r["row_key"] for r in rows] == ["a", "b"]
    assert all(r["first_seen_at"] == r["recorded_at"] == "2026-11-12T06:30:00Z"
               for r in rows)


def test_a_second_run_over_the_same_rows_writes_nothing(store):
    store.record_quiver_rows("layoff", [row("a")],
                             recorded_at="2026-11-12T06:30:00Z")
    assert store.record_quiver_rows(
        "layoff", [row("a")], recorded_at="2026-11-13T06:30:00Z") == (0, 0)
    assert len(stored(store)) == 1
    assert revisions(store) == []


def test_a_changed_payload_files_one_revision_and_keeps_the_original(store):
    store.record_quiver_rows("layoff", [row("a", workers=100)],
                             recorded_at="2026-11-12T06:30:00Z")
    assert store.record_quiver_rows(
        "layoff", [row("a", workers=120)],
        recorded_at="2026-11-13T06:30:00Z") == (0, 1)
    assert '"workers":100' in stored(store)[0]["payload"]
    [rev] = revisions(store)
    assert '"workers":100' in rev["old_payload"]
    assert '"workers":120' in rev["new_payload"]
    assert rev["noticed_at"] == "2026-11-13T06:30:00Z"


def test_the_same_change_seen_again_is_not_a_second_revision(store):
    store.record_quiver_rows("layoff", [row("a", workers=100)],
                             recorded_at="2026-11-12T06:30:00Z")
    store.record_quiver_rows("layoff", [row("a", workers=120)],
                             recorded_at="2026-11-13T06:30:00Z")
    assert store.record_quiver_rows(
        "layoff", [row("a", workers=120)],
        recorded_at="2026-11-14T06:30:00Z") == (0, 0)
    assert len(revisions(store)) == 1


def test_a_change_only_in_a_volatile_field_files_nothing(store):
    volatile = ("list", "net_worth_estimate")
    store.record_quiver_rows(
        "insider_trade", [row("a", list="ceo", net_worth_estimate=5)],
        volatile=volatile, recorded_at="2026-11-12T06:30:00Z")
    assert store.record_quiver_rows(
        "insider_trade", [row("a", list="all", net_worth_estimate=9)],
        volatile=volatile, recorded_at="2026-11-13T06:30:00Z") == (0, 0)
    assert revisions(store, "insider_trade") == []


def test_first_seen_at_never_moves(store):
    store.record_quiver_rows("layoff", [row("a", workers=100)],
                             recorded_at="2026-11-12T06:30:00Z")
    store.record_quiver_rows("layoff", [row("a", workers=120)],
                             recorded_at="2026-11-13T06:30:00Z")
    store.record_quiver_rows("layoff", [row("a", workers=130)],
                             recorded_at="2026-11-14T06:30:00Z")
    [r] = stored(store)
    assert r["first_seen_at"] == r["recorded_at"] == "2026-11-12T06:30:00Z"


def test_an_as_of_read_excludes_a_row_recorded_later(store):
    store.record_quiver_rows("layoff", [row("early", event_date="2026-11-09")],
                             recorded_at="2026-11-10T06:30:00Z")
    store.record_quiver_rows("layoff", [row("late", event_date="2026-11-09")],
                             recorded_at="2026-11-12T06:30:00Z")
    keys = [r["row_key"] for r in
            store.quiver_records_as_of("layoff", "2026-11-11")]
    assert keys == ["early"]


def test_an_as_of_read_applies_only_revisions_noticed_by_then(store):
    store.record_quiver_rows("layoff", [row("a", ticker=None)],
                             recorded_at="2026-11-10T06:30:00Z")
    store.record_quiver_rows("layoff", [row("a", ticker="ACME")],
                             recorded_at="2026-11-12T06:30:00Z")
    [before] = store.quiver_records_as_of("layoff", "2026-11-11")
    [after] = store.quiver_records_as_of("layoff", "2026-11-12")
    assert before["ticker"] is None and before["revised"] is False
    assert after["ticker"] == "ACME" and after["revised"] is True
    assert after["payload"]["ticker"] == "ACME"
    assert after["first_seen_at"] == "2026-11-10T06:30:00Z"
    assert [r["row_key"] for r in store.quiver_records_as_of(
        "layoff", "2026-11-12", ticker="acme")] == ["a"]
    assert store.quiver_records_as_of("layoff", "2026-11-11",
                                      ticker="ACME") == []


def test_a_snapshot_is_written_once_per_week(store):
    rows = [{"row_key": "m1", "ticker": None, "payload": {"name": "A"}}]
    assert store.record_quiver_snapshot(
        "congress_net_worth", "2026-W46", rows,
        recorded_at="2026-11-12T06:30:00Z") == 1
    assert store.record_quiver_snapshot(
        "congress_net_worth", "2026-W46", rows,
        recorded_at="2026-11-13T06:30:00Z") == 0
    assert store.has_quiver_snapshot("congress_net_worth", "2026-W46")
    assert not store.has_quiver_snapshot("congress_net_worth", "2026-W47")


def _ran(store, job, day):
    run_id = store.start_run(job, as_of_date=day)
    store.finish_run(rows_written=1, status="ok", run_id=run_id)


def test_a_daily_job_misses_a_sunday(store):
    # 2026-11-06 is a Friday; the 8th is the Sunday with no run.
    for day in ("2026-11-06", "2026-11-07", "2026-11-09"):
        _ran(store, "quiver_feeds", day)
    assert store.missing_days("quiver_feeds", "2026-11-06", "2026-11-09",
                              every="daily") == ["2026-11-08"]


def test_the_weekday_default_is_unchanged(store):
    _ran(store, "scan", "2026-11-06")
    assert store.missing_days("scan", "2026-11-06", "2026-11-09",
                              every="weeknight") == ["2026-11-09"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_store.py -q -p no:cacheprovider`
Expected: FAIL. Most tests fail with `AttributeError: module 'research.pit_store' has no attribute 'record_quiver_rows'`; `test_a_daily_job_misses_a_sunday` fails because `["2026-11-08"]` is not reported (weekend days are not due). `test_the_weekday_default_is_unchanged` passes.

- [ ] **Step 3: Implement**

In `research/pit_store.py`, change the imports (lines 34-38) to:

```python
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
```

Append to `_SCHEMA`, immediately before its closing `"""` (after `ON xbrl_class_eps(accession);`):

```sql

-- QuiverQuant, scraped by research/quiver_record.py. Quiver publishes rolling
-- windows, so a row that leaves the window is gone from the source. These are
-- the only record that it was there and when we first saw it.
CREATE TABLE IF NOT EXISTS quiver_record (
    dataset       TEXT NOT NULL,
    row_key       TEXT NOT NULL,
    ticker        TEXT,
    event_date    TEXT,
    payload       TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (dataset, row_key)
);
CREATE INDEX IF NOT EXISTS idx_quiver_record_ticker
    ON quiver_record(ticker, dataset);

-- A row seen again with a different payload. The stored row stands.
CREATE TABLE IF NOT EXISTS quiver_revision (
    dataset     TEXT NOT NULL,
    row_key     TEXT NOT NULL,
    old_payload TEXT NOT NULL,
    new_payload TEXT NOT NULL,
    new_ticker  TEXT,
    noticed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quiver_revision_key
    ON quiver_revision(dataset, row_key);

-- Lists that change slowly, copied once per ISO week.
CREATE TABLE IF NOT EXISTS quiver_snapshot (
    dataset     TEXT NOT NULL,
    as_of_week  TEXT NOT NULL,
    row_key     TEXT NOT NULL,
    ticker      TEXT,
    payload     TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (dataset, as_of_week, row_key)
);
```

In `missing_days`, replace the schedule branch:

```python
        if every == "Saturday":
            due = day.weekday() == 5
        elif every == "monthly":
            due = day.day == 1
        else:
            due = day.weekday() < 5
```

with:

```python
        if every == "Saturday":
            due = day.weekday() == 5
        elif every == "monthly":
            due = day.day == 1
        elif every == "daily":
            due = True
        else:
            due = day.weekday() < 5
```

and add one sentence to its docstring, after "A Saturday job is due on Saturdays and a monthly one on the 1st;": `a daily one every calendar day, weekends included;`.

Append to the end of `research/pit_store.py`:

```python
# ------------------------------------------------------------ QuiverQuant
#
# Written by research/quiver_record.py. A payload is stored as canonical JSON
# so two nights' copies of one row compare equal when nothing changed.

def _quiver_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)


def _quiver_compared(text: str, volatile: Sequence[str]) -> str:
    body = json.loads(text)
    return _quiver_json({k: v for k, v in body.items() if k not in volatile})


def record_quiver_rows(dataset: str, rows: Iterable[Dict[str, Any]], *,
                       volatile: Sequence[str] = (),
                       recorded_at: Optional[str] = None) -> Tuple[int, int]:
    """Append Quiver rows. Returns (new, revised).

    A row seen before keeps its first copy and its `first_seen_at`. When its
    payload differs from the latest one known -- the newest revision, else the
    stored row -- the difference is filed in `quiver_revision`, once. Fields
    in `volatile` (an estimate that moves with prices, the list a row came
    from) are stored but ignored when deciding whether a row changed, or every
    night would file a revision for every row.
    """
    stamp = iso_utc(recorded_at) or _now()
    new = revised = 0
    with connect() as conn:
        for row in rows:
            payload = _quiver_json(row["payload"])
            held = conn.execute(
                "SELECT payload FROM quiver_record "
                "WHERE dataset = ? AND row_key = ?",
                (dataset, row["row_key"])).fetchone()
            if held is None:
                conn.execute(
                    """INSERT INTO quiver_record
                       (dataset, row_key, ticker, event_date, payload,
                        first_seen_at, recorded_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (dataset, row["row_key"], row.get("ticker"),
                     row.get("event_date"), payload, stamp, stamp))
                new += 1
                continue
            latest = conn.execute(
                """SELECT new_payload FROM quiver_revision
                   WHERE dataset = ? AND row_key = ?
                   ORDER BY rowid DESC LIMIT 1""",
                (dataset, row["row_key"])).fetchone()
            known = latest["new_payload"] if latest else held["payload"]
            if _quiver_compared(known, volatile) == _quiver_compared(
                    payload, volatile):
                continue
            conn.execute(
                """INSERT INTO quiver_revision
                   (dataset, row_key, old_payload, new_payload, new_ticker,
                    noticed_at)
                   VALUES (?,?,?,?,?,?)""",
                (dataset, row["row_key"], known, payload, row.get("ticker"),
                 stamp))
            revised += 1
    return new, revised


def quiver_records_as_of(dataset: str, as_of: str,
                         ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """Quiver rows as they were known on `as_of`, newest event first.

    A row recorded after `as_of` is left out, and so is a revision noticed
    after it: a ticker Quiver matched to a layoff a week later is found under
    that ticker from the night the match was noticed, never before.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT * FROM quiver_record
               WHERE dataset = ? AND date(recorded_at) <= ?""",
            (dataset, as_of)).fetchall()
        changes = conn.execute(
            """SELECT row_key, new_payload, new_ticker FROM quiver_revision
               WHERE dataset = ? AND date(noticed_at) <= ?
               ORDER BY rowid""",
            (dataset, as_of)).fetchall()
    latest = {change["row_key"]: change for change in changes}
    wanted = ticker.upper() if ticker else None
    out: List[Dict[str, Any]] = []
    for row in rows:
        change = latest.get(row["row_key"])
        row_ticker = change["new_ticker"] if change else row["ticker"]
        if wanted and (row_ticker or "").upper() != wanted:
            continue
        out.append({
            "dataset": dataset,
            "row_key": row["row_key"],
            "ticker": row_ticker,
            "event_date": row["event_date"],
            "payload": json.loads(change["new_payload"] if change
                                  else row["payload"]),
            "first_seen_at": row["first_seen_at"],
            "revised": change is not None,
        })
    out.sort(key=lambda r: (r["event_date"] or "", r["row_key"]), reverse=True)
    return out


def record_quiver_snapshot(dataset: str, as_of_week: str,
                           rows: Iterable[Dict[str, Any]],
                           recorded_at: Optional[str] = None) -> int:
    """One week's copy of a slow-moving list. A second copy of the same week
    writes nothing."""
    stamp = iso_utc(recorded_at) or _now()
    written = 0
    with connect() as conn:
        for row in rows:
            cur = conn.execute(
                """INSERT OR IGNORE INTO quiver_snapshot
                   (dataset, as_of_week, row_key, ticker, payload, recorded_at)
                   VALUES (?,?,?,?,?,?)""",
                (dataset, as_of_week, row["row_key"], row.get("ticker"),
                 _quiver_json(row["payload"]), stamp))
            written += cur.rowcount
    return written


def has_quiver_snapshot(dataset: str, as_of_week: str) -> bool:
    with connect() as conn:
        return conn.execute(
            "SELECT 1 FROM quiver_snapshot "
            "WHERE dataset = ? AND as_of_week = ? LIMIT 1",
            (dataset, as_of_week)).fetchone() is not None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_store.py testing/test_status.py testing/test_pit_store*.py -q -p no:cacheprovider`
Expected: all pass (the `test_pit_store*` glob may match nothing; that is fine).

- [ ] **Step 5: Commit**

```bash
git add research/pit_store.py testing/test_quiver_store.py
git commit -m "Give pit.db a place for QuiverQuant rows

Three tables: quiver_record keeps the first copy of each row and when it was
first seen, quiver_revision keeps each later change once, and quiver_snapshot
holds one copy per ISO week of the slow-moving lists. As-of reads apply only
the revisions noticed by that date. missing_days learns a daily schedule, so a
job that runs every night is due at weekends too."
```

---

### Task 2: Feeds and row keys

**Files:**
- Create: `research/quiver_record.py`
- Test: `testing/test_quiver_record.py` (create)

**Interfaces:**
- Produces:
  - `Feed` (frozen dataclass): `dataset: str, method: str, rows_key: str, identity: Tuple[str, ...] = (), id_field: Optional[str] = None, ticker_field: Optional[str] = "ticker", date_field: Optional[str] = None, volatile: Tuple[str, ...] = ()`
  - `FEEDS: Tuple[Feed, ...]` (7) and `LISTS: Tuple[Feed, ...]` (3)
  - `keyed_rows(feed: Feed, rows: Sequence[dict]) -> List[dict]`; output dicts have `row_key, ticker, event_date, payload` (the shape `pit_store.record_quiver_rows` takes).

- [ ] **Step 1: Write the failing tests**

Create `testing/test_quiver_record.py`:

```python
"""The nightly QuiverQuant recorder: row keys, stages and the entry point."""
from datetime import datetime, timedelta, timezone

import pytest

from research import pit_store
from research import quiver_record as qr
from tools.quiver_server.client import QuiverError


def feed(name):
    return next(f for f in qr.FEEDS + qr.LISTS if f.dataset == name)


def test_the_feeds_are_the_seven_the_spec_names():
    assert [f.dataset for f in qr.FEEDS] == [
        "congress_trade", "insider_trade", "layoff", "contract_award",
        "gov_spending", "lobbying_filing", "revolving_door"]
    assert [f.dataset for f in qr.LISTS] == [
        "congress_net_worth", "election_contribution", "lobbying_top_spender"]


def test_every_feed_names_how_its_rows_are_identified():
    for f in qr.FEEDS + qr.LISTS:
        assert f.id_field or f.identity, f.dataset


def test_congress_rows_are_keyed_by_quivers_own_id():
    [r] = qr.keyed_rows(feed("congress_trade"), [
        {"tx_id": "Senate-C001047-750", "ticker": "aapl",
         "filed_date": "2026-11-04"}])
    assert r["row_key"] == "Senate-C001047-750"
    assert r["ticker"] == "AAPL"
    assert r["event_date"] == "2026-11-04"


def test_a_congress_row_without_an_id_is_refused():
    with pytest.raises(ValueError, match="tx_id"):
        qr.keyed_rows(feed("congress_trade"), [{"ticker": "AAPL"}])


def test_two_identical_lobbying_filings_are_two_rows():
    filing = {"ticker": "EVGO", "amount": "$30,000", "date": "2026-11-11"}
    keys = [r["row_key"] for r in
            qr.keyed_rows(feed("lobbying_filing"), [filing, dict(filing)])]
    assert len(set(keys)) == 2


def test_the_occurrence_index_is_stable_across_nights():
    a = {"ticker": "EVGO", "amount": "$30,000", "date": "2026-11-11"}
    b = {"ticker": "META", "amount": "$10,000", "date": "2026-11-11"}
    first = {r["row_key"] for r in
             qr.keyed_rows(feed("lobbying_filing"), [a, dict(a), b])}
    second = {r["row_key"] for r in
              qr.keyed_rows(feed("lobbying_filing"), [b, a, dict(a)])}
    assert first == second


def test_a_field_quiver_fills_in_later_does_not_change_the_key():
    base = {"employer": "Acme", "notice_date": "2026-11-09", "city": "Austin",
            "state": "TX", "workers": 120, "kind": "Layoff", "ticker": None,
            "ticker_match": None}
    later = {**base, "ticker": "ACME", "ticker_match": "fuzzy"}
    [first] = qr.keyed_rows(feed("layoff"), [base])
    [second] = qr.keyed_rows(feed("layoff"), [later])
    assert first["row_key"] == second["row_key"]
    assert first["ticker"] is None and second["ticker"] == "ACME"


def test_a_contract_award_carries_no_ticker():
    [r] = qr.keyed_rows(feed("contract_award"), [
        {"recipient": "Acme", "date_awarded": "2026-11-05", "amount": "$5",
         "agency": "DOD", "description": "parts"}])
    assert r["ticker"] is None
    assert r["event_date"] == "2026-11-05"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_record.py -q -p no:cacheprovider`
Expected: collection error, `ModuleNotFoundError: No module named 'research.quiver_record'`.

- [ ] **Step 3: Implement**

Create `research/quiver_record.py`:

```python
"""A nightly record of what QuiverQuant publishes.

Quiver's pages publish rolling windows: the latest 300 congressional trades,
the latest 100 contract awards, about 2,000 WARN notices. A row that leaves the
window is gone from the source, and nothing else records when it first
appeared. This job writes each feed into pit.db every night and three
slow-moving lists once per ISO week. It never overwrites a row.

Each feed fails alone and by name. A feed that parses no rows is a failure,
never a quiet night: "nothing new" means rows were read and none were new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from research import pit_store
from tools.quiver_server.client import QuiverError, QuiverWebClient

JOB = "research-quiver"
FEEDS_STAGE = "quiver_feeds"
LISTS_STAGE = "quiver_lists"


@dataclass(frozen=True)
class Feed:
    dataset: str
    method: str                       # QuiverWebClient method, called with no arguments
    rows_key: str                     # the list of rows in that method's result
    identity: Tuple[str, ...] = ()    # fields hashed into row_key
    id_field: Optional[str] = None    # or Quiver's own id, used as row_key
    ticker_field: Optional[str] = "ticker"
    date_field: Optional[str] = None
    volatile: Tuple[str, ...] = ()    # stored, ignored when comparing


# Identity fields leave out anything Quiver may fill in later (a ticker match,
# its quality), so filling it in is a revision of the row and not a new row.
FEEDS: Tuple[Feed, ...] = (
    Feed("congress_trade", "get_congress_trades", "trades",
         id_field="tx_id", date_field="filed_date"),
    # The fields the client already deduplicates its two insider lists on.
    Feed("insider_trade", "get_insider_trades", "trades",
         identity=("owner", "ticker", "transaction_date", "shares", "tx_type"),
         date_field="filed_date",
         volatile=("list", "net_worth_estimate")),
    Feed("layoff", "get_layoffs", "notices",
         identity=("employer", "notice_date", "city", "state", "workers",
                   "kind"),
         date_field="notice_date"),
    Feed("contract_award", "get_gov_contract_awards", "awards",
         identity=("recipient", "date_awarded", "amount", "agency",
                   "description"),
         ticker_field=None, date_field="date_awarded"),
    Feed("gov_spending", "get_gov_spending", "transactions",
         identity=("award_id", "date", "amount_usd", "recipient", "agency"),
         date_field="date"),
    Feed("lobbying_filing", "get_lobbying_filings", "filings",
         identity=("ticker", "amount", "date"), date_field="date"),
    Feed("revolving_door", "get_revolving_door", "registrations",
         identity=("lobbyist", "client", "firm", "filed_date", "filing_type",
                   "period"),
         date_field="filed_date", volatile=("is_new",)),
)

LISTS: Tuple[Feed, ...] = (
    Feed("congress_net_worth", "get_congress_net_worth", "members",
         identity=("bioguide_id", "name"), ticker_field=None),
    Feed("election_contribution", "get_election_contributions",
         "contributions", identity=("ticker", "company", "source", "cycle")),
    Feed("lobbying_top_spender", "get_lobbying_filings", "top_spenders",
         identity=("ticker",)),
)


def keyed_rows(feed: Feed, rows: Sequence[Dict[str, Any]]
               ) -> List[Dict[str, Any]]:
    """Each row with its key, ticker and event date, ready for pit_store.

    A hashed key carries an occurrence index: the n-th row in one fetch with
    the same identity gets n in its hash, so two identical lobbying filings
    stay two rows. The rows are identical, so the index is stable across
    nights whichever of them Quiver lists first.
    """
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if feed.id_field:
            raw = row.get(feed.id_field)
            if not raw:
                raise ValueError(
                    f"{feed.dataset}: a row has no {feed.id_field}, so it "
                    f"cannot be told apart from any other: {row!r}")
            key = str(raw)
        else:
            identity = json.dumps([row.get(f) for f in feed.identity],
                                  default=str)
            occurrence = seen.get(identity, 0)
            seen[identity] = occurrence + 1
            key = hashlib.sha256(
                f"{identity}#{occurrence}".encode()).hexdigest()[:32]
        ticker = row.get(feed.ticker_field) if feed.ticker_field else None
        out.append({
            "row_key": key,
            "ticker": str(ticker).upper() if ticker else None,
            "event_date": row.get(feed.date_field) if feed.date_field else None,
            "payload": row,
        })
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_record.py -q -p no:cacheprovider`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add research/quiver_record.py testing/test_quiver_record.py
git commit -m "Key each QuiverQuant row by what identifies it

Congress trades carry Quiver's own id. Every other feed hashes the fields that
identify a row, leaving out ones Quiver fills in later such as a ticker match,
so a later match is a revision and not a second row. Identical rows in one
fetch get an occurrence index, so two lobbying filings with the same ticker,
amount and date stay two rows."
```

---

### Task 3: The two stages

**Files:**
- Modify: `research/quiver_record.py` (append)
- Test: `testing/test_quiver_record.py` (append)

**Interfaces:**
- Consumes: `FEEDS`, `LISTS`, `keyed_rows` (Task 2); `pit_store.record_quiver_rows`, `record_quiver_snapshot`, `has_quiver_snapshot`, `start_run`, `finish_run`, `iso_utc`, `init_schema` (Task 1 and existing).
- Produces:
  - `iso_week(now: datetime) -> str` e.g. `"2026-W46"`
  - `record_feeds(client, now: datetime) -> dict` with `status, written, feeds: {dataset: {read, new, revised}}, failed: {dataset: reason}`
  - `record_lists(client, now: datetime) -> dict` with `status, week, written, lists: {dataset: {read, written}}, already_held: [dataset], failed: {dataset: reason}`
  - `run(client=None, now: Optional[datetime] = None) -> dict` with `status ("ok"|"partial"|"failed"), as_of, written, quiver_feeds, quiver_lists, failed: {"stage/dataset": reason}`

- [ ] **Step 1: Write the failing tests**

Append to `testing/test_quiver_record.py`:

```python
# A Thursday, ISO week 2026-W46.
NOW = datetime(2026, 11, 12, 6, 30, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


class Scripted:
    """Answers each client method from a dict; an exception value is raised."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.closed = False

    def __getattr__(self, name):
        if not name.startswith("get_"):
            raise AttributeError(name)

        def call(**kwargs):
            self.calls.append(name)
            answer = self.answers[name]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return call

    def close(self):
        self.closed = True


def answers(**overrides):
    base = {
        "get_congress_trades": {"trades": [
            {"tx_id": "T1", "ticker": "AAPL", "filed_date": "2026-11-10"}]},
        "get_insider_trades": {"trades": [
            {"owner": "Jane", "ticker": "MSFT", "transaction_date": "2026-11-01",
             "shares": 10, "tx_type": "Sale", "filed_date": "2026-11-03",
             "list": "all", "net_worth_estimate": 1}]},
        "get_layoffs": {"notices": [
            {"employer": "Acme", "notice_date": "2026-11-09", "city": "Austin",
             "state": "TX", "workers": 120, "kind": "Layoff", "ticker": None}]},
        "get_gov_contract_awards": {"awards": [
            {"recipient": "Acme", "date_awarded": "2026-11-05", "amount": "$5",
             "agency": "DOD", "description": "parts"}]},
        "get_gov_spending": {"transactions": [
            {"award_id": "A1", "date": "2026-10-01", "amount_usd": 5.0,
             "recipient": "Acme", "agency": "DOD", "ticker": "ACME"}]},
        "get_lobbying_filings": {
            "filings": [{"ticker": "EVGO", "amount": "$30,000",
                         "date": "2026-11-11"}],
            "top_spenders": [{"ticker": "META", "amount": "$5,880,000"}]},
        "get_revolving_door": {"registrations": [
            {"lobbyist": "D", "client": "C", "firm": "F",
             "filed_date": "2026-11-11", "filing_type": "Q3",
             "period": "Q3 2026", "is_new": False, "ticker": None}]},
        "get_congress_net_worth": {"members": [
            {"bioguide_id": "S001229", "name": "Jefferson Shreve",
             "net_worth_usd": 7.2e8}]},
        "get_election_contributions": {"contributions": [
            {"ticker": "SPCX", "company": "SpaceX", "source": "employee",
             "cycle": "2026", "total_usd": 1.0}]},
    }
    base.update(overrides)
    return base


def last_run(store, job):
    with store.connect() as conn:
        return dict(conn.execute(
            "SELECT * FROM run_log WHERE job = ? ORDER BY run_id DESC LIMIT 1",
            (job,)).fetchone())


def test_the_week_is_the_utc_iso_week():
    assert qr.iso_week(NOW) == "2026-W46"


def test_a_full_night_records_every_feed_and_list(store):
    result = qr.run(Scripted(answers()), NOW)
    assert result["status"] == "ok"
    assert result["as_of"] == "2026-11-12"
    assert set(result["quiver_feeds"]["feeds"]) == {f.dataset for f in qr.FEEDS}
    assert set(result["quiver_lists"]["lists"]) == {f.dataset for f in qr.LISTS}
    assert result["written"] == 7 + 3
    assert last_run(store, "quiver_feeds")["status"] == "ok"
    assert last_run(store, "quiver_lists")["status"] == "ok"
    [trade] = store.quiver_records_as_of("congress_trade", "2026-11-12")
    assert trade["first_seen_at"] == "2026-11-12T06:30:00Z"


def test_a_second_night_over_the_same_rows_records_nothing_new(store):
    qr.run(Scripted(answers()), NOW)
    second = qr.run(Scripted(answers()), NOW + timedelta(days=1))
    assert second["status"] == "ok"
    assert second["quiver_feeds"]["written"] == 0
    assert all(c["read"] == 1 and c["new"] == 0
               for c in second["quiver_feeds"]["feeds"].values())


def test_one_failed_feed_is_named_and_the_others_still_record(store):
    result = qr.run(Scripted(answers(
        get_layoffs=QuiverError("rate_limited", "HTTP 429 after one retry"))),
        NOW)
    assert result["status"] == "partial"
    assert result["quiver_feeds"]["failed"]["layoff"].startswith("rate_limited")
    assert "quiver_feeds/layoff" in result["failed"]
    assert store.quiver_records_as_of("congress_trade", "2026-11-12")
    run = last_run(store, "quiver_feeds")
    assert run["status"] == "partial"
    assert "layoff: rate_limited" in run["error"]


def test_an_empty_feed_is_a_failure_not_a_quiet_night(store):
    result = qr.run(Scripted(answers(get_gov_contract_awards={"awards": []})),
                    NOW)
    assert result["status"] == "partial"
    assert result["quiver_feeds"]["failed"]["contract_award"].startswith("empty")


def test_a_feed_with_no_row_list_is_a_parse_failure(store):
    result = qr.run(Scripted(answers(get_layoffs={"unexpected": 1})), NOW)
    assert result["quiver_feeds"]["failed"]["layoff"].startswith("parse_failed")


def test_every_feed_failing_fails_the_stage(store):
    down = QuiverError("provider_unavailable", "connect timeout")
    everything_down = {name: down for name in answers()}
    result = qr.run(Scripted(everything_down), NOW)
    assert result["status"] == "failed"
    assert last_run(store, "quiver_feeds")["status"] == "failed"
    assert last_run(store, "quiver_lists")["status"] == "failed"


def test_weekly_lists_are_written_once_per_week(store):
    first = qr.run(Scripted(answers()), NOW)
    second = qr.run(Scripted(answers()), NOW + timedelta(days=1))
    assert first["quiver_lists"]["written"] == 3
    assert second["quiver_lists"]["written"] == 0
    assert sorted(second["quiver_lists"]["already_held"]) == sorted(
        f.dataset for f in qr.LISTS)
    assert last_run(store, "quiver_lists")["status"] == "ok"


def test_a_failed_list_is_retried_the_next_night(store):
    qr.run(Scripted(answers(get_congress_net_worth=QuiverError(
        "provider_unavailable", "connect timeout"))), NOW)
    second = qr.run(Scripted(answers()), NOW + timedelta(days=1))
    assert second["quiver_lists"]["lists"]["congress_net_worth"]["written"] == 1


def test_a_new_week_is_copied_again(store):
    qr.run(Scripted(answers()), NOW)
    later = qr.run(Scripted(answers()), NOW + timedelta(days=7))
    assert later["quiver_lists"]["week"] == "2026-W47"
    assert later["quiver_lists"]["written"] == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_record.py -q -p no:cacheprovider`
Expected: the new tests fail with `AttributeError: module 'research.quiver_record' has no attribute 'iso_week'` / `'run'`; the 8 tests from Task 2 still pass.

- [ ] **Step 3: Implement**

Append to `research/quiver_record.py`:

```python
def iso_week(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def _fetch(client: Any, feed: Feed) -> List[Dict[str, Any]]:
    result = getattr(client, feed.method)()
    rows = result.get(feed.rows_key) if isinstance(result, dict) else None
    if not isinstance(rows, list):
        raise QuiverError(
            "parse_failed",
            f"{feed.method} returned no {feed.rows_key!r} list")
    if not rows:
        raise QuiverError(
            "empty",
            f"{feed.method} parsed 0 {feed.rows_key}. An empty feed is a "
            f"failure, not a night with nothing new.")
    return rows


def _reason(exc: Exception) -> str:
    if isinstance(exc, QuiverError):
        return f"{exc.reason}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def _stage_status(failures: Dict[str, str], succeeded: int) -> str:
    if not failures:
        return "ok"
    return "partial" if succeeded else "failed"


def _finish(run_id: int, written: int, status: str,
            failures: Dict[str, str]) -> None:
    error = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
    pit_store.finish_run(rows_written=written, status=status,
                         error=error or None, run_id=run_id)


def record_feeds(client: Any, now: datetime) -> Dict[str, Any]:
    """The seven feeds into quiver_record. Each fails alone, by name."""
    stamp = pit_store.iso_utc(now)
    run_id = pit_store.start_run(FEEDS_STAGE, as_of_date=now.date().isoformat())
    counts: Dict[str, Dict[str, int]] = {}
    failures: Dict[str, str] = {}
    for feed in FEEDS:
        try:
            rows = keyed_rows(feed, _fetch(client, feed))
            new, revised = pit_store.record_quiver_rows(
                feed.dataset, rows, volatile=feed.volatile, recorded_at=stamp)
        except Exception as exc:  # noqa: BLE001 - named in the run and the exit
            failures[feed.dataset] = _reason(exc)
            continue
        counts[feed.dataset] = {"read": len(rows), "new": new,
                                "revised": revised}
    written = sum(c["new"] + c["revised"] for c in counts.values())
    status = _stage_status(failures, len(counts))
    _finish(run_id, written, status, failures)
    return {"status": status, "written": written, "feeds": counts,
            "failed": failures}


def record_lists(client: Any, now: datetime) -> Dict[str, Any]:
    """The weekly lists into quiver_snapshot, once per ISO week.

    Checked every night, so a night whose fetch failed is retried by the next
    one instead of leaving the week empty.
    """
    week = iso_week(now)
    stamp = pit_store.iso_utc(now)
    run_id = pit_store.start_run(LISTS_STAGE, as_of_date=now.date().isoformat())
    counts: Dict[str, Dict[str, int]] = {}
    held: List[str] = []
    failures: Dict[str, str] = {}
    for feed in LISTS:
        if pit_store.has_quiver_snapshot(feed.dataset, week):
            held.append(feed.dataset)
            continue
        try:
            rows = keyed_rows(feed, _fetch(client, feed))
            written = pit_store.record_quiver_snapshot(
                feed.dataset, week, rows, recorded_at=stamp)
        except Exception as exc:  # noqa: BLE001 - named in the run and the exit
            failures[feed.dataset] = _reason(exc)
            continue
        counts[feed.dataset] = {"read": len(rows), "written": written}
    written = sum(c["written"] for c in counts.values())
    status = _stage_status(failures, len(counts) + len(held))
    _finish(run_id, written, status, failures)
    return {"status": status, "week": week, "written": written,
            "lists": counts, "already_held": held, "failed": failures}


def run(client: Any = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    pit_store.init_schema()
    owned = client is None
    client = client or QuiverWebClient()
    try:
        feeds = record_feeds(client, now)
        lists = record_lists(client, now)
    finally:
        if owned:
            client.close()
    failed = {f"{stage}/{name}": reason
              for stage, result in ((FEEDS_STAGE, feeds), (LISTS_STAGE, lists))
              for name, reason in result["failed"].items()}
    statuses = {feeds["status"], lists["status"]}
    if statuses == {"ok"}:
        status = "ok"
    elif statuses == {"failed"}:
        status = "failed"
    else:
        status = "partial"
    return {"status": status, "as_of": now.date().isoformat(),
            "written": feeds["written"] + lists["written"],
            FEEDS_STAGE: feeds, LISTS_STAGE: lists, "failed": failed}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_record.py -q -p no:cacheprovider`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add research/quiver_record.py testing/test_quiver_record.py
git commit -m "Record the QuiverQuant feeds and weekly lists

Two stages, each with its own run_log row. A feed that raises, comes back
without its row list, or parses 0 rows is named with its reason, and the
other feeds still record. The weekly lists are checked every night and written
only when the ISO week has no copy, so a failed night is retried the next."
```

---

### Task 4: Entry point and wiring (Discord, status, compose, docs)

**Files:**
- Modify: `research/quiver_record.py` (append `main`)
- Modify: `research/discord_post.py` (`JOB_CHANNELS` at ~line 69; `messages_for` branch chain at ~line 509; add `_quiver` next to `_sync` at ~line 724)
- Modify: `research/status.py` (`JOBS` tuple, ~line 61)
- Modify: `deploy/docker-compose.yml` (new service after `research-status`)
- Modify: `deploy/README.md` (cron block ~line 359; env table ~line 170; new subsection after "Congressional purchase decisions")
- Modify: `README.md` (job table ~line 330)
- Modify: `testing/test_discord_wiring.py` (scenario + `JOBS` + `FAILING`)
- Test: `testing/test_quiver_record.py` (append)

**Interfaces:**
- Consumes: `run()` from Task 3.
- Produces: `main(argv: Optional[Sequence[str]] = None) -> int`; module runnable as `python -m research.quiver_record`.

- [ ] **Step 1: Write the failing tests**

Append to `testing/test_quiver_record.py`:

```python
def test_a_clean_night_exits_zero(store, monkeypatch):
    monkeypatch.setattr(qr, "run", lambda: {
        "status": "ok", "as_of": "2026-11-12", "written": 10, "failed": {}})
    assert qr.main([]) == 0


def test_a_partial_night_exits_non_zero(store, monkeypatch):
    monkeypatch.setattr(qr, "run", lambda: {
        "status": "partial", "as_of": "2026-11-12", "written": 9,
        "failed": {"quiver_feeds/layoff": "rate_limited: HTTP 429"}})
    assert qr.main([]) == 1


def test_a_crash_exits_non_zero_and_says_why(store, monkeypatch, capsys):
    def boom():
        raise RuntimeError("store is locked")
    monkeypatch.setattr(qr, "run", boom)
    assert qr.main([]) == 1
    assert "store is locked" in capsys.readouterr().out
```

In `testing/test_discord_wiring.py`, add a scenario next to `_sync` (after it):

```python
def _quiver(monkeypatch, tmp_path, ok=True):
    from research import quiver_record

    monkeypatch.setattr(quiver_record, "run", lambda: {
        "status": "ok" if ok else "partial", "as_of": "2026-11-12",
        "written": 42,
        "failed": {} if ok else {
            "quiver_feeds/layoff": "rate_limited: HTTP 429"}})
    return quiver_record.main([])
```

Add to the `JOBS` dict:

```python
    "research-quiver": (_quiver, ["ops"]),
```

and change `FAILING` to:

```python
FAILING = {"research-copy": _copy, "research-scan": _scan,
           "research-watch": _watch, "research-quiver": _quiver}
```

Append to `testing/test_bot_cards.py` (the owner reads `/status` on a phone; a
job with no plain name falls back to its internal id):

```python
def test_every_job_the_status_screen_knows_has_a_plain_name():
    from research import status

    missing = [job["job"] for job in status.JOBS
               if job["job"] not in post.JOB_NAMES]
    assert not missing, f"no plain name for {missing}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_record.py testing/test_discord_wiring.py -q -p no:cacheprovider`
Expected: the three new `test_quiver_record` tests fail with `AttributeError: ... has no attribute 'main'`; `test_every_job_in_the_table_is_a_job_the_messages_know` fails because `research-quiver` is not in `JOB_CHANNELS`; the `research-quiver` parametrizations fail. `test_every_job_the_status_screen_knows_has_a_plain_name` passes now (all twelve current jobs have names); it guards Step 5, and Task 5's mutation table proves it.

- [ ] **Step 3: Implement `main`**

Append to `research/quiver_record.py`:

```python
def main(argv: Optional[Sequence[str]] = None) -> int:
    argparse.ArgumentParser(
        description="Record QuiverQuant's published feeds into pit.db."
    ).parse_args(argv)
    from research import discord_post
    elapsed = discord_post.stopwatch()
    try:
        result = run()
    except Exception as exc:  # noqa: BLE001 - reported and exits non-zero
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "written": 0, "failed": {}}
    print(json.dumps(result, indent=2))
    code = 0 if result["status"] == "ok" else 1
    return discord_post.finish(JOB, code, result, seconds=elapsed())


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement the Discord card**

In `research/discord_post.py`, add to `JOB_CHANNELS` after `"research-seed": ("ops",),`:

```python
    "research-quiver": ("ops",),
```

In `messages_for`, add a branch after the `congress-sync` branch:

```python
    elif job == "research-quiver":
        messages = _quiver(result, failed, seconds, when, error)
```

Add after `_sync`:

```python
def _quiver(result, failed, seconds, when, error):
    written = int(result.get("written") or 0)
    failures = result.get("failed") or {}
    summary = f"{_count(written, 'new record')} from QuiverQuant"
    if failures:
        summary += f", {_count(len(failures), 'feed')} failed"
    detail = error or "\n".join(
        f"{name}: {reason}" for name, reason in failures.items())
    return [("ops", ops_message(
        "research-quiver", summary, ok=not failed, quiet=not failed,
        seconds=seconds, error=detail if failed else "", when=when))]
```

- [ ] **Step 4b: Give the two stages plain names on the owner's `/status` card**

In `research/discord_post.py`, add to `JOB_NAMES` after `"fill_collect": "Checking orders",`:

```python
    "quiver_feeds": "Trades and filings from QuiverQuant",
    "quiver_lists": "Weekly lists from QuiverQuant",
```

- [ ] **Step 5: Register the stages with the status screen**

In `research/status.py`, add to `JOBS` after the `congress_copy` entry:

```python
    # QuiverQuant (research/quiver_record.py). Both stages log a run every
    # night, the weekly lists included (0 rows once the week is held), so
    # both are due every calendar day.
    {"job": "quiver_feeds", "module": "research.quiver_record",
     "every": "daily", "stale_after_days": 2,
     "max_run_seconds": 600, "partial_is_normal": False},
    {"job": "quiver_lists", "module": "research.quiver_record",
     "every": "daily", "stale_after_days": 2,
     "max_run_seconds": 600, "partial_is_normal": False},
```

- [ ] **Step 6: Add the compose service**

In `deploy/docker-compose.yml`, add after the `research-status` service block (after its `logging:` block, before the `# Congressional decisions` comment):

```yaml
  # QuiverQuant's published feeds, recorded nightly (research/quiver_record.py).
  # Quiver publishes rolling windows -- the latest 300 congressional trades, the
  # latest 100 contract awards -- so a row that leaves the window is gone from
  # the source, and this is the only record that it was there. No key: the
  # client reads Quiver's public pages, and no recorded feed needs the session
  # cookie the quiver server takes.
  #
  #   docker compose --env-file ../.env run --rm research-quiver
  #
  # After congress-sync, well before research-copy:
  #   30 6 * * * cd /srv/nemo/deploy && { date -u; flock -n -E 99 /var/lock/nemo-research-quiver.lock docker compose --env-file ../.env run --rm research-quiver; } >> /srv/nemo/logs/research-quiver.log 2>&1 || { rc=$?; tail -n 100 /srv/nemo/logs/research-quiver.log; docker compose --env-file ../.env run --rm research-notify --job research-quiver --log /srv/nemo/logs/research-quiver.log --code $rc; }
  research-quiver:
    image: ${NEMO_IMAGE:-nemo-data:local}
    profiles: ["sync"]
    restart: "no"
    entrypoint: ["python", "-m", "research.quiver_record"]
    environment:
      <<: *clock-env
      NEMO_PIT_DB: /app/data/pit.db
      NEMO_DISCORD_OPS_WEBHOOK_URL: ${NEMO_DISCORD_OPS_WEBHOOK_URL:-}
    volumes:
      - research-data:/app/data
    tmpfs:
      - /root/.edgar:rw,size=16m
      - /app/db_cache:rw,size=16m
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

- [ ] **Step 7: Document it**

In `deploy/README.md`, in the cron block of "## The scheduled jobs", add this line directly after the ` 0  6 * * * ... congress-sync ...` line:

```
30  6 * * * cd /srv/nemo/deploy && { date -u; flock -n -E 99 /var/lock/nemo-research-quiver.lock docker compose --env-file ../.env run --rm research-quiver; } >> /srv/nemo/logs/research-quiver.log 2>&1 || { rc=$?; tail -n 100 /srv/nemo/logs/research-quiver.log; docker compose --env-file ../.env run --rm research-notify --job research-quiver --log /srv/nemo/logs/research-quiver.log --code $rc; }
```

In the env table ("| Service | What it is handed |"), add after the `research-backup` row:

```
| research-quiver | nothing — Quiver's pages take no key |
```

After the "### Congressional purchase decisions" section (before "### Where a cron line's output goes"), add:

```markdown
### QuiverQuant's feeds, recorded nightly

`research-quiver` runs at 06:30 UTC every day, after `congress-sync`. It
records seven of Quiver's feeds into `pit.db` — congressional trades, insider
trades, WARN layoff notices, contract awards, federal spending, lobbying
filings and revolving-door registrations — and once per ISO week three lists
that change slowly: congress net worth, corporate donations and the top
lobbying spenders.

Quiver publishes windows, not registers: the latest 300 congressional trades,
the latest 100 contract awards. A row that leaves the window is gone from
Quiver. The job keeps the first copy of each row with the time it was first
seen, and files a later change as a revision instead of overwriting it.

Each feed fails alone. A rate limit, a page that changed shape, or a feed that
parses no rows is named in the run and in #ops, the other feeds still record,
and the job exits non-zero. The status screen shows the two stages as
`quiver_feeds` and `quiver_lists`.

    docker compose --env-file ../.env run --rm research-quiver
```

In `README.md`, add to the job table after the `research-copy` row:

```
| `research-quiver` | records QuiverQuant's congressional, insider, layoff, contract, spending and lobbying feeds, and three weekly lists | `30 6 * * *` |
```

- [ ] **Step 8: Run the affected tests**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_record.py testing/test_quiver_store.py testing/test_discord_wiring.py testing/test_bot_cards.py testing/test_status.py testing/test_readme_counts.py testing/test_compose_env_minimality.py testing/test_readme_env_vars.py testing/test_cli.py -q -p no:cacheprovider`
Expected: all pass. If `test_readme_env_vars.py` or the README STE checks object to wording, fix the wording, not the test.

- [ ] **Step 9: Commit**

```bash
git add research/quiver_record.py research/discord_post.py research/status.py \
        deploy/docker-compose.yml deploy/README.md README.md \
        testing/test_quiver_record.py testing/test_discord_wiring.py \
        testing/test_bot_cards.py
git commit -m "Schedule the QuiverQuant recorder

research-quiver runs at 06:30 UTC every day, after congress-sync. It posts one
#ops card per run, silent when clean and red with each failed feed when not,
and exits non-zero on a partial night so the cron line reports it. The status
screen knows both stages on a daily schedule, and the runbook and README carry
the cron line."
```

---

### Task 5: Verify, ship, deploy

**Files:** none new.

- [ ] **Step 1: Mutation checks — each rule must fail its own test by name**

For each mutation: make the change, run the named test, confirm it FAILS, then `git checkout -- <file>`.

| Mutation | File | Test that must fail |
| --- | --- | --- |
| Delete the `known = latest[...] if latest else ...` line and compare against `held["payload"]` | `research/pit_store.py` | `test_the_same_change_seen_again_is_not_a_second_revision` |
| Pass `volatile=()` inside `_quiver_compared` calls | `research/pit_store.py` | `test_a_change_only_in_a_volatile_field_files_nothing` |
| Drop `AND date(recorded_at) <= ?` (and its parameter) | `research/pit_store.py` | `test_an_as_of_read_excludes_a_row_recorded_later` |
| Drop `AND date(noticed_at) <= ?` (and its parameter) | `research/pit_store.py` | `test_an_as_of_read_applies_only_revisions_noticed_by_then` |
| Remove the `elif every == "daily"` branch | `research/pit_store.py` | `test_a_daily_job_misses_a_sunday` |
| Remove `#{occurrence}` from the hash input | `research/quiver_record.py` | `test_two_identical_lobbying_filings_are_two_rows` |
| Add `"ticker"` to the layoff identity | `research/quiver_record.py` | `test_a_field_quiver_fills_in_later_does_not_change_the_key` |
| Remove the `if not rows:` raise in `_fetch` | `research/quiver_record.py` | `test_an_empty_feed_is_a_failure_not_a_quiet_night` |
| Remove the `has_quiver_snapshot` skip in `record_lists` | `research/quiver_record.py` | `test_weekly_lists_are_written_once_per_week` |
| Change `code = 0 if ...` to `code = 0` | `research/quiver_record.py` | `test_a_partial_night_exits_non_zero` |
| Delete the `"quiver_feeds"` line from `JOB_NAMES` | `research/discord_post.py` | `test_every_job_the_status_screen_knows_has_a_plain_name` |

Command per row, with the test name from the table in place of `TEST_NAME`:
`SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_quiver_store.py testing/test_quiver_record.py testing/test_bot_cards.py -q -p no:cacheprovider -k TEST_NAME`

- [ ] **Step 2: Full offline suite and lint**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --ignore=testing/output -p no:cacheprovider | tail -1`
Expected: a line matching `^[0-9]+ passed, [0-9]+ skipped`, no failures.
Run: `uvx ruff@0.15.14 check .`
Expected: `All checks passed!`

- [ ] **Step 3: One live run against Quiver into a scratch store**

```bash
NEMO_PIT_DB=/private/tmp/claude-501/-Users-jacobdang-Projects-Nemo-IB/34c4c992-fa56-4c01-b7fd-054d7270844d/scratchpad/quiver-live.db \
NEMO_DISCORD_OPS_WEBHOOK_URL= \
.venv/bin/python -c "import json; from research import quiver_record as q; r = q.run(); print(json.dumps({k: r[k] for k in ('status','written','failed')}, indent=1)); print({d: c for d, c in r['quiver_feeds']['feeds'].items()})"
```

Expected: `status` `ok`; every feed `read` > 0 and `new == read` minus any duplicate `tx_id`s. Then run it a second time with the same `NEMO_PIT_DB`: every feed `new` 0 (or a handful, if Quiver published in between), `revised` near 0, `quiver_lists.written` 0. A large `revised` count on the second run means a field is volatile and missing from that feed's `volatile` tuple: find it by diffing one revision's `old_payload` and `new_payload`, add it, add a test, and repeat. Delete the scratch database afterwards.

- [ ] **Step 4: Push and open the PR**

The PR body says what changed and why, nothing else, with no attribution line.
Fill the two bracketed numbers from Steps 2 and 3 before running:

```bash
git push -u origin quiver-recorder
gh pr create --title "Record QuiverQuant's feeds nightly into pit.db" --body "$(cat <<'EOF'
Quiver publishes rolling windows (the latest 300 congressional trades, the latest 100 contract awards), so a row that leaves the window is gone from the source. This records seven feeds into pit.db every night and three slow-moving lists once per ISO week.

- `pit_store`: `quiver_record` keeps the first copy of each row and when it was first seen; `quiver_revision` keeps each later change once; `quiver_snapshot` holds the weekly lists. As-of reads apply only revisions noticed by that date.
- `research/quiver_record.py`: each feed fails alone and by name; a feed that parses 0 rows is a failure, not a quiet night. Weekly lists are checked nightly, so a failed night is retried the next.
- `research-quiver` runs at 06:30 UTC daily, after congress-sync. One #ops card per run. The status screen knows both stages on a new `daily` schedule, which `missing_days` now counts on weekends too.

Spec: `docs/superpowers/specs/2026-11-12-quiver-recorder-design.md`. Offline suite: [suite line]. Live run into a scratch store: [rows per feed].
EOF
)"
```

- [ ] **Step 5: Merge behind green CI**

`gh` resolves the PR from the current branch:

```bash
gh pr checks --watch
gh pr merge --merge
```

- [ ] **Step 6: Deploy to the VM and schedule it**

```bash
ssh jacob@100.118.253.108 'cd /srv/nemo && git pull --ff-only && docker build -q -t nemo-data:local . && docker build -q --target fills -t nemo-fills:local .'
ssh jacob@100.118.253.108 'crontab -l > ~/crontab.before-quiver'
```

Add the `30 6 * * *` line from Step 7 of Task 4 to the VM's crontab, keeping every existing line (the fills lines and the 21:15 portfolio line are not in the runbook's main block; do not replace the crontab from the runbook). Then run it once by hand and confirm the run log:

```bash
ssh jacob@100.118.253.108 'cd /srv/nemo/deploy && docker compose --env-file ../.env run --rm research-quiver | tail -20'
```

Expected: `"status": "ok"`. Then `docker compose --env-file ../.env --profile bot up -d nemo-bot` (plain `up -d` does not touch the bot).
