# QuiverQuant Recorder — Design

**Date:** 2026-11-12
**Status:** approved
**Depends on:** the QuiverQuant server (PR #208, deployed 2026-11-12)
**Scope:** record Quiver's published feeds into `pit.db` every night. The congress
gap filler and the held-name alerts come after this and read what it records;
neither is designed here.

## Problem

PR #208 made fifteen Quiver tools callable, but an agent only sees what Quiver
publishes at the moment it asks. Most of Quiver's surfaces publish a window: the
latest 300 congressional trades, the latest 100 contract awards, about 2,000
WARN notices. A row that leaves the window is gone from Quiver, and nothing
records when it first appeared. A forward record cannot be made later, which is
the reason `pit_store` exists; this extends it to Quiver.

## What was measured

Live, 2026-11-12, with `QuiverWebClient` and no limit:

| Feed | Rows | Window by its own date | Busiest day |
| --- | ---: | --- | ---: |
| congress trades | 300 | filed 2026-10-12 .. 2026-11-10 | 64 |
| insider trades (union of two lists) | 1,733 | filed over 22 distinct days | 552 |
| WARN layoff notices | 2,000 | notice 2026-02-27 .. 2026-11-09 | 35 |
| contract awards | 100 | awarded 2026-11-05 .. 2026-11-08 | 58 |
| federal spending transactions | 500 | 2026-10-01 .. 2026-10-31 (monthly batch) | 134 |
| lobbying filings | 100 | 2026-10-22 .. 2026-11-11 | 20 |
| revolving-door registrations | 1,500 | dated filings, quarterly periods | — |

The tightest window is contract awards, at four days. One run a night, every
day of the week, misses nothing.

Two of these carry no ticker: contract awards name only the recipient, and
lobbying filings carry only ticker, amount and date (no registrant), so two
filings for one company on one day with one amount are indistinguishable.

Not recorded:

- **WallStreetBets.** Quiver's `data_horizon` is 2025-02-21. A nightly copy of a
  series that stopped twenty-one months ago records nothing new.
- **Bills, patents, 13F, ticker snapshot.** Each answers one query. None
  publishes a feed.

## The job

`research/quiver_record.py`, run as compose service `research-quiver` under the
`sync` profile, on the `research-data` volume with `NEMO_PIT_DB`. It imports
`tools.quiver_server.client.QuiverWebClient` directly, so it needs no MCP token
and no gateway, and it inherits the client's 200 ms request spacing and single
retry on 429. It reads no API key. `QUIVER_SESSION_COOKIE` is not passed; no
recorded feed needs it.

Cron, after `congress-sync` (06:00) and before `research-copy` (12:30):

```
30 6 * * * cd /srv/nemo/deploy && { date -u; flock -n -E 99 /var/lock/nemo-research-quiver.lock docker compose --env-file ../.env run --rm research-quiver; } >> /srv/nemo/logs/research-quiver.log 2>&1 || { rc=$?; tail -n 100 /srv/nemo/logs/research-quiver.log; docker compose --env-file ../.env run --rm research-notify --job research-quiver --log /srv/nemo/logs/research-quiver.log --code $rc; }
```

The run's `as_of_date` and the snapshot's ISO week both come from
`datetime.now(timezone.utc)` at the start of the run, never a local clock.

Two stages, each with its own `run_log` row:

- **`quiver_feeds`** — every run. The seven feeds above, into `quiver_record`.
- **`quiver_lists`** — once per ISO week. Congress net worth (494 members),
  election contributions (200 rows, PAC and employee), lobbying top spenders
  (200), into `quiver_snapshot`. It runs every night and writes only when the
  current ISO week has no snapshot for that list, so a failed night is retried
  by the next one instead of waiting a week.

## Schema

Added to `pit_store._SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS quiver_record (
    dataset       TEXT NOT NULL,
    row_key       TEXT NOT NULL,
    ticker        TEXT,
    event_date    TEXT,
    payload       TEXT NOT NULL,   -- the row as Quiver gave it, JSON, sorted keys
    first_seen_at TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (dataset, row_key)
);
CREATE INDEX IF NOT EXISTS idx_quiver_record_ticker
    ON quiver_record(ticker, dataset);

CREATE TABLE IF NOT EXISTS quiver_revision (
    dataset     TEXT NOT NULL,
    row_key     TEXT NOT NULL,
    old_payload TEXT NOT NULL,
    new_payload TEXT NOT NULL,
    new_ticker  TEXT,
    noticed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiver_snapshot (
    dataset     TEXT NOT NULL,
    as_of_week  TEXT NOT NULL,     -- ISO week, e.g. 2026-W46
    row_key     TEXT NOT NULL,
    ticker      TEXT,
    payload     TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (dataset, as_of_week, row_key)
);
```

## Keys

| Dataset | `row_key` | `ticker` | `event_date` | Volatile (ignored when comparing) |
| --- | --- | --- | --- | --- |
| `congress_trade` | Quiver's `tx_id` | `ticker` | `filed_date` | — |
| `insider_trade` | hash of `owner`, `ticker`, `transaction_date`, `shares`, `tx_type` — the fields the client already deduplicates the two lists on | `ticker` | `filed_date` | `list`, `net_worth_estimate` |
| `layoff` | hash of `employer`, `notice_date`, `city`, `state`, `workers`, `kind` | `ticker` | `notice_date` | — |
| `contract_award` | hash of `recipient`, `date_awarded`, `amount`, `agency`, `description` | none | `date_awarded` | — |
| `gov_spending` | hash of `award_id`, `date`, `amount_usd`, `recipient`, `agency` | `ticker` | `date` | — |
| `lobbying_filing` | hash of `ticker`, `amount`, `date` | `ticker` | `date` | — |
| `revolving_door` | hash of `lobbyist`, `client`, `firm`, `filed_date`, `filing_type`, `period` | `ticker` | `filed_date` | `is_new` |
| `congress_net_worth` (snapshot) | `bioguide_id`, else `name` | — | — | — |
| `election_contribution` (snapshot) | `ticker` else `company`, `source`, `cycle` | `ticker` | — | — |
| `lobbying_top_spender` (snapshot) | `ticker` | `ticker` | — | — |

**Identical rows.** A hash key carries an occurrence index: the n-th row in one
fetch with the same identity fields gets `n` in its hash. Two lobbying filings
for one company on one day with one amount are two rows, not one. Because the
rows are identical, the index is stable across nights: whichever of them Quiver
lists first is still occurrence 0.

**Fields left out of the hash.** A field Quiver may fill in later — a ticker
match, a match quality — is not part of the identity, so filling it in is a
revision of the row and not a new row.

**Volatile fields.** `net_worth_estimate` moves with prices and `list` says which
of two insider lists carried the row that night; `is_new` flips as a quarter
ages. They are stored in the payload but ignored when deciding whether a row
changed, or every night would file 1,733 revisions that mean nothing.

## Writes and reads

`pit_store.record_quiver_rows(dataset, rows, recorded_at=None) -> (new, revised)`:

- Not seen before: insert, `first_seen_at = recorded_at = now`.
- Seen, payload equal apart from volatile fields: nothing.
- Seen, payload different from the latest known payload (the newest revision,
  else the stored row): the stored row stands; one `quiver_revision` row holds
  old, new and the new ticker. The same change seen again the next night is not
  a second revision. `first_seen_at` is never written after the insert.

`pit_store.record_quiver_snapshot(dataset, as_of_week, rows, recorded_at=None) -> int`
inserts with `INSERT OR IGNORE` on `(dataset, as_of_week, row_key)`.

`pit_store.quiver_records_as_of(dataset, as_of, ticker=None)` returns rows with
`date(recorded_at) <= as_of`, newest `event_date` first, each with the payload
and ticker of the latest revision noticed by `as_of` (or the original) and a
`revised` flag. A layoff Quiver matches to a ticker a week after we first saw it
is found under that ticker from the night the match was noticed, and not
before. The filler and the alerts read through it and nothing else.

`pit_store.has_quiver_snapshot(dataset, as_of_week)` answers whether a weekly
list is already held.

## Failure

Fail loud, one feed at a time:

- A feed that raises (`QuiverError` with `provider_unavailable`,
  `parse_failed`, a 429 after its retry) is named with its reason. The other
  feeds still record.
- **A feed that parses 0 rows is a failure**, reported as `empty`. "Nothing new"
  means N rows read and 0 of them new, and the run says both numbers.
- Any failed feed makes the stage `partial`, with the failed feeds and reasons
  in `run_log.error`, and the job exits non-zero so the cron line posts to #ops.
- Every successful feed reports `read`, `new` and `revised` in the run's JSON.
- One #ops card per run through `discord_post.finish`, like every other job:
  "**N** new records from QuiverQuant", silent when the night is clean, red
  with each failed feed and its reason when it is not.

## Status

Both stages run, and log a `run_log` row, every night: `quiver_lists` logs `ok`
with 0 rows on a night when this week's copies already exist. So both are on a
new `daily` schedule. Added to `research/status.JOBS`:

- `quiver_feeds`: module `research.quiver_record`, every `daily`,
  `stale_after_days` 2, `max_run_seconds` 600.
- `quiver_lists`: same module, every `daily`, `stale_after_days` 2,
  `max_run_seconds` 600.

`pit_store.missing_days` falls back to weekdays for any schedule it does not
name, which would hide a missed Saturday or Sunday. It learns `daily`: every
calendar day is due. `status._age` already counts calendar days for any
schedule outside `WEEKDAY_SCHEDULES`.

`testing/test_status.py` already fails when a compose job is missing from that
list.

## Tests (written first)

`testing/test_quiver_record.py`, with a scripted client — no network:

1. First sight writes every row with `first_seen_at == recorded_at`.
2. A second run over the same rows writes nothing.
3. A changed payload files one revision and leaves the stored row as it was.
4. A change only in a volatile field files nothing.
5. `first_seen_at` does not move on a later run.
6. Two identical lobbying rows in one fetch are two rows.
7. `quiver_records_as_of` excludes a row recorded after `as_of`.
8. One feed raising: stage `partial`, that feed and its reason in `error`, every
   other feed recorded, non-zero exit.
9. One feed returning 0 rows: reported as `empty`, same as 8.
10. Weekly lists: written on the first run of an ISO week, not on the second;
    written on the next night when the first night's fetch failed.
11. `missing_days(..., every="daily")` reports a missed Sunday; the weekday
    default is unchanged for every existing job.

Each rule is pinned by mutation: remove it and run the suite; its own test must
fail by name.

## Docs and registration

- `deploy/docker-compose.yml`: `research-quiver` service, with the cron line in
  its comment block.
- `deploy/README.md`: the job in the scheduled-jobs section and the env table
  (no keys).
- `README.md`: the cron line and job table (`testing/test_readme_counts.py`
  checks both).
- `research/discord_post.py`: `research-quiver` in `JOB_CHANNELS` (ops only)
  and its card; `testing/test_discord_wiring.py` gets its scenario.
- `research/cli.py` needs nothing; it reads services from compose.
