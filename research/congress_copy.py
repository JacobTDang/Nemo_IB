"""Record congressional purchase decisions separately from the earnings book."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from research import daily_job, pit_store, scanner
from tools.altdata_server import congress_store

JOB = 'congress_copy'
# Weekday cron plus a House posting lag of up to 4 days measured.
LOOKBACK_DAYS = 7


def enabled() -> bool:
    """An empty compose value is unset; any other value must be on or off."""
    value = os.environ.get('NEMO_COPY_TRADING', '').lower()
    if value not in ('', 'on', 'off'):
        raise ValueError('NEMO_COPY_TRADING must be on or off')
    return value == 'on'


def _integer(raw: str, name: str, default: int, minimum: int) -> int:
    if raw == '':
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer >= {minimum}') from exc
    if value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


@dataclass(frozen=True)
class Settings:
    enabled: bool
    min_amount: int
    dollars: int
    hold_days: int
    max_open: int
    max_trade_age_days: int
    members: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        flag = enabled()
        raw = os.environ.get('NEMO_COPY_MEMBERS', '')
        members = tuple(name.strip().casefold() for name in raw.split(',')) if raw else ()
        if any(not name for name in members):
            raise ValueError('NEMO_COPY_MEMBERS must contain comma-separated member names')
        return cls(
            flag,
            _integer(os.environ.get('NEMO_COPY_MIN_AMOUNT', ''),
                     'NEMO_COPY_MIN_AMOUNT', 15001, 0),
            _integer(os.environ.get('NEMO_COPY_DOLLARS', ''),
                     'NEMO_COPY_DOLLARS', 5000, 1),
            _integer(os.environ.get('NEMO_COPY_HOLD_DAYS', ''),
                     'NEMO_COPY_HOLD_DAYS', 20, 1),
            _integer(os.environ.get('NEMO_COPY_MAX_OPEN', ''),
                     'NEMO_COPY_MAX_OPEN', 10, 1),
            _integer(os.environ.get('NEMO_COPY_MAX_TRADE_AGE_DAYS', ''),
                     'NEMO_COPY_MAX_TRADE_AGE_DAYS', 45, 0),
            members)


def _instant(value: str, field: str) -> datetime:
    if not value:
        raise ValueError(f'missing {field}')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError(f'{field} must have a UTC offset: {value!r}')
    return stamp.astimezone(timezone.utc)


def _ticker(value: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', value.upper())


def _issuer(ticker: str, cik: str | None) -> str:
    # An absent CIK is not an issuer shared by every unmapped ticker.
    return 'cik:' + str(cik).lstrip('0') if cik else 'ticker:' + _ticker(ticker)


def _source_rows(now: datetime) -> list[dict]:
    # congress_store.connect() applies migrations and sets WAL. A reader must
    # neither migrate nor create a missing store, so this opens it mode=ro. The
    # mount stays writable: SQLite needs the WAL shared-memory file even to read.
    uri = Path(congress_store.current_db_path()).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as conn:
        conn.row_factory = sqlite3.Row
        last = conn.execute("SELECT MAX(last_synced_at) FROM sync_state "
                            "WHERE source GLOB 'house_*'").fetchone()[0]
        if last is None:
            raise RuntimeError('House sync is missing; run congress-sync')
        if now - _instant(last, 'House last_synced_at') > timedelta(
                days=congress_store.CONGRESS_SYNC_STALE_DAYS):
            raise RuntimeError(f'House sync is stale: last success {last}')
        since = now - timedelta(days=LOOKBACK_DAYS)
        return [dict(row) for row in conn.execute(
            """SELECT t.*, f.chamber, f.filed_date, f.parsed_at,
                      m.full_name AS member
                 FROM transactions t JOIN filings f ON f.filing_id=t.filing_id
                 LEFT JOIN members m ON m.member_id=t.member_id
                WHERE t.transaction_type='purchase'
                  AND t.asset_type_code IN ('ST', 'Stock')
                ORDER BY f.filed_date, f.filing_id, t.row_index, t.txn_id""")
            if _instant(row['parsed_at'], 'parsed_at') >= since]


def _decisions(now: datetime, settings: Settings) -> list[dict]:
    as_of = now.date().isoformat()
    source = _source_rows(now)
    previous = pit_store.copy_orders_as_of(now.isoformat())
    known = {row['txn_id'] for row in previous}
    universe = {_ticker(t): (t, cik) for t, cik in daily_job.eligible_names(as_of)}
    held = set()
    open_count = 0
    for row in previous:
        if not row['accepted']:
            continue
        expiry = row['intended_session']
        for _ in range(settings.hold_days):
            expiry = scanner._next_session(expiry)
        if as_of < expiry:
            held.add(_issuer(row['ticker'], row['cik']))
            open_count += 1
    # scanner names the session strictly after its argument. Moving the input
    # back one date gives the first session on or after the UTC decision date.
    session = scanner._next_session((now.date() - timedelta(days=1)).isoformat())
    selected = set()
    decisions = []
    for source_row in source:
        if source_row['txn_id'] in known:
            continue
        learned = _instant(source_row['parsed_at'], 'parsed_at')
        if learned >= now:
            continue
        row = {**source_row, 'learned_at': source_row['parsed_at'],
               'as_of_date': as_of, 'decided_at': now.isoformat(),
               'source_ticker': source_row['ticker'], 'cik': None,
               'side': None, 'target_dollars': None, 'intended_session': None}
        ticker = _ticker(row['ticker'] or '')
        reason = None
        issuer = None
        if not ticker:
            reason = 'no_ticker'
        elif row['amount_min'] is None:
            raise ValueError(f"{row['txn_id']}: missing amount_min")
        elif row['amount_min'] < settings.min_amount:
            reason = 'below_amount_floor'
        elif settings.members and (row['member'] or '').casefold() not in settings.members:
            reason = 'member_not_followed'
        elif not row['transaction_date']:
            raise ValueError(f"{row['txn_id']}: missing transaction_date")
        elif (learned.date() - date.fromisoformat(row['transaction_date'])).days > settings.max_trade_age_days:
            reason = 'trade_too_old'
        elif ticker not in universe:
            reason = 'not_eligible'
        else:
            row['ticker'], row['cik'] = universe[ticker]
            issuer = _issuer(row['ticker'], row['cik'])
            if issuer in held:
                reason = 'already_held'
            elif issuer in selected:
                reason = 'duplicate_issuer'
            elif open_count + len(selected) >= settings.max_open:
                reason = 'book_full'
        row.update(accepted=reason is None, reason=reason)
        if reason is None:
            selected.add(issuer)
            row.update(side='long', target_dollars=settings.dollars,
                       intended_session=session)
        decisions.append(row)
    return decisions


def run() -> dict:
    """One UTC decision pass; configuration errors name the invalid setting."""
    settings = Settings.from_env()
    if not settings.enabled:
        return {'status': 'off', 'rows_written': 0}
    now = datetime.now(timezone.utc)
    as_of = now.date().isoformat()
    pit_store.init_schema()
    run_id = pit_store.start_run(JOB, as_of_date=as_of)
    try:
        decisions = _decisions(now, settings)
        written = pit_store.record_copy_orders(decisions)
    except Exception as exc:  # noqa: BLE001 - failure is recorded and exits nonzero
        error = f'{type(exc).__name__}: {exc}'
        pit_store.finish_run(status='failed', error=error, run_id=run_id)
        return {'status': 'failed', 'as_of': as_of, 'error': error, 'rows_written': 0}
    pit_store.finish_run(rows_written=written, status='ok', run_id=run_id)
    return {'status': 'ok', 'as_of': as_of, 'rows_written': written}


def main() -> int:
    if enabled():
        pit_store.init_schema()
    result = run()
    print(json.dumps(result, indent=2))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
