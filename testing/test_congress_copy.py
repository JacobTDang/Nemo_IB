"""Congress decisions use the real source and destination schemas."""
from datetime import datetime, timezone
import importlib
from pathlib import Path

import pytest
import yaml

from research import pit_store
from tools.altdata_server import congress_store

NOW = datetime(2026, 11, 5, 12, 30, tzinfo=timezone.utc)
SETTINGS = ('TRADING', 'MIN_AMOUNT', 'DOLLARS', 'HOLD_DAYS', 'MAX_OPEN',
            'MAX_TRADE_AGE_DAYS', 'MEMBERS')


@pytest.fixture
def book(tmp_path, monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv('NEMO_COPY_' + name, raising=False)
    monkeypatch.setenv('NEMO_COPY_TRADING', 'on')
    monkeypatch.setenv('NEMO_PIT_DB', str(tmp_path / 'pit.db'))
    monkeypatch.setenv('NEMO_CONGRESS_DB', str(tmp_path / 'congress.db'))
    pit_store.init_schema()
    congress_store.init_schema()
    with congress_store.connect() as conn:
        conn.execute("INSERT INTO sync_state(source,last_synced_at) VALUES (?,?)",
                     ('house_2026', '2026-11-05T06:00:00+00:00'))
        conn.execute("INSERT INTO members(member_id,chamber,full_name) VALUES (?,?,?)",
                     ('member1', 'house', 'Jane Smith'))
    pit_store.record_universe('2026-11-04', [
        {'ticker': t, 'cik': c, 'eligible': True}
        for t, c in [('AAA', '1'), ('AAB', '1'), ('BBB', '2'),
                     ('BRK-B', '3'), ('NOCIK', None), ('OTHER', None)]],
        recorded_at='2026-11-04T22:00:00Z')
    mod = importlib.import_module('research.congress_copy')

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return NOW

    monkeypatch.setattr(mod, 'datetime', Clock)
    return mod


def transaction(txn_id='t1', **changes):
    filing = dict(filing_id='f' + txn_id, chamber='house', doc_id=txn_id,
                  member_id='member1', filed_date='2026-11-04',
                  parse_status='parsed', parsed_at='2026-11-05T06:00:00+00:00')
    row = dict(txn_id=txn_id, filing_id=filing['filing_id'], member_id='member1',
               ticker='AAA', asset_type_code='ST', owner='self',
               transaction_type='purchase', transaction_date='2026-10-15',
               amount_min=15001, amount_max=50000)
    for key, value in changes.items():
        (filing if key in filing else row)[key] = value
    with congress_store.connect() as conn:
        for table, values in [('filings', filing), ('transactions', row)]:
            conn.execute(f"INSERT INTO {table} ({','.join(values)}) VALUES "
                         f"({','.join('?' for _ in values)})", tuple(values.values()))


def decisions():
    with pit_store.connect() as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM copy_order ORDER BY txn_id')]


def last_run():
    with pit_store.connect() as conn:
        return dict(conn.execute("SELECT * FROM run_log WHERE job='congress_copy' "
                                 'ORDER BY run_id DESC LIMIT 1').fetchone())


def test_defaults_are_declared(book, monkeypatch):
    monkeypatch.delenv('NEMO_COPY_TRADING')
    settings = book.Settings.from_env()
    assert (settings.enabled, settings.min_amount, settings.dollars,
            settings.hold_days, settings.max_open, settings.max_trade_age_days,
            settings.members) == (False, 15001, 5000, 20, 10, 45, ())


@pytest.mark.parametrize('value,enabled', [('ON', True), ('oN', True),
                                          ('OFF', False), ('off', False), ('', False)])
def test_flag_accepts_case_insensitive_on_off(book, monkeypatch, value, enabled):
    monkeypatch.setenv('NEMO_COPY_TRADING', value)
    assert book.Settings.from_env().enabled is enabled


@pytest.mark.parametrize('value', ['yes', 'true', '0', 'of', ' on '])
def test_unknown_flag_fails_loud(book, monkeypatch, value):
    monkeypatch.setenv('NEMO_COPY_TRADING', value)
    with pytest.raises(ValueError, match='NEMO_COPY_TRADING'):
        book.run()


@pytest.mark.parametrize('name', SETTINGS[1:-1])
@pytest.mark.parametrize('value', ['bad', '-1', 'nan', 'inf', '1.5'])
def test_unreadable_numeric_setting_names_itself(book, monkeypatch, name, value):
    monkeypatch.setenv('NEMO_COPY_' + name, value)
    with pytest.raises(ValueError, match='NEMO_COPY_' + name):
        book.run()


@pytest.mark.parametrize('name', ['DOLLARS', 'HOLD_DAYS', 'MAX_OPEN'])
def test_positive_settings_refuse_zero(book, monkeypatch, name):
    monkeypatch.setenv('NEMO_COPY_' + name, '0')
    with pytest.raises(ValueError, match='NEMO_COPY_' + name):
        book.run()


def test_empty_numeric_values_use_defaults_for_compose(book, monkeypatch):
    for name in SETTINGS[1:-1]:
        monkeypatch.setenv('NEMO_COPY_' + name, '')
    assert book.Settings.from_env().max_open == 10


def test_malformed_member_list_fails_loud(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_MEMBERS', 'Jane Smith,,John Smith')
    with pytest.raises(ValueError, match='NEMO_COPY_MEMBERS'):
        book.run()


def test_off_does_not_open_stores_or_write_run(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_TRADING', 'off')
    monkeypatch.setattr(pit_store, 'init_schema', lambda: pytest.fail('opened PIT store'))
    assert book.run()['status'] == 'off'
    with pit_store.connect() as conn:
        assert conn.execute('SELECT count(*) FROM run_log').fetchone()[0] == 0


def test_purchase_records_provenance_and_target(book):
    transaction()
    assert book.run()['status'] == 'ok'
    row = decisions()[0]
    assert row['accepted'] == 1 and row['reason'] is None
    assert {k: row[k] for k in ['txn_id', 'filing_id', 'member', 'amount_min',
                               'amount_max', 'transaction_date', 'filed_date']} == {
        'txn_id': 't1', 'filing_id': 'ft1', 'member': 'Jane Smith',
        'amount_min': 15001, 'amount_max': 50000,
        'transaction_date': '2026-10-15', 'filed_date': '2026-11-04'}
    assert row['learned_at'] == '2026-11-05T06:00:00+00:00'
    assert datetime.fromisoformat(row['decided_at']) == NOW
    assert row['as_of_date'] == '2026-11-05'
    assert row['side'] == 'long' and row['target_dollars'] == 5000
    assert row['intended_session'] == '2026-11-05'
    assert last_run()['rows_written'] == 1


@pytest.mark.parametrize('owner', ['self', 'spouse', 'joint', 'dependent_child'])
def test_all_household_owners_are_copied(book, owner):
    transaction(owner=owner)
    book.run()
    assert decisions()[0]['accepted'] == 1


@pytest.mark.parametrize('kind', ['sale', 'sale_full', 'sale_partial', 'exchange'])
def test_only_purchases_are_candidates(book, kind):
    transaction(transaction_type=kind)
    book.run()
    assert decisions() == []


@pytest.mark.parametrize('asset', ['GS', 'OT', 'CS', 'OP', None])
def test_only_stock_is_a_candidate(book, asset):
    transaction(asset_type_code=asset)
    book.run()
    assert decisions() == []


def test_senate_stock_can_be_copied(book):
    transaction(chamber='senate', asset_type_code='Stock')
    book.run()
    assert decisions()[0]['accepted'] == 1


@pytest.mark.parametrize('stamp', ['2026-10-29T12:29:59Z', '2026-10-28T12:30:00Z'])
def test_filings_outside_seven_day_window_are_not_candidates(book, stamp):
    transaction(parsed_at=stamp)
    book.run()
    assert decisions() == []


def test_seven_day_boundary_is_included(book):
    transaction(parsed_at='2026-10-29T12:30:00Z')
    book.run()
    assert decisions()[0]['accepted'] == 1


def test_filing_parsed_on_weekend_after_late_post_is_decided_on_monday(book, monkeypatch):
    monday = datetime(2026, 11, 9, 12, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(book.datetime, 'now', classmethod(lambda cls, tz=None: monday))
    with congress_store.connect() as conn:
        conn.execute("UPDATE sync_state SET last_synced_at='2026-11-09T06:00:00+00:00'")
    transaction(filed_date='2026-11-05', parsed_at='2026-11-07T06:00:00+00:00',
                transaction_date='2026-10-25')
    assert book.run()['status'] == 'ok'
    order = decisions()[0]
    assert order['accepted'] == 1 and order['reason'] is None
    assert order['filed_date'] == '2026-11-05'
    assert order['learned_at'] == '2026-11-07T06:00:00+00:00'


def test_old_filed_date_with_recent_parse_is_decided_and_rejected_when_trade_too_old(book):
    transaction(filed_date='2026-09-01',
                parsed_at='2026-11-05T06:00:00+00:00',
                transaction_date='2026-08-01')
    book.run()
    assert len(decisions()) == 1
    assert decisions()[0]['accepted'] == 0
    assert decisions()[0]['reason'] == 'trade_too_old'


def test_transaction_decided_on_earlier_run_is_not_decided_again_on_later_run(book, monkeypatch):
    transaction('t1')
    assert book.run()['rows_written'] == 1
    assert len(decisions()) == 1

    friday = datetime(2026, 11, 6, 12, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(book.datetime, 'now', classmethod(lambda cls, tz=None: friday))
    with congress_store.connect() as conn:
        conn.execute("UPDATE sync_state SET last_synced_at='2026-11-06T06:00:00+00:00'")

    settings = book.Settings.from_env()
    assert book._decisions(friday, settings) == []
    assert book.run()['rows_written'] == 0
    assert len(decisions()) == 1


@pytest.mark.parametrize('stamp', ['2026-11-05T12:30:01Z',
                                   '2026-11-05T07:31:00-05:00',
                                   '2026-11-05T12:30:00Z'])
def test_future_or_simultaneous_parse_is_not_copied(book, stamp):
    transaction(parsed_at=stamp)
    book.run()
    assert decisions() == []


def test_parse_time_offsets_are_compared_as_instants(book):
    transaction(parsed_at='2026-11-05T13:00:00+01:00')
    book.run()
    assert decisions()[0]['accepted'] == 1


def test_missing_parse_time_fails_instead_of_inventing_knowledge(book):
    transaction(parsed_at=None)
    assert book.run()['status'] == 'failed'
    assert 'parsed_at' in last_run()['error']


def test_no_ticker_is_rejected(book):
    transaction(ticker=None)
    book.run()
    assert decisions()[0]['reason'] == 'no_ticker'


def test_below_amount_floor_is_rejected(book):
    transaction(amount_min=15000)
    book.run()
    assert decisions()[0]['reason'] == 'below_amount_floor'


def test_member_outside_list_is_rejected(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_MEMBERS', 'John Smith')
    transaction()
    book.run()
    assert decisions()[0]['reason'] == 'member_not_followed'


def test_member_list_matches_full_names_without_case(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_MEMBERS', ' jane smith , John Smith')
    transaction()
    book.run()
    assert decisions()[0]['accepted'] == 1


def test_trade_age_is_measured_at_learning_time(book):
    transaction(transaction_date='2026-09-20')
    book.run()
    assert decisions()[0]['reason'] == 'trade_too_old'


def test_age_limit_boundary_at_learning_is_accepted(book):
    transaction(transaction_date='2026-09-20', parsed_at='2026-11-04T06:00:00Z')
    book.run()
    assert decisions()[0]['accepted'] == 1


def test_ineligible_ticker_is_rejected(book):
    transaction(ticker='ABSENT')
    book.run()
    assert decisions()[0]['reason'] == 'not_eligible'


def test_ticker_separators_are_removed(book):
    transaction(ticker='BRK.B')
    book.run()
    assert decisions()[0]['ticker'] == 'BRK-B'
    assert decisions()[0]['accepted'] == 1


def test_second_purchase_of_same_cik_is_rejected(book):
    transaction('t1', ticker='AAA')
    transaction('t2', ticker='AAB')
    book.run()
    assert [r['reason'] for r in decisions()] == [None, 'duplicate_issuer']


def test_missing_ciks_do_not_collapse_unrelated_tickers(book):
    transaction('t1', ticker='NOCIK')
    transaction('t2', ticker='OTHER')
    book.run()
    assert all(r['accepted'] for r in decisions())


def test_already_held_issuer_is_rejected_on_later_run(book):
    transaction()
    book.run()
    transaction('t2', ticker='AAB')
    book.run()
    assert decisions()[1]['reason'] == 'already_held'


def test_book_full_rejects_another_issuer(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_MAX_OPEN', '1')
    transaction()
    book.run()
    transaction('t2', ticker='BBB')
    book.run()
    assert decisions()[1]['reason'] == 'book_full'


def test_current_run_acceptances_use_capacity(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_MAX_OPEN', '1')
    transaction('t1')
    transaction('t2', ticker='BBB')
    book.run()
    assert decisions()[1]['reason'] == 'book_full'


@pytest.mark.parametrize('session,reason', [('2026-11-03', 'already_held'),
                                           ('2026-11-02', None)])
def test_hold_expires_after_configured_sessions(book, monkeypatch, session, reason):
    monkeypatch.setenv('NEMO_COPY_HOLD_DAYS', '3')
    transaction()
    book.run()
    with pit_store.connect() as conn:
        conn.execute('UPDATE copy_order SET intended_session=?', (session,))
    transaction('t2', ticker='AAB')
    book.run()
    assert decisions()[1]['reason'] == reason


def test_calendar_comes_from_scanner(book, monkeypatch):
    from research import scanner
    calls = []

    def next_session(day):
        calls.append(day)
        return '2026-11-09'

    monkeypatch.setattr(scanner, '_next_session', next_session)
    transaction()
    book.run()
    assert decisions()[0]['intended_session'] == '2026-11-09'
    assert '2026-11-04' in calls


def test_rejected_and_accepted_decisions_are_not_overwritten(book):
    transaction('t1', amount_min=10)
    transaction('t2', ticker='BBB')
    book.run()
    before = decisions()
    book.run()
    assert decisions() == before
    assert last_run()['rows_written'] == 0


def test_stale_house_sync_fails_run(book):
    with congress_store.connect() as conn:
        conn.execute("UPDATE sync_state SET last_synced_at='2026-11-02T12:29:59Z'")
    transaction()
    assert book.run()['status'] == 'failed'
    assert 'House' in last_run()['error'] and 'stale' in last_run()['error']
    assert decisions() == []


def test_sync_exactly_three_days_old_is_current(book):
    with congress_store.connect() as conn:
        conn.execute("UPDATE sync_state SET last_synced_at='2026-11-02T12:30:00Z'")
    assert book.run()['status'] == 'ok'


def test_missing_house_sync_fails_even_with_current_senate(book):
    with congress_store.connect() as conn:
        conn.execute("UPDATE sync_state SET source='senate_ptr'")
    assert book.run()['status'] == 'failed'


def test_current_sync_without_purchases_is_success(book):
    assert book.run()['status'] == 'ok'
    assert last_run()['rows_written'] == 0
    assert last_run()['finished_at'] is not None


def test_missing_congress_database_is_failure_not_new_database(book, monkeypatch, tmp_path):
    path = tmp_path / 'missing.db'
    monkeypatch.setenv('NEMO_CONGRESS_DB', str(path))
    assert book.run()['status'] == 'failed'
    assert not path.exists()


def test_copy_service_mounts_congress_data_writable_and_passes_settings():
    """The store is WAL. SQLite keeps a reader's shared-memory file beside it,
    so on a read-only mount even a mode=ro open fails with "unable to open
    database file" -- found by running the job against the VM's volume. The
    reader's mode=ro URI, not the mount, is what keeps it from writing."""
    root = Path(__file__).resolve().parents[1]
    services = yaml.safe_load((root / 'deploy/docker-compose.yml').read_text())['services']
    service = services['research-copy']
    assert service['profiles'] == ['sync']
    assert service['entrypoint'] == ['python', '-m', 'research.congress_copy']
    assert 'research-data:/app/data' in service['volumes']
    assert 'congress-data:/app/congress' in service['volumes']
    assert 'congress-data:/app/congress:ro' not in service['volumes']
    assert service['environment']['NEMO_CONGRESS_DB'] == '/app/congress/congress.db'
    for target in ['research-copy', 'research-status']:
        for setting in SETTINGS:
            name = 'NEMO_COPY_' + setting
            assert services[target]['environment'][name] == '${' + name + ':-}'


def test_copy_reader_filters_decision_date_and_time(book):
    transaction()
    book.run()
    assert pit_store.copy_orders_as_of('2026-11-04') == []
    assert pit_store.copy_orders_as_of('2026-11-05T12:29:59Z') == []
    assert len(pit_store.copy_orders_as_of('2026-11-05T12:30:00Z')) == 1
    with pit_store.connect() as conn:
        conn.execute("UPDATE copy_order SET as_of_date='2026-11-06'")
    assert pit_store.copy_orders_as_of('2026-11-05') == []


def test_copy_writer_keeps_first_decision(book):
    transaction()
    book.run()
    original = decisions()[0]
    changed = {**original, 'accepted': False, 'reason': 'book_full'}
    assert pit_store.record_copy_orders([changed]) == 0
    assert decisions() == [original]


def test_copy_schema_can_be_added_to_existing_store(book):
    with pit_store.connect() as conn:
        conn.execute('DROP TABLE copy_order')
    pit_store.set_cursor('kept', 7)
    pit_store.init_schema()
    pit_store.init_schema()
    assert decisions() == []
    assert pit_store.cursor_for('kept') == 7


def test_hold_period_counts_sessions_instead_of_calendar_days(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_HOLD_DAYS', '5')
    transaction()
    book.run()
    with pit_store.connect() as conn:
        conn.execute("UPDATE copy_order SET intended_session='2026-10-30'")
    transaction('t2', ticker='AAB')
    book.run()
    assert decisions()[1]['reason'] == 'already_held'


def test_failed_job_exits_nonzero(book, capsys):
    with congress_store.connect() as conn:
        conn.execute('DELETE FROM sync_state')
    assert book.main() == 1
    assert 'House' in capsys.readouterr().out


def test_successful_empty_job_exits_zero(book, monkeypatch):
    monkeypatch.setattr(
        'research.discord_post.finish',
        lambda job, code, result=None, **kwargs: code)
    assert book.main() == 0


def test_cron_runs_copy_at_1230_with_lock_and_log():
    root = Path(__file__).resolve().parents[1]
    doc = (root / 'deploy/README.md').read_text()
    line = next(line for line in doc.splitlines()
                if line.startswith('30 12 * * 1-5 '))
    assert 'date -u;' in line
    assert 'flock -n /var/lock/nemo-research-copy.lock' in line
    assert 'run --rm research-copy;' in line
    assert '>> /srv/nemo/logs/research-copy.log 2>&1' in line


def test_copy_reader_preserves_submillisecond_decision_cutoff(book):
    transaction(parsed_at='2026-11-05T12:29:59Z')
    book.run()
    with pit_store.connect() as conn:
        conn.execute("UPDATE copy_order SET decided_at='2026-11-05T12:30:00.000100Z'")
    assert pit_store.copy_orders_as_of('2026-11-05T12:30:00Z') == []


def test_copy_reader_keeps_parse_just_before_decision(book):
    transaction(parsed_at='2026-11-05T12:29:59.999999Z')
    book.run()
    assert len(pit_store.copy_orders_as_of('2026-11-05')) == 1


def test_off_entrypoint_does_not_create_store(book, monkeypatch):
    monkeypatch.setenv('NEMO_COPY_TRADING', 'off')
    monkeypatch.setattr(pit_store, 'init_schema', lambda: pytest.fail('opened PIT store'))
    monkeypatch.setattr(
        'research.discord_post.finish',
        lambda job, code, result=None, **kwargs: code)
    assert book.main() == 0
