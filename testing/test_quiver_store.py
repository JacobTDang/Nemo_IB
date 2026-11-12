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
