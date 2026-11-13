"""Who a company says it buys from, and the two ways that fact can lie.

The extractor is sound at one thing: it reads the Business section of a 10-K
and matches the names in it against a curated list of large caps, putting each
one on a side. Everything that can go wrong is around that reading, and the
tests here are about the around:

  * the fact belongs to the filing, so it is dated to the filing and not to
    the night a job got round to reading it -- and when those differ, the row
    says so;
  * a filing that named nobody and a filing that could not be read are not the
    same fact, and only one of them is coverage;
  * a 10-K comes once a year, so the pacing has to skip a name that is current
    without skipping it forever.

Everything here runs offline: `_extract` is the seam the nightly pass uses,
and it is replaced in every test that reads a name.
"""
from datetime import date

import pytest

from research import pit_store, supply_chain


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _payload(ticker="AAA", filed="2026-11-03",
             links=(("TSM", "supplier"),), success=True, error=None):
    """One name's answer, in the shape `get_supply_chain` returns."""
    return {
        "ticker": ticker, "success": success, "error": error,
        # The extractor hands back a `datetime.date`, not a string.
        "filing_date": date.fromisoformat(filed) if filed else None,
        "related_companies": [
            {"name_matched": related.lower(), "ticker": related,
             "mention_count": 3, "relationship": side,
             "relationship_basis": "cue phrase near the mention",
             "sample_context": "…"}
            for related, side in links],
        "suppliers": [related for related, side in links
                      if side == "supplier"],
        "customers": [related for related, side in links
                      if side == "customer"],
        "competitors": [related for related, side in links
                        if side == "competitor"],
        "partners": [related for related, side in links
                     if side == "partner"],
        "unclassified": [related for related, side in links
                         if side == "unknown"],
        "related_count": len(links),
        "trigger_sentences": [],
    }


def _run(monkeypatch, payloads):
    """The extractor, answering per ticker from `payloads` (or a callable)."""
    def extract(ticker):
        if callable(payloads):
            return payloads(ticker)
        return payloads.get(ticker, _payload(ticker=ticker))
    monkeypatch.setattr(supply_chain, "_extract", extract)
    return extract


# ------------------------------------------------------------- what is stored

def test_a_link_is_stamped_the_day_the_filing_was_public(store, monkeypatch):
    """The fact belongs to the 10-K, not to the night the job read it.

    A reader standing on the filing date could have read that document, so it
    sees the link; one standing the day before cannot, because the filing did
    not exist. The read happened a day later and the row says so.
    """
    _run(monkeypatch, {"AAA": _payload()})

    result = supply_chain.record(["AAA"], as_of="2026-11-04")

    assert result["written"] == 1, result
    row = store.supply_chain_as_of("AAA", "2026-11-04")[0]
    assert row["filing_date"] == "2026-11-03", row
    assert row["relationship"] == "supplier", row
    assert row["basis"] == "cue phrase near the mention", row
    assert row["backfilled"] == 1, row
    assert store.supply_chain_as_of("AAA", "2026-11-03")
    assert store.supply_chain_as_of("AAA", "2026-11-02") == []


def test_a_link_read_on_the_filing_date_is_not_a_reconstruction(store,
                                                                monkeypatch):
    _run(monkeypatch, {"AAA": _payload()})

    supply_chain.record(["AAA"], as_of="2026-11-03")

    assert store.supply_chain_as_of("AAA", "2026-11-03")[0]["backfilled"] == 0


def test_the_side_lists_are_not_counted_twice(store, monkeypatch):
    """`related_companies` carries each match with its side; the top-level
    `suppliers`/`customers` lists are the same names as strings. Reading both
    would double every edge, which looks like twice as much coverage."""
    payload = _payload(links=(("TSM", "supplier"), ("MU", "supplier")))
    assert payload["suppliers"] == ["TSM", "MU"]
    _run(monkeypatch, {"AAA": payload})

    assert supply_chain.record(["AAA"], as_of="2026-11-03")["written"] == 2


def test_a_foreign_listing_is_kept_as_the_filing_named_it(store, monkeypatch):
    """`000660.KS` cannot be priced here and is still true about the filing.
    Dropping it would quietly shrink the graph to the names that trade."""
    _run(monkeypatch, {"AAA": _payload(links=(("000660.KS", "supplier"),
                                              ("2317.TW", "supplier")))})

    supply_chain.record(["AAA"], as_of="2026-11-03")

    related = {r["related_ticker"]
               for r in store.supply_chain_as_of("AAA", "2026-11-03")}
    assert related == {"000660.KS", "2317.TW"}, related


def test_a_company_does_not_supply_itself(store):
    """The extractor excludes the filer from its own list. A row that survived
    would read as a data point rather than as the bug it is."""
    written = store.record_supply_chain("AAA", "2026-11-03", [{
        "related_ticker": "AAA", "relationship": "competitor",
        "mention_count": 5}])

    assert written == 0
    assert store.supply_chain_as_of("AAA", "2026-11-03") == []


def test_a_newer_filing_replaces_what_a_reader_stands_on(store, monkeypatch):
    """A company that has filed five 10-Ks has described its suppliers five
    times. A flat list of all of them would say it buys from a company it
    stopped buying from four years ago."""
    _run(monkeypatch, {"AAA": _payload(filed="2025-02-25",
                                       links=(("TSM", "supplier"),))})
    supply_chain.record(["AAA"], as_of="2025-02-25")
    _run(monkeypatch, {"AAA": _payload(filed="2026-02-25",
                                       links=(("MU", "supplier"),))})
    supply_chain.record(["AAA"], as_of="2026-02-25")

    assert [r["related_ticker"]
            for r in store.supply_chain_as_of("AAA", "2026-02-26")] == ["MU"]
    # And the older read still answers for the older date: a new filing is a
    # new set of rows, not a rewrite of the last one.
    assert [r["related_ticker"]
            for r in store.supply_chain_as_of("AAA", "2025-06-01")] == ["TSM"]


def test_the_reverse_view_is_each_filers_newest_filing(store):
    """The read-through direction: a supplier's print matters to the names
    downstream of it, and each of them contributes one filing, not five."""
    store.record_supply_chain("AAA", "2025-02-25", [{
        "related_ticker": "TSM", "relationship": "supplier",
        "mention_count": 1}])
    store.record_supply_chain("AAA", "2026-02-25", [{
        "related_ticker": "TSM", "relationship": "supplier",
        "mention_count": 3}])
    store.record_supply_chain("BBB", "2026-03-01", [{
        "related_ticker": "TSM", "relationship": "customer",
        "mention_count": 2}])

    edges = store.supply_chain_edges_as_of("TSM", "2026-06-01")

    assert {(e["ticker"], e["relationship"], e["mention_count"])
            for e in edges} == {("AAA", "supplier", 3),
                                ("BBB", "customer", 2)}, edges
    # Before either filing, nothing.
    assert store.supply_chain_edges_as_of("TSM", "2025-01-01") == []


# ------------------------------------------------- empty is not a failure

def test_a_filing_that_names_nobody_is_ok_and_says_zero(store, monkeypatch):
    """Most filings name nobody on a 150-name list. That is an answer about
    the extractor's coverage, and writing it as a missing row would make a
    read that found nothing look like a read that never happened."""
    _run(monkeypatch, {"AAA": _payload(links=())})

    result = supply_chain.record(["AAA"], as_of="2026-11-03")

    assert result["status"] == "ok", result
    assert result["captured"] == 1 and result["zero_link"] == 1, result
    assert store.supply_chain_as_of("AAA", "2026-11-03") == []
    capture = store.supply_chain_captures()["AAA"]
    assert capture["outcome"] == "ok", capture
    assert capture["links"] == 0, capture
    assert capture["filing_date"] == "2026-11-03", capture


def test_a_failed_read_is_recorded_as_one_and_retried_in_a_month(
        store, monkeypatch):
    """Not captured as zero links: a filing that could not be read is not a
    filing that named nobody, and the retry is what tells them apart."""
    _run(monkeypatch, {"AAA": _payload(success=False,
                                       error="Item 1 body too short")})

    result = supply_chain.record(["AAA"], as_of="2026-11-03")

    assert result["status"] == "failed", result
    assert "Item 1 body too short" in result["failed"][0], result
    capture = store.supply_chain_captures()["AAA"]
    assert capture["outcome"] == "failed", capture
    assert "Item 1 body too short" in capture["reason"], capture
    assert capture["filing_date"] is None, capture
    # Not again tonight, and not the status of never having read it.
    assert supply_chain.due(["AAA"], "2026-11-04") == []
    assert supply_chain.due(["AAA"], "2026-12-03") == ["AAA"]


def test_an_extractor_that_raises_is_a_failure_not_a_crash(store,
                                                           monkeypatch):
    def boom(ticker):
        raise ConnectionError("EDGAR answered 503")

    _run(monkeypatch, boom)

    result = supply_chain.record(["AAA"], as_of="2026-11-03")

    assert result["status"] == "failed", result
    assert "EDGAR answered 503" in result["failed"][0], result


def test_one_broken_name_does_not_take_the_slice(store, monkeypatch):
    def extract(ticker):
        if ticker == "BBB":
            raise ConnectionError("EDGAR answered 503")
        return _payload(ticker=ticker)

    _run(monkeypatch, extract)

    result = supply_chain.record(["AAA", "BBB", "CCC"], as_of="2026-11-03")

    assert result["status"] == "partial", result
    assert result["captured"] == 2 and result["written"] == 2, result
    assert result["failed"][0].startswith("BBB: "), result
    with store.connect() as conn:
        run = conn.execute("SELECT * FROM run_log WHERE job = 'supply_chain'"
                           ).fetchone()
    assert run["status"] == "partial", dict(run)
    assert "BBB" in run["error"], dict(run)


def test_a_10k_dated_after_the_date_asked_for_is_not_written(store,
                                                             monkeypatch):
    """The extractor always returns the newest filing, so a replay of June
    would otherwise capture November's suppliers under a June run. The row
    would be dated November and invisible to June -- but the run would say it
    had read June."""
    _run(monkeypatch, {"AAA": _payload(filed="2026-11-03")})

    result = supply_chain.record(["AAA"], as_of="2026-06-01")

    assert result["status"] == "failed", result
    assert "after the 2026-06-01" in result["failed"][0], result
    assert store.supply_chain_as_of("AAA", "2026-06-01") == []
    assert store.supply_chain_captures()["AAA"]["outcome"] == "failed"


def test_a_payload_without_a_filing_date_is_refused(store, monkeypatch):
    """Without a date there is nothing to stamp the facts with, and inventing
    today would put a filing on a day it was not filed."""
    _run(monkeypatch, {"AAA": _payload(filed=None)})

    result = supply_chain.record(["AAA"], as_of="2026-11-03")

    assert result["status"] == "failed", result
    assert "no filing date" in result["failed"][0], result
    assert store.supply_chain_as_of("AAA", "2026-11-03") == []


# ------------------------------------------------------------------- pacing

def _captured(store, ticker, filed, captured_at):
    store.record_supply_chain_capture(ticker, filed, 3, "ok",
                                      captured_at=captured_at)


def test_a_filing_inside_its_year_is_not_read_again(store):
    _captured(store, "AAA", "2026-02-25", "2026-02-26T21:00:00Z")

    assert supply_chain.due(["AAA"], "2026-11-03") == []
    # A year after the filing, the next 10-K is expected: read it again.
    assert supply_chain.due(["AAA"], "2027-02-26") == ["AAA"]


def test_past_the_anniversary_it_is_asked_monthly_not_nightly(store):
    """A company can file late, move its fiscal year end, or stop filing. A
    nightly retry of the same document would spend the whole slice on it."""
    _captured(store, "AAA", "2026-02-25", "2027-03-01T21:00:00Z")

    assert supply_chain.due(["AAA"], "2027-03-02") == []
    assert supply_chain.due(["AAA"], "2027-03-31") == ["AAA"]


def test_a_company_that_has_not_filed_again_writes_nothing_new(store,
                                                               monkeypatch):
    """The re-read is not a no-op in the log: it moves the clock forward, so
    the name is not asked again for another month."""
    _run(monkeypatch, {"AAA": _payload()})
    supply_chain.record(["AAA"], as_of="2026-11-03")

    result = supply_chain.record(["AAA"], as_of="2027-11-05")

    assert result["asked"] == 1, result
    assert result["written"] == 0, result
    assert result["captured"] == 1, result
    capture = store.supply_chain_captures()["AAA"]
    assert capture["captured_at"].startswith("2027-11-05"), capture
    assert supply_chain.due(["AAA"], "2027-11-06") == []


def test_never_read_names_come_first_then_the_stalest(store):
    """The eligible universe is ordered by ticker, so a slice that took the
    first due names without sorting would spend the first month on the A's."""
    _captured(store, "OLD", "2025-01-01", "2025-01-02T21:00:00Z")
    _captured(store, "MID", "2025-06-01", "2025-06-02T21:00:00Z")

    assert supply_chain.due(["MID", "NEW", "OLD"], "2026-11-03") == \
        ["NEW", "OLD", "MID"]


def test_the_pass_is_bounded_to_the_slice_it_was_given(store, monkeypatch):
    _run(monkeypatch, {"AAA": _payload(ticker="AAA"),
                       "BBB": _payload(ticker="BBB"),
                       "CCC": _payload(ticker="CCC")})

    result = supply_chain.record(["AAA", "BBB", "CCC"], as_of="2026-11-03",
                                 limit=2)

    assert result["universe"] == 3 and result["asked"] == 2, result
    assert store.supply_chain_captures().keys() == {"AAA", "BBB"}


def test_a_night_with_nothing_due_still_logs_a_run(store, monkeypatch):
    """A stage that writes nothing and says nothing is indistinguishable from
    one that is not scheduled, and the status screen reads the run log."""
    _run(monkeypatch, lambda ticker: pytest.fail(
        "read a name that was not due"))
    _captured(store, "AAA", "2026-11-03", "2026-11-03T21:00:00Z")

    result = supply_chain.record(["AAA"], as_of="2026-11-04")

    assert result["status"] == "ok" and result["asked"] == 0, result
    with store.connect() as conn:
        run = conn.execute("SELECT status, rows_written FROM run_log "
                           "WHERE job = 'supply_chain'").fetchone()
    assert run["status"] == "ok" and run["rows_written"] == 0, dict(run)


def test_the_names_default_to_the_universe_the_store_holds(store, monkeypatch):
    store.record_universe("2026-11-03", [
        {"ticker": "AAA", "cik": "1", "eligible": True},
        {"ticker": "ZZZ", "cik": "2", "eligible": False,
         "exclusion_reason": "insufficient history"}],
        recorded_at="2026-11-03T21:00:00Z")
    _run(monkeypatch, {"AAA": _payload()})

    result = supply_chain.record(as_of="2026-11-03")

    assert result["universe"] == 1, result
    assert store.supply_chain_as_of("AAA", "2026-11-03")


# -------------------------------------------------------------- entry point

def test_the_entry_point_exits_non_zero_when_every_name_failed(
        store, monkeypatch, capsys):
    _run(monkeypatch, {"AAA": _payload(success=False, error="empty text")})

    assert supply_chain.main(["--tickers", "AAA", "--as-of",
                             "2026-11-03"]) == 1
    assert "empty text" in capsys.readouterr().out


def test_the_entry_point_exits_zero_on_a_partial_night(store, monkeypatch):
    """Two names read and one EDGAR timeout is not a failed job; the exit code
    is what cron mails about."""
    _run(monkeypatch, lambda ticker: (_payload(ticker=ticker)
                                      if ticker != "BBB"
                                      else (_ for _ in ()).throw(
                                          ConnectionError("EDGAR 503"))))

    assert supply_chain.main(["--tickers", "AAA", "BBB",
                             "--as-of", "2026-11-03"]) == 0
