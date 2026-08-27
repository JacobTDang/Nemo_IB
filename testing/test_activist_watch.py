"""A 13D watcher, and the two ways it can be wrong without looking wrong.

A Schedule 13D says someone has taken a >5% stake and intends to influence
management. Brav, Jiang, Partnoy & Thomas measure 6.3-8% abnormal returns
around the filing, and the 2024 rule change -- initial 13D now due in 5
business days rather than 10 calendar, amendments in 2 -- leaves less room to
accumulate quietly before disclosure, so proportionally more of the move lands
at publication. The edge is detection latency, not analysis: seeing the filing
when it posts rather than reading about it the next morning.

Which makes the two failure modes below the whole of this file.

**A quiet day and a broken watcher look identical from outside.** "No new 13Ds
today" and "EDGAR did not answer" are the same empty list and mean opposite
things -- one is information, the other is an outage that will be discovered
weeks later when someone asks why the record has a hole. Every pass therefore
writes a run-log entry saying which of the two happened, and a pass that
reached nobody is never logged as coverage.

**A company's EDGAR folder holds both sides of the relationship.** Filings
where it is the SUBJECT (someone took a stake in it) sit beside filings where
it is the FILER (it took a stake in someone else). Measured on INTC: 71 of the
first 100 rows were Intel filing on MariaDB, Mobileye, Joby and Vuzix, and
`activist_count` read 124 when the truth was 0 activists. A watcher that
repeats that mistake does not report a wrong number, it reports an activist
campaign that does not exist.

The rule that separates them is not reinvented here. EDGAR gives the SUBJECT of
a Schedule 13 an `005-` file number, present in the subject's own folder and
blank on filings it made about others -- verified 28 of 28 against header
ground truth on INTC. The submissions index carries it, so the whole folder is
classified for free, and only a filing that passes the free proxy is worth
spending a document fetch to verify against its header, which is the authority.
"""
import pytest

from research import activist_watch, daily_job, pit_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    """A live pass paces itself between SEC calls. Nothing in this file reaches
    SEC, so the sleep would buy nothing but a slower suite."""
    monkeypatch.setattr(activist_watch, "_be_gentle", lambda: None)


# --- fakes shaped like the submissions index and the filing header ---------

class _Info:
    def __init__(self, name, cik):
        self.name = name
        self.cik = cik


class _Party:
    def __init__(self, name, cik):
        self.company_information = _Info(name, cik)


class _Header:
    def __init__(self, subject, filer):
        self.subject_companies = [_Party(*subject)] if subject else []
        self.filers = [_Party(*filer)] if filer else []


class _Filing:
    """One row of a company's submissions index.

    `header` is a property that counts its own reads, because on a real filing
    it is a document fetch against SEC and the free file-number proxy exists
    precisely so that fetch is not spent on rows already known to be
    filer-side.
    """

    def __init__(self, accession, form, filing_date, file_number,
                 subject=None, filer=None, acceptance_datetime=None):
        self.accession_number = accession
        self.form = form
        self.filing_date = filing_date
        self.file_number = file_number
        self.acceptance_datetime = acceptance_datetime
        self.filing_url = f"https://sec.gov/{accession}.htm"
        self._header = _Header(subject, filer)
        self.header_reads = 0

    @property
    def header(self):
        self.header_reads += 1
        return self._header


INTC = ("INTEL CORP", "0000050863")
INTC_CIK = "50863"
VUZIX = ("Vuzix Corp", "0001463972")
ICAHN = ("ICAHN CARL C", "0000921669")


def _feed(monkeypatch, by_ticker, cik=INTC_CIK):
    """Install the one network seam: a ticker's Schedule 13 folder."""
    def fetch(ticker):
        if ticker not in by_ticker:
            raise LookupError(f"no fixture for {ticker}")
        return cik, by_ticker[ticker]

    monkeypatch.setattr(activist_watch, "_fetch_company_filings", fetch)


def _activist_13d(accession="a1", form="SC 13D", date="2026-08-26",
                  accepted="2026-08-26T20:31:15Z", subject=INTC):
    """Someone taking a stake in Intel: file number present, Intel is subject."""
    return _Filing(accession, form, date, "005-19567", subject, ICAHN,
                   acceptance_datetime=accepted)


def _intel_filing_on_someone_else(accession="b1", form="SC 13D",
                                  date="2021-01-29"):
    """Intel taking a stake in Vuzix: no `005-` number, Intel is the filer."""
    return _Filing(accession, form, date, "", VUZIX, INTC,
                   acceptance_datetime="2021-01-29T21:00:00Z")


# --- the headline property: a quiet day is not a broken watcher ------------

def test_a_quiet_day_is_a_successful_pass(store, monkeypatch):
    """Nobody filed a 13D on Intel today. That is a fact about the market and
    the record must carry it as one, or a real gap in coverage months later is
    indistinguishable from a market in which nothing happened."""
    _feed(monkeypatch, {"INTC": []})

    result = activist_watch.watch_pass(["INTC"], as_of="2026-08-26")

    assert result["status"] == "ok"
    assert result["new_events"] == 0
    assert result["covered"] == 1
    assert store.missing_days("activist_watch", "2026-08-26", "2026-08-26") \
        == [], "a quiet day was recorded as a day the watcher did not run"


def test_a_feed_that_did_not_answer_is_not_a_quiet_day(store, monkeypatch):
    """The failure this whole module is built around. An SEC rate limit returns
    the same empty list as a day with no activism, and treating them alike puts
    a silent outage into the record as evidence that nothing happened."""
    def throttled(ticker):
        raise ConnectionError("SEC returned 429")

    monkeypatch.setattr(activist_watch, "_fetch_company_filings", throttled)

    result = activist_watch.watch_pass(["INTC", "AAPL"], as_of="2026-08-26")

    assert result["status"] == "failed"
    assert result["new_events"] == 0
    assert "429" in str(result.get("error"))
    assert store.missing_days("activist_watch", "2026-08-26", "2026-08-26") \
        == ["2026-08-26"], "a pass that reached nobody was counted as coverage"


def test_a_partly_answered_pass_says_so(store, monkeypatch):
    """One name out of two is neither a working day nor an outage, and calling
    it either one throws away the number that tells them apart."""
    def half(ticker):
        if ticker == "AAPL":
            raise ConnectionError("SEC returned 429")
        return INTC_CIK, []

    monkeypatch.setattr(activist_watch, "_fetch_company_filings", half)

    result = activist_watch.watch_pass(["INTC", "AAPL"], as_of="2026-08-26")

    assert result["status"] == "partial"
    assert result["covered"] == 1
    assert result["requested"] == 2
    assert "AAPL" in str(result["error"]), "the name that failed went unnamed"


def test_an_empty_universe_is_a_failed_pass_not_a_quiet_one(store):
    """Watching nobody finds nothing, which is exactly what a working watcher
    reports on a quiet day. A pass with an empty watchlist has not observed the
    market at all and must not be able to claim it did."""
    result = activist_watch.watch_pass(as_of="2026-08-26")

    assert result["status"] == "failed"
    assert "universe" in str(result["error"]).lower()
    assert store.missing_days("activist_watch", "2026-08-26", "2026-08-26") \
        == ["2026-08-26"]


# --- never claim a stake from a filing the company made about someone else --

def test_a_filing_the_company_made_about_someone_else_is_not_an_event(
        store, monkeypatch):
    """The INTC shape, directly. Intel's 13D on Vuzix lives in Intel's own
    EDGAR folder, and reading it as an event would announce an activist in
    Intel on the strength of a stake Intel took in someone else."""
    _feed(monkeypatch, {"INTC": [
        _intel_filing_on_someone_else("b1"),
        _intel_filing_on_someone_else("b2", form="SC 13D/A", date="2021-06-30"),
    ]})

    result = activist_watch.watch_pass(["INTC"], as_of="2026-08-26")

    assert result["new_events"] == 0, (
        "a stake Intel took in Vuzix was reported as an activist in Intel")
    assert store.activist_filings_as_of("2026-08-26", ticker="INTC") == []
    assert result["status"] == "ok", (
        "correctly finding nothing was reported as a failure")
    assert result["filed_by_this_company"] == 2, (
        "the filer-side rows vanished without trace")


def test_the_filer_side_costs_no_document_fetch(store, monkeypatch):
    """Gentleness, made testable. The `005-` file number comes free with the
    submissions index; the header is a document fetch. Spending one on a row
    the free proxy has already rejected is how a watcher over a real watchlist
    earns a rate limit, and SEC rate-limits hard."""
    theirs = _intel_filing_on_someone_else("b1")
    _feed(monkeypatch, {"INTC": [theirs]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26")

    assert theirs.header_reads == 0, (
        "a document was fetched for a filing the file number already excluded")


def test_the_header_overrides_the_file_number_proxy(store, monkeypatch):
    """The proxy is free and was right 28 of 28 times on INTC. The header is
    the authority. Where they disagree the header wins and the disagreement is
    reported, so a drift in the heuristic surfaces instead of quietly seeding
    the store with events that never happened."""
    misfiled = _Filing("c1", "SC 13D", "2026-08-26", "005-19567",
                       subject=VUZIX, filer=INTC,
                       acceptance_datetime="2026-08-26T20:31:15Z")
    _feed(monkeypatch, {"INTC": [misfiled]})

    result = activist_watch.watch_pass(["INTC"], as_of="2026-08-26")

    assert result["new_events"] == 0, (
        "the header said Vuzix was the subject and the row was kept anyway")
    assert result["subject_filter_disagreements"] == 1, (
        "the heuristic drifted and nothing said so")


def test_a_subject_we_cannot_confirm_is_flagged_not_dropped(store, monkeypatch):
    """A header that will not parse is our failure, not an absence of filings.
    Dropping the row would make a watcher that cannot read anything
    indistinguishable from a company nobody has filed on."""
    unreadable = _Filing("d1", "SC 13D", "2026-08-26", "005-19567",
                         subject=None, filer=None,
                         acceptance_datetime="2026-08-26T20:31:15Z")
    _feed(monkeypatch, {"INTC": [unreadable]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    rows = store.activist_filings_as_of("2026-08-26", ticker="INTC")
    assert len(rows) == 1, "an unverifiable row was silently discarded"
    assert rows[0]["subject_verified"] is False


def test_the_store_refuses_a_filer_side_row(store):
    """The store is the last place a false event can be stopped. Whatever
    classifies filings today may be replaced; everything still writes through
    here, and a row that says Intel is both the filer and the subject of a
    stake in Vuzix cannot be true."""
    with pytest.raises(ValueError, match="subject"):
        store.record_activist_filings([{
            "accession": "b1", "subject_ticker": "INTC",
            "subject_cik": "1463972", "subject_name": "Vuzix Corp",
            "filer_name": "INTEL CORP", "filer_cik": "50863",
            "form": "SC 13D", "is_amendment": False,
            "filing_date": "2021-01-29", "accepted_at": None,
            "is_subject": False, "subject_verified": True,
        }], detected_at="2026-08-26T20:35:00Z")


# --- an initial 13D and an amendment are different events -------------------

def test_an_amendment_is_not_recorded_as_a_new_position(store, monkeypatch):
    """A 13D says a stake was built. A 13D/A says something about a stake
    already disclosed -- an increase, a sale, a settlement, sometimes an exit.
    The expected reaction differs in sign, so a consumer that cannot tell them
    apart is trading a coin flip."""
    _feed(monkeypatch, {"INTC": [
        _activist_13d("a1", form="SC 13D", date="2026-08-20",
                      accepted="2026-08-20T20:31:15Z"),
        _activist_13d("a2", form="SC 13D/A", date="2026-08-26",
                      accepted="2026-08-26T20:31:15Z"),
    ]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    rows = {r["accession"]: r
            for r in store.activist_filings_as_of("2026-08-26", ticker="INTC")}
    assert rows["a1"]["form"] == "SC 13D"
    assert rows["a1"]["is_amendment"] is False
    assert rows["a2"]["form"] == "SC 13D/A"
    assert rows["a2"]["is_amendment"] is True


def test_every_recorded_event_names_both_sides(store, monkeypatch):
    """Subject and filer, on every row. Without both, the one question this
    store exists to answer -- who took a stake in whom -- has to be re-derived
    from a URL, and the derivation is the part that was wrong before."""
    _feed(monkeypatch, {"INTC": [_activist_13d("a1")]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    row = store.activist_filings_as_of("2026-08-26", ticker="INTC")[0]
    assert row["subject_name"] == "INTEL CORP"
    assert row["subject_cik"] == "50863"
    assert row["filer_name"] == "ICAHN CARL C"
    assert row["filer_cik"] == "921669"
    assert row["subject_verified"] is True


# --- detection latency is the product ---------------------------------------

def test_detection_latency_is_measured_from_acceptance_to_detection(
        store, monkeypatch):
    """The number the whole watcher exists to make small. EDGAR's
    acceptanceDateTime is genuinely UTC despite the `Z` being the kind of
    suffix that is often decorative -- verified against the acceptance-hour
    distribution on INTC's 1000 most recent filings, which is empty from 03h to
    09h UTC, exactly the complement of the 06:00-22:00 ET filing window. Read
    as Eastern it would put every latency out by four hours."""
    _feed(monkeypatch, {"INTC": [
        _activist_13d("a1", accepted="2026-08-26T20:31:15Z")]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    row = store.activist_filings_as_of("2026-08-26", ticker="INTC")[0]
    assert row["accepted_at"] == "2026-08-26T20:31:15Z"
    assert row["detected_at"] == "2026-08-26T20:33:45Z"
    assert row["latency_seconds"] == 150.0


def test_a_filing_with_no_acceptance_time_has_no_latency_rather_than_zero(
        store, monkeypatch):
    """A latency of zero is a perfect catch. An unknown acceptance time is not
    one, and the two must never share a value in the column that measures how
    well this works."""
    _feed(monkeypatch, {"INTC": [_activist_13d("a1", accepted=None)]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    row = store.activist_filings_as_of("2026-08-26", ticker="INTC")[0]
    assert row["accepted_at"] is None
    assert row["latency_seconds"] is None
    assert row["is_backfill"] is None, "unknown was rounded to a verdict"


def test_history_found_on_a_first_look_is_not_a_slow_detection(
        store, monkeypatch):
    """The first pass over a ticker sees every 13D ever filed on it, and each
    one is new to us. Counting a 2019 filing as a detection five years late
    would bury the live number the watcher is judged on -- and it is not a
    detection at all, it is the baseline."""
    _feed(monkeypatch, {"INTC": [
        _activist_13d("old", date="2019-05-02",
                      accepted="2019-05-02T20:31:15Z"),
        _activist_13d("new", date="2026-08-26",
                      accepted="2026-08-26T20:31:15Z"),
    ]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    rows = {r["accession"]: r
            for r in store.activist_filings_as_of("2026-08-26", ticker="INTC")}
    assert rows["old"]["is_backfill"] is True
    assert rows["new"]["is_backfill"] is False

    report = activist_watch.latency_report(as_of="2026-08-26")
    assert report["live_detections"] == 1
    assert report["backfilled"] == 1
    assert report["median_latency_seconds"] == 150.0, (
        "a five-year-old filing was averaged into the detection latency")


# --- append-only, and the as-of discipline ----------------------------------

def test_a_filing_already_seen_is_not_reported_again(store, monkeypatch):
    """The watcher runs on a schedule, so it sees the same folder repeatedly.
    Re-reporting is worse than noise: a consumer sizing a position off "new
    13D" would take the trade every time the timer fires."""
    _feed(monkeypatch, {"INTC": [_activist_13d("a1")]})

    first = activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                                      detected_at="2026-08-26T20:33:45Z")
    second = activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                                       detected_at="2026-08-26T21:33:45Z")

    assert first["new_events"] == 1
    assert second["new_events"] == 0
    assert second["status"] == "ok", "a quiet re-run was reported as a failure"

    rows = store.activist_filings_as_of("2026-08-26", ticker="INTC")
    assert len(rows) == 1
    assert rows[0]["detected_at"] == "2026-08-26T20:33:45Z", (
        "the second pass overwrote when we first saw it -- which is the one "
        "number that cannot be re-derived later")


def test_a_detection_is_invisible_to_a_date_before_it_was_recorded(
        store, monkeypatch):
    """The store's whole discipline, applied to events. A 2019 filing learned
    today must not appear to a simulation standing in 2019, or the backtest
    that judges this signal is handed the answer."""
    _feed(monkeypatch, {"INTC": [
        _activist_13d("old", date="2019-05-02",
                      accepted="2019-05-02T20:31:15Z")]})

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                              detected_at="2026-08-26T20:33:45Z")

    assert store.activist_filings_as_of("2019-06-01", ticker="INTC") == [], (
        "a filing back-filled today was visible to a 2019 query")
    assert len(store.activist_filings_as_of("2026-08-26", ticker="INTC")) == 1


# --- who gets watched -------------------------------------------------------

def test_the_watchlist_defaults_to_the_eligible_universe(store, monkeypatch):
    """The universe is a recorded, dated fact rather than a list in a config
    file, so the set of names watched on any past day can be reconstructed --
    including the ones that have since been delisted."""
    store.record_universe("2026-08-26", [
        {"ticker": "INTC", "cik": "50863", "eligible": True},
        {"ticker": "TINY", "cik": "999", "eligible": False,
         "exclusion_reason": "below $500k median dollar volume"},
    ])
    asked = []

    def fetch(ticker):
        asked.append(ticker)
        return INTC_CIK, []

    monkeypatch.setattr(activist_watch, "_fetch_company_filings", fetch)

    result = activist_watch.watch_pass(as_of="2026-08-26")

    assert asked == ["INTC"], "an ineligible name was watched"
    assert result["status"] == "ok"


def test_a_capped_pass_is_never_reported_as_full_coverage(store, monkeypatch):
    """A cap keeps a universe sweep from fanning out into a rate limit, which
    is worth having. Reporting the truncated sweep as a complete one is not:
    the names past the cap were not watched and their silence means nothing."""
    store.record_universe("2026-08-26", [
        {"ticker": t, "cik": str(i), "eligible": True}
        for i, t in enumerate(["AAA", "BBB", "CCC"])])
    monkeypatch.setattr(activist_watch, "_fetch_company_filings",
                        lambda ticker: (INTC_CIK, []))

    result = activist_watch.watch_pass(as_of="2026-08-26", max_tickers=2)

    assert result["status"] == "partial"
    assert result["requested"] == 3
    assert "cap" in str(result["error"]).lower()


def test_the_universe_is_read_as_it_stood_on_the_day(store, monkeypatch):
    """A name eligible in March and delisted in June was watched in March. A
    watcher that reads today's universe when replaying a March pass silently
    rewrites who was being observed."""
    store.record_universe("2026-03-02", [{"ticker": "GONE", "cik": "1",
                                          "eligible": True}])
    store.record_universe("2026-08-26", [{"ticker": "INTC", "cik": "50863",
                                          "eligible": True}])
    asked = []
    monkeypatch.setattr(activist_watch, "_fetch_company_filings",
                        lambda ticker: (asked.append(ticker), (INTC_CIK, []))[1])

    activist_watch.watch_pass(as_of="2026-03-02")

    assert asked == ["GONE"]


def test_one_unreadable_filing_does_not_take_down_the_whole_sweep(
        store, monkeypatch):
    """A pass covers a watchlist, and the names are independent. One filing
    EDGAR describes in a way we cannot parse should cost that name and no
    other -- otherwise a single malformed row anywhere in the watchlist
    silences every name after it, and the pass dies without a finished run
    while the market is exactly where a 13D might be landing."""
    def feed(ticker):
        if ticker == "BAD":
            # Genuinely subject-side -- header and file number both agree --
            # so it reaches the point where the timestamp is read.
            return INTC_CIK, [_activist_13d("x1", accepted="the day before")]
        return INTC_CIK, [_activist_13d("a1")]

    monkeypatch.setattr(activist_watch, "_fetch_company_filings", feed)

    result = activist_watch.watch_pass(["BAD", "INTC"], as_of="2026-08-26",
                                       detected_at="2026-08-26T20:33:45Z")

    assert result["status"] == "partial"
    assert "BAD" in str(result["error"])
    assert result["new_events"] == 1, "the healthy name was lost with the bad one"
    assert len(store.activist_filings_as_of("2026-08-26", ticker="INTC")) == 1
    assert activist_watch.last_run()["finished_at"], (
        "the pass died leaving a run that never finished")


def test_a_reported_event_matches_what_was_stored(store, monkeypatch):
    """The pass returns the events it found and the store holds them. A
    consumer that reads one and joins against the other must not have to
    normalise a timestamp first -- the same field in two shapes is how a
    latency gets computed against a string in one place and a datetime in
    another."""
    from datetime import datetime, timezone

    accepted = datetime(2026, 8, 26, 20, 31, 15, tzinfo=timezone.utc)
    _feed(monkeypatch, {"INTC": [_activist_13d("a1", accepted=accepted)]})

    result = activist_watch.watch_pass(["INTC"], as_of="2026-08-26",
                                       detected_at="2026-08-26T20:33:45Z")

    stored = store.activist_filings_as_of("2026-08-26", ticker="INTC")[0]
    assert result["events"][0]["accepted_at"] == stored["accepted_at"]
    assert stored["accepted_at"] == "2026-08-26T20:31:15Z"


def test_a_ticker_edgar_cannot_resolve_is_a_failure_not_a_quiet_day(store):
    """Found live. `Company("PARA")` does not raise for a ticker EDGAR no
    longer maps -- it hands back a company whose CIK is the sentinel
    -999999999 and whose filing list is empty. Empty is the same shape a
    genuinely quiet name returns, so an unresolvable ticker would be recorded
    as "nobody filed a 13D on it" every pass, forever, and would count as
    coverage while doing it. Exactly the confusion this module exists to
    prevent, arriving through the vendor rather than the network."""
    with pytest.raises(LookupError, match="PARA"):
        activist_watch._resolved_cik("PARA", -999999999)

    assert activist_watch._resolved_cik("INTC", 50863) == "50863"


# --- the run log tells the difference ---------------------------------------

def test_each_pass_leaves_its_own_run_log_entry(store, monkeypatch):
    """Two passes in a day are normal -- the watcher is a timer, not a daily
    job. Each records separately so a day where the 09:00 pass worked and the
    16:00 pass was throttled is visible as exactly that."""
    calls = {"n": 0}

    def flaky(ticker):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("SEC returned 429")
        return INTC_CIK, []

    monkeypatch.setattr(activist_watch, "_fetch_company_filings", flaky)

    activist_watch.watch_pass(["INTC"], as_of="2026-08-26")
    activist_watch.watch_pass(["INTC"], as_of="2026-08-26")

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT status FROM run_log WHERE job = 'activist_watch' "
            "ORDER BY run_id").fetchall()
    assert [r["status"] for r in rows] == ["ok", "failed"]
    assert activist_watch.last_run()["status"] == "failed"
