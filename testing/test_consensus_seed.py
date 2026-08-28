"""Filling in the consensus history that cannot be recorded backwards.

The analyst surprise needs eight quarters of what the street expected before
each print. Finnhub serves four quarters and no more -- verified at limit=12
and limit=30 alike -- so the recorder fills the rest one day at a time and
sue_af refuses for two years while it does.

The four it does serve carry both legs: a vendor estimate and a vendor actual,
on one basis. The question is whether that estimate is the one that stood
before the print or one revised afterwards, because a revised estimate has
converged toward the actual and would shrink every surprise and every sigma --
inflating the signal, which is the direction that flatters.

It is frozen. Ten estimates recorded from the forward calendar on 2026-08-26,
before those companies reported, come back byte-identical from the surprises
endpoint after they reported: NVDA 2.1283, CRWD 0.2984, OKTA 0.9841, DY 4.8221,
MOV 0.3560, PLAB 0.4114, BJ 1.1951, DG 2.0559, HPQ 0.6639, CRM 3.3057. Ten of
ten, to four decimal places, checked against this project's own point-in-time
recording rather than against the vendor's word for it.

So seeding is sound, and it still has to be visible. A seeded row is a
reconstruction stamped at a date we were not watching, and nothing downstream
may treat it as something we saw happen.
"""
import pytest

from research import pit_store, seed_consensus


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


SURPRISES = [
    {"year": 2026, "quarter": 4, "period": "2026-06-30",
     "estimate_eps": 4.3274, "actual_eps": 4.74},
    {"year": 2026, "quarter": 3, "period": "2026-03-31",
     "estimate_eps": 4.1432, "actual_eps": 4.27},
]

# The filer's own dates, which is where the stamps come from. The vendor's
# `period` field is a calendar bucket and cannot be used for this.
FILED = {
    "2026Q4": {"period_end": "2026-06-30", "known_at": "2026-07-29"},
    "2026Q3": {"period_end": "2026-03-31", "known_at": "2026-04-29"},
}


@pytest.fixture(autouse=True)
def _filings(monkeypatch):
    monkeypatch.setattr(seed_consensus, "_filing_dates",
                        lambda t, as_of=None: FILED)
    # No release on record by default, so the filing date stands and the tests
    # written before announcement-dating keep their expectations. The ones
    # that are about the release override this.
    monkeypatch.setattr(seed_consensus, "_announcements",
                        lambda t, as_of=None: {})


def test_a_seeded_quarter_carries_both_legs(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)

    seed_consensus.seed(["MSFT"])

    assert pit_store.actual_as_of("MSFT", "2026Q4", as_of="2026-12-31") == 4.74
    snap = pit_store.consensus_as_of("MSFT", "2026Q4", as_of="2026-12-31")
    assert snap["eps_estimate"] == 4.3274


def test_the_estimate_is_readable_before_the_print(store, monkeypatch):
    """Which is the whole point of it. Stamped after the announcement it would
    be invisible to the only lookup that wants it."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    seed_consensus.seed(["MSFT"])

    snap = pit_store.consensus_as_of("MSFT", "2026Q4", as_of="2026-06-30")
    assert snap is not None and snap["eps_estimate"] == 4.3274


def test_the_actual_is_not_readable_before_the_print(store, monkeypatch):
    """It had not happened. A seeded row may be a reconstruction, but it must
    not reconstruct the future."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    seed_consensus.seed(["MSFT"])

    assert pit_store.actual_as_of("MSFT", "2026Q4", as_of="2026-06-30") is None
    assert pit_store.actual_as_of("MSFT", "2026Q4", as_of="2026-07-29") == 4.74


def test_a_seeded_row_says_it_was_seeded(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    seed_consensus.seed(["MSFT"])

    with pit_store.connect() as conn:
        sources = {r["source"] for r in
                   conn.execute("SELECT source FROM consensus_snapshot")}
    assert sources == {"seeded"}


def test_a_recorded_row_says_recorded(store):
    """The default, so the daily job needs no change and nothing already in a
    store is retroactively relabelled as a reconstruction."""
    store.record_consensus("2026-03-01", "AAPL", "2026Q2", eps_estimate=1.10,
                           recorded_at="2026-03-01T21:00:00Z")
    with pit_store.connect() as conn:
        row = conn.execute("SELECT source FROM consensus_snapshot").fetchone()
    assert row["source"] == "recorded"


def test_seeding_never_overwrites_something_that_was_watched(store,
                                                             monkeypatch):
    """A real observation outranks a reconstruction of the same quarter."""
    store.record_consensus("2026-06-30", "MSFT", "2026Q4", eps_estimate=9.99,
                           recorded_at="2026-06-30T21:00:00Z")
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)

    seed_consensus.seed(["MSFT"])

    # The read that matters is the pre-print one, because that is where the
    # analyst surprise takes its estimate from. The observed row stands there.
    before = pit_store.consensus_as_of("MSFT", "2026Q4", as_of="2026-07-01")
    assert before["eps_estimate"] == 9.99
    assert before["source"] == "recorded"


def test_seeding_twice_changes_nothing(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    first = seed_consensus.seed(["MSFT"])
    second = seed_consensus.seed(["MSFT"])
    assert first["written"] > 0
    assert second["written"] == 0


def test_a_row_without_both_legs_is_not_seeded(store, monkeypatch):
    """Half a pair cannot make a surprise, and a row with one leg would look
    like coverage this does not have."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises", lambda t: [
        {"year": 2026, "quarter": 4, "period": "2026-06-30",
         "estimate_eps": None, "actual_eps": 4.74},
        {"year": 2026, "quarter": 3, "period": "2026-03-31",
         "estimate_eps": 4.1432, "actual_eps": None},
    ])

    out = seed_consensus.seed(["MSFT"])
    assert out["written"] == 0
    assert out["incomplete"] == 2


# --- the vendor's period field is a calendar bucket -------------------------
#
# Finnhub's `year` and `quarter` are the filer's own, but its `period` is not:
# it reports NVDA's fiscal 2027 Q2 with period=2027-06-30, a quarter that
# actually ended 2026-07-26 and was announced a month later. Stamping a seeded
# estimate at that date puts it a year in the future, where the lookup that
# wants it cannot see it -- which is exactly what happened live: NVDA, COST,
# WMT, CRM and ORCL all came back "no consensus was recorded before that print"
# from a store that had just seeded them.
#
# The filer's real dates are in the XBRL series, keyed on the same fiscal
# identity, and that is what the stamps come from.

OFF_CALENDAR = [
    {"year": 2027, "quarter": 2, "period": "2027-06-30",
     "estimate_eps": 2.1283, "actual_eps": 2.22},
]

XBRL_QUARTERS = {
    "2027Q2": {"period_end": "2026-07-26", "known_at": "2026-08-26"},
}


def test_a_seeded_stamp_comes_from_the_filing_not_the_bucket(store,
                                                             monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: OFF_CALENDAR)
    monkeypatch.setattr(seed_consensus, "_filing_dates",
                        lambda t, as_of=None: XBRL_QUARTERS)  # overrides the fixture

    seed_consensus.seed(["NVDA"], as_of="2026-08-27")

    # Readable on the day the scan runs, which the calendar bucket would not be.
    snap = pit_store.consensus_as_of("NVDA", "2027Q2", as_of="2026-08-27")
    assert snap is not None, "the estimate was stamped into the future"
    assert snap["eps_estimate"] == 2.1283
    # ...and before the print, which is where the surprise lookup reads it.
    assert pit_store.consensus_as_of("NVDA", "2027Q2",
                                     as_of="2026-08-01") is not None


def test_the_seeded_actual_lands_on_the_filing_date(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: OFF_CALENDAR)
    monkeypatch.setattr(seed_consensus, "_filing_dates",
                        lambda t, as_of=None: XBRL_QUARTERS)  # overrides the fixture

    seed_consensus.seed(["NVDA"], as_of="2026-08-27")

    assert pit_store.actual_as_of("NVDA", "2027Q2", "2026-08-25") is None
    assert pit_store.actual_as_of("NVDA", "2027Q2", "2026-08-27") == 2.22


def test_a_quarter_the_filings_do_not_cover_is_skipped(store, monkeypatch):
    """Without the filer's own dates there is nothing to stamp it at, and
    guessing is what put an estimate a year into the future."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: OFF_CALENDAR)
    monkeypatch.setattr(seed_consensus, "_filing_dates",
                        lambda t, as_of=None: {})  # overrides the fixture

    out = seed_consensus.seed(["NVDA"], as_of="2026-08-27")
    assert out["written"] == 0
    assert out["undated"] == 1


# --- the gap a conservative skip leaves -------------------------------------

def test_a_quarter_missing_only_its_actual_is_still_seeded(store, monkeypatch):
    """The daily job records estimates for everything on the calendar. If it
    was down on the day a quarter reported, that quarter has an estimate and no
    actual -- and skipping on the mere presence of a row leaves it half-filled
    forever, which is one quarter fewer for a window that needs six.
    """
    store.record_consensus("2026-06-01", "MSFT", "2026Q4", eps_estimate=4.3274,
                           recorded_at="2026-06-01T21:00:00Z")
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)

    out = seed_consensus.seed(["MSFT"])

    assert pit_store.actual_as_of("MSFT", "2026Q4", as_of="2026-12-31") == 4.74
    assert out["written"] > 0


def test_a_quarter_that_already_has_its_actual_is_left_alone(store, monkeypatch):
    """A real observation outranks a reconstruction of the same thing."""
    store.record_consensus("2026-07-29", "MSFT", "2026Q4", eps_estimate=4.30,
                           eps_actual=4.70, recorded_at="2026-07-29T21:00:00Z")
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)

    seed_consensus.seed(["MSFT"])

    assert pit_store.actual_as_of("MSFT", "2026Q4", as_of="2026-12-31") == 4.70
    with pit_store.connect() as conn:
        rows = conn.execute(
            "SELECT source FROM consensus_snapshot WHERE fiscal_period='2026Q4'"
        ).fetchall()
    assert all(r["source"] == "recorded" for r in rows)


def test_the_recorded_estimate_survives_the_seeding_of_its_actual(store,
                                                                  monkeypatch):
    """Filling the missing leg must not restate the leg that was observed."""
    store.record_consensus("2026-06-01", "MSFT", "2026Q4", eps_estimate=9.99,
                           recorded_at="2026-06-01T21:00:00Z")
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)

    seed_consensus.seed(["MSFT"])

    before = pit_store.consensus_as_of("MSFT", "2026Q4", as_of="2026-06-15")
    assert before["eps_estimate"] == 9.99
    assert before["source"] == "recorded"


def test_a_name_whose_labels_never_match_is_named(store, monkeypatch):
    """Finnhub's year and quarter are usually the filer's own, and for some
    names they are not: DG's vendor labels run a full year ahead of its XBRL
    ones, so not one of its four quarters matches and it can never be seeded.
    Skipping is right -- guessing a stamp is what put an estimate a year in the
    future. Skipping silently is not: the count said `undated: 9` and nothing
    said which names would stay uncovered forever."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: OFF_CALENDAR)
    monkeypatch.setattr(seed_consensus, "_filing_dates",
                        lambda t, as_of=None: {})

    out = seed_consensus.seed(["DG"], as_of="2026-08-27")

    assert out["undated"] == 1
    assert "DG" in out["unmatched"]
    assert "2027Q2" in out["unmatched"]["DG"]


def test_a_name_that_matched_is_not_listed_as_unmatched(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    out = seed_consensus.seed(["MSFT"])
    assert out["unmatched"] == {}


def test_two_vendor_rows_for_one_quarter_seed_neither(store, monkeypatch):
    """Finnhub returns TGT's 2027Q2 twice, with different calendar buckets and
    the same actual. Which one is the quarter is not answerable from the
    payload, and taking whichever arrives first would attach an estimate to a
    quarter on a coin flip. The server already counts these as
    `duplicate_fiscal_periods`; here they are simply not seeded."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises", lambda t: [
        {"year": 2027, "quarter": 2, "period": "2026-09-30",
         "estimate_eps": 2.40, "actual_eps": 2.46},
        {"year": 2027, "quarter": 2, "period": "2025-09-30",
         "estimate_eps": 2.31, "actual_eps": 2.46},
        {"year": 2026, "quarter": 4, "period": "2026-03-31",
         "estimate_eps": 4.1432, "actual_eps": 4.27},
    ])
    monkeypatch.setattr(seed_consensus, "_filing_dates", lambda t, as_of=None: {
        "2027Q2": {"period_end": "2026-08-01", "known_at": "2026-08-20"},
        "2026Q4": {"period_end": "2026-03-31", "known_at": "2026-04-29"},
    })

    out = seed_consensus.seed(["TGT"], as_of="2026-08-27")

    assert pit_store.consensus_as_of("TGT", "2027Q2", "2026-12-31") is None
    assert out["duplicates"]["TGT"] == ["2027Q2"]
    # ...and the unambiguous quarter is still seeded.
    assert pit_store.actual_as_of("TGT", "2026Q4", "2026-12-31") == 4.27


def test_a_clean_payload_reports_no_duplicates(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises", lambda t: SURPRISES)
    assert seed_consensus.seed(["MSFT"])["duplicates"] == {}


# --- dating the actual by the announcement, not the filing ------------------
#
# A seeded actual was stamped at the 10-Q filing date, because that was the
# only date on hand. The market learned the figure at the Item 2.02 8-K, a
# median of 8 days earlier on the names measured -- 23 for JPM, 28 for TGT.
#
# It is the single reason a replay entered late. Everything downstream reads
# `actual_as_of`, so the stamp on this row decides when a study is allowed to
# act, and eight days into a drift that is largest in its first days is most
# of the effect.

ANNOUNCED = {
    "2026Q4": {"announced_date": "2026-07-14", "timing": "bmo"},
    "2026Q3": {"announced_date": "2026-04-14", "timing": "bmo"},
}


def test_a_seeded_actual_is_known_on_the_announcement(store, monkeypatch):
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    monkeypatch.setattr(seed_consensus, "_announcements",
                        lambda t, as_of=None: ANNOUNCED)

    seed_consensus.seed(["MSFT"], as_of="2026-08-27")

    # Readable the day of the release, not three weeks later at the filing.
    assert pit_store.actual_as_of("MSFT", "2026Q4", "2026-07-14") == 4.74
    assert pit_store.actual_as_of("MSFT", "2026Q4", "2026-07-13") is None


def test_the_estimate_still_predates_the_announcement(store, monkeypatch):
    """Moving the actual earlier must not drag the estimate past the print --
    an estimate read after the announcement is the answer to the question."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    monkeypatch.setattr(seed_consensus, "_announcements",
                        lambda t, as_of=None: ANNOUNCED)

    seed_consensus.seed(["MSFT"], as_of="2026-08-27")

    snap = pit_store.consensus_as_of("MSFT", "2026Q4", "2026-07-13")
    assert snap is not None and snap["eps_estimate"] == 4.3274


def test_without_an_announcement_it_falls_back_to_the_filing(store,
                                                             monkeypatch):
    """Late is the safe direction. A quarter with no 2.02 on record is still
    worth seeding; it is simply timed conservatively, and the row says so."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    monkeypatch.setattr(seed_consensus, "_announcements",
                        lambda t, as_of=None: {})

    seed_consensus.seed(["MSFT"], as_of="2026-08-27")
    assert pit_store.actual_as_of("MSFT", "2026Q4", "2026-07-29") == 4.74
    assert pit_store.actual_as_of("MSFT", "2026Q4", "2026-07-14") is None


def test_the_seeding_reports_how_many_were_announcement_dated(store,
                                                              monkeypatch):
    """The difference between the two datings is the difference between a
    replay that measures drift and one that measures its own lateness, so a
    run has to say which it produced."""
    monkeypatch.setattr(seed_consensus, "_fetch_surprises",
                        lambda t: SURPRISES)
    monkeypatch.setattr(seed_consensus, "_announcements",
                        lambda t, as_of=None: {
                            "2026Q4": ANNOUNCED["2026Q4"]})

    out = seed_consensus.seed(["MSFT"], as_of="2026-08-27")
    assert out["announcement_dated"] == 1
    assert out["filing_dated"] == 1


# --- one leaked session per name --------------------------------------------
#
# `asyncio.run` opens a loop, runs the call, and closes the loop. The aiohttp
# session created inside it survives, holding a connector bound to a loop that
# no longer exists. Nothing notices for a while; then the garbage collector
# reaches one, its destructor tries to schedule cleanup on that dead loop, and
# the run fails with "Event loop is closed".
#
# Twelve names in a row is fine, which is why this never showed up in a test.
# It surfaced on the six hundredth.

def test_the_client_is_closed_after_each_fetch(monkeypatch):
    """The session has to be closed inside the loop that made it. Nothing else
    can close it afterwards, because that loop is gone."""
    closed = []

    class Client:
        async def close(self):
            closed.append(True)

    class Server:
        def __init__(self):
            self.client = Client()

        async def get_earnings_surprises(self, ticker):
            import json

            from mcp.types import TextContent
            return [TextContent(type="text", text=json.dumps(
                {"data": {"quarters": []}}))]

    monkeypatch.setattr(seed_consensus, "_finnhub_server", Server)

    seed_consensus._fetch_surprises("AAPL")
    assert closed == [True], "the session was left open"


def test_the_client_is_closed_even_when_the_call_fails(monkeypatch):
    """A failing name must not leak either -- over a universe the failures are
    where the leak accumulates fastest."""
    closed = []

    class Client:
        async def close(self):
            closed.append(True)

    class Server:
        def __init__(self):
            self.client = Client()

        async def get_earnings_surprises(self, ticker):
            raise ConnectionError("Finnhub returned 503")

    monkeypatch.setattr(seed_consensus, "_finnhub_server", Server)

    with pytest.raises(ConnectionError):
        seed_consensus._fetch_surprises("AAPL")
    assert closed == [True]
