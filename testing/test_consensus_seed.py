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

    snap = pit_store.consensus_as_of("MSFT", "2026Q4", as_of="2026-12-31")
    assert snap["eps_estimate"] == 9.99
    assert snap["source"] == "recorded"


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
