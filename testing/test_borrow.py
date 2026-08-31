"""What it costs to be short, and what happens when nobody knows.

`round_trip_cost` prices a crossing: half a spread each way plus impact. Being
short is not a crossing, it is a position held open, and the stock loan bills
for every calendar day of it. At the 40bp the cross-sectional variant claims
over twenty sessions, a 3% borrow is 23bp -- sixty percent of the edge -- and a
10% borrow is more than the edge itself.

The tests here are mostly about the refusal. A borrow rate is not free data,
and the failure that matters is not an inaccurate charge: it is a zero charge
that makes the least borrowable name in the universe look like the cheapest
short on the book.
"""
from datetime import date, timedelta

import pytest

from research import borrow, pit_store, spread


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _weekdays(n, start="2025-06-02"):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _weekdays(400)
AS_OF = DAYS[299]


@pytest.fixture
def calendar(store):
    """A session calendar, which is the only thing that says how long a
    twenty-session hold is in the days a lender charges for."""
    store.record_bars(spread.REFERENCE_TICKER, [
        {"trade_date": d, "open": 50.0, "high": 50.0, "low": 50.0,
         "close": 50.0, "volume": 1_000_000} for d in DAYS[:300]],
        recorded_at=f"{AS_OF}T21:00:00Z")
    return AS_OF


# --- the long side ----------------------------------------------------------

def test_a_long_carries_no_borrow(calendar):
    out = borrow.carry_cost("ANY", calendar, side="long")

    assert out["cost"] == 0.0
    assert out["reason"] is None
    assert out["rate_source"] == "not_short"


# --- the refusal ------------------------------------------------------------

def test_a_short_with_no_rate_is_refused_rather_than_charged_zero(calendar):
    out = borrow.carry_cost("HTB", calendar, side="short")

    assert out["cost"] is None
    assert out["annual_rate"] is None
    assert out["rate_source"] is None
    assert "HTB" in out["reason"]
    assert "borrow" in out["reason"].lower()


def test_a_short_with_no_session_calendar_is_refused_not_charged_zero(store):
    """No calendar means nobody knows how many days the hold covers. A cost of
    zero there is the same lie by a different route."""
    store.record_borrow_rates(AS_OF, [{"ticker": "HTB", "annual_rate": 0.03}],
                              recorded_at=f"{AS_OF}T21:00:00Z")

    out = borrow.carry_cost("HTB", AS_OF, side="short")

    assert out["cost"] is None
    assert spread.REFERENCE_TICKER in out["reason"]


# --- the charge -------------------------------------------------------------

def test_a_recorded_rate_is_charged_over_the_calendar_span(store, calendar):
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.03}],
                              recorded_at=f"{calendar}T21:00:00Z")

    out = borrow.carry_cost("HTB", calendar, side="short", horizon_days=20)

    # Twenty sessions of a weekday calendar is four weeks: 28 days, not 20.
    assert out["calendar_days"] == 28
    assert out["cost"] == pytest.approx(0.03 * 28 / 360)
    assert out["annual_rate"] == 0.03
    assert out["rate_source"] == "recorded"
    assert out["reason"] is None


def test_borrow_accrues_on_the_weekend_the_position_is_still_open(store,
                                                                  calendar):
    """Charging twenty sessions rather than twenty-eight days undercharges by
    forty percent, in the direction that flatters the short."""
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.03}],
                              recorded_at=f"{calendar}T21:00:00Z")

    out = borrow.carry_cost("HTB", calendar, side="short", horizon_days=20)

    assert out["calendar_days"] > 20
    assert out["cost"] > 0.03 * 20 / 360


def test_the_charge_scales_with_the_horizon(store, calendar):
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.03}],
                              recorded_at=f"{calendar}T21:00:00Z")

    short = borrow.carry_cost("HTB", calendar, side="short", horizon_days=20)
    long_hold = borrow.carry_cost("HTB", calendar, side="short",
                                  horizon_days=40)

    assert long_hold["cost"] == pytest.approx(2 * short["cost"], rel=0.05)


def test_a_ten_percent_borrow_exceeds_the_edge_the_tail_variant_claims(
        store, calendar):
    """The arithmetic the issue was filed on, pinned so it cannot drift."""
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.10}],
                              recorded_at=f"{calendar}T21:00:00Z")

    out = borrow.carry_cost("HTB", calendar, side="short", horizon_days=20)

    assert out["cost"] * 10_000 > 40.0


# --- where the rate comes from ----------------------------------------------

def test_a_declared_rate_is_used_when_nothing_is_recorded_and_says_so(
        calendar):
    out = borrow.carry_cost("HTB", calendar, side="short",
                            declared_rate=0.05)

    assert out["annual_rate"] == 0.05
    assert out["rate_source"] == "declared"
    assert out["cost"] == pytest.approx(0.05 * 28 / 360)


def test_a_recorded_rate_beats_a_declared_one(store, calendar):
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.22}],
                              recorded_at=f"{calendar}T21:00:00Z")

    out = borrow.carry_cost("HTB", calendar, side="short",
                            declared_rate=0.05)

    assert out["annual_rate"] == 0.22
    assert out["rate_source"] == "recorded"


def test_the_most_recent_rate_on_or_before_the_date_is_the_one_charged(
        store, calendar):
    older = DAYS[290]
    store.record_borrow_rates(older, [{"ticker": "HTB", "annual_rate": 0.02}],
                              recorded_at=f"{older}T21:00:00Z")
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.19}],
                              recorded_at=f"{calendar}T21:00:00Z")

    assert borrow.annual_rate("HTB", older)["annual_rate"] == 0.02
    assert borrow.annual_rate("HTB", calendar)["annual_rate"] == 0.19


# --- point in time ----------------------------------------------------------

def test_a_rate_dated_after_the_decision_is_not_visible_to_it(store, calendar):
    later = DAYS[305]
    store.record_borrow_rates(later, [{"ticker": "HTB", "annual_rate": 0.30}],
                              recorded_at=f"{later}T21:00:00Z")

    out = borrow.carry_cost("HTB", calendar, side="short")

    assert out["cost"] is None


def test_a_rate_written_after_the_decision_is_not_visible_to_it(store,
                                                                calendar):
    """Dated for the decision date but written a week later -- the lookahead
    the recorded_at column exists to stop."""
    store.record_borrow_rates(calendar,
                              [{"ticker": "HTB", "annual_rate": 0.30}],
                              recorded_at=f"{DAYS[305]}T21:00:00Z")

    out = borrow.carry_cost("HTB", calendar, side="short")

    assert out["cost"] is None


# --- what the store will accept ---------------------------------------------

def test_a_negative_rate_is_refused_at_the_door(store):
    with pytest.raises(ValueError):
        store.record_borrow_rates(AS_OF,
                                  [{"ticker": "HTB", "annual_rate": -0.01}])


def test_a_missing_rate_is_refused_at_the_door(store):
    with pytest.raises(ValueError):
        store.record_borrow_rates(AS_OF,
                                  [{"ticker": "HTB", "annual_rate": None}])


def test_a_rate_already_on_record_is_not_overwritten(store):
    store.record_borrow_rates(AS_OF, [{"ticker": "HTB", "annual_rate": 0.03}],
                              recorded_at=f"{AS_OF}T21:00:00Z")
    second = store.record_borrow_rates(
        AS_OF, [{"ticker": "HTB", "annual_rate": 0.99}],
        recorded_at=f"{AS_OF}T22:00:00Z")

    assert second == 0
    assert store.borrow_rate_as_of("HTB", AS_OF)["annual_rate"] == 0.03


def test_a_bad_side_is_a_caller_bug_not_a_free_short(calendar):
    with pytest.raises(ValueError):
        borrow.carry_cost("HTB", calendar, side="buy")


# --- getting rates into the store -------------------------------------------

# The loader stamps `recorded_at` with the wall clock, as it must -- a rate
# typed in today is not something last month's decision could have used. So
# these load for today and read as of today; the point-in-time behaviour when
# the two dates differ is pinned above.
TODAY = date.today().isoformat()


def _csv(tmp_path, text):
    path = tmp_path / "rates.csv"
    path.write_text(text)
    return str(path)


def test_a_csv_of_rates_is_recorded_for_a_date(store, tmp_path, capsys):
    path = _csv(tmp_path, "ticker,annual_rate\nHTB,0.035\nGC,0.003\n")

    rc = borrow.main(["--as-of", TODAY, "--from-csv", path,
                      "--units", "fraction", "--source", "broker-file"])

    assert rc == 0
    row = store.borrow_rate_as_of("HTB", TODAY)
    assert row["annual_rate"] == pytest.approx(0.035)
    assert row["source"] == "broker-file"


def test_percent_units_are_converted_and_fractions_are_not(store, tmp_path):
    borrow.main(["--as-of", TODAY, "--units", "percent",
                 "--from-csv", _csv(tmp_path, "ticker,annual_rate\nHTB,3.5\n")])

    assert store.borrow_rate_as_of("HTB", TODAY)["annual_rate"] == \
        pytest.approx(0.035)


def test_the_units_must_be_said_because_three_and_0_03_differ_by_a_hundred(
        store, tmp_path):
    """The single most likely way to be wrong here by two orders of magnitude,
    and no default can be safe when both readings are plausible."""
    with pytest.raises(SystemExit):
        borrow.main(["--as-of", AS_OF,
                     "--from-csv", _csv(tmp_path,
                                        "ticker,annual_rate\nHTB,3.5\n")])


def test_a_csv_missing_the_rate_column_is_refused(store, tmp_path, capsys):
    rc = borrow.main(["--as-of", AS_OF, "--units", "fraction",
                      "--from-csv", _csv(tmp_path, "ticker,fee\nHTB,0.035\n")])

    assert rc == 1
    assert store.borrow_rate_as_of("HTB", AS_OF) is None


def test_one_bad_row_records_nothing(store, tmp_path):
    """Half a file is worse than none: the names that landed get priced and the
    ones that did not get refused, and nothing says which was which."""
    rc = borrow.main(["--as-of", AS_OF, "--units", "fraction",
                      "--from-csv", _csv(
                          tmp_path,
                          "ticker,annual_rate\nHTB,0.035\nBAD,not-a-number\n")])

    assert rc == 1
    assert store.borrow_rate_as_of("HTB", AS_OF) is None


def test_the_recorded_range_is_printed_so_a_unit_error_is_visible(
        store, tmp_path, capsys):
    borrow.main(["--as-of", AS_OF, "--units", "fraction",
                 "--from-csv", _csv(
                     tmp_path, "ticker,annual_rate\nA,0.003\nB,0.35\n")])

    out = capsys.readouterr().out
    assert "0.003" in out and "0.35" in out
