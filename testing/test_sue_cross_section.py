"""The analyst surprise, standardised across names instead of across time.

`sue_af` divides a surprise by that company's own dispersion over eight
quarters, and eight quarters of recorded consensus is nine months away at best.
Seeding buys four; the recorder has to supply the rest one day at a time.

Cross-sectional standardisation does not need any of that. It scales the
surprise by price -- which makes it comparable between a $600 stock and a $6
one -- and ranks it against the other companies that reported in the same
window. It needs one quarter of data, not eight, and that quarter already
exists: 52 vendor actual-and-estimate pairs were recorded on a single day.

Both forms appear in the drift literature. This one is not a substitute for the
other and does not pretend to be: it measures where a surprise sits among its
peers, where sue_af measures where it sits against the company's own history. A
company that always beats by a penny looks unremarkable to one and unremarkable
to the other for opposite reasons.

The failure modes are its own:

  A cohort of three is not a distribution, and a percentile against it is a
  number with no information in it.

  The cohort must be names that had already reported. Ranking against companies
  that report next week is lookahead wearing a peer group's clothes.

  Scaling by the estimate rather than by price divides by something that
  approaches zero, and the names where it does are exactly the ones whose
  surprise then looks infinite.
"""
import pytest

from research import pit_store, sue_cs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _reported(store, ticker, period, day, estimate, actual, price=100.0):
    store.record_consensus(day, ticker, period, eps_estimate=estimate,
                           eps_actual=actual, recorded_at=f"{day}T21:00:00Z")
    store.record_bars(ticker, [{"trade_date": day, "open": price,
                                "high": price, "low": price, "close": price,
                                "volume": 1_000_000}],
                      recorded_at=f"{day}T21:00:00Z")


def _cohort(store, day="2026-03-02", n=12):
    """A spread of surprises, so a percentile means something."""
    for i in range(n):
        _reported(store, f"N{i:02d}", "2026Q1", day,
                  estimate=1.00, actual=1.00 + (i - n / 2) * 0.02)


# --- the measurement --------------------------------------------------------

def test_a_bigger_surprise_ranks_higher(store):
    _cohort(store)
    _reported(store, "BIG", "2026Q1", "2026-03-02", estimate=1.00, actual=1.30)
    _reported(store, "SMALL", "2026Q1", "2026-03-02", estimate=1.00, actual=1.01)

    big = sue_cs.surprise_rank("BIG", as_of="2026-03-03")
    small = sue_cs.surprise_rank("SMALL", as_of="2026-03-03")

    assert big["success"] and small["success"]
    assert big["percentile"] > small["percentile"]
    assert big["z"] > small["z"]


def test_the_surprise_is_scaled_by_price_not_by_the_estimate(store):
    """Two identical percentage beats on very different share prices are not
    the same trade, and dividing by an estimate that can approach zero makes
    the smallest companies look like the biggest surprises."""
    _cohort(store)
    _reported(store, "CHEAP", "2026Q1", "2026-03-02", estimate=1.00,
              actual=1.10, price=10.0)
    _reported(store, "DEAR", "2026Q1", "2026-03-02", estimate=1.00,
              actual=1.10, price=1000.0)

    cheap = sue_cs.surprise_rank("CHEAP", as_of="2026-03-03")
    dear = sue_cs.surprise_rank("DEAR", as_of="2026-03-03")

    assert cheap["scaled_surprise"] > dear["scaled_surprise"] * 10
    assert cheap["percentile"] > dear["percentile"]


def test_a_near_zero_estimate_does_not_explode(store):
    _cohort(store)
    _reported(store, "TINY", "2026Q1", "2026-03-02", estimate=0.001,
              actual=0.02, price=50.0)

    out = sue_cs.surprise_rank("TINY", as_of="2026-03-03")
    assert out["success"]
    assert abs(out["scaled_surprise"]) < 1.0


# --- the cohort -------------------------------------------------------------

def test_a_cohort_too_small_is_refused(store):
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)
    _reported(store, "BBB", "2026Q1", "2026-03-02", estimate=1.0, actual=1.1)

    out = sue_cs.surprise_rank("AAA", as_of="2026-03-03")
    assert out["success"] is False
    assert "cohort" in out["error"].lower()


def test_the_cohort_is_only_names_that_had_already_reported(store):
    """Ranking against companies that report next week is lookahead wearing a
    peer group's clothes."""
    _cohort(store)
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)
    # A much bigger beat, recorded a week later.
    _reported(store, "LATER", "2026Q1", "2026-03-09", estimate=1.0, actual=9.0)

    out = sue_cs.surprise_rank("AAA", as_of="2026-03-03")
    assert "LATER" not in out["cohort_tickers"]


def test_a_stale_print_is_not_in_the_cohort(store):
    _cohort(store)
    _reported(store, "OLD", "2025Q3", "2025-09-01", estimate=1.0, actual=1.5)
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)

    out = sue_cs.surprise_rank("AAA", as_of="2026-03-03")
    assert "OLD" not in out["cohort_tickers"]


def test_a_name_that_has_not_reported_is_refused(store):
    _cohort(store)
    out = sue_cs.surprise_rank("NOPE", as_of="2026-03-03")
    assert out["success"] is False
    assert out["z"] is None


# --- provenance -------------------------------------------------------------

def test_both_legs_come_from_the_vendor(store):
    """The whole reason this is possible without eight quarters of history:
    the estimate and the actual are recorded side by side on one basis, so the
    subtraction is a surprise rather than a definitional gap."""
    _cohort(store)
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.00, actual=1.20)

    out = sue_cs.surprise_rank("AAA", as_of="2026-03-03")
    assert out["estimate"] == 1.00
    assert out["actual"] == 1.20
    assert out["surprise"] == pytest.approx(0.20)


# --- one bad quarter should not rescale everyone else -----------------------
#
# Live on 26 recorded names, QFIN missed by 2,964bp -- estimate 5.998, actual
# 3.23 -- and its z came out at -4.82. That one number roughly doubles the
# standard deviation of the cohort, so the largest beat in the set scored +0.62
# where it should read as a clear outlier. The percentile was unaffected,
# because a rank does not care how far away the tail is.
#
# Earnings surprises are fat-tailed by nature, so this is the normal case and
# not an accident of one quarter. A robust z, taken from the median and the
# median absolute deviation, is not moved by it.

def test_a_single_outlier_does_not_compress_everyone_elses_z(store):
    _cohort(store, n=20)
    _reported(store, "GOOD", "2026Q1", "2026-03-02", estimate=1.00, actual=1.25)

    before = sue_cs.surprise_rank("GOOD", as_of="2026-03-03")

    # One catastrophic miss joins the cohort.
    _reported(store, "AWFUL", "2026Q1", "2026-03-02", estimate=1.00,
              actual=-20.0)
    after = sue_cs.surprise_rank("GOOD", as_of="2026-03-03")

    assert after["z"] < before["z"] * 0.6, (
        "the plain z should be visibly compressed; if not, the fixture is not "
        "exercising the problem")
    assert after["robust_z"] == pytest.approx(before["robust_z"], rel=0.25), (
        f"the robust z moved from {before['robust_z']:.2f} to "
        f"{after['robust_z']:.2f} because of one other company's quarter")


def test_the_robust_z_still_orders_names_the_same_way(store):
    _cohort(store, n=16)
    _reported(store, "BIG", "2026Q1", "2026-03-02", estimate=1.00, actual=1.40)
    _reported(store, "MID", "2026Q1", "2026-03-02", estimate=1.00, actual=1.10)

    big = sue_cs.surprise_rank("BIG", as_of="2026-03-03")
    mid = sue_cs.surprise_rank("MID", as_of="2026-03-03")
    assert big["robust_z"] > mid["robust_z"]


def test_a_cohort_with_no_spread_refuses_the_robust_z_too(store):
    """Every name beating by exactly the same amount leaves nothing to rank."""
    for i in range(10):
        _reported(store, f"S{i}", "2026Q1", "2026-03-02", estimate=1.0,
                  actual=1.1)

    out = sue_cs.surprise_rank("S0", as_of="2026-03-03")
    assert out["success"] is False
    assert out["robust_z"] is None


def test_the_module_says_which_statistic_to_rank_on(store):
    """Neither z survives contact with the real distribution. On 26 recorded
    names the plain z put the largest beat at +0.62, compressed by one
    catastrophic miss; the robust z put it at +21.78 and that same miss at
    -260. Both are arithmetic, neither is a sigma, and the difference between
    them is the point -- a reader handed one number would take it for a
    standard deviation. The rank is the one that behaves."""
    _cohort(store, n=16)
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)

    out = sue_cs.surprise_rank("AAA", as_of="2026-03-03")
    assert out["rank_on"] == "percentile"
    assert 0.0 <= out["percentile"] <= 1.0


def test_a_rank_carries_the_date_the_print_became_known(store):
    """The scanner places every signal in time before it does anything else,
    and a signal with no date is rejected as unplaceable. sue_cs returned
    none: every cross-sectional candidate would have been refused in a real
    scan while the scanner's own tests passed, because those stubbed a date
    the module never produced."""
    _cohort(store)
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)

    out = sue_cs.surprise_rank("AAA", as_of="2026-03-03")
    assert out["known_at"] == "2026-03-02"


def test_the_date_is_when_the_actual_landed_not_when_it_was_read(store):
    _cohort(store, day="2026-03-02")
    _reported(store, "LATE", "2026Q1", "2026-03-10", estimate=1.0, actual=1.4)

    out = sue_cs.surprise_rank("LATE", as_of="2026-03-20")
    assert out["known_at"] == "2026-03-10"


def test_the_real_output_satisfies_the_scanner_gate(store):
    """A contract test between the two, because the bug above was a stub that
    had a field the module did not. Anything the scanner requires of a signal
    has to be present on the thing the module actually returns."""
    from research import scanner

    _cohort(store, n=20)
    _reported(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.9)

    real = {**sue_cs.surprise_rank("AAA", as_of="2026-03-03"), "variant": "cs"}
    assert real["success"] is True, real["error"]
    assert scanner._signal_problem(real, "2026-03-03") is None, (
        scanner._signal_problem(real, "2026-03-03"))
