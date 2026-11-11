"""The pre-registered long-lag screen on the release-timed surprise (#115).

Rule, fixed before any held-out data was seen: trade a release-timed print
only when the issuer's median release-to-10-Q lag over its last four quarters
filed on or before the decision date is 7 calendar days or more. Fewer than
four quarters is a refusal. The measurement is `usual_filing_lag`; these
tests pin the screen that uses it.

No real issuer data. Synthetic fixtures only. The held-out test waits on #114.
"""
import pytest

from research import release_eps, scanner, scoring

# Same five-quarter filer as test_usual_filing_lag: enough history to pass or
# fail the four-quarter bar, and a later 10-Q that must stay invisible early.
RELEASES = {
    "2025Q1": "2025-04-24", "2025Q2": "2025-07-24", "2025Q3": "2025-10-23",
    "2025Q4": "2026-02-05", "2026Q1": "2026-04-23",
}
FILINGS = {
    "2025Q1": "2025-05-08",  # 14
    "2025Q2": "2025-07-25",  # 1
    "2025Q3": "2025-11-06",  # 14
    "2025Q4": "2026-02-26",  # 21
    "2026Q1": "2026-05-07",  # 14
}


def _release_signal(ticker="AAA", sue=3.0, known_at="2026-03-02",
                    period="2026Q1"):
    return {
        "ticker": ticker, "success": True, "error": None, "sue": sue,
        "fiscal_period": period, "known_at": known_at, "sigma": 0.10,
        "sigma_quarters": 8, "basis_changes": [], "eps": 1.0,
        "eps_year_ago": 0.9, "concept": "EarningsPerShareDiluted",
        "variant": "ts_release",
    }


def _lag_for(lag_days, reason=None, quarters=None):
    return {
        "ticker": "AAA", "as_of": "2026-03-03", "lag_days": lag_days,
        "quarters": quarters or [], "reason": reason,
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    from research import pit_store
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


@pytest.fixture
def liquid_universe(store):
    from datetime import date, timedelta

    def sessions(n, start_close=100.0, volume=5_000_000):
        out, d = [], date(2026, 3, 2)
        while len(out) < n:
            if d.weekday() < 5:
                out.append({"trade_date": d.isoformat(), "open": start_close,
                            "high": start_close * 1.01, "low": start_close * 0.99,
                            "close": start_close, "volume": volume})
            d -= timedelta(days=1)
        return list(reversed(out))

    for ticker in ("AAA", "BBB"):
        store.record_bars(ticker, sessions(300),
                          recorded_at="2026-03-02T21:00:00Z")
    store.record_bars(scanner.REGIME_TICKER, sessions(300),
                      recorded_at="2026-03-02T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "AAA", "cik": "1", "eligible": True},
        {"ticker": "BBB", "cik": "2", "eligible": True},
    ], recorded_at="2026-03-02T21:00:00Z")
    return store


def _reported(store, ticker, period, as_of, actual=1.0):
    store.record_consensus(as_of, ticker, period, eps_estimate=0.9,
                           eps_actual=actual,
                           recorded_at=f"{as_of}T21:00:00Z")


def _banded_cost(high, low=0.00002, resolved=False):
    return lambda t, as_of, dollars: {
        "cost": high, "cost_floor": low, "reason": None, "spread": high / 2,
        "resolved": resolved, "resolution": "measured" if resolved else "bound",
        "tick_floor": low, "impact_cost": 0.0,
    }


# --- the boundary -----------------------------------------------------------

def test_a_median_lag_of_seven_days_passes(monkeypatch):
    """Seven calendar days is the floor of the pre-registered rule, inclusive."""
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(release_eps, "usual_filing_lag_for",
                        lambda t, as_of=None, **kw: _lag_for(7.0))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: _release_signal(t))

    out = scanner._signal_for("AAA", "2026-03-03")

    assert out["success"] is True
    assert out["sue"] == 3.0
    assert out["variant"] == "ts_release_long_lag"


def test_a_median_lag_of_six_days_is_refused(monkeypatch):
    """One day under the floor is a refusal, not a soft miss."""
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(release_eps, "usual_filing_lag_for",
                        lambda t, as_of=None, **kw: _lag_for(6.0))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: pytest.fail(
                            "the release signal must not run under the floor"))

    out = scanner._signal_for("AAA", "2026-03-03")

    assert out["success"] is False
    assert out["sue"] is None
    assert "6" in (out["error"] or "")
    assert str(scanner.LONG_LAG_DAYS) in (out["error"] or "")
    assert out["variant"] == "ts_release_long_lag"


# --- fewer than four quarters -----------------------------------------------

def test_fewer_than_four_quarters_is_refused_with_the_count(monkeypatch):
    """The measurement refuses short history; the screen must pass that on."""
    reason = ("3 quarter(s) have both a release and a filing on or before "
              "2025-12-31; 4 are needed")
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(release_eps, "usual_filing_lag_for",
                        lambda t, as_of=None, **kw: _lag_for(
                            None, reason=reason))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: pytest.fail(
                            "a short lag history must not call the release "
                            "signal"))

    out = scanner._signal_for("AAA", "2025-12-31")

    assert out["success"] is False
    assert "3" in out["error"] and "4" in out["error"]


# --- as-of: a later 10-Q must not change the decision ------------------------

def test_a_filing_after_the_decision_date_does_not_change_the_screen(
        monkeypatch):
    """On 2026-05-01 the 2026Q1 10-Q is not filed. Its 14-day lag must not
    enter the median. The four earlier quarters median to 14, which passes."""
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(
        release_eps, "usual_filing_lag_for",
        lambda t, as_of=None, quarters=release_eps.LAG_QUARTERS, **kw:
            {"ticker": t, "as_of": as_of,
             **release_eps.usual_filing_lag(
                 RELEASES, FILINGS, as_of=as_of, quarters=quarters)})
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: _release_signal(
                            t, known_at="2026-04-23", period="2026Q1"))

    # Before the 2026Q1 10-Q: four earlier quarters, median 14 → pass.
    early = scanner._signal_for("AAA", "2026-05-01")
    assert early["success"] is True

    # Same inputs, decision after that 10-Q: still four most recent, still 14.
    late = scanner._signal_for("AAA", "2026-06-30")
    assert late["success"] is True

    # And if we drop to a date with only three filed quarters, refuse.
    short = scanner._signal_for("AAA", "2025-12-31")
    assert short["success"] is False
    assert "3" in short["error"] and "4" in short["error"]


def test_the_decision_date_is_what_the_lag_helper_sees(monkeypatch):
    """The screen must ask for the lag as of the scan date, not today."""
    seen = {}
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(
        release_eps, "usual_filing_lag_for",
        lambda t, as_of=None, **kw: seen.update(as_of=as_of) or _lag_for(14.0))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: _release_signal(t))

    scanner._signal_for("AAA", "2026-05-01")

    assert seen["as_of"] == "2026-05-01"


# --- the refusal reaches the scan's rejected rows ---------------------------

def test_a_short_lag_refusal_reaches_the_scan_rejected_rows(
        liquid_universe, monkeypatch):
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(release_eps, "usual_filing_lag_for",
                        lambda t, as_of=None, **kw: _lag_for(6.0))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: pytest.fail(
                            "under-floor lag must refuse before the release "
                            "signal"))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    out = scanner.scan(as_of="2026-03-03")

    assert out["candidates"] == []
    rejected = [r for r in out["rejected"] if r["ticker"] == "AAA"]
    assert rejected, out["rejected"]
    assert "6" in rejected[0]["reason"]
    assert str(scanner.LONG_LAG_DAYS) in rejected[0]["reason"]


def test_an_unknown_lag_is_a_refusal_not_a_pass(monkeypatch):
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(release_eps, "usual_filing_lag_for",
                        lambda t, as_of=None, **kw: _lag_for(
                            None, reason="SEC_EMAIL is not set"))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: pytest.fail(
                            "an unknown lag must not call the release signal"))

    out = scanner._signal_for("AAA", "2026-03-03")

    assert out["success"] is False
    assert "SEC_EMAIL" in out["error"]


# --- constants and defaults -------------------------------------------------

def test_the_threshold_is_the_pre_registered_seven_days():
    assert scanner.LONG_LAG_DAYS == 7


def test_the_screen_reuses_the_measurement_quarter_count():
    """Do not invent a second N. The measurement's LAG_QUARTERS is the rule."""
    assert release_eps.LAG_QUARTERS == 4


def test_the_production_default_stays_time_series():
    assert scanner.SIGNAL_VARIANT == "ts"


def test_the_scorer_prices_the_long_lag_variant_as_a_sigma():
    assert scoring.UNIT_OF["ts_release_long_lag"] == "sigma"


# --- replay accepts the variant the same way ts_release is run --------------

def test_replay_accepts_the_long_lag_variant(store, monkeypatch):
    """`signal_for=None` lets the scanner's dispatch run, which is how
    `ts_release` and `cs` are replayed. The new variant must be in that
    dispatch so the #114 held-out run only needs data."""
    from research import replay

    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release_long_lag")
    monkeypatch.setattr(release_eps, "usual_filing_lag_for",
                        lambda t, as_of=None, **kw: _lag_for(14.0))
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: _release_signal(
                            t, known_at="2026-01-05"))
    # No names, no dates: exercise the wiring, not a market. A ValueError on
    # an unknown variant would surface here before any date loop work.
    out = scanner._signal_for("AAA", "2026-03-03")
    assert out["variant"] == "ts_release_long_lag"
    assert out["success"] is True

    # And the replay runner must accept signal_for=None without rejecting the
    # variant name up front.
    monkeypatch.setattr(replay, "_refresh_universe_for",
                        lambda tickers, as_of: 0)
    monkeypatch.setattr(scanner, "scan", lambda **kw: {
        "candidates": [], "rejected": [], "undetermined": []})
    result = replay.run(["2026-03-03"], tickers=["AAA"], signal_for=None,
                        comparisons=6)
    assert result["comparisons"] == 6
    assert result["orders"] == 0
