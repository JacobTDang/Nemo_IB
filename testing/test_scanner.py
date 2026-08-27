"""The step where a signal becomes a position, and the ways that step lies.

Everything upstream of here measures something. This module decides, and a
decision has failure modes that measurement does not:

  A signal that does not clear its own trading cost. PEAD is a few tens of
  basis points over weeks; a 60bp round trip in a thin name eats it entirely.
  A scanner that ranks on the signal alone finds its best ideas in exactly the
  names where the cost is largest, because illiquidity is what made the
  surprise look big.

  A fill that was never available. A company reporting after the close cannot
  be bought at that close, and the close is the price sitting in the store when
  the scan runs. Using it turns the hardest part of the strategy into free
  money.

  A size nobody could trade. $100k of a name that trades $200k a day is half
  the volume, and the impact model stops meaning anything long before that.

  A missing input filled with a house average. There is no house spread here.
  A name whose cost cannot be measured is not a name with an average cost.

  A rejection nobody recorded, which is how a scanner that quietly stopped
  finding candidates looks identical to a market with nothing in it.
"""
import pytest

from research import pit_store, scanner


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _sessions(n, start_close=100.0, volume=5_000_000):
    """n weekday sessions ending 2026-03-02, flat and liquid."""
    from datetime import date, timedelta
    out, d = [], date(2026, 3, 2)
    while len(out) < n:
        if d.weekday() < 5:
            out.append({"trade_date": d.isoformat(), "open": start_close,
                        "high": start_close * 1.01, "low": start_close * 0.99,
                        "close": start_close, "volume": volume})
        d -= timedelta(days=1)
    return list(reversed(out))


@pytest.fixture
def liquid_universe(store):
    for ticker in ("AAA", "BBB"):
        store.record_bars(ticker, _sessions(300),
                          recorded_at="2026-03-02T21:00:00Z")
    store.record_bars(scanner.REGIME_TICKER, _sessions(300),
                      recorded_at="2026-03-02T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "AAA", "cik": "1", "eligible": True},
        {"ticker": "BBB", "cik": "2", "eligible": True},
    ])
    return store


def _signal(ticker, sue, known_at="2026-03-02", period="2026Q1", **over):
    out = {"ticker": ticker, "success": True, "error": None, "sue": sue,
           "fiscal_period": period, "known_at": known_at, "sigma": 0.10,
           "sigma_quarters": 8, "basis_changes": [], "eps": 1.0,
           "eps_year_ago": 0.9, "concept": "EarningsPerShareDiluted"}
    out.update(over)
    return out


def _patch_signals(monkeypatch, mapping):
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, as_of: mapping.get(t)
                        or _signal(t, None, success=False,
                                   error="no filings"))


# --- cost is not an afterthought -------------------------------------------

def test_a_signal_that_does_not_clear_its_cost_is_not_a_trade(liquid_universe,
                                                              monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.05, "reason": None, "spread": 0.04,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    assert [c["ticker"] for c in result["candidates"]] == []
    rejected = {r["ticker"]: r for r in result["rejected"]}
    assert "cost" in rejected["AAA"]["reason"].lower()


def test_ranking_is_on_edge_after_cost_not_on_the_signal(liquid_universe,
                                                         monkeypatch):
    """The whole point. BBB has the smaller surprise and the cheaper spread,
    and it is the better trade -- a scanner ranking on SUE alone puts the
    expensive one first every time, because thin names produce big surprises."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 4.0),
                                 "BBB": _signal("BBB", 3.0)})
    costs = {"AAA": 0.0030, "BBB": 0.0005}
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": costs[t], "reason": None,
                            "spread": costs[t] / 2, "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    assert [c["ticker"] for c in result["candidates"]] == ["BBB", "AAA"]
    assert result["candidates"][0]["net_edge_bps"] > \
        result["candidates"][1]["net_edge_bps"]


def test_an_unmeasurable_cost_is_a_refusal_not_an_average(liquid_universe,
                                                          monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": None, "reason": "insufficient history",
                            "spread": None, "resolved": False})

    result = scanner.scan(as_of="2026-03-03")

    assert result["candidates"] == []
    assert "insufficient history" in result["rejected"][0]["reason"]


# --- the fill has to have been available ------------------------------------

def test_the_intended_fill_is_the_next_session_not_todays_close(
        liquid_universe, monkeypatch):
    """A company that reported after Monday's close cannot be bought at it."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")
    candidate = result["candidates"][0]

    assert candidate["intended_session"] > "2026-03-03"
    assert "fill_price" not in candidate, (
        "a price was recorded for a session that has not happened")


def test_a_signal_known_after_the_scan_date_is_invisible(liquid_universe,
                                                         monkeypatch):
    """Anti-lookahead, at the last place it can still get in."""
    _patch_signals(monkeypatch,
                   {"AAA": _signal("AAA", 3.0, known_at="2026-03-10")})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    assert result["candidates"] == []
    assert "not yet known" in result["rejected"][0]["reason"].lower()


def test_a_stale_surprise_is_not_a_fresh_one(liquid_universe, monkeypatch):
    """The drift is a few weeks long. A surprise from last quarter is not a
    reason to buy today, and nothing in the signal itself says so."""
    _patch_signals(monkeypatch,
                   {"AAA": _signal("AAA", 3.0, known_at="2025-11-01")})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    assert result["candidates"] == []
    assert "stale" in result["rejected"][0]["reason"].lower()


# --- the signal has to be worth trusting ------------------------------------

def test_a_sigma_from_too_few_quarters_is_refused(liquid_universe, monkeypatch):
    _patch_signals(monkeypatch,
                   {"AAA": _signal("AAA", 3.0, sigma_quarters=3)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")
    assert result["candidates"] == []
    assert "quarters" in result["rejected"][0]["reason"].lower()


def test_a_basis_change_inside_the_window_invalidates_the_series(
        liquid_universe, monkeypatch):
    """A split mid-window makes the standard deviation a measure of the
    redenomination rather than of how much this company surprises."""
    _patch_signals(monkeypatch, {
        "AAA": _signal("AAA", 3.0,
                       sigma_periods=["2026Q1", "2025Q4", "2025Q3", "2025Q2",
                                      "2025Q1", "2024Q4", "2024Q3", "2024Q2"],
                       basis_changes=[{"between": ["2025-07-30", "2025-10-29"],
                                       "ratio": 0.1}])})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")
    assert result["candidates"] == []
    assert "basis" in result["rejected"][0]["reason"].lower()


def test_an_old_basis_change_does_not_disqualify_a_name(liquid_universe,
                                                        monkeypatch):
    """AAPL's are from 2014 and 2020, and sue rebases the series onto one basis
    before computing anything. Rejecting on the mere presence of a historical
    split threw out AAPL, AMZN, GOOGL and NVDA in a live scan -- four of the
    most liquid names on the tape -- for events a decade outside the window
    that were already corrected for."""
    _patch_signals(monkeypatch, {
        "AAA": _signal("AAA", 3.0,
                       sigma_periods=["2026Q1", "2025Q4", "2025Q3", "2025Q2",
                                      "2025Q1", "2024Q4", "2024Q3", "2024Q2"],
                       basis_changes=[{"between": ["2014-04-24", "2014-07-23"],
                                       "ratio": 0.1428}])})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")
    assert [c["ticker"] for c in result["candidates"]] == ["AAA"]


# --- a cost nobody can measure is a band, not a number ----------------------
#
# EDGE cannot resolve a mega-cap's spread from daily bars. Live at 252
# sessions, every one of MSFT, AAPL and JPM came back resolved=False, so the
# cost model charged the 95% upper bound: 19.8bp, 39.6bp and 54.0bp against
# true spreads around a basis point. Those are not cost differences, they are
# sampling error, and ranking on drift-minus-cost ranks on it -- JPM, the
# cheapest name on the list, was rejected for being too expensive to trade.
#
# The floor is knowable: one tick against the price. So the cost is a band,
# and a name excluded by the top of that band was excluded by the measurement
# rather than by the strategy. Those two things must not look alike.


def _banded_cost(high, low=0.00002, resolved=False):
    return lambda t, as_of, dollars: {
        "cost": high, "cost_floor": low, "reason": None,
        "spread": None if not resolved else high / 2, "resolved": resolved}


def test_a_name_excluded_only_by_the_bound_is_not_a_rejection(liquid_universe,
                                                              monkeypatch):
    """45bp expected against a 20bp-to-0.4bp band: it clears comfortably if
    the spread is anywhere near its floor and fails at the bound."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0060))

    result = scanner.scan(as_of="2026-03-03")

    assert result["candidates"] == []
    assert result["undetermined"], (
        "a name the measurement could not judge was filed as a rejection")
    row = result["undetermined"][0]
    assert row["ticker"] == "AAA"
    assert row["cost_bps_high"] > row["cost_bps_low"]
    assert "unresolved" in row["reason"].lower()


def test_a_name_that_fails_even_at_the_floor_is_a_real_rejection(
        liquid_universe, monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 1.2)})
    monkeypatch.setattr(scanner, "_cost_for",
                        _banded_cost(0.0300, low=0.0100))

    result = scanner.scan(as_of="2026-03-03")

    assert result["undetermined"] == []
    assert "does not clear" in result["rejected"][0]["reason"]


def test_a_name_that_clears_the_bound_is_a_candidate(liquid_universe,
                                                     monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")

    assert [c["ticker"] for c in result["candidates"]] == ["AAA"]
    assert result["undetermined"] == []


def test_the_scan_says_how_much_of_its_cost_is_measured(liquid_universe,
                                                        monkeypatch):
    """If nothing was measured, the ranking is a ranking of floors and the
    reader has to be told."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0),
                                 "BBB": _signal("BBB", 2.5)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")

    assert result["costs_measured"] == 0
    assert result["costs_total"] == 2


def test_a_name_outside_the_universe_is_never_considered(store, monkeypatch):
    store.record_bars("XXX", _sessions(300), recorded_at="2026-03-02T21:00:00Z")
    store.record_bars(scanner.REGIME_TICKER, _sessions(300),
                      recorded_at="2026-03-02T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "XXX", "cik": "9", "eligible": False,
         "exclusion_reason": "below $500k median dollar volume"}])
    _patch_signals(monkeypatch, {"XXX": _signal("XXX", 5.0)})

    result = scanner.scan(as_of="2026-03-03")
    assert result["candidates"] == []


# --- size has to be tradeable ------------------------------------------------

def test_a_position_is_capped_by_what_the_name_trades(store, monkeypatch):
    """$100k into a name doing $200k a day is not a position, it is the day."""
    store.record_bars("THIN", _sessions(300, start_close=10.0, volume=20_000),
                      recorded_at="2026-03-02T21:00:00Z")
    store.record_bars(scanner.REGIME_TICKER, _sessions(300),
                      recorded_at="2026-03-02T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "THIN", "cik": "3", "eligible": True}])
    _patch_signals(monkeypatch, {"THIN": _signal("THIN", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    if result["candidates"]:
        c = result["candidates"][0]
        mdv = 10.0 * 20_000
        assert c["target_dollars"] <= mdv * scanner.spread.MAX_PARTICIPATION \
            * 1.0001, f"sized at {c['target_dollars']} against {mdv} a day"


# --- the regime changes the gross, not the ranking ---------------------------

def test_high_volatility_shrinks_the_book(liquid_universe, monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    monkeypatch.setattr(scanner, "_regime_scale", lambda as_of: (1.0, "calm"))
    calm = scanner.scan(as_of="2026-03-03")
    monkeypatch.setattr(scanner, "_regime_scale", lambda as_of: (0.4, "stressed"))
    stressed = scanner.scan(as_of="2026-03-03")

    assert stressed["gross_target"] < calm["gross_target"]
    assert stressed["candidates"][0]["target_dollars"] < \
        calm["candidates"][0]["target_dollars"]
    assert stressed["regime"] == "stressed"


def test_the_regime_cannot_read_a_future_session(liquid_universe):
    """It is computed from the index's own history, which makes it exactly as
    leakable as everything else here."""
    scale_early, _ = scanner._regime_scale("2026-01-15")
    assert 0 < scale_early <= 1.0


# --- what happened has to be recoverable ------------------------------------

def test_every_rejection_carries_a_reason(liquid_universe, monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 0.2)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    assert result["rejected"], "a scan with no candidates explained nothing"
    for row in result["rejected"]:
        assert row["reason"], f"{row['ticker']} rejected with no reason"


def test_a_scan_is_recorded_so_it_can_be_scored_later(liquid_universe,
                                                      monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0),
                                 "BBB": _signal("BBB", 0.1)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    scanner.record_scan(as_of="2026-03-03")

    orders = pit_store.paper_orders_as_of("2026-03-03")
    accepted = [o for o in orders if o["accepted"]]
    assert [o["ticker"] for o in accepted] == ["AAA"]
    assert accepted[0]["target_dollars"] > 0
    assert accepted[0]["intended_session"] > "2026-03-03"
    # The rejects are kept too: a scan that stops finding anything must not
    # look like a market with nothing in it.
    assert any(not o["accepted"] for o in orders)


def test_a_recorded_scan_is_invisible_to_an_earlier_date(liquid_universe,
                                                         monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    scanner.record_scan(as_of="2026-03-03")
    assert pit_store.paper_orders_as_of("2026-03-02") == []


def test_rerunning_a_scan_does_not_double_the_book(liquid_universe, monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    scanner.record_scan(as_of="2026-03-03")
    scanner.record_scan(as_of="2026-03-03")

    orders = [o for o in pit_store.paper_orders_as_of("2026-03-03")
              if o["accepted"]]
    assert len(orders) == 1


def test_the_expected_drift_is_a_stated_assumption(liquid_universe, monkeypatch):
    """It is not measured here and must not read as though it were. The store
    exists so this number can eventually be calibrated on recorded history
    rather than borrowed from a paper about a different decade."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for",
                        lambda t, as_of, dollars: {
                            "cost": 0.001, "reason": None, "spread": 0.0005,
                            "resolved": True})

    result = scanner.scan(as_of="2026-03-03")

    assert result["assumptions"]["drift_bps_per_sue"] == \
        scanner.DRIFT_BPS_PER_SUE
    assert result["assumptions"]["calibrated"] is False


# --- a surprise too big to be a surprise ------------------------------------
#
# A live scan ranked GOOGL and AMZN first on SUEs of 10.98 and 12.68, and the
# data was right: Alphabet's XBRL really does tag 9.11 for the three months to
# 2026-06-30 against 2.31 a year earlier. Nothing was misparsed.
#
# That is the problem. A ten-sigma move in GAAP EPS is not an earnings
# surprise, it is a non-operating item -- Alphabet runs equity revaluations
# through EPS -- and post-earnings drift is the market underreacting to
# OPERATING news. Every published drift coefficient is fit over roughly
# |SUE| <= 4. Multiplying 15bp by 12.68 claims 190bp of edge by extrapolating a
# linear model six standard deviations past anything it was estimated on, and
# it put the two least trustworthy signals at the top of the book.


def test_an_implausibly_large_surprise_is_refused(liquid_universe, monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 12.68)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")

    assert result["candidates"] == []
    reason = result["rejected"][0]["reason"].lower()
    assert "12.68" in reason
    assert "operating" in reason or "outside" in reason


def test_a_large_but_credible_surprise_still_trades(liquid_universe,
                                                    monkeypatch):
    """The gate is for outliers, not for good quarters."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.5)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")
    assert [c["ticker"] for c in result["candidates"]] == ["AAA"]


def test_the_edge_never_extrapolates_past_the_gate(liquid_universe,
                                                   monkeypatch):
    """Whatever the cap is, the expected edge at the cap is the largest this
    module will ever claim -- so a bad sigma cannot become a large position."""
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", scanner.MAX_ABS_SUE)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0001))

    result = scanner.scan(as_of="2026-03-03")
    ceiling = scanner.MAX_ABS_SUE * scanner.DRIFT_BPS_PER_SUE
    assert result["candidates"][0]["expected_edge_bps"] <= ceiling


# --- its own entry point ----------------------------------------------------
#
# Separate from the recorder's on purpose. Recording is not repeatable -- a day
# missed is a day gone -- while a scan is a decision over an existing record and
# can be re-run against the same day as many times as the parameters change.
# Sharing one entry point would mean re-recording to re-decide.

def test_a_scan_that_finds_nothing_still_exits_zero(liquid_universe,
                                                    monkeypatch):
    """An empty tape is not a failure, and paging someone for one teaches them
    to ignore the pager."""
    _patch_signals(monkeypatch, {})
    assert scanner.main(["--as-of", "2026-03-03"]) == 0


def test_a_scan_that_cannot_run_exits_non_zero(liquid_universe, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(scanner, "record_scan", boom)
    assert scanner.main(["--as-of", "2026-03-03"]) == 1


def test_the_entry_point_files_the_scan(liquid_universe, monkeypatch):
    _patch_signals(monkeypatch, {"AAA": _signal("AAA", 3.0)})
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    assert scanner.main(["--as-of", "2026-03-03"]) == 0
    assert pit_store.paper_orders_as_of("2026-03-03", accepted_only=True)


# --- asking EDGAR about 2,435 companies to reject 2,300 of them -------------
#
# scan() fetched a signal for every eligible name. Measured on the real
# universe that is 2,435 companyconcept requests and about 19 minutes, and
# nearly all of it is spent learning that a company last reported in May --
# which the staleness gate then throws out. This project has already earned
# real SEC 429s once.
#
# The store already knows who reported: the consensus recorder captures the
# vendor's actual within days of each print. Narrowing on that before spending
# a request is the same test applied earlier, against data already paid for.


def _reported(store, ticker, period, as_of, actual=1.0):
    store.record_consensus(as_of, ticker, period, eps_estimate=0.9,
                           eps_actual=actual,
                           recorded_at=f"{as_of}T21:00:00Z")


def test_only_names_that_recently_reported_are_asked_about(liquid_universe,
                                                           monkeypatch):
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")

    asked = []
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: asked.append(t) or _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    scanner.scan(as_of="2026-03-03")

    assert asked == ["AAA"], (
        f"asked EDGAR about {asked}; only AAA has a print on the record")


def test_a_name_whose_print_is_old_is_not_asked_about(liquid_universe,
                                                      monkeypatch):
    """The staleness gate, moved in front of the request instead of behind."""
    _reported(liquid_universe, "AAA", "2025Q3", "2025-09-01")

    asked = []
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: asked.append(t) or _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")
    assert asked == []
    assert result["candidates"] == []


def test_an_estimate_without_an_actual_is_not_a_print(liquid_universe,
                                                      monkeypatch):
    """A company on next week's calendar has a consensus recorded and has not
    reported. Treating that as a print would scan it before the news exists."""
    liquid_universe.record_consensus("2026-03-02", "AAA", "2026Q1",
                                     eps_estimate=0.9, eps_actual=None,
                                     recorded_at="2026-03-02T21:00:00Z")

    asked = []
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: asked.append(t) or _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    scanner.scan(as_of="2026-03-03")
    assert asked == []


def test_a_store_with_no_prints_recorded_says_so_rather_than_finding_nothing(
        liquid_universe, monkeypatch):
    """The recorder only accumulates forward, so a young store knows about no
    prints at all. Silently scanning nothing would look exactly like a quiet
    tape; silently scanning everything would spend 2,435 requests without
    saying why. It scans the universe and states that it had to."""
    asked = []
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: asked.append(t) or _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")

    assert set(asked) == {"AAA", "BBB"}
    assert result["narrowed_by"] is None
    assert "no prints" in result["narrowing_note"].lower() or \
        "announcement" in result["narrowing_note"].lower()


def test_the_scan_reports_how_far_it_narrowed(liquid_universe, monkeypatch):
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    result = scanner.scan(as_of="2026-03-03")

    assert result["screened"] == 2
    assert result["considered"] == 1
    assert result["narrowed_by"] == "recorded prints"


def test_a_scan_that_had_nothing_to_do_leaves_a_record_saying_why(
        liquid_universe, monkeypatch):
    """A day with no orders and a day the scan never ran look identical in the
    paper_order table, because both are empty. The run log is the only place
    the difference can live, so the reason goes there."""
    _reported(liquid_universe, "AAA", "2026Q1", "2026-01-01")

    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    scanner.record_scan(as_of="2026-03-03")

    from research import daily_job

    run = daily_job.last_run("scan")
    assert run["status"] == "ok"
    assert run["error"], "a scan with no candidates recorded no reason"
    assert "eligible" in run["error"] or "print" in run["error"]


# --- the cost the scanner charges -------------------------------------------
#
# It charged EDGE's 95% upper bound, which for a mega-cap is the estimator's
# bias rather than the name's spread: 39.7bp for AAPL, 21.7bp for MSFT, 39.1bp
# for SPY itself, whose market is one cent wide. Nothing cleared, and the
# liquidity gradient the kill criteria are supposed to test for was absent --
# every name in the universe came back inside the same 20-50bp band.

def test_the_scanner_charges_the_adaptive_basis(liquid_universe, monkeypatch):
    seen = {}

    def cost(ticker, as_of, dollars):
        from research import spread as sp
        seen["basis"] = "adaptive"
        return sp.round_trip_cost(ticker, as_of, dollars,
                                  window=sp.RESOLVING_WINDOW,
                                  basis="adaptive")

    monkeypatch.setattr(scanner, "_cost_for", cost)
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: _signal(t, 3.0))

    scanner.scan(as_of="2026-03-03")
    assert seen.get("basis") == "adaptive"


def test_the_real_cost_seam_asks_for_the_adaptive_basis():
    """Pinning the production seam rather than a stub of it."""
    import inspect
    source = inspect.getsource(scanner._cost_for)
    assert 'basis="adaptive"' in source or "basis='adaptive'" in source


def test_the_scan_reports_which_costs_were_measured(liquid_universe,
                                                    monkeypatch):
    """`resolved` asks whether EDGE's estimate differs from zero, which a
    biased estimator passes easily -- SPY comes back resolved at 41bp. Counting
    it told the reader nothing. What matters is whether the charge is this
    name's spread or the tick it was floored to."""
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: _signal(t, 3.0))
    monkeypatch.setattr(scanner, "_cost_for", lambda t, a, d: {
        "cost": 0.0002, "cost_floor": 0.00002, "reason": None,
        "spread": 0.0001, "resolved": False,
        "resolution": "at_resolution_floor"})

    result = scanner.scan(as_of="2026-03-03")

    assert result["costs_measured"] == 0
    assert result["costs_floored"] == 1
    assert result["costs_total"] == 1


# --- one print, one trade ---------------------------------------------------
#
# The signal stays fresh for 45 days, so a name that reported on Monday is a
# candidate again on Tuesday, and every session after that until the window
# closes. Nothing stopped it: the scanner has no notion of having already acted
# on a print. In production that is the same position proposed forty-five
# nights running; in a study it is one earnings event counted as forty-five
# independent trades, which is how a sample gets a t-statistic it has not
# earned.

def test_a_print_already_acted_on_is_not_proposed_again(liquid_universe,
                                                        monkeypatch):
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: _signal(t, 3.0, period="2026Q1"))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    first = scanner.record_scan(as_of="2026-03-03")
    assert [c["ticker"] for c in first["candidates"]] == ["AAA"]

    second = scanner.scan(as_of="2026-03-04")
    assert second["candidates"] == [], (
        "the same print was proposed again the next session")
    reason = next(r["reason"] for r in second["rejected"]
                  if r["ticker"] == "AAA")
    assert "already" in reason.lower()


def test_the_next_quarter_is_a_new_print(liquid_universe, monkeypatch):
    """The rule is one trade per print, not one trade per name ever."""
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: _signal(t, 3.0, period="2026Q1"))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))
    scanner.record_scan(as_of="2026-03-03")

    _reported(liquid_universe, "AAA", "2026Q2", "2026-06-02")
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: _signal(t, 3.0, period="2026Q2",
                                             known_at="2026-06-02"))
    later = scanner.scan(as_of="2026-06-03")
    assert [c["ticker"] for c in later["candidates"]] == ["AAA"]


def test_a_caller_can_supply_what_has_already_been_acted_on(liquid_universe,
                                                            monkeypatch):
    """Replay does not file into the live book, so it keeps its own set."""
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: _signal(t, 3.0, period="2026Q1"))
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))

    out = scanner.scan(as_of="2026-03-03",
                       already_acted={("AAA", "2026Q1")})
    assert out["candidates"] == []


def test_the_signal_source_can_be_supplied(liquid_universe, monkeypatch):
    """Replay precomputes every quarter in one EDGAR pass per name and needs
    the scan to use that table. Reaching in and rebinding the module's own seam
    works until something else rebinds it back; a parameter cannot be undone by
    ordering."""
    _reported(liquid_universe, "AAA", "2026Q1", "2026-03-02")
    monkeypatch.setattr(scanner, "_cost_for", _banded_cost(0.0010))
    monkeypatch.setattr(scanner, "_signal_for",
                        lambda t, a: pytest.fail("the default seam was used"))

    out = scanner.scan(as_of="2026-03-03",
                       signal_for=lambda t, a: _signal(t, 3.0))
    assert [c["ticker"] for c in out["candidates"]] == ["AAA"]
