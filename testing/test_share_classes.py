"""One issuer's earnings print is one trade, whatever it lists under.

A live scan against the real registrant list put six Morgan Stanley preferred
share classes into a twenty-name book -- MS-PA, MS-PF, MS-PI, MS-PO, MS-PP and
MS-PQ, all CIK 895421, all carrying the same SUE of 3.95 off the same 2026Q2
filing. Thirty percent of gross on one bank's quarter, in a book whose
`MAX_NAMES` exists to spread it across twenty independent bets. The twenty rows
were fifteen distinct issuers.

Two separate holes let that through and both are covered here.

The screen admitted instruments that are not common equity. A preferred pays a
fixed dividend and does not participate in earnings growth; post-earnings drift
is a phenomenon of the common. Eighty-three of 1,565 eligible names matched a
preferred pattern, and the screen had exactly three filters -- history, dollar
volume, price -- none of them about what the instrument is.

And the de-duplication was keyed on the ticker. `already_acted` correctly stops
one ticker trading twice on one print and says nothing about six tickers on one
issuer's print. That bites ordinary multi-class common too: GOOG and GOOGL are
one earnings surprise and would take two slots.

The second is the one that generalises. The real feed also carries fourteen
ProShares ETFs under a single sponsor CIK and a utility's baby bonds beside its
common, and no ticker-suffix rule catches those.
"""
from datetime import date, timedelta

import pytest

from research import daily_job, pit_store, scanner, spread


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


# --- what the screen lets through -------------------------------------------

@pytest.mark.parametrize("ticker,instrument", [
    # Every one of these suffixes appears in the real SEC feed, at the
    # frequency noted from a screen of 10,391 registrants.
    ("MS-PA", "preferred"),     # -PA is the commonest of all, 74 names
    ("MS-P", "preferred"),      # bare -P, 5 names
    ("PCG-PX", "preferred"),    # runs to -PZ
    ("SPCE-WT", "warrant"),     # 62 names
    ("ABC-WTA", "warrant"),
    ("XYZ-RW", "warrant"),
    ("DWAC-UN", "unit"),        # 50 names
    ("ABC-RI", "right"),        # 18 names
])
def test_an_instrument_that_is_not_common_equity_is_excluded(ticker,
                                                             instrument):
    problem = daily_job._share_class_problem(ticker)

    assert problem is not None, f"{ticker} was admitted as common equity"
    assert instrument in problem


@pytest.mark.parametrize("ticker", [
    # Class suffixes on genuine common. These carry the earnings and must
    # survive: excluding BRK-B would drop Berkshire from the universe.
    "BRK-B", "BF-B", "LEN-B", "HEI-A", "MOG-A", "CWEN-A", "PBR-A",
    # And the ordinary case.
    "MSFT", "AAPL", "GOOGL",
])
def test_common_equity_is_kept(ticker):
    assert daily_job._share_class_problem(ticker) is None


def test_the_reason_says_it_is_reading_the_ticker_not_a_data_field():
    """It is a heuristic on a string. The row it writes has to say so, or a
    reader six months from now will take it for a fact the filing carried."""
    problem = daily_job._share_class_problem("MS-PA")

    assert "ticker" in problem.lower() or "suffix" in problem.lower()


def test_the_screen_refuses_a_preferred_before_it_looks_at_any_bars(store):
    """Cheapest first, and it is a property of the ticker rather than of the
    tape: a preferred with a decade of history is still a preferred."""
    out = daily_job._screen("MS-PA", "2026-03-03")

    assert out["eligible"] is False
    assert "preferred" in out["exclusion_reason"]


def test_a_common_share_still_gets_the_liquidity_screen(store):
    """The new gate must not shadow the old ones."""
    out = daily_job._screen("MSFT", "2026-03-03")

    assert out["eligible"] is False
    assert "insufficient history" in out["exclusion_reason"]


# --- one issuer is one trade ------------------------------------------------

def _days(n, start="2025-06-02"):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _days(320)
AS_OF = DAYS[299]


def _series(store, ticker, price=100.0):
    store.record_bars(ticker, [
        {"trade_date": d, "open": price, "high": price * 1.01,
         "low": price * 0.99, "close": price, "volume": 2_000_000}
        for d in DAYS[:300]], recorded_at=f"{AS_OF}T21:00:00Z")


@pytest.fixture
def two_classes(store):
    """One issuer, two listed tickers -- the shape that produced the bug."""
    for ticker in ("BANK", "BANK-PA", "OTHER"):
        _series(store, ticker)
    _series(store, spread.REFERENCE_TICKER, price=50.0)
    store.record_universe(AS_OF, [
        {"ticker": "BANK", "cik": "895421", "eligible": True},
        {"ticker": "BANK-PA", "cik": "895421", "eligible": True},
        {"ticker": "OTHER", "cik": "111111", "eligible": True},
    ], recorded_at=f"{AS_OF}T21:00:00Z")
    return AS_OF


def _signal(sue_by_ticker):
    def go(ticker, as_of):
        return {"ticker": ticker, "success": True, "error": None,
                "sue": sue_by_ticker[ticker], "fiscal_period": "2026Q2",
                "known_at": as_of, "sigma_quarters": 8, "sigma_periods": [],
                "basis_changes": [], "variant": "ts"}
    return go


def _cost(bps_by_ticker):
    return lambda t, a, d: {"cost": bps_by_ticker[t] / 1e4,
                            "cost_floor": 0.00002, "reason": None,
                            "spread": 0.0001, "resolved": True,
                            "resolution": "measured"}


def test_two_share_classes_of_one_issuer_take_one_slot(two_classes,
                                                       monkeypatch):
    monkeypatch.setattr(scanner, "_signal_for",
                        _signal({"BANK": 3.0, "BANK-PA": 3.0, "OTHER": 2.0}))
    monkeypatch.setattr(scanner, "_cost_for",
                        _cost({"BANK": 5.0, "BANK-PA": 5.0, "OTHER": 5.0}))

    out = scanner.scan(as_of=two_classes)
    tickers = [c["ticker"] for c in out["candidates"]]

    assert sorted(tickers) == ["BANK", "OTHER"], (
        f"one issuer's print took more than one slot: {tickers}")


def test_the_better_priced_class_is_the_one_kept(two_classes, monkeypatch):
    """Collapsed after ranking, not before. Taking whichever came first in
    ticker order would hand the book the worse execution."""
    monkeypatch.setattr(scanner, "_signal_for",
                        _signal({"BANK": 3.0, "BANK-PA": 3.0, "OTHER": 2.0}))
    # The preferred is cheaper to cross here, so it wins on net edge.
    monkeypatch.setattr(scanner, "_cost_for",
                        _cost({"BANK": 40.0, "BANK-PA": 5.0, "OTHER": 5.0}))

    out = scanner.scan(as_of=two_classes)

    assert [c["ticker"] for c in out["candidates"] if c["ticker"].startswith(
        "BANK")] == ["BANK-PA"]


def test_the_sibling_is_rejected_by_name_not_silently_dropped(two_classes,
                                                              monkeypatch):
    monkeypatch.setattr(scanner, "_signal_for",
                        _signal({"BANK": 3.0, "BANK-PA": 3.0, "OTHER": 2.0}))
    monkeypatch.setattr(scanner, "_cost_for",
                        _cost({"BANK": 5.0, "BANK-PA": 9.0, "OTHER": 5.0}))

    out = scanner.scan(as_of=two_classes)
    dropped = [r for r in out["rejected"] if r["ticker"] == "BANK-PA"]

    assert dropped, "the sibling vanished from the record entirely"
    assert "BANK" in dropped[0]["reason"]
    assert "895421" in dropped[0]["reason"] or "issuer" in dropped[0]["reason"]


def test_an_issuer_already_in_the_book_is_not_bought_under_another_ticker(
        two_classes, monkeypatch):
    """Across scans, not just within one. The filed book holds the common;
    tomorrow's scan must not add the preferred on the same quarter."""
    store = pit_store
    store.record_paper_orders(
        DAYS[298],
        [{"ticker": "BANK", "side": "long", "sue": 3.0, "variant": "ts",
          "strength": 3.0, "fiscal_period": "2026Q2", "expected_edge_bps": 45.0,
          "cost_bps": 5.0, "net_edge_bps": 40.0, "target_dollars": 5000.0,
          "intended_session": AS_OF, "rank": 1}],
        recorded_at=f"{DAYS[298]}T21:00:00Z")

    monkeypatch.setattr(scanner, "_signal_for",
                        _signal({"BANK": 3.0, "BANK-PA": 3.0, "OTHER": 2.0}))
    monkeypatch.setattr(scanner, "_cost_for",
                        _cost({"BANK": 5.0, "BANK-PA": 5.0, "OTHER": 5.0}))

    out = scanner.scan(as_of=two_classes)

    assert [c["ticker"] for c in out["candidates"]] == ["OTHER"]


def test_a_ticker_with_no_recorded_cik_is_not_merged_with_anything(store,
                                                                   monkeypatch):
    """Absent is not equal. Two names whose issuer nobody recorded must not
    collapse into each other just because both are unknown."""
    for ticker in ("AAA", "BBB"):
        _series(store, ticker)
    _series(store, spread.REFERENCE_TICKER, price=50.0)
    store.record_universe(AS_OF, [
        {"ticker": "AAA", "cik": None, "eligible": True},
        {"ticker": "BBB", "cik": None, "eligible": True},
    ], recorded_at=f"{AS_OF}T21:00:00Z")

    monkeypatch.setattr(scanner, "_signal_for", _signal({"AAA": 3.0, "BBB": 2.0}))
    monkeypatch.setattr(scanner, "_cost_for", _cost({"AAA": 5.0, "BBB": 5.0}))

    out = scanner.scan(as_of=AS_OF)

    assert sorted(c["ticker"] for c in out["candidates"]) == ["AAA", "BBB"]


def test_different_quarters_of_one_issuer_are_different_prints(two_classes,
                                                               monkeypatch):
    """The rule is one print, not one issuer forever."""
    def signal(ticker, as_of):
        period = "2026Q2" if ticker == "BANK" else "2026Q1"
        return {"ticker": ticker, "success": True, "error": None, "sue": 3.0,
                "fiscal_period": period, "known_at": as_of,
                "sigma_quarters": 8, "sigma_periods": [], "basis_changes": [],
                "variant": "ts"}

    monkeypatch.setattr(scanner, "_signal_for", signal)
    monkeypatch.setattr(scanner, "_cost_for",
                        _cost({"BANK": 5.0, "BANK-PA": 5.0, "OTHER": 5.0}))

    out = scanner.scan(as_of=two_classes)

    assert "BANK" in [c["ticker"] for c in out["candidates"]]


# --- one line per issuer, at the screen ------------------------------------
#
# The suffix rule cannot see everything. AGNC Investment's five preferred
# series list as AGNCL, AGNCM, AGNCN, AGNCO and AGNCP -- no hyphen, a trailing
# letter -- and all five passed the screen, each wearing the common's EPS
# surprise. The scanner's collapse protected the book; nothing protected the
# bar fetch, the signal build or the watcher's sweep, which paid a request per
# line. So the universe itself keeps one line per CIK, the most liquid, and
# the scanner's collapse becomes a second line of defence. See issue #91.

def _bars(store, ticker, volume):
    store.record_bars(ticker, [
        {"trade_date": d, "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": volume} for d in DAYS[:300]],
        recorded_at=f"{AS_OF}T21:00:00Z")


def _registrants(monkeypatch, rows):
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: [
        {"ticker": t, "cik": cik, "name": t} for t, cik in rows])


def test_the_universe_keeps_one_line_per_issuer_the_most_liquid(store,
                                                               monkeypatch):
    """BRK-A is listed first and is the shorter-looking symbol; BRK-B is the
    line that trades. Whichever came first, or whichever is shortest, would
    both keep the wrong one."""
    _bars(store, "BRK-A", volume=40_000)     # $4M a day: clears the floor
    _bars(store, "BRK-B", volume=4_000_000)
    _bars(store, "OTHER", volume=4_000_000)
    _registrants(monkeypatch, [("BRK-A", "1067983"), ("BRK-B", "1067983"),
                               ("OTHER", "111111")])

    out = daily_job.refresh_universe(as_of=AS_OF)
    members = {m["ticker"]: m for m in pit_store.universe_as_of(AS_OF)}

    assert out["collapsed"] == 1
    assert out["eligible"] == 2
    assert members["BRK-B"]["eligible"] is True
    assert members["OTHER"]["eligible"] is True
    assert members["BRK-A"]["eligible"] is False
    assert "BRK-B" in members["BRK-A"]["exclusion_reason"]
    assert "1067983" in members["BRK-A"]["exclusion_reason"]


def test_five_hyphenless_preferreds_fall_to_the_common():
    lines = [{"ticker": t, "cik": "1423689", "eligible": True,
              "exclusion_reason": None, "median_dollar_volume": mdv}
             for t, mdv in (("AGNCL", 2e6), ("AGNC", 3e8), ("AGNCM", 2e6),
                            ("AGNCN", 3e6), ("AGNCO", 2e6), ("AGNCP", 1e6))]

    dropped = daily_job._one_line_per_issuer(lines)
    kept = [e["ticker"] for e in lines if e["eligible"]]

    assert dropped == 5
    assert kept == ["AGNC"]
    for e in lines:
        if e["ticker"] != "AGNC":
            assert "AGNC" in e["exclusion_reason"]


def test_lines_with_no_cik_are_not_collapsed_at_the_screen():
    """Absent is not equal, here as in the scanner."""
    lines = [{"ticker": t, "cik": cik, "eligible": True,
              "exclusion_reason": None, "median_dollar_volume": 1e7}
             for t, cik in (("AAA", ""), ("BBB", ""), ("CCC", None),
                            ("DDD", None))]

    assert daily_job._one_line_per_issuer(lines) == 0
    assert all(e["eligible"] for e in lines)


def test_a_sibling_the_screen_already_rejected_is_left_with_its_own_reason():
    """Only eligible lines compete. A thin sibling keeps the reason the
    liquidity screen gave it, and does not knock the common out either."""
    lines = [
        {"ticker": "BANK", "cik": "895421", "eligible": True,
         "exclusion_reason": None, "median_dollar_volume": 5e8},
        {"ticker": "BANKL", "cik": "895421", "eligible": False,
         "exclusion_reason": "median dollar volume 40,000 below the floor"},
    ]

    assert daily_job._one_line_per_issuer(lines) == 0
    assert lines[0]["eligible"] is True
    assert lines[1]["exclusion_reason"].startswith("median dollar volume")
