"""The signal, and the ways a plausible number can be the wrong one.

SUE is the one anomaly in this strategy that survived replication: in the
post-2005 sample it was the only statistically significant determinant of
returns, while book-to-market, profitability and momentum all shrank at least
40% and several flipped sign. Everything else here is filtering. So the bar for
this module is not "does it return a number" -- it is "is the number it returns
the one the filings support".

Five ways it could quietly not be, each with tests below:

  **A calendar bucket instead of a fiscal identity.** AMAT's 13 August print is
  labelled 2026-09-30 by the vendor calendar. Key a series on that and the
  join returns nothing; key it on the wrong quarter and the seasonal
  comparison lands a quarter off, which measures seasonality rather than
  surprise.

  **A share-basis change read as an earnings collapse.** NVDA's diluted EPS is
  5.98 for the April 2024 quarter as originally filed and 0.60 for the same
  quarter as restated after the 10-for-1 split. Difference those against each
  other across the split and the surprise is -5.38 where it should be +0.09.
  SUE is a ratio and therefore scale-free, so the split must leave it exactly
  where it was.

  **A shorter window substituted to produce a number.** Eight trailing
  announcements is the denominator; six is the floor. Quietly falling back to
  three would divide by a standard deviation estimated from noise and print a
  large SUE for a company we know nothing about.

  **Lookahead through the filing date.** An EPS figure is not knowable before
  the filing that carries it. A series filtered only on period end lets the
  quarter that has happened but not yet been reported into a simulation
  standing before the print -- which is the whole trade.

  **A quarterly series for a filer that has none.** A foreign private issuer
  files 20-F and 6-K; 6-K exhibits carry no XBRL. There is no quarterly EPS
  anywhere for TSM, and a partial or annual-only series dressed as one is
  worse than a refusal.
"""
import statistics
from datetime import date, timedelta

import pytest

from research import pit_store, sue


# --------------------------------------------------------------- fixtures

def _plus(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def _calendar_quarter(fy: int, fq: int):
    """(start, end) for a December fiscal-year-end filer."""
    bounds = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"),
              3: ("07-01", "09-30"), 4: ("10-01", "12-31")}
    start, end = bounds[fq]
    return f"{fy}-{start}", f"{fy}-{end}"


def _eps_map(start_fy: int, values):
    """A flat list of quarterly EPS, keyed by (fiscal year, fiscal quarter)."""
    return {(start_fy + i // 4, i % 4 + 1): v for i, v in enumerate(values)}


def concept_payload(eps, quarter_lag=30, annual_lag=55, unit="USD/shares",
                    drop_comparatives=(), drop_ytd=(), split=None):
    """One filer's EPS facts, shaped exactly as data.sec.gov hands them back.

    Every 10-Q carries four durations -- the quarter, the year to date, and
    both of the prior year's equivalents -- because that is what a real one
    carries, and the prior-year duration is the only evidence in XBRL that a
    share basis has changed.

    `split` is (effective_date, ratio): every fact filed on or after that date
    is restated onto the new basis, which is what a stock split does to a
    filer's tagged EPS and nothing else in the API marks.
    """
    rows = []

    def add(start, end, val, form, fy, fp, filed, accn):
        rows.append({"start": start, "end": end, "val": val, "accn": accn,
                     "fy": fy, "fp": fp, "form": form, "filed": filed})

    years = sorted({fy for fy, _ in eps})
    for fy in years:
        for fq in (1, 2, 3):
            if (fy, fq) not in eps:
                continue
            accn = f"000-{fy}-Q{fq}"
            q_start, q_end = _calendar_quarter(fy, fq)
            filed = _plus(q_end, quarter_lag)
            year_start = _calendar_quarter(fy, 1)[0]
            add(q_start, q_end, eps[(fy, fq)], "10-Q", fy, f"Q{fq}", filed, accn)
            if fq > 1 and (fy, fq) not in drop_ytd and \
                    all((fy, i) in eps for i in range(1, fq + 1)):
                add(year_start, q_end,
                    sum(eps[(fy, i)] for i in range(1, fq + 1)),
                    "10-Q", fy, f"Q{fq}", filed, accn)
            have_prior = all((fy - 1, i) in eps for i in range(1, fq + 1))
            if have_prior and (fy, fq) not in drop_comparatives:
                p_start, p_end = _calendar_quarter(fy - 1, fq)
                add(p_start, p_end, eps[(fy - 1, fq)], "10-Q", fy, f"Q{fq}",
                    filed, accn)
                if fq > 1:
                    add(_calendar_quarter(fy - 1, 1)[0], p_end,
                        sum(eps[(fy - 1, i)] for i in range(1, fq + 1)),
                        "10-Q", fy, f"Q{fq}", filed, accn)

        if all((fy, i) in eps for i in (1, 2, 3, 4)):
            accn = f"000-{fy}-FY"
            filed = _plus(f"{fy}-12-31", annual_lag)
            add(f"{fy}-01-01", f"{fy}-12-31",
                sum(eps[(fy, i)] for i in (1, 2, 3, 4)),
                "10-K", fy, "FY", filed, accn)
            if all((fy - 1, i) in eps for i in (1, 2, 3, 4)) and \
                    (fy, 4) not in drop_comparatives:
                add(f"{fy - 1}-01-01", f"{fy - 1}-12-31",
                    sum(eps[(fy - 1, i)] for i in (1, 2, 3, 4)),
                    "10-K", fy, "FY", filed, accn)

    if split is not None:
        effective, ratio = split
        for row in rows:
            if row["filed"] >= effective:
                row["val"] = row["val"] * ratio

    return {"units": {unit: [dict(r) for r in rows]}}


# Twenty-four quarters. The year-on-year change alternates 0.30 / 0.50 so the
# trailing standard deviation is a number the test can state rather than one
# the module hands back about itself.
_BASE_VALUES = [1.00, 1.20, 1.10, 1.40]
_STEPS = [0.30, 0.50, 0.30, 0.50, 0.50, 0.30, 0.50, 0.30]


def _steady_values(n_years=6):
    values = list(_BASE_VALUES)
    for year in range(1, n_years):
        for fq in range(4):
            step = _STEPS[((year - 1) * 4 + fq) % len(_STEPS)]
            values.append(round(values[-4] + step, 2))
    return values


STEADY = _eps_map(2020, _steady_values())


@pytest.fixture
def filer(monkeypatch):
    """Install one company's facts behind the network seam.

    Returns a setter so a test can state the filing history it is about and
    nothing else. The CIK map is stubbed too: a test that reached EDGAR to
    resolve a ticker would be a network test wearing a unit test's clothes.
    """
    state = {"payload": {}, "calls": []}

    def fetch(cik, taxonomy, tag):
        state["calls"].append(f"{taxonomy}:{tag}")
        return state["payload"].get(f"{taxonomy}:{tag}")

    monkeypatch.setattr(sue, "_fetch_cik_map", lambda: {"TEST": "0000000001"})
    monkeypatch.setattr(sue, "_fetch_company_concept", fetch)
    monkeypatch.setattr(sue, "_fetch_company_facts",
                        lambda cik: state.get("facts"))
    sue._reset_caches()

    def install(payload, concept="us-gaap:EarningsPerShareDiluted"):
        state["payload"][concept] = payload
        return state

    install.state = state
    return install


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


# --- the expectation model --------------------------------------------------

def test_the_expectation_is_the_same_quarter_a_year_ago(filer):
    """A seasonal random walk, not a random walk.

    Earnings are seasonal: a retailer's December quarter dwarfs its March one.
    Differencing against the previous quarter measures that seasonality and
    calls it a surprise, which is why the model compares like fiscal quarter
    with like.
    """
    filer(concept_payload(STEADY))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["success"], result.get("error")
    assert result["comparison_period"] == "2024Q3"
    assert result["eps"] == pytest.approx(STEADY[(2025, 3)])
    assert result["eps_year_ago"] == pytest.approx(STEADY[(2024, 3)])
    assert result["delta"] == pytest.approx(
        STEADY[(2025, 3)] - STEADY[(2024, 3)])


def test_the_signal_divides_by_the_trailing_eight_changes(filer):
    """The denominator is the dispersion of this company's own surprises.

    A 20c beat is enormous for a utility and noise for a semiconductor. Scaling
    by the company's own trailing dispersion is what makes one number
    comparable across the universe -- and it is why the window's length is a
    rule rather than a preference.
    """
    filer(concept_payload(STEADY))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    expected = [STEADY[(fy, fq)] - STEADY[(fy - 1, fq)] for fy, fq in
                [(2025, 2), (2025, 1), (2024, 4), (2024, 3),
                 (2024, 2), (2024, 1), (2023, 4), (2023, 3)]]
    assert result["sigma_quarters"] == 8
    assert result["sigma"] == pytest.approx(statistics.stdev(expected))
    assert result["sue"] == pytest.approx(
        result["delta"] / statistics.stdev(expected))


def test_the_current_surprise_is_not_in_its_own_denominator(filer):
    """Including it shrinks exactly the quarters the signal exists to find.

    A record-breaking surprise would inflate the standard deviation it is
    divided by, pulling the largest SUEs back toward the middle -- the one
    place the strategy makes its money.
    """
    filer(concept_payload(STEADY))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    including = [STEADY[(fy, fq)] - STEADY[(fy - 1, fq)] for fy, fq in
                 [(2025, 3), (2025, 2), (2025, 1), (2024, 4), (2024, 3),
                  (2024, 2), (2024, 1), (2023, 4), (2023, 3)]]
    assert result["sigma"] != pytest.approx(statistics.stdev(including))
    assert result["sigma_periods"][0] == "2025Q2"
    assert "2025Q3" not in result["sigma_periods"]


# --- fiscal identity, never a calendar bucket -------------------------------

def test_fiscal_identity_comes_from_the_filing_not_the_calendar(filer):
    """AMAT's 13 August print is labelled 2026-09-30 by the vendor calendar.

    A quarter that ends 26 July is the filer's third fiscal quarter of 2026 and
    nothing else. Bucketing it by the calendar quarter its end date falls in
    puts it under Q3 of a calendar year that has not finished, and joining a
    price series or a consensus snapshot on that label returns nothing at all.
    """
    filer({"units": {"USD/shares": [
        {"start": "2026-04-27", "end": "2026-07-26", "val": 2.46,
         "accn": "a", "fy": 2027, "fp": "Q2", "form": "10-Q",
         "filed": "2026-08-26"},
        {"start": "2026-01-26", "end": "2026-07-26", "val": 4.85,
         "accn": "a", "fy": 2027, "fp": "Q2", "form": "10-Q",
         "filed": "2026-08-26"},
    ]}})

    series = sue.eps_series("TEST", as_of="2026-09-01")

    quarter = series["quarters"][-1]
    assert quarter["fiscal_period"] == "2027Q2"
    assert quarter["period_end"] == "2026-07-26"
    assert quarter["fiscal_year"] == 2027 and quarter["fiscal_quarter"] == 2


def test_a_fifty_two_week_year_keeps_its_quarter_identity(filer):
    """Twelve-week quarters and a sixteen-week fourth one, as Costco files.

    A duration window tight enough to call 91 days "a quarter" throws away the
    83-day and 112-day ones, and a filer whose year ends on the Sunday nearest
    31 August never lands on a calendar boundary at all.
    """
    filer({"units": {"USD/shares": [
        {"start": "2025-09-01", "end": "2026-05-10", "val": 14.01, "accn": "q3",
         "fy": 2026, "fp": "Q3", "form": "10-Q", "filed": "2026-06-03"},
        {"start": "2026-02-16", "end": "2026-05-10", "val": 4.93, "accn": "q3",
         "fy": 2026, "fp": "Q3", "form": "10-Q", "filed": "2026-06-03"},
        {"start": "2025-09-01", "end": "2026-08-30", "val": 20.10, "accn": "fy",
         "fy": 2026, "fp": "FY", "form": "10-K", "filed": "2026-10-08"},
    ]}})

    series = sue.eps_series("TEST", as_of="2026-11-01")

    by_period = {q["fiscal_period"]: q for q in series["quarters"]}
    assert by_period["2026Q3"]["period_start"] == "2026-02-16"
    assert by_period["2026Q4"]["eps"] == pytest.approx(20.10 - 14.01)
    assert by_period["2026Q4"]["period_end"] == "2026-08-30"


# --- the fourth quarter -----------------------------------------------------

def test_the_fourth_quarter_is_derived_and_says_so(filer):
    """No 10-Q covers it, so it is the year minus the nine months.

    Both inputs are the filer's own reported EPS, so the multi-class trap does
    not arise -- but the subtraction of two figures each rounded to a cent can
    land a cent away from the number the company announced, and a caller
    weighing a marginal SUE deserves to know which quarters carry that.
    """
    filer(concept_payload(STEADY))

    series = sue.eps_series("TEST", as_of="2026-06-01")
    by_period = {q["fiscal_period"]: q for q in series["quarters"]}

    assert by_period["2025Q4"]["source"] == "derived"
    assert by_period["2025Q3"]["source"] == "reported"
    assert by_period["2025Q4"]["eps"] == pytest.approx(STEADY[(2025, 4)])
    assert "annual" in by_period["2025Q4"]["derivation"]


def test_the_fourth_quarter_is_absent_when_the_nine_months_are(filer):
    """Not zero, and not the whole year mislabelled as a quarter.

    Without the third-quarter year-to-date figure there is nothing to subtract,
    and an annual EPS sitting in a quarterly slot is a 4x overstatement that
    would read as the largest surprise in the universe.
    """
    eps = dict(STEADY)
    del eps[(2025, 3)]
    filer(concept_payload(eps))

    series = sue.eps_series("TEST", as_of="2025-12-01")
    periods = {q["fiscal_period"] for q in series["quarters"]}

    assert "2025Q4" not in periods
    assert "2025Q3" not in periods
    assert "2024Q4" in periods


# --- the window is a rule ---------------------------------------------------

def test_fewer_than_six_trailing_changes_is_no_signal(filer):
    """Six of the eight, or nothing.

    A standard deviation from three observations is not an estimate of
    dispersion, it is an estimate of noise -- and dividing by it produces the
    largest SUEs for the companies with the shortest history.
    """
    filer(concept_payload(_eps_map(2023, _steady_values()[:12])))

    result = sue.sue_ts("TEST", as_of="2026-06-01", fiscal_period="2025Q2")

    assert result["success"] is False
    assert result["sue"] is None
    assert "6" in result["error"] and "8" in result["error"]


def test_a_shorter_window_is_never_substituted(filer):
    """The refusal names the count so it cannot be mistaken for a quiet zero."""
    filer(concept_payload(_eps_map(2023, _steady_values()[:12])))

    result = sue.sue_ts("TEST", as_of="2026-06-01", fiscal_period="2025Q2")

    assert result["sigma"] is None
    assert result["sigma_quarters"] is not None
    assert result["sigma_quarters"] < 6


def test_six_of_the_eight_is_enough(filer):
    """The floor is a floor, not a target: a hole or two does not disqualify."""
    eps = dict(STEADY)
    del eps[(2022, 3)]
    filer(concept_payload(eps))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["success"] is True, result.get("error")
    assert result["sigma_quarters"] == 6


def test_the_window_never_reaches_past_the_eighth_quarter(filer):
    """Missing quarters are missing, not an invitation to look further back.

    Reaching to the ninth and tenth to make the count up would silently widen
    the window on exactly the companies whose history is patchy, so their
    denominators would come from a different regime than everyone else's.
    """
    eps = dict(STEADY)
    del eps[(2022, 3)]
    filer(concept_payload(eps))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["sigma_quarters"] == 6
    assert "2023Q2" not in result["sigma_periods"]
    assert "2023Q1" not in result["sigma_periods"]


def test_a_zero_denominator_refuses_rather_than_dividing(filer):
    """Eight identical year-on-year changes leave no scale to divide by.

    The arithmetic would raise, and the tempting repair -- a floor on sigma --
    would invent a scale nothing in the filings supports and hand back a very
    large SUE for a company whose earnings have never varied.
    """
    values = [1.00, 1.20, 1.10, 1.40]
    for _ in range(5):
        values += [round(v + 0.25, 2) for v in values[-4:]]
    filer(concept_payload(_eps_map(2020, values)))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["success"] is False
    assert result["sue"] is None
    assert "zero" in result["error"].lower() or \
        "dispersion" in result["error"].lower()


# --- point in time ----------------------------------------------------------

def test_a_quarter_is_invisible_before_the_filing_that_carries_it(filer):
    """The period ended; the number was not knowable.

    A series filtered on period end alone hands a simulation the September
    quarter in early October, weeks before anybody could read it -- and the
    whole trade is what happens in the days after the print.
    """
    filer(concept_payload(STEADY))

    before = sue.eps_series("TEST", as_of="2025-10-20")
    after = sue.eps_series("TEST", as_of="2025-11-05")

    assert "2025Q3" not in {q["fiscal_period"] for q in before["quarters"]}
    assert "2025Q3" in {q["fiscal_period"] for q in after["quarters"]}


def test_the_signal_carries_the_date_it_became_computable(filer):
    """`known_at` is the filing date, not the period end.

    A backtest that enters on the period end rather than the filing date buys
    the drift before the announcement that causes it, which is not a strategy
    anyone could have run.
    """
    filer(concept_payload(STEADY))

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["known_at"] == _plus("2025-09-30", 30)
    assert result["period_end"] == "2025-09-30"


def test_asking_for_a_quarter_not_yet_filed_refuses(filer):
    filer(concept_payload(STEADY))

    result = sue.sue_ts("TEST", as_of="2025-10-20", fiscal_period="2025Q3")

    assert result["success"] is False
    assert "2025Q3" in result["error"]


def test_the_latest_quarter_is_the_latest_one_known(filer):
    filer(concept_payload(STEADY))

    result = sue.sue_ts("TEST", as_of="2025-10-20")

    assert result["fiscal_period"] == "2025Q2"


# --- restatement ------------------------------------------------------------

def test_the_original_filing_wins_over_a_later_amendment(filer):
    """XBRL as filed is what was known at the time; an amended figure is not.

    A restatement that lands in March cannot be what a February simulation
    traded on, and taking the newest value for a period silently rewrites every
    surprise in the history.
    """
    payload = concept_payload(STEADY)
    payload["units"]["USD/shares"].append(
        {"start": "2025-07-01", "end": "2025-09-30", "val": 9.99,
         "accn": "amend", "fy": 2025, "fp": "Q3", "form": "10-Q/A",
         "filed": "2026-02-15"})
    filer(payload)

    series = sue.eps_series("TEST", as_of="2026-06-01")
    by_period = {q["fiscal_period"]: q for q in series["quarters"]}

    assert by_period["2025Q3"]["eps"] == pytest.approx(STEADY[(2025, 3)])
    assert by_period["2025Q3"]["known_at"] == _plus("2025-09-30", 30)


# --- share basis ------------------------------------------------------------

def test_a_split_leaves_the_signal_exactly_where_it_was(filer):
    """SUE is a ratio, so a ten-for-one split must not move it at all.

    Nothing in the XBRL API marks a split. It shows up only as the same fiscal
    period carrying 5.98 in the filing that first reported it and 0.60 in the
    filing a year later -- NVDA, April 2024 quarter, verified live. Difference
    the originals across that boundary and the surprise is -5.38 against a true
    +0.09: the largest miss in the universe, invented by arithmetic.
    """
    # Ten times STEADY, so that after a ten-for-one the figures sit where a
    # real post-split EPS sits. At a tenth of this the rounding floor bites
    # first and the refusal is the correct answer, which is a different test.
    pre_split = {key: value * 10 for key, value in STEADY.items()}
    filer(concept_payload(pre_split))
    plain = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    filer.state["payload"].clear()
    sue._reset_caches()
    filer(concept_payload(pre_split, split=("2024-06-01", 0.1)))
    split = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert plain["success"] is True, plain.get("error")
    assert split["success"] is True, split.get("error")
    assert split["sue"] == pytest.approx(plain["sue"], rel=1e-9)
    assert split["basis_changes"], "the split was never noticed"
    assert split["basis_changes"][0]["ratio"] == pytest.approx(0.1, rel=0.05)


def test_a_split_that_cannot_be_placed_leaves_the_quarter_out(filer):
    """Bracketed to a year is not placed.

    A filer that tags no year-to-date figure gives up the within-year evidence
    that normally settles this, leaving only the restated comparatives -- which
    arrive four quarters late. The change could then have happened at any of
    several filings, and rebasing on a guess is a tenfold error on whichever
    quarters fall the wrong side. The quarter is dropped and the reason kept.
    """
    def one(accn, start_date, end_date, value, fy, fp, filed):
        return {"start": start_date, "end": end_date, "val": value,
                "accn": accn, "fy": fy, "fp": fp, "form": "10-Q",
                "filed": filed}

    filer({"units": {"USD/shares": [
        one("a1", "2024-01-01", "2024-03-31", 1.00, 2024, "Q1", "2024-04-30"),
        one("a2", "2024-04-01", "2024-06-30", 1.20, 2024, "Q2", "2024-07-30"),
        one("a3", "2024-07-01", "2024-09-30", 1.10, 2024, "Q3", "2024-10-30"),
        one("a4", "2025-01-01", "2025-03-31", 0.13, 2025, "Q1", "2025-04-30"),
        one("a4", "2024-01-01", "2024-03-31", 0.10, 2025, "Q1", "2025-04-30"),
        one("a5", "2025-04-01", "2025-06-30", 0.15, 2025, "Q2", "2025-07-30"),
        one("a5", "2024-04-01", "2024-06-30", 0.12, 2025, "Q2", "2025-07-30"),
    ]}})

    series = sue.eps_series("TEST", as_of="2025-09-30")

    assert [u["fiscal_period"] for u in series["basis_unresolved"]] == ["2024Q3"]
    assert "2024Q3" not in {q["fiscal_period"] for q in series["quarters"]}
    assert series["basis_unresolved"][0]["ratio"] == pytest.approx(0.1, rel=0.2)


def test_a_penny_of_rounding_is_not_a_split(filer):
    """Filers round EPS to a cent and two filings can round differently.

    Treating every disagreement as a basis change would rebase the series by
    1.007 on names that never split, and the drift compounds across a decade.
    """
    payload = concept_payload(STEADY)
    for row in payload["units"]["USD/shares"]:
        if row["accn"] == "000-2025-Q3" and row["end"] == "2024-09-30":
            row["val"] = round(row["val"] + 0.01, 2)
    filer(payload)

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["success"] is True, result.get("error")
    assert result["basis_changes"] == []


# --- what is not there ------------------------------------------------------

def test_a_foreign_private_issuer_refuses_and_says_why(filer):
    """TSM files 20-F annually and 6-K for interims, and 6-K carries no XBRL.

    Verified live: every EPS fact TSM tags covers a full calendar year. There
    is no quarterly figure to be had from anywhere, so a tool that returns a
    partial series here is describing a company that does not exist.
    """
    filer(None, concept="us-gaap:EarningsPerShareDiluted")
    filer({"units": {"TWD/shares": [
        {"start": "2023-01-01", "end": "2023-12-31", "val": 32.85,
         "accn": "f1", "fy": 2023, "fp": "FY", "form": "20-F",
         "filed": "2024-04-18"},
        {"start": "2024-01-01", "end": "2024-12-31", "val": 44.67,
         "accn": "f2", "fy": 2024, "fp": "FY", "form": "20-F",
         "filed": "2025-04-17"},
    ]}}, concept="ifrs-full:DilutedEarningsLossPerShare")

    result = sue.sue_ts("TEST", as_of="2026-06-01")

    assert result["success"] is False
    assert "20-F" in result["error"]
    assert "6-K" in result["error"] or "quarterly" in result["error"].lower()


def test_a_filer_that_tags_no_eps_at_all_says_that_instead(filer):
    """Distinct from a foreign issuer, and distinct from a bad ticker."""
    result = sue.sue_ts("TEST", as_of="2026-06-01")

    assert result["success"] is False
    assert "TEST" in result["error"]
    assert result["concepts_tried"]


def test_a_ticker_that_is_not_a_registrant_refuses(filer):
    result = sue.sue_ts("NOTREAL", as_of="2026-06-01")

    assert result["success"] is False
    assert "NOTREAL" in result["error"]
    assert result["concepts_tried"] == []


# --- the concept chain ------------------------------------------------------

def test_one_concept_carries_the_whole_series(filer):
    """Basic and diluted EPS differ by the dilution, which drifts over time.

    Splicing one onto the other puts a step change in the series that looks
    exactly like a surprise, at whichever quarter the filer changed its tagging.
    """
    filer(concept_payload(STEADY))
    filer(concept_payload({k: v * 1.05 for k, v in STEADY.items()}),
          concept="us-gaap:EarningsPerShareBasic")

    series = sue.eps_series("TEST", as_of="2026-06-01")

    assert series["concept"] == "us-gaap:EarningsPerShareDiluted"
    assert {q["concept"] for q in series["quarters"]} == {series["concept"]}


def test_the_series_never_asks_for_a_share_count(filer):
    """EPS is the filer's own, never net income over shares.

    Share counts differ by class -- BRK's two classes are 1500:1 -- so a
    derived per-share figure is wrong by whichever class the count came from,
    and plausibly so.
    """
    filer(concept_payload(STEADY))

    sue.eps_series("TEST", as_of="2026-06-01")

    assert filer.state["calls"], "nothing was fetched at all"
    for call in filer.state["calls"]:
        tag = call.split(":")[-1]
        assert "PerShare" in tag or "PerDilutedShare" in tag, call
        assert "SharesOutstanding" not in tag
        assert "NetIncome" not in tag


def test_a_stray_unit_does_not_join_the_series(filer):
    """Costco's 2010 10-K tags eleven EPS facts under `pure`, verified live.

    Mixing units mixes scales. The series takes the unit the filer uses for
    the concept and leaves the other alone.
    """
    payload = concept_payload(STEADY)
    payload["units"]["pure"] = [
        {"start": "2025-07-01", "end": "2025-09-30", "val": 0.0021,
         "accn": "pure", "fy": 2025, "fp": "Q3", "form": "10-K",
         "filed": "2025-10-30"}]
    filer(payload)

    series = sue.eps_series("TEST", as_of="2026-06-01")
    by_period = {q["fiscal_period"]: q for q in series["quarters"]}

    assert series["unit"] == "USD/shares"
    assert by_period["2025Q3"]["eps"] == pytest.approx(STEADY[(2025, 3)])


# --- history ----------------------------------------------------------------

def test_the_history_is_every_quarter_the_filings_support(filer):
    filer(concept_payload(STEADY))

    history = sue.sue_ts_history("TEST", as_of="2026-01-01")

    assert history["success"] is True, history.get("error")
    periods = [row["fiscal_period"] for row in history["signals"]]
    assert periods == sorted(periods)
    assert "2025Q3" in periods
    assert all(row["sue"] is not None for row in history["signals"])


# --- the analyst leg --------------------------------------------------------

def test_the_analyst_variant_refuses_until_the_record_is_deep(filer, store):
    """The history does not exist to be fetched, only to be accumulated.

    Finnhub returns four quarters at limit=12 and at limit=30 -- verified -- so
    an analyst-based surprise for a 2024 print cannot be reconstructed today.
    Manufacturing one from today's consensus would be lookahead of the purest
    kind: the estimate as revised after the fact.
    """
    filer(concept_payload(STEADY))

    result = sue.sue_af("TEST", as_of="2026-01-01", fiscal_period="2025Q3",
                        actuals={})

    assert result["success"] is False
    assert result["sue"] is None
    assert "consensus" in result["error"].lower()
    assert result["surprises_available"] == 0


def test_the_analyst_variant_will_not_mix_gaap_with_a_street_estimate(filer,
                                                                     store):
    """A street estimate is a non-GAAP number and XBRL only has the GAAP one.

    Verified live on 2026-08-27: Finnhub reports MSFT's 2026Q2 actual as 4.14
    against the 5.16 diluted EPS in the 10-Q, and NVDA's 2026Q2 as 1.05
    against 1.08. Subtract the estimate from the wrong actual and every
    surprise carries a dollar of definitional gap that varies quarter to
    quarter -- larger than the surprises themselves, and pointing wherever the
    one-offs happen to point.
    """
    filer(concept_payload(STEADY))

    result = sue.sue_af("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["success"] is False
    assert "non-GAAP" in result["error"]
    assert result["sue"] is None


def _seed_consensus(store, series, days_before=3):  # noqa: D401
    """One consensus snapshot per quarter, a few days before the print.

    The misses vary quarter to quarter on purpose: a filer that beats by
    exactly five cents eight times running has zero dispersion of surprise,
    which is a refusal rather than a signal and would hide the arithmetic
    this test is about.
    """
    misses = [0.05, 0.02, 0.09, 0.04]
    seeded, actuals = {}, {}
    for index, quarter in enumerate(series["quarters"]):
        miss = misses[index % len(misses)]
        period = quarter["fiscal_period"]
        # The vendor's own actual, a few cents off the GAAP figure the way a
        # non-GAAP one is, so the test cannot pass by accident on the XBRL
        # number standing in for it.
        actuals[period] = round(quarter["eps"] - 0.03, 2)
        seeded[period] = round(actuals[period] - miss, 2)
        store.record_consensus(_plus(quarter["known_at"], -days_before),
                               "TEST", period, eps_estimate=seeded[period],
                               analyst_count=9)
    return seeded, actuals


def test_the_analyst_variant_computes_once_the_record_is_deep(filer, store):
    """The seam is real: it starts working when the recorder has run long
    enough, and needs no change here to do so."""
    filer(concept_payload(STEADY))
    series = sue.eps_series("TEST", as_of="2026-01-01")
    seeded, actuals = _seed_consensus(store, series)

    result = sue.sue_af("TEST", as_of="2026-01-01", fiscal_period="2025Q3",
                        actuals=actuals)

    assert result["success"] is True, result.get("error")
    assert result["consensus"] == pytest.approx(seeded["2025Q3"])
    assert result["surprise"] == pytest.approx(
        actuals["2025Q3"] - seeded["2025Q3"])
    assert result["surprises_available"] >= 6
    assert result["sue"] is not None


def test_the_analyst_leg_never_reads_a_consensus_set_after_the_print(filer,
                                                                    store):
    """An estimate revised the day after the announcement is not a surprise.

    The consensus that matters is the last one standing before the print. Read
    the one recorded after it and the surprise collapses toward zero, because
    the street has already marked to the actual.
    """
    filer(concept_payload(STEADY))
    series = sue.eps_series("TEST", as_of="2026-01-01")
    seeded, actuals = _seed_consensus(store, series)
    for quarter in series["quarters"]:
        store.record_consensus(_plus(quarter["known_at"], 1), "TEST",
                               quarter["fiscal_period"],
                               eps_estimate=quarter["eps"], analyst_count=9)

    result = sue.sue_af("TEST", as_of="2026-01-01", fiscal_period="2025Q3",
                        actuals=actuals)

    assert result["consensus"] == pytest.approx(seeded["2025Q3"])
    assert result["surprise"] != pytest.approx(0.0, abs=1e-9)


# --- what the sweep over real filers turned up -------------------------------

def test_a_ten_q_that_mislabels_its_period_is_read_from_its_periods(filer):
    """Dell's first two FY2024 10-Qs declare a fiscal period focus of `FY`.

    Verified live: accessions 0001571996-23-000019 and -000032, both 10-Q, both
    tagged FY. Dropping a filing whose focus is not Q1, Q2 or Q3 costs two real
    quarters and, through the year-ago term, two more -- which is most of a
    trailing window. The periods the filing carries say which quarter it is:
    the year to date runs one, two or three quarters long.
    """
    filer({"units": {"USD/shares": [
        {"start": "2023-02-04", "end": "2023-08-04", "val": 1.42, "accn": "q2",
         "fy": 2024, "fp": "FY", "form": "10-Q", "filed": "2023-09-12"},
        {"start": "2023-05-06", "end": "2023-08-04", "val": 0.63, "accn": "q2",
         "fy": 2024, "fp": "FY", "form": "10-Q", "filed": "2023-09-12"},
        {"start": "2023-02-04", "end": "2023-05-05", "val": 0.79, "accn": "q1",
         "fy": 2024, "fp": "FY", "form": "10-Q", "filed": "2023-06-12"},
    ]}})

    series = sue.eps_series("TEST", as_of="2024-01-01")
    by_period = {q["fiscal_period"]: q["eps"] for q in series["quarters"]}

    assert by_period["2024Q1"] == pytest.approx(0.79)
    assert by_period["2024Q2"] == pytest.approx(0.63)


def test_a_restatement_is_not_a_rescaling(filer):
    """Dell's FY2025 10-K restates three FY2024 quarters by 4%, 5% and 9%.

    A split rescales every prior period by one exact factor; a restatement
    moves each period by its own amount. Treating the latter as a share basis
    stretches the whole history onto a figure nobody traded on -- and, worse,
    leaves the quarters between the two filings unplaceable and therefore
    unusable, which is how one small restatement cost a real name its signal.
    """
    payload = concept_payload(STEADY)
    restated = {"2024-01-01/2024-03-31": 1.088, "2024-04-01/2024-06-30": 1.051}
    for row in payload["units"]["USD/shares"]:
        key = f"{row['start']}/{row['end']}"
        if key in restated and row["filed"] > "2024-12-31":
            row["val"] = round(row["val"] * restated[key], 2)
    filer(payload)

    result = sue.sue_ts("TEST", as_of="2026-01-01", fiscal_period="2025Q3")

    assert result["basis_changes"] == []
    assert result["success"] is True, result.get("error")


def test_a_concept_the_filer_abandoned_does_not_win_the_chain(filer):
    """Ford's continuing-operations EPS stops in 2011; its diluted EPS is
    current. Coca-Cola is the same shape.

    Length is not currency. Picking the chain member with the most quarters
    picked Ford's dead tag and reported a 2011 quarter as today's signal --
    a number that is not wrong so much as fifteen years out of date.
    """
    filer(concept_payload(_eps_map(2010, _steady_values())))
    filer(concept_payload(_eps_map(2020, _steady_values())),
          concept="us-gaap:EarningsPerShareBasicAndDiluted")

    series = sue.eps_series("TEST", as_of="2026-01-01")

    assert series["concept"] == "us-gaap:EarningsPerShareBasicAndDiluted"
    assert series["quarters"][-1]["fiscal_period"] == "2025Q3"


def test_an_empty_company_concept_falls_back_to_company_facts(filer):
    """SEC's own two endpoints disagree, verified live on 2026-08-27.

    `companyconcept` for Coca-Cola's `EarningsPerShareDiluted` returns the
    concept header with both unit arrays empty; `companyfacts` for the same
    CIK carries 229 facts of it, the most recent filed three months earlier.
    Ford is the same. Trusting the cheap endpoint alone silently drops filers
    that report perfectly well.
    """
    payload = concept_payload(STEADY)
    filer({"units": {"pure": [], "USD/shares": []}})
    filer.state["facts"] = {"facts": {"us-gaap": {
        "EarningsPerShareDiluted": payload}}}

    series = sue.eps_series("TEST", as_of="2026-01-01")

    assert series["success"] is True, series.get("error")
    assert series["source"] == "companyfacts"
    assert series["quarters"][-1]["fiscal_period"] == "2025Q3"


def test_a_filer_with_no_consolidated_eps_says_which_forms_it_files(filer):
    """Visa and Berkshire tag no undimensioned EPS at all, verified live.

    Both report earnings per share by share class, and SEC's company APIs
    return only consolidated facts, so the concept is absent rather than
    partial. "Berkshire has no EPS" would be a startling claim about the
    company; what is true is that none is readable this way.
    """
    filer.state["facts"] = {"facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"start": "2026-01-01", "end": "2026-03-31", "val": 1.0,
             "accn": "a", "fy": 2026, "fp": "Q1", "form": "10-Q",
             "filed": "2026-04-30"}]}}},
        "us-gaap": {}}}

    result = sue.sue_ts("TEST", as_of="2026-06-01")

    assert result["success"] is False
    assert "10-Q" in result["error"]
    assert "class" in result["error"].lower()


def test_a_short_history_refusal_says_how_short(filer):
    """SEC's ticker file maps XOM to CIK 0002115436, a successor registrant
    with one 10-Q on EDGAR; the history sits under the predecessor CIK.

    "No year-ago quarter" reads as a data gap. "One quarter, 2026Q2 to 2026Q2,
    under CIK 0002115436" reads as the wrong registrant, which is what it is.
    """
    filer({"units": {"USD/shares": [
        {"start": "2026-04-01", "end": "2026-06-30", "val": 2.0, "accn": "a",
         "fy": 2026, "fp": "Q2", "form": "10-Q", "filed": "2026-07-30"},
    ]}})

    result = sue.sue_ts("TEST", as_of="2026-08-27")

    assert result["success"] is False
    assert "1 quarter" in result["error"]
    assert "0000000001" in result["error"]


def test_a_series_that_stopped_says_when_it_stopped(filer):
    """Berkshire's readable EPS ends in 2014, when it moved to per-class facts.

    Reporting a 2010 quarter as today's signal is worse than refusing: the
    number is real, the date is sixteen years stale, and nothing in the result
    says so. A domestic filer still reporting files a 10-Q within 45 days of
    every quarter end, so a long silence is a fact about the filer or about
    what is readable -- either way, not a signal.
    """
    filer(concept_payload(_eps_map(2008, _steady_values())))

    result = sue.sue_ts("TEST", as_of="2026-08-27")

    assert result["success"] is False
    assert "2013Q4" in result["error"] or "2013Q3" in result["error"]
    assert "stopped filing" in result["error"]


def test_one_split_seen_twice_is_still_one_split(filer):
    """Salesforce's 4-for-1 measures 0.2516 through one period and 0.2857
    through another, verified live: a quarter whose EPS rounds to a couple of
    cents cannot express a quarter of itself any more precisely.

    Counted as two events they multiply, rebasing the pre-split history by
    0.072 instead of 0.25, and the quarters between the two brackets become
    unplaceable and drop out of every window that needs them.
    """
    payload = concept_payload(STEADY, split=("2024-06-01", 0.25))
    for row in payload["units"]["USD/shares"]:
        if row["accn"] == "000-2025-Q1" and row["end"] == "2024-03-31":
            row["val"] = row["val"] * 0.2857 / 0.25
    filer(payload)

    series = sue.eps_series("TEST", as_of="2026-01-01")

    assert len(series["basis_changes"]) == 1
    assert series["basis_unresolved"] == []


def test_a_split_is_placed_from_the_filing_whose_year_it_broke(filer):
    """One quarter after the split, not five.

    The restated comparative is a year late, so on its own it brackets NVDA's
    June 2024 split to the twelve months to August 2024 and leaves the three
    filings inside unplaceable -- taking the signal away for three quarters,
    starting with the print right after the split. The filing's own year to
    date settles it a quarter later: 1.27 less 0.67 is 0.60 for a quarter it
    had filed as 5.98.
    """
    filer(concept_payload({k: v * 10 for k, v in STEADY.items()},
                          split=("2024-06-01", 0.1)))

    series = sue.eps_series("TEST", as_of="2024-09-30")

    assert series["basis_unresolved"] == []
    assert len(series["basis_changes"]) == 1
    low, high = series["basis_changes"][0]["between"]
    assert low == "2024-04-30" and high == "2024-07-30"
    by_period = {q["fiscal_period"]: q for q in series["quarters"]}
    assert by_period["2024Q1"]["basis_factor"] == pytest.approx(0.1)
    assert by_period["2024Q2"]["basis_factor"] == pytest.approx(1.0)
    assert by_period["2024Q1"]["eps"] == pytest.approx(
        by_period["2024Q1"]["eps_as_filed"] * 0.1)


def test_a_cent_of_rounding_on_a_small_eps_is_not_a_split(filer):
    """Amazon's 2012 quarters round to a few cents, and 0.13 against 0.20 is
    a ratio of 1.54 -- past the bar a 3-for-2 split would clear.

    Verified live: two such pairs three months apart were read as a rescaling
    and its near-inverse, 1.5420x then 0.6364x. Both are the cent the filer
    rounds to, seen through a denominator small enough to magnify it. A
    rescaling has to clear the bar even when the rounding is worst.
    """
    payload = concept_payload(STEADY)
    for row in payload["units"]["USD/shares"]:
        if row["end"] == "2022-03-31":
            row["val"] = 0.20 if row["filed"] > "2022-12-31" else 0.13
    filer(payload)

    series = sue.eps_series("TEST", as_of="2026-01-01")

    assert series["basis_changes"] == []
    assert series["basis_unresolved"] == []


def test_two_measurements_of_one_split_never_multiply(filer):
    """Salesforce's 4-for-1 measures 0.2222 through one period and 0.2857
    through another -- 29% apart, further than a rounding allowance can carry.

    Left as two events they compound: 0.0635 rather than 0.25, and the whole
    pre-split history is rescaled four times too far. Two brackets that end up
    covering the same stretch of filing dates are one event, whatever their
    ratios say, because a filer does not split twice between two filings.
    """
    payload = concept_payload(STEADY, split=("2024-06-01", 0.25))
    for row in payload["units"]["USD/shares"]:
        if row["accn"] == "000-2025-Q1" and row["end"] == "2024-03-31":
            row["val"] = row["val"] * 0.2857 / 0.25
        if row["accn"] == "000-2025-Q2" and row["end"] == "2024-06-30":
            row["val"] = row["val"] * 0.2222 / 0.25
    filer(payload)

    series = sue.eps_series("TEST", as_of="2026-01-01")

    assert len(series["basis_changes"]) == 1
    assert series["basis_changes"][0]["ratio"] == pytest.approx(0.25, rel=0.2)


def test_two_penny_figures_assert_nothing_either_way(filer):
    """Half a cent against a one-cent figure is a factor of three.

    A pair like that can neither establish a rescaling nor rule one out, and
    the failure that showed up live was the first: identical one-cent values
    were read as a basis change of exactly 1.0000x, which then made the
    quarters around it unplaceable for no reason at all.
    """
    payload = concept_payload(STEADY)
    for row in payload["units"]["USD/shares"]:
        if row["end"] == "2022-03-31":
            row["val"] = 0.01
    filer(payload)

    series = sue.eps_series("TEST", as_of="2026-01-01")

    assert series["basis_changes"] == []
    assert series["basis_unresolved"] == []
