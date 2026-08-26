"""A stock split is not dilution, and the cover-page tag cannot tell them apart.

`dei:EntityCommonStockSharesOutstanding` states the shares that existed on the
day the cover page was signed, in the units of that day. NVDA's Q3 FY2024 cover
says 2.47bn and its Q1 FY2027 cover says 24.2bn; the 10-for-1 split of June 2024
lives entirely in the gap. Differencing the two reported **+879.76%** and called
it `"dilution"` -- for a company that repurchased $40.1bn of its own stock in the
trailing year and whose split-adjusted count fell about 2%.

That is the worst shape a data error can take. Not a crash, not an absurd number,
but a confident verdict pointing the opposite way on a thesis-level input. Four
of the five split cases checked (NVDA, CMG, WMT, LRCX) inverted the verdict, and
two other tools in the same deployment already knew better -- `get_buyback_history`
reports the $40.1bn and `get_corporate_actions` reports the 10:1 ratio -- so the
answer was contradicted from inside its own toolbox.

The rules these tests pin, in order of importance:

1. A direction is never stated when it cannot be supported. A refusal is a worse
   answer than a correct one and a far better answer than an inverted one.
2. A genuine large issuance is still reported as dilution. Suppressing real
   dilution as a "suspected split" would trade one silent error for another.
3. The adjustment is auditable: which ratio, between which dates, over which raw
   figures. A silently adjusted number is as hard to check as a silently wrong one.
4. `by_class` gets the same treatment as the total. A multi-class filer splits
   every class, and a caller reading the breakdown must not be handed the
   unadjusted series the headline was fixed to avoid.
"""
import os

import pytest

from tools.web_search_server import dilution
from tools.web_search_server.sec_series import ConceptFact, FilingPoint

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"

CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"


def network(func):
    """The live-EDGAR marker plus the offline skip, matching test_dilution.py."""
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


# --------------------------------------------------------------------- fixtures

def _point(filing_date, as_of, shares, member=None):
    """One filing's cover-page count, newest-first ordering supplied by caller."""
    if not isinstance(shares, (list, tuple)):
        shares = [(shares, member)]
    facts = [
        ConceptFact(value=float(value), period=as_of, context_ref=f"c-{i}",
                    concept=dilution.SHARES_CONCEPT, unit="shares",
                    dimensions={CLASS_AXIS: klass} if klass else {})
        for i, (value, klass) in enumerate(shares)
    ]
    return FilingPoint(filing_date=filing_date, form="10-Q",
                       accession=f"acc-{filing_date}", facts=facts)


def _nvda_series():
    """NVDA's eight 10-Q covers as filed, verified against EDGAR 2026-08-26.

    The two oldest are pre-split; the June 2024 10-for-1 sits between
    2024-05-24 and 2024-08-23.
    """
    return [
        _point("2026-05-20", "2026-05-15", 24_200_000_000),
        _point("2025-11-19", "2025-11-14", 24_300_000_000),
        _point("2025-08-27", "2025-08-22", 24_300_000_000),
        _point("2025-05-28", "2025-05-23", 24_400_000_000),
        _point("2024-11-20", "2024-11-15", 24_490_000_000),
        _point("2024-08-28", "2024-08-23", 24_530_000_000),
        _point("2024-05-29", "2024-05-24", 2_460_000_000),
        _point("2023-11-21", "2023-11-17", 2_470_000_000),
    ]


def _cmg_series():
    """CMG's covers around the 50-for-1 of June 2024. Raw change: +4507%."""
    return [
        _point("2026-07-31", "2026-07-24", 1_265_418_000),
        _point("2026-04-30", "2026-04-24", 1_282_734_000),
        _point("2025-10-30", "2025-10-27", 1_322_278_000),
        _point("2024-07-25", "2024-07-22", 1_369_476_000),
        _point("2024-04-25", "2024-04-22", 27_467_000),
    ]


def _msft_series():
    """MSFT: no split since 2003. Nothing here may move."""
    return [
        _point("2026-04-29", "2026-04-23", 7_428_434_704),
        _point("2026-01-28", "2026-01-22", 7_425_629_076),
        _point("2025-10-29", "2025-10-23", 7_432_377_655),
        _point("2024-01-30", "2024-01-25", 7_430_436_229),
    ]


@pytest.fixture
def series(monkeypatch):
    """The filings, offline -- and the quote provider with them.

    A multi-class filer now has its class weights checked against a market
    capitalisation, because summing classes at 1:1 is wrong by a third for a
    filer whose Class A converts into 1,500 Class B. These tests are about
    split adjustment, so the check is left unreachable rather than answered:
    an unreachable source is reported, never taken for agreement.
    """
    def install(points):
        monkeypatch.setattr(dilution, "fetch_concept_series",
                            lambda *a, **k: list(points))
        monkeypatch.setattr(dilution, "fetch_market_share_count",
                            _no_quote_provider)
    return install


def _no_quote_provider(ticker):
    raise dilution.MarketShareCountUnavailable(
        f"no quote provider is reachable from the offline suite ({ticker})")


@pytest.fixture
def splits(monkeypatch):
    """Stand in for the split calendar without touching the network."""
    def install(events):
        monkeypatch.setattr(
            dilution, "fetch_split_history",
            lambda ticker: [{"date": d, "ratio": float(r)} for d, r in events])
    return install


@pytest.fixture
def unreachable_splits(monkeypatch):
    def install(message="HTTPError: 503 from the split calendar"):
        def boom(ticker):
            raise RuntimeError(message)
        monkeypatch.setattr(dilution, "fetch_split_history", boom)
        return message
    return install


@pytest.fixture
def calendar_calls(monkeypatch):
    """Every consultation of the split calendar, recorded rather than made.

    Recorded and not raised: the consultation is wrapped in `except Exception`
    so that a source outage is reported instead of crashing the series, which
    means an AssertionError raised here would be swallowed into `source_error`
    and the test would pass on the very code path it meant to forbid.
    """
    calls = []

    def record(ticker):
        calls.append(ticker)
        return []

    monkeypatch.setattr(dilution, "fetch_split_history", record)
    return calls


# ------------------------------------------------------- the inverted verdict

def test_a_ten_for_one_split_is_not_reported_as_dilution(series, splits):
    """The headline defect. Without this, NVDA reads +879.76% "dilution" in the
    same window it retired $40.1bn of stock."""
    series(_nvda_series())
    splits([("2024-06-10", 10.0)])

    result = dilution.get_share_count_series("NVDA")

    assert result["direction"] == "buyback", (
        f"NVDA repurchased stock over this window; the tool says "
        f"{result['direction']!r} at {result['change_pct']}%")
    assert result["change_pct"] == pytest.approx(-2.02, abs=0.05)


def test_a_fifty_for_one_split_is_not_reported_as_dilution(series, splits):
    """CMG is the extreme case: +4507% raw, and every dollar of it a split."""
    series(_cmg_series())
    splits([("2024-06-26", 50.0)])

    result = dilution.get_share_count_series("CMG")

    assert result["direction"] == "buyback"
    assert result["change_pct"] == pytest.approx(-7.86, abs=0.05)


def test_a_reverse_split_does_not_read_as_a_buyback(series, splits):
    """The mirror error, and the one that flatters a serial diluter: a 1-for-10
    reverse cuts the count 90%, which reads as the largest buyback ever run."""
    series([_point("2026-05-01", "2026-04-25", 110_000_000),
            _point("2025-05-01", "2025-04-25", 1_000_000_000)])
    splits([("2025-09-15", 0.1)])

    result = dilution.get_share_count_series("REVERSE")

    assert result["direction"] == "dilution", (
        f"a 1-for-10 reverse split left 110m shares where 100m were expected; "
        f"that is 10% dilution, reported as {result['direction']!r}")
    assert result["change_pct"] == pytest.approx(10.0, abs=0.01)
    message = result["warnings"][0]["message"]
    assert "1-for-10" in message, (
        f"a reverse split is a 1-for-10, never a 0.1-for-1: {message}")


def test_the_latest_total_is_still_the_count_as_filed(series, splits):
    """Adjustment normalises history onto today's basis, never the reverse.
    `latest_total` has to stay comparable to the share count the market quotes,
    or every market-cap and per-share figure built on it breaks."""
    series(_nvda_series())
    splits([("2024-06-10", 10.0)])

    result = dilution.get_share_count_series("NVDA")

    assert result["latest_total"] == 24_200_000_000


# -------------------------------------------------------------- the audit trail

def test_the_unadjusted_change_is_still_reported(series, splits):
    """An adjusted number nobody can check is only a better class of wrong.
    The raw figure stays so a reader can reproduce the arithmetic."""
    series(_nvda_series())
    splits([("2024-06-10", 10.0)])

    result = dilution.get_share_count_series("NVDA")

    assert result["raw_change_pct"] == pytest.approx(879.757, abs=0.01)


def test_the_applied_ratio_and_its_date_are_reported(series, splits):
    """Which split, and applied where -- that is the whole audit. A ratio
    baked silently into a percentage cannot be argued with."""
    series(_nvda_series())
    splits([("2024-06-10", 10.0)])

    adjustment = dilution.get_share_count_series("NVDA")["split_adjustment"]

    assert adjustment["adjusted"] is True
    assert adjustment["splits_applied"] == [{"date": "2024-06-10", "ratio": 10.0}]
    assert adjustment["raw_oldest_total"] == 2_470_000_000
    assert adjustment["adjusted_oldest_total"] == pytest.approx(24_700_000_000)
    assert adjustment["raw_latest_total"] == 24_200_000_000


def test_each_filing_carries_the_factor_applied_to_it(series, splits):
    """Per-row, so a reader can see exactly where the discontinuity was closed
    rather than being told a single number about the window."""
    series(_nvda_series())
    splits([("2024-06-10", 10.0)])

    rows = {r["filing_date"]: r for r in
            dilution.get_share_count_series("NVDA")["total_series"]}

    assert rows["2023-11-21"]["split_factor"] == pytest.approx(10.0)
    assert rows["2023-11-21"]["total"] == 2_470_000_000, "the as-filed count"
    assert rows["2023-11-21"]["total_split_adjusted"] == pytest.approx(24_700_000_000)
    assert rows["2026-05-20"]["split_factor"] == pytest.approx(1.0)
    assert rows["2026-05-20"]["total_split_adjusted"] == 24_200_000_000


def test_a_split_adjusted_series_says_so_at_the_top_level(series, splits):
    """A caller that reads one field must still learn the series was touched."""
    series(_nvda_series())
    splits([("2024-06-10", 10.0)])

    result = dilution.get_share_count_series("NVDA")

    assert result["split_adjusted"] is True
    codes = [w["code"] for w in result["warnings"]]
    assert "split_adjusted" in codes, (
        f"the series was rebased by 10x and nothing warns: {result['warnings']}")


# ------------------------------------------- telling a split from real dilution

def test_a_genuine_large_issuance_is_still_called_dilution(series, splits):
    """The failure mode on the other side. A company that really did issue 40%
    more shares must not have that suppressed as a suspected split -- that
    trades an inverted verdict for a silenced one."""
    series([_point("2026-05-01", "2026-04-25", 140_000_000),
            _point("2025-05-01", "2025-04-25", 100_000_000)])
    splits([])

    result = dilution.get_share_count_series("ISSUER")

    assert result["direction"] == "dilution"
    assert result["change_pct"] == pytest.approx(40.0, abs=0.01)
    assert result["split_adjusted"] is False


@pytest.mark.parametrize("newest,expected_pct", [
    (140_000_000, 40.0),
    (160_000_000, 60.0),
    (190_000_000, 90.0),
])
def test_no_unround_jump_is_ever_mistaken_for_a_split(series, splits,
                                                      newest, expected_pct):
    """Splits land on round ratios; equity raises do not. 1.4x, 1.6x and 1.9x
    are all large enough to look dramatic and none is near a split ratio, so
    each must survive as dilution with its magnitude intact."""
    series([_point("2026-05-01", "2026-04-25", newest),
            _point("2025-05-01", "2025-04-25", 100_000_000)])
    splits([])

    result = dilution.get_share_count_series("ISSUER")

    assert result["direction"] == "dilution"
    assert result["change_pct"] == pytest.approx(expected_pct, abs=0.01)


def test_a_split_the_calendar_missed_is_refused_not_inverted(series, splits):
    """The residual check, and the reason the calendar is not trusted blindly.
    A 10.00x jump between two consecutive covers is a split whatever the
    calendar says; with no ratio to stand behind, the tool must decline to name
    a direction rather than report a 900% "dilution"."""
    series([_point("2026-05-01", "2026-04-25", 1_000_000_000),
            _point("2025-05-01", "2025-04-25", 100_000_000)])
    splits([])

    result = dilution.get_share_count_series("MISSED")

    assert result["direction"] == "split_suspected_undetermined", (
        f"a 10x jump with no ratio available was reported as "
        f"{result['direction']!r}")
    assert result["change_pct"] is None, (
        "a change_pct beside an undetermined direction is the same wrong "
        "number wearing a disclaimer")
    assert result["raw_change_pct"] == pytest.approx(900.0)


def test_the_suspected_split_is_described_rather_than_just_flagged(series, splits):
    """Warning that something is wrong somewhere is not actionable. The
    reader needs the interval and the implied ratio to go and look it up."""
    series([_point("2026-05-01", "2026-04-25", 1_000_000_000),
            _point("2025-05-01", "2025-04-25", 100_000_000)])
    splits([])

    suspected = dilution.get_share_count_series(
        "MISSED")["split_adjustment"]["unexplained_jumps"]

    assert len(suspected) == 1
    assert suspected[0]["implied_ratio"] == pytest.approx(10.0)
    assert suspected[0]["from_date"] == "2025-04-25"
    assert suspected[0]["to_date"] == "2026-04-25"


# ------------------------------------------------------ when the source is gone

def test_an_unreachable_calendar_refuses_rather_than_guessing(
        series, unreachable_splits):
    """Falling back to the unadjusted series here is the original bug with an
    extra step. The window contains a 10x jump and the ratio cannot be
    confirmed, so no direction is supportable."""
    unreachable_splits()
    series(_nvda_series())

    result = dilution.get_share_count_series("NVDA")

    assert result["direction"] == "split_suspected_undetermined"
    assert result["change_pct"] is None


def test_an_unreachable_calendar_is_reported_not_swallowed(
        series, unreachable_splits):
    """A silent source failure turns "adjusted" into a claim the tool cannot
    keep. The error text has to reach the caller."""
    message = unreachable_splits()
    series(_nvda_series())

    result = dilution.get_share_count_series("NVDA")

    assert message in result["split_adjustment"]["source_error"]
    assert "split_source_unavailable" in [w["code"] for w in result["warnings"]]


def test_an_outage_does_not_silence_a_filer_whose_jumps_are_not_split_shaped(
        series, unreachable_splits):
    """The refusal has to stay narrow. A 20% jump is not a split ratio at any
    tolerance, so an unreachable calendar costs the caller a warning, not the
    answer -- otherwise one flaky dependency mutes every serial diluter."""
    unreachable_splits()
    series([_point("2026-05-01", "2026-04-25", 120_000_000),
            _point("2025-05-01", "2025-04-25", 100_000_000)])

    result = dilution.get_share_count_series("DILUTER")

    assert result["direction"] == "dilution"
    assert result["change_pct"] == pytest.approx(20.0)
    assert "split_source_unavailable" in [w["code"] for w in result["warnings"]]


def test_a_quiet_series_never_needs_the_calendar(series, calendar_calls):
    """MSFT's count moves fractions of a percent between filings. A split large
    enough to matter always shows as a discontinuity in the raw counts, so a
    series without one has nothing to look up -- and paying for a network call
    on every quiet filer, then having to decide what to do when that call
    fails, would be its own defect."""
    series(_msft_series())

    result = dilution.get_share_count_series("MSFT")

    assert calendar_calls == [], (
        "the split calendar was consulted for a series whose largest move "
        "between filings is under a tenth of a percent")
    assert result["direction"] == "flat"
    assert result["split_adjusted"] is False
    assert result["change_pct"] == pytest.approx(result["raw_change_pct"])


def test_a_series_with_no_split_in_the_window_is_left_alone(series, splits):
    """The 2003 split is in MSFT's history and outside this window. Applying it
    would rebase a series that needs no rebasing."""
    series(_msft_series())
    splits([("2003-02-18", 2.0)])

    result = dilution.get_share_count_series("MSFT")

    assert result["split_adjusted"] is False
    assert result["split_adjustment"]["splits_applied"] == []
    assert result["change_pct"] == pytest.approx(-0.0269, abs=0.001)


# ------------------------------------------------------------ the class breakdown

def _two_class_series():
    """A two-class filer straddling a 4-for-1. Class A splits with Class B;
    every class of a split filer is restated, not just the headline."""
    return [
        _point("2026-05-01", "2026-04-25",
               [(3_960_000_000, "us-gaap:CommonClassAMember"),
                (400_000_000, "us-gaap:CommonClassBMember")]),
        _point("2025-05-01", "2025-04-25",
               [(1_000_000_000, "us-gaap:CommonClassAMember"),
                (100_000_000, "us-gaap:CommonClassBMember")]),
    ]


def test_every_class_row_carries_its_adjusted_count(series, splits):
    """`by_class` is the same exposure as the total. A reader differencing the
    Class A rows by hand gets +296% out of a 4-for-1 unless the rows say so."""
    series(_two_class_series())
    splits([("2025-09-10", 4.0)])

    by_class = dilution.get_share_count_series("TWOCLASS")["by_class"]
    older = by_class["Class A"][-1]

    assert older["shares"] == 1_000_000_000, "the as-filed count stays"
    assert older["split_factor"] == pytest.approx(4.0)
    assert older["shares_split_adjusted"] == pytest.approx(4_000_000_000)


def test_each_class_reports_its_own_adjusted_direction(series, splits):
    """Classes diverge -- buybacks usually run in one class only -- so a single
    headline direction cannot stand in for all of them."""
    series(_two_class_series())
    splits([("2025-09-10", 4.0)])

    changes = dilution.get_share_count_series("TWOCLASS")["by_class_change"]

    assert changes["Class A"]["change_pct"] == pytest.approx(-1.0, abs=0.01)
    assert changes["Class A"]["direction"] == "buyback"
    assert changes["Class B"]["change_pct"] == pytest.approx(0.0, abs=0.01)
    assert changes["Class B"]["direction"] == "flat"


def test_the_class_rows_still_sum_to_the_reported_total(series, splits):
    """The invariant test_dilution_dedup.py pinned, re-pinned against the new
    per-class fields: the breakdown must not drift from the headline."""
    series(_two_class_series())
    splits([("2025-09-10", 4.0)])

    result = dilution.get_share_count_series("TWOCLASS")
    newest = max(r["period"] for rows in result["by_class"].values() for r in rows)
    latest = sum(r["shares"] for rows in result["by_class"].values()
                 for r in rows if r["period"] == newest)

    assert latest == result["latest_total"]


def test_a_class_direction_is_refused_when_the_total_is(series, splits):
    """The classes cannot be more certain than the series they came from."""
    series(_two_class_series())
    splits([])

    changes = dilution.get_share_count_series("TWOCLASS")["by_class_change"]

    assert changes["Class A"]["direction"] == "split_suspected_undetermined"
    assert changes["Class A"]["change_pct"] is None


# ------------------------------------------------------------------- boundaries

def test_a_split_dated_on_the_cover_date_is_already_reflected(series, splits):
    """A forward split's shares are distributed before the ex-date opens, so a
    cover page dated the day of the split already counts the new shares.
    Applying the ratio to that cover as well would invent a tenfold error in
    the other direction -- and the series here is built so it would show:
    rebasing the middle observation leaves two discontinuities where there
    were none."""
    series([_point("2026-05-01", "2026-04-25", 1_000_000_000),
            _point("2025-09-15", "2025-09-10", 1_000_000_000),
            _point("2025-05-15", "2025-05-10", 100_000_000)])
    splits([("2025-09-10", 10.0)])

    result = dilution.get_share_count_series("BOUNDARY")

    assert result["split_adjustment"]["splits_applied"] == [
        {"date": "2025-09-10", "ratio": 10.0}]
    assert result["change_pct"] == pytest.approx(0.0)
    assert result["direction"] == "flat"


def test_a_single_filing_states_no_direction_and_asks_nothing_of_the_calendar(
        series, calendar_calls):
    """One observation supports no change at all, so there is nothing to adjust
    and no reason to spend a network call finding out."""
    series([_point("2026-05-01", "2026-04-25", 1_000_000_000)])

    result = dilution.get_share_count_series("NEWIPO")

    assert calendar_calls == []
    assert result["direction"] == "insufficient_history"
    assert result["change_pct"] is None
    assert result["raw_change_pct"] is None


def test_a_failed_series_carries_the_new_fields_too(monkeypatch):
    """A caller reading `split_adjusted` must not get a KeyError on the error
    path; a shape that changes between success and failure is its own bug."""
    def raise_not_covered(*a, **k):
        from tools.web_search_server.sec_series import NotCovered
        raise NotCovered("no such concept")
    monkeypatch.setattr(dilution, "fetch_concept_series", raise_not_covered)

    result = dilution.get_share_count_series("NOTAGS")

    assert result["success"] is False
    assert result["split_adjusted"] is False
    assert result["raw_change_pct"] is None


# ---------------------------------------------------------------- the detector

def test_the_detector_knows_a_round_ratio_from_a_capital_raise():
    """The line between the two failure modes, stated directly."""
    assert dilution._round_split_ratio(10.0, strict=True) == pytest.approx(10.0)
    assert dilution._round_split_ratio(0.1, strict=True) == pytest.approx(0.1)
    assert dilution._round_split_ratio(1.4, strict=True) is None
    assert dilution._round_split_ratio(1.4, strict=False) is None
    assert dilution._round_split_ratio(1.08, strict=False) is None, (
        "an 8% quarter of option exercises is not a split")


def test_the_detector_tolerates_the_drift_a_real_quarter_adds():
    """The count either side of a split also moves for buybacks and option
    exercises, so the implied ratio is never exactly 10.000."""
    assert dilution._round_split_ratio(9.97, strict=True) == pytest.approx(10.0)
    assert dilution._round_split_ratio(50.3, strict=True) == pytest.approx(50.0)


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_nvda_reads_as_a_buyback_against_live_filings():
    """The end-to-end check, and the one that has to agree with
    get_buyback_history ($40.1bn repurchased) and get_corporate_actions (10:1)."""
    result = dilution.get_share_count_series("NVDA", limit=8)

    assert result["success"] is True
    assert result["direction"] == "buyback"
    assert -6.0 < result["change_pct"] < 0.0
    assert result["split_adjustment"]["splits_applied"], (
        "the June 2024 10-for-1 falls inside eight NVDA 10-Qs and must be applied")


@network
def test_msft_is_untouched_against_live_filings():
    """A filer with no split in the window must come back exactly as before."""
    result = dilution.get_share_count_series("MSFT", limit=4)

    assert result["success"] is True
    assert result["split_adjusted"] is False
    assert result["change_pct"] == pytest.approx(result["raw_change_pct"])


@network
def test_googl_classes_are_intact_and_unadjusted():
    """The multi-class path with no split to apply: three classes, no rebasing,
    and the per-class breakdown still present."""
    result = dilution.get_share_count_series("GOOGL", limit=4)

    assert len(result["classes_found"]) == 3
    assert result["split_adjusted"] is False
    assert set(result["by_class_change"]) == set(result["classes_found"])


# --- a calendar that fails quietly rather than loudly -----------------------
#
# `fetch_split_history` documents that it "raises rather than returning an
# empty list when the source cannot be read", because "no splits" and "could
# not find out" lead to opposite conclusions about the same series. The tests
# above prove the behaviour for a source that RAISES.
#
# yfinance does not raise. Asked for a symbol it cannot resolve it prints
# `HTTP Error 404` to stderr and hands back an empty series -- indistinguishable
# from Berkshire Hathaway, which has genuinely never split. The promise was
# therefore unenforced on the only failure the source actually produces, and a
# Yahoo outage would have relabelled every split as dilution while the response
# said `split_adjusted` with an empty calendar and no `source_error`.


def test_a_symbol_that_did_not_resolve_is_not_a_symbol_without_splits():
    """The 404 case, which is what a Yahoo outage looks like from here."""
    from tools.web_search_server import dilution

    class _Unresolved:
        splits = []
        history_metadata = {}          # populated only when Yahoo answered

    import pytest
    with pytest.raises(Exception) as caught:
        dilution._splits_from_yfinance_ticker(_Unresolved(), "ZZZZ")
    assert "ZZZZ" in str(caught.value)


def test_a_filer_that_genuinely_never_split_returns_an_empty_calendar():
    """Berkshire has never split its A shares. Empty here is the true answer,
    and must not be mistaken for the failure above."""
    from tools.web_search_server import dilution

    class _Resolved:
        splits = []
        history_metadata = {"symbol": "BRK-A", "currency": "USD"}

    assert dilution._splits_from_yfinance_ticker(_Resolved(), "BRK-A") == []


@pytest.mark.network
def test_the_two_cases_are_distinguished_against_live_yahoo():
    from tools.web_search_server.dilution import fetch_split_history

    assert fetch_split_history("BRK-A") == [], "BRK-A has never split"

    with pytest.raises(Exception):
        fetch_split_history("ZZZZNOTAREALTICKER")
