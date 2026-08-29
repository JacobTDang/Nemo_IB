"""Relative volume, average dollar volume, and average true range.

The OHLCV bars behind these were already being fetched by get_price_history;
only the derived metrics were missing. Every number here is checkable by hand,
which is the point -- a position-sizing input nobody can verify is worse than
no input at all.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from unittest.mock import patch

import pandas as pd
import pytest

from tools.financial_modeling_engine import trading_metrics as tm

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live yfinance test")(func)


# --------------------------------------------------------------------- fakes

def _bars(rows, start=None):
    """rows: list of (open, high, low, close, volume). Index is business days.

    Anchored to the current year so that get_price_history's year-to-date
    branch sees the bars as this year's, whatever year the suite runs in.
    """
    start = start or f"{datetime.now(ZoneInfo('America/New_York')).year}-01-02"
    index = pd.bdate_range(start=start, periods=len(rows), tz="America/New_York")
    return pd.DataFrame(
        {
            "Open":   [r[0] for r in rows],
            "High":   [r[1] for r in rows],
            "Low":    [r[2] for r in rows],
            "Close":  [r[3] for r in rows],
            "Volume": [r[4] for r in rows],
        },
        index=index,
    )


def _flat(n, close=100.0, volume=1_000_000, half_range=0.5):
    return [(close, close + half_range, close - half_range, close, volume)
            for _ in range(n)]


def _install(monkeypatch, frame):
    monkeypatch.setattr(tm, "fetch_daily_bars", lambda *a, **k: frame)


# ----------------------------------------------------------------- true range

def test_atr_uses_wilder_smoothing_not_a_simple_mean(monkeypatch):
    """21 bars: the first 14 true ranges are 1.0, the last 6 are 10.0.

    Wilder's recurrence seeds on the first 14 and converges toward 10 from
    below: 10 - 9*(13/14)^6 = 4.2306. A simple 14-bar mean of the same series
    gives (8*1 + 6*10)/14 = 4.857. The two differ enough that this test tells
    you which one shipped.
    """
    rows = _flat(15, half_range=0.5) + _flat(6, half_range=5.0)
    _install(monkeypatch, _bars(rows))

    result = tm.get_trading_metrics("FAKE", atr_period=14)

    assert result["success"] is True
    assert result["atr"]["method"] == "wilder"
    assert result["atr"]["atr"] == pytest.approx(4.2306, abs=0.001)
    assert result["atr"]["atr_pct_of_price"] == pytest.approx(4.2306, abs=0.001)


def test_true_range_accounts_for_gaps_beyond_the_bar_high_low(monkeypatch):
    """A gap up leaves high-low understating the real move.

    Bar 2 opens and trades entirely above bar 1's close: high 120, low 118,
    prior close 100. high-low is 2, but the true range is 20. A tool that used
    high-low would report a stock as eight times calmer than it is on gap days.
    """
    rows = [(100, 100.5, 99.5, 100, 1_000_000),
            (119, 120.0, 118.0, 119, 1_000_000)]
    _install(monkeypatch, _bars(rows))

    result = tm.get_trading_metrics("GAPPY", atr_period=1)

    assert result["atr"]["atr"] == pytest.approx(20.0)


def test_atr_is_none_with_a_reason_when_there_are_too_few_bars(monkeypatch):
    _install(monkeypatch, _bars(_flat(5)))

    result = tm.get_trading_metrics("SHORT", atr_period=14)

    assert result["success"] is True
    assert result["atr"]["atr"] is None
    assert result["atr"]["atr_pct_of_price"] is None
    assert "14" in result["atr"]["note"] and "5" in result["atr"]["note"]


# ---------------------------------------------------------------------- rvol

def test_rvol_compares_the_latest_session_against_its_trailing_average(monkeypatch):
    rows = _flat(20, volume=1_000_000) + _flat(1, volume=3_000_000)
    _install(monkeypatch, _bars(rows))

    result = tm.get_trading_metrics("SPIKE", rvol_lookback=20)

    assert result["rvol"]["latest_volume"] == 3_000_000
    assert result["rvol"]["average_volume"] == pytest.approx(1_000_000)
    assert result["rvol"]["ratio"] == pytest.approx(3.0)


def test_rvol_baseline_excludes_the_session_being_measured(monkeypatch):
    """Folding today into its own baseline drags the ratio toward 1.

    Twenty quiet days then one 20x day: the honest ratio is 20.0. Including
    the spike in its own 21-bar average gives 20 / 1.95 = 10.3 -- a doubled
    volume day would read as roughly average.
    """
    rows = _flat(20, volume=1_000_000) + _flat(1, volume=20_000_000)
    _install(monkeypatch, _bars(rows))

    result = tm.get_trading_metrics("SPIKE", rvol_lookback=20)

    assert result["rvol"]["ratio"] == pytest.approx(20.0)


def test_rvol_is_none_with_a_reason_when_the_baseline_is_short(monkeypatch):
    _install(monkeypatch, _bars(_flat(4)))

    result = tm.get_trading_metrics("SHORT", rvol_lookback=20)

    assert result["rvol"]["ratio"] is None
    assert "20" in result["rvol"]["note"]


def test_a_zero_volume_baseline_reports_none_rather_than_dividing(monkeypatch):
    rows = _flat(20, volume=0) + _flat(1, volume=500)
    _install(monkeypatch, _bars(rows))

    result = tm.get_trading_metrics("HALTED", rvol_lookback=20)

    assert result["rvol"]["ratio"] is None
    assert result["rvol"]["average_volume"] == 0


# ----------------------------------------------------------------------- adv

def test_adv_is_dollars_not_shares(monkeypatch):
    """$100 stock trading a million shares is $100m a day, not 1m of anything.

    The share count is reported alongside because the two get confused, but
    the dollar figure is the one that decides whether a position can be exited.
    """
    _install(monkeypatch, _bars(_flat(70, close=100.0, volume=1_000_000)))

    result = tm.get_trading_metrics("LIQUID")

    assert result["adv"]["20d"]["dollar_volume"] == pytest.approx(100_000_000)
    assert result["adv"]["20d"]["share_volume"] == pytest.approx(1_000_000)
    assert result["adv"]["60d"]["dollar_volume"] == pytest.approx(100_000_000)


def test_adv_averages_the_window_rather_than_summing_it(monkeypatch):
    """A sum reads as a plausible ADV for a large cap and is 20x too big."""
    _install(monkeypatch, _bars(_flat(30, close=50.0, volume=2_000_000)))

    result = tm.get_trading_metrics("MID")

    assert result["adv"]["20d"]["dollar_volume"] == pytest.approx(100_000_000)


def test_adv_prices_each_session_at_its_own_close(monkeypatch):
    """Ten sessions at $10 and ten at $110 average $600m, not $1.2bn.

    Using the latest close against the whole window would double it here.
    """
    rows = _flat(10, close=10.0, volume=10_000_000) + \
        _flat(10, close=110.0, volume=10_000_000)
    _install(monkeypatch, _bars(rows))

    result = tm.get_trading_metrics("REPRICED")

    assert result["adv"]["20d"]["dollar_volume"] == pytest.approx(600_000_000)


def test_a_window_longer_than_the_history_reports_none_not_a_short_average(monkeypatch):
    _install(monkeypatch, _bars(_flat(25)))

    result = tm.get_trading_metrics("SHORT")

    assert result["adv"]["20d"]["dollar_volume"] is not None
    assert result["adv"]["60d"]["dollar_volume"] is None
    assert "60" in result["adv"]["note"]


# ------------------------------------------------------- partial live session

def _freeze(monkeypatch, when):
    monkeypatch.setattr(tm, "_now_eastern", lambda: when)


def test_an_in_progress_session_is_flagged_because_rvol_understates_it(monkeypatch):
    """yfinance hands back a bar for the day in progress.

    Half a session's volume against a full-session baseline halves RVOL. The
    number is still the honest live read, so it is returned -- but a caller
    told nothing would size a position off it.
    """
    rows = _flat(20, volume=1_000_000) + _flat(1, volume=400_000)
    frame = _bars(rows)
    _install(monkeypatch, frame)
    _freeze(monkeypatch, frame.index[-1].to_pydatetime().replace(hour=11, minute=30))

    result = tm.get_trading_metrics("MIDDAY")

    assert result["latest_bar_is_partial_session"] is True
    assert result["rvol"]["ratio"] == pytest.approx(0.4)
    assert "partial" in result["rvol"]["note"].lower()


def test_a_partial_session_is_kept_out_of_atr_and_adv(monkeypatch):
    """Those two describe completed sessions; a half day belongs in neither.

    Twenty $100 sessions on a million shares, then a partial day printing
    100k shares. ADV stays $100m. Fold the stub in and it reads $95.5m.
    """
    rows = _flat(20, close=100.0, volume=1_000_000) + \
        _flat(1, close=100.0, volume=100_000)
    frame = _bars(rows)
    _install(monkeypatch, frame)
    _freeze(monkeypatch, frame.index[-1].to_pydatetime().replace(hour=10, minute=0))

    result = tm.get_trading_metrics("MIDDAY")

    assert result["adv"]["20d"]["dollar_volume"] == pytest.approx(100_000_000)
    assert result["adv"]["excluded_partial_session"] is True


def test_a_closed_session_is_not_flagged_as_partial(monkeypatch):
    rows = _flat(21)
    frame = _bars(rows)
    _install(monkeypatch, frame)
    _freeze(monkeypatch, frame.index[-1].to_pydatetime().replace(hour=17, minute=0))

    result = tm.get_trading_metrics("CLOSED")

    assert result["latest_bar_is_partial_session"] is False


# ------------------------------------------------------------------- failures

def test_an_empty_frame_is_a_failure_not_a_row_of_zeros(monkeypatch):
    _install(monkeypatch, pd.DataFrame())

    result = tm.get_trading_metrics("DELISTED")

    assert result["success"] is False
    assert "no price history" in result["error"].lower()


def test_a_fetch_error_is_reported_not_swallowed(monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("yfinance unavailable")
    monkeypatch.setattr(tm, "fetch_daily_bars", explode)

    result = tm.get_trading_metrics("BOOM")

    assert result["success"] is False
    assert "yfinance unavailable" in result["error"]


def test_a_missing_volume_column_is_reported_rather_than_guessed(monkeypatch):
    frame = _bars(_flat(30)).drop(columns=["Volume"])
    _install(monkeypatch, frame)

    result = tm.get_trading_metrics("NOVOL")

    assert result["success"] is False
    assert "volume" in result["error"].lower()


# ------------------------------------------------------------- one fetch path

def test_price_history_and_trading_metrics_share_one_bar_fetch(monkeypatch):
    """Two yfinance calls with different periods would silently disagree.

    Both tools must go through utils.fetch_daily_bars so that a fix to the
    fetch -- retries, adjustment policy, a cache -- lands in both.
    """
    from tools.financial_modeling_engine import utils

    calls = []

    def spy(ticker, period="2y"):
        calls.append((ticker, period))
        return _bars(_flat(30))

    monkeypatch.setattr(utils, "fetch_daily_bars", spy)
    utils.get_price_history("FAKE", period="6mo", include_recent_bars=5)

    assert calls == [("FAKE", "6mo")]


# ------------------------------------------------------------- live golden set

@network
def test_spy_average_dollar_volume_is_in_the_tens_of_billions():
    """SPY is the most heavily traded equity instrument in the world.

    Its ADV has not printed below $10bn in years. A units error -- shares
    instead of dollars, or a sum instead of a mean -- lands far outside this.
    """
    result = tm.get_trading_metrics("SPY")

    assert result["success"] is True, result.get("error")
    adv = result["adv"]["20d"]["dollar_volume"]
    assert 1e10 < adv < 1e12, f"SPY 20d ADV of ${adv:,.0f} is not credible"


@network
def test_a_mega_cap_and_a_micro_cap_are_orders_of_magnitude_apart():
    """The whole point of ADV is telling a $50m position it cannot exit.

    AAPL trades billions a day; a small biotech trades millions. If the tool
    cannot separate those two it cannot size anything.
    """
    mega = tm.get_trading_metrics("AAPL")["adv"]["20d"]["dollar_volume"]
    micro = tm.get_trading_metrics("BOOM")["adv"]["20d"]["dollar_volume"]

    assert mega > 1e9, f"AAPL ADV ${mega:,.0f}"
    assert micro is not None and micro < mega / 100, f"BOOM ADV ${micro:,.0f}"


def _metrics_or_skip(ticker: str) -> dict:
    """Fetch, or skip naming the vendor's own reason.

    These are @network tests, so a transient vendor failure is not a finding
    about the calculation. Reading straight through a refusal is how a clean
    run failed with `KeyError: 'atr'` and pointed at the ATR band rather than
    at the fetch that never happened.
    """
    result = tm.get_trading_metrics(ticker)
    if not result.get("success"):
        pytest.skip(f"{ticker}: the vendor did not answer -- "
                    f"{result.get('error')}")
    return result


@network
def test_share_volume_agrees_with_yfinances_own_three_month_average():
    """An independent check on the volume path.

    yfinance reports averageVolume as a ~3-month mean, so our 60-day figure
    should sit in the same neighbourhood. A factor-of-20 error (summing) or a
    split-adjustment mismatch shows up immediately; normal drift does not.
    """
    import yfinance as yf

    ours = _metrics_or_skip("MSFT")["adv"]["60d"]["share_volume"]
    theirs = yf.Ticker("MSFT").info.get("averageVolume")

    assert theirs, "yfinance did not return averageVolume; cannot cross-check"
    assert 0.5 < ours / theirs < 2.0, f"ours {ours:,.0f} vs yfinance {theirs:,.0f}"


@network
def test_atr_percent_lands_in_a_plausible_band_for_a_mega_cap():
    """A mega cap does not have a 0.05% or a 40% average daily range.

    Recomputing ATR here would just restate the implementation, so this
    asserts the band that any correct implementation must land inside and
    that an off-by-100 scaling error cannot.
    """
    result = _metrics_or_skip("MSFT")

    atr_pct = result["atr"]["atr_pct_of_price"]
    assert 0.3 < atr_pct < 10.0, f"MSFT ATR is {atr_pct}% of price"
    assert result["atr"]["atr"] < result["last_close"]


@network
def test_rvol_of_a_liquid_name_sits_near_one_on_an_ordinary_day():
    """RVOL is a ratio against the stock's own baseline, so it is O(1).

    A value of 250,000 would mean shares were compared against dollars.
    """
    result = tm.get_trading_metrics("AAPL")

    assert result["rvol"]["ratio"] is not None
    assert 0.05 < result["rvol"]["ratio"] < 20.0, result["rvol"]


# ------------------------------------------------------- refusal shape
#
# A clean full-suite run failed these two tests with `KeyError: 'atr'` rather
# than with a value disagreement, while both passed in isolation -- so the
# trigger is a transient vendor failure, and what the caller met was a refusal
# missing the keys the success path documents.

_REFUSAL_PATHS = (
    ("fetch raised", lambda: patch.object(
        tm, "fetch_daily_bars", side_effect=RuntimeError("vendor throttled"))),
    ("empty frame", lambda: patch.object(
        tm, "fetch_daily_bars", return_value=pd.DataFrame())),
)


@pytest.mark.parametrize("label,make_patch", _REFUSAL_PATHS,
                         ids=[p[0] for p in _REFUSAL_PATHS])
def test_a_refusal_still_carries_the_documented_keys(label, make_patch):
    """`result["atr"]` must not raise on a fetch that never happened.

    The success path returns `atr`; every transport refusal returned only
    ticker/success/error. A caller reaching for it got a KeyError instead of
    the refusal it was handed, and one writing `.get("atr", {})` read "no ATR
    for this name" out of a vendor timeout.

    `None` is not a claim about the security -- an `atr_pct_of_price` of 0.0
    would be. The module already returns `"atr": None` for the
    insufficient-data case; these paths just did not use it.
    """
    with make_patch():
        result = tm.get_trading_metrics("MSFT")

    assert result["success"] is False, "this fixture should produce a refusal"
    assert result.get("error"), "a refusal must say what went wrong"
    for key in ("atr", "rvol", "adv"):
        assert key in result, (
            f"{label}: the refusal drops {key!r}, which the success path "
            f"documents; a caller reading it gets a KeyError")
        assert result[key] is None, (
            f"{label}: {key!r} carries a value on a fetch that never happened")
