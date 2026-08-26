"""A "52-week" low must come from 52 weeks, or not be reported.

`get_price_history` computed its 52-week stats as `df.iloc[-252:] if len(df)
>= 252 else df`. The `else df` silently substitutes whatever the request
fetched and keeps the field name. Same for `max_drawdown_12m`, which rides on
the same frame, and for YTD, which anchors on the first bar IN THE FRAME on or
after 1 January rather than the first trading day of the year.

Measured live on AMD, same ticker, same second:

    period="6mo"   fifty_two_week.low  188.22 (2026-03-03)   ytd 140.21%
    period="2y"    fifty_two_week.low  149.22 (2025-09-08)   ytd 115.21%

The 2y values are right -- `get_basic_financials` independently reports a
52-week low of 149.22 on 2025-09-08. So a six-month request reported a
"52-week low" 26% too high and a YTD off by 25 points, with no warning.

The correct behaviour already exists in the same function: `returns_pct.6m`,
`returns_pct.1y`, `realized_vol.180d` and `realized_vol.1y` all go null when
the window cannot support them. These three fields were simply never given the
same treatment.

52 weeks is checked as a date span rather than a 252-row count, because
`period="1y"` returns 251 bars and a row test would null the field on the most
natural request for it.
"""
import pytest


@pytest.mark.network
@pytest.mark.parametrize("period", ["3mo", "6mo"])
def test_a_short_window_does_not_report_a_52_week_low(period):
    from tools.financial_modeling_engine.utils import get_price_history

    result = get_price_history("AMD", period=period, include_recent_bars=1)
    assert result.get("fifty_two_week") is None, (
        f"period={period} reported {result.get('fifty_two_week')} as a "
        f"52-week range")
    assert result.get("max_drawdown_12m") is None


@pytest.mark.network
@pytest.mark.parametrize("period", ["1y", "2y"])
def test_a_sufficient_window_still_reports_it(period):
    from tools.financial_modeling_engine.utils import get_price_history

    result = get_price_history("AMD", period=period, include_recent_bars=1)
    window = result.get("fifty_two_week")
    assert window, f"period={period} should support a 52-week range"
    assert window["low"] == pytest.approx(149.22, abs=1.0), (
        f"the true 52-week low is 149.22 on 2025-09-08; got {window}")


@pytest.mark.network
def test_the_short_window_says_which_fields_it_dropped():
    from tools.financial_modeling_engine.utils import get_price_history

    result = get_price_history("AMD", period="6mo", include_recent_bars=1)
    codes = [w.get("code") for w in (result.get("warnings") or [])]
    assert "window_too_short_for_field" in codes, (
        f"fields vanished with no explanation: {codes}")


@pytest.mark.network
def test_ytd_is_measured_from_the_first_trading_day_of_the_year():
    """On a 6-month frame the first bar on/after 1 January is late February,
    so 'YTD' measured five months and reported 140% against a true 115%."""
    from tools.financial_modeling_engine.utils import get_price_history

    short = get_price_history("AMD", period="6mo", include_recent_bars=1)
    full = get_price_history("AMD", period="2y", include_recent_bars=1)

    if short["returns_pct"].get("ytd") is not None:
        assert short["returns_pct"]["ytd"] == pytest.approx(
            full["returns_pct"]["ytd"], abs=1.0), (
            "YTD moved with the requested window")
