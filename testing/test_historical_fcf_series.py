"""get_historical_fcf should be historical.

It reads one filing and returns one period, so anything trusting the name gets
a point estimate where it expects a trend. That is the same class of mistake as
`equity-deep-research/SKILL.md` reaching for it to judge "FCF vs reported
earnings divergence" -- a divergence is a shape over time, and one period
cannot show one.

The existing single-period fields stay exactly as they are. Consumers read
them, and a series is additive.
"""
import os

import pytest

from tools.web_search_server.sec_utils import get_historical_fcf

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_a_series_is_returned():
    result = get_historical_fcf("MSFT")
    assert result["success"] is True
    assert len(result["series"]) > 1, (
        "a tool called historical returned a single period")


@network
def test_the_latest_series_entry_matches_the_top_level_fields():
    """The existing shape is the newest period, so consumers reading the flat
    fields keep getting what they got."""
    result = get_historical_fcf("MSFT")
    latest = result["series"][0]
    assert latest["free_cash_flow"] == result["free_cash_flow"]
    assert latest["operating_cash_flow"] == result["operating_cash_flow"]
    assert latest["period_end"] == result["period_end"]


@network
def test_periods_run_newest_first_and_do_not_repeat():
    result = get_historical_fcf("MSFT")
    ends = [row["period_end"] for row in result["series"]]
    assert ends == sorted(ends, reverse=True)
    assert len(ends) == len(set(ends)), f"a period repeats: {ends}"


@network
def test_a_filer_tagging_no_capex_still_reports_its_cash_flow():
    """Banks tag no capex element. The series must not invent a zero -- that
    is the bug that made Amazon's FCF read 18x its real value."""
    result = get_historical_fcf("JPM")
    assert result.get("coverage") == "not_covered"
    assert result.get("free_cash_flow") is None
    assert result.get("operating_cash_flow") is not None
