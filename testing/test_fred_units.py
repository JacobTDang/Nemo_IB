"""A change is reported in the unit the series is actually measured in.

Two unit defects, both the same class as the calculators' rate coercion: a
conversion applied without asking what the number is.

`get_credit_spreads` labelled the raw FRED value `current_bps`. FRED quotes
BAMLH0A0HYM2 in percent, so high-yield OAS came back as **2.69** under a name
promising basis points. It is 269bp. Anything pricing debt off that field is
out by 100x. In the same object `3m_change_bps` genuinely was basis points, so
one response used two units under one suffix.

`get_macro_snapshot` multiplied every change by 100 and called it `_bps`,
regardless of what the series measures:

    PAYEMS   158,858 vs 158,798 thousand jobs -> "3m_change_bps": 6000.0
    ICSA     206,000 vs 198,000 claims        -> "3m_change_bps": 800000.0
    UMCSENT  49.5 vs 53.3 index points        -> "3m_change_bps": -380.0

A 60,000-job change is not 6,000 basis points of anything.

A basis point is one hundredth of a percentage point, so the conversion is
only meaningful for a series quoted in percent. Everything else reports its
change in its own units.
"""
import pytest

from tools.news_agregator.fred_server import (
    SERIES_UNITS, is_percent_series, change_field_name)


class TestSeriesUnits:
    @pytest.mark.parametrize("series", ["DGS10", "DGS2", "FEDFUNDS", "T10Y2Y",
                                        "UNRATE", "BAMLH0A0HYM2",
                                        "BAMLC0A4CBBB", "A191RL1Q225SBEA"])
    def test_rate_series_are_percent(self, series):
        assert is_percent_series(series) is True, (
            f"{series} is quoted in percent; a basis-point change is meaningful")

    @pytest.mark.parametrize("series", ["PAYEMS", "ICSA", "UMCSENT", "NFCI",
                                        "CPIAUCSL", "PCEPILFE"])
    def test_level_and_index_series_are_not(self, series):
        assert is_percent_series(series) is False, (
            f"{series} is not a percentage, so a basis-point change of it is "
            f"a number with no meaning")

    def test_every_snapshot_series_declares_a_unit(self):
        """An undeclared series would silently fall back to some default,
        which is how the wrong conversion got applied to all of them."""
        from tools.news_agregator.fred_server import SNAPSHOT_SERIES
        undeclared = [s for s in SNAPSHOT_SERIES if s not in SERIES_UNITS]
        assert not undeclared, f"no unit declared for {undeclared}"


class TestChangeNaming:
    def test_a_percent_series_reports_basis_points(self):
        assert change_field_name("DGS10", "3m") == "3m_change_bps"

    @pytest.mark.parametrize("series", ["PAYEMS", "ICSA", "UMCSENT"])
    def test_a_non_percent_series_does_not_claim_basis_points(self, series):
        name = change_field_name(series, "3m")
        assert not name.endswith("_bps"), (
            f"{series} reports its change as {name!r}; a basis point is a "
            f"hundredth of a percentage point and this series is not a "
            f"percentage")
        assert name == "3m_change"


@pytest.mark.network
class TestAgainstFred:
    """The rule end to end. Needs FRED_API_KEY."""

    def test_credit_spreads_report_genuine_basis_points(self):
        import asyncio, json
        from tools.news_agregator.fred_server import FredServer
        server = FredServer()
        try:
            body = json.loads(asyncio.run(server.get_credit_spreads())[0].text)
        finally:
            asyncio.run(server.client.close())
        hy = body["data"]["BAMLH0A0HYM2"]
        assert hy["current_bps"] > 50, (
            f"HY OAS reported as {hy['current_bps']}bp. FRED quotes this series "
            f"in percent; a high-yield spread below 50bp has never happened.")

    def test_a_payrolls_change_is_not_called_basis_points(self):
        import asyncio, json
        from tools.news_agregator.fred_server import FredServer
        server = FredServer()
        try:
            body = json.loads(asyncio.run(server.get_macro_snapshot())[0].text)
        finally:
            asyncio.run(server.client.close())
        payems = body["data"]["PAYEMS"]
        assert not any(k.endswith("_bps") for k in payems), (
            f"PAYEMS still reports {[k for k in payems if k.endswith('_bps')]}")
