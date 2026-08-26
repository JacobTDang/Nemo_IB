"""A warning that always fires is a warning nobody reads.

`get_debt_maturity_schedule("MSFT")` returns `coverage: "full"`,
`buckets_found: 6`, all six values verified against SEC — and carried
`incomplete_coverage: "Coverage is materially incomplete."` anyway, because
the caveat was attached to the tool rather than to the answer.

It contradicts `coverage: "full"` in the same object. Worse, it trains a
reader to skip the warnings array, so the genuinely incomplete response the
caveat exists for no longer stands out. An unconditional warning is not a
conservative choice; it is the destruction of the signal.

The same applies to the macro snapshot's as-of dates: the components are
correct individually and are drawn from different observation dates, so the
spread shown is not the difference of the two yields shown.
"""
import pytest


@pytest.mark.network
class TestDebtMaturity:
    def test_a_complete_schedule_carries_no_incompleteness_warning(self):
        from tools.web_search_server.debt_maturity import get_debt_maturity_schedule
        result = get_debt_maturity_schedule("MSFT")
        if result.get("coverage") != "full":
            pytest.skip("MSFT's coverage is not full in this run")

        codes = [w.get("code") for w in (result.get("warnings") or [])]
        assert "incomplete_coverage" not in codes, (
            f"coverage is 'full' with all buckets tagged, and the response "
            f"still warns that coverage is materially incomplete: {codes}")

    def test_an_incomplete_schedule_still_warns(self):
        """Removing the noise must not remove the signal."""
        from tools.web_search_server.debt_maturity import get_debt_maturity_schedule
        result = get_debt_maturity_schedule("NVDA")
        if result.get("coverage") == "full":
            pytest.skip("NVDA's coverage is full in this run")

        codes = [w.get("code") for w in (result.get("warnings") or [])]
        assert "incomplete_coverage" in codes, (
            f"coverage is {result.get('coverage')!r} and nothing warns: {codes}")

    def test_the_warning_and_the_coverage_field_never_contradict(self):
        from tools.web_search_server.debt_maturity import get_debt_maturity_schedule
        for ticker in ("MSFT", "AAPL", "NVDA", "JPM"):
            result = get_debt_maturity_schedule(ticker)
            codes = [w.get("code") for w in (result.get("warnings") or [])]
            if result.get("coverage") == "full":
                assert "incomplete_coverage" not in codes, ticker


@pytest.mark.network
class TestMacroAsOf:
    def test_the_snapshot_says_whether_its_series_share_a_date(self):
        """T10Y2Y (0.47, as of 08-25) sat beside DGS10 and DGS2 (as of 08-24)
        whose difference is 0.46. One basis point, but the response presented
        a spread that is not the difference of the two yields it showed, with
        no single as-of to warn a reader that they are different days."""
        import asyncio, json
        from tools.news_agregator.fred_server import FredServer
        server = FredServer()
        try:
            body = json.loads(asyncio.run(server.get_macro_snapshot())[0].text)
        finally:
            asyncio.run(server.client.close())

        assert body.get("as_of_range") or body.get("data_as_of"), (
            "the snapshot gives every series its own as_of and no summary, so "
            "nothing tells a reader the components are from different days")

    def test_a_mixed_snapshot_is_flagged(self):
        import asyncio, json
        from tools.news_agregator.fred_server import FredServer
        server = FredServer()
        try:
            body = json.loads(asyncio.run(server.get_macro_snapshot())[0].text)
        finally:
            asyncio.run(server.client.close())

        dates = {v.get("as_of") for v in body["data"].values() if v.get("as_of")}
        if len(dates) <= 1:
            pytest.skip("every series shares an as_of in this run")
        assert body.get("as_of_mixed") is True, (
            f"{len(dates)} distinct observation dates in one snapshot and "
            f"nothing says so")
