"""Share counts on different bases, sitting side by side with nothing to say so.

Every defect here produces an error that is an exact multiple, never a small
discrepancy, and every one of them is a number a caller would reasonably
compute from two fields of the same response.

1. **`marketCap` and `sharesOutstanding` in one payload, on different bases.**

       GOOGL  currentPrice 342.00 x sharesOutstanding 5,867,155,790 = $2.007tn
              reported marketCap                                    = $4.183tn
              gap 52.03% (2.0845x)

       BRK-B  currentPrice 504.91 x sharesOutstanding 1,408,035,161 = $711bn
              reported marketCap                                    = $1,081bn
              gap 34.23% (1.5204x)

   `marketCap / currentPrice` for GOOGL implies 12,229,935,104 shares against
   an SEC all-class total of 12,230,000,000 -- a 0.0005% match. Yahoo's
   `sharesOutstanding` is Class A alone. `pe_ratio` and `pb_ratio` are built
   from `marketCap` and are correct; `sharesOutstanding` is the one field
   inconsistent with the rest of its own payload, and it is the field
   `calculate_dcf` divides an equity value by to get a price per share.

   The same field feeds `get_short_interest`, where GOOGL's `float_shares`
   (10.88bn) comes back larger than its `shares_outstanding` (5.87bn) -- a
   state that cannot exist, from which insider ownership computes to -85%.

2. **`get_share_count_series` adds share classes worth 1,500:1.**

   BRK's `latest_total` is 488,450 Class A plus 1,408,035,161 Class B, summed
   at 1:1. A Class A share converts into 1,500 Class B, so the economically
   equivalent total is 2,140,710,161 and the reported one is 34.20% short.

   The worst part is that the obvious cross-check passes. SEC `latest_total`
   1,408,523,611 and Yahoo `sharesOutstanding` 1,408,035,161 agree to 0.03% --
   two independent sources concurring, both wrong by a third, because one
   dropped Class A and the other under-weighted it. The analyst who runs the
   sanity check you would prescribe comes away *more* confident.

   GOOGL is fine: its A/B/C classes are 1:1. The conversion ratio is a
   per-filer fact, so it is established against the market rather than assumed,
   and where it cannot be established the tool says so instead of assuming 1:1.

3. **`get_price_history` never says its prices are split-adjusted.**

   `auto_adjust=True` back-adjusts every close for splits and dividends, and
   the response says so nowhere. Paired with `get_share_count_series`:

       NVDA 2024-05-24  close 106.29 x total 2,460,000,000        = $261bn
                        close 106.29 x total_split_adjusted 24.6bn = $2,615bn
       exactly 10.0x -- the split ratio.  AAPL 2020-08-28: exactly 4.0x.

   `total` is the unqualified, default-looking field and it pairs with nothing
   this toolset can produce. `get_corporate_actions.dividends` carries the same
   silent adjustment in the other direction: AAPL's 2020-08-07 dividend is
   reported 0.205 against an as-declared $0.82.

The rule all of these are held to: a caller cannot combine two numbers from
these responses and land on a multiple-of-N error without being told. Stating
the basis is enough where the two are reconcilable; where they are not, the
response names the gap, the multiple, and the field that does have a valid
counterpart.
"""
import os

import pandas as pd
import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
    """The live-provider marker plus the offline skip, as elsewhere in testing/."""
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live market-data test")(func)


def _codes(payload):
    return [w["code"] for w in payload.get("warnings") or []]


def _message(payload, code):
    for entry in payload.get("warnings") or []:
        if entry["code"] == code:
            return entry["message"]
    raise AssertionError(
        f"no {code!r} warning; got {_codes(payload)}")


# --------------------------------------------------------------------------
# 1. marketCap and sharesOutstanding
# --------------------------------------------------------------------------

class TestShareCountBasisArithmetic:
    """The reconciliation itself, on the figures measured live 2026-08-26."""

    def test_a_single_class_filer_reconciles_exactly(self):
        """NVDA, MSFT and AAPL agree to within a rounding error today, and the
        guard must leave them alone. A basis check that fires on the filers it
        has nothing to say about is a caveat attached to the tool rather than
        to the answer, and a reader learns to skip the array."""
        from tools.financial_modeling_engine.utils import share_count_basis

        basis = share_count_basis(market_cap=5_078_174_924_800,
                                  price=209.66,
                                  shares_outstanding=24_221_000_000,
                                  provider_implied=24_221_000_000)

        assert basis["basis"] == "matches_market_cap"
        assert abs(basis["gap_pct"]) < 0.01
        assert basis["shares_outstanding_all_classes"] == 24_221_000_000

    def test_a_market_cap_that_needs_more_shares_than_reported_is_flagged(self):
        """GOOGL: marketCap implies 12.23bn shares beside a sharesOutstanding
        of 5.87bn. Nothing in the response said the two were different things."""
        from tools.financial_modeling_engine.utils import share_count_basis

        basis = share_count_basis(market_cap=4_182_637_805_568,
                                  price=342.00,
                                  shares_outstanding=5_867_155_790,
                                  provider_implied=12_229_934_831)

        assert basis["basis"] == "narrower_than_market_cap", (
            f"GOOGL's sharesOutstanding covers one class and its marketCap "
            f"covers all of them; the basis reads {basis['basis']!r}")
        assert basis["gap_pct"] == pytest.approx(108.45, abs=0.5)
        assert basis["market_cap_implied_shares"] == pytest.approx(
            12_229_935_104, rel=1e-6)
        assert basis["shares_outstanding_all_classes"] == pytest.approx(
            12_229_934_831, rel=1e-6)

    def test_the_all_class_count_reproduces_the_market_cap_it_sits_beside(self):
        """The point of publishing it: it has a valid counterpart. Multiplying
        it by the price in the same payload returns the marketCap in the same
        payload, which is exactly what sharesOutstanding fails to do."""
        from tools.financial_modeling_engine.utils import share_count_basis

        basis = share_count_basis(market_cap=1_080_865_783_808,
                                  price=504.91,
                                  shares_outstanding=1_408_035_161,
                                  provider_implied=2_140_709_794)

        rebuilt = basis["shares_outstanding_all_classes"] * 504.91
        assert rebuilt == pytest.approx(1_080_865_783_808, rel=1e-6)

    def test_an_absent_market_cap_leaves_the_basis_unverified_not_matched(self):
        """"We could not check" and "we checked and they agree" are different
        claims. Defaulting the unknown one to agreement is how a basis error
        gets a clean bill of health from a check that never ran."""
        from tools.financial_modeling_engine.utils import share_count_basis

        for absent in ({"market_cap": None, "price": 100.0},
                       {"market_cap": 1e12, "price": None},
                       {"market_cap": 1e12, "price": 0.0}):
            basis = share_count_basis(shares_outstanding=1e9, **absent)
            assert basis["basis"] == "unverified", absent
            assert basis["gap_pct"] is None

    def test_the_warning_names_the_multiple_a_caller_would_be_wrong_by(self):
        """A percentage gap does not tell an analyst what their per-share
        number is wrong by. The multiple does, and it is the whole reason this
        class of defect matters: 2.0845x, not "roughly half"."""
        from tools.financial_modeling_engine.utils import (
            share_count_basis, share_basis_warning)

        basis = share_count_basis(market_cap=4_182_637_805_568, price=342.00,
                                  shares_outstanding=5_867_155_790,
                                  provider_implied=12_229_934_831)
        entry = share_basis_warning("GOOGL", basis)

        assert entry is not None
        assert "2.084" in entry["message"], entry["message"]
        assert "shares_outstanding_all_classes" in entry["message"]

    def test_no_warning_is_raised_for_a_filer_that_reconciles(self):
        from tools.financial_modeling_engine.utils import (
            share_count_basis, share_basis_warning)

        basis = share_count_basis(market_cap=3_685_818_040_320, price=496.37,
                                  shares_outstanding=7_425_545_491,
                                  provider_implied=7_425_545_491)
        assert share_basis_warning("MSFT", basis) is None


class TestMarketDataPayload:
    @network
    @pytest.mark.parametrize("ticker", ["NVDA", "AAPL", "MSFT"])
    def test_a_single_class_filer_is_untouched(self, ticker):
        """The regression guard. These three reconcile to 0.0000% today and
        must keep coming back clean, or the fix has traded one false reading
        for a different one."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)

        assert data["shares_outstanding_basis"] == "matches_market_cap"
        assert "share_count_basis_mismatch" not in _codes(data)
        assert data["shares_outstanding_all_classes"] == pytest.approx(
            data["sharesOutstanding"], rel=1e-3)

    @network
    @pytest.mark.parametrize("ticker", ["GOOGL", "BRK-B"])
    def test_a_multi_class_filer_says_the_two_fields_disagree(self, ticker):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)

        assert data["shares_outstanding_basis"] == "narrower_than_market_cap"
        message = _message(data, "share_count_basis_mismatch")
        assert "sharesOutstanding" in message and "marketCap" in message

    @network
    @pytest.mark.parametrize("ticker", ["GOOGL", "BRK-B"])
    def test_the_published_all_class_count_rebuilds_the_market_cap(self, ticker):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)

        rebuilt = data["shares_outstanding_all_classes"] * data["currentPrice"]
        assert rebuilt == pytest.approx(data["marketCap"], rel=0.01), (
            f"{ticker}: the count published for per-share arithmetic does not "
            f"reproduce the market cap in its own payload")

    @network
    @pytest.mark.parametrize("ticker", ["GOOGL", "BRK-B", "NVDA", "AAPL", "MSFT"])
    def test_the_multiples_built_on_market_cap_are_left_alone(self, ticker):
        """pe_ratio and pb_ratio divide marketCap, never sharesOutstanding, so
        they were never wrong and must not become so."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)

        if data.get("pe_ratio") is not None:
            assert data["pe_ratio"] == pytest.approx(
                data["marketCap"] / data["netIncomeToCommon"], rel=1e-9)

    @network
    def test_short_interest_carries_the_same_basis(self):
        """`get_short_interest` reads the same provider field, and publishes
        `float_shares` beside it. For GOOGL the float is larger than the shares
        outstanding, which is not a state a company can be in."""
        from tools.financial_modeling_engine.utils import get_short_interest
        data = get_short_interest("GOOGL")

        assert data["shares_outstanding_basis"] == "narrower_than_market_cap"
        message = _message(data, "float_exceeds_shares_outstanding")
        assert "float_shares" in message

    @network
    @pytest.mark.parametrize("ticker", ["NVDA", "AAPL", "MSFT"])
    def test_short_interest_is_quiet_for_a_single_class_filer(self, ticker):
        from tools.financial_modeling_engine.utils import get_short_interest
        data = get_short_interest(ticker)

        assert data["shares_outstanding_basis"] == "matches_market_cap"
        assert _codes(data) == []


# --------------------------------------------------------------------------
# 2. summing share classes worth 1,500:1
# --------------------------------------------------------------------------

CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"


def _point(filing_date, as_of, shares):
    from tools.web_search_server.sec_series import ConceptFact, FilingPoint
    from tools.web_search_server import dilution

    facts = [
        ConceptFact(value=float(value), period=as_of, context_ref=f"c-{i}",
                    concept=dilution.SHARES_CONCEPT, unit="shares",
                    dimensions={CLASS_AXIS: member} if member else {})
        for i, (value, member) in enumerate(shares)
    ]
    return FilingPoint(filing_date=filing_date, form="10-Q",
                       accession=f"acc-{filing_date}", facts=facts)


def _brk_series():
    """Berkshire's cover page as filed: 488,450 Class A and 1.408bn Class B."""
    return [
        _point("2026-08-02", "2026-07-28",
               [(488_450, "us-gaap:CommonClassAMember"),
                (1_408_035_161, "us-gaap:CommonClassBMember")]),
        _point("2026-05-03", "2026-04-28",
               [(490_100, "us-gaap:CommonClassAMember"),
                (1_412_000_000, "us-gaap:CommonClassBMember")]),
    ]


def _googl_series():
    """Alphabet's three classes, which do convert one for one."""
    return [
        _point("2026-07-23", "2026-07-15",
               [(5_868_000_000, "us-gaap:CommonClassAMember"),
                (835_000_000, "us-gaap:CommonClassBMember"),
                (5_527_000_000, "goog:CapitalClassCMember")]),
        _point("2026-04-23", "2026-04-15",
               [(5_880_000_000, "us-gaap:CommonClassAMember"),
                (846_000_000, "us-gaap:CommonClassBMember"),
                (5_540_000_000, "goog:CapitalClassCMember")]),
    ]


@pytest.fixture
def filings(monkeypatch):
    from tools.web_search_server import dilution

    def install(points):
        monkeypatch.setattr(dilution, "fetch_concept_series",
                            lambda *a, **k: list(points))
        monkeypatch.setattr(dilution, "fetch_split_history", lambda ticker: [])
    return install


@pytest.fixture
def market_total(monkeypatch):
    """The quoted-basis share count, without touching Yahoo."""
    from tools.web_search_server import dilution

    def install(total):
        if isinstance(total, Exception):
            def boom(ticker):
                raise total
            monkeypatch.setattr(dilution, "fetch_market_share_count", boom)
        else:
            monkeypatch.setattr(dilution, "fetch_market_share_count",
                                lambda ticker: total)
    return install


class TestMultiClassTotals:
    def test_the_unweighted_sum_is_never_presented_as_the_only_total(
            self, filings, market_total):
        """Berkshire's 34.20% understatement. `latest_total` stays as it was --
        callers depend on it -- but it stops being the only number in the
        response, and it stops being unlabelled."""
        from tools.web_search_server import dilution
        filings(_brk_series())
        market_total(2_140_709_794)

        result = dilution.get_share_count_series("BRK-B")

        assert result["latest_total"] == 1_408_523_611
        assert result["latest_total_basis"] == "sum_of_classes_unweighted"
        assert result["share_basis"]["economically_equivalent"] is False
        assert result["share_basis"]["quote_equivalent_total"] == pytest.approx(
            2_140_709_794, rel=1e-6)

    def test_the_conversion_ratio_is_derived_not_hardcoded(
            self, filings, market_total):
        """1,500 is a fact about Berkshire, not a constant. It falls out of the
        market's own count: (2,140,709,794 - 1,408,035,161) / 488,450 = 1500.0,
        and it is published only because it lands on a round multiple."""
        from tools.web_search_server import dilution
        filings(_brk_series())
        market_total(2_140_709_794)

        weights = dilution.get_share_count_series("BRK-B")["share_basis"][
            "implied_class_weights"]

        assert weights["Class A"] == pytest.approx(1500.0)
        assert weights["Class B"] == pytest.approx(1.0)

    def test_the_warning_names_the_understatement_and_the_correct_total(
            self, filings, market_total):
        from tools.web_search_server import dilution
        filings(_brk_series())
        market_total(2_140_709_794)

        message = _message(dilution.get_share_count_series("BRK-B"),
                           "multi_class_unweighted_total")

        assert "1,500" in message, message
        assert "2,140,709,794" in message, message

    def test_a_one_for_one_multi_class_filer_stays_quiet(
            self, filings, market_total):
        """GOOGL's classes really are equivalent, and the market says so to
        within 0.0005%. Warning here would make the warning meaningless
        everywhere."""
        from tools.web_search_server import dilution
        filings(_googl_series())
        market_total(12_229_934_831)

        result = dilution.get_share_count_series("GOOGL")

        assert result["share_basis"]["economically_equivalent"] is True
        assert "multi_class_unweighted_total" not in _codes(result)
        assert result["latest_total"] == 12_230_000_000

    def test_a_single_class_filer_never_consults_the_market(
            self, filings, market_total, monkeypatch):
        """One class cannot be mixed with another, so there is nothing to check
        and no network call to pay for. NVDA, AAPL and MSFT go down this path."""
        from tools.web_search_server import dilution
        filings([_point("2026-05-20", "2026-05-15", [(24_200_000_000, None)]),
                 _point("2025-05-28", "2025-05-23", [(24_400_000_000, None)])])

        calls = []
        monkeypatch.setattr(dilution, "fetch_market_share_count",
                            lambda ticker: calls.append(ticker))

        result = dilution.get_share_count_series("NVDA")

        assert calls == []
        assert result["latest_total_basis"] == "single_class"
        assert result["share_basis"]["economically_equivalent"] is True
        assert _codes(result) == []

    def test_an_unreachable_market_is_reported_not_assumed_equivalent(
            self, filings, market_total):
        """The failure this brief singles out: assuming 1:1 is exactly how a
        1,500:1 filer comes back looking reconciled. Not knowing is stated."""
        from tools.web_search_server import dilution
        filings(_brk_series())
        market_total(RuntimeError("HTTPError: 503 from the quote provider"))

        result = dilution.get_share_count_series("BRK-B")

        assert result["share_basis"]["economically_equivalent"] is None
        assert result["share_basis"]["source_error"]
        message = _message(result, "multi_class_basis_unverified")
        assert "1:1" in message, message

    def test_no_weight_is_published_when_the_residual_is_not_a_round_ratio(
            self, filings, market_total):
        """A gap can also mean a stale cover page or a fast diluter. Only a
        residual that lands on a round multiple is a conversion ratio, and a
        ratio that cannot be stood behind is withheld rather than qualified."""
        from tools.web_search_server import dilution
        filings(_brk_series())
        market_total(1_620_000_000)          # +15%, no round weight explains it

        share_basis = dilution.get_share_count_series("BRK-B")["share_basis"]

        assert share_basis["economically_equivalent"] is False
        assert share_basis["implied_class_weights"] is None
        assert share_basis["quote_equivalent_total"] == 1_620_000_000

    @network
    def test_brk_is_flagged_against_live_filings(self):
        from tools.web_search_server import dilution
        result = dilution.get_share_count_series("BRK-B", limit=4)

        assert result["success"] is True
        assert result["latest_total_basis"] == "sum_of_classes_unweighted"
        assert result["share_basis"]["economically_equivalent"] is False
        assert result["share_basis"]["quote_equivalent_total"] > result[
            "latest_total"] * 1.4

    @network
    @pytest.mark.parametrize("ticker", ["NVDA", "AAPL", "MSFT"])
    def test_single_class_filers_are_unchanged_against_live_filings(self, ticker):
        from tools.web_search_server import dilution
        result = dilution.get_share_count_series(ticker, limit=4)

        assert result["latest_total_basis"] == "single_class"
        assert "multi_class_unweighted_total" not in _codes(result)


# --------------------------------------------------------------------------
# 3. prices that never say what basis they are on
# --------------------------------------------------------------------------

def _bars(rows):
    """A yfinance-shaped daily frame: (date, close, dividend, split)."""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York")
                              for d, *_ in rows])
    return pd.DataFrame(
        {
            "Open":  [c for _, c, _, _ in rows],
            "High":  [c for _, c, _, _ in rows],
            "Low":   [c for _, c, _, _ in rows],
            "Close": [c for _, c, _, _ in rows],
            "Volume": [1_000_000] * len(rows),
            "Dividends": [d for _, _, d, _ in rows],
            "Stock Splits": [s for _, _, _, s in rows],
        },
        index=index,
    )


class TestPriceAdjustment:
    def test_a_split_inside_the_window_is_named_with_its_ratio(self):
        from tools.financial_modeling_engine.utils import price_adjustment

        adjustment = price_adjustment(_bars([
            ("2024-05-23", 105.0, 0.0, 0.0),
            ("2024-06-10", 120.1, 0.0, 10.0),
            ("2024-06-11", 121.0, 0.0, 0.0),
        ]))

        assert adjustment["splits_in_window"] == [
            {"date": "2024-06-10", "ratio": 10.0}]
        assert adjustment["cumulative_split_factor"] == pytest.approx(10.0)

    def test_a_window_with_no_split_has_a_factor_of_one(self):
        from tools.financial_modeling_engine.utils import price_adjustment

        adjustment = price_adjustment(_bars([
            ("2026-08-24", 209.0, 0.0, 0.0),
            ("2026-08-25", 210.0, 0.01, 0.0),
        ]))

        assert adjustment["splits_in_window"] == []
        assert adjustment["cumulative_split_factor"] == 1.0

    def test_the_dividends_taken_out_of_the_price_path_are_quantified(self):
        """auto_adjust strips dividends as well as splits. AAPL's dividends
        after 2020-08-28 are 4.75% of that bar, so a market cap built from
        these closes understates by that much -- unstated, until now."""
        from tools.financial_modeling_engine.utils import price_adjustment

        adjustment = price_adjustment(_bars([
            ("2025-08-25", 100.0, 0.0, 0.0),
            ("2025-11-10", 102.0, 1.00, 0.0),
            ("2026-02-10", 104.0, 1.50, 0.0),
        ]))

        assert adjustment["dividends_in_window"] == 2
        assert adjustment["dividends_per_share_removed"] == pytest.approx(2.5)
        assert adjustment["dividends_pct_of_oldest_close"] == pytest.approx(2.5)

    def test_the_split_warning_names_the_field_that_pairs_with_these_prices(self):
        """The exact-multiple error: `get_share_count_series.total` is the
        as-filed cover-page count and these closes are back-adjusted, so the
        product is out by the split ratio. `total_split_adjusted` is the one
        that pairs."""
        from tools.financial_modeling_engine.utils import (
            price_adjustment, price_adjustment_warnings)

        adjustment = price_adjustment(_bars([
            ("2024-05-23", 105.0, 0.0, 0.0),
            ("2024-06-10", 120.1, 0.0, 10.0),
        ]))
        entries = price_adjustment_warnings("NVDA", adjustment)

        assert [e["code"] for e in entries] == ["prices_split_adjusted"]
        message = entries[0]["message"]
        assert "total_split_adjusted" in message, message
        assert "10" in message

    def test_no_split_warning_fires_on_a_quiet_window(self):
        from tools.financial_modeling_engine.utils import (
            price_adjustment, price_adjustment_warnings)

        adjustment = price_adjustment(_bars([
            ("2026-08-24", 209.0, 0.0, 0.0),
            ("2026-08-25", 210.0, 0.0, 0.0),
        ]))
        assert price_adjustment_warnings("NVDA", adjustment) == []

    @network
    @pytest.mark.parametrize("ticker", ["GOOGL", "BRK-B", "NVDA", "AAPL", "MSFT"])
    def test_every_price_history_declares_its_basis(self, ticker):
        """Unconditional, because it is a property of every bar in the response
        and not an exception to be warned about."""
        from tools.financial_modeling_engine.utils import get_price_history
        result = get_price_history(ticker, period="1y", include_recent_bars=3)

        assert result["success"] is True
        assert result["price_basis"] == "split_and_dividend_adjusted"
        assert result["price_adjustment"]["auto_adjust"] is True

    @network
    def test_a_window_spanning_a_split_says_so(self):
        """NVDA's 10-for-1 of June 2024 sits inside a 5y window."""
        from tools.financial_modeling_engine.utils import get_price_history
        result = get_price_history("NVDA", period="5y", include_recent_bars=3)

        assert result["price_adjustment"]["cumulative_split_factor"] == \
            pytest.approx(10.0)
        assert "prices_split_adjusted" in _codes(result)

    @network
    def test_a_recent_window_carries_no_split_warning(self):
        from tools.financial_modeling_engine.utils import get_price_history
        result = get_price_history("NVDA", period="6mo", include_recent_bars=3)

        assert "prices_split_adjusted" not in _codes(result)


# --------------------------------------------------------------------------
# 4. dividends that are silently split-adjusted
# --------------------------------------------------------------------------

class TestDividendBasis:
    def test_a_dividend_paid_before_a_split_reports_its_as_declared_amount(self):
        """AAPL declared $0.82 on 2020-08-07 and the provider reports 0.205,
        because the 4-for-1 of 2020-08-31 has been applied to it. Multiplying
        0.205 by an as-filed share count from the same quarter is out by 4x in
        the opposite direction to every price in this toolset."""
        from tools.financial_modeling_engine.corporate_actions import (
            as_declared_factor)

        splits = [{"date": "2020-08-31", "ratio": 4.0}]
        assert as_declared_factor("2020-08-07", splits) == pytest.approx(4.0)
        assert as_declared_factor("2020-09-07", splits) == pytest.approx(1.0)

    def test_a_split_on_the_dividend_date_itself_is_not_applied_twice(self):
        """Strictly after, matching the share-count rebasing: a payment on the
        split date is already stated in post-split units."""
        from tools.financial_modeling_engine.corporate_actions import (
            as_declared_factor)

        assert as_declared_factor(
            "2020-08-31", [{"date": "2020-08-31", "ratio": 4.0}]) == 1.0

    def test_successive_splits_compound(self):
        from tools.financial_modeling_engine.corporate_actions import (
            as_declared_factor)

        splits = [{"date": "2014-06-09", "ratio": 7.0},
                  {"date": "2020-08-31", "ratio": 4.0}]
        assert as_declared_factor("2013-01-01", splits) == pytest.approx(28.0)

    @network
    def test_aapl_dividends_carry_both_bases(self):
        from tools.financial_modeling_engine.corporate_actions import (
            get_corporate_actions)
        result = get_corporate_actions("AAPL", years=10)

        assert result["dividend_basis"] == "split_adjusted"
        pre_split = [d for d in result["dividends"]
                     if d["date"].startswith("2020-08-07")]
        assert pre_split, "expected AAPL's 2020-08-07 dividend in a 10y window"
        assert pre_split[0]["amount"] == pytest.approx(0.205, abs=0.001)
        assert pre_split[0]["amount_as_declared"] == pytest.approx(0.82, abs=0.001)
        assert "dividends_split_adjusted" in _codes(result)

    @network
    def test_a_filer_with_no_split_in_the_window_is_left_alone(self):
        """MSFT last split in 2003. Every as-declared amount equals the
        reported one and nothing is warned about."""
        from tools.financial_modeling_engine.corporate_actions import (
            get_corporate_actions)
        result = get_corporate_actions("MSFT", years=10)

        assert result["split_count"] == 0
        assert all(d["amount_as_declared"] == d["amount"]
                   for d in result["dividends"])
        assert "dividends_split_adjusted" not in _codes(result)
