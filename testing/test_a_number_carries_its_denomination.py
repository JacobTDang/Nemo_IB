"""A number whose currency, unit or source is not in the response.

Every figure these tools return is one a caller will put beside a figure from
another tool. That only works if each side says what it is measured in. Four
places it did not, all verified live on 2026-08-26:

1. TSM's EPS, twice, 6.11x apart and neither labelled.

       get_earnings_surprises(TSM)  actual_eps  27.25
       get_forward_estimates(TSM)   eps 0q avg   4.45834

   The first is TWD per ordinary share, the second USD per ADR. Finnhub
   resolves the ADR to the local listing -- `/stock/earnings` for TSM echoes
   `symbol: "2330.TW"`, and `/stock/profile2` answers `currency: "TWD"` with
   25,932.37m shares against the ADR's 5,186m. Chained without labels, TSM's
   EPS "collapses 84%" into the print.

   Same field, same tool, two currencies: `revenue_B 0q` is 1454.9601 for TSM
   (TWD billions, ~$46bn) and 44.4452 for DELL (USD billions). The name
   asserts billions and says nothing about of what, so TSM reads 33x DELL.

2. Finnhub reports market cap and enterprise value in millions while every
   other tool in the stack returns raw currency units.

       get_basic_financials(NVDA)  marketCapitalization  5422978
       get_market_data(NVDA)       marketCap             5078174924800

   Read literally the first says NVDA is worth $5.4 million. Nothing in the
   Finnhub response carries a unit, and `data_as_of` is null, so the 6.79%
   disagreement between those two figures at the same instant -- $5.423tn
   against $5.078tn, against 0.38% for MSFT -- is invisible too.

3. `get_corporate_actions` and `extract_13f_holdings` claimed
   `provider: "SEC EDGAR"` while reading yfinance. That is worse than a
   mis-credit: `get_share_count_series.split_adjustment.source` says
   "yfinance", so a caller cross-checking a split ratio between the two
   believed they had two independent providers agreeing. They had one source
   read twice, and the EDGAR label is what sold the illusion.

4. `get_earnings_surprises` period labels do not identify the period. AMAT
   reported its Q3 on 2026-08-13 for a quarter ended 2026-07-26; the row is
   labelled `period: "2026-09-30"` -- neither date, and five weeks in the
   future. It is a calendar-quarter bucket. Joining on it returns nothing.

The rule these tests hold: a figure carries its currency, its scale and the
source that produced it, or it says plainly that it could not be established.
Labelling only -- no tool here invents an FX rate. Where a provider publishes
its own translated figure that is reportable as theirs, which is the standard
`get_revenue_base` already set for foreign filers.
"""
import json
import os

import pytest

from tools.news_agregator.finnhub_server import (
    FinnhubServer,
    _condense_basic_financials,
    _condense_earnings_surprises,
    _label_forward_denomination,
)
from testing._gates import requires_finnhub

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"

network = pytest.mark.skipif(SKIP_NETWORK, reason="live network test")

CALENDAR_QUARTER_ENDS = ("-03-31", "-06-30", "-09-30", "-12-31")


@pytest.fixture(autouse=True)
def _no_cached_denominations():
    """The currency lookup is memoised per process, which is right in
    production -- a listing's currency does not change and re-asking spends a
    call restating a constant -- and wrong across tests, where one case's
    stubbed profile would answer the next case's question."""
    from tools.news_agregator import finnhub_utils
    finnhub_utils._DENOMINATION_CACHE.clear()
    yield
    finnhub_utils._DENOMINATION_CACHE.clear()


# Verified live 2026-08-26 against /stock/profile2.
TSM_DENOMINATION = {
    "requested_symbol": "TSM",
    "finnhub_symbol": "2330.TW",
    "currency": "TWD",
    "shares_outstanding_millions": 25932.37,
    "error": None,
}
NVDA_DENOMINATION = {
    "requested_symbol": "NVDA",
    "finnhub_symbol": "NVDA",
    "currency": "USD",
    "shares_outstanding_millions": 24200.0,
    "error": None,
}
UNKNOWN_DENOMINATION = {
    "requested_symbol": "ZZZZNOTREAL",
    "finnhub_symbol": None,
    "currency": None,
    "shares_outstanding_millions": None,
    "error": "Finnhub: HTTP 403",
}


def _server(payload, *, profile=None):
    """A FinnhubServer whose upstream answers `payload`, or `profile` for
    /stock/profile2 when one is given."""
    server = FinnhubServer.__new__(FinnhubServer)

    class _Client:
        async def get(self, endpoint, params=None):
            if profile is not None and endpoint == "/stock/profile2":
                return profile
            return payload

    server.client = _Client()
    return server


def _envelope(contents):
    return json.loads(contents[0].text)


def _codes(envelope):
    return [w.get("code") for w in (envelope.get("warnings") or [])]


def _text(envelope):
    return " ".join(str(w.get("message", ""))
                    for w in (envelope.get("warnings") or [])).lower()


# ---------------------------------------------------------------------------
# 2. The scale: Finnhub answers in millions, everything else in raw units
# ---------------------------------------------------------------------------

class TestBasicFinancialsCarriesItsScale:
    def test_market_cap_and_enterprise_value_are_named_as_millions(self):
        """5422978 is $5.42tn or $5.42m depending on a fact the response did
        not carry. `get_market_data.marketCap` is raw dollars, so a caller
        mixing the two is off by 10^6 with nothing to catch it."""
        out = _condense_basic_financials(
            {"metric": {"marketCapitalization": 5422978,
                        "enterpriseValue": 5418211, "epsTTM": 6.53}},
            denomination=NVDA_DENOMINATION)
        scaled = out["denomination"]["scaled_fields"]
        assert scaled["marketCapitalization"] == "millions"
        assert scaled["enterpriseValue"] == "millions"

    def test_the_currency_is_reported(self):
        out = _condense_basic_financials(
            {"metric": {"marketCapitalization": 5422978}},
            denomination=NVDA_DENOMINATION)
        assert out["denomination"]["currency"] == "USD"

    def test_a_foreign_filer_is_not_reported_in_dollars(self):
        """TSM's 63,145,320 is NT$63.1tn. Called dollars it is $63tn, which
        would make TSM larger than every listed company combined."""
        out = _condense_basic_financials(
            {"metric": {"marketCapitalization": 63145320, "epsTTM": 87.3818}},
            denomination=TSM_DENOMINATION)
        assert out["denomination"]["currency"] == "TWD"

    def test_the_listing_finnhub_actually_answered_about_is_named(self):
        """Finnhub resolves TSM to 2330.TW and reports the Taiwan ordinary
        share. The condenser used to drop that symbol, so nothing in the
        response said the figures were not about the ADR the caller asked
        for."""
        out = _condense_basic_financials(
            {"metric": {"epsTTM": 87.3818}, "symbol": "2330.TW"},
            denomination=TSM_DENOMINATION)
        assert out["denomination"]["finnhub_symbol"] == "2330.TW"
        assert out["denomination"]["requested_symbol"] == "TSM"

    def test_an_unknown_currency_is_never_defaulted_to_dollars(self):
        """The securities whose metadata is thinnest are exactly the ones a
        default would be wrong about. Unknown is reported as unknown."""
        out = _condense_basic_financials(
            {"metric": {"marketCapitalization": 1.0}},
            denomination=UNKNOWN_DENOMINATION)
        assert out["denomination"]["currency"] is None
        assert out["denomination"]["error"]


class TestBasicFinancialsWarnsBeforeItIsCombined:
    async def test_a_foreign_listing_says_so_in_a_warning(self):
        envelope = _envelope(await _server(
            {"metric": {"marketCapitalization": 63145320, "epsTTM": 87.3818},
             "symbol": "2330.TW"},
            profile={"ticker": "2330.TW", "currency": "TWD",
                     "shareOutstanding": 25932.37},
        ).get_basic_financials("TSM"))
        assert "reported_on_the_local_listing" in _codes(envelope), envelope
        assert "twd" in _text(envelope)

    async def test_a_foreign_currency_alone_is_enough_to_warn(self):
        """BABA is why the resolved symbol cannot be the test. It stays
        'BABA' on the NYSE and Finnhub still reports it in CNY, on 19,063.63m
        ordinary shares against the ADS count a US quote carries. A guard that
        fired only when the symbol changed would pass BABA straight through.
        """
        envelope = _envelope(await _server(
            {"metric": {"marketCapitalization": 279285, "epsTTM": 3.8044},
             "symbol": "BABA"},
            profile={"ticker": "BABA", "currency": "CNY",
                     "shareOutstanding": 19063.63},
        ).get_basic_financials("BABA"))
        assert "reported_on_the_local_listing" in _codes(envelope), envelope
        assert "cny" in _text(envelope)

    async def test_the_warning_does_not_assert_a_share_basis(self):
        """Finnhub is not consistent about it. TSM's EPS is per ordinary
        share of 2330.TW; BABA's is per ADS while its epsTTM is per ordinary
        share. Naming one basis would be wrong for whichever issuer is not
        the one the sentence was written about, so the warning hands over the
        share count and says the basis is unstated."""
        envelope = _envelope(await _server(
            {"metric": {"marketCapitalization": 279285}, "symbol": "BABA"},
            profile={"ticker": "BABA", "currency": "CNY",
                     "shareOutstanding": 19063.63},
        ).get_basic_financials("BABA"))
        entry = next(w for w in envelope["warnings"]
                     if w["code"] == "reported_on_the_local_listing")
        assert entry["shares_outstanding_millions"] == 19063.63
        assert "does not state the share basis" in entry["message"]

    async def test_a_domestic_filer_is_not_warned_about_currency(self):
        """The guard must not cost US names a clean response."""
        envelope = _envelope(await _server(
            {"metric": {"marketCapitalization": 5422978}, "symbol": "NVDA"},
            profile={"ticker": "NVDA", "currency": "USD",
                     "shareOutstanding": 24200},
        ).get_basic_financials("NVDA"))
        assert "reported_on_the_local_listing" not in _codes(envelope)

    async def test_a_currency_that_could_not_be_established_is_flagged(self):
        envelope = _envelope(await _server(
            {"metric": {"marketCapitalization": 5422978}, "symbol": "NVDA"},
            profile={"error": "HTTP 403: no access"},
        ).get_basic_financials("NVDA"))
        assert "currency_unknown" in _codes(envelope), envelope

    async def test_the_millions_caveat_is_attached_to_the_tool(self):
        """Static, so it belongs on the dispatcher rather than being rebuilt
        per response -- the same place get_short_interest's staleness lives."""
        config = _finnhub_annotation_config()
        codes = [w["code"]
                 for w in config["warnings_per_tool"]["get_basic_financials"]]
        assert "units_are_millions" in codes, codes
        assert "no_as_of_timestamp" in codes, codes

    async def test_the_cross_source_gap_is_stated(self):
        """Finnhub had NVDA at $5.423tn against get_market_data's $5.078tn at
        the same instant -- 6.79%, against 0.38% for MSFT. With data_as_of
        null the size of that gap is unknowable per call, so the caveat names
        it and says not to mix the two."""
        config = _finnhub_annotation_config()
        messages = " ".join(
            w["message"]
            for w in config["warnings_per_tool"]["get_basic_financials"])
        assert "get_market_data" in messages, messages


def _finnhub_annotation_config():
    from testing.test_known_limits import _annotation_config
    return _annotation_config(FinnhubServer())


# ---------------------------------------------------------------------------
# 1. The currency and the share basis behind an EPS
# ---------------------------------------------------------------------------

class TestEarningsSurprisesCarriesItsCurrency:
    def test_the_currency_reaches_the_response(self):
        out = _condense_earnings_surprises(
            [{"period": "2026-06-30", "year": 2026, "quarter": 2,
              "actual": 27.25, "estimate": 24.5662, "surprisePercent": 10.92,
              "symbol": "2330.TW"}],
            denomination=TSM_DENOMINATION)
        assert out["denomination"]["currency"] == "TWD"

    def test_the_listing_the_eps_belongs_to_is_named(self):
        """27.25 is TWD per ordinary share. The caller asked about TSM, whose
        ADR is five ordinary shares, and nothing said so."""
        out = _condense_earnings_surprises(
            [{"period": "2026-06-30", "year": 2026, "quarter": 2,
              "actual": 27.25, "estimate": 24.5662, "surprisePercent": 10.92,
              "symbol": "2330.TW"}],
            denomination=TSM_DENOMINATION)
        assert out["denomination"]["finnhub_symbol"] == "2330.TW"
        assert out["denomination"]["requested_symbol"] == "TSM"

    async def test_a_foreign_eps_is_warned_about_before_it_is_chained(self):
        envelope = _envelope(await _server(
            [{"period": "2026-06-30", "year": 2026, "quarter": 2,
              "actual": 27.25, "estimate": 24.5662, "surprisePercent": 10.92,
              "symbol": "2330.TW"}],
            profile={"ticker": "2330.TW", "currency": "TWD",
                     "shareOutstanding": 25932.37},
        ).get_earnings_surprises("TSM"))
        assert "reported_on_the_local_listing" in _codes(envelope), envelope


# ---------------------------------------------------------------------------
# 4. The period label
# ---------------------------------------------------------------------------

class TestThePeriodLabelSaysWhatItIs:
    def test_the_field_is_explained(self):
        out = _condense_earnings_surprises(
            [{"period": "2026-09-30", "year": 2026, "quarter": 3,
              "actual": 3.5, "estimate": 3.4544, "surprisePercent": 1.32}])
        label = out["period_label"]
        assert label["field"] == "period"
        assert label["means"]

    def test_it_is_not_offered_as_a_fiscal_period_end_or_a_report_date(self):
        """AMAT's quarter ended 2026-07-26 and it reported on 2026-08-13. The
        row says 2026-09-30, which is neither, and joining a fiscal period end
        against it returns nothing at all."""
        out = _condense_earnings_surprises(
            [{"period": "2026-09-30", "year": 2026, "quarter": 3,
              "actual": 3.5, "estimate": 3.4544, "surprisePercent": 1.32}])
        excluded = " ".join(out["period_label"]["is_not"]).lower()
        assert "fiscal period end" in excluded
        assert "report" in excluded

    def test_the_fiscal_identity_is_named(self):
        out = _condense_earnings_surprises(
            [{"period": "2026-09-30", "year": 2026, "quarter": 3,
              "actual": 3.5, "estimate": 3.4544, "surprisePercent": 1.32}])
        assert out["period_label"]["fiscal_identity"] == ["year", "quarter"]

    def test_a_repeated_fiscal_quarter_is_flagged_not_silently_counted(self):
        """TGT, live 2026-08-26: FY2027 Q2 appears twice, actual 2.46 both
        times, once as period 2026-09-30 and once as 2025-09-30. Left alone it
        double-counts a beat and drags the average surprise toward one
        quarter. It is not our row to delete, but it is ours to declare."""
        out = _condense_earnings_surprises([
            {"period": "2026-09-30", "year": 2027, "quarter": 2,
             "actual": 2.46, "estimate": 2.3095, "surprisePercent": 6.5166},
            {"period": "2025-09-30", "year": 2027, "quarter": 2,
             "actual": 2.46, "estimate": 2.3095, "surprisePercent": 6.5166},
            {"period": "2026-06-30", "year": 2027, "quarter": 1,
             "actual": 1.71, "estimate": 1.477, "surprisePercent": 15.7752},
        ])
        assert out["duplicate_fiscal_periods"], out
        assert [2027, 2] in [list(d["fiscal"]) for d in
                             out["duplicate_fiscal_periods"]]

    def test_a_clean_history_reports_no_duplicates(self):
        out = _condense_earnings_surprises([
            {"period": "2026-06-30", "year": 2027, "quarter": 1,
             "actual": 1.87, "estimate": 1.7922, "surprisePercent": 4.341},
            {"period": "2026-03-31", "year": 2026, "quarter": 4,
             "actual": 1.62, "estimate": 1.5634, "surprisePercent": 3.62},
        ])
        assert out["duplicate_fiscal_periods"] == []

    async def test_the_bucket_caveat_is_attached_to_the_tool(self):
        config = _finnhub_annotation_config()
        codes = [w["code"]
                 for w in config["warnings_per_tool"]["get_earnings_surprises"]]
        assert "period_is_a_calendar_bucket" in codes, codes


# ---------------------------------------------------------------------------
# 1 (continued). Forward estimates: one response, two currencies
# ---------------------------------------------------------------------------

class TestForwardEstimatesCarryTheirCurrency:
    def test_revenue_is_named_in_billions_of_a_named_currency(self):
        """`revenue_B` asserts billions and nothing else. TSM's 1454.9601 is
        TWD billions (~$46bn) beside DELL's 44.4452 USD billions, so the
        unlabelled field makes TSM look 33x DELL."""
        condensed = _label_forward_denomination(
            {"eps": {"periods": [{"period": "0q", "avg": 4.45834}],
                     "_source": "yfinance_fallback"},
             "revenue_B": {"periods": [{"period": "0q", "avg": 1454.9601}],
                           "_source": "yfinance_fallback"},
             "ebitda_B": {"error": "no yfinance equivalent"}},
            finnhub_currency="TWD",
            yf_quote_currency="USD",
            yf_reporting_currency="TWD")
        assert condensed["revenue_B"]["_currency"] == "TWD"
        assert "billions of TWD" in condensed["revenue_B"]["_unit"]

    def test_a_domestic_filer_gets_a_definite_eps_currency(self):
        condensed = _label_forward_denomination(
            {"eps": {"periods": [{"period": "0q", "avg": 2.09161}],
                     "_source": "yfinance_fallback"},
             "revenue_B": {"periods": [{"period": "0q", "avg": 92.1766}],
                           "_source": "yfinance_fallback"},
             "ebitda_B": {"error": "x"}},
            finnhub_currency="USD",
            yf_quote_currency="USD",
            yf_reporting_currency="USD")
        assert condensed["eps"]["_currency"] == "USD"
        assert "_currency_candidates" not in condensed["eps"]

    def test_an_adr_eps_currency_is_reported_unknown_rather_than_guessed(self):
        """yfinance does not say which of its two currencies an EPS estimate
        is in, and it is not consistently one of them. Live 2026-08-26:

            TSM   eps 0q 4.45834   ADR $417.69  -> USD (quote currency)
            SONY  eps 0q 0.33459   ADR  $24.12  -> USD (quote currency)
            BABA  eps 0q 10.90073  ADS $119.83  -> CNY (reporting currency)

        BABA at 10.90 USD/quarter would put it on a P/E of 2.7. Picking the
        quote currency because it is right three times in four is how a 7.15x
        error gets shipped, so an ambiguous one is reported ambiguous."""
        condensed = _label_forward_denomination(
            {"eps": {"periods": [{"period": "0q", "avg": 4.45834}],
                     "_source": "yfinance_fallback"},
             "revenue_B": {"error": "x"}, "ebitda_B": {"error": "x"}},
            finnhub_currency="TWD",
            yf_quote_currency="USD",
            yf_reporting_currency="TWD")
        assert condensed["eps"]["_currency"] is None
        assert set(condensed["eps"]["_currency_candidates"]) == {"USD", "TWD"}

    def test_a_finnhub_served_field_takes_finnhubs_own_currency(self):
        """No `_source` means Finnhub answered, and Finnhub answers on the
        listing /stock/profile2 names."""
        condensed = _label_forward_denomination(
            {"eps": {"periods": [{"period": "0q", "avg": 27.0}]},
             "revenue_B": {"periods": [{"period": "0q", "avg": 1454.96}]},
             "ebitda_B": {"error": "x"}},
            finnhub_currency="TWD",
            yf_quote_currency=None,
            yf_reporting_currency=None)
        assert condensed["eps"]["_currency"] == "TWD"
        assert condensed["revenue_B"]["_currency"] == "TWD"

    def test_a_field_nobody_answered_is_not_given_a_currency(self):
        condensed = _label_forward_denomination(
            {"eps": {"error": "Finnhub: HTTP 403"},
             "revenue_B": {"error": "x"}, "ebitda_B": {"error": "x"}},
            finnhub_currency="USD",
            yf_quote_currency="USD",
            yf_reporting_currency="USD")
        assert "_currency" not in condensed["eps"]

    def test_the_derived_ebitda_inherits_the_revenue_currency(self):
        """It is revenue multiplied by a margin, so it is whatever revenue
        was."""
        condensed = _label_forward_denomination(
            {"eps": {"error": "x"},
             "revenue_B": {"periods": [{"period": "0q", "avg": 1454.96}],
                           "_source": "yfinance_fallback"},
             "ebitda_B": {"periods": [{"period": "0q", "avg": 950.0}],
                          "_source": "yfinance_fallback_inferred"}},
            finnhub_currency="TWD",
            yf_quote_currency="USD",
            yf_reporting_currency="TWD")
        assert condensed["ebitda_B"]["_currency"] == "TWD"


# ---------------------------------------------------------------------------
# The labels must not turn an empty answer into a full one
# ---------------------------------------------------------------------------

class TestLabellingDoesNotManufactureContent:
    async def test_an_unknown_symbol_is_still_not_covered(self):
        """`denomination` describes how to read an answer, not the answer. If
        it counted as content, a symbol Finnhub does not carry would come back
        looking answered and the not_covered label would never fire again."""
        envelope = _envelope(await _server(
            {"metric": {}, "series": {}, "symbol": "ZZZZNOTREAL"},
            profile={},
        ).get_basic_financials("ZZZZNOTREAL"))
        assert envelope["coverage"] == "not_covered", envelope

    async def test_an_empty_earnings_history_is_still_not_covered(self):
        envelope = _envelope(
            await _server([], profile={}).get_earnings_surprises("ZZZZNOTREAL"))
        assert envelope["coverage"] == "not_covered", envelope


# ---------------------------------------------------------------------------
# 3. Provider labels: name what actually answered
# ---------------------------------------------------------------------------

def _financial_annotation_config():
    from tools.financial_modeling_engine.analysis_tools import Financial_Analysis
    from testing.test_known_limits import _annotation_config
    return _annotation_config(Financial_Analysis())


class TestTheProviderIsTheOneThatAnswered:
    @pytest.mark.parametrize("tool", ["get_corporate_actions",
                                      "extract_13f_holdings"])
    def test_neither_claims_sec_edgar(self, tool):
        """corporate_actions.py calls `yf.Ticker(symbol)`;
        get_institutional_holdings reads `yf.Ticker(ticker).major_holders`.
        The response shape agrees independently -- tz-aware timestamps and
        split-adjusted dividends, neither of which EDGAR publishes."""
        provider = _financial_annotation_config()["per_tool"].get(tool)
        assert provider is not None, f"{tool}: no explicit provider"
        assert "EDGAR" not in provider, provider

    @pytest.mark.parametrize("tool", ["get_corporate_actions",
                                      "extract_13f_holdings"])
    def test_both_name_yfinance(self, tool):
        provider = _financial_annotation_config()["per_tool"][tool]
        assert "yfinance" in provider.lower(), provider

    def test_the_split_cross_check_is_declared_non_independent(self):
        """`get_share_count_series.split_adjustment.source` is "yfinance". A
        caller comparing a split ratio across the two believes two providers
        agree; they have one provider read twice, and the EDGAR label was what
        made that believable."""
        entries = _financial_annotation_config()["warnings_per_tool"][
            "get_corporate_actions"]
        message = " ".join(w["message"] for w in entries).lower()
        assert "get_share_count_series" in message, message
        assert "yfinance" in message

    def test_the_13f_aggregation_is_declared(self):
        entries = _financial_annotation_config()["warnings_per_tool"][
            "extract_13f_holdings"]
        message = " ".join(w["message"] for w in entries).lower()
        assert "13f" in message
        assert "yahoo" in message or "yfinance" in message


# ---------------------------------------------------------------------------
# Live. The class was found live and stays checked live.
# ---------------------------------------------------------------------------

async def _live(tool: str, ticker: str, *args):
    """One live call against Finnhub, on a client that is closed afterwards.

    A server shared across tests cannot be: aiohttp binds its session to the
    event loop that created it and pytest-asyncio gives each test a fresh one,
    so the second test onward talks to a closed loop.
    """
    server = FinnhubServer()
    try:
        contents = await getattr(server, tool)(ticker, *args)
    finally:
        await server.client.close()
    return json.loads(contents[0].text)


@pytest.mark.network
@requires_finnhub
class TestAgainstLiveIssuers:
    @pytest.mark.parametrize("ticker,currency", [
        ("TSM", "TWD"), ("NVDA", "USD"), ("DELL", "USD"), ("BABA", "CNY"),
    ])
    async def test_basic_financials_names_the_currency(self, ticker, currency):
        body = await _live("get_basic_financials", ticker)
        assert body["data"]["denomination"]["currency"] == currency

    async def test_tsm_is_reported_on_its_taiwan_listing(self):
        body = await _live("get_basic_financials", "TSM")
        assert body["data"]["denomination"]["finnhub_symbol"] == "2330.TW"

    @pytest.mark.parametrize("ticker,currency", [
        ("TSM", "TWD"), ("NVDA", "USD"), ("DELL", "USD"),
    ])
    async def test_earnings_surprises_names_the_currency(self, ticker,
                                                         currency):
        body = await _live("get_earnings_surprises", ticker)
        assert body["data"]["denomination"]["currency"] == currency

    @pytest.mark.parametrize("ticker", ["AMAT", "WMT", "NVDA", "DELL"])
    async def test_the_period_is_a_bucket_and_not_a_period_end(self, ticker):
        """The evidence for the label, both halves.

        These four close their quarters in January, April, July and October,
        so not one of their real period ends can be a calendar quarter end --
        and every `period` Finnhub returns for them is. If Finnhub ever
        started returning real fiscal period ends the label would become a
        lie, and this is what would say so.
        """
        import yfinance as yf
        fiscal_ends = {str(c.date()) for c in
                       yf.Ticker(ticker).quarterly_income_stmt.columns}
        assert fiscal_ends, f"{ticker}: no fiscal period ends to compare"

        body = await _live("get_earnings_surprises", ticker)
        periods = {row["period"] for row in body["data"]["quarters"]}
        assert periods, f"{ticker}: no quarters returned"
        for period in periods:
            assert period.endswith(CALENDAR_QUARTER_ENDS), period
        assert not (periods & fiscal_ends), (
            f"{ticker}: {sorted(periods & fiscal_ends)} look like fiscal "
            f"period ends, so the calendar-bucket label may be wrong")

    async def test_tsm_revenue_estimate_is_labelled_twd(self):
        """1454.9601 is NT$1.45tn, about $46bn, sitting in a field named
        revenue_B beside DELL's 44.4452 US dollars."""
        body = await _live("get_forward_estimates", "TSM")
        revenue = body["data"]["revenue_B"]
        assert revenue.get("_currency") == "TWD", revenue
        assert "billions of TWD" in revenue["_unit"]

    async def test_dell_revenue_estimate_is_labelled_usd(self):
        body = await _live("get_forward_estimates", "DELL")
        revenue = body["data"]["revenue_B"]
        assert revenue.get("_currency") == "USD", revenue

    async def test_tsm_forward_eps_is_not_silently_called_dollars(self):
        """The whole reason TSM's EPS could be chained 6.11x wrong."""
        body = await _live("get_forward_estimates", "TSM")
        eps = body["data"]["eps"]
        assert eps.get("_currency") is None, eps
        assert set(eps["_currency_candidates"]) == {"USD", "TWD"}
        assert "currency_not_established" in _codes(body), body["warnings"]

    async def test_dell_forward_eps_is_labelled_usd(self):
        body = await _live("get_forward_estimates", "DELL")
        assert body["data"]["eps"].get("_currency") == "USD"

    @pytest.mark.parametrize("ticker,expected", [
        ("TSM", True), ("BABA", True), ("NVDA", False)])
    async def test_the_foreign_filer_warning_fires_only_where_it_applies(
            self, ticker, expected):
        body = await _live("get_basic_financials", ticker)
        fired = "reported_on_the_local_listing" in _codes(body)
        assert fired is expected, body.get("warnings")
