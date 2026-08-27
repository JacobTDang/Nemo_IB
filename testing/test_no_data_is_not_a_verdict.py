"""A verdict derived from no rows is not a verdict.

`build_envelope` already labels an empty Finnhub payload `not_covered`, and it
works -- profile, peers, basic financials and company news all carry the label
against a symbol Finnhub does not answer for. Four tools slipped past it
because they do not hand `build_envelope` the empty thing. They hand it a
summary computed over the empty thing, and a summary always has content:

    get_insider_transactions  signal "neutral", total_bought 0, net_shares 0
    get_insider_sentiment     months [], signal "neutral"
    get_earnings_surprises    quarters [], beat_count 0, miss_count 0
    get_analyst_rating_trend  error "no recommendation data"

`signal: "neutral"` is the worst of them. It is the same field that reads
"net_selling" for NVDA, so nothing downstream can separate "insiders were
balanced" from "we got nothing back", and a screen filtering on
`signal != "net_selling"` silently admits every ticker the plan cannot see.
`beat_count: 0` beside `quarters: []` reads as a fact about the company.

The rule these tests hold: a field derived from a collection is only reported
when the collection has rows. With no rows there is no total, no count and no
signal -- the source array is the empty thing, and handing that to
`build_envelope` unadorned is what earns the coverage label and the warning.

The warning deliberately does not say why the response was empty. Verified
live 2026-08-26: `/stock/insider-transactions` answers `{"data": [], "symbol":
"X"}` for SHOP, NVO and SAP -- three real, covered companies with no Form 4
rows -- and byte-for-byte the same body for ZZZZNOTREAL. Finnhub also returns
an empty body for a symbol outside the plan's entitlement. Three causes, one
response, so the label names all three and asserts none.

`get_financial_statements` is the same defect wearing a different hat. Finnhub
answers `/stock/financials` with HTTP 403 on this plan -- for NVDA as much as
for ZZZZNOTREAL -- and the condenser diagnosed that as "Unrecognized response
format". The format was recognizable; it was a refusal. `get_forward_estimates`
already handles its own 403s correctly, keeping the provider's words in
`metadata.errors` and naming the source that actually answered, and that is
the shape copied here.
"""
import json
import os

import pytest

from tools.news_agregator import finnhub_server
from tools.news_agregator.finnhub_server import (
    FinnhubServer,
    _condense_earnings_surprises,
    _condense_financial_statements,
    _condense_insider_data,
    _condense_insider_sentiment,
)

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"

network = pytest.mark.skipif(SKIP_NETWORK, reason="live network test")

_403 = "HTTP 403: {\"error\":\"You don't have access to this resource.\"}"


def _server(payload, profile=None):
    """A FinnhubServer whose single upstream call returns `payload`.

    `profile` answers /stock/profile2 separately. Basic financials and
    earnings surprises now ask it what currency they are reporting in -- an
    EPS of 27.25 means nothing until you know it is TWD per ordinary share --
    and a stub that answered the earnings list to that question would have
    every response here reporting a currency it could not establish.
    """
    server = FinnhubServer.__new__(FinnhubServer)

    class _Client:
        async def get(self, endpoint, params=None):
            if endpoint == "/stock/profile2":
                return profile if profile is not None else payload
            return payload

    server.client = _Client()
    return server


def _envelope(contents):
    return json.loads(contents[0].text)


def _messages(envelope):
    return " ".join(str(w.get("message", ""))
                    for w in (envelope.get("warnings") or [])).lower()


# --------------------------------------------------------------------------
# The condensers: no rows in, no derived figure out
# --------------------------------------------------------------------------

class TestInsiderTransactions:
    def test_no_transactions_yields_no_signal(self):
        """"neutral" is a reading. It belongs to a window that had buying and
        selling that cancelled, not to a window we never saw."""
        out = _condense_insider_data({"data": []})
        assert out["signal"] is None, (
            f"signal={out['signal']!r} over zero transactions occupies the "
            f"same field that says 'net_selling' for a real reading")

    def test_no_transactions_yields_no_totals(self):
        """`net_shares: 0` says insiders moved no stock. Nobody measured."""
        out = _condense_insider_data({"data": []})
        for field in ("total_bought", "total_sold", "net_shares",
                      "buy_count", "sell_count"):
            assert out[field] is None, (
                f"{field}={out[field]!r} states a figure derived from an "
                f"empty transaction list")

    def test_no_transactions_yields_no_recency_buckets(self):
        out = _condense_insider_data({"data": []})
        for bucket in ("recent_30d", "recent_90d", "prior_90d"):
            assert out[bucket] is None, f"{bucket}={out[bucket]!r}"

    def test_the_source_array_stays_an_empty_array(self):
        """`build_envelope` reads the payload to decide coverage, so the
        emptiness has to survive into it rather than being summarised away."""
        out = _condense_insider_data({"data": []})
        assert out["top_insiders"] == []

    def test_a_real_reading_keeps_its_signal_and_totals(self):
        """Removing the false verdict must not cost the true one."""
        out = _condense_insider_data({"data": [
            {"name": "COOK", "change": -50000, "transactionCode": "S",
             "transactionDate": "2025-12-15"},
            {"name": "LEVINSON", "change": -340000, "transactionCode": "S",
             "transactionDate": "2025-09-30"},
        ]})
        assert out["signal"] == "net_selling"
        assert out["total_sold"] == 390000
        assert out["total_bought"] == 0
        assert out["net_shares"] == -390000

    def test_a_genuine_zero_inside_a_real_reading_survives(self):
        """A window of pure selling really did have zero purchases. That is a
        measurement, and it stays a zero."""
        out = _condense_insider_data({"data": [
            {"name": "COOK", "change": -50000, "transactionCode": "S",
             "transactionDate": "2025-12-15"},
        ]})
        assert out["total_bought"] == 0
        assert out["buy_count"] == 0


class TestInsiderSentiment:
    def test_no_months_yields_no_signal(self):
        out = _condense_insider_sentiment({"data": []})
        assert out["signal"] is None, (
            f"signal={out['signal']!r} with months=[] reports an MSPR verdict "
            f"over no MSPR readings")
        assert out["months"] == []
        assert out["avg_mspr"] is None

    def test_months_with_no_mspr_value_yield_no_signal(self):
        """The same rule one level in: rows arrived, the number did not."""
        out = _condense_insider_sentiment({"data": [
            {"year": 2026, "month": 7, "mspr": None, "change": -100},
        ]})
        assert out["avg_mspr"] is None
        assert out["signal"] is None

    def test_a_real_reading_keeps_its_signal(self):
        out = _condense_insider_sentiment({"data": [
            {"year": 2026, "month": 7, "mspr": -100, "change": -500000},
            {"year": 2026, "month": 6, "mspr": -95.84, "change": -2742641},
        ]})
        assert out["signal"] == "net_selling"
        assert out["avg_mspr"] == pytest.approx(-97.92, abs=0.01)


class TestEarningsSurprises:
    def test_no_quarters_yields_no_counts(self):
        """`beat_count: 0` beside `quarters: []` reads as "this company has
        never beaten", which is a claim about the company."""
        out = _condense_earnings_surprises([])
        assert out["quarters"] == []
        assert out["beat_count"] is None, f"beat_count={out['beat_count']!r}"
        assert out["miss_count"] is None, f"miss_count={out['miss_count']!r}"
        assert out.get("avg_surprise_pct") is None

    def test_a_real_history_still_counts_beats_and_misses(self):
        out = _condense_earnings_surprises([
            {"period": "2026-06-30", "year": 2027, "quarter": 1,
             "actual": 1.87, "estimate": 1.7922, "surprisePercent": 4.341},
            {"period": "2026-03-31", "year": 2026, "quarter": 4,
             "actual": 1.40, "estimate": 1.50, "surprisePercent": -6.67},
        ])
        assert out["beat_count"] == 1
        assert out["miss_count"] == 1
        assert out["total_periods"] == 2

    def test_a_clean_beat_streak_still_reports_zero_misses(self):
        """Four quarters were read and none was a miss. That zero is an
        answer and must not be nulled with the fabricated ones."""
        out = _condense_earnings_surprises([
            {"period": "2026-06-30", "surprisePercent": 4.3},
            {"period": "2026-03-31", "surprisePercent": 3.6},
        ])
        assert out["beat_count"] == 2
        assert out["miss_count"] == 0


class TestFinancialStatementsDiagnosis:
    def test_a_403_is_not_called_an_unrecognized_format(self):
        """The body was a perfectly recognizable entitlement refusal. Calling
        it a parsing problem sends the reader to fix the wrong thing, and
        loses the one sentence that names the fixable one."""
        out = _condense_financial_statements({"error": _403}, "ic", "annual")
        assert "403" in out["error"], out["error"]
        assert "unrecognized" not in out["error"].lower(), (
            f"a 403 diagnosed as a format problem: {out['error']!r}")

    def test_the_providers_own_words_survive(self):
        out = _condense_financial_statements({"error": _403}, "ic", "annual")
        assert "access to this resource" in out["error"], out["error"]

    def test_a_genuinely_unrecognized_body_still_says_so(self):
        """Fixing the false diagnosis must not delete the true one."""
        out = _condense_financial_statements({"surprise": [1, 2, 3]}, "ic",
                                             "annual")
        assert "nrecognized" in out["error"], out["error"]

    def test_a_recognized_body_is_parsed_as_before(self):
        out = _condense_financial_statements(
            {"financials": {"annual": {"ic": [
                {"period": "2026-01-31", "revenue": 215938000000.0,
                 "netIncome": 120067000000.0}]}}},
            "ic", "annual")
        assert out["count"] == 1
        assert out["periods"][0]["revenue"] == 215938000000.0
        assert "error" not in out


# --------------------------------------------------------------------------
# The envelopes: the emptiness reaches the caller labelled
# --------------------------------------------------------------------------

class TestEmptyEnvelopesAreLabelled:
    async def test_insider_transactions(self):
        envelope = _envelope(
            await _server({"data": [], "symbol": "ZZZZNOTREAL"})
            .get_insider_transactions("ZZZZNOTREAL"))
        assert envelope["coverage"] == "not_covered", envelope
        assert envelope["warnings"], "nothing explains the empty response"
        assert envelope["data"]["signal"] is None

    async def test_insider_sentiment(self):
        envelope = _envelope(
            await _server({"data": [], "symbol": "ZZZZNOTREAL"})
            .get_insider_sentiment("ZZZZNOTREAL"))
        assert envelope["coverage"] == "not_covered", envelope
        assert envelope["warnings"]
        assert envelope["data"]["signal"] is None

    async def test_earnings_surprises(self):
        envelope = _envelope(
            await _server([]).get_earnings_surprises("ZZZZNOTREAL"))
        assert envelope["coverage"] == "not_covered", envelope
        assert envelope["warnings"]

    async def test_analyst_revisions_history(self):
        """It answered `error: "no recommendation data"` -- a sentence with no
        code, no coverage label and nothing in metadata.errors."""
        envelope = _envelope(
            await _server([]).get_analyst_rating_trend("ZZZZNOTREAL"))
        assert envelope["coverage"] == "not_covered", envelope
        assert envelope["warnings"]
        assert envelope["data"].get("signal") is None

    @pytest.mark.parametrize("tool", [
        "get_insider_transactions", "get_insider_sentiment",
        "get_earnings_surprises", "get_analyst_rating_trend",
    ])
    async def test_the_label_never_claims_the_company_lacks_the_data(self, tool):
        """Three causes produce this body and the response cannot tell them
        apart, so the label must not pick one."""
        payload = [] if tool in ("get_earnings_surprises",
                                 "get_analyst_rating_trend") \
            else {"data": [], "symbol": "ZZZZNOTREAL"}
        envelope = _envelope(
            await getattr(_server(payload), tool)("ZZZZNOTREAL"))
        message = _messages(envelope)
        assert "finnhub" in message, message
        for claim in ("does not disclose", "has no insider", "no such company",
                      "has never", "insiders did not"):
            assert claim not in message, (
                f"{tool} made a claim about the company: {message[:200]}")

    async def test_the_label_names_a_real_symbol_with_nothing_to_report(self):
        """SHOP, NVO and SAP return this exact body. A label that offers only
        "unknown symbol" or "not entitled" is wrong for all three."""
        envelope = _envelope(
            await _server({"data": [], "symbol": "SHOP"})
            .get_insider_transactions("SHOP"))
        message = _messages(envelope)
        assert "nothing to report" in message or "no rows" in message, (
            f"the label offers no cause that fits a covered company with an "
            f"empty window: {message[:250]}")


class TestRealDataIsUntouched:
    async def test_a_real_insider_reading_carries_no_warning(self):
        envelope = _envelope(
            await _server({"data": [
                {"name": "HUANG JEN HSUN", "change": -100000,
                 "transactionCode": "S", "transactionDate": "2026-07-01"},
            ]}).get_insider_transactions("NVDA"))
        assert envelope.get("warnings", []) == [], envelope["warnings"]
        assert envelope.get("coverage") != "not_covered"
        assert envelope["data"]["signal"] == "net_selling"

    async def test_a_real_earnings_history_carries_no_warning(self):
        envelope = _envelope(await _server(
            [{"period": "2026-06-30", "year": 2027, "quarter": 1,
              "actual": 1.87, "estimate": 1.7922, "surprisePercent": 4.341}],
            profile={"ticker": "NVDA", "currency": "USD"},
        ).get_earnings_surprises("NVDA"))
        assert envelope.get("warnings", []) == [], envelope["warnings"]
        assert envelope["data"]["beat_count"] == 1

    async def test_a_real_rating_history_carries_no_warning(self):
        """lookback_months matches the history on offer, so there is no
        shortfall to report. Asking for 12 and receiving 2 is a different
        response and carries `history_shorter_than_requested`."""
        envelope = _envelope(await _server([
            {"period": "2026-08-01", "strongBuy": 23, "buy": 41, "hold": 3,
             "sell": 1, "strongSell": 0},
            {"period": "2026-07-01", "strongBuy": 24, "buy": 40, "hold": 4,
             "sell": 1, "strongSell": 0},
        ]).get_analyst_rating_trend("NVDA", lookback_months=2))
        assert envelope.get("warnings", []) == [], envelope["warnings"]
        assert envelope["data"]["periods_returned"] == 2


class TestFinancialStatementsEnvelope:
    async def test_the_403_reaches_metadata_errors(self, monkeypatch):
        """Where a caller looks. `get_forward_estimates` already puts it
        there; this one buried it in `data.error` under a false diagnosis."""
        monkeypatch.setattr(finnhub_server, "_yf_financial_statements",
                            lambda t, s, f: {"statement": s, "freq": f,
                                             "periods": [],
                                             "_source": "yfinance_fallback"})
        envelope = _envelope(
            await _server({"error": _403})
            .get_financial_statements("ZZZZNOTREAL", "ic", "annual"))
        errors = envelope["metadata"]["errors"]
        assert errors and any("403" in str(e) for e in errors), envelope
        codes = [w["code"] for w in envelope.get("warnings", [])]
        assert "primary_source_unavailable" in codes, codes
        assert "Finnhub" not in envelope["provider"], (
            f"nothing was retrieved and the response credits "
            f"{envelope['provider']!r} for it")

    async def test_the_403_is_recorded_even_when_the_fallback_answers(
            self, monkeypatch):
        """NVDA gets real statements from yfinance, and the response credited
        Finnhub with `errors: []` -- a fixable entitlement problem reported as
        a silent success."""
        monkeypatch.setattr(
            finnhub_server, "_yf_financial_statements",
            lambda t, s, f: {"statement": s, "freq": f, "count": 1,
                             "periods": [{"period": "2026-01-31",
                                          "revenue": 215938000000.0}],
                             "_source": "yfinance_fallback"})
        envelope = _envelope(
            await _server({"error": _403})
            .get_financial_statements("NVDA", "ic", "annual"))
        assert envelope["data"]["periods"], "the fallback's data was lost"
        errors = envelope["metadata"]["errors"]
        assert errors and any("403" in str(e) for e in errors), envelope
        assert "yfinance" in envelope["provider"], envelope["provider"]

    async def test_a_finnhub_answer_still_credits_finnhub(self, monkeypatch):
        def _must_not_run(*args, **kwargs):
            raise AssertionError("fallback ran on a successful primary call")

        monkeypatch.setattr(finnhub_server, "_yf_financial_statements",
                            _must_not_run)
        envelope = _envelope(
            await _server({"financials": {"annual": {"ic": [
                {"period": "2026-01-31", "revenue": 215938000000.0}]}}})
            .get_financial_statements("NVDA", "ic", "annual"))
        assert envelope["metadata"]["errors"] == []
        assert envelope.get("warnings", []) == []
        assert envelope["provider"] == "Finnhub"


class TestTheAnalysisPromptSaysItToo:
    """A null that reaches the prompt as "0" undoes the fix at the last step.

    The analysis agent renders the tool payload into the text the model reads,
    with `data.get('net_shares', 0):,` and `data.get('signal', 'neutral')`.
    Nulling the fabricated fields would have re-fabricated them there -- or
    raised a TypeError formatting None -- so the renderer states the emptiness
    in the same words the envelope does.
    """

    @staticmethod
    def _render(tool, data):
        from agent.Financial_Analysis_Agent import Financial_Analysis_Agent
        agent = Financial_Analysis_Agent.__new__(Financial_Analysis_Agent)
        return agent._format_market_intel(tool, data)

    def test_empty_insider_transactions_are_not_rendered_as_zero(self):
        text = self._render("get_insider_transactions",
                            _condense_insider_data({"data": []}))
        assert "Net shares: 0" not in text, text
        assert "no insider transactions" in text.lower(), text

    def test_empty_insider_sentiment_is_not_rendered_as_neutral(self):
        text = self._render("get_insider_sentiment",
                            _condense_insider_sentiment({"data": []}))
        assert "NEUTRAL" not in text, text
        assert "no monthly mspr" in text.lower(), text

    def test_a_real_insider_reading_still_renders_its_figures(self):
        text = self._render("get_insider_transactions",
                            _condense_insider_data({"data": [
                                {"name": "COOK", "change": -50000,
                                 "transactionCode": "S",
                                 "transactionDate": "2025-12-15"}]}))
        assert "NET SELLING" in text, text
        assert "-50,000" in text, text


# --------------------------------------------------------------------------
# Live, against the real Finnhub API
# --------------------------------------------------------------------------

@network
class TestLive:
    @pytest.fixture
    async def live(self):
        from testing._gates import requires_finnhub  # noqa: F401 -- gate
        from tools.news_agregator.finnhub_utils import FinnhubClient
        server = FinnhubServer.__new__(FinnhubServer)
        server.client = FinnhubClient()
        try:
            yield server
        finally:
            await server.client.close()

    async def test_an_unknown_symbol_gets_no_verdict(self, live):
        for coro, field in (
            (live.get_insider_transactions("ZZZZNOTREAL"), "signal"),
            (live.get_insider_sentiment("ZZZZNOTREAL"), "signal"),
            (live.get_earnings_surprises("ZZZZNOTREAL"), "beat_count"),
        ):
            envelope = _envelope(await coro)
            assert envelope["coverage"] == "not_covered", envelope
            assert envelope["data"].get(field) is None, envelope["data"]

    @pytest.mark.parametrize("ticker", ["NVDA", "AAPL", "MSFT"])
    async def test_a_real_company_keeps_its_payload_and_no_warnings(
            self, live, ticker):
        envelope = _envelope(await live.get_insider_transactions(ticker))
        assert envelope.get("warnings", []) == [], envelope["warnings"]
        assert envelope.get("coverage") != "not_covered"
        assert envelope["data"]["signal"] in ("net_buying", "net_selling",
                                              "neutral")

        envelope = _envelope(await live.get_earnings_surprises(ticker))
        assert envelope.get("warnings", []) == [], envelope["warnings"]
        assert isinstance(envelope["data"]["beat_count"], int)

    async def test_a_covered_company_with_an_empty_window_is_not_accused(
            self, live):
        """SHOP files no Form 4s. Finnhub returns the same empty body it
        returns for a symbol it does not carry, so the response may report
        that it got nothing -- and may not report a balanced insider window,
        nor say the company discloses nothing."""
        envelope = _envelope(await live.get_insider_transactions("SHOP"))
        if envelope["data"].get("top_insiders"):
            pytest.skip("SHOP has insider rows in this run")
        assert envelope["data"]["signal"] is None, envelope["data"]
        message = _messages(envelope)
        assert "nothing to report" in message, message

    async def test_the_403_on_financial_statements_is_named(self, live):
        envelope = _envelope(
            await live.get_financial_statements("NVDA", "ic", "annual"))
        errors = envelope["metadata"]["errors"]
        assert errors and any("403" in str(e) for e in errors), envelope
        blob = json.dumps(envelope).lower()
        assert "unrecognized response format" not in blob, (
            "a 403 is still being reported as a parsing failure")
