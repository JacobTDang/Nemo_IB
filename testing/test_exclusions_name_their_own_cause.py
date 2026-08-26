"""A count of exclusions is only half the answer; the cause is the other half.

`comparable_company_analysis` dropped peers correctly and then explained every
one of them with a single fixed string:

    "N peer(s) reported no comparable multiple -- a foreign issuer whose
     multiples are suppressed across currencies, or a filer tagging nothing"

Run live against MU, BRK-B, ZZZZNOTREAL and AMD, the three real causes were:

    MU           a US filer whose provider marketCap came back null
    BRK-B        a negative enterprise value, so EV multiples are refused
    ZZZZNOTREAL  not a company at all -- the symbol did not resolve

None of them is a foreign issuer and none of them is a filer tagging nothing.
The counts were right and the stated reason was wrong for every peer, which is
worse than saying nothing: an analyst reading "foreign issuer" goes looking
for a currency problem that is not there, and never learns that one of their
four comparables does not exist.

`get_market_data` already refuses an unresolved symbol outright, so the
information was available and was being thrown away at the boundary.

The second half of this file is the same defect one layer down.
`get_market_data("MU")` dropped pe_ratio and pb_ratio because the provider
reported no marketCap, and set `multiples_suppressed_reason: null` beside
them -- the field built to explain suppression explained nothing, because it
was wired only to the cross-currency and negative-EV paths. A null reason
reads as "we could not obtain this", which is the one thing it was not.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.financial_modeling_engine.analysis_tools as at
from tools.financial_modeling_engine.analysis_tools import peer_distribution


# --- peers, as get_data would return them -----------------------------------

AMD = {
    "ticker": "AMD", "marketCap": 2.6e11, "netIncomeToCommon": 1.6e9,
    "enterpriseValue": 2.6e11, "revenue": 2.5e10, "EBITDA": 4.0e9,
    "EBIT": 2.0e9, "pe_ratio": 162.5, "pb_ratio": 4.4, "ev_revenue": 10.4,
    "ev_ebitda": 65.0, "ev_ebit": 130.0, "multiples_suppressed_reason": None,
}

MU = {
    "ticker": "MU", "marketCap": None, "netIncomeToCommon": 8.5e9,
    "enterpriseValue": 1.3e11, "revenue": 3.7e10, "EBITDA": 1.7e10,
    "EBIT": 1.0e10, "ev_revenue": 3.5, "ev_ebitda": 7.6, "ev_ebit": 13.0,
    "provider_trailing_pe": 21.249,
    "multiples_suppressed_reason": (
        "pe_ratio and pb_ratio are not reported for MU: the market-data "
        "provider returned no market capitalisation."),
}

BRK_B = {
    "ticker": "BRK-B", "marketCap": 1.05e12, "netIncomeToCommon": 8.9e10,
    "enterpriseValue": -2.339e11, "revenue": 3.7e11, "EBITDA": 1.2e11,
    "EBIT": 1.1e11, "pe_ratio": 11.8, "pb_ratio": 1.6,
    "multiples_suppressed_reason": (
        "EV multiples are not reported: the provider's enterprise value for "
        "BRK-B is -233,900,000,000, which is not positive."),
}

NOT_REAL = {
    "ticker": "ZZZZNOTREAL", "success": False,
    "error": ("'ZZZZNOTREAL' did not resolve to a listed security at the "
              "market-data provider. This is a failed lookup, not a company "
              "without financials."),
}

PEERS = {"AMD": AMD, "MU": MU, "BRK-B": BRK_B, "ZZZZNOTREAL": NOT_REAL}


@pytest.fixture
def comps(monkeypatch):
    monkeypatch.setattr(at, "get_data", lambda ticker: PEERS[ticker])
    blocks = asyncio.run(at.Financial_Analysis().comparable_company_analysis(
        list(PEERS)))
    return json.loads(blocks[0].text)


def _excluded_for(payload, block):
    return {e["ticker"]: e.get("reason") or "" for e in payload[block]["excluded_peers"]}


class TestEachExclusionCarriesItsOwnReason:
    def test_the_counts_are_still_right(self, comps):
        """The counting was never the defect and must not regress."""
        assert comps["pe_ratio"]["included_count"] == 2      # AMD, BRK-B
        assert comps["pe_ratio"]["excluded_count"] == 2      # MU, ZZZZNOTREAL
        assert comps["ev_ebitda_data"]["included_count"] == 2  # AMD, MU
        assert comps["ev_ebitda_data"]["excluded_count"] == 2  # BRK-B, ZZZZNOTREAL

    def test_every_excluded_peer_is_named(self, comps):
        assert set(_excluded_for(comps, "pe_ratio")) == {"MU", "ZZZZNOTREAL"}
        assert set(_excluded_for(comps, "ev_ebitda_data")) == {"BRK-B", "ZZZZNOTREAL"}

    def test_the_missing_market_cap_is_blamed_on_the_market_cap(self, comps):
        reason = _excluded_for(comps, "pe_ratio")["MU"].lower()
        assert "market cap" in reason or "marketcap" in reason, reason
        assert "foreign" not in reason and "currency" not in reason, (
            f"MU is a US filer and was blamed on a currency problem: {reason}")

    def test_the_negative_enterprise_value_is_blamed_on_the_enterprise_value(
            self, comps):
        reason = _excluded_for(comps, "ev_ebitda_data")["BRK-B"].lower()
        assert "enterprise value" in reason, reason
        assert "foreign" not in reason and "currency" not in reason, reason

    def test_a_symbol_that_did_not_resolve_says_so(self, comps):
        """Folded into `excluded_absent` with no mention that the lookup
        failed, a non-existent comparable reads as a real company that
        discloses nothing -- and the comp set silently has one fewer member
        than the analyst believes."""
        for block in ("pe_ratio", "pb_data", "ev_revenue_data",
                      "ev_ebitda_data", "ev_ebit_data"):
            reason = _excluded_for(comps, block)["ZZZZNOTREAL"].lower()
            assert "did not resolve" in reason or "not resolve" in reason, (
                f"{block}: {reason}")

    def test_no_peer_is_given_a_cause_that_is_not_its_own(self, comps):
        """The whole defect: one fixed string for every exclusion."""
        for block in ("pe_ratio", "ev_ebitda_data"):
            reasons = set(_excluded_for(comps, block).values())
            assert len(reasons) == 2, (
                f"{block}: two different causes were reported identically")

    def test_the_aggregate_reason_still_exists_for_a_reader_who_wants_one(
            self, comps):
        summary = comps["pe_ratio"]["excluded_reason"]
        assert summary
        assert "MU" in summary and "ZZZZNOTREAL" in summary


class TestPeerDistributionWithoutContext:
    """Called with bare values it has no cause to report, and must not invent
    one. The old string asserted 'a foreign issuer or a filer tagging nothing'
    on no evidence at all."""

    def test_a_bare_call_still_counts_and_explains(self):
        stats = peer_distribution([25.0, 30.0, None, None])
        assert stats["included_count"] == 2
        assert stats["excluded_count"] == 2
        assert stats["excluded_reason"]

    def test_a_bare_call_does_not_claim_a_cause_it_was_not_given(self):
        reason = peer_distribution([25.0, None]).get("excluded_reason", "")
        assert "foreign" not in reason.lower(), (
            f"a cause was asserted with no evidence for it: {reason}")

    def test_a_supplied_reason_is_carried_through(self):
        stats = peer_distribution(
            [25.0, None],
            tickers=["AMD", "MU"],
            reasons={"MU": "the provider returned no market capitalisation"})
        assert stats["excluded_peers"] == [
            {"ticker": "MU", "value": None,
             "reason": "the provider returned no market capitalisation"}]

    def test_an_implausible_value_is_attributed_to_its_own_value(self):
        stats = peer_distribution([25.0, -20767.83], tickers=["AMD", "INTC"])
        excluded = {e["ticker"]: e["reason"] for e in stats["excluded_peers"]}
        assert "INTC" in excluded
        assert "-20767" in excluded["INTC"] or "denominator" in excluded["INTC"]


# --- one layer down: get_market_data itself ---------------------------------

class TestSuppressionAlwaysCarriesAReason:
    def test_the_note_names_the_input_that_was_missing(self):
        from tools.financial_modeling_engine.utils import missing_inputs_note
        note = missing_inputs_note("MU", "pe_ratio", ["marketCap"])
        assert "MU" in note and "pe_ratio" in note and "marketCap" in note

    def test_a_filer_with_no_market_cap_says_why_its_ratios_are_gone(self):
        """`marketCap: null` killed pe_ratio and pb_ratio and set
        `multiples_suppressed_reason: null` beside them."""
        data = _fake_market_data(market_cap=None)
        assert data.get("pe_ratio") is None and data.get("pb_ratio") is None
        reason = data.get("multiples_suppressed_reason")
        assert reason, (
            "the ratios were dropped and the field built to explain the "
            "suppression said nothing")
        assert "market cap" in reason.lower() or "marketcap" in reason.lower()

    def test_the_detail_is_per_multiple(self):
        detail = _fake_market_data(market_cap=None).get(
            "multiples_suppressed_detail") or {}
        assert "pe_ratio" in detail and "pb_ratio" in detail
        assert "ev_ebitda" not in detail, (
            "EV multiples were computable and must not be reported as "
            "suppressed")

    def test_the_providers_own_pe_is_still_passed_through(self):
        """It is computed upstream in one currency; refusing ours should not
        lose it, and it is the substitute the note should leave available."""
        assert _fake_market_data(market_cap=None)["provider_trailing_pe"] == 21.249

    def test_a_complete_filer_reports_no_suppression(self):
        """The guard must not cost a healthy name its clean payload."""
        data = _fake_market_data()
        assert data["pe_ratio"] is not None and data["pb_ratio"] is not None
        assert data.get("multiples_suppressed_reason") is None
        assert not data.get("multiples_suppressed_detail")

    def test_a_negative_enterprise_value_keeps_its_own_reason(self):
        data = _fake_market_data(enterprise_value=-2.339e11)
        assert data.get("ev_ebitda") is None
        assert "enterprise value" in data["multiples_suppressed_reason"].lower()


# --- a provider stand-in, so this runs offline -------------------------------

def _fake_market_data(market_cap=1.3e11, enterprise_value=1.3e11):
    """get_data against a stubbed provider. Only the fields the multiples read.

    Stubbed rather than live so the failure is the code's, not the weather's;
    the live MU case is covered by the network test below.
    """
    import pandas as pd

    from tools.financial_modeling_engine import utils

    period = pd.Timestamp("2025-08-28")
    income = pd.DataFrame(
        {period: [1.0e10, 5.0e8]},
        index=["Operating Income", "Interest Expense"])
    balance = pd.DataFrame({period: [6.5e10]}, index=["Stockholders Equity"])

    class _Ticker:
        history_metadata = {"symbol": "MU"}
        info = {
            "symbol": "MU", "marketCap": market_cap, "currentPrice": 118.0,
            "totalRevenue": 3.7e10, "ebitda": 1.7e10,
            "netIncomeToCommon": 8.5e9, "enterpriseValue": enterprise_value,
            "totalCash": 9.6e9, "totalDebt": 1.5e10,
            "sharesOutstanding": None if market_cap is None else 1.1e9,
            "beta": 1.3, "currency": "USD", "financialCurrency": "USD",
            "trailingPE": 21.249,
        }
        income_stmt = income
        balance_sheet = balance

        def __init__(self, ticker):
            pass

    original = utils.yf
    try:
        class _Module:
            Ticker = _Ticker
        utils.yf = _Module
        return utils.get_data("MU")
    finally:
        utils.yf = original


@pytest.mark.network
def test_the_live_filer_that_found_this_explains_itself():
    from tools.financial_modeling_engine.utils import get_data
    data = get_data("MU")
    if data.get("pe_ratio") is not None:
        pytest.skip("the provider is reporting MU's market cap again")
    assert data.get("multiples_suppressed_reason"), (
        "MU's ratios are absent with no reason given")
