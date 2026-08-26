"""A symbol the provider cannot resolve is not a company with no data.

`get_data("ZZZZNOTREAL")` returns a fully-formed payload -- every field
present, every value null -- and no failure of any kind. The dispatcher then
infers `success: true`, because inference reads an absent `error` key as
success. A caller who typos a ticker gets a confident answer about a company
that does not exist.

It is worse than a bare empty. The response also carries:

    multiples_suppressed_reason: "Multiples are not reported: the quote is in
    an unknown currency while the financials are reported in an unknown
    currency..."

which is the cross-currency guard firing on two unknowns and explaining the
absence with a reason that is not the reason. The caller is told why the
multiples are missing and never told that the company is.

yfinance does not raise here: it logs `HTTP Error 404` and hands back an empty
info dict, exactly as it does for a split calendar. `history_metadata` is
populated only when the provider actually answered, which separates "no such
symbol" from "a real company whose fields we could not read".
"""
import pytest

from tools.financial_modeling_engine import utils


class _Unresolved:
    """What yfinance actually hands back for a symbol it cannot find.

    Measured, not assumed: a single `trailingPegRatio: None` key and an empty
    `history_metadata`. An emptiness check on `info` does NOT catch this --
    the first version of this guard used one and passed the unit test while
    doing nothing live.
    """
    info = {"trailingPegRatio": None}
    history_metadata = {}

    def __init__(self, *a, **k):
        pass

    def get_income_stmt(self, *a, **k):
        return None

    def get_balance_sheet(self, *a, **k):
        return None

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Resolved(_Unresolved):
    """A real company that happens to tag little: resolved, mostly empty."""
    info = {"marketCap": 1_000_000, "currency": "USD"}
    history_metadata = {"symbol": "REAL", "currency": "USD"}


def test_an_unresolvable_symbol_is_reported_as_a_failure(monkeypatch):
    monkeypatch.setattr(utils.yf, "Ticker", _Unresolved)
    result = utils.get_data("ZZZZNOTREAL")

    assert result.get("success") is False, (
        "a symbol the provider could not resolve was reported as a success")
    assert result.get("error"), "the failure carries no reason"
    assert "ZZZZNOTREAL" in str(result["error"])


def test_the_failure_is_not_explained_as_a_currency_problem(monkeypatch):
    """Two unknown currencies are a symptom of the missing company, not the
    cause of the missing multiples."""
    monkeypatch.setattr(utils.yf, "Ticker", _Unresolved)
    result = utils.get_data("ZZZZNOTREAL")

    reason = str(result.get("multiples_suppressed_reason") or "")
    assert "unknown currency" not in reason, (
        f"the response blames the currency for a company that does not "
        f"exist: {reason[:160]}")


def test_a_real_company_with_thin_data_still_succeeds(monkeypatch):
    """The guard must not turn a sparse filer into a lookup failure."""
    monkeypatch.setattr(utils.yf, "Ticker", _Resolved)
    result = utils.get_data("REAL")

    assert result.get("success") is not False, (
        "a resolved company with few fields was reported as unresolvable")


# --- the same shape in three more tools --------------------------------------
#
# A live probe with a symbol that does not exist found three more market-data
# tools reporting success. `get_corporate_actions` is the worst of them: it
# answers `pays_dividend: false`, `split_count: 0`, `ttm_dividend: 0.0` --
# concrete claims about a company that is not there. A caller checking whether
# a holding pays a dividend gets "no" rather than "no such security".


@pytest.mark.parametrize("module,fname", [
    ("tools.financial_modeling_engine.utils", "get_data"),
    ("tools.financial_modeling_engine.utils", "get_short_interest"),
    ("tools.financial_modeling_engine.utils", "get_institutional_holdings"),
    ("tools.financial_modeling_engine.corporate_actions",
     "get_corporate_actions"),
])
def test_no_market_data_tool_claims_success_for_a_missing_symbol(
        module, fname, monkeypatch):
    import importlib

    mod = importlib.import_module(module)
    monkeypatch.setattr(mod.yf, "Ticker", _Unresolved, raising=False)

    result = getattr(mod, fname)("ZZZZNOTREAL")
    assert isinstance(result, dict)
    assert result.get("success") is False, (
        f"{fname} reported success for a symbol that does not exist")
    assert result.get("error"), f"{fname} failed without a reason"


def test_corporate_actions_does_not_answer_the_dividend_question(monkeypatch):
    """"Does it pay a dividend?" must not be answered "no" for a security
    that does not exist."""
    from tools.financial_modeling_engine import corporate_actions as ca

    monkeypatch.setattr(ca.yf, "Ticker", _Unresolved, raising=False)
    result = ca.get_corporate_actions("ZZZZNOTREAL")

    assert result.get("pays_dividend") is not False or result.get("success") is False
    assert result.get("success") is False
