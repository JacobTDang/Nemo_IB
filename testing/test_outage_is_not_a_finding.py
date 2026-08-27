"""An outage must never be reported as a fact about the filer.

`earnings_quality._series` already states the rule this file enforces:
"Only NotCovered is swallowed, and only to try the next concept. A network
failure or an unknown ticker propagates: reporting an outage as 'this filer
does not disclose it' is the one answer worse than an error."

Six concept-walking tools broke it. Each wrapped its whole fetch in
`except Exception: continue`, walked off the end of the concept list, and
returned the not-covered message written for a filer that genuinely omits
the tag. Asked about NVDA during a 503 they answered, in their own words,
"NVDA does not tag long-term debt maturities in its 10-K" and "NVDA does
not disaggregate revenue by geography" -- affirmative claims about a real
company's filings, phrased for an agent to repeat.

The distinction is the exception type, not the outcome: NotCovered means
try the next concept, anything else means say what went wrong.
"""
import importlib

import pytest

CASES = [
    ("sbc", "get_sbc_series"),
    ("debt_maturity", "get_debt_maturity_schedule"),
    ("forward_metrics", "get_contracted_revenue"),
    ("forward_metrics", "get_geographic_revenue"),
    ("forward_metrics", "get_public_float"),
    ("earnings_quality", "get_accruals_quality"),
    ("earnings_quality", "get_working_capital_trends"),
    ("earnings_quality", "get_operating_leases"),
    ("foreign_issuer", "get_annual_revenue"),
]

# Phrasings that assert something about the filer rather than about the fetch.
CLAIMS_ABOUT_THE_FILER = (
    "does not tag", "does not disaggregate", "does not report",
    "not covered", "no consolidated revenue concept found",
    "was not disclosed", "does not disclose",
)


@pytest.mark.parametrize("modname,fname", CASES,
                         ids=[f"{m}.{f}" for m, f in CASES])
def test_a_fetch_failure_is_reported_as_a_failure(modname, fname, monkeypatch):
    mod = importlib.import_module(f"tools.web_search_server.{modname}")
    # Some paths resolve the SEC contact address before they fetch, and that
    # refusal would mask the behaviour under test.
    monkeypatch.setenv("SEC_EMAIL", "test@example.invalid")

    def boom(*args, **kwargs):
        raise ConnectionError("SEC returned 503")

    monkeypatch.setattr(mod, "fetch_concept_series", boom)
    result = getattr(mod, fname)("NVDA")

    assert result["success"] is False
    message = (result.get("error") or result.get("note") or "").lower()

    claimed = [p for p in CLAIMS_ABOUT_THE_FILER if p in message]
    assert not claimed, (
        f"{modname}.{fname} answered a 503 with {claimed} -- a statement "
        f"about what NVDA discloses, not about what failed: {message[:200]}")
    assert "503" in message or "failed" in message, (
        f"{modname}.{fname} does not say what went wrong: {message[:200]}")


@pytest.mark.parametrize("modname,fname", CASES,
                         ids=[f"{m}.{f}" for m, f in CASES])
def test_a_genuine_absence_still_reads_as_an_absence(modname, fname, monkeypatch):
    """The control. Making an outage loud must not make every gap loud too.

    NotCovered is the signal that this filer does not tag the concept, and it
    must still produce the not-covered answer rather than an error -- that
    answer is the useful one, and it is what the other 21 filers rely on.
    """
    mod = importlib.import_module(f"tools.web_search_server.{modname}")
    monkeypatch.setenv("SEC_EMAIL", "test@example.invalid")

    from tools.web_search_server.sec_series import NotCovered

    def not_covered(*args, **kwargs):
        raise NotCovered("NVDA does not tag this concept")

    monkeypatch.setattr(mod, "fetch_concept_series", not_covered)
    result = getattr(mod, fname)("NVDA")

    assert result["success"] is False
    message = (result.get("error") or result.get("note") or "").lower()
    assert "503" not in message, (
        f"{modname}.{fname} reported a genuine absence as a fetch failure: "
        f"{message[:200]}")
    assert message, f"{modname}.{fname} gave no reason at all"
