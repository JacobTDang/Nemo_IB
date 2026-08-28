"""Layer 1 + Layer 2 tests for nemo_altdata tools.

Layer 1: call runner scripts directly via subprocess.run (fastest feedback,
         no MCP server needed).
Layer 2: test pure-Python helpers (MOPS parser, job postings, options math)
         by calling them as functions.

Tests that require live network (Google Trends, MOPS, Greenhouse) are marked
with @pytest.mark.network and skipped in CI if SKIP_NETWORK_TESTS=1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from testing._gates import skip_if_provider_unavailable

# Resolve paths relative to repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PY = os.path.join(_REPO, ".venv", "Scripts", "python.exe")
if not os.path.isfile(_VENV_PY):
    _VENV_PY = os.path.join(_REPO, ".venv", "bin", "python")
if not os.path.isfile(_VENV_PY):
    _VENV_PY = sys.executable


SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "0") == "1"


def network(func):
    """Apply the real `network` marker plus the offline skip.

    This name used to be bound to a bare pytest.mark.skipif. A skipif is not a
    registered marker, so `-m network` and `-m "not network"` collected nothing
    here -- the tests were selectable only by file path. Registering the marker
    as well makes the selection work; the skipif keeps the offline behaviour
    identical.
    """
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="network tests skipped")(func)


def _run(runner, tool_name, kwargs, timeout=60):
    proc = subprocess.run(
        [_VENV_PY, runner, tool_name, json.dumps(kwargs)],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    )
    return proc


# ---------------------------------------------------------------------------
# Google Trends runner — Layer 1
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Job postings — live network test
# ---------------------------------------------------------------------------

@network
def test_job_postings_greenhouse_stripe():
    # Stripe is a known Greenhouse customer (nvidia uses its own portal)
    from tools.altdata_server.server import _fetch_job_postings
    result = _fetch_job_postings("stripe", "greenhouse", None)
    assert "error" not in result, result.get("error")
    assert result["total"] > 0
    assert isinstance(result["by_department"], dict)
    assert result["ats"] in {"greenhouse", "lever"}


@network
def test_job_postings_unknown_slug_returns_error():
    from tools.altdata_server.server import _fetch_job_postings
    result = _fetch_job_postings("thisslugshouldnotexist99999xyz", "greenhouse", None)
    assert "error" in result


# ---------------------------------------------------------------------------
# Taiwan MOPS — live network test
# ---------------------------------------------------------------------------

@network
def test_taiwan_revenue_finmind_tsmc():
    from tools.altdata_server.server import _fetch_taiwan_revenue_finmind
    result = _fetch_taiwan_revenue_finmind(["2330"], months=3)
    tsmc = result["companies"].get("2330", {})
    assert "error" not in tsmc, f"FinMind returned error: {tsmc.get('error')}"
    assert tsmc["months_returned"] > 0
    assert tsmc["source"] == "finmind"
    last = tsmc["months"][-1]
    assert last["revenue_ntd_m"] is not None
    assert last["revenue_ntd_m"] > 0
    assert last["yoy_pct"] is not None  # FinMind returns enough history for YoY


@network
def test_taiwan_revenue_finmind_yoy_computed():
    from tools.altdata_server.server import _fetch_taiwan_revenue_finmind
    result = _fetch_taiwan_revenue_finmind(["2330"], months=6)
    tsmc = result["companies"].get("2330", {})
    assert "error" not in tsmc, tsmc.get("error")
    months_with_yoy = [m for m in tsmc["months"] if m["yoy_pct"] is not None]
    assert len(months_with_yoy) > 0, "expected at least one month with YoY computed"


# ---------------------------------------------------------------------------
# Fix B — Google Trends SQLite cache (Layer 2, no network)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Fix C — Job postings: Workday probe + ATS fingerprinting (Layer 2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 5 — job postings: discovery/fetch split + no thread leak
# ---------------------------------------------------------------------------

def test_workday_discovery_split_functions_exist():
    """Discovery (probes only) is separated from the full fetch."""
    from tools.altdata_server.server import (
        _discover_workday_tenant, _try_workday_discovery, _workday_fetch_full,
    )
    assert callable(_discover_workday_tenant)
    assert callable(_try_workday_discovery)
    assert callable(_workday_fetch_full)


def test_discover_workday_tenant_garbage_returns_none():
    """Discovery returns None (metadata absent) for an unknown tenant — no fetch."""
    from tools.altdata_server.server import _discover_workday_tenant
    assert _discover_workday_tenant("absolutelyfakecompany999xyz") is None


@network
def test_job_postings_no_thread_leak():
    """After a job-postings call returns, worker threads must drain back to
    ~baseline (the old design left the Workday fetch running past teardown)."""
    import threading
    import time
    from tools.altdata_server.server import _fetch_job_postings
    baseline = threading.active_count()
    result = _fetch_job_postings("stripe", "greenhouse", None)
    assert isinstance(result, dict)
    # Allow brief drain for any bounded probe threads to exit.
    deadline = time.time() + 12
    while threading.active_count() > baseline + 2 and time.time() < deadline:
        time.sleep(0.5)
    leaked = threading.active_count() - baseline
    assert leaked <= 2, f"thread leak: {leaked} threads above baseline"


def test_workday_probe_returns_none_on_404():
    """A nonsense tenant must return None, not raise."""
    from tools.altdata_server.server import _workday_probe
    result = _workday_probe("thisdoesnotexist999xyz", 5, "External_Career_Site")
    assert result is None


def test_workday_probe_returns_none_on_bad_wd_num():
    """A real company name with a non-existent wd number must not raise."""
    from tools.altdata_server.server import _workday_probe
    result = _workday_probe("stripe", 99, "External_Career_Site")
    assert result is None


def test_ats_fingerprint_pattern_greenhouse():
    """Regex patterns correctly identify Greenhouse from a mock HTML fragment."""
    import re
    from tools.altdata_server.server import _ATS_PATTERNS
    html = '<a href="https://boards.greenhouse.io/stripe/jobs/12345">Apply</a>'
    found = None
    for pattern, ats_type in _ATS_PATTERNS:
        if pattern.search(html):
            found = ats_type
            break
    assert found == "greenhouse", f"expected greenhouse, detected: {found}"


def test_ats_fingerprint_pattern_lever():
    """Regex patterns correctly identify Lever from a mock HTML fragment."""
    from tools.altdata_server.server import _ATS_PATTERNS
    html = '<a href="https://jobs.lever.co/openai/abc-123">Apply here</a>'
    found = None
    for pattern, ats_type in _ATS_PATTERNS:
        if pattern.search(html):
            found = ats_type
            break
    assert found == "lever"


def test_ats_fingerprint_pattern_workday():
    """Regex patterns correctly identify Workday from a mock HTML fragment."""
    from tools.altdata_server.server import _ATS_PATTERNS
    html = 'Redirect to https://oracle.wd5.myworkdayjobs.com/External_Career_Site for jobs'
    found = None
    for pattern, ats_type in _ATS_PATTERNS:
        if pattern.search(html):
            found = ats_type
            break
    assert found == "workday"


def test_job_postings_totally_unknown_slug_clean_error():
    """A completely made-up company returns a structured error dict, not a traceback."""
    from tools.altdata_server.server import _fetch_job_postings
    result = _fetch_job_postings("zxqthiscompanydoesnotexist9999", "greenhouse", None)
    assert "error" in result, "expected error key in result"
    assert "Traceback" not in result["error"]


@network
def test_workday_discovery_fails_cleanly_for_garbage_slug():
    """_try_workday_discovery times out cleanly for a garbage slug (no hang)."""
    from tools.altdata_server.server import _try_workday_discovery
    result = _try_workday_discovery("absolutelyfakecompanyname999xyz", None)
    assert result is None, "should return None for an unknown tenant"


# ---------------------------------------------------------------------------
# Capex announcement helpers — pure math (Layer 2, no network)
# ---------------------------------------------------------------------------

def test_extract_dollar_amounts_billion():
    from tools.altdata_server.server import _extract_dollar_amounts
    amounts = _extract_dollar_amounts("Company plans to invest $2.5 billion in new factory")
    assert len(amounts) == 1
    assert amounts[0] == pytest.approx(2_500_000_000)


def test_extract_dollar_amounts_trillion():
    from tools.altdata_server.server import _extract_dollar_amounts
    amounts = _extract_dollar_amounts("US government allocates $1.2 trillion for infrastructure")
    assert len(amounts) == 1
    assert amounts[0] == pytest.approx(1_200_000_000_000)


def test_extract_dollar_amounts_short_suffixes():
    from tools.altdata_server.server import _extract_dollar_amounts
    text = "Invested $500M in chips and raised $3B in funding"
    amounts = _extract_dollar_amounts(text)
    assert len(amounts) == 2
    assert 500_000_000 in amounts
    assert 3_000_000_000 in amounts


def test_extract_dollar_amounts_none_present():
    from tools.altdata_server.server import _extract_dollar_amounts
    amounts = _extract_dollar_amounts("Company announced new products without disclosing costs")
    assert amounts == []


def test_classify_capex_text_bullish():
    from tools.altdata_server.server import _classify_capex_text
    text = "Samsung will invest $17 billion in a new semiconductor factory in Texas"
    assert _classify_capex_text(text) == "bullish"


def test_classify_capex_text_bearish():
    from tools.altdata_server.server import _classify_capex_text
    text = "Intel will cancel construction of its Ohio chip plant and restructure operations"
    assert _classify_capex_text(text) == "bearish"


def test_classify_capex_text_neutral():
    from tools.altdata_server.server import _classify_capex_text
    text = "Company held its annual meeting and discussed quarterly earnings"
    assert _classify_capex_text(text) == "neutral"


def test_classify_capex_text_mixed_favors_majority():
    from tools.altdata_server.server import _classify_capex_text
    # More bullish keywords than bearish
    text = "Company expands, invests in new facility and announces new data center, despite cutting one old plant"
    result = _classify_capex_text(text)
    assert result == "bullish"


# ---------------------------------------------------------------------------
# Phase 2 — capex word-boundary classifier + direction-aware signal
# ---------------------------------------------------------------------------

def test_capex_classify_no_false_bearish_from_substring():
    """'circuit' must not trigger 'cut'; 'disclosed' must not trigger 'close'."""
    from tools.altdata_server.server import _classify_capex_text
    assert _classify_capex_text("executive discussed integrated circuit boards") == "neutral"
    assert _classify_capex_text("the company disclosed quarterly results") == "neutral"


def test_capex_classify_conjugations_match():
    from tools.altdata_server.server import _classify_capex_text
    assert _classify_capex_text("the firm invests heavily and expands output") == "bullish"
    assert _classify_capex_text("the firm cancelled the project and is divesting") == "bearish"


def test_capex_signal_bearish_major_not_bullish():
    """A cancelled $2B plant must read bearish, never bullish (the old bug)."""
    from tools.altdata_server.server import _capex_signal
    announcements = [
        {"direction": "bearish", "max_amount_usd": 2_000_000_000},
    ]
    assert _capex_signal(announcements) == "bearish"


def test_capex_signal_bullish_major():
    from tools.altdata_server.server import _capex_signal
    announcements = [
        {"direction": "bullish", "max_amount_usd": 5_000_000_000},
    ]
    assert _capex_signal(announcements) == "bullish"


def test_capex_signal_neutral_large_does_not_force_direction():
    """A $5B item with no direction verb stays neutral, not auto-bullish."""
    from tools.altdata_server.server import _capex_signal
    announcements = [
        {"direction": "neutral", "max_amount_usd": 5_000_000_000},
    ]
    assert _capex_signal(announcements) == "neutral"


def test_capex_signal_count_majority():
    from tools.altdata_server.server import _capex_signal
    announcements = [
        {"direction": "bullish", "max_amount_usd": 100_000_000},
        {"direction": "bullish", "max_amount_usd": 50_000_000},
        {"direction": "bullish", "max_amount_usd": 0},
        {"direction": "bearish", "max_amount_usd": 0},
    ]
    assert _capex_signal(announcements) == "bullish"  # 3 bull > 2x1 bear


def test_capex_name_tokens_drops_generic():
    from tools.altdata_server.server import _company_name_tokens
    assert _company_name_tokens("NextEra Energy") == ["nextera"]
    assert _company_name_tokens("Exxon Mobil Corporation") == ["exxon", "mobil"]
    assert "mcdonald" in _company_name_tokens("McDonald's Corporation")


def test_capex_relevance_filters_macro_headline():
    """Macro headline without the company name is dropped; on-topic kept."""
    from tools.altdata_server.server import _article_is_relevant, _company_name_tokens
    toks = _company_name_tokens("Exxon Mobil Corporation")
    assert _article_is_relevant("US secures $18T of investment in factories", toks) is False
    assert _article_is_relevant("Exxon to build $10B refinery expansion", toks) is True


def test_capex_relevance_no_tokens_keeps_all():
    from tools.altdata_server.server import _article_is_relevant
    # No distinctive tokens -> do not over-filter
    assert _article_is_relevant("any article text", []) is True


# ---------------------------------------------------------------------------
# New tools — live network tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 0 — gov-contracts signal logic (pure, no network)
# ---------------------------------------------------------------------------

def test_gov_signal_decline_is_not_bullish():
    """Regression guard: a YoY decline must NOT read bullish even at huge size.
    (The old >=$1B override force-bulled every megacap regardless of trend.)"""
    from tools.altdata_server.server import _gov_contracts_signal
    # $45B trailing, $49B prior, -8% YoY — exactly the LMT case that was wrong
    assert _gov_contracts_signal(45e9, 49e9, -8.0) == "neutral"
    assert _gov_contracts_signal(45e9, 60e9, -25.0) == "bearish"


def test_gov_signal_flat_band_is_neutral():
    from tools.altdata_server.server import _gov_contracts_signal
    for yoy in (-15.0, -10.0, 0.0, 10.0, 15.0):
        assert _gov_contracts_signal(50e9, 50e9, yoy) == "neutral", f"yoy={yoy}"


def test_gov_signal_growth_is_bullish():
    from tools.altdata_server.server import _gov_contracts_signal
    assert _gov_contracts_signal(60e9, 40e9, 50.0) == "bullish"


def test_gov_signal_tiny_is_not_applicable():
    from tools.altdata_server.server import _gov_contracts_signal
    assert _gov_contracts_signal(5_000_000, 0, None) == "not_applicable"


def test_gov_signal_new_business_is_bullish():
    from tools.altdata_server.server import _gov_contracts_signal
    # prior 0, trailing above floor, no computable YoY → newly winning business
    assert _gov_contracts_signal(50_000_000, 0, None) == "bullish"


def test_gov_signal_collapse_from_real_base_is_bearish():
    """Total collapse from a >=$10M prior base must read bearish — the tiny-
    absolute gate must not mask declines (the same masking class we removed
    on the upside)."""
    from tools.altdata_server.server import _gov_contracts_signal
    assert _gov_contracts_signal(0, 19_500_000, -100.0) == "bearish"
    # but both-tiny stays not_applicable
    assert _gov_contracts_signal(5_000_000, 2_000_000, None) == "not_applicable"


def test_gov_signal_prior_unknown_never_bullish():
    """A FAILED prior fetch (None) must not read as 'newly winning business'."""
    from tools.altdata_server.server import _gov_contracts_signal
    assert _gov_contracts_signal(50_000_000, None, None) == "neutral"
    assert _gov_contracts_signal(5_000_000, None, None) == "not_applicable"


def test_gov_windows_calendar_accurate_no_overlap():
    """12 months = a true 365-day year (not 360), and the prior window ends one
    day before the trailing window starts (inclusive API, no double count)."""
    from datetime import datetime
    from tools.altdata_server.server import _gov_windows
    t_start, end, p_start, p_end = _gov_windows(datetime(2026, 6, 5), 12)
    assert end == "2026-06-05"
    assert t_start == "2025-06-05"           # exactly one year
    assert p_end == "2025-06-04"             # day before trailing start
    assert p_start == "2024-06-05"
    assert p_end < t_start


def test_run_subprocess_tolerates_stray_stdout_lines(tmp_path):
    """A library printing to stdout before the JSON line must not break the
    runner protocol — the LAST line is the contract."""
    from tools.altdata_server.server import _run_subprocess
    stub = tmp_path / "stub_runner.py"
    stub.write_text(
        "import sys, json\n"
        "print('yfinance deprecation noise')\n"
        "print(json.dumps({'success': True, 'data': {'ok': 1}}))\n",
        encoding="utf-8")
    result = _run_subprocess(str(stub), "any_tool", {}, sub_timeout=15)
    assert result["success"] is True
    assert result["data"]["ok"] == 1


def _skip_on_usaspending_timeout(result):
    """Skip test if USASpending.gov was unreachable (flaky free API)."""
    if "error" in result:
        err = result["error"]
        if any(k in err for k in ("Timeout", "timeout", "ConnectError", "ConnectionError")):
            pytest.skip(f"USASpending.gov unavailable: {err[:100]}")


@network
def test_government_contracts_consumer_company_not_applicable():
    """A consumer goods company should have negligible federal contracts → not_applicable signal."""
    from tools.altdata_server.server import _fetch_government_contracts
    result = _fetch_government_contracts("MCD", "McDonald's Corporation", months=12, include_grants=False)
    _skip_on_usaspending_timeout(result)
    assert "error" not in result, result.get("error")
    assert result["signal"] in {"not_applicable", "neutral", "bullish", "bearish"}
    assert "trailing_awards_usd" in result
    assert result["source"] == "usaspending.gov"


@network
def test_government_contracts_has_required_fields():
    """Result always contains the required output schema fields."""
    from tools.altdata_server.server import _fetch_government_contracts
    result = _fetch_government_contracts("LMT", "Lockheed Martin Corporation", months=12, include_grants=False)
    _skip_on_usaspending_timeout(result)
    assert "error" not in result, result.get("error")
    required = ["company_name", "ticker", "period_months", "trailing_awards_usd",
                "trailing_award_count", "prior_period_awards_usd", "yoy_change_pct",
                "signal", "top_agencies", "source", "basis"]
    for field in required:
        assert field in result, f"missing field: {field}"
    assert result["signal"] in {"bullish", "bearish", "neutral", "not_applicable"}


@network
def test_government_contracts_defense_is_bullish_or_neutral():
    """Lockheed Martin (major defense contractor) should not be not_applicable."""
    from tools.altdata_server.server import _fetch_government_contracts
    result = _fetch_government_contracts("LMT", "Lockheed Martin Corporation", months=12, include_grants=False)
    _skip_on_usaspending_timeout(result)
    assert "error" not in result, result.get("error")
    assert result["trailing_awards_usd"] > 10_000_000, "LMT must have >$10M in trailing awards"
    assert result["signal"] != "not_applicable"


# ---------------------------------------------------------------------------
# Phase 3 — policy bill scoring (pure, no network)
# ---------------------------------------------------------------------------

def test_bill_score_banking_not_bearish():
    """'Banking Modernization Act' must not trigger the 'ban' bearish term."""
    from tools.altdata_server.server import _score_bill_title
    score = _score_bill_title("H.R. 100: Banking Modernization Act", "introduced")
    assert score >= 0, f"banking bill scored bearish: {score}"


def test_bill_score_investigation_not_double_counted():
    """'investigation' must score bearish only, not also bullish via 'invest'."""
    from tools.altdata_server.server import _score_bill_title
    score = _score_bill_title("S. 50: Investigation into Drug Pricing Act", "introduced")
    assert score < 0, f"investigation bill should be net bearish, got {score}"


def test_bill_score_ban_whole_word_is_bearish():
    from tools.altdata_server.server import _score_bill_title
    score = _score_bill_title("H.R. 7: Ban on Chip Exports Act", "introduced")
    assert score < 0


def test_bill_score_bullish_terms():
    from tools.altdata_server.server import _score_bill_title
    score = _score_bill_title("S. 9: Semiconductor Manufacturing Investment Incentive Act", "introduced")
    assert score > 0


def test_bill_status_weighting():
    from tools.altdata_server.server import _score_bill_title
    enacted = _score_bill_title("Ban on Exports Act", "enacted_signed")
    introduced = _score_bill_title("Ban on Exports Act", "introduced")
    assert abs(enacted) > abs(introduced)  # enacted weighted far higher


def test_bill_status_case_insensitive():
    from tools.altdata_server.server import _score_bill_title
    a = _score_bill_title("Investment Incentive Act", "ENACTED_SIGNED")
    b = _score_bill_title("Investment Incentive Act", "enacted_signed")
    assert a == b and a != 0


def test_sector_keyword_coverage_all_gics():
    """Regression guard: every yfinance GICS sector must have bill keywords."""
    from tools.altdata_server.server import SECTOR_BILL_KEYWORDS
    gics = {
        "Technology", "Basic Materials", "Communication Services",
        "Consumer Cyclical", "Consumer Defensive", "Energy",
        "Financial Services", "Healthcare", "Industrials",
        "Real Estate", "Utilities",
    }
    missing = gics - set(SECTOR_BILL_KEYWORDS)
    assert not missing, f"sectors missing bill keywords: {missing}"


@network
def test_policy_signals_returns_required_fields():
    """Policy signals for any ticker must return the required schema fields."""
    from tools.altdata_server.server import _fetch_policy_signals
    result = _fetch_policy_signals("NVDA", "Technology", lookback_days=180)
    skip_if_provider_unavailable(result, "GovTrack")
    assert "error" not in result, result.get("error")
    required = ["ticker", "sector", "signal", "bill_count", "bills"]
    for field in required:
        assert field in result, f"missing field: {field}"
    assert result["signal"] in {"bullish", "bearish", "neutral", "data_gap", None}
    assert isinstance(result["bills"], list)

    # Regression guard: GovTrack dedup must actually return bills (the old
    # obj.get("id") key was always None and silently dropped every bill).
    #
    # A bare `bill_count > 0` could not tell that regression from GovTrack
    # simply answering with nothing, so a flaky provider failed as a code
    # change (issue #18). The fields added there separate the two:
    # `provider_rows_returned` is what GovTrack handed over, before dedup and
    # before the lookback filter, and `bills_before_lookback_filter` is what
    # survived dedup. The guard is now made against those, and the count is
    # only asserted over a provider that actually sent something.
    if result["provider_rows_returned"] == 0:
        pytest.skip("GovTrack returned no bill for any keyword: there is "
                    "nothing for dedup to have dropped")
    assert result["bills_before_lookback_filter"] > 0, (
        f"GovTrack sent {result['provider_rows_returned']} rows and dedup "
        f"kept none of them — dedup/fetch regression")
    if result["bill_count"] == 0:
        pytest.skip("every bill GovTrack returned predates the lookback "
                    "window; dedup is intact")
    assert result["bill_count"] > 0, "no bills found — dedup/fetch regression"


@network
def test_policy_signals_uses_govtrack_without_api_key(monkeypatch):
    """Without CONGRESS_API_KEY, must fall back to GovTrack without error."""
    import os
    monkeypatch.delitem(os.environ, "CONGRESS_API_KEY", raising=False)
    from tools.altdata_server.server import _fetch_policy_signals
    result = _fetch_policy_signals("MSFT", "Technology", lookback_days=180)
    skip_if_provider_unavailable(result, "GovTrack")
    # Should not error even without API key
    assert "error" not in result, result.get("error")
    # `None` is the answer when GovTrack returned nothing at all: a provider
    # that said nothing is not a neutral legislative climate (issue #18).
    assert result["signal"] in {"bullish", "bearish", "neutral", "data_gap", None}


@network
def test_capex_announcements_returns_required_fields():
    """Capex announcements for a major semiconductor company must return expected structure."""
    from tools.altdata_server.server import _fetch_capex_announcements
    result = _fetch_capex_announcements("TSMC", "Taiwan Semiconductor Manufacturing", lookback_days=180)
    assert "error" not in result, result.get("error")
    required = ["ticker", "company_name", "lookback_days", "announcement_count",
                "capex_total_usd", "capex_total_basis", "figures",
                "amounts_by_category", "signal", "announcements"]
    for field in required:
        assert field in result, f"missing field: {field}"
    assert result["signal"] in {"bullish", "bearish", "neutral", "data_gap"}


@network
def test_capex_announcements_semiconductor_has_activity():
    """TSMC or Intel should show capex news in the past 180 days."""
    from tools.altdata_server.server import _fetch_capex_announcements
    result = _fetch_capex_announcements("INTC", "Intel Corporation", lookback_days=180)
    assert "error" not in result
    # Intel is heavily covered for fab/capex news — should find at least some articles
    # (allow data_gap only if truly no news was returned)
    if result["signal"] != "data_gap":
        assert result["announcement_count"] >= 1


# ---------------------------------------------------------------------------
# Policy signals: the Congress.gov credential, and telling an empty provider
# from a filtered-out one. Issues #59 (section 1) and #18.
#
# Every test below stubs `requests.get`. GovTrack and Congress.gov are never
# reached: the behaviour under test is what our own code does with what a
# provider hands back, and pinning that to a live upstream would make it a
# weather report.
# ---------------------------------------------------------------------------

# Shaped like a Congress.gov key -- 40 characters, alphanumeric -- but typed
# out here, so anything matching it could only have come from this file. The
# real key is never read: a test that loads the live credential in order to
# prove it does not escape is itself the escape.
#
# Deliberately free of English words. `_leaked_fragment` below hunts for short
# runs of it, and a sentinel containing "congress" would match the word in
# every honest response.
_CONGRESS_SENTINEL = "kx7q2vzhr9m4tbn6wjd8scfl3pgy5aue1oi0xzqv"


class _FakeResponse:
    """The two pieces of a requests.Response the bill fetchers touch."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _leaked_fragment(text, secret, minimum=4):
    """The longest run of `secret` of at least `minimum` characters that
    appears in `text`, or "".

    Asserting only on the whole credential is the mistake the 120-character
    truncation already made: a cut that keeps three characters of the key has
    still disclosed three characters of the key.
    """
    for size in range(len(secret), minimum - 1, -1):
        for start in range(0, len(secret) - size + 1):
            if secret[start:start + size] in text:
                return secret[start:start + size]
    return ""


def _raising_get(url, params=None, **kwargs):
    """requests.get that fails the way raise_for_status() does.

    The message that method builds renders the whole request URL, query string
    included -- which is where the Congress.gov key travels.
    """
    import requests

    query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    raise requests.exceptions.HTTPError(
        f"403 Client Error: Forbidden for url: {url}?{query}")


def test_the_congress_fetcher_takes_no_bare_string_credential():
    """The signature itself, so the parameter cannot come back under another
    name. pytest prints a frame's arguments at the head of every traceback
    entry, so a rendered credential parameter is a disclosure on the first
    failure anywhere below it."""
    import inspect
    from tools.altdata_server import server as srv

    params = inspect.signature(srv._congress_api_fetch_bills).parameters
    assert "api_key" not in params, \
        "_congress_api_fetch_bills grew a bare-string credential parameter again"
    annotations = [str(p.annotation) for p in params.values()]
    assert any("Secret" in a for a in annotations), \
        "the credential parameter is no longer typed as Secret"


@pytest.mark.parametrize("keyword", ["chips", "semiconductor export controls"])
def test_a_congress_gov_failure_never_reports_the_key(monkeypatch, keyword):
    """Issue #59: the key is a query parameter, so `raise_for_status()` builds
    a message with it in the middle, and that message was truncated to 120
    characters and returned to the caller in `partial_errors`.

    The truncation was never a guard. Measured on the real URL shape, the key
    starts at character 111 with the keyword "chips" (inside the cut) and at
    135 with a longer one (outside it), so whether the credential shipped
    depended on the caller's search term. Both keywords are exercised here.
    """
    import requests
    from tools.altdata_server import server as srv

    monkeypatch.setattr(requests, "get", _raising_get)
    bills, errors, queried, provider = srv._congress_api_fetch_bills(
        [keyword], 119, srv.Secret(_CONGRESS_SENTINEL))

    assert bills == []
    assert errors, "the failure was swallowed"
    reported = " ".join(errors)
    assert not _leaked_fragment(reported, _CONGRESS_SENTINEL), \
        "part of the credential reached the returned error"
    assert "api_key" not in reported and "https://" not in reported, \
        "the request URL is still in the error, so the key is one edit away"
    assert "HTTPError" in reported and keyword in reported, \
        "scrubbing ate the diagnosis"


def test_a_policy_signals_response_never_carries_the_key(monkeypatch):
    """The end of the path: `partial_errors` is returned to the MCP caller."""
    import requests
    from tools.altdata_server import server as srv

    monkeypatch.setenv("CONGRESS_API_KEY", _CONGRESS_SENTINEL)
    monkeypatch.setattr(requests, "get", _raising_get)

    result = srv._fetch_policy_signals("NVDA", "Technology", lookback_days=180)

    assert result["success"] is False
    assert result["reason"] == "provider_unavailable"
    rendered = json.dumps(result)
    assert not _leaked_fragment(rendered, _CONGRESS_SENTINEL), \
        "part of the credential reached the caller"
    assert "api_key" not in rendered


@pytest.mark.parametrize(
    "render",
    [repr, str, lambda s: f"{s}", lambda s: f"{s!r}", lambda s: str([s]),
     lambda s: str({"key": s})],
    ids=["repr", "str", "fstring", "fstring_r", "in_list", "in_dict"],
)
def test_no_way_of_rendering_the_credential_shows_its_value(render):
    from tools.altdata_server.server import Secret

    rendered = render(Secret(_CONGRESS_SENTINEL))
    assert not _leaked_fragment(rendered, _CONGRESS_SENTINEL), \
        "part of the value survived rendering"


def test_the_credential_type_matches_the_one_it_was_copied_from():
    """`Secret` is defined twice on purpose, so this pins the copy to the
    original. Two implementations of a redaction primitive that drift apart
    are worse than one, and the divergence would only show up in the leak."""
    from agent.openrouter_template import Secret as Shared
    from tools.altdata_server.server import Secret as Local

    assert Local(_CONGRESS_SENTINEL).reveal() == Shared(_CONGRESS_SENTINEL).reveal()
    assert repr(Local(_CONGRESS_SENTINEL)) == repr(Shared(_CONGRESS_SENTINEL))
    assert bool(Local("")) is bool(Shared("")) is False
    echoed = f"401: key {_CONGRESS_SENTINEL} rejected"
    assert (Local(_CONGRESS_SENTINEL).scrub(echoed)
            == Shared(_CONGRESS_SENTINEL).scrub(echoed))


def test_the_server_binds_no_unwrapped_congress_credential():
    """Under --showlocals a local renders exactly like a parameter, so moving
    the key out of the signature and into a variable would have looked like a
    fix and disclosed the same value.

    Narrow on purpose: only CONGRESS_API_KEY. The other credential sites named
    in issue #59 live in other modules and are not this change.
    """
    import ast

    from tools.altdata_server import server as srv

    source = ast.parse(open(srv.__file__).read())
    offenders = []
    for assignment in ast.walk(source):
        if not isinstance(assignment, ast.Assign):
            continue
        wrapped = set()
        for node in ast.walk(assignment.value):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "Secret"):
                wrapped.update(id(n) for n in ast.walk(node))
        for node in ast.walk(assignment.value):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "getenv")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "CONGRESS_API_KEY"
                    and id(node) not in wrapped):
                offenders.append(assignment.lineno)

    assert not offenders, (
        f"CONGRESS_API_KEY is assigned unwrapped in {srv.__file__} at lines "
        f"{offenders}. Wrap it in Secret(...) at the read.")


# --------------------------------------------- an empty provider, issue #18

def _govtrack_payload(objects, total_count=None):
    return {"objects": objects,
            "meta": {"total_count": total_count
                     if total_count is not None else len(objects)}}


def _govtrack_bill(bill_id, title, activity_date):
    return {"title": title, "short_title": "", "current_status": "introduced",
            "introduced_date": activity_date,
            "current_status_date": activity_date,
            "link": f"https://www.govtrack.us/congress/bills/{bill_id}"}


def test_an_empty_provider_is_not_reported_as_a_neutral_climate(monkeypatch):
    """Issue #18: GovTrack answering 200-with-zero for every keyword produced
    `signal: "neutral"`, which a consumer reads as an affirmative claim about
    the legislative climate. The provider said nothing; that is not a finding
    that nothing is happening."""
    import requests
    from tools.altdata_server import server as srv

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _FakeResponse(_govtrack_payload([])))

    result = srv._fetch_policy_signals("NVDA", "Technology", lookback_days=180)

    assert result["success"] is True
    assert result["signal"] is None, \
        "a provider that returned nothing was reported as a neutral climate"
    assert result["provider_rows_returned"] == 0
    assert result["bills_before_lookback_filter"] == 0
    assert result["partial_errors"] == []
    assert any("returned no bill" in d for d in result["degraded"]), \
        f"the empty provider was not named in degraded: {result['degraded']}"


def test_a_filtered_out_result_is_distinguishable_from_an_empty_provider(
        monkeypatch):
    """The other half of the pair. Forty bills, all older than the lookback
    window, is a real answer about a real legislative record -- and it used to
    produce a response byte-identical to the one above."""
    import requests
    from tools.altdata_server import server as srv

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    # Four distinct bills, so the fetcher does not fall through to the prior
    # congress and the counts below describe one round.
    stale = [_govtrack_bill(i, f"An Old Act No {i}", "2019-01-15")
             for i in range(4)]
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _FakeResponse(_govtrack_payload(stale)))

    result = srv._fetch_policy_signals("NVDA", "Technology", lookback_days=30)

    assert result["success"] is True
    assert result["bill_count"] == 0
    assert result["signal"] == "neutral"
    assert result["provider_rows_returned"] > 0
    assert result["bills_before_lookback_filter"] > 0, \
        "the bills the lookback filter removed were not counted anywhere"


def test_the_provider_row_count_survives_the_lookback_filter(monkeypatch):
    """`meta.total_count` is what GovTrack says matched, which is not what it
    sent: the per-keyword limit caps the page. Both travel, so a caller can
    tell 'nothing matched' from 'more matched than fitted'."""
    import requests
    from tools.altdata_server import server as srv

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    recent = [_govtrack_bill(i, f"A Semiconductor Investment Act {i}",
                             "2026-08-01") for i in range(3)]
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse(_govtrack_payload(recent, total_count=137)))

    result = srv._fetch_policy_signals("NVDA", "Technology", lookback_days=180)

    keywords = len(result["keywords_searched"])
    assert result["provider_rows_returned"] == 3 * keywords, \
        "the rows the provider actually sent were not counted"
    assert result["provider_total_count"] == 137 * keywords
    assert result["bills_before_lookback_filter"] == 3, \
        "the same three bills on every keyword must dedup to three"
    assert result["bill_count"] == 3


def test_companion_bills_sharing_a_title_are_counted_separately(monkeypatch):
    """Issue #18: congress.gov rows hard-coded `link` to "", so dedup fell back
    to the title and a House bill and its Senate companion -- which share one
    -- were silently collapsed into one row."""
    import requests
    from tools.altdata_server import server as srv

    payload = {"bills": [
        {"title": "CHIPS and Science Act of 2026",
         "url": "https://api.congress.gov/v3/bill/119/hr/1234",
         "latestAction": {"actionDate": "2026-08-01", "text": "Referred"}},
        {"title": "CHIPS and Science Act of 2026",
         "url": "https://api.congress.gov/v3/bill/119/s/567",
         "latestAction": {"actionDate": "2026-08-02", "text": "Referred"}},
    ]}
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _FakeResponse(payload))

    bills, errors, queried, provider = srv._congress_api_fetch_bills(
        ["chips"], 119, srv.Secret(_CONGRESS_SENTINEL))

    assert errors == []
    assert len(bills) == 2, \
        "the House bill and its Senate companion were collapsed into one"
    assert {b["link"] for b in bills} == {
        "https://api.congress.gov/v3/bill/119/hr/1234",
        "https://api.congress.gov/v3/bill/119/s/567"}
