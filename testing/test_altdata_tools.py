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
    assert result["signal"] in {"bullish", "bearish", "neutral", "data_gap"}
    assert isinstance(result["bills"], list)
    # Regression guard: GovTrack dedup must actually return bills (the old
    # obj.get("id") key was always None and silently dropped every bill).
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
    assert result["signal"] in {"bullish", "bearish", "neutral", "data_gap"}


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
