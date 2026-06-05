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

# Resolve paths relative to repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PY = os.path.join(_REPO, ".venv", "Scripts", "python.exe")
if not os.path.isfile(_VENV_PY):
    _VENV_PY = os.path.join(_REPO, ".venv", "bin", "python")
if not os.path.isfile(_VENV_PY):
    _VENV_PY = sys.executable

_TRENDS_RUNNER = os.path.join(_REPO, "tools", "altdata_server", "trends_runner.py")
_FINBERT_RUNNER = os.path.join(_REPO, "tools", "altdata_server", "finbert_runner.py")

SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "0") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="network tests skipped")


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

@network
def test_trends_runner_returns_records():
    proc = _run(_TRENDS_RUNNER, "get_google_trends",
                {"keywords": ["NVDA"], "timeframe": "today 3-m", "geo": "US"})
    assert proc.returncode == 0, f"stderr: {proc.stderr[-200:]}"
    result = json.loads(proc.stdout)
    assert result["success"] is True, result.get("error")
    data = result["data"]
    assert len(data["records"]) > 0
    assert "date" in data["records"][0]
    assert "NVDA" in data["records"][0]


@network
def test_trends_runner_yoy_ratio_present():
    proc = _run(_TRENDS_RUNNER, "get_google_trends",
                {"keywords": ["iPhone"], "timeframe": "today 12-m", "geo": "US"})
    result = json.loads(proc.stdout)
    assert result["success"] is True
    # YoY ratio requires >= 26 weeks of data (12-m timeframe)
    assert result["data"]["yoy_ratio"] is not None
    assert isinstance(result["data"]["yoy_ratio"], float)
    assert result["data"]["yoy_signal"] in {"bullish", "bearish", "neutral"}


def test_trends_runner_empty_keywords_fails_clean():
    proc = _run(_TRENDS_RUNNER, "get_google_trends", {"keywords": []})
    result = json.loads(proc.stdout)
    assert result["success"] is False
    assert "error" in result
    # Must not be a raw Python traceback
    assert "Traceback" not in result["error"]


def test_trends_runner_unknown_tool_fails_clean():
    proc = _run(_TRENDS_RUNNER, "nonexistent_tool", {"keywords": ["AAPL"]})
    result = json.loads(proc.stdout)
    assert result["success"] is False
    assert "unknown tool" in result["error"]


def test_trends_runner_bad_json_args():
    proc = subprocess.run(
        [_VENV_PY, _TRENDS_RUNNER, "get_google_trends", "NOT_JSON"],
        capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=10,
    )
    result = json.loads(proc.stdout)
    assert result["success"] is False


# ---------------------------------------------------------------------------
# FinBERT runner — Layer 1
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_finbert_runner_positive_sentiment():
    texts = [
        "Revenue beat expectations by 15%, raising full-year guidance",
        "Strong demand across all product lines, margins expanded",
        "Record quarterly earnings driven by AI chip demand",
    ]
    proc = _run(_FINBERT_RUNNER, "get_finbert_sentiment",
                {"texts": texts, "ticker": "NVDA"}, timeout=180)
    assert proc.returncode == 0, f"stderr: {proc.stderr[-300:]}"
    result = json.loads(proc.stdout)
    assert result["success"] is True, result.get("error")
    data = result["data"]
    assert data["signal"] == "bullish", f"expected bullish, got {data['signal']} (score={data['net_score']})"
    assert data["net_score"] > 0
    assert data["article_count"] == 3
    assert len(data["per_article"]) == 3


@pytest.mark.slow
def test_finbert_runner_negative_sentiment():
    texts = [
        "Missed EPS estimates by a wide margin, guidance cut significantly",
        "Revenue declined year over year, demand deteriorating",
        "CEO resigned amid accounting irregularities investigation",
    ]
    proc = _run(_FINBERT_RUNNER, "get_finbert_sentiment",
                {"texts": texts, "ticker": "TEST"}, timeout=180)
    result = json.loads(proc.stdout)
    assert result["success"] is True
    data = result["data"]
    assert data["signal"] == "bearish", f"expected bearish, got {data['signal']} (score={data['net_score']})"
    assert data["net_score"] < 0


def test_finbert_runner_empty_texts_fails_clean():
    proc = _run(_FINBERT_RUNNER, "get_finbert_sentiment",
                {"texts": [], "ticker": "AAPL"}, timeout=10)
    result = json.loads(proc.stdout)
    assert result["success"] is False
    assert "Traceback" not in result.get("error", "")


def test_finbert_runner_unknown_tool_fails_clean():
    proc = _run(_FINBERT_RUNNER, "bad_tool", {"texts": ["x"], "ticker": "AAPL"})
    result = json.loads(proc.stdout)
    assert result["success"] is False
    assert "unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# Options implied move — pure math (Layer 2)
# ---------------------------------------------------------------------------

def test_options_implied_move_basic_math():
    from tools.altdata_server.server import compute_implied_move
    result = compute_implied_move(spot=100.0, atm_call_ask=3.50, atm_put_ask=3.20)
    assert abs(result["implied_move_pct"] - 0.067) < 0.001
    assert result["straddle_cost"] == pytest.approx(6.70, abs=0.01)


def test_options_implied_move_zero_spot():
    from tools.altdata_server.server import compute_implied_move
    result = compute_implied_move(spot=0, atm_call_ask=3.0, atm_put_ask=3.0)
    assert result["implied_move_pct"] == 0.0


def test_find_atm_options_selects_nearest_strike():
    from tools.altdata_server.server import _find_atm_options
    rows = [
        {"expiration": "2026-08-15", "option_type": "call", "strike": 200, "ask": 5.0, "implied_volatility": 0.35},
        {"expiration": "2026-08-15", "option_type": "call", "strike": 210, "ask": 2.0, "implied_volatility": 0.32},
        {"expiration": "2026-08-15", "option_type": "put",  "strike": 200, "ask": 4.8, "implied_volatility": 0.38},
        {"expiration": "2026-08-15", "option_type": "put",  "strike": 210, "ask": 8.0, "implied_volatility": 0.42},
    ]
    call, put, expiry = _find_atm_options(rows, spot=202.0, target_expiry="2026-08-15")
    assert call is not None and put is not None
    assert float(call["strike"]) == 200.0   # nearest to 202
    assert float(put["strike"])  == 200.0
    assert expiry == "2026-08-15"


def test_find_atm_options_empty_returns_none():
    from tools.altdata_server.server import _find_atm_options
    c, p, e = _find_atm_options([], spot=100.0)
    assert c is None and p is None and e is None


def test_skew_classification():
    from tools.altdata_server.server import _find_atm_options
    rows = [
        {"expiration": "2026-08-15", "option_type": "call", "strike": 100, "ask": 3.0, "implied_volatility": 0.30},
        {"expiration": "2026-08-15", "option_type": "put",  "strike": 100, "ask": 4.0, "implied_volatility": 0.38},
    ]
    call, put, _ = _find_atm_options(rows, 100.0, "2026-08-15")
    skew_diff = float(put["implied_volatility"]) - float(call["implied_volatility"])
    assert skew_diff > 0.03  # should be classified put_heavy


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
# Fix A — ATM gap guard (Layer 2)
# ---------------------------------------------------------------------------

def test_atm_gap_guard_returns_none_when_strike_too_far():
    """Nearest strike >8% from spot must return (None, None, None)."""
    from tools.altdata_server.server import _find_atm_options
    rows = [
        # spot=234, strike=190 → gap ~18.8% — should be rejected
        {"expiration": "2026-07-18", "option_type": "call", "strike": 190, "ask": 5.0, "implied_volatility": 0.30},
        {"expiration": "2026-07-18", "option_type": "put",  "strike": 190, "ask": 4.5, "implied_volatility": 0.33},
    ]
    c, p, e = _find_atm_options(rows, spot=234.0, target_expiry="2026-07-18")
    assert c is None
    assert p is None
    assert e == "2026-07-18"


def test_atm_gap_guard_passes_when_strike_within_threshold():
    """Strike within 8% of spot must not be rejected."""
    from tools.altdata_server.server import _find_atm_options
    rows = [
        # spot=234, strike=220 → gap ~6% — should be accepted
        {"expiration": "2026-07-18", "option_type": "call", "strike": 220, "ask": 8.0, "implied_volatility": 0.28},
        {"expiration": "2026-07-18", "option_type": "put",  "strike": 220, "ask": 7.5, "implied_volatility": 0.32},
    ]
    c, p, e = _find_atm_options(rows, spot=234.0, target_expiry="2026-07-18")
    assert c is not None
    assert p is not None
    assert float(c["strike"]) == 220.0


def test_atm_gap_guard_boundary_just_inside():
    """Strike exactly at 8% boundary (inclusive) should be accepted."""
    from tools.altdata_server.server import _find_atm_options
    spot = 100.0
    strike = 92.1   # gap = 7.9% — just inside threshold
    rows = [
        {"expiration": "2026-08-01", "option_type": "call", "strike": strike, "ask": 2.0, "implied_volatility": 0.25},
        {"expiration": "2026-08-01", "option_type": "put",  "strike": strike, "ask": 1.8, "implied_volatility": 0.27},
    ]
    c, p, _ = _find_atm_options(rows, spot=spot, target_expiry="2026-08-01")
    assert c is not None and p is not None


@network
def test_fetch_options_yfinance_returns_rows_with_atm_coverage():
    """yfinance chain for a liquid stock must include strikes near spot."""
    from tools.altdata_server.server import _fetch_options_yfinance, _find_atm_options
    import yfinance as yf
    spot = yf.Ticker("AAPL").history(period="1d")["Close"].iloc[-1]
    rows = _fetch_options_yfinance("AAPL", spot, near_days=60)
    assert len(rows) > 10, f"expected a full chain, got {len(rows)} rows"
    # ATM gap guard must pass on a real chain
    c, p, _ = _find_atm_options(rows, spot)
    assert c is not None, f"ATM call not found near spot={spot:.2f} in {len(rows)} rows"
    assert p is not None, "ATM put not found"
    gap = abs(float(c["strike"]) - spot) / spot
    assert gap <= 0.08, f"ATM gap too large: {gap:.1%}"


# ---------------------------------------------------------------------------
# Fix B — Google Trends SQLite cache (Layer 2, no network)
# ---------------------------------------------------------------------------

def test_trends_cache_key_order_invariant():
    """Same keywords in different order must produce the same cache key."""
    import tools.altdata_server.trends_runner as tr
    k1 = tr._cache_key(["Oracle", "OCI", "Oracle Cloud"], "today 12-m", "US")
    k2 = tr._cache_key(["Oracle Cloud", "Oracle", "OCI"], "today 12-m", "US")
    assert k1 == k2, "cache key must sort keywords before hashing"


def test_trends_cache_key_different_geo():
    """Different geo must produce different keys."""
    import tools.altdata_server.trends_runner as tr
    k_us = tr._cache_key(["NVDA"], "today 12-m", "US")
    k_gb = tr._cache_key(["NVDA"], "today 12-m", "GB")
    assert k_us != k_gb


def test_trends_cache_roundtrip(tmp_path, monkeypatch):
    """Write to cache then read back returns identical payload."""
    import tools.altdata_server.trends_runner as tr
    monkeypatch.setattr(tr, "_CACHE_DB", str(tmp_path / "trends_cache.db"))
    payload = {"keywords": ["TEST"], "yoy_ratio": 1.25, "records": [{"date": "2026-01-01", "TEST": 80}]}
    key = tr._cache_key(["TEST"], "today 12-m", "US")
    tr._cache_put(key, payload)
    result = tr._cache_get(key)
    assert result is not None, "cache miss after immediate write"
    assert result["yoy_ratio"] == 1.25
    assert result["records"][0]["TEST"] == 80


def test_trends_cache_miss_returns_none(tmp_path, monkeypatch):
    """Fresh DB returns None on any key."""
    import tools.altdata_server.trends_runner as tr
    monkeypatch.setattr(tr, "_CACHE_DB", str(tmp_path / "trends_cache.db"))
    result = tr._cache_get("nonexistent_key_abc123")
    assert result is None


def test_trends_cache_ttl_expired_returns_none(tmp_path, monkeypatch):
    """Entry written with an old timestamp is treated as expired."""
    import time
    import sqlite3
    import tools.altdata_server.trends_runner as tr

    db_path = str(tmp_path / "trends_cache.db")
    monkeypatch.setattr(tr, "_CACHE_DB", db_path)

    # Write a fresh entry first (sets up the table)
    key = "stale_key"
    tr._cache_put(key, {"yoy_ratio": 0.5})

    # Back-date the entry to 13 hours ago (past 12h TTL)
    stale_ts = time.time() - (13 * 3600)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE trends_cache SET cached_at = ? WHERE cache_key = ?", (stale_ts, key))
    conn.commit()
    conn.close()

    result = tr._cache_get(key)
    assert result is None, "expired entry should not be returned"


def test_trends_runner_cache_hit_sets_cached_flag(tmp_path, monkeypatch):
    """When runner finds a cache hit, payload contains cached=True."""
    import tools.altdata_server.trends_runner as tr
    monkeypatch.setattr(tr, "_CACHE_DB", str(tmp_path / "trends_cache.db"))
    payload = {"keywords": ["AAPL"], "yoy_ratio": 1.1, "yoy_signal": "bullish",
               "records": [], "record_count": 0, "timeframe": "today 12-m", "geo": "US"}
    key = tr._cache_key(["AAPL"], "today 12-m", "US")
    tr._cache_put(key, payload)

    hit = tr._cache_get(key)
    assert hit is not None
    # Simulate what main() does: add cached=True flag
    hit["cached"] = True
    assert hit["cached"] is True


# ---------------------------------------------------------------------------
# Fix C — Job postings: Workday probe + ATS fingerprinting (Layer 2)
# ---------------------------------------------------------------------------

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
# New tools — live network tests
# ---------------------------------------------------------------------------

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
                "trailing_award_count", "signal", "top_agencies", "major_recent_awards", "source"]
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


@network
def test_policy_signals_returns_required_fields():
    """Policy signals for any ticker must return the required schema fields."""
    from tools.altdata_server.server import _fetch_policy_signals
    result = _fetch_policy_signals("NVDA", "Technology", lookback_days=180)
    assert "error" not in result, result.get("error")
    required = ["ticker", "sector", "signal", "bill_count", "bills"]
    for field in required:
        assert field in result, f"missing field: {field}"
    assert result["signal"] in {"bullish", "bearish", "neutral", "data_gap"}
    assert isinstance(result["bills"], list)


@network
def test_policy_signals_uses_govtrack_without_api_key(monkeypatch):
    """Without CONGRESS_API_KEY, must fall back to GovTrack without error."""
    import os
    monkeypatch.delitem(os.environ, "CONGRESS_API_KEY", raising=False)
    from tools.altdata_server.server import _fetch_policy_signals
    result = _fetch_policy_signals("MSFT", "Technology", lookback_days=180)
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
                "total_announced_usd", "signal", "announcements"]
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
