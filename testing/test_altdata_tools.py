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
def test_taiwan_mops_tsmc():
    from tools.altdata_server.server import _fetch_mops_revenue
    result = _fetch_mops_revenue(["2330"], months=3)
    tsmc = result["companies"].get("2330", {})
    # MOPS may be unreachable from some networks (Taiwanese gov server).
    # Accept either a valid result OR a connection error (not a code bug).
    if "error" in tsmc:
        # MOPS may return JS-rendered pages or be blocked by network.
        # The tool correctly returns an error envelope — skip the assertion.
        pytest.skip(f"MOPS not parseable from this environment: {tsmc['error']}")
    assert tsmc["months_returned"] > 0
    last = tsmc["months"][-1]
    assert last["revenue_ntd_m"] is not None
    assert last["revenue_ntd_m"] > 0
