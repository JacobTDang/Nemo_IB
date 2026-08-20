"""Temporary: assert the merged get_options_metrics agrees with the altdata
implementation it replaces. Deleted once the old path is removed."""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

pytestmark = pytest.mark.network

if os.environ.get("SKIP_NETWORK_TESTS") == "1":
    pytest.skip("network tests disabled", allow_module_level=True)

TICKERS = ["MSFT", "AAPL", "NVDA"]


@pytest.mark.parametrize("ticker", TICKERS)
def test_merged_matches_altdata(ticker):
    from tools.altdata_server.options_runner import fetch_options_chain
    from tools.altdata_server.server import (
        _find_atm_options as old_find,
        _leg_price as old_leg,
        compute_implied_move as old_move,
    )
    from tools.financial_modeling_engine.utils import get_options_metrics

    new = get_options_metrics(ticker)
    assert new.get("success"), f"{ticker}: merged tool failed: {new.get('error')}"
    spot = new["spot_price"]

    # Refuse to pass on sentinel data. yfinance serves bid=ask=0 and IV
    # sentinels outside market hours; two implementations reading the same
    # degraded feed agree trivially. Agreement is only evidence when the
    # underlying quotes are real.
    ts = new.get("term_structure", {})
    ivs = [v.get("atm_iv") for v in ts.values()
           if isinstance(v, dict) and v.get("atm_iv") is not None]
    assert ivs, f"{ticker}: no ATM IV in term structure — cannot verify"
    assert max(ivs) > 0.08, (
        f"{ticker}: IV sentinel data (max atm_iv={max(ivs):.4f}) — run this test "
        f"during US market hours (09:30-16:00 ET). Agreement on sentinel quotes "
        f"is not evidence the merge is correct."
    )
    assert len(set(round(v, 6) for v in ivs)) > 1, (
        f"{ticker}: identical ATM IV {ivs[0]} across all tenors — sentinel feed, "
        f"not a real term structure."
    )

    merged = new["implied_move"]
    assert "error" not in merged, f"{ticker}: merged implied_move errored: {merged}"

    # The two implementations pick front expiry by DIFFERENT policies, and that
    # difference is deliberate: the merged tool reuses the term structure's 7d
    # bucket (floored at 7 DTE), while the old one took the nearest listed expiry
    # with no floor. On a Thursday a 1-DTE Friday weekly makes them diverge by
    # ~58% -- not a math error, a policy choice. The 7-DTE floor was kept because
    # implied_move_pct governs the >20% binary-event gate (SKILL.md:267), and a
    # 1-DTE straddle systematically understates the move, leaving that safety gate
    # under-firing.
    #
    # So this test compares the MATH, not the policy: force the old implementation
    # onto the merged tool's chosen expiry and assert they agree there.
    rows = fetch_options_chain(ticker, near_days=60)
    o_call, o_put, o_expiry = old_find(rows, spot, target_expiry=merged["front_expiry"])
    if o_call is None or o_put is None:
        pytest.skip(f"{ticker}: old path found no ATM strike; nothing to compare")

    old_px = old_move(spot, old_leg(o_call)[0], old_leg(o_put)[0])

    assert o_expiry == merged["front_expiry"], (
        f"{ticker}: forcing the old path onto {merged['front_expiry']} did not take "
        f"— got {o_expiry}. The chain lacks that expiry; the comparison is invalid."
    )

    # Tolerance, not equality: the two fetch the chain at slightly different
    # moments, so quotes can move between calls. A 15% relative gap means the
    # math diverged, not that the market ticked.
    a, b = merged["implied_move_pct"], old_px["implied_move_pct"]
    assert a > 0 and b > 0, f"{ticker}: zero implied move — merged {a}, old {b}"
    assert abs(a - b) / max(a, b) < 0.15, (
        f"{ticker}: implied move diverged — merged {a:.4f} vs old {b:.4f}"
    )
