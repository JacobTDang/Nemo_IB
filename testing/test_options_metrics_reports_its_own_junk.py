"""An options payload must not hide a hole, and must not call junk "ok".

`get_options_metrics("DLNG")`, live 2026-08-26, answered `success: true` with:

    implied_move   {"error": "no ATM strike within threshold"}
    term_structure 7d   atm_call_iv 0.875   atm_put_iv 4.0391
                   30d  atm_call_iv 0.5     atm_put_iv 0.25
                   60d  atm_call_iv 0.5508  atm_put_iv 0.25
                   90d  atm_call_iv 0.5508  atm_put_iv 0.25
    data_quality   {"iv_status": "ok", "notes": []}

Two defects, both live.

1. `implied_move` is a bare error object sitting where every other ticker has
   a numeric dict. TGT returns `{"implied_move_pct": 0.0387, "straddle_cost":
   6.35, ...}`. A caller doing `.get("implied_move_pct")` gets None from
   DLNG's block -- not because the move is zero, and not because the field
   was refused, but because the key is absent. `or 0` turns it into a 0%
   implied move, and the `implied_move_pct > 0.20` binary-event gate in
   .claude/skills/preearnings-research reads that as a calm setup on a name
   whose chain could not be priced at all. Nothing at the top level said so.

2. `iv_status: "ok"` over a put IV of 404%. 4.0391 is not an implied
   volatility; it is what a bisection solver returns when it cannot solve.
   It sits at the same strike and expiry as a call at 0.875 -- put-call
   parity forces same-strike IVs to agree closely, so a 4.6x gap means at
   least one leg is not a solve of a real quote. And atm_put_iv is 0.25 at
   every expiry in the sweep, which a term structure never does. The tool
   description promises a sentinel detector; the one that shipped only asked
   whether the *averaged* atm_iv was below 0.08 or identical across tenors,
   and 2.457 / 0.375 / 0.4004 / 0.4004 is neither.

Thresholds, measured live 2026-08-26 across NVDA, TGT, AAPL, MSFT, SPY, F,
PLUG, AMC, TLRY and RIG -- the widest same-expiry call/put IV ratio any of
them showed was 1.377 (SPY), and none produced an ATM leg IV above 1.0 or
repeated a leg's IV across two different expiry dates:

    implausible level      any ATM leg IV > 3.0  (DLNG: 4.0391)
    parity divergence      max/min leg IV >= 2.0 at one expiry (DLNG: 4.616)
    pinned leg             one leg, identical to 4dp, at >= 2 distinct
                           expiry DATES (DLNG: put 0.25 at 2026-10-16 and
                           2027-01-15)

Expiry DATES, not the 7d/30d/60d/90d bucket labels: DLNG's 60d and 90d
buckets both resolve to 2027-01-15, so counting buckets would call one
reading two and fire on every short chain in the market.

What must not change: the two original rules still fire, `data_quality` keeps
its `iv_status` / `notes` / `volume_data_usable` keys, every suspect status
keeps the `suspect_` prefix the preearnings-research skill matches on, and
`implied_move` keeps the `error` key that same skill documents.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.options_quality import (
    audit_iv,
    audit_options_metrics,
    normalize_implied_move,
)


# Verbatim from get_options_metrics("DLNG"), 2026-08-26.
DLNG = {
    "ticker": "DLNG", "success": True, "error": None,
    "spot_price": 3.78, "expirations_available": 3,
    "term_structure": {
        "7d":  {"expiry": "2026-09-18", "dte": 23,
                "atm_call_iv": 0.875, "atm_put_iv": 4.0391, "atm_iv": 2.457},
        "30d": {"expiry": "2026-10-16", "dte": 51,
                "atm_call_iv": 0.5, "atm_put_iv": 0.25, "atm_iv": 0.375},
        "60d": {"expiry": "2027-01-15", "dte": 142,
                "atm_call_iv": 0.5508, "atm_put_iv": 0.25, "atm_iv": 0.4004},
        "90d": {"expiry": "2027-01-15", "dte": 142,
                "atm_call_iv": 0.5508, "atm_put_iv": 0.25, "atm_iv": 0.4004},
    },
    "implied_move": {"error": "no ATM strike within threshold"},
    "put_call_skew_30d": {"value": -0.25, "put_iv_90pct": 0.25,
                          "call_iv_110pct": 0.5, "expiry": "2026-10-16",
                          "note": "0.9*spot put IV minus 1.1*spot call IV; "
                                  "positive=downside fear, negative=upside speculation"},
    "nearest_expiry_activity": {"expiry": "2026-09-18",
                                "call_open_interest": 36, "put_open_interest": 1,
                                "put_call_oi_ratio": 0.028,
                                "call_volume": 31, "put_volume": 0,
                                "put_call_volume_ratio": 0.0},
    "data_quality": {"iv_status": "ok", "notes": [], "volume_data_usable": True},
}

# Verbatim from get_options_metrics("TGT"), same session -- a chain that works.
TGT_TERM = {
    "7d":  {"expiry": "2026-09-04", "dte": 9,
            "atm_call_iv": 0.2905, "atm_put_iv": 0.2897, "atm_iv": 0.2901},
    "30d": {"expiry": "2026-09-25", "dte": 30,
            "atm_call_iv": 0.312, "atm_put_iv": 0.2907, "atm_iv": 0.3013},
    "60d": {"expiry": "2026-11-20", "dte": 86,
            "atm_call_iv": 0.3616, "atm_put_iv": 0.3567, "atm_iv": 0.3591},
    "90d": {"expiry": "2026-12-18", "dte": 114,
            "atm_call_iv": 0.3663, "atm_put_iv": 0.3375, "atm_iv": 0.3519},
}

TGT_IMPLIED_MOVE = {"implied_move_pct": 0.0387, "straddle_cost": 6.35,
                    "front_expiry": "2026-09-04", "quotes_stale": False}


# --------------------------------------------------------------------------
# 1. implied_move: a hole must not be shaped like a number
# --------------------------------------------------------------------------

def test_an_unpriceable_chain_does_not_answer_with_a_missing_key():
    """`.get("implied_move_pct")` must not be the caller's only signal."""
    block, _ = normalize_implied_move(DLNG["implied_move"])
    assert block["available"] is False, block
    assert "implied_move_pct" in block, (
        "the field a caller reads is absent, so None means 'no such key' and "
        "'not computed' and 'zero' all at once")
    assert block["implied_move_pct"] is None
    assert block["straddle_cost"] is None


def test_the_reason_survives_and_so_does_the_documented_error_key():
    """preearnings-research documents `implied_move` carrying an `error` key."""
    block, _ = normalize_implied_move(DLNG["implied_move"])
    assert block.get("error") == "no ATM strike within threshold"
    assert "no ATM strike" in str(block.get("reason") or block.get("error"))


def test_an_unpriceable_chain_is_visible_at_the_top_level():
    out = audit_options_metrics(DLNG)
    codes = {w.get("code") for w in (out.get("warnings") or [])}
    assert "implied_move_unavailable" in codes, (
        f"nothing at the top level says the straddle could not be priced: {out.get('warnings')}")
    assert out.get("coverage") == "partial", (
        "a payload missing one of its three headline outputs is still "
        f"reported as complete: coverage={out.get('coverage')!r}")


def test_a_priced_chain_is_left_alone():
    """The guard must not label a working chain."""
    block, warn = normalize_implied_move(TGT_IMPLIED_MOVE)
    assert warn is None
    assert block["available"] is True
    assert block["implied_move_pct"] == 0.0387
    assert block["straddle_cost"] == 6.35
    assert block["front_expiry"] == "2026-09-04"


# --------------------------------------------------------------------------
# 2. the sentinel detector the description promises
# --------------------------------------------------------------------------

def test_a_404_percent_atm_iv_is_not_ok():
    audit = audit_iv(DLNG["term_structure"])
    assert audit["iv_status"] != "ok", audit
    assert audit["iv_status"].startswith("suspect"), (
        "the preearnings-research skill matches on a `suspect` status; "
        f"got {audit['iv_status']!r}")
    assert "implausible_iv_level" in audit["iv_findings"], audit


def test_same_strike_call_and_put_iv_cannot_disagree_by_2x():
    """Put-call parity: same strike, same expiry, the IVs must nearly agree."""
    audit = audit_iv(DLNG["term_structure"])
    assert "parity_divergence" in audit["iv_findings"], audit
    joined = " ".join(audit["notes"]).lower()
    assert "4.0391" in " ".join(audit["notes"]) or "4.62" in joined or "4.6" in joined, (
        f"the note does not quantify the divergence: {audit['notes']}")


def test_a_leg_pinned_to_one_value_across_expiries_is_flagged():
    """A term structure that does not move with tenor is not a term structure."""
    audit = audit_iv(DLNG["term_structure"])
    assert "pinned_leg_iv" in audit["iv_findings"], audit


def test_every_finding_is_explained_in_notes():
    audit = audit_iv(DLNG["term_structure"])
    assert len(audit["notes"]) == len(audit["iv_findings"]) >= 3, audit


def test_a_working_chain_is_still_ok():
    """The calibration guard: none of the new rules may fire on TGT."""
    audit = audit_iv(TGT_TERM)
    assert audit["iv_status"] == "ok", audit
    assert audit["notes"] == []
    assert audit["iv_findings"] == []


# --------------------------------------------------------------------------
# 3. the rules that already shipped must keep working
# --------------------------------------------------------------------------

def test_the_all_below_008_sentinel_rule_still_fires():
    term = {"7d":  {"atm_call_iv": 0.0625, "atm_put_iv": 0.0625, "atm_iv": 0.0625,
                    "expiry": "2026-09-04"},
            "30d": {"atm_call_iv": 0.03125, "atm_put_iv": 0.03125, "atm_iv": 0.03125,
                    "expiry": "2026-09-25"}}
    audit = audit_iv(term)
    assert audit["iv_status"] == "suspect_iv_sentinel", audit


def test_the_identical_across_tenors_rule_still_fires():
    term = {"7d":  {"atm_call_iv": 0.30, "atm_put_iv": 0.30, "atm_iv": 0.30,
                    "expiry": "2026-09-04"},
            "30d": {"atm_call_iv": 0.30, "atm_put_iv": 0.30, "atm_iv": 0.30,
                    "expiry": "2026-09-25"}}
    audit = audit_iv(term)
    assert audit["iv_status"].startswith("suspect"), audit
    assert "constant_atm_iv" in audit["iv_findings"], audit


def test_no_extractable_iv_is_still_reported():
    term = {"7d": {"expiry": "2026-09-04", "dte": 9, "atm_call_iv": None,
                   "atm_put_iv": None, "atm_iv": None}}
    audit = audit_iv(term)
    assert audit["iv_status"] == "no_iv_data", audit


def test_data_quality_keeps_the_keys_it_already_published():
    out = audit_options_metrics(DLNG)
    dq = out["data_quality"]
    for key in ("iv_status", "notes", "volume_data_usable"):
        assert key in dq, f"data_quality.{key} disappeared"
    assert isinstance(dq["notes"], list)


def test_the_audit_does_not_mutate_its_input():
    before = json.dumps(DLNG, sort_keys=True)
    audit_options_metrics(DLNG)
    assert json.dumps(DLNG, sort_keys=True) == before, (
        "audit_options_metrics rewrote the payload it was handed")


# --------------------------------------------------------------------------
# 4. the MCP tool actually applies it
# --------------------------------------------------------------------------

def test_the_mcp_tool_serves_the_audited_payload(monkeypatch):
    from tools.financial_modeling_engine import analysis_tools

    monkeypatch.setattr(analysis_tools, "get_options_metrics",
                        lambda ticker: json.loads(json.dumps(DLNG)))
    server = analysis_tools.Financial_Analysis()
    payload = asyncio.run(server.get_options_metrics("DLNG"))
    out = json.loads(payload[0].text)

    assert out["data_quality"]["iv_status"].startswith("suspect"), out["data_quality"]
    assert out["implied_move"]["available"] is False
    assert {w["code"] for w in out["warnings"]} >= {"implied_move_unavailable",
                                                    "suspect_iv"}
    assert out["coverage"] == "partial"


def test_a_failed_lookup_is_passed_through_untouched(monkeypatch):
    """A response that never got a chain has nothing to audit."""
    from tools.financial_modeling_engine import analysis_tools

    failed = {"ticker": "ZZZZ", "success": False, "error": "no options listed"}
    monkeypatch.setattr(analysis_tools, "get_options_metrics", lambda ticker: dict(failed))
    server = analysis_tools.Financial_Analysis()
    out = json.loads(asyncio.run(server.get_options_metrics("ZZZZ"))[0].text)
    assert out == failed
