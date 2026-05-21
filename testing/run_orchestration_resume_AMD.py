"""Resume the AMD orchestration from STEP 6 -- thesis 267 already inserted.

Run after `run_full_orchestration_AMD.py` if the stdio rail hung. Idempotent:
re-runs risk-check, exposure analysis, falsifier pre-check, and writes the
bundle file.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from state.theses import latest_thesis, get_thesis
from state.schema import init_schema
from agent.exposure_analyzer import analyze_exposures
from state.theses import active_theses
from daemons.falsifier_watcher import evaluate_thesis as falsifier_evaluate_thesis
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tools.financial_modeling_engine.utils import get_data as get_market_data


TICKER = "AMD"
OUT_BUNDLE = os.path.join(THIS, "output", "orchestration_AMD_resume.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_stdio_risk_check(synthetic: dict) -> dict:
    print("Spawning alpaca stdio server...")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    env.setdefault("ALPACA_PAPER_KEY", "orchestration_fake")
    env.setdefault("ALPACA_PAPER_SECRET", "orchestration_fake")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.alpaca.server", "server"],
        env=env,
    )
    out: dict = {"call_args": synthetic}
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=30)
                tools = await asyncio.wait_for(session.list_tools(), timeout=10)
                out["server_tool_count"] = len(tools.tools)
                out["server_tool_names"] = sorted(t.name for t in tools.tools)
                # ping first -- confirms stdio framing alive
                pr = await asyncio.wait_for(session.call_tool("ping_alpaca", {}),
                                             timeout=15)
                out["ping_alpaca"] = json.loads(pr.content[0].text)
                r = await asyncio.wait_for(
                    session.call_tool("risk_check_proposed_trade", synthetic),
                    timeout=45,
                )
                decision = json.loads(r.content[0].text)
                out["decision_via_stdio"] = decision
    except Exception as exc:
        out["stdio_error"] = f"{type(exc).__name__}: {exc}"
        out["stdio_traceback"] = traceback.format_exc(limit=4)
    return out


def run_local_risk_check(synthetic: dict) -> dict:
    """Direct Risk_Officer call -- same business logic the MCP tool wraps."""
    from agent.Arbiter_Agent import ArbiterVerdict
    from agent.Risk_Officer import Risk_Officer
    from state.positions import portfolio_stats, open_positions
    verdict = ArbiterVerdict(
        final_recommendation=synthetic["recommendation"],
        confidence=float(synthetic["confidence"]),
        bull_strength=float(synthetic["bull_strength"]),
        bear_strength=float(synthetic["bear_strength"]),
        decisive_factors=[synthetic.get("rationale", "(no rationale)")],
        acknowledged_risks=["(risk check evaluation only)"],
        conditions_to_change_mind=["(evaluated via local rail)"],
        position_sizing_guidance=synthetic["position_sizing"],
        rationale=synthetic.get("rationale", ""),
    )
    portfolio = portfolio_stats(paper=True)
    basket = [p["ticker"].upper() for p in open_positions(paper=True) or []]
    ro = Risk_Officer()
    d = ro.evaluate(
        proposed_quantity=float(synthetic["quantity"]),
        proposed_price=float(synthetic["price"]),
        arbiter_verdict=verdict,
        portfolio=portfolio,
        proposed_ticker=str(synthetic["ticker"]).upper(),
        open_basket=basket,
    )
    return {
        "approve":              d.approve,
        "reasons":              d.reasons,
        "adjusted_quantity":    d.adjusted_quantity,
        "adjusted_dollar_size": d.adjusted_dollar_size,
    }


async def main():
    init_schema()
    os.makedirs(os.path.dirname(OUT_BUNDLE), exist_ok=True)

    th = latest_thesis(TICKER)
    if not th:
        print(f"FATAL: no active {TICKER} thesis on file. Run "
              f"run_full_orchestration_AMD.py first.")
        sys.exit(1)
    print(f"Active AMD thesis: id={th['thesis_id']}  conf={th['confidence']}  "
          f"falsifiers={len(th.get('falsifiers') or [])}")

    bundle = {
        "ticker":    TICKER,
        "started_at": _now_iso(),
        "thesis_id": th["thesis_id"],
        "thesis_confidence": th["confidence"],
    }

    # Market data for the synthetic price (cheap call)
    print("\nFetching market data for synthetic verdict price...")
    md = get_market_data(TICKER)
    current_price = (md.get("currentPrice")
                     or md.get("regularMarketPrice") or 165.0)
    print(f"  current_price = {current_price}")
    synth_qty = max(1, int(1000.0 / float(current_price)))
    sizing = "cautious"
    synthetic_verdict = {
        "ticker":          TICKER,
        "side":            "buy",
        "quantity":        synth_qty,
        "price":           float(current_price),
        "recommendation":  "BUY",
        "confidence":      float(th["confidence"]),
        "bull_strength":   0.65,
        "bear_strength":   0.45,
        "position_sizing": sizing,
        "rationale":       (f"AMD cautious BUY (thesis_id={th['thesis_id']}); "
                            "orchestration verification rail."),
    }

    # ---- Step 5: exposure ----
    print("\n=== STEP 5  exposure analysis ===")
    exposures = analyze_exposures(active_theses())
    amd_buckets = []
    for fname, entries in exposures["factors"].items():
        if any(e["ticker"] == TICKER for e in entries):
            amd_buckets.append(fname)
    print(f"  AMD lands in: {amd_buckets}")
    bundle["step5_exposures"] = {
        "theses_analyzed":     exposures["theses_analyzed"],
        "factor_count":        exposures["factor_count"],
        "amd_buckets":         amd_buckets,
        "warnings":            exposures["warnings"],
        "top_concentrations":  [{
            "factor":      c["factor"],
            "thesis_count": c["thesis_count"],
            "tickers":     c["tickers"],
        } for c in exposures["top_concentrations"]],
    }

    # ---- Step 6: risk-check rail (stdio + local fallback) ----
    print("\n=== STEP 6  risk-check rail ===")
    bundle["step6_risk_check"] = await run_stdio_risk_check(synthetic_verdict)
    print(f"  stdio result: {('OK' if 'decision_via_stdio' in bundle['step6_risk_check'] else 'ERR')}")
    if bundle["step6_risk_check"].get("stdio_error"):
        print(f"  stdio_error: {bundle['step6_risk_check']['stdio_error']}")
    # Always also run local Risk_Officer so the rail decision is captured
    local = run_local_risk_check(synthetic_verdict)
    bundle["step6_risk_check"]["decision_local"] = local
    print(f"  [local] approve = {local['approve']}")
    for reason in local["reasons"]:
        print(f"     {reason}")
    bundle["step6_risk_check"]["decision"] = (
        bundle["step6_risk_check"].get("decision_via_stdio") or local
    )

    # ---- Step 7: falsifier pre-check ----
    print("\n=== STEP 7  falsifier pre-check ===")
    th_full = get_thesis(th["thesis_id"])
    fw_summary = falsifier_evaluate_thesis(
        th_full, observed={}, lookback_hours=48,
        log_fn=lambda m: print("  " + m),
    )
    print(f"  triggers: {len(fw_summary.get('triggers', []))} "
          f"(new: {len(fw_summary.get('new_triggers', []))})")
    bundle["step7_falsifier_precheck"] = fw_summary

    bundle["finished_at"] = _now_iso()
    with open(OUT_BUNDLE, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, default=str)
    print(f"\n[bundle] saved to {OUT_BUNDLE}")

    # ---- Final summary ----
    print("\n" + "#" * 70)
    print("# RESUME FINAL REPORT")
    print("#" * 70)
    risk_dec = bundle["step6_risk_check"]["decision"]
    print(f"Thesis        : id={th['thesis_id']} conf={th['confidence']}")
    print(f"Exposure      : AMD -> {amd_buckets}")
    print(f"Risk (stdio)  : "
          f"{'OK' if 'decision_via_stdio' in bundle['step6_risk_check'] else 'hung/err'}")
    print(f"Risk (local)  : approve={local['approve']}  "
          f"reasons={local['reasons']}")
    print(f"Falsifier pre : {len(fw_summary.get('triggers', []))} triggers "
          f"(expected 0 for clean entry)")


if __name__ == "__main__":
    asyncio.run(main())
