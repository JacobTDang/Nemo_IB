"""Phase B3 pilot driver.

Runs the IB analyst playbook (CLAUDE.md) against a single ticker using the
Nemo MCP servers in SEC-first ordering, then dumps every tool call's raw
JSON to testing/output/phase_B3_pilot.json so the write-up can cite real
data.

This is a one-shot harness, not a recurring job. The analyst synthesis
(probe -> gather -> reason -> verdict) lives in the resulting log file,
not in this script.

Usage:
  python -m testing.run_phase_B3_pilot [TICKER]
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

from agent.MCP_manager import MCPConnectionManager


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_PATH = os.path.join(OUT_DIR, "phase_B3_pilot.json")


def _now_iso() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_error_response(result) -> bool:
  """Distinguish a real error envelope from a tool result that happens to
  carry an `error: null` field. SEC tools use `{success: true, error: null}`
  on success, which earlier mis-classified them as errors."""
  if not isinstance(result, dict):
    return False
  err = result.get("error")
  if err in (None, False, "", 0):
    # A `success` flag, if present, is authoritative
    if result.get("success") is False:
      return True
    return False
  return True


async def _call(mcp: MCPConnectionManager, tool: str, args: dict, log: list) -> dict:
  """Wrapper around mcp.call_tool that captures both success and failure
  into the run log without aborting the pilot."""
  started = _now_iso()
  try:
    result = await mcp.call_tool(tool, args)
    entry = {
      "tool": tool,
      "args": args,
      "started_at": started,
      "finished_at": _now_iso(),
      "ok": not _is_error_response(result),
      "result": result,
    }
  except Exception as exc:
    entry = {
      "tool": tool,
      "args": args,
      "started_at": started,
      "finished_at": _now_iso(),
      "ok": False,
      "result": {"error": f"{type(exc).__name__}: {exc}"},
    }
  log.append(entry)
  status = "OK " if entry["ok"] else "ERR"
  print(f"  [{status}] {tool} {json.dumps(args, default=str)[:80]}", flush=True)
  return entry["result"]


def _resolve_peers(profile_result: dict, peers_result: dict, target: str) -> list:
  """comparable_company_analysis needs a `companies` list that includes the
  target and 3-5 peers. Pull peers from get_company_peers and drop duplicates
  / the target itself, then prepend the target."""
  peers = []
  if isinstance(peers_result, dict):
    raw = peers_result.get("data") or peers_result
    if isinstance(raw, list):
      peers = raw
  cleaned = []
  for p in peers:
    if not isinstance(p, str):
      continue
    p = p.upper().strip()
    if p == target.upper() or p in cleaned:
      continue
    cleaned.append(p)
  return [target.upper()] + cleaned[:5]


def _scrape(payload, *path, default=None):
  """Defensive nested-dict pluck so we don't crash when a tool returns an
  error envelope mid-pilot."""
  cur = payload
  for key in path:
    if isinstance(cur, dict) and key in cur:
      cur = cur[key]
    else:
      return default
  return cur


async def _run_alpaca_block(ticker: str, price: float, log: list,
                            conviction_decision: dict) -> dict:
  """Open a fresh stdio session to the alpaca MCP server, reconcile paper
  positions, then either route the proposed trade through the Risk_Officer
  or document why we skipped trading."""
  env = {**os.environ}
  project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
  env["PYTHONUNBUFFERED"] = "1"

  params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "tools.alpaca.server", "server"],
    env=env,
  )

  block = {"reconciliation": None, "risk_check": None, "order": None,
           "skipped": False, "skip_reason": None}

  try:
    async with stdio_client(params) as (read, write):
      async with ClientSession(read, write) as session:
        await session.initialize()
        print("Connected to Alpaca MCP server", flush=True)

        # Health check first
        resp = await session.call_tool("ping_alpaca", {})
        try:
          ping = json.loads(resp.content[0].text)
        except Exception:
          ping = {"status": "unknown"}
        log.append({"tool": "ping_alpaca", "args": {}, "result": ping,
                    "finished_at": _now_iso(), "ok": True})

        # Reconciliation step (required by playbook before any trading)
        resp = await session.call_tool("get_paper_positions", {})
        positions = json.loads(resp.content[0].text)
        block["reconciliation"] = positions
        log.append({"tool": "get_paper_positions", "args": {},
                    "result": positions, "finished_at": _now_iso(),
                    "ok": "error" not in positions})

        if not positions.get("reconciled", False):
          block["skipped"] = True
          block["skip_reason"] = (
            "get_paper_positions returned reconciled=False — "
            "playbook requires halting before further orders."
          )
          print(f"  HALT: {block['skip_reason']}", flush=True)
          return block

        if not conviction_decision["should_trade"]:
          block["skipped"] = True
          block["skip_reason"] = conviction_decision["reason"]
          print(f"  SKIP TRADE: {block['skip_reason']}", flush=True)
          return block

        proposal = conviction_decision["proposal"]
        proposal["price"] = float(price)
        risk_args = {k: v for k, v in proposal.items() if k != "thesis_id"}

        resp = await session.call_tool("risk_check_proposed_trade", risk_args)
        risk = json.loads(resp.content[0].text)
        block["risk_check"] = risk
        log.append({"tool": "risk_check_proposed_trade", "args": risk_args,
                    "result": risk, "finished_at": _now_iso(),
                    "ok": "error" not in risk})

        if not risk.get("approve", False):
          block["skipped"] = True
          block["skip_reason"] = (
            "Risk_Officer rejected: " + "; ".join(risk.get("reasons", []))
          )
          print(f"  REJECTED: {block['skip_reason']}", flush=True)
          return block

        # Use adjusted quantity if Risk_Officer asked us to shrink the size
        order_args = dict(risk_args)
        if "adjusted_quantity" in risk and risk["adjusted_quantity"]:
          order_args["quantity"] = float(risk["adjusted_quantity"])
        order_args["rationale"] = proposal.get("rationale", "")

        resp = await session.call_tool("place_paper_order", order_args)
        order = json.loads(resp.content[0].text)
        block["order"] = order
        log.append({"tool": "place_paper_order", "args": order_args,
                    "result": order, "finished_at": _now_iso(),
                    "ok": order.get("success", False)})
        print(f"  ORDER: success={order.get('success')} "
              f"qty={order.get('qty')} id={order.get('order_id')}", flush=True)
  except Exception as exc:
    block["skipped"] = True
    block["skip_reason"] = f"Alpaca session error: {type(exc).__name__}: {exc}"
    print(f"  ALPACA ERROR: {block['skip_reason']}", flush=True)
  return block


async def run(ticker: str) -> dict:
  ticker = ticker.upper()
  os.makedirs(OUT_DIR, exist_ok=True)

  bundle = {
    "ticker": ticker,
    "started_at": _now_iso(),
    "playbook_phase": "B3 pilot",
    "calls": [],         # full ordered tool log
    "alpaca_block": None,
  }
  log = bundle["calls"]

  async with MCPConnectionManager() as mcp:
    # =========================
    # Phase 1 -- SEC XBRL first
    # =========================
    print(f"\n=== Phase 1: SEC XBRL (highest trust) for {ticker} ===")
    sec_calls = [
      ("get_revenue_base",    {"ticker": ticker}),
      ("get_ebitda_margin",   {"ticker": ticker}),
      ("get_capex_pct_revenue", {"ticker": ticker}),
      ("get_tax_rate",        {"ticker": ticker}),
      ("get_depreciation",    {"ticker": ticker}),
      ("get_margin_breakdown", {"ticker": ticker}),
      ("get_historical_fcf",  {"ticker": ticker}),
      ("get_working_capital", {"ticker": ticker}),
      ("get_latest_filing",   {"ticker": ticker}),
      ("extract_8k_events",   {"ticker": ticker}),
      ("extract_governance_data", {"ticker": ticker}),
    ]
    sec_results = {}
    for tool, args in sec_calls:
      sec_results[tool] = await _call(mcp, tool, args, log)

    # =========================
    # Phase 2 -- Vendor intel (Finnhub + market data)
    # =========================
    print(f"\n=== Phase 2: Vendor intel for {ticker} ===")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    thirty_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    six_months_out = (datetime.now(timezone.utc) + timedelta(days=180)).strftime("%Y-%m-%d")

    vendor_results = {}
    vendor_calls = [
      ("get_company_profile",         {"ticker": ticker}),
      ("get_market_data",             {"ticker": ticker}),
      ("get_basic_financials",        {"ticker": ticker}),
      ("get_analyst_recommendations", {"ticker": ticker}),
      ("get_forward_estimates",       {"ticker": ticker}),
      ("get_earnings_surprises",      {"ticker": ticker}),
      ("get_earnings_calendar",       {"from_date": today, "to_date": six_months_out}),
      ("get_company_peers",           {"ticker": ticker}),
      ("get_insider_transactions",    {"ticker": ticker}),
      ("get_insider_sentiment",       {"ticker": ticker,
                                        "from_date": one_year_ago,
                                        "to_date": today}),
      ("get_company_news",            {"ticker": ticker,
                                        "from_date": thirty_ago,
                                        "to_date": today}),
      ("get_financial_statements",    {"ticker": ticker,
                                        "statement": "ic",
                                        "freq": "annual"}),
    ]
    for tool, args in vendor_calls:
      vendor_results[tool] = await _call(mcp, tool, args, log)

    # Cash-flow + balance-sheet variants of get_financial_statements — needed
    # for real buyback / dividend yields in calculate_capital_returns;
    # otherwise both report 0% (B3 pilot finding). Stored under separate
    # dict keys so they don't overwrite the income-statement result.
    vendor_results["get_financial_statements_cf"] = await _call(
      mcp, "get_financial_statements",
      {"ticker": ticker, "statement": "cf", "freq": "annual"}, log)
    vendor_results["get_financial_statements_bs"] = await _call(
      mcp, "get_financial_statements",
      {"ticker": ticker, "statement": "bs", "freq": "annual"}, log)

    # =========================
    # Phase 3 -- Calculations (comps + DCF + credit + capital returns)
    # =========================
    print(f"\n=== Phase 3: Calculations for {ticker} ===")
    calc_results = {}

    peer_list = _resolve_peers(
      vendor_results.get("get_company_profile", {}),
      vendor_results.get("get_company_peers", {}),
      ticker,
    )
    calc_results["_peer_list"] = peer_list
    calc_results["comparable_company_analysis"] = await _call(
      mcp, "comparable_company_analysis", {"companies": peer_list}, log
    )

    # WACC auto-resolution only works inside the agent's execution engine
    # (the variable store). When called via MCP directly we have to feed
    # the inputs ourselves from the prior tool results.
    md_raw = vendor_results.get("get_market_data", {}) or {}
    md_beta = md_raw.get("beta") or 0
    md_market_cap = md_raw.get("marketCap") or 0
    md_total_debt = md_raw.get("totalDebt") or 0
    md_cash = md_raw.get("cash") or 0
    md_shares = md_raw.get("sharesOutstanding") or 0
    md_interest = md_raw.get("interestExpense")
    md_price = md_raw.get("currentPrice") or md_raw.get("price") or 0
    if md_total_debt and md_interest and not (md_interest != md_interest):  # NaN check
      md_cod = md_interest / md_total_debt
    else:
      md_cod = 0.05  # fallback IG corporate
    tax_rate_pct = sec_results.get("get_tax_rate", {}).get("effective_tax_rate") or 0
    macro_raw = None  # filled in phase 4 below; defer WACC if needed
    calc_results["calculate_wacc"] = await _call(
      mcp, "calculate_wacc",
      {"beta": md_beta, "risk_free_rate": 0.042,
       "equity_risk_premium": 0.06,
       "cost_of_debt": md_cod, "tax_rate": tax_rate_pct / 100.0,
       "market_cap": md_market_cap, "total_debt": md_total_debt},
      log,
    )

    # Pull scalars from the actual flat-dict shapes the servers return.
    # SEC tools: top-level keys. Finnhub tools: nested under `data`.
    sec_rev = sec_results.get("get_revenue_base", {}) or {}
    sec_ebitda = sec_results.get("get_ebitda_margin", {}) or {}
    sec_capex = sec_results.get("get_capex_pct_revenue", {}) or {}
    sec_tax = sec_results.get("get_tax_rate", {}) or {}
    sec_dep = sec_results.get("get_depreciation", {}) or {}

    rev_base = sec_rev.get("revenue_base") or 0
    # SEC tools return percent values; convert to decimal for the calcs
    ebitda_margin = (sec_ebitda.get("ebitda_margin_percent") or 0) / 100.0
    capex_pct = (sec_capex.get("capex_pct_revenue") or 0) / 100.0
    tax_rate = (sec_tax.get("effective_tax_rate") or 0) / 100.0
    dep_pct = (sec_dep.get("d&a_pct") or 0) / 100.0

    fwd_est = vendor_results.get("get_forward_estimates", {}) or {}
    consensus_growth = _scrape(fwd_est, "data", "consensus_revenue_growth")
    if not isinstance(consensus_growth, list) or len(consensus_growth) < 3:
      # Bridge: anchor on the implied basic_financials TTM growth, fade
      bf_growth = _scrape(vendor_results.get("get_basic_financials", {}),
                          "data", "metric", "revenueGrowthTTMYoy") or 10.0
      g0 = max(0.04, min(0.20, float(bf_growth) / 100.0))
      consensus_growth = [round(g0 - 0.01 * i, 4) for i in range(5)]

    base_growth = (consensus_growth + [consensus_growth[-1]] * 5)[:5]
    bear_growth = [max(0.0, g - 0.04) for g in base_growth]
    bull_growth = [g + 0.03 for g in base_growth]

    base_margin = ebitda_margin if ebitda_margin else 0.45
    bear_margin = max(0.20, base_margin - 0.07)
    bull_margin = min(0.65, base_margin + 0.03)

    # get_market_data returns a flat dict with camelCase keys
    cash = md_cash
    debt = md_total_debt
    shares = md_shares
    market_cap = md_market_cap
    price = md_price

    bf = vendor_results.get("get_basic_financials", {}) or {}
    term_multiple = _scrape(bf, "data", "metric", "evEbitdaTTM") or 0

    fin_stmt = vendor_results.get("get_financial_statements", {}) or {}
    interest_expense = (md_interest if md_interest and md_interest == md_interest
                        else 0)
    dep_abs = (dep_pct * rev_base) if dep_pct and rev_base else 0
    capex_abs = (capex_pct * rev_base) if capex_pct and rev_base else 0
    ebitda_abs = (ebitda_margin * rev_base) if ebitda_margin and rev_base else 0

    calc_results["calculate_scenario_dcf"] = await _call(
      mcp, "calculate_scenario_dcf",
      {
        "ticker": ticker,
        "bear_growth": bear_growth,
        "base_growth": base_growth,
        "bull_growth": bull_growth,
        "bear_margin": bear_margin,
        "base_margin": base_margin,
        "bull_margin": bull_margin,
        "revenue_base": rev_base,
        "capex_pct_revenue": capex_pct,
        "tax_rate": tax_rate,
        "depreciation": dep_pct,
        "wacc": 0,           # auto-resolved
        "terminal_growth": 0.025,
        "terminal_multiple": term_multiple or 18.0,
        "cash": cash, "debt": debt, "shares_outstanding": shares,
      },
      log,
    )

    if ebitda_abs and dep_abs and capex_abs and tax_rate:
      calc_results["calculate_credit_profile"] = await _call(
        mcp, "calculate_credit_profile",
        {
          "ticker": ticker,
          "total_debt": debt,
          "cash": cash,
          "ebitda": ebitda_abs,
          "interest_expense": interest_expense or 1,
          "depreciation_abs": dep_abs,
          "capex_abs": capex_abs,
          "tax_rate": tax_rate,
          "market_cap": market_cap,
        },
        log,
      )
    else:
      calc_results["calculate_credit_profile"] = {
        "skipped": "missing scalar inputs (ebitda/dep/capex/tax)"
      }

    # Cash-flow data drives buyback + dividend yields in capital returns.
    # Finnhub returns camelCase keys; calculate_capital_returns wants snake_case.
    cf_stmt = vendor_results.get("get_financial_statements_cf", {}) or {}
    cf_periods = _scrape(cf_stmt, "data", "periods") or []
    cf_latest = cf_periods[0] if cf_periods else {}
    dividends_paid = cf_latest.get("dividendsPaid") or 0
    shares_repurchased = cf_latest.get("repurchaseOfCapitalStock") or 0
    cap_returns_source = "cf_statement" if dividends_paid or shares_repurchased else "none"

    # SEC-XBRL buyback fallback (highest trust per playbook). Finnhub free
    # tier returns 0 periods for /stock/financials, so cf-statement buybacks
    # land as 0; SEC's PaymentsForRepurchaseOfCommonStock concept gives a
    # real annual figure from the latest 10-K.
    if not shares_repurchased:
      bb_result = await _call(mcp, "get_buyback_history", {"ticker": ticker}, log)
      if bb_result.get("success") and bb_result.get("ttm_repurchase"):
        shares_repurchased = float(bb_result["ttm_repurchase"])
        cap_returns_source = (
          "sec_xbrl_plus_imputed_dividends" if cap_returns_source == "none"
          else f"sec_xbrl+{cap_returns_source}"
        )

    # Yield-based fallback when CF statement is unavailable (e.g. Finnhub
    # free tier returns 403 for /stock/financials). currentDividendYieldTTM
    # is available on free tier, so impute dividends_paid from it.
    if not dividends_paid and market_cap:
      div_yield_pct = _scrape(
        vendor_results.get("get_basic_financials", {}) or {},
        "data", "metric", "currentDividendYieldTTM",
      )
      if div_yield_pct and isinstance(div_yield_pct, (int, float)):
        dividends_paid = round(market_cap * (div_yield_pct / 100.0), 2)
        if "sec_xbrl" in cap_returns_source:
          cap_returns_source = "sec_xbrl_buybacks+imputed_dividends"
        else:
          cap_returns_source = "imputed_from_yield"
    if ebitda_abs and capex_abs and tax_rate and dep_abs:
      result = await _call(
        mcp, "calculate_capital_returns",
        {
          "ticker": ticker,
          "market_cap": market_cap,
          "ebitda": ebitda_abs,
          "capex_abs": capex_abs,
          "tax_rate": tax_rate,
          "depreciation_abs": dep_abs,
          "dividends_paid": dividends_paid,
          "shares_repurchased": shares_repurchased,
          "shares_outstanding": shares,
        },
        log,
      )
      # Tag the source so the write-up can disclose imputed vs sourced figures.
      if isinstance(result, dict):
        result.setdefault("_driver_meta", {})["capital_returns_source"] = cap_returns_source
      calc_results["calculate_capital_returns"] = result
    else:
      calc_results["calculate_capital_returns"] = {
        "skipped": "missing scalar inputs"
      }

    # =========================
    # Phase 4 -- Macro context (FRED)
    # =========================
    print(f"\n=== Phase 4: Macro context ===")
    macro_results = {
      "get_macro_snapshot":   await _call(mcp, "get_macro_snapshot", {}, log),
      "get_treasury_yields":  await _call(mcp, "get_treasury_yields", {}, log),
      "get_credit_spreads":   await _call(mcp, "get_credit_spreads", {}, log),
    }

    bundle["sec_results"]    = sec_results
    bundle["vendor_results"] = vendor_results
    bundle["calc_results"]   = calc_results
    bundle["macro_results"]  = macro_results
    bundle["scalars"] = {
      "price": price, "market_cap": market_cap, "shares_outstanding": shares,
      "cash": cash, "debt": debt,
      "revenue_base": rev_base, "ebitda_margin": ebitda_margin,
      "capex_pct_revenue": capex_pct, "tax_rate": tax_rate,
      "depreciation_pct": dep_pct,
      "ebitda_abs": ebitda_abs, "capex_abs": capex_abs, "dep_abs": dep_abs,
      "consensus_growth_used": consensus_growth,
      "peer_list_for_comps": peer_list,
    }

    # Persist NOW before any trading rail. Anything below this line is
    # additive -- the analyst can still write up the verdict from the
    # bundle even if the alpaca block crashes on Windows asyncio quirks.
    bundle["finished_data_capture_at"] = _now_iso()
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
      json.dump(bundle, fh, indent=2, default=str)
    print(f"\nData capture written to {OUT_PATH}")

    # =========================
    # Phase 5 -- Trade gating (Alpaca, runs inside the MCP scope to avoid
    # the anyio cancel-scope mismatch on Windows after MCPConnectionManager
    # cleanup).
    # =========================
    conviction_env = os.environ.get("PHASE_B3_CONVICTION")
    decision = {"should_trade": False,
                "reason": "no conviction decision provided; "
                          "this driver is data-capture only -- "
                          "analyst makes the trade call in the log."}
    if conviction_env:
      try:
        decision = json.loads(conviction_env)
      except Exception as exc:
        decision = {"should_trade": False,
                    "reason": f"PHASE_B3_CONVICTION parse failed: {exc}"}

    bundle["alpaca_block"] = await _run_alpaca_block(ticker, price, log, decision)

  bundle["finished_at"] = _now_iso()
  with open(OUT_PATH, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, indent=2, default=str)
  print(f"\nFinal pilot bundle written to {OUT_PATH}")
  return bundle


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("ticker", nargs="?", default="MSFT")
  args = parser.parse_args()
  asyncio.run(run(args.ticker))


if __name__ == "__main__":
  main()
