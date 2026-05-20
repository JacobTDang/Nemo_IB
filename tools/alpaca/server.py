"""Alpaca MCP server — exposes paper-trading tools gated by Risk_Officer.

Tools (added incrementally per the Phase A punchlist):
  - ping_alpaca         [A1] health check
  - get_paper_account   [A2] equity/cash/buying_power from the broker
  - get_paper_positions [A3] open positions with local-vs-broker reconciliation
  - risk_check_proposed_trade [A4] Risk_Officer.evaluate() — no broker call
  - place_paper_order   [A5] risk-checked order placement
  - close_paper_position [A6] opposing market order to flatten

Entry: python -m tools.alpaca.server server
"""
from typing import Any, Dict, List
import asyncio
import json
import sys
from datetime import date, datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions


def _json_default(obj):
  if isinstance(obj, (date, datetime)):
    return obj.isoformat()
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _safe_dumps(obj) -> str:
  return json.dumps(obj, default=_json_default)


class AlpacaServer:
  """MCP server that bridges Claude Code to the existing Execution_Agent
  and Risk_Officer. Tool handlers are added incrementally across A2-A6."""

  def __init__(self):
    self.server = Server("nemo_alpaca")
    self._setup_handlers()

  async def list_tools_descriptors(self) -> List[Tool]:
    """Public mirror of the @list_tools handler — exposes the tool list
    to unit tests without bouncing through MCP framing."""
    return [
      Tool(
        name="ping_alpaca",
        description=(
          "Health check for the Alpaca MCP server. Returns "
          "{\"status\": \"pong\"}. Use to verify server is reachable."
        ),
        inputSchema={
          "type": "object",
          "properties": {},
          "required": [],
        },
      ),
      Tool(
        name="get_paper_account",
        description=(
          "Fetches the current paper-trading account summary from Alpaca: "
          "equity, cash, buying_power, portfolio_value, and account status. "
          "Numeric fields are coerced to float. On broker failure, returns "
          "{\"error\": str, \"paper\": true}. Use to confirm account is "
          "active and check available capital before proposing trades."
        ),
        inputSchema={
          "type": "object",
          "properties": {},
          "required": [],
        },
      ),
      Tool(
        name="risk_check_proposed_trade",
        description=(
          "Evaluates a proposed paper trade against the deterministic "
          "Python Risk_Officer. Returns "
          "{approve: bool, reasons: [str], adjusted_quantity?: float, "
          "adjusted_dollar_size?: float}. This tool ONLY evaluates — it "
          "does NOT place the order. Use this before place_paper_order to "
          "pre-check whether Risk_Officer will allow the trade. If approve "
          "is False, do NOT call place_paper_order. If approve is True with "
          "adjusted_quantity set, use the adjusted quantity (smaller than "
          "requested) when calling place_paper_order."
        ),
        inputSchema={
          "type": "object",
          "properties": {
            "ticker":           {"type": "string"},
            "side":             {"type": "string", "enum": ["buy", "sell"]},
            "quantity":         {"type": "number"},
            "price":            {"type": "number"},
            "recommendation":   {"type": "string",
                                  "enum": ["BUY", "SELL", "HOLD", "NEUTRAL"]},
            "confidence":       {"type": "number"},
            "bull_strength":    {"type": "number"},
            "bear_strength":    {"type": "number"},
            "position_sizing":  {"type": "string",
                                  "enum": ["aggressive", "normal",
                                            "cautious", "no_position"]},
            "rationale":        {"type": "string"},
          },
          "required": ["ticker", "side", "quantity", "price",
                        "recommendation", "confidence", "bull_strength",
                        "bear_strength", "position_sizing"],
        },
      ),
      Tool(
        name="place_paper_order",
        description=(
          "Places a paper order with MANDATORY internal Risk_Officer check. "
          "This tool calls risk_check_proposed_trade FIRST; if Risk_Officer "
          "rejects, the order is refused and the broker is never contacted. "
          "If Risk_Officer returns an adjusted_quantity (size cap), the "
          "broker is called with the smaller quantity. Returns "
          "{success: bool, order_id?: str, client_order_id?: str, "
          "qty: number, error?: str, risk_decision: {...}}. "
          "Always inspect risk_decision in the response for audit."
        ),
        inputSchema={
          "type": "object",
          "properties": {
            "ticker":             {"type": "string"},
            "side":               {"type": "string", "enum": ["buy", "sell"]},
            "quantity":           {"type": "number"},
            "price":              {"type": "number"},
            "recommendation":     {"type": "string",
                                    "enum": ["BUY", "SELL", "HOLD", "NEUTRAL"]},
            "confidence":         {"type": "number"},
            "bull_strength":      {"type": "number"},
            "bear_strength":      {"type": "number"},
            "position_sizing":    {"type": "string",
                                    "enum": ["aggressive", "normal",
                                              "cautious", "no_position"]},
            "rationale":          {"type": "string"},
            "thesis_id":          {"type": ["integer", "null"]},
            "arbiter_verdict_id": {"type": ["integer", "null"]},
          },
          "required": ["ticker", "side", "quantity", "price",
                        "recommendation", "confidence", "bull_strength",
                        "bear_strength", "position_sizing"],
        },
      ),
      Tool(
        name="close_paper_position",
        description=(
          "Closes an open paper position by submitting an opposing market "
          "order (sell to flatten a long, buy to flatten a short). No "
          "Risk_Officer gate is applied — reducing exposure is always safe. "
          "`reason` is REQUIRED for the audit trail (e.g., 'stop_loss_hit', "
          "'thesis_broken', 'profit_target', 'manual_override'). Returns "
          "{success: bool, side?: str, order_id?: str, error?: str}."
        ),
        inputSchema={
          "type": "object",
          "properties": {
            "ticker": {"type": "string"},
            "reason": {"type": "string",
                        "description": "audit log reason; non-empty required"},
          },
          "required": ["ticker", "reason"],
        },
      ),
      Tool(
        name="get_paper_positions",
        description=(
          "Returns open paper positions reconciled across two sources: the "
          "broker (Alpaca) and the local SQLite audit log. Response shape: "
          "{broker_positions: [...], local_positions: [...], "
          "reconciled: bool, discrepancies: [str]}. Discrepancies are tagged "
          "'missing_locally:<TICKER>' (broker has it, DB doesn't) or "
          "'missing_at_broker:<TICKER>' (DB has it, broker doesn't). "
          "Use to detect divergence — any unreconciled state needs audit "
          "before further trading."
        ),
        inputSchema={
          "type": "object",
          "properties": {},
          "required": [],
        },
      ),
    ]

  @staticmethod
  def _build_verdict_and_portfolio(args: Dict[str, Any]):
    """Common helper: build an ArbiterVerdict from MCP args + pull current
    portfolio_stats + open basket. Shared between risk_check_proposed_trade
    (A4) and place_paper_order (A5)."""
    from agent.Arbiter_Agent import ArbiterVerdict
    from state.positions import portfolio_stats, open_positions
    verdict = ArbiterVerdict(
      final_recommendation=args["recommendation"],
      confidence=float(args["confidence"]),
      bull_strength=float(args["bull_strength"]),
      bear_strength=float(args["bear_strength"]),
      decisive_factors=[args.get("rationale", "")] if args.get("rationale") else ["(no rationale provided)"],
      acknowledged_risks=["(risk check evaluation only)"],
      conditions_to_change_mind=["(evaluated via MCP)"],
      position_sizing_guidance=args["position_sizing"],
      rationale=args.get("rationale", ""),
    )
    portfolio = portfolio_stats(paper=True)
    basket = [p["ticker"].upper() for p in open_positions(paper=True) or []]
    return verdict, portfolio, basket

  async def risk_check_proposed_trade(self, args: Dict[str, Any]) -> List[TextContent]:
    """Evaluate-only Risk_Officer wrapper. Never calls submit_order.

    Returns the RiskDecision serialized as JSON. Tool callers should respect
    `approve` strictly: a False approve means do NOT call place_paper_order.
    """
    try:
      from agent.Risk_Officer import Risk_Officer
      verdict, portfolio, basket = self._build_verdict_and_portfolio(args)
      ro = Risk_Officer()
      decision = ro.evaluate(
        proposed_quantity=float(args["quantity"]),
        proposed_price=float(args["price"]),
        arbiter_verdict=verdict,
        portfolio=portfolio,
        proposed_ticker=str(args["ticker"]).upper(),
        open_basket=basket,
      )
      out = {
        "approve": decision.approve,
        "reasons": decision.reasons,
        "adjusted_quantity": decision.adjusted_quantity,
        "adjusted_dollar_size": decision.adjusted_dollar_size,
      }
    except Exception as e:
      out = {
        "approve": False,
        "reasons": [f"risk_check_failed: {type(e).__name__}: {e}"],
        "error": True,
      }
    return [TextContent(type="text", text=_safe_dumps(out))]

  async def place_paper_order(self, args: Dict[str, Any]) -> List[TextContent]:
    """Risk_Officer-gated order placement.

    CRITICAL INVARIANT: Risk_Officer.evaluate() is called FIRST. If it
    returns approve=False, this method returns immediately with
    success=False and DOES NOT touch the broker. There is no bypass path.
    """
    from agent.Risk_Officer import Risk_Officer
    from agent.Execution_Agent import Execution_Agent

    try:
      verdict, portfolio, basket = self._build_verdict_and_portfolio(args)
    except Exception as e:
      return [TextContent(type="text", text=_safe_dumps({
        "success": False,
        "error": f"verdict_build_failed: {type(e).__name__}: {e}",
        "risk_decision": {"approve": False,
                           "reasons": ["could not build verdict"]},
      }))]

    ro = Risk_Officer()
    try:
      decision = ro.evaluate(
        proposed_quantity=float(args["quantity"]),
        proposed_price=float(args["price"]),
        arbiter_verdict=verdict,
        portfolio=portfolio,
        proposed_ticker=str(args["ticker"]).upper(),
        open_basket=basket,
      )
    except Exception as e:
      return [TextContent(type="text", text=_safe_dumps({
        "success": False,
        "error": f"risk_evaluation_failed: {type(e).__name__}: {e}",
        "risk_decision": {"approve": False, "reasons": [str(e)]},
      }))]

    risk_decision_json = {
      "approve": decision.approve,
      "reasons": decision.reasons,
      "adjusted_quantity": decision.adjusted_quantity,
      "adjusted_dollar_size": decision.adjusted_dollar_size,
    }

    # HARD GATE: rejection means no broker call. Period.
    if not decision.approve:
      return [TextContent(type="text", text=_safe_dumps({
        "success": False,
        "error": "risk_rejected: " + "; ".join(decision.reasons or ["no reason"]),
        "risk_decision": risk_decision_json,
      }))]

    # Approved. Use adjusted_quantity if Risk_Officer capped/halved the size.
    effective_qty = (
      decision.adjusted_quantity
      if decision.adjusted_quantity is not None
      else float(args["quantity"])
    )

    def _blocking_place():
      ea = Execution_Agent(paper=True)
      return ea.place_order(
        ticker=str(args["ticker"]),
        quantity=effective_qty,
        side=str(args["side"]),
        order_type="market",
        thesis_id=args.get("thesis_id"),
        arbiter_verdict_id=args.get("arbiter_verdict_id"),
      )

    try:
      broker_result = await asyncio.wait_for(
        asyncio.to_thread(_blocking_place), timeout=20.0
      )
    except asyncio.TimeoutError:
      broker_result = {"success": False, "error": "broker_timeout_20s"}
    except Exception as e:
      broker_result = {"success": False,
                        "error": f"{type(e).__name__}: {e}"}

    # Merge broker result with risk_decision for audit.
    out = dict(broker_result)
    out["risk_decision"] = risk_decision_json
    out["qty"] = effective_qty
    return [TextContent(type="text", text=_safe_dumps(out))]

  async def close_paper_position(self, args: Dict[str, Any]) -> List[TextContent]:
    """Wraps Execution_Agent.close_position_for(). `reason` required for
    audit; the underlying agent already handles the no-open-position case."""
    ticker = args.get("ticker", "")
    reason = (args.get("reason") or "").strip()
    if not reason:
      return [TextContent(type="text", text=_safe_dumps({
        "success": False,
        "error": "reason_required: a non-empty `reason` field is required for audit",
      }))]
    def _blocking_close():
      from agent.Execution_Agent import Execution_Agent
      ea = Execution_Agent(paper=True)
      return ea.close_position_for(ticker=ticker, reason=reason)

    try:
      result = await asyncio.wait_for(
        asyncio.to_thread(_blocking_close), timeout=20.0
      )
    except asyncio.TimeoutError:
      result = {"success": False, "error": "broker_timeout_20s"}
    except Exception as e:
      result = {"success": False,
                 "error": f"{type(e).__name__}: {e}"}
    return [TextContent(type="text", text=_safe_dumps(result))]

  async def get_paper_positions(self) -> List[TextContent]:
    """Reconcile broker positions against the local SQLite audit log.
    Either side failing surfaces as `error` but the other side's data is
    still returned so callers can decide what to trust."""
    broker_positions: List[Dict[str, Any]] = []
    broker_error: str | None = None

    def _fetch_broker_positions():
      from agent.Execution_Agent import Execution_Agent
      agent = Execution_Agent(paper=True)
      client = agent._get_client()
      out = []
      for p in client.get_all_positions() or []:
        out.append({
          "symbol": str(getattr(p, "symbol", "")).upper(),
          "qty": float(getattr(p, "qty", 0) or 0),
          "side": str(getattr(p, "side", "long")),
          "market_value": float(getattr(p, "market_value", 0) or 0),
          "avg_entry_price": float(getattr(p, "avg_entry_price", 0) or 0),
        })
      return out

    try:
      broker_positions = await asyncio.wait_for(
        asyncio.to_thread(_fetch_broker_positions), timeout=15.0
      )
    except asyncio.TimeoutError:
      broker_error = "broker_timeout_15s"
    except Exception as e:
      broker_error = f"{type(e).__name__}: {e}"

    local_positions: List[Dict[str, Any]] = []
    local_error: str | None = None
    try:
      from state.positions import open_positions
      for p in open_positions(paper=True) or []:
        local_positions.append({
          "position_id": p.get("position_id"),
          "ticker": str(p.get("ticker", "")).upper(),
          "side": p.get("side"),
          "quantity": p.get("quantity"),
          "entry_price": p.get("entry_price"),
          "thesis_id": p.get("thesis_id"),
        })
    except Exception as e:
      local_error = f"{type(e).__name__}: {e}"

    broker_tickers = {b["symbol"] for b in broker_positions}
    local_tickers = {l["ticker"] for l in local_positions}
    missing_locally = sorted(broker_tickers - local_tickers)
    missing_at_broker = sorted(local_tickers - broker_tickers)
    discrepancies = (
      [f"missing_locally:{t}" for t in missing_locally]
      + [f"missing_at_broker:{t}" for t in missing_at_broker]
    )
    reconciled = (not discrepancies) and (broker_error is None) and (local_error is None)

    out: Dict[str, Any] = {
      "broker_positions": broker_positions,
      "local_positions": local_positions,
      "reconciled": reconciled,
      "discrepancies": discrepancies,
    }
    if broker_error or local_error:
      out["error"] = "; ".join(filter(None, [broker_error, local_error]))
    return [TextContent(type="text", text=_safe_dumps(out))]

  async def get_paper_account(self) -> List[TextContent]:
    """Wraps Execution_Agent.get_account_summary() in an MCP TextContent
    envelope. The broker call is synchronous and could block the asyncio
    event loop for many seconds against a hung/unreachable Alpaca host,
    so we offload to a worker thread with a 15s timeout. The MCP server's
    JSON-RPC keepalive would otherwise drop the connection."""
    def _blocking():
      from agent.Execution_Agent import Execution_Agent
      agent = Execution_Agent(paper=True)
      return agent.get_account_summary()
    try:
      summary = await asyncio.wait_for(asyncio.to_thread(_blocking), timeout=15.0)
    except asyncio.TimeoutError:
      summary = {"paper": True, "error": "broker_timeout_15s"}
    except Exception as e:
      summary = {"paper": True, "error": f"{type(e).__name__}: {e}"}
    return [TextContent(type="text", text=_safe_dumps(summary))]

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return await parent.list_tools_descriptors()

    @self.server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
      if name == "ping_alpaca":
        return [TextContent(type="text", text=_safe_dumps({"status": "pong"}))]
      if name == "get_paper_account":
        return await parent.get_paper_account()
      if name == "get_paper_positions":
        return await parent.get_paper_positions()
      if name == "risk_check_proposed_trade":
        return await parent.risk_check_proposed_trade(arguments)
      if name == "place_paper_order":
        return await parent.place_paper_order(arguments)
      if name == "close_paper_position":
        return await parent.close_paper_position(arguments)
      return [TextContent(
        type="text",
        text=_safe_dumps({"error": f"unknown tool: {name}"}),
      )]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(
          read_stream,
          write_stream,
          InitializationOptions(
            server_name="nemo_alpaca",
            server_version="0.1.0",
            capabilities=ServerCapabilities(),
          ),
        )
        print("Successfully created alpaca process", file=sys.stderr, flush=True)
    except Exception:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.alpaca.server server", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "server":
    print("Starting alpaca process", file=sys.stderr, flush=True)
    try:
      server = AlpacaServer()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.alpaca.server server", file=sys.stderr)
    sys.exit(1)
