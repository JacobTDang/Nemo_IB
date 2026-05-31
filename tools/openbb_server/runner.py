"""Subprocess runner for OpenBB SDK calls.

Invoked by the MCP server as an isolated child process so that OpenBB's
heavy SDK state never shares an asyncio event loop with the MCP framework.
Running in a fresh process avoids the Windows asyncio/thread deadlock that
caused nemo_openbb tool calls to hang indefinitely when dispatched via
asyncio.to_thread inside the MCP server process.

Usage (internal — called by server.py):
  python runner.py <tool_name> <json_args>

Writes exactly one JSON line to stdout:
  {"success": true, "data": ...}    on success
  {"success": false, "error": "..."}  on failure

OpenBB's startup output (extension greetings, etc.) goes to stderr and is
not captured by the calling MCP server process.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Serialization helpers (duplicated from server.py to keep runner standalone)
# ---------------------------------------------------------------------------

def _to_records(obbject) -> List[Dict[str, Any]]:
    if obbject is None:
        return []
    d = None
    if hasattr(obbject, 'model_dump'):
        d = obbject.model_dump()
    elif isinstance(obbject, dict):
        d = obbject
    if not d:
        return []
    results = d.get('results')
    if isinstance(results, list):
        out = []
        for r in results:
            if hasattr(r, 'model_dump'):
                out.append(r.model_dump())
            elif isinstance(r, dict):
                out.append(r)
        return out
    if isinstance(results, dict):
        return [results]
    return []


def _serialize_safe(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)


def _ok(data: Any) -> None:
    print(json.dumps({"success": True, "data": data}, default=_serialize_safe),
          flush=True)


def _fail(msg: str) -> None:
    print(json.dumps({"success": False, "error": msg}), flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 3:
        _fail("usage: runner.py <tool_name> <json_args>")
        return 1

    tool_name = sys.argv[1]
    try:
        args = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        _fail(f"invalid json args: {exc}")
        return 1

    try:
        from openbb import obb  # noqa: PLC0415
    except Exception as exc:
        _fail(f"openbb import failed: {type(exc).__name__}: {exc}")
        return 1

    try:
        if tool_name == "obb_insider_trading":
            ticker = args["ticker"]
            limit = int(args.get("limit", 50))
            result = obb.equity.ownership.insider_trading(symbol=ticker, limit=limit)
            _ok(_to_records(result))

        elif tool_name == "obb_news_company":
            ticker = args["ticker"]
            limit = int(args.get("limit", 30))
            result = obb.news.company(symbol=ticker, limit=limit)
            _ok(_to_records(result))

        elif tool_name == "obb_options_chain":
            ticker = args["ticker"]
            result = obb.derivatives.options.chains(symbol=ticker)
            records = _to_records(result)
            # Truncate to 200 rows — full chain can be 2000+ rows
            _ok({
                "rows": records[:200],
                "total_rows": len(records),
                "returned": min(len(records), 200),
            })

        elif tool_name == "obb_analyst_consensus":
            ticker = args["ticker"]
            result = obb.equity.estimates.consensus(symbol=ticker)
            _ok(_to_records(result))

        else:
            _fail(f"unknown tool: {tool_name}")
            return 1

        return 0

    except Exception as exc:
        _fail(f"{type(exc).__name__}: {str(exc)[:300]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
