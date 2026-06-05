"""Subprocess runner for FinBERT financial sentiment scoring.

Invoked by altdata_server/server.py as an isolated child process so the
440MB transformer model never lives in the MCP server process.

Usage (internal):
  python finbert_runner.py get_finbert_sentiment <json_args>

Writes exactly one JSON line to stdout:
  {"success": true, "data": {...}}    on success
  {"success": false, "error": "..."}  on failure

Model: ProsusAI/finbert (downloaded to HuggingFace cache on first call,
~440MB). Subsequent calls load from disk in ~3-5s on CPU.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _ok(data: Any) -> None:
    print(json.dumps({"success": True, "data": data}, default=str), flush=True)


def _fail(msg: str) -> None:
    print(json.dumps({"success": False, "error": msg}), flush=True)


def _classify(net_score: float) -> str:
    if net_score > 0.15:
        return "bullish"
    if net_score < -0.15:
        return "bearish"
    return "neutral"


def main() -> int:
    if len(sys.argv) < 3:
        _fail("usage: finbert_runner.py <tool_name> <json_args>")
        return 1

    tool_name = sys.argv[1]
    try:
        args = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        _fail(f"invalid json args: {exc}")
        return 1

    if tool_name != "get_finbert_sentiment":
        _fail(f"unknown tool: {tool_name}")
        return 1

    texts = args.get("texts")
    ticker = str(args.get("ticker", ""))

    if not texts or not isinstance(texts, list):
        _fail("texts must be a non-empty list of strings")
        return 1

    texts = [str(t) for t in texts[:50]]  # cap at 50 articles

    try:
        from transformers import pipeline
    except ImportError as exc:
        _fail(f"transformers not installed: {exc}")
        return 1

    try:
        # Suppress HuggingFace progress bars in subprocess output
        import os
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,          # CPU
            max_length=512,
            truncation=True,
        )
        results = pipe(texts, batch_size=8)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {str(exc)[:400]}")
        return 1

    # Aggregate: weighted net score
    pos_score = sum(r["score"] for r in results if r["label"] == "positive")
    neg_score = sum(r["score"] for r in results if r["label"] == "negative")
    net_score = round((pos_score - neg_score) / len(results), 4)

    by_label = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    _ok({
        "ticker": ticker.upper(),
        "article_count": len(texts),
        "net_score": net_score,
        "signal": _classify(net_score),
        "label_counts": by_label,
        "per_article": [{"label": r["label"], "score": round(r["score"], 4)}
                        for r in results],
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
