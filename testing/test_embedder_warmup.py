"""Benchmark: verify the embedder warmup thread eliminates cold-start
latency on the first rag_search MCP call.

Two scenarios:
  1. No warmup (baseline): spawn web_search server, immediately call
     rag_search, measure latency. Expected: ~5-10s on a warm-cache
     machine (deserializing the 80MB model from disk).
  2. With warmup: spawn server, wait briefly for the warmup thread to
     run, then call rag_search. Expected: < 200ms (model already loaded
     into the singleton from the daemon thread).

Run:
  ./.venv/Scripts/python.exe testing/test_embedder_warmup.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition, hint: str = ''):
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _section(t):
  print(f"\n=== {t} ===")


async def _measure_first_rag_search(warmup_wait_s: float) -> float:
  """Spawn the web_search MCP server, wait warmup_wait_s seconds, then
  measure the latency of the first rag_search call. Returns seconds."""
  env = {
    **os.environ,
    'PYTHONPATH': r'C:\Users\UsoSe\OneDrive\Desktop\Projects\Nemo_IB',
    'PYTHONUNBUFFERED': '1',
  }
  params = StdioServerParameters(
    command=sys.executable,
    args=['-u', '-m', 'tools.web_search_server.web_search', 'server'],
    env=env,
  )
  async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
      await session.initialize()
      if warmup_wait_s > 0:
        # Sleep long enough for the warmup daemon thread to import
        # sentence-transformers (~12s torch import) AND load the model
        # (~5s deserialize) AND run a single warmup encode (~2s) on a
        # Windows-CPU host. Total ~20s cold-cache or ~7s warm-cache.
        # The lock in embedder.py serializes any racing main-thread call.
        await asyncio.sleep(warmup_wait_s)
      t0 = time.time()
      r = await session.call_tool(
        'rag_search',
        {'query': 'Azure cloud capex', 'top_k': 3},
      )
      elapsed = time.time() - t0
      # Sanity: must return valid payload
      payload = json.loads(r.content[0].text)
      assert 'results' in payload or 'results_count' in payload, \
        f'unexpected payload shape: {list(payload.keys())[:5]}'
      return elapsed


def main():
  print("\nEmbedder warmup benchmark\n")
  _section("Scenario A: warmup thread had 25s to run before query")
  t_warm = asyncio.run(_measure_first_rag_search(warmup_wait_s=25.0))
  print(f"  first rag_search latency: {t_warm * 1000:.0f}ms")
  _check(
    "  warmed first-call latency < 1000ms",
    t_warm < 1.0,
    f"got {t_warm:.2f}s — warmup thread may not be running",
  )
  _check(
    "  warmed first-call latency < 200ms (target)",
    t_warm < 0.2,
    f"got {t_warm:.2f}s — model loaded but might be CPU-bound",
  )

  _section("Scenario B: no warmup wait — query immediately on connect")
  t_cold = asyncio.run(_measure_first_rag_search(warmup_wait_s=0.0))
  print(f"  first rag_search latency: {t_cold * 1000:.0f}ms")
  # Cold path inherits the same load via the shared lock — no double-load
  _check(
    "  worst-case (no wait) latency < 30s",
    t_cold < 30.0,
    f"got {t_cold:.2f}s — model load took longer than expected",
  )

  _section("Speedup")
  if t_warm > 0:
    speedup = t_cold / t_warm
    print(f"  warmup speedup: {speedup:.1f}x ({t_cold*1000:.0f}ms -> {t_warm*1000:.0f}ms)")

  print(f"\n=== Summary ===\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  if _results['failures']:
    for n, h in _results['failures']:
      print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
