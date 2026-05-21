"""End-to-end stress tests for the RAG search layer.

Covers:
  1. Tiny known corpus seed -> known-fact retrieval
  2. Negative case: query with no relevant content scores low
  3. Metadata filters: ticker filter narrows results
  4. Metadata filters: doc_type filter narrows results
  5. MCP roundtrip: launch web_search server via stdio and call rag_search
  6. Performance: p50 query latency < 200ms on the tiny corpus

All test docs use the SEARCH_TEST_ doc_id prefix so cleanup is targeted and
the production / bootstrapped corpus (if any) is never touched.

The ingest layer is being built in parallel by another agent. To stay
independent we seed using the lower-level primitives directly:
chunker.chunk_text -> embedder.embed -> store.insert_chunk. That mirrors what
ingest_document will eventually do internally.

Run via:
  ./.venv/Scripts/python.exe testing/test_rag_search.py
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from agent.rag.chunker import chunk_text
from agent.rag.embedder import embed
from agent.rag.store import insert_chunk, delete_by_doc_id, count_chunks
from agent.rag.search import rag_search


# ---------------------------------------------------------------------------
# Test infrastructure (same shape as test_falsifier_watcher_e2e.py)
# ---------------------------------------------------------------------------

_results = {'pass': 0, 'fail': 0, 'failures': []}
_DOC_ID_PREFIX = 'SEARCH_TEST_'
_seeded_doc_ids: List[str] = []


def _check(name: str, condition, hint: str = ''):
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _section(title: str):
  print(f"\n=== {title} ===")


def _seed_doc(
    doc_suffix: str,
    text: str,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    source_tool: Optional[str] = 'test_rag_search',
    section_heading: Optional[str] = None,
) -> str:
  """Insert one synthetic document. Returns its doc_id.

  We bypass the (yet-to-be-built) ingest_document function and call
  insert_chunk directly so this test doesn't depend on the parallel
  workstream. The chunker is still exercised so we get realistic chunk
  shapes.
  """
  doc_id = _DOC_ID_PREFIX + doc_suffix
  # Tiny test docs fit comfortably in a single chunk; chunk_text still runs
  # cleanly because it handles short text as one paragraph.
  chunks = chunk_text(text, target_tokens=500, overlap_tokens=0,
                      section_heading=section_heading)
  if not chunks:
    # Single-paragraph fallback for very short text the chunker treats as empty.
    chunks = [{
      'chunk_text':      text,
      'chunk_offset':    0,
      'chunk_sequence':  0,
      'section_heading': section_heading,
      'token_count':     len(text.split()),
    }]

  for ch in chunks:
    chunk_dict = {
      'doc_id':          doc_id,
      'ticker':          ticker,
      'source_tool':     source_tool,
      'doc_type':        doc_type,
      'filing_date':     None,
      'item_number':     None,
      'section_heading': ch.get('section_heading'),
      'chunk_text':      ch['chunk_text'],
      'chunk_offset':    ch.get('chunk_offset', 0),
      'chunk_sequence':  ch.get('chunk_sequence', 0),
    }
    vec = embed(ch['chunk_text'])
    insert_chunk(chunk_dict, vec)

  _seeded_doc_ids.append(doc_id)
  return doc_id


def _cleanup_seeded():
  """Remove every SEARCH_TEST_ doc we created. Idempotent."""
  for doc_id in list(_seeded_doc_ids):
    try:
      delete_by_doc_id(doc_id)
    except Exception as exc:
      print(f"  cleanup: failed to delete {doc_id}: {exc}", file=sys.stderr)
  # Belt-and-suspenders: sweep any orphans that slipped through (e.g. from
  # a previous failed run on the same machine).
  conn = get_connection()
  try:
    rows = conn.execute(
      "SELECT doc_id FROM rag_chunks WHERE doc_id LIKE ?",
      (_DOC_ID_PREFIX + '%',)
    ).fetchall()
    leftover_ids = [r['doc_id'] for r in rows]
  finally:
    conn.close()
  for doc_id in leftover_ids:
    try:
      delete_by_doc_id(doc_id)
    except Exception:
      pass
  _seeded_doc_ids.clear()


# ---------------------------------------------------------------------------
# Tiny known corpus
# ---------------------------------------------------------------------------

# 5 synthetic docs with very different topics so cosine separation is clean.
_CORPUS = [
  {
    'suffix':   'MSFT_AZURE',
    'text':     ('Microsoft Azure cloud revenue accelerated to 21% year-over-year '
                 'growth in fiscal Q3, marking the third consecutive quarter of '
                 're-acceleration after the post-pandemic digestion period. '
                 'Hyperscaler capex continues to climb to support AI inference workloads.'),
    'ticker':   'MSFT',
    'doc_type': 'analyst_writeup',
  },
  {
    'suffix':   'AAPL_IPHONE_CHINA',
    'text':     ('Apple iPhone unit sales weakness in mainland China has now '
                 'persisted for multiple consecutive quarters. Local competitors '
                 'Huawei and Xiaomi have recaptured premium share. Wait times for '
                 'the latest flagship in Tier 1 cities suggest soft demand.'),
    'ticker':   'AAPL',
    'doc_type': 'analyst_writeup',
  },
  {
    'suffix':   'DOTCOM_1999',
    'text':     ('At the 1999 dot-com peak, aggregate technology-sector capex '
                 'ratios as a percentage of revenue exceeded every prior cycle '
                 'in postwar U.S. economic history. The unwind in 2000-2002 '
                 'compressed multiples by more than 70 percent peak to trough.'),
    'ticker':   None,
    'doc_type': 'analogue',
  },
  {
    'suffix':   'FED_RATES',
    'text':     ('The Federal Reserve held the target federal funds range steady '
                 'at 4.25 to 4.50 percent following the latest FOMC meeting. '
                 'Powell signaled patience on cuts pending further evidence that '
                 'core services inflation is moving sustainably toward target.'),
    'ticker':   None,
    'doc_type': 'rule',
  },
  {
    'suffix':   'PERSHING_MSFT',
    'text':     ('Pershing Square Capital Management initiated a position in '
                 'Microsoft during the first quarter, per the latest 13F-HR '
                 'filing. Ackman framed the position around durable enterprise '
                 'software economics and the Azure AI optionality.'),
    'ticker':   'MSFT',
    'doc_type': 'analyst_writeup',
  },
]


def _seed_corpus():
  """Seed the 5-doc baseline used by tests 1, 2, 3, 6."""
  for doc in _CORPUS:
    _seed_doc(
      doc_suffix=doc['suffix'],
      text=doc['text'],
      ticker=doc['ticker'],
      doc_type=doc['doc_type'],
    )


# ---------------------------------------------------------------------------
# 1. Known-fact retrieval
# ---------------------------------------------------------------------------

def test_known_fact_retrieval():
  _section("1. Known-fact retrieval")

  azure = rag_search("Azure cloud growth", top_k=5)
  top_ids = [r['doc_id'] for r in azure['results']]
  _check("  'Azure cloud growth' -> MSFT_AZURE in top-2",
         _DOC_ID_PREFIX + 'MSFT_AZURE' in top_ids[:2],
         f"got top_ids={top_ids}")
  if azure['results']:
    _check("  top result has similarity > 0.4",
           azure['results'][0]['similarity'] > 0.4,
           f"top similarity={azure['results'][0]['similarity']:.3f}")

  iphone = rag_search("iPhone sales China", top_k=5)
  top_iphone = [r['doc_id'] for r in iphone['results']]
  _check("  'iPhone sales China' -> AAPL_IPHONE_CHINA is top-1",
         top_iphone[:1] == [_DOC_ID_PREFIX + 'AAPL_IPHONE_CHINA'],
         f"got top_ids={top_iphone}")

  pershing = rag_search("Pershing Square portfolio additions", top_k=5)
  top_pers = [r['doc_id'] for r in pershing['results']]
  _check("  'Pershing Square portfolio additions' -> PERSHING_MSFT in top-2",
         _DOC_ID_PREFIX + 'PERSHING_MSFT' in top_pers[:2],
         f"got top_ids={top_pers}")


# ---------------------------------------------------------------------------
# 2. Negative case
# ---------------------------------------------------------------------------

def test_negative_case():
  _section("2. Negative case — query with no match")

  res = rag_search("Tesla Model Y delivery numbers Q4 2024", top_k=5)
  if res['results']:
    top_sim = res['results'][0]['similarity']
  else:
    top_sim = 0.0
  _check("  unrelated query: top similarity < 0.5",
         top_sim < 0.5,
         f"top similarity={top_sim:.3f}")

  # With a high min_score the corpus should produce 0 hits and a note.
  strict = rag_search("Tesla Model Y delivery numbers Q4 2024",
                      top_k=5, min_score=0.9)
  _check("  high min_score returns 0 results",
         strict['results_count'] == 0,
         f"got {strict['results_count']} results")
  _check("  empty result envelope carries 'note'",
         'note' in strict,
         f"keys={list(strict.keys())}")


# ---------------------------------------------------------------------------
# 3. Ticker filter
# ---------------------------------------------------------------------------

def test_ticker_filter():
  _section("3. Ticker filter")

  res = rag_search("growth", ticker='MSFT', top_k=10)
  tickers = {r['ticker'] for r in res['results']}
  _check("  ticker='MSFT' returns only MSFT docs",
         tickers.issubset({'MSFT'}) and len(tickers) > 0,
         f"got tickers={tickers}")
  _check("  ticker filter recorded in response",
         res['filters']['ticker'] == 'MSFT')

  res_aapl = rag_search("growth", ticker='AAPL', top_k=10)
  tickers_aapl = {r['ticker'] for r in res_aapl['results']}
  _check("  ticker='AAPL' returns only AAPL docs",
         tickers_aapl.issubset({'AAPL'}) and len(tickers_aapl) > 0,
         f"got tickers={tickers_aapl}")


# ---------------------------------------------------------------------------
# 4. Doc-type filter
# ---------------------------------------------------------------------------

def test_doc_type_filter():
  _section("4. doc_type filter")

  res = rag_search("market cycle", doc_type='analogue', top_k=10)
  types = {r['doc_type'] for r in res['results']}
  _check("  doc_type='analogue' returns only analogue docs",
         types.issubset({'analogue'}) and len(types) > 0,
         f"got types={types}")

  res_wu = rag_search("market cycle", doc_type='analyst_writeup', top_k=10)
  types_wu = {r['doc_type'] for r in res_wu['results']}
  _check("  doc_type='analyst_writeup' returns only analyst_writeup",
         types_wu.issubset({'analyst_writeup'}) and len(types_wu) > 0,
         f"got types={types_wu}")


# ---------------------------------------------------------------------------
# 5. MCP roundtrip
# ---------------------------------------------------------------------------

async def _mcp_roundtrip_async() -> Dict[str, Any]:
  """Launch the web_search MCP server as a subprocess and call rag_search."""
  from mcp import ClientSession, StdioServerParameters
  from mcp.client.stdio import stdio_client

  project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  env = {**os.environ, "PYTHONUNBUFFERED": "1"}
  env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

  params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "tools.web_search_server.web_search", "server"],
    env=env,
  )
  async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
      await session.initialize()
      tools = await session.list_tools()
      tool_names = {t.name for t in tools.tools}
      if 'rag_search' not in tool_names:
        return {'error': 'rag_search not advertised by server',
                'available': sorted(tool_names)}
      resp = await session.call_tool('rag_search', {
        'query': 'Azure cloud growth',
        'top_k': 5,
      })
      payload_text = resp.content[0].text
      return {'tool_names': sorted(tool_names),
              'payload': json.loads(payload_text)}


def test_mcp_roundtrip():
  _section("5. MCP roundtrip — server stdio")

  try:
    out = asyncio.run(_mcp_roundtrip_async())
  except Exception as exc:
    _check("  MCP roundtrip completed without exception", False, str(exc))
    return

  if 'error' in out:
    _check("  rag_search advertised in list_tools", False, out['error'])
    return

  _check("  rag_search + rag_ingest advertised in list_tools",
         'rag_search' in out['tool_names'] and 'rag_ingest' in out['tool_names'],
         f"tools={out['tool_names']}")
  payload = out['payload']
  _check("  response payload parses as JSON",
         isinstance(payload, dict))
  _check("  response includes results list",
         isinstance(payload.get('results'), list),
         f"keys={list(payload.keys())}")
  results = payload.get('results', [])
  top_ids = [r.get('doc_id') for r in results]
  _check("  MSFT_AZURE retrievable via MCP rag_search",
         _DOC_ID_PREFIX + 'MSFT_AZURE' in top_ids,
         f"top_ids={top_ids}")


# ---------------------------------------------------------------------------
# 6. Performance
# ---------------------------------------------------------------------------

def test_performance_latency():
  _section("6. Performance — query latency")

  queries = [
    "Azure cloud growth",
    "iPhone sales China",
    "1999 dot-com bubble",
    "Federal Reserve rate decision",
    "Pershing Square Microsoft position",
    "hyperscaler capex acceleration",
    "premium share recapture",
    "core services inflation",
  ]
  # Warm-up: first call pays the SentenceTransformer load cost; we don't
  # want that in the percentile.
  rag_search("warmup query", top_k=3)

  latencies_ms: List[float] = []
  for q in queries:
    t0 = time.perf_counter()
    rag_search(q, top_k=5)
    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

  p50 = statistics.median(latencies_ms)
  p95 = (sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95) - 1)]
         if latencies_ms else 0.0)
  print(f"  -- p50={p50:.1f}ms  p95={p95:.1f}ms  n={len(latencies_ms)}")
  _check("  p50 query latency under 200ms (tiny corpus)",
         p50 < 200.0,
         f"p50={p50:.1f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  print("\nRAG search — end-to-end stress tests\n")
  init_schema()

  # Pre-clean any leftover SEARCH_TEST_ rows from a previous failed run.
  _cleanup_seeded()

  pre_count = count_chunks()
  print(f"  baseline rag_chunks count = {pre_count}")

  try:
    _seed_corpus()
    seeded_count = count_chunks(filter={'source_tool': 'test_rag_search'})
    print(f"  seeded test chunks = {seeded_count}")
    if seeded_count == 0:
      print("  FATAL: corpus seeding produced 0 chunks", file=sys.stderr)
      return 1

    test_known_fact_retrieval()
    test_negative_case()
    test_ticker_filter()
    test_doc_type_filter()
    test_mcp_roundtrip()
    test_performance_latency()
  finally:
    _cleanup_seeded()

  print(f"\n=== Summary ===")
  print(f"  PASS: {_results['pass']}")
  print(f"  FAIL: {_results['fail']}")
  if _results['failures']:
    print("\nFailures:")
    for name, hint in _results['failures']:
      print(f"  - {name}: {hint}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
