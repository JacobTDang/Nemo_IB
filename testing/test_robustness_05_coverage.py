"""Coverage matrix: full agent end-to-end against diverse ticker classes.

Runs serially. Each ticker is one full agent invocation (~5-10 min).
Pass criteria per ticker:
- Workflow completes within 15 min timeout
- analysis_report is non-empty markdown
- Sensible markdown shape (has at least 'CONCLUSION' section)

Usage:
  python testing/test_robustness_05_coverage.py                # all 10
  python testing/test_robustness_05_coverage.py AAPL JPM       # subset
"""
import asyncio, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.MCP_manager import MCPConnectionManager
from agent.workflows.analysis_workflow import WorkFlow


TICKER_MATRIX = [
  # (ticker, class, expected_difficulty)
  ('AAPL', 'large-cap tech',     'easy'),
  ('JPM',  'bank',                'medium'),  # no GrossProfit XBRL concept
  ('KO',   'consumer staple',     'easy'),    # tests DDM
  ('WMT',  'retail',              'medium'),  # negative NWC
  ('NVDA', 'momentum tech',       'easy'),
  ('XOM',  'energy',              'medium'),  # commodity cyclical
  ('JNJ',  'pharma',              'medium'),  # R&D heavy
  ('PG',   'consumer staple',     'easy'),    # tests DDM
  ('O',    'REIT',                'hard'),    # different cash flow shape
  ('TSLA', 'high-PE volatile',    'medium'),
]


async def run_one(ticker: str, label: str, timeout_s: int = 900):
  print(f"\n{'='*70}\n{ticker} ({label}) starting...\n{'='*70}", flush=True)
  t0 = time.time()
  async with MCPConnectionManager() as mcp:
    wf = WorkFlow(mcp=mcp)
    try:
      report_md = await asyncio.wait_for(
        wf.run(user_query=f"Run a comprehensive analysis on {ticker}", ticker=''),
        timeout=timeout_s,
      )
    except asyncio.TimeoutError:
      print(f"\nFAIL: {ticker} — timed out after {timeout_s}s")
      return False
    except Exception as e:
      print(f"\nFAIL: {ticker} — exception: {type(e).__name__}: {e}")
      return False
  elapsed = time.time() - t0
  if not report_md or len(report_md) < 300:
    print(f"\nFAIL: {ticker} — empty/short report "
          f"({len(report_md or '')} chars), {elapsed:.0f}s")
    return False
  if 'CONCLUSION' not in report_md.upper():
    print(f"\nFAIL: {ticker} — no CONCLUSION section in {len(report_md)} chars")
    return False
  print(f"\nPASS: {ticker} ({label}) in {elapsed:.0f}s, {len(report_md)} chars")
  return True


async def main():
  if len(sys.argv) > 1:
    selected = sys.argv[1:]
    matrix = [t for t in TICKER_MATRIX if t[0] in selected]
    if not matrix:
      print(f"No matching tickers in selection: {selected}")
      print(f"Available: {[t[0] for t in TICKER_MATRIX]}")
      sys.exit(1)
  else:
    matrix = TICKER_MATRIX

  results = {}
  for ticker, label, _ in matrix:
    results[ticker] = await run_one(ticker, label)

  print("\n" + "=" * 70)
  print("COVERAGE MATRIX RESULTS")
  print("=" * 70)
  for ticker, ok in results.items():
    print(f"  {ticker}: {'PASS' if ok else 'FAIL'}")
  total_pass = sum(results.values())
  total = len(results)
  print(f"\nTotal: {total_pass}/{total} passed ({100*total_pass//total}%)")
  # Threshold: 80% pass = "production-ready" assertion for the full matrix
  threshold = max(1, int(0.8 * total))
  sys.exit(0 if total_pass >= threshold else 1)


if __name__ == "__main__":
  asyncio.run(main())
