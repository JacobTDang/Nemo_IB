"""Phase 2: Probing_Agent extracts ticker from the user query.

Three real Probing_Agent calls (~30s each on the OpenRouter Nemotron model).
Run sparingly. Resolves the Future/Pending Work item from MEMORY.md:
   "LLM-driven ticker extraction: remove input('ticker: ') from main.py and
    have the Master Orchestrator/Probing Agent extract the ticker directly."
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Probing_Agent import Probing_Agent, ProbingResult


def _probe(query: str) -> dict:
  agent = Probing_Agent()
  # ticker arg deliberately omitted so we exercise the extraction path
  return agent.probe(user_query=query, ticker=None, modeling_tools_context='')


def test_probe_extracts_explicit_ticker_symbol():
  """When the user mentions a ticker symbol directly, extraction must
  populate result.ticker with that symbol."""
  result = _probe("Analyze AAPL fundamentals and give me a recommendation")
  assert 'error' not in result, f"probe parse failed: {result.get('error')}"
  assert result.get('ticker') == 'AAPL', \
    f"explicit AAPL mention should yield ticker='AAPL', got {result.get('ticker')!r}"
  print(f"PASS: extracted ticker={result.get('ticker')!r} from explicit AAPL mention")


def test_probe_infers_ticker_from_company_name():
  """When the user mentions a well-known company by NAME only, the probe
  should infer the ticker. Acceptable: AAPL or None — both are defensible
  for an LLM (some models won't infer without an explicit symbol). The
  hard constraint is that it does not hallucinate the WRONG ticker."""
  result = _probe("What's Apple's Q3 outlook looking like?")
  assert 'error' not in result, f"probe parse failed: {result.get('error')}"
  t = result.get('ticker')
  assert t in ('AAPL', None, ''), \
    f"company-name probe should yield AAPL or null; got {t!r}"
  print(f"PASS: company-name probe yielded ticker={t!r}")


def test_probe_returns_none_when_no_ticker_in_query():
  """Generic queries with no specific company must NOT fabricate a ticker."""
  result = _probe("Explain how DCF valuation works for tech companies in general")
  assert 'error' not in result, f"probe parse failed: {result.get('error')}"
  t = result.get('ticker')
  assert t in (None, '', 'null'), \
    f"generic query must not fabricate a ticker; got {t!r}"
  print(f"PASS: generic query correctly yielded ticker={t!r}")


if __name__ == "__main__":
  test_probe_extracts_explicit_ticker_symbol()
  test_probe_infers_ticker_from_company_name()
  test_probe_returns_none_when_no_ticker_in_query()
  print("\nAll Phase 2 probe ticker extraction tests passed.")
