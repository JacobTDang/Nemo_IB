"""Item 2: structured AnalysisReport replaces regex-parsed markdown."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Analysis_Agent import AnalysisReport, Financial_Analysis_Agent


def test_construct_and_render():
  report = AnalysisReport(
    executive_summary="AAPL is fairly valued at current price given strong margins.",
    recommendation="HOLD",
    signal="neutral",
    valuation="P/E = 35.1, EV/EBITDA = 27.1, DCF fair value = $128",
    financial_performance="Revenue $416B, EBITDA margin 34.8%, ROE 146%",
    macro_context="10Y yield 4.41%, mild risk-off",
    sentiment="Analyst Buy 72%, insider net selling",
    risks=["Competitive pressure in AI", "Valuation premium leaves no room for error"],
    assumptions=["Revenue growth: 8% Y1 declining to 5% Y5 because consensus median"],
    data_gaps=["Segment revenue breakdown not provided"],
    confidence=0.78,
    conclusion="HOLD — fairly valued; wait for compression or catalyst clarity.",
  )
  md = report.render_markdown()
  assert "## EXECUTIVE SUMMARY" in md
  assert "## RECOMMENDATION: HOLD" in md
  assert "## VALUATION" in md
  assert "## DATA GAPS" in md
  assert "Segment revenue breakdown" in md
  assert "## CONFIDENCE: 0.78" in md
  assert "## CONCLUSION" in md
  print(f"PASS: render_markdown produces {len(md)} chars with all sections")


def test_optional_sections_skipped():
  report = AnalysisReport(
    executive_summary="Quick factual lookup answer.",
    recommendation="INFO",
    signal="n/a",
    valuation="(not applicable)",
    financial_performance="Revenue $5B",
    # macro_context and sentiment omitted -> should not appear in markdown
    risks=[],
    assumptions=[],
    data_gaps=[],
    confidence=0.95,
    conclusion="Factual answer delivered.",
  )
  md = report.render_markdown()
  assert "## MACRO CONTEXT" not in md
  assert "## SENTIMENT" not in md
  assert "## DATA GAPS" in md and "None" in md  # always shown even when empty
  print("PASS: optional sections (macro, sentiment) skipped when empty")


def test_fallback_report_recovers_from_markdown():
  """If the LLM emits markdown instead of JSON, the fallback constructor should
  still produce a usable AnalysisReport."""
  raw_md = """## VALUATION
P/E = 35x, premium to sector
DCF says $128 fair value

## CONCLUSION
HOLD — premium multiple needs growth delivery to justify.

## DATA GAPS
- Segment breakdown not available
- Forward guidance not disclosed
"""
  report = Financial_Analysis_Agent._fallback_report_from_raw(raw_md, "Analyze AAPL")
  assert report.recommendation == "HOLD"
  assert report.signal == "neutral"
  assert len(report.data_gaps) >= 2
  assert "Segment" in report.data_gaps[0] or "Segment" in report.data_gaps[1]
  assert report.confidence == 0.4  # low because structure was missing
  print(f"PASS: fallback parsed {len(report.data_gaps)} gaps from raw markdown; "
        f"recommendation={report.recommendation}")


def test_fallback_detects_bullish_signal():
  raw = "Strong BUY recommendation given the bullish setup and earnings beat history."
  report = Financial_Analysis_Agent._fallback_report_from_raw(raw, "test")
  assert report.recommendation == "BUY"
  assert report.signal == "bullish"
  print("PASS: fallback heuristic detects BUY/bullish")


def test_schema_validates_required_fields():
  """Pydantic should reject missing required fields."""
  from pydantic import ValidationError
  try:
    AnalysisReport(
      executive_summary="x",
      recommendation="BUY",
      signal="bullish",
      # valuation missing
      financial_performance="x",
      confidence=0.5,
      conclusion="x",
    )
    assert False, "should have raised ValidationError"
  except ValidationError:
    print("PASS: schema rejects missing required fields")


if __name__ == "__main__":
  test_construct_and_render()
  test_optional_sections_skipped()
  test_fallback_report_recovers_from_markdown()
  test_fallback_detects_bullish_signal()
  test_schema_validates_required_fields()
  print("\nAll tests passed.")
