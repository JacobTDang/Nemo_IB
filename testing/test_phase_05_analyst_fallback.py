"""Phase 5 (analyst fallback): _fallback_report_from_raw recognizes a
broader directional-vocabulary set.

Pre-fix the fallback's keyword list was narrow (`STRONG BUY`, `BULLISH`,
`STRONG SELL`, `BEARISH`, `HOLD`, `NEUTRAL`). Anything outside fell to
`INFO`, which then suppresses the Bull/Bear debate in analyze_node. Real
directional analyses that used `outperform`, `appreciate`, `attractive`,
etc. were silently degraded.

These tests are zero-LLM — they call the static fallback directly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Analysis_Agent import Financial_Analysis_Agent


def _fb(raw: str):
  return Financial_Analysis_Agent._fallback_report_from_raw(raw, "test query")


def test_fallback_extracts_attractive_as_buy():
  r = _fb("AAPL looks attractive at current levels given Services growth and "
          "buyback capacity. Consensus underestimates Q3 print.")
  assert r.recommendation == 'BUY', \
    f"'attractive' should yield BUY, got {r.recommendation!r}"
  assert r.signal == 'bullish'
  print(f"PASS: 'attractive' -> BUY/bullish")


def test_fallback_extracts_overvalued_as_sell():
  r = _fb("NVDA appears overvalued given the deceleration in hyperscaler "
          "capex and the multiple compression risk.")
  assert r.recommendation == 'SELL', \
    f"'overvalued' should yield SELL, got {r.recommendation!r}"
  assert r.signal == 'bearish'
  print(f"PASS: 'overvalued' -> SELL/bearish")


def test_fallback_extracts_outperform_as_buy():
  r = _fb("The semiconductor sector is set to outperform broader tech over "
          "the next 12 months as AI capex accelerates.")
  assert r.recommendation == 'BUY', \
    f"'outperform' should yield BUY, got {r.recommendation!r}"
  assert r.signal == 'bullish'
  print(f"PASS: 'outperform' -> BUY/bullish")


def test_fallback_extracts_underperform_as_sell():
  r = _fb("Likely to underperform peers in the coming quarters due to "
          "margin pressure and decelerating bookings.")
  assert r.recommendation == 'SELL', \
    f"'underperform' should yield SELL, got {r.recommendation!r}"
  assert r.signal == 'bearish'
  print(f"PASS: 'underperform' -> SELL/bearish")


def test_fallback_extracts_undervalued_as_buy():
  r = _fb("KO appears undervalued relative to its dividend yield profile "
          "and recent organic growth re-acceleration.")
  assert r.recommendation == 'BUY', \
    f"'undervalued' should yield BUY, got {r.recommendation!r}"
  print(f"PASS: 'undervalued' -> BUY")


def test_fallback_keeps_info_for_truly_directionless():
  r = _fb("Apple was founded in 1976 in Cupertino, California, by Steve "
          "Jobs, Steve Wozniak, and Ronald Wayne. The company is headquartered "
          "in Apple Park and employs about 165,000 people.")
  assert r.recommendation == 'INFO', \
    f"directionless factual text should stay INFO, got {r.recommendation!r}"
  assert r.signal == 'n/a'
  print("PASS: factual directionless text -> INFO/n/a")


def test_fallback_existing_bullish_keyword_still_works():
  """Regression guard: the original 'BULLISH' keyword path still works."""
  r = _fb("Our view on AAPL remains BULLISH with a 12-month target of $230.")
  assert r.recommendation == 'BUY'
  assert r.signal == 'bullish'
  print("PASS: existing BULLISH keyword still yields BUY (regression guard)")


def test_fallback_existing_hold_keyword_still_works():
  r = _fb("We HOLD our position pending Q3 confirmation of the inflection.")
  assert r.recommendation == 'HOLD'
  assert r.signal == 'neutral'
  print("PASS: existing HOLD keyword still yields HOLD (regression guard)")


if __name__ == "__main__":
  test_fallback_extracts_attractive_as_buy()
  test_fallback_extracts_overvalued_as_sell()
  test_fallback_extracts_outperform_as_buy()
  test_fallback_extracts_underperform_as_sell()
  test_fallback_extracts_undervalued_as_buy()
  test_fallback_keeps_info_for_truly_directionless()
  test_fallback_existing_bullish_keyword_still_works()
  test_fallback_existing_hold_keyword_still_works()
  print("\nAll Phase 5 analyst fallback tests passed.")
