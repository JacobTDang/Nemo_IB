"""Phase 1b: Materiality_Classifier — real Groq API calls against fixture headlines.

Strong test goals:
1. Material events are flagged is_material=True
2. Noise is flagged is_material=False
3. Categories match the heuristic table
4. Tickers are extracted from text, not hallucinated
5. Bullish vs bearish direction is identified correctly
6. Urgency reflects time-criticality

This test makes ~12 Groq API calls (~$0.00 on free tier). Run sparingly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Materiality_Classifier import Materiality_Classifier, MaterialityResult


def _is_result(r):
  assert r is not None, "classifier returned None — parse failed or API down"
  assert isinstance(r, MaterialityResult), f"wrong type: {type(r)}"


# --- Fixtures: (label, headline, summary, expected_material, expected_category_hint, expected_signal_hint)
MATERIAL_FIXTURES = [
  ("earnings_beat",
   "Apple beats Q3 estimates with $89.5B revenue, EPS $2.05 vs $1.91 expected",
   "Apple Inc (AAPL) reported third-quarter earnings that exceeded Wall Street expectations. "
   "Revenue of $89.5B beat consensus of $85.2B. iPhone revenue rose 12% YoY. "
   "Services hit an all-time high of $24.2B.",
   True, "earnings", "bullish"),

  ("earnings_miss",
   "Tesla misses Q2 deliveries, guides down full-year",
   "Tesla (TSLA) delivered 423K vehicles vs 445K expected. CEO Elon Musk warned that "
   "demand softness will continue through Q3. Lowers full-year guidance.",
   True, "earnings", "bearish"),

  ("m_and_a",
   "Microsoft to acquire Activision Blizzard for $68.7 billion",
   "Microsoft (MSFT) announced a definitive agreement to acquire Activision Blizzard (ATVI) "
   "in an all-cash deal valuing ATVI at $95/share. Expected to close in fiscal 2023.",
   True, "m_and_a", "bullish"),

  ("ceo_departure",
   "JPMorgan CEO Jamie Dimon to step down by end of year, succession plan unclear",
   "Jamie Dimon, longtime CEO of JPMorgan Chase (JPM), announced he will step down by year-end. "
   "The board has not yet identified a successor.",
   True, "exec_change", None),  # signal could be bearish or neutral — both defensible

  ("fed_rate_decision",
   "Fed holds rates steady, signals one more hike possible this year",
   "The Federal Reserve held the federal funds rate at 5.25-5.50%. Powell said one more "
   "25bp hike remains on the table if inflation does not continue cooling.",
   True, "macro", None),  # signal is mixed/neutral — don't strictly check
]

NOISE_FIXTURES = [
  ("opinion_piece",
   "Why I think Apple's stock might do something interesting this year",
   "An opinion column speculating about Apple without citing earnings, product news, "
   "or specific catalysts. Author thinks the stock is interesting.",
   False),

  ("routine_coverage",
   "Apple Watch features list updated on company website",
   "The company refreshed its product page listing standard Apple Watch features. "
   "No new product or feature was announced.",
   False),

  ("vague_movement",
   "Tesla stock up 2% on no specific news",
   "Shares of TSLA rose 2% in early trading. No corporate catalyst or news event "
   "was identified as the driver.",
   False),
]


def test_material_fixtures():
  cls = Materiality_Classifier()
  failures = []
  for label, headline, summary, expected_mat, exp_cat, exp_signal in MATERIAL_FIXTURES:
    print(f"\n--- {label} ---")
    r = cls.classify(headline, summary, source="test:fixture")
    _is_result(r)
    print(f"  is_material={r.is_material} cat={r.category} sig={r.directional_signal} "
          f"urg={r.urgency} conf={r.confidence:.2f}")
    print(f"  reason: {r.one_line_reason[:120]}")
    if r.is_material != expected_mat:
      failures.append(f"{label}: expected is_material={expected_mat}, got {r.is_material}")
    if exp_cat and exp_cat not in r.category:
      # softer match: category should contain or equal the expected
      if r.category != exp_cat:
        failures.append(f"{label}: expected category~={exp_cat}, got {r.category}")
    if exp_signal and r.directional_signal != exp_signal:
      failures.append(f"{label}: expected signal={exp_signal}, got {r.directional_signal}")
  assert not failures, f"Material fixtures had {len(failures)} failures:\n  " + "\n  ".join(failures)
  print(f"\nPASS: all {len(MATERIAL_FIXTURES)} material fixtures classified correctly")


def test_noise_fixtures():
  cls = Materiality_Classifier()
  failures = []
  for label, headline, summary, expected_mat in NOISE_FIXTURES:
    print(f"\n--- {label} ---")
    r = cls.classify(headline, summary, source="test:fixture")
    _is_result(r)
    print(f"  is_material={r.is_material} conf={r.confidence:.2f}")
    print(f"  reason: {r.one_line_reason[:120]}")
    if r.is_material != expected_mat:
      failures.append(f"{label}: expected is_material={expected_mat}, got {r.is_material} (conf={r.confidence})")
  assert not failures, f"Noise fixtures had {len(failures)} failures:\n  " + "\n  ".join(failures)
  print(f"\nPASS: all {len(NOISE_FIXTURES)} noise fixtures classified as non-material")


def test_no_hallucinated_tickers():
  """Unknown company should NOT produce a fake ticker."""
  cls = Materiality_Classifier()
  r = cls.classify(
    headline="Acme Anvil Corp reports quarterly earnings",
    summary="Acme Anvil, a private maker of cartoon-inspired safety equipment, "
            "reported $12M in revenue.",
    source="test:fixture"
  )
  _is_result(r)
  # If a primary_ticker is set, it must not be a real public company we know
  # Most importantly, AAPL/MSFT/GOOGL shouldn't appear since they aren't mentioned
  known_real = {'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'KO', 'XOM'}
  assert not any(t in known_real for t in r.affected_tickers), \
    f"hallucinated real ticker for fake company: {r.affected_tickers}"
  print(f"PASS: no hallucinated tickers (got affected={r.affected_tickers}, primary={r.primary_ticker})")


def test_ticker_extraction_when_present():
  """Tickers mentioned in text should be extracted."""
  cls = Materiality_Classifier()
  r = cls.classify(
    headline="Both NVDA and AMD report strong AI chip demand",
    summary="Nvidia (NVDA) and Advanced Micro Devices (AMD) cited record demand for AI "
            "accelerators in their respective earnings calls.",
    source="test:fixture"
  )
  _is_result(r)
  assert 'NVDA' in r.affected_tickers, f"NVDA missing from affected: {r.affected_tickers}"
  assert 'AMD' in r.affected_tickers, f"AMD missing from affected: {r.affected_tickers}"
  print(f"PASS: extracted both NVDA and AMD (affected={r.affected_tickers})")


def test_urgency_levels_distinct():
  """Earnings announcement = immediate; routine industry trend = watch."""
  cls = Materiality_Classifier()
  r_now = cls.classify(
    "Nvidia smashes Q2 earnings, guides Q3 above consensus",
    "Nvidia (NVDA) reported $30B in Q2 revenue and guided Q3 to $33B vs $31.5B consensus. "
    "Stock up 8% after-hours.",
    "test:fixture"
  )
  r_slow = cls.classify(
    "Software industry sees gradual shift toward AI co-pilots over next 3 years",
    "A long-term sector report predicts that AI co-pilot tools will gradually replace "
    "some traditional SaaS features over a multi-year horizon.",
    "test:fixture"
  )
  _is_result(r_now); _is_result(r_slow)
  assert r_now.urgency in ('immediate', 'hours'), f"earnings should be immediate/hours, got {r_now.urgency}"
  assert r_slow.urgency in ('days', 'watch'), f"long-term trend should be days/watch, got {r_slow.urgency}"
  print(f"PASS: urgency distinction (earnings={r_now.urgency}, trend={r_slow.urgency})")


if __name__ == "__main__":
  test_material_fixtures()
  test_noise_fixtures()
  test_no_hallucinated_tickers()
  test_ticker_extraction_when_present()
  test_urgency_levels_distinct()
  print("\nAll Phase 1b materiality classifier tests passed.")
