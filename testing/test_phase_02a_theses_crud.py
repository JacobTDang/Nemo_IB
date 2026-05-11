"""Phase 2a: theses table CRUD + supersede invariants."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.theses import (
  insert_thesis, latest_thesis, supersede_thesis,
  thesis_history, active_theses, get_thesis,
)


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM theses WHERE ticker LIKE 'TEST_%'")
    conn.commit()
  finally:
    conn.close()


def test_insert_and_latest_round_trip():
  init_schema(); _clean()
  tid = insert_thesis(
    ticker="TEST_AAPL", recommendation="BUY", signal="bullish",
    target_price=210.0, stop_loss=170.0, confidence=0.78,
    analysis_summary="Strong services growth outweighs hardware cyclicality.",
    key_assumptions=["iPhone -5% to -2% YoY", "Services +12% YoY"],
    data_gaps=["China revenue breakdown"],
    full_report_md="## RECOMMENDATION: BUY\n..."
  )
  assert isinstance(tid, int) and tid > 0
  latest = latest_thesis("TEST_AAPL")
  assert latest is not None
  assert latest['thesis_id'] == tid
  assert latest['recommendation'] == 'BUY'
  assert latest['target_price'] == 210.0
  assert latest['superseded_by'] is None
  # JSON fields are deserialized back to lists
  assert latest['key_assumptions'] == ["iPhone -5% to -2% YoY", "Services +12% YoY"]
  assert latest['data_gaps'] == ["China revenue breakdown"]
  print(f"PASS: insert + latest_thesis round-trip works (id={tid})")


def test_supersede_only_yields_one_active():
  init_schema(); _clean()
  t1 = insert_thesis(ticker="TEST_MSFT", recommendation="BUY", signal="bullish",
                     target_price=400.0, stop_loss=350.0, confidence=0.7,
                     analysis_summary="v1", key_assumptions=["a"], data_gaps=[],
                     full_report_md="r1")
  t2 = insert_thesis(ticker="TEST_MSFT", recommendation="HOLD", signal="neutral",
                     target_price=405.0, stop_loss=355.0, confidence=0.6,
                     analysis_summary="v2", key_assumptions=["b"], data_gaps=[],
                     full_report_md="r2")
  supersede_thesis(t1, t2)
  active = active_theses()
  msft_active = [t for t in active if t['ticker'] == 'TEST_MSFT']
  assert len(msft_active) == 1, f"expected 1 active MSFT thesis, got {len(msft_active)}"
  assert msft_active[0]['thesis_id'] == t2
  # The superseded one should still be retrievable by id
  old = get_thesis(t1)
  assert old['superseded_by'] == t2
  # Latest finds only t2
  l = latest_thesis("TEST_MSFT")
  assert l['thesis_id'] == t2
  print(f"PASS: supersede chains correctly (t1={t1} -> t2={t2})")


def test_history_preserves_chain():
  init_schema(); _clean()
  ids = []
  for i, rec in enumerate(['BUY', 'HOLD', 'SELL']):
    tid = insert_thesis(
      ticker="TEST_NVDA", recommendation=rec, signal="n/a",
      target_price=None, stop_loss=None, confidence=0.5 + i * 0.1,
      analysis_summary=f"v{i+1}", key_assumptions=[], data_gaps=[],
      full_report_md=f"r{i+1}"
    )
    ids.append(tid)
    if i > 0:
      supersede_thesis(ids[i-1], tid)
  hist = thesis_history("TEST_NVDA")
  assert len(hist) == 3
  # History is newest-first
  assert hist[0]['thesis_id'] == ids[2]
  assert hist[2]['thesis_id'] == ids[0]
  print(f"PASS: thesis_history returns full chain newest-first")


def test_info_recommendation_can_be_stored_explicitly():
  """If someone explicitly stores INFO, it should round-trip (the workflow
  skips INFO at the persistence call site)."""
  init_schema(); _clean()
  tid = insert_thesis(
    ticker="TEST_KO", recommendation="INFO", signal="n/a",
    target_price=None, stop_loss=None, confidence=0.3,
    analysis_summary="factual lookup", key_assumptions=[], data_gaps=[],
    full_report_md="r"
  )
  l = latest_thesis("TEST_KO")
  assert l is not None and l['recommendation'] == 'INFO'
  print("PASS: INFO thesis storable when persisted explicitly")


def test_ticker_normalization_to_upper():
  init_schema(); _clean()
  tid = insert_thesis(ticker="test_jpm", recommendation="HOLD", signal="neutral",
                     target_price=None, stop_loss=None, confidence=0.5,
                     analysis_summary="s", key_assumptions=[], data_gaps=[],
                     full_report_md="r")
  # Latest lookup is case-insensitive
  l = latest_thesis("test_jpm")
  l2 = latest_thesis("TEST_JPM")
  assert l and l2 and l['thesis_id'] == l2['thesis_id']
  assert l['ticker'] == 'TEST_JPM', f"expected uppercase, got {l['ticker']}"
  print("PASS: ticker is normalized to uppercase on insert and lookup")


def test_missing_ticker_returns_none():
  init_schema(); _clean()
  assert latest_thesis("DOES_NOT_EXIST_TEST") is None
  assert get_thesis(999999999) is None
  print("PASS: missing ticker/id returns None safely")


if __name__ == "__main__":
  test_insert_and_latest_round_trip()
  test_supersede_only_yields_one_active()
  test_history_preserves_chain()
  test_info_recommendation_can_be_stored_explicitly()
  test_ticker_normalization_to_upper()
  test_missing_ticker_returns_none()
  _clean()
  print("\nAll Phase 2a theses CRUD tests passed.")
