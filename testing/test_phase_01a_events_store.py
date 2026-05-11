"""Phase 1a: events store CRUD + dedup invariants.

Tests written for correctness, NOT to pass. If logic is wrong here, the entire
news pipeline silently drops or duplicates events. Strong tests required.
"""
import sys, os
import json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.events_store import (
  event_id, store_event, seen, unprocessed_events,
  mark_processed, recent_events_for_ticker
)


def _clean_test_events():
  """Remove only events tagged with our test source."""
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE source LIKE 'test:%'")
    conn.commit()
  finally:
    conn.close()


def test_event_id_is_deterministic():
  a = event_id("test:src1", "Apple reports record earnings", "2026-05-10T10:00:00")
  b = event_id("test:src1", "Apple reports record earnings", "2026-05-10T10:00:00")
  c = event_id("test:src2", "Apple reports record earnings", "2026-05-10T10:00:00")  # different source
  d = event_id("test:src1", "Apple reports OK earnings", "2026-05-10T10:00:00")  # different headline
  e = event_id("test:src1", "Apple reports record earnings", "2026-05-11T10:00:00")  # different date
  assert a == b, "same inputs must give same id"
  assert a != c, "different source must give different id"
  assert a != d, "different headline must give different id"
  assert a != e, "different published date must give different id"
  assert len(a) == 16, f"id length must be 16, got {len(a)}"
  print("PASS: event_id is deterministic and discriminating")


def test_store_event_and_seen():
  init_schema()
  _clean_test_events()
  eid = store_event(
    source="test:phase1a", ticker="AAPL",
    headline="Apple reports record earnings beat",
    body="EPS $2.05 vs $1.91 consensus", url="http://x.com/a",
    published_at="2026-05-10T16:00:00",
    materiality="high", category="earnings",
    affected_tickers=["AAPL"], primary_ticker="AAPL",
    directional_signal="bullish", urgency="immediate",
    classifier_reason="EPS beat"
  )
  assert eid is not None and len(eid) == 16
  assert seen("test:phase1a", "Apple reports record earnings beat", "2026-05-10T16:00:00")
  assert not seen("test:phase1a", "Different headline", "2026-05-10T16:00:00")
  print(f"PASS: store + seen round-trip works (eid={eid})")


def test_dedup_via_insert_or_ignore():
  init_schema()
  _clean_test_events()
  args = dict(
    source="test:phase1a", ticker="AAPL",
    headline="duplicate test headline",
    body="body", url="http://x.com",
    published_at="2026-05-10T16:00:00",
    materiality="medium", category="other",
    affected_tickers=["AAPL"], primary_ticker="AAPL",
    directional_signal="neutral", urgency="days",
    classifier_reason="r"
  )
  eid1 = store_event(**args)
  eid2 = store_event(**args)
  eid3 = store_event(**args)
  assert eid1 == eid2 == eid3, "same content must yield same id"
  conn = get_connection()
  try:
    count = conn.execute(
      "SELECT COUNT(*) c FROM events WHERE source='test:phase1a' AND headline='duplicate test headline'"
    ).fetchone()['c']
    assert count == 1, f"expected 1 row after 3 inserts, got {count}"
  finally:
    conn.close()
  print("PASS: INSERT OR IGNORE dedupes correctly across 3 calls")


def test_different_sources_same_content_dedup_per_source():
  """Same headline from 3 different sources = 3 events (different source = different id)."""
  init_schema()
  _clean_test_events()
  for src in ['test:wire1', 'test:wire2', 'test:wire3']:
    store_event(
      source=src, ticker="AAPL", headline="Apple announces buyback",
      body="b", url="u", published_at="2026-05-10T16:00:00",
      materiality="high", category="other",
      affected_tickers=["AAPL"], primary_ticker="AAPL",
      directional_signal="bullish", urgency="hours",
      classifier_reason="buyback"
    )
  conn = get_connection()
  try:
    count = conn.execute(
      "SELECT COUNT(*) c FROM events WHERE headline='Apple announces buyback' AND source LIKE 'test:wire%'"
    ).fetchone()['c']
    assert count == 3, f"expected 3 rows (one per source), got {count}"
  finally:
    conn.close()
  print("PASS: source is part of dedup key (same content from N sources = N rows)")


def test_unprocessed_filters_by_materiality():
  init_schema()
  _clean_test_events()
  # 3 noise, 2 low, 2 medium, 2 high
  for i, mat in enumerate(['noise', 'noise', 'noise', 'low', 'low',
                            'medium', 'medium', 'high', 'high']):
    store_event(
      source=f"test:phase1a:f{i}", ticker="T",
      headline=f"h{i}", body="b", url="u",
      published_at=f"2026-05-10T10:00:{i:02d}",
      materiality=mat, category="other",
      affected_tickers=["T"], primary_ticker="T",
      directional_signal="neutral", urgency="watch",
      classifier_reason="r"
    )
  med_plus = [e for e in unprocessed_events(min_materiality='medium', limit=100)
              if e['source'].startswith('test:phase1a:')]
  high_only = [e for e in unprocessed_events(min_materiality='high', limit=100)
               if e['source'].startswith('test:phase1a:')]
  assert len(med_plus) == 4, f"expected 4 (2 med + 2 high), got {len(med_plus)}"
  assert len(high_only) == 2, f"expected 2 high, got {len(high_only)}"
  print(f"PASS: materiality filter correct (med+={len(med_plus)}, high={len(high_only)})")


def test_mark_processed():
  init_schema()
  _clean_test_events()
  eid = store_event(
    source="test:phase1a:proc", ticker="X", headline="h", body="b", url="u",
    published_at="2026-05-10T11:00:00",
    materiality="high", category="other",
    affected_tickers=["X"], primary_ticker="X",
    directional_signal="bullish", urgency="hours",
    classifier_reason="r"
  )
  before = [e for e in unprocessed_events(min_materiality='medium') if e['event_id'] == eid]
  assert len(before) == 1, "event should be unprocessed initially"
  mark_processed([eid])
  after = [e for e in unprocessed_events(min_materiality='medium') if e['event_id'] == eid]
  assert len(after) == 0, "event should not appear in unprocessed after marking"
  print("PASS: mark_processed removes events from unprocessed query")


def test_recent_events_for_ticker_matches_primary_and_affected():
  init_schema()
  _clean_test_events()
  # Direct primary_ticker match
  store_event(source="test:phase1a:r1", ticker="NVDA",
              headline="NVDA news", body="b", url="u",
              published_at=datetime.now().isoformat(),
              materiality="high", category="earnings",
              affected_tickers=["NVDA"], primary_ticker="NVDA",
              directional_signal="bullish", urgency="immediate",
              classifier_reason="r")
  # Affected list match (primary is something else)
  store_event(source="test:phase1a:r2", ticker="",
              headline="AMD news mentions NVDA",
              body="b", url="u",
              published_at=datetime.now().isoformat(),
              materiality="medium", category="sector",
              affected_tickers=["AMD", "NVDA"], primary_ticker="AMD",
              directional_signal="mixed", urgency="days",
              classifier_reason="r")
  hits = [e for e in recent_events_for_ticker("NVDA", hours=2)
          if e['source'].startswith('test:phase1a:r')]
  assert len(hits) >= 2, f"expected both NVDA events, got {len(hits)}"
  print(f"PASS: recent_events_for_ticker finds primary and affected matches (n={len(hits)})")


def test_recent_events_respects_time_window():
  init_schema()
  _clean_test_events()
  # Recent
  store_event(source="test:phase1a:t_new", ticker="KO",
              headline="recent KO", body="b", url="u",
              published_at=datetime.now().isoformat(),
              materiality="high", category="other",
              affected_tickers=["KO"], primary_ticker="KO",
              directional_signal="neutral", urgency="days",
              classifier_reason="r")
  # Stale (insert event but then set ingested_at backdated)
  store_event(source="test:phase1a:t_old", ticker="KO",
              headline="old KO", body="b", url="u",
              published_at="2020-01-01T00:00:00",
              materiality="high", category="other",
              affected_tickers=["KO"], primary_ticker="KO",
              directional_signal="neutral", urgency="watch",
              classifier_reason="r")
  # Backdate ingested_at for the "old" event
  old_ingested = (datetime.now() - timedelta(days=10)).isoformat()
  conn = get_connection()
  try:
    conn.execute("UPDATE events SET ingested_at = ? WHERE source = 'test:phase1a:t_old'",
                 (old_ingested,))
    conn.commit()
  finally:
    conn.close()
  hits = recent_events_for_ticker("KO", hours=24)
  recent_sources = [e['source'] for e in hits if e['source'].startswith('test:phase1a:t_')]
  assert 'test:phase1a:t_new' in recent_sources
  assert 'test:phase1a:t_old' not in recent_sources, "stale event must be excluded"
  print("PASS: recent_events_for_ticker respects time window")


def test_body_truncation_does_not_crash():
  init_schema()
  _clean_test_events()
  long_body = "x" * 50_000  # well beyond 5000-char cap
  eid = store_event(source="test:phase1a:trunc", ticker="AAPL",
                    headline="long body test",
                    body=long_body, url="u",
                    published_at="2026-05-10T18:00:00",
                    materiality="high", category="other",
                    affected_tickers=["AAPL"], primary_ticker="AAPL",
                    directional_signal="neutral", urgency="watch",
                    classifier_reason="r")
  conn = get_connection()
  try:
    row = conn.execute("SELECT body FROM events WHERE event_id = ?", (eid,)).fetchone()
    assert len(row['body']) <= 5000, f"body should be truncated to <=5000, got {len(row['body'])}"
  finally:
    conn.close()
  print(f"PASS: long body truncated safely (50000 -> {len(row['body'])} chars)")


if __name__ == "__main__":
  test_event_id_is_deterministic()
  test_store_event_and_seen()
  test_dedup_via_insert_or_ignore()
  test_different_sources_same_content_dedup_per_source()
  test_unprocessed_filters_by_materiality()
  test_mark_processed()
  test_recent_events_for_ticker_matches_primary_and_affected()
  test_recent_events_respects_time_window()
  test_body_truncation_does_not_crash()
  _clean_test_events()
  print("\nAll Phase 1a events_store tests passed.")
