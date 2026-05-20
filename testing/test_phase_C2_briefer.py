"""Phase C2: assemble_brief_inputs reads SQLite into a brief-ready dict.

Seeds events + a position + a thesis, calls the helper, asserts shape.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.events_store import store_event
from state.positions import open_position
from state.theses import insert_thesis


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE ticker LIKE 'C2_%' OR primary_ticker LIKE 'C2_%'")
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'C2_%'")
    conn.execute("DELETE FROM theses WHERE ticker LIKE 'C2_%'")
    conn.commit()
  finally:
    conn.close()


def _seed():
  store_event(source='reuters', ticker='C2_AAPL',
              headline='C2_AAPL beats Q3 estimates',
              body='Apple posted record services revenue.',
              url='https://example.com/aapl', published_at='2026-05-19T20:00:00',
              materiality='high', category='earnings',
              affected_tickers=['C2_AAPL'], primary_ticker='C2_AAPL',
              directional_signal='bullish', urgency='hours',
              classifier_reason='positive earnings surprise')
  store_event(source='bloomberg', ticker='C2_NVDA',
              headline='C2_NVDA announces new GPU',
              body='Nvidia unveils next-gen Blackwell SKU.',
              url='https://example.com/nvda', published_at='2026-05-19T18:00:00',
              materiality='medium', category='product',
              affected_tickers=['C2_NVDA'], primary_ticker='C2_NVDA',
              directional_signal='bullish', urgency='days',
              classifier_reason='new product launch')
  open_position('C2_AAPL', 'long', 10, 195.0, paper=True)
  insert_thesis('C2_NVDA', recommendation='BUY', signal='long',
                target_price=950.0, stop_loss=750.0, confidence=0.78,
                analysis_summary='AI demand sustains high gross margin.',
                key_assumptions=['data center capex elevated through 2027'],
                data_gaps=[],
                full_report_md='## NVDA report\nlong thesis')


def test_assemble_brief_inputs_shape():
  init_schema(); _clean(); _seed()
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs(['C2_AAPL', 'C2_NVDA'], hours_back=72)
  # Top-level keys
  for k in ('date', 'watchlist', 'events_per_ticker', 'active_theses',
            'open_positions', 'macro_snapshot'):
    assert k in brief, f"missing key: {k}"
  assert brief['watchlist'] == ['C2_AAPL', 'C2_NVDA']
  print(f"PASS: shape includes all 6 top-level keys")
  _clean()


def test_events_per_ticker_populated():
  init_schema(); _clean(); _seed()
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs(['C2_AAPL', 'C2_NVDA'], hours_back=72)
  ept = brief['events_per_ticker']
  assert 'C2_AAPL' in ept and len(ept['C2_AAPL']) >= 1, "no AAPL events"
  assert 'C2_NVDA' in ept and len(ept['C2_NVDA']) >= 1, "no NVDA events"
  # Event rows should carry headline + materiality
  aapl_event = ept['C2_AAPL'][0]
  assert 'beats Q3' in aapl_event['headline']
  assert aapl_event['materiality'] == 'high'
  print(f"PASS: events surfaced per ticker ({len(ept['C2_AAPL'])} AAPL, {len(ept['C2_NVDA'])} NVDA)")
  _clean()


def test_open_positions_included():
  init_schema(); _clean(); _seed()
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs(['C2_AAPL', 'C2_NVDA'], hours_back=72)
  positions = brief['open_positions']
  tickers = [p['ticker'] for p in positions]
  assert 'C2_AAPL' in tickers, f"AAPL position missing; got {tickers}"
  print(f"PASS: open positions surfaced ({len(positions)} rows)")
  _clean()


def test_active_theses_per_ticker():
  init_schema(); _clean(); _seed()
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs(['C2_AAPL', 'C2_NVDA'], hours_back=72)
  theses = brief['active_theses']
  assert 'C2_NVDA' in theses, f"NVDA thesis missing; got {list(theses.keys())}"
  assert theses['C2_NVDA']['recommendation'] == 'BUY'
  # AAPL has no thesis -> key absent OR explicit None
  if 'C2_AAPL' in theses:
    assert theses['C2_AAPL'] is None
  print(f"PASS: theses keyed by ticker, None for no-thesis case")
  _clean()


def test_hours_back_window_respected():
  init_schema(); _clean()
  # Seed one old event (>72h ago) and one fresh
  store_event(source='reuters', ticker='C2_MSFT',
              headline='C2_MSFT old news',
              body='', url='https://example.com/o',
              published_at='2026-05-01T10:00:00',
              materiality='medium', category='other',
              affected_tickers=['C2_MSFT'], primary_ticker='C2_MSFT')
  # Hack: force ingested_at into the past for the old event
  conn = get_connection()
  try:
    conn.execute(
      "UPDATE events SET ingested_at = ? WHERE primary_ticker = 'C2_MSFT'",
      ('2026-05-01T10:00:00',)
    )
    conn.commit()
  finally:
    conn.close()
  store_event(source='bloomberg', ticker='C2_MSFT',
              headline='C2_MSFT fresh news',
              body='', url='https://example.com/f',
              published_at='2026-05-19T10:00:00',
              materiality='medium', category='other',
              affected_tickers=['C2_MSFT'], primary_ticker='C2_MSFT')
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs(['C2_MSFT'], hours_back=24)
  msft_events = brief['events_per_ticker'].get('C2_MSFT', [])
  headlines = [e['headline'] for e in msft_events]
  assert any('fresh' in h for h in headlines), f"fresh event missing: {headlines}"
  assert not any('old' in h for h in headlines), f"old event leaked: {headlines}"
  print(f"PASS: hours_back filter works (fresh kept, old dropped)")
  _clean()


def test_empty_watchlist_returns_empty_sections():
  init_schema(); _clean()
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs([], hours_back=24)
  assert brief['watchlist'] == []
  assert brief['events_per_ticker'] == {}
  assert brief['active_theses'] == {}
  print(f"PASS: empty watchlist returns empty per-ticker sections")


def test_date_is_iso_today():
  init_schema(); _clean()
  from datetime import datetime
  from agent.Pre_Market_Briefer import assemble_brief_inputs
  brief = assemble_brief_inputs(['C2_X'], hours_back=24)
  assert brief['date'] == datetime.now().date().isoformat()
  print(f"PASS: date = {brief['date']}")


if __name__ == "__main__":
  test_assemble_brief_inputs_shape()
  test_events_per_ticker_populated()
  test_open_positions_included()
  test_active_theses_per_ticker()
  test_hours_back_window_respected()
  test_empty_watchlist_returns_empty_sections()
  test_date_is_iso_today()
  print("\nAll Phase C2 briefer tests passed.")
