"""Phase 9: hardening — pre-mortem, correlation, audit trail.

Pre-mortem uses real Groq (1 call). Correlation uses mocked yfinance.
Audit trail uses seeded DB.
"""
import sys, os
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from agent.Pre_Mortem_Agent import Pre_Mortem_Agent, PreMortemReport
from agent.correlation import (
  correlation_matrix, avg_correlation_to_basket, correlation_decision
)
from agent.Risk_Officer import Risk_Officer
from agent.Arbiter_Agent import ArbiterVerdict
from state.schema import init_schema, get_connection
from state.theses import insert_thesis
from state.positions import record_order
from state.events_store import store_event
from fastapi.testclient import TestClient
from dashboard.app import app


client = TestClient(app)


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'H9_%'")
    conn.execute("DELETE FROM theses WHERE ticker LIKE 'H9_%'")
    conn.execute("DELETE FROM orders WHERE ticker LIKE 'H9_%' OR client_order_id LIKE 'h9-%'")
    conn.execute("DELETE FROM events WHERE source LIKE 'h9-test:%'")
    conn.commit()
  finally:
    conn.close()


# ---- Pre-mortem -----------------------------------------------------------

def test_pre_mortem_produces_structured_output():
  pma = Pre_Mortem_Agent()
  report = pma.envision(
    ticker="AAPL", recommendation="BUY",
    analyst_report_md="Apple BUY: services growth 14% YoY, iPhone -2%, target $215.",
    variables={"services_growth_yoy": 0.14, "iphone_growth_yoy": -0.02,
               "pe_trailing": 31, "current_price": 195.0}
  )
  assert report is not None, "pre-mortem returned None (parse failure)"
  assert isinstance(report, PreMortemReport)
  print(f"  failure_modes: {len(report.failure_modes)}")
  print(f"  early_warnings: {len(report.early_warnings)}")
  print(f"  hedge_or_exit: {len(report.hedge_or_exit)}")
  print(f"  worst_case_loss: {report.worst_case_loss_pct}%")
  assert len(report.failure_modes) >= 3, \
    f"need at least 3 failure modes, got {len(report.failure_modes)}"
  assert len(report.early_warnings) >= 1
  assert report.worst_case_loss_pct > 0
  print(f"PASS: pre-mortem produced {len(report.failure_modes)} failure modes "
        f"with worst case {report.worst_case_loss_pct}%")


def test_pre_mortem_failure_modes_are_specific():
  pma = Pre_Mortem_Agent()
  report = pma.envision(
    ticker="NVDA", recommendation="BUY",
    analyst_report_md="NVDA BUY: hyperscaler AI capex tailwind, $1000 target.",
    variables={"hyperscaler_capex_growth_yoy": 0.30}
  )
  assert report
  # Each failure mode should be > 40 chars (not "things go bad")
  short = [m for m in report.failure_modes if len(m.strip()) < 40]
  assert not short, f"failure modes too short: {short}"
  print(f"PASS: all {len(report.failure_modes)} failure modes are substantive")


# ---- Correlation ----------------------------------------------------------

def _make_fake_returns(tickers, n=90, correlations=None):
  """Build a fake returns DataFrame with known correlation structure."""
  np.random.seed(42)
  base = np.random.randn(n)
  data = {}
  for t in tickers:
    corr = (correlations or {}).get(t, 0.0)
    noise = np.random.randn(n)
    data[t] = corr * base + np.sqrt(1 - corr**2) * noise
  return pd.DataFrame(data, index=pd.date_range('2025-05-10', periods=n, freq='B'))


def test_correlation_matrix_returns_dataframe():
  fake_returns = _make_fake_returns(['A', 'B', 'C'], correlations={'A': 0.9, 'B': 0.9, 'C': 0.1})
  with patch('agent.correlation._daily_returns_panel_cached', return_value=fake_returns):
    corr = correlation_matrix(['A', 'B', 'C'])
  assert corr is not None
  # A and B share base signal -> high correlation
  assert corr.loc['A', 'B'] > 0.7
  # C is independent
  assert corr.loc['A', 'C'] < 0.3
  print(f"PASS: correlation matrix computes (A-B={corr.loc['A','B']:.2f}, A-C={corr.loc['A','C']:.2f})")


def test_avg_correlation_to_basket():
  fake_returns = _make_fake_returns(['CAND', 'X', 'Y'], correlations={'CAND': 0.9, 'X': 0.9, 'Y': 0.9})
  with patch('agent.correlation._daily_returns_panel_cached', return_value=fake_returns):
    avg = avg_correlation_to_basket('CAND', ['X', 'Y'])
  assert avg is not None
  assert avg > 0.6
  print(f"PASS: avg correlation to basket = {avg:.2f}")


def test_correlation_decision_blocks_high_corr():
  fake_returns = _make_fake_returns(['NEW', 'OLD1', 'OLD2'],
                                     correlations={'NEW': 0.95, 'OLD1': 0.95, 'OLD2': 0.95})
  with patch('agent.correlation._daily_returns_panel_cached', return_value=fake_returns):
    d = correlation_decision('NEW', ['OLD1', 'OLD2'], threshold=0.7)
  assert not d['ok']
  assert d['avg_correlation'] >= 0.7
  print(f"PASS: high correlation blocks (avg={d['avg_correlation']})")


def test_correlation_decision_passes_low_corr():
  fake_returns = _make_fake_returns(['NEW', 'OLD1', 'OLD2'],
                                     correlations={'NEW': 0.1, 'OLD1': 0.95, 'OLD2': 0.95})
  with patch('agent.correlation._daily_returns_panel_cached', return_value=fake_returns):
    d = correlation_decision('NEW', ['OLD1', 'OLD2'], threshold=0.7)
  assert d['ok']
  print(f"PASS: low correlation passes (avg={d['avg_correlation']})")


def test_correlation_decision_skips_on_unavailable_data():
  with patch('agent.correlation._daily_returns_panel_cached', return_value=None):
    d = correlation_decision('NEW', ['X', 'Y'], threshold=0.7)
  assert d['ok'], "missing data should not block trades"
  assert 'correlation_unavailable' in d['reason']
  print("PASS: missing correlation data does not block trades")


def test_correlation_empty_basket_ok():
  d = correlation_decision('FIRST_TRADE', [], threshold=0.7)
  assert d['ok']
  print("PASS: empty basket -> no correlation problem")


def test_risk_officer_rejects_on_correlation():
  fake_returns = _make_fake_returns(['NEW', 'EXISTING'],
                                     correlations={'NEW': 0.95, 'EXISTING': 0.95})
  verdict = ArbiterVerdict(
    final_recommendation='BUY', confidence=0.75,
    bull_strength=0.78, bear_strength=0.45,
    decisive_factors=['ok'], acknowledged_risks=['ok'],
    conditions_to_change_mind=['ok'],
    position_sizing_guidance='normal', rationale='test'
  )
  portfolio = {'total_value': 100_000, 'starting_value': 100_000,
                'daily_pnl_pct': 0, 'positions_opened_today': 0}
  ro = Risk_Officer()
  with patch('agent.correlation._daily_returns_panel_cached', return_value=fake_returns):
    d = ro.evaluate(10, 200, verdict, portfolio,
                     proposed_ticker='NEW', open_basket=['EXISTING'])
  assert not d.approve, f"high correlation should reject; got {d}"
  print(f"PASS: Risk Officer rejects high-correlation trade ({d.reasons})")


# ---- Audit trail ----------------------------------------------------------

def test_audit_order_returns_chain():
  init_schema(); _clean()
  store_event(
    source="h9-test:wsj", ticker="H9_AAPL",
    headline="H9 AAPL earnings beat",
    body="dummy", url="http://x/1",
    published_at="2026-04-01T10:00:00",
    materiality="high", category="earnings",
    affected_tickers=["H9_AAPL"], primary_ticker="H9_AAPL",
    directional_signal="bullish", urgency="immediate",
    classifier_reason="EPS beat"
  )
  thesis_id = insert_thesis(
    ticker="H9_AAPL", recommendation="BUY", signal="bullish",
    target_price=210.0, stop_loss=170.0, confidence=0.72,
    analysis_summary="audit test thesis",
    key_assumptions=["x"], data_gaps=[],
    full_report_md="r"
  )
  record_order(
    order_id="audit-test-order", client_order_id="h9-cli-1",
    ticker="H9_AAPL", side='buy', order_type='market', quantity=10,
    status='filled', thesis_id=thesis_id,
  )

  r = client.get("/audit/order/audit-test-order")
  assert r.status_code == 200
  j = r.json()
  assert j['order']['order_id'] == 'audit-test-order'
  assert j['thesis'] is not None
  assert j['thesis']['thesis_id'] == thesis_id
  assert j['chain_complete'] is True
  assert len(j['related_events']) >= 1
  assert any('H9 AAPL' in e['headline'] for e in j['related_events'])
  print(f"PASS: audit trail traced order -> thesis #{thesis_id} -> {len(j['related_events'])} events")


def test_audit_order_missing_returns_error():
  r = client.get("/audit/order/does-not-exist")
  assert r.status_code == 200
  j = r.json()
  assert 'error' in j
  print("PASS: audit /unknown returns clean error")


def test_audit_order_without_thesis_still_returns():
  """An order that wasn't linked to a thesis still has an audit row."""
  init_schema(); _clean()
  record_order(
    order_id="orphan-order", client_order_id="h9-orph",
    ticker="H9_X", side='buy', order_type='market', quantity=1,
    status='filled', thesis_id=None,
  )
  r = client.get("/audit/order/orphan-order")
  j = r.json()
  assert j['order']['order_id'] == 'orphan-order'
  assert j['thesis'] is None
  assert j['chain_complete'] is False
  print("PASS: orphan order returns order info with chain_complete=False")


if __name__ == "__main__":
  test_pre_mortem_produces_structured_output()
  test_pre_mortem_failure_modes_are_specific()
  test_correlation_matrix_returns_dataframe()
  test_avg_correlation_to_basket()
  test_correlation_decision_blocks_high_corr()
  test_correlation_decision_passes_low_corr()
  test_correlation_decision_skips_on_unavailable_data()
  test_correlation_empty_basket_ok()
  test_risk_officer_rejects_on_correlation()
  test_audit_order_returns_chain()
  test_audit_order_missing_returns_error()
  test_audit_order_without_thesis_still_returns()
  _clean()
  print("\nAll Phase 9 hardening tests passed.")
