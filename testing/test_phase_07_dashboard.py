"""Phase 7: FastAPI dashboard.

Uses FastAPI's TestClient to hit endpoints without spinning up uvicorn.
Tests read-only invariant by inspecting code, then exercises each route.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from state.schema import init_schema, get_connection, add_to_watchlist
from state.theses import insert_thesis
from state.events_store import store_event
from state.positions import open_position
from dashboard.app import app


client = TestClient(app)


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'DASH_%'")
    conn.execute("DELETE FROM theses WHERE ticker LIKE 'DASH_%'")
    conn.execute("DELETE FROM events WHERE source LIKE 'dash-test:%'")
    conn.execute("DELETE FROM watchlist WHERE ticker LIKE 'DASH_%'")
    conn.commit()
  finally:
    conn.close()


def _seed():
  init_schema()
  _clean()
  add_to_watchlist("DASH_AAPL", priority=2)
  insert_thesis(
    ticker="DASH_AAPL", recommendation="BUY", signal="bullish",
    target_price=210.0, stop_loss=170.0, confidence=0.72,
    analysis_summary="DASH test thesis: services growth + buyback.",
    key_assumptions=["Services +14% YoY"], data_gaps=[],
    full_report_md="## RECOMMENDATION: BUY\n..."
  )
  store_event(
    source="dash-test:wsj", ticker="DASH_AAPL",
    headline="DASH AAPL beats Q2 estimates",
    body="dummy", url="http://x/1",
    published_at="2026-05-10T16:00:00",
    materiality="high", category="earnings",
    affected_tickers=["DASH_AAPL"], primary_ticker="DASH_AAPL",
    directional_signal="bullish", urgency="immediate",
    classifier_reason="EPS beat"
  )
  open_position("DASH_AAPL", "long", 10, 195.0, thesis_id=1)


def test_home_renders():
  _seed()
  r = client.get("/")
  assert r.status_code == 200
  body = r.text
  assert "Nemo IB" in body
  assert "DASH_AAPL" in body, "seeded ticker should appear in watchlist or theses"
  print(f"PASS: / renders ({len(body)} bytes)")


def test_health_returns_json():
  _seed()
  r = client.get("/health")
  assert r.status_code == 200
  j = r.json()
  assert j['status'] == 'ok'
  assert j['watchlist_size'] >= 1
  assert j['open_positions'] >= 1
  assert 'news_watcher_stale' in j
  print(f"PASS: /health returns {j}")


def test_ticker_detail_renders():
  _seed()
  r = client.get("/ticker/DASH_AAPL")
  assert r.status_code == 200
  body = r.text
  assert "DASH_AAPL" in body
  assert "BUY" in body
  assert "services growth" in body.lower()
  print("PASS: /ticker/DASH_AAPL renders with thesis")


def test_ticker_detail_missing_ticker_still_renders():
  _seed()
  r = client.get("/ticker/UNSEEN_TICKER")
  assert r.status_code == 200
  # Should render the page with "no active thesis" copy
  assert "no active thesis" in r.text.lower() or "UNSEEN_TICKER" in r.text
  print("PASS: /ticker/<unseen> renders without crashing")


def test_positions_page_renders():
  _seed()
  r = client.get("/positions")
  assert r.status_code == 200
  assert "DASH_AAPL" in r.text
  assert "Open Positions" in r.text
  print("PASS: /positions renders with the seeded long position")


def test_events_page_renders():
  _seed()
  r = client.get("/events")
  assert r.status_code == 200
  assert "DASH AAPL beats" in r.text
  assert "earnings" in r.text
  print("PASS: /events renders with seeded high-materiality event")


def test_partial_endpoints_return_html():
  _seed()
  for path in ("/_partials/positions", "/_partials/events", "/_partials/theses"):
    r = client.get(path)
    assert r.status_code == 200
    assert "<" in r.text and ">" in r.text  # HTML-ish
  print("PASS: all 3 HTMX partial endpoints return HTML")


def test_no_dashboard_route_is_a_post():
  """Read-only invariant: no POST/PUT/DELETE routes in dashboard.app."""
  unsafe = [r for r in app.routes
            if hasattr(r, 'methods') and (r.methods or set()) & {'POST', 'PUT', 'DELETE', 'PATCH'}]
  assert not unsafe, f"dashboard should be read-only, found unsafe routes: {[r.path for r in unsafe]}"
  print(f"PASS: dashboard is read-only (no POST/PUT/DELETE routes)")


def test_app_starts_via_lifespan():
  """The dashboard should boot via the modern FastAPI lifespan handler, not
  the deprecated `@app.on_event('startup')`. `with TestClient(app)` forces
  the lifespan context manager to execute on enter/exit."""
  with TestClient(app) as c:
    r = c.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j['status'] == 'ok'
  # Verify the deprecated decorator was NOT used (regression guard)
  import dashboard.app as app_mod
  assert getattr(app_mod, '_lifespan', None) is not None, \
    "expected _lifespan context manager on the module"
  # FastAPI internal: lifespan-aware apps store the context on the router
  assert app.router.lifespan_context is not None
  print(f"PASS: app boots via lifespan handler (health -> {j['status']})")


def test_static_css_served():
  r = client.get("/static/style.css")
  assert r.status_code == 200
  assert 'background' in r.text  # CSS loaded
  print("PASS: /static/style.css served")


if __name__ == "__main__":
  test_home_renders()
  test_health_returns_json()
  test_ticker_detail_renders()
  test_ticker_detail_missing_ticker_still_renders()
  test_positions_page_renders()
  test_events_page_renders()
  test_partial_endpoints_return_html()
  test_no_dashboard_route_is_a_post()
  test_app_starts_via_lifespan()
  test_static_css_served()
  _clean()
  print("\nAll Phase 7 dashboard tests passed.")
