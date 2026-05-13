"""FastAPI dashboard. READ-ONLY.

No dashboard endpoint writes to the DB. The trading system writes; the
dashboard observes. This separation makes the dashboard safe to expose
locally without privilege concerns.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root on path when launched as `uvicorn dashboard.app:app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state.schema import init_schema, get_watchlist, get_connection
from state.theses import active_theses, thesis_history, latest_thesis, get_thesis
from state.events_store import unprocessed_events, recent_events_for_ticker
from state.positions import open_positions, portfolio_stats, recent_orders


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
  # Startup: ensure schema exists. Teardown: nothing to do (read-only).
  init_schema()
  yield


app = FastAPI(title="Nemo IB", docs_url="/api/docs", redoc_url=None,
              lifespan=_lifespan)

# Static assets (CSS)
static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def _recent_events_df(limit: int = 30):
  """Pull recent material events for the home page."""
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT * FROM events WHERE materiality IN ('high','medium')
      ORDER BY ingested_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]
  finally:
    conn.close()


def _all_recent_events(limit: int = 100):
  conn = get_connection()
  try:
    rows = conn.execute("""
      SELECT * FROM events ORDER BY ingested_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]
  finally:
    conn.close()


# ---- Pages ----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
  ctx = {
    "request": request,
    "watchlist": get_watchlist(),
    "theses": active_theses(limit=30),
    "events": _recent_events_df(limit=20),
    "positions": open_positions(),
    "portfolio": portfolio_stats(),
  }
  return templates.TemplateResponse("home.html", ctx)


@app.get("/ticker/{ticker}", response_class=HTMLResponse)
async def ticker_detail(request: Request, ticker: str):
  ticker = ticker.upper()
  ctx = {
    "request": request,
    "ticker": ticker,
    "latest": latest_thesis(ticker),
    "history": thesis_history(ticker, limit=20),
    "events": recent_events_for_ticker(ticker, hours=24 * 14),  # 2 weeks
  }
  return templates.TemplateResponse("ticker.html", ctx)


@app.get("/positions", response_class=HTMLResponse)
async def positions_page(request: Request):
  ctx = {
    "request": request,
    "positions": open_positions(),
    "portfolio": portfolio_stats(),
    "recent_orders": recent_orders(limit=30),
  }
  return templates.TemplateResponse("positions.html", ctx)


@app.get("/events", response_class=HTMLResponse)
async def events_page(request: Request):
  ctx = {
    "request": request,
    "events": _all_recent_events(limit=100),
  }
  return templates.TemplateResponse("events.html", ctx)


@app.get("/health", response_class=JSONResponse)
async def health():
  """Lightweight liveness + DB connectivity check + last-event freshness."""
  conn = get_connection()
  try:
    last_ev = conn.execute(
      "SELECT MAX(ingested_at) AS m FROM events"
    ).fetchone()['m']
  finally:
    conn.close()
  last_ev_age_sec = None
  if last_ev:
    try:
      last_ev_age_sec = (datetime.now() - datetime.fromisoformat(last_ev)).total_seconds()
    except Exception:
      pass
  watchlist_size = len(get_watchlist())
  open_count = len(open_positions())
  return {
    "status": "ok",
    "watchlist_size": watchlist_size,
    "open_positions": open_count,
    "last_event_at": last_ev,
    "last_event_age_seconds": last_ev_age_sec,
    "news_watcher_stale": (
      last_ev_age_sec is not None and last_ev_age_sec > 600
    ),
  }


# ---- HTMX partials (auto-refresh fragments) -------------------------------

@app.get("/audit/order/{order_id}", response_class=JSONResponse)
async def audit_order(order_id: str):
  """Trace an order back through thesis -> events. Full causal chain."""
  conn = get_connection()
  try:
    order = conn.execute("SELECT * FROM orders WHERE order_id = ?",
                         (order_id,)).fetchone()
    if not order:
      return {'error': 'order_not_found', 'order_id': order_id}
    o = dict(order)
    thesis = None
    if o.get('thesis_id'):
      t = conn.execute("SELECT * FROM theses WHERE thesis_id = ?",
                       (o['thesis_id'],)).fetchone()
      thesis = dict(t) if t else None
    events = []
    if thesis:
      # Pull events for the ticker that arrived BEFORE this thesis was created
      events_rows = conn.execute("""
        SELECT * FROM events
        WHERE (primary_ticker = ? OR ticker = ?)
          AND ingested_at <= ?
        ORDER BY ingested_at DESC LIMIT 20
      """, (thesis['ticker'], thesis['ticker'], thesis['thesis_date'])).fetchall()
      events = [dict(r) for r in events_rows]
  finally:
    conn.close()
  return {
    'order': o,
    'thesis': thesis,
    'related_events': events,
    'chain_complete': bool(thesis),
  }


@app.get("/_partials/positions", response_class=HTMLResponse)
async def partial_positions(request: Request):
  ctx = {"request": request, "positions": open_positions(),
          "portfolio": portfolio_stats()}
  return templates.TemplateResponse("partials/positions_block.html", ctx)


@app.get("/_partials/events", response_class=HTMLResponse)
async def partial_events(request: Request):
  ctx = {"request": request, "events": _recent_events_df(limit=20)}
  return templates.TemplateResponse("partials/events_block.html", ctx)


@app.get("/_partials/theses", response_class=HTMLResponse)
async def partial_theses(request: Request):
  ctx = {"request": request, "theses": active_theses(limit=30)}
  return templates.TemplateResponse("partials/theses_block.html", ctx)
