"""Pre-market briefer: assemble the raw inputs Claude Code narrates.

Non-LLM helper. Pulls from local SQLite + the FRED MCP if reachable. The
narration step (turning these dicts into a Markdown brief) is the LLM's
job per the playbook in CLAUDE.md.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from state.events_store import recent_events_for_ticker
from state.positions import open_positions
from state.theses import latest_thesis


def assemble_brief_inputs(
  watchlist: List[str],
  hours_back: int = 24,
  paper: bool = True,
  macro_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Read-only snapshot of state for the daily brief.

  Args:
    watchlist: tickers to summarize (case preserved in the output but
      events_store handles case via affected_tickers JSON match).
    hours_back: look-back window for events_per_ticker.
    paper: filter positions by paper/live (default paper).
    macro_snapshot: optional pre-fetched FRED snapshot (caller-supplied).
      Briefer never makes its own MCP calls — keep it pure-Python.

  Returns:
    Dict with keys: date, watchlist, events_per_ticker, active_theses,
    open_positions, macro_snapshot.
  """
  events_per_ticker: Dict[str, List[Dict[str, Any]]] = {}
  theses_per_ticker: Dict[str, Optional[Dict[str, Any]]] = {}
  for ticker in watchlist:
    events_per_ticker[ticker] = recent_events_for_ticker(ticker, hours=hours_back)
    theses_per_ticker[ticker] = latest_thesis(ticker)
  return {
    'date': datetime.now().date().isoformat(),
    'watchlist': list(watchlist),
    'events_per_ticker': events_per_ticker,
    'active_theses': theses_per_ticker,
    'open_positions': open_positions(paper=paper),
    'macro_snapshot': macro_snapshot or {},
  }
