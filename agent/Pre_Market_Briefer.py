"""Pre-market briefer: assemble the raw inputs Claude Code narrates.

Non-LLM helper. Pulls from local SQLite + the FRED MCP if reachable. The
narration step (turning these dicts into a Markdown brief) is the LLM's
job per the playbook in CLAUDE.md.
"""
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from state.events_store import recent_events_for_ticker
from state.positions import open_positions
from state.theses import latest_thesis


DISCORD_MESSAGE_LIMIT = 2000
SAFE_CHUNK_BUDGET = 1900  # leave headroom under Discord's 2000-char message cap


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


def build_brief_chunks(brief_md: str, max_chars: int = SAFE_CHUNK_BUDGET) -> List[str]:
  """Split a Markdown brief into <=max_chars Discord-message chunks.

  Splitting strategy:
    1. If the whole brief fits, return it as a single chunk.
    2. Split on `\n### ` (subsection) boundaries first.
    3. If a single subsection still exceeds the budget, fall back to
       paragraph (`\n\n`) splits within that subsection.
    4. If a single paragraph still exceeds the budget, hard-split at the
       budget boundary.

  Every chunk after the first starts with a `##` or `###` header so the
  reader can pick up cold in the next message.
  """
  if not brief_md:
    return []
  if len(brief_md) <= max_chars:
    return [brief_md]

  sections: List[str] = []
  current = ""
  for piece in _section_split(brief_md):
    candidate = (current + piece) if current else piece
    if len(candidate) <= max_chars:
      current = candidate
      continue
    if current:
      sections.append(current)
      current = ""
    if len(piece) <= max_chars:
      current = piece
    else:
      sections.extend(_paragraph_split(piece, max_chars))
      current = ""
  # If any chunk after the first doesn't start with a header (because its
  # source section overflowed and we paragraph-split it), prepend a
  # continuation header so readers don't pick up cold mid-paragraph.
  fixed: List[str] = []
  current_header = ""
  for i, sec in enumerate(sections):
    stripped = sec.lstrip()
    if stripped.startswith('## ') or stripped.startswith('### '):
      current_header = stripped.split('\n', 1)[0]
      fixed.append(sec)
    else:
      if i == 0 or not current_header:
        fixed.append(sec)
      else:
        fixed.append(f"{current_header} (continued)\n{sec}")
  sections = fixed
  if current:
    sections.append(current)
  return [s.rstrip() for s in sections if s.strip()]


def _section_split(md: str) -> List[str]:
  """Split on `\n### ` and `\n## ` boundaries, keeping the header with its body."""
  out: List[str] = []
  buf = ""
  for line in md.splitlines(keepends=True):
    if (line.startswith('### ') or line.startswith('## ')) and buf:
      out.append(buf)
      buf = line
    else:
      buf += line
  if buf:
    out.append(buf)
  return out


def _paragraph_split(piece: str, max_chars: int) -> List[str]:
  """Last-resort paragraph splitter for an oversized section."""
  out: List[str] = []
  current = ""
  for para in piece.split('\n\n'):
    candidate = (current + '\n\n' + para) if current else para
    if len(candidate) <= max_chars:
      current = candidate
      continue
    if current:
      out.append(current)
    if len(para) <= max_chars:
      current = para
    else:
      # Hard-split if even a single paragraph is too long
      for i in range(0, len(para), max_chars):
        out.append(para[i:i + max_chars])
      current = ""
  if current:
    out.append(current)
  return out


def send_brief(brief_md: str, webhook_url: Optional[str] = None) -> bool:
  """Deliver the brief to Discord, paginating into ~1900-char chunks.

  Returns True only if every chunk posts successfully. Missing webhook,
  empty brief, or import errors return False without raising.
  """
  if not brief_md or not brief_md.strip():
    print("[brief] empty brief; nothing to send", file=sys.stderr, flush=True)
    return False
  if not webhook_url:
    from alerts.discord import _webhook_url
    webhook_url = _webhook_url()
  if not webhook_url:
    print("[brief] DISCORD_WEBHOOK_URL not set; brief NOT delivered",
          file=sys.stderr, flush=True)
    return False
  try:
    from discord_webhook import DiscordWebhook
  except ImportError as e:
    print(f"[brief] discord_webhook not installed: {e}", file=sys.stderr, flush=True)
    return False

  chunks = build_brief_chunks(brief_md)
  if not chunks:
    return False

  all_ok = True
  for chunk in chunks:
    try:
      hook = DiscordWebhook(url=webhook_url, content=chunk, rate_limit_retry=True)
      resp = hook.execute()
      if isinstance(resp, list):
        ok = all(getattr(r, 'status_code', 500) < 300 for r in resp)
      else:
        ok = getattr(resp, 'status_code', 500) < 300
      if not ok:
        print(f"[brief] chunk send returned non-2xx: {resp}",
              file=sys.stderr, flush=True)
      all_ok = all_ok and ok
    except Exception as e:
      print(f"[brief] chunk send raised: {e}", file=sys.stderr, flush=True)
      all_ok = False
  return all_ok
