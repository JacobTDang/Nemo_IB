"""Phase C1: lint the Daily Pre-Market Brief playbook in CLAUDE.md.

C1 specifies the playbook only; the executable helper lands in C2 and the
Discord delivery in C3. This test guards the playbook is present and
references real state/alerts module entrypoints so Claude Code knows what
functions exist to drive a brief.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUDE_MD = os.path.join(ROOT, 'CLAUDE.md')

REQUIRED_TOP_HEADER = "## Daily Pre-Market Brief Playbook"
REQUIRED_SUBSECTIONS = [
  "Goal",
  "Inputs",
  "Steps",
  "Output",
  "Hard rules",
]

# Module-level entrypoints the briefer must use.
REQUIRED_API_REFERENCES = [
  "events_store.unprocessed_events",
  "events_store.recent_events_for_ticker",
  "positions.open_positions",
  "theses.active_theses",
  "Thesis_Maintainer",
  "discord.send_thesis_alert",  # or similar discord delivery function
]


def _read_md():
  with open(CLAUDE_MD, 'r', encoding='utf-8') as f:
    return f.read()


def _extract_section(md, header):
  start = md.find(header)
  if start < 0:
    return None
  rest = md[start + len(header):]
  m = re.search(r'\n## [^\n]', rest)
  end = m.start() if m else len(rest)
  return rest[:end]


def test_brief_section_present():
  md = _read_md()
  assert REQUIRED_TOP_HEADER in md, f"missing top header: {REQUIRED_TOP_HEADER!r}"
  print("PASS: brief section header present")


def test_subsections_present():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER) or ""
  missing = [s for s in REQUIRED_SUBSECTIONS if s not in body]
  assert not missing, f"missing sub-sections: {missing}"
  print(f"PASS: all {len(REQUIRED_SUBSECTIONS)} sub-sections present")


def test_api_references_present():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER) or ""
  missing = [a for a in REQUIRED_API_REFERENCES if a not in body]
  assert not missing, f"missing api references: {missing}"
  print(f"PASS: all {len(REQUIRED_API_REFERENCES)} api references present")


def test_no_trade_placement_in_brief():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER) or ""
  body_lower = body.lower()
  # Hard rule: brief mode never places trades
  forbids_trades = any(p in body_lower for p in (
    'do not place', 'never place', 'no trading', 'no orders',
    'never trade', 'never calls `place_paper_order',
    'never call place_paper_order',
  ))
  assert forbids_trades, "playbook must explicitly forbid placing trades during brief"
  # If place_paper_order is named at all, it must be inside a sentence that
  # also contains a negation word (never / not / no / don't / forbid / etc.)
  for line in body.splitlines():
    if 'place_paper_order' in line:
      low = line.lower()
      has_negation = any(neg in low for neg in (
        'never', 'not ', 'no ', "don't", 'forbid', 'do not', 'must not'
      ))
      assert has_negation, f"place_paper_order mentioned without negation: {line.strip()!r}"
  print("PASS: brief explicitly forbids placing trades")


def test_required_modules_importable():
  for spec in REQUIRED_API_REFERENCES:
    if '.' in spec:
      mod_part, _attr = spec.rsplit('.', 1)
      try:
        # Try both 'state.events_store' and 'discord' style references
        try:
          mod = __import__(f'state.{mod_part}', fromlist=[''])
        except ImportError:
          mod = __import__(f'alerts.{mod_part}', fromlist=[''])
      except Exception as e:
        raise AssertionError(f"could not import for {spec!r}: {type(e).__name__}: {e}")
    else:
      try:
        mod = __import__(f'agent.{spec}', fromlist=[''])
      except Exception as e:
        raise AssertionError(f"could not import {spec!r}: {type(e).__name__}: {e}")
  print(f"PASS: all referenced modules are importable")


if __name__ == "__main__":
  test_brief_section_present()
  test_subsections_present()
  test_api_references_present()
  test_no_trade_placement_in_brief()
  test_required_modules_importable()
  print("\nAll Phase C1 brief playbook lint tests passed.")
