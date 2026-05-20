"""Phase B1: lint CLAUDE.md for the IB analyst playbook section.

Asserts:
- The top-level section header exists
- All 5 sub-sections exist
- References real MCP tool names that exist in the codebase
- The Risk_Officer rule is stated (no bypass)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLAUDE_MD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CLAUDE.md')

REQUIRED_TOP_HEADER = "## Investment Banking Agent Playbook (Claude Code Mode)"
REQUIRED_SUBSECTIONS = [
  "When to invoke",
  "Tool-use discipline",
  "Flow",
  "Trading discipline",
  "What NOT to do",
]

# A representative subset of REAL tool names — playbook should reference at
# least these so Claude Code knows what's callable.
REQUIRED_TOOL_NAMES = [
  "get_revenue_base",        # SEC
  "get_ebitda_margin",       # SEC
  "get_basic_financials",    # Finnhub
  "get_analyst_recommendations",  # Finnhub
  "get_insider_transactions",     # Finnhub
  "get_company_news",        # Finnhub
  "get_macro_snapshot",      # FRED
  "get_market_data",         # Financial
  "risk_check_proposed_trade",    # Alpaca — critical
  "place_paper_order",       # Alpaca — critical
]


def _read_md():
  with open(CLAUDE_MD, 'r', encoding='utf-8') as f:
    return f.read()


def _extract_section(md, header):
  """Return text between `header` and the next `## ` header (or EOF)."""
  start = md.find(header)
  if start < 0:
    return None
  rest = md[start + len(header):]
  m = re.search(r'\n## [^\n]', rest)
  end = m.start() if m else len(rest)
  return rest[:end]


def test_playbook_section_present():
  md = _read_md()
  assert REQUIRED_TOP_HEADER in md, f"missing top header: {REQUIRED_TOP_HEADER!r}"
  print(f"PASS: top section header present")


def test_all_subsections_present():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER)
  assert body, "could not extract playbook body"
  missing = [s for s in REQUIRED_SUBSECTIONS if s not in body]
  assert not missing, f"missing sub-sections: {missing}"
  print(f"PASS: all {len(REQUIRED_SUBSECTIONS)} sub-sections present")


def test_required_tool_names_referenced():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER)
  missing = [t for t in REQUIRED_TOOL_NAMES if t not in body]
  assert not missing, f"playbook does not reference these tools: {missing}"
  print(f"PASS: all {len(REQUIRED_TOOL_NAMES)} required tool names referenced")


def test_risk_officer_no_bypass_rule_stated():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER) or ""
  body_lower = body.lower()
  # Must mention risk_check_proposed_trade and the ordering constraint
  assert 'risk_check_proposed_trade' in body
  must_state_ordering = ('before' in body_lower and 'place_paper_order' in body)
  assert must_state_ordering, "playbook must say risk_check runs BEFORE place_paper_order"
  # Must say Risk_Officer is not to be argued with / bypassed
  assert any(kw in body_lower for kw in ('never bypass', 'no bypass', 'never argue')), \
    "playbook must explicitly forbid bypassing the Risk_Officer"
  print(f"PASS: Risk_Officer no-bypass rule stated")


def test_paper_only_constraint_present():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER) or ""
  assert 'paper' in body.lower(), "playbook should state paper-only constraint"
  print(f"PASS: paper-only constraint mentioned")


def test_no_date_invention_rule():
  md = _read_md()
  body = _extract_section(md, REQUIRED_TOP_HEADER) or ""
  body_lower = body.lower()
  # The Round 4 fix said: no invented calendar dates in narrative
  rule_words = ('date' in body_lower and ('invent' in body_lower or 'fabricat' in body_lower))
  assert rule_words, "playbook must forbid inventing dates/catalysts"
  print(f"PASS: no-date-invention rule stated")


if __name__ == "__main__":
  test_playbook_section_present()
  test_all_subsections_present()
  test_required_tool_names_referenced()
  test_risk_officer_no_bypass_rule_stated()
  test_paper_only_constraint_present()
  test_no_date_invention_rule()
  print("\nAll Phase B1 playbook lint tests passed.")
