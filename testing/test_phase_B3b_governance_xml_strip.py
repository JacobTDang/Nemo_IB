"""Phase B3b: governance extractor handles SEC's XML declaration.

The B3 pilot found that `extract_governance_data` was returning
`success=False, board_members=[]` for MSFT. Root cause was an unhandled
combination on Python >=3.10:

  - `pd.read_html(StringIO(html))` defaults to the lxml flavor, which
    rejects unicode strings that begin with an `<?xml ... ?>` processing
    instruction (which SEC's HTML files all do).
  - The pre-existing fallback to `flavor='html5lib'` cannot execute
    because the latest released `html5lib` (1.1) does
    `from collections import Mapping`, which was removed in Python 3.10.

Fix in `tools/web_search_server/8K_and_DEF14A_utils.py:_extract_board_from_tables`:
strip the XML declaration before invoking pd.read_html so lxml accepts the
input, and fall back to the `bs4` flavor (works without html5lib) instead
of the broken `html5lib` flavor when lxml still fails.

This test is intentionally narrow: it pins the parsing-level fix without
making an HTTP call to EDGAR. A captured-fixture run is more honest than
hitting the network in CI.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# A minimal SEC-style HTML containing the XML processing instruction that
# the lxml string parser previously rejected.
_SEC_STYLE_HTML = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/></head>
<body>
<table>
<thead>
<tr><th>Name</th><th>Age</th><th>Independent</th><th>Director Since</th></tr>
</thead>
<tbody>
<tr><td>Alice Director</td><td>62</td><td>Yes</td><td>2015</td></tr>
<tr><td>Bob Chair</td><td>58</td><td>No</td><td>2010</td></tr>
<tr><td>Carol Member</td><td>55</td><td>Yes</td><td>2018</td></tr>
<tr><td>Dan Member</td><td>61</td><td>Yes</td><td>2020</td></tr>
</tbody>
</table>
</body>
</html>'''


def test_xml_declaration_strip_keeps_lxml_path_alive():
  """Direct sanity check that the helper's strip logic produces a string
  lxml can parse via pd.read_html."""
  import pandas as pd
  from io import StringIO
  # Pre-fix behavior: this raises on Py>=3.10
  raised = False
  try:
    pd.read_html(StringIO(_SEC_STYLE_HTML))
  except (ValueError, ImportError):
    raised = True
  assert raised, "test premise wrong: raw lxml string parse already works"
  # Post-fix behavior: stripping the declaration unblocks lxml
  clean = _SEC_STYLE_HTML
  if clean.lstrip().startswith('<?xml') and '?>' in clean:
    clean = clean.split('?>', 1)[1].lstrip()
  dfs = pd.read_html(StringIO(clean))
  assert dfs and len(dfs) == 1
  assert len(dfs[0]) == 4
  print("PASS: stripping XML declaration unblocks pd.read_html (lxml)")


def test_extract_board_from_tables_no_longer_silent_fails_on_xml_decl():
  """Before the fix, the extractor returned (members=[], df=None) and
  swallowed the ImportError silently because pd.read_html was inside a
  try block. Verify that XML-declared HTML now actually invokes pandas
  parsing (whether or not the synthetic table matches the parser's
  column-identification heuristics is a separate concern; what matters
  is that parsing isn't silently failing on the XML declaration anymore)."""
  import importlib
  mod = importlib.import_module('tools.web_search_server.8K_and_DEF14A_utils')
  parser = mod.SECFilingParser()
  # If parse layer was still broken, the inner pd.read_html call would
  # raise and the outer except in the method would also catch — so we
  # cannot directly assert non-empty results from a synthetic fixture.
  # Instead patch pd.read_html to capture the input it actually received.
  from unittest.mock import patch
  import pandas as pd
  captured = {"input": None}
  orig = pd.read_html
  def spy(*args, **kwargs):
    if args:
      buf = args[0]
      try:
        captured["input"] = buf.getvalue() if hasattr(buf, 'getvalue') else str(buf)
      except Exception:
        pass
    return orig(*args, **kwargs)
  with patch.object(mod.pd, 'read_html', side_effect=spy):
    parser._extract_board_from_tables(_SEC_STYLE_HTML, debug=False)
  assert captured["input"] is not None, "pd.read_html was never called — parse layer still broken"
  assert not captured["input"].lstrip().startswith('<?xml'), \
    f"XML declaration leaked into pd.read_html: {captured['input'][:80]!r}"
  print("PASS: XML declaration stripped before pd.read_html invocation")


def test_no_xml_declaration_passes_through_unchanged():
  """Regression guard: HTML without the XML declaration must pass to
  pd.read_html unmodified — the strip logic must only fire for declared
  inputs."""
  plain_html = _SEC_STYLE_HTML.split('?>', 1)[1].lstrip()
  import importlib
  mod = importlib.import_module('tools.web_search_server.8K_and_DEF14A_utils')
  parser = mod.SECFilingParser()
  from unittest.mock import patch
  captured = {"input": None}
  orig = mod.pd.read_html
  def spy(*args, **kwargs):
    if args:
      buf = args[0]
      try:
        captured["input"] = buf.getvalue() if hasattr(buf, 'getvalue') else str(buf)
      except Exception:
        pass
    return orig(*args, **kwargs)
  with patch.object(mod.pd, 'read_html', side_effect=spy):
    parser._extract_board_from_tables(plain_html, debug=False)
  assert captured["input"] == plain_html, "plain HTML was modified unexpectedly"
  print("PASS: plain HTML passes through unmodified")


if __name__ == "__main__":
  test_xml_declaration_strip_keeps_lxml_path_alive()
  test_extract_board_from_tables_no_longer_silent_fails_on_xml_decl()
  test_no_xml_declaration_passes_through_unchanged()
  print("\nAll Phase B3b governance XML-strip tests passed.")
