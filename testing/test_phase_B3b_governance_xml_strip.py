"""Phase B3b: governance extractor handles SEC's XML declaration.

The B3 pilot found that `extract_governance_data` was returning
`success=False, board_members=[]` for MSFT. Root cause: SEC HTML files begin
with an `<?xml ... encoding="..."?>` processing instruction, and lxml's string
parser rejects those outright ("Unicode strings with encoding declaration are
not supported").

Fix in `tools/web_search_server/8K_and_DEF14A_utils.py`: strip the XML
declaration before invoking pd.read_html so lxml accepts the input.

An earlier version of this file asserted that a plain
`pd.read_html(StringIO(sec_html))` *raises*, to prove the strip was needed.
That premise was wrong -- and wrong in a way worth pinning, because it made
the workaround look dead. `pd.read_html`'s default `flavor` is the pair
("lxml", "bs4") tried in order, with ValueError caught between them. So the
default call succeeds on declared HTML by silently downgrading to bs4; it is
only `flavor="lxml"` that still raises. The strip is therefore live and
load-bearing: it is the difference between SEC tables being parsed by lxml
and being parsed by the weaker fallback parser.

These tests are intentionally narrow: they pin the parsing-level behaviour
without making an HTTP call to EDGAR.
"""
import os
import sys

import pytest

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


def test_lxml_flavor_still_rejects_the_xml_declaration():
  """The limitation the strip exists for is still real.

  If this ever starts passing without raising, lxml has changed and the strip
  can be reconsidered -- but not before.
  """
  import pandas as pd
  from io import StringIO
  with pytest.raises(ValueError, match="encoding declaration"):
    pd.read_html(StringIO(_SEC_STYLE_HTML), flavor='lxml')
  print("PASS: flavor='lxml' still rejects an encoding declaration")


def test_default_flavor_masks_the_failure_by_downgrading_to_bs4():
  """Why the old premise looked right and was wrong.

  The default flavor is ("lxml", "bs4"); pandas catches the lxml ValueError
  and retries with bs4, so the declared HTML parses -- via the fallback
  parser, not via lxml. That silent downgrade is exactly what the strip
  avoids, so this behaviour is pinned rather than relied upon.
  """
  import pandas as pd
  from io import StringIO
  dfs = pd.read_html(StringIO(_SEC_STYLE_HTML))
  assert len(dfs) == 1 and len(dfs[0]) == 4
  print("PASS: default flavor parses declared HTML by falling back to bs4")


def test_stripping_the_declaration_unblocks_lxml():
  """Post-fix behavior: after the strip, the lxml flavor itself succeeds."""
  import pandas as pd
  from io import StringIO
  clean = _SEC_STYLE_HTML
  if clean.lstrip().startswith('<?xml') and '?>' in clean:
    clean = clean.split('?>', 1)[1].lstrip()
  dfs = pd.read_html(StringIO(clean), flavor='lxml')
  assert dfs and len(dfs) == 1
  assert len(dfs[0]) == 4
  print("PASS: stripping the XML declaration unblocks the lxml flavor")


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
  test_lxml_flavor_still_rejects_the_xml_declaration()
  test_default_flavor_masks_the_failure_by_downgrading_to_bs4()
  test_stripping_the_declaration_unblocks_lxml()
  test_extract_board_from_tables_no_longer_silent_fails_on_xml_decl()
  test_no_xml_declaration_passes_through_unchanged()
  print("\nAll Phase B3b governance XML-strip tests passed.")
