"""Company-issued guidance statements out of 8-K earnings releases.

Every string in this file is verbatim from a real filing. That matters more
here than anywhere else in the suite, because the whole question this tool had
to answer first was whether guidance is extractable at all, and the answer
turned on what the filings actually look like once edgartools has flattened
them.

Two things it looks like:

  * Prose. "For the third quarter of 2026, AMD expects revenue to be
    approximately $13 billion, plus or minus $300 million." Clean, and the
    number means what it says.
  * A flattened table. Salesforce's EPS guidance renders as
    "GAAP diluted net income per share range(1)(2) Fiscal 2027 $1.74 - $7.93 -
    Plus Q2 $1.76 FY27 $7.99". The real table is two columns -- Q2 at
    $1.74-$1.76 and FY27 at $7.93-$7.99 -- and the flattening interleaves
    them. "$1.74 - $7.93" is a range no one guided.

So the tool returns the sentence as filed and declines the table. It makes no
beat/miss claim: see test_the_tool_does_not_claim_a_beat_or_a_miss.
"""
import os

import pytest

from tools.web_search_server import guidance as g

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Apply the real `network` marker plus the offline skip."""
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


@pytest.fixture(scope="module", autouse=True)
def _load_env():
  from dotenv import load_dotenv
  load_dotenv()


def _texts(result):
  return [s["text"] for s in result["statements"]]


# ------------------------------------------------------------ prose is kept

def test_a_plain_guidance_sentence_is_kept_with_its_own_period():
  """AMD, 8-K of 2026-08-04. The easy case, and it must stay easy."""
  text = ("For the third quarter of 2026, AMD expects revenue to be "
          "approximately $13 billion, plus or minus $300 million.")
  found = g._scan_guidance(text)
  assert len(found["statements"]) == 1
  statement = found["statements"][0]
  assert statement["text"] == text
  assert "third quarter of 2026" in statement["period_label"]
  assert statement["period_source"] == "in_statement"
  assert statement["caveats"] == []


def test_a_bullet_inherits_the_period_from_its_outlook_lead_in():
  """NVIDIA's outlook bullets carry no period of their own.

  The filing says "NVIDIA's outlook for the second quarter of fiscal 2027 is
  as follows:" and then bullets "Revenue is expected to be $91.0 billion".
  Read alone, that bullet is a $91bn guide for an unnamed period; the same
  filing also mentions fiscal 2027 as a whole, and attaching the quarterly
  number to the year overstates it roughly fourfold.
  """
  text = ("NVIDIA's outlook for the second quarter of fiscal 2027 is as "
          "follows: • Revenue is expected to be $91.0 billion, plus or "
          "minus 2%.")
  found = g._scan_guidance(text)
  statements = [s for s in found["statements"] if "$91.0 billion" in s["text"]]
  assert statements, f"lost the revenue bullet: {found}"
  statement = statements[0]
  assert "second quarter of fiscal 2027" in statement["period_label"]
  assert statement["period_source"] == "section_lead_in"
  assert "period_inherited_from_section_lead_in" in statement["caveats"]


def test_a_statement_with_no_period_anywhere_reports_none_not_a_guess():
  """Deere, 8-K of 2026-08-20: "Net income guidance improved to $4.75 billion
  to $5.00 billion." A null period is a smaller error than the wrong one."""
  found = g._scan_guidance(
      "Net income guidance improved to $4.75 billion to $5.00 billion.")
  assert len(found["statements"]) == 1
  assert found["statements"][0]["period_label"] is None
  assert found["statements"][0]["period_source"] is None


# --------------------------------------------------- flattened tables are not

def test_the_salesforce_eps_table_is_refused_rather_than_parsed():
  """The interleaved-column case, verbatim.

  A range of "$1.74 - $7.93" is the GAAP quarterly low bolted to the
  full-year high. Nothing in the flattened text distinguishes it from a real
  range, so the only safe answer is to decline the region.
  """
  text = ("The following is a per share reconciliation of GAAP diluted net "
          "income per share to non-GAAP diluted net income per share guidance "
          "for the next quarter and the full year: GAAP diluted net income per "
          "share range(1)(2) Fiscal 2027 $1.74 - $7.93 - Plus Q2 $1.76 FY27 "
          "$7.99")
  found = g._scan_guidance(text)
  assert found["statements"] == []
  assert found["rejected"].get("table_layout")


def test_two_amounts_with_only_whitespace_between_them_are_a_table_row():
  """Intel, 8-K of 2026-07-23: "Gross margin 41.0% 42.0%".

  Those are the GAAP and non-GAAP columns, not a 41-to-42 range. Prose always
  puts a word between two numbers -- "74.9% and 75.0%" -- so the absence of
  one is the signature.
  """
  found = g._scan_guidance(
      "Intel's guidance for the third quarter of 2026: Revenue $15.8-16.8 "
      "billion Gross margin 41.0% 42.0% Tax Rate 1% 11%")
  assert found["statements"] == []
  assert found["rejected"].get("table_layout")


def test_a_current_versus_prior_column_pair_is_a_table_row():
  """Coca-Cola, 8-K of 2026-07-28, verbatim.

  "Approx. 5% growth" is the current guide and "4% to 5% growth" is the one it
  replaced. Read as prose, the superseded range wins because it looks like a
  range.
  """
  found = g._scan_guidance(
      "Full Year 2026 Guidance Current Prior Organic revenues Approx. 5% "
      "growth 4% to 5% growth (non-GAAP)")
  assert found["statements"] == []
  assert found["rejected"].get("table_layout")


def test_a_reconciliation_caption_marks_a_rendered_table():
  """Target, 8-K of 2026-08-19, verbatim.

  This one cleared every structural guard -- no box rule, no adjacent
  amounts, no column-header pair -- and its numbers happen to be right. It is
  still a table, and the next filer's version of it will not be.
  """
  found = g._scan_guidance(
      "Reconciliation of Non-GAAP Adjusted EPS Guidance (per share) "
      "(unaudited) Full Year 2026 GAAP diluted earnings per share guidance "
      "$9.90 - $10.90 Estimated adjustments Other Adjusted diluted earnings "
      "per share $9.90 - $10.90 guidance")
  assert found["statements"] == []
  assert found["rejected"].get("table_layout")


def test_box_drawing_rules_mark_a_rendered_table():
  found = g._scan_guidance(
      "Outlook Revenue is expected to be $10.0 billion "
      "──────── GAAP $ 6,509 $ 3,542")
  assert found["statements"] == []


def test_a_truncated_cell_ellipsis_marks_a_rendered_table():
  """edgartools abbreviates overflowing cells with a horizontal ellipsis.

  NVIDIA's summary table renders as "R… $8… $6… $4… 20 % 85 %".
  """
  found = g._scan_guidance(
      "Q1 Fiscal 2027 Summary G… ($ in m… expects R… $8… $6… 20 % 85 %")
  assert found["statements"] == []


# --------------------------------------------------- not-guidance is not kept

def test_a_backward_reference_to_an_old_guide_is_not_a_new_guide():
  """Microsoft, 8-K of 2026-07-29, verbatim -- and the only numeric hit in the
  whole exhibit. Microsoft guides on the call, not in the release, so a tool
  that took this would report Microsoft's guidance as $0.27."""
  found = g._scan_guidance(
      "This compares with the guidance provided on April 29, 2026, resulting "
      "in a benefit of $0.27 on diluted earnings per share.")
  assert found["statements"] == []
  assert found["rejected"].get("backward_reference")


def test_a_reported_result_is_not_a_forecast():
  """Target, 8-K of 2026-08-19, verbatim.

  Every amount here sits in a past-tense clause. The sentence is in a section
  that also discusses guidance, which is how it got picked up at all.
  """
  found = g._scan_guidance(
      "The Company paid dividends of $518 million in the second quarter, "
      "compared with $509 million last year, reflecting a 1.8 percent "
      "increase in the dividend per share. Full year guidance follows.")
  assert not any("paid dividends" in t for t in _texts(found))


def test_a_chief_executive_quotation_is_not_guidance():
  """Eli Lilly, 8-K of 2026-08-05, verbatim.

  "raised our full-year guidance" plus "48% revenue growth" trips every cue,
  and the 48% is a reported result. The real guide is restated in the outlook
  section, so dropping quotations costs nothing.
  """
  found = g._scan_guidance(
      '"Lilly’s momentum continues, as we delivered 48% revenue growth '
      'and raised our full-year guidance," said David A. Ricks.')
  assert found["statements"] == []
  assert found["rejected"].get("quotation")


def test_safe_harbour_boilerplate_is_not_guidance():
  found = g._scan_guidance(
      "NVIDIA's financial and business outlook for the second quarter of "
      "fiscal 2027 and beyond; projected market growth of 20% and trends are "
      "forward-looking statements within the meaning of Section 27A of the "
      "Securities Act of 1933.")
  assert found["statements"] == []
  assert found["rejected"].get("boilerplate")


def test_a_headline_run_into_its_dateline_is_split_before_scanning():
  """Home Depot, 8-K of 2026-08-18, verbatim.

  The exhibit opens with the headline, the dateline and the first result
  sentence as one unbroken run. "Reaffirms Fiscal 2026 Guidance" is a real
  cue and "$47.9 billion" is a reported quarterly sales figure; left joined,
  the actual becomes the guide.
  """
  text = ("Exhibit 99.1 The Home Depot Announces Second Quarter Fiscal 2026 "
          "Results; Reaffirms Fiscal 2026 Guidance ATLANTA, August 18, 2026 "
          "-- The Home Depot, the world's largest home improvement retailer, "
          "today reported sales of $47.9 billion for the second quarter.")
  found = g._scan_guidance(text)
  assert not any("47.9" in t for t in _texts(found)), (
      f"a reported sales figure was returned as guidance: {_texts(found)}")


def test_a_swallowed_section_heading_is_not_a_statement():
  """Salesforce, 8-K of 2026-05-27: a bullet about Q1 ARR that ran into the
  "Guidance" heading immediately after it."""
  found = g._scan_guidance(
      "Public Sector Industry Cloud ARR surpasses $2 billion, up 23% Y/Y in "
      "Q1, with Public Sector AWUs up nearly 400% Q/Q Guidance")
  assert found["statements"] == []


# ------------------------------------------------ nothing found, and why not

def _stub_releases(monkeypatch, payload):
  monkeypatch.setattr(g, "get_earnings_releases", lambda *a, **k: payload)


def test_no_guidance_language_is_distinct_from_no_source():
  """The distinction the whole tool turns on.

  Apple files an earnings exhibit and puts no guidance in it. Procter &
  Gamble files one whose text edgartools could not extract. Both yield zero
  statements and they are not the same finding: the first is evidence about
  Apple, the second is evidence about nothing.
  """
  apple = {"ticker": "AAPL", "success": True, "error": None, "releases": [
      {"filing_date": "2026-07-30", "accession_number": "a1", "text":
       "Apple today announced financial results. Revenue was $94 billion.",
       "filing_url": "u1", "attachment_doc": "ex99.htm"}]}
  monkey = pytest.MonkeyPatch()
  monkey.setattr(g, "get_earnings_releases", lambda *a, **k: apple)
  result = g.extract_guidance("AAPL")
  monkey.undo()
  assert result["success"] is True
  assert result["guidance_found"] is False
  assert result["no_guidance_reason"] == "no_guidance_language_found"

  pg = {"ticker": "PG", "success": True, "error": None, "releases": [
      {"filing_date": "2026-07-29", "accession_number": "a2", "text": None,
       "filing_url": "u2", "attachment_doc": None}]}
  monkey = pytest.MonkeyPatch()
  monkey.setattr(g, "get_earnings_releases", lambda *a, **k: pg)
  result = g.extract_guidance("PG")
  monkey.undo()
  assert result["guidance_found"] is False
  assert result["no_guidance_reason"] == "release_text_unavailable"


def test_no_releases_at_all_is_its_own_reason(monkeypatch):
  """Caterpillar returned no 8-K Item 2.02 in the window. That says nothing
  about whether Caterpillar guides."""
  _stub_releases(monkeypatch, {"ticker": "CAT", "success": False,
                               "error": "No 8-K Item 2.02 filings found",
                               "releases": []})
  result = g.extract_guidance("CAT")
  assert result["guidance_found"] is False
  assert result["no_guidance_reason"] == "no_earnings_releases_found"
  assert "No 8-K Item 2.02" in result["error"]


def test_a_wall_of_rejected_tables_with_no_prose_says_so(monkeypatch):
  """Coca-Cola guides, in a table, and this tool will not read tables.

  Reporting a bare "no guidance found" for a company that guides every
  quarter is the wrong answer. The count of refused table regions is the
  evidence that separates "does not guide" from "guides where I cannot look".
  """
  _stub_releases(monkeypatch, {"ticker": "KO", "success": True, "error": None,
                               "releases": [{
                                   "filing_date": "2026-07-28",
                                   "accession_number": "a3",
                                   "filing_url": "u3",
                                   "attachment_doc": "ex99.htm",
                                   "text": " ".join([
                                       "Full Year 2026 Guidance Current Prior "
                                       "Organic revenues Approx. 5% growth 4% "
                                       "to 5% growth (non-GAAP)"] * 12)}]})
  result = g.extract_guidance("KO")
  assert result["guidance_found"] is False
  assert result["guidance_may_be_table_only"] is True


def test_a_release_with_no_tables_and_no_guidance_is_not_flagged_table_only(monkeypatch):
  _stub_releases(monkeypatch, {"ticker": "AAPL", "success": True, "error": None,
                               "releases": [{
                                   "filing_date": "2026-07-30",
                                   "accession_number": "a4",
                                   "filing_url": "u4",
                                   "attachment_doc": "ex99.htm",
                                   "text": "Apple today announced results for "
                                           "its fiscal 2026 third quarter."}]})
  result = g.extract_guidance("AAPL")
  assert result["guidance_found"] is False
  assert result["guidance_may_be_table_only"] is False


# ------------------------------------------------------------- the contract

def test_the_failure_path_keeps_every_documented_key(monkeypatch):
  """extract_customer_concentration shipped an error path that dropped
  has_concentration, so a caller reading the documented field got a KeyError
  instead of an answer. Failure must not change the shape."""
  def explode(*a, **k):
    raise RuntimeError("EDGAR unreachable")
  monkeypatch.setattr(g, "get_earnings_releases", explode)

  result = g.extract_guidance("BOOM")
  assert result["success"] is False
  assert "EDGAR unreachable" in result["error"]
  for key in ("guidance_found", "statements", "statement_count", "sources",
              "no_guidance_reason", "guidance_may_be_table_only",
              "limitations"):
    assert key in result, f"failure path dropped {key!r}"
  assert result["guidance_found"] is False
  assert result["statements"] == []


def test_every_statement_carries_the_filing_it_came_from(monkeypatch):
  """A guidance sentence with no provenance cannot be checked, and checking it
  is the only thing the caller can do that the tool will not."""
  _stub_releases(monkeypatch, {"ticker": "AMD", "success": True, "error": None,
                               "releases": [{
                                   "filing_date": "2026-08-04",
                                   "accession_number": "0000002488-26-000123",
                                   "filing_url": "https://sec.gov/x.htm",
                                   "attachment_doc": "q22026991.htm",
                                   "text": "For the third quarter of 2026, AMD "
                                           "expects revenue to be approximately "
                                           "$13 billion, plus or minus $300 "
                                           "million."}]})
  result = g.extract_guidance("AMD")
  assert result["guidance_found"] is True
  statement = result["statements"][0]
  assert statement["filing_date"] == "2026-08-04"
  assert statement["accession"] == "0000002488-26-000123"
  assert statement["source_url"] == "https://sec.gov/x.htm"


def test_the_tool_does_not_claim_a_beat_or_a_miss(monkeypatch):
  """Deliberate, and the reason is in `limitations`.

  Grading a guide needs the actual on the same basis for the same fiscal
  period. Salesforce's GAAP and non-GAAP EPS differ by a factor of four, and
  the only actuals source here labels fiscal quarters with calendar-quarter
  ends -- off by up to 60 days for a fiscal-offset filer. Two fuzzy joins
  stacked on a regex is how a tool ends up reporting that management missed a
  number it never gave.
  """
  _stub_releases(monkeypatch, {"ticker": "AMD", "success": True, "error": None,
                               "releases": [{
                                   "filing_date": "2026-08-04",
                                   "accession_number": "a5",
                                   "filing_url": "u5",
                                   "attachment_doc": "d.htm",
                                   "text": "For the third quarter of 2026, AMD "
                                           "expects revenue to be approximately "
                                           "$13 billion."}]})
  result = g.extract_guidance("AMD")
  serialised = repr(result).lower()
  for forbidden in ("beat", "missed", "met_guidance", "verdict"):
    assert forbidden not in result, f"the tool must not publish a {forbidden}"
  assert "does not" in result["limitations"].lower()
  assert "beat" in serialised  # only inside the limitations note


def test_sources_are_listed_even_when_nothing_was_found(monkeypatch):
  """"I looked at these four filings and found nothing" is a finding.
  "I found nothing" is not."""
  _stub_releases(monkeypatch, {"ticker": "AAPL", "success": True, "error": None,
                               "releases": [
                                   {"filing_date": "2026-07-30", "accession_number": "a",
                                    "filing_url": "u", "attachment_doc": "d",
                                    "text": "No forward language here."},
                                   {"filing_date": "2026-04-30", "accession_number": "b",
                                    "filing_url": "v", "attachment_doc": "e",
                                    "text": "None here either."}]})
  result = g.extract_guidance("AAPL")
  assert len(result["sources"]) == 2
  assert result["sources"][0]["filing_date"] == "2026-07-30"
  assert result["sources"][0]["text_available"] is True


# ------------------------------------------------------------- live golden set

@network
def test_nvidia_guides_revenue_for_the_coming_quarter():
  """NVIDIA has opened its Outlook section the same way every quarter for
  years: a named quarter, then a revenue bullet. The figure changes; the
  shape does not, so this asserts the shape and the period, not the number."""
  result = g.extract_guidance("NVDA", quarters=1)

  assert result["success"] is True, result["error"]
  assert result["guidance_found"] is True
  revenue = [s for s in result["statements"]
             if "revenue" in s["text"].lower() and "$" in s["text"]]
  assert revenue, f"no revenue guide found: {_texts(result)}"
  assert any("quarter" in (s["period_label"] or "").lower() for s in revenue), (
      f"no quarterly period attached: {[s['period_label'] for s in revenue]}")


@network
def test_apple_is_reported_as_giving_no_guidance_in_its_release():
  """Apple withdrew formal guidance in 2020 and has not restored it. The
  release carries none, and this must come back as an absence of guidance
  language rather than an absence of data."""
  result = g.extract_guidance("AAPL", quarters=1)

  assert result["success"] is True, result["error"]
  assert result["guidance_found"] is False, _texts(result)
  assert result["no_guidance_reason"] == "no_guidance_language_found"
  assert result["sources"] and result["sources"][0]["text_available"] is True


@network
def test_salesforce_guides_the_full_year_in_prose():
  """Salesforce restates every guide as a prose bullet before tabulating it,
  which is why the full-year line survives while the EPS table does not."""
  result = g.extract_guidance("CRM", quarters=1)

  assert result["guidance_found"] is True
  labels = [(s["period_label"] or "").lower() for s in result["statements"]]
  assert any("full year" in label or "full-year" in label for label in labels), (
      f"no full-year guide: {labels}")


@network
@pytest.mark.parametrize("ticker", ["NVDA", "CRM", "INTC", "AMD", "KO", "UPS"])
def test_no_returned_statement_ever_carries_table_wreckage(ticker):
  """The invariant that makes the output safe to read.

  If a box rule, a truncated cell or a pair of amounts with nothing but
  whitespace between them reaches the caller, the column interleaving reached
  the caller too.
  """
  result = g.extract_guidance(ticker, quarters=1)

  for statement in result["statements"]:
    text = statement["text"]
    assert "─" not in text and "…" not in text, text
    assert not g.ADJACENT_AMOUNTS.search(text), text
