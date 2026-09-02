"""The surprise, dated from the release rather than the filing.

The time-series surprise is computed from XBRL, and XBRL does not exist until
the 10-Q is filed -- a median of eight days after the earnings 8-K. The
published drift is dated from the announcement and most of it happens in the
first days. A full replay over 230 dates and 2,216 names made the cost of that
lag concrete: the XBRL-timed arm returned a median trade of -36bp and a
coefficient whose interval contained zero, while the release-timed
cross-sectional arm on the same dates returned +122bp mean against -20bp.

Only one number has to move earlier. The standardisation needs eight quarters
of history, and those are months old and safely XBRL. The quarter being
announced is the one whose EPS must come from the 8-K, with the 8-K's date as
`known_at`. That is what this module does, and the whole risk is in reading
the right number out of prose.

Two real releases fix the shape of that risk. Apple's is clean. Photronics'
GAAP sentence carries the current figure, two prior-period comparatives and,
one sentence on, a non-GAAP figure that a naive regex picks up instead:

    GAAP Net income ... was $28.9 million, or $0.49 per diluted share,
    compared with $22.9 million, or $0.39 per diluted share, in the third
    quarter of 2025 and $31.4 million, or $0.54 per diluted share, in the
    second quarter of 2026. Non-GAAP Net income ... was $29.4 million, or
    $0.50 per diluted share ...

So the extractor is held to those, and every live extraction is checked
against the 10-Q's XBRL once that exists. Agreement is the extractor's
measured accuracy; disagreement refuses the row rather than guessing.
"""
from __future__ import annotations

import pytest

from research import release_eps, scoring

APPLE = ("Company gross margin was 50.1 percent, including a favorable impact "
         "of approximately 2 percentage points from tariff refunds. Diluted "
         "earnings per share was $2.02, up 29 percent year over year, and "
         "included a favorable impact of $0.11 from tariff refunds. "
         "“Today, Apple is proud to report our strongest June quarter "
         "ever.”")

PHOTRONICS = (
    "Revenue was $217.2 million, an increase of 2.7% year-over-year and 2.9% "
    "sequentially. GAAP Net income attributable to Photronics, Inc. "
    "shareholders was $28.9 million, or $0.49 per diluted share, compared "
    "with $22.9 million, or $0.39 per diluted share, in the third quarter of "
    "2025 and $31.4 million, or $0.54 per diluted share, in the second "
    "quarter of 2026. Non-GAAP Net income attributable to Photronics, Inc. "
    "shareholders was $29.4 million, or $0.50 per diluted share, compared "
    "with $29.4 million, or $0.51 per diluted share in the third quarter of "
    "2025 and $24.9 million, or $0.42 per diluted share, in the second "
    "quarter of 2026. IC revenue was $154.7 million.")


# --- reading the number out of prose ----------------------------------------

def test_a_clean_headline_sentence_is_read():
    out = release_eps.extract_diluted_eps(APPLE)

    assert out["eps"] == pytest.approx(2.02)
    assert out["reason"] is None
    assert "2.02" in out["evidence"]


def test_the_current_gaap_figure_wins_over_comparatives_and_non_gaap():
    """The first figure adjacent to the phrase, in the first GAAP sentence.
    $0.39 and $0.54 are prior periods, $0.50 is non-GAAP."""
    out = release_eps.extract_diluted_eps(PHOTRONICS)

    assert out["eps"] == pytest.approx(0.49)


def test_a_release_that_only_reports_non_gaap_is_refused():
    text = ("Non-GAAP diluted earnings per share was $1.10, up 12% year over "
            "year. Adjusted EBITDA was $400 million.")

    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] is None
    assert "GAAP" in out["reason"]


def test_a_loss_is_negative():
    for text in ("Net loss per diluted share was $(0.12) for the quarter.",
                 "Diluted net loss per share of $0.12, compared with net "
                 "income of $0.30 per diluted share a year ago.",
                 "GAAP diluted EPS was -$0.12."):
        out = release_eps.extract_diluted_eps(text)
        assert out["eps"] == pytest.approx(-0.12), text


def test_a_release_with_no_per_share_figure_is_refused():
    out = release_eps.extract_diluted_eps(
        "Revenue was $217.2 million, an increase of 2.7% year-over-year. "
        "Gross margin was 38%.")

    assert out["eps"] is None
    assert out["reason"]


def test_basic_eps_is_not_taken_for_diluted():
    """Basic and diluted differ, and the XBRL series is diluted."""
    out = release_eps.extract_diluted_eps(
        "Basic earnings per share was $1.05. Diluted earnings per share was "
        "$1.01.")

    assert out["eps"] == pytest.approx(1.01)


def test_empty_text_is_a_refusal_not_a_crash():
    assert release_eps.extract_diluted_eps("")["eps"] is None
    assert release_eps.extract_diluted_eps(None)["eps"] is None


# --- substituting one quarter into the XBRL series --------------------------

def _series():
    """Eight prior quarters of a flat-ish series plus the quarter being
    announced, as eps_series would build them from XBRL."""
    quarters = []
    fy, fq = 2024, 1
    # Not a straight line: identical year-on-year deltas give a sigma of
    # zero, which the signal correctly refuses -- and a refusal on both sides
    # of a comparison passes it vacuously.
    eps = [1.00, 1.07, 1.10, 1.18, 1.20, 1.29, 1.33, 1.35, 1.42, 1.44, 1.53,
           1.55]
    for i, value in enumerate(eps):
        quarters.append({
            "fiscal_period": f"{fy}Q{fq}", "fiscal_year": fy,
            "fiscal_quarter": fq, "eps": value, "eps_as_filed": value,
            "basis_factor": 1.0, "source": "xbrl", "concept": "EPS",
            "known_at": f"{fy}-{fq * 3:02d}-15", "accession": f"acc-{i}",
            "form": "10-Q", "derivation": None,
            "period_end": f"{fy}-{fq * 3:02d}-01"})
        fq += 1
        if fq == 5:
            fy, fq = fy + 1, 1
    return {"ticker": "AAA", "cik": "1", "success": True, "error": None,
            "concept": "EPS", "concepts_tried": ["EPS"], "basis_changes": [],
            "quarters": quarters}


def test_the_announced_quarter_takes_the_release_eps_and_date():
    series = _series()
    # 2026Q4's XBRL row is dated 2026-12-15; the release came 2026-11-30.
    signal = release_eps.signal_from_release(
        series, fiscal_period="2026Q4", release_eps=1.75,
        announced_date="2026-11-30")

    assert signal["success"] is True
    assert signal["eps"] == pytest.approx(1.75)
    assert signal["known_at"] == "2026-11-30"
    assert signal["variant"] == "ts_release"
    assert signal["source"] == "release"


def test_the_history_is_still_xbrl_and_the_sigma_is_unchanged(monkeypatch):
    """Only the announced quarter moves. Substituting the release figure into
    the history too would change the standardisation with hindsight."""
    from research import sue

    series = _series()
    xbrl = sue._signal_from_series(series, (2026, 4), "2026-12-31")
    rel = release_eps.signal_from_release(
        series, fiscal_period="2026Q4", release_eps=series["quarters"][-1]["eps"],
        announced_date="2026-11-30")

    assert xbrl["sigma"] and xbrl["sigma"] > 0, "the fixture must standardise"
    assert rel["sigma"] == pytest.approx(xbrl["sigma"])
    assert rel["sigma_quarters"] == xbrl["sigma_quarters"]
    assert rel["sue"] == pytest.approx(xbrl["sue"])


def test_the_release_figure_carries_the_quarters_split_basis():
    """XBRL EPS is rescaled for splits announced later. The release figure is
    on the share basis of its own day, exactly as the XBRL fact was as filed,
    so it takes the same factor."""
    series = _series()
    series["quarters"][-1]["basis_factor"] = 0.5   # a 2-for-1 came later

    signal = release_eps.signal_from_release(
        series, fiscal_period="2026Q4", release_eps=3.10,
        announced_date="2026-11-30")

    assert signal["eps"] == pytest.approx(1.55)


def test_agreement_with_the_later_xbrl_is_recorded_not_assumed():
    series = _series()     # 2026Q4 XBRL eps is 1.55

    agrees = release_eps.signal_from_release(
        series, "2026Q4", release_eps=1.55, announced_date="2026-11-30")
    off = release_eps.signal_from_release(
        series, "2026Q4", release_eps=1.75, announced_date="2026-11-30")

    assert agrees["xbrl_eps"] == pytest.approx(1.55)
    assert agrees["agrees_with_xbrl"] is True
    assert off["agrees_with_xbrl"] is False
    assert off["xbrl_eps"] == pytest.approx(1.55)


def test_a_cent_of_rounding_still_agrees():
    series = _series()
    signal = release_eps.signal_from_release(
        series, "2026Q4", release_eps=1.56, announced_date="2026-11-30")
    assert signal["agrees_with_xbrl"] is True


def test_a_quarter_the_xbrl_does_not_hold_yet_is_still_signalled():
    """The live case: the 8-K is out and the 10-Q is not. The standardisation
    uses the eight quarters before it, and agreement is unknown, not False."""
    series = _series()
    series["quarters"] = series["quarters"][:-1]       # drop 2026Q4

    signal = release_eps.signal_from_release(
        series, "2026Q4", release_eps=1.60, announced_date="2026-11-30")

    assert signal["success"] is True
    assert signal["known_at"] == "2026-11-30"
    assert signal["xbrl_eps"] is None
    assert signal["agrees_with_xbrl"] is None


def test_a_release_date_after_the_filing_is_refused():
    """A release dated after the 10-Q is not the announcement of that
    quarter; it is some other event, and using it would date the signal
    later than XBRL already does."""
    series = _series()
    signal = release_eps.signal_from_release(
        series, "2026Q4", release_eps=1.55, announced_date="2027-01-20")

    assert signal["success"] is False
    assert "after" in signal["error"]


# --- the variant is a first-class citizen downstream ------------------------

def test_the_scorer_prices_the_release_variant_as_a_sigma():
    assert scoring.UNIT_OF["ts_release"] == "sigma"


def test_the_scanner_can_be_set_to_the_release_variant(monkeypatch):
    from research import scanner

    seen = {}
    monkeypatch.setattr(release_eps, "sue_ts_release",
                        lambda t, as_of=None: seen.update(t=t) or {
                            "ticker": t, "success": True, "sue": 1.0,
                            "variant": "ts_release"})
    monkeypatch.setattr(scanner, "SIGNAL_VARIANT", "ts_release")

    out = scanner._signal_for("AAA", "2026-03-03")

    assert seen["t"] == "AAA"
    assert out["variant"] == "ts_release"


# --- one bad fetch is one bad row -------------------------------------------

def test_a_fetch_that_fails_costs_one_release_not_the_name(monkeypatch):
    """EDGAR timed out on one of Photronics' 8-Ks and the whole history for
    the name went with it -- and in a replay driver, the whole run. A failed
    fetch is a refusal for that quarter with the reason on the record, and
    every other quarter is still read."""
    from research import announcements, sue

    series = _series()
    monkeypatch.setattr(sue, "eps_series", lambda t, as_of=None: series)
    monkeypatch.setattr(announcements, "for_quarters",
                        lambda t, as_of=None, quarters=None: {
                            "2026Q3": {"accession": "good", "announced_date": "2026-08-30"},
                            "2026Q4": {"accession": "bad", "announced_date": "2026-11-30"}})

    def fetch(ticker, accession, period_end=None, filings=None):
        if accession == "bad":
            raise TimeoutError("The read operation timed out")
        return release_eps.extract_diluted_eps(
            "Diluted earnings per share was $1.50.")

    monkeypatch.setattr(release_eps, "read_release", fetch)

    class _Company:
        def get_filings(self, form=None):
            return []

    monkeypatch.setattr(release_eps, "_company", lambda ticker: _Company())

    out = release_eps.release_history("AAA")

    assert out["error"] is None
    assert [s["fiscal_period"] for s in out["signals"]] == ["2026Q3"]
    bad = next(r for r in out["extractions"] if r["fiscal_period"] == "2026Q4")
    assert "TimeoutError" in bad["reason"]


# --- the three ways real releases defeated the first extractor ---------------
#
# Measured on four names against the 10-Q's XBRL: Apple 6 for 6, then
# Photronics wrong once, JPMorgan read zero of twelve, Rivian read zero of
# nineteen. Each was a systematic shape, not noise, and each is fixed by a rule
# that a fixture from the real release pins.

PHOTRONICS_Q4 = (
    "Fiscal 2025 revenue was $867.1 million, up 1% year-over-year. GAAP net "
    "income attributable to Photronics, Inc. shareholders was $136.4 million, "
    "or $2.28 per diluted share, compared with $130.7 million, or $2.05 per "
    "diluted share in 2024. Favorable impact associated with the deferred tax "
    "valuation allowance reduction of $16.8 million. Non-GAAP net income "
    "attributable to Photronics, Inc. shareholders was $120.6 million, or "
    "$2.01 per diluted share, compared with $127.6 million, or $2.05 per "
    "diluted share in 2024. Fourth quarter revenue was $221.5 million, up 2.6% "
    "sequentially. GAAP Net income attributable to Photronics, Inc. "
    "shareholders was $61.8 million, or $1.07 per diluted share, compared with "
    "$33.9 million, or $0.54 per diluted share, in the fourth quarter of 2024 "
    "and $22.9 million, or $0.39 per diluted share, in the third quarter of "
    "2025.")

JPM_HEADLINE = (
    "JPMORGANCHASE REPORTS SECOND-QUARTER 2026 NET INCOME OF $21.2 BILLION "
    "( $7.70 PER SHARE), NET INCOME EXCLUDING SIGNIFICANT ITEMS OF $16.9 "
    "BILLION ($6.14 PER SHARE) SECOND-QUARTER 2026 RESULTS. Common dividend of "
    "$4.0 billion or $1.50 per share. Book value per share of $133.01, up 9% "
    "YoY; tangible book value per share of $113.35, up 10% YoY.")

RIVIAN_TABLE = (
    "Condensed Consolidated Statements of Operations (in millions, except per "
    "share amounts) (unaudited) Three Months Ended March 31, 2025 2026 "
    "Revenues 1,240 1,520 Net loss -541 -416 Net loss attributable to common "
    "$ -545 $ -416 stockholders, basic and diluted Net loss per share "
    "attributable to Class A and Class B $ -0.48 $ -0.33 common stockholders, "
    "basic and diluted Weighted-average common shares 1,137 1,249 outstanding, "
    "basic and diluted")


def test_a_fourth_quarter_release_yields_the_quarter_not_the_year():
    """Q4 releases lead with the fiscal year. The first figure was $2.28 --
    the year -- and XBRL said 1.05 for the quarter. A sentence that names the
    quarter outranks one that does not."""
    out = release_eps.extract_diluted_eps(PHOTRONICS_Q4)

    assert out["eps"] == pytest.approx(1.07)


def test_a_bank_headline_with_no_word_diluted_is_read_as_the_gaap_figure():
    """Banks say "per share" and never "diluted". The GAAP figure is the first
    one; the one after "excluding significant items" is adjusted; the dividend
    and book value lines are not earnings at all."""
    out = release_eps.extract_diluted_eps(JPM_HEADLINE)

    assert out["eps"] == pytest.approx(7.70)
    assert out["basis"] == "per_share"


def test_a_dividend_or_book_value_per_share_is_never_earnings():
    for text in ("The Board declared a quarterly dividend of $1.50 per share.",
                 "Book value per share of $133.01, up 9% YoY.",
                 "The company repurchased shares at an average of $210.40 per share."):
        assert release_eps.extract_diluted_eps(text)["eps"] is None, text


def test_a_table_is_read_by_matching_the_column_to_the_period_end():
    """Rivian's loss lives only in a table whose columns run prior-year first.
    Which column is "this quarter" cannot be known from the row; it is the one
    whose header date is the quarter's period end."""
    out = release_eps.extract_diluted_eps(RIVIAN_TABLE, period_end="2026-03-31")

    assert out["eps"] == pytest.approx(-0.33)
    assert out["basis"] == "table"


def test_the_same_table_read_for_the_prior_quarter_gives_the_other_column():
    out = release_eps.extract_diluted_eps(RIVIAN_TABLE, period_end="2025-03-31")

    assert out["eps"] == pytest.approx(-0.48)


def test_a_table_without_a_period_end_to_match_is_refused_not_guessed():
    out = release_eps.extract_diluted_eps(RIVIAN_TABLE)

    assert out["eps"] is None
    assert "column" in out["reason"] or "period" in out["reason"]


def test_a_table_whose_columns_do_not_hold_the_period_end_is_refused():
    out = release_eps.extract_diluted_eps(RIVIAN_TABLE, period_end="2026-06-30")

    assert out["eps"] is None


def test_prose_wins_over_a_table_when_both_are_present():
    text = APPLE + " " + RIVIAN_TABLE
    out = release_eps.extract_diluted_eps(text, period_end="2026-03-31")

    assert out["eps"] == pytest.approx(2.02)
    assert out["basis"] == "gaap"


# --- which exhibit, and one retry ------------------------------------------

class _Attachment:
    def __init__(self, document, description, text):
        self.document, self.description, self._text = document, description, text

    def text(self):
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class _Homepage:
    def __init__(self, documents):
        self.documents = documents


class _Filing:
    """Shaped like edgartools' Filing as the fetch layer uses it: the index
    page lists the documents. `attachments` is deliberately absent, because
    reaching for it parses the whole submission."""
    def __init__(self, accession_no, documents):
        self.accession_no, self.homepage = accession_no, _Homepage(documents)


def _company_with(filings, monkeypatch):
    class _Company:
        def __init__(self, ticker):
            pass

        def get_filings(self, form=None):
            return filings
    monkeypatch.setattr(release_eps, "_company", lambda ticker: _Company(ticker))


def test_exhibit_99_1_is_preferred_and_a_substring_match_on_99_2_is_not_enough(
        monkeypatch):
    """JPMorgan names its exhibits `exhibit991narrative` and `ex992supplement`.
    A substring check on `ex99` matched the supplement -- tables only -- and
    the narrative press release was never read."""
    filing = _Filing("acc", [
        _Attachment("a2q26erfex992supplement.htm", "EARNINGS RELEASE FINANCIAL SUPPLEMENT",
                    "Earnings Per Share and Related Information Page 11"),
        _Attachment("a2q26erfexhibit991narrative.htm", "EARNINGS RELEASE - SECOND QUARTER",
                    JPM_HEADLINE),
    ])
    _company_with([filing], monkeypatch)

    text = release_eps._release_text("JPM", "acc")

    assert "$7.70 PER SHARE" in text


def test_every_ex99_exhibit_is_tried_until_one_yields_a_figure(monkeypatch):
    filing = _Filing("acc", [
        _Attachment("ex99-1.htm", "EX-99.1", "Revenue was $1 billion."),
        _Attachment("ex99-2.htm", "EX-99.2", APPLE),
    ])
    _company_with([filing], monkeypatch)

    out = release_eps.read_release("AAPL", "acc")

    assert out["eps"] == pytest.approx(2.02)


def test_a_read_timeout_is_retried_once(monkeypatch):
    calls = {"n": 0}

    class _Flaky(_Attachment):
        def text(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("The read operation timed out")
            return APPLE

    _company_with([_Filing("acc", [_Flaky("ex99-1.htm", "EX-99.1", None)])],
                  monkeypatch)
    monkeypatch.setattr(release_eps.time, "sleep", lambda s: None)

    assert "$2.02" in release_eps._release_text("AAPL", "acc")
    assert calls["n"] == 2


def test_a_second_timeout_is_raised_not_swallowed(monkeypatch):
    class _Dead(_Attachment):
        def text(self):
            raise TimeoutError("The read operation timed out")

    _company_with([_Filing("acc", [_Dead("ex99-1.htm", "EX-99.1", None)])],
                  monkeypatch)
    monkeypatch.setattr(release_eps.time, "sleep", lambda s: None)

    with pytest.raises(TimeoutError):
        release_eps._release_text("AAPL", "acc")


def test_a_release_with_only_a_basic_figure_is_refused():
    """Basic is not diluted, and the XBRL series is diluted. With a diluted
    sentence present the ranking prefers it; with none present the basic
    figure must not be read as if it were."""
    out = release_eps.extract_diluted_eps(
        "Net income was $40 million. Basic earnings per share was $1.05, up "
        "from $0.90. Revenue grew 4%.")

    assert out["eps"] is None


# --- three more shapes, from the disagreements the first pass left ----------

APPLE_FY24_Q4 = (
    "Apple today announced financial results for its fiscal 2024 fourth "
    "quarter ended September 28, 2024. The Company posted quarterly revenue "
    "of $94.9 billion, up 6 percent year over year, and quarterly diluted "
    "earnings per share of $0.97. Diluted earnings per share was $1.64, up 12 "
    "percent year over year when excluding the one-time charge recognized "
    "during the fourth quarter of 2024 related to the impact of the reversal "
    "of the European General Court's State Aid decision.")

PHOTRONICS_BULLETS = (
    "Fourth quarter and fiscal year 2023 highlights • Fifth consecutive year "
    "of record revenue GAAP net income attributable to Photronics, Inc. "
    "shareholders was $125.5 • million, or $2.03 per diluted share, compared "
    "with $118.8 million, or $1.94 per diluted share in 2022 Non-GAAP net "
    "income attributable to Photronics, Inc. shareholders was • $126.0 "
    "million, or $2.04 per diluted share, compared with $101.7 million, or "
    "$1.66 per diluted share in 2022 • IC revenue was $651.3 million, up 10% "
    "• Fourth quarter revenue was $227.1 million • GAAP net income "
    "attributable to Photronics, Inc. shareholders was $44.7 million, or "
    "$0.72 per diluted share, compared with $37.4 million, or $0.60 per "
    "diluted share, in the fourth quarter of 2022 • Non-GAAP net income was "
    "$45.1 million, or $0.73 per diluted share")


def test_an_adjusted_figure_qualified_after_the_number_loses_to_the_gaap_one():
    """"Diluted earnings per share was $1.64 ... when excluding the one-time
    charge": the qualifier comes after the figure, so a check on the text
    before the phrase let it through, and "fourth quarter" in the same
    sentence then outranked the plain GAAP sentence before it."""
    out = release_eps.extract_diluted_eps(APPLE_FY24_Q4)

    assert out["eps"] == pytest.approx(0.97)


def test_a_flattened_bullet_list_is_split_at_the_bullets():
    """Older Photronics releases carry no full stops at all -- the page is
    bullets -- so to a sentence splitter the whole thing is one sentence and
    the year's $2.03 is always the first figure in it."""
    out = release_eps.extract_diluted_eps(PHOTRONICS_BULLETS)

    assert out["eps"] == pytest.approx(0.72)


def test_a_table_cell_in_parentheses_is_negative():
    table = ("Three Months Ended March 31, 2021 2022 Net loss per share, "
             "basic and diluted $ (0.90) $ (1.77) Weighted-average shares "
             "basic and diluted 800 900")
    # The row regex wants the diluted word after the cells; this layout has it
    # before. Both are real, so both must read.
    out = release_eps.extract_diluted_eps(table, period_end="2022-03-31")

    assert out["eps"] == pytest.approx(-1.77)


def test_exhibits_come_from_the_filing_index_not_the_full_submission(
        monkeypatch):
    """`filing.attachments` parses the entire SGML submission -- for Rivian,
    28 embedded images -- to list the exhibits: 17 seconds a release, and the
    read timeouts came from there. The filing's homepage lists the same
    documents from a small index page."""
    class _NoSGML(_Filing):
        @property
        def attachments(self):
            raise AssertionError("the full submission was parsed")

    _company_with([_NoSGML("acc", [_Attachment("ex-991q26.htm", "EX-99.1",
                                                APPLE)])], monkeypatch)

    assert release_eps.read_release("AAPL", "acc")["eps"] == pytest.approx(2.02)


def test_a_bullet_inside_a_sentence_does_not_split_it():
    """Photronics wraps lines with bullets: "$28.9 • million, or $0.49 per
    diluted share". Splitting there strands the non-GAAP sentence's figure
    in a fragment with no qualifier, which then reads as GAAP. Seventeen
    misreads on one name. A bullet is a boundary only where a new item
    starts -- an uppercase letter follows it."""
    text = ("Non-GAAP Net income attributable to shareholders was $34.7 • "
            "million, or $0.55 per diluted share, compared with $0.41 per "
            "diluted share in the third quarter of 2024. GAAP Net income "
            "attributable to shareholders was $24.5 million, or $0.39 per "
            "diluted share, compared with $0.30 in the third quarter of 2024 "
            "• IC revenue was $150 million.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.39)


def test_the_filings_list_is_fetched_once_per_name(monkeypatch):
    """`Company(ticker).get_filings` per accession was 14 seconds a release
    on JPMorgan. A name's releases all live in the one list."""
    from research import announcements, sue

    calls = {"n": 0}
    series = _series()
    monkeypatch.setattr(sue, "eps_series", lambda t, as_of=None: series)
    monkeypatch.setattr(announcements, "for_quarters",
                        lambda t, as_of=None, quarters=None: {
                            "2026Q3": {"accession": "a", "announced_date": "2026-08-30"},
                            "2026Q4": {"accession": "b", "announced_date": "2026-11-30"}})

    class _Company:
        def get_filings(self, form=None):
            calls["n"] += 1
            return [_Filing("a", [_Attachment("ex99-1.htm", "EX-99.1", APPLE)]),
                    _Filing("b", [_Attachment("ex99-1.htm", "EX-99.1", APPLE)])]

    monkeypatch.setattr(release_eps, "_company", lambda ticker: _Company())

    out = release_eps.release_history("AAA")

    assert len(out["signals"]) == 2
    assert calls["n"] == 1


def test_a_flattened_statement_header_is_not_read_as_a_sentence():
    """Coca-Cola's income statement flattens to "(In millions except per Three
    June share Months 27, June 28, ... Ended 2025 2024)" and the prose path
    matched "per share" inside it, then took the six-month column's $1.65 for
    a quarter XBRL puts at $0.88. A statement region is a table, and only the
    table reader may read it -- by column."""
    # Flattened cell order is not row order. The six-month figure lands
    # right after the diluted row label, which is where prose looks.
    text = ("Consolidated Statements of Income (In millions except per share "
            "data) Six Months Ended June 27, 2025 June 28, 2024 Diluted net "
            "income per share $ 1.65 $ 1.50 Three Months Ended June 27, 2025 "
            "June 28, 2024 $ 0.88 $ 0.71 Diluted weighted-average shares "
            "4,310 4,320")

    out = release_eps.extract_diluted_eps(text, period_end="2025-06-27")

    # Refused or read right; never the six-month column.
    assert out["eps"] != pytest.approx(1.65)


def test_a_year_to_date_sentence_ranks_below_the_quarter():
    text = ("Net income per diluted share was $1.65 for the six months ended "
            "June 27, 2025. For the quarter, diluted earnings per share was "
            "$0.88, up 4 percent.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.88)


def test_a_headline_that_names_its_own_quarter_is_still_prose():
    """"Months ended" is how a headline paragraph names its quarter. A guard
    that took it for a statement header refused all 23 of Coca-Cola's
    releases."""
    text = ("For the three months ended June 27, 2025, net income per diluted "
            "share was $0.88, compared with $0.71 a year ago.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.88)


def test_a_figure_reached_through_a_growth_verb_is_read():
    """Coca-Cola: "EPS grew 18% to $0.91". A verb and a percentage sit between
    the phrase and the figure, and an adjacency rule that allowed only
    "was", "of" or a colon refused all 23 of its releases. "Comparable" is
    its word for non-GAAP."""
    for text, want in (
        ("Earnings per share: EPS grew 18% to $0.91, and comparable EPS "
         "(non-GAAP) grew 18% to $0.86.", 0.91),
        ("EPS Grew 18% to $0.91; Comparable EPS (Non-GAAP) Grew 18% to "
         "$0.86", 0.91),
        ("Diluted earnings per share increased 12% to $2.02.", 2.02),
        ("Net income per diluted share rose to $1.10 from $0.95.", 1.10),
        ("Diluted EPS declined 5% to $0.40.", 0.40),
    ):
        assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(want), text


def test_comparable_is_non_gaap():
    text = "Comparable EPS (non-GAAP) grew 18% to $0.86."

    assert release_eps.extract_diluted_eps(text)["eps"] is None


# --- the shapes a hundred-name audit added -----------------------------------
#
# 2,337 releases, 69% agreement on reads. The unqualified "per share" fallback
# was 57% right; the diluted phrase 78%. Each shape below cost tens of reads.

def test_a_component_of_earnings_is_not_the_earnings():
    """Aflac: the sentence naming the quarter described a component --
    "included pretax net realized investment losses of $322 million, or
    $0.42 per diluted share" -- and outranked the headline "$0.69 per diluted
    share" that did not name the quarter. 25 of 30 releases wrong."""
    text = ("Net earnings were $525 million, or $0.69 per diluted share, "
            "compared with $2.6 billion, or $3.27 per diluted share a year ago. "
            "Net earnings in the fourth quarter of 2018 included pretax net "
            "realized investment losses of $322 million, or $0.42 per diluted "
            "share, compared with pretax net gains of $58 million, or $0.07 "
            "per diluted share a year ago.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.69)


def test_continuing_operations_loses_to_a_total_figure():
    """XBRL's diluted EPS is the total. "From continuing operations" is a
    different basis when the company had discontinued ones."""
    text = ("GAAP net income from continuing operations was $23.4 million or "
            "$0.61 per diluted share. GAAP net income was $31.5 million, or "
            "$0.82 per diluted share.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.82)


def test_continuing_operations_is_accepted_when_it_is_all_there_is():
    text = ("GAAP net income from continuing operations was $23.4 million or "
            "$0.61 per diluted share, compared with $0.40 in the prior quarter.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.61)


def test_a_net_loss_sentence_makes_an_unqualified_figure_negative():
    """Alcoa: "Net loss attributable to Alcoa Corporation was $746 million,
    or $4.17 per share" read as +4.17, because the sign came only from the
    phrase and the phrase was "per share". 63 sign flips in a hundred names."""
    text = ("Net loss attributable to Alcoa Corporation was $746 million, or "
            "$4.17 per share, primarily due to the decline in aluminum prices.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(-4.17)


def test_a_parenthetical_loss_label_does_not_force_a_negative():
    """"Net income (loss) per diluted share of $0.30" is a row label with
    both words in it. The figure is positive unless the digits say so."""
    text = "Net income (loss) per diluted share was $0.30, up from $0.12."

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(0.30)


def test_cents_are_read():
    """Agilent writes every figure in cents. 214 "phrase found, no figure
    beside it" refusals in a hundred names."""
    for text, want in (
        ("GAAP net income of $101 million, or 32 cents per share.", 0.32),
        ("Diluted earnings per share of 85 cents, up from 70 cents.", 0.85),
        ("GAAP net loss of $12 million, or 5 cents per diluted share.", -0.05),
        ("Diluted earnings per share were $.85, compared with $.70.", 0.85),
    ):
        assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(want), text


def test_a_reit_or_insurer_metric_per_share_is_not_earnings():
    """AGNC: "net spread and dollar roll income per common share" is the
    headline, and it is non-GAAP by another name. Five preferred tickers of
    the same issuer read it 125 times."""
    for text in (
        "Net spread and dollar roll income of $0.44 per common share.",
        "Comprehensive income of $0.12 per common share for the quarter.",
        "Distributable earnings of $1.10 per share.",
        "Funds from operations (FFO) of $2.05 per diluted share.",
        "Core earnings of $0.95 per share.",
        "Operating earnings per share were $1.20.",
    ):
        assert release_eps.extract_diluted_eps(text)["eps"] is None, text


def test_the_unqualified_fallback_needs_net_income_wording():
    """"Earnings" alone let too much through: 57% right against 78% for the
    diluted phrase. The fallback reads "per share" only beside net income,
    net earnings or net loss."""
    assert release_eps.extract_diluted_eps(
        "Earnings of $1.20 per share reflect strong demand.")["eps"] is None
    assert release_eps.extract_diluted_eps(
        "Net income of $1.2 billion, or $1.20 per share.")["eps"] == pytest.approx(1.20)


def test_a_filing_missing_from_the_list_says_so(monkeypatch):
    """113 refusals said "no EX-99 exhibit attached" for filings that plainly
    carry one. The filing was not in the list the lookup searched, and the
    reason blamed the exhibit."""
    class _Company:
        def get_filings(self, form=None):
            return [_Filing("other", [_Attachment("ex99-1.htm", "EX-99.1", APPLE)])]

    monkeypatch.setattr(release_eps, "_company", lambda ticker: _Company())
    monkeypatch.setattr(release_eps, "_filing_by_accession", lambda acc: None)

    out = release_eps.read_release("AAPL", "acc-not-listed")

    assert out["eps"] is None
    assert "not in" in out["reason"] or "not found" in out["reason"]


def test_a_filing_missing_from_the_list_is_resolved_the_slow_way(monkeypatch):
    """The list is the fast path. A filing it does not hold -- an old one past
    the page, a list that came back short -- is resolved through the
    quarterly index before the release is given up on."""
    class _Company:
        def get_filings(self, form=None):
            return [_Filing("other", [])]

    monkeypatch.setattr(release_eps, "_company", lambda ticker: _Company())
    monkeypatch.setattr(
        release_eps, "_filing_by_accession",
        lambda acc: _Filing(acc, [_Attachment("ex99-1.htm", "EX-99.1", APPLE)]))

    assert release_eps.read_release("AAPL", "old-acc")["eps"] == pytest.approx(2.02)


# --- the quarter word belongs to the comparison ------------------------------
#
# Each test below is arranged so that only its own rule can save it: the
# sibling rules are kept out of reach by sentence order and wording. A first
# version of these passed with every rule removed.

def test_a_comparison_opener_demotes_even_with_the_quarter_word_first():
    """Advanced Energy, 32 of 33 wrong. "In the fourth quarter of 2025, this
    compares with ... $1.94" has the quarter word before the figure, so the
    position rule cannot help; only recognising a comparison can."""
    text = ("GAAP net income was $62 million or $1.58 per diluted share. In the "
            "fourth quarter of 2025, this compares with $75 million or $1.94 per "
            "diluted share.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(1.58)


def test_a_quarter_word_after_the_figure_names_the_comparison():
    """No comparison opener to catch here; the quarter word simply follows
    the figure, and that is where a prior-period figure puts it."""
    text = ("GAAP net income was $62 million or $1.58 per diluted share. Net "
            "income was $75 million or $1.94 per diluted share in the fourth "
            "quarter of 2025.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(1.58)


def test_a_quarter_word_before_the_figure_still_counts():
    text = ("For the fiscal year, diluted earnings per share were $6.10. "
            "Fourth quarter diluted earnings per share were $1.58, compared "
            "with $1.94 in the prior quarter.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(1.58)


def test_a_line_item_dressed_as_a_headline_is_a_component():
    """AGNC, 23 of 31 wrong. The line-item sentence comes FIRST here and
    names the quarter before its figure, so neither order nor position can
    save it; only recognising "in other ..., net" as a component can."""
    text = ("OTHER GAIN (LOSS), NET For the first quarter, the Company recorded "
            "a net loss of $(433) million in other gain (loss), net, or $(0.39) "
            "per common share. For the first quarter, net loss available to "
            "common stockholders was $(155) million, or $(0.17) per common "
            "share.")

    assert release_eps.extract_diluted_eps(text)["eps"] == pytest.approx(-0.17)
