"""A rate and an average are claims about a company, not counts of rows.

`get_earnings_surprises("CREG")`, live 2026-08-26, returned four "quarters"
spanning 2011-06-30 to 2026-03-31 -- a fifteen-year hole presented as
contiguous history -- and summarised them:

    quarters  2026 Q1  actual -0.03   estimate null
              2023 Q3  actual -0.02   estimate null
              2011 Q3  actual  1600   estimate 918   surprise +74.29%
              2011 Q2  actual   700   estimate 561   surprise +24.78%
    beat_count 2   miss_count 0   total_periods 4
    avg_surprise_pct 49.53
    beat_rate_pct    50.0
    success true, warnings []

Three separate defects in two fields:

1. `avg_surprise_pct: 49.53` is (74.29 + 24.78) / 2 -- computed entirely from
   the two 2011 rows. Nothing the company has done since 2011 contributes to
   it, and nothing in the response says so. `Financial_Analysis_Agent` prints
   it as "Avg surprise: 49.53%" into the analysis prompt.

2. `beat_rate_pct: 50.0` is 2 beats over `total_periods: 4`, and two of those
   four rows have `estimate_eps: null` -- they could not beat or miss. The
   denominator counts rows that were never graded. The graded population is
   two, both beats, so the arithmetic answer is 100.0 and the honest answer
   is neither, because those two quarters are fifteen years old. This is not
   cosmetic: `Financial_Modeling_Agent` raises bull_growth_y1 by 200bps when
   `beat_rate_pct > 75` and cuts it when `< 40`, so 50.0 lands in the dead
   band that a corrected 100.0 would leave.

3. `actual_eps` of 1600 and 700 sit in the same array as -0.03 and -0.02.
   Those are pre-reverse-split figures: 80,000x apart on a per-share basis,
   so the array does not have one share basis and no statistic across it is
   defined. This is `get_earnings_surprises`'s own denomination rule -- "a
   figure carries its currency, its scale and the source that produced it" --
   applied within a single response instead of across two.

The approach follows the fix already landed for TGT's duplicate fiscal
periods: the rows are left exactly as Finnhub sent them and the defect is
DECLARED, because which row is wrong is Finnhub's to say and dropping one
would silently change counts other callers already read. What differs here is
that the derived statistics are withheld rather than emitted-and-flagged. A
duplicate quarter overweights an average that still means something; a hole
and a share-basis break mean the average has no referent at all, and the
repository's rule is to refuse rather than emit a plausible wrong number.

Also checked here: FDX. Finnhub's newest earnings row for FDX is fiscal 2026
Q1, bucket 2025-09-30, while `extract_guidance` shows FDX earnings 8-Ks on
2025-12-18, 2026-03-19 and 2026-06-23. Verified against the raw endpoint on
2026-08-26: `/stock/earnings` returns four rows for FDX at limit=12 AND at
limit=30, and the same call for TGT returns a 2026-09-30 bucket -- so this is
Finnhub not having published three FDX prints, not our cap and not our
condenser. A provider coverage limit, then; the defect is that the response
says nothing about it and the envelope's `data_as_of` is null, so a chain
reading FDX earnings alongside FDX filings cannot see that one of the two is
four calendar quarters behind.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.news_agregator.finnhub_utils import (
    fiscal_period_gaps,
    share_basis_discontinuity,
    summarize_earnings_surprises,
)

# Verbatim from /stock/earnings, 2026-08-26, condensed to the entry shape
# `_condense_earnings_surprises` builds.
CREG = [
    {"period": "2026-03-31", "year": 2026, "quarter": 1,
     "actual_eps": -0.03, "estimate_eps": None},
    {"period": "2023-09-30", "year": 2023, "quarter": 3,
     "actual_eps": -0.02, "estimate_eps": None},
    {"period": "2011-09-30", "year": 2011, "quarter": 3, "actual_eps": 1600,
     "estimate_eps": 918, "surprise_pct": 74.29, "result": "beat"},
    {"period": "2011-06-30", "year": 2011, "quarter": 2, "actual_eps": 700,
     "estimate_eps": 561, "surprise_pct": 24.78, "result": "beat"},
]

FDX = [
    {"period": "2025-09-30", "year": 2026, "quarter": 1, "actual_eps": 3.83,
     "estimate_eps": 3.6297, "surprise_pct": 5.52, "result": "beat"},
    {"period": "2025-06-30", "year": 2025, "quarter": 4, "actual_eps": 6.07,
     "estimate_eps": 5.8715, "surprise_pct": 3.38, "result": "beat"},
    {"period": "2025-03-31", "year": 2025, "quarter": 3, "actual_eps": 4.51,
     "estimate_eps": 4.5797, "surprise_pct": -1.52, "result": "miss"},
    {"period": "2024-12-31", "year": 2025, "quarter": 2, "actual_eps": 4.05,
     "estimate_eps": 3.9353, "surprise_pct": 2.91, "result": "beat"},
]

# TGT's duplicate fiscal quarter, already declared by _duplicate_fiscal_periods.
# It must keep its summary: a repeated quarter overweights an average that
# still describes the company, which is a different defect from a hole.
TGT = [
    {"period": "2026-09-30", "year": 2027, "quarter": 2, "actual_eps": 2.46,
     "estimate_eps": 2.3095, "surprise_pct": 6.52, "result": "beat"},
    {"period": "2025-09-30", "year": 2027, "quarter": 2, "actual_eps": 2.46,
     "estimate_eps": 2.3095, "surprise_pct": 6.52, "result": "beat"},
    {"period": "2026-06-30", "year": 2027, "quarter": 1, "actual_eps": 1.71,
     "estimate_eps": 1.477, "surprise_pct": 15.78, "result": "beat"},
    {"period": "2026-03-31", "year": 2026, "quarter": 4, "actual_eps": 2.44,
     "estimate_eps": 2.177, "surprise_pct": 12.08, "result": "beat"},
]

TODAY = "2026-08-26"


# --------------------------------------------------------------------------
# the hole
# --------------------------------------------------------------------------

def test_a_fifteen_year_hole_is_found_and_measured():
    gaps = fiscal_period_gaps(CREG)
    assert gaps, "a series running 2011 Q2 -> 2026 Q1 was read as contiguous"
    missing = {g["quarters_missing"] for g in gaps}
    assert 47 in missing, (
        f"the 2011 Q3 -> 2023 Q3 hole was not measured: {gaps}")
    assert 9 in missing, (
        f"the 2023 Q3 -> 2026 Q1 hole was not measured: {gaps}")


def test_a_contiguous_series_has_no_gaps():
    assert fiscal_period_gaps(FDX) == []


def test_a_repeated_quarter_is_not_a_gap():
    """TGT files fiscal 2027 Q2 twice. A repeat is a duplicate, not a hole."""
    assert fiscal_period_gaps(TGT) == []


# --------------------------------------------------------------------------
# the share basis
# --------------------------------------------------------------------------

def test_pre_reverse_split_eps_beside_post_split_eps_is_found():
    break_ = share_basis_discontinuity(CREG)
    assert break_ is not None, (
        "actual_eps of 1600 and -0.03 were read as one share basis")
    assert break_["ratio"] >= 1000
    assert break_["max_abs_actual_eps"] == 1600


def test_an_ordinary_eps_series_is_not_a_share_basis_break():
    assert share_basis_discontinuity(FDX) is None
    assert share_basis_discontinuity(TGT) is None


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------

def test_no_average_surprise_is_reported_over_a_hole():
    summary = summarize_earnings_surprises(CREG, today=TODAY)
    assert summary["avg_surprise_pct"] is None, (
        "an average computed entirely from 2011 was reported as CREG's "
        f"surprise history: {summary['avg_surprise_pct']}")
    assert summary["avg_surprise_pct_unavailable"], (
        "the average was withheld without saying why")


def test_no_beat_rate_is_reported_over_a_hole():
    summary = summarize_earnings_surprises(CREG, today=TODAY)
    assert summary["beat_rate_pct"] is None
    assert summary["beat_rate_pct_unavailable"]


def test_the_counts_survive_because_they_count_rows_not_the_company():
    """Declared, not deduplicated -- the TGT precedent. The rows are facts."""
    summary = summarize_earnings_surprises(CREG, today=TODAY)
    assert summary["beat_count"] == 2
    assert summary["miss_count"] == 0
    assert summary["total_periods"] == 4


def test_rows_with_no_estimate_are_counted_separately():
    """A row Finnhub never priced could not beat and could not miss."""
    summary = summarize_earnings_surprises(CREG, today=TODAY)
    assert summary["graded_periods"] == 2
    assert summary["ungraded_periods"] == 2


def test_a_clean_series_still_gets_its_summary():
    summary = summarize_earnings_surprises(FDX, today=TODAY)
    assert summary["avg_surprise_pct"] == 2.57
    assert summary["beat_rate_pct"] == 75.0
    assert summary["graded_periods"] == 4
    assert summary["ungraded_periods"] == 0


def test_the_beat_rate_denominator_is_the_graded_population():
    """The bug behind CREG's 50.0: rows with no estimate were in the divisor."""
    partly = [FDX[0], FDX[1], FDX[2],
              {"period": "2024-12-31", "year": 2025, "quarter": 2,
               "actual_eps": 4.05, "estimate_eps": None}]
    summary = summarize_earnings_surprises(partly, today=TODAY)
    assert summary["graded_periods"] == 3
    # 2 beats of 3 graded, not 2 of 4 rows.
    assert summary["beat_rate_pct"] == 66.7, summary
    assert "3" in summary["beat_rate_basis"], summary["beat_rate_basis"]


def test_every_reported_statistic_says_what_it_was_computed_over():
    summary = summarize_earnings_surprises(FDX, today=TODAY)
    assert summary["avg_surprise_pct_basis"], summary
    basis = summary["avg_surprise_pct_basis"]
    assert basis["rows"] == 4
    assert basis["fiscal_first"] == [2025, 2]
    assert basis["fiscal_last"] == [2026, 1]


def test_tgts_declared_duplicate_keeps_its_summary():
    """The already-landed fix declares the duplicate; it does not withhold."""
    summary = summarize_earnings_surprises(TGT, today=TODAY)
    assert summary["avg_surprise_pct"] is not None
    assert summary["beat_rate_pct"] == 100.0


# --------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------

def test_an_earnings_history_four_quarters_behind_the_calendar_says_so():
    """FDX: Finnhub's newest bucket is 2025-09-30 on a 2026-08-26 session."""
    summary = summarize_earnings_surprises(FDX, today=TODAY)
    assert summary["calendar_quarters_behind"] == 4, summary
    assert summary["history_is_stale"] is True


def test_a_current_history_is_not_flagged_stale():
    summary = summarize_earnings_surprises(TGT, today=TODAY)
    assert summary["calendar_quarters_behind"] == 0
    assert summary["history_is_stale"] is False


def test_the_newest_period_is_named_by_its_fiscal_identity():
    """`period` is a calendar bucket, not a date the company reached, so the
    latest period is reported on the fiscal identity the response already
    documents as the thing to join on."""
    summary = summarize_earnings_surprises(FDX, today=TODAY)
    latest = summary["latest_fiscal_period"]
    assert latest["year"] == 2026 and latest["quarter"] == 1
    assert latest["period_bucket"] == "2025-09-30"


# --------------------------------------------------------------------------
# empties
# --------------------------------------------------------------------------

def test_no_rows_means_no_statistics_at_all():
    """test_no_data_is_not_a_verdict: a verdict from no rows is not a verdict."""
    summary = summarize_earnings_surprises([], today=TODAY)
    for field in ("avg_surprise_pct", "beat_rate_pct", "beat_count",
                  "miss_count", "latest_fiscal_period",
                  "calendar_quarters_behind"):
        assert summary[field] is None, f"{field} was reported over zero rows"
