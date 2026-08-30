"""The band a TTM cross-check may use, derived from the filer's own growth.

`test_ttm_revenue_agrees_between_vendors` compares two vendor TTM revenues.
Neither vendor publishes where its window ends, so on a fast-moving name the
check measures the cutoff difference rather than agreement about a fact.

Measured: NVDA's own adjacent TTM windows are 229.426bn and 281.585bn, 22.7%
apart -- wider than the 19.5% between the two vendors, so a one-quarter
difference in cutoff is a sufficient explanation and no extraction error is
needed to produce the gap. The flat 3% band cannot express that.

This is not "widen the tolerance until it passes", which the reconciliation
module rightly forbids. The band is a measured property of the filer: it stays
at 3% for a name whose revenue is flat, and only opens for one whose own
quarters say a cutoff difference is worth that much.
"""
import pytest

from testing import test_cross_source_reconciliation as rec


def _quarters(values):
    """Newest-first quarterly revenues, as the helper expects."""
    return list(values)


def test_a_flat_filer_keeps_the_floor():
    """Four identical quarters: adjacent TTM windows are the same number."""
    band = rec._ttm_growth_band(_quarters([100.0] * 6))
    assert band == pytest.approx(0.0, abs=1e-9)


def test_the_band_is_the_adjacent_window_spread():
    """NVDA's real quarters, newest first, in billions."""
    band = rec._ttm_growth_band(
        _quarters([96.221, 81.615, 57.006, 46.743, 44.062, 35.082]))
    # TTM 281.585 against 229.426 one quarter back.
    assert band == pytest.approx(0.2273, abs=0.002)


def test_too_few_quarters_gives_no_band():
    """Five are needed for two adjacent windows. Fewer must not loosen
    anything -- a missing measurement is not licence to accept more drift."""
    assert rec._ttm_growth_band(_quarters([10.0, 11.0, 12.0])) is None
    assert rec._ttm_growth_band([]) is None


def test_a_shrinking_filer_gets_a_band_too():
    """Direction does not matter; the size of the step does."""
    band = rec._ttm_growth_band(_quarters([50.0, 60.0, 70.0, 80.0, 90.0]))
    assert band > 0.1


def test_a_zero_window_cannot_divide():
    """A filer with no revenue in the older window must not raise."""
    assert rec._ttm_growth_band(_quarters([5.0, 0.0, 0.0, 0.0, 0.0])) is None
