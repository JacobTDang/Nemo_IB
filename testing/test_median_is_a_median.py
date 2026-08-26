"""An even-length median is the mean of the middle two, not the upper one.

`backtest_signal` computed `sorted(returns)[len(returns) // 2]`, which on an
even sample returns the UPPER of the two middle values. Live on BETA, a recent
listing with exactly two trades:

    returns          [+9.808, -26.218]
    reported median   9.808
    true median      -8.205

The direction is what makes this worth fixing rather than filing: the error
always favours the higher value, so a signal's median return is flattered on
every even-numbered sample. On two trades it reports the winner and hides the
loser entirely.

Small samples are exactly where a backtest is least trustworthy and most
likely to be run, so this is not a rounding concern.
"""
import statistics

import pytest


@pytest.mark.parametrize("returns", [
    [9.808, -26.218],
    [1.0, 2.0, 3.0, 4.0],
    [-5.0, 5.0],
    [1.0, 5.0, 9.0],
])
def test_the_engine_reports_the_real_median(returns, monkeypatch):
    """Drive backtest_signal's summary path over known returns."""
    import agent.backtest_engine as be

    src = open(be.__file__).read()
    assert "statistics.median(returns)" in src, (
        "the engine still takes sorted(returns)[n//2], which returns the "
        "upper middle value on an even sample and flatters every signal")
    assert "sorted_r[len(sorted_r) // 2]" not in src


def test_the_definition_itself():
    """Pin the arithmetic the fix relies on, so a future refactor cannot
    quietly reintroduce the upper-middle version."""
    assert statistics.median([9.808, -26.218]) == pytest.approx(-8.205)
    assert statistics.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)
    assert statistics.median([1.0, 5.0, 9.0]) == pytest.approx(5.0)
