"""The pool must outwait the attempts it is waiting on.

Greenhouse is tried twice: `?content=true`, which carries the departments,
then the plain listing as a fallback. Both run inside one stage-1 pool
alongside Lever. When the pool's own wait was shorter than the two attempts
it supervises, a board that was merely slow was reported as "no public job
board answered" -- the failure this code exists to state truthfully, given
for a provider that was about to answer.

That was a 12s pool over an 18s worst case. Nothing in the code said the
three numbers were related, so the next person to tune one had no way to
know. This is that statement.
"""
from tools.altdata_server import server


def test_the_pool_outwaits_both_greenhouse_attempts():
    attempts = (server._GREENHOUSE_CONTENT_TIMEOUT_S
                + server._GREENHOUSE_LISTING_TIMEOUT_S)
    assert attempts <= server._ATS_POOL_TIMEOUT_S, (
        f"Greenhouse can spend {attempts}s across its two attempts but the "
        f"pool gives up at {server._ATS_POOL_TIMEOUT_S}s, so a slow board is "
        f"reported as no provider at all")


def test_the_budget_leaves_no_dead_wait():
    """A pool far wider than its attempts stalls every caller for nothing."""
    attempts = (server._GREENHOUSE_CONTENT_TIMEOUT_S
                + server._GREENHOUSE_LISTING_TIMEOUT_S)
    assert server._ATS_POOL_TIMEOUT_S - attempts <= 5, (
        "the pool waits well past the point either provider can still answer")
