"""One filing walk, many concepts -- and a wall-clock bound on it.

Every concept-chain tool in this package reads a *set* of concepts from the
same handful of filings. `get_working_capital_trends` asks for 19,
`get_operating_leases` for 18, `get_sbc_series` for 8, `get_annual_revenue`
for up to 7. Each of them evaluates the whole chain rather than stopping at
the first hit, for the reason each one's own docstring gives, and that is not
the thing to change.

What was expensive is that each concept went through its own
`fetch_concept_series`, and that walks the filings from scratch: a fresh
`Company`, a fresh filing list, and a fresh `filing.xbrl()` per filing. The
parse is the smaller half of the cost. edgartools enriches every fact in a
filing on the first query against an XBRL object and memoises the result *on
that object*, so a new object per concept rebuilds the whole fact table --
for a bank's 10-K that dominates everything else.

Measured live, before this module existed:

    get_working_capital_trends("JPM")   36 parses of 2 filings   189s
    get_sbc_series("GS")                40 parses of 5 filings    67s
    get_debt_maturity_schedule("GS")    12 parses of 1 filing     66s

`sec_series.concept_point` was written for exactly this caller and says so:
parse the filing once, read every concept out of the one parsed object. This
module is the shared plumbing that does it, so the tools that need it do not
each grow their own copy.

The caller passes its own `fetch_concept_series` binding in. That name is the
seam the tests replace -- 29 of them in `test_earnings_quality.py` alone -- so
the walk has to be able to see that it has been replaced and stand down.
Reading the caller's module global is what makes that possible: a walk that
consulted `sec_series.fetch_concept_series` instead would ignore the
replacement and quietly talk to EDGAR from inside a unit test.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Tuple

from . import sec_series
from .sec_series import NotCovered


class ToolTimeout(Exception):
    """One call ran past its wall-clock budget.

    Separate from every other failure here because it is the only one that
    says nothing whatsoever about the filer. It must never be reported as a
    coverage gap, and it must never come back attached to a partial answer.
    """


# A call that outlives its client helps nobody: an MCP client has usually given
# up long before, so the work is spent producing an answer nothing is waiting
# for -- and against the SEC it is a sustained request rate with no reader.
DEFAULT_BUDGET_SECONDS = 120.0


def budget_seconds() -> float:
    """Wall-clock budget for one call, from NEMO_SEC_TOOL_BUDGET_S.

    An explicit `0` removes the bound, for an operator who genuinely wants an
    unbounded sweep and knows they are waiting for it. Anything unparseable or
    negative falls back to the default rather than being read as "no limit":
    a typo in a tuning variable must not silently restore the seven-minute
    call this bound exists to stop.
    """
    raw = os.environ.get("NEMO_SEC_TOOL_BUDGET_S", "").strip()
    if not raw:
        return DEFAULT_BUDGET_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_BUDGET_SECONDS
    if value < 0:
        return DEFAULT_BUDGET_SECONDS
    return value


class Deadline:
    """A wall-clock bound, checked between units of work.

    Checked between filings and between concepts rather than interrupting one
    in flight: a parse already running is left to finish. The overshoot is
    therefore bounded by a single filing parse rather than by the whole walk,
    which is the honest trade -- cancelling a parse would need a thread nobody
    can join, and a half-parsed filing is not a result.
    """

    def __init__(self, seconds: float, label: str):
        self._budget = seconds
        self._label = label
        self._started = time.monotonic()
        self.filings_parsed = 0
        self.concepts_read = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def check(self, doing: str) -> None:
        if self._budget <= 0:
            return
        elapsed = self.elapsed
        if elapsed <= self._budget:
            return
        # Phrased to say only what the fetch did, never what the filer
        # discloses -- the rule test_outage_is_not_a_finding enforces. A chain
        # abandoned to the clock was not evaluated, and an empty result here
        # would read as a coverage gap the filings were never asked about.
        raise ToolTimeout(
            f"{self._label} exceeded its {self._budget:g}s budget: still "
            f"{doing} after {elapsed:.1f}s, having parsed "
            f"{self.filings_parsed} filing(s) and read "
            f"{self.concepts_read} concept(s). No partial result is returned: "
            f"the concept chains it never reached would be indistinguishable "
            f"from empty ones once written into the output. Raise "
            f"NEMO_SEC_TOOL_BUDGET_S (or set it to 0 for no bound) to allow "
            f"longer.")


class SharedFilings:
    """Filings for one ticker, parsed once and read by every concept.

    Keyed two ways on purpose. The filing *list* depends on (form, limit), so
    that is what caches a list. The parsed XBRL is keyed by accession, so
    `get_operating_leases` -- which asks for `limit` filings for the balance
    concepts and 1 for each maturity bucket -- parses the newest filing once
    and not twice. `get_debt_maturity_schedule` is the same shape with twelve
    concepts against one filing.
    """

    def __init__(self, ticker: str, deadline: Deadline):
        self._ticker = ticker
        self._deadline = deadline
        self._lists: Dict[Tuple[str, int], List[Tuple[Any, Any]]] = {}
        self._parsed: Dict[str, Any] = {}

    def _walk(self, form: str, limit: int) -> List[Tuple[Any, Any]]:
        """(filing, parsed xbrl) for each filing, parsed at most once."""
        cached = self._lists.get((form, limit))
        if cached is not None:
            return cached

        sec_series._require_identity()
        company = sec_series.Company(self._ticker)
        # amendments=False for the reason fetch_concept_series gives: a 10-K/A
        # carrying only Part III takes a slot in the walk and hides the real
        # 10-K behind it.
        filings = company.get_filings(
            form=form, amendments=False).head(limit)

        walked: List[Tuple[Any, Any]] = []
        for filing in filings:
            self._deadline.check(
                f"parsing {self._ticker}'s {form} filings")
            accession = str(getattr(filing, "accession_no", "") or "")
            if accession and accession in self._parsed:
                walked.append((filing, self._parsed[accession]))
                continue
            sec_series._throttle()
            try:
                xbrl = filing.xbrl()
            except Exception:
                # One unparseable filing must not sink the series -- the same
                # rule fetch_concept_series applies, applied once per filing
                # here instead of once per filing per concept.
                continue
            if xbrl is None:
                continue
            if accession:
                self._parsed[accession] = xbrl
            self._deadline.filings_parsed += 1
            walked.append((filing, xbrl))

        self._lists[(form, limit)] = walked
        return walked

    def series(self, concept: str, form: str, limit: int) -> List[Any]:
        """Every fact for one concept, across the filings already walked.

        Raises NotCovered exactly where `fetch_concept_series` does, so every
        caller keeps its "swallow NotCovered, propagate everything else"
        contract unchanged.
        """
        points: List[Any] = []
        for filing, xbrl in self._walk(form, limit):
            self._deadline.check(
                f"reading {concept} from {self._ticker}'s {form} filings")
            try:
                point = sec_series.concept_point(
                    xbrl, concept,
                    filing_date=str(filing.filing_date),
                    form=str(filing.form),
                    accession=str(getattr(filing, "accession_no", "")))
            except Exception:
                continue
            self._deadline.concepts_read += 1
            if point is not None:
                points.append(point)

        if not points:
            raise NotCovered(
                f"{self._ticker}: concept {concept!r} found in none of the "
                f"last {limit} {form} filings")
        return points


# Per-thread, because the MCP server hands each tool call to its own worker via
# asyncio.to_thread. Cleared in a finally: pool threads are reused, and a walk
# left behind would serve one ticker's filings to the next ticker's call.
ACTIVE = threading.local()


@contextmanager
def shared_filings(ticker: str, deadline: Deadline,
                   fetch: Callable[..., Any]) -> Iterator[None]:
    """Share one filing walk across every concept read inside this block.

    `fetch` is the calling module's current `fetch_concept_series` binding.
    Skipped when that name has been replaced. The tests swap it for a stub
    serving canned FilingPoints, and several of them assert what happens when
    it raises; honouring the replacement matters more than the speedup, and
    there are no real filings to share in that case anyway.

    Nested blocks are supported by saving and restoring, not by refusing: a
    tool that opens a walk and then calls another tool that opens its own must
    end up with the outer walk still active, or the outer call's remaining
    concepts would silently re-fetch.
    """
    if fetch is not sec_series.fetch_concept_series:
        yield
        return
    previous = getattr(ACTIVE, "walk", None)
    ACTIVE.walk = SharedFilings(ticker, deadline)
    try:
        yield
    finally:
        ACTIVE.walk = previous


def concept_series(ticker: str, concept: str, form: str, limit: int,
                   fetch: Callable[..., Any]) -> List[Any]:
    """One concept's series, reusing filings already parsed for this call.

    With no shared walk active this is `fetch` verbatim, which is what keeps
    `fetch_concept_series` the seam the tests replace -- and what keeps the
    chain helpers callable on their own, as several tests call them.

    The replacement check is repeated here rather than left to
    `shared_filings` alone. The two are equivalent for a tool called on its
    own, and they differ only for a tool called from inside another tool's
    walk: there the outer walk is active while the inner module's seam has
    been replaced, and reading the walk would quietly ignore the replacement.
    Whoever replaced the name wins, wherever the call came from.
    """
    walk = getattr(ACTIVE, "walk", None)
    if walk is None or fetch is not sec_series.fetch_concept_series:
        return fetch(ticker, concept, form=form, limit=limit)
    return walk.series(concept, form, limit)
