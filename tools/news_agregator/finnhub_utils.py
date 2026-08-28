"""
Finnhub API utilities -- HTTP client with rate limiting and response envelope.
this is a deterministic data runner.
"""
import os
import time
import asyncio
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# This key travels as a *query parameter*, so it reaches the URL aiohttp builds,
# the error text aiohttp renders from that URL, and from there the `error` field
# this client returns to its MCP caller -- which leaves the process rather than
# merely reaching a log. Secret keeps it out of all three, and common/secret.py
# imports nothing, so importing it here crosses no package boundary.
from common.secret import Secret


_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def get_api_key() -> Secret:
  """Load FINNHUB_API_KEY from .env file, wrapped so it cannot be rendered.

  Building the Secret in the same expression that reads the environment is
  deliberate: an intermediate `key = os.getenv(...)` would put the raw value
  in this frame, and returning a bare `str` would put it in every caller's
  frame as well.
  """
  load_dotenv(dotenv_path=_DOTENV_PATH)
  credential = Secret(os.getenv("FINNHUB_API_KEY") or "")
  if not credential:
    raise RuntimeError("FINNHUB_API_KEY not found in environment. Add it to .env")
  return credential


class RateLimiter:
  """Sliding window rate limiter: max_calls within window_seconds.

  Finnhub free tier: 60 calls / 60 seconds.
  """
  def __init__(self, max_calls: int = 60, window_seconds: float = 60.0):
    self.max_calls = max_calls
    self.window_seconds = window_seconds
    self._timestamps: list[float] = []
    self._lock = asyncio.Lock()

  async def acquire(self):
    """Wait until a call slot is available, then record the call."""
    async with self._lock:
      now = time.monotonic()
      # Evict timestamps outside the window
      cutoff = now - self.window_seconds
      self._timestamps = [t for t in self._timestamps if t > cutoff]

      if len(self._timestamps) >= self.max_calls:
        # Sleep until the oldest call exits the window
        sleep_for = self._timestamps[0] - cutoff
        await asyncio.sleep(sleep_for)
        # Re-evict after sleep
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

      self._timestamps.append(time.monotonic())


class FinnhubClient:
  """Async HTTP client for Finnhub API with rate limiting.

  Creates an aiohttp session lazily on first request.
  Single retry on 429 with 2s backoff.
  """
  BASE_URL = "https://finnhub.io/api/v1"

  def __init__(self):
    self._api_key = get_api_key()
    self._session: Optional[aiohttp.ClientSession] = None
    self._rate_limiter = RateLimiter()

  async def _get_session(self) -> aiohttp.ClientSession:
    if self._session is None or self._session.closed:
      self._session = aiohttp.ClientSession()
    return self._session

  async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Rate-limited GET request to Finnhub API.

    Args:
      endpoint: API path (e.g. '/company-news')
      params: Query parameters (api key appended automatically)

    Returns:
      Parsed JSON dict, or {"error": "..."} on failure
    """
    params = dict(params) if params else {}
    url = f"{self.BASE_URL}{endpoint}"

    session = await self._get_session()

    for attempt in range(2):
      await self._rate_limiter.acquire()
      try:
        # The credential is revealed into the call itself and never bound
        # to a name: a `params` dict holding it is a local, and a local is
        # what a rendered traceback prints.
        async with session.get(url, params={**params, "token": self._api_key.reveal()},
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
          if resp.status == 429:
            if attempt == 0:
              await asyncio.sleep(2)
              continue
            return {"error": f"Rate limited (429) after retry"}
          if resp.status != 200:
            text = await resp.text()
            return {"error": self._api_key.scrub(f"HTTP {resp.status}: {text[:200]}")}
          return await resp.json()
      except asyncio.TimeoutError:
        return {"error": "Request timed out (15s)"}
      except aiohttp.ClientError as e:
        # aiohttp renders the request URL into several of its errors, and the
        # credential is a query parameter of that URL. Scrubbed before it goes
        # back to the caller, which is further than a log line travels.
        return {"error": self._api_key.scrub(f"HTTP client error: {str(e)}")}

    return {"error": "Unexpected: exhausted retries"}

  async def close(self):
    """Close the underlying HTTP session."""
    if self._session and not self._session.closed:
      await self._session.close()
      self._session = None


# ---------------------------------------------------------------- denomination

# Currency does not change, so one lookup per symbol per process is plenty and
# the alternative -- a /stock/profile2 call on every metric and earnings
# request -- would spend a third of a 60-call minute restating a constant.
_DENOMINATION_CACHE: Dict[str, Dict[str, Any]] = {}


def denomination_shape(requested: str, *, finnhub_symbol=None, currency=None,
                       shares_outstanding_millions=None, error=None) -> Dict[str, Any]:
  """One shape for "what are these numbers measured in", filled or empty."""
  return {
    "requested_symbol": requested,
    "finnhub_symbol": finnhub_symbol,
    "currency": currency,
    "shares_outstanding_millions": shares_outstanding_millions,
    "error": error,
  }


async def get_denomination(client, ticker: str) -> Dict[str, Any]:
  """Which listing Finnhub answers about for `ticker`, and in what currency.

  Finnhub resolves an ADR to the company's *local* listing and reports there:
  TSM answers as 2330.TW in TWD on 25,932.37m ordinary shares, against the
  ADR's 5,186m; SAP answers as SAP.DE in EUR; SONY as 6758.T in JPY. Nothing
  in `/stock/metric` or `/stock/earnings` carries the currency, so an EPS of
  27.25 and an EPS of 4.46 for the same company in the same quarter both
  arrive bare and 6.11x apart.

  The symbol alone is not enough to tell, which is why this asks the provider
  rather than reading a suffix: BABA stays 'BABA' on the NYSE and is still
  reported in CNY. `/stock/profile2` is the only Finnhub endpoint that states
  the currency outright.

  A failed lookup returns `currency: None` with the provider's words in
  `error`. It never falls back to USD -- the securities whose metadata is
  thinnest are exactly the ones a default would be wrong about, and a wrong
  currency is worse than an absent one because it is actionable.
  """
  key = (ticker or "").strip().upper()
  if not key:
    return denomination_shape(ticker, error="no symbol given")
  if key in _DENOMINATION_CACHE:
    return dict(_DENOMINATION_CACHE[key])

  profile = await client.get("/stock/profile2", {"symbol": key})

  if not isinstance(profile, dict):
    return denomination_shape(
      key, error=f"Finnhub: unrecognized /stock/profile2 response "
                 f"({type(profile).__name__})")
  if profile.get("error"):
    return denomination_shape(key, error=f"Finnhub: {profile['error']}")

  currency = profile.get("currency") or None
  resolved = profile.get("ticker") or None
  shares = profile.get("shareOutstanding")
  if not currency:
    # An empty profile is Finnhub's answer for an unknown symbol, for one
    # outside the plan, and for a covered company it holds no profile for.
    # Uncached: any of those can change without the symbol changing.
    return denomination_shape(
      key, finnhub_symbol=resolved, shares_outstanding_millions=shares,
      error="Finnhub /stock/profile2 returned no currency for this symbol")

  found = denomination_shape(key, finnhub_symbol=resolved, currency=currency,
                             shares_outstanding_millions=shares)
  _DENOMINATION_CACHE[key] = dict(found)
  return found


# Keys that restate the request or describe the lookup rather than answer it.
# `symbol` and `metricType` are Finnhub echoing the arguments back. The news
# page's window and counts are ours, and are the same kind of thing: a payload
# holding only "you asked about these eight days and we returned 0 of 0
# articles" is still a payload with nothing in it. Without this the counts
# added to close the silent-cap defect would have permanently satisfied
# `_has_content`, and the not_covered label would never fire for news again.
# `total_months` is insider sentiment's equivalent of `total_articles`: a
# count of the monthly MSPR rows that fell inside the requested window, which
# is zero exactly when the window could not be filled. Without it here, "we
# looked at your window and found 0 months" would read as content and the
# not_covered label would stop firing for that tool.
# `denomination` and `period_label` join them one step removed: they say how
# to READ an answer -- what currency, what scale, which listing, what a date
# field means -- and a payload carrying nothing but "we could not establish
# the currency of the nothing we found" is still a payload with nothing in it.
# Without this, adding them would have permanently satisfied `_has_content`
# and retired the not_covered label for basic financials and earnings
# surprises.
_LOOKUP_ECHO_KEYS = frozenset({
  "symbol", "metricType",
  "window_requested", "window_returned", "total_articles", "total_months",
  "returned", "truncated",
  "denomination", "period_label", "duplicate_fiscal_periods",
})


def _has_content(node: Any) -> bool:
  """Does this payload contain anything a caller could act on?

  Nested because Finnhub answers an unknown symbol with structure but no
  content -- `{"metric": {}, "series": {}, "symbol": "ZZZZ"}` is shaped like a
  result and holds none. A zero counts as content: a metric that is genuinely
  zero is an answer, unless it is a count of what we did rather than a
  measurement of the company.
  """
  if isinstance(node, dict):
    return any(_has_content(value) for key, value in node.items()
               if key not in _LOOKUP_ECHO_KEYS)
  if isinstance(node, (list, tuple)):
    return any(_has_content(value) for value in node)
  if isinstance(node, str):
    return bool(node.strip())
  return node is not None


def build_envelope(
  data: Any,
  ticker: str,
  tool_name: str,
  api_calls_made: int = 1,
  errors: list = None
) -> Dict[str, Any]:
  """Wrap a Finnhub response in a standardized envelope.

  Every tool result goes through this so downstream consumers
  (execution engine, analysis agent) see a consistent shape.

  An empty payload is labelled rather than left to be inferred. Asked about a
  symbol that does not exist, these tools answered `success: true, data: {}`,
  and a successful response with no content reads as "this company has no
  profile" -- a claim about the company, made from our own empty hand.

  The label stops short of saying why. Finnhub returns the same empty body for
  an unknown symbol, for one outside the plan's entitlement, and for a covered
  company with nothing to report in the window asked for -- verified:
  `/stock/insider-transactions` answers `{"data": [], "symbol": X}` for SHOP,
  NVO and SAP exactly as it does for ZZZZNOTREAL. The server already reasons
  this way for forward estimates, where a 403 is kept in the provider's own
  words rather than flattened to "no data".

  This only fires on a payload with nothing in it, which is why a tool that
  summarises its rows has to hand over the empty rows rather than a summary
  computed over them: `{"quarters": [], "beat_count": 0}` has content, and the
  content is a claim nothing measured.

  `success` is untouched: a news window with no articles is a real empty.
  """
  envelope = {
    "domain": "market_intel",
    "ticker": ticker,
    "tool": tool_name,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "data": data,
    "metadata": {
      "api_calls_made": api_calls_made,
      "errors": errors or [],
    }
  }

  if not _has_content(data):
    envelope["coverage"] = "not_covered"
    envelope["warnings"] = [{
      "code": "finnhub_returned_nothing",
      "message": (
        f"Finnhub returned no content for {ticker!r}. That happens for a "
        f"symbol it does not carry, for one outside this plan's entitlement, "
        f"and for a covered company with nothing to report in the window "
        f"asked for, and the response does not distinguish them, so this is "
        f"not evidence about what the company discloses."),
    }]

  return envelope


# ------------------------------------------------------- earnings integrity
#
# A count is a fact about the rows Finnhub returned. A rate and an average are
# claims about the company, and they are only defined when the rows they run
# over form one continuous series on one share basis.
#
# CREG, live 2026-08-26: four rows spanning 2011-06-30 to 2026-03-31, with
# actual_eps of 1600 and 700 (pre-reverse-split) beside -0.03 and -0.02, and
# two rows Finnhub never priced. It reported avg_surprise_pct 49.53 -- both
# 2011 rows and nothing since -- and beat_rate_pct 50.0, which is 2 beats over
# 4 rows when only 2 of those rows could beat or miss at all.
#
# The rows are left exactly as Finnhub sent them, following the fix already
# landed for TGT's duplicate fiscal periods: which row is wrong is Finnhub's
# to say, and dropping one would silently change counts other callers read.
# What is withheld is the derived statistics, because a hole and a share-basis
# break leave them with no referent.

# A per-share figure 1000x another in the same array is a corporate action,
# not a business result. Deliberately far above any organic swing -- a company
# going from a $0.01 quarter to a $3.00 quarter is 300x and real -- and far
# below CREG's 80,000x.
_SHARE_BASIS_BREAK_RATIO = 1000.0

# A quarterly filer's newest bucket sits at or just behind the current
# calendar quarter. `period` is a bucket that can lead the fiscal close by
# weeks, so one quarter of slack is normal and two is tolerable; three or more
# means the provider has not published prints the company has already made.
_STALE_CALENDAR_QUARTERS = 3


def _quarter_index(year, quarter):
  """A single ordinal for a fiscal (year, quarter), or None."""
  if not isinstance(year, int) or not isinstance(quarter, int):
    return None
  if not 1 <= quarter <= 4:
    return None
  return year * 4 + (quarter - 1)


def _indexed(quarters):
  """(index, entry) for every row carrying a usable fiscal identity, newest first."""
  rows = []
  for entry in quarters or []:
    if not isinstance(entry, dict):
      continue
    index = _quarter_index(entry.get("year"), entry.get("quarter"))
    if index is not None:
      rows.append((index, entry))
  return sorted(rows, key=lambda pair: pair[0], reverse=True)


def fiscal_period_gaps(quarters):
  """Quarters missing between consecutive rows of the returned series.

  A repeat is not a gap: TGT files fiscal 2027 Q2 under two different `period`
  buckets, and `_duplicate_fiscal_periods` already declares that. A hole is a
  step of more than one quarter between adjacent rows.
  """
  rows = _indexed(quarters)
  gaps = []
  for (newer_index, newer), (older_index, older) in zip(rows, rows[1:]):
    step = newer_index - older_index
    if step > 1:
      gaps.append({
        "between": [[older.get("year"), older.get("quarter")],
                    [newer.get("year"), newer.get("quarter")]],
        "period_buckets": [older.get("period"), newer.get("period")],
        "quarters_missing": step - 1,
      })
  return gaps


def share_basis_discontinuity(quarters):
  """Is this array's actual_eps on more than one share basis?

  Returns the measurement, or None. A reverse split between two rows makes
  every cross-row statistic undefined, and Finnhub restates neither side.
  """
  values = []
  for entry in quarters or []:
    if not isinstance(entry, dict):
      continue
    actual = entry.get("actual_eps")
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
      continue
    if abs(float(actual)) > 0:
      values.append((abs(float(actual)), entry))
  if len(values) < 2:
    return None
  smallest, small_row = min(values, key=lambda pair: pair[0])
  largest, large_row = max(values, key=lambda pair: pair[0])
  ratio = largest / smallest
  if ratio < _SHARE_BASIS_BREAK_RATIO:
    return None
  return {
    "max_abs_actual_eps": large_row.get("actual_eps"),
    "min_abs_actual_eps": small_row.get("actual_eps"),
    "ratio": round(ratio, 1),
    "fiscal": [[large_row.get("year"), large_row.get("quarter")],
               [small_row.get("year"), small_row.get("quarter")]],
    "threshold": _SHARE_BASIS_BREAK_RATIO,
  }


def _calendar_quarter_index(iso_date):
  """The quarter ordinal of a YYYY-MM-DD string, or None."""
  if not isinstance(iso_date, str) or len(iso_date) < 7:
    return None
  try:
    year = int(iso_date[0:4])
    month = int(iso_date[5:7])
  except ValueError:
    return None
  if not 1 <= month <= 12:
    return None
  return year * 4 + (month - 1) // 3


def summarize_earnings_surprises(quarters, today=None):
  """Counts, rates and averages over a condensed surprise table.

  Counts are always reported: they describe the rows in hand. Rates and
  averages are reported only when the series is continuous and on one share
  basis, and each carries the population it was computed over -- the defect
  behind `beat_rate_pct: 50.0` was a denominator (`total_periods`) that
  counted two rows Finnhub never priced and which therefore could neither
  beat nor miss.

  `avg_surprise_pct` averages the `surprise_pct` values as published in
  data.quarters, which are rounded to 2dp. The figure it replaces averaged
  Finnhub's unrounded `surprisePercent`, so a response can move by 0.01 --
  TSM went 7.14 -> 7.13. Deliberate: the average now reconciles against the
  column a reader can add up, and an average that disagrees with its own
  table by a cent is a question nobody can answer from the response.
  """
  rows = [entry for entry in (quarters or []) if isinstance(entry, dict)]
  empty = {
    "beat_count": None, "miss_count": None, "inline_count": None,
    "total_periods": None, "graded_periods": None, "ungraded_periods": None,
    "beat_rate_pct": None, "beat_rate_basis": None,
    "beat_rate_pct_unavailable": None,
    "avg_surprise_pct": None, "avg_surprise_pct_basis": None,
    "avg_surprise_pct_unavailable": None,
    "fiscal_period_gaps": [], "share_basis_discontinuity": None,
    "latest_fiscal_period": None, "calendar_quarters_behind": None,
    "history_is_stale": None,
  }
  if not rows:
    # test_no_data_is_not_a_verdict: `beat_count: 0` beside `quarters: []`
    # reads as a fact about the filer, asserted from an empty hand.
    return empty

  graded = [entry for entry in rows if entry.get("surprise_pct") is not None]
  beat_count = sum(1 for entry in graded if entry["surprise_pct"] > 0)
  miss_count = sum(1 for entry in graded if entry["surprise_pct"] < 0)
  inline_count = len(graded) - beat_count - miss_count

  gaps = fiscal_period_gaps(rows)
  basis_break = share_basis_discontinuity(rows)

  blockers = []
  if gaps:
    worst = max(gaps, key=lambda gap: gap["quarters_missing"])
    blockers.append(
      f"the series is not continuous -- {worst['quarters_missing']} fiscal "
      f"quarters are missing between {worst['between'][0]} and "
      f"{worst['between'][1]}, so a figure averaged across these rows "
      f"describes no period a reader could name")
  if basis_break:
    blockers.append(
      f"actual_eps runs from {basis_break['min_abs_actual_eps']} to "
      f"{basis_break['max_abs_actual_eps']} ({basis_break['ratio']:g}x), which "
      f"is a corporate action rather than a business result -- these rows are "
      f"not on one share basis and Finnhub restates neither side")
  if not graded:
    blockers.append(
      "no row carries an estimate, so nothing in this response was graded "
      "against one")

  reason = None
  if blockers:
    reason = ("Withheld: " + "; and ".join(blockers) +
              ". The per-quarter rows are unchanged and are in data.quarters.")

  indexed = _indexed(rows)
  latest = None
  behind = None
  if indexed:
    _, newest = indexed[0]
    latest = {"year": newest.get("year"), "quarter": newest.get("quarter"),
              "period_bucket": newest.get("period")}
    newest_bucket = _calendar_quarter_index(newest.get("period"))
    now = _calendar_quarter_index(
      today or datetime.now(timezone.utc).date().isoformat())
    if newest_bucket is not None and now is not None:
      behind = max(0, now - newest_bucket)

  summary = dict(empty)
  summary.update({
    "beat_count": beat_count,
    "miss_count": miss_count,
    "inline_count": inline_count,
    "total_periods": len(rows),
    "graded_periods": len(graded),
    "ungraded_periods": len(rows) - len(graded),
    "fiscal_period_gaps": gaps,
    "share_basis_discontinuity": basis_break,
    "latest_fiscal_period": latest,
    "calendar_quarters_behind": behind,
    "history_is_stale": (None if behind is None
                         else behind >= _STALE_CALENDAR_QUARTERS),
  })

  if reason is not None:
    summary["avg_surprise_pct_unavailable"] = reason
    summary["beat_rate_pct_unavailable"] = reason
    return summary

  summary["beat_rate_pct"] = round(beat_count / len(graded) * 100, 1)
  summary["beat_rate_basis"] = (
    f"{beat_count} beats of {len(graded)} graded quarters. Rows with no "
    f"estimate are excluded from the denominator: they could neither beat "
    f"nor miss.")
  summary["avg_surprise_pct"] = round(
    sum(entry["surprise_pct"] for entry in graded) / len(graded), 2)
  graded_indexed = _indexed(graded)
  summary["avg_surprise_pct_basis"] = {
    "rows": len(graded),
    "of_total_periods": len(rows),
    "fiscal_first": [graded_indexed[-1][1].get("year"),
                     graded_indexed[-1][1].get("quarter")],
    "fiscal_last": [graded_indexed[0][1].get("year"),
                    graded_indexed[0][1].get("quarter")],
  }
  return summary
