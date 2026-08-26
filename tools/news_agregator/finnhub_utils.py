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


_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def get_api_key() -> str:
  """Load FINNHUB_API_KEY from .env file."""
  load_dotenv(dotenv_path=_DOTENV_PATH)
  key = os.getenv("FINNHUB_API_KEY")
  if not key:
    raise RuntimeError("FINNHUB_API_KEY not found in environment. Add it to .env")
  return key


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
    params["token"] = self._api_key
    url = f"{self.BASE_URL}{endpoint}"

    session = await self._get_session()

    for attempt in range(2):
      await self._rate_limiter.acquire()
      try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
          if resp.status == 429:
            if attempt == 0:
              await asyncio.sleep(2)
              continue
            return {"error": f"Rate limited (429) after retry"}
          if resp.status != 200:
            text = await resp.text()
            return {"error": f"HTTP {resp.status}: {text[:200]}"}
          return await resp.json()
      except asyncio.TimeoutError:
        return {"error": "Request timed out (15s)"}
      except aiohttp.ClientError as e:
        return {"error": f"HTTP client error: {str(e)}"}

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
