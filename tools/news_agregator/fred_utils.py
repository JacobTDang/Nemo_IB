"""
FRED API utilities -- HTTP client with rate limiting and response envelope.
Federal Reserve Economic Data: interest rates, inflation, GDP, employment, yield curve.
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
  """Load FRED_API_KEY from .env file, wrapped so it cannot be rendered.

  Building the Secret in the same expression that reads the environment is
  deliberate: an intermediate `key = os.getenv(...)` would put the raw value
  in this frame, and returning a bare `str` would put it in every caller's
  frame as well.
  """
  load_dotenv(dotenv_path=_DOTENV_PATH)
  credential = Secret(os.getenv("FRED_API_KEY") or "")
  if not credential:
    raise RuntimeError("FRED_API_KEY not found in environment. Add it to .env")
  return credential


class RateLimiter:
  """Sliding window rate limiter: max_calls within window_seconds.

  FRED API: ~120 requests per minute.
  """
  def __init__(self, max_calls: int = 120, window_seconds: float = 60.0):
    self.max_calls = max_calls
    self.window_seconds = window_seconds
    self._timestamps: list[float] = []
    self._lock = asyncio.Lock()

  async def acquire(self):
    """Wait until a call slot is available, then record the call."""
    async with self._lock:
      now = time.monotonic()
      cutoff = now - self.window_seconds
      self._timestamps = [t for t in self._timestamps if t > cutoff]

      if len(self._timestamps) >= self.max_calls:
        sleep_for = self._timestamps[0] - cutoff
        await asyncio.sleep(sleep_for)
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

      self._timestamps.append(time.monotonic())


class FredClient:
  """Async HTTP client for FRED API with rate limiting.

  Creates an aiohttp session lazily on first request.
  Single retry on 429 with 2s backoff.
  """
  BASE_URL = "https://api.stlouisfed.org/fred"

  def __init__(self):
    self._api_key = get_api_key()
    self._session: Optional[aiohttp.ClientSession] = None
    self._rate_limiter = RateLimiter()

  async def _get_session(self) -> aiohttp.ClientSession:
    if self._session is None or self._session.closed:
      self._session = aiohttp.ClientSession()
    return self._session

  async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Rate-limited GET request to FRED API.

    Args:
      endpoint: API path (e.g. '/series/observations')
      params: Query parameters (api_key and file_type appended automatically)

    Returns:
      Parsed JSON dict, or {"error": "..."} on failure
    """
    params = dict(params) if params else {}
    params["file_type"] = "json"
    url = f"{self.BASE_URL}{endpoint}"

    session = await self._get_session()

    for attempt in range(2):
      await self._rate_limiter.acquire()
      try:
        # The credential is revealed into the call itself and never bound
        # to a name: a `params` dict holding it is a local, and a local is
        # what a rendered traceback prints.
        async with session.get(url, params={**params, "api_key": self._api_key.reveal()},
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
          if resp.status == 429:
            if attempt == 0:
              await asyncio.sleep(2)
              continue
            return {"error": "Rate limited (429) after retry"}
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


def build_envelope(
  data: Any,
  context_label: str,
  tool_name: str,
  api_calls_made: int = 1,
  errors: list = None
) -> Dict[str, Any]:
  """Wrap a FRED response in a standardized envelope.

  Every tool result goes through this so downstream consumers
  (execution engine, analysis agent) see a consistent shape.
  """
  return {
    "domain": "macro",
    "context": context_label,
    "tool": tool_name,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "data": data,
    "metadata": {
      "api_calls_made": api_calls_made,
      "errors": errors or [],
    }
  }
