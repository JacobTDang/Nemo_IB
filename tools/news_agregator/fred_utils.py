"""
FRED API utilities -- HTTP client with rate limiting and response envelope.
Federal Reserve Economic Data: interest rates, inflation, GDP, employment, yield curve.
"""
import os
import time
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def get_api_key() -> str:
  """Load FRED_API_KEY from .env file."""
  load_dotenv()
  key = os.getenv("FRED_API_KEY")
  if not key:
    raise RuntimeError("FRED_API_KEY not found in environment. Add it to .env")
  return key


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
    params["api_key"] = self._api_key
    params["file_type"] = "json"
    url = f"{self.BASE_URL}{endpoint}"

    session = await self._get_session()

    for attempt in range(2):
      await self._rate_limiter.acquire()
      try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
          if resp.status == 429:
            if attempt == 0:
              await asyncio.sleep(2)
              continue
            return {"error": "Rate limited (429) after retry"}
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
