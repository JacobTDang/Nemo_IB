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


_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class Secret:
  """A credential that renders as a placeholder instead of as itself.

  pytest prints a frame's arguments at the head of every traceback entry, and
  every local under --showlocals, so a key bound to a name is written to
  stdout by the first test that fails anywhere below it. This key is worse
  placed than most: it travels as a *query parameter*, so it is in the URL
  aiohttp builds, in the error text aiohttp renders from that URL, and from
  there in the `error` field this client returns to its MCP caller -- which
  leaves the process entirely rather than merely reaching a log.

  Mirrored from agent/openrouter_template.py (issue #17) rather than imported
  from it. Nothing under tools/news_agregator imports from agent/ today, and
  agent.openrouter_template is the LLM layer: importing it here would put
  openai, ollama and httpx into a data-source image that will never run any of
  them, which is the coupling testing/test_agent_package_boundary.py exists to
  prevent. See that file for the full reasoning behind the type.
  """
  __slots__ = ("_value",)

  PLACEHOLDER = "<redacted>"

  def __init__(self, value: str = ""):
    self._value = value or ""

  def reveal(self) -> str:
    """The raw credential.

    Call this at the point of use -- inside the request call -- and never bind
    the result to a name, or the value is back in a frame.
    """
    return self._value

  def scrub(self, text: str) -> str:
    """`text` with the credential replaced by the placeholder.

    Provider and transport error text is returned to the caller, so this runs
    before that text is put in an `error` field.
    """
    if not self._value:
      return text
    return text.replace(self._value, self.PLACEHOLDER)

  def __repr__(self) -> str:
    return self.PLACEHOLDER

  __str__ = __repr__

  def __bool__(self) -> bool:
    return bool(self._value)


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
