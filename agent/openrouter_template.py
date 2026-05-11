from openai import OpenAI, APIError, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError, NotFoundError
import httpx
import os, sys, json, re, time
from threading import Lock

def _strip_non_ascii(text: str) -> str:
  """Remove non-ASCII characters from a streaming chunk before printing.
  Same logic as Financial_Analysis_Agent._strip_unicode_artifacts but applied
  per-chunk so the terminal output is clean even before the full response is assembled."""
  return re.sub(r'[^\x00-\x7F]+', '', text)
from dotenv import load_dotenv
try:
  from ollama import chat as ollama_chat
  _OLLAMA_AVAILABLE = True
except ImportError:
  _OLLAMA_AVAILABLE = False

# Fix Windows console encoding — always use 'replace' to handle emojis/unicode
if hasattr(sys.stdout, 'reconfigure'):
  sys.stdout.reconfigure(errors='replace')
  sys.stderr.reconfigure(errors='replace')


def _verify_model_alive(model_id: str, api_key: str, timeout: float = 10.0) -> bool:
  """Send a 1-token completion to check the model endpoint exists.

  Returns True if alive (or if the error is non-404 — auth, rate limit, timeout
  all indicate the model name itself is at least known). Returns False only on
  explicit 404 NotFoundError.
  """
  try:
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=timeout)
    client.chat.completions.create(
      model=model_id,
      messages=[{"role": "user", "content": "ping"}],
      max_tokens=1,
    )
    return True
  except NotFoundError:
    return False
  except Exception:
    return True  # rate limit, auth, etc. don't prove the model is dead


# ---------------------------------------------------------------------------
# Reasoning model pool
# ---------------------------------------------------------------------------
# Module-level state. All access goes through the helpers below.
_MODEL_POOL: list = []                  # alive models, in preference order
_MODEL_LAST_USED: dict = {}             # model -> unix timestamp last picked
_MODEL_DEMOTED_UNTIL: dict = {}         # model -> unix timestamp it's banned
_POOL_LOCK = Lock()
_DEMOTE_SECONDS = 90                    # how long a model is banned after a 429


def _build_reasoning_pool() -> list:
  """Ping each candidate, return the alive ones in preference order.

  Reads OPENROUTER_API_KEY. If unset, returns the ultimate fallback only.
  Honors PRIMARY_REASONING_MODEL env var as a top-priority override.
  """
  load_dotenv()
  api_key = os.getenv("OPENROUTER_API_KEY")
  ultimate_fallback = 'z-ai/glm-4.5-air:free'

  if not api_key:
    return [ultimate_fallback]

  candidates = [
    os.getenv("PRIMARY_REASONING_MODEL"),    # explicit override stays at top
    'deepseek/deepseek-chat-v3.1:free',
    'deepseek/deepseek-r1-distill-llama-70b:free',
    'qwen/qwq-32b-preview:free',
    'meta-llama/llama-3.3-70b-instruct:free',
    'z-ai/glm-4.5-air:free',
  ]
  alive = []
  seen = set()
  for c in candidates:
    if c and c not in seen and _verify_model_alive(c, api_key):
      alive.append(c)
      seen.add(c)
  if not alive:
    alive = [ultimate_fallback]
  print(f"[OpenRouter] Pool initialized with {len(alive)} models: {alive}",
        file=sys.stderr, flush=True)
  return alive


def _pick_next_model() -> str:
  """Return the least-recently-used non-demoted model from the pool."""
  with _POOL_LOCK:
    if not _MODEL_POOL:
      return 'z-ai/glm-4.5-air:free'
    now = time.time()
    eligible = [m for m in _MODEL_POOL if _MODEL_DEMOTED_UNTIL.get(m, 0) < now]
    if not eligible:
      # All demoted -- return the one that demotes soonest
      eligible = sorted(_MODEL_POOL, key=lambda m: _MODEL_DEMOTED_UNTIL.get(m, 0))[:1]
    pick = min(eligible, key=lambda m: _MODEL_LAST_USED.get(m, 0))
    _MODEL_LAST_USED[pick] = now
    return pick


def _demote_model(model: str, seconds: float = None) -> None:
  """Mark a model unhealthy for N seconds (called after 429 / connection error)."""
  if seconds is None:
    seconds = _DEMOTE_SECONDS
  with _POOL_LOCK:
    _MODEL_DEMOTED_UNTIL[model] = time.time() + seconds
    print(f"[OpenRouter] Demoted {model} for {seconds:.0f}s",
          file=sys.stderr, flush=True)


# Initialize pool at import; PRIMARY_REASONING_MODEL stays as a convenience alias
# pointing at the first-preference alive model (used by existing constructor defaults).
_MODEL_POOL = _build_reasoning_pool()
PRIMARY_REASONING_MODEL = _MODEL_POOL[0]


class OpenRouterModel:
  """
  OpenRouter API base class using the OpenAI-compatible SDK.
  Same interface as OllamaModel so agents can swap between backends.

  Subclasses set response_schema to a Pydantic BaseModel for structured JSON output.
  Uses OpenRouter's chat completions API with streaming and optional reasoning.
  """
  response_schema = None

  MAX_RETRIES = 5
  RETRY_BASE_DELAY = 1  # seconds — OpenRouter allows 120 req/min, no need to wait long
  CLIENT_TIMEOUT = 120.0  # 2 minutes — enough for streaming; drops are connection errors not timeouts
  FALLBACK_MODEL = 'z-ai/glm-4.5-air:free'
  OLLAMA_FALLBACK_MODEL = 'llama3.1:8b'
  MAX_OUTPUT_TOKENS = 2048  # Subclasses can override (e.g., verifier needs more room after thinking)
  REASONING_EFFORT = "low"  # Subclasses can set to None to disable reasoning (e.g., orchestrator just needs JSON)

  def __init__(self, model_name: str = None, api_key_env: str = "OPENROUTER_API_KEY"):
    load_dotenv()
    # Default to the verified primary reasoning model resolved at import time
    if model_name is None:
      model_name = PRIMARY_REASONING_MODEL
    # Try the requested env var first, then fall back to the main key.
    # This means a single OPENROUTER_API_KEY is always enough to run the system --
    # model-specific keys (OPENROUTER_NEMOTRON, OPENROUTER_GLM) are optional extras.
    api_key = os.getenv(api_key_env) or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
      raise ValueError(f"No API key found. Set OPENROUTER_API_KEY (or {api_key_env}) in your .env file.")
    self.client = OpenAI(
      api_key=api_key,
      base_url="https://openrouter.ai/api/v1",
      timeout=self.CLIENT_TIMEOUT
    )
    self.model_name = model_name
    self.conversatoin_history = []

    # Fallback client: prefer OPENROUTER_GLM, otherwise reuse the main key.
    # Reusing the main key is fine -- fallback only triggers if primary model fails.
    fallback_key = os.getenv("OPENROUTER_GLM") or api_key
    self.fallback_client = OpenAI(
      api_key=fallback_key,
      base_url="https://openrouter.ai/api/v1",
      timeout=self.CLIENT_TIMEOUT
    )

  @staticmethod
  def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.
    Handles both closed tags and unclosed tags (truncated mid-thought)."""
    # First: strip closed <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Second: strip unclosed <think> (model ran out of tokens mid-thought)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()

  def generate_response(self, prompt: str, system_prompt: str = "You are a professional Investment Banker from wallstreet. Never use emojis in your responses.", schema=None):
    active_schema = schema or self.response_schema

    # Build messages: system + history + current prompt
    messages = [{"role": "system", "content": system_prompt}]

    for msg in self.conversatoin_history:
      messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": prompt})

    # Build kwargs for the API call
    kwargs = {
      "model": self.model_name,
      "messages": messages,
      "stream": True,
      "max_tokens": self.MAX_OUTPUT_TOKENS,
    }

    # Build extra_body: provider routing + optional reasoning budget control
    extra_body = {
      "provider": {
        "allow_fallbacks": True,
        "sort": "throughput",
      },
    }
    if self.REASONING_EFFORT:
      extra_body["reasoning"] = {"effort": self.REASONING_EFFORT}
    kwargs["extra_body"] = extra_body

    # If a Pydantic schema is set, request structured JSON output
    if active_schema:
      json_schema = active_schema.model_json_schema()
      kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
          "name": active_schema.__name__,
          "strict": True,
          "schema": json_schema
        }
      }

    # Stream response with retry on transient errors
    assistant_response = self._stream_with_retry(kwargs)

    # Store full response in history (with thinking), but return cleaned version
    self.conversatoin_history.append({"role": "user", "content": prompt})
    self.conversatoin_history.append({"role": "assistant", "content": assistant_response})

    # Strip <think> blocks so downstream consumers get clean output
    return self._strip_thinking(assistant_response)

  def _stream_with_retry(self, kwargs: dict) -> str:
    """Stream a chat completion with exponential backoff retry on transient errors.
    Tries primary client first, then falls back to GLM-4.5-Air if retries fail.
    If the stream started producing content before dropping, skips remaining
    retries and goes straight to fallback (retrying will just repeat the drop)."""
    last_error = None
    got_partial = False  # True if we received any content before a drop

    # Try primary model
    for attempt in range(1, self.MAX_RETRIES + 1):
      try:
        assistant_response = ""
        thinking_started = False
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
          delta = chunk.choices[0].delta if chunk.choices else None
          if not delta:
            continue
          # Show R1 reasoning on stderr (visible but not part of response)
          reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
          if reasoning:
            if not thinking_started:
              print("\n[Thinking]", file=sys.stderr, flush=True)
              thinking_started = True
            print(reasoning, end='', flush=True, file=sys.stderr)
          # Capture actual output content
          if delta.content:
            if thinking_started:
              print("\n[Output]", file=sys.stderr, flush=True)
              thinking_started = False
            assistant_response += delta.content
            print(_strip_non_ascii(delta.content), end='', flush=True)
        return assistant_response

      except (AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError, APIError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        last_error = e
        error_type = type(e).__name__
        # 401 auth errors will never recover with retries -- skip straight to fallback
        if isinstance(e, AuthenticationError):
          print(f"\n[Auth error] {e}. Skipping retries, switching to fallback.", file=sys.stderr, flush=True)
          break
        # If we got thinking or content before the drop, retrying the same model
        # will likely produce the same result. Skip to fallback immediately.
        if assistant_response or thinking_started:
          got_partial = True
          print(f"\n[Partial stream drop] {error_type} after receiving content. "
                f"Skipping to fallback.", file=sys.stderr, flush=True)
          break
        if attempt == self.MAX_RETRIES:
          break
        # Pool rotation: on rate-limit / connection errors, demote the current
        # model and pick a different alive one BEFORE wasting more retries.
        # Only rotates when there are 2+ pool members AND the error is one
        # that rotating will help with (429s, connection issues).
        if isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError)) and len(_MODEL_POOL) > 1:
          _demote_model(self.model_name)
          new_model = _pick_next_model()
          if new_model != self.model_name:
            print(f"\n[Pool rotate] {self.model_name} -> {new_model}",
                  file=sys.stderr, flush=True)
            self.model_name = new_model
            kwargs['model'] = new_model
        delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
        print(f"\n[Retry {attempt}/{self.MAX_RETRIES}] {error_type}: {e}. "
              f"Retrying in {delay}s...", file=sys.stderr, flush=True)
        time.sleep(delay)

    # Primary exhausted — try fallback model with separate key
    if self.fallback_client:
      print(f"\n[Fallback] Primary model failed after {self.MAX_RETRIES} attempts. "
            f"Switching to {self.FALLBACK_MODEL}...", file=sys.stderr, flush=True)
      fallback_kwargs = {**kwargs, "model": self.FALLBACK_MODEL}
      # GLM supports reasoning — keep it in extra_body
      # Remove json_schema response_format — GLM doesn't support structured outputs
      if "response_format" in fallback_kwargs:
        del fallback_kwargs["response_format"]

      for attempt in range(1, self.MAX_RETRIES + 1):
        try:
          assistant_response = ""
          thinking_started = False
          stream = self.fallback_client.chat.completions.create(**fallback_kwargs)
          for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
              continue
            reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
            if reasoning:
              if not thinking_started:
                print("\n[Fallback Thinking]", file=sys.stderr, flush=True)
                thinking_started = True
              print(reasoning, end='', flush=True, file=sys.stderr)
            if delta.content:
              if thinking_started:
                print("\n[Fallback Output]", file=sys.stderr, flush=True)
                thinking_started = False
              assistant_response += delta.content
              print(_strip_non_ascii(delta.content), end='', flush=True)
          return assistant_response

        except (AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError, APIError, httpx.ReadError, httpx.RemoteProtocolError) as e:
          last_error = e
          if isinstance(e, AuthenticationError) or attempt == self.MAX_RETRIES:
            break
          delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
          error_type = type(e).__name__
          print(f"\n[Fallback retry {attempt}/{self.MAX_RETRIES}] {error_type}: {e}. "
                f"Retrying in {delay}s...", file=sys.stderr, flush=True)
          time.sleep(delay)

    # Both OpenRouter tiers exhausted -- try local Ollama as last resort
    if _OLLAMA_AVAILABLE:
      print(f"\n[Ollama fallback] OpenRouter unavailable. Trying local {self.OLLAMA_FALLBACK_MODEL}...",
            file=sys.stderr, flush=True)
      try:
        ollama_kwargs = {
          'model': self.OLLAMA_FALLBACK_MODEL,
          'messages': kwargs['messages'],
          'stream': True,
          'keep_alive': 0,
          'options': {'num_gpu': -1, 'gpu_memory_utilization': 0.9},
        }
        # Pass structured output schema if one is set
        active_schema = self.response_schema
        if active_schema:
          ollama_kwargs['format'] = active_schema.model_json_schema()
        assistant_response = ""
        stream = ollama_chat(**ollama_kwargs)
        for chunk in stream:
          content = chunk['message']['content']
          assistant_response += content
          print(_strip_non_ascii(content), end='', flush=True)
        return assistant_response
      except Exception as ollama_error:
        print(f"\n[Ollama fallback] Failed: {ollama_error}", file=sys.stderr, flush=True)

    raise last_error

  def parse_response(self, response: str, schema=None):
    """Parse a response using the active schema. Returns a validated Pydantic model instance.

    Applies several repair passes before validation to handle common LLM JSON defects:
    - Non-ASCII bleed (CJK, Cyrillic, Arabic characters mid-output)
    - Trailing commas before ] or } (invalid in strict JSON)
    - JSON embedded in prose or markdown code fences
    """
    active_schema = schema or self.response_schema
    if not active_schema:
      raise ValueError("No schema set. Set response_schema on the class or pass schema= argument.")

    # 1. Strip thinking tags
    clean = self._strip_thinking(response)

    # 2. Strip non-ASCII artifacts (CJK, Cyrillic, Arabic, etc. that bleed from multilingual models)
    clean = re.sub(r'[^\x00-\x7F]+', '', clean)

    # 3. Fix trailing commas before closing brackets/braces (e.g. ["a", "b",] -> ["a", "b"])
    clean = re.sub(r',(\s*[}\]])', r'\1', clean)

    # 4. Extract the first complete JSON object in case the response has surrounding prose
    json_match = re.search(r'\{.*\}', clean, re.DOTALL)
    if json_match:
      clean = json_match.group(0)

    return active_schema.model_validate_json(clean)
