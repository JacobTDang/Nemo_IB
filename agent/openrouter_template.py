from openai import OpenAI, APIError, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError, NotFoundError
from .groq_template import CredentialsMissing
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


_MODEL_ID_RE = re.compile(r'^[^\s/:#]+/[^\s/:#]+(?::[^\s/:#]+)?$')


def _is_valid_model_id(model_id) -> bool:
  """True if `model_id` looks like an OpenRouter id: `vendor/model` or
  `vendor/model:tag`.

  Rejects None, empty/whitespace-only strings, anything starting with '#', and
  anything without a vendor separator. This exists because a malformed override
  (e.g. a trailing `# comment` that dotenv read as the VALUE of
  PRIMARY_REASONING_MODEL) must never be pinged, let alone admitted to the pool.
  """
  if not isinstance(model_id, str):
    return False
  candidate = model_id.strip()
  if not candidate or candidate.startswith('#'):
    return False
  return _MODEL_ID_RE.match(candidate) is not None


class CredentialRejected(RuntimeError):
    """The provider refused the API key, so nothing could be verified.

    Distinct from a model being absent. Raised rather than returned so a pool
    cannot be assembled out of failed probes.
    """


class Secret:
  """A credential that renders as a placeholder instead of as itself.

  pytest prints a frame's arguments at the head of every traceback entry, and
  every local under --showlocals. So a key sitting in a parameter or a variable
  is written to stdout by the first test that fails anywhere below it: the live
  OpenRouter key went into CI logs, pasted terminals and captured artefacts
  that way (issue #17).

  Suppressing frame rendering in pytest configuration would have fixed pytest
  and nothing else -- a log line, a debugger, a crash reporter and a print
  written next year are the same disclosure by another route. Keeping the value
  behind reveal() leaves nothing renderable to render, which closes all of them
  at once.
  """
  __slots__ = ("_value",)

  PLACEHOLDER = "<redacted>"

  def __init__(self, value: str = ""):
    self._value = value or ""

  def reveal(self) -> str:
    """The raw credential.

    Call this at the point of use -- an SDK constructor, a request header --
    and never bind the result to a name, or the value is back in a frame.
    """
    return self._value

  def scrub(self, text: str) -> str:
    """`text` with the credential replaced by the placeholder.

    Provider error bodies are quoted into the messages we raise and print. A
    provider that echoed the offending credential back would otherwise put it
    straight into our own diagnostics.
    """
    if not self._value:
      return text
    return text.replace(self._value, self.PLACEHOLDER)

  def __repr__(self) -> str:
    return self.PLACEHOLDER

  __str__ = __repr__

  def __bool__(self) -> bool:
    return bool(self._value)


def _openrouter_credential() -> Secret:
  """The configured OpenRouter key, wrapped so it cannot be rendered.

  Reading the key here rather than accepting it as an argument is the other
  half of the fix: a credential that never crosses a function boundary cannot
  be printed as that function's arguments.
  """
  load_dotenv()
  return Secret(os.getenv("OPENROUTER_API_KEY") or "")


def _verify_model_alive(model_id: str, credential: 'Secret | None' = None,
                        timeout: float = 10.0) -> bool:
  """Send a 1-token completion to check the model endpoint exists.

  Returns False immediately — without any network call — for an id that is not
  shaped like an OpenRouter model id; a malformed id is definitively unusable
  and pinging it would only produce an error we would have to interpret.

  Otherwise returns True if alive, or if the error is non-404 (auth, rate limit,
  timeout all indicate the model name itself is at least known and none of them
  prove the model is dead). Returns False on explicit 404 NotFoundError.

  `credential` is a `Secret`, never a bare string, and may be omitted to read
  the configured key here. A bare string is refused rather than quietly
  wrapped, because wrapping it on this side would leave the raw value in the
  *caller's* frame -- which is where it leaked from.
  """
  if credential is not None and not isinstance(credential, Secret):
    raise TypeError(
      f"_verify_model_alive needs a Secret, got {type(credential).__name__}. "
      f"Wrap the credential with Secret(...) where it is read, or omit the "
      f"argument and let the probe read OPENROUTER_API_KEY itself.")
  if not _is_valid_model_id(model_id):
    print(f"[OpenRouter] Rejecting malformed model id {model_id!r} "
          f"(expected 'vendor/model[:tag]'); not adding it to the pool.",
          file=sys.stderr, flush=True)
    return False
  if credential is None:
    credential = _openrouter_credential()
  if not credential:
    # Refused here rather than handed on. An empty key makes the SDK's own
    # constructor raise, the generic handler below reads any non-404 as "keep
    # it in the pool", and the model is then marked verified without having
    # been probed -- the same fiction CredentialRejected exists to prevent,
    # arriving through a missing key instead of a rejected one.
    raise CredentialsMissing(
      "OPENROUTER_API_KEY is not set, so no model can be probed. Set it in "
      "your .env file -- a valid key begins 'sk-or-v1-'.")
  try:
    client = OpenAI(api_key=credential.reveal(),
                    base_url="https://openrouter.ai/api/v1", timeout=timeout)
    client.chat.completions.create(
      model=model_id,
      messages=[{"role": "user", "content": "ping"}],
      max_tokens=1,
    )
    return True
  except NotFoundError:
    print(f"[OpenRouter] {model_id} returned 404; treating it as dead.",
          file=sys.stderr, flush=True)
    return False
  except AuthenticationError as exc:
    # A rejected credential answers a question about the KEY, not the model.
    # Treating it as "not a 404, so alive" marked every model in the list
    # alive and built the pool entirely out of 401s -- observed live, a
    # rejected key still logged "Pool initialized with 5 models" without one
    # of them having been reached. A pool that reports five verified models
    # when zero were verified is our own outage wearing a fact about the
    # world.
    raise CredentialRejected(
      f"OpenRouter rejected the API key while probing {model_id}: "
      f"{credential.scrub(str(exc))}. "
      f"No model can be verified with a credential the provider will not "
      f"accept, so the pool is not built rather than filled with unverified "
      f"entries. Check OPENROUTER_API_KEY -- a valid key begins 'sk-or-v1-'."
    ) from exc
  except Exception as exc:
    # Rate limit, timeout etc. don't prove the model is dead, so it stays
    # in the pool -- but report what happened rather than swallowing it.
    print(f"[OpenRouter] {model_id} probe hit {type(exc).__name__}: "
          f"{credential.scrub(str(exc))}. Not a 404, so keeping it in the pool.",
          file=sys.stderr, flush=True)
    return True


# ---------------------------------------------------------------------------
# Reasoning model pool
# ---------------------------------------------------------------------------
# Module-level state. All access goes through the helpers below.
_MODEL_POOL: list = []                  # alive models, in preference order
_POOL_CREDENTIAL_ERROR: str = ''        # why the pool is empty, if it is
_MODEL_LAST_USED: dict = {}             # model -> unix timestamp last picked
_MODEL_DEMOTED_UNTIL: dict = {}         # model -> unix timestamp it's banned
_POOL_LOCK = Lock()
_DEMOTE_SECONDS = 90                    # how long a model is banned after a 429


def _build_reasoning_pool() -> list:
  """Ping each candidate, return the alive ones in preference order.

  Reads OPENROUTER_API_KEY. If unset, returns the ultimate fallback only.
  Honors PRIMARY_REASONING_MODEL env var as a top-priority override.
  """
  credential = _openrouter_credential()
  ultimate_fallback = 'z-ai/glm-4.5-air:free'

  if not credential:
    return [ultimate_fallback]

  # An explicit override stays at the top of the pool, but only if it is
  # actually shaped like a model id. A malformed value here (most commonly a
  # trailing `# comment` that dotenv read as the value) is reported and dropped
  # rather than silently becoming the default model for every agent.
  override = os.getenv("PRIMARY_REASONING_MODEL")
  if override is not None and not _is_valid_model_id(override):
    print(f"[OpenRouter] Ignoring malformed PRIMARY_REASONING_MODEL={override!r}. "
          f"Expected 'vendor/model[:tag]'. Check for a comment on the same line "
          f"as the assignment in your .env.",
          file=sys.stderr, flush=True)
    override = None

  candidates = [
    override,
    'deepseek/deepseek-chat-v3.1:free',
    'deepseek/deepseek-r1-distill-llama-70b:free',
    'qwen/qwq-32b-preview:free',
    'meta-llama/llama-3.3-70b-instruct:free',
    'z-ai/glm-4.5-air:free',
  ]
  global _POOL_CREDENTIAL_ERROR
  alive = []
  seen = set()
  for c in candidates:
    try:
      ok = _verify_model_alive(c, credential)
    except CredentialRejected as exc:
      # Recorded, not raised here. Importing this module for a helper or a
      # type must not explode because a credential is bad -- but resolving a
      # model must, because that is the point at which a caller is about to
      # rely on it. The pool stays EMPTY rather than falling back, so nothing
      # downstream mistakes an unverified name for a working one.
      _POOL_CREDENTIAL_ERROR = str(exc)
      print(f"[OpenRouter] {exc}", file=sys.stderr, flush=True)
      return []
    if c and c not in seen and ok:
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
_MODEL_POOL: list = []
_POOL_BUILT = False


def _configured_candidate() -> str:
  """The first configured model name, with no claim that it is reachable.

  Used only when the credential was rejected, so that importing a name does
  not fail while resolving one still does.
  """
  configured = os.getenv("PRIMARY_REASONING_MODEL", "").strip()
  if _is_valid_model_id(configured):
    return configured
  return 'z-ai/glm-4.5-air:free'   # same ultimate fallback the pool uses


def _ensure_pool() -> list:
  """Build the pool on first use.

  This used to run at import, which meant importing the module for a type or a
  helper cost five OpenRouter pings and ~0.7s -- and made an offline test run
  not strictly offline. Resolving a model is what should cost, not importing.
  """
  global _MODEL_POOL, _POOL_BUILT
  with _POOL_LOCK:
    if not _POOL_BUILT:
      _MODEL_POOL = _build_reasoning_pool()
      _POOL_BUILT = True
  return _MODEL_POOL


def primary_reasoning_model() -> str:
  """First-preference alive model, resolving the pool if needed.

  Raises when the credential was rejected. An empty pool there is not "no
  models available" -- it is "we could not ask", and handing back a fallback
  name would be an unverified model wearing a verified one's place.
  """
  pool = _ensure_pool()
  if not pool and _POOL_CREDENTIAL_ERROR:
    raise CredentialRejected(_POOL_CREDENTIAL_ERROR)
  return pool[0]


def __getattr__(name: str):
  """PEP 562 module-level attribute access.

  Keeps `PRIMARY_REASONING_MODEL` working as a module attribute and as a
  from-import for the several consumers that read it, while deferring the
  network work until something actually asks.
  """
  if name == "PRIMARY_REASONING_MODEL":
    # The attribute is a NAME; primary_reasoning_model() is a claim that the
    # name resolves to something live. Only the latter refuses when the
    # credential was rejected -- otherwise importing this module for a helper
    # takes the whole test suite down over a key almost nothing needs, which
    # buries every other signal. The name is still marked unverified, and any
    # actual completion fails with the provider's own 401.
    try:
      return primary_reasoning_model()
    except CredentialRejected as exc:
      print(f"[OpenRouter] PRIMARY_REASONING_MODEL is UNVERIFIED: {exc}",
            file=sys.stderr, flush=True)
      return _configured_candidate()
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
      model_name = primary_reasoning_model()
    # Try the requested env var first, then fall back to the main key.
    # This means a single OPENROUTER_API_KEY is always enough to run the system --
    # model-specific keys (OPENROUTER_NEMOTRON, OPENROUTER_GLM) are optional extras.
    self._api_key_env = api_key_env
    self._client = None
    self._fallback_client = None
    self.model_name = model_name
    self.conversatoin_history = []

  def _resolve_credential(self) -> Secret:
    """The configured key, wrapped. See Secret for why it is never bare.

    Building the Secret in the same expression that reads the environment is
    deliberate: an intermediate `api_key = os.getenv(...)` would put the raw
    value in this frame, which is exactly what a rendered traceback prints.
    """
    credential = Secret(os.getenv(self._api_key_env)
                        or os.getenv("OPENROUTER_API_KEY") or "")
    if not credential:
      raise CredentialsMissing(
        f"No API key found. Set OPENROUTER_API_KEY (or {self._api_key_env}) in your .env file.")
    return credential

  def validate_credentials(self) -> None:
    """Fail fast at process start. See GroqModel.validate_credentials."""
    self._resolve_credential()

  @property
  def client(self) -> OpenAI:
    if self._client is None:
      self._client = OpenAI(
        api_key=self._resolve_credential().reveal(),
        base_url="https://openrouter.ai/api/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    return self._client

  @property
  def fallback_client(self) -> OpenAI:
    # Prefer OPENROUTER_GLM, otherwise reuse the main key. Reusing it is fine --
    # the fallback only triggers when the primary model fails.
    # _resolve_credential is still called when OPENROUTER_GLM is absent so the
    # fallback path cannot become a way to skip the credential check.
    if self._fallback_client is None:
      credential = (Secret(os.getenv("OPENROUTER_GLM") or "")
                    or self._resolve_credential())
      self._fallback_client = OpenAI(
        api_key=credential.reveal(),
        base_url="https://openrouter.ai/api/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    return self._fallback_client

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
