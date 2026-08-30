from openai import OpenAI, APIError, APIStatusError, RateLimitError, APIConnectionError, APITimeoutError
from ollama import chat as ollama_chat
import httpx
import os, sys, json, re, time
from dotenv import load_dotenv

# openrouter_template imports *this* module for CredentialsMissing, which is why
# Secret was mirrored here rather than imported from there. The shared home
# imports nothing at all, so there is no cycle to make.
from common.secret import Secret

# Fix Windows console encoding — always use 'replace' to handle emojis/unicode
if hasattr(sys.stdout, 'reconfigure'):
  sys.stdout.reconfigure(errors='replace')
  sys.stderr.reconfigure(errors='replace')



class CredentialsMissing(ValueError):
  """No usable API key is configured.

  Distinct from every other failure because it will not fix itself: a caller
  looping over work items gets the identical failure on every one. Subclasses
  ValueError so existing `except ValueError` handlers keep working.
  """


class GroqModel:
  """
  Groq API base class using the OpenAI-compatible SDK.
  Same interface as OllamaModel/OpenRouterModel so agents can swap between backends.

  Subclasses set response_schema to a Pydantic BaseModel for structured JSON output.
  Uses Groq's chat completions API with streaming.

  Primary: llama-3.3-70b-versatile (fast, reliable)
  Fallback: qwen/qwen3-32b (Groq alternate)
  Last resort: Ollama local (llama3.1:8b) when Groq is rate-limited
  """
  response_schema = None

  MAX_RETRIES = 3
  RETRY_BASE_DELAY = 2  # seconds — Groq free tier: 30 req/min, need slightly longer backoff
  CLIENT_TIMEOUT = 120.0
  FALLBACK_MODEL = 'qwen/qwen3-32b'
  OLLAMA_FALLBACK_MODEL = 'llama3.1:8b'  # Local last-resort when Groq is rate-limited
  MAX_OUTPUT_TOKENS = 2048
  REASONING_EFFORT = None  # Groq doesn't support reasoning effort param; R1 reasons via <think> tags

  def __init__(self, model_name: str = 'llama-3.3-70b-versatile', api_key_env: str = "GROQ_API_KEY"):
    load_dotenv()
    self._api_key_env = api_key_env
    self._client = None
    self.model_name = model_name
    self.conversatoin_history = []  # Typo kept for codebase consistency

  def _resolve_credential(self) -> Secret:
    """The configured key, wrapped. See Secret for why it is never bare.

    Building the Secret in the same expression that reads the environment is
    deliberate: an intermediate `api_key = os.getenv(...)` would put the raw
    value in this frame, and returning a bare `str` would put it in every
    caller's frame as well.
    """
    credential = Secret(os.getenv(self._api_key_env) or "")
    if not credential:
      raise CredentialsMissing(
        f"{self._api_key_env} not found in environment. Add it to your .env file.")
    return credential

  def _scrub(self, text: str) -> str:
    """Provider text with the configured key taken back out.

    Reads the environment again rather than calling _resolve_credential:
    every caller is inside an `except` block, and raising CredentialsMissing
    there would replace the provider's diagnosis with an unrelated one.
    """
    return Secret(os.getenv(self._api_key_env) or "").scrub(text)

  def validate_credentials(self) -> None:
    """Fail fast at process start. Daemon entrypoints call this on boot so a
    missing key surfaces immediately rather than partway through a run.

    Construction deliberately does not call this: subclasses carry
    deterministic methods -- prompt builders, guards, scoring maths -- that
    never touch the API, and validating in __init__ held those hostage to a
    credential they never use."""
    self._resolve_credential()

  @property
  def client(self) -> OpenAI:
    if self._client is None:
      self._client = OpenAI(
        api_key=self._resolve_credential().reveal(),
        base_url="https://api.groq.com/openai/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    return self._client

  @staticmethod
  def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.
    Handles both closed tags and unclosed tags (truncated mid-thought)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
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

    # If a Pydantic schema is set, request JSON output
    # Groq supports json_object mode broadly; json_schema only on some models
    # Groq REQUIRES the word "json" in messages when using json_object format
    if active_schema:
      kwargs["response_format"] = {"type": "json_object"}
      schema_hint = f"\nRespond with valid JSON matching this schema: {json.dumps(active_schema.model_json_schema())}"
      kwargs["messages"][0]["content"] += schema_hint

    # Stream response with retry on transient errors
    assistant_response = self._stream_with_retry(kwargs)

    # Store full response in history (with thinking), but return cleaned version
    self.conversatoin_history.append({"role": "user", "content": prompt})
    self.conversatoin_history.append({"role": "assistant", "content": assistant_response})

    # Strip <think> blocks so downstream consumers get clean output
    return self._strip_thinking(assistant_response)

  def _stream_with_retry(self, kwargs: dict) -> str:
    """Stream with 3-tier fallback: Groq primary -> Groq fallback -> Ollama local.
    Exponential backoff retry on each tier. Ollama is last resort for rate limiting."""
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
          # R1 distill on Groq puts reasoning in content as <think> tags
          # Some Groq models may use reasoning_content field
          reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
          if reasoning:
            if not thinking_started:
              print("\n[Thinking]", file=sys.stderr, flush=True)
              thinking_started = True
            print(reasoning, end='', flush=True, file=sys.stderr)
          # Capture actual output content
          if delta.content:
            # Detect inline <think> tags from R1 distill
            if '<think>' in delta.content and not thinking_started:
              thinking_started = True
              print("\n[Thinking]", file=sys.stderr, flush=True)
              # Print the thinking part to stderr, keep in response for strip later
              assistant_response += delta.content
              print(delta.content.replace('<think>', ''), end='', flush=True, file=sys.stderr)
              continue
            if thinking_started and '</think>' not in delta.content:
              assistant_response += delta.content
              print(delta.content, end='', flush=True, file=sys.stderr)
              continue
            if '</think>' in delta.content:
              thinking_started = False
              assistant_response += delta.content
              print("\n[Output]", file=sys.stderr, flush=True)
              # Print any content after the closing tag
              after = delta.content.split('</think>', 1)[-1]
              if after:
                print(after, end='', flush=True)
              continue

            assistant_response += delta.content
            print(delta.content, end='', flush=True)
        return assistant_response

      except (RateLimitError, APIConnectionError, APITimeoutError, APIError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        last_error = e
        error_type = type(e).__name__
        # 413 = request too large for Groq TPM limit — no Groq model will handle it
        if isinstance(e, APIStatusError) and e.status_code == 413:
          print(f"\n[Request too large for Groq] {self._scrub(str(e))}. "
                f"Skipping to Ollama.", file=sys.stderr, flush=True)
          break  # Falls through Groq fallback to Ollama
        # Partial stream: skip retries, go to fallback
        if assistant_response or thinking_started:
          print(f"\n[Partial stream drop] {error_type} after receiving content. "
                f"Skipping to fallback.", file=sys.stderr, flush=True)
          break
        if attempt == self.MAX_RETRIES:
          break
        delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
        print(f"\n[Retry {attempt}/{self.MAX_RETRIES}] {error_type}: "
              f"{self._scrub(str(e))}. Retrying in {delay}s...",
              file=sys.stderr, flush=True)
        time.sleep(delay)

    # Primary exhausted — try fallback (same client, different model)
    # Skip Groq fallback entirely if request is too large for any Groq model
    _skip_groq_fallback = isinstance(last_error, APIStatusError) and last_error.status_code == 413
    if not _skip_groq_fallback:
      print(f"\n[Fallback] Primary model failed after {self.MAX_RETRIES} attempts. "
            f"Switching to {self.FALLBACK_MODEL}...", file=sys.stderr, flush=True)
      fallback_kwargs = {**kwargs, "model": self.FALLBACK_MODEL}

      for attempt in range(1, self.MAX_RETRIES + 1):
        try:
          assistant_response = ""
          stream = self.client.chat.completions.create(**fallback_kwargs)
          for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
              continue
            if delta.content:
              assistant_response += delta.content
              print(delta.content, end='', flush=True)
          return assistant_response

        except (RateLimitError, APIConnectionError, APITimeoutError, APIError, httpx.ReadError, httpx.RemoteProtocolError) as e:
          last_error = e
          if attempt == self.MAX_RETRIES:
            break
          delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
          error_type = type(e).__name__
          print(f"\n[Fallback retry {attempt}/{self.MAX_RETRIES}] {error_type}: "
                f"{self._scrub(str(e))}. Retrying in {delay}s...",
                file=sys.stderr, flush=True)
          time.sleep(delay)

    # Both Groq models exhausted — try Ollama local as last resort
    print(f"\n[Ollama Fallback] Groq exhausted. Trying local {self.OLLAMA_FALLBACK_MODEL}...",
          file=sys.stderr, flush=True)
    try:
      # Convert OpenAI-format kwargs to Ollama format
      ollama_kwargs = {
        'model': self.OLLAMA_FALLBACK_MODEL,
        'messages': kwargs['messages'],
        'stream': True,
        'keep_alive': 0,
        'options': {
          'num_gpu': -1,
          'gpu_memory_utilization': 0.9,
        }
      }

      # Ollama uses 'format' with the raw JSON schema dict, not response_format
      if 'response_format' in kwargs:
        active_schema = self.response_schema
        if active_schema:
          ollama_kwargs['format'] = active_schema.model_json_schema()

      assistant_response = ""
      stream = ollama_chat(**ollama_kwargs)
      for chunk in stream:
        content = chunk['message']['content']
        assistant_response += content
        print(content, end='', flush=True)
      return assistant_response

    except Exception as ollama_error:
      print(f"\n[Ollama Fallback Failed] {type(ollama_error).__name__}: {ollama_error}",
            file=sys.stderr, flush=True)
      # Raise the original Groq error — Ollama is just a bonus attempt
      raise last_error

  def parse_response(self, response: str, schema=None):
    """Parse a response using the active schema. Returns a validated Pydantic model instance."""
    active_schema = schema or self.response_schema
    if not active_schema:
      raise ValueError("No schema set. Set response_schema on the class or pass schema= argument.")
    # Strip thinking tags in case they weren't already removed
    clean = self._strip_thinking(response)
    return active_schema.model_validate_json(clean)
