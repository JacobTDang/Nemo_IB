from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
import httpx
import os, sys, json, re, time
from dotenv import load_dotenv

# Fix Windows console encoding — always use 'replace' to handle emojis/unicode
if hasattr(sys.stdout, 'reconfigure'):
  sys.stdout.reconfigure(errors='replace')
  sys.stderr.reconfigure(errors='replace')


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
  MAX_OUTPUT_TOKENS = 2048  # Subclasses can override (e.g., verifier needs more room after thinking)
  REASONING_EFFORT = "low"  # Subclasses can set to None to disable reasoning (e.g., orchestrator just needs JSON)

  def __init__(self, model_name: str = 'deepseek/deepseek-r1-0528:free', api_key_env: str = "OPENROUTER_API_KEY"):
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
      raise ValueError(f"{api_key_env} not found in environment. Add it to your .env file.")
    self.client = OpenAI(
      api_key=api_key,
      base_url="https://openrouter.ai/api/v1",
      timeout=self.CLIENT_TIMEOUT
    )
    self.model_name = model_name
    self.conversatoin_history = []

    # Fallback client uses a separate API key for GLM-4.5-Air
    fallback_key = os.getenv("OPENROUTER_GLM")
    if fallback_key:
      self.fallback_client = OpenAI(
        api_key=fallback_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    else:
      self.fallback_client = None

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
            print(delta.content, end='', flush=True)
        return assistant_response

      except (RateLimitError, APIConnectionError, APITimeoutError, APIError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        last_error = e
        error_type = type(e).__name__
        # If we got thinking or content before the drop, retrying the same model
        # will likely produce the same result. Skip to fallback immediately.
        if assistant_response or thinking_started:
          got_partial = True
          print(f"\n[Partial stream drop] {error_type} after receiving content. "
                f"Skipping to fallback.", file=sys.stderr, flush=True)
          break
        if attempt == self.MAX_RETRIES:
          break
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
              print(delta.content, end='', flush=True)
          return assistant_response

        except (RateLimitError, APIConnectionError, APITimeoutError, APIError, httpx.ReadError, httpx.RemoteProtocolError) as e:
          last_error = e
          if attempt == self.MAX_RETRIES:
            break
          delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
          error_type = type(e).__name__
          print(f"\n[Fallback retry {attempt}/{self.MAX_RETRIES}] {error_type}: {e}. "
                f"Retrying in {delay}s...", file=sys.stderr, flush=True)
          time.sleep(delay)

    raise last_error

  def parse_response(self, response: str, schema=None):
    """Parse a response using the active schema. Returns a validated Pydantic model instance."""
    active_schema = schema or self.response_schema
    if not active_schema:
      raise ValueError("No schema set. Set response_schema on the class or pass schema= argument.")
    # Strip thinking tags in case they weren't already removed
    clean = self._strip_thinking(response)
    return active_schema.model_validate_json(clean)
