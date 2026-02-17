from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
import httpx
import os, sys, json, re, time
from dotenv import load_dotenv

# Fix Windows console encoding — always use 'replace' to handle emojis/unicode
if hasattr(sys.stdout, 'reconfigure'):
  sys.stdout.reconfigure(errors='replace')
  sys.stderr.reconfigure(errors='replace')


class GroqModel:
  """
  Groq API base class using the OpenAI-compatible SDK.
  Same interface as OllamaModel/OpenRouterModel so agents can swap between backends.

  Subclasses set response_schema to a Pydantic BaseModel for structured JSON output.
  Uses Groq's chat completions API with streaming.

  Primary: deepseek-r1-distill-llama-70b (reasoning model, outputs <think> tags)
  Fallback: llama-3.3-70b-versatile (fast, no reasoning overhead)
  """
  response_schema = None

  MAX_RETRIES = 3
  RETRY_BASE_DELAY = 2  # seconds — Groq free tier: 30 req/min, need slightly longer backoff
  CLIENT_TIMEOUT = 120.0
  FALLBACK_MODEL = 'qwen/qwen3-32b'
  MAX_OUTPUT_TOKENS = 2048
  REASONING_EFFORT = None  # Groq doesn't support reasoning effort param; R1 reasons via <think> tags

  def __init__(self, model_name: str = 'llama-3.3-70b-versatile', api_key_env: str = "GROQ_API_KEY"):
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
      raise ValueError(f"{api_key_env} not found in environment. Add it to your .env file.")
    self.client = OpenAI(
      api_key=api_key,
      base_url="https://api.groq.com/openai/v1",
      timeout=self.CLIENT_TIMEOUT
    )
    self.model_name = model_name
    self.conversatoin_history = []  # Typo kept for codebase consistency

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
    """Stream a chat completion with exponential backoff retry on transient errors.
    Tries primary model first, then falls back to Llama 3.3 70B.
    Same client (same API key) for both — just swaps model name."""
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
        # Partial stream: skip retries, go to fallback
        if assistant_response or thinking_started:
          print(f"\n[Partial stream drop] {error_type} after receiving content. "
                f"Skipping to fallback.", file=sys.stderr, flush=True)
          break
        if attempt == self.MAX_RETRIES:
          break
        delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
        print(f"\n[Retry {attempt}/{self.MAX_RETRIES}] {error_type}: {e}. "
              f"Retrying in {delay}s...", file=sys.stderr, flush=True)
        time.sleep(delay)

    # Primary exhausted — try fallback (same client, different model)
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
