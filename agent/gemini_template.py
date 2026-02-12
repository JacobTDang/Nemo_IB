from google import genai
from google.genai import types
import os
from dotenv import load_dotenv


class GeminiModel:
  """
  Gemini API base class using the new google.genai SDK.
  Same interface as OllamaModel so agents can swap between backends.

  Subclasses set response_schema to a Pydantic BaseModel for structured JSON output.
  """
  response_schema = None

  def __init__(self, model_name: str = 'gemini-2.5-flash'):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY not found in environment. Add it to your .env file.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.conversatoin_history = []

  def generate_response(self, prompt: str, system_prompt: str = "You are a professional Investment Banker from wallstreet", schema=None):
    active_schema = schema or self.response_schema

    # Build config with system instruction and optional structured output
    if active_schema:
      config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type='application/json',
        response_schema=active_schema
      )
    else:
      config = types.GenerateContentConfig(
        system_instruction=system_prompt
      )

    # Build contents: history + current prompt
    # Gemini uses role 'model' (not 'assistant') and 'user'
    contents = []
    for msg in self.conversatoin_history:
      role = 'model' if msg['role'] == 'assistant' else msg['role']
      contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['content'])]))
    contents.append(types.Content(role='user', parts=[types.Part.from_text(text=prompt)]))

    # Stream response
    assistant_response = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=config
    ):
      if chunk.text:
        assistant_response += chunk.text
        print(chunk.text, end='', flush=True)

    self.conversatoin_history.append({'role': 'user', 'content': prompt})
    self.conversatoin_history.append({'role': 'assistant', 'content': assistant_response})

    return assistant_response

  def parse_response(self, response: str, schema=None):
    """Parse a response using the active schema. Returns a validated Pydantic model instance."""
    active_schema = schema or self.response_schema
    if not active_schema:
      raise ValueError("No schema set. Set response_schema on the class or pass schema= argument.")
    return active_schema.model_validate_json(response)
