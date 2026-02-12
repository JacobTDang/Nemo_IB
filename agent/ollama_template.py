from ollama import chat

class OllamaModel():
  # ollama they could never make me hate you
  response_schema = None  # Subclasses set this to a Pydantic BaseModel to get structured output

  def __init__(self, model_name:str='DeepSeek-R1-Distill-Llama-8B:latest'):
    self.model_name = model_name
    self.conversatoin_history = []

  def generate_response(self, prompt:str, system_prompt:str = "You are a professional Investment Banker from wallstreet", schema=None):
    self.conversatoin_history.append({'role': 'user', 'content': prompt})
    messages = [
      {'role': 'system', 'content': system_prompt},
      *self.conversatoin_history
      ]

    # Use schema priority: explicit arg > class-level > none
    active_schema = schema or self.response_schema

    kwargs = {
      'model': self.model_name,
      'messages': messages,
      'stream': True,
      'keep_alive': 0,
      'options': {
        'num_gpu': -1,
        'gpu_memory_utilization': 0.9,
      }
    }

    if active_schema:
      kwargs['format'] = active_schema.model_json_schema()

    stream = chat(**kwargs)
    assistant_response = ""
    for chunk in stream:
      content = chunk['message']['content']
      assistant_response += content
      print(content, end='', flush=True)

    self.conversatoin_history.append({'role': 'assistant', 'content': assistant_response})

    return assistant_response

  def parse_response(self, response: str, schema=None):
    """Parse a response using the active schema. Returns a validated Pydantic model instance."""
    active_schema = schema or self.response_schema
    if not active_schema:
      raise ValueError("No schema set. Set response_schema on the class or pass schema= argument.")
    return active_schema.model_validate_json(response)

