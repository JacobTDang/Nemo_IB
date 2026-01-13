from ollama import chat
import json
import sys
import re

class OllamaModel():
  # ollama they could never make me hate you
  def __init__(self, model_name:str='DeepSeek-R1-Distill-Llama-8B:latest'):
    self.model_name = model_name
    self.conversatoin_history = []

  def generate_response(self, prompt:str, system_prompt:str = "You are a professional Investment Banker from wallstreet"):
    self.conversatoin_history.append({'role': 'user', 'content': prompt})
    messages = [
      {'role': 'system', 'content': system_prompt},
      *self.conversatoin_history
      ]
    stream = chat(
      model=self.model_name,
      messages=messages,
      stream=True,
      keep_alive=0,
      options={
        'num_gpu': -1,
        'gpu_memory_utilization': 0.9,
      }
    )
    assistant_response = ""
    for chunk in stream:
      content = chunk['message']['content']
      assistant_response += content
      print(content, end='', flush=True)

    self.conversatoin_history.append({'role': 'assistant', 'content': assistant_response})

    return assistant_response

  def _parse__tool_response(self, response: str):
    """Parse single tool call response (backward compatibility)"""
    # Clean up response - remove markdown code blocks
    response = response.strip()
    response = response.replace('```json', '').replace('```', '')

    # Find JSON by counting braces to handle nested objects
    start = response.find('{')
    if start == -1:
      print(f"No opening brace found in response: {response[:200]}...", file=sys.stderr, flush=True)
      return None, None

    brace_count = 0
    for i, char in enumerate(response[start:], start=start):
      if char == '{':
        brace_count += 1
      elif char == '}':
        brace_count -= 1
        if brace_count == 0:
          # Found matching closing brace
          json_str = response[start:i+1]
          try:
            tool_call = json.loads(json_str)
            if 'tool' in tool_call and 'arguments' in tool_call:
              return tool_call.get('tool'), tool_call.get('arguments', {})
            else:
              print(f"JSON missing 'tool' or 'arguments': {json_str}", file=sys.stderr, flush=True)
              return None, None
          except Exception as e:
            print(f"JSON parse error: {e}", file=sys.stderr, flush=True)
            print(f"Attempted to parse: {json_str}", file=sys.stderr, flush=True)
            return None, None

    print(f"No matching closing brace found in response: {response[:200]}...", file=sys.stderr, flush=True)
    return None, None

  def _parse_plan_response(self, response: str):
    """Parse orchestrator plan with multiple tool calls

    Returns:
      dict with keys: task_type, ticker, reasoning, tools_sequence
      or None if parsing fails
    """
    # Clean up response - remove markdown and thinking tags
    response = response.strip()
    response = response.replace('```json', '').replace('```', '')

    # Remove thinking tags if present
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()

    # Find JSON by counting braces
    start = response.find('{')
    if start == -1:
      print(f"No opening brace found in plan response: {response[:200]}...", file=sys.stderr, flush=True)
      return None

    brace_count = 0
    for i, char in enumerate(response[start:], start=start):
      if char == '{':
        brace_count += 1
      elif char == '}':
        brace_count -= 1
        if brace_count == 0:
          # Found matching closing brace
          json_str = response[start:i+1]
          try:
            plan = json.loads(json_str)

            # Validate plan structure
            required_keys = ['task_type', 'tools_sequence']
            if not all(key in plan for key in required_keys):
              print(f"Plan missing required keys: {required_keys}", file=sys.stderr, flush=True)
              print(f"Got: {list(plan.keys())}", file=sys.stderr, flush=True)
              return None

            # Validate tools_sequence is a list
            if not isinstance(plan['tools_sequence'], list):
              print(f"tools_sequence must be a list, got: {type(plan['tools_sequence'])}", file=sys.stderr, flush=True)
              return None

            # Validate each tool in sequence has 'tool' and 'arguments'
            for idx, tool_call in enumerate(plan['tools_sequence']):
              if 'tool' not in tool_call or 'arguments' not in tool_call:
                print(f"Tool #{idx} missing 'tool' or 'arguments': {tool_call}", file=sys.stderr, flush=True)
                return None

            print(f"\nParsed plan successfully:", file=sys.stderr, flush=True)
            print(f"  Task Type: {plan['task_type']}", file=sys.stderr, flush=True)
            print(f"  Ticker: {plan.get('ticker', 'N/A')}", file=sys.stderr, flush=True)
            print(f"  Tools: {len(plan['tools_sequence'])}", file=sys.stderr, flush=True)
            if 'reasoning' in plan:
              print(f"  Reasoning: {plan['reasoning']}", file=sys.stderr, flush=True)

            return plan

          except json.JSONDecodeError as e:
            print(f"JSON parse error in plan: {e}", file=sys.stderr, flush=True)
            print(f"Attempted to parse: {json_str[:500]}...", file=sys.stderr, flush=True)
            return None

    print(f"No matching closing brace found in plan response: {response[:200]}...", file=sys.stderr, flush=True)
    return None
