from typing import Dict, Any, Optional, Self
from mcp import StdioServerParameters, ClientSession, stdio_server
from mcp.client.stdio import stdio_client
from ollama import chat
import asyncio
import sys
import os
import torch
import json, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
      options={
        'num_gpu': -1,
        'gpu_memory_utilization': 0.9
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

class HuggingFaceModel():
  """this class was made specifically for Hugging Face Models to load in"""
  def __init__(self,cuda: bool = torch.cuda.is_available(),
              model_name = 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1',
              quantization_yeaconfig: Optional[BitsAndBytesConfig] = None):

    self.model_name = model_name
    self.quantization_config = quantization_yeaconfig
    self.device = "cuda" if cuda else "cpu"
    self.tokenizer = None
    self.model = None

    # load model and set to gpu
    self.load_model()

  def load_model(self):
    """load in the model from hugging face"""

    # model hasn't been cached yet, so download to agents/models dir
    cache_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(cache_dir, exist_ok=True)

    # load the tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)

    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # load the model with quantization
    if self.quantization_config is not None:
      try:
        self.model = AutoModelForCausalLM.from_pretrained(
          self.model_name,
          quantization_config=self.quantization_config,
          device_map='auto', # delgate model to use GPU + cpu
          dtype=torch.bfloat16, # tradeoff: accuracy for speed
          trust_remote_code=True, # this allows downloading and executing python code from hugging face repos
          cache_dir=cache_dir #  use model if cached else download to agents/models
        )
        print(f"{self.model_name} loaded successfully")
      except Exception as e:
        self.model=None
        print(f"Error loading model {e}")
    else:
      # Load without quantization but with aggressive CPU delegation
      self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        cache_dir=cache_dir,
        device_map='auto',
        dtype=torch.float16,  # Half precision to save memory
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Reduce memory usage during loading
        max_memory={0: "5.5GB", "cpu": "8GB"}  # Use 4GB GPU, leave 2GB buffer
      )


  def generate_response(self, prompt: str) -> Optional[str]:
    if not self.model:
      print(f'{self.model} not loaded')
      return None

    # structure the input as a chat
    # following boiler plate code from: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1?library=transformers
    messages = [{'role': 'system', 'content': "You are a Professional Wallstreet analyst"},
                {'role': 'user', 'content': prompt}
               ]
    if self.tokenizer is not None:
      # apply chat will automatically format to the model's expected input/output format
      inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, # adds "assistant" at the end
        tokenize=True, # convert to numbers
        return_dict=True, # returns as a dict
        return_tensors="pt", # use the pytorch format
      ).to(self.model.device)
      # result: {"input_ids": [1, 512, 1234, 5678, ...]}

      # this creates a printer that converts numbers back to text and then shows them immediately
      streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)

      with torch.inference_mode(): # an optimization, use "fast mode"
        outputs = self.model.generate(
          **inputs, # the numbers from inputs
          max_new_tokens=1024, # stop after 1024 new words
          do_sample=True, # tells model to be creative
          temperature=0.7, # how creative should the model be 0.1 = boring, 1.0 = sooper creative
          streamer=streamer, # print the words as they appear
          pad_token_id=self.tokenizer.eos_token_id # what mean stop
        )

      return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    else:
      print("Tokenizer not loaded")
      return None

class Orchestrator_Agent(OllamaModel):
  """this will first recieve the users query, then create a plan using the tools provided"""
  def __init__(self, model_name: str = 'orchestrator:latest'):
    super().__init__(model_name=model_name)

  def build_orchestrator_prompt(self, tool_list: Dict[str, Dict]) -> str:
    """Build complete system prompt with reasoning framework and available tools"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are an AI Task Orchestrator for financial analysis. Today: {current_date}

              YOUR ROLE:
              Given a user's request and available tools, create an execution plan by reasoning about:
              1. What information is needed to answer the question?
              2. Which tools can provide that information?
              3. What's missing or unclear? (Use web search for current data, context, or missing pieces)
              4. Do tools depend on each other? (Some tools output data needed by other tools)

              REASONING FRAMEWORK:
              - Historical financial data → Use SEC filing tools (revenue, margins, etc.)
              - Current market data, analyst opinions, context → Use search
              - Company strategy/decisions unclear → Use disclosures or search
              - Tool output contains IDs/names/URLs → Next tool might need those as input
              - Unsure or data incomplete → Add search to fill gaps

              EXAMPLES OF DYNAMIC THINKING:

              "Run a DCF on AAPL":
              - Need: Revenue (get_revenue_base)
              - Need: Profitability (get_ebitda_margin, get_tax_rate)
              - Need: CapEx & D&A (get_capex_pct_revenue, get_depreciation)
              - MISSING: WACC? Growth rate? Terminal value assumptions?
                → Search: {{"ticker": "AAPL", "query": {{"wacc": "AAPL WACC {datetime.now().year}", "growth": "AAPL revenue growth forecast", "target": "AAPL analyst price target"}}}}
              - MISSING: Recent events affecting valuation?
                → Search: {{"ticker": "AAPL", "query": {{"news": "AAPL news {datetime.now().year}", "events": "AAPL material events"}}}}
                → Or extract_8k_events for official filings

              "Is TSLA a good buy right now?":
              - Need: Current valuation multiples (search "TSLA P/E ratio")
              - Need: Recent financial performance (get_revenue_base, get_ebitda_margin)
              - Need: Market sentiment (search "TSLA analyst ratings {datetime.now().year}")
              - Need: Recent material events (extract_8k_events)
              - Need: Peer comparison (comparable_company_analysis)
              - Context: What's happening in EV market? (search "EV market trends {datetime.now().year}")

              "Why did NVDA stock drop last month?":
              - Need: Recent material events (extract_8k_events)
              - Need: News and market reaction (search "NVDA stock January {datetime.now().year}")
              - Need: Recent financial performance (get_latest_filing)
              - Context: Broader market conditions (search "semiconductor industry news {datetime.now().year}")

              TWO-PART TOOLS (output → input):
              - search gives URLs → get_urls_content needs those URLs (use placeholder "FROM_SEARCH")
              - get_disclosures_names gives list → extract_disclosure_data needs specific name
              - Think: "Does this tool's output help the next tool?"

              PRINCIPLES:
              - Be flexible and adaptive
              - Use search liberally when you need current/contextual data
              - Chain tools when outputs feed inputs
              - Don't rigidly follow patterns - reason about the actual need

              Available tools:

              """

    # Add dynamic tool list
    for tool_name, tool_info in tool_list.items():
      schema = tool_info['parameters']
      params = []

      if 'properties' in schema:
        required = set(schema.get('required', []))
        for param_name, param_info in schema['properties'].items():
          param_type = param_info.get('type', 'any')
          if param_name in required:
            params.append(f'{param_name}: {param_type}')
          else:
            params.append(f'{param_name}?: {param_type}')

      params_str = ", ".join(params)
      prompt += f"- {tool_name}({params_str}): {tool_info['description']}\n"

    prompt += """

      SPECIAL TOOL FORMATS:

      search tool - query parameter must be a dict with keys (any names) and search strings as values:
        CORRECT: {"ticker": "MSFT", "query": {"q1": "MSFT WACC 2026", "q2": "MSFT revenue forecast"}}
        WRONG: {"ticker": "MSFT", "query": {"terms": ["MSFT WACC 2026"]}}

      get_urls_content tool - urls must be a list of strings:
        CORRECT: {"urls": ["https://example.com", "https://example2.com"]}
        Use "FROM_SEARCH" placeholder if URLs come from previous search tool

      OUTPUT FORMAT:
      Respond with ONLY valid JSON. No thinking tags, no explanations, JUST JSON.

      JSON Structure:
      {
        "task_type": "DCF|Comps|Research|Valuation|Sentiment|...",
        "ticker": "AAPL or N/A",
        "reasoning": "Why these tools, in this order, what gaps search fills, what chains together",
        "tools_sequence": [
          {
            "tool": "tool_name",
            "arguments": {"param": "value"}
          }
        ]
      }

      CRITICAL: Each tool must have "tool" and "arguments" keys. All parameters go inside "arguments".

      RULES:
      - Output ONLY JSON, no text before or after
      - Tool names must match available tools exactly
      - For two-part tools: use "FROM_SEARCH" for URLs, leave disclosure_name empty to auto-inject
      - Reasoning should explain your thinking, not just list tools
      - Sequence matters - think about dependencies
      - Default form_type to "10-K" unless specified

      Remember: You're planning, not analyzing. Be adaptive, not rigid. Use search when you need context."""

    return prompt

  def create_plan(self, user_query: str, tool_list: Dict[str, Dict]):
    """
    Create execution plan for user query

    Args:
      user_query: User's request (e.g., "Run a DCF on AAPL")
      tool_list: Dict of available tools with descriptions and parameters

    Returns:
      dict: Execution plan with task_type, ticker, reasoning, tools_sequence
      or None if planning fails
    """
    # Build system prompt with tools
    system_prompt = self.build_orchestrator_prompt(tool_list)
    # Get plan from orchestrator model
    print(f"\nOrchestrator planning...\n", file=sys.stderr, flush=True)
    response = self.generate_response(prompt=user_query, system_prompt=system_prompt)
    # Parse the plan
    plan = self._parse_plan_response(response)

    return plan

class Financial_Analysis_Agent(OllamaModel):
  """this is the mcp tool setup and logic handling for the inherited model"""
  def __init__(self, model_name:str='DeepSeek-R1-Distill-Llama-8B:latest'):
    # runs on ollama model
    super().__init__(model_name)

    self.web_client_connection=None
    self.web_session = None  # Keep a single session alive

    self.financial_client = None
    self.web_client = None
    self.excel_client = None

  # using context manager pattern
  async def __aenter__(self) -> Self:
    await self.connect_to_servers()
    return self

  async def __aexit__(self,
                      exec_type: Optional[type[BaseException]],
                      exc_val: Optional[BaseException],
                      exc_tb: Optional[object]
                      ):
    await self.disconnect_from_servers()

  def __parse_CallToolResult(self, response):
    if hasattr(response, 'content'):
      text_content = response.content[0]
      try:
        json_string = text_content.text
        data = json.loads(json_string)
        return data
      except json.JSONDecodeError as e:
        print(f"Error: {str(e)}", file=sys.stderr, flush=True)
        # Return the text string, not the TextContent object
        return {"error": "JSON parse failed", "raw_text": text_content.text}
    else:
      raise AttributeError("Unable to find 'content' attribute in response")

  async def connect_to_servers(self):
    # Add project root to PYTHONPATH so subprocess can find modules
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_with_path = {**os.environ}
    if 'PYTHONPATH' in env_with_path:
      env_with_path['PYTHONPATH'] = f"{project_root}{os.pathsep}{env_with_path['PYTHONPATH']}"
    else:
      env_with_path['PYTHONPATH'] = project_root
    env_with_path['PYTHONUNBUFFERED'] = '1'

    financial_params = StdioServerParameters(
      command=sys.executable,
      args=["-m", "tools.financial_modeling_engine.analysis_tools", "server"],
      env=env_with_path
    )

    web_params = StdioServerParameters(
      command=sys.executable,
      args=['-m', "tools.web_search_server.web_search", "server"],
      env=env_with_path
    )

    excel_params = StdioServerParameters(
      command=sys.executable,
      args=['-m', "tools.excel_server.excel_tools", 'server'],
      env=env_with_path
    )

    try:
      self.web_client = stdio_client(web_params)
      self.web_client_connection = await self.web_client.__aenter__()

      # Create a single ClientSession that stays alive for the agent's lifetime
      self.web_session = ClientSession(*self.web_client_connection)
      await self.web_session.__aenter__()
      await self.web_session.initialize()
    except Exception as e:
      print(f"Unable to start servers: {str(e)}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc()
      raise
    #self.excel_client = stdio_client(excel_params)
    #self.financial_client = stdio_client(financial_params)

  async def disconnect_from_servers(self):
    if self.web_session:
      await self.web_session.__aexit__(None, None, None)
    if self.web_client:
      await self.web_client.__aexit__(None, None, None)


  async def call_tool(self, tool_name: str, args: Dict[str, Any]):
    if self.web_session is None:
      raise RuntimeError("Not connected! Please connect to server first")

    response = await self.web_session.call_tool(tool_name, args)
    res = self.__parse_CallToolResult(response)

    return res

  async def list_tools(self) -> Dict[str, Any]:
    if self.web_session is None:
      raise RuntimeError("Not connected! Please connect to the server first")

    web_response = await self.web_session.list_tools()

    tools = {}
    for tool in web_response.tools:
      tools[tool.name] = {"description":tool.description,
                          "parameters" :tool.inputSchema}

    return tools

  async def build_system_tool_prompt(self) -> str:
    """builds the system prompt using the list of tools"""
    tool_list = await self.list_tools()
    system_prompt = """You are a professional Investment Banker AI with access to financial analysis tools.

                    STRICT RULES:
                    1. When a user asks you to analyze a company, you MUST call the appropriate tool
                    2. Respond with ONLY this JSON format: {"tool": "tool_name", "arguments": {...}}
                    3. Output NOTHING else - no explanations, no markdown, just pure JSON
                    4. Do NOT say you cannot perform tasks - you CAN by calling tools
                    5. Do NOT make up tools - only use the tools listed below

                    Available Tools:
                    """

    for tool_name, tool_info in tool_list.items():
      schema = tool_info['parameters']
      params = []
      if 'properties' in schema:
        required = set(schema.get('required', []))
        for param_name, param_info in schema['properties'].items():
          param_type = param_info.get('type', 'any')
          default_value = param_info.get('default', None)
          if param_name in required:
            params.append(f'{param_name}: {param_type}')
          else:
            if default_value:
              params.append(f'{param_name}?: {param_type} = "{default_value}"')
            else:
              params.append(f'{param_name}?: {param_type}')
      params_str = ", ".join(params)
      system_prompt += f'- {tool_name}({params_str}): {tool_info["description"]}\n'

    system_prompt += """
                    EXAMPLES OF CORRECT RESPONSES:

                    User: "What is Apple's revenue?"
                    YOU RESPOND: {"tool": "get_revenue_base", "arguments": {"ticker": "AAPL"}}

                    User: "Get Microsoft's EBITDA margin"
                    YOU RESPOND: {"tool": "get_ebitda_margin", "arguments": {"ticker": "MSFT", "form_type": "10-K"}}

                    User: "Search for Tesla earnings"
                    YOU RESPOND: {"tool": "search", "arguments": {"ticker": "TSLA", "query": {"earnings": "Latest Earnings Report"}}}

                    Remember: Output ONLY the JSON, nothing else!
                    """

    return system_prompt

  async def run_with_tools(self, user_query: str):
    """Main agent loop with tool calling"""

    # Clear conversation history for fresh context each query
    self.conversatoin_history = []

    # Build system prompt with tools
    system_prompt = await self.build_system_tool_prompt()

    # Get model's tool decision
    print("\nAgent thinking...\n", file=sys.stderr, flush=True)
    response = self.generate_response(
        prompt=user_query,
        system_prompt=system_prompt
    )

    print(f"\n\nRaw model output:\n{response}\n", file=sys.stderr, flush=True)

    # Parse for tool call
    tool_name, tool_args = self._parse__tool_response(response)
    if tool_name:
        print(f"\nDetected tool call: {tool_name}", file=sys.stderr, flush=True)
        print(f"With arguments: {json.dumps(tool_args, indent=2)}\n", file=sys.stderr, flush=True)

        try:
            # Execute the tool
            if tool_args and tool_name:
              print(f"Executing tool...\n", file=sys.stderr, flush=True)
              tool_result = await self.call_tool(tool_name, tool_args)

            print(f"Tool result:\n{json.dumps(tool_result, indent=2)}\n", file=sys.stderr, flush=True)

            # Feed result back for analysis
            follow_up = f"""The tool '{tool_name}' returned this data:

            {json.dumps(tool_result, indent=2)}

            Based on this data, provide a clear analysis for the user's question: "{user_query}"
            Be concise and professional."""

            print("\nGenerating final analysis...\n", file=sys.stderr, flush=True)
            final = self.generate_response(
                prompt=follow_up,
                system_prompt="You are a professional Investment Banker. Analyze the data and provide clear insights."
            )

            return final

        except Exception as e:
            print(f"Error executing tool: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
    else:
        print("\nNo tool call detected - returning conversational response\n", file=sys.stderr, flush=True)
        return response

  async def run_with_planner(self, user_query: str):
    """Two-phase execution: Plan → Execute → Analyze"""

    # Clear conversation history for fresh context
    self.conversatoin_history = []

    # Phase 1: Create execution plan with orchestrator
    orchestrator = Orchestrator_Agent()
    tool_list = await self.list_tools()

    plan = orchestrator.create_plan(user_query, tool_list)

    if not plan:
        print("\nFailed to create execution plan. Falling back to single-tool mode.\n", file=sys.stderr, flush=True)
        return await self.run_with_tools(user_query)

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"EXECUTION PLAN", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)
    print(f"Task Type: {plan['task_type']}", file=sys.stderr, flush=True)
    print(f"Ticker: {plan.get('ticker', 'N/A')}", file=sys.stderr, flush=True)
    if 'reasoning' in plan:
        print(f"Reasoning: {plan['reasoning']}", file=sys.stderr, flush=True)
    print(f"Steps: {len(plan['tools_sequence'])} tools", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    # Phase 2: Execute all tools in sequence with smart result injection
    results = []
    result_store = {}  # Store results by tool name for dynamic chaining

    for idx, tool_call in enumerate(plan['tools_sequence'], 1):
        tool_name = tool_call['tool']
        tool_args = tool_call['arguments'].copy()  # Copy to avoid modifying plan
        # SMART INJECTION: Auto-detect what's needed from previous results
        # If tool needs URLs and placeholder detected
        if 'urls' in tool_args:
            if tool_args['urls'] == "FROM_SEARCH" or not tool_args['urls']:
                # Find most recent search result
                for prev_tool, prev_result in reversed(list(result_store.items())):
                    if prev_tool == 'search' and 'search_result' in prev_result:
                        urls = [item['link'] for item in prev_result['search_result'][:5]]
                        tool_args['urls'] = urls
                        print(f"Auto-injected {len(urls)} URLs from search", file=sys.stderr, flush=True)
                        break

        # If tool needs disclosure_name and it's missing or empty
        if tool_name == 'extract_disclosure_data':
            if 'disclosure_name' not in tool_args or not tool_args.get('disclosure_name'):
                # Find most recent get_disclosures_names result
                for prev_tool, prev_result in reversed(list(result_store.items())):
                    if prev_tool == 'get_disclosures_names' and 'disclosure_names' in prev_result:
                        if prev_result['disclosure_names']:
                            tool_args['disclosure_name'] = prev_result['disclosure_names'][0]
                            print(f"Auto-injected disclosure: {tool_args['disclosure_name']}", file=sys.stderr, flush=True)
                            break

        print(f"\n[Step {idx}/{len(plan['tools_sequence'])}] Executing: {tool_name}", file=sys.stderr, flush=True)
        print(f"Arguments: {json.dumps(tool_args, indent=2)}", file=sys.stderr, flush=True)

        try:
            result = await self.call_tool(tool_name, tool_args)

            # Store result for potential use by subsequent tools
            result_store[tool_name] = result

            results.append({
                'step': idx,
                'tool': tool_name,
                'arguments': tool_args,
                'result': result,
                'success': True
            })
            print(f"Result: Success", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)
            results.append({
                'step': idx,
                'tool': tool_name,
                'arguments': tool_args,
                'error': str(e),
                'success': False
            })

    # Phase 3: Analyze with DeepSeek-R1
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"ANALYSIS PHASE", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    # Build comprehensive analysis prompt with date context
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    analysis_prompt = f"""CURRENT DATE: {current_date}

                      User Request: {user_query}

                      Task Type: {plan['task_type']}
                      Ticker: {plan.get('ticker', 'N/A')}
                      """

    if 'reasoning' in plan:
        analysis_prompt += f"Plan Reasoning: {plan['reasoning']}\n"

    analysis_prompt += f"\nData Gathered ({len(results)} tools executed):\n\n"

    for r in results:
        if r['success']:
            analysis_prompt += f"--- {r['tool']} ---\n"
            analysis_prompt += f"{json.dumps(r['result'], indent=2)}\n\n"
        else:
            analysis_prompt += f"--- {r['tool']} (FAILED) ---\n"
            analysis_prompt += f"Error: {r['error']}\n\n"

    analysis_prompt += f"""
                    CRITICAL: Today is {current_date}. Use this when discussing market conditions, recent filings, and forward estimates.

                    YOUR TASK:
                    Analyze the data gathered above to answer the user's question. The orchestrator selected these tools based on its reasoning - use that context to guide your analysis.

                    REASONING FRAMEWORK:
                    1. Review what the orchestrator planned and why (see "Plan Reasoning" above)
                    2. Examine the data collected from each tool
                    3. Synthesize insights that directly address the user's question
                    4. Support your conclusions with specific data points
                    5. Be adaptive - the analysis type depends on the question asked and data gathered

                    ANALYSIS GUIDELINES:
                    - If financial metrics were gathered (revenue, margins, multiples): Provide valuation insights
                    - If web search data was included: Incorporate current market context, analyst opinions, recent news
                    - If SEC filings were retrieved: Use historical trends and official disclosures
                    - If recent events (8-K) were extracted: Discuss material impacts on the business
                    - If multiple companies were analyzed: Provide comparative analysis with peer context
                    - If DCF inputs were gathered: Calculate Free Cash Flow, apply WACC, provide valuation range
                    - If comp multiples were retrieved: Compare to peers, identify premium/discount, suggest fair value range

                    OUTPUT REQUIREMENTS:
                    - Be professional, precise, and actionable
                    - Support every conclusion with specific data from the tools
                    - Provide numerical analysis where appropriate (calculations, ratios, ranges)
                    - If data is missing or incomplete, acknowledge what else would be needed
                    - Structure your response clearly with sections/headers
                    - Use the orchestrator's reasoning to stay aligned with the intended analysis approach

                    Remember: You're a Senior Investment Banker. Provide institutional-grade analysis that directly answers what was asked."""

    final_analysis = self.generate_response(
        prompt=analysis_prompt,
        system_prompt=f"You are a Senior Investment Banker providing institutional-grade financial analysis. Today is {current_date}."
    )

    return final_analysis


if __name__ == "__main__":
  async def main():
    async with Financial_Analysis_Agent() as agent:
      print("Financial Analysis Agent started (with Orchestrator Planning)")
      print("Type 'exit' or 'quit' to stop.\n")
      print("Mode: /planner (multi-tool) or /single (single-tool)")
      print("Default: /planner\n")

      mode = 'planner'  # Default mode

      while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['exit', 'quit']:
          break

        # Mode switching
        if user_input == '/planner':
          mode = 'planner'
          print("Switched to planner mode (multi-tool execution)")
          continue
        elif user_input == '/single':
          mode = 'single'
          print("Switched to single-tool mode")
          continue


        print()

        # Execute based on mode
        if mode == 'planner':
          await agent.run_with_planner(user_input)
        else:
          await agent.run_with_tools(user_input)

        print("\n")

  asyncio.run(main())
