from typing import Dict, Any, Optional, Self
from mcp import StdioServerParameters, ClientSession, stdio_server
from mcp.client.stdio import stdio_client

import asyncio
import sys
import os
import json
from Orchestrator_Agent import Orchestrator_Agent
from huggingface_template import HuggingFaceModel
from ollama_template import OllamaModel
# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
