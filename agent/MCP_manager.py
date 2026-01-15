from typing import Dict, Any, Optional, Self
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import sys
import os
import json

class MCPConnectionManager:
  """Manages connections to MCP servers and tool execution"""
  def __init__(self):
      self.web_client = None
      self.web_client_connection = None
      self.web_session = None

      self.financial_client = None
      self.financial_session = None

      self.excel_client = None
      self.excel_session = None

  async def __aenter__(self) -> Self:
      await self.connect_to_servers()
      return self

  async def __aexit__(self, exc_type = None, exc_val=None, exc_tb=None):
      await self.disconnect_from_servers()

  def _get_env_with_pythonpath(self):
      """Prepare environment with PYTHONPATH for subprocess"""
      project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      env_with_path = {**os.environ}
      if 'PYTHONPATH' in env_with_path:
          env_with_path['PYTHONPATH'] = f"{project_root}{os.pathsep}{env_with_path['PYTHONPATH']}"
      else:
          env_with_path['PYTHONPATH'] = project_root
      env_with_path['PYTHONUNBUFFERED'] = '1'
      return env_with_path

  async def connect_to_servers(self, servers=['web', 'financial', 'excel']):
      """
      Connect to specified MCP servers

      Args:
          servers: List of server names to connect to ('web', 'financial', 'excel')
      """
      env = self._get_env_with_pythonpath()

      try:
          # Web Search Server
          if 'web' in servers:
              web_params = StdioServerParameters(
                  command=sys.executable,
                  args=['-m', "tools.web_search_server.web_search", "server"],
                  env=env
              )
              self.web_client = stdio_client(web_params)
              self.web_client_connection = await self.web_client.__aenter__()
              self.web_session = ClientSession(*self.web_client_connection)
              await self.web_session.__aenter__()
              await self.web_session.initialize()
              print("Connected to Web Search Server", file=sys.stderr, flush=True)

          # # Financial Analysis Server
          # if 'financial' in servers:
          #     financial_params = StdioServerParameters(
          #         command=sys.executable,
          #         args=["-m", "tools.financial_modeling_engine.analysis_tools", "server"],
          #         env=env
          #     )
          #     self.financial_client = stdio_client(financial_params)
          #     financial_connection = await self.financial_client.__aenter__()
          #     self.financial_session = ClientSession(*financial_connection)
          #     await self.financial_session.__aenter__()
          #     await self.financial_session.initialize()
          #     print("Connected to Financial Analysis Server", file=sys.stderr, flush=True)

          # # Excel Server
          # if 'excel' in servers:
          #     excel_params = StdioServerParameters(
          #         command=sys.executable,
          #         args=['-m', "tools.excel_server.excel_tools", 'server'],
          #         env=env
          #     )
          #     self.excel_client = stdio_client(excel_params)
          #     excel_connection = await self.excel_client.__aenter__()
          #     self.excel_session = ClientSession(*excel_connection)
          #     await self.excel_session.__aenter__()
          #     await self.excel_session.initialize()
          #     print("Connected to Excel Server", file=sys.stderr, flush=True)

      except Exception as e:
          print(f"Unable to start servers: {str(e)}", file=sys.stderr, flush=True)
          import traceback
          traceback.print_exc()
          raise

  async def disconnect_from_servers(self):
      """Disconnect from all connected MCP servers"""
      if self.web_session:
          await self.web_session.__aexit__(None, None, None)
      if self.web_client:
          await self.web_client.__aexit__(None, None, None)

      # if self.financial_session:
      #     await self.financial_session.__aexit__(None, None, None)
      # if self.financial_client:
      #     await self.financial_client.__aexit__(None, None, None)

      # if self.excel_session:
      #     await self.excel_session.__aexit__(None, None, None)
      # if self.excel_client:
      #     await self.excel_client.__aexit__(None, None, None)

      print("Disconnected from all MCP servers", file=sys.stderr, flush=True)

  def _parse_call_tool_result(self, response):
      """Parse MCP tool response"""
      if hasattr(response, 'content'):
          text_content = response.content[0]
          try:
              json_string = text_content.text
              data = json.loads(json_string)
              return data
          except json.JSONDecodeError as e:
              print(f"Error: {str(e)}", file=sys.stderr, flush=True)
              return {"error": "JSON parse failed", "raw_text": text_content.text}
      else:
          raise AttributeError("Unable to find 'content' attribute in response")

  async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
      """
      Call a tool on any connected server

      Args:
          tool_name: Name of the tool to call
          args: Arguments for the tool

      Returns:
          Parsed tool result
      """
      # Try each session until we find the tool
      sessions = [
          ("web", self.web_session),
          ("financial", self.financial_session),
          ("excel", self.excel_session)
      ]

      for server_name, session in sessions:
          if session is None:
              continue

          try:
              response = await session.call_tool(tool_name, args)
              return self._parse_call_tool_result(response)
          except Exception as e:
              # Tool not found in this server, try next
              if "Unknown tool" in str(e) or "not found" in str(e).lower():
                  continue
              else:
                  # Real error, propagate it
                  raise

      raise RuntimeError(f"Tool '{tool_name}' not found in any connected server")

  async def list_tools(self) -> Dict[str, Any]:
      """
      List all tools from all connected servers

      Returns:
          Dict mapping tool names to their descriptions and schemas
      """
      tools = {}

      sessions = [
          ("web", self.web_session),
          ("financial", self.financial_session),
          ("excel", self.excel_session)
      ]

      for server_name, session in sessions:
          if session is None:
              continue

          try:
              response = await session.list_tools()
              for tool in response.tools:
                  tools[tool.name] = {
                      "description": tool.description,
                      "parameters": tool.inputSchema,
                      "server": server_name
                  }
          except Exception as e:
              print(f"Warning: Could not list tools from {server_name}: {e}",
                    file=sys.stderr, flush=True)

      return tools
