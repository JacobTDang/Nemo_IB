from typing import Dict, Any, Self
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import sys
import os
import json
import asyncio

class MCPConnectionManager:
  """Manages connections to MCP servers and tool execution.

  Builds a tool registry at connect time so call_tool routes directly
  to the correct server without trial-and-error.
  """
  def __init__(self):
      self.web_client = None
      self.web_client_connection = None
      self.web_session = None

      self.financial_client = None
      self.financial_session = None

      # Maps server_name -> ClientSession
      self._sessions: Dict[str, ClientSession] = {}
      # Maps tool_name -> server_name (built at connect time)
      self._tool_registry: Dict[str, str] = {}

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

  async def connect_to_servers(self, servers=['web', 'financial']):
      """
      Connect to specified MCP servers and build tool registry.

      Args:
          servers: List of server names to connect to ('web', 'financial')
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
              self._sessions['web'] = self.web_session
              print("Connected to Web Search Server", file=sys.stderr, flush=True)

          # Financial Analysis Server
          if 'financial' in servers:
              financial_params = StdioServerParameters(
                  command=sys.executable,
                  args=["-m", "tools.financial_modeling_engine.analysis_tools", "server"],
                  env=env
              )
              self.financial_client = stdio_client(financial_params)
              financial_connection = await self.financial_client.__aenter__()
              self.financial_session = ClientSession(*financial_connection)
              await self.financial_session.__aenter__()
              await self.financial_session.initialize()
              self._sessions['financial'] = self.financial_session
              print("Connected to Financial Analysis Server", file=sys.stderr, flush=True)

      except Exception as e:
          print(f"Unable to start servers: {str(e)}", file=sys.stderr, flush=True)
          import traceback
          traceback.print_exc()
          raise

      # Build tool registry from all connected servers
      await self._build_tool_registry()

  async def _build_tool_registry(self):
      """Query each server for its tools and build name -> server mapping."""
      self._tool_registry = {}

      for server_name, session in self._sessions.items():
          try:
              response = await session.list_tools()
              for tool in response.tools:
                  self._tool_registry[tool.name] = server_name
          except Exception as e:
              print(f"Warning: Could not list tools from {server_name}: {e}",
                    file=sys.stderr, flush=True)

      print(f"Tool registry: {len(self._tool_registry)} tools across {len(self._sessions)} servers",
            file=sys.stderr, flush=True)

  async def disconnect_from_servers(self):
      """Disconnect from all connected MCP servers"""
      servers = [
          ("web", self.web_session, self.web_client),
          ("financial", self.financial_session, self.financial_client),
      ]

      for name, session, client in servers:
          try:
              if session:
                  await session.__aexit__(None, None, None)
          except Exception as e:
              print(f"Warning: {name} session cleanup failed: {e}", file=sys.stderr, flush=True)
          try:
              if client:
                  # Timeout the client cleanup - anyio's process.wait() can hang
                  await asyncio.wait_for(client.__aexit__(None, None, None), timeout=3.0)
          except (asyncio.TimeoutError, asyncio.CancelledError):
              print(f"Warning: {name} client cleanup timed out, forcing shutdown", file=sys.stderr, flush=True)
          except Exception as e:
              print(f"Warning: {name} client cleanup failed: {e}", file=sys.stderr, flush=True)

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
      Call a tool using the registry to route to the correct server.

      Args:
          tool_name: Name of the tool to call
          args: Arguments for the tool

      Returns:
          Parsed tool result
      """
      server_name = self._tool_registry.get(tool_name)

      if server_name is None:
          # Log what we know for debugging
          print(f"Tool {tool_name} not listed by server, cannot validate any structured content", flush=True)
          raise RuntimeError(f"Tool '{tool_name}' not found in registry. Available: {list(self._tool_registry.keys())}")

      session = self._sessions.get(server_name)
      if session is None:
          raise RuntimeError(f"Server '{server_name}' for tool '{tool_name}' is not connected")

      response = await session.call_tool(tool_name, args)
      return self._parse_call_tool_result(response)

  async def list_tools(self) -> Dict[str, Any]:
      """
      List all tools from all connected servers.

      Returns:
          Dict mapping tool names to their descriptions and schemas
      """
      tools = {}

      for server_name, session in self._sessions.items():
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
