from typing import Dict, Any, Optional
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
import asyncio
import sys

"""This is a class template for different models, either on cloud or locally"""
class Financial_Analysis_Agent():

  def __init__(self, model_name:str='deepseek-r1:14b-mini'):
    # runs on ollama model
    self.model_name = model_name
    self.model = None

    self.financial_client = None
    self.web_client = None
    self.excel_client = None

  async def connect_to_servers(self):
    financial_params = StdioServerParameters(
      command=sys.executable,
      args=["-m", "tools.financial_modeling_engine.analysis_tools"]
    )

    web_params = StdioServerParameters(
      command=sys.executable,
      args=['-m', "tools.web_search_server.web_search"]
    )

    excel_params = StdioServerParameters(
      command=sys.executable,
      args=['-m', "tools.excel_server.excel_tools"]
    )

    # connect to the actual servers
    self.financial_client = stdio_client(financial_params)
    self.web_client = stdio_client(web_params)
    self.excel_client = stdio_client(excel_params)


  async def query(self, prompt: str):
    # send a query request for llm to process
    response = ollama.chat(
      model=self.model_name,
      messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

  async def generate_dcf_assumptions(self, query: str):
    # get all the assumptions from our websearch server
    pass
