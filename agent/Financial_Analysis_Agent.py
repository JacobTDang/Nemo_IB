from typing import Dict, Any, Optional, Self
from mcp import StdioServerParameters, ClientSession, stdio_server
from mcp.client.stdio import stdio_client
from ollama import chat
import asyncio
import sys
import os
import torch
import json
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
    messages = [
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': prompt}
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

    for chunk in stream:
      print(chunk['message']['content'], end='', flush=True)

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
        return text_content
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

  async def list_tool(self) -> Dict[str, str]:
    if self.web_session is None:
      raise RuntimeError("Not connected! Please connect to the server first")

    web_response = await self.web_session.list_tools()

    tools = {}
    for tool in web_response.tools:
      tools[tool.name] = str(tool.description)
    return tools

if __name__ == "__main__":
  async def main():
    ticker = "AAPL"
    async with Financial_Analysis_Agent() as agent:

    #   # List available tools
    #   print(await agent.list_tool(), file=sys.stderr, flush=True)

    #   print(await agent.call_tool('search', {"ticker" :ticker, "query": {
    #   "earnings": "Latest Earnings Report",
    #   "financial": "Q4 2024"
    # }}), file=sys.stderr, flush=True)

      # print(f'REVENUE DATA: {await agent.call_tool('get_revenue_base', {'ticker': ticker})}', file=sys.stderr, flush=True)
      # print(f'EBITDA MARGIN: {await agent.call_tool('get_ebitda_margin', {'ticker': ticker, 'form_type': '10-K'})}', file=sys.stderr, flush=True)
      # print(f'CAPEX REVENUE PCT: {await agent.call_tool('get_capex_pct_revenue', {'ticker': ticker, 'form_type': '10-K'})}', file=sys.stderr, flush=True)
      # print(f'TAX RATE: {await agent.call_tool('get_tax_rate', {'ticker': ticker, 'form_type': '10-K'})}', file=sys.stderr, flush=True)
      # print(f'DEPRECIATION: {await agent.call_tool('get_depreciation', {'ticker': ticker, 'form_type': '10-K'})}', file=sys.stderr, flush=True)

      response = await agent.call_tool('get_disclosures_names', {'ticker': ticker, 'form_type': '10-K'})

      for disclosure in response['disclosure_names']:
        print(await agent.call_tool('extract_disclosure_data', {'ticker': ticker,'disclosure_name': disclosure, 'form_type':'10-K'}))

  asyncio.run(main())
