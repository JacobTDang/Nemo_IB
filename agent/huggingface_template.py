from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
import torch
from typing import Dict, Optional
import os, sys

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
