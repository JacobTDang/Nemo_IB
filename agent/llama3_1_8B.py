import ollama
import asyncio
from typing import Dict, Any, Optional

class FinancialAnalysisAgent:
  def __init__(self, model_name='llama3.1:8b'):
    self.model = model_name

  async def analyze_company(self, ticker: str):
    prompt = f"'analyze {ticker}'s growth prospects..."

    response = ollama.chat(
      model=self.model,
      messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']

  async def generate_assumptions(self, ticker: str, company_data: Dict[str, Any]):
    prompt = f"""
          Analyze {ticker} and provide DCF assumptions in JSON format:

          Based on the company's financials and industry, provide:
          - revenue_growth_rates: [5-year projections as decimals]
          - ebitda_margins: [5-year margin projections as decimals]
          - capex_pct_revenue: [decimal]
          - tax_rate: [decimal]
          - terminal_growth: [decimal]

          Company data: {company_data}

          Return only JSON format, NOT EXPLAINATIONS NEEDED.
          """

    response = ollama.chat(
      model=self.model,
      messages=[{'role':'user', 'content': prompt}]
    )

    return response

if __name__ == "__main__":
  pass
