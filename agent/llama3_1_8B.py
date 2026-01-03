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
          Analyze {ticker} and provide DCF assumptions in JSON format with these exact keys:

          Based on the company's financials and industry, provide:
          - revenue_base: current year revenue in millions
          - revenue_growth: [5-year growth rates as decimals]
          - ebitda_margin: average EBITDA margin as decimal
          - capex_pct_revenue: capex as % of revenue as decimal
          - tax_rate: corporate tax rate as decimal
          - depreciation: depreciation as % of revenue as decimal
          - terminal_growth: long-term growth rate as decimal
          - terminal_multiple: EV/EBITDA exit multiple as number
          - wacc: weighted average cost of capital as decimal
          - revenue_year5: projected year 5 revenue in millions

          Company data: {company_data}

          Return ONLY valid JSON with numeric values. NO calculations, expressions, or formulas inside the JSON.
          """

    response = ollama.chat(
      model=self.model,
      messages=[{'role':'user', 'content': prompt}]
    )

    # Parse JSON from response content
    import json
    try:
      content = response['message']['content']
      # Extract JSON from markdown code blocks if present
      if '```json' in content:
        json_start = content.find('```json') + 7
        json_end = content.find('```', json_start)
        json_str = content[json_start:json_end].strip()
      else:
        json_str = content.strip()

      return json.loads(json_str)
    except (json.JSONDecodeError, KeyError) as e:
      print(f"Error parsing LLM response: {e}")
      print(f"Response content: {content}")
      # Return default assumptions if parsing fails
      return {
        'revenue_base': company_data.get('revenue', 1000) / 1_000_000,  # Convert to millions
        'revenue_growth': [0.05, 0.04, 0.035, 0.03, 0.025],
        'ebitda_margin': 0.25,
        'capex_pct_revenue': 0.03,
        'tax_rate': 0.21,
        'depreciation': 0.02,
        'terminal_growth': 0.025,
        'terminal_multiple': 12.0,
        'wacc': 0.09,
        'revenue_year5': company_data.get('revenue', 1000) * 1.2 / 1_000_000
      }

if __name__ == "__main__":
  pass
