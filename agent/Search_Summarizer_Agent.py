from .ollama_template import OllamaModel
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import json
import sys


class DataPoint(BaseModel):
  metric: str
  value: str
  source: Optional[str] = None
  date: Optional[str] = None


class SummaryResult(BaseModel):
  relevant: bool
  ticker: Optional[str] = None
  data_points: Optional[List[DataPoint]] = None
  key_facts: Optional[List[str]] = None
  sentiment: Optional[str] = None
  data_quality: Optional[str] = None
  reason: Optional[str] = None


class Search_Summarizer_Agent(OllamaModel):
  response_schema = SummaryResult

  def __init__(self, model_name: str = "llama3.1:8b"):
    super().__init__(model_name=model_name)

  def build_system_prompt(self) -> str:
    prompt = """You are a Financial Data Extractor. Your job is to extract useful information from web content.

DEFAULT: relevant = true. This content was found by a targeted financial search, so it is almost always relevant.

Set relevant = false ONLY if the content is:
- A 404 error page, login wall, or cookie notice
- Entirely about an unrelated topic (sports, entertainment, etc.)
- Empty or contains no readable text

RULES:
1. Keep output under 500 tokens
2. Preserve exact numbers (don't paraphrase "$234.5B" as "large revenue")
3. Extract ANY useful information -- numbers, facts, context, opinions
4. Even if content is general market/economic news, extract what's useful for the search intent
5. If content has raw data (SEC filings, financial tables), extract the key metrics

EXTRACT PRIORITY:
1. Numerical metrics (interest rates, yields, growth rates, multiples, prices, revenue, margins)
2. Analyst opinions (ratings, price targets, recommendations)
3. Recent events (earnings, guidance, M&A, product launches, policy changes)
4. Market/economic context (Treasury yields, inflation, Fed policy, sector trends)"""

    return prompt

  def summarize_single(self, content: Dict[str, Any], ticker: str, search_intent: str) -> Dict[str, Any]:
    """Summarize a single scraped content result"""

    if not content.get('success', False):
      return {
        "relevant": False,
        "reason": content.get('error', 'Failed to retrieve'),
        "url": content.get('url', 'unknown')
      }

    raw_content = content.get('content', '')

    if isinstance(raw_content, str) and len(raw_content) > 4000:
      raw_content = raw_content[:4000] + "... [truncated]"
    elif isinstance(raw_content, dict):
      raw_content = json.dumps(raw_content)[:4000]

    user_prompt = f"""Ticker: {ticker}
Search Intent: {search_intent}
Source: {content.get('title', content.get('url', 'N/A'))}

Content:
{raw_content}

Extract ONLY information relevant to {ticker} financial analysis."""

    print(f"  [Summarizer] Processing: {content.get('url', 'unknown')[:60]}...", file=sys.stderr, flush=True)

    response = self.generate_response(user_prompt, self.build_system_prompt())

    try:
      result = self.parse_response(response)
      parsed = result.model_dump()
      parsed['source_url'] = content.get('url', 'unknown')
      parsed['source_title'] = content.get('title', 'unknown')
      return parsed
    except Exception as e:
      print(f"  [Summarizer] Parse failed: {e}", file=sys.stderr, flush=True)
      return {
        "relevant": False,
        "error": f"Parse failed: {e}",
        "source_url": content.get('url', 'unknown'),
        "source_title": content.get('title', 'unknown')
      }

