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
    prompt = """You are a Financial Data Extractor. Distill raw web content into concise structured data with ONLY financially relevant information.

RULES:
1. Keep output under 500 tokens
2. Preserve exact numbers (don't paraphrase "$234.5B" as "large revenue")
3. If content is NOT about the ticker or irrelevant to finance, set relevant to false with a reason
4. Ignore ads, navigation, boilerplate, unrelated news

EXTRACT PRIORITY:
1. Numerical metrics (WACC, growth rates, multiples, prices, revenue, margins)
2. Analyst opinions (ratings, price targets, recommendations)
3. Recent events (earnings, guidance, M&A, product launches)
4. Market context (market share, competitive position)"""

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

  def summarize_tool_outputs(self, tool_outputs: List[Dict], ticker: str) -> List[Dict[str, Any]]:
    """
    Summarize all tool outputs, filtering and condensing search results.

    Args:
      tool_outputs: List of tool results from execution_node
      ticker: The ticker being analyzed

    Returns:
      List of summarized/filtered results ready for analysis agent
    """
    summarized = []

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"SUMMARIZER: Processing {len(tool_outputs)} tool outputs for {ticker}", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)

    for output in tool_outputs:
      tool_name = output.get('tool', '')

      if tool_name in ['get_revenue_base', 'get_ebitda_margin', 'get_capex_pct_revenue',
                       'get_depreciation', 'get_tax_rate', 'extract_8k_events',
                       'get_disclosures_names', 'extract_disclosure_data', 'get_latest_filing']:
        print(f"  [Pass-through] {tool_name}: structured SEC data", file=sys.stderr, flush=True)
        summarized.append({
          "type": "sec_data",
          "tool": tool_name,
          "data": output.get('result', {})
        })

      elif tool_name == 'search':
        search_results = output.get('result', {}).get('search_result', [])
        query_info = output.get('arguments', {}).get('query', {})

        snippets = []
        for item in search_results[:5]:
          title = item.get('title', '')
          snippet = item.get('snippet', '')
          if title or snippet:
            snippets.append({
              "title": title,
              "snippet": snippet,
              "link": item.get('link', '')
            })

        if snippets:
          print(f"  [Search] Kept {len(snippets)}/{len(search_results)} snippets", file=sys.stderr, flush=True)
          summarized.append({
            "type": "search_snippets",
            "ticker": ticker,
            "queries": query_info,
            "results": snippets
          })

      elif 'get_urls_content' in tool_name:
        results = output.get('result', {}).get('results', [])
        print(f"  [Scrape] Processing {len(results)} URLs...", file=sys.stderr, flush=True)

        for item in results:
          if item.get('success') and item.get('content'):
            search_intent = "financial data, metrics, analyst opinions"
            summary = self.summarize_single(item, ticker, search_intent)

            if summary.get('relevant', False):
              print(f"    [OK] Relevant: {item.get('title', 'unknown')[:50]}", file=sys.stderr, flush=True)
              summarized.append({"type": "web_content", **summary})
            else:
              reason = summary.get('reason', summary.get('error', 'not relevant'))
              print(f"    [X] Filtered: {reason[:50]}", file=sys.stderr, flush=True)
          else:
            print(f"    [X] Failed: {item.get('error', 'unknown error')[:50]}", file=sys.stderr, flush=True)

      else:
        print(f"  [Unknown] {tool_name}: passing through", file=sys.stderr, flush=True)
        summarized.append({
          "type": "other",
          "tool": tool_name,
          "data": output.get('result', {})
        })

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"SUMMARIZER COMPLETE: {len(summarized)} items retained", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    return summarized
