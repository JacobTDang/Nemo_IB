"""
Test the News Processing Agent in isolation.
Fetches real articles from Finnhub via MCP, then runs them through
the per-article sentiment analysis pipeline.

Usage: python -m testing.test_news_agent
"""
import asyncio
import sys
import json

sys.path.insert(0, ".")

from datetime import datetime, timedelta
from agent.MCP_manager import MCPConnectionManager
from agent.News_Processing_Agent import News_Processing_Agent


async def main():
  ticker = "AAPL"
  today = datetime.now().strftime("%Y-%m-%d")
  thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

  news_agent = News_Processing_Agent("llama3.1:8b")

  async with MCPConnectionManager() as mcp:

    # --- Test 1: Company news ---
    print(f"\n{'='*60}")
    print(f"TEST 1: Company news for {ticker}")
    print(f"{'='*60}")

    result = await mcp.call_tool("get_company_news", {
      "ticker": ticker, "from_date": thirty_days_ago, "to_date": today
    })

    articles = result.get("data", [])
    print(f"Fetched {len(articles)} articles from Finnhub\n")

    # Only run 5 articles for speed
    test_articles = articles[:5]
    analysis = news_agent.analyze_news(test_articles, ticker, "get_company_news")

    print(f"\n--- Company News Result ---")
    print(f"Overall sentiment: {analysis['overall_sentiment']}")
    print(f"Sentiment score: {analysis['sentiment_score']:+.2f}")
    print(f"Articles analyzed: {analysis['articles_analyzed']}")
    print(f"Articles relevant: {analysis['articles_relevant']}")
    print(f"Key themes ({len(analysis['key_themes'])}):")
    for theme in analysis['key_themes']:
      print(f"  - {theme}")
    print(f"\nPer-article assessments:")
    for a in analysis['article_assessments']:
      icon = "+" if a['sentiment'] == 'bullish' else "-" if a['sentiment'] == 'bearish' else "~"
      rel = "REL" if a['relevant'] else "---"
      print(f"  [{icon}] [{rel}] [{a['impact']}] {a['headline'][:70]}")
      print(f"       {a['reason']}")

    # --- Test 2: Market news ---
    print(f"\n{'='*60}")
    print(f"TEST 2: Market news (general)")
    print(f"{'='*60}")

    result = await mcp.call_tool("get_market_news", {"category": "general"})
    articles = result.get("data", [])
    print(f"Fetched {len(articles)} articles from Finnhub\n")

    test_articles = articles[:5]
    analysis = news_agent.analyze_news(test_articles, ticker, "get_market_news")

    print(f"\n--- Market News Result ---")
    print(f"Overall sentiment: {analysis['overall_sentiment']}")
    print(f"Sentiment score: {analysis['sentiment_score']:+.2f}")
    print(f"Articles analyzed: {analysis['articles_analyzed']}")
    print(f"Articles relevant: {analysis['articles_relevant']}")
    print(f"Key themes ({len(analysis['key_themes'])}):")
    for theme in analysis['key_themes']:
      print(f"  - {theme}")
    print(f"\nPer-article assessments:")
    for a in analysis['article_assessments']:
      icon = "+" if a['sentiment'] == 'bullish' else "-" if a['sentiment'] == 'bearish' else "~"
      rel = "REL" if a['relevant'] else "---"
      print(f"  [{icon}] [{rel}] [{a['impact']}] {a['headline'][:70]}")
      print(f"       {a['reason']}")

  print(f"\n{'='*60}")
  print("NEWS AGENT TEST COMPLETE")
  print(f"{'='*60}")


if __name__ == "__main__":
  asyncio.run(main())
