from .groq_template import GroqModel
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys


class ArticleAssessment(BaseModel):
  headline: str
  relevant: bool
  sentiment: str    # "bullish" | "bearish" | "neutral"
  impact: str       # "high" | "medium" | "low"
  reason: str


class NewsAnalysisResult(BaseModel):
  overall_sentiment: str    # "bullish" | "bearish" | "neutral" | "mixed"
  sentiment_score: float    # -1.0 to +1.0
  key_themes: List[str]
  article_assessments: List[ArticleAssessment]
  articles_analyzed: int
  articles_relevant: int


IMPACT_WEIGHTS = {"high": 0.3, "medium": 0.15, "low": 0.05}


class News_Processing_Agent(GroqModel):
  response_schema = ArticleAssessment

  def __init__(self, model_name: str = "llama-3.1-8b-instant"):
    super().__init__(model_name=model_name)

  def _build_company_prompt(self, ticker: str) -> str:
    return f"""You are a Financial News Analyst assessing a single news article about {ticker}.

For this article, determine:
1. RELEVANT: Is this article directly relevant to {ticker}'s stock price, business operations, competitive position, or valuation? Mark irrelevant if it only mentions {ticker} in passing or is about an unrelated topic.
2. SENTIMENT: What is the directional signal for {ticker}'s stock?
   - "bullish" = positive catalyst (earnings beat, new product, upgrade, expansion, positive guidance)
   - "bearish" = negative catalyst (earnings miss, lawsuit, downgrade, loss of customer, negative guidance)
   - "neutral" = informational with no clear directional signal
3. IMPACT: How material is this for {ticker}'s valuation?
   - "high" = earnings, M&A, major product launch, regulatory action, guidance change
   - "medium" = analyst opinion, partnership, market share data, management change
   - "low" = general mention, minor update, routine event
4. REASON: 1-2 sentences explaining your assessment.

Be precise. Echo back the exact headline."""

  def _build_market_prompt(self, ticker: str = "") -> str:
    ticker_context = f" Consider implications for {ticker} if applicable." if ticker and ticker != "UNKNOWN" else ""
    return f"""You are a Financial News Analyst assessing a single market/macro news article.

For this article, determine:
1. RELEVANT: Is this article relevant to equity markets or the broader economy? Default to true -- the orchestrator specifically requested market news. Only mark irrelevant if the article is a duplicate, paywall notice, or completely non-financial.
2. SENTIMENT: What is the directional signal for equity markets?
   - "bullish" = dovish Fed, strong economic data, trade deal, stimulus, rate cuts
   - "bearish" = hawkish Fed, weak economic data, trade war, inflation spike, geopolitical risk
   - "neutral" = mixed signals or purely informational
3. IMPACT: How material is this for markets?
   - "high" = Fed decision, CPI/jobs data, tariff announcement, major geopolitical event
   - "medium" = sector rotation, earnings season trends, commodity moves
   - "low" = routine reports, minor policy updates
4. REASON: 1-2 sentences explaining your assessment.{ticker_context}

Be precise. Echo back the exact headline."""

  def assess_article(self, article: Dict[str, Any], system_prompt: str) -> Optional[ArticleAssessment]:
    """Assess a single article. Returns ArticleAssessment or None on parse failure."""
    headline = article.get("headline", "")
    summary = article.get("summary", "")
    source = article.get("source", "")
    dt = article.get("datetime", "")

    user_prompt = f"""Headline: {headline}
Summary: {summary}
Source: {source}
Date: {dt}

Assess this article."""

    # Clear history -- each article is independent
    self.conversatoin_history = []

    response = self.generate_response(prompt=user_prompt, system_prompt=system_prompt)

    try:
      result = self.parse_response(response)
      return result
    except Exception as e:
      print(f"    [X] Parse failed for '{headline[:50]}': {e}", file=sys.stderr, flush=True)
      return None

  def analyze_news(self, articles: List[Dict], ticker: str, tool_name: str) -> Dict[str, Any]:
    """Analyze news articles one by one and produce an aggregated sentiment result.

    Args:
      articles: List of slimmed article dicts (headline, summary, source, datetime, url)
      ticker: Ticker symbol or category for context
      tool_name: "get_company_news" or "get_market_news" -- determines prompt style

    Returns:
      NewsAnalysisResult as a dict
    """
    if not articles:
      return NewsAnalysisResult(
        overall_sentiment="neutral",
        sentiment_score=0.0,
        key_themes=[],
        article_assessments=[],
        articles_analyzed=0,
        articles_relevant=0
      ).model_dump()

    # Choose prompt based on tool
    if tool_name == "get_company_news":
      system_prompt = self._build_company_prompt(ticker)
    else:
      system_prompt = self._build_market_prompt(ticker)

    print(f"\n  [News Agent] Analyzing {len(articles)} articles for {ticker}...", file=sys.stderr, flush=True)

    assessments: List[ArticleAssessment] = []
    raw_score = 0.0
    has_bullish = False
    has_bearish = False

    for i, article in enumerate(articles, 1):
      headline = article.get("headline", "")[:60]
      print(f"    [{i}/{len(articles)}] {headline}...", file=sys.stderr, flush=True, end="")

      result = self.assess_article(article, system_prompt)
      if result is None:
        # Parse failure -- skip this article
        print(" SKIP", file=sys.stderr, flush=True)
        continue

      assessments.append(result)

      if result.relevant:
        weight = IMPACT_WEIGHTS.get(result.impact, 0.05)
        if result.sentiment == "bullish":
          raw_score += weight
          has_bullish = True
          print(f" +{weight:.2f} ({result.impact})", file=sys.stderr, flush=True)
        elif result.sentiment == "bearish":
          raw_score -= weight
          has_bearish = True
          print(f" -{weight:.2f} ({result.impact})", file=sys.stderr, flush=True)
        else:
          print(f" ~0 (neutral)", file=sys.stderr, flush=True)
      else:
        print(" irrelevant", file=sys.stderr, flush=True)

    # Clamp score
    sentiment_score = max(-1.0, min(1.0, raw_score))

    # Determine overall sentiment
    if sentiment_score > 0.2:
      overall_sentiment = "bullish"
    elif sentiment_score < -0.2:
      overall_sentiment = "bearish"
    elif has_bullish and has_bearish:
      overall_sentiment = "mixed"
    else:
      overall_sentiment = "neutral"

    # Extract themes from high/medium impact relevant articles
    key_themes = []
    for a in assessments:
      if a.relevant and a.impact in ("high", "medium"):
        key_themes.append(a.reason)

    articles_relevant = sum(1 for a in assessments if a.relevant)

    print(f"  [News Agent] Done: score={sentiment_score:+.2f}, sentiment={overall_sentiment}, "
          f"{articles_relevant}/{len(assessments)} relevant", file=sys.stderr, flush=True)

    return NewsAnalysisResult(
      overall_sentiment=overall_sentiment,
      sentiment_score=sentiment_score,
      key_themes=key_themes,
      article_assessments=assessments,
      articles_analyzed=len(assessments),
      articles_relevant=articles_relevant
    ).model_dump()
