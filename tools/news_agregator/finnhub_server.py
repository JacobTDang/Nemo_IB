"""
Finnhub MCP Server -- 7 tools for market intelligence data.

Provides forward-looking signals: news, insider activity, analyst sentiment,
earnings calendars, peer companies, and key financial metrics.

Entry point: python -m tools.news_agregator.finnhub_server server
"""
from typing import Any, Dict, List
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.news_agregator.finnhub_utils import FinnhubClient, build_envelope


def json_serializer(obj):
  if isinstance(obj, (date, datetime)):
    return obj.isoformat()
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def safe_json_dumps(obj):
  return json.dumps(obj, default=json_serializer)


# Tool descriptions
company_news_description = """Retrieves recent news articles for a specific company from Finnhub.
Should use: When you need current news, press releases, or media coverage for a company to understand recent events, sentiment shifts, or catalysts.
Should NOT use: For historical SEC filing data (use SEC tools) or broad market news (use get_market_news)."""

market_news_description = """Retrieves broad market news by category (general, forex, crypto, merger).
Should use: When you need market-wide context, sector trends, or macroeconomic news that could affect a company's valuation.
Should NOT use: For company-specific news (use get_company_news instead)."""

insider_transactions_description = """Retrieves insider trading activity (Form 4 filings) for a company from Finnhub.
Should use: When analyzing insider sentiment -- whether executives and directors are buying or selling shares. Important signal for investment decisions.
Should NOT use: For institutional holdings or 13F data. This covers only insider (officer/director) transactions."""

earnings_calendar_description = """Retrieves upcoming and recent earnings dates and EPS estimates for companies.
Should use: When you need to know when a company reports earnings, consensus EPS estimates, or to identify earnings catalysts in a date range.
Should NOT use: For actual reported financial results (use SEC tools or get_market_data instead)."""

analyst_recommendations_description = """Retrieves analyst recommendation trends (buy/hold/sell/strong buy/strong sell) over time for a company.
Should use: When you need Wall Street consensus sentiment, analyst rating changes, or to gauge institutional opinion on a stock.
Should NOT use: For price targets or detailed analyst reports. This provides aggregate recommendation counts."""

company_peers_description = """Retrieves a list of peer/comparable company tickers for a given company from Finnhub.
Should use: When you need comparable companies for relative valuation, comp analysis, or to understand a company's competitive landscape.
Should NOT use: If you already know the peer group. This returns Finnhub's pre-computed peer list based on industry classification."""

basic_financials_description = """Retrieves key financial metrics and ratios for a company from Finnhub (52-week high/low, beta, PE, margins, ROE, etc.).
Should use: When you need a broad set of financial ratios and metrics for quick screening or to supplement yfinance data with additional metrics.
Should NOT use: For detailed financial statements or historical data (use SEC tools). This provides current snapshot metrics."""

earnings_surprises_description = """Retrieves historical EPS earnings surprises (actual vs. consensus estimate) for the last 12 reported quarters.
Includes beat/miss rates and average surprise percent to assess management execution reliability.
Should use: When assessing earnings quality, management guidance credibility, or to anchor forward EPS assumptions.
Should NOT use: For forward EPS estimates (use get_forward_estimates) or full income statements (use get_financial_statements)."""

forward_estimates_description = """Retrieves Wall Street analyst consensus estimates for EPS, Revenue, and EBITDA for the next 4-6 quarters.
Combines three Finnhub endpoints (eps-estimate, revenue-estimate, ebitda-estimate) in one call.
Should use: When building DCF assumptions from consensus forecasts, or comparing your projections to the street consensus.
Should NOT use: For historical reported results (use get_earnings_surprises or SEC tools)."""


financial_statements_description = """Retrieves standardized historical financial statements for a company from Finnhub.
Parameters:
  - statement: 'ic' (income statement), 'bs' (balance sheet), 'cf' (cash flow)
  - freq: 'annual' (last 5 years) or 'quarterly' (last 8 quarters)
Should use: When you need full historical financials for trend analysis, margin expansion/compression, or to supplement SEC XBRL data.
Should NOT use: For forward estimates (use get_forward_estimates) or key ratios (use get_basic_financials)."""

company_profile_description = """Retrieves company profile: name, exchange, sector, industry, country, employee count, IPO date, and business description.
Should use: At the start of any analysis to understand the business, sector classification, and company fundamentals.
Should NOT use: For financial data (use get_basic_financials or get_financial_statements)."""

insider_sentiment_description = """Retrieves the monthly share purchase ratio (MSPR) insider sentiment signal for a company.
MSPR aggregates insider buy vs. sell activity into a single signal (-1 to +1) by month.
Should use: As a quick aggregate insider signal to confirm or contradict the detailed get_insider_transactions data.
Should NOT use: For individual transaction details (use get_insider_transactions)."""

def _condense_insider_data(raw: Dict[str, Any]) -> Dict[str, Any]:
  """Aggregate raw insider transactions into a compact signal summary.

  Input: Finnhub's {"data": [list of transactions]} where each has
  name, share, change, transactionDate, transactionCode (P=purchase, S=sale, etc.)

  Returns a condensed dict with totals, top insiders, recency buckets, and signal.
  """
  transactions = raw.get("data", [])
  if not transactions:
    return {"total_bought": 0, "total_sold": 0, "net_shares": 0,
            "buy_count": 0, "sell_count": 0, "top_insiders": [],
            "recent_30d": {"bought": 0, "sold": 0, "net": 0},
            "recent_90d": {"bought": 0, "sold": 0, "net": 0},
            "prior_90d": {"bought": 0, "sold": 0, "net": 0},
            "prior_period_avg_per_90d_sold": None,
            "current_vs_baseline_ratio": None,
            "period_start": None, "period_end": None,
            "signal": "neutral"}

  total_bought = 0
  total_sold = 0
  buy_count = 0
  sell_count = 0

  # Per-insider accumulator: {name: {"net": int, "count": int}}
  insider_activity = defaultdict(lambda: {"net": 0, "count": 0})

  # Recency buckets. prior_90 covers days 91-180 — the immediate baseline
  # comparison window for the current 90 days.
  now = datetime.now(timezone.utc).date()
  r30 = {"bought": 0, "sold": 0}
  r90 = {"bought": 0, "sold": 0}
  prior_90 = {"bought": 0, "sold": 0}

  for txn in transactions:
    code = txn.get("transactionCode", "")
    change = txn.get("change", 0) or 0
    name = txn.get("name", "Unknown")
    txn_date_str = txn.get("transactionDate", "")

    # Parse transaction date for recency
    txn_date = None
    if txn_date_str:
      try:
        txn_date = datetime.strptime(txn_date_str, "%Y-%m-%d").date()
      except ValueError:
        pass

    abs_change = abs(change)
    days_ago = (now - txn_date).days if txn_date else None

    if code == "P":  # Purchase
      total_bought += abs_change
      buy_count += 1
      insider_activity[name]["net"] += abs_change
      insider_activity[name]["count"] += 1
      if days_ago is not None:
        if days_ago <= 30:
          r30["bought"] += abs_change
        if days_ago <= 90:
          r90["bought"] += abs_change
        if 90 < days_ago <= 180:
          prior_90["bought"] += abs_change

    elif code in ("S", "F"):  # Sale or tax withholding
      total_sold += abs_change
      sell_count += 1
      insider_activity[name]["net"] -= abs_change
      insider_activity[name]["count"] += 1
      if days_ago is not None:
        if days_ago <= 30:
          r30["sold"] += abs_change
        if days_ago <= 90:
          r90["sold"] += abs_change
        if 90 < days_ago <= 180:
          prior_90["sold"] += abs_change

  # Top 5 insiders by absolute net activity
  sorted_insiders = sorted(
    insider_activity.items(), key=lambda x: abs(x[1]["net"]), reverse=True
  )[:5]
  top_insiders = [
    {"name": name, "net_shares": data["net"], "transaction_count": data["count"]}
    for name, data in sorted_insiders
  ]

  net_shares = total_bought - total_sold

  # Signal determination
  if net_shares > 0:
    signal = "net_buying"
  elif net_shares < 0:
    signal = "net_selling"
  else:
    signal = "neutral"

  # Derive the actual date range present in the data so downstream agents
  # don't fabricate qualifiers like "since Q1 2024". Finnhub returns ~1
  # year of transactions but doesn't guarantee a fixed window.
  valid_dates = []
  for txn in transactions:
    s = txn.get("transactionDate", "")
    if s:
      try:
        valid_dates.append(datetime.strptime(s, "%Y-%m-%d").date())
      except ValueError:
        continue
  period_start = min(valid_dates).isoformat() if valid_dates else None
  period_end = max(valid_dates).isoformat() if valid_dates else None

  # Baseline: extrapolate avg shares sold per 90 days from the pre-recent-90
  # window, so the consumer can ground claims like "loud selling" in a ratio
  # rather than vibes. Programmatic 10b5-1 selling is steady — only deviations
  # from baseline carry signal.
  prior_period_avg_per_90d_sold = None
  current_vs_baseline_ratio = None
  if len(valid_dates) >= 2:
    total_period_days = (max(valid_dates) - min(valid_dates)).days
    prior_days_covered = total_period_days - 90
    if prior_days_covered >= 30:
      prior_window_sold = total_sold - r90["sold"]
      # Scale prior-window sold to a 90-day equivalent
      prior_period_avg_per_90d_sold = round(prior_window_sold * 90 / prior_days_covered, 2)
      if prior_period_avg_per_90d_sold > 0:
        current_vs_baseline_ratio = round(r90["sold"] / prior_period_avg_per_90d_sold, 3)

  return {
    "total_bought": total_bought,
    "total_sold": total_sold,
    "net_shares": net_shares,
    "buy_count": buy_count,
    "sell_count": sell_count,
    "top_insiders": top_insiders,
    "recent_30d": {"bought": r30["bought"], "sold": r30["sold"], "net": r30["bought"] - r30["sold"]},
    "recent_90d": {"bought": r90["bought"], "sold": r90["sold"], "net": r90["bought"] - r90["sold"]},
    "prior_90d": {"bought": prior_90["bought"], "sold": prior_90["sold"],
                  "net": prior_90["bought"] - prior_90["sold"]},
    "prior_period_avg_per_90d_sold": prior_period_avg_per_90d_sold,
    "current_vs_baseline_ratio": current_vs_baseline_ratio,
    "period_start": period_start,
    "period_end": period_end,
    "signal": signal
  }


def _condense_recommendations(raw: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Aggregate raw analyst recommendation periods into a compact consensus summary.

  Input: Finnhub returns a list of monthly periods:
  [{"buy": 20, "hold": 5, "sell": 2, "strongBuy": 10, "strongSell": 1, "period": "2025-01-01"}, ...]

  Returns condensed dict with latest/prior periods, consensus, trend, and total analysts.
  """
  # Source attribution surfaces methodology so Bull/Bear agents don't
  # conflict with externally-cited analyst counts (Yahoo / TipRanks
  # de-duplicate; Finnhub counts each firm-rating row).
  _SOURCE = "Finnhub /stock/recommendation"
  _METHODOLOGY = (
    "Counts reflect Finnhub's aggregated firm-rating buckets for the "
    "given period and may exceed the number of distinct active analysts "
    "reported by other sources (e.g., Yahoo, TipRanks). Use with that "
    "caveat when comparing to externally-cited consensus counts."
  )

  if not raw:
    return {"latest": None, "prior": None, "consensus": "unknown",
            "trend": "unknown", "total_analysts": 0,
            "source": _SOURCE, "methodology_note": _METHODOLOGY}

  def _normalize_period(period_data: Dict) -> Dict[str, Any]:
    return {
      "strong_buy": period_data.get("strongBuy", 0),
      "buy": period_data.get("buy", 0),
      "hold": period_data.get("hold", 0),
      "sell": period_data.get("sell", 0),
      "strong_sell": period_data.get("strongSell", 0),
      "period": period_data.get("period", "")
    }

  latest = _normalize_period(raw[0])
  prior = _normalize_period(raw[1]) if len(raw) > 1 else None

  # Consensus = category with most votes in latest period
  categories = {
    "strong_buy": latest["strong_buy"],
    "buy": latest["buy"],
    "hold": latest["hold"],
    "sell": latest["sell"],
    "strong_sell": latest["strong_sell"]
  }
  consensus = max(categories, key=categories.get)

  total_analysts = sum(categories.values())

  # Trend = compare bullish sentiment (buy + strong_buy) between latest and prior
  if prior:
    latest_bullish = latest["strong_buy"] + latest["buy"]
    prior_bullish = prior["strong_buy"] + prior["buy"]
    if latest_bullish > prior_bullish:
      trend = "upgrading"
    elif latest_bullish < prior_bullish:
      trend = "downgrading"
    else:
      trend = "stable"
  else:
    trend = "unknown"

  return {
    "latest": latest,
    "prior": prior,
    "consensus": consensus,
    "trend": trend,
    "total_analysts": total_analysts,
    "source": _SOURCE,
    "methodology_note": _METHODOLOGY,
  }


def _condense_earnings_calendar(raw: Dict[str, Any]) -> Dict[str, Any]:
  """Aggregate raw earnings calendar into a date-grouped summary.

  Input: Finnhub's {"earningsCalendar": [list of events]} where each has
  date, symbol, epsEstimate, revenueEstimate, hour, quarter, year, epsActual, revenueActual.

  Returns condensed dict with total count, per-date counts, and a capped event list
  with only the essential fields.
  """
  events = raw.get("earningsCalendar", [])
  if not events:
    return {"total_companies": 0, "by_date": [], "events": []}

  # Group by date for summary counts
  date_counts: Dict[str, int] = {}
  for event in events:
    d = event.get("date", "unknown")
    date_counts[d] = date_counts.get(d, 0) + 1

  by_date = [{"date": d, "count": c} for d, c in sorted(date_counts.items())]

  # Slim events: drop nulls and keep only useful fields, cap at 15
  slimmed_events = []
  for event in events[:15]:
    slim = {"symbol": event.get("symbol", ""), "date": event.get("date", "")}
    if event.get("epsEstimate") is not None:
      slim["eps_estimate"] = event["epsEstimate"]
    if event.get("revenueEstimate") is not None:
      slim["revenue_estimate"] = event["revenueEstimate"]
    if event.get("epsActual") is not None:
      slim["eps_actual"] = event["epsActual"]
    if event.get("revenueActual") is not None:
      slim["revenue_actual"] = event["revenueActual"]
    if event.get("hour"):
      slim["hour"] = event["hour"]
    slimmed_events.append(slim)

  return {
    "total_companies": len(events),
    "by_date": by_date,
    "events": slimmed_events
  }


# Key metrics an IB analyst actually needs from Finnhub's 132-metric dump
KEY_METRICS = {
  # Valuation
  'peTTM', 'forwardPE', 'pegTTM', 'evEbitdaTTM', 'evRevenueTTM',
  'pbQuarterly', 'pfcfShareTTM', 'psTTM',
  # Profitability
  'grossMarginTTM', 'operatingMarginTTM', 'netProfitMarginTTM',
  'roeTTM', 'roaTTM', 'roiTTM',
  # Growth
  'epsGrowthTTMYoy', 'epsGrowth5Y', 'revenueGrowthTTMYoy', 'revenueGrowth5Y',
  'ebitdaCagr5Y', 'revenueGrowthQuarterlyYoy',
  # Per-share
  'epsTTM', 'bookValuePerShareQuarterly', 'currentDividendYieldTTM',
  'dividendPerShareTTM', 'cashFlowPerShareTTM', 'revenuePerShareTTM',
  # Leverage & liquidity
  'currentRatioQuarterly', 'quickRatioQuarterly',
  'totalDebt/totalEquityQuarterly', 'longTermDebt/equityQuarterly',
  'netInterestCoverageTTM',
  # Size & risk
  'marketCapitalization', 'enterpriseValue', 'beta',
  # Price context
  '52WeekHigh', '52WeekLow', '52WeekHighDate', '52WeekLowDate',
}


def _condense_basic_financials(raw: Dict[str, Any]) -> Dict[str, Any]:
  """Filter Finnhub's 132 metrics + historical series down to IB-essential metrics.

  Keeps ~35 key metrics, drops the massive 'series' section entirely.
  """
  metrics = raw.get("metric", {})
  if not metrics:
    return raw

  filtered = {k: v for k, v in metrics.items() if k in KEY_METRICS and v is not None}

  return {
    "metric": filtered,
    "metric_count": len(filtered)
  }


def _condense_earnings_surprises(raw: List[Dict[str, Any]], limit: int = 12) -> Dict[str, Any]:
  """Condense historical EPS surprises into beat/miss summary.

  Input: Finnhub list of {actual, estimate, surprise, surprisePercent, period, year, quarter}
  Returns: per-quarter table, beat_count, miss_count, avg_surprise_pct, beat_rate_pct
  """
  if not isinstance(raw, list) or not raw:
    return {"quarters": [], "beat_count": 0, "miss_count": 0, "avg_surprise_pct": None}

  quarters = []
  beat_count = 0
  miss_count = 0
  surprise_pcts = []

  for item in raw[:limit]:
    actual = item.get("actual")
    estimate = item.get("estimate")
    surprise_pct = item.get("surprisePercent")

    entry = {
      "period": item.get("period", ""),
      "year": item.get("year"),
      "quarter": item.get("quarter"),
      "actual_eps": actual,
      "estimate_eps": estimate,
    }
    if surprise_pct is not None:
      entry["surprise_pct"] = round(surprise_pct, 2)
      surprise_pcts.append(surprise_pct)
      if surprise_pct > 0:
        beat_count += 1
        entry["result"] = "beat"
      elif surprise_pct < 0:
        miss_count += 1
        entry["result"] = "miss"
      else:
        entry["result"] = "inline"
    quarters.append(entry)

  result = {
    "quarters": quarters,
    "beat_count": beat_count,
    "miss_count": miss_count,
    "total_periods": len(quarters),
  }
  if surprise_pcts:
    result["avg_surprise_pct"] = round(sum(surprise_pcts) / len(surprise_pcts), 2)
    result["beat_rate_pct"] = round(beat_count / max(len(quarters), 1) * 100, 1)
  return result


def _condense_forward_estimates(eps_raw: Any, rev_raw: Any, ebitda_raw: Any) -> Dict[str, Any]:
  """Combine EPS, Revenue, and EBITDA forward estimates into one compact structure.

  Input: three Finnhub responses from eps-estimate, revenue-estimate, ebitda-estimate.
  Revenue and EBITDA are returned in raw USD (billions scale applied here for readability).
  """
  result = {}

  def _extract(raw: Any, avg_key: str, high_key: str, low_key: str, scale: float = 1.0) -> Dict:
    if not isinstance(raw, dict) or "data" not in raw:
      return {"error": "no data"}
    periods = []
    for item in (raw["data"] or [])[:6]:
      avg = item.get(avg_key)
      high = item.get(high_key)
      low = item.get(low_key)
      n = item.get("numberAnalysts")
      entry = {"period": item.get("period", "")}
      if avg is not None:
        entry["avg"] = round(avg / scale, 4) if scale != 1.0 else avg
      if high is not None:
        entry["high"] = round(high / scale, 4) if scale != 1.0 else high
      if low is not None:
        entry["low"] = round(low / scale, 4) if scale != 1.0 else low
      if n is not None:
        entry["analysts"] = n
      periods.append(entry)
    return {"periods": periods}

  BILLION = 1e9
  result["eps"] = _extract(eps_raw, "epsAvg", "epsHigh", "epsLow")
  result["revenue_B"] = _extract(rev_raw, "revenueAvg", "revenueHigh", "revenueLow", scale=BILLION)
  result["ebitda_B"] = _extract(ebitda_raw, "ebitdaAvg", "ebitdaHigh", "ebitdaLow", scale=BILLION)
  return result


def _yf_forward_estimates(ticker: str) -> Dict[str, Any]:
  """yfinance fallback for forward estimates. Synchronous — must be awaited
  via asyncio.to_thread by the caller.

  Returns the same shape as `_condense_forward_estimates` so consumers can
  swap sub-fields when Finnhub returns 'no data' (free-tier 403). Each sub-
  field is tagged with `_source: yfinance_fallback` (or `_inferred` for the
  EBITDA case, which yfinance does not surface natively and is derived from
  revenue * info['ebitdaMargins']).
  """
  import yfinance as yf

  BILLION = 1e9
  out = {
    "eps": {"error": "no yfinance data"},
    "revenue_B": {"error": "no yfinance data"},
    "ebitda_B": {"error": "no yfinance equivalent"},
  }

  def _is_num(x):
    return x is not None and x == x  # NaN check via self-equality

  def _periods_from_df(df, scale: float):
    periods = []
    for label, row in df.iterrows():
      entry = {"period": str(label)}
      avg = row.get("avg")
      low = row.get("low")
      high = row.get("high")
      n = row.get("numberOfAnalysts")
      if _is_num(avg):
        entry["avg"] = round(float(avg) / scale, 4) if scale != 1.0 else float(avg)
      if _is_num(high):
        entry["high"] = round(float(high) / scale, 4) if scale != 1.0 else float(high)
      if _is_num(low):
        entry["low"] = round(float(low) / scale, 4) if scale != 1.0 else float(low)
      if _is_num(n):
        entry["analysts"] = int(n)
      periods.append(entry)
    return periods

  try:
    t = yf.Ticker(ticker)
  except Exception as exc:
    err = f"yfinance Ticker init failed: {type(exc).__name__}: {exc}"
    return {k: {"error": err} for k in out}

  try:
    eps_df = t.earnings_estimate
    if eps_df is not None and not eps_df.empty:
      out["eps"] = {"periods": _periods_from_df(eps_df, scale=1.0),
                    "_source": "yfinance_fallback"}
  except Exception as exc:
    out["eps"] = {"error": f"yfinance eps: {type(exc).__name__}: {exc}"}

  try:
    rev_df = t.revenue_estimate
    if rev_df is not None and not rev_df.empty:
      out["revenue_B"] = {"periods": _periods_from_df(rev_df, scale=BILLION),
                          "_source": "yfinance_fallback"}
  except Exception as exc:
    out["revenue_B"] = {"error": f"yfinance revenue: {type(exc).__name__}: {exc}"}

  # EBITDA inferred from revenue * info['ebitdaMargins'].
  # Order matters: t.info is the slow call (can hang 10-30s under Yahoo
  # throttling), so skip it entirely when revenue failed — without revenue
  # periods the inference can't produce anything anyway.
  rev_periods = out["revenue_B"].get("periods") if isinstance(out["revenue_B"], dict) else None
  if rev_periods:
    try:
      margin = t.info.get("ebitdaMargins")
      if margin:
        ebitda_periods = []
        for p in rev_periods:
          e = {"period": p["period"]}
          for k in ("avg", "low", "high"):
            if k in p:
              e[k] = round(p[k] * margin, 4)
          if "analysts" in p:
            e["analysts"] = p["analysts"]
          ebitda_periods.append(e)
        out["ebitda_B"] = {"periods": ebitda_periods,
                           "_source": "yfinance_fallback_inferred",
                           "_inferred_margin": round(float(margin), 4)}
    except Exception as exc:
      out["ebitda_B"] = {"error": f"yfinance ebitda inference: {type(exc).__name__}: {exc}"}

  return out


def _yf_financial_statements(ticker: str, statement: str, freq: str) -> Dict[str, Any]:
  """yfinance fallback for `get_financial_statements`. Used when Finnhub
  returns HTTP 403 (free-tier scope) or an unrecognized response.

  Maps yfinance DataFrame row labels to the same camelCase keys that
  `_condense_financial_statements` produces from Finnhub's standardized
  response so downstream consumers don't need to branch on source. Tags
  the result with `_source: yfinance_fallback`.

  Synchronous — must be awaited via `asyncio.to_thread` by the caller.
  """
  import yfinance as yf
  import pandas as pd

  attr_map = {
    ("ic", "annual"):     "income_stmt",
    ("ic", "quarterly"):  "quarterly_income_stmt",
    ("cf", "annual"):     "cashflow",
    ("cf", "quarterly"):  "quarterly_cashflow",
    ("bs", "annual"):     "balance_sheet",
    ("bs", "quarterly"):  "quarterly_balance_sheet",
  }
  attr = attr_map.get((statement, freq))
  if not attr:
    return {"statement": statement, "freq": freq,
            "error": f"unsupported statement/freq: {statement}/{freq}"}

  LABEL_MAP = {
    "ic": {
      "Total Revenue":      "revenue",
      "Cost Of Revenue":    "costOfRevenue",
      "Gross Profit":       "grossProfit",
      "Operating Expense":  "operatingExpense",
      "Operating Income":   "operatingIncome",
      "EBITDA":             "ebitda",
      "EBIT":               "ebit",
      "Net Income":         "netIncome",
      "Basic EPS":          "eps",
      "Diluted EPS":        "epsDiluted",
    },
    "cf": {
      "Operating Cash Flow":         "operatingCashFlow",
      "Capital Expenditure":         "capitalExpenditures",
      "Free Cash Flow":              "freeCashFlow",
      "Cash Dividends Paid":         "dividendsPaid",
      "Common Stock Dividend Paid":  "dividendsPaid",
      "Repurchase Of Capital Stock": "repurchaseOfCapitalStock",
      "Changes In Cash":             "netChangeInCash",
    },
    "bs": {
      "Total Assets":                            "totalAssets",
      "Current Assets":                          "totalCurrentAssets",
      "Cash And Cash Equivalents":               "cashAndEquivalents",
      "Cash Cash Equivalents And Short Term Investments": "cashAndEquivalents",
      "Other Short Term Investments":            "shortTermInvestments",
      "Short Term Investments":                  "shortTermInvestments",
      "Total Liabilities Net Minority Interest": "totalLiabilities",
      "Current Liabilities":                     "totalCurrentLiabilities",
      "Long Term Debt":                          "longTermDebt",
      "Current Debt":                            "shortTermDebt",
      "Total Debt":                              "totalDebt",
      "Total Equity Gross Minority Interest":    "totalEquity",
      "Stockholders Equity":                     "stockholdersEquity",
      "Goodwill":                                "goodwill",
      "Other Intangible Assets":                 "intangibleAssets",
    },
  }
  mapping = LABEL_MAP.get(statement, {})

  try:
    t = yf.Ticker(ticker)
    df = getattr(t, attr, None)
  except Exception as exc:
    return {"statement": statement, "freq": freq,
            "error": f"yfinance Ticker {attr} failed: {type(exc).__name__}: {exc}"}

  if df is None or df.empty:
    return {"statement": statement, "freq": freq,
            "error": "yfinance returned empty statement"}

  cap = 5 if freq == "annual" else 8
  periods = []
  for col in list(df.columns)[:cap]:
    period_label = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
    row = {"period": period_label}
    for yf_label, key in mapping.items():
      if yf_label in df.index:
        v = df.loc[yf_label, col]
        if pd.notna(v):
          # yfinance reports flows positive; CF statement values that are
          # outflows (capex, dividends, buybacks) come back negative in the
          # yfinance shape (matching the cash-flow-statement convention).
          # Convert to positive for buybacks/dividends so downstream consumers
          # match Finnhub's convention (Finnhub returns these as positive
          # outflow values in /stock/financials).
          if key in ("capitalExpenditures", "dividendsPaid",
                     "repurchaseOfCapitalStock") and v < 0:
            v = abs(v)
          row[key] = float(v)
    periods.append(row)

  return {
    "statement": statement, "freq": freq, "periods": periods,
    "count": len(periods), "_source": "yfinance_fallback",
  }


def _condense_financial_statements(raw: Dict[str, Any], statement: str, freq: str) -> Dict[str, Any]:
  """Extract key line items from Finnhub standardized financial statements.

  Handles both common response formats:
  - Format A: {"financials": {"annual": {"ic": [{period, revenue, ...}, ...]}}}
  - Format B: {"data": [{"endDate": ..., "report": {"ic": [{concept, value}, ...]}}]}

  Returns last 5 annual or 8 quarterly periods with key fields only.
  """
  # Fields to keep for each statement type (Finnhub standardized camelCase)
  KEEP = {
    "ic": {"revenue", "costOfRevenue", "grossProfit", "operatingExpense",
           "operatingIncome", "ebitda", "ebit", "netIncome",
           "eps", "epsDiluted", "period"},
    "bs": {"totalAssets", "totalCurrentAssets", "cashAndEquivalents",
           "shortTermInvestments", "totalLiabilities", "totalCurrentLiabilities",
           "longTermDebt", "shortTermDebt", "totalDebt",
           "totalEquity", "stockholdersEquity", "goodwill",
           "intangibleAssets", "period"},
    "cf": {"operatingCashFlow", "capitalExpenditures", "freeCashFlow",
           "dividendsPaid", "repurchaseOfCapitalStock", "netChangeInCash", "period"},
  }
  keep_fields = KEEP.get(statement, set())
  cap = 5 if freq == "annual" else 8

  # Try Format A
  financials = raw.get("financials", {})
  freq_data = financials.get("annual" if freq == "annual" else "quarterly", {})
  periods_raw = freq_data.get(statement, [])

  if periods_raw and isinstance(periods_raw, list):
    periods = []
    for p in periods_raw[:cap]:
      if isinstance(p, dict):
        filtered = {k: v for k, v in p.items() if k in keep_fields and v is not None}
        if filtered:
          periods.append(filtered)
    if periods:
      return {"statement": statement, "freq": freq, "periods": periods, "count": len(periods)}

  # Try Format B (Financials As Reported style)
  data_list = raw.get("data", [])
  if isinstance(data_list, list) and data_list:
    periods = []
    for item in data_list[:cap]:
      report = item.get("report", {})
      stmt_items = report.get(statement, [])
      period_label = item.get("endDate") or item.get("period", "")
      if isinstance(stmt_items, list):
        row = {"period": period_label}
        for li in stmt_items:
          if isinstance(li, dict):
            concept = li.get("concept", "").lower()
            value = li.get("value")
            label = li.get("label", "")
            key = concept or label.replace(" ", "").replace("/", "_")
            if key and value is not None:
              row[key] = value
        if row:
          periods.append(row)
    if periods:
      return {"statement": statement, "freq": freq, "periods": periods, "count": len(periods)}

  # Neither format recognized -- return raw condensed
  return {"statement": statement, "freq": freq, "raw_preview": str(raw)[:500], "error": "Unrecognized response format"}


def _condense_insider_sentiment(raw: Dict[str, Any]) -> Dict[str, Any]:
  """Condense Finnhub insider sentiment (MSPR) into a compact monthly summary.

  Input: {"data": [{"year": int, "month": int, "mspr": float, "change": int, "msprChange": float}]}
  MSPR: Monthly Share Purchase Ratio. +1 = all insiders buying, -1 = all insiders selling.
  Returns: last 6 months of MSPR with overall signal.
  """
  data = raw.get("data", []) if isinstance(raw, dict) else []
  if not data:
    return {"months": [], "signal": "neutral", "avg_mspr": None}

  # Sort by year desc, month desc and take last 6
  sorted_data = sorted(data, key=lambda x: (x.get("year", 0), x.get("month", 0)), reverse=True)[:6]

  months = []
  mspr_values = []
  for item in sorted_data:
    mspr = item.get("mspr")
    months.append({
      "year": item.get("year"),
      "month": item.get("month"),
      "mspr": round(mspr, 4) if mspr is not None else None,
      "change_shares": item.get("change"),
    })
    if mspr is not None:
      mspr_values.append(mspr)

  avg_mspr = round(sum(mspr_values) / len(mspr_values), 4) if mspr_values else None

  if avg_mspr is not None:
    if avg_mspr > 0.2:
      signal = "net_buying"
    elif avg_mspr < -0.2:
      signal = "net_selling"
    else:
      signal = "neutral"
  else:
    signal = "neutral"

  return {"months": months, "signal": signal, "avg_mspr": avg_mspr}


def _slim_articles(articles: List[Dict[str, Any]], cap: int = 20) -> List[Dict[str, Any]]:
  """Strip news articles to essential fields and cap count.

  Keeps: headline, summary, source, datetime (ISO), url
  Drops: id, image, related, category
  """
  slimmed = []
  for article in articles[:cap]:
    dt = article.get("datetime")
    if isinstance(dt, (int, float)):
      dt = datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
    slimmed.append({
      "headline": article.get("headline", ""),
      "summary": article.get("summary", ""),
      "source": article.get("source", ""),
      "datetime": dt,
      "url": article.get("url", "")
    })
  return slimmed


def _warm_yfinance_session() -> None:
  """Pre-fetch Yahoo's crumb/cookie so the first real yfinance call in this
  process isn't blocked on the 30+ second handshake. fast_info is a lazy
  accessor and does NOT trigger the crumb fetch — a small yf.download() is
  the cheapest reliable way to force the cookie handshake. Failures are
  silent: the subsequent fallback path will still time out gracefully if
  Yahoo refuses us entirely."""
  try:
    import yfinance as yf
    _ = yf.download("AAPL", period="5d", progress=False, auto_adjust=True)
  except Exception:
    pass


class FinnhubServer:
  def __init__(self):
    self.server = Server("finnhub")
    self.client = FinnhubClient()
    self._setup_handlers()
    # Warm yfinance in a daemon thread so the first get_forward_estimates
    # fallback call doesn't pay the cold-start tax (~30s for Yahoo crumb).
    import threading
    threading.Thread(target=_warm_yfinance_session, daemon=True).start()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="get_company_news",
          description=company_news_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL)"
              },
              "from_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format"
              },
              "to_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format"
              }
            },
            "required": ["ticker", "from_date", "to_date"]
          }
        ),
        Tool(
          name="get_market_news",
          description=market_news_description,
          inputSchema={
            "type": "object",
            "properties": {
              "category": {
                "type": "string",
                "description": "News category: general, forex, crypto, or merger",
                "enum": ["general", "forex", "crypto", "merger"]
              }
            },
            "required": ["category"]
          }
        ),
        Tool(
          name="get_insider_transactions",
          description=insider_transactions_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL)"
              }
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_earnings_calendar",
          description=earnings_calendar_description,
          inputSchema={
            "type": "object",
            "properties": {
              "from_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format"
              },
              "to_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format"
              }
            },
            "required": ["from_date", "to_date"]
          }
        ),
        Tool(
          name="get_analyst_recommendations",
          description=analyst_recommendations_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL)"
              }
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_analyst_revisions_history",
          description=(
            "Full time series of analyst rating-bucket counts (strong buy / buy / hold / sell / strong sell) over the last N months. "
            "Detects upgrade/downgrade momentum invisible in a single-period snapshot. Returns per-month counts, net_bullish score "
            "((strong_buy+buy) - (sell+strong_sell)), 1mo/3mo/6mo deltas, and a momentum signal classifier "
            "(upgrading_strong / upgrading / neutral / downgrading / downgrading_strong). "
            "Use to detect institutional re-rating before it shows up in price."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol"},
              "lookback_months": {"type": "integer", "description": "Months of history to include", "default": 12}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_company_peers",
          description=company_peers_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL)"
              }
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_basic_financials",
          description=basic_financials_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL)"
              }
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_earnings_surprises",
          description=earnings_surprises_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_forward_estimates",
          description=forward_estimates_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_financial_statements",
          description=financial_statements_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
              "statement": {
                "type": "string",
                "description": "Statement type: 'ic' (income), 'bs' (balance sheet), 'cf' (cash flow)",
                "enum": ["ic", "bs", "cf"]
              },
              "freq": {
                "type": "string",
                "description": "Frequency: 'annual' or 'quarterly'",
                "enum": ["annual", "quarterly"]
              }
            },
            "required": ["ticker", "statement", "freq"]
          }
        ),
        Tool(
          name="get_company_profile",
          description=company_profile_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
          }
        ),
        Tool(
          name="get_insider_sentiment",
          description=insider_sentiment_description,
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
              "from_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format (default: 1 year ago)"
              },
              "to_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format (default: today)"
              }
            },
            "required": ["ticker"]
          }
        ),
      ]

    @self.server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
      match name:
        case "get_company_news":
          return await parent.get_company_news(
            arguments["ticker"], arguments["from_date"], arguments["to_date"]
          )
        case "get_market_news":
          return await parent.get_market_news(arguments["category"])
        case "get_insider_transactions":
          return await parent.get_insider_transactions(arguments["ticker"])
        case "get_earnings_calendar":
          return await parent.get_earnings_calendar(
            arguments["from_date"], arguments["to_date"]
          )
        case "get_analyst_recommendations":
          return await parent.get_analyst_recommendations(arguments["ticker"])
        case "get_analyst_revisions_history":
          return await parent.get_analyst_revisions_history(arguments["ticker"], arguments.get("lookback_months", 12))
        case "get_company_peers":
          return await parent.get_company_peers(arguments["ticker"])
        case "get_basic_financials":
          return await parent.get_basic_financials(arguments["ticker"])
        case "get_earnings_surprises":
          return await parent.get_earnings_surprises(arguments["ticker"])
        case "get_forward_estimates":
          return await parent.get_forward_estimates(arguments["ticker"])
        case "get_financial_statements":
          return await parent.get_financial_statements(
            arguments["ticker"], arguments["statement"], arguments.get("freq", "annual")
          )
        case "get_company_profile":
          return await parent.get_company_profile(arguments["ticker"])
        case "get_insider_sentiment":
          return await parent.get_insider_sentiment(
            arguments["ticker"],
            arguments.get("from_date"),
            arguments.get("to_date")
          )
        case _:
          return [TextContent(
            type="text",
            text=safe_json_dumps({"error": f"Unknown tool: {name}"})
          )]

  # -- Tool implementations --

  async def get_company_news(self, ticker: str, from_date: str, to_date: str) -> List[TextContent]:
    result = await self.client.get("/company-news", {
      "symbol": ticker, "from": from_date, "to": to_date
    })
    if isinstance(result, list):
      result = _slim_articles(result, cap=20)
    envelope = build_envelope(result, ticker, "get_company_news")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_market_news(self, category: str) -> List[TextContent]:
    result = await self.client.get("/news", {"category": category})
    if isinstance(result, list):
      result = _slim_articles(result, cap=20)
    envelope = build_envelope(result, category, "get_market_news")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_insider_transactions(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/insider-transactions", {"symbol": ticker})
    condensed = _condense_insider_data(result)
    envelope = build_envelope(condensed, ticker, "get_insider_transactions")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_earnings_calendar(self, from_date: str, to_date: str) -> List[TextContent]:
    result = await self.client.get("/calendar/earnings", {
      "from": from_date, "to": to_date
    })
    condensed = _condense_earnings_calendar(result) if isinstance(result, dict) else result
    envelope = build_envelope(condensed, "calendar", "get_earnings_calendar")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_analyst_recommendations(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/recommendation", {"symbol": ticker})
    condensed = _condense_recommendations(result) if isinstance(result, list) else result
    envelope = build_envelope(condensed, ticker, "get_analyst_recommendations")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_analyst_revisions_history(self, ticker: str, lookback_months: int = 12) -> List[TextContent]:
    """Full time series of analyst rating-bucket counts over lookback_months.
    Lets the consumer detect upgrade/downgrade momentum that's invisible
    in a single-period snapshot."""
    result = await self.client.get("/stock/recommendation", {"symbol": ticker})
    if not isinstance(result, list) or not result:
      envelope = build_envelope({"error": "no recommendation data"}, ticker,
                                "get_analyst_revisions_history")
      return [TextContent(type="text", text=safe_json_dumps(envelope))]

    # Finnhub returns most-recent first
    periods = []
    for row in result[:lookback_months]:
      if not isinstance(row, dict):
        continue
      sb = int(row.get("strongBuy") or 0)
      b = int(row.get("buy") or 0)
      h = int(row.get("hold") or 0)
      s = int(row.get("sell") or 0)
      ss = int(row.get("strongSell") or 0)
      total = sb + b + h + s + ss
      # Net upgrade score: (strong_buy + buy) - (sell + strong_sell), normalized
      net_bullish = (sb + b) - (s + ss)
      pct_bullish = round((sb + b) / total * 100, 1) if total else 0
      pct_bearish = round((s + ss) / total * 100, 1) if total else 0
      periods.append({
        "period":         row.get("period", ""),
        "strong_buy":     sb,
        "buy":            b,
        "hold":           h,
        "sell":           s,
        "strong_sell":    ss,
        "total":          total,
        "net_bullish":    net_bullish,
        "pct_bullish":    pct_bullish,
        "pct_bearish":    pct_bearish,
      })

    # Trend deltas (latest vs N months ago)
    momentum = {}
    if len(periods) >= 2:
      latest = periods[0]
      prior = periods[1]
      momentum["1mo_strong_buy_delta"] = latest["strong_buy"] - prior["strong_buy"]
      momentum["1mo_buy_delta"] = latest["buy"] - prior["buy"]
      momentum["1mo_sell_delta"] = latest["sell"] - prior["sell"]
      momentum["1mo_net_bullish_delta"] = latest["net_bullish"] - prior["net_bullish"]
    if len(periods) >= 4:
      latest = periods[0]
      m3 = periods[3]
      momentum["3mo_net_bullish_delta"] = latest["net_bullish"] - m3["net_bullish"]
    if len(periods) >= 7:
      latest = periods[0]
      m6 = periods[6]
      momentum["6mo_net_bullish_delta"] = latest["net_bullish"] - m6["net_bullish"]

    # Signal classifier
    signal = "neutral"
    if momentum.get("3mo_net_bullish_delta") is not None:
      d3 = momentum["3mo_net_bullish_delta"]
      if d3 >= 5:
        signal = "upgrading_strong"
      elif d3 >= 2:
        signal = "upgrading"
      elif d3 <= -5:
        signal = "downgrading_strong"
      elif d3 <= -2:
        signal = "downgrading"

    out = {
      "ticker":           ticker.upper(),
      "periods":          periods,
      "periods_returned": len(periods),
      "momentum":         momentum,
      "signal":           signal,
      "source":           "Finnhub /stock/recommendation",
      "note":             "Counts reflect Finnhub's firm-rating buckets per month. Net_bullish = (strong_buy+buy)-(sell+strong_sell). Signal classifier uses 3mo net_bullish delta.",
    }
    envelope = build_envelope(out, ticker, "get_analyst_revisions_history")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_company_peers(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/peers", {"symbol": ticker})
    envelope = build_envelope(result, ticker, "get_company_peers")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_basic_financials(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/metric", {
      "symbol": ticker, "metric": "all"
    })
    condensed = _condense_basic_financials(result) if isinstance(result, dict) else result
    envelope = build_envelope(condensed, ticker, "get_basic_financials")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_earnings_surprises(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/earnings", {"symbol": ticker, "limit": 12})
    condensed = _condense_earnings_surprises(result) if isinstance(result, list) else result
    envelope = build_envelope(condensed, ticker, "get_earnings_surprises")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_forward_estimates(self, ticker: str) -> List[TextContent]:
    eps_result, rev_result, ebitda_result = await asyncio.gather(
      self.client.get("/stock/eps-estimate", {"symbol": ticker, "freq": "quarterly"}),
      self.client.get("/stock/revenue-estimate", {"symbol": ticker, "freq": "quarterly"}),
      self.client.get("/stock/ebitda-estimate", {"symbol": ticker, "freq": "quarterly"}),
    )
    condensed = _condense_forward_estimates(eps_result, rev_result, ebitda_result)

    # yfinance fallback when Finnhub free-tier returns no data on any sub-field.
    # Only fetch yfinance once per call, and only if at least one sub-field is missing.
    # Wrap in wait_for: Ticker.info is known to hang under Yahoo throttling.
    needs_fallback = any(
      isinstance(condensed.get(k), dict) and condensed[k].get("error")
      for k in ("eps", "revenue_B", "ebitda_B")
    )
    if needs_fallback:
      try:
        # 30s budget — yfinance's earnings_estimate / revenue_estimate calls
        # run ~3s standalone but can hit Yahoo throttling under MCP-subprocess
        # contention (other yfinance calls in flight via get_market_data).
        yf_data = await asyncio.wait_for(
          asyncio.to_thread(_yf_forward_estimates, ticker),
          timeout=30.0,
        )
        for k in ("eps", "revenue_B", "ebitda_B"):
          if isinstance(condensed.get(k), dict) and condensed[k].get("error"):
            condensed[k] = yf_data.get(k, condensed[k])
      except asyncio.TimeoutError:
        for k in ("eps", "revenue_B", "ebitda_B"):
          if isinstance(condensed.get(k), dict) and condensed[k].get("error"):
            condensed[k] = {"error": "no data (Finnhub) + yfinance_timeout"}

    envelope = build_envelope(condensed, ticker, "get_forward_estimates", api_calls_made=3)
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_financial_statements(self, ticker: str, statement: str, freq: str) -> List[TextContent]:
    result = await self.client.get("/stock/financials", {
      "symbol": ticker, "statement": statement, "freq": freq
    })
    condensed = _condense_financial_statements(result, statement, freq) if isinstance(result, dict) else result

    # yfinance fallback when Finnhub returns 403 / empty / unrecognized format
    # (free-tier `/stock/financials` is paywalled). Same pattern as
    # get_forward_estimates — single fallback call, 30s timeout to absorb
    # Yahoo cold-start handshake, tag _source so consumers see the imputation.
    needs_fallback = (
      not isinstance(condensed, dict)
      or condensed.get("error")
      or not condensed.get("periods")
    )
    if needs_fallback:
      try:
        yf_data = await asyncio.wait_for(
          asyncio.to_thread(_yf_financial_statements, ticker, statement, freq),
          timeout=30.0,
        )
        if yf_data.get("periods"):
          condensed = yf_data
      except asyncio.TimeoutError:
        if isinstance(condensed, dict):
          condensed["error"] = (condensed.get("error") or "no data") + " + yfinance_timeout"

    envelope = build_envelope(condensed, ticker, "get_financial_statements")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_company_profile(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/profile2", {"symbol": ticker})
    # Keep only the fields useful for analysis context
    if isinstance(result, dict):
      keep = {"name", "ticker", "exchange", "finnhubIndustry", "gics", "gicsSubIndustry",
              "country", "currency", "ipo", "weburl", "shareOutstanding", "marketCapitalization",
              "employeeTotal", "description"}
      condensed = {k: v for k, v in result.items() if k in keep and v is not None}
    else:
      condensed = result
    envelope = build_envelope(condensed, ticker, "get_company_profile")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_insider_sentiment(self, ticker: str, from_date: str = None, to_date: str = None) -> List[TextContent]:
    if not from_date:
      from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not to_date:
      to_date = datetime.now().strftime("%Y-%m-%d")
    result = await self.client.get("/stock/insider-sentiment", {
      "symbol": ticker, "from": from_date, "to": to_date
    })
    condensed = _condense_insider_sentiment(result) if isinstance(result, dict) else result
    envelope = build_envelope(condensed, ticker, "get_insider_sentiment")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(
          read_stream,
          write_stream,
          self.server.create_initialization_options(),
        )
        print("Successfully created finnhub process", file=sys.stderr, flush=True)
    except Exception as e:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise
    finally:
      await self.client.close()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.news_agregator.finnhub_server server", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "server":
    print("Starting finnhub process", file=sys.stderr, flush=True)
    try:
      server = FinnhubServer()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.news_agregator.finnhub_server server", file=sys.stderr)
    sys.exit(1)
