"""
Finnhub MCP Server -- 7 tools for market intelligence data.

Provides forward-looking signals: news, insider activity, analyst sentiment,
earnings calendars, peer companies, and key financial metrics.

Entry point: python -m tools.news_agregator.finnhub_server server
"""
from tools.response_meta import annotating, warning
from typing import Any, Dict, List
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.news_agregator.finnhub_utils import (
  FinnhubClient, build_envelope, denomination_shape, get_denomination,
)


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
Counts open-market purchases (code P) as buying, and sales plus tax withholding
(S, F) as selling; share grants (A) and gifts (G) are neither.
The recency buckets are anchored on today and the feed is not: `as_of`,
`data_lag_days` and each bucket's own `window` say how far the data actually
reaches, and a bucket the feed never reaches is null rather than zero.
Should use: When analyzing insider sentiment -- whether executives and directors are buying or selling shares. Important signal for investment decisions.
Should NOT use: For institutional holdings or 13F data. This covers only insider (officer/director) transactions."""

earnings_calendar_description = """Retrieves upcoming and recent earnings dates and EPS estimates for companies.
Should use: When you need to know when a company reports earnings, consensus EPS estimates, or to identify earnings catalysts in a date range.
Should NOT use: For actual reported financial results (use SEC tools or get_market_data instead)."""

ipo_calendar_description = """Retrieves upcoming and recent IPO listings in a date range.
Should use: When discovering new public companies that may be misunderstood for 6-12 months post-listing. Each entry includes symbol, company name, IPO date, exchange, expected price range, share count, and expected market cap.
Should NOT use: For already-public companies (use get_market_data). For S-1 / prospectus content (use SEC tools)."""

analyst_recommendations_description = """Retrieves analyst recommendation trends (buy/hold/sell/strong buy/strong sell) over time for a company.
Should use: When you need Wall Street consensus sentiment, analyst rating changes, or to gauge institutional opinion on a stock.
Should NOT use: For price targets or detailed analyst reports. This provides aggregate recommendation counts."""

company_peers_description = """Retrieves a list of peer/comparable company tickers for a given company from Finnhub.
Should use: When you need comparable companies for relative valuation, comp analysis, or to understand a company's competitive landscape.
Should NOT use: If you already know the peer group. This returns Finnhub's pre-computed peer list based on industry classification."""

basic_financials_description = """Retrieves key financial metrics and ratios for a company from Finnhub (52-week high/low, beta, PE, margins, ROE, etc.).
Units: marketCapitalization and enterpriseValue are in MILLIONS, unlike every other market-cap figure in this toolset. data.denomination names the currency, the listing Finnhub actually answered about, and which fields are scaled — read it before combining any figure here with one from another tool. For an ADR, Finnhub reports the local listing in the local currency (TSM answers as 2330.TW in TWD).
Should use: When you need a broad set of financial ratios and metrics for quick screening or to supplement yfinance data with additional metrics.
Should NOT use: For detailed financial statements or historical data (use SEC tools). This provides current snapshot metrics. Do not mix its market cap with get_market_data's — they are different snapshots at an unstated time and disagreed by 6.8% on NVDA."""

earnings_surprises_description = """Retrieves historical EPS earnings surprises (actual vs. consensus estimate) for the last 12 reported quarters.
Includes beat/miss rates and average surprise percent to assess management execution reliability.
Units: EPS is in the currency and on the share basis named in data.denomination, which for an ADR is the ordinary share of the local listing, not the receipt (TSM answers 27.25 TWD per 2330.TW share).
Periods: `period` is the calendar quarter Finnhub files a fiscal quarter under — not the fiscal period end, not the report date, and sometimes in the future. Join on (year, quarter); see data.period_label.
Should use: When assessing earnings quality, management guidance credibility, or to anchor forward EPS assumptions.
Should NOT use: For forward EPS estimates (use get_forward_estimates) or full income statements (use get_financial_statements)."""

forward_estimates_description = """Retrieves Wall Street analyst consensus estimates for EPS, Revenue, and EBITDA for the next 4-6 quarters.
Combines three Finnhub endpoints (eps-estimate, revenue-estimate, ebitda-estimate) in one call.
Units: each field carries its own `_currency`, and they need not agree — for an ADR the revenue estimate is the filer's reporting currency while the EPS estimate may be either. revenue_B and ebitda_B are billions of `_currency`; `_unit` states it. Where the currency could not be established it is null with `_currency_candidates` listed, never guessed.
Should use: When building DCF assumptions from consensus forecasts, or comparing your projections to the street consensus.
Should NOT use: For historical reported results (use get_earnings_surprises or SEC tools). Do not chain against get_earnings_surprises EPS without checking both denominations — for TSM they are 6.11x apart."""


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
Window: `from_date`/`to_date` are honoured to whole months. Finnhub itself filters
only on the year, so the month-level window is applied here, and a month only
partly inside the request is left out because a monthly ratio cannot be
pro-rated. The response states `window_requested` next to `window_returned`.
Reconciling with get_insider_transactions: the two count different Form 4
transaction codes. MSPR counts share grants (code A), which get_insider_transactions
excludes -- it counts only open-market purchases (P) against sales and tax
withholding (S, F). AMAT's 2026-03 MSPR of +100 on +6,669 shares is nine
director grants of 741 shares, not a purchase, which is why that month reads
as buying here and as zero buys there.
Should use: As a quick aggregate insider signal to confirm or contradict the detailed get_insider_transactions data.
Should NOT use: For individual transaction details (use get_insider_transactions)."""

def _covered_bucket(counts: Dict[str, int], window_start, window_end,
                    observed_dates: list):
  """A recency bucket, or None when the data never reaches into its window.

  The buckets are anchored on today, which is right -- "the last 30 days"
  means the last 30 days. The transaction feed is not anchored on today.
  Finnhub's insider transactions for AMAT stop at 2026-07-01 while today is
  2026-08-26, so the 30-day window covers 2026-07-27 to 2026-08-26 and the
  data speaks to none of it. `{bought: 0, sold: 0, net: 0}` there is a
  statement about how insiders behaved last month, made entirely out of our
  own gap: a screen reading `recent_30d.net == 0` as a quiet month was
  reading the absence of a feed.

  So a window with no observations in it at all returns None -- the same
  shape this condenser already uses when there are no transactions -- and a
  window the data reaches only part of carries the span it actually covers.
  `window` next to the bucket's name is the shortfall: "recent_90d" over
  "2026-05-28 -> 2026-07-01" is 34 days of a 90-day claim, and a reader can
  see that without doing the subtraction from period_end.
  """
  if not observed_dates:
    return None
  covered_start = max(window_start, min(observed_dates))
  covered_end = min(window_end, max(observed_dates))
  if covered_start > covered_end:
    return None
  return {
    "bought": counts["bought"],
    "sold": counts["sold"],
    "net": counts["bought"] - counts["sold"],
    "window": f"{covered_start.isoformat()} -> {covered_end.isoformat()}",
  }


def _condense_insider_data(raw: Dict[str, Any]) -> Dict[str, Any]:
  """Aggregate raw insider transactions into a compact signal summary.

  Input: Finnhub's {"data": [list of transactions]} where each has
  name, share, change, transactionDate, transactionCode (P=purchase, S=sale, etc.)

  Returns a condensed dict with totals, top insiders, recency buckets, and signal.
  """
  transactions = raw.get("data", [])
  if not transactions:
    # Every field below is derived from these rows, so with no rows there is
    # no total, no count and no signal. `signal: "neutral"` here occupied the
    # same field that reads "net_selling" for a real window, so nothing
    # downstream could separate "insiders were balanced" from "we got nothing
    # back", and a screen filtering on `signal != "net_selling"` admitted
    # every ticker the plan cannot see. The empty `top_insiders` is left as
    # the only shape in the payload, which is what lets `build_envelope`
    # recognise the emptiness and label the coverage.
    return {"total_bought": None, "total_sold": None, "net_shares": None,
            "buy_count": None, "sell_count": None, "top_insiders": [],
            "as_of": None,
            "data_lag_days": None,
            "recent_30d": None,
            "recent_90d": None,
            "prior_90d": None,
            "prior_period_avg_per_90d_sold": None,
            "current_vs_baseline_ratio": None,
            "period_start": None, "period_end": None,
            "signal": None}

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
    "as_of": now.isoformat(),
    "data_lag_days": (now - max(valid_dates)).days if valid_dates else None,
    "recent_30d": _covered_bucket(r30, now - timedelta(days=30), now, valid_dates),
    "recent_90d": _covered_bucket(r90, now - timedelta(days=90), now, valid_dates),
    "prior_90d": _covered_bucket(prior_90, now - timedelta(days=180),
                                 now - timedelta(days=91), valid_dates),
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
    # Which quarter the print covers. Without it the row is a ticker and a
    # date -- joinable to nothing, and this codebase has already been bitten
    # by an announcement keyed on the vendor's calendar bucket instead of on
    # fiscal identity.
    if event.get("quarter") is not None:
      slim["quarter"] = event["quarter"]
    if event.get("year") is not None:
      slim["year"] = event["year"]
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

  out = {
    "total_companies": len(events),
    "by_date": by_date,
    "events": slimmed_events
  }
  # A cap is fine. A cap the reader cannot see is not: 15 events out of 288
  # looks exactly like a quiet week unless the truncation is stated. Callers
  # needing the whole set pass `symbol`, or read the endpoint directly.
  if len(events) > len(slimmed_events):
    out["events_truncated"] = True
    out["events_shown"] = len(slimmed_events)
  return out


def _condense_ipo_calendar(raw: Dict[str, Any]) -> Dict[str, Any]:
  """Condense Finnhub /calendar/ipo response.

  Input: {"ipoCalendar": [list]} where each entry has symbol, name, date,
  exchange, numberOfShares, totalSharesValue, price (e.g. "10.00-12.00"),
  status (expected/priced/filed/withdrawn).

  Computes expected_market_cap = price_mid * numberOfShares so downstream
  filters (e.g., >= $1B) can apply without re-parsing the price field.
  """
  events = raw.get("ipoCalendar", []) or []
  out_events = []

  def _parse_price_mid(p):
    if not p:
      return None
    s = str(p)
    # Finnhub returns "10.00-12.00" or "10.00" or null
    try:
      if '-' in s:
        lo, hi = s.split('-', 1)
        return (float(lo) + float(hi)) / 2.0
      return float(s)
    except (ValueError, TypeError):
      return None

  for event in events:
    symbol = event.get('symbol', '')
    if not symbol:
      continue
    shares = event.get('numberOfShares') or 0
    price_mid = _parse_price_mid(event.get('price'))
    expected_mcap = (price_mid * shares) if (price_mid and shares) else None

    slim = {
      'symbol':              symbol,
      'name':                event.get('name', ''),
      'ipo_date':            event.get('date', ''),
      'exchange':            event.get('exchange', ''),
      'status':              event.get('status', ''),
      'price_range':         event.get('price', ''),
      'price_mid':           round(price_mid, 4) if price_mid else None,
      'shares_outstanding':  shares,
      'total_shares_value':  event.get('totalSharesValue'),
      'expected_market_cap': round(expected_mcap, 2) if expected_mcap else None,
    }
    out_events.append(slim)

  return {
    'total_listings': len(out_events),
    'events': out_events,
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


# The two metrics Finnhub scales and nothing in the response says so.
# `get_market_data.marketCap` for NVDA is 5,078,174,924,800 -- raw dollars --
# against Finnhub's 5,422,978. Same concept, same instant, 10^6 apart, and a
# caller reading the smaller one literally has NVDA worth $5.4 million.
_MILLIONS_METRICS = ("marketCapitalization", "enterpriseValue")


def _denomination_block(denomination: Dict[str, Any] = None,
                        *, requested: str = None,
                        scaled_fields: Dict[str, str] = None,
                        note: str = None) -> Dict[str, Any]:
  """The `denomination` sub-object: what these numbers are measured in."""
  block = dict(denomination or denomination_shape(requested))
  if scaled_fields:
    block["scaled_fields"] = dict(scaled_fields)
  if note:
    block["note"] = note
  return block


def _append_warnings(envelope: Dict[str, Any], entries: List[Dict[str, Any]]) -> None:
  """Add to whatever warnings the envelope already carries.

  Appending rather than assigning: `build_envelope` sets its own warning when
  a payload has no content, and overwriting that would trade "Finnhub returned
  nothing" for "this figure is in TWD" -- losing the more important of the two.
  """
  if not entries:
    return
  existing = list(envelope.get("warnings") or [])
  for entry in entries:
    if entry not in existing:
      existing.append(entry)
  envelope["warnings"] = existing


def _denomination_warnings(ticker: str,
                           denomination: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Warn when the figures are not in the units a caller would assume.

  Two cases, and only two, because a warning on every response is a warning
  nobody reads:

  * Finnhub answered about a different listing, or in a currency that is not
    the dollar every other tool in this stack reports in. That is the TSM
    case: 27.25 TWD per 2330.TW ordinary share sitting in a field a caller is
    about to divide by a USD ADR figure.
  * The currency could not be established at all, which is not the same as
    dollars and must not be read as dollars.
  """
  currency = (denomination or {}).get("currency")
  resolved = (denomination or {}).get("finnhub_symbol")
  requested = (ticker or "").strip().upper()

  if not currency:
    return [warning(
      "currency_unknown",
      f"Finnhub did not state a currency for {requested!r}, so every figure "
      f"below is a number without a unit. Do not assume US dollars: Finnhub "
      f"reports TSM in TWD, SAP in EUR and BABA in CNY. "
      f"({(denomination or {}).get('error') or 'no /stock/profile2 currency'})")]

  if currency.upper() != "USD" or (resolved and resolved.upper() != requested):
    shares = (denomination or {}).get("shares_outstanding_millions")
    return [warning(
      "reported_on_the_local_listing",
      f"Finnhub answers {requested} as {resolved or requested} and reports in "
      f"{currency}, not USD. It also does not state the share basis of a "
      f"per-share figure, and it is not consistent: TSM's EPS is per ordinary "
      f"share of 2330.TW, five to an ADR, while BABA's is per ADS. Finnhub "
      f"carries {shares}m shares outstanding for this company -- compare that "
      f"against the count behind whatever you are combining it with. Nothing "
      f"here converts a currency or a share basis.",
      currency=currency, finnhub_symbol=resolved,
      shares_outstanding_millions=shares)]

  return []


def _condense_basic_financials(raw: Dict[str, Any],
                               denomination: Dict[str, Any] = None) -> Dict[str, Any]:
  """Filter Finnhub's 132 metrics + historical series down to IB-essential metrics.

  Keeps ~35 key metrics, drops the massive 'series' section entirely.

  `denomination` says what the survivors are measured in, because every one of
  them is a figure a caller will put beside another tool's. Two facts were
  missing and both are actionable:

  * marketCapitalization and enterpriseValue are in MILLIONS. Nothing else in
    this toolset is.
  * the currency is the filer's, not the caller's. Finnhub resolved TSM to
    2330.TW and answered marketCapitalization 63,145,320 -- NT$63.1tn, roughly
    $2.0tn. Read as dollars that is $63tn.

  The resolved symbol used to be dropped here as request echo. For a domestic
  filer it is; for TSM it is Finnhub telling you it answered about a different
  listing than the one you asked for, which is the single most useful fact in
  the response.
  """
  metrics = raw.get("metric", {})
  if not metrics:
    # An empty body is Finnhub's answer for a symbol it does not carry, for
    # one outside the plan, and for a covered company with nothing to report.
    # It is handed back untouched so build_envelope still labels it, but the
    # denomination goes on regardless: "we could not establish the currency"
    # is itself worth saying.
    out = dict(raw)
    out["denomination"] = _denomination_block(
      denomination, requested=raw.get("symbol"))
    return out

  filtered = {k: v for k, v in metrics.items() if k in KEY_METRICS and v is not None}
  block = _denomination_block(
    denomination,
    requested=raw.get("symbol"),
    scaled_fields={name: "millions" for name in _MILLIONS_METRICS
                   if name in filtered},
  )
  currency = block.get("currency") or "an unestablished currency"
  listing = block.get("finnhub_symbol") or block.get("requested_symbol") or "this listing"
  block["note"] = (
    f"marketCapitalization and enterpriseValue are in MILLIONS of {currency}; "
    f"get_market_data.marketCap is raw currency units, so the two are 10^6 "
    f"apart. Per-share and price metrics (epsTTM, bookValuePerShare*, "
    f"52WeekHigh/Low, dividendPerShare*) are {currency} on the {listing} "
    f"listing, whose share count Finnhub gives as "
    f"{block.get('shares_outstanding_millions')}m. Ratios, margins and growth "
    f"rates are unitless.")

  return {
    "metric": filtered,
    "metric_count": len(filtered),
    "denomination": block,
  }


# What Finnhub's `period` actually is, stated once. Measured 2026-08-26 across
# AMAT, WMT, NVDA, DELL, TSM, AAPL, ORCL, COST, ADBE, CSCO, MU, HPQ and NKE:
# every value is a calendar quarter end, and for a filer whose quarters do not
# close on one it is the calendar quarter end on or after the fiscal close --
# AMAT's quarter ended 2026-07-26 and reported 2026-08-13 is labelled
# 2026-09-30. So it is neither of the two dates a caller would assume, and it
# can sit weeks in the future. `year` and `quarter` are the fiscal designators
# and are the identity to join on.
_PERIOD_LABEL = {
  "field": "period",
  "means": ("the end of the calendar quarter Finnhub files this fiscal "
            "quarter under -- a bucket, not a date the company reached"),
  "is_not": ["the fiscal period end", "the date the company reported"],
  "fiscal_identity": ["year", "quarter"],
  "example": ("AMAT reported its fiscal Q3 on 2026-08-13 for a quarter ended "
              "2026-07-26; the row is labelled 2026-09-30"),
}


def _duplicate_fiscal_periods(quarters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """(year, quarter) pairs Finnhub returned more than once.

  TGT, live 2026-08-26: fiscal 2027 Q2 arrives twice, actual 2.46 and
  surprise +6.52% on both, once labelled period 2026-09-30 and once
  2025-09-30. The rows are counted twice in beat_count and pull
  avg_surprise_pct toward that one quarter, and `period` -- being a bucket
  rather than an identity -- cannot tell a reader they are the same quarter.

  Declared rather than deduplicated. Which of the two rows is the wrong one is
  Finnhub's to say, and dropping one would silently change counts that other
  callers already read.
  """
  seen = defaultdict(list)
  for entry in quarters:
    year, quarter = entry.get("year"), entry.get("quarter")
    if year is None or quarter is None:
      continue
    seen[(year, quarter)].append(entry.get("period"))
  return [{"fiscal": [year, quarter], "periods": periods}
          for (year, quarter), periods in seen.items() if len(periods) > 1]


def _condense_earnings_surprises(raw: List[Dict[str, Any]], limit: int = 12,
                                 denomination: Dict[str, Any] = None) -> Dict[str, Any]:
  """Condense historical EPS surprises into beat/miss summary.

  Input: Finnhub list of {actual, estimate, surprise, surprisePercent, period, year, quarter}
  Returns: per-quarter table, beat_count, miss_count, avg_surprise_pct, beat_rate_pct

  `denomination` names the currency and the listing the EPS belongs to.
  TSM's 27.25 is TWD per ordinary share of 2330.TW; get_forward_estimates
  answers 4.46 for the same company and quarter in USD per ADR. Chained
  without labels that is an 84% collapse in EPS which never happened, and
  neither response used to carry anything that could stop it.
  """
  if not isinstance(raw, list) or not raw:
    # `beat_count: 0` beside `quarters: []` reads as "this company has never
    # beaten" -- a fact about the filer, asserted from an empty hand. The
    # counts below are only reported once there are quarters to count.
    return {"quarters": [], "beat_count": None, "miss_count": None,
            "avg_surprise_pct": None,
            "denomination": _denomination_block(denomination)}

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

  block = _denomination_block(
    denomination, requested=(raw[0].get("symbol") if isinstance(raw[0], dict) else None))
  currency = block.get("currency") or "an unestablished currency"
  listing = block.get("finnhub_symbol") or block.get("requested_symbol") or "this listing"
  block["note"] = (
    f"actual_eps and estimate_eps are in {currency}, on the {listing} listing. "
    f"Finnhub does not state whether a per-share figure is per ordinary share "
    f"or per ADR and is not consistent across issuers, so establish the basis "
    f"before combining these with a per-share figure from anywhere else. "
    f"surprise_pct is a percentage.")

  result = {
    "quarters": quarters,
    "beat_count": beat_count,
    "miss_count": miss_count,
    "total_periods": len(quarters),
    "denomination": block,
    "period_label": dict(_PERIOD_LABEL),
    "duplicate_fiscal_periods": _duplicate_fiscal_periods(quarters),
  }
  if surprise_pcts:
    result["avg_surprise_pct"] = round(sum(surprise_pcts) / len(surprise_pcts), 2)
    result["beat_rate_pct"] = round(beat_count / max(len(quarters), 1) * 100, 1)
  return result


def _condense_forward_estimates(eps_raw: Any, rev_raw: Any, ebitda_raw: Any) -> Dict[str, Any]:
  """Combine EPS, Revenue, and EBITDA forward estimates into one compact structure.

  Input: three Finnhub responses from eps-estimate, revenue-estimate, ebitda-estimate.
  Revenue and EBITDA are returned in raw USD (billions scale applied here for readability).

  When Finnhub declines the request, its own words are kept. All three of
  these endpoints answer HTTP 403 "You don't have access to this resource" on
  the free tier; flattening that to "no data" made an entitlement problem look
  like a company with no analyst coverage, and left the caller nothing to act
  on.
  """
  result = {}

  def _extract(raw: Any, avg_key: str, high_key: str, low_key: str, scale: float = 1.0) -> Dict:
    if isinstance(raw, dict) and raw.get("error"):
      return {"error": f"Finnhub: {raw['error']}"}
    if not isinstance(raw, dict) or "data" not in raw:
      return {"error": f"Finnhub: unrecognized response ({type(raw).__name__})"}
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


# Which provider each `_source` tag names. A field with no `_source` came from
# the primary path, which is the only case that may be credited to Finnhub.
# The derived EBITDA still credits yfinance -- its input is yfinance revenue --
# because "derived" is a property of the field, not of the upstream, and the
# field says so itself via `_source`, `_derived` and the response warning.
_SOURCE_PROVIDERS = {
  None: "Finnhub",
  "yfinance_fallback": "yfinance",
  "yfinance_fallback_inferred": "yfinance",
}

_FORWARD_FIELDS = ("eps", "revenue_B", "ebitda_B")


def _forward_estimates_provenance(condensed: Dict[str, Any]):
  """(provider, {field: _source}) for a condensed forward-estimates payload.

  The response said `provider: "Finnhub"` on every call while serving
  yfinance for all three fields. A caller who credits Finnhub cannot audit the
  number, cannot reproduce it, and has no way to know the EBITDA line was
  arithmetic rather than a survey. This names what actually answered, per
  field and in one summary string, from the `_source` tags the fields already
  carry -- one source of truth rather than a second opinion that can drift.
  """
  sources = {}
  for field in _FORWARD_FIELDS:
    value = condensed.get(field)
    sources[field] = value.get("_source") if isinstance(value, dict) else None

  providers = []
  for field in _FORWARD_FIELDS:
    value = condensed.get(field)
    # A field that errored credits nobody; an empty period list is still an
    # answer, and the source that gave it is the one to name.
    if not isinstance(value, dict) or "periods" not in value:
      continue
    name = _SOURCE_PROVIDERS.get(sources[field], sources[field])
    if name not in providers:
      providers.append(name)

  if not providers:
    return "none (no forward estimates retrieved)", sources
  return " + ".join(providers), sources


# Fields whose value is a currency amount rather than a per-share figure, and
# which this tool has already divided by 1e9.
_FORWARD_BILLIONS_FIELDS = ("revenue_B", "ebitda_B")


def _label_forward_denomination(condensed: Dict[str, Any], *,
                                finnhub_currency=None,
                                yf_quote_currency=None,
                                yf_reporting_currency=None) -> Dict[str, Any]:
  """Tag each forward-estimate field with the currency it is actually in.

  One response, two currencies, and neither stated. Live 2026-08-26 for TSM:
  `eps 0q avg` is 4.45834 and `revenue_B 0q avg` is 1454.9601. The first is
  USD, the second TWD billions -- roughly $46bn, which is why the field name
  asserting billions still left TSM looking 33x DELL's 44.4452.

  Where the figure came from decides which metadata answers for it:

  * No `_source` means Finnhub served it, and Finnhub answers on the local
    listing `/stock/profile2` names -- TWD for TSM, EUR for SAP.
  * A yfinance `_source` on revenue (and the EBITDA derived from revenue)
    means the filer's reporting currency, `info['financialCurrency']`.
    Verified against TSM (TWD 1454.96bn), SONY (JPY 3148.04bn), BABA
    (CNY 270.32bn) and SAP (EUR 10.08bn) -- four for four.
  * A yfinance `_source` on EPS means one of two currencies and yfinance does
    not say which. TSM 4.45834 against a $417.69 ADR is USD, the quote
    currency; SONY 0.33459 against $24.12 is USD; BABA 10.90073 against a
    $119.83 ADS is CNY, the reporting currency -- 10.90 dollars a quarter
    would put BABA on a P/E of 2.7. Three in four is not a rule, so where the
    two currencies differ this reports the currency unknown and names both
    candidates rather than picking the likelier and being 7.15x wrong on the
    fourth.

  Nothing here converts. A field nobody answered gets no currency at all,
  because a currency on an error is a claim about a number that does not
  exist.
  """
  quote = (yf_quote_currency or "").strip().upper() or None
  reporting = (yf_reporting_currency or "").strip().upper() or None
  finnhub = (finnhub_currency or "").strip().upper() or None

  for field in _FORWARD_FIELDS:
    value = condensed.get(field)
    if not isinstance(value, dict) or "periods" not in value:
      continue                      # errored, or not a payload we can label
    source = value.get("_source")

    if source is None:
      currency, candidates = finnhub, None
    elif field in _FORWARD_BILLIONS_FIELDS:
      currency, candidates = reporting, None
    elif quote and reporting and quote != reporting:
      currency, candidates = None, [quote, reporting]
    else:
      currency, candidates = quote or reporting, None

    value["_currency"] = currency
    if candidates:
      value["_currency_candidates"] = candidates
    if field in _FORWARD_BILLIONS_FIELDS:
      value["_unit"] = (f"billions of {currency}" if currency
                        else "billions of an unestablished currency")

  return condensed


def _forward_denomination_warnings(ticker: str,
                                   condensed: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Say so when the fields of one response are not in the same currency.

  TSM live 2026-08-26 is the case: `eps` in USD or TWD (yfinance does not say
  which), `revenue_B` in TWD billions. A caller who takes both as dollars gets
  a company doing $1.45tn of quarterly revenue on $4.46 of quarterly EPS.
  """
  entries = []
  labelled = {}
  for field in _FORWARD_FIELDS:
    value = condensed.get(field)
    if isinstance(value, dict) and "periods" in value:
      labelled[field] = value.get("_currency")

  ambiguous = {field: condensed[field]["_currency_candidates"]
               for field in labelled
               if condensed[field].get("_currency_candidates")}
  if ambiguous:
    entries.append(warning(
      "currency_not_established",
      f"yfinance served these estimates for {ticker} and quotes it in one "
      f"currency while the company reports in another. yfinance does not say "
      f"which applies to an EPS estimate and it is not consistently either -- "
      f"TSM's is the quote currency, BABA's is the reporting currency -- so "
      f"the currency is left unset rather than guessed. Establish it before "
      f"combining these with any other figure.",
      fields=ambiguous))

  known = {c for c in labelled.values() if c}
  if len(known) > 1:
    entries.append(warning(
      "mixed_currencies_in_one_response",
      f"The fields below are not all in the same currency: "
      f"{ {f: c for f, c in labelled.items() if c} }. Each carries its own "
      f"`_currency`. Nothing here has been converted.",
      currencies=sorted(known)))

  return entries


def _infer_ebitda_periods(rev_periods: list, margin: float) -> list:
  """Revenue estimates multiplied by a single trailing EBITDA margin.

  This is not an estimate anybody published. yfinance surfaces no forward
  EBITDA, so each period is `revenue_avg * ebitdaMargins` with the current TTM
  margin held flat -- verified against the live response: 395.7282 * 0.6529 =
  258.39.

  The analyst count is deliberately dropped. It was copied across from the
  revenue period, so every derived EBITDA figure reported `analysts: 54` and
  claimed 54 analysts had published an EBITDA estimate when none had. A number
  this process computed must not carry a count of people who did not compute
  it; `_derived` and `_derived_from` say where it came from instead.
  """
  periods = []
  for p in rev_periods or []:
    entry = {"period": p.get("period", "")}
    for key in ("avg", "low", "high"):
      if key in p:
        entry[key] = round(p[key] * margin, 4)
    entry["_derived"] = True
    entry["_derived_from"] = "revenue estimate * trailing ebitdaMargins"
    periods.append(entry)
  return periods


def _yf_forward_estimates(ticker: str) -> Dict[str, Any]:
  """yfinance fallback for forward estimates. Synchronous — must be awaited
  via asyncio.to_thread by the caller.

  Returns the same shape as `_condense_forward_estimates` so consumers can
  swap sub-fields when Finnhub returns 'no data' (free-tier 403). Each sub-
  field is tagged with `_source: yfinance_fallback` (or `_inferred` for the
  EBITDA case, which yfinance does not surface natively and is derived from
  revenue * info['ebitdaMargins']).

  `_currencies` carries yfinance's own two currency fields so the caller can
  label each estimate. They are not the same thing and for an ADR they are not
  the same value: TSM quotes in USD and reports in TWD, which is the whole
  reason its EPS estimate and its revenue estimate arrive in different
  currencies inside one response.
  """
  import yfinance as yf

  BILLION = 1e9
  out = {
    "eps": {"error": "no yfinance data"},
    "revenue_B": {"error": "no yfinance data"},
    "ebitda_B": {"error": "no yfinance equivalent"},
    "_currencies": {"quote": None, "reporting": None},
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
    for field in _FORWARD_FIELDS:
      out[field] = {"error": err}
    return out

  # `t.info` fetched at most once and reused. It is the slow call -- it can
  # hang 10-30s under Yahoo throttling -- and it is now wanted for two things
  # rather than one: the EBITDA margin, and the pair of currency fields that
  # say what any of these estimates are denominated in.
  _info_cache = {}

  def _info():
    if "value" not in _info_cache:
      _info_cache["value"] = t.info or {}
    return _info_cache["value"]

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
      margin = _info().get("ebitdaMargins")
      if margin:
        out["ebitda_B"] = {"periods": _infer_ebitda_periods(rev_periods, margin),
                           "_source": "yfinance_fallback_inferred",
                           "_derived": True,
                           "_inferred_margin": round(float(margin), 4)}
    except Exception as exc:
      out["ebitda_B"] = {"error": f"yfinance ebitda inference: {type(exc).__name__}: {exc}"}

  # Read only when something was actually retrieved. On an all-errors response
  # there is no figure to denominate, and paying for `t.info` to label nothing
  # would spend the slow call for no answer.
  if any(isinstance(out.get(f), dict) and "periods" in out[f]
         for f in _FORWARD_FIELDS):
    try:
      info = _info()
      out["_currencies"] = {"quote": info.get("currency"),
                            "reporting": info.get("financialCurrency")}
    except Exception as exc:
      out["_currencies"] = {
        "quote": None, "reporting": None,
        "error": f"yfinance currency: {type(exc).__name__}: {exc}"}

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

  When Finnhub declines the request, its own words are kept -- the same rule
  `_condense_forward_estimates` follows. `/stock/financials` answers HTTP 403
  "You don't have access to this resource" on this plan for every symbol, NVDA
  included, and calling that "Unrecognized response format" was a false
  diagnosis: the format was a perfectly recognizable entitlement refusal. It
  sent a reader to fix a parser and lost the one sentence naming the fixable
  problem.
  """
  if isinstance(raw, dict) and raw.get("error"):
    return {"statement": statement, "freq": freq,
            "error": f"Finnhub: {raw['error']}"}

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


# MSPR is a monthly aggregate, so the finest window this tool can answer is a
# whole month. The cap bounds the payload for a multi-year request; it is
# reported through `total_months`/`returned`/`truncated` rather than applied
# silently, which is what the old fixed six-month slice did.
_SENTIMENT_MONTH_CAP = 24

# Rating-trend classifier thresholds, in percentage points of the net_bullish
# SHARE of covering analysts. The classifier used to read the raw count delta
# with bands at |2| and |5| ratings, which made 65-of-68 analysts falling to
# 63-of-68 a "downgrading" verdict -- 2.9pp of drift on a firm covered by
# sixty-eight houses. Percentage points are the only unit in which a move at
# 6 analysts and a move at 68 are the same statement.
_RATING_TREND_PP = 5.0
_RATING_TREND_STRONG_PP = 10.0


def _month_span(year: int, month: int):
  """First and last calendar day of a month."""
  first = date(year, month, 1)
  last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
  return first, last


def _parse_iso_date(value: Any):
  """An ISO date, or None for anything that is not one."""
  try:
    return datetime.strptime(value, "%Y-%m-%d").date()
  except (TypeError, ValueError):
    return None


def _condense_insider_sentiment(raw: Dict[str, Any], from_date: str = None,
                                to_date: str = None) -> Dict[str, Any]:
  """Condense Finnhub insider sentiment (MSPR) into a monthly summary of the
  window that was asked for, and say which window that turned out to be.

  Input: {"data": [{"year": int, "month": int, "mspr": float, "change": int, "msprChange": float}]}
  MSPR: Monthly Share Purchase Ratio. +1 = all insiders buying, -1 = all insiders selling.

  The window has to be applied here because Finnhub cannot apply it. Verified
  against the live endpoint on 2026-08-26: `/stock/insider-sentiment` reads
  only the *year* of `from` and `to`. `from=2025-10-01&to=2026-07-01` and
  `from=2025-08-26&to=2026-08-26` both return the identical 16 rows spanning
  2025-01 to 2026-07, and `from=2024-01-01&to=2024-06-30` returns all twelve
  months of 2024. Passing the caller's dates through and returning whatever
  came back made the two arguments decoration: the response for AMAT was
  byte-identical across a six-month and a ten-month request, down to
  `avg_mspr: -66.6667`.

  It then kept the six most recent rows regardless, which erased any window
  older than six months and left the coverage to be reverse-engineered from
  the months array. That is what made this tool irreconcilable with
  `get_insider_transactions`, which states its own period_start/period_end.

  A month only partly inside the request is left out. There is no way to
  pro-rate a ratio, so including July for a window that ends on July 1 would
  answer with thirty days the caller excluded. `window_requested` and
  `window_returned` sit beside each other so the shortfall is visible without
  arithmetic -- the same pair `_news_page` reports.
  """
  data = raw.get("data", []) if isinstance(raw, dict) else []
  window_start = _parse_iso_date(from_date)
  window_end = _parse_iso_date(to_date)
  window_requested = (f"{from_date} -> {to_date}"
                      if window_start and window_end else None)

  in_window = []
  for item in data:
    year, month = item.get("year"), item.get("month")
    if not isinstance(year, int) or not isinstance(month, int):
      continue
    if not 1 <= month <= 12:
      continue
    first, last = _month_span(year, month)
    if window_start and first < window_start:
      continue
    if window_end and last > window_end:
      continue
    in_window.append(item)

  total_months = len(in_window)
  page = sorted(in_window, key=lambda x: (x["year"], x["month"]),
                reverse=True)[:_SENTIMENT_MONTH_CAP]

  months = []
  mspr_values = []
  for item in page:
    mspr = item.get("mspr")
    months.append({
      "year": item["year"],
      "month": item["month"],
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
    # Either no months survived the window -- so there is no average and no
    # verdict over them, and the empty `months` is what `build_envelope`
    # reads to label the coverage -- or months arrived and every MSPR was
    # null. The same rule either way: no average, no classification.
    signal = None

  window_returned = (
    f"{page[-1]['year']}-{page[-1]['month']:02d} -> "
    f"{page[0]['year']}-{page[0]['month']:02d}") if page else None

  return {
    "window_requested": window_requested,
    "window_returned": window_returned,
    "total_months": total_months,
    "returned": len(months),
    "truncated": total_months > len(months),
    "months": months,
    "signal": signal,
    "avg_mspr": avg_mspr,
  }


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


def _news_page(articles: List[Dict[str, Any]], from_date: str, to_date: str,
               cap: int = 20) -> Dict[str, Any]:
  """A capped slice of a news window, plus what the cap cost.

  NVDA over an eight-day window returned 20 articles all stamped the final
  day; Finnhub had 246 and the preceding seven days vanished. As a bare list
  that reads as "here is the news for your window" when it is "here are the
  20 most recent" -- for a catalyst workflow, the difference between
  "nothing happened last week" and "we did not look".

  So the page names both windows: the one asked for and the one the returned
  articles actually span.
  """
  returned = _slim_articles(articles, cap=cap)
  stamps = sorted(a["datetime"] for a in returned if a["datetime"])
  covered = (f"{stamps[0][:10]} -> {stamps[-1][:10]}") if stamps else None

  return {
    "window_requested": f"{from_date} -> {to_date}",
    "window_returned": covered,
    "total_articles": len(articles),
    "returned": len(returned),
    "truncated": len(returned) < len(articles),
    "articles": returned,
  }


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
              },
              "symbol": {
                "type": "string",
                "description": "Optional: filter to one ticker (e.g. ORCL). "
                               "Use this to confirm a single company's earnings "
                               "date — events are never lost to the summary cap."
              }
            },
            "required": ["from_date", "to_date"]
          }
        ),
        Tool(
          name="get_ipo_calendar",
          description=ipo_calendar_description,
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
          name="get_analyst_rating_trend",
          description=(
            "Monthly time series of analyst RATING-BUCKET counts (strong buy / buy / hold / sell / strong sell) from "
            "Finnhub /stock/recommendation. This measures how firms are rated, NOT how they are modelling the company: "
            "it contains no EPS or revenue numbers and no price targets, so it cannot answer 'are estimates being taken "
            "up into the print'. Finnhub's estimate and upgrade/downgrade feeds are not on this plan (verified 403). "
            "Returns per-month bucket counts, net_bullish ((strong_buy+buy)-(sell+strong_sell)), the same figure as a "
            "share of covering analysts, deltas where the history supports them, and a signal classifier "
            "(upgrading_strong / upgrading / neutral / downgrading / downgrading_strong) measured on the change in that "
            "SHARE so a two-analyst move means something different at 6 analysts than at 68. Finnhub typically serves "
            "only the last ~4 monthly snapshots regardless of lookback_months; the shortfall is reported as a warning."
          ),
          inputSchema={
            "type": "object",
            "properties": {
              "ticker": {"type": "string", "description": "Stock ticker symbol"},
              "lookback_months": {"type": "integer", "description": "Months of history to include. Finnhub usually returns fewer; the response names the shortfall.", "default": 12}
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
                "description": "Start date in YYYY-MM-DD format (default: 1 year "
                               "ago). Honoured to whole months: a month must fall "
                               "entirely inside the window to be reported."
              },
              "to_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format (default: today). "
                               "Honoured to whole months, so the current partial "
                               "month is excluded from a window ending today."
              }
            },
            "required": ["ticker"]
          }
        ),
      ]

    @self.server.call_tool()
    @annotating(
      "Finnhub",
warnings_per_tool={
        "get_earnings_calendar": [
          warning("dates_can_be_wrong",
                  "Finnhub earnings dates can be stale or incorrect. Confirm "
                  "against the filer before relying on a date."),
        ],
        # True of every response, so stated once here rather than rebuilt per
        # call -- the same place get_short_interest's staleness lives on the
        # financial server.
        "get_basic_financials": [
          warning("units_are_millions",
                  "marketCapitalization and enterpriseValue are in MILLIONS "
                  "of the currency named in data.denomination. Every other "
                  "market-cap figure in this stack is raw currency units -- "
                  "get_market_data.marketCap for NVDA is 5078174924800 "
                  "against this tool's 5422978. Read literally, the second "
                  "says NVDA is worth $5.4 million."),
          warning("no_as_of_timestamp",
                  "Finnhub stamps no as-of time on these metrics, so "
                  "data_as_of is null and staleness is invisible. Measured "
                  "2026-08-26 against get_market_data at the same instant, "
                  "market cap to market cap: NVDA 6.79% apart "
                  "($5.423tn against $5.078tn), MSFT 0.38%. It is part stale "
                  "price and part a different share count, the size is "
                  "ticker-dependent, and none of that can be read off the "
                  "response -- so do not mix the two sources in one "
                  "calculation."),
        ],
        "get_earnings_surprises": [
          warning("period_is_a_calendar_bucket",
                  "`period` is the calendar quarter Finnhub files a fiscal "
                  "quarter under. It is neither the fiscal period end nor the "
                  "report date, and it can be weeks in the future: AMAT "
                  "reported its fiscal Q3 on 2026-08-13 for a quarter ended "
                  "2026-07-26 and the row is labelled 2026-09-30. Joining "
                  "anything on it against a real period end returns nothing. "
                  "The fiscal identity is (year, quarter)."),
        ],
      })
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
            arguments["from_date"], arguments["to_date"],
            arguments.get("symbol", "")
          )
        case "get_ipo_calendar":
          return await parent.get_ipo_calendar(
            arguments["from_date"], arguments["to_date"]
          )
        case "get_analyst_recommendations":
          return await parent.get_analyst_recommendations(arguments["ticker"])
        case "get_analyst_rating_trend":
          return await parent.get_analyst_rating_trend(arguments["ticker"], arguments.get("lookback_months", 12))
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
      result = _news_page(result, from_date, to_date, cap=20)
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

  async def get_earnings_calendar(self, from_date: str, to_date: str,
                                  symbol: str = "") -> List[TextContent]:
    params = {"from": from_date, "to": to_date}
    if symbol:
      # Finnhub supports symbol filtering server-side; a single ticker's
      # events are never lost to the condensed 15-event cap.
      params["symbol"] = symbol.upper()
    result = await self.client.get("/calendar/earnings", params)
    condensed = _condense_earnings_calendar(result) if isinstance(result, dict) else result
    envelope = build_envelope(condensed, symbol.upper() if symbol else "calendar",
                              "get_earnings_calendar")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_ipo_calendar(self, from_date: str, to_date: str) -> List[TextContent]:
    """Wraps Finnhub's /calendar/ipo. Returns per-entry symbol, name,
    date, exchange, expected price range, share count, and computed
    expected market cap (price_mid * shares)."""
    result = await self.client.get("/calendar/ipo", {
      "from": from_date, "to": to_date
    })
    condensed = _condense_ipo_calendar(result) if isinstance(result, dict) else result
    envelope = build_envelope(condensed, "calendar", "get_ipo_calendar")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_analyst_recommendations(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/recommendation", {"symbol": ticker})
    condensed = _condense_recommendations(result) if isinstance(result, list) else result
    envelope = build_envelope(condensed, ticker, "get_analyst_recommendations")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_analyst_rating_trend(self, ticker: str, lookback_months: int = 12) -> List[TextContent]:
    """Monthly analyst RATING-BUCKET counts over lookback_months.

    Named for what it measures. It was called `get_analyst_revisions_history`,
    which reads as "how are analysts revising their estimates" -- and a
    pre-earnings workflow asking that question got a strongBuy/buy/hold/sell
    distribution back and could not tell. Verified against this key on
    2026-08-26: `/stock/eps-estimate`, `/stock/revenue-estimate`,
    `/stock/upgrade-downgrade` and `/stock/price-target` all answer 403 "You
    don't have access to this resource." There is no estimate-revision feed on
    this plan, so the name could not be made true and had to change instead.

    The classifier reads the change in net_bullish as a SHARE of covering
    analysts rather than the raw count. A two-analyst move is a re-rating at 6
    analysts and noise at 68, and the old thresholds (>= |2| ratings) called
    65-of-68 going to 63-of-68 a downgrade. Sharing also stops new coverage
    reading as an upgrade: AMAT went 43 analysts to 45 and picked up 3 net
    bullish, two of which were initiations, and was reported "upgrading" the
    day before it gapped -6.57%.
    """
    result = await self.client.get("/stock/recommendation", {"symbol": ticker})

    # Two different nothings, told apart. Finnhub declining the request is an
    # error with a reason worth reporting; an empty list is a coverage fact
    # `build_envelope` already knows how to label. `{"error": "no
    # recommendation data"}` collapsed both into a sentence with no code, no
    # coverage label and nothing in metadata.errors.
    if isinstance(result, dict) and result.get("error"):
      envelope = build_envelope({"periods": [],
                                 "error": f"Finnhub: {result['error']}"},
                                ticker, "get_analyst_rating_trend",
                                errors=[f"Finnhub: {result['error']}"])
      envelope["warnings"] = [warning(
        "primary_source_unavailable",
        "Finnhub did not answer the recommendation request, so no rating "
        "history is reported. See metadata.errors for what it said; this is "
        "not evidence that the company has no analyst coverage.")]
      return [TextContent(type="text", text=safe_json_dumps(envelope))]
    if not isinstance(result, list) or not result:
      envelope = build_envelope({"periods": []}, ticker,
                                "get_analyst_rating_trend")
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
        # net_bullish as a share of the analysts covering that month. The
        # count alone cannot be compared across firms, or across months in
        # which coverage changed.
        "net_bullish_pct": (round(net_bullish / total * 100, 1)
                            if total else None),
        "pct_bullish":    pct_bullish,
        "pct_bearish":    pct_bearish,
      })

    # Trend deltas (latest vs N months ago). A delta the description promises
    # and the history cannot support is NAMED rather than left absent: an
    # absent key and a delta of zero look identical to anything reading the
    # dict with .get().
    momentum = {}
    momentum_unavailable = {}
    if len(periods) >= 2:
      latest = periods[0]
      prior = periods[1]
      momentum["1mo_strong_buy_delta"] = latest["strong_buy"] - prior["strong_buy"]
      momentum["1mo_buy_delta"] = latest["buy"] - prior["buy"]
      momentum["1mo_sell_delta"] = latest["sell"] - prior["sell"]
      momentum["1mo_net_bullish_delta"] = latest["net_bullish"] - prior["net_bullish"]
    else:
      momentum_unavailable["1mo_net_bullish_delta"] = (
        f"needs 2 monthly snapshots; Finnhub returned {len(periods)}")

    share_delta_3mo = None
    if len(periods) >= 4:
      latest = periods[0]
      m3 = periods[3]
      momentum["3mo_net_bullish_delta"] = latest["net_bullish"] - m3["net_bullish"]
      momentum["3mo_total_analysts_delta"] = latest["total"] - m3["total"]
      if latest["net_bullish_pct"] is not None and m3["net_bullish_pct"] is not None:
        share_delta_3mo = round(
          latest["net_bullish_pct"] - m3["net_bullish_pct"], 1)
      momentum["3mo_net_bullish_share_delta_pp"] = share_delta_3mo
    else:
      momentum_unavailable["3mo_net_bullish_delta"] = (
        f"needs 4 monthly snapshots; Finnhub returned {len(periods)}")

    if len(periods) >= 7:
      latest = periods[0]
      m6 = periods[6]
      momentum["6mo_net_bullish_delta"] = latest["net_bullish"] - m6["net_bullish"]
      momentum["6mo_total_analysts_delta"] = latest["total"] - m6["total"]
      if latest["net_bullish_pct"] is not None and m6["net_bullish_pct"] is not None:
        momentum["6mo_net_bullish_share_delta_pp"] = round(
          latest["net_bullish_pct"] - m6["net_bullish_pct"], 1)
    else:
      momentum_unavailable["6mo_net_bullish_delta"] = (
        f"needs 7 monthly snapshots; Finnhub returned {len(periods)}")

    # Signal classifier. It reads one input, the 3-month change in net_bullish
    # as a share of coverage, and fewer than four periods means that input does
    # not exist -- "neutral" there reported no momentum where none was measured.
    signal = None
    signal_basis = None
    if share_delta_3mo is None:
      signal_basis = (
        "not classified: " + momentum_unavailable.get(
          "3mo_net_bullish_delta",
          "no covering analysts in one of the two months compared"))
    else:
      signal = "neutral"       # measured and flat, which is a reading
      if share_delta_3mo >= _RATING_TREND_STRONG_PP:
        signal = "upgrading_strong"
      elif share_delta_3mo >= _RATING_TREND_PP:
        signal = "upgrading"
      elif share_delta_3mo <= -_RATING_TREND_STRONG_PP:
        signal = "downgrading_strong"
      elif share_delta_3mo <= -_RATING_TREND_PP:
        signal = "downgrading"
      signal_basis = (
        f"net_bullish share moved {share_delta_3mo:+.1f}pp over 3 months "
        f"({periods[3]['net_bullish']} of {periods[3]['total']} analysts -> "
        f"{periods[0]['net_bullish']} of {periods[0]['total']}); "
        f"thresholds +/-{_RATING_TREND_PP}pp and "
        f"+/-{_RATING_TREND_STRONG_PP}pp")

    out = {
      "ticker":           ticker.upper(),
      "measures":         "analyst_rating_buckets",
      "periods":          periods,
      "periods_returned": len(periods),
      "lookback_months_requested": lookback_months,
      "momentum":         momentum,
      "momentum_unavailable": momentum_unavailable,
      "signal":           signal,
      "signal_basis":     signal_basis,
      "source":           "Finnhub /stock/recommendation",
      "note":             ("Recommendation buckets per month -- how firms RATE "
                           "the company, not what they MODEL for it. No EPS, "
                           "revenue or price-target figure is in this response, "
                           "and none is available on this Finnhub plan. "
                           "Net_bullish = (strong_buy+buy)-(sell+strong_sell); "
                           "net_bullish_pct expresses it as a share of covering "
                           "analysts, and the signal classifier reads the "
                           "3-month change in that share."),
    }
    envelope = build_envelope(out, ticker, "get_analyst_rating_trend")

    # The window asked for and the window covered were both in the response
    # and a reader had to spot the gap themselves. lookback_months=12 returns
    # 4 periods for every ticker tried.
    if periods and len(periods) < lookback_months:
      _append_warnings(envelope, [warning(
        "history_shorter_than_requested",
        f"{lookback_months} months of rating history were requested and "
        f"Finnhub returned {len(periods)} monthly snapshots. Deltas needing "
        f"more history than that are listed in data.momentum_unavailable.",
        requested_months=lookback_months,
        returned_months=len(periods))])

    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_company_peers(self, ticker: str) -> List[TextContent]:
    result = await self.client.get("/stock/peers", {"symbol": ticker})
    envelope = build_envelope(result, ticker, "get_company_peers")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_basic_financials(self, ticker: str) -> List[TextContent]:
    result, denomination = await asyncio.gather(
      self.client.get("/stock/metric", {"symbol": ticker, "metric": "all"}),
      get_denomination(self.client, ticker),
    )
    condensed = (_condense_basic_financials(result, denomination)
                 if isinstance(result, dict) else result)
    envelope = build_envelope(condensed, ticker, "get_basic_financials",
                              api_calls_made=2)
    _append_warnings(envelope, _denomination_warnings(ticker, denomination))
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_earnings_surprises(self, ticker: str) -> List[TextContent]:
    result, denomination = await asyncio.gather(
      self.client.get("/stock/earnings", {"symbol": ticker, "limit": 12}),
      get_denomination(self.client, ticker),
    )
    condensed = (_condense_earnings_surprises(result, denomination=denomination)
                 if isinstance(result, list) else result)
    envelope = build_envelope(condensed, ticker, "get_earnings_surprises",
                              api_calls_made=2)
    warnings = _denomination_warnings(ticker, denomination)
    duplicates = (condensed.get("duplicate_fiscal_periods")
                  if isinstance(condensed, dict) else None)
    if duplicates:
      warnings.append(warning(
        "duplicate_fiscal_period",
        "Finnhub returned the same fiscal quarter more than once, under "
        "different `period` buckets. beat_count, miss_count, total_periods "
        "and avg_surprise_pct count each copy, so they overweight the "
        "repeated quarter. The rows are left as Finnhub sent them; see "
        "data.duplicate_fiscal_periods for which quarters repeat.",
        duplicates=duplicates))
    _append_warnings(envelope, warnings)
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_forward_estimates(self, ticker: str) -> List[TextContent]:
    # The denomination lookup rides along in the same gather, so it costs one
    # call and no latency. It is what labels the Finnhub-served path; the
    # yfinance path is labelled from yfinance's own currency fields below.
    eps_result, rev_result, ebitda_result, denomination = await asyncio.gather(
      self.client.get("/stock/eps-estimate", {"symbol": ticker, "freq": "quarterly"}),
      self.client.get("/stock/revenue-estimate", {"symbol": ticker, "freq": "quarterly"}),
      self.client.get("/stock/ebitda-estimate", {"symbol": ticker, "freq": "quarterly"}),
      get_denomination(self.client, ticker),
    )
    condensed = _condense_forward_estimates(eps_result, rev_result, ebitda_result)
    yf_currencies = {}

    # What the primary path said, kept before the fallback overwrites it. The
    # response used to report `errors: []` while every field was served by
    # yfinance, so nothing recorded that Finnhub answered 403 three times --
    # a fixable entitlement problem that read as a silent success.
    primary_errors = [
      f"{field}: {condensed[field]['error']}"
      for field in _FORWARD_FIELDS
      if isinstance(condensed.get(field), dict) and condensed[field].get("error")
    ]

    # yfinance fallback when Finnhub free-tier returns no data on any sub-field.
    # Only fetch yfinance once per call, and only if at least one sub-field is missing.
    # Wrap in wait_for: Ticker.info is known to hang under Yahoo throttling.
    if primary_errors:
      try:
        # 30s budget — yfinance's earnings_estimate / revenue_estimate calls
        # run ~3s standalone but can hit Yahoo throttling under MCP-subprocess
        # contention (other yfinance calls in flight via get_market_data).
        yf_data = await asyncio.wait_for(
          asyncio.to_thread(_yf_forward_estimates, ticker),
          timeout=30.0,
        )
        for k in _FORWARD_FIELDS:
          if isinstance(condensed.get(k), dict) and condensed[k].get("error"):
            condensed[k] = yf_data.get(k, condensed[k])
        yf_currencies = yf_data.get("_currencies") or {}
      except asyncio.TimeoutError:
        for k in _FORWARD_FIELDS:
          if isinstance(condensed.get(k), dict) and condensed[k].get("error"):
            condensed[k] = {"error": condensed[k]["error"] + " + yfinance_timeout"}

    _label_forward_denomination(
      condensed,
      finnhub_currency=(denomination or {}).get("currency"),
      yf_quote_currency=yf_currencies.get("quote"),
      yf_reporting_currency=yf_currencies.get("reporting"))

    provider, sources = _forward_estimates_provenance(condensed)
    warnings = _forward_denomination_warnings(ticker, condensed)
    if primary_errors:
      warnings.append(warning(
        "primary_source_unavailable",
        "Finnhub returned no forward estimates for this ticker; the values "
        "below come from the fallback named in `provider` and in each field's "
        "`_source`. See metadata.errors for what Finnhub said.",
        finnhub_errors=primary_errors))
    if any(str(s or "").endswith("_inferred") for s in sources.values()):
      warnings.append(warning(
        "derived_not_an_analyst_estimate",
        "`ebitda_B` is not an analyst estimate. No forward EBITDA consensus "
        "was available, so each period is the revenue estimate multiplied by "
        "the trailing EBITDA margin held flat (`_inferred_margin`). It "
        "carries no analyst count because no analyst published it.",
        field="ebitda_B"))

    envelope = build_envelope(condensed, ticker, "get_forward_estimates",
                              api_calls_made=4, errors=primary_errors)
    # `provider` and `warnings` are set on the body deliberately. The
    # dispatcher's @annotating decorator fills both with setdefault, so a value
    # already present here survives -- which is the only way to correct a
    # per-response provider without editing the decorator's single static
    # "Finnhub" for the whole server.
    envelope["provider"] = provider
    envelope["sources"] = sources
    _append_warnings(envelope, warnings)
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_financial_statements(self, ticker: str, statement: str, freq: str) -> List[TextContent]:
    result = await self.client.get("/stock/financials", {
      "symbol": ticker, "statement": statement, "freq": freq
    })
    condensed = _condense_financial_statements(result, statement, freq) if isinstance(result, dict) else result

    # What the primary path said, kept before the fallback overwrites it --
    # the same bookkeeping get_forward_estimates does. `/stock/financials`
    # answers 403 on this plan for every symbol, and the response reported
    # `errors: []` while yfinance served NVDA's income statement, so nothing
    # recorded a fixable entitlement problem.
    primary_errors = []
    if isinstance(condensed, dict) and condensed.get("error"):
      primary_errors.append(f"{statement}/{freq}: {condensed['error']}")

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

    has_periods = isinstance(condensed, dict) and bool(condensed.get("periods"))
    source = condensed.get("_source") if isinstance(condensed, dict) else None

    envelope = build_envelope(condensed, ticker, "get_financial_statements",
                              errors=primary_errors)
    # Set on the body deliberately: @annotating fills provider and warnings
    # with setdefault, so a value already here survives, which is the only way
    # to correct a per-response provider without editing the decorator's one
    # static "Finnhub" for the whole server. A response with no periods was
    # answered by nobody, and naming a provider there would credit a source
    # for a figure it did not supply.
    envelope["provider"] = (
      _SOURCE_PROVIDERS.get(source, source or "Finnhub") if has_periods
      else "none (no financial statements retrieved)")
    if primary_errors:
      envelope["warnings"] = list(envelope.get("warnings") or []) + [warning(
        "primary_source_unavailable",
        "Finnhub declined the financial-statements request. Any periods below "
        "come from the fallback named in `provider` and in `_source`; if "
        "there are none, nothing was retrieved, which is not evidence that "
        "the company files no statements. See metadata.errors for what "
        "Finnhub said.",
        finnhub_errors=primary_errors)]
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
    condensed = (_condense_insider_sentiment(result, from_date, to_date)
                 if isinstance(result, dict) else result)
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
    except Exception:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise
    finally:
      await self.client.close()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.news_agregator.finnhub_server [server|http]", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "http":
    # Streamable HTTP, for a host a client connects to rather than one
    # that spawns it. stdio stays the default for local use.
    from tools.mcp_http import run_http
    print("Starting finnhub over streamable HTTP", file=sys.stderr, flush=True)
    run_http(FinnhubServer().server)

  elif sys.argv[1] == "server":
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
