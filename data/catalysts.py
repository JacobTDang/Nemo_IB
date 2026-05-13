"""Catalyst calendar aggregation.

Combines per-ticker catalysts (earnings, ex-dividend) with macro catalysts
(FOMC meetings, CPI/NFP releases) into a single sorted timeline used by the
analyst to weight near-term price-moving events.

Hard schedules (FOMC, key macro releases) are embedded for 2026 since the FRED
release calendar API is rate-limited and these dates are publicly fixed. Update
this table at the start of each year.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import calendar
import os
import sys


# FOMC meeting dates 2026 (final two-day meetings, statement on day 2).
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES_2026 = [
  "2026-01-28", "2026-03-18", "2026-04-29",
  "2026-06-17", "2026-07-29", "2026-09-16",
  "2026-10-28", "2026-12-09",
]

# Macro release dates (typical day-of-month — exact dates vary).
# For prod use the FRED /releases/dates API; this is a fallback timeline.
# NFP is computed separately as the first Friday of each month — see
# _first_friday_of_month() below.
MACRO_PATTERNS = {
  'CPI':       {'day_of_month': 12, 'impact': 'rate-sensitive'},
  'GDP':       {'day_of_month': 26, 'impact': 'macro-broad'},     # advance
  'RetailSales': {'day_of_month': 15, 'impact': 'consumer'},
}


def _first_friday_of_month(year: int, month: int) -> Optional[datetime]:
  """Return the first Friday of the given month, or None if year/month invalid.
  Uses calendar.monthcalendar with the default Monday-start week (Friday is
  index 4). The first week may begin in the prior month — in that case its
  Friday entry is 0, and we fall through to the second week."""
  try:
    weeks = calendar.monthcalendar(year, month)
  except (ValueError, calendar.IllegalMonthError):
    return None
  for week in weeks:
    fri = week[4]  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
    if fri != 0:
      return datetime(year, month, fri)
  return None


def upcoming_catalysts(ticker: str, days_out: int = 60) -> List[Dict[str, Any]]:
  """Aggregate upcoming events that could move the ticker.

  Returns sorted list of {type, date, description, impact, ticker?}.
  Sources tried in order; failures are logged and skipped.
  """
  now = datetime.now()
  end = now + timedelta(days=days_out)
  out: List[Dict[str, Any]] = []

  # 1. Earnings date via Finnhub
  try:
    out.extend(_earnings_finnhub(ticker, now, end))
  except Exception as e:
    print(f"[catalysts] earnings lookup failed: {e}", file=sys.stderr, flush=True)

  # 2. Ex-dividend date via yfinance
  try:
    out.extend(_ex_dividend_yfinance(ticker, now, end))
  except Exception as e:
    print(f"[catalysts] ex-div lookup failed: {e}", file=sys.stderr, flush=True)

  # 3. FOMC schedule
  for d in FOMC_DATES_2026:
    dt = datetime.fromisoformat(d)
    if now <= dt <= end:
      out.append({
        'type': 'fomc', 'date': d,
        'description': 'FOMC interest rate decision',
        'impact': 'rate-sensitive', 'ticker': None,
      })

  # 4. Macro release pattern (approximate)
  for name, info in MACRO_PATTERNS.items():
    # Next occurrence in window
    for month_offset in range(0, (days_out // 30) + 2):
      target_month = (now.month + month_offset - 1) % 12 + 1
      target_year = now.year + (now.month + month_offset - 1) // 12
      try:
        candidate = datetime(target_year, target_month, info['day_of_month'])
      except ValueError:
        continue
      if now <= candidate <= end:
        out.append({
          'type': 'macro_release', 'date': candidate.date().isoformat(),
          'description': f'{name} release (approximate)',
          'impact': info['impact'], 'ticker': None,
        })

  # 4b. NFP — always the first Friday of each month in the window.
  for month_offset in range(0, (days_out // 30) + 2):
    target_month = (now.month + month_offset - 1) % 12 + 1
    target_year = now.year + (now.month + month_offset - 1) // 12
    nfp_date = _first_friday_of_month(target_year, target_month)
    if nfp_date is None:
      continue
    if now <= nfp_date <= end:
      out.append({
        'type': 'macro_release', 'date': nfp_date.date().isoformat(),
        'description': 'NFP release',
        'impact': 'rate-sensitive', 'ticker': None,
      })

  return sorted(out, key=lambda x: x['date'])


def _earnings_finnhub(ticker: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
  """Pull next earnings date from Finnhub /calendar/earnings."""
  api_key = os.getenv("FINNHUB_API_KEY")
  if not api_key:
    return []
  import requests
  url = "https://finnhub.io/api/v1/calendar/earnings"
  params = {
    'symbol': ticker.upper(),
    'from': start.date().isoformat(),
    'to':   end.date().isoformat(),
    'token': api_key,
  }
  resp = requests.get(url, params=params, timeout=10)
  if resp.status_code != 200:
    return []
  data = resp.json() or {}
  out = []
  for e in data.get('earningsCalendar', []):
    d = e.get('date')
    if not d:
      continue
    out.append({
      'type': 'earnings', 'date': d, 'ticker': ticker.upper(),
      'description': f"{ticker.upper()} earnings",
      'impact': 'company-specific',
      'expected_eps': e.get('epsEstimate'),
      'expected_revenue': e.get('revenueEstimate'),
      'hour': e.get('hour'),  # 'bmo' | 'amc' | 'dmh'
    })
  return out


def _ex_dividend_yfinance(ticker: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
  """Pull next ex-dividend date from yfinance info."""
  import yfinance as yf
  info = yf.Ticker(ticker.upper()).info
  ex = info.get('exDividendDate')
  if not ex:
    return []
  # yfinance returns a unix timestamp here
  try:
    dt = datetime.fromtimestamp(int(ex))
  except (TypeError, ValueError):
    return []
  if not (start <= dt <= end):
    return []
  return [{
    'type': 'ex_dividend', 'date': dt.date().isoformat(),
    'ticker': ticker.upper(),
    'description': f"{ticker.upper()} ex-dividend",
    'impact': 'price-discount-equal-to-dividend',
  }]


def summarize_for_analyst(events: List[Dict[str, Any]]) -> str:
  """Compact text summary suitable for inclusion in the analyst prompt."""
  if not events:
    return "No catalysts in window."
  lines = []
  for e in events[:15]:
    parts = [f"{e['date']}", f"{e['type']}", e.get('description', '')]
    if e.get('expected_eps') is not None:
      parts.append(f"EPS est ${e['expected_eps']:.2f}")
    if e.get('hour'):
      parts.append({'bmo': 'pre-open', 'amc': 'after-close'}.get(e['hour'], e['hour']))
    lines.append("  " + " - ".join(str(p) for p in parts if p))
  return "UPCOMING CATALYSTS:\n" + "\n".join(lines)
