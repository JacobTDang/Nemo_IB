import yfinance as yf
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import re
import sys

def get_data(ticker: str) -> Dict[str, Any]:
  data = {}

  company = yf.Ticker(ticker)
  book_value = None
  operating_income = None

  # CONSTANT POSSIBLE KEYS
  BOOK_VAL_KEYS : List = ['Stockholders Equity', 'Total Stockholder Equity','Total Equity Gross Minority Interest', 'Common Stock Equity',]
  OPERATING_INCOME_KEYS : List = ['Operating Income','Ebit']
  REVENUE_KEYS : List[str] = ['Total Revenue','Revenue','Net Sales']
  NET_INCOME_KEYS: List[str] = ['Net Income','Net Income To Common', 'Net Income From Continuing Ops']

  # get necessary information
  info = company.info
  data['ticker'] = ticker
  data['marketCap'] = info.get('marketCap')
  data['currentPrice'] = info.get('currentPrice') or info.get('regularMarketPrice')
  # yfinance `totalRevenue` and `ebitda` ARE trailing-twelve-month values.
  # Expose them under explicit `*_ttm` aliases so the analyst prompt can
  # distinguish TTM (these) from latest-annual values (from SEC tools like
  # get_revenue_base / get_ebitda_margin which return last 10-K's FY total).
  # The legacy `revenue` / `EBITDA` keys are kept for back-compat with the
  # DCF/credit/LBO callers that already read them.
  data['revenue'] = info.get('totalRevenue')
  data['revenue_ttm'] = data['revenue']
  data['EBITDA'] = info.get('ebitda')
  data['ebitda_ttm'] = data['EBITDA']
  data['netIncomeToCommon'] = info.get('netIncomeToCommon')
  data['net_income_ttm'] = data['netIncomeToCommon']
  data['enterpriseValue'] = info.get('enterpriseValue')
  data['cash'] = info.get('totalCash', 0)
  data['totalDebt'] = info.get('totalDebt', 0)
  data['sharesOutstanding'] = info.get('sharesOutstanding')
  data['beta'] = info.get('beta')

  # safely get interest expense from income statement
  INTEREST_EXPENSE_KEYS: List[str] = ['Interest Expense', 'Interest Expense Non Operating', 'Net Interest Income']
  try:
    income_stmt = company.income_stmt
    ie_key = find_key(INTEREST_EXPENSE_KEYS, income_stmt.index)
    if ie_key:
      interest_expense = income_stmt.loc[ie_key].iloc[0]
      data['interestExpense'] = abs(float(interest_expense)) if interest_expense is not None else None
    else:
      data['interestExpense'] = None
  except Exception as e:
    print(f'Could not get interest expense for {ticker}: {str(e)}', file=sys.stderr)
    data['interestExpense'] = None

  # safely get the balancesheet and book_value
  try:
    balance_sheet = company.balance_sheet
    key = find_key(BOOK_VAL_KEYS, balance_sheet.index)
    # .loc finds the row with key and .iloc will get the first col / most recent year
    book_value = balance_sheet.loc[key].iloc[0]
  except Exception as e:
    print(f'Could not get book value for {ticker} : {str(e)}', file=sys.stderr)

  data['EBIT'] = None

  # safely get the operating_income from the income statement
  try:
    income_statement = company.income_stmt
    key = find_key(OPERATING_INCOME_KEYS, income_statement.index)
    operating_income = income_statement.loc[key].iloc[0]
    data['EBIT'] = operating_income

  except Exception as e:
    print(f"Could not get the operating income from income statement for {ticker} : {str(e)}", file=sys.stderr)


  # calculate multiples -- each wrapped independently so one failure doesn't skip all
  try:
    if data['marketCap'] is not None and data['netIncomeToCommon'] is not None:
      data['pe_ratio'] = data['marketCap'] / data['netIncomeToCommon']
  except Exception as e:
    print(f'Error calculating P/E ratio for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['marketCap'] is not None and book_value is not None:
      data['pb_ratio'] = data['marketCap'] / book_value
  except Exception as e:
    print(f'Error calculating P/B ratio for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['enterpriseValue'] is not None and data['revenue'] is not None:
      data['ev_revenue'] = data['enterpriseValue'] / data['revenue']
  except Exception as e:
    print(f'Error calculating EV/Revenue for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['enterpriseValue'] is not None and data['EBITDA'] is not None:
      data['ev_ebitda'] = data['enterpriseValue'] / data['EBITDA']
  except Exception as e:
    print(f'Error calculating EV/EBITDA for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['EBIT'] is not None and data['enterpriseValue'] is not None:
      data['ev_ebit'] = data['enterpriseValue'] / data['EBIT']
  except Exception as e:
    print(f'Error calculating EV/EBIT for {ticker}: {str(e)}', file=sys.stderr)

  return data


def find_key(possible_key : List[str], indexes: pd.Index) -> Optional[str]:
  # find key with in the list of indexes
  for key in possible_key:
    if key in indexes:
      return str(key)

  # if that fails then we use a llm fallback - TODO later when I implement quantiazation for models
  print(f'Unable to find key, using llm to compare indexes to possible keys: {possible_key}', file=sys.stderr)

  # complete failure
  print('complete failure, key DNE', file=sys.stderr)
  return None

# Curated theme-to-ETF map. Each theme maps to one or more ETF tickers that
# offer concentrated exposure. ETFs chosen by AUM + holdings purity.
_THEME_TO_ETFS: Dict[str, List[str]] = {
  # Tech / AI / semis
  'semiconductors':    ['SMH', 'SOXX', 'XSD'],
  'semis':             ['SMH', 'SOXX', 'XSD'],
  'ai':                ['BOTZ', 'AIQ', 'ROBO', 'ARTY'],
  'artificial intelligence': ['BOTZ', 'AIQ', 'ROBO'],
  'robotics':          ['BOTZ', 'ROBO', 'IRBO'],
  'cloud':             ['SKYY', 'WCLD', 'CLOU'],
  'cybersecurity':     ['HACK', 'CIBR', 'BUG'],
  'fintech':           ['FINX', 'IPAY', 'KFIN'],
  'software':          ['IGV', 'XSW'],
  'internet':          ['FDN', 'PNQI'],
  'tech':              ['XLK', 'VGT', 'QQQ'],
  # Energy / commodities
  'energy':            ['XLE', 'XOP', 'IEO'],
  'oil':               ['XOP', 'OIH', 'IEZ'],
  'natural gas':       ['UNG', 'FCG'],
  'uranium':           ['URA', 'URNM'],
  'lithium':           ['LIT', 'BATT'],
  'battery':           ['LIT', 'BATT'],
  'clean energy':      ['ICLN', 'QCLN', 'TAN'],
  'solar':             ['TAN', 'PBW'],
  'wind':              ['FAN'],
  'utilities':         ['XLU', 'VPU'],
  # Financials
  'banks':             ['KBE', 'KRE', 'XLF'],
  'financials':        ['XLF', 'VFH'],
  # Healthcare / biotech
  'healthcare':        ['XLV', 'VHT'],
  'biotech':           ['IBB', 'XBI', 'BBH'],
  'pharma':            ['IHE', 'PJP'],
  'medical devices':   ['IHI'],
  # Consumer
  'consumer discretionary': ['XLY', 'VCR'],
  'consumer staples':       ['XLP', 'VDC'],
  'retail':                 ['XRT', 'RTH'],
  # Industrials / infra
  'industrials':       ['XLI', 'VIS'],
  'aerospace':         ['ITA', 'PPA'],
  'defense':           ['ITA', 'PPA', 'XAR'],
  'infrastructure':    ['PAVE', 'IFRA'],
  'rare earth':        ['REMX'],
  # Macro / geographic
  'gold':              ['GLD', 'IAU', 'GDX'],
  'silver':            ['SLV', 'SIL'],
  'china':             ['FXI', 'KWEB', 'MCHI'],
  'india':             ['INDA', 'EPI'],
  'japan':             ['EWJ'],
  'emerging markets':  ['EEM', 'VWO', 'IEMG'],
  # Real estate
  'reit':              ['VNQ', 'IYR', 'XLRE'],
  'real estate':       ['VNQ', 'IYR', 'XLRE'],
  'data centers':      ['DTCR', 'SRVR'],
  # Themes
  'electric vehicles': ['DRIV', 'KARS', 'IDRV'],
  'ev':                ['DRIV', 'KARS', 'IDRV'],
  'genomics':          ['ARKG'],
  'space':             ['UFO', 'ARKX'],
  'esports':           ['HERO', 'GAMR'],
  'metaverse':         ['META', 'METV'],
}


_ANALOGUES_CACHE: List[Dict[str, Any]] = []


def _load_analogues() -> List[Dict[str, Any]]:
  """Load the analogues knowledge file into a list of {name, tags, body}.
  Cached in-process — file is checked into the repo, doesn't change at
  runtime."""
  global _ANALOGUES_CACHE
  if _ANALOGUES_CACHE:
    return _ANALOGUES_CACHE
  import re as _re
  from pathlib import Path as _P
  path = _P(__file__).resolve().parents[2] / 'knowledge' / 'analogues.md'
  if not path.exists():
    return []
  text = path.read_text(encoding='utf-8')
  # Split on numbered section headers: ## N. Title
  sections = _re.split(r'\n##\s+\d+\.\s+', text)
  out = []
  for sec in sections[1:]:  # skip preamble
    # First line is the title
    lines = sec.split('\n', 1)
    title = lines[0].strip()
    body = lines[1] if len(lines) > 1 else ''
    # Extract tags line
    tag_match = _re.search(r'\*\*Tags:\*\*\s*([^\n]+)', body)
    tags = []
    if tag_match:
      tags = [t.strip().rstrip(',').lower()
              for t in tag_match.group(1).split(',')]
      tags = [t for t in tags if t]
    out.append({
      'name': title,
      'tags': tags,
      'body': body.strip(),
    })
  _ANALOGUES_CACHE = out
  return out


def get_historical_analogue(thesis_description: str,
                            top_n: int = 3) -> Dict[str, Any]:
  """Match a current thesis description against the curated catalog of
  historical investment periods. Returns top N matches by tag-overlap
  score plus the catalog's lesson for each match.

  The thesis description should contain structural keywords matching
  the analogue tag schema (capex_cycle, valuation_expansion,
  margin_compression, supply_constrained, etc.). The tool scans the
  description case-insensitively for tag tokens and ranks analogues by
  how many tags they share with the thesis.
  """
  analogues = _load_analogues()
  if not analogues:
    return {
      'success': False,
      'error': 'analogues.md not found or empty',
    }

  td = (thesis_description or '').lower()

  # Score each analogue by token overlap with the description.
  # Token = a tag name, but we also treat each whitespace word as a token
  # so descriptive phrases like "AI capex peak" can match the
  # 'capex_peak' tag too.
  description_tokens = set(re.findall(r'[a-z_]+', td))
  # Convert multi-word phrases (e.g. "capex peak") to underscore form for
  # tag matching
  td_underscored = re.sub(r'\s+', '_', td)
  underscored_tokens = set(re.findall(r'[a-z_]{3,}', td_underscored))
  description_tokens.update(underscored_tokens)

  scored = []
  for a in analogues:
    matches = []
    for tag in a['tags']:
      # Tag like "capex_peak" — check if any component matches
      tag_norm = tag.strip().lower()
      if tag_norm in td_underscored or tag_norm in td:
        matches.append(tag_norm)
        continue
      parts = tag_norm.split('_')
      if all(p in description_tokens for p in parts if p):
        matches.append(tag_norm)
    # Also boost when sector name appears in description
    sector_tags = {'tech', 'energy', 'financials', 'commodities',
                   'consumer', 'real_estate', 'biotech'}
    sector_matched = any(t in sector_tags for t in matches)
    score = len(matches) + (2 if sector_matched else 0)
    if score > 0:
      scored.append({
        'name': a['name'],
        'score': score,
        'matched_tags': matches,
        'all_tags': a['tags'],
        'body_excerpt': a['body'][:1500],
      })

  scored.sort(key=lambda r: r['score'], reverse=True)

  return {
    'success': True,
    'thesis_description': thesis_description,
    'top_matches': scored[:top_n],
    'analogues_catalog_size': len(analogues),
    'note': "Matching is by tag-token overlap, not semantic similarity. Tag taxonomy is documented in knowledge/analogues.md preamble. Adjust thesis description to include structural tags (capex_cycle, valuation_expansion, supply_constrained, etc.) for better matches.",
  }


def get_industry_etfs(theme: str, top_holdings_per_etf: int = 10) -> Dict[str, Any]:
  """Map a research theme (e.g. 'AI semis', 'energy', 'cloud') to relevant
  ETFs and return their top holdings + weights.

  Acts as the bridge from top-down thematic conviction to bottom-up
  ticker selection. Searches the theme map; if no exact match, tries
  substring match across theme keys. Returns up to 3 ETFs per theme with
  their top N holdings.
  """
  out: Dict[str, Any] = {
    'theme_query': theme,
    'success': True,
    'error': None,
  }

  theme_norm = theme.lower().strip()
  # Exact match first
  etfs = _THEME_TO_ETFS.get(theme_norm)
  matched_themes = [theme_norm] if etfs else []

  # Fuzzy / substring match across keys
  if not etfs:
    matches = {}
    for key, etf_list in _THEME_TO_ETFS.items():
      if theme_norm in key or key in theme_norm:
        matches[key] = etf_list
    if matches:
      # Combine ETF lists (dedup, preserve order)
      seen = set()
      etfs = []
      for k, lst in matches.items():
        for e in lst:
          if e not in seen:
            seen.add(e)
            etfs.append(e)
      matched_themes = list(matches.keys())

  if not etfs:
    return {
      'theme_query': theme,
      'success': False,
      'error': f'No ETF mapping for theme {theme!r}',
      'available_themes': sorted(_THEME_TO_ETFS.keys()),
    }

  out['matched_themes'] = matched_themes
  out['etfs_matched'] = etfs[:5]  # cap at 5 to keep payload bounded

  # Fetch holdings for each ETF
  etf_details = []
  for etf_symbol in out['etfs_matched']:
    detail: Dict[str, Any] = {'symbol': etf_symbol}
    try:
      t = yf.Ticker(etf_symbol)
      info = t.info
      detail['name'] = info.get('longName') or info.get('shortName')
      detail['category'] = info.get('category')
      detail['total_assets'] = info.get('totalAssets')
      detail['expense_ratio'] = info.get('annualReportExpenseRatio')

      fd = t.funds_data
      if fd:
        try:
          th = fd.top_holdings
          holdings = []
          if hasattr(th, 'iterrows'):
            for sym, row in th.head(top_holdings_per_etf).iterrows():
              holdings.append({
                'symbol': str(sym),
                'name': str(row.get('Name', '')),
                'weight_pct': round(float(row.get('Holding Percent', 0)) * 100, 2),
              })
          detail['top_holdings'] = holdings
        except Exception as exc:
          detail['holdings_error'] = f'{type(exc).__name__}: {exc}'

        try:
          sw = fd.sector_weightings
          if isinstance(sw, dict) and sw:
            detail['sector_weightings'] = {k: round(float(v) * 100, 2) for k, v in sw.items()}
        except Exception:
          pass
    except Exception as exc:
      detail['error'] = f'{type(exc).__name__}: {exc}'
    etf_details.append(detail)

  out['etfs'] = etf_details
  return out


def get_price_history(ticker: str, period: str = '2y',
                      include_recent_bars: int = 20) -> Dict[str, Any]:
  """Historical OHLCV summary from yfinance.

  Returns aggregate metrics rather than the raw bars (which would blow out
  MCP payload size): returns over 1M/3M/6M/YTD/1Y/3Y, realized volatility
  over the same windows, 52-week high/low with dates, max drawdown from
  trailing-12-month peak, and the most recent N daily OHLCV bars for
  technical reference.
  """
  from datetime import datetime, timedelta

  try:
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=True)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'yfinance history fetch failed: {type(e).__name__}: {e}'}

  if df is None or df.empty:
    return {'ticker': ticker, 'success': False, 'error': 'no price history returned'}

  close = df['Close']
  returns = close.pct_change().dropna()

  def _ret_over(days: int):
    if len(close) <= days:
      return None
    return round(((close.iloc[-1] / close.iloc[-days - 1]) - 1) * 100, 2)

  def _vol_over(days: int):
    if len(returns) < days:
      return None
    sub = returns.iloc[-days:]
    return round(float(sub.std() * (252 ** 0.5) * 100), 2)

  # YTD return: from first trading day of this calendar year
  today = datetime.now()
  jan1 = datetime(today.year, 1, 1)
  ytd_idx = df.index[df.index >= pd.Timestamp(jan1, tz=df.index.tz)] if df.index.tz else \
            df.index[df.index >= jan1]
  ytd_ret = None
  if len(ytd_idx) > 0:
    ytd_start = close.loc[ytd_idx[0]]
    if ytd_start:
      ytd_ret = round(((close.iloc[-1] / ytd_start) - 1) * 100, 2)

  # 52w stats (last 252 trading days)
  win52 = df.iloc[-252:] if len(df) >= 252 else df
  high_idx = win52['High'].idxmax()
  low_idx = win52['Low'].idxmin()

  # Max drawdown over trailing 12 months
  running_max = win52['Close'].expanding().max()
  drawdown = (win52['Close'] / running_max - 1) * 100
  max_dd = float(drawdown.min())
  max_dd_date = drawdown.idxmin()
  max_dd_peak_close = float(running_max.loc[max_dd_date])

  # Recent bars
  recent = df.tail(include_recent_bars).copy()
  recent_bars = []
  for ts, row in recent.iterrows():
    recent_bars.append({
      'date': ts.strftime('%Y-%m-%d'),
      'open': round(float(row['Open']), 2),
      'high': round(float(row['High']), 2),
      'low': round(float(row['Low']), 2),
      'close': round(float(row['Close']), 2),
      'volume': int(row['Volume']),
    })

  return {
    'ticker':            ticker.upper(),
    'success':           True,
    'error':             None,
    'period_requested':  period,
    'bars_returned':     len(df),
    'date_range': {
      'start': df.index[0].strftime('%Y-%m-%d'),
      'end':   df.index[-1].strftime('%Y-%m-%d'),
    },
    'current_close':     round(float(close.iloc[-1]), 2),
    'returns_pct': {
      '1m':  _ret_over(21),
      '3m':  _ret_over(63),
      '6m':  _ret_over(126),
      'ytd': ytd_ret,
      '1y':  _ret_over(252),
      '3y':  _ret_over(252 * 3),
    },
    'realized_vol_annualized_pct': {
      '30d':  _vol_over(30),
      '90d':  _vol_over(90),
      '180d': _vol_over(180),
      '1y':   _vol_over(252),
    },
    'fifty_two_week': {
      'high':     round(float(win52['High'].max()), 2),
      'high_date': high_idx.strftime('%Y-%m-%d'),
      'low':      round(float(win52['Low'].min()), 2),
      'low_date': low_idx.strftime('%Y-%m-%d'),
    },
    'max_drawdown_12m': {
      'drawdown_pct':       round(max_dd, 2),
      'trough_date':        max_dd_date.strftime('%Y-%m-%d'),
      'peak_close_before':  round(max_dd_peak_close, 2),
      'trough_close':       round(float(win52['Close'].loc[max_dd_date]), 2),
    },
    'recent_bars':       recent_bars,
  }


def get_short_interest(ticker: str) -> Dict[str, Any]:
  """Short-interest snapshot from yfinance (underlying source: FINRA biweekly).

  Returns shares short, short ratio (days to cover), percent of float, and
  MoM trend (current vs prior-month shares short). Crowded shorts can
  signal squeeze risk or strong bear conviction; low shorts on a high-
  quality name indicate institutional acceptance.
  """
  from datetime import datetime, timezone

  try:
    t = yf.Ticker(ticker)
    info = t.info
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'yfinance init failed: {type(e).__name__}: {e}'}

  def _epoch_to_iso(v):
    try:
      return datetime.fromtimestamp(int(v), tz=timezone.utc).strftime('%Y-%m-%d')
    except (TypeError, ValueError):
      return None

  shares_short = info.get('sharesShort')
  shares_short_prior = info.get('sharesShortPriorMonth')
  mom_change_pct = None
  if shares_short and shares_short_prior:
    mom_change_pct = round(((shares_short / shares_short_prior) - 1) * 100, 2)

  short_pct_float = info.get('shortPercentOfFloat')
  short_ratio = info.get('shortRatio')

  # Sentiment label based on % of float
  signal = 'unknown'
  if short_pct_float is not None:
    if short_pct_float < 0.02:
      signal = 'low_short_interest'
    elif short_pct_float < 0.05:
      signal = 'moderate_short_interest'
    elif short_pct_float < 0.10:
      signal = 'elevated_short_interest'
    else:
      signal = 'crowded_short_squeeze_risk'

  return {
    'ticker': ticker.upper(),
    'success': True,
    'error': None,
    'shares_short':              int(shares_short) if shares_short else None,
    'shares_short_prior_month':  int(shares_short_prior) if shares_short_prior else None,
    'mom_change_pct':            mom_change_pct,
    'short_ratio_days_to_cover': float(short_ratio) if short_ratio else None,
    'short_pct_of_float':        round(float(short_pct_float) * 100, 3) if short_pct_float else None,
    'float_shares':              int(info.get('floatShares', 0)) or None,
    'shares_outstanding':        int(info.get('sharesOutstanding', 0)) or None,
    'as_of_date':                _epoch_to_iso(info.get('dateShortInterest')),
    'prior_month_date':          _epoch_to_iso(info.get('sharesShortPreviousMonthDate')),
    'signal':                    signal,
    'source':                    'yfinance (underlying: FINRA biweekly short interest)',
  }


def get_options_metrics(ticker: str) -> Dict[str, Any]:
  """Compute key options-market metrics from yfinance option chains.

  Returns:
    - term structure (ATM IV at ~7d/30d/60d/90d expirations)
    - 30d put/call skew (downside vs. upside IV)
    - nearest-expiry open interest + volume put/call ratios

  Filters out illiquid contracts (bid == 0) to avoid yfinance's garbage IV
  values on deep ITM/OTM strikes. ATM is defined as the strike closest to
  spot. Skew compares 0.9*spot put IV to 1.1*spot call IV.
  """
  from datetime import datetime, date as _date

  out: Dict[str, Any] = {'ticker': ticker.upper(), 'success': True, 'error': None}

  try:
    t = yf.Ticker(ticker)
    info = t.info
    spot = info.get('currentPrice') or info.get('regularMarketPrice')
    if not spot:
      return {'ticker': ticker, 'success': False, 'error': 'no spot price'}
    out['spot_price'] = float(spot)

    exps = list(t.options)
    if not exps:
      return {'ticker': ticker, 'success': False, 'error': 'no options listed'}
    out['expirations_available'] = len(exps)

    today = _date.today()
    exp_dates = []
    for e in exps:
      try:
        d = (datetime.strptime(e, '%Y-%m-%d').date() - today).days
        exp_dates.append((e, d))
      except ValueError:
        continue

    def _find_expiry(target_dte: int) -> tuple:
      future = [(e, d) for e, d in exp_dates if d >= target_dte]
      pool = future if future else exp_dates
      return min(pool, key=lambda x: abs(x[1] - target_dte))

    def _atm_iv(chain, side: str) -> Optional[float]:
      df = chain.calls if side == 'call' else chain.puts
      # Filter out yfinance's IV=0.00001 sentinel for inactive contracts but
      # keep bid==0 contracts (some ATM strikes show bid=0 but valid IV).
      df = df[df['impliedVolatility'] > 0.01]
      if df.empty:
        return None
      idx = (df['strike'] - spot).abs().idxmin()
      return float(df.loc[idx, 'impliedVolatility'])

    # Term structure
    term_structure = {}
    for label, target_days in [('7d', 7), ('30d', 30), ('60d', 60), ('90d', 90)]:
      exp, dte = _find_expiry(target_days)
      try:
        chain = t.option_chain(exp)
      except Exception:
        term_structure[label] = {'expiry': exp, 'dte': dte, 'error': 'chain fetch failed'}
        continue
      call_iv = _atm_iv(chain, 'call')
      put_iv = _atm_iv(chain, 'put')
      atm_iv = None
      if call_iv is not None and put_iv is not None:
        atm_iv = (call_iv + put_iv) / 2
      elif call_iv is not None:
        atm_iv = call_iv
      elif put_iv is not None:
        atm_iv = put_iv
      term_structure[label] = {
        'expiry':         exp,
        'dte':            dte,
        'atm_call_iv':    round(call_iv, 4) if call_iv is not None else None,
        'atm_put_iv':     round(put_iv, 4) if put_iv is not None else None,
        'atm_iv':         round(atm_iv, 4) if atm_iv is not None else None,
      }
    out['term_structure'] = term_structure

    # 30d skew
    exp_30, _ = _find_expiry(30)
    try:
      chain30 = t.option_chain(exp_30)
      calls = chain30.calls[chain30.calls['impliedVolatility'] > 0.01]
      puts = chain30.puts[chain30.puts['impliedVolatility'] > 0.01]
      otm_put_idx = (puts['strike'] - 0.9 * spot).abs().idxmin() if not puts.empty else None
      otm_call_idx = (calls['strike'] - 1.1 * spot).abs().idxmin() if not calls.empty else None
      put_iv_90 = float(puts.loc[otm_put_idx, 'impliedVolatility']) if otm_put_idx is not None else None
      call_iv_110 = float(calls.loc[otm_call_idx, 'impliedVolatility']) if otm_call_idx is not None else None
      skew = None
      if put_iv_90 is not None and call_iv_110 is not None:
        skew = round(put_iv_90 - call_iv_110, 4)
      out['put_call_skew_30d'] = {
        'value':       skew,
        'put_iv_90pct': round(put_iv_90, 4) if put_iv_90 is not None else None,
        'call_iv_110pct': round(call_iv_110, 4) if call_iv_110 is not None else None,
        'expiry':      exp_30,
        'note':        '0.9*spot put IV minus 1.1*spot call IV; positive=downside fear, negative=upside speculation',
      }
    except Exception as exc:
      out['put_call_skew_30d'] = {'error': str(exc)}

    # Nearest-expiry OI and volume aggregates
    try:
      chain_near = t.option_chain(exps[0])
      c_oi = int(chain_near.calls['openInterest'].fillna(0).sum())
      p_oi = int(chain_near.puts['openInterest'].fillna(0).sum())
      c_vol = int(chain_near.calls['volume'].fillna(0).sum())
      p_vol = int(chain_near.puts['volume'].fillna(0).sum())
      out['nearest_expiry_activity'] = {
        'expiry':                  exps[0],
        'call_open_interest':      c_oi,
        'put_open_interest':       p_oi,
        'put_call_oi_ratio':       round(p_oi / c_oi, 3) if c_oi else None,
        'call_volume':             c_vol,
        'put_volume':              p_vol,
        'put_call_volume_ratio':   round(p_vol / c_vol, 3) if c_vol else None,
      }
    except Exception as exc:
      out['nearest_expiry_activity'] = {'error': str(exc)}

    # Data-quality check. yfinance sometimes returns sentinel IV values
    # (powers of 2 fractions like 0.0156, 0.0625, 0.125) when the underlying
    # market snapshot is stale or contracts are illiquid. Real ATM IVs for
    # US large-caps live in the 0.15-0.60 range; anything below 0.08 is
    # almost certainly bad data.
    iv_vals = [
      entry.get('atm_iv') for entry in term_structure.values()
      if isinstance(entry, dict) and entry.get('atm_iv') is not None
    ]
    iv_quality = 'ok'
    iv_quality_notes = []
    if iv_vals:
      if max(iv_vals) < 0.08:
        iv_quality = 'suspect_iv_sentinel'
        iv_quality_notes.append(
          'all ATM IVs < 0.08 — likely yfinance sentinel values, not real implied volatility')
      elif len(set(round(v, 4) for v in iv_vals)) == 1:
        iv_quality = 'suspect_iv_constant'
        iv_quality_notes.append('all ATM IVs identical across tenors — suspect data')
    else:
      iv_quality = 'no_iv_data'
      iv_quality_notes.append('no ATM IV could be extracted')

    # Volume ratios are usually still reliable even when IV is sentinel
    out['data_quality'] = {
      'iv_status':   iv_quality,
      'notes':       iv_quality_notes,
      'volume_data_usable': True,
    }

    return out

  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'get_options_metrics failed: {type(e).__name__}: {e}'}


def get_institutional_holdings(ticker: str, top_n: int = 10) -> Dict[str, Any]:
  """Pull aggregated 13F institutional holdings via yfinance.

  Source layering: Yahoo aggregates SEC 13F-HR filings server-side, so the
  underlying data is SEC-tier but the aggregation/freshness is Yahoo's.
  Tagged as vendor-tier in the playbook hierarchy.

  Returns top N institutional holders + top N mutual fund holders with
  shares, market value, percent of shares outstanding, and quarter-over-
  quarter percent change in position. Also returns aggregate institutional
  ownership stats (institutions %, insiders %, total institution count).
  """
  out: Dict[str, Any] = {
    "ticker": ticker.upper(),
    "success": True,
    "error": None,
    "source": "yfinance (aggregates SEC 13F-HR)",
  }

  try:
    t = yf.Ticker(ticker)
  except Exception as exc:
    return {"ticker": ticker, "success": False,
            "error": f"yfinance Ticker init failed: {type(exc).__name__}: {exc}"}

  # Aggregate stats (institutions %, insiders %, count)
  try:
    mh = t.major_holders
    if mh is not None and not mh.empty:
      vals = {}
      for idx, row in mh.iterrows():
        try:
          vals[str(idx)] = float(row['Value'])
        except (KeyError, TypeError, ValueError):
          continue
      out['aggregate'] = {
        'insiders_pct': vals.get('insidersPercentHeld'),
        'institutions_pct': vals.get('institutionsPercentHeld'),
        'institutions_float_pct': vals.get('institutionsFloatPercentHeld'),
        'institutions_count': int(vals['institutionsCount']) if 'institutionsCount' in vals else None,
      }
  except Exception as exc:
    out['aggregate_error'] = f"{type(exc).__name__}: {exc}"

  def _holders_to_list(df, cap: int) -> List[Dict[str, Any]]:
    rows = []
    for _, r in df.head(cap).iterrows():
      pct_change = r.get('pctChange')
      try:
        pct_change_f = float(pct_change) if pd.notna(pct_change) else None
      except (TypeError, ValueError):
        pct_change_f = None
      rows.append({
        'holder':         str(r.get('Holder', '')),
        'date_reported':  str(r.get('Date Reported', '')),
        'pct_held':       float(r.get('pctHeld', 0)) if pd.notna(r.get('pctHeld', 0)) else None,
        'shares':         int(r.get('Shares', 0)) if pd.notna(r.get('Shares', 0)) else None,
        'value_usd':      float(r.get('Value', 0)) if pd.notna(r.get('Value', 0)) else None,
        'pct_change_qoq': pct_change_f,
      })
    return rows

  # Institutional (13F) holders
  try:
    ih = t.institutional_holders
    if ih is not None and not ih.empty:
      out['institutional_holders'] = _holders_to_list(ih, top_n)
  except Exception as exc:
    out['institutional_error'] = f"{type(exc).__name__}: {exc}"

  # Mutual fund holders (NPORT-P filings, also aggregated by Yahoo)
  try:
    mf = t.mutualfund_holders
    if mf is not None and not mf.empty:
      out['mutualfund_holders'] = _holders_to_list(mf, top_n)
  except Exception as exc:
    out['mutualfund_error'] = f"{type(exc).__name__}: {exc}"

  return out


def calculate_percentiles(data: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
  percentiles = {}
  # build the list of values based on key
  values = [d[key] for d in data if d.get(key) is not None]

  if not values:
    print(f'No valid data found for key: {str(key)}', file=sys.stderr)
    return {}

  # calcaute statistics
  percentiles['mean'] = np.mean(values)
  percentiles['median'] = np.median(values)
  percentiles['q1'] = np.percentile(values, 25)
  percentiles['q3'] = np.percentile(values, 75)
  percentiles['low'] = np.min(values)
  percentiles['high'] = np.max(values)

  return percentiles
if __name__ == "__main__":
  data = get_data("MSFT")
  print(data['cash'])
  print(data['totalDebt'])
  print(data['sharesOutstanding'])
