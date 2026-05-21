"""Hedge fund position / quarterly-letter aggregator.

Sources its data from SEC EDGAR 13F-HR filings — same underlying disclosures
that quarterly letters describe. The advantage: 13Fs are structured, fully
machine-readable, and free. The disadvantage: 13Fs lag 45 days from quarter-
end and don't include narrative ("why").

For the narrative side, fund websites occasionally post letters with
predictable URL patterns; that path is fragile (sites redesign), so this
tool focuses on the structured 13F data and surfaces position-change
deltas which are the highest-signal datapoint.

Capability:
  - list_known_funds()     -> curated list of (name, cik)
  - get_fund_holdings(fund_name_or_cik, n_filings=2)
       returns latest 2 13F-HR with parsed holdings tables
  - compare_fund_holdings(...)
       returns {new, added, trimmed, exited, top_holdings} between two periods
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from edgar import Company, set_identity


# Default identity for SEC requests (sec_utils handles this too, but we
# call set_identity defensively here so the tool works standalone)
NAME = os.getenv('NAME', 'Investment Analyst')
SEC_EMAIL = os.getenv('SEC_EMAIL', 'analyst@example.com')


# Curated registry of major institutional 13F filers. Names normalized
# to lowercase, ASCII. CIKs verified against EDGAR.
KNOWN_FUNDS: Dict[str, Dict[str, Any]] = {
  # Large-cap value / superinvestors
  'berkshire hathaway':    {'cik': '0001067983', 'manager': 'Warren Buffett'},
  'berkshire':             {'cik': '0001067983', 'manager': 'Warren Buffett'},
  # Activist / event-driven
  'pershing square':       {'cik': '0001336528', 'manager': 'Bill Ackman'},
  'ackman':                {'cik': '0001336528', 'manager': 'Bill Ackman'},
  'third point':           {'cik': '0001040273', 'manager': 'Dan Loeb'},
  'loeb':                  {'cik': '0001040273', 'manager': 'Dan Loeb'},
  'icahn enterprises':     {'cik': '0000921669', 'manager': 'Carl Icahn'},
  'trian fund management': {'cik': '0001345471', 'manager': 'Nelson Peltz'},
  # Long/short / hedge
  'greenlight capital':    {'cik': '0001079114', 'manager': 'David Einhorn'},
  'einhorn':               {'cik': '0001079114', 'manager': 'David Einhorn'},
  'appaloosa':             {'cik': '0001656456', 'manager': 'David Tepper'},
  'tepper':                {'cik': '0001656456', 'manager': 'David Tepper'},
  'duquesne':              {'cik': '0001536411', 'manager': 'Stanley Druckenmiller'},
  'druckenmiller':         {'cik': '0001536411', 'manager': 'Stanley Druckenmiller'},
  'soros fund management': {'cik': '0001029160', 'manager': 'Soros family office'},
  # Value-deep
  'pabrai investment funds': {'cik': '0001549575', 'manager': 'Mohnish Pabrai'},
  'pabrai':                  {'cik': '0001549575', 'manager': 'Mohnish Pabrai'},
  'smead capital management': {'cik': '0001512912', 'manager': 'Bill Smead'},
  # Tech / growth
  'tiger global':          {'cik': '0001167483', 'manager': 'Chase Coleman'},
  'coatue management':     {'cik': '0001135730', 'manager': 'Philippe Laffont'},
  # Famous shorts / contrarian
  'scion asset management': {'cik': '0001649339', 'manager': 'Michael Burry'},
  'burry':                  {'cik': '0001649339', 'manager': 'Michael Burry'},
}


def list_known_funds() -> List[Dict[str, str]]:
  """De-dup the alias-rich registry to canonical entries."""
  seen = set()
  out = []
  for name, info in KNOWN_FUNDS.items():
    if info['cik'] in seen:
      continue
    seen.add(info['cik'])
    out.append({'fund': name, 'cik': info['cik'], 'manager': info['manager']})
  return out


def _resolve_fund(fund_name_or_cik: str) -> Optional[Dict[str, str]]:
  """Return {cik, manager, name} for a fund name or CIK. None if not found."""
  s = (fund_name_or_cik or '').strip().lower()
  if not s:
    return None
  # CIK direct match (with or without leading zeros)
  if s.isdigit() or (s.startswith('0') and s.replace('0','').isdigit()):
    cik = s.zfill(10)
    for k, v in KNOWN_FUNDS.items():
      if v['cik'].zfill(10) == cik:
        return {'name': k, 'cik': v['cik'], 'manager': v['manager']}
    return {'name': cik, 'cik': cik, 'manager': '(unknown)'}
  # Name lookup with fuzzy fallback
  if s in KNOWN_FUNDS:
    info = KNOWN_FUNDS[s]
    return {'name': s, 'cik': info['cik'], 'manager': info['manager']}
  for k, info in KNOWN_FUNDS.items():
    if s in k or k in s:
      return {'name': k, 'cik': info['cik'], 'manager': info['manager']}
  return None


def get_fund_holdings(fund_name_or_cik: str, n_filings: int = 2) -> Dict[str, Any]:
  """Pull the last N 13F-HR filings for a fund, with parsed holdings tables."""
  resolved = _resolve_fund(fund_name_or_cik)
  if not resolved:
    return {'success': False,
            'error': f'Unknown fund {fund_name_or_cik!r}. Use list_known_funds() to see available.',
            'available': [f['fund'] for f in list_known_funds()]}

  try:
    set_identity(f"{NAME} {SEC_EMAIL}")
    company = Company(resolved['cik'])
    filings = list(company.get_filings(form='13F-HR').head(n_filings))
  except Exception as e:
    return {'success': False,
            'error': f'EDGAR fetch failed for {resolved["name"]}: {type(e).__name__}: {e}',
            'fund': resolved}

  if not filings:
    return {'success': False,
            'error': f'No 13F-HR filings found for {resolved["name"]}',
            'fund': resolved}

  parsed = []
  for f in filings:
    try:
      do = f.data_object()
      holdings_df = do.holdings if do and do.has_infotable else None
    except Exception:
      holdings_df = None

    holdings_list = []
    if holdings_df is not None and not holdings_df.empty:
      for _, row in holdings_df.iterrows():
        try:
          holdings_list.append({
            'issuer':  str(row.get('Issuer', '')),
            'ticker':  str(row.get('Ticker', '')).strip(),
            'cusip':   str(row.get('Cusip', '')),
            'class':   str(row.get('Class', '')),
            'shares':  int(row.get('SharesPrnAmount', 0) or 0),
            'value':   float(row.get('Value', 0) or 0),
            'put_call': str(row.get('PutCall', '')).strip(),
          })
        except Exception:
          continue
      # Sort by value descending
      holdings_list.sort(key=lambda h: h['value'], reverse=True)

    total_value = sum(h['value'] for h in holdings_list)
    parsed.append({
      'filing_date':       str(f.filing_date),
      'accession_number':  f.accession_number,
      'total_holdings':    len(holdings_list),
      'total_value_usd':   total_value,
      'top_holdings':      holdings_list[:20],
      'all_holdings':      holdings_list,
    })

  return {
    'success':  True,
    'fund':     resolved,
    'filings':  parsed,
    'filings_count': len(parsed),
  }


def compare_fund_holdings(fund_name_or_cik: str) -> Dict[str, Any]:
  """Compare the latest 13F-HR to the prior one; surface position changes."""
  pulled = get_fund_holdings(fund_name_or_cik, n_filings=2)
  if not pulled.get('success'):
    return pulled
  filings = pulled.get('filings', [])
  if len(filings) < 2:
    return {'success': False,
            'error': 'Need at least 2 consecutive 13F filings to compare',
            'fund': pulled.get('fund'),
            'available_filings': len(filings)}

  current, prior = filings[0], filings[1]
  cur_by_cusip = {h['cusip']: h for h in current['all_holdings']}
  pri_by_cusip = {h['cusip']: h for h in prior['all_holdings']}

  new_positions: List[Dict] = []
  added: List[Dict] = []
  trimmed: List[Dict] = []
  exited: List[Dict] = []

  for cusip, cur in cur_by_cusip.items():
    pri = pri_by_cusip.get(cusip)
    if pri is None:
      new_positions.append({
        'ticker': cur['ticker'], 'issuer': cur['issuer'],
        'value_usd': cur['value'], 'shares': cur['shares'],
        'cusip': cusip,
      })
    else:
      delta_shares = cur['shares'] - pri['shares']
      if delta_shares > 0:
        delta_pct = (delta_shares / pri['shares']) * 100 if pri['shares'] else None
        added.append({
          'ticker': cur['ticker'], 'issuer': cur['issuer'],
          'shares_before': pri['shares'], 'shares_after': cur['shares'],
          'delta_shares': delta_shares,
          'delta_pct': round(delta_pct, 2) if delta_pct is not None else None,
          'value_after': cur['value'],
        })
      elif delta_shares < 0:
        delta_pct = (delta_shares / pri['shares']) * 100 if pri['shares'] else None
        trimmed.append({
          'ticker': cur['ticker'], 'issuer': cur['issuer'],
          'shares_before': pri['shares'], 'shares_after': cur['shares'],
          'delta_shares': delta_shares,
          'delta_pct': round(delta_pct, 2) if delta_pct is not None else None,
          'value_after': cur['value'],
        })

  for cusip, pri in pri_by_cusip.items():
    if cusip not in cur_by_cusip:
      exited.append({
        'ticker': pri['ticker'], 'issuer': pri['issuer'],
        'shares_before': pri['shares'], 'value_before': pri['value'],
        'cusip': cusip,
      })

  # Sort each category by magnitude
  new_positions.sort(key=lambda r: r['value_usd'], reverse=True)
  added.sort(key=lambda r: r.get('delta_pct') or 0, reverse=True)
  trimmed.sort(key=lambda r: r.get('delta_pct') or 0)  # most negative first
  exited.sort(key=lambda r: r['value_before'], reverse=True)

  return {
    'success': True,
    'fund': pulled['fund'],
    'current_filing_date': current['filing_date'],
    'prior_filing_date':   prior['filing_date'],
    'current_total_value': current['total_value_usd'],
    'prior_total_value':   prior['total_value_usd'],
    'current_positions':   current['total_holdings'],
    'prior_positions':     prior['total_holdings'],
    'new_positions':       new_positions[:25],
    'added':               added[:25],
    'trimmed':             trimmed[:25],
    'exited':              exited[:25],
    'top_holdings_current': current['top_holdings'][:10],
  }
