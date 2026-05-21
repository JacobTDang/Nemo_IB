"""Cross-thesis exposure analyzer.

Decomposes a portfolio of active theses into latent factor buckets so the
analyst can see hidden concentration — thinking they have 10 diversified
bets when they actually have 10 bets on the same factor.

V1 approach: pattern-match each thesis's ticker, sector, and thesis text
against a curated factor taxonomy. Factors are coarse on purpose
(AI capex, rate sensitivity, USD strength, China consumer, oil beta,
commodity supply, etc.) — the goal is to surface obvious dependencies,
not to run formal PCA.

Output per factor: list of theses with that exposure, sum of position
weights (if available), and the dominant signal direction.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# Factor → matching rules. Each rule is a dict with:
#   - tickers: explicit tickers that have this exposure
#   - keywords: keywords in the thesis text that imply this exposure
#   - sectors: GICS-like sector keywords
# A thesis matches a factor if ANY of these match.
FACTOR_TAXONOMY: Dict[str, Dict[str, Any]] = {
  'ai_capex_long': {
    'description': "Long exposure to AI infrastructure capex cycle",
    'tickers': {'NVDA', 'AMD', 'TSM', 'AVGO', 'MU', 'MSFT', 'GOOG', 'GOOGL',
                'META', 'AMZN', 'ARM', 'SNOW', 'PLTR', 'ASML', 'LRCX', 'KLAC',
                'AMAT', 'CRWV', 'SMCI'},
    'keywords': ['ai capex', 'hyperscaler', 'gpu', 'hbm', 'data center',
                 'cloud capex', 'azure', 'aws', 'generative ai',
                 'inference', 'training', 'cowos', 'compute demand'],
  },
  'memory_cycle': {
    'description': "Memory / DRAM / HBM cycle exposure",
    'tickers': {'MU', 'WDC', 'STX'},
    'keywords': ['dram', 'hbm', 'memory cycle', 'flash memory', 'nand'],
  },
  'rate_sensitive_long_duration': {
    'description': "Long-duration assets — multiple compression risk in rising-rate regime",
    'tickers': {'TSLA', 'PLTR', 'SHOP', 'SNOW', 'CRWD', 'NET', 'DDOG', 'MDB',
                'ARKK', 'AFRM', 'COIN'},
    'keywords': ['high multiple', 'long duration', 'growth stock',
                 'unprofitable growth', 'high p/s', 'terminal value'],
  },
  'rate_sensitive_short_duration': {
    'description': "Rate-beneficiary names (banks, insurers, value)",
    'tickers': {'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'V', 'MA',
                'BLK', 'SCHW'},
    'keywords': ['net interest margin', 'nii benefit', 'bank earnings'],
  },
  'energy_long': {
    'description': "Long oil/gas/energy",
    'tickers': {'XOM', 'CVX', 'COP', 'EOG', 'PXD', 'FANG', 'HES', 'OXY',
                'SLB', 'HAL', 'BKR'},
    'keywords': ['oil price', 'wti', 'brent', 'opec', 'natural gas', 'shale'],
  },
  'china_exposure': {
    'description': "Direct or indirect China demand exposure",
    'tickers': {'BABA', 'JD', 'PDD', 'TCEHY', 'NIO', 'LI', 'XPEV', 'BIDU',
                'LVS', 'WYNN', 'YUMC', 'TME'},
    'keywords': ['china demand', 'beijing', 'shanghai', 'tariff',
                 'china property', 'china consumer'],
  },
  'commodity_supply_constrained': {
    'description': "Long commodities where supply is constrained",
    'tickers': {'CCJ', 'NXE', 'UEC', 'URA', 'URNM', 'FCX', 'BHP', 'RIO',
                'VALE', 'NEM', 'GLD', 'GDX'},
    'keywords': ['supply constrained', 'capex drought', 'reserve replacement',
                 'uranium', 'copper', 'lithium'],
  },
  'gold_macro_hedge': {
    'description': "Gold / dollar / macro hedge exposure",
    'tickers': {'GLD', 'IAU', 'GDX', 'GDXJ', 'NEM', 'AEM', 'KGC'},
    'keywords': ['gold', 'dollar weakness', 'real yields', 'inflation hedge'],
  },
  'consumer_discretionary': {
    'description': "Discretionary consumer demand exposure",
    'tickers': {'AMZN', 'TSLA', 'HD', 'LOW', 'TGT', 'NKE', 'SBUX',
                'MCD', 'DIS', 'BKNG'},
    'keywords': ['consumer spending', 'discretionary'],
  },
  'crypto_beta': {
    'description': "Crypto beta exposure",
    'tickers': {'COIN', 'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'BITO',
                'IBIT', 'FBTC'},
    'keywords': ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth', 'blockchain'],
  },
  'biotech_speculative': {
    'description': "Speculative biotech / drug-pipeline binary",
    'tickers': {'MRNA', 'BNTX', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'XBI'},
    'keywords': ['phase 3', 'fda approval', 'pdufa', 'clinical trial',
                 'breakthrough designation'],
  },
}


def _classify_thesis(thesis: Dict[str, Any]) -> List[str]:
  """Return the list of factor names this thesis matches."""
  ticker = (thesis.get('ticker') or '').upper()
  text_blob = ' '.join([
    str(thesis.get('analysis_summary') or ''),
    str(thesis.get('variant_perception') or ''),
    ' '.join(thesis.get('key_assumptions') or []),
  ]).lower()

  matches: List[str] = []
  for factor, rules in FACTOR_TAXONOMY.items():
    matched_by_ticker = ticker in rules.get('tickers', set())
    matched_by_kw = False
    for kw in rules.get('keywords', []):
      if kw.lower() in text_blob:
        matched_by_kw = True
        break
    if matched_by_ticker or matched_by_kw:
      matches.append(factor)
  return matches


def analyze_exposures(theses: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Aggregate exposures across a portfolio of theses.

  Each thesis is classified into 0+ factors. We then aggregate:
    - factor -> list of (ticker, confidence, recommendation)
    - top concentrated factors
    - unclassified theses (analyst should review)
  """
  if not theses:
    return {
      'theses_analyzed': 0, 'factors': {}, 'top_concentrations': [],
      'unclassified': [],
    }

  by_factor: Dict[str, List[Dict[str, Any]]] = {}
  unclassified: List[Dict[str, Any]] = []

  for th in theses:
    factors = _classify_thesis(th)
    th_entry = {
      'ticker':         th.get('ticker'),
      'thesis_id':      th.get('thesis_id'),
      'confidence':     th.get('confidence'),
      'recommendation': th.get('recommendation'),
      'signal':         th.get('signal'),
    }
    if not factors:
      unclassified.append(th_entry)
      continue
    for f in factors:
      by_factor.setdefault(f, []).append(th_entry)

  # Concentration ranking: number of theses + sum of confidence in that factor
  concentration_ranked = []
  for factor, entries in by_factor.items():
    total_conf = sum(float(e.get('confidence') or 0) for e in entries)
    concentration_ranked.append({
      'factor':        factor,
      'description':   FACTOR_TAXONOMY[factor]['description'],
      'thesis_count':  len(entries),
      'total_confidence_weighted': round(total_conf, 2),
      'tickers':       [e['ticker'] for e in entries],
      'theses':        entries,
    })
  concentration_ranked.sort(
    key=lambda r: (r['thesis_count'], r['total_confidence_weighted']),
    reverse=True,
  )

  # Hidden-concentration warnings
  warnings = []
  for entry in concentration_ranked:
    if entry['thesis_count'] >= 3:
      warnings.append(
        f"{entry['factor']}: {entry['thesis_count']} theses "
        f"({', '.join(entry['tickers'])}) — diversification illusion risk"
      )
    if entry['total_confidence_weighted'] >= 2.0 and entry['thesis_count'] >= 2:
      warnings.append(
        f"{entry['factor']}: conviction-weighted exposure {entry['total_confidence_weighted']:.2f} "
        f"across {entry['thesis_count']} positions — single-factor regret risk"
      )

  return {
    'theses_analyzed':     len(theses),
    'classified_theses':   len(theses) - len(unclassified),
    'factor_count':        len(by_factor),
    'factors':             by_factor,
    'top_concentrations':  concentration_ranked,
    'unclassified':        unclassified,
    'warnings':            warnings,
  }
