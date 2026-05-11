"""Diff Risk Factor disclosures between consecutive 10-K filings.

When management quietly *adds* a risk paragraph (e.g., "supply chain
disruption" or "key customer concentration") to its 10-K, that's a leading
indicator the market may not have priced in. This module produces a structured
diff.

Strategy: pull current and prior year's 10-K Risk Factors section via
edgartools, split into paragraphs, fuzzy-match across years (Jaccard on
3-grams), and return new/removed/modified paragraphs.
"""
import sys
from typing import List, Dict, Any, Optional, Tuple


def _paragraph_split(text: str, min_chars: int = 80) -> List[str]:
  """Split a Risk Factors section into individual risk paragraphs."""
  if not text:
    return []
  # Split on double newlines; clean and filter short fragments
  paras = [p.strip() for p in text.replace('\r', '\n').split('\n\n')]
  return [p for p in paras if len(p) >= min_chars]


def _ngrams(text: str, n: int = 3) -> set:
  """Generate normalized n-gram set for fuzzy matching."""
  words = text.lower().split()
  if len(words) < n:
    return set(words)
  return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
  if not a or not b:
    return 0.0
  return len(a & b) / len(a | b)


def _matches_any(paragraph: str, others: List[str], threshold: float = 0.5) -> Optional[Tuple[int, float]]:
  """Find the closest matching paragraph in `others`. Returns (index, score)
  if any match exceeds threshold, else None."""
  p_ng = _ngrams(paragraph)
  best_idx = -1
  best_score = 0.0
  for i, o in enumerate(others):
    score = _jaccard(p_ng, _ngrams(o))
    if score > best_score:
      best_score = score
      best_idx = i
  if best_score >= threshold:
    return best_idx, best_score
  return None


def diff_risk_factors(current_text: str, prior_text: str,
                      similarity_threshold: float = 0.5) -> Dict[str, Any]:
  """Compare two 10-K Risk Factors sections.

  Returns:
    new_risks: paragraphs in current with no close match in prior
    removed_risks: paragraphs in prior with no close match in current
    modified_risks: paragraphs that matched but had material rewording
    paragraph_count_delta: current_count - prior_count
  """
  current_paras = _paragraph_split(current_text)
  prior_paras = _paragraph_split(prior_text)

  if not current_paras or not prior_paras:
    return {'error': 'insufficient_text',
            'current_paragraphs': len(current_paras),
            'prior_paragraphs': len(prior_paras)}

  new_risks: List[str] = []
  modified_risks: List[Dict[str, Any]] = []
  matched_prior_indices = set()

  for p in current_paras:
    match = _matches_any(p, prior_paras, similarity_threshold)
    if match is None:
      new_risks.append(p)
    else:
      idx, score = match
      matched_prior_indices.add(idx)
      # If similarity is in the "modified" band (0.5-0.85), flag as modified
      if score < 0.85:
        modified_risks.append({
          'current': p[:600], 'prior_index': idx,
          'similarity': round(score, 3),
        })

  removed_risks = [
    p for i, p in enumerate(prior_paras) if i not in matched_prior_indices
  ]

  return {
    'new_risks_count': len(new_risks),
    'removed_risks_count': len(removed_risks),
    'modified_risks_count': len(modified_risks),
    'paragraph_count_delta': len(current_paras) - len(prior_paras),
    # Cap each section to top 10 to keep prompts manageable
    'new_risks': [p[:800] for p in new_risks[:10]],
    'removed_risks': [p[:800] for p in removed_risks[:10]],
    'modified_risks': modified_risks[:10],
    'method': 'jaccard_3gram_paragraph_matching',
    'similarity_threshold': similarity_threshold,
  }


def fetch_and_diff_10k_risks(ticker: str) -> Dict[str, Any]:
  """Fetch the two most recent 10-K Risk Factors via edgartools and diff them.

  Edgartools is rate-limited (SEC etiquette). Result is cached at callsite by
  the Session_Cache pattern when invoked through the MCP server.
  """
  try:
    from edgar import Company, set_identity
    set_identity("nemo-ib agent ops@example.com")  # SEC requires UA header
  except Exception as e:
    return {'error': f'edgartools unavailable: {e}'}

  try:
    company = Company(ticker.upper())
    filings = company.get_filings(form="10-K").latest(2)
    if len(filings) < 2:
      return {'error': 'need_two_10K_filings', 'found': len(filings)}
    current = filings[0]
    prior = filings[1]

    def _risk_text(filing):
      try:
        # edgartools exposes Item 1A as a section
        tenk = filing.obj()
        return tenk.risk_factors if hasattr(tenk, 'risk_factors') else ""
      except Exception as e:
        print(f"[risk_diff] failed to parse {filing.form} for {ticker}: {e}",
              file=sys.stderr, flush=True)
        return ""

    current_text = _risk_text(current)
    prior_text = _risk_text(prior)

    diff = diff_risk_factors(current_text, prior_text)
    diff['ticker'] = ticker.upper()
    diff['current_filing_date'] = str(current.filing_date)
    diff['prior_filing_date'] = str(prior.filing_date)
    return diff
  except Exception as e:
    return {'error': f'fetch_failed: {type(e).__name__}: {e}'}
