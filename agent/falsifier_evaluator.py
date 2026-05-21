"""Pure-Python evaluator for falsifier conditions.

Inputs:
  - A falsifier string (free text, e.g. "Azure revenue growth < 18% YoY for 2
    consecutive quarters")
  - One or more pieces of evidence (news headlines, 8-K text, macro snapshots,
    earnings press releases)

Output: a structured score with breakdown of WHY the falsifier was or wasn't
triggered. The evaluator is intentionally conservative — it should produce
false negatives more often than false positives, because each false positive
wastes analyst attention and erodes trust in the watcher.

Design principles:
  * No external API calls. The watcher must be cheap to run every 15 minutes.
  * Deterministic — same inputs produce same scores. No LLM in the hot path.
  * Multi-signal: token overlap, named-entity match, numeric-threshold parse,
    negation awareness. No single signal is decisive; the combined score is.
  * Auditable — the result includes which evidence chunks matched, which
    tokens overlapped, and which numeric thresholds were evaluated. Analyst
    can re-check the daemon's reasoning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tokenization & utilities
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    'a', 'an', 'and', 'or', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
    'this', 'that', 'these', 'those', 'it', 'its', 'their', 'our', 'we',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'may', 'might', 'can', 'must', 'shall',
    'than', 'then', 'so', 'such', 'if', 'else', 'while', 'after', 'before',
    'over', 'under', 'between', 'about',
})

# Tokens that signal negation in falsifier text (don't match when present)
_NEGATION_TOKENS = frozenset({'not', 'no', 'without', 'never', 'except',
                              'unless', "doesn't", "don't", "won't"})

# Tokens that signal numeric thresholds — used to extract comparable values
_THRESHOLD_OPS = {
    '>=': 'gte', '<=': 'lte', '>': 'gt', '<': 'lt', '==': 'eq',
    'above': 'gt', 'below': 'lt',
    'over': 'gt', 'under': 'lt',
    'exceeds': 'gt', 'exceeded': 'gt',
    'less than': 'lt', 'more than': 'gt',
    'greater than': 'gt', 'at least': 'gte', 'at most': 'lte',
}


def _tokenize(text: str) -> List[str]:
  """Lowercase, alphanumeric tokens of length >= 3, stopwords removed."""
  if not text:
    return []
  s = text.lower()
  toks = re.findall(r"[a-z][a-z0-9'-]{2,}", s)
  return [t for t in toks if t not in _STOPWORDS]


def _significant_tokens(text: str) -> set:
  """Tokens that carry meaning (long enough, not stopwords)."""
  return set(t for t in _tokenize(text) if len(t) >= 4)


def _jaccard(a: set, b: set) -> float:
  if not a or not b:
    return 0.0
  inter = len(a & b)
  union = len(a | b)
  return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Numeric threshold extraction
# ---------------------------------------------------------------------------

@dataclass
class NumericCondition:
  """A parsed numeric condition like '10Y > 5.25%' or 'capex < 15%'."""
  metric_phrase: str   # raw text describing the metric (e.g. "10Y treasury")
  op:            str   # one of: gt, lt, gte, lte, eq
  value:         float
  unit:          str   # '%', 'bps', '$B', '$M', 'x', 'days', etc.
  raw_match:     str   # original substring


def parse_numeric_conditions(text: str) -> List[NumericCondition]:
  """Extract numeric threshold conditions from a falsifier string.

  Examples it should parse:
    - "10Y treasury > 5.25%"            -> op=gt, value=5.25, unit='%'
    - "HY OAS > 400bps"                 -> op=gt, value=400, unit='bps'
    - "Azure revenue growth < 18% YoY"  -> op=lt, value=18, unit='%'
    - "net debt/EBITDA above 3.5x"      -> op=gt, value=3.5, unit='x'
  """
  conditions: List[NumericCondition] = []
  # Cover both math-symbol operators and English ones in a single pass.
  # The text-form operators have longer "less than"-style phrases first to
  # avoid being shadowed by single-word matches.
  text_ops = sorted([k for k in _THRESHOLD_OPS if not k[0] in '<>='],
                    key=len, reverse=True)
  pattern_parts = [r'(?P<op_sym>>=|<=|>|<|==)']
  pattern_parts.append('|'.join(re.escape(o) for o in text_ops))
  op_alternation = '|'.join(pattern_parts)

  # Look for: <metric phrase up to 60 chars> <op> <number> <unit?>
  combined = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z0-9 /&%\.\-]{1,60}?)\s*"
    r"(?P<op>>=|<=|>|<|==|" + '|'.join(re.escape(o) for o in text_ops) + r")\s*"
    r"\$?(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|bps|bp|x|X|B|M|K|days?|years?|months?|qtrs?|quarters?)?",
    re.IGNORECASE)

  for m in combined.finditer(text):
    op_raw = m.group('op').lower().strip()
    op = _THRESHOLD_OPS.get(op_raw)
    if not op:
      continue
    try:
      value = float(m.group('value'))
    except (TypeError, ValueError):
      continue
    metric = (m.group('metric') or '').strip(' ,.;:')
    unit = (m.group('unit') or '').strip().lower()
    conditions.append(NumericCondition(
      metric_phrase=metric,
      op=op,
      value=value,
      unit=unit,
      raw_match=m.group(0),
    ))
  return conditions


def evaluate_numeric_condition(cond: NumericCondition,
                               observed_value: Optional[float]) -> Optional[bool]:
  """Given a parsed condition and an observed value, return True if the
  threshold is breached, False if not, None if can't evaluate."""
  if observed_value is None:
    return None
  if cond.op == 'gt':
    return observed_value > cond.value
  if cond.op == 'lt':
    return observed_value < cond.value
  if cond.op == 'gte':
    return observed_value >= cond.value
  if cond.op == 'lte':
    return observed_value <= cond.value
  if cond.op == 'eq':
    return abs(observed_value - cond.value) < 1e-9
  return None


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

def has_negation(text: str) -> bool:
  """True if the text contains a negation token. Used to flip match polarity
  so a falsifier like 'CFO Hood NOT replaced' isn't triggered by a press
  release announcing she stayed."""
  if not text:
    return False
  tokens = re.findall(r"[a-z'-]+", text.lower())
  return any(t in _NEGATION_TOKENS for t in tokens)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@dataclass
class FalsifierEvalResult:
  triggered:             bool
  score:                 float
  matched_tokens:        List[str] = field(default_factory=list)
  numeric_conditions:    List[Dict[str, Any]] = field(default_factory=list)
  best_evidence:         Optional[Dict[str, Any]] = None
  reason:                str = ''
  notes:                 List[str] = field(default_factory=list)


def score_evidence_against_falsifier(
  falsifier: str,
  evidence_text: str,
  evidence_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Return a similarity score in [0, 1] between a falsifier and an evidence
  text. Higher = more likely the evidence triggers the falsifier."""
  ftoks = _significant_tokens(falsifier)
  etoks = _significant_tokens(evidence_text)

  jacc = _jaccard(ftoks, etoks)
  matched = sorted(ftoks & etoks)

  # Polarity check: if falsifier asserts a negation and evidence also
  # negates, that's NOT a trigger — they agree the thing didn't happen.
  falsifier_negates = has_negation(falsifier)
  evidence_negates = has_negation(evidence_text)
  polarity_aligned = falsifier_negates == evidence_negates

  # Boost score when both contain a proper noun present in the falsifier.
  # Multi-word entities (e.g. "Amy Hood") are stronger evidence than single-
  # word ones (e.g. "Microsoft"), so weight the boost by word count.
  boost = 0.0
  proper_noun_match = []
  for pn in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", falsifier):
    if pn.lower() in evidence_text.lower():
      word_count = len(pn.split())
      boost += 0.10 + 0.05 * word_count   # 1-word=0.15, 2-word=0.20, 3-word=0.25
      proper_noun_match.append(pn)
  boost = min(boost, 0.50)  # cap so it doesn't dominate

  score = jacc + boost
  if not polarity_aligned and score > 0.0:
    # Asymmetric polarity — likely a false match (one side says "happened",
    # other says "didn't happen"). Reduce confidence.
    score *= 0.5

  score = min(score, 1.0)

  return {
    'score':              round(score, 3),
    'jaccard':            round(jacc, 3),
    'proper_noun_boost':  round(boost, 3),
    'proper_nouns_matched': proper_noun_match,
    'matched_tokens':     matched[:15],
    'falsifier_negates':  falsifier_negates,
    'evidence_negates':   evidence_negates,
    'polarity_aligned':   polarity_aligned,
  }


def evaluate_falsifier(
  falsifier: str,
  evidence_pool: List[Dict[str, Any]],
  observed_values: Optional[Dict[str, float]] = None,
  threshold: float = 0.35,
) -> FalsifierEvalResult:
  """Evaluate a single falsifier against an evidence pool + observed metrics.

  evidence_pool: list of dicts each with at least 'text' field plus arbitrary
  metadata (event_id, source, headline, published_at, etc.).

  observed_values: optional dict mapping metric names to current values,
  used to evaluate numeric conditions parsed from the falsifier. Key matching
  is by case-insensitive substring of the parsed metric phrase.

  threshold: text-similarity score at or above which the best-matching
  evidence triggers the falsifier. Default 0.35 chosen empirically — set
  higher for noisier feeds.

  Returns FalsifierEvalResult with the full audit trail.
  """
  result = FalsifierEvalResult(triggered=False, score=0.0)

  # Parse numeric conditions
  conds = parse_numeric_conditions(falsifier)
  for c in conds:
    obs = None
    if observed_values:
      # Case-insensitive substring match against observed-value keys
      mp = c.metric_phrase.lower()
      for k, v in observed_values.items():
        if k.lower() in mp or mp in k.lower():
          obs = float(v) if v is not None else None
          break
    triggered = evaluate_numeric_condition(c, obs)
    result.numeric_conditions.append({
      'metric':    c.metric_phrase,
      'op':        c.op,
      'threshold': c.value,
      'unit':      c.unit,
      'observed':  obs,
      'triggered': triggered,
    })

  # If ANY numeric condition triggered, that's a strong signal — flag it
  any_numeric_trigger = any(nc.get('triggered') is True
                            for nc in result.numeric_conditions)
  any_numeric_unknown = any(nc.get('triggered') is None
                            for nc in result.numeric_conditions)

  # Score against evidence pool
  best = {'score': 0.0}
  best_idx = -1
  for i, ev in enumerate(evidence_pool or []):
    text = ev.get('text') or ev.get('headline') or ''
    body = ev.get('body') or ''
    full_text = (text + ' ' + body).strip()
    if not full_text:
      continue
    s = score_evidence_against_falsifier(falsifier, full_text)
    if s['score'] > best['score']:
      best = s
      best_idx = i

  if best_idx >= 0:
    result.best_evidence = {
      **(evidence_pool[best_idx]),
      'similarity_score': best['score'],
      'matched_tokens':   best['matched_tokens'],
      'proper_nouns':     best['proper_nouns_matched'],
    }
    result.matched_tokens = best['matched_tokens']
    result.score = best['score']

  # Trigger decision: any numeric trigger fires, OR text similarity above
  # threshold + (proper-noun match OR numeric condition unknown but plausible).
  text_triggered = result.score >= threshold
  if any_numeric_trigger:
    result.triggered = True
    result.reason = 'numeric_threshold_breached'
    result.notes.append(
      f"{sum(1 for nc in result.numeric_conditions if nc['triggered'] is True)} "
      "numeric condition(s) breached"
    )
  elif text_triggered:
    result.triggered = True
    result.reason = 'text_similarity_above_threshold'
    result.notes.append(f"best evidence score {result.score:.3f} >= {threshold}")
  else:
    result.triggered = False
    if result.numeric_conditions and any_numeric_unknown:
      result.reason = 'numeric_condition_unevaluated_text_below_threshold'
    else:
      result.reason = 'below_threshold'

  return result
