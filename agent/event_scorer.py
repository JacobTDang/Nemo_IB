"""Heuristic event scorer for the Sentry triage layer.

Reads `events` table rows that have already been classified upstream by
edgar_firehose / news_watcher (materiality, category, directional_signal,
urgency, primary_ticker, affected_tickers). Outputs a 5-component signal
score that the triage daemon uses to decide whether to queue the event
for Claude review.

The scorer is intentionally heuristic — Phase 5 of the Sentry plan
tunes the weights based on observed Slack noise vs hit rate.

Component definitions:
  novelty           — how unusual is this event type for this ticker?
                      Cluster of similar events recently → low novelty.
  magnitude         — material 8-K > earnings beat > generic news.
                      Derived from event.materiality + category.
  reliability       — SEC > Finnhub vendor > scraped news.
                      Derived from event.source.
  market_awareness  — INVERSE — low value means market hasn't priced this yet.
                      Heuristic via event.urgency + age + reliability.
  thesis_relevance  — does this affect an active thesis in `theses` table?
                      Active thesis on this ticker → high relevance.

Overall:
  overall = 0.30 * magnitude
          + 0.20 * novelty
          + 0.15 * reliability
          + 0.20 * (1 - market_awareness)   # low awareness → higher score
          + 0.15 * thesis_relevance

Thresholds the triage daemon uses:
  overall >= 0.70  →  high signal — queue regardless of budget
  0.50-0.70        →  queue if budget allows
  < 0.50           →  log only, don't queue
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from state.schema import get_connection


# -- materiality → magnitude weight ------------------------------------------
# Aligns with how edgar_firehose / news_watcher tag events.
_MATERIALITY_WEIGHT = {
  'critical': 1.00,
  'high':     0.85,
  'medium':   0.60,
  'low':      0.30,
  None:       0.40,   # untagged — default to moderate
}

# -- category → magnitude multiplier -----------------------------------------
# Some categories are intrinsically higher signal than others.
_CATEGORY_BONUS = {
  '8-K':                      0.15,   # 8-K is required disclosure of material events
  'earnings':                 0.10,
  'guidance':                 0.10,
  'guidance_cut':             0.20,   # rare and high signal
  'guidance_raise':           0.15,
  'management_change':        0.15,
  'activist_position':        0.15,
  'insider_buying':           0.10,
  'insider_selling':          0.05,
  'merger':                   0.20,
  'acquisition':              0.20,
  'regulatory_approval':      0.15,
  'regulatory_risk':          0.15,
  'litigation_update':        0.05,
  'macro_shock':              0.10,
  'press_release':            0.00,
  'general_news':             0.00,
}

# -- source → reliability ----------------------------------------------------
# SEC filings are the highest-trust source; scraped news the lowest.
_SOURCE_RELIABILITY = {
  'sec_edgar':       1.00,
  'edgar_firehose':  1.00,
  'finnhub':         0.80,
  'fred':            0.95,
  'press_release':   0.85,
  'gdelt':           0.65,
  'rss':             0.70,
  'news_watcher':    0.70,
  'scraped':         0.50,
  'unknown':         0.50,
  None:              0.50,
}

# -- urgency → market_awareness ----------------------------------------------
# High urgency events typically already moved the price → market is aware.
# Lower urgency events may be slow-bleeders the market hasn't priced.
_URGENCY_AWARENESS = {
  'breaking':   0.85,   # market already moved; awareness high
  'high':       0.70,
  'medium':     0.50,
  'low':        0.30,
  None:         0.45,
}


def score_event(event: Dict[str, Any]) -> Dict[str, float]:
  """Score one event row from the `events` table.

  Args:
    event: dict-like row with keys including materiality, category, source,
           urgency, primary_ticker, affected_tickers, ingested_at, published_at.

  Returns dict with novelty, magnitude, reliability, market_awareness,
  thesis_relevance, overall_score. All values in [0.0, 1.0].
  """
  magnitude = _compute_magnitude(event)
  reliability = _compute_reliability(event)
  market_awareness = _compute_market_awareness(event, reliability)
  novelty = _compute_novelty(event)
  thesis_relevance = _compute_thesis_relevance(event)

  # Weighted sum, then gated by magnitude^0.3 — a "is this even important?"
  # multiplier that gently suppresses low-magnitude events even when they
  # score well on novelty / reliability / market_awareness. Without this,
  # generic Finnhub news on an under-covered ticker would clear the 0.50
  # threshold purely on novelty + reliability, which is noise.
  weighted = (
    0.30 * magnitude
    + 0.20 * novelty
    + 0.15 * reliability
    + 0.20 * (1.0 - market_awareness)
    + 0.15 * thesis_relevance
  )
  magnitude_gate = max(0.10, magnitude) ** 0.3  # floor at 0.10 so a small bug doesn't zero everything
  overall = weighted * magnitude_gate

  return {
    'novelty':          round(novelty, 3),
    'magnitude':        round(magnitude, 3),
    'reliability':      round(reliability, 3),
    'market_awareness': round(market_awareness, 3),
    'thesis_relevance': round(thesis_relevance, 3),
    'overall_score':    round(min(1.0, max(0.0, overall)), 3),
  }


def _compute_magnitude(event: Dict[str, Any]) -> float:
  """Combine materiality + category bonus, clipped to [0, 1]."""
  base = _MATERIALITY_WEIGHT.get(event.get('materiality'), 0.40)
  cat = event.get('category')
  bonus = _CATEGORY_BONUS.get(cat, 0.0) if cat else 0.0
  return min(1.0, base + bonus)


def _compute_reliability(event: Dict[str, Any]) -> float:
  """Map source to reliability score."""
  src = event.get('source')
  return _SOURCE_RELIABILITY.get(src, 0.50)


def _compute_market_awareness(event: Dict[str, Any], reliability: float) -> float:
  """Heuristic — how much has the market already priced in this event?

  High awareness when:
    - urgency is breaking/high (market reacted on the day)
    - event is from a vendor source the market also consumes (Finnhub)
    - event is old (already had time to be priced)

  Low awareness when:
    - urgency is low (slow-bleeder)
    - source is SEC raw filings (most market participants don't read raw)
    - event is fresh (within last few hours)
  """
  urgency_awareness = _URGENCY_AWARENESS.get(event.get('urgency'), 0.45)

  # Vendor sources like Finnhub are widely consumed → higher awareness
  # SEC raw filings → lower awareness (most don't parse them)
  src = event.get('source')
  if src in ('finnhub', 'gdelt'):
    source_bump = 0.10
  elif src in ('sec_edgar', 'edgar_firehose'):
    source_bump = -0.15   # raw filings are less-priced in real-time
  else:
    source_bump = 0.0

  # Age modifier — older events have had time to be priced
  age_bump = _age_awareness_bump(event)

  awareness = urgency_awareness + source_bump + age_bump
  return min(1.0, max(0.0, awareness))


def _age_awareness_bump(event: Dict[str, Any]) -> float:
  """Older events → higher awareness (already had time to circulate).

  Returns a bump in [-0.10, +0.15] based on hours since published_at.
  """
  pub = event.get('published_at')
  if not pub:
    return 0.0
  try:
    # Handle both ISO strings and datetime objects
    if isinstance(pub, str):
      pub_dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
    else:
      pub_dt = pub
    if pub_dt.tzinfo is None:
      pub_dt = pub_dt.replace(tzinfo=timezone.utc)
    hours = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600.0
  except (ValueError, TypeError):
    return 0.0

  if hours < 1.0:
    return -0.10    # very fresh — market hasn't fully processed yet
  if hours < 6.0:
    return -0.05
  if hours < 24.0:
    return 0.0
  if hours < 72.0:
    return 0.05
  return 0.15       # older than 3 days — market has fully chewed on it


def _compute_novelty(event: Dict[str, Any]) -> float:
  """How unusual is this event type for this ticker recently?

  Lookup: count of events with same (ticker, category) in the last 7 days.
  Many similar events → low novelty. First in 7+ days → high novelty.
  """
  ticker = event.get('primary_ticker') or event.get('ticker')
  category = event.get('category')
  if not ticker or not category:
    return 0.5    # missing data — neutral novelty

  cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
  current_event_id = event.get('event_id')

  conn = get_connection()
  try:
    if current_event_id:
      cur = conn.execute(
        """SELECT COUNT(*) AS n FROM events
           WHERE primary_ticker = ? AND category = ?
             AND ingested_at >= ? AND event_id != ?""",
        (ticker, category, cutoff, current_event_id),
      )
    else:
      cur = conn.execute(
        """SELECT COUNT(*) AS n FROM events
           WHERE primary_ticker = ? AND category = ? AND ingested_at >= ?""",
        (ticker, category, cutoff),
      )
    similar_count = int(cur.fetchone()['n'])
  except Exception:
    return 0.5
  finally:
    conn.close()

  if similar_count == 0:
    return 1.0     # first in 7+ days
  if similar_count <= 2:
    return 0.75
  if similar_count <= 5:
    return 0.50
  if similar_count <= 10:
    return 0.30
  return 0.15      # 10+ similar events in past week → noise


def _compute_thesis_relevance(event: Dict[str, Any]) -> float:
  """Does this event affect an active thesis?

  Checks `theses` table for any active (non-superseded) thesis on the
  primary ticker. High relevance → high score: events affecting our
  open positions deserve priority over random discovery.
  """
  ticker = event.get('primary_ticker') or event.get('ticker')
  if not ticker:
    return 0.0

  conn = get_connection()
  try:
    cur = conn.execute(
      """SELECT thesis_id, confidence FROM theses
         WHERE ticker = ? AND superseded_by IS NULL
         ORDER BY thesis_date DESC LIMIT 1""",
      (ticker,),
    )
    row = cur.fetchone()
  except Exception:
    return 0.0
  finally:
    conn.close()

  if row is None:
    return 0.0   # no active thesis on this ticker

  # Scale relevance by thesis confidence — high-conviction thesis → higher relevance
  conf = row['confidence'] or 0.5
  return min(1.0, 0.7 + 0.3 * conf)
