"""The analyst surprise, standardised across names instead of across time.

`sue.sue_af` divides a surprise by that company's own dispersion over eight
quarters, and eight quarters of recorded consensus is nine months away at best:
seeding buys four and the recorder supplies the rest one day at a time.

Cross-sectional standardisation needs none of it. It scales the surprise by
price -- which is what makes a beat on a $600 stock comparable with one on a $6
stock -- and ranks it against the other companies that had already reported in
the same window. One quarter of data, not eight, and that quarter exists today.

Both forms are used in the drift literature and this is not a substitute for
the other. `sue_af` asks whether a company beat by more than it usually does;
this asks whether it beat by more than its peers did this season. A company
that beats by a penny every quarter is unremarkable to both, for opposite
reasons, and a company whose first-ever beat is average for the season is
remarkable to one and not the other.

What it inherits from `sue_af` is the part that matters: both legs come out of
`consensus_snapshot`, recorded side by side on the vendor's own basis, so the
subtraction is a surprise rather than the gap between a non-GAAP estimate and a
GAAP filing -- which for MSFT is a dollar, larger than any surprise in it.

Three failure modes of its own, all refused rather than approximated:

  A cohort of three is not a distribution. A percentile against it carries no
  information and would rank a name first or last on nothing.

  The cohort is names that had already reported on the date being asked about.
  Ranking against companies that report next week is lookahead wearing a peer
  group's clothes.

  Scaling by the estimate divides by something that approaches zero, and the
  names where it does are exactly the ones whose surprise would then look
  infinite. Price is the denominator.

Two z-scores come back, for a reason the first live run made obvious. Among 26
recorded names, QFIN missed by 2,964 basis points -- estimate 5.998 against an
actual of 3.23 -- and scored -4.82. That single number roughly doubles the
cohort's standard deviation, so the largest beat in the set came out at +0.62
when it plainly deserves to read as an outlier. Earnings surprises are
fat-tailed by nature, so this is the ordinary case rather than one strange
quarter. `robust_z` is taken from the median and the median absolute deviation
and is not moved by it; `z` is kept beside it because the difference between
them is itself worth seeing.

That difference turned out to be the finding. On the same 26 names the plain z
put the largest beat at +0.62 and the robust one put it at +21.78, with QFIN at
-260. Both are arithmetic and neither is a standard deviation in any useful
sense -- the cross-section of earnings surprises has tails heavy enough that no
scale estimated from it means what a sigma means. So `rank_on` says
`percentile`, and it is not a formality: a reader handed a z would take it for
a sigma and size a position on it.
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from research import pit_store

# Below this the cohort is not a distribution and a percentile against it says
# nothing. Deliberately not "more than one".
MIN_COHORT = 8

# How far back a print still counts as this season's. The same window the
# scanner treats a signal as fresh over.
COHORT_WINDOW_DAYS = 45


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _shell(ticker: str, as_of: str) -> Dict[str, Any]:
    return {"ticker": ticker.upper(), "as_of": as_of, "success": False,
            "error": None, "fiscal_period": None, "estimate": None,
            "actual": None, "surprise": None, "scaled_surprise": None,
            # When the print landed, which is what places this in time. The
            # scanner rejects a signal it cannot date before it looks at
            # anything else.
            "known_at": None,
            "z": None, "robust_z": None, "percentile": None,
            # Which of the three to order on. Neither z is a sigma here; see
            # the module docstring for the numbers that settled it.
            "rank_on": "percentile", "cohort_size": 0,
            "cohort_tickers": []}


def _price_at(ticker: str, as_of: str) -> Optional[float]:
    bars = pit_store.bars_as_of(ticker, as_of)
    if not bars:
        return None
    close = bars[-1].get("close")
    return float(close) if close else None


def _scaled(ticker: str, fiscal: str, as_of: str,
            known_at: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """One name's price-scaled surprise, or None if either leg is missing."""
    actual = pit_store.actual_as_of(ticker, fiscal, as_of)
    if actual is None:
        return None
    snapshot = pit_store.consensus_as_of(ticker, fiscal, as_of)
    estimate = (snapshot or {}).get("eps_estimate")
    if estimate is None:
        return None
    price = _price_at(ticker, as_of)
    if not price or price <= 0:
        return None
    surprise = float(actual) - float(estimate)
    return {"ticker": ticker, "fiscal_period": fiscal, "known_at": known_at,
            "estimate": float(estimate), "actual": float(actual),
            "surprise": surprise, "scaled_surprise": surprise / price}


def cohort(as_of: Optional[str] = None,
           window_days: int = COHORT_WINDOW_DAYS) -> List[Dict[str, Any]]:
    """Every name whose print was on the record by `as_of`, scaled.

    `reporters_since` already filters on `recorded_at`, so a company that
    reports next week cannot appear here however the window is drawn.
    """
    as_of = as_of or _today()
    since = (date.fromisoformat(as_of)
             - timedelta(days=window_days)).isoformat()
    out = []
    for row in pit_store.reporters_since(since, as_of):
        scaled = _scaled(row["ticker"], row["fiscal_period"], as_of,
                         known_at=row.get("as_of_date"))
        if scaled is not None:
            out.append(scaled)
    return out


def surprise_rank(ticker: str, as_of: Optional[str] = None,
                  window_days: int = COHORT_WINDOW_DAYS,
                  peers: Optional[List[Dict[str, Any]]] = None
                  ) -> Dict[str, Any]:
    """Where this name's surprise sits among the ones already reported.

    `peers` takes a cohort the caller already built. It is the same list for
    every name on a given date, by construction, and rebuilding it per name is
    quadratic: on the real store a cohort of 305 takes 386ms to assemble, so a
    date with 305 reporters spent two minutes rebuilding one list, and a replay
    over 652 dates would have run for about twenty-one hours.
    """
    as_of = as_of or _today()
    ticker = ticker.upper()
    result = _shell(ticker, as_of)

    if peers is None:
        peers = cohort(as_of, window_days=window_days)
    result["cohort_size"] = len(peers)
    result["cohort_tickers"] = [p["ticker"] for p in peers]

    mine = next((p for p in peers if p["ticker"] == ticker), None)
    if mine is None:
        result["error"] = (
            f"{ticker} has no print on the record within {window_days} days of "
            f"{as_of}, or no price to scale it by")
        return result

    result.update({k: mine[k] for k in
                   ("fiscal_period", "known_at", "estimate", "actual",
                    "surprise", "scaled_surprise")})

    if len(peers) < MIN_COHORT:
        result["error"] = (
            f"{len(peers)} names reported within {window_days} days of "
            f"{as_of} and {MIN_COHORT} are required; a percentile against a "
            f"cohort this small carries no information about where this "
            f"surprise sits")
        return result

    values = [p["scaled_surprise"] for p in peers]
    spread = statistics.pstdev(values)
    if not spread:
        result["error"] = (
            f"every one of the {len(peers)} names in the cohort has the same "
            f"scaled surprise, so there is nothing to rank against")
        return result

    below = sum(1 for v in values if v < mine["scaled_surprise"])
    result["z"] = (mine["scaled_surprise"] - statistics.fmean(values)) / spread
    result["percentile"] = below / (len(values) - 1)

    # 1.4826 makes the median absolute deviation an unbiased estimate of the
    # standard deviation for a normal sample, so the two scales are comparable
    # when there is no tail to distort the first one.
    centre = statistics.median(values)
    mad = statistics.median([abs(v - centre) for v in values])
    if mad:
        result["robust_z"] = (mine["scaled_surprise"] - centre) / (1.4826 * mad)
    else:
        # Over half the cohort sits on one value, so a deviation from the
        # median is not a scale. The plain z still stands.
        result["robust_z"] = None
    result["success"] = True
    return result
