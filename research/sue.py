"""Standardised unexpected earnings -- the signal the rest of this is filtering.

Post-earnings announcement drift is the one anomaly in this strategy that
survived replication. In the post-2005 sample SUE was the only statistically
significant determinant of returns; book-to-market, profitability and price
momentum all shrank by at least 40% and several flipped sign. Everything else
built here decides which names are tradeable. This decides which are
interesting.

    SUE_ts = (EPS_q - EPS_q-4) / sigma(dEPS over the trailing 8 announcements)

Two things about that formula are load-bearing and easy to get wrong.

**The expectation is seasonal.** EPS_q-4 is the same fiscal quarter a year
earlier, not the quarter before. Earnings are seasonal -- a retailer's December
quarter dwarfs its March one -- so differencing consecutive quarters measures
the seasonality and reports it as a surprise.

**The denominator excludes the quarter being measured.** Sigma is estimated
over the eight announcements *preceding* this one. Fold the current surprise
into its own scale and the largest surprises get pulled back toward the middle,
which is the one part of the distribution the strategy is paid for.

Eight trailing announcements, and fewer than six is no signal. A standard
deviation from three observations is an estimate of noise, and dividing by it
prints the biggest SUEs for the companies we know least about. The window is
never quietly widened past eight either: doing so would give the patchiest
histories a denominator drawn from a different regime than everyone else's.

Where the numbers come from
---------------------------
`data.sec.gov/api/xbrl/companyconcept` -- one request returns every fact a
filer ever tagged for one concept, each carrying its own period, the form it
came from, the fiscal year and period the filing declared, and the date it was
filed. That last field is the point-in-time discipline: nothing is visible to a
question asked before the filing that carried it, which is the same rule
`pit_store.bars_as_of` enforces with `recorded_at`.

Traps this cost real time to find, all verified live against EDGAR:

**SEC's two endpoints disagree.** `companyconcept` for Coca-Cola's
`EarningsPerShareDiluted` returns the concept header with both unit arrays
empty; `companyfacts` for the same CIK carries 229 facts of it, the most recent
filed three months earlier. Ford is the same. So the cheap endpoint is the
default and the five-megabyte one is the fallback, taken whenever the chain
comes up empty or ends in a quarter too old to be this filer's latest.

**`fy` and `fp` belong to the filing, not to the fact.** MSFT's January 2026
10-Q reports its own December quarter and, tagged with the same fy=2026 fp=Q2,
the December 2024 comparative. Reading fiscal identity straight off the fact
puts last year's number in this year's slot. Identity is therefore taken from
the filing and assigned only to the fact whose period ends when the filing's
does -- which also means a quarter is whatever length the filer says it is.
Costco's are 12 weeks and its fourth is 16; Apple's are 13 and occasionally 14.
A duration window tight enough to call 91 days "a quarter" throws both away.

**And `fp` is sometimes simply wrong.** Dell's first two FY2024 10-Qs declare a
fiscal period focus of `FY`. Dropping them costs two quarters and, through the
year-ago term, two more. The periods the filing carries settle it instead: the
year to date runs one, two or three quarters long.

**No 10-Q covers the fourth quarter.** It is the annual figure less the nine
months to date, both the filer's own reported EPS. That subtraction of two
figures each rounded to a cent can land a cent from the number the company
announced -- Apple's FY2025 fourth quarter derives to 1.84 against an announced
1.85 -- so those quarters are labelled `derived` and carry the arithmetic.

**Nothing in XBRL marks a stock split.** It shows up only as one fiscal period
carrying different values in two filings: NVDA's April 2024 quarter is 5.98 as
filed and 0.60 as restated after the 10-for-1. Difference the two originals
across that boundary and the surprise is -5.38 against a true +0.09 -- the
largest miss in the universe, invented by arithmetic. Restated comparatives
arrive four quarters late and on their own bracket the change to a year, which
would take the signal away for the three prints after every split. A filing's
own year to date closes that: NVDA's Q2 FY2025 10-Q gives 1.27 - 0.67 = 0.60
for a quarter it had itself filed as 5.98, which places the split between two
consecutive filings. Facts on the old side are rebased; a change that still
cannot be placed drops those quarters rather than guessing, because guessing is
a tenfold error on whichever ones fall the wrong side.

**A restatement is not a rescaling.** Dell's FY2025 10-K restates three FY2024
quarters by 9%, 5% and 4% -- each period by its own amount, which is what
distinguishes it from a split. Only a rescaling of 1.45x or more, agreed by at
least two periods, counts; and it must clear that bar at the worst of the
rounding, because 0.13 against 0.20 is a ratio of 1.54 from nothing but the
cent Amazon rounded its 2012 quarters to.

**EPS is the filer's own, never net income over shares.** Share counts differ
by class and BRK's two are 1500:1, so a derived per-share figure is wrong by
whichever class the count came from and plausibly so. One concept carries the
whole series, too: basic and diluted differ by the dilution, which drifts, and
splicing them puts a step change where the filer changed its tagging. The chain
is ranked on currency before depth -- Ford's continuing-operations tag has more
quarters than its diluted one and stops in 2011.

**Some filers have no consolidated EPS at all.** Visa and Berkshire tag
earnings per share against each share class, and SEC's company APIs return only
undimensioned facts, so the concept is absent rather than partial. That is a
refusal naming the cause, never a series built from whichever class happened to
be readable.

**A foreign private issuer has no quarterly EPS either.** It files 20-F or 40-F
annually and 6-K for interims, and 6-K exhibits carry no XBRL -- verified for
TSM, ASML and BABA. Every EPS fact TSM tags covers a full calendar year. Same
refusal shape as `foreign_issuer.get_annual_revenue`, never a partial series.

**A ticker can point at the wrong registrant.** SEC's own ticker file maps XOM
to CIK 0002115436, a successor with one 10-Q on EDGAR; the history sits under
the predecessor CIK and no ticker points at it. Nothing here can repair that,
so the refusal reports how many quarters the CIK actually carries, which reads
as the wrong registrant rather than as a gap in the data.

The analyst leg
---------------
The literature (Livnat & Mendenhall; Doyle, Lundholm & Soliman) finds
analyst-based surprises drift further than time-series ones and that a two-way
sort on both is strongest. `sue_af` implements it and today it refuses, for two
reasons, only one of which goes away by waiting.

The consensus history cannot be fetched: Finnhub returns four quarters at
limit=12 and at limit=30 alike -- verified. It accrues one snapshot per name
per day in `pit_store.consensus_snapshot`, and `sue_af` starts answering on its
own once eight quarters of it have gone by. Nothing here fakes it from data
that does not exist.

The second is the one the brief did not anticipate: a street estimate is a
non-GAAP number and XBRL carries only the GAAP one. Verified live on
2026-08-27, Finnhub's actual EPS against the same quarter's diluted EPS in the
filing -- MSFT 2026Q2: 4.14 against 5.16. NVDA 2026Q2: 1.05 against 1.08. AAPL
2026Q3: 1.91 against 2.02. Subtracting the estimate from the wrong actual gives
every surprise a definitional gap that moves quarter to quarter and reaches a
dollar, which is larger than the surprises themselves. So `sue_af` takes the
actuals as an argument and refuses without them, and what the daily job needs
to start recording is the vendor's own `actual_eps` beside its estimate.

Fiscal identity does line up, which was worth checking before relying on it:
Finnhub's `year` and `quarter` are the filer's own -- AAPL's June 2026 quarter
is 2026Q3 there and here -- while its `period` field is a calendar bucket
(2026-06-30 for a quarter that ended 2026-06-27), which is exactly the label
`pit_store.record_announcement` already refuses to key on.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from research import pit_store

# --- the model ------------------------------------------------------------
SIGMA_QUARTERS = 8
MIN_SIGMA_QUARTERS = 6

# Concepts are tried in order and the first that yields a usable series carries
# all of it. Diluted first because it is what a filer leads with and what the
# street quotes; the combined tag is what a single-class filer with no dilutive
# securities uses; the continuing-operations tag is what a filer mid-divestment
# tags instead; basic is the last resort.
US_GAAP_EPS_CONCEPTS = (
    "us-gaap:EarningsPerShareDiluted",
    "us-gaap:EarningsPerShareBasicAndDiluted",
    "us-gaap:IncomeLossFromContinuingOperationsPerDilutedShare",
    "us-gaap:EarningsPerShareBasic",
)
# Tried only to tell a foreign private issuer apart from a filer that tags
# nothing. IFRS filers report EPS annually and semi-annually at best, so this
# chain produces a reason rather than a series.
IFRS_EPS_CONCEPTS = (
    "ifrs-full:DilutedEarningsLossPerShare",
    "ifrs-full:BasicEarningsLossPerShare",
)

# Periodic reports only. An 8-K earnings release occasionally carries tagged
# EPS, but its fiscal-period focus is not reliably the period it reports, and
# the accession-level rule below depends on that focus being right.
_PERIOD_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")

# A foreign private issuer files these instead, and none of them carries a
# tagged quarter: 20-F and 40-F are annual, and 6-K exhibits have no XBRL.
FOREIGN_ANNUAL_FORMS = ("20-F", "40-F")

# A rescaling of less than 1.45x is not a split: the smallest common one is
# 3-for-2, and what lands under that bar is accounting restatement, which moves
# each period by its own amount rather than all of them by one factor -- Dell's
# FY2025 10-K restates three FY2024 quarters by 9%, 5% and 4%, verified live.
# The bar is applied to the *worst* case of the rounding, so that a couple of
# cents of EPS cannot manufacture one; see `_rescale_bounds`.
_MIN_BASIS_RESCALE = 1.45
# Once restatements are out, two brackets 15% apart are still one event: a
# period whose EPS rounds to a couple of cents measures a 4-for-1 as 0.2857
# rather than 0.25, and Salesforce's April 2013 split shows up exactly that
# way. Distinct splits stay apart because their brackets cannot intersect.
_RATIO_MATCH = 0.15
# How many distinct fiscal periods must agree on a ratio before it counts as a
# share basis rather than one restated line.
_BASIS_CORROBORATION = 2
# Filers round EPS to the cent, so a figure carries half a cent of uncertainty
# and a difference of two of them carries a whole one. Trailing changes that
# agree to inside that have no dispersion this data can measure.
_EPS_QUANTUM = 0.005
# A fact ending when the annual period does and running under this share of its
# length is the filer tagging its own fourth quarter, not a second annual.
_Q4_SPAN_SHARE = 0.45

# A filer with no quarterly EPS filed within this many days of the as-of date
# has either stopped reporting or is being read through a concept it abandoned.
# A 10-Q is due within 45 days of a quarter end, so a live filer is never
# quieter than about 135 days.
_STALE_DAYS = 200

_CONCEPT_URL = ("https://data.sec.gov/api/xbrl/companyconcept/"
                "CIK{cik}/{taxonomy}/{tag}.json")
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC fair access allows roughly 10 requests a second. Kept local rather than
# imported from tools.web_search_server.sec_series so that `research` stays
# importable without edgartools, which that module loads at import time.
_MIN_REQUEST_GAP_S = 0.15
_last_request_at = 0.0
_throttle_lock = threading.Lock()

_cik_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()


def _reset_caches() -> None:
    """Drop the ticker-to-CIK map. For tests; nothing else should call it."""
    with _cache_lock:
        _cik_cache.clear()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - elapsed)
        _last_request_at = time.monotonic()


def _user_agent() -> str:
    """SEC fair access wants a real contact address, or we do not ask.

    Defaulting to a placeholder misrepresents the caller, so an unset value
    stops us here rather than at a 403 several thousand requests later.
    """
    email = os.environ.get("SEC_EMAIL", "").strip()
    if not email:
        raise ValueError(
            "SEC_EMAIL is not set. SEC fair access requires a real contact "
            "address in the User-Agent header.")
    return f"Nemo_IB research {email}"


# --------------------------------------------------------------- seams

def _fetch_cik_map() -> Dict[str, str]:
    """Ticker -> zero-padded CIK, from SEC's own file.

    SEC's list rather than a vendor's for the same reason `daily_job` uses it:
    a vendor's universe has already dropped whatever delisted.
    """
    request = urllib.request.Request(
        _TICKERS_URL, headers={"User-Agent": _user_agent()})
    _throttle()
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())

    out: Dict[str, str] = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).strip().upper()
        if ticker:
            out[ticker] = str(row.get("cik_str", "")).zfill(10)
    return out


def _fetch_company_concept(cik: str, taxonomy: str,
                           tag: str) -> Optional[Dict[str, Any]]:
    """Every fact this filer ever tagged for one concept, or None if it never
    tagged it.

    One request for a whole history, against roughly one filing download per
    quarter through the edgartools path. None for a 404 specifically: that is
    "this filer does not use this element", which the concept chain needs to
    tell apart from an outage. Anything else raises.
    """
    url = _CONCEPT_URL.format(cik=cik, taxonomy=taxonomy, tag=tag)
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    _throttle()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _fetch_company_facts(cik: str) -> Optional[Dict[str, Any]]:
    """Every concept this filer tags, in one request.

    The fallback, not the default, because it is five to eight megabytes
    against a hundred kilobytes for one concept. It exists because SEC's two
    endpoints disagree: on 2026-08-27 `companyconcept` for Coca-Cola's
    `EarningsPerShareDiluted` returned the concept header with both unit arrays
    empty, while `companyfacts` for the same CIK carried 229 facts of it, the
    most recent filed three months earlier. Ford is the same. A filer that
    reports perfectly well would otherwise read as one that tags no earnings.
    """
    request = urllib.request.Request(_FACTS_URL.format(cik=cik),
                                     headers={"User-Agent": _user_agent()})
    _throttle()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _cik_for(ticker: str) -> Optional[str]:
    key = ticker.strip().upper()
    with _cache_lock:
        cached = dict(_cik_cache)
    if not cached:
        cached = _fetch_cik_map()
        with _cache_lock:
            _cik_cache.update(cached)
    return cached.get(key) or cached.get(key.replace(".", "-"))


# ------------------------------------------------------------- fiscal keys

def _period_key(fiscal_year: int, fiscal_quarter: int) -> str:
    return f"{fiscal_year}Q{fiscal_quarter}"


def _parse_period(text: str) -> Optional[Tuple[int, int]]:
    try:
        year, quarter = str(text).upper().split("Q")
        return int(year), int(quarter)
    except (ValueError, AttributeError):
        return None


def _shift(fiscal_year: int, fiscal_quarter: int, back: int) -> Tuple[int, int]:
    """`back` fiscal quarters earlier, carrying across the fiscal year."""
    index = fiscal_year * 4 + (fiscal_quarter - 1) - back
    return index // 4, index % 4 + 1


# ------------------------------------------------------------------ facts

@dataclass(frozen=True)
class _Fact:
    start: str
    end: str
    value: float
    accession: str
    form: str
    fiscal_year: int
    fiscal_period: str
    filed: str

    @property
    def days(self) -> int:
        return (date.fromisoformat(self.end) - date.fromisoformat(self.start)).days


def _pick_unit(units: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """The unit the filer actually reports this concept in.

    Costco's 2010 10-K tags eleven EPS facts under `pure` alongside 295 under
    `USD/shares` -- verified live. Mixing units mixes scales, so the series
    takes one: a per-share denomination if there is one, and otherwise
    whichever carries the most facts. On a tie a non-USD unit wins, matching
    `sec_series.reporting_currency`: a filer presenting in dollars has no
    second currency to be confused with, so a USD unit sitting beside another
    is the convenience translation.
    """
    if not units:
        return None
    per_share = {name: rows for name, rows in units.items()
                 if name.lower().endswith("/shares")}
    candidates = per_share or units
    return max(candidates.items(),
               key=lambda kv: (len(kv[1]), kv[0].upper() != "USD/SHARES"))[0]


def _parse_facts(payload: Dict[str, Any], as_of: str
                 ) -> Tuple[List[_Fact], Optional[str], set]:
    """Facts knowable on `as_of`, the unit they are in, and the forms seen.

    `filed <= as_of` is the whole point-in-time discipline of this module. A
    quarter that has ended is not a quarter that has been reported, and a
    series filtered on period end alone hands a simulation the September
    quarter in early October -- weeks before anyone could read it, and days
    before the announcement the strategy trades.
    """
    units = (payload or {}).get("units") or {}
    unit = _pick_unit(units)
    if unit is None:
        return [], None, set()

    forms_seen: set = set()
    facts: List[_Fact] = []
    for row in units[unit]:
        filed = str(row.get("filed") or "")
        if not filed or filed > as_of:
            continue
        form = str(row.get("form") or "")
        forms_seen.add(form)
        start, end = str(row.get("start") or ""), str(row.get("end") or "")
        if form not in _PERIOD_FORMS or not start or not end:
            continue
        try:
            value = float(row["val"])
            fiscal_year = int(row["fy"])
        except (KeyError, TypeError, ValueError):
            continue
        facts.append(_Fact(start=start, end=end, value=value,
                           accession=str(row.get("accn") or ""), form=form,
                           fiscal_year=fiscal_year,
                           fiscal_period=str(row.get("fp") or ""), filed=filed))
    return facts, unit, forms_seen


# ------------------------------------------------------------- share basis

def _rescale_bounds(earlier: float, later: float) -> Tuple[float, float]:
    """How large a rescaling these two figures could be, at worst and at best.

    EPS is rounded to the cent, so each figure carries half a cent either way
    and the ratio between them is a range rather than a number. Against a
    denominator of a few cents that range is wide: Amazon's 2012 quarters gave
    1.54x, which is past the bar a 3-for-2 split clears, from nothing but
    rounding. A rescaling has to clear the bar at the bottom of its range; the
    top of the range decides whether "no change" can be asserted instead.
    """
    low, high = abs(earlier), abs(later)
    floor = max(low - _EPS_QUANTUM, _EPS_QUANTUM)
    ratios = [(high - _EPS_QUANTUM) / (low + _EPS_QUANTUM),
              (high + _EPS_QUANTUM) / floor]
    magnitudes = [max(r, 1.0 / r) for r in ratios if r > 0] or [1.0]
    return min(magnitudes), max(magnitudes)


def _implied_prior_stubs(filings: List[Dict[str, Any]]
                         ) -> List[Tuple[Tuple[str, str], str, float]]:
    """Each filing's own arithmetic, restating the quarters before it.

    A 10-Q carries the quarter and the year to date, both in that filing's
    basis, so their difference is the earlier part of the year as this filing
    would state it -- and it covers exactly the period the previous filing
    reported. NVDA's Q2 FY2025 10-Q gives 1.27 - 0.67 = 0.60 for a first
    quarter it had itself filed as 5.98 three months earlier.

    That matters because the comparative route arrives four quarters late and
    brackets a split to a year, leaving three filings inside it unplaceable --
    which took NVDA's signal away for three quarters after the June 2024
    split, exactly when it was worth having. This narrows the bracket to the
    gap between two consecutive filings.
    """
    stubs: List[Tuple[Tuple[str, str], str, float]] = []
    for filing in filings:
        year, quarter = filing["longest"], filing["shortest"]
        if year is quarter or quarter.start <= year.start:
            continue
        stub_end = (date.fromisoformat(quarter.start)
                    - timedelta(days=1)).isoformat()
        stubs.append(((year.start, stub_end), filing["filed"],
                      year.value - quarter.value))
    return stubs


def _basis_events(facts: List[_Fact],
                  filings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Where a share basis changed, bracketed between two filing dates.

    Nothing in XBRL says "split". The evidence is one fiscal period carrying
    different values in two filings -- either because a later filing restated
    it, which takes four quarters to arrive, or because a filing's own year to
    date implies it, which takes one. Several periods bracket the same change,
    and intersecting their brackets narrows it: for NVDA's 10-for-1 that lands
    on 29 May to 28 August 2024 against an actual 7 June.
    """
    by_period: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for fact in facts:
        by_period[(fact.start, fact.end)].setdefault(fact.filed, fact.value)
    for period, filed, value in _implied_prior_stubs(filings):
        if period in by_period:
            by_period[period].setdefault(filed, value)

    raw: List[Dict[str, Any]] = []
    quiet: List[Tuple[str, str]] = []
    for period, observations in by_period.items():
        ordered = sorted(observations.items())
        for (earlier_filed, earlier), (later_filed, later) in zip(
                ordered, ordered[1:]):
            if earlier == 0 or later == 0:
                continue
            ratio = later / earlier
            if ratio <= 0:
                # A sign flip is a restatement, not a rescaling, and it says
                # nothing either way about the share basis.
                continue
            worst, best = _rescale_bounds(earlier, later)
            stated = max(ratio, 1.0 / ratio)
            if stated >= _MIN_BASIS_RESCALE and worst >= _MIN_BASIS_RESCALE:
                raw.append({"low": earlier_filed, "high": later_filed,
                            "ratio": ratio, "period": period})
            elif best < _MIN_BASIS_RESCALE:
                # Two filings agreeing about a period is evidence in its own
                # right: no share basis changed between them. That is what
                # places a filing sitting inside a bracket the restated
                # comparatives could only draw a quarter too wide.
                quiet.append((earlier_filed, later_filed))
            # Otherwise the cent the filer rounds to is large enough, against
            # this denominator, to leave the question open. Neither an event
            # nor evidence against one.

    clusters: List[Dict[str, Any]] = []
    for event in sorted(raw, key=lambda e: (e["low"], e["high"])):
        for cluster in clusters:
            low = max(cluster["low"], event["low"])
            high = min(cluster["high"], event["high"])
            if low < high and abs(event["ratio"] / cluster["ratio"] - 1.0) \
                    < _RATIO_MATCH:
                cluster["low"], cluster["high"] = low, high
                cluster["ratios"].append(event["ratio"])
                cluster["ratio"] = statistics.median(cluster["ratios"])
                cluster["periods"].add(event["period"])
                break
        else:
            clusters.append({"low": event["low"], "high": event["high"],
                             "ratio": event["ratio"],
                             "ratios": [event["ratio"]],
                             "periods": {event["period"]}})

    # One period restated on its own is a restatement, not a share basis: a
    # split rescales every period before it, so its evidence turns up in the
    # quarter, the year to date and the year alike, all by the same ratio.
    # Rescaling the series off a single disagreement would take the amended
    # figure -- which nobody could have traded on -- and stretch the whole
    # history to match it.
    kept = [c for c in clusters if len(c["periods"]) >= _BASIS_CORROBORATION]
    for cluster in kept:
        _narrow(cluster, quiet)
    return _coalesce(kept)


def _coalesce(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold together brackets that ended up covering the same filings.

    Narrowing can bring two brackets onto the same stretch of dates after the
    ratio test had kept them apart -- Salesforce's 4-for-1 reads as 0.2222
    through one period and 0.2857 through another, 29% apart. Left separate
    they multiply, rescaling the pre-split history by 0.0635 instead of 0.25.
    A filer does not split twice between two consecutive filings, so one
    stretch is one event.
    """
    merged: List[Dict[str, Any]] = []
    for cluster in sorted(clusters, key=lambda c: (c["low"], c["high"])):
        for other in merged:
            if max(other["low"], cluster["low"]) < min(other["high"],
                                                       cluster["high"]):
                other["low"] = max(other["low"], cluster["low"])
                other["high"] = min(other["high"], cluster["high"])
                other["ratios"].extend(cluster["ratios"])
                other["ratio"] = statistics.median(other["ratios"])
                other["periods"] |= cluster["periods"]
                break
        else:
            merged.append(cluster)
    return merged


def _narrow(cluster: Dict[str, Any], quiet: List[Tuple[str, str]]) -> None:
    """Trim a bracket by the stretches nothing changed in.

    Palo Alto's 3-for-1 sits between its 10-K of 6 September 2022 and the
    filing that first restated a comparative, five months later -- and the
    November 10-Q in between is left unplaceable. But that November filing and
    the one a year after it report the same quarter at the same value, so
    nothing changed across that stretch, which puts the split before it.

    A quiet stretch is only used when it touches one end of the bracket.
    One landing strictly inside would cut the bracket in two, and two
    candidate dates for one event is not an improvement on a wide one.
    """
    for start, end in quiet:
        low, high = cluster["low"], cluster["high"]
        if start <= low < end <= high:
            cluster["low"] = end
        elif low <= start < high <= end:
            cluster["high"] = start
        if cluster["low"] >= cluster["high"]:
            # Contradictory evidence. Put the bracket back rather than invent
            # an empty one, and let the ambiguity be reported as ambiguity.
            cluster["low"], cluster["high"] = low, high
            return


def _basis_factor(clusters: List[Dict[str, Any]],
                  filed: str) -> Tuple[Optional[float], Optional[Dict]]:
    """What to multiply a filing's figures by to reach today's share basis.

    A filing dated at or before a change's lower bracket is on the old basis
    and gets the factor; one dated at or after the upper bracket already
    reflects it. One dated strictly inside cannot be placed, and that is
    returned as a refusal rather than resolved by assumption -- being wrong
    here is a tenfold error, not a rounding one.
    """
    factor = 1.0
    for cluster in clusters:
        if filed <= cluster["low"]:
            factor *= cluster["ratio"]
        elif filed < cluster["high"]:
            return None, cluster
    return factor, None


# -------------------------------------------------------------- assembly

def _quarter_from_periods(filing: Dict[str, Any]) -> Optional[int]:
    """Which quarter a 10-Q covers, read off the periods it carries.

    Dell's first two FY2024 10-Qs declare a fiscal period focus of `FY` --
    accessions 0001571996-23-000019 and -000032, verified live. Dropping a
    filing whose focus is unusable costs that quarter and, through the year-ago
    term, the one four quarters later: most of a trailing window for one
    tagging slip.

    The year to date runs one, two or three quarters long, so its length over
    the quarter's says which. This works for a 12-week quarter and a 16-week
    one alike, which a fixed day count does not.
    """
    quarter_days = filing["shortest"].days
    if quarter_days <= 0:
        return None
    inferred = round(filing["longest"].days / quarter_days)
    return inferred if 1 <= inferred <= 3 else None


def _filings(facts: List[_Fact]) -> List[Dict[str, Any]]:
    """One entry per accession, with the periods that filing is *about*.

    `fy` and `fp` describe the filing, and every fact in it inherits them --
    including the prior-year comparatives. The facts the filing is reporting
    are the ones ending when its own period does; of those the shortest is the
    quarter and the longest the year to date, which is what makes this work for
    a 12-week quarter and a 16-week one alike without a duration window.
    """
    groups: Dict[str, List[_Fact]] = defaultdict(list)
    for fact in facts:
        groups[fact.accession].append(fact)

    out: List[Dict[str, Any]] = []
    for accession, group in groups.items():
        period_end = max(f.end for f in group)
        at_end = [f for f in group if f.end == period_end]
        if not at_end:
            continue
        head = group[0]
        out.append({
            "accession": accession, "form": head.form,
            "fiscal_year": head.fiscal_year, "fiscal_period": head.fiscal_period,
            "filed": head.filed, "period_end": period_end,
            "shortest": min(at_end, key=lambda f: f.days),
            "longest": max(at_end, key=lambda f: f.days),
        })
    return sorted(out, key=lambda g: (g["filed"], g["accession"]))


def _entry(fact: _Fact, clusters, source: str, concept: str,
           known_at: Optional[str] = None,
           derivation: Optional[str] = None) -> Optional[Dict[str, Any]]:
    factor, unresolved = _basis_factor(clusters, fact.filed)
    if factor is None:
        return {"unresolved": unresolved, "known_at": known_at or fact.filed}
    return {
        "period_start": fact.start, "period_end": fact.end,
        "eps": fact.value * factor, "eps_as_filed": fact.value,
        "basis_factor": factor, "source": source, "concept": concept,
        "known_at": known_at or fact.filed, "accession": fact.accession,
        "form": fact.form, "derivation": derivation,
    }


def _build_quarters(facts: List[_Fact], concept: str
                    ) -> Tuple[Dict[Tuple[int, int], Dict], List[Dict], List[Dict]]:
    """The quarterly EPS series, the basis changes found, and what they cost.

    Returns (quarters keyed on fiscal identity, basis clusters, periods that
    could not be placed on a share basis).
    """
    filings = _filings(facts)
    clusters = _basis_events(facts, filings)
    quarters: Dict[Tuple[int, int], Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    year_to_date: Dict[Tuple[int, int], _Fact] = {}
    annual: Dict[int, _Fact] = {}

    def place(key: Tuple[int, int], built: Optional[Dict[str, Any]]) -> None:
        if built is None:
            return
        if "unresolved" in built:
            unresolved.append({
                "fiscal_period": _period_key(*key),
                "known_at": built["known_at"],
                "between": [built["unresolved"]["low"],
                            built["unresolved"]["high"]],
                "ratio": built["unresolved"]["ratio"]})
            return
        existing = quarters.get(key)
        # The original filing wins. XBRL as filed is what was known at the
        # time; an amendment landing months later is not, and taking the
        # newest value silently rewrites every surprise in the history.
        if existing is None or built["known_at"] < existing["known_at"]:
            quarters[key] = built

    for filing in filings:
        base_form = filing["form"].split("/")[0]
        focus = filing["fiscal_period"].upper()
        fiscal_year = filing["fiscal_year"]

        if base_form == "10-Q":
            fiscal_quarter = (int(focus[1]) if focus in ("Q1", "Q2", "Q3")
                              else _quarter_from_periods(filing))
            if fiscal_quarter is None:
                continue
            year_to_date.setdefault((fiscal_year, fiscal_quarter),
                                    filing["longest"])
            place((fiscal_year, fiscal_quarter),
                  _entry(filing["shortest"], clusters, "reported", concept))
        elif base_form == "10-K":
            annual.setdefault(fiscal_year, filing["longest"])
            quarter = filing["shortest"]
            if quarter is not filing["longest"] and \
                    quarter.days <= _Q4_SPAN_SHARE * filing["longest"].days:
                place((fiscal_year, 4),
                      _entry(quarter, clusters, "reported", concept))

    for fiscal_year, year in annual.items():
        if (fiscal_year, 4) in quarters:
            continue
        nine = year_to_date.get((fiscal_year, 3))
        if nine is None or nine.start != year.start:
            # Without a nine-month figure covering the same fiscal year there
            # is nothing to subtract. An annual EPS in a quarterly slot is a
            # fourfold overstatement that would read as the largest surprise
            # in the universe, so the quarter is simply absent.
            continue
        year_factor, year_block = _basis_factor(clusters, year.filed)
        nine_factor, nine_block = _basis_factor(clusters, nine.filed)
        if year_factor is None or nine_factor is None:
            block = year_block or nine_block
            unresolved.append({
                "fiscal_period": _period_key(fiscal_year, 4),
                "known_at": max(year.filed, nine.filed),
                "between": [block["low"], block["high"]],
                "ratio": block["ratio"]})
            continue
        adjusted_year = year.value * year_factor
        adjusted_nine = nine.value * nine_factor
        quarters[(fiscal_year, 4)] = {
            "period_start": (date.fromisoformat(nine.end)
                             + timedelta(days=1)).isoformat(),
            "period_end": year.end,
            "eps": adjusted_year - adjusted_nine,
            # As the 10-K's own basis would have expressed it, which is what a
            # consensus estimate standing at that date was denominated in.
            "eps_as_filed": (adjusted_year - adjusted_nine) / year_factor,
            "basis_factor": year_factor,
            "source": "derived",
            "concept": concept,
            "known_at": max(year.filed, nine.filed),
            "accession": year.accession,
            "form": year.form,
            "derivation": (
                f"annual {adjusted_year:.4f} less the nine months to "
                f"{nine.end} ({adjusted_nine:.4f}); no 10-Q covers a fourth "
                f"quarter, and two figures each rounded to a cent can differ "
                f"by one from the announced number"),
        }

    return quarters, clusters, unresolved


# ------------------------------------------------------------ the series

def _series_shell(ticker: str, as_of: str) -> Dict[str, Any]:
    return {"ticker": ticker.upper(), "success": False, "error": None,
            "as_of": as_of, "cik": None, "source": None, "concept": None,
            "unit": None, "quarters": [], "basis_changes": [],
            "basis_unresolved": [], "concepts_tried": []}


def _is_stale(candidate: Dict[str, Any], as_of: str) -> bool:
    """Whether the newest quarter here is too old to be this filer's latest.

    A live domestic filer files a 10-Q within 45 days of each quarter end, so
    it is never quiet for long. Silence means one of two things and both need
    another look: the concept was abandoned -- Ford's continuing-operations EPS
    stops in 2011 while its diluted EPS is current -- or the cheap endpoint
    returned nothing, which SEC's does for filers `companyfacts` carries in
    full.
    """
    cutoff = (date.fromisoformat(as_of) - timedelta(days=_STALE_DAYS)).isoformat()
    return candidate["latest_known_at"] < cutoff


def _candidate(payload, as_of: str, concept: str
               ) -> Tuple[Optional[Dict[str, Any]], set]:
    facts, unit, forms = _parse_facts(payload, as_of)
    if not facts:
        return None, forms
    quarters, clusters, unresolved = _build_quarters(facts, concept)
    if not quarters:
        return None, forms
    return {"concept": concept, "unit": unit, "quarters": quarters,
            "clusters": clusters, "unresolved": unresolved,
            "latest_known_at": max(q["known_at"] for q in quarters.values()),
            }, forms


def _rank(candidate: Dict[str, Any]) -> Tuple[str, int]:
    """Currency first, then depth.

    Length is not currency. Ranking on quarter count alone picked Ford's dead
    continuing-operations tag over its live diluted one and reported a 2011
    quarter as today's signal -- a number that is not wrong so much as fifteen
    years out of date.
    """
    return candidate["latest_known_at"], len(candidate["quarters"])


def _walk_concepts(cik: str, as_of: str, tried: List[str],
                   payload_for) -> Tuple[Optional[Dict], set]:
    forms_seen: set = set()
    best: Optional[Dict[str, Any]] = None
    for concept in US_GAAP_EPS_CONCEPTS + IFRS_EPS_CONCEPTS:
        if concept not in tried:
            tried.append(concept)
        taxonomy, tag = concept.split(":", 1)
        payload = payload_for(taxonomy, tag)
        if payload is None:
            continue
        candidate, forms = _candidate(payload, as_of, concept)
        forms_seen |= forms
        if candidate is None:
            continue
        if best is None or _rank(candidate) > _rank(best):
            best = candidate
        # One concept carries the whole series; the chain exists to find which
        # one the filer uses, not to pool them. A current, deep series ends the
        # search and keeps the common case at a single request.
        if not _is_stale(candidate, as_of) and \
                len(candidate["quarters"]) >= SIGMA_QUARTERS + 5:
            break
    return best, forms_seen


def _forms_filed(facts_payload: Dict[str, Any]) -> set:
    """Every form this filer has tagged anything under.

    Read from the cover-page `dei` facts, which exist in every filing, so a
    filer that tags no EPS at all can still be told apart from one that files
    nothing quarterly.
    """
    forms: set = set()
    for body in (facts_payload.get("facts") or {}).get("dei", {}).values():
        for rows in (body.get("units") or {}).values():
            for row in rows:
                form = str(row.get("form") or "")
                if form:
                    forms.add(form)
    return forms


def _no_series_reason(ticker: str, cik: str, as_of: str, forms: set,
                      tried: List[str]) -> str:
    annual = sorted(f for f in forms if f in FOREIGN_ANNUAL_FORMS)
    if annual:
        return (f"{ticker} is a foreign private issuer: its EPS is tagged in "
                f"{', '.join(annual)} filings and it reports interim results "
                f"on 6-K, whose exhibits carry no XBRL. No quarterly EPS "
                f"exists for this filer anywhere on EDGAR, so a time-series "
                f"SUE cannot be built for it -- not from a shorter window, and "
                f"not from the annual figures.")
    if "10-Q" in forms:
        return (f"{ticker} files 10-Q but tags no consolidated EPS under any "
                f"of {', '.join(tried)}. SEC's company APIs return only "
                f"undimensioned facts, and a filer with several share classes "
                f"tags earnings per share against each class instead -- Visa "
                f"and Berkshire both do -- so the concept is absent here "
                f"rather than partial. This is a gap in what is readable, not "
                f"a company without earnings.")
    if forms:
        return (f"{ticker} tags EPS but no quarterly period could be "
                f"identified in the forms seen ({', '.join(sorted(forms))}). "
                f"No time-series SUE can be built from that.")
    return (f"{ticker} (CIK {cik}) tags none of the EPS concepts tried "
            f"({', '.join(tried)}) in any filing on or before {as_of}.")


def eps_series(ticker: str, as_of: Optional[str] = None) -> Dict[str, Any]:
    """Quarterly EPS keyed on fiscal identity, as it was knowable on `as_of`.

    Oldest first. Every entry carries `known_at` -- the filing date that made
    it computable -- because the period end is not when the number existed, and
    the drift being traded happens after the announcement, not after the
    quarter.
    """
    as_of = as_of or _today()
    result = _series_shell(ticker, as_of)

    try:
        cik = _cik_for(ticker)
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        result["error"] = (f"could not resolve {ticker.upper()} to a CIK: "
                           f"{type(exc).__name__}: {exc}")
        return result
    if not cik:
        result["error"] = (
            f"{ticker.upper()} does not appear in SEC's ticker-to-CIK file, so "
            f"it is not an SEC registrant under that symbol. Nothing was "
            f"fetched.")
        return result
    result["cik"] = cik
    name = ticker.upper()

    try:
        best, forms_seen = _walk_concepts(
            cik, as_of, result["concepts_tried"],
            lambda taxonomy, tag: _fetch_company_concept(cik, taxonomy, tag))
    except Exception as exc:  # noqa: BLE001 - an outage is not an answer
        result["error"] = (f"SEC companyconcept lookup failed for {name}: "
                           f"{type(exc).__name__}: {exc}. This is an outage, "
                           f"not a fact about the filer.")
        result["fetch_failed"] = True
        return result
    source = "companyconcept"

    if best is None or _is_stale(best, as_of):
        # The expensive endpoint, and only when the cheap one has come up
        # empty or stale. See `_fetch_company_facts`: SEC's two endpoints
        # disagree, and the cheap one is the one that loses facts.
        try:
            payload = _fetch_company_facts(cik)
        except Exception as exc:  # noqa: BLE001
            result["error"] = (f"SEC companyfacts lookup failed for {name}: "
                               f"{type(exc).__name__}: {exc}. This is an "
                               f"outage, not a fact about the filer.")
            result["fetch_failed"] = True
            return result
        if payload is not None:
            taxonomies = payload.get("facts") or {}
            fallback, forms = _walk_concepts(
                cik, as_of, result["concepts_tried"],
                lambda taxonomy, tag: taxonomies.get(taxonomy, {}).get(tag))
            forms_seen |= forms | _forms_filed(payload)
            if fallback is not None and (best is None
                                         or _rank(fallback) > _rank(best)):
                best, source = fallback, "companyfacts"

    if best is None:
        result["error"] = _no_series_reason(name, cik, as_of, forms_seen,
                                            result["concepts_tried"])
        return result

    if _is_stale(best, as_of):
        # Both endpoints have now been asked, so this is the filer's own
        # silence rather than a gap in the cheap one. A domestic filer that
        # is still reporting files a 10-Q within 45 days of every quarter end.
        latest = max(best["quarters"])
        result["error"] = (
            f"{name} (CIK {cik}): the most recent readable quarterly EPS is "
            f"{_period_key(*latest)}, filed {best['latest_known_at']}, which "
            f"is more than {_STALE_DAYS} days before {as_of}. A filer still "
            f"reporting is never that quiet, so this is a registrant that has "
            f"stopped filing, or one whose EPS moved to a tagging this cannot "
            f"read -- Berkshire's went to per-share-class facts in 2014, and "
            f"SEC's company APIs return only undimensioned ones.")
        return result

    result["success"] = True
    result["source"] = source
    result["concept"] = best["concept"]
    result["unit"] = best["unit"]
    result["basis_unresolved"] = best["unresolved"]
    result["basis_changes"] = [
        {"between": [c["low"], c["high"]], "ratio": c["ratio"],
         "note": (f"the same fiscal period is reported {c['ratio']:.4f} times "
                  f"larger after {c['high']} than it was on or before "
                  f"{c['low']}; figures filed on or before {c['low']} are "
                  f"rebased onto the later basis")}
        for c in best["clusters"]]
    result["quarters"] = [
        {"fiscal_period": _period_key(fy, fq), "fiscal_year": fy,
         "fiscal_quarter": fq, **entry}
        for (fy, fq), entry in sorted(best["quarters"].items())]
    return result


# ------------------------------------------------------------- the signal

def _signal_shell(ticker: str, as_of: str) -> Dict[str, Any]:
    return {"ticker": ticker.upper(), "success": False, "error": None,
            "as_of": as_of, "fiscal_period": None, "comparison_period": None,
            "period_end": None, "known_at": None, "eps": None,
            "eps_year_ago": None, "delta": None, "sigma": None,
            "sigma_quarters": None, "sigma_periods": [],
            "trailing_mean_delta": None, "sue": None, "concept": None,
            "source": None, "basis_changes": [], "concepts_tried": []}


def _quantum(entry: Dict[str, Any]) -> float:
    """How much of this figure is rounding, in the basis it now carries.

    Half a cent for a reported quarter and a whole one for a derived fourth,
    which is the difference of two separately rounded figures -- and scaled by
    the basis factor, because a cent on the pre-split basis is a tenth of one
    after a 10-for-1.
    """
    base = _EPS_QUANTUM * (2 if entry.get("source") == "derived" else 1)
    return base * abs(entry.get("basis_factor") or 1.0)


def _resolve_period(quarters: Dict[str, Dict], fiscal_period: Optional[str],
                    ticker: str, as_of: str) -> Tuple[Optional[Tuple[int, int]],
                                                      Optional[str]]:
    if fiscal_period is None:
        if not quarters:
            return None, f"{ticker}: no quarterly EPS was filed on or before {as_of}."
        return max(_parse_period(p) for p in quarters), None
    key = _parse_period(fiscal_period)
    if key is None:
        return None, f"{fiscal_period!r} is not a fiscal period like '2026Q3'."
    if _period_key(*key) not in quarters:
        return None, (
            f"{ticker} {fiscal_period}: no EPS for that fiscal quarter had "
            f"been filed by {as_of}.")
    return key, None


def _unresolved_note(series: Dict[str, Any], periods) -> Optional[str]:
    blocked = [u for u in series["basis_unresolved"]
               if u["fiscal_period"] in periods]
    if not blocked:
        return None
    first = blocked[0]
    return (f"{len(blocked)} of them sit on an unresolved share basis: a "
            f"change of {first['ratio']:.4f}x somewhere between "
            f"{first['between'][0]} and {first['between'][1]} could not be "
            f"placed against a filing, so rebasing them would be a guess")


def _signal_from_series(series: Dict[str, Any], key: Tuple[int, int],
                        as_of: str) -> Dict[str, Any]:
    ticker = series["ticker"]
    result = _signal_shell(ticker, as_of)
    result["concept"] = series["concept"]
    result["concepts_tried"] = series["concepts_tried"]
    result["basis_changes"] = series["basis_changes"]

    quarters = {q["fiscal_period"]: q for q in series["quarters"]}
    period = _period_key(*key)
    comparison = _period_key(*_shift(*key, back=4))
    result["fiscal_period"] = period
    result["comparison_period"] = comparison

    current = quarters[period]
    result["period_end"] = current["period_end"]
    result["known_at"] = current["known_at"]
    result["eps"] = current["eps"]
    result["source"] = current["source"]

    prior = quarters.get(comparison)
    if prior is None:
        note = _unresolved_note(series, {comparison})
        span = sorted(quarters)
        # The span is in the message because a short one usually means the
        # wrong registrant rather than a gap: SEC's ticker file maps XOM to
        # CIK 0002115436, a successor with one 10-Q on EDGAR, while the
        # history sits under the predecessor CIK and no ticker points at it.
        result["error"] = (
            f"{ticker} {period}: the same fiscal quarter a year earlier "
            f"({comparison}) is not in the series, so there is no seasonal "
            f"expectation to difference against. CIK {series['cik']} carries "
            f"{len(span)} quarter{'' if len(span) == 1 else 's'} on EDGAR "
            f"({span[0]} to {span[-1]})"
            + (f"; {note}" if note else "") + ".")
        return result
    result["eps_year_ago"] = prior["eps"]
    result["delta"] = current["eps"] - prior["eps"]

    # The trailing window, newest first, and never wider than eight. Reaching
    # past it to make the count up would give the patchiest histories a
    # denominator drawn from a different regime than everyone else's.
    window = [_shift(*key, back=step)
              for step in range(1, SIGMA_QUARTERS + 1)]
    deltas: List[float] = []
    used: List[str] = []
    quanta: List[float] = []
    for step_key in window:
        this = quarters.get(_period_key(*step_key))
        that = quarters.get(_period_key(*_shift(*step_key, back=4)))
        if this is None or that is None:
            continue
        deltas.append(this["eps"] - that["eps"])
        quanta.append(_quantum(this) + _quantum(that))
        used.append(_period_key(*step_key))

    result["sigma_quarters"] = len(deltas)
    result["sigma_periods"] = used
    if len(deltas) < MIN_SIGMA_QUARTERS:
        wanted = {_period_key(*k) for k in window}
        wanted |= {_period_key(*_shift(*k, back=4)) for k in window}
        note = _unresolved_note(series, wanted)
        result["error"] = (
            f"{ticker} {period}: only {len(deltas)} of the {SIGMA_QUARTERS} "
            f"trailing quarters have a year-on-year change, and "
            f"{MIN_SIGMA_QUARTERS} are required. A standard deviation from "
            f"fewer is an estimate of noise, and the window is not widened to "
            f"make the count up"
            + (f". {note}" if note else "") + ".")
        return result

    sigma = statistics.stdev(deltas)
    floor = max(quanta)
    result["trailing_mean_delta"] = statistics.fmean(deltas)
    if sigma <= floor:
        result["sigma"] = sigma
        result["error"] = (
            f"{ticker} {period}: the last {len(deltas)} year-on-year changes "
            f"agree to within the cent EPS is rounded to (dispersion "
            f"{sigma:.5f} against a rounding floor of {floor:.5f}), so there "
            f"is no scale this data can measure. Flooring sigma here would "
            f"invent one and print a very large SUE for a company whose "
            f"earnings have never varied.")
        return result

    result["sigma"] = sigma
    result["sue"] = result["delta"] / sigma
    result["success"] = True
    return result


def sue_ts(ticker: str, as_of: Optional[str] = None,
           fiscal_period: Optional[str] = None) -> Dict[str, Any]:
    """The time-series surprise for one quarter, or a refusal with a reason.

    `fiscal_period` defaults to the most recent quarter filed on or before
    `as_of` -- which is the question a daily job asks, and the one a backtest
    standing on a past date asks too.
    """
    as_of = as_of or _today()
    series = eps_series(ticker, as_of=as_of)
    if not series["success"]:
        result = _signal_shell(ticker, as_of)
        result["error"] = series["error"]
        result["concepts_tried"] = series["concepts_tried"]
        return result

    quarters = {q["fiscal_period"]: q for q in series["quarters"]}
    key, problem = _resolve_period(quarters, fiscal_period,
                                   series["ticker"], as_of)
    if key is None:
        result = _signal_shell(ticker, as_of)
        result["error"] = problem
        result["concept"] = series["concept"]
        result["concepts_tried"] = series["concepts_tried"]
        result["basis_changes"] = series["basis_changes"]
        result["fiscal_period"] = fiscal_period
        return result
    return _signal_from_series(series, key, as_of)


def sue_ts_history(ticker: str, as_of: Optional[str] = None) -> Dict[str, Any]:
    """Every quarter the filings support a signal for, oldest first.

    One fetch for the whole history: the expensive part is the request, and a
    backtest wants every quarter rather than the latest one.
    """
    as_of = as_of or _today()
    series = eps_series(ticker, as_of=as_of)
    out = {"ticker": series["ticker"], "success": series["success"],
           "error": series["error"], "as_of": as_of,
           "concept": series["concept"], "signals": [], "refusals": []}
    if not series["success"]:
        return out

    for quarter in series["quarters"]:
        key = _parse_period(quarter["fiscal_period"])
        signal = _signal_from_series(series, key, as_of)
        if signal["success"]:
            out["signals"].append(signal)
        else:
            out["refusals"].append({"fiscal_period": quarter["fiscal_period"],
                                    "error": signal["error"]})
    return out


# --------------------------------------------------------- the analyst leg

def _announced_before(ticker: str, fiscal_period: str, known_at: str,
                      as_of: str) -> str:
    """The last date a consensus could have stood before this print.

    The recorded announcement date when the store has one -- it is the event
    the market reacts to -- and otherwise the day before the filing, which is
    the latest date the figure was certainly not yet public. An estimate read
    after the print is not a surprise: the street has already marked to the
    actual, and the measured miss collapses toward zero.
    """
    for row in pit_store.announcements_as_of(ticker, as_of):
        if row["fiscal_period"] == fiscal_period:
            return (date.fromisoformat(row["announced_date"])
                    - timedelta(days=1)).isoformat()
    return (date.fromisoformat(known_at) - timedelta(days=1)).isoformat()


def sue_af(ticker: str, as_of: Optional[str] = None,
           fiscal_period: Optional[str] = None,
           actuals: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The analyst-based surprise, from the point-in-time consensus record.

    Same shape as `sue_ts`, with the consensus standing before each print in
    place of the year-ago quarter. It refuses today, and is expected to, for
    two separate reasons -- both worth knowing because only one of them goes
    away by waiting.

    **The consensus history cannot be fetched.** Finnhub returns four quarters
    at limit=12 and at limit=30 alike -- verified. It accrues one day at a time
    in `pit_store.consensus_snapshot`, and this starts answering on its own
    once eight quarters of it have gone by. Nothing here substitutes today's
    consensus for a past one; that would be lookahead of the purest kind, the
    estimate as revised after the fact.

    **A street estimate is not a GAAP number.** This is the one the brief did
    not anticipate. Verified live on 2026-08-27: Finnhub reports MSFT's 2026Q2
    actual as 4.14 against the 5.16 diluted EPS in the 10-Q, NVDA's 2026Q2 as
    1.05 against 1.08, and AAPL's 2026Q3 as 1.91 against 2.02. Subtracting a
    non-GAAP estimate from a GAAP actual gives every surprise a definitional
    gap that varies quarter to quarter and is, for MSFT, a dollar -- larger
    than the surprises themselves and pointing wherever the one-offs point.

    So `actuals` is required: {fiscal_period: reported EPS on the same basis
    the estimates are quoted on}, which is what the vendor's own `actual_eps`
    field carries. Until the daily job records that alongside the estimate,
    this refuses rather than mixing the two.

    Fiscal identity does line up, which was worth checking: Finnhub's `year`
    and `quarter` are the filer's own -- AAPL's June 2026 quarter is 2026Q3
    there and here -- even though its `period` field is a calendar bucket
    (2026-06-30 for a quarter that ended 2026-06-27), which is the AMAT trap
    `pit_store.record_announcement` already refuses to key on.
    """
    as_of = as_of or _today()
    series = eps_series(ticker, as_of=as_of)
    result = _signal_shell(ticker, as_of)
    result["consensus"] = None
    result["surprise"] = None
    result["surprises_available"] = 0
    if not series["success"]:
        result["error"] = series["error"]
        result["concepts_tried"] = series["concepts_tried"]
        return result

    result["concept"] = series["concept"]
    result["concepts_tried"] = series["concepts_tried"]
    quarters = {q["fiscal_period"]: q for q in series["quarters"]}
    key, problem = _resolve_period(quarters, fiscal_period,
                                   series["ticker"], as_of)
    if key is None:
        result["error"] = problem
        return result

    name = series["ticker"]

    def consensus_for(step_key) -> Optional[Tuple[Dict[str, Any], float]]:
        quarter = quarters.get(_period_key(*step_key))
        if quarter is None:
            return None
        cutoff = _announced_before(name, quarter["fiscal_period"],
                                   quarter["known_at"], as_of)
        snapshot = pit_store.consensus_as_of(
            name, quarter["fiscal_period"], as_of=cutoff)
        if snapshot is None or snapshot.get("eps_estimate") is None:
            return None
        return quarter, float(snapshot["eps_estimate"])

    period = _period_key(*key)
    current = quarters[period]
    result["fiscal_period"] = period
    result["period_end"] = current["period_end"]
    result["known_at"] = current["known_at"]
    result["eps"] = current["eps"]
    result["source"] = current["source"]

    window = [_shift(*key, back=step) for step in range(1, SIGMA_QUARTERS + 1)]
    covered = [k for k in window if consensus_for(k) is not None]
    result["surprises_available"] = len(covered)
    result["sigma_periods"] = [_period_key(*k) for k in covered]
    result["sigma_quarters"] = len(covered)

    if actuals is None:
        result["error"] = (
            f"{name} {period}: no actual on the same basis as the estimates "
            f"was supplied. A street estimate is a non-GAAP figure and XBRL "
            f"carries only the GAAP one -- Finnhub reports MSFT's 2026Q2 "
            f"actual as 4.14 against the 5.16 in the 10-Q -- so subtracting "
            f"one from the other measures the definition, not the surprise. "
            f"Pass `actuals` keyed on fiscal period, or record the vendor's "
            f"own actual alongside the estimate. The point-in-time consensus "
            f"record holds {len(covered)} of the {SIGMA_QUARTERS} trailing "
            f"prints for this name.")
        return result

    def surprise_for(step_key) -> Optional[Tuple[float, float]]:
        pair = consensus_for(step_key)
        if pair is None:
            return None
        quarter, estimate = pair
        actual = actuals.get(quarter["fiscal_period"])
        if actual is None:
            return None
        # Both numbers were quoted in the share basis of their day; the series
        # has been rebased onto today's. Scaling the pair the same way keeps
        # sigma on one basis across a split.
        return (actual - estimate,
                (actual - estimate) * quarter["basis_factor"])

    trailing, used = [], []
    for step_key in window:
        pair = surprise_for(step_key)
        if pair is None:
            continue
        trailing.append(pair[1])
        used.append(_period_key(*step_key))
    result["surprises_available"] = len(trailing)
    result["sigma_periods"] = used
    result["sigma_quarters"] = len(trailing)

    head = surprise_for(key)
    if head is None:
        result["error"] = (
            f"{name} {period}: no consensus was recorded before that print, or "
            f"no actual was supplied for it. The analyst leg reads only the "
            f"point-in-time record, which accrues forward from the day the "
            f"recorder started.")
        return result
    result["surprise"] = head[0]
    result["consensus"] = actuals[period] - head[0]

    if len(trailing) < MIN_SIGMA_QUARTERS:
        result["error"] = (
            f"{name} {period}: {len(trailing)} of the {SIGMA_QUARTERS} "
            f"trailing prints have a consensus recorded before them and "
            f"{MIN_SIGMA_QUARTERS} are required. Analyst history cannot be "
            f"reconstructed -- Finnhub returns four quarters at limit=12 and "
            f"at limit=30 alike -- so this fills in going forward and refuses "
            f"until it has.")
        return result

    sigma = statistics.stdev(trailing)
    result["trailing_mean_delta"] = statistics.fmean(trailing)
    if sigma <= 0:
        result["sigma"] = sigma
        result["error"] = (
            f"{name} {period}: the last {len(trailing)} surprises are "
            f"identical, so there is no dispersion to divide by.")
        return result

    result["sigma"] = sigma
    result["delta"] = head[1]
    result["sue"] = head[1] / sigma
    result["success"] = True
    return result
