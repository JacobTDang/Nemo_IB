"""Share count, dilution, and shelf activity.

Dilution is the blind spot this module closes. Share count previously existed
only as a point-in-time snapshot from yfinance, so a company could grow its
denominator steadily and every per-share metric would drift in the flattering
direction with nothing to show it.

Three things make this harder than reading one number:

1. **Multi-class filers.** GOOGL tags Class A, B, and C as separate facts with
   identical labels. Taking the first yields 5.868bn against a true 12.23bn.
   Every result therefore reports its per-class breakdown alongside the total,
   so a caller can see which classes were found rather than trusting a bare sum.

   Summing them is only right when the classes are worth the same. Berkshire's
   Class A converts into 1,500 Class B, so 488,450 + 1,408,035,161 = 1.4085bn
   is 34% short of the company, and the obvious cross-check hides it: Yahoo's
   `sharesOutstanding` agrees with that sum to 0.03%, because one source
   dropped Class A and the other under-weighted it. The sum is therefore
   weighed against a market capitalisation, which cannot make either mistake,
   and where the two disagree the total is labelled rather than quietly used.
   The ratio is a per-filer fact: 1,500 is derived here, never assumed, and
   where it cannot be derived the tool says so instead of assuming 1:1.

2. **Company-specific member tags.** Alphabet's Class C is
   `goog:CapitalClassCMember`, not `us-gaap:CommonClassCMember`. Labels are
   derived from whatever the filer used; a whitelist of standard members would
   silently drop the class.

3. **Stock splits.** The cover-page tag states the shares outstanding *at that
   time*, in the units of that time. NVDA's November 2023 cover says 2.47bn and
   its May 2026 cover says 24.2bn, and the whole difference is the 10-for-1 of
   June 2024. Differencing them reported +879.76% "dilution" for a company that
   repurchased $40.1bn of stock over the same window and whose adjusted count
   fell 2%. Four of the five split cases checked -- NVDA, CMG, WMT, LRCX --
   inverted the verdict, which makes this worse than a missing number: it is a
   confident answer pointing the wrong way on a thesis-level input.

   The series is therefore rebased onto the newest filing's basis, and the
   ratio, its date and the raw figures are all reported so the adjustment can
   be checked. Where a split is evident but no ratio can be stood behind, the
   tool declines to name a direction. A refusal is a worse answer than a
   correct one and a far better one than an inverted one.

Shelf activity is the other half. A share count that rose is history; an
effective S-3 with 424B5 takedowns is the mechanism, and it tells you the
dilution is ongoing rather than finished.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from math import gcd
from typing import Any, Dict, List, Optional, Tuple

from edgar import Company

from .foreign_issuer import form_mismatch_note, not_covered_reason
from .sec_series import NotCovered, _require_identity, fetch_concept_series

SHARES_CONCEPT = "dei:EntityCommonStockSharesOutstanding"
CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"

# Below this, a change is rounding and share-count noise from option exercises
# rather than a signal worth naming.
_FLAT_THRESHOLD_PCT = 0.1

# A share count that moves more than this between two consecutive filings is
# either a split or a genuinely dramatic financing event, and both are worth
# confirming against the split calendar. Below it there is nothing to look up,
# which keeps the quiet filer -- nearly every filer, nearly every quarter --
# free of a network call it does not need. The cost of the floor: a stock
# dividend smaller than 15% (a 21-for-20, say) is not detected and still reads
# as dilution. That is a bounded error of a few percent rather than an
# order-of-magnitude inversion.
_SPLIT_LOOKUP_JUMP = 0.15

# Below this, a round-looking ratio is a coincidence. Companies do issue 20% or
# 40% more shares in a quarter; they do not do 2-for-1 splits by accident.
_SPLIT_MIN_RATIO = 1.5

# The count either side of a split also moves for buybacks and option
# exercises, so an implied ratio is never exactly 10.000. LRCX's straddling
# pair implies 9.84 across a six-month gap -- the widest drift measured on the
# 2024 split cohort (NVDA 9.97, CMG 49.86, WMT 2.99, AVGO 10.03).
_SPLIT_RATIO_TOLERANCE = 0.02

# Nothing splits by more than this. Bounds the candidate search rather than
# expressing a view: CMG's 50-for-1 is the largest in recent memory.
_MAX_SPLIT_MULTIPLE = 200

# Fractional splits (3-for-2 and friends). Only considered when the split
# calendar could not be read: with a source in hand, positive evidence of "no
# split here" should not be overturned by a jump that merely resembles 3:2,
# because a genuine 50% equity raise resembles it exactly.
_FRACTIONAL_SPLITS = tuple(sorted({
    p / q
    for q in (2, 3, 4)
    for p in range(3, 4 * q + 1)
    if gcd(p, q) == 1 and p / q >= _SPLIT_MIN_RATIO
}))

# How far the SEC cover-page sum and the market's own share count may differ
# before they are measuring different things. A cover page is up to a quarter
# old, so ordinary buybacks and option exercises move it a percent or two;
# GOOGL's three classes agree to 0.0005% and Berkshire's two are 51.98% apart.
# Nothing real has been measured in between.
_MARKET_BASIS_TOLERANCE_PCT = 5.0

# A share class converts into another at a ratio written into a charter: 1,500
# for Berkshire, 10 for a typical dual-class founder share. Charters use round
# numbers, and with a class as small as Berkshire's 488,450 Class A shares
# almost any residual divides into something near a whole number -- 433.96 for
# a gap that is really an equity raise. Requiring two significant figures is
# what separates a conversion ratio from a coincidence.
_CONVERSION_RATIO_TOLERANCE = 0.005
_MIN_CONVERSION_RATIO = 2.0

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

UNDETERMINED = "split_suspected_undetermined"


def _class_label(member: Optional[str]) -> str:
    """Turn an XBRL member tag into something a human reads.

    `us-gaap:CommonClassAMember` becomes "Class A" and
    `goog:CapitalClassCMember` becomes "Capital Class C". Derived from the tag
    rather than looked up, because filers invent their own members and a lookup
    table would drop the ones it had not seen.
    """
    if not member:
        return "Common"
    local = member.split(":", 1)[-1]
    if local.endswith("Member"):
        local = local[: -len("Member")]
    words = re.findall(r"[A-Z][a-z0-9.]*|[A-Z]+(?![a-z])", local) or [local]
    # "CommonClassA" reads better as "Class A" than "Common Class A".
    if len(words) > 1 and words[0] == "Common" and words[1] == "Class":
        words = words[1:]
    return " ".join(words)


class MarketShareCountUnavailable(RuntimeError):
    """The market's own share count could not be read.

    Distinct from a filer whose classes are equivalent. Raised rather than
    returned so that "we could not check" can never come back as "we checked
    and they agree" -- which is precisely how a 1,500:1 filer would pass.
    """


def fetch_market_share_count(ticker: str) -> float:
    """Shares outstanding across every class, in quoted-share equivalents.

    The SEC cover page states each class in its own units and nothing in the
    filing says what those units are worth relative to each other. The market
    does: a quote and a market capitalisation are both denominated in the
    listed class, so their quotient is the whole company measured in the units
    the listed class trades in. For Berkshire that is 2,140,709,794 against a
    cover-page sum of 1,408,523,611; for Alphabet the two agree to 0.0005%.

    yfinance is the source -- already a dependency, and `fetch_split_history`
    in this module reads the same provider. Its own `impliedSharesOutstanding`
    is preferred where it agrees with market cap over price; the quotient is
    the fallback because it is consistent with the market cap by construction.
    """
    import yfinance as yf  # imported here so this module loads without it

    return _market_share_count_from_info(yf.Ticker(ticker).info or {}, ticker)


def _market_share_count_from_info(info: Dict[str, Any], ticker: str) -> float:
    """The quoted-basis count inside a yfinance info dict. Split out for tests."""
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    quotient = market_cap / price if market_cap and price else None
    provider = info.get("impliedSharesOutstanding") or None

    if quotient is None and provider is None:
        raise MarketShareCountUnavailable(
            f"the quote provider returned neither a market capitalisation nor "
            f"an implied share count for {ticker!r}, so the share basis its "
            f"classes are stated on could not be established.")
    if quotient is None:
        return float(provider)
    if provider is not None and abs(provider - quotient) / quotient <= 0.005:
        return float(provider)
    return float(quotient)


class SplitCalendarUnavailable(RuntimeError):
    """The split calendar could not be read.

    Distinct from a filer that has never split. Raised rather than returned so
    an unadjusted series can never come back wearing an "adjusted" label.
    """


def fetch_split_history(ticker: str) -> List[Dict[str, Any]]:
    """Stock splits for `ticker`, oldest first, as {"date", "ratio"}.

    yfinance is the source: it is already a dependency, `get_corporate_actions`
    reads the same series, and a split calendar is not something SEC XBRL
    exposes reliably (`StockholdersEquityNoteStockSplitConversionRatio` is
    tagged by some filers and not others, and costs a filing fetch each).

    Ratios are forward multiples: 10.0 is a 10-for-1, 0.1 a 1-for-10 reverse.

    Raises rather than returning an empty list when the source cannot be read.
    "No splits" and "could not find out" lead to opposite conclusions about the
    same series, and conflating them is exactly how an unadjusted series would
    come back wearing an "adjusted" label.
    """
    import yfinance as yf  # imported here so this module loads without it

    return _splits_from_yfinance_ticker(yf.Ticker(ticker), ticker)


def _splits_from_yfinance_ticker(handle: Any, ticker: str) -> List[Dict[str, Any]]:
    """The split events on an already-constructed yfinance handle.

    Split out so the empty-versus-unresolved distinction can be tested without
    Yahoo. yfinance does not raise on a symbol it cannot resolve: it prints
    `HTTP Error 404` and returns an empty series, which is byte-identical to
    Berkshire Hathaway genuinely never having split. `history_metadata` is
    populated only when Yahoo actually answered, so it separates the two at no
    extra network cost -- both branches have already paid for the one request.
    """
    series = handle.splits
    events: List[Dict[str, Any]] = []
    if series is None or len(series) == 0:
        metadata = getattr(handle, "history_metadata", None)
        if not metadata:
            raise SplitCalendarUnavailable(
                f"Yahoo returned no data for {ticker!r}, so its split calendar "
                f"could not be read. This is not the same as {ticker!r} having "
                f"never split, and the two lead to opposite conclusions about "
                f"the same share count.")
        return events
    for stamp, value in series.items():
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            continue
        # A ratio of exactly one is "a split that changed nothing", which is a
        # different claim from "no split" and not one worth rebasing on --
        # `get_corporate_actions` refuses to report 1.0 for the same reason.
        if ratio != ratio or ratio <= 0 or ratio == 1.0:
            continue
        moment = _first_date(str(stamp))
        if moment is None:
            continue
        events.append({"date": moment, "ratio": ratio})
    events.sort(key=lambda event: event["date"])
    return events


def _first_date(text: Any) -> Optional[str]:
    """The ISO date inside a label, or None.

    Period labels arrive as a bare date, as "instant_2024-05-24", and as a
    pandas timestamp with a timezone attached. All three carry the date in the
    same shape.
    """
    match = _DATE_PATTERN.search(str(text or ""))
    return match.group(0) if match else None


def _as_of(point: Any) -> Optional[str]:
    """The date a filing's cover-page count speaks for.

    The fact's own instant, not the filing date: NVDA signs its cover five days
    before it files, and a split landing in those five days would be applied to
    the wrong side of itself. Falls back to the filing date when the instant is
    unreadable, since a date a few days late still places the observation
    correctly relative to a split.
    """
    instants = [d for d in (_first_date(f.period) for f in point.deduplicated())
                if d]
    if instants:
        return max(instants)
    return _first_date(point.filing_date)


def _round_split_ratio(ratio: float, *, strict: bool) -> Optional[float]:
    """The round split ratio `ratio` sits on, or None if it sits on none.

    This is the line between the two failure modes. Too eager and a real 40%
    equity raise is suppressed as a "suspected split"; too slack and a 10-for-1
    is reported as 900% dilution. Splits land on round ratios and financings do
    not, so roundness -- within the drift a real quarter adds -- is the test,
    and a floor of 1.5x keeps the ordinary noise of option exercises out of it.

    `strict` narrows the candidates to whole multiples and their reverses. It
    is used when the split calendar was read successfully: having positive
    evidence that no split occurred, only an unmistakably round jump should
    override it, and a 50% raise is indistinguishable from a 3-for-2.
    """
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return None
    if ratio != ratio or ratio <= 0:
        return None
    forward = ratio if ratio >= 1.0 else 1.0 / ratio
    if forward < _SPLIT_MIN_RATIO - 1e-9:
        return None

    candidates = [float(n) for n in range(2, _MAX_SPLIT_MULTIPLE + 1)]
    if not strict:
        candidates.extend(_FRACTIONAL_SPLITS)
    best = min(candidates, key=lambda c: abs(forward - c) / c)
    if abs(forward - best) / best > _SPLIT_RATIO_TOLERANCE:
        return None
    return best if ratio >= 1.0 else 1.0 / best


def _jumps(totals: List[Tuple[str, float]]) -> List[Tuple[str, str, float]]:
    """Consecutive (from_date, to_date, ratio) triples, oldest pair first."""
    out: List[Tuple[str, str, float]] = []
    for (older_date, older), (newer_date, newer) in zip(totals, totals[1:]):
        if not older:
            continue
        out.append((older_date, newer_date, newer / older))
    return out


def _needs_the_calendar(totals: List[Tuple[str, float]]) -> bool:
    """Whether any interval moved far enough to be worth looking up."""
    high = 1.0 + _SPLIT_LOOKUP_JUMP
    return any(ratio > high or ratio < 1.0 / high
               for _, _, ratio in _jumps(totals))


def _consult_split_calendar(
        ticker: str,
        totals: List[Tuple[str, float]]) -> Tuple[Optional[List[Dict[str, Any]]],
                                                  Optional[str]]:
    """(events, error). `None` events means the calendar could not be read.

    Distinct from `[]`, which means it was read and holds no split. The caller
    treats the two very differently, which is the entire point of separating
    them.
    """
    if not _needs_the_calendar(totals):
        return [], None
    try:
        return fetch_split_history(ticker), None
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return None, f"{type(exc).__name__}: {exc}"


def _factor_for(as_of: Optional[str],
                applied: List[Dict[str, Any]]) -> float:
    """Multiple that puts a count stated on `as_of` onto the newest basis.

    Strictly after: a forward split's shares are distributed before the ex-date
    opens, so a cover page dated the day of the split already counts them.
    Applying the ratio there too would invent the same error in reverse.
    """
    factor = 1.0
    for event in applied:
        if as_of is None or event["date"] > as_of:
            factor *= event["ratio"]
    return factor


def _direction(change_pct: Optional[float]) -> str:
    if change_pct is None:
        return "insufficient_history"
    if change_pct > _FLAT_THRESHOLD_PCT:
        return "dilution"
    if change_pct < -_FLAT_THRESHOLD_PCT:
        return "buyback"
    return "flat"


def _pct_change(oldest: Optional[float],
                latest: Optional[float]) -> Optional[float]:
    if not oldest or latest is None:
        return None
    return (latest - oldest) / oldest * 100.0


def _round_conversion_ratio(ratio: float) -> Optional[float]:
    """The round conversion ratio `ratio` sits on, or None if it sits on none.

    Two significant figures, because that is what a charter contains. 1,500 and
    10 and 200 pass; 434 -- what a 15% equity raise implies against Berkshire's
    Class A count -- does not, and that is the case this check exists to reject.
    """
    from math import floor, log10

    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return None
    if ratio != ratio or ratio < _MIN_CONVERSION_RATIO:
        return None
    magnitude = 10 ** (floor(log10(ratio)) - 1)
    candidate = round(ratio / magnitude) * magnitude
    if candidate < _MIN_CONVERSION_RATIO:
        return None
    if abs(ratio - candidate) / candidate > _CONVERSION_RATIO_TOLERANCE:
        return None
    return float(candidate)


def _implied_class_weights(latest_by_class: Dict[str, float],
                           quote_equivalent_total: float
                           ) -> Optional[Dict[str, float]]:
    """What each class is worth in quoted-share units, or None.

    Solvable only for two classes: the listed one carries a weight of one and
    the residual falls entirely on the other. With three the residual has more
    than one way to be split and any answer would be a choice dressed as a
    measurement.

    The listed class is taken to be the larger count, which is what makes a
    class liquid enough to quote. The result is published only when it lands on
    a round ratio *and* rebuilds the market's own total, so the arithmetic has
    to agree with itself twice before a number this load-bearing is stated.
    """
    if len(latest_by_class) != 2:
        return None
    ordered = sorted(latest_by_class.items(), key=lambda kv: kv[1], reverse=True)
    (listed_label, listed_count), (other_label, other_count) = ordered
    if not other_count or not listed_count:
        return None

    weight = (quote_equivalent_total - listed_count) / other_count
    rounded = _round_conversion_ratio(weight)
    if rounded is None:
        return None
    rebuilt = rounded * other_count + listed_count
    if abs(rebuilt - quote_equivalent_total) / quote_equivalent_total > 0.001:
        return None
    return {other_label: rounded, listed_label: 1.0}


def _empty_share_basis() -> Dict[str, Any]:
    return {
        "total_basis": None,
        "classes_found": [],
        "economically_equivalent": None,
        "quote_equivalent_total": None,
        "quote_equivalent_source": None,
        "implied_class_weights": None,
        "gap_pct": None,
        "source_error": None,
    }


def _share_basis(ticker: str, latest_by_class: Dict[str, float],
                 latest_total: Optional[float]
                 ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """(basis, warnings) for the units `latest_total` is stated in.

    `latest_total` adds the classes at one to one. For Alphabet that is right;
    for Berkshire it is wrong by a third, because a Class A share converts into
    1,500 Class B and the sum counts it as one. The two mistakes cancel into
    something that looks reassuring: Berkshire's cover-page sum 1,408,523,611
    and Yahoo's `sharesOutstanding` 1,408,035,161 agree to 0.03%, so the
    obvious cross-check between two independent sources passes while both are
    a third short -- one dropped Class A, the other under-weighted it.

    So the check is made against a source that cannot make the same mistake: a
    market capitalisation is the whole company, whatever its classes are worth
    to each other. A single-class filer has nothing to check and pays nothing
    for the check -- no lookup is made at all.
    """
    basis = _empty_share_basis()
    classes = sorted(latest_by_class)
    basis["classes_found"] = classes

    if len(classes) <= 1:
        basis.update({
            "total_basis": "single_class",
            "economically_equivalent": True,
            "quote_equivalent_total": latest_total,
            "quote_equivalent_source": "sec_cover_page",
        })
        return basis, []

    basis["total_basis"] = "sum_of_classes_unweighted"
    try:
        market_total = fetch_market_share_count(ticker)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        basis["source_error"] = f"{type(exc).__name__}: {exc}"
        market_total = None

    if not market_total or not latest_total:
        return basis, [{
            "code": "multi_class_basis_unverified",
            "message": (
                f"{ticker}: latest_total adds {len(classes)} share classes "
                f"({', '.join(classes)}) at 1:1, and whether they are worth "
                f"1:1 is a fact about this filer that the filings do not "
                f"state. It could not be established here "
                f"({basis['source_error'] or 'no market share count available'}). "
                f"A Berkshire Class A share converts into 1,500 Class B, "
                f"which makes the same sum 34% short. Do not multiply "
                f"latest_total by a share price until the basis is known."),
            "classes_found": classes,
        }]

    gap_pct = (market_total - latest_total) / latest_total * 100.0
    basis.update({
        "quote_equivalent_total": market_total,
        "quote_equivalent_source": "market_capitalisation_over_price",
        "gap_pct": gap_pct,
    })

    if abs(gap_pct) <= _MARKET_BASIS_TOLERANCE_PCT:
        basis["economically_equivalent"] = True
        return basis, []

    basis["economically_equivalent"] = False
    weights = _implied_class_weights(latest_by_class, market_total)
    basis["implied_class_weights"] = weights

    if weights:
        named = ", ".join(
            f"one {label} carries {weight:,.0f} quoted shares"
            for label, weight in sorted(weights.items(),
                                        key=lambda kv: kv[1], reverse=True)
            if weight != 1.0)
        cause = (f" The gap is explained exactly by a conversion ratio: "
                 f"{named}, which rebuilds the market's total to within "
                 f"0.1%.")
    else:
        cause = (" The classes may convert at a ratio other than 1:1, or the "
                 "cover page may simply predate an issuance; neither can be "
                 "settled from these filings.")

    return basis, [{
        "code": "multi_class_unweighted_total",
        "message": (
            f"{ticker}: latest_total ({latest_total:,.0f}) adds "
            f"{', '.join(classes)} at 1:1, and the market prices the company "
            f"as {market_total:,.0f} shares of the listed class -- "
            f"{gap_pct:+.2f}% away.{cause} Multiplying latest_total by a share "
            f"price understates the market capitalisation by "
            f"{(1 - latest_total / market_total) * 100:.2f}%; "
            f"quote_equivalent_total is the count that pairs with a quote."),
        "classes_found": classes,
        "quote_equivalent_total": market_total,
        "implied_class_weights": weights,
    }]


def _empty_adjustment() -> Dict[str, Any]:
    return {
        "adjusted": False,
        "source": "yfinance",
        "source_error": None,
        "splits_applied": [],
        "cumulative_factor": 1.0,
        "basis": None,
        "raw_oldest_total": None,
        "raw_latest_total": None,
        "adjusted_oldest_total": None,
        "raw_change_pct": None,
        "unexplained_jumps": [],
    }


def _failure(ticker: str, message: str,
             form: Optional[str] = None) -> Dict[str, Any]:
    """Every miss in this module funnels here, so the form guard sits here too.

    A foreign private issuer reports on 6-K, which carries no XBRL, so a share
    count walked over 10-Q filings finds nothing. "Not covered" would read as
    a filer that does not disclose its share count.
    """
    mismatch = form_mismatch_note(ticker, form) if form else None
    return {
        "ticker": ticker,
        "success": False,
        "wrong_form": bool(mismatch),
        "error": not_covered_reason(ticker, form, message) if form else message,
        "latest_total": None,
        "latest_total_basis": None,
        "share_basis": _empty_share_basis(),
        "by_class": {},
        "classes_found": [],
        "total_series": [],
        "change_pct": None,
        "raw_change_pct": None,
        "direction": "not_covered",
        "split_adjusted": False,
        "split_adjustment": _empty_adjustment(),
        "by_class_change": {},
        "warnings": [],
    }


def get_share_count_series(ticker: str, limit: int = 8,
                           form: str = "10-Q") -> Dict[str, Any]:
    """Shares outstanding across the last `limit` filings, newest first.

    Returns the per-class breakdown, the summed total per filing, and the change
    from oldest to newest in the window. A filer that does not tag the concept
    gets an explicit failure rather than a zero, because zero shares outstanding
    is a meaningful and very different claim.

    The change is **split-adjusted**: history is rebased onto the newest
    filing's share basis so a 10-for-1 does not read as 900% dilution.
    `latest_total` stays as filed, which keeps it comparable to the count the
    market quotes. `raw_change_pct` and the per-filing `total` keep the
    unadjusted figures, and `split_adjustment` names the ratio applied and the
    date it was applied from, so the arithmetic can be reproduced.

    When a split is evident in the counts but no ratio can be stood behind --
    the calendar was unreachable, or it disagrees with what the filings plainly
    show -- `direction` is `split_suspected_undetermined` and `change_pct` is
    None. A percentage beside an undetermined direction is the same wrong
    number wearing a disclaimer, so it is withheld rather than qualified.
    """
    try:
        points = fetch_concept_series(ticker, SHARES_CONCEPT, form=form, limit=limit)
    except NotCovered as exc:
        return _failure(ticker, f"share count not covered: {exc}", form)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return _failure(ticker, f"{type(exc).__name__}: {exc}", form)

    # deduplicated(), not facts: Biogen emits one share-count fact twice, and
    # total() already discounts the repeat. Walking the raw list listed every
    # Biogen period twice and, for anyone adding the classes up, handed back
    # the doubled count the total was fixed to avoid. Distinct classes carry
    # distinct contexts and survive.
    observations = [{
        "point": point,
        "as_of": _as_of(point),
        "facts": point.deduplicated(),
        "total": point.total(),
    } for point in points]

    # Oldest first for the split arithmetic; the payload stays newest first.
    oldest_first = list(reversed(observations))
    dated_totals = [(o["as_of"], o["total"]) for o in oldest_first if o["total"]]

    events, source_error = _consult_split_calendar(ticker, dated_totals)
    adjustment = _empty_adjustment()
    adjustment["source_error"] = source_error

    # Only splits inside the window matter. One before the oldest observation
    # is already in every count here, and one after the newest cancels out of
    # a comparison rebased onto that newest count. An undatable window end
    # leaves the series unadjusted, and the check below then refuses a
    # direction rather than guessing which side of the split a filing sits on.
    applied: List[Dict[str, Any]] = []
    if events and dated_totals:
        oldest_as_of, latest_as_of = dated_totals[0][0], dated_totals[-1][0]
        if oldest_as_of is not None and latest_as_of is not None:
            applied = [e for e in events
                       if oldest_as_of < e["date"] <= latest_as_of]

    for observation in observations:
        observation["factor"] = _factor_for(observation["as_of"], applied)

    by_class: Dict[str, List[Dict[str, Any]]] = {}
    total_series: List[Dict[str, Any]] = []
    for observation in observations:
        point, factor = observation["point"], observation["factor"]
        for fact in observation["facts"]:
            label = _class_label(fact.dimension_member(CLASS_AXIS))
            by_class.setdefault(label, []).append({
                "filing_date": point.filing_date,
                "period": fact.period,
                "shares": fact.value,
                "split_factor": factor,
                "shares_split_adjusted": fact.value * factor,
            })
        total = observation["total"]
        total_series.append({
            "filing_date": point.filing_date,
            "form": point.form,
            "as_of": observation["as_of"],
            "total": total,
            "split_factor": factor,
            "total_split_adjusted": None if total is None else total * factor,
        })

    latest_total = total_series[0]["total"] if total_series else None

    # The classes as of the newest filing, which is the filing latest_total
    # speaks for. Read from that observation rather than from by_class, because
    # a class the newest cover page does not carry must not be weighed into a
    # total the newest cover page produced.
    latest_by_class: Dict[str, float] = {}
    if observations:
        for fact in observations[0]["facts"]:
            label = _class_label(fact.dimension_member(CLASS_AXIS))
            latest_by_class[label] = latest_by_class.get(label, 0.0) + fact.value
    share_basis, basis_warnings = _share_basis(ticker, latest_by_class,
                                               latest_total)

    # The safety net, and the reason the calendar is not trusted blindly. A
    # discontinuity that survives the adjustment is a split the calendar got
    # wrong, missed, or dated differently from the filings -- and a ratio that
    # cannot be pinned down is not one to publish a direction on. Strict while
    # the calendar was readable: with positive evidence of no split, only an
    # unmistakably round jump may override it, or a genuine 50% equity raise
    # would be silenced as a suspected 3-for-2.
    adjusted_totals = [(o["as_of"], o["total"] * o["factor"])
                       for o in oldest_first if o["total"]]
    unexplained: List[Dict[str, Any]] = []
    for older, newer, ratio in _jumps(adjusted_totals):
        round_ratio = _round_split_ratio(ratio, strict=source_error is None)
        if round_ratio is None:
            continue
        unexplained.append({"from_date": older, "to_date": newer,
                            "implied_ratio": ratio,
                            "nearest_split_ratio": round_ratio})

    # One observation is a snapshot, not a series: comparing it with itself
    # would report 0.0%, which reads as "no dilution" rather than "no history".
    has_history = len(dated_totals) > 1
    oldest_raw = dated_totals[0][1] if has_history else None
    oldest_adjusted = adjusted_totals[0][1] if has_history else None
    raw_change_pct = _pct_change(oldest_raw, latest_total)
    change_pct = _pct_change(oldest_adjusted, latest_total)
    direction = _direction(change_pct)

    if unexplained:
        # Never state a direction that cannot be supported. The raw figures
        # stay in the payload; what is withheld is the verdict.
        change_pct = None
        direction = UNDETERMINED

    adjustment.update({
        "adjusted": bool(applied),
        "splits_applied": [{"date": e["date"], "ratio": e["ratio"]}
                           for e in applied],
        "cumulative_factor": _factor_for(dated_totals[0][0] if dated_totals
                                         else None, applied),
        "basis": dated_totals[-1][0] if dated_totals else None,
        "raw_oldest_total": oldest_raw,
        "raw_latest_total": latest_total,
        "adjusted_oldest_total": oldest_adjusted,
        "raw_change_pct": raw_change_pct,
        "unexplained_jumps": unexplained,
    })

    return {
        "ticker": ticker,
        "success": True,
        "latest_total": latest_total,
        "latest_total_basis": share_basis["total_basis"],
        "share_basis": share_basis,
        "by_class": by_class,
        "classes_found": sorted(by_class.keys()),
        "by_class_change": _class_changes(by_class, undetermined=bool(unexplained)),
        "total_series": total_series,
        "change_pct": change_pct,
        "raw_change_pct": raw_change_pct,
        "direction": direction,
        "periods_examined": len(total_series),
        "split_adjusted": bool(applied),
        "split_adjustment": adjustment,
        "warnings": _split_warnings(ticker, adjustment, unexplained,
                                    source_error) + basis_warnings,
    }


def _class_changes(by_class: Dict[str, List[Dict[str, Any]]],
                   *, undetermined: bool) -> Dict[str, Dict[str, Any]]:
    """Per-class change over the window, on the same adjusted basis.

    `by_class` carries the same exposure the total did: a reader differencing
    the Class A rows by hand gets the split back. Every class of a split filer
    is restated, and classes diverge -- a buyback usually runs in one class
    only -- so one headline direction cannot stand in for all of them.

    Rows arrive newest first, matching the filing order they were built from.
    """
    changes: Dict[str, Dict[str, Any]] = {}
    for label, rows in by_class.items():
        newest, oldest = rows[0], rows[-1]
        raw = _pct_change(oldest["shares"], newest["shares"]) if len(rows) > 1 else None
        adjusted = (_pct_change(oldest["shares_split_adjusted"],
                                newest["shares_split_adjusted"])
                    if len(rows) > 1 else None)
        if undetermined:
            # The classes cannot be more certain than the series they came from.
            changes[label] = {"change_pct": None, "raw_change_pct": raw,
                              "direction": UNDETERMINED}
            continue
        changes[label] = {"change_pct": adjusted, "raw_change_pct": raw,
                          "direction": _direction(adjusted)}
    return changes


def _split_warnings(ticker: str, adjustment: Dict[str, Any],
                    unexplained: List[Dict[str, Any]],
                    source_error: Optional[str]) -> List[Dict[str, Any]]:
    """Warnings in the house shape, raised only when they apply.

    A caveat attached to the tool rather than to the answer trains a reader to
    skip the array, so none of these fires on a filer with no split near it.
    """
    def _shape(event: Dict[str, Any]) -> str:
        # A reverse split is a ratio below one, and "0.1-for-1" is not how
        # anybody says it.
        ratio = event["ratio"]
        name = f"{ratio:g}-for-1" if ratio >= 1.0 else f"1-for-{1.0 / ratio:g}"
        return f"{name} on {event['date']}"

    warnings: List[Dict[str, Any]] = []
    if adjustment["splits_applied"]:
        described = ", ".join(_shape(e) for e in adjustment["splits_applied"])
        warnings.append({
            "code": "split_adjusted",
            "message": (
                f"{ticker}: share counts before {adjustment['basis']} were "
                f"rebased for {described}. change_pct is split-adjusted; "
                f"raw_change_pct ({adjustment['raw_change_pct']:.2f}%) is the "
                f"unadjusted difference between cover pages."),
            "splits_applied": adjustment["splits_applied"],
        })
    if source_error is not None:
        warnings.append({
            "code": "split_source_unavailable",
            "message": (
                f"{ticker}: the split calendar could not be read "
                f"({source_error}), so no split could be confirmed or ruled "
                f"out for a window whose share count moved sharply."),
        })
    if unexplained:
        described = "; ".join(
            f"{j['implied_ratio']:.3g}x between {j['from_date']} and "
            f"{j['to_date']} (nearest split ratio {j['nearest_split_ratio']:g})"
            for j in unexplained)
        warnings.append({
            "code": UNDETERMINED,
            "message": (
                f"{ticker}: the share count jumps by {described}, which is a "
                f"split ratio rather than a plausible issuance, and no ratio "
                f"could be confirmed for it. No direction is stated; "
                f"raw_change_pct is the unadjusted figure and measures the "
                f"split, not dilution."),
            "unexplained_jumps": unexplained,
        })
    return warnings


def get_shelf_activity(ticker: str, lookback_days: int = 730) -> Dict[str, Any]:
    """Shelf registrations and takedowns in the window.

    An S-3 is the authorisation to sell shares; a 424B5 is an actual sale off
    that shelf. Neither form had any coverage in this codebase, which meant the
    mechanism of dilution was invisible even when its effect was not.
    """
    try:
        _require_identity()
        company = Company(ticker)
        cutoff = date.today() - timedelta(days=lookback_days)

        def _recent(form_name: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            try:
                filings = company.get_filings(form=form_name)
            except Exception:
                return out
            for filing in filings:
                filed = getattr(filing, "filing_date", None)
                if filed is None:
                    continue
                filed_date = filed if isinstance(filed, date) else None
                if filed_date is None:
                    try:
                        filed_date = date.fromisoformat(str(filed))
                    except ValueError:
                        continue
                if filed_date < cutoff:
                    break  # EDGAR returns newest first
                out.append({
                    "form": form_name,
                    "filing_date": str(filed_date),
                    "accession": str(getattr(filing, "accession_no", "")),
                })
            return out

        registrations = _recent("S-3")
        takedowns = _recent("424B5")
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {
            "ticker": ticker,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "s3_registrations": [],
            "b5_takedowns": [],
        }

    return {
        "ticker": ticker,
        "success": True,
        "lookback_days": lookback_days,
        "s3_registrations": registrations,
        "b5_takedowns": takedowns,
        "has_active_shelf": bool(registrations),
        "takedown_count": len(takedowns),
    }
