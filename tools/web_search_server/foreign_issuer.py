"""Foreign private issuers: which form, which taxonomy, which currency.

The SEC layer read 10-K and 10-Q and nothing else, so every ADR returned
silence. Measured before this module existed:

    get_debt_maturity_schedule("TSM")
      -> "TSM does not tag long-term debt maturities in its 10-K"
    get_geographic_revenue("TSM")
      -> "TSM does not disaggregate revenue by geography in its 10-K"

TSMC has no 10-K. Its 20-F disaggregates revenue across four regions and
discloses a full bond maturity ladder. Both answers were confident statements
about a filing that does not exist, and both read as findings about the
company rather than gaps in the tool. That is the failure this module exists
to end -- more than any figure it extracts.

Three things were probed live against EDGAR before any of it was written, and
two contradict the assumption you would start from.

**Taxonomy belongs to the filer, not the form.** The 20-F basket splits both
ways: TSM, SAP and NVO tag `ifrs-full`; ASML and BABA tag `us-gaap` in the
same form, and both say so in `dei:DocumentAccountingStandard`. Selecting
concepts by form would have missed two of five. Both chains are therefore
tried for every filer, ordered by what the filing actually declares.

**6-K carries no XBRL.** Not sparse -- absent. edgartools reports "No XBRL
attachments found" for every recent 6-K from TSM, ASML and BABA. A foreign
issuer's interim results exist only as an untagged exhibit, so there is no
quarterly tagged-data path for these filers at all. A 10-Q-shaped tool cannot
be made to work here and must say so instead of returning nothing.

**Status is the latest annual form, not the filing history.** Shopify filed
20-F in 2016 and 40-F through February 2024, then graduated to domestic
forms and now files 10-K and 10-Q. "Has ever filed 40-F" and "has a 10-K" are
both wrong tests.

Coverage measured live over the basket is reported by `get_annual_revenue`
per call rather than claimed here: revenue resolves for TSM, ASML, SAP, NVO
and BABA. Balance-sheet and cash-flow concepts under IFRS are a larger
mapping job and are deliberately not attempted -- an unmapped concept returns
"not covered", which is the honest answer, and never a US-GAAP tag's value
under an IFRS filer's name.
"""
from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .sec_series import (
    NotCovered,
    _period_rank,
    _require_identity,
    _throttle,
    currency_of,
    fetch_concept_series,
)
from .shared_filings import (
    Deadline,
    ToolTimeout,
    budget_seconds,
    concept_series,
    shared_filings,
)

# Ordered by how commonly they appear, which is also the order a mismatch
# message reads best in.
ANNUAL_FORMS = ("10-K", "20-F", "40-F")

# A foreign private issuer files 6-K in place of 10-Q. Verified live: 6-K
# exhibits carry no XBRL, so this mapping tells a caller what exists, not
# what can be read.
FOREIGN_ANNUAL_FORMS = ("20-F", "40-F")
INTERIM_FORM = {"10-K": "10-Q", "20-F": "6-K", "40-F": "6-K"}

IFRS_REVENUE_CONCEPTS = (
    # SAP, NVO and BCE carry the consolidated total here. TSM tags the same
    # element but every one of its facts is dimensioned by geography, so the
    # chain has to fall through on "no undimensioned fact" rather than on
    # "concept absent".
    "ifrs-full:Revenue",
    "ifrs-full:RevenueFromContractsWithCustomers",
    "ifrs-full:RevenueFromSaleOfGoods",
    "ifrs-full:RevenueFromRenderingOfServices",
)

US_GAAP_REVENUE_CONCEPTS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
)

_ACCOUNTING_STANDARD_CONCEPT = "dei:DocumentAccountingStandard"
_DEI_FIELDS = {
    "dei:EntityRegistrantName": "registrant_name",
    "dei:EntityIncorporationStateCountryCode": "incorporation_country_code",
    "dei:EntityAddressCountry": "address_country",
}

_index_cache: Dict[str, Dict[str, str]] = {}
_index_lock = threading.Lock()


def _reset_caches() -> None:
    """Drop the per-ticker forms cache. For tests; nothing else should call it."""
    with _index_lock:
        _index_cache.clear()


def _normalise_concept(value: Any) -> str:
    """Concept names arrive with either ':' or '_' separating the prefix."""
    return str(value or "").strip().replace("_", ":", 1)


class AnnualIndexUnavailable(RuntimeError):
    """EDGAR could not be asked which annual forms this filer uses.

    Distinct from an empty index, which means the filer files none of them.
    Kept separate so the caller reports an outage instead of a fact about the
    company.
    """


def _fetch_annual_filing_index(ticker: str) -> Dict[str, str]:
    """{form: date of its most recent filing} across 10-K, 20-F and 40-F.

    Costs the submissions index and nothing more -- no XBRL parse -- because
    `form_mismatch_note` runs on every failure path in the SEC layer and a
    filing download per annotation would not be affordable.

    A form the filer never used comes back as an empty list, not an exception.
    So an exception never meant "a form this filer never used" -- it meant the
    lookup broke -- and swallowing it answered a rate limit with "NVDA has no
    annual report on EDGAR".
    """
    from edgar import Company

    _require_identity()
    try:
        company = Company(ticker)
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        raise AnnualIndexUnavailable(
            f"could not look up {ticker} on EDGAR: "
            f"{type(exc).__name__}: {exc}") from exc

    index: Dict[str, str] = {}
    for form in ANNUAL_FORMS:
        _throttle()
        try:
            filings = company.get_filings(form=form, amendments=False).head(1)
        except Exception as exc:  # noqa: BLE001 - reported, never masked
            raise AnnualIndexUnavailable(
                f"could not list {form} filings for {ticker}: "
                f"{type(exc).__name__}: {exc}") from exc
        for filing in filings:
            index[form] = str(filing.filing_date)
    return index


def _annual_filing_index(ticker: str) -> Dict[str, str]:
    key = ticker.upper()
    with _index_lock:
        cached = _index_cache.get(key)
    if cached is not None:
        return cached
    # Raises rather than returning {} when EDGAR could not be reached, so a
    # failed lookup is never memoised and replayed as an answer.
    index = _fetch_annual_filing_index(key)
    with _index_lock:
        _index_cache[key] = index
    return index


def _latest_annual_form(ticker: str) -> Tuple[Optional[str], Dict[str, str]]:
    """The form this filer currently reports on, and the whole index.

    Decided by filing date rather than by precedence. Shopify has all three
    forms in its history and only the most recent one describes it today.
    """
    index = _annual_filing_index(ticker)
    if not index:
        return None, index
    return max(index.items(), key=lambda kv: kv[1])[0], index


def _annual_facts(ticker: str, form: str):
    """(filing_date, accession, facts dataframe) for the latest `form` filing.

    None when the filing has no parseable XBRL. CNI's 40-F is the live case:
    41 cover-page facts and no financial statements at all.
    """
    from edgar import Company

    _require_identity()
    company = Company(ticker)
    _throttle()
    filings = company.get_filings(form=form, amendments=False).head(1)
    for filing in filings:
        try:
            xbrl = filing.xbrl()
            if xbrl is None:
                return None
            frame = xbrl.facts.query().to_dataframe()
        except Exception:  # noqa: BLE001 - an unparseable filing is not a crash
            return None
        if frame is None or len(frame) == 0:
            return None
        return (str(filing.filing_date),
                str(getattr(filing, "accession_no", "")), frame)
    return None


def _dei_value(frame: Any, concept: str) -> Optional[str]:
    if "concept" not in getattr(frame, "columns", []):
        return None
    rows = frame[frame["concept"].map(lambda c: _normalise_concept(c) == concept)]
    for _, row in rows.iterrows():
        text = str(row.get("value") or "").strip()
        if text and text.lower() != "nan":
            return text
    return None


def _taxonomy_of(frame: Any) -> Optional[str]:
    """Whichever financial-reporting taxonomy carries the most facts.

    `dei` and the filer's own prefix are excluded: every filing has both, and
    a company-specific prefix outnumbering the standard one would say nothing
    about the accounting basis.
    """
    if "concept" not in getattr(frame, "columns", []):
        return None
    counts = Counter()
    for value in frame["concept"]:
        prefix = _normalise_concept(value).split(":", 1)[0]
        if prefix in ("ifrs-full", "us-gaap"):
            counts[prefix] += 1
    return counts.most_common(1)[0][0] if counts else None


def _currencies_of(frame: Any) -> Dict[str, int]:
    """Fact counts per currency across the whole filing.

    The reporting currency is the one nearly everything is tagged in; a
    convenience translation is a couple of hundred facts against several
    thousand. Non-monetary units resolve to None and are dropped.
    """
    if "unit_ref" not in getattr(frame, "columns", []):
        return {}
    counts = Counter()
    for value in frame["unit_ref"]:
        code = currency_of(value)
        if code:
            counts[code] += 1
    return dict(counts)


def get_foreign_filer_profile(ticker: str) -> Dict[str, Any]:
    """Which forms this filer uses, under which accounting standard, in which
    currency -- so a caller can tell a foreign issuer from an uncovered one.

    `is_foreign_private_issuer` is decided by the most recent annual form:
    20-F or 40-F means yes, 10-K means no. None means no annual filing was
    found at all, which is a lookup failure rather than an answer.

    `taxonomy` is read from the filing rather than inferred from the form,
    because the two do not agree: ASML and BABA file 20-F under US GAAP while
    TSM, SAP and NVO file it under IFRS.

    `interim_xbrl` false is the single most useful field here. A foreign
    issuer reports interim results on 6-K, and 6-K exhibits carry no XBRL, so
    no quarterly tagged figure exists for it anywhere.
    """
    try:
        form, index = _latest_annual_form(ticker)
    except AnnualIndexUnavailable as exc:
        return {
            "ticker": ticker,
            "success": False,
            "is_foreign_private_issuer": None,
            "error": f"EDGAR lookup failed for {ticker}: {exc}",
            "annual_form": None,
            "annual_filing_date": None,
        }
    if form is None:
        return {
            "ticker": ticker,
            "success": False,
            "is_foreign_private_issuer": None,
            "error": (f"{ticker}: no 10-K, 20-F or 40-F filing found on EDGAR. "
                      f"Either the ticker does not map to an SEC registrant, or "
                      f"it files none of the three annual forms."),
            "annual_form": None,
            "annual_filing_date": None,
        }

    is_foreign = form in FOREIGN_ANNUAL_FORMS
    former = {f: d for f, d in index.items() if f != form}

    profile: Dict[str, Any] = {
        "ticker": ticker,
        "success": True,
        "is_foreign_private_issuer": is_foreign,
        "annual_form": form,
        "annual_filing_date": index[form],
        "annual_accession": None,
        "interim_form": INTERIM_FORM[form],
        "interim_xbrl": form == "10-K",
        "taxonomy": None,
        "accounting_standard": None,
        "accounting_standard_source": None,
        "reporting_currency": None,
        "currencies_present": {},
        "usd_convenience_translation": False,
        "registrant_name": None,
        "incorporation_country_code": None,
        "address_country": None,
        "former_annual_forms": former,
    }

    notes: List[str] = []
    if is_foreign:
        notes.append(
            f"{ticker} is a foreign private issuer: it files {form} annually "
            f"and {INTERIM_FORM[form]} for interim reports. 6-K exhibits carry "
            f"no XBRL, so no quarterly tagged figures exist for this filer -- "
            f"a 10-Q-based tool cannot be made to work on it.")
    if former:
        notes.append(
            "It previously filed " +
            ", ".join(f"{f} (latest {d})" for f, d in sorted(former.items())) +
            ", so filings older than that are on a different form.")

    facts = None
    try:
        facts = _annual_facts(ticker, form)
    except Exception:  # noqa: BLE001 - the form answer is useful without XBRL
        facts = None

    if facts is None:
        notes.append(
            f"The latest {form} has no parseable XBRL, so taxonomy and "
            f"reporting currency could not be read. Some 40-F filers tag only "
            f"the cover page and file their statements as an untagged exhibit.")
        profile["note"] = " ".join(notes)
        return profile

    filing_date, accession, frame = facts
    profile["annual_filing_date"] = filing_date
    profile["annual_accession"] = accession

    profile["taxonomy"] = _taxonomy_of(frame)
    declared = _dei_value(frame, _ACCOUNTING_STANDARD_CONCEPT)
    if declared:
        profile["accounting_standard"] = declared
        profile["accounting_standard_source"] = _ACCOUNTING_STANDARD_CONCEPT
    elif profile["taxonomy"]:
        profile["accounting_standard"] = (
            "International Financial Reporting Standards"
            if profile["taxonomy"] == "ifrs-full" else "U.S. GAAP")
        profile["accounting_standard_source"] = "concept prefixes"

    currencies = _currencies_of(frame)
    profile["currencies_present"] = currencies
    if currencies:
        reporting = max(currencies.items(),
                        key=lambda kv: (kv[1], kv[0] != "USD"))[0]
        profile["reporting_currency"] = reporting
        profile["usd_convenience_translation"] = (
            reporting != "USD" and "USD" in currencies)

    for concept, key in _DEI_FIELDS.items():
        profile[key] = _dei_value(frame, concept)

    if profile["reporting_currency"] and profile["reporting_currency"] != "USD":
        notes.append(
            f"Figures from this filer are denominated in "
            f"{profile['reporting_currency']}, not dollars.")
    if profile["usd_convenience_translation"]:
        notes.append(
            "It also tags a US-dollar convenience translation of the most "
            "recent year, at its own chosen rate. That is the filer's number, "
            "not a live conversion, and it covers only the latest period.")
    profile["note"] = " ".join(notes) if notes else (
        f"{ticker} is a domestic filer: 10-K annually, 10-Q quarterly, "
        f"us-gaap concepts in USD. The standard SEC tools apply unchanged.")
    return profile


def form_mismatch_note(ticker: str, form_type: str) -> Optional[str]:
    """Why a tool found nothing, when the reason is the form it read.

    Returns None when there is no mismatch, which is the overwhelming majority
    of calls -- a guard that speaks up on the normal path is noise. Callers
    put the returned string in place of their own "not covered" message, since
    that message is the misleading one.

    Never raises. This annotates an error path, so failing to annotate has to
    be survivable: an EDGAR timeout here must not convert a partial answer
    into no answer.
    """
    try:
        form, index = _latest_annual_form(ticker)
    except Exception:  # noqa: BLE001 - see docstring
        return None
    if form is None:
        return None

    requested = str(form_type or "").strip().upper()
    is_foreign = form in FOREIGN_ANNUAL_FORMS

    if requested in ("10-Q", "6-K"):
        if not is_foreign:
            return None
        return (
            f"{ticker} is a foreign private issuer: it files {form} annually "
            f"(latest {index[form]}) and reports interim results on 6-K, not "
            f"10-Q. 6-K exhibits carry no XBRL, so there are no quarterly "
            f"tagged figures for this filer to read -- this is an absence of "
            f"tagged data, not an absence of results. Use the annual {form} "
            f"instead, and call get_foreign_filer_profile('{ticker}') for the "
            f"taxonomy and reporting currency it uses.")

    if requested not in ANNUAL_FORMS or requested == form:
        return None

    former = f" It last filed {requested} on {index[requested]}." if requested in index else ""
    subject = ("is a foreign private issuer and" if is_foreign
               else "is a domestic filer and")
    return (
        f"{ticker} {subject} files {form} annually (latest {index[form]}), not "
        f"{requested}. This tool read {requested} filings and found none, which "
        f"is a gap in the tool rather than a finding about the company.{former} "
        f"Re-run with form='{form}', and call "
        f"get_foreign_filer_profile('{ticker}') for the taxonomy and reporting "
        f"currency its filings use.")


def not_covered_reason(ticker: str, form_type: str, fallback: str) -> str:
    """The honest reason a tool found nothing, given who this filer is.

    Three cases, in order. A form mismatch is the loudest and gets the whole
    message. Otherwise, a foreign private issuer asked on its own form still
    warrants a caveat, because every concept chain in this package outside
    this module is us-gaap: an IFRS filer does not tag those elements at all,
    which is a gap in the tool and reads exactly like a company that chose
    not to disclose. Otherwise the caller's own message stands -- Ford tagging
    no debt maturities is a real finding about Ford.

    Never raises; a failure to explain must not become a second failure.
    """
    try:
        mismatch = form_mismatch_note(ticker, form_type)
        if mismatch:
            return mismatch
        form, _ = _latest_annual_form(ticker)
        if form not in FOREIGN_ANNUAL_FORMS:
            return fallback
        return (
            f"{fallback} NOTE: {ticker} is a foreign private issuer filing "
            f"{form}, and this tool reads us-gaap concepts. Roughly half of "
            f"20-F filers report under IFRS instead (TSM, SAP and NVO do; "
            f"ASML and BABA do not), and an IFRS filer does not tag us-gaap "
            f"elements at all -- so an empty result here may be this tool's "
            f"limit rather than the company's silence. Call "
            f"get_foreign_filer_profile('{ticker}') for the taxonomy before "
            f"concluding anything.")
    except Exception:  # noqa: BLE001 - see docstring
        return fallback


def _shared_filings(ticker: str, deadline: Deadline):
    """`shared_filings`, bound to this module's `fetch_concept_series`.

    Read here rather than passed in from the call site so the name stays a
    module global a test can replace -- which is what the walk checks before
    it stands down.
    """
    return shared_filings(ticker, deadline, fetch_concept_series)


def _concept_series(ticker: str, concept: str, form: str, limit: int) -> list:
    """One concept's series, reusing filings already parsed for this call.

    With no shared walk open this is `fetch_concept_series` verbatim, which is
    what keeps that name the seam the tests replace.
    """
    return concept_series(ticker, concept, form, limit, fetch_concept_series)


def _revenue_chains(taxonomy: Optional[str]) -> Tuple[Tuple[str, ...], ...]:
    """Both concept chains, best guess first.

    Both are always tried. A 20-F filer may be on either basis, and guessing
    from the form gets ASML and BABA wrong.
    """
    ifrs = ("ifrs-full", IFRS_REVENUE_CONCEPTS)
    gaap = ("us-gaap", US_GAAP_REVENUE_CONCEPTS)
    return (ifrs, gaap) if taxonomy == "ifrs-full" else (gaap, ifrs)


def get_annual_revenue(ticker: str, limit: int = 3,
                       taxonomy_hint: Optional[str] = None) -> Dict[str, Any]:
    """Consolidated annual revenue in the currency it was reported in.

    Works for domestic and foreign filers alike: the annual form is resolved
    from EDGAR (10-K, 20-F or 40-F), and both the IFRS and US-GAAP concept
    chains are tried because 20-F filers use both.

    `currency` is never assumed. TSM reports NT$3.8 trillion and BABA RMB
    1.02 trillion; read without the code attached, either is a wildly
    different company. `latest_revenue_usd` is populated only when the filer
    itself tagged a US-dollar convenience translation -- at its own rate, for
    the latest period only -- or when it reports in dollars to begin with;
    `usd_is_filer_translation` distinguishes the two. It is never computed
    here, because a rate this module picked would look exactly like a
    disclosed one.

    A chain member is skipped when it has no *undimensioned* fact, not merely
    when it is absent: TSM tags `ifrs-full:Revenue` six times and every one is
    a geography, so taking the concept's presence as an answer reports
    NT$352bn against a real NT$3,809bn.
    """
    try:
        form, index = _latest_annual_form(ticker)
    except AnnualIndexUnavailable as exc:
        return {
            "ticker": ticker, "success": False,
            "error": f"EDGAR lookup failed for {ticker}: {exc}",
            "latest_revenue": None, "currency": None, "form": None,
            "series": [], "concept_used": None, "concepts_tried": [],
        }
    if form is None:
        return {
            "ticker": ticker, "success": False,
            "error": (f"{ticker}: no 10-K, 20-F or 40-F filing found on EDGAR, "
                      f"so there is no annual report to read revenue from."),
            "latest_revenue": None, "currency": None, "form": None,
            "series": [], "concept_used": None, "concepts_tried": [],
        }

    if taxonomy_hint is None and form != "10-K":
        # Only worth an XBRL parse for a form that could be either. A domestic
        # registrant must report under US GAAP, so a 10-K needs no probe and
        # the common path stays one filing fetch rather than two.
        try:
            taxonomy_hint = get_foreign_filer_profile(ticker).get("taxonomy")
        except Exception:  # noqa: BLE001 - chain ordering is an optimisation
            taxonomy_hint = None

    tried: List[str] = []
    deadline = Deadline(budget_seconds(), f"get_annual_revenue({ticker})")
    try:
        with _shared_filings(ticker, deadline):
            answer = _revenue_chain_walk(ticker, form, index, taxonomy_hint,
                                         limit, tried)
    except ToolTimeout as exc:
        return {
            "ticker": ticker,
            "success": False,
            "timed_out": True,
            "form": form,
            "annual_filing_date": index.get(form),
            "error": str(exc),
            "latest_revenue": None, "currency": None, "series": [],
            "concept_used": None, "concepts_tried": tried,
        }
    if answer is not None:
        return answer

    return {
        "ticker": ticker,
        "success": False,
        "form": form,
        "annual_filing_date": index[form],
        "error": (f"{ticker}: no consolidated revenue concept found in the last "
                  f"{limit} {form} filings. Both the IFRS and US-GAAP chains "
                  f"were tried and neither produced an undimensioned fact. "
                  f"This filer tags revenue under an element not in either "
                  f"chain, or only with dimensions."),
        "latest_revenue": None, "currency": None, "series": [],
        "concept_used": None, "concepts_tried": tried,
    }


def _revenue_chain_walk(ticker: str, form: str, index: Dict[str, str],
                        taxonomy_hint: Optional[str], limit: int,
                        tried: List[str]) -> Optional[Dict[str, Any]]:
    """The first chain element answering in the most recent filing, or None.

    Split out so the shared filing walk above wraps the fetching and nothing
    else. Up to seven elements are asked of the same `limit` filings; parsing
    them once per element cost GS 21 parses of 3 filings and 46.4 seconds.

    `tried` is appended to rather than returned because the caller needs it
    whichever way this ends -- including when the clock stops the walk, where
    "which elements were asked" is the only honest thing left to report.
    """
    for taxonomy, concepts in _revenue_chains(taxonomy_hint):
        for concept in concepts:
            tried.append(concept)
            try:
                points = _concept_series(ticker, concept, form, limit)
            # Only NotCovered is swallowed, and only to try the next concept.
            # A network failure or an unknown ticker propagates: reporting an
            # outage as "this filer does not disclose it" is the one answer
            # worse than an error.
            except NotCovered:
                continue
            # The clock says nothing about the filer, so it is not turned into
            # a message about one here. The caller reports it in its own words.
            except ToolTimeout:
                raise
            except Exception as exc:  # noqa: BLE001 - surface it, never mask it
                return {
                    "ticker": ticker,
                    "success": False,
                    "form": form,
                    "annual_filing_date": index.get(form),
                    "error": f"fetching revenue concepts failed: {exc}",
                    "latest_revenue": None, "currency": None, "series": [],
                    "concept_used": None, "concepts_tried": tried,
                }

            series: List[Dict[str, Any]] = []
            covers_latest_filing = False
            for point in points:
                fact = point.latest_undimensioned()
                if fact is None:
                    continue
                if point.filing_date == index[form]:
                    covers_latest_filing = True
                usd = point.latest_undimensioned(currency="USD")
                series.append({
                    "filing_date": point.filing_date,
                    "period": fact.period,
                    "value": fact.value,
                    "currency": fact.currency,
                    "value_usd": usd.value if usd is not None else None,
                })
            # A concept answers only if it answers in the *most recent* annual
            # filing. TSM tagged ifrs-full:Revenue undimensioned in its 2024
            # and 2025 20-Fs and stopped in the 2026 one, where all six facts
            # are geographies. Accepting "the chain produced rows" reported
            # FY2024's NT$2,894bn as the latest revenue -- a full year stale
            # and 24% low, with the right currency and a plausible face.
            if not series or not covers_latest_filing:
                continue

            series.sort(key=lambda row: _period_rank(row["period"]), reverse=True)
            latest = series[0]
            currency = latest["currency"]
            reports_in_usd = currency == "USD"
            return {
                "ticker": ticker,
                "success": True,
                "form": form,
                "annual_filing_date": index[form],
                "taxonomy_used": taxonomy,
                "concept_used": concept,
                "concepts_tried": tried,
                "currency": currency,
                "latest_revenue": latest["value"],
                "latest_period": latest["period"],
                "latest_revenue_usd": (latest["value"] if reports_in_usd
                                       else latest["value_usd"]),
                "usd_is_filer_translation": (not reports_in_usd
                                             and latest["value_usd"] is not None),
                "series": series,
                "note": (
                    f"Revenue is denominated in {currency or 'an untagged unit'}. "
                    f"latest_revenue_usd, when present, is the filer's own "
                    f"convenience translation at its own rate for the latest "
                    f"period only -- never a live conversion, and null does not "
                    f"mean the company is small."),
            }
    return None
