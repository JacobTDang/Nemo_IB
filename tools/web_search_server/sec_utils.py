from typing import Any, Dict, List, Optional
from edgar import Company
from edgar.xbrl import XBRL
from collections import OrderedDict
import pandas as pd
import logging
import os
import re
import sys
import threading
# useful documentation for edgartools xbrl: https://edgartools.readthedocs.io/en/latest/getting-xbrl/

# SEC identity is resolved on use by sec_series._require_identity, which
# refuses to invent a contact address. Defaulting it here misrepresented the
# caller to the SEC on every request.
from tools.web_search_server.sec_series import _require_identity

# Structured warnings, built with the same helper the MCP dispatcher validates
# against, so a caveat raised inside an extractor survives annotation instead
# of being dropped or rejected for shape.
from tools.response_meta import warning

# Single-flight LRU cache for SEC filing fetches. The previous plain-dict cache
# had a check-then-fill race: N concurrent threads asking for the same
# (ticker, form_type) all saw "miss" and all hit SEC EDGAR in parallel, which
# triggers rate limiting on heavy 10-Ks (AMD, MSFT). Now a per-key lock
# serializes concurrent callers — only one thread fetches, the rest block and
# return the cached result. OrderedDict bounds growth across long sessions.
logger = logging.getLogger(__name__)

_FILING_CACHE_MAX = 32
_filing_cache_lru: "OrderedDict[tuple, Any]" = OrderedDict()
_filing_key_locks: Dict[tuple, threading.Lock] = {}
_filing_locks_master = threading.Lock()


def _get_filing_lock(key: tuple) -> threading.Lock:
  """Return (creating if needed) the per-key lock for a cache key. The master
  lock is held only for the dict insert, so it never blocks an SEC fetch."""
  with _filing_locks_master:
    lock = _filing_key_locks.get(key)
    if lock is None:
      lock = threading.Lock()
      _filing_key_locks[key] = lock
    return lock


# Revenue elements, broadest first. Order is load-bearing, and the previous
# order -- ASC 606 first -- was wrong wherever a filer earns anything outside a
# customer contract:
#
#   AMT   reported   935,900,000 against 10,644,600,000. Tower rents are lease
#         income under ASC 842; AMT labels the ASC 606 element "Total non-lease
#         revenue" and it is 8.8% of revenue.
#   WFC   reported 10,498,000,000 against 83,699,000,000 -- the fact WFC labels
#         "Fee income".
#   WMT   reported 706,413,000,000 against 713,163,000,000.
#   GE    reported 30,163,000,000 against 45,855,000,000.
#
# `RevenuesNetOfInterestExpense` is the total-revenue line on a bank's income
# statement and is the only one GS and WFC tag undimensioned; it used to be
# reached by accident, through the prefix match on `us-gaap:Revenues`, and
# reported under that name. GS does not tag `us-gaap:Revenues` at all.
#
# Reordering alone is not enough and makes two filers worse on its own: the
# largest `us-gaap:Revenues` fact is 452.209bn for XOM against a consolidated
# 332.238bn, and 48.024bn for GE against 45.855bn. It only works because
# `filter_annual_data` now returns the undimensioned fact rather than the
# largest one.
#
# The unprefixed spellings that used to trail this list ('Revenues', 'Revenue',
# 'TotalRevenues', 'SalesRevenueNet') are gone. No filing tags them; they only
# ever matched through the prefix search, which no longer answers.
REVENUE_CONCEPTS = (
  'us-gaap:Revenues',
  'us-gaap:RevenuesNetOfInterestExpense',
  'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
  'us-gaap:SalesRevenueNet',
)

# Depreciation and amortisation, combined totals first. The bare
# `us-gaap:Depreciation` used to sit second and reached the combined elements
# by prefix, so it looked like the right answer while naming the wrong element.
# Matched exactly it returns depreciation only: AMT 1,100,000,000 against
# 2,041,600,000 of depreciation, amortisation and accretion, CHTR
# 8,100,000,000 against 8,711,000,000, WFC 1,700,000,000 against 7,713,000,000.
# It stays last, for the filers -- MSFT, GOOGL -- that tag nothing else.
DA_CONCEPTS = (
  'us-gaap:DepreciationDepletionAndAmortization',
  'us-gaap:DepreciationAmortizationAndAccretionNet',
  'us-gaap:DepreciationAndAmortization',
  'us-gaap:DepreciationDepletionAndAmortizationExcludingAmortizationOfDebtIssuanceCosts',
  'us-gaap:Depreciation',
)

# Operating income, for EBITDA. One element, and it stays one element.
#
# A bare `OperatingIncomeLoss` and a bare `IncomeLossFromContinuingOperations`
# used to trail this chain, and the prefix match answered both with whatever
# longer element a filer happened to tag: pre-tax income for JPM
# (72,595,000,000) and XOM (45,969,000,000), and for GS
# `gs:GeographicReportingInformationPercentageOfOperatingIncomeLoss` -- four
# facts of 0.69, 0.23, 0.08 and 1.00, of which `idxmax` took the 1.00 and
# reported it as Goldman's operating income in dollars.
#
# Eleven of the 32 filers in the audit basket tag nothing here: JPM, BAC, WFC,
# GS, O, BIIB, XOM, CVX, GE, FOXA, LEN. Every alternative was measured against
# their live filings on 2026-08-24 and none is operating income:
#
#   * `us-gaap:OperatingIncomeLossIncludingEquityMethodInvestments` and
#     `us-gaap:GrossProfit` are tagged by none of the eleven.
#   * `...BeforeIncomeTaxes...` is pre-tax income. It is what the prefix match
#     used to return, and it is the defect, not the fix.
#   * Revenue less `us-gaap:CostsAndExpenses` is not operating income either,
#     and the filings disagree about why. O's expense total carries its
#     financing interest, tagged `us-gaap:InterestExpenseOperating`
#     (1,134,879,000). BIIB nests `us-gaap:NonoperatingIncomeExpense`
#     (-305,600,000) inside it, where the sign the presentation applies is not
#     recoverable from the fact. XOM's non-operating income is inside
#     `us-gaap:Revenues` with no element of its own to remove it by. GE is the
#     case that settles it: 45,855 - 37,342 + 1,487 reconciles to GE's tagged
#     pre-tax income of 10,000,000,000 exactly, and the 8,513,000,000 in the
#     middle is still struck after `ge:InterestAndOtherFinancialCharges`
#     (843,000,000) and a non-operating pension credit (-788,000,000). A
#     reconciling build-up is not a correct one.
#   * O tags no FFO or AFFO concept anywhere in its 10-K -- 672 distinct
#     concepts, none of them NAREIT. The REIT measure is in the earnings
#     release, not the filing's XBRL.
#
# So the eleven are refused by name of what was tried. See
# `_ebitda_not_covered`.
OPERATING_INCOME_CONCEPTS = (
  'us-gaap:OperatingIncomeLoss',
)

# Elements that appear on a bank's or broker-dealer's income statement and
# essentially nowhere else. Used only to pick which refusal to write, never to
# choose or alter a number, so a false positive costs wording rather than
# accuracy. All four banks in the basket tag both.
BANK_INCOME_STATEMENT_CONCEPTS = (
  'us-gaap:InterestIncomeExpenseNet',
  'us-gaap:NoninterestExpense',
)

TAX_EXPENSE_CONCEPTS = (
  'us-gaap:IncomeTaxExpenseBenefit',
  'us-gaap:ProvisionForIncomeTaxes',
  'us-gaap:CurrentIncomeTaxExpense',
  'us-gaap:IncomeTaxesPaid',
)

# Pre-tax income. AMZN, CAT and CVX tag only the
# MinorityInterestAndIncomeLossFromEquityMethodInvestments variant --
# 97,311,000,000, 11,541,000,000 and 19,743,000,000 -- which the prefix match
# used to reach from the shorter name beside it.
PRETAX_INCOME_CONCEPTS = (
  'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
  'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
  'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
  'us-gaap:EarningsBeforeIncomeTaxes',
)

OPERATING_CASH_FLOW_CONCEPTS = (
  'us-gaap:NetCashProvidedByUsedInOperatingActivities',
  'us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
)

# Capital expenditure, in the order a filer is most likely to mean it. The
# first two are the whole chain this project shipped with, and between them
# they miss what AMZN, T, NVDA, CVX, HD, REGN and SPG tag
# (PaymentsToAcquireProductiveAssets) and what PLD tags
# (PaymentsToDevelopRealEstateAssets).
#
# PaymentsToAcquireRealEstate and PaymentsToAcquireCommercialRealEstate are
# deliberately absent: buying a finished building is closer to an acquisition
# than to capital expenditure, and whether a REIT's property purchases belong
# in free cash flow is a judgement call rather than an identity.
CAPEX_CONCEPTS = (
  'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
  'us-gaap:PaymentsForCapitalImprovements',
  'us-gaap:PaymentsToAcquireProductiveAssets',
  'us-gaap:PaymentsToAcquireOtherProductiveAssets',
  'us-gaap:PaymentsToDevelopRealEstateAssets',
)

# Period length, in days, that each form's primary reporting period runs for.
# A fiscal year is 364 or 365 days and a 52/53-week retailer's runs 371, so the
# annual floor sits below the shortest of those and above any half-year. A
# quarter is 89-92 days; the window has to exclude the year-to-date duration a
# 10-Q carries beside it, which ends on the same day.
_PERIOD_SPAN_DAYS = {
  '10-K': (350, None),
  '20-F': (350, None),
  '40-F': (350, None),
  '10-Q': (80, 95),
}


def _consolidated_fact(xbrl, concept: str, span_days=None):
  """The undimensioned fact for `concept`, or None if the filer tags none.

  The single selection path behind every reader in this module. It exists
  because the two it replaces were wrong in the same two ways, and both ways
  produce a plausible number rather than a crash:

  1. `xbrl.facts.query().by_concept(name)` matches by **prefix**. Asking for
     `us-gaap:Assets` also returns `us-gaap:AssetsCurrent`, and asking for
     `us-gaap:Revenues` also returns `us-gaap:RevenuesNetOfInterestExpense`.
     Goldman does not tag `us-gaap:Revenues` at all, yet the old path reported
     its net revenues under that name -- a caller reconciling against the
     filing would find nothing there.
  2. Ties were broken with `idxmax`, on the stated assumption that the
     consolidated total is the largest fact for the period. It is not. A
     segment aggregate is struck before intersegment eliminations (CVX:
     231.4bn against 189.0bn), an unconsolidated joint venture's revenue is
     not the filer's revenue at all (SPG: 12.5bn against 6.4bn), and where the
     consolidated figure is negative the largest fact is the parent-company-
     only Schedule I figure, so the sign flips (JPM's operating cash flow:
     +44.5bn against -147.8bn).

  `sec_series` already solved both -- exact-concept filtering in
  `concept_point`, consolidated selection by *absence of dimensions* in
  `FilingPoint.undimensioned()` -- and every tool built on it reconciled
  cleanly in both audit sweeps. This routes through that rather than adding a
  third mechanism.

  None means "this filer does not tag this concept undimensioned", which is
  what lets a caller's concept chain move on to the element that it does tag.
  It is never a reason to substitute a dimensioned fact.
  """
  if xbrl is None:
    return None
  from .sec_series import concept_point
  try:
    point = concept_point(xbrl, concept, filing_date='', form='')
  except Exception:
    return None
  if point is None:
    return None
  return point.latest_undimensioned(span_days=span_days)


def _fact_result(fact, concept: str) -> Optional[Dict[str, Any]]:
  """A selected fact in the shape every caller in this module reads.

  `concept_used` names the element the value is actually tagged under rather
  than the one that was asked for. Those differed whenever the prefix match
  answered, and provenance that names an absent element is wrong even when the
  number beside it is right.
  """
  if fact is None:
    return None
  from .sec_series import _period_rank
  period_end, duration_days = _period_rank(fact.period)
  return {
    'value': fact.value,
    'concept_used': fact.concept or concept,
    'period_end': period_end,
    'duration_days': duration_days,
    # Currency matters the moment a foreign filer is in scope: TSM tags TWD,
    # SAP and ASML EUR, NVO DKK, BABA CNY. None means the unit was not a plain
    # amount of money, or was not tagged at all.
    'currency': fact.currency,
  }


def filter_instant_data(xbrl, concept: str) -> Optional[Dict[str, Any]]:
  """The consolidated instant fact for a concept -- balance sheet items.

  Balance sheet items are tagged as an instant rather than a duration, so the
  span window is exactly zero days. See `_consolidated_fact` for why selection
  is by absence of dimensions rather than by size.
  """
  return _fact_result(_consolidated_fact(xbrl, concept, span_days=(0, 0)),
                      concept)


def filter_annual_data(xbrl, concept: str, form_type: str = '10-K') -> Optional[Dict[str, Any]]:
  """The consolidated fact for a concept over the period `form_type` implies.

  - 10-K / 20-F / 40-F: the fiscal year (350+ days)
  - 10-Q: the quarter (80-95 days)
  - Others: the most recent period of any length

  20-F and 40-F are the annual reports of foreign private issuers and carry
  the same 12-month durations as a 10-K.

  Returns None when the filer tags no undimensioned fact for this concept in
  the window. That is a coverage fact the caller needs, not a reason to hand
  back a segment or a parent-company-only figure -- see `_consolidated_fact`.
  """
  span = _PERIOD_SPAN_DAYS.get(str(form_type).strip().upper())
  return _fact_result(_consolidated_fact(xbrl, concept, span_days=span),
                      concept)

def _filing_miss(ticker: str, form_type: str, fallback: str) -> str:
  """Say "this filer uses a different form" when that is why nothing was found.

  Every function here defaults to form_type='10-K'. A foreign private issuer
  has no 10-K at all, so `get_latest_filing` returned None and the caller
  reported "No 10-K filing found for TSM" -- which reads as a fact about
  Taiwan Semiconductor rather than about this tool. TSMC files 20-F.

  Falls back to the caller's own message when there is no mismatch, and the
  guard never raises, so annotating an error can never replace one error with
  another.
  """
  try:
    from .foreign_issuer import form_mismatch_note
    return form_mismatch_note(ticker, form_type) or fallback
  except Exception:  # noqa: BLE001 - an annotation must not become the failure
    return fallback


def get_latest_filing(ticker: str, form_type: str = '10-K') -> Optional[Dict[str, Any]]:
  """Get latest SEC filing with XBRL data.

  Single-flight LRU cache: when N tools call this concurrently for the same
  (ticker, form_type), only one thread performs the SEC download; the rest
  block on a per-key lock and return the cached result. Cache is bounded to
  _FILING_CACHE_MAX entries (LRU eviction) so long deep-research sessions
  don't grow memory indefinitely.

  An absence is cached -- a company that files no 10-K will not start doing
  so mid-session, and re-asking on every call is the retry storm this cache
  exists to prevent. A *failure* is not: a 503 or a dropped connection says
  nothing about the filer, and caching it told every later caller the filing
  did not exist until the process restarted.
  """
  cache_key = (ticker.upper(), form_type)

  # Fast path: cache hit, no lock acquisition needed.
  if cache_key in _filing_cache_lru:
    _filing_cache_lru.move_to_end(cache_key)
    return _filing_cache_lru[cache_key]

  lock = _get_filing_lock(cache_key)
  with lock:
    # Re-check inside the lock — a peer thread may have filled the cache
    # while we were waiting on the lock.
    if cache_key in _filing_cache_lru:
      _filing_cache_lru.move_to_end(cache_key)
      return _filing_cache_lru[cache_key]

    _require_identity()

    result = None
    cacheable = True
    try:
      company = Company(ticker)
      # An amendment carries whatever the filer is correcting, which is often
      # Part III proxy information and no financial statements. Altria's most
      # recent 10-K is such a 10-K/A, and it made get_revenue_base report that
      # Altria does not tag revenue. fetch_concept_series already excludes
      # them; this is the other path into EDGAR, and it feeds ten tools.
      filings = company.get_filings(form=form_type, amendments=False)

      if filings:
        latest_filing = filings[0]
        try:
          xbrl_data = latest_filing.xbrl()
        except Exception as exc:  # noqa: BLE001 - reason is logged, not swallowed
          # Cached, this is worse than a failed fetch: the filing looks
          # present and every caller reads xbrl_data as None, i.e. "this
          # company tags nothing". Retry instead.
          logger.warning("XBRL fetch failed for %s %s (%s): %s",
                         ticker, form_type, latest_filing.accession_number, exc)
          xbrl_data = None
          cacheable = False

        url = None
        for attr in ['filing_url', 'url', 'filing_details_url', 'document_url']:
          if hasattr(latest_filing, attr):
            url = getattr(latest_filing, attr)
            break

        result = {
          'filing_date': latest_filing.filing_date,
          'url': url,
          'accession_number': latest_filing.accession_number,
          'filing_object': latest_filing,
          'xbrl_data': xbrl_data,
        }
    except Exception as exc:  # noqa: BLE001 - reason is logged, not swallowed
      logger.warning("SEC filing fetch failed for %s %s: %s",
                     ticker, form_type, exc)
      result = None
      cacheable = False

    # A company that files no 10-K is a fact and is worth caching; a 503 is
    # not. Caching the 503 answered every later caller with "no such filing"
    # for the life of the process, which for a long-running server means
    # until someone restarts it.
    if cacheable:
      _filing_cache_lru[cache_key] = result
      # LRU eviction: drop oldest entries (and their per-key locks) above cap.
      while len(_filing_cache_lru) > _FILING_CACHE_MAX:
        evicted_key, _ = _filing_cache_lru.popitem(last=False)
        with _filing_locks_master:
          _filing_key_locks.pop(evicted_key, None)

    return result

def get_disclosures_names(ticker:str, form_type: str = '10-K') -> Dict[str, Any]:
  # get the disclosure name for agent to use
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if filing_data and filing_data['xbrl_data']:
      xbrl=filing_data['xbrl_data']
      disclosures = []
      try:

        statements = xbrl.statements

        # Get all disclosure statements
        disclosure_statements = statements.disclosures()

        for disclosure in disclosure_statements:
          # Get role/type and clean it up for readability
          if hasattr(disclosure, 'role_or_type'):
            role = disclosure.role_or_type
            # Extract the disclosure name from the URL
            if '/' in role:
              disclosure_name = role.split('/')[-1]
            else:
              disclosure_name = role
            disclosures.append(disclosure_name)

        if disclosures:
          return {
            'ticker': ticker,
            'success': True,
            'error': None,
            'disclosure_names': disclosures
          }
        else:
          return{
            'ticker': ticker,
            'success': False,
            'error': f'Unable to find any disclosure in {ticker} statements',
            'disclosure_names': None
          }

      except Exception as e:
        return{
          'ticker': ticker,
          'success': False,
          'error': f"Unable to find disclosure in disclosure concepts for {ticker}: {str(e)}",
          'disclosure_names': None
        }


  except Exception as e:
    return {
      'ticker': ticker,
      'success': False,
      'error': f'Unable to get disclosures for {ticker}: {str(e)}',
      'disclosure_names': None
    }
  return {
    'ticker': ticker,
    'success': False,
    'error': _filing_miss(ticker, form_type,
                          f'Unable to get disclosures for {ticker}'),
    'disclosure_names': None
  }

def extract_disclosure_data(ticker: str, disclosure_name: str, form_type: str = '10-K') -> Dict[str, Any]:

  try:
    latest_filing = get_latest_filing(ticker, form_type)
    if latest_filing and latest_filing['xbrl_data']:
      xbrl = latest_filing['xbrl_data']

      try:
        statement = xbrl.statements
        disclosures =  statement.disclosures()

        # Find the specific disclosure by name
        target_disclosure = None
        available_names = []
        for disclosure in disclosures:
          if hasattr(disclosure, 'role_or_type'):
            role = disclosure.role_or_type
            # Extract the disclosure name from the URL
            if '/' in role:
              current_name = role.split('/')[-1]
            else:
              current_name = role

            available_names.append(current_name)
            if current_name == disclosure_name:
              target_disclosure = disclosure
              break

        if target_disclosure:
          print(f'Found disclosure: {disclosure_name}', file=sys.stderr, flush=True)

          # Get summary info about the disclosure
          disclosure_summary = {
            'name': disclosure_name,
            'role_or_type': target_disclosure.role_or_type if hasattr(target_disclosure, 'role_or_type') else None,
            'primary_concept': target_disclosure.primary_concept if hasattr(target_disclosure, 'primary_concept') else None,
            'success': True
          }

          # Try to get DataFrame but filter out text-heavy data
          if hasattr(target_disclosure, 'to_dataframe'):
            try:
              df = target_disclosure.to_dataframe()
              print(f'DataFrame shape: {df.shape if df is not None else None}', file=sys.stderr, flush=True)

              if df is not None and not df.empty:
                disclosure_summary['data_shape'] = df.shape
                disclosure_summary['columns'] = df.columns.tolist()

                # Check if this is mostly text data (like HTML)
                text_heavy = False
                for col in df.columns:
                  if df[col].dtype == 'object':  # String columns
                    sample_text = str(df[col].iloc[0]) if not df[col].isna().iloc[0] else ""
                    if len(sample_text) > 1000 or '<' in sample_text:  # HTML or very long text
                      text_heavy = True
                      break

                if text_heavy:
                  print("This disclosure contains mostly text/HTML data - extracting clean text", file=sys.stderr, flush=True)
                  disclosure_summary['data_type'] = 'text_heavy'

                  # Extract clean text from HTML
                  for col in df.columns:
                    if df[col].dtype == 'object' and not df[col].isna().iloc[0]:
                      raw_content = str(df[col].iloc[0])
                      if '<' in raw_content:  # HTML content
                        # Remove HTML tags but keep the text content
                        clean_text = re.sub(r'<[^>]+>', ' ', raw_content)
                        # Clean up extra whitespace
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        # Remove special characters like \xa0
                        clean_text = re.sub(r'[\xa0\u00a0]', ' ', clean_text)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                        disclosure_summary[f'clean_text_{col}'] = clean_text
                        print(f"Extracted clean text length: {len(clean_text)} characters", file=sys.stderr, flush=True)

                else:
                  # Only include actual data for numerical/structured disclosures
                  disclosure_summary['data_type'] = 'structured'
                  disclosure_summary['sample_data'] = df.head(3).to_dict("records")

              print(f'Disclosure summary: {disclosure_summary}', file=sys.stderr, flush=True)

            except Exception as e:
              print(f'Error converting to dataframe: {e}', file=sys.stderr, flush=True)

          return disclosure_summary

        # Disclosure names are per-filing taxonomy roles, so a guessed or
        # borrowed name misses routinely. Returning {} here made a miss look
        # like a company that discloses nothing; name the miss and hand back
        # the roles this filing actually carries.
        return {
          'ticker': ticker,
          'success': False,
          'error': (f"No disclosure named '{disclosure_name}' in {ticker}'s "
                    f"latest {form_type}. Pick one from "
                    "available_disclosure_names, or call get_disclosures_names."),
          'available_disclosure_names': available_names,
        }

      except Exception as e:
        return {
          'ticker': ticker,
          'success': False,
          'error': (f"Unable to read the XBRL statements index for {ticker}: "
                    f"{type(e).__name__}: {e}"),
        }
    else:
      # failed to get the filing data
      return {
        'ticker': ticker,
        'success': False,
        'error': _filing_miss(ticker, form_type,
                              f"Unable to get latest filing for {ticker}")
      }
  except Exception as e:
    return {
      'ticker': ticker,
      'error': f"Unable to get {disclosure_name} for {ticker}: {str(e)}",
      'success': False
    }

def _foreign_revenue_base(ticker: str) -> Dict[str, Any]:
  """get_revenue_base's shape, filled from the foreign-issuer reader."""
  from .foreign_issuer import get_annual_revenue
  result = get_annual_revenue(ticker)
  if not result.get('success'):
    return {'ticker': ticker, 'success': False,
            'error': result.get('error'), 'revenue_base': None}
  currency = result.get('currency')
  return {
    'ticker': ticker,
    'revenue_base': float(result['latest_revenue']),
    'currency': currency,
    'revenue_base_usd': result.get('latest_revenue_usd'),
    'concept_used': result.get('concept_used'),
    'period_end': result.get('latest_period'),
    'filing_date': result.get('annual_filing_date'),
    'form_type': result.get('form'),
    'taxonomy': result.get('taxonomy_used'),
    'note': (f"Denominated in {currency}, NOT dollars. revenue_base_usd, when "
             f"present, is the filer's own convenience translation at its own "
             f"rate -- never a live conversion."
             if currency not in (None, 'USD') else None),
    'success': True,
  }


def get_revenue_base(ticker: str, form_type: str= "10-K") -> Dict[str, Any]:
  """Consolidated annual revenue -- the starting point for nearly all analysis.

  Foreign private issuers are routed to `get_annual_revenue`, which reads the
  same filing through the exact-concept, undimensioned selection in
  sec_series. `filter_annual_data` below cannot serve them: it takes the
  largest fact for the latest period, and an IFRS filer tags adjusted
  variants on dimension axes. Measured on SAP's FY2025 20-F, the max is
  EUR 37,804,000,000 -- a CONSTANT-CURRENCY segment figure -- against a real
  EUR 36,800,000,000, with `ifrs-full:RevenueOfCombinedEntity` (a pro-forma
  acquisition number) sitting at 36,861,000,000 in between. All three look
  like revenue and only one is.

  `currency` is never assumed. TSM reports NT$3.8tn and BABA RMB 1.02tn; a
  DCF built on either without converting the share price is off by ~30x.
  """
  if str(form_type).strip().upper() in ('20-F', '40-F'):
    return _foreign_revenue_base(ticker)
  try:
    filing_data = get_latest_filing(ticker, form_type)

    if filing_data and filing_data['xbrl_data']:
      xbrl = filing_data['xbrl_data']


      # Try different revenue concept names - prioritize Google's specific concepts
      for concept in REVENUE_CONCEPTS:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          currency = result.get('currency')
          return {
            'ticker': ticker,
            'revenue_base': float(result['value']),  # raw units, not scaled
            'currency': currency,
            'concept_used': result['concept_used'],
            'period_end': result['period_end'],
            'filing_date': filing_data['filing_date'],
            'form_type': form_type,
            'note': (None if currency in (None, 'USD') else
                     f'Denominated in {currency}, NOT dollars. Any valuation '
                     f'built on this must use a matching share count and '
                     f'price, or convert both consistently.'),
            'success': True
          }

      return {
        'ticker': ticker,
        'error': _filing_miss(ticker, form_type,
                              'No revenue concept found'),
        'success': False
      }

    return {
      'ticker': ticker,
      'error': _filing_miss(ticker, form_type, 'No XBRL data available'),
      'success': False
    }

  except Exception as e:
    return {
      'ticker': ticker,
      'error': f"Unable to get filing data: {str(e)}",
      'success': False
    }

def _ebitda_result(ticker: str, coverage: str, error: Optional[str],
                   operating_income=None, operating_income_concept=None,
                   d_and_a=None, d_and_a_concept=None, revenue=None,
                   currency=None, period_end=None) -> Dict[str, Any]:
  """One shape for every outcome, served or refused.

  A caller reading `coverage` must not have to branch on whether the key is
  there, and the halves that were read are handed back either way -- knowing
  a filer's D&A and revenue is useful even when its operating income is
  missing, and `get_historical_fcf` already answers that way for capex.
  """
  served = coverage == 'full'
  ebitda_amount = (float(operating_income) + float(d_and_a)) if served else None
  margin = (ebitda_amount / float(revenue) * 100.0) if served and revenue else None
  return {
    'ticker': ticker,
    'success': served,
    'coverage': coverage,
    'error': error,
    'concepts_tried': list(OPERATING_INCOME_CONCEPTS),
    'ebitda_margin_percent': margin,
    'ebitda_amount': ebitda_amount,
    'operating_income': None if operating_income is None else float(operating_income),
    'd&a': None if d_and_a is None else float(d_and_a),
    'revenue': None if revenue is None else float(revenue),
    'currency': currency,
    'operating_income_concept_used': operating_income_concept,
    'd&a_concept_used': d_and_a_concept,
    'period_end': period_end,
  }


def _is_bank_income_statement(xbrl, form_type: str) -> bool:
  """Whether this filing's income statement is a bank's.

  A bank reports net interest income and non-interest expense and has no
  operating income line at all. Read from the filing rather than from a list
  of tickers, so a filer this project has never seen gets the same answer.
  """
  return all(filter_annual_data(xbrl, concept, form_type) is not None
             for concept in BANK_INCOME_STATEMENT_CONCEPTS)


def _ebitda_not_covered(ticker: str, form_type: str, xbrl) -> str:
  """Why this filer cannot be given an EBITDA, in its own terms.

  Two reasons, and the difference matters to whoever reads it. A bank will
  never have an operating income line and EBITDA would be meaningless if it
  did, so pointing at a missing element invites someone to go find another
  one. Everybody else here simply presents no operating income subtotal, and
  the answer is that no substitute is honest.
  """
  tried = ', '.join(OPERATING_INCOME_CONCEPTS)
  if _is_bank_income_statement(xbrl, form_type):
    return (
      f"{ticker} files a bank income statement and EBITDA is not a meaningful "
      f"measure for a bank. Interest is a bank's raw material, not a financing "
      f"cost: the revenue line is struck net of interest expense "
      f"(us-gaap:RevenuesNetOfInterestExpense), already net of the very charge "
      f"EBITDA exists to add back, and the statement carries no operating "
      f"income line ({tried} is not tagged). "
      f"Adding interest back would report funding cost as profit. Use net "
      f"interest margin, the efficiency ratio (us-gaap:NoninterestExpense over "
      f"revenue), or return on tangible common equity instead.")
  return (
    f"{ticker} tags no operating income in its {form_type} ({tried}), so EBITDA "
    f"cannot be computed from this filing. No substitute is used: the only "
    f"subtotal above net income this filer tags is pre-tax income "
    f"(us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes...), which "
    f"carries interest and every other non-operating item, and revenue less "
    f"us-gaap:CostsAndExpenses is not operating income either -- filers put "
    f"financing interest and non-operating pension inside that total. Either "
    f"would be plausible and wrong. This is a coverage gap, not a zero margin.")


def get_ebitda_margin(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """EBITDA margin: operating income plus D&A, over revenue.

  Operating income comes from `us-gaap:OperatingIncomeLoss` and from nothing
  else -- see `OPERATING_INCOME_CONCEPTS` for the eleven filers that tag it
  nowhere and why none of the alternatives is defensible. They are refused
  with a named reason rather than given a number built on pre-tax income.
  """
  try:
    filing = get_latest_filing(ticker, form_type)
    if not filing or not filing['xbrl_data']:
      return _ebitda_result(
        ticker, 'not_covered',
        _filing_miss(ticker, form_type,
                     f'Unable to get latest filing for {ticker}'))

    xbrl = filing['xbrl_data']

    operating_income = operating_income_concept = period_end = None
    for concept in OPERATING_INCOME_CONCEPTS:
      result = filter_annual_data(xbrl, concept, form_type)
      if result:
        operating_income = result['value']
        operating_income_concept = result['concept_used']
        period_end = result['period_end']
        break

    # D&A off the cash flow statement. Combined totals only; the bare
    # `us-gaap:Depreciation` that ends DA_CONCEPTS is depreciation without
    # amortisation and would understate EBITDA rather than fail to find it.
    d_and_a = d_and_a_concept = None
    for concept in DA_CONCEPTS[:-1]:
      result = filter_annual_data(xbrl, concept, form_type)
      if result:
        d_and_a = result['value']
        d_and_a_concept = result['concept_used']
        break

    if d_and_a is None:
      # Some filers tag depreciation and amortisation separately and no total.
      depreciation = amortization = 0
      concepts_used = []
      for concept in ('us-gaap:Depreciation',
                      'us-gaap:AmortizationOfIntangibleAssets'):
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          if 'Depreciation' in concept:
            depreciation = result['value']
          else:
            amortization = result['value']
          concepts_used.append(result['concept_used'])
      if concepts_used:
        d_and_a = depreciation + amortization
        d_and_a_concept = ' + '.join(concepts_used)

    revenue_data = get_revenue_base(ticker, form_type)
    revenue = revenue_data['revenue_base'] if revenue_data['success'] else None
    currency = revenue_data.get('currency') if revenue_data['success'] else None

    if operating_income is None:
      return _ebitda_result(
        ticker, 'not_covered', _ebitda_not_covered(ticker, form_type, xbrl),
        d_and_a=d_and_a, d_and_a_concept=d_and_a_concept, revenue=revenue,
        currency=currency)

    if d_and_a is None:
      tried = ', '.join(DA_CONCEPTS[:-1])
      return _ebitda_result(
        ticker, 'not_covered',
        f"{ticker} tags no depreciation and amortisation total in its "
        f"{form_type} ({tried}), nor us-gaap:Depreciation beside "
        f"us-gaap:AmortizationOfIntangibleAssets. Operating income alone is "
        f"EBIT, not EBITDA, and returning it as EBITDA would understate the "
        f"margin by the whole of D&A.",
        operating_income=operating_income,
        operating_income_concept=operating_income_concept,
        revenue=revenue, currency=currency, period_end=period_end)

    if not revenue:
      return _ebitda_result(
        ticker, 'not_covered', revenue_data.get('error'),
        operating_income=operating_income,
        operating_income_concept=operating_income_concept,
        d_and_a=d_and_a, d_and_a_concept=d_and_a_concept,
        period_end=period_end)

    return _ebitda_result(
      ticker, 'full', None,
      operating_income=operating_income,
      operating_income_concept=operating_income_concept,
      d_and_a=d_and_a, d_and_a_concept=d_and_a_concept,
      revenue=revenue, currency=currency, period_end=period_end)

  except Exception as e:  # noqa: BLE001 - reported, never swallowed
    return _ebitda_result(
      ticker, 'not_covered',
      f'Unable to get EBITDA margin for {ticker}: {type(e).__name__}: {e}')

def get_capex_pct_revenue(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # function to get capital expenditures: Capex is the money that the company spends to buy, maintain, or upgrade physical assets
  # this metric will show CapEX as percentage of revenue, it shows how much a company is reinvesting back into its assets
  # CapEx % of revenue = capital expedeitures / total revenue
  # can find it on cash flow statement under 'cash flow from investing activities'
  try:
    filing = get_latest_filing(ticker, form_type)

    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      # The shared chain, so this tool and get_historical_fcf cannot disagree
      # about what capex a filer tags. Before it, HD, GE, CVX and T all fell
      # through to the component sum below and came back "Unable to find any
      # concepts" -- each of them tags PaymentsToAcquireProductiveAssets.
      primary_capex_concepts = CAPEX_CONCEPTS
      total_capex = 0
      capex_concept_used = None

      for concept in primary_capex_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
         total_capex = abs(result['value'])
         capex_concept_used = result['concept_used']
         break

       # unable to find anything in the primary concepts, so we move to components
       # thinking about this, there are many issues that can arise from this
       # issue 1. could be more concepts outside of component concepts
       # issue 2. could overlap when adding capital expenditures
       # please fix in future, this will be just a place holder
      if total_capex == 0:
        component_concepts = [
        'us-gaap:PaymentsToAcquireBuildings',
        'us-gaap:PaymentsToAcquireMachineryAndEquipment',
        'us-gaap:PaymentsToAcquireComputerSoftwareAndEquipment',
        'us-gaap:PaymentsToAcquireOtherPropertyPlantAndEquipment'
        ]
        print(f'WARNING for {ticker}: Might not account for all capital expenditures. Possible overlap of capital expenditures. Using concepts: {component_concepts}', file=sys.stderr)
        for concept in component_concepts:
          result = filter_annual_data(xbrl, concept, form_type)
          if result:
            total_capex += result['value']

        if total_capex == 0:
          return{
            'error': f'Unable to find any concepts for {ticker}',
            'success': False
          }

      # now that we have the capex value we can get the percentage
      revenue_data = get_revenue_base(ticker, form_type)
      if not revenue_data['success']:
        return revenue_data # return the revenue error
      revenue = revenue_data['revenue_base']  # already in raw dollars
      capex_pct = (total_capex / revenue) * 100

      return{
        'error': None,
        'success': True,
        'ticker': ticker,
        'total_capex': float(total_capex),  # keep in raw dollars
        'revenue': float(revenue),  # keep in raw dollars
        'capex_pct_revenue': float(capex_pct),
        'capex_concept_used': capex_concept_used,
        'period_end': revenue_data['period_end']
      }
    else:
      return {
        'error': _filing_miss(ticker, form_type,
                              f"Unable to get xbrl data for: {ticker}"),
        'success': False
      }

  except Exception:
    return{
      'error': f'Unable to get filing for {ticker}',
      'success': False
    }


def get_tax_rate(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # returns the effective/actual tax rate that the company pays on its profits
  # can find it on the income statement in 'income before provision for income taxes or similar wording' and 'provision for income taxes
  # formula: Effective tax rate = provision for income taxes / earnings before taxes
  try:
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      tax_expense = None
      tax_concept_used = None
      for concept in TAX_EXPENSE_CONCEPTS:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          tax_expense = float(result['value'])
          tax_concept_used = result['concept_used']
          break

      pretax_income = 0.0
      pretax_concept_used = None
      for concept in PRETAX_INCOME_CONCEPTS:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          pretax_income = float(result['value'])
          pretax_concept_used = result['concept_used']
          break

      # A tax expense that was never found is not a tax expense of zero. The
      # default used to be 0.0, which returned an effective rate of 0% for a
      # filer whose tax line the chain could not reach.
      if tax_expense is None:
        return{
          'error': (f'{ticker} tags no income tax expense concept in its '
                    f'{form_type}. Tried: {", ".join(TAX_EXPENSE_CONCEPTS)}'),
          'success': False
        }

      # calulate the effective tax rate: Effective tax rate = provision for income taxes / earnings before taxes
      if pretax_income != 0: # prevent divide by 0 error
        effective_tax_rate = (tax_expense / pretax_income) * 100
      else:
        return{
          'error': f"pretax income is 0, unable to calculate effective tax rate",
          'success': False
        }

      return{
        'error': None,
        'success': True,
        'effective_tax_rate': effective_tax_rate,
        'tax_expense': tax_expense,
        'tax_concept_used': tax_concept_used,
        'pretax_income': pretax_income,
        'pretax_concept_used': pretax_concept_used
      }

    else:
      return{
        'error': _filing_miss(ticker, form_type,
                              f'Unable to get xbrl data for {ticker}'),
        'success': False
      }
  except Exception as e:
    return{
      'error': f'Unable to get filing for {ticker}: {str(e)}',
      'success': False
    }


def get_depreciation(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # this is the accounting method of allocating the cost of a physical asset over its uselife. It is a non cash expense is will be expressed as a percentage of revenue
  # formula: depreication % of revenue = depreication & amorization / total revenue
  # this will be helpful beacuse it helps us find the age and cost structure of a company's assets
  # find it on the cash flow statement, usually under "cash flow from operating activities"
  try:
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      d_a_value = 0.0
      d_a_concept = None
      for concept in DA_CONCEPTS:
        results = filter_annual_data(xbrl, concept, form_type)
        if results:
          d_a_value = float(results['value'])
          d_a_concept = results['concept_used']
          break

      if d_a_value == 0.0:
        return{
          'error': f"Unable to find concept for {ticker}: Concepts used = {list(DA_CONCEPTS)}",
          'success': False
        }

      revenue_data = get_revenue_base(ticker, form_type)

      if not revenue_data['success']:
        return revenue_data # just return revenue error

      revenue = revenue_data['revenue_base']  # already in raw dollars

      # now we have the d_a value and revenue so we can calulate deprceication %
      d_a_pct = (d_a_value / revenue) * 100

      return{
        'error': None,
        'success': True,
        'd&a_pct': d_a_pct,
        'concept': d_a_concept,
        'd&a': d_a_value,
        'revenue': revenue
      }

    else:
      return{
        'error': _filing_miss(ticker, form_type,
                              f"Unable to get filing for {ticker}"),
        'success': False
      }

  except Exception as e:
    return{
      'error': f"Unable to get filing for {ticker}",
      'success':False
    }


def _get_revenue_from_xbrl(xbrl, form_type: str):
  """Revenue for the filing's own period; (value, period_end) or None."""
  for concept in REVENUE_CONCEPTS:
    d = filter_annual_data(xbrl, concept, form_type)
    if d:
      return d['value'], d['period_end']
  return None


# The combined element, tried first and alone. A filer that tags it has
# already done the addition; summing its halves on top would double-count.
# AAPL tags all three -- 27,601 combined against 8,077 + 19,524 -- and the
# halves reconcile to the combined element exactly.
SGA_COMBINED_CONCEPTS = ('us-gaap:SellingGeneralAndAdministrativeExpense',)

# The selling half, in the order filers use it. Alternation, not a sum: UBER
# tags SellingAndMarketingExpense (4,898, its "Sales and marketing" line) AND
# MarketingExpense (1,600, a narrower element), and adding both would invent
# 1.6bn of expense. MarketingExpense is last for that reason, but it is in the
# chain at all because for AMZN it IS the income statement line -- "Marketing",
# 47,129 -- and no broader selling element is tagged.
#
# MarketingAndAdvertisingExpense is deliberately absent. It is what a bank
# tags (JPM: 5,531) and it is an advertising budget, not a selling-and-
# administrative line.
SGA_SELLING_CONCEPTS = ('us-gaap:SellingAndMarketingExpense',
                        'us-gaap:MarketingAndSalesExpense',
                        'us-gaap:SellingExpense',
                        'us-gaap:MarketingExpense')

SGA_ADMIN_CONCEPTS = ('us-gaap:GeneralAndAdministrativeExpense',)


def _select_sga(read) -> Optional[Dict[str, Any]]:
  """Selling + G&A, summing the halves when the filer tags no combined element.

  `read` is a concept -> fact-dict (or None) callable, so this is testable
  without a filing.

  The chain used to be "combined, else G&A". Microsoft tags no combined
  element at all, so it answered with GeneralAndAdministrativeExpense --
  7,956,000,000, 2.40% of revenue -- and silently dropped
  SellingAndMarketingExpense, 26,710,000,000. True selling+G&A is
  34,666,000,000, 10.4%. That made MSFT look like it spent a third of Apple's
  SG&A ratio when it in fact spends more, which is the opposite of the truth
  and exactly the comparison this tool exists to support.

  A selling line with no administrative line beside it is NOT accepted as
  SG&A. The split pattern this handles is "selling + G&A"; a lone marketing
  element on a filer that tags no G&A at all is a narrower disclosure than
  SG&A, and reporting it as SG&A would put a new error where there was an
  honest absence -- JPM would acquire a 5.5bn "SG&A" against roughly 100bn of
  noninterest expense. G&A alone IS accepted: plenty of filers have no
  separate selling line, and the operating-income reconciliation below is what
  tells that case apart from the MSFT one.

  Returns None when the filer tags nothing in the chain.
  """
  for concept in SGA_COMBINED_CONCEPTS:
    fact = read(concept)
    if fact:
      return {'value': fact['value'],
              'concept_used': fact['concept_used'],
              'components': [{'concept': fact['concept_used'],
                              'value': fact['value']}]}

  def _first(concepts):
    for concept in concepts:
      fact = read(concept)
      if fact:
        return fact
    return None

  admin = _first(SGA_ADMIN_CONCEPTS)
  if admin is None:
    return None
  selling = _first(SGA_SELLING_CONCEPTS)

  components = []
  if selling:
    components.append({'concept': selling['concept_used'],
                       'value': selling['value']})
  components.append({'concept': admin['concept_used'], 'value': admin['value']})
  return {
    'value': sum(c['value'] for c in components),
    'concept_used': ' + '.join(c['concept'] for c in components),
    'components': components,
  }


# A residual worth more than this share of revenue is material enough that a
# reader comparing SG&A ratios across filers would be misled by it. MSFT's
# omitted selling line was 8.05% of revenue.
_RECONCILE_TOLERANCE_PCT_REVENUE = 1.0


def _reconcile_operating_income(revenue, gross_profit, sga, rnd,
                                operating_income) -> Optional[Dict[str, Any]]:
  """Check gross profit - SG&A - R&D against the operating income the filer tags.

  This is the check that would have caught the SG&A defect without anyone
  looking at a filing: 225,465 - 7,956 - 35,562 = 181,947 against MSFT's
  reported 155,237, a gap of exactly the 26,710 of Sales & Marketing that the
  concept chain dropped. AAPL and NVDA reconciled to the dollar throughout,
  so the gap was a property of the extraction, not of the arithmetic.

  A residual is not automatically an error. Many filers carry operating
  expense lines this tool does not read -- restructuring, impairment,
  amortisation of acquired intangibles -- so `reconciles` false means "there
  is something here that is not SG&A or R&D", which is a fact the caller
  needs either way.

  Returns None when either input is absent (banks tag no GrossProfit), because
  a reconciliation against a missing number is not a reconciliation.
  """
  if gross_profit is None or operating_income is None or not revenue:
    return None
  implied = gross_profit - (sga or 0) - (rnd or 0)
  residual = implied - operating_income
  residual_pct = (residual / revenue) * 100
  return {
    'basis': 'gross_profit - sga - rnd',
    'gross_profit': gross_profit,
    'sga': sga,
    'rnd': rnd,
    'implied_operating_income': implied,
    'operating_income': operating_income,
    'residual': residual,
    'residual_pct_revenue': residual_pct,
    'tolerance_pct_revenue': _RECONCILE_TOLERANCE_PCT_REVENUE,
    'reconciles': abs(residual_pct) <= _RECONCILE_TOLERANCE_PCT_REVENUE,
  }


def get_margin_breakdown(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract gross margin, SG&A %, R&D % from the latest filing.

  Returns ticker, revenue, gross_profit, sga, rnd, *_pct_revenue values, plus
  concepts_used for traceability. Banks (no COGS) typically have no GrossProfit
  XBRL concept; absence is expected, not an error.

  `sga` sums the selling and administrative halves when the filer tags no
  combined element -- see `_select_sga`. `reconciliation` shows the arithmetic
  against the filer's own operating income so the number can be checked
  without leaving the response, and a material gap is reported as a structured
  warning rather than passing silently.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': _filing_miss(ticker, form_type,
                                    f'No filing found for {ticker}'),
              'success': False}

    xbrl = filing_data['xbrl_data']
    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)
    if rev_tuple is None:
      return {'ticker': ticker, 'error': 'No revenue concept found', 'success': False}
    revenue, period_end = rev_tuple

    result = {'ticker': ticker, 'revenue': revenue, 'period_end': period_end, 'success': True}
    concepts_used = {}

    for c in ('us-gaap:GrossProfit',):
      gp = filter_annual_data(xbrl, c, form_type)
      if gp:
        result['gross_profit'] = gp['value']
        result['gross_margin_pct'] = (gp['value'] / revenue) * 100
        concepts_used['gross_profit'] = gp['concept_used']
        break

    sga = _select_sga(lambda c: filter_annual_data(xbrl, c, form_type))
    if sga:
      result['sga'] = sga['value']
      result['sga_pct_revenue'] = (sga['value'] / revenue) * 100
      result['sga_components'] = sga['components']
      concepts_used['sga'] = sga['concept_used']

    # BIIB and VRTX tag only the ExcludingAcquiredInProcessCost element --
    # 1,778,600,000 and 3,909,500,000 -- which the prefix match used to reach
    # from a query for us-gaap:ResearchAndDevelopmentExpense.
    #
    # ADBE tags only the ...SoftwareExcludingAcquiredInProcessCost element,
    # 4,294,000,000, and so reported no R&D at all. The reconciliation check
    # below is what found it: 21,218 - 8,061 - 0 implied operating income of
    # 13,157 against the 8,706 Adobe tags, an 18.7%-of-revenue residual that
    # closes to 0.7% once the element is read.
    for c in ('us-gaap:ResearchAndDevelopmentExpense',
              'us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost',
              'us-gaap:ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost'):
      rnd = filter_annual_data(xbrl, c, form_type)
      if rnd:
        result['rnd'] = rnd['value']
        result['rnd_pct_revenue'] = (rnd['value'] / revenue) * 100
        concepts_used['rnd'] = rnd['concept_used']
        break

    oi = filter_annual_data(xbrl, 'us-gaap:OperatingIncomeLoss', form_type)
    if oi:
      result['operating_income'] = oi['value']
      concepts_used['operating_income'] = oi['concept_used']

    result['concepts_used'] = concepts_used

    warnings: list = []
    recon = _reconcile_operating_income(
      revenue=revenue,
      gross_profit=result.get('gross_profit'),
      sga=result.get('sga'),
      rnd=result.get('rnd'),
      operating_income=result.get('operating_income'))
    if recon is not None:
      result['reconciliation'] = recon
      if not recon['reconciles']:
        warnings.append(warning(
          'operating_income_does_not_reconcile',
          f"gross_profit - sga - rnd implies operating income of "
          f"{recon['implied_operating_income']:,.0f} against the "
          f"{recon['operating_income']:,.0f} {ticker} tags: a residual of "
          f"{recon['residual']:,.0f} ({recon['residual_pct_revenue']:.2f}% of "
          f"revenue). Either an operating expense line is not read here "
          f"(restructuring, impairment, amortisation of acquired intangibles) "
          f"or the SG&A concept chain missed an element this filer tags.",
          residual=recon['residual'],
          residual_pct_revenue=round(recon['residual_pct_revenue'], 4)))
    result['warnings'] = warnings

    if 'gross_profit' not in result:
      print(f"[Validate SEC] {ticker}: gross_profit XBRL concept not found (expected for banks/financials)",
            file=sys.stderr, flush=True)
    return result

  except Exception as e:
    return {'ticker': ticker, 'error': f'get_margin_breakdown failed: {e}', 'success': False}


def _consolidated_series(xbrl, concepts, min_span=300, max_span=400):
  """Every annual undimensioned period for the first covered concept.

  A 10-K's cash-flow statement carries three comparative years, so the trend is
  available from one filing without walking filings. The span window keeps
  quarterly durations out: a 10-K tags those beside the annual ones and they
  end on the same day.

  Returns (list of {period_end, value} newest first, concept_used).
  """
  from tools.web_search_server.sec_series import _period_rank, concept_point
  for concept in concepts:
    try:
      point = concept_point(xbrl, concept, "", "")
    except Exception:  # noqa: BLE001 - try the next concept
      continue
    if point is None:
      continue

    # One period can carry two roundings; keep the precise one.
    best = {}
    for fact in point.undimensioned():
      end, span = _period_rank(fact.period)
      if not end or not (min_span <= span <= max_span):
        continue
      prior = best.get(end)
      if prior is None or (fact.decimals or -999) > (prior[1] or -999):
        best[end] = (fact.value, fact.decimals)

    if best:
      series = [{'period_end': end, 'value': value}
                for end, (value, _) in sorted(best.items(), reverse=True)]
      return series, concept
  return [], None


def get_historical_fcf(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Operating cash flow, capex, and free cash flow from the latest filing.

  A filer that tags no capital-expenditure element gets `success: False` and
  `coverage: 'not_covered'` rather than a free cash flow. The previous
  `fcf = ocf - (capex or 0)` turned a missing input into a zero one, which
  reports operating cash flow as free cash flow: Amazon read 139,514,000,000
  against a real 7,695,000,000, an 18x overstatement, because its capex is
  tagged under an element the two-concept chain did not try. Widening the
  chain fixes the eight filers in the audit basket that did tag capex; the
  loud failure is for the ones that genuinely do not, which is every bank in
  it. The operating cash flow that was read is still returned, so nothing
  found is thrown away.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': _filing_miss(ticker, form_type,
                                    f'No filing found for {ticker}'),
              'success': False}

    xbrl = filing_data['xbrl_data']
    ocf = None
    ocf_concept = None
    for c in OPERATING_CASH_FLOW_CONCEPTS:
      d = filter_annual_data(xbrl, c, form_type)
      if d:
        ocf = d['value']
        ocf_concept = d['concept_used']
        break

    capex = None
    capex_concept = None
    for c in CAPEX_CONCEPTS:
      d = filter_annual_data(xbrl, c, form_type)
      if d:
        capex = abs(d['value'])  # capex is reported negative on the CF statement
        capex_concept = d['concept_used']
        break

    if ocf is None:
      return {'ticker': ticker, 'error': 'OCF concept not found', 'success': False}

    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)
    revenue = rev_tuple[0] if rev_tuple else None
    period_end = rev_tuple[1] if rev_tuple else None

    # The name promises history, so return it. A single period cannot show a
    # divergence, which is what callers reach for this tool to see.
    ocf_series, _ = _consolidated_series(xbrl, OPERATING_CASH_FLOW_CONCEPTS)
    capex_series, _ = _consolidated_series(xbrl, CAPEX_CONCEPTS)
    capex_by_end = {row['period_end']: abs(row['value']) for row in capex_series}

    series = []
    for row in ocf_series:
      end = row['period_end']
      period_ocf = row['value']
      period_capex = capex_by_end.get(end)
      series.append({
        'period_end': end,
        'operating_cash_flow': period_ocf,
        'capex': period_capex,
        # None, never a zero standing in for a missing input -- that is what
        # made Amazon's free cash flow read 18x its real value.
        'free_cash_flow': (period_ocf - period_capex)
                          if period_capex is not None else None,
      })

    result = {
      'ticker': ticker,
      'operating_cash_flow': ocf,
      'operating_cash_flow_concept_used': ocf_concept,
      'capex': capex,
      'capex_concept_used': capex_concept,
      'period_end': period_end,
      'series': series,
    }

    if capex is None:
      result.update({
        'free_cash_flow': None,
        'fcf_margin_pct': None,
        'coverage': 'not_covered',
        'success': False,
        'error': (f'{ticker} tags no capex concept in its {form_type}; free '
                  f'cash flow cannot be computed. Tried: '
                  f'{", ".join(CAPEX_CONCEPTS)}'),
      })
      return result

    fcf = ocf - capex
    result.update({
      'free_cash_flow': fcf,
      'fcf_margin_pct': (fcf / revenue * 100) if revenue else None,
      'coverage': 'full',
      'success': True,
      'error': None,
    })
    return result
  except Exception as e:
    return {'ticker': ticker, 'error': f'get_historical_fcf failed: {e}', 'success': False}


def get_working_capital(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract current assets/liabilities and compute NWC + NWC % of revenue.

  Balance sheet items are XBRL instant facts (point-in-time), not duration facts,
  so this uses filter_instant_data rather than filter_annual_data.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': _filing_miss(ticker, form_type, 'No filing'),
              'success': False}

    xbrl = filing_data['xbrl_data']
    ca = filter_instant_data(xbrl, 'us-gaap:AssetsCurrent')
    cl = filter_instant_data(xbrl, 'us-gaap:LiabilitiesCurrent')
    ar = filter_instant_data(xbrl, 'us-gaap:AccountsReceivableNetCurrent')
    # A long-term-contract manufacturer nets customer advances and progress
    # billings against inventory and tags that element instead. BA's
    # 84,679,000,000 used to be reached from `us-gaap:InventoryNet` by prefix.
    inv = (filter_instant_data(xbrl, 'us-gaap:InventoryNet')
           or filter_instant_data(
               xbrl,
               'us-gaap:InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings'))
    ap = filter_instant_data(xbrl, 'us-gaap:AccountsPayableCurrent')
    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)

    if not (ca and cl):
      return {'ticker': ticker, 'error': 'Current assets/liabilities not found', 'success': False}

    nwc = ca['value'] - cl['value']
    revenue = rev_tuple[0] if rev_tuple else None

    return {
      'ticker': ticker,
      'current_assets': ca['value'],
      'current_liabilities': cl['value'],
      'net_working_capital': nwc,
      'nwc_pct_revenue': (nwc / revenue * 100) if revenue else None,
      'accounts_receivable': ar['value'] if ar else None,
      'inventory': inv['value'] if inv else None,
      'accounts_payable': ap['value'] if ap else None,
      'period_end': ca['period_end'],
      'success': True
    }
  except Exception as e:
    return {'ticker': ticker, 'error': f'get_working_capital failed: {e}', 'success': False}


# Curated ticker lookup for supply-chain extraction. Maps display-name
# fragments (case-insensitive) to ticker symbols. Covers mega-caps that
# appear most frequently in 10-K Business sections. Add entries as
# coverage gaps surface in real research.
_COMPANY_NAME_TO_TICKER: Dict[str, str] = {
  # Mega-cap tech
  'apple':                'AAPL',
  'alphabet':             'GOOGL',
  'google':               'GOOGL',
  'meta platforms':       'META',
  'meta':                 'META',
  'facebook':             'META',
  'microsoft':            'MSFT',
  'amazon':               'AMZN',
  'tesla':                'TSLA',
  'nvidia':               'NVDA',
  'oracle':               'ORCL',
  'salesforce':           'CRM',
  'adobe':                'ADBE',
  'sap':                  'SAP',
  'cisco':                'CSCO',
  'vmware':               'VMW',
  'snowflake':            'SNOW',
  'palantir':             'PLTR',
  'workday':              'WDAY',
  'servicenow':           'NOW',
  'datadog':              'DDOG',
  'crowdstrike':          'CRWD',
  'palo alto networks':   'PANW',
  'zscaler':              'ZS',
  'fortinet':             'FTNT',
  'cloudflare':           'NET',
  'mongodb':              'MDB',
  'shopify':              'SHOP',
  'square':               'SQ',
  'paypal':               'PYPL',
  'block':                'SQ',
  'twilio':               'TWLO',
  'zoom':                 'ZM',
  'docusign':             'DOCU',
  'atlassian':            'TEAM',
  'roblox':               'RBLX',
  'unity software':       'U',
  'ibm':                  'IBM',
  'sony':                 'SONY',
  'nintendo':             'NTDOY',
  # Semis
  'taiwan semiconductor': 'TSM',
  'tsmc':                 'TSM',
  'samsung electronics':  'SSNLF',
  'samsung':              'SSNLF',
  'sk hynix':             '000660.KS',
  'micron':               'MU',
  'micron technology':    'MU',
  'intel':                'INTC',
  'amd':                  'AMD',
  'advanced micro':       'AMD',
  'asml':                 'ASML',
  'qualcomm':             'QCOM',
  'broadcom':             'AVGO',
  'marvell':              'MRVL',
  'arm holdings':         'ARM',
  'arm':                  'ARM',
  'lam research':         'LRCX',
  'applied materials':    'AMAT',
  'kla':                  'KLAC',
  'texas instruments':    'TXN',
  'analog devices':       'ADI',
  'nxp':                  'NXPI',
  'on semiconductor':     'ON',
  # OEMs / EMS / contract manufacturers
  'foxconn':              '2317.TW',
  'hon hai':              '2317.TW',
  'pegatron':             '4938.TW',
  'compal electronics':   '2324.TW',
  'wistron':              '3231.TW',
  'flex':                 'FLEX',
  'jabil':                'JBL',
  'celestica':            'CLS',
  # Mega-cap industrials / autos / energy
  'general motors':       'GM',
  'ford motor':           'F',
  'ford':                 'F',
  'stellantis':           'STLA',
  'toyota':               'TM',
  'volkswagen':           'VWAGY',
  'boeing':               'BA',
  'lockheed martin':      'LMT',
  'general electric':     'GE',
  'caterpillar':          'CAT',
  'deere':                'DE',
  'honeywell':            'HON',
  'raytheon':             'RTX',
  # 'rtx' alias removed — collides with NVIDIA's RTX GPU product line
  'exxon mobil':          'XOM',
  'exxon':                'XOM',
  'chevron':              'CVX',
  'conocophillips':       'COP',
  'shell':                'SHEL',
  'bp':                   'BP',
  'totalenergies':        'TTE',
  'nextera energy':       'NEE',
  'duke energy':          'DUK',
  # Healthcare / pharma
  'johnson & johnson':    'JNJ',
  'pfizer':               'PFE',
  'merck':                'MRK',
  'eli lilly':            'LLY',
  'lilly':                'LLY',
  'novo nordisk':         'NVO',
  'bristol-myers':        'BMY',
  'abbvie':               'ABBV',
  'astrazeneca':          'AZN',
  'gilead':               'GILD',
  'amgen':                'AMGN',
  'moderna':              'MRNA',
  'biontech':             'BNTX',
  'unitedhealth':         'UNH',
  'cvs health':           'CVS',
  # Financials
  'jpmorgan':             'JPM',
  'jp morgan':            'JPM',
  'bank of america':      'BAC',
  'wells fargo':          'WFC',
  'citigroup':            'C',
  'goldman sachs':        'GS',
  'morgan stanley':       'MS',
  'blackrock':            'BLK',
  'berkshire hathaway':   'BRK.B',
  'visa':                 'V',
  'mastercard':           'MA',
  'american express':     'AXP',
  # Retail / consumer
  'walmart':              'WMT',
  'costco':               'COST',
  'target corporation':   'TGT',  # require "Corporation" to avoid "target" as common word
  'home depot':           'HD',
  "lowe's":               'LOW',
  'nike':                 'NKE',
  'starbucks':            'SBUX',
  "mcdonald's":           'MCD',
  'mcdonalds':            'MCD',
  'coca-cola':            'KO',
  'pepsico':              'PEP',
  'procter & gamble':     'PG',
  'unilever':             'UL',
  # Streaming / media
  'netflix':              'NFLX',
  'walt disney':          'DIS',
  'disney':               'DIS',
  'paramount':            'PARA',
  'comcast':              'CMCSA',
  'warner bros discovery':'WBD',
  'spotify':              'SPOT',
}


def _schedule13_party(filing, attribute: str):
  """(name, cik) of a filing's subject or filer, or (None, None)."""
  try:
    for party in (getattr(filing.header, attribute, None) or []):
      info = getattr(party, 'company_information', None)
      if info is None:
        continue
      cik = getattr(info, 'cik', None)
      return (getattr(info, 'name', None),
              str(int(cik)) if cik not in (None, '') else None)
  except Exception:  # noqa: BLE001 - an unreadable header is reported, not fatal
    pass
  return (None, None)


def get_schedule_13d_filings(ticker: str, limit: int = 15,
                             include_passive: bool = True) -> Dict[str, Any]:
  """Return SC 13D (activist) and SC 13G (passive) filings naming the
  target ticker as subject company.

  SC 13D = institutional holder with >5% stake AND intent to influence
  management (activist). SC 13G = >5% stake, passive (index funds,
  long-only). 13D/A and 13G/A are amendments.

  Activist 13D filings are highly informative for thesis-building —
  knowing Ackman or Loeb has built a position is decisive context. Even
  passive 13G filings show concentration of institutional ownership
  (Vanguard, BlackRock, State Street typically dominate).

  Attempts to extract stake percentage from the filing body via regex
  scan; surfaces as `stake_pct` when found. Falls back to filer name +
  date + URL when stake parse fails.
  """
  _require_identity()

  try:
    company = Company(ticker)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'Company lookup failed: {type(e).__name__}: {e}'}

  company_cik = str(int(company.cik))

  forms_to_pull = ['SC 13D', 'SC 13D/A']
  if include_passive:
    forms_to_pull.extend(['SC 13G', 'SC 13G/A'])

  rows: list = []
  whole_set: dict = {}       # accession -> is_activist, subject-side only
  filed_by_company: set = set()   # this company's own stakes in other issuers
  heuristic_disagreements: list = []   # where the header contradicts the proxy
  for form in forms_to_pull:
    try:
      filings = company.get_filings(form=form)
    except Exception as e:
      # A swallowed failure here reported `activist_count: 0` -- "no activist
      # investors" -- when the real cause was an SEC rate limit. The one form
      # most likely to be missing is the one whose absence inverts the answer,
      # so a failed query refuses rather than under-counting.
      return {'ticker': ticker, 'success': False,
              'error': (f'Could not list {form} filings for {ticker}: '
                        f'{type(e).__name__}: {e}. Counts are omitted rather '
                        f'than under-reported -- a missing form reads as an '
                        f'absence of filings.'),
              'filings': [], 'failed_form': form}

    # Accession, form and file number come from the submissions index, so the
    # SET can be classified and counted without fetching a single document.
    #
    # A CIK's folder holds both sides of a Schedule 13 relationship: filings
    # where the company is the SUBJECT, and filings it made about OTHER
    # issuers. Counting both answered "are there activists in Intel?" with 124
    # when 71 of the first 100 rows were Intel filing on MariaDB, Mobileye,
    # Joby and Vuzix. EDGAR gives the subject an `005-` file number, present
    # and constant in the subject's own folder and blank on the filings it
    # made about others -- verified against header ground truth on INTC, 28 of
    # 28 in agreement.
    subject_side = []
    for f in filings:
      try:
        accession = f.accession_number
      except Exception:
        continue
      if not str(getattr(f, 'file_number', '') or '').strip():
        filed_by_company.add(accession)
        continue
      whole_set.setdefault(accession, form.startswith('SC 13D'))
      subject_side.append(f)

    # Page the subject-side filings, not the raw folder. Slicing first meant
    # asking for 3 rows on INTC returned 1, because three quarters of the
    # folder is Intel filing on other issuers.
    for f in subject_side[:limit]:
      # The page already pays for a document fetch, so its rows are verified
      # against the filing header, which is the authority. The file number is
      # only a free proxy, used for the set.
      filer_name, filer_cik = _schedule13_party(f, 'filers')
      subject_name, subject_cik = _schedule13_party(f, 'subject_companies')
      subject_verified = subject_cik is not None
      is_subject = (subject_cik == company_cik) if subject_verified else None
      if is_subject is False:
        # The header is the authority: this is a stake the company took in
        # someone else. Note where it contradicts the file-number proxy, so a
        # drift in that heuristic surfaces instead of quietly skewing counts.
        if f.accession_number in whole_set:
          heuristic_disagreements.append(f.accession_number)
        continue
      if is_subject is None and f.accession_number not in whole_set:
        # The header would not parse and the proxy says this filing is one the
        # company made about another issuer. Nothing contradicts the proxy, so
        # it stands -- and the row stays out of both the page and the count
        # rather than being in one and not the other.
        continue
      if is_subject and f.accession_number not in whole_set:
        heuristic_disagreements.append(f.accession_number)
        whole_set.setdefault(f.accession_number, form.startswith('SC 13D'))
        filed_by_company.discard(f.accession_number)

      # Try to extract stake percentage from filing body
      stake_pct = None
      try:
        text = f.text()
        # Common phrasings:
        #   "Percent of class represented by amount in row (11): 5.7%"
        #   "Percentage of Class: 6.2%"
        pat = re.compile(
          r'(?:percent\s+of\s+(?:class|shares)|percentage\s+of\s+class|aggregate\s+percentage)[\s:\-]{0,80}([0-9]{1,2}(?:\.[0-9]{1,2})?)\s*%',
          re.IGNORECASE)
        m = pat.search(text)
        if m:
          stake_pct = float(m.group(1))
      except Exception:
        pass

      rows.append({
        'form':             form,
        'filing_date':      str(f.filing_date),
        'accession':        f.accession_number,
        'filer_name':       filer_name,
        'filer_cik':        filer_cik,
        'subject_name':     subject_name,
        'subject_cik':      subject_cik,
        'is_subject':       is_subject,
        # False means the header would not parse. The row is kept and flagged
        # rather than dropped: an absence created by our own failure would be
        # indistinguishable from a company nobody has filed on.
        'subject_verified': subject_verified,
        'stake_pct':        stake_pct,
        'url':              getattr(f, 'filing_url', None),
        'is_amendment':     form.endswith('/A'),
        'is_activist':      form.startswith('SC 13D'),
      })

  if not rows:
    return {'ticker': ticker, 'success': True,
            'error': None,
            'filings': [],
            'count': 0,
            'rows_returned': 0,
            'truncated': False,
            'activist_count': 0,
            'passive_count': 0,
            'filed_by_this_company_count': len(filed_by_company),
            'note': 'No Schedule 13D/G filings found — company may be too small to have a 5%-stake holder, or coverage gap.'}

  # Dedupe by accession — edgartools returns the same filing under both
  # "SC 13G" and "SC 13G/A" when it's an amendment.
  seen_acc = set()
  deduped = []
  for r in rows:
    if r['accession'] in seen_acc:
      continue
    seen_acc.add(r['accession'])
    deduped.append(r)
  rows = deduped

  # Sort newest first
  rows.sort(key=lambda r: r['filing_date'], reverse=True)

  # Count the SET, then take the page. Counting after truncating answered
  # "are there activists in INTC?" with activist_count 0 at the default limit
  # and 31 at limit=100 -- not a smaller version of the answer, the opposite
  # one.
  matched = len(whole_set)
  activist_count = sum(1 for is_activist in whole_set.values() if is_activist)
  passive_count = matched - activist_count
  rows = rows[:limit]

  return {
    'ticker':         ticker,
    'success':        True,
    'error':          None,
    'filings':        rows,
    'count':          matched,
    'rows_returned':  len(rows),
    'truncated':      matched > len(rows),
    'activist_count': activist_count,
    'passive_count':  passive_count,
    # Kept rather than discarded: a stake this company took in another issuer
    # is real information, just the answer to a different question.
    'filed_by_this_company_count': len(filed_by_company),
    # The set is classified from the submissions index; the page is verified
    # against filing headers. They agreed on 28 of 28 filings when the proxy
    # was chosen, and a non-zero figure here means it has drifted.
    'subject_filter_disagreements': len(heuristic_disagreements),
    'note':           'Stake percentage extracted via regex on common phrasings; null = parse failed (analyst should check URL).',
  }


def _extract_section_from_filing_obj(filing_obj, item: str) -> Optional[str]:
  """Helper for diff_10k: extract Item 1A or Item 7 body text from any
  10-K filing object. Mirrors logic in extract_risk_factors / extract_mda
  but operates on an arbitrary filing (not just the latest)."""
  try:
    text = filing_obj.text()
  except Exception:
    return None
  if not text:
    return None

  if item == '1A':
    # Body header (skip TOC): require uppercase "ITEM 1A" + "RIS K|RISK FACTORS"
    m = re.search(r'ITEM\s+1A\.?\s+(?:RIS\s?K|RISK)\s+FACTORS', text, re.IGNORECASE)
    if m and m.start() < 30000:
      m2 = re.search(r'ITEM\s+1A\.?\s+(?:RIS\s?K|RISK)\s+FACTORS',
                     text[30000:], re.IGNORECASE)
      if m2:
        start = 30000 + m2.start()
      else:
        return None
    elif m:
      start = m.start()
    else:
      return None
    end_m = re.search(r'ITEM\s+1B\b', text[start + 200:], re.IGNORECASE)
    end = start + 200 + end_m.start() if end_m else min(start + 200000, len(text))
    return text[start:end]

  if item == '7':
    m = re.search(r'ITEM\s+7\.?\s+MANAGEMENT.{0,5}S?\s+DISCUSSION',
                  text, re.IGNORECASE)
    if not m:
      return None
    start = m.start()
    if start < 30000:
      m2 = re.search(r'ITEM\s+7\.?\s+MANAGEMENT.{0,5}S?\s+DISCUSSION',
                     text[30000:], re.IGNORECASE)
      if m2:
        start = 30000 + m2.start()
    end_m = re.search(r'ITEM\s+7A\b', text[start + 200:], re.IGNORECASE)
    if not end_m:
      end_m = re.search(r'ITEM\s+8\b', text[start + 200:], re.IGNORECASE)
    end = start + 200 + end_m.start() if end_m else min(start + 250000, len(text))
    return text[start:end]

  return None


def diff_10k(ticker: str, item: str = '1A',
             current_year: Optional[int] = None,
             prior_year: Optional[int] = None,
             max_changes: int = 20) -> Dict[str, Any]:
  """Diff Item 1A (risk factors) or Item 7 (MD&A) across two years of 10-K
  filings. Returns added and removed paragraphs.

  Use for: detecting new risk factors a company has added vs prior year
  (e.g. AI safety, supply chain disruption, regulatory exposure) — the
  filing tells you what management thinks has changed, before consensus.

  Default behavior: diff latest 10-K vs prior 10-K. Override with
  current_year/prior_year for specific comparisons.
  """
  _require_identity()

  try:
    company = Company(ticker)
    filings = list(company.get_filings(form='10-K').head(10))
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'company/filings lookup failed: {type(e).__name__}: {e}'}

  if len(filings) < 2:
    return {'ticker': ticker, 'success': False,
            'error': _filing_miss(
              ticker, '10-K',
              f'Need 2+ 10-K filings to diff, found {len(filings)}')}

  def _filing_year(f):
    return int(str(f.filing_date)[:4])

  current_f = None
  prior_f = None
  if current_year is not None or prior_year is not None:
    for f in filings:
      y = _filing_year(f)
      if current_year is not None and y == current_year and current_f is None:
        current_f = f
      if prior_year is not None and y == prior_year and prior_f is None:
        prior_f = f
  if current_f is None:
    current_f = filings[0]
  if prior_f is None:
    # Pick the filing that's not the current one and has a different year
    cy = _filing_year(current_f)
    for f in filings[1:]:
      if _filing_year(f) != cy:
        prior_f = f
        break
    if prior_f is None:
      prior_f = filings[1]

  if item not in ('1A', '7'):
    return {'ticker': ticker, 'success': False,
            'error': f'Unsupported item {item!r} — currently 1A and 7 are supported'}

  cur_text = _extract_section_from_filing_obj(current_f, item)
  pri_text = _extract_section_from_filing_obj(prior_f, item)
  if not cur_text or not pri_text:
    return {'ticker': ticker, 'success': False,
            'error': 'Could not extract Item section from one or both filings',
            'current_section_extracted': cur_text is not None,
            'prior_section_extracted': pri_text is not None}

  # Paragraph-level diff. Split on blank lines (\n\s*\n) then normalize
  # whitespace so the diff isn't dominated by formatting drift.
  def _paragraphs(text: str) -> list:
    paras = re.split(r'\n\s*\n', text)
    out = []
    for p in paras:
      n = re.sub(r'\s+', ' ', p).strip()
      if len(n) > 40:  # skip page headers, short labels
        out.append(n)
    return out

  cur_paras = _paragraphs(cur_text)
  pri_paras = _paragraphs(pri_text)

  import difflib
  matcher = difflib.SequenceMatcher(a=pri_paras, b=cur_paras, autojunk=False)
  added: list = []
  removed: list = []
  changed: list = []
  for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == 'insert':
      added.extend(cur_paras[j1:j2])
    elif tag == 'delete':
      removed.extend(pri_paras[i1:i2])
    elif tag == 'replace':
      # Treat as change pairs (best-effort align)
      pri_chunk = pri_paras[i1:i2]
      cur_chunk = cur_paras[j1:j2]
      n = min(len(pri_chunk), len(cur_chunk))
      for k in range(n):
        changed.append({'before': pri_chunk[k][:600], 'after': cur_chunk[k][:600]})
      if len(cur_chunk) > n:
        added.extend(cur_chunk[n:])
      if len(pri_chunk) > n:
        removed.extend(pri_chunk[n:])

  return {
    'ticker':                  ticker,
    'success':                 True,
    'error':                   None,
    'item':                    item,
    'current_filing_date':     str(current_f.filing_date),
    'prior_filing_date':       str(prior_f.filing_date),
    'current_section_length':  len(cur_text),
    'prior_section_length':    len(pri_text),
    'current_paragraph_count': len(cur_paras),
    'prior_paragraph_count':   len(pri_paras),
    'added_count':             len(added),
    'removed_count':           len(removed),
    'changed_count':           len(changed),
    'added_paragraphs':        [p[:600] for p in added[:max_changes]],
    'removed_paragraphs':      [p[:600] for p in removed[:max_changes]],
    'changed_paragraphs':      changed[:max_changes],
  }


def get_supply_chain(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract supply-chain / competitor mentions from 10-K Item 1 (Business).

  Two extraction layers:
    1. Curated-name match — scans Item 1 body text for a list of well-known
       company names (~150 entries). Returns matched tickers with mention
       counts and a sample context sentence each.
    2. Trigger-phrase extraction — returns sentences containing supply-
       chain language ('compete with', 'rely on', 'suppliers include',
       'customers include') so the analyst can see context even when no
       specific company names match.

  Note: software/services companies often describe competitors by category
  ('identity vendors', 'security solution vendors') rather than by name —
  use the trigger sentences in those cases. Hardware/semi/auto companies
  tend to name suppliers and customers explicitly.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data:
      return {'ticker': ticker, 'success': False,
              'error': _filing_miss(ticker, form_type,
                                    f'No {form_type} filing found for {ticker}')}
    filing_obj = filing_data.get('filing_object')
    if filing_obj is None:
      return {'ticker': ticker, 'success': False, 'error': 'no filing object'}
    text = filing_obj.text()
    if not text:
      return {'ticker': ticker, 'success': False, 'error': 'empty text'}

    # Locate Item 1 body: skip past TOC (offset > 7500), end at Item 1A body header
    m1a = re.search(r'ITEM\s+1A\.?\s+(?:RIS\s?K|RISK)\s+FACTORS',
                    text[30000:], re.IGNORECASE)
    item1_end = 30000 + m1a.start() if m1a else min(60000, len(text))
    item1_start = 7500
    item1 = text[item1_start:item1_end]
    if len(item1) < 1000:
      return {'ticker': ticker, 'success': False,
              'error': f'Item 1 body too short ({len(item1)} chars) — header pattern mismatch'}

    # Layer 1: curated name match
    self_name = ticker.upper()
    related_companies = []
    seen_tickers = set([self_name])
    for name_lower, mapped_ticker in _COMPANY_NAME_TO_TICKER.items():
      if mapped_ticker == self_name:
        continue
      # Use word boundary, escape regex chars
      pat = re.compile(rf'\b{re.escape(name_lower)}\b', re.IGNORECASE)
      matches = list(pat.finditer(item1))
      if not matches:
        continue
      # Sample context: first 200 chars around first mention
      m0 = matches[0]
      ctx_start = max(0, m0.start() - 100)
      ctx_end = min(len(item1), m0.end() + 200)
      context = re.sub(r'\s+', ' ', item1[ctx_start:ctx_end]).strip()
      if mapped_ticker in seen_tickers:
        # Aggregate: bump count on existing entry
        for r in related_companies:
          if r['ticker'] == mapped_ticker:
            r['mention_count'] += len(matches)
        continue
      seen_tickers.add(mapped_ticker)
      related_companies.append({
        'name_matched':   name_lower,
        'ticker':         mapped_ticker,
        'mention_count':  len(matches),
        'sample_context': context[:400],
      })
    related_companies.sort(key=lambda r: r['mention_count'], reverse=True)

    # Layer 2: trigger-phrase sentences
    triggers = [
      ('compete with',          r'[^.\n]*\bcompete[sd]?\s+with\b[^.\n]*\.'),
      ('competitors include',   r'[^.\n]*\bcompetitors\s+include\b[^.\n]*\.'),
      ('suppliers include',     r'[^.\n]*\bsuppliers?\s+include\b[^.\n]*\.'),
      ('customers include',     r'[^.\n]*\bcustomers?\s+include\b[^.\n]*\.'),
      ('rely on',               r'[^.\n]*\brely\s+on\b[^.\n]*\.'),
      ('partner with',          r'[^.\n]*\bpartner(?:s|ed|ing)?\s+with\b[^.\n]*\.'),
    ]
    trigger_sentences = []
    for label, pat in triggers:
      for m in re.finditer(pat, item1, re.IGNORECASE):
        s = re.sub(r'\s+', ' ', m.group(0)).strip()
        if 30 < len(s) < 500:
          trigger_sentences.append({'trigger': label, 'sentence': s})
        if len(trigger_sentences) >= 25:
          break
      if len(trigger_sentences) >= 25:
        break

    return {
      'ticker':           ticker.upper(),
      'success':          True,
      'error':            None,
      'item1_length_chars': len(item1),
      'related_companies':  related_companies,
      'related_count':      len(related_companies),
      'trigger_sentences':  trigger_sentences,
      'trigger_count':      len(trigger_sentences),
      'filing_date':        filing_data.get('filing_date'),
      'note':               'Curated name match covers ~150 mega-caps. Software/services 10-Ks often describe competitors by category, not by name — trigger_sentences captures that context.',
    }

  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'get_supply_chain failed: {type(e).__name__}: {e}'}


def get_company_filings_history(ticker: str, form_type: str = '10-K',
                                n: int = 5) -> Dict[str, Any]:
  """Return the last N filings of a given form type for a company.

  Generalizes get_latest_filing to return historical filings — useful for
  YoY 10-K comparisons (e.g. detecting new risk factors), tracking 8-K
  cadence, or backfilling time-series financial data from older filings.

  Returns metadata only (date, accession, URL, form, has_xbrl) — does not
  download or parse XBRL/text. Use the other extractors with specific
  accession numbers for content.
  """
  _require_identity()

  try:
    company = Company(ticker)
    filings = company.get_filings(form=form_type)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'company/filings fetch failed: {type(e).__name__}: {e}'}

  if not filings:
    return {'ticker': ticker, 'success': False,
            'error': _filing_miss(
                ticker, form_type,
                f'No {form_type} filings found for {ticker}')}

  out_filings = []
  try:
    for f in filings.head(n):
      # URL access
      url = None
      for attr in ['filing_url', 'url', 'filing_details_url', 'document_url']:
        if hasattr(f, attr):
          try:
            v = getattr(f, attr)
            if isinstance(v, str) and v:
              url = v
              break
          except Exception:
            pass
      # XBRL availability — try briefly
      has_xbrl = False
      try:
        x = f.xbrl()
        has_xbrl = x is not None
      except Exception:
        has_xbrl = False
      out_filings.append({
        'filing_date':      str(f.filing_date),
        'form':             f.form,
        'accession_number': f.accession_number,
        'url':              url,
        'has_xbrl_data':    has_xbrl,
      })
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'filings iteration failed: {type(e).__name__}: {e}',
            'partial': out_filings}

  return {
    'ticker':           ticker,
    'success':          True,
    'error':            None,
    'form_type':        form_type,
    'filings_returned': len(out_filings),
    'filings':          out_filings,
  }


def get_patent_filings(company_name: str, years_back: int = 5,
                       sample_count: int = 5) -> Dict[str, Any]:
  """Patent filing counts and recent samples from Google Patents.

  Google Patents aggregates USPTO + EPO + WIPO + national patents and
  exposes a public JSON endpoint at /xhr/query. This tool returns total
  patent count for the assignee, year-by-year counts for the last N years
  (R&D output proxy), and a small sample of recent patents.

  Note: patents publish ~18 months after filing, so 'recent' patent
  counts lag real R&D output. Use the trend across years (not the
  absolute most-recent count) as the signal.
  """
  import requests as _req
  from datetime import datetime as _dt

  if not company_name:
    return {'company_name': company_name, 'success': False, 'error': 'no company_name'}

  url = 'https://patents.google.com/xhr/query'
  headers = {'User-Agent': 'Mozilla/5.0 (compatible; nemo-ib/1.0)'}

  def _query(qs: str) -> Dict[str, Any]:
    """{'ok': True, 'payload': ...} or {'ok': False, 'reason': '<cause>'}.

    Google throttles this endpoint with a 503 "Sorry..." page. Collapsing that,
    a 404 and a DNS failure into a bare None made every failure read the same
    and left the caller unable to tell "retry later" from "wrong assignee".
    """
    try:
      r = _req.get(url, params={'url': qs, 'exp': ''},
                   headers=headers, timeout=20)
    except Exception as exc:
      return {'ok': False, 'reason': f'{type(exc).__name__}: {str(exc)[:150]}'}
    if r.status_code != 200:
      return {'ok': False, 'reason': f'HTTP {r.status_code}'}
    try:
      return {'ok': True, 'payload': r.json()}
    except ValueError as exc:
      return {'ok': False,
              'reason': f'HTTP 200 with a non-JSON body: {str(exc)[:150]}'}

  # Total count for assignee
  base_q = f'assignee={company_name}'
  total = _query(base_q + '&num=10')
  if not total['ok']:
    return {'company_name': company_name, 'success': False,
            'error': f"Google Patents query failed: {total['reason']}"}
  total_payload = total['payload']

  total_results = total_payload.get('results', {}).get('total_num_results', 0)

  # Year-by-year breakdown (last N years of grant dates)
  this_year = _dt.now().year
  year_counts = []
  failed_years = []
  for y in range(this_year - years_back, this_year + 1):
    # Patents granted within calendar year y
    yq = f'assignee={company_name}&after=publication:{y}0101&before=publication:{y}1231&num=1'
    res = _query(yq)
    if res['ok']:
      year_counts.append({
        'year': y,
        'count': res['payload'].get('results', {}).get('total_num_results', 0),
      })
    else:
      # A skipped year silently truncates the R&D trend, which is the whole
      # signal this tool exists to produce. Name the gap instead.
      failed_years.append({'year': y, 'reason': res['reason']})

  # Recent sample
  recent = []
  for cluster in total_payload.get('results', {}).get('cluster', []):
    for r in cluster.get('result', [])[:sample_count]:
      patent = r.get('patent', {})
      recent.append({
        'id':           r.get('id'),
        'title':        (patent.get('title') or '').strip(),
        'snippet':      (patent.get('snippet') or '').strip()[:300],
        'publication_date': patent.get('publication_date'),
        'priority_date':    patent.get('priority_date'),
        'assignee':         patent.get('assignee'),
        'inventor':         patent.get('inventor'),
      })
      if len(recent) >= sample_count:
        break
    if len(recent) >= sample_count:
      break

  return {
    'company_name':   company_name,
    'success':        True,
    'error':          None,
    'total_patents':  total_results,
    'year_counts':    year_counts,
    'failed_years':   failed_years,
    'recent_sample':  recent,
    'source':         'Google Patents /xhr/query (USPTO + EPO + WIPO + national)',
    'note':           'Patents publish ~18 months after filing. Year_counts reflect publication year, not filing year. Trend across years is the cleaner R&D signal than absolute most-recent year.',
  }


# Lexicon for earnings-release sentiment scoring. Lists below are loose —
# the goal is YoY tonal change, not absolute classification. Word lists
# adapted from Loughran-McDonald financial-text sentiment work plus
# observed earnings-release patterns.
_CONFIDENT_TERMS = (
  'record', 'strong', 'robust', 'momentum', 'accelerate', 'accelerating',
  'outperform', 'exceed', 'exceeded', 'beat', 'expanded', 'expansion',
  'achievement', 'milestone', 'breakthrough', 'leadership', 'optimistic',
  'pleased', 'confident', 'best', 'highest', 'increased', 'growth',
  'opportunity', 'opportunities', 'differentiated', 'innovative',
  'demand', 'attracting', 'winning', 'gained', 'gain', 'recovery',
)
_HEDGING_TERMS = (
  'uncertain', 'uncertainty', 'cautious', 'softness', 'soft', 'weakness',
  'weak', 'slow', 'slower', 'slowdown', 'declined', 'decline', 'decreased',
  'pressure', 'headwind', 'headwinds', 'challenging', 'challenges',
  'difficult', 'volatile', 'volatility', 'disrupt', 'disruption',
  'mixed', 'transition', 'rebalance', 'reduced', 'reduction', 'lower',
  'below', 'miss', 'shortfall', 'impair', 'impairment', 'restructur',
  'layoff', 'workforce reduction', 'pull-forward', 'pull-in',
)
_FUTURE_TERMS = (
  'will', 'expect', 'expects', 'anticipate', 'plan', 'forecast',
  'guidance', 'outlook', 'next year', 'coming year', 'fiscal',
  'long-term', 'medium-term', 'next quarter', 'going forward',
)


def extract_call_sentiment(ticker: str, quarters: int = 4) -> Dict[str, Any]:
  """Score sentiment over the last N quarterly earnings releases.

  Counts confident terms (record, strong, momentum) vs hedging terms
  (uncertainty, softness, headwinds) per release. Computes a net score
  (confident - hedging) per quarter and a YoY tonal shift signal.

  Limitations: regex word-counting, not real NLP. Captures gross tone
  shifts (e.g. CFO switching to "challenging environment" from "record
  quarter") which is what's most actionable. Subtle sentiment is missed.
  """
  releases_result = get_earnings_releases(ticker, max_quarters=quarters,
                                          max_chars_per_release=200000)
  if not releases_result.get('success'):
    return releases_result

  scores = []
  for rel in releases_result.get('releases', []):
    text = rel.get('text') or ''
    if not text:
      continue
    text_lower = text.lower()
    # Word-boundary count for each lexicon term
    def _count(terms):
      total = 0
      hits = {}
      for term in terms:
        n = len(re.findall(rf'\b{re.escape(term)}\b', text_lower))
        total += n
        if n > 0:
          hits[term] = n
      return total, hits

    conf_total, conf_hits = _count(_CONFIDENT_TERMS)
    hedge_total, hedge_hits = _count(_HEDGING_TERMS)
    future_total, _ = _count(_FUTURE_TERMS)

    # Word count for normalization
    words = len(re.findall(r'\b[a-z]+\b', text_lower))
    word_count_kw = max(words, 1) / 1000.0

    net = conf_total - hedge_total
    # Top hedging terms surface what's worrying management
    top_hedges = sorted(hedge_hits.items(), key=lambda kv: -kv[1])[:5]
    top_confs = sorted(conf_hits.items(), key=lambda kv: -kv[1])[:5]

    scores.append({
      'filing_date':       rel.get('filing_date'),
      'confident_count':   conf_total,
      'hedging_count':     hedge_total,
      'future_count':      future_total,
      'net_score':         net,
      'confident_per_1k_words': round(conf_total / word_count_kw, 2),
      'hedging_per_1k_words':   round(hedge_total / word_count_kw, 2),
      'word_count':        words,
      'top_hedging_terms': dict(top_hedges),
      'top_confident_terms': dict(top_confs),
    })

  if len(scores) < 2:
    return {
      'ticker': ticker, 'success': True, 'error': None,
      'quarters_scored': len(scores), 'scores': scores,
      'note': 'Need 2+ quarters to compute YoY tonal shift.',
    }

  # YoY tonal shift: compare latest quarter to ~4 quarters ago
  latest = scores[0]
  yoy_ref = scores[3] if len(scores) >= 4 else scores[-1]
  qoq_ref = scores[1]
  net_yoy_delta = latest['net_score'] - yoy_ref['net_score']
  hedging_yoy_delta = latest['hedging_per_1k_words'] - yoy_ref['hedging_per_1k_words']
  confident_yoy_delta = latest['confident_per_1k_words'] - yoy_ref['confident_per_1k_words']

  # Signal classifier
  signal = 'stable'
  if hedging_yoy_delta >= 1.0 and confident_yoy_delta <= -1.0:
    signal = 'tone_deteriorating_strong'
  elif hedging_yoy_delta >= 0.5 or confident_yoy_delta <= -0.5:
    signal = 'tone_deteriorating'
  elif hedging_yoy_delta <= -0.5 and confident_yoy_delta >= 0.5:
    signal = 'tone_improving_strong'
  elif hedging_yoy_delta <= -0.3 or confident_yoy_delta >= 0.3:
    signal = 'tone_improving'

  return {
    'ticker': ticker,
    'success': True,
    'error': None,
    'quarters_scored': len(scores),
    'scores': scores,
    'yoy_shift': {
      'net_score_delta':     net_yoy_delta,
      'hedging_per_1k_delta': round(hedging_yoy_delta, 2),
      'confident_per_1k_delta': round(confident_yoy_delta, 2),
      'compared_periods':    f'{latest["filing_date"]} vs {yoy_ref["filing_date"]}',
    },
    'signal': signal,
    'note': "Regex word counting; YoY delta in hedging-words-per-1k-words is the cleanest tonal shift signal. tone_deteriorating = CFO using more hedging language YoY.",
  }


def get_earnings_releases(ticker: str, max_quarters: int = 4,
                          max_chars_per_release: int = 50000) -> Dict[str, Any]:
  """Fetch the last N quarterly earnings press releases as filed with the SEC.

  Source path: companies file an 8-K with Item 2.02 (Results of Operations
  and Financial Condition) attaching the press release as EX-99.1. This is
  the SEC-authoritative equivalent of a paid transcript service's
  prepared-remarks section — same prose, written by the company, filed
  publicly under SEC penalty of perjury.

  Q&A from the analyst call is NOT in the 8-K — that lives in paid
  transcript databases (AlphaSense, Refinitiv) or syndicated services
  (Motley Fool, Seeking Alpha). This tool returns the prepared remarks
  + the key-metrics table that always opens the release.
  """
  _require_identity()

  try:
    company = Company(ticker)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'Company lookup failed: {type(e).__name__}: {e}'}

  releases = []
  try:
    filings = company.get_filings(form='8-K').head(30)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'get_filings failed: {type(e).__name__}: {e}'}

  for f in filings:
    if len(releases) >= max_quarters:
      break
    try:
      do = f.data_object()
      items = list(do.items) if do and do.items else []
    except Exception:
      items = []

    has_results = any('2.02' in i for i in items)
    if not has_results:
      continue

    # Find EX-99.1 (the earnings release attachment)
    ex99_text = None
    ex99_doc = None
    try:
      for a in f.attachments:
        doc = (getattr(a, 'document', '') or '').lower()
        descr = (getattr(a, 'description', '') or '').lower()
        if 'ex99' in doc or 'ex-99' in descr:
          try:
            ex99_text = a.text() if callable(getattr(a, 'text', None)) else None
            ex99_doc = a.document
          except Exception:
            ex99_text = None
          break
    except Exception:
      pass

    releases.append({
      'filing_date':       str(f.filing_date),
      'accession_number':  f.accession_number,
      'items':             items,
      'attachment_doc':    ex99_doc,
      'text':              (ex99_text[:max_chars_per_release] if ex99_text else None),
      'text_length_chars': len(ex99_text) if ex99_text else 0,
      'text_truncated':    bool(ex99_text and len(ex99_text) > max_chars_per_release),
      'filing_url':        getattr(f, 'filing_url', None),
    })

  if not releases:
    return {'ticker': ticker, 'success': False,
            'error': f'No 8-K Item 2.02 filings found in last 30 8-Ks for {ticker}'}

  return {
    'ticker': ticker,
    'success': True,
    'error': None,
    'source': '8-K Item 2.02 (Results of Operations) — EX-99.1 press release attachment',
    'releases': releases,
    'release_count': len(releases),
    'note': 'Prepared remarks only. Analyst Q&A requires a paid transcript service.',
  }


def extract_mda(ticker: str, form_type: str = '10-K',
                max_chars: int = 80000) -> Dict[str, Any]:
  """Extract 10-K Item 7 (Management's Discussion and Analysis) full text and
  detect sub-section headings.

  Item 7 covers Executive Summary, Results of Operations, Liquidity &
  Capital Resources, Critical Accounting Estimates. Companies vary heading
  format and ordering, so detection is keyword-based. Returns full text
  bounded to max_chars plus heading list with offsets.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data:
      return {'ticker': ticker, 'success': False,
              'error': _filing_miss(
                  ticker, form_type,
                  f'No {form_type} filing found for {ticker}')}

    filing_obj = filing_data.get('filing_object')
    if filing_obj is None:
      return {'ticker': ticker, 'success': False,
              'error': 'No filing object in cache'}

    text = filing_obj.text()
    if not text:
      return {'ticker': ticker, 'success': False, 'error': 'Empty filing text'}

    # Body header. Apostrophe in "Management's" is sometimes Unicode 0x2019,
    # sometimes 0x27, sometimes encoded as '?' after charset translation.
    # Allow any single char (or none) between "MANAGEMENT" and "S DISCUSSION".
    header_match = re.search(
      r'ITEM\s+7\.?\s+MANAGEMENT.{0,5}S?\s+DISCUSSION', text, re.IGNORECASE)
    if not header_match:
      header_match = re.search(r'ITEM\s+7\b\s*\n', text[50000:])
      if header_match:
        header_match = type('M', (), {
          'start': lambda self=None, off=50000 + header_match.start(): off
        })()
    if not header_match:
      return {'ticker': ticker, 'success': False,
              'error': 'Could not locate Item 7 header in filing text'}

    # Filings can have the heading appear in the TOC and again in the body.
    # If the first match is before offset 30k, find a later occurrence.
    start = header_match.start()
    if start < 30000:
      next_m = re.search(r'ITEM\s+7\.?\s+MANAGEMENT.{0,5}S?\s+DISCUSSION',
                         text[30000:], re.IGNORECASE)
      if next_m:
        start = 30000 + next_m.start()

    # End at Item 7A or Item 8 (Financial Statements)
    end_m = re.search(r'ITEM\s+7A\b', text[start + 200:], re.IGNORECASE)
    if not end_m:
      end_m = re.search(r'ITEM\s+8\b', text[start + 200:], re.IGNORECASE)
    end = (start + 200 + end_m.start()) if end_m else min(start + 250000, len(text))
    section = text[start:end]

    # Sub-section heading detection. MD&A headings are typically Title Case
    # (not all-caps like risk factors), so we look for both.
    MDA_KEYWORDS = (
      'OVERVIEW', 'EXECUTIVE SUMMARY', 'HIGHLIGHTS', 'RESULTS OF OPERATIONS',
      'OPERATING SEGMENT', 'SEGMENT RESULTS', 'PRODUCTIVITY', 'INTELLIGENT CLOUD',
      'MORE PERSONAL', 'OPERATING EXPENSES', 'COST OF REVENUE',
      'OPERATING INCOME', 'LIQUIDITY', 'CAPITAL RESOURCES',
      'CASH FLOW', 'CONTRACTUAL OBLIGATIONS', 'OFF-BALANCE',
      'CRITICAL ACCOUNTING', 'RECENT ACCOUNTING', 'ECONOMIC CONDITIONS',
      'METRICS', 'GROSS MARGIN', 'COMMITMENTS', 'CAPITAL EXPENDITURES',
      'NON-GAAP', 'INFLATION', 'CONSTANT CURRENCY', 'REVENUE',
    )
    headings = []
    seen = set()
    char_offset = 0
    for line in section.split('\n'):
      s = line.strip()
      if 4 <= len(s) <= 150 and not any(c.isdigit() for c in s):
        upper = s.upper()
        # Match keyword AND require the line to look like a heading
        # (not a sentence) — short, no trailing punctuation other than colon
        if any(kw in upper for kw in MDA_KEYWORDS) and not s.endswith('.'):
          # Skip page-header noise
          if re.match(r'ITEM\s+7$', s, re.IGNORECASE) or len(s) < 5:
            char_offset += len(line) + 1
            continue
          key = re.sub(r'\s+', ' ', s).strip().upper()
          if key not in seen:
            seen.add(key)
            headings.append({
              'heading': re.sub(r'\s+', ' ', s).strip(),
              'offset_in_section': char_offset,
            })
      char_offset += len(line) + 1

    truncated = len(section) > max_chars
    text_out = section[:max_chars]

    return {
      'ticker': ticker,
      'success': True,
      'error': None,
      'item': '7',
      'section_length_chars': len(section),
      'text': text_out,
      'text_truncated': truncated,
      'section_headings': headings,
      'heading_count': len(headings),
      'filing_date': filing_data.get('filing_date'),
      'filing_url': filing_data.get('url'),
    }

  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'extract_mda failed: {type(e).__name__}: {e}'}


def extract_risk_factors(ticker: str, form_type: str = '10-K',
                         max_chars: int = 80000) -> Dict[str, Any]:
  """Extract 10-K Item 1A Risk Factors with optional sub-section detection.

  Locates the body header (handles both 'RISK FACTORS' and SEC's
  letter-spaced 'RIS K FACTORS' variant) and slices to the next Item 1B.
  Detects uppercase sub-section headings (e.g. 'CYBERSECURITY, DATA
  PRIVACY, AND PLATFORM ABUSE RISKS') so consumers can navigate without
  re-parsing.

  Returns full text (truncated to max_chars to keep MCP payloads bounded)
  plus a list of detected section headings with character offsets.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data:
      return {'ticker': ticker, 'success': False,
              'error': _filing_miss(
                  ticker, form_type,
                  f'No {form_type} filing found for {ticker}')}

    filing_obj = filing_data.get('filing_object')
    if filing_obj is None:
      return {'ticker': ticker, 'success': False,
              'error': 'No filing object in cache'}

    text = filing_obj.text()
    if not text:
      return {'ticker': ticker, 'success': False, 'error': 'Empty filing text'}

    # Body header — SEC filers sometimes letter-space "RISK" as "RIS K" in
    # HTML; allow optional whitespace inside.
    header_match = re.search(r'ITEM\s+1A\.?\s+(?:RIS\s?K|RISK)\s+FACTORS',
                             text, re.IGNORECASE)
    if not header_match:
      # Fallback: any standalone "RISK FACTORS" header after offset 30k (past TOC)
      tail = text[30000:]
      m = re.search(r'(?:RIS\s?K|RISK)\s+FACTORS', tail)
      if m:
        header_match = type('M', (), {'start': lambda self=None, off=30000 + m.start(): off})()
      else:
        return {'ticker': ticker, 'success': False,
                'error': 'Could not locate Item 1A header in filing text'}

    start = header_match.start()
    # End at Item 1B (Unresolved Staff Comments) — first occurrence after start
    end_m = re.search(r'ITEM\s+1B\b', text[start + 200:], re.IGNORECASE)
    end = (start + 200 + end_m.start()) if end_m else min(start + 200000, len(text))
    section = text[start:end]

    # Sub-section heading detection. Look for lines that:
    #   - have 70%+ uppercase letters
    #   - length 8-150 chars
    #   - don't include digits (skip "Item 1A" markers / pagination)
    #   - typically contain 'RISKS' or known risk-category keywords
    KEYWORDS = ('RISK', 'OPERATIONS', 'PRODUCT', 'BUSINESS', 'STRATEGIC',
                'LEGAL', 'CYBER', 'PRIVACY', 'REGULATORY', 'FINANCIAL',
                'INTERNATIONAL', 'COMPETITION', 'TALENT', 'GOVERNANCE',
                'CLIMATE', 'INTELLECTUAL', 'SECURITY')
    headings = []
    seen = set()
    char_offset_in_section = 0
    for line in section.split('\n'):
      s = line.strip()
      if 8 <= len(s) <= 150 and not any(c.isdigit() for c in s):
        letters = [c for c in s if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.70:
          # Skip the "ITEM 1A" repeated page header
          if re.match(r'ITEM\s+1A', s, re.IGNORECASE) and len(s) < 30:
            continue
          if any(kw in s.upper() for kw in KEYWORDS):
            key = re.sub(r'\s+', ' ', s).strip().upper()
            if key not in seen and len(key) > 6:
              seen.add(key)
              headings.append({
                'heading': re.sub(r'\s+', ' ', s).strip(),
                'offset_in_section': char_offset_in_section,
              })
      # advance offset by line length + 1 for the newline
      char_offset_in_section += len(line) + 1

    # Truncate output text to bounded length for MCP transport
    truncated = len(section) > max_chars
    text_out = section[:max_chars]

    return {
      'ticker': ticker,
      'success': True,
      'error': None,
      'item': '1A',
      'section_length_chars': len(section),
      'text': text_out,
      'text_truncated': truncated,
      'section_headings': headings,
      'heading_count': len(headings),
      'filing_date': filing_data.get('filing_date'),
      'filing_url': filing_data.get('url'),
    }

  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'extract_risk_factors failed: {type(e).__name__}: {e}'}


def track_segment_growth(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Compute YoY growth + multi-year CAGR per segment from the existing
  5-year segment history. Detects acceleration / deceleration by comparing
  the most recent YoY growth to the trailing 2y CAGR.

  Lets the analyst see at a glance:
    - which segments are accelerating (latest YoY > 2y CAGR)
    - which are decelerating (latest YoY < 2y CAGR)
    - operating-leverage signal (op income growth > revenue growth)
  """
  seg_result = get_segment_financials(ticker, form_type)
  if not seg_result.get('success'):
    return seg_result

  out_segments = []
  for seg in seg_result.get('segments', []):
    rev = seg.get('revenue', [])
    op = seg.get('operating_income', [])

    # Compute YoY series
    rev_yoy_series = []
    for i in range(len(rev) - 1):
      cur = rev[i]['value']
      prev = rev[i + 1]['value']
      if prev and cur:
        rev_yoy_series.append({
          'period_end': rev[i]['period_end'],
          'yoy_pct': round(((cur / prev) - 1) * 100, 2),
        })

    op_yoy_series = []
    for i in range(len(op) - 1):
      cur = op[i]['value']
      prev = op[i + 1]['value']
      if prev and cur:
        op_yoy_series.append({
          'period_end': op[i]['period_end'],
          'yoy_pct': round(((cur / prev) - 1) * 100, 2),
        })

    # CAGR over the full available history
    def _cagr(series):
      if len(series) < 2:
        return None
      latest = series[0]['value']
      oldest = series[-1]['value']
      years = len(series) - 1
      if not oldest or oldest <= 0 or not latest:
        return None
      return round(((latest / oldest) ** (1.0 / years) - 1) * 100, 2)

    rev_cagr = _cagr(rev)
    op_cagr = _cagr(op)

    # Operating margin trend
    op_margin_series = []
    for i in range(len(rev)):
      r_val = rev[i]['value']
      o_val = op[i]['value'] if i < len(op) else None
      if r_val and o_val:
        op_margin_series.append({
          'period_end': rev[i]['period_end'],
          'op_margin_pct': round((o_val / r_val) * 100, 2),
        })

    # Acceleration signal: compare latest YoY to multi-year CAGR
    latest_yoy = rev_yoy_series[0]['yoy_pct'] if rev_yoy_series else None
    accel_signal = 'unknown'
    accel_delta = None
    if latest_yoy is not None and rev_cagr is not None:
      accel_delta = round(latest_yoy - rev_cagr, 2)
      if accel_delta >= 3:
        accel_signal = 'accelerating'
      elif accel_delta <= -3:
        accel_signal = 'decelerating'
      else:
        accel_signal = 'stable'

    # Operating leverage: op growth > rev growth in latest period
    leverage_signal = 'unknown'
    if rev_yoy_series and op_yoy_series:
      r_yoy = rev_yoy_series[0]['yoy_pct']
      o_yoy = op_yoy_series[0]['yoy_pct']
      if o_yoy - r_yoy >= 2:
        leverage_signal = 'positive_operating_leverage'
      elif r_yoy - o_yoy >= 2:
        leverage_signal = 'margin_compression'
      else:
        leverage_signal = 'in_line'

    out_segments.append({
      'segment':              seg['segment'],
      'segment_member':       seg['segment_member'],
      'years_of_history':     len(rev),
      'revenue_series':       rev,
      'op_income_series':     op,
      'revenue_yoy_series':   rev_yoy_series,
      'op_income_yoy_series': op_yoy_series,
      'op_margin_series':     op_margin_series,
      'revenue_cagr_pct':     rev_cagr,
      'op_income_cagr_pct':   op_cagr,
      'latest_yoy_pct':       latest_yoy,
      'acceleration_delta':   accel_delta,
      'acceleration_signal':  accel_signal,
      'leverage_signal':      leverage_signal,
    })

  # Sort by acceleration delta — fastest accelerating first
  out_segments.sort(
    key=lambda s: (s.get('acceleration_delta') if s.get('acceleration_delta') is not None else -999),
    reverse=True
  )

  return {
    'ticker':       ticker,
    'success':      True,
    'error':        None,
    'segments':     out_segments,
    'segment_count': len(out_segments),
    'filing_date':  seg_result.get('filing_date'),
    'note':         "Acceleration signal compares latest YoY revenue growth to multi-year CAGR (delta >= +3 = accelerating, <= -3 = decelerating). Leverage signal compares op-income YoY to revenue YoY in the latest period.",
  }


# --------------------------------------------------------------- segment axis

# The axis a filer uses to say "this fact is the segment column of the
# reconciliation table" rather than a breakdown of it. Standard us-gaap, and
# the only extra dimension a segment's own revenue is allowed to carry: AAPL,
# BA, HON, JPM, NVDA and WFC tag every segment fact this way and nothing else,
# so a rule that demanded the segment axis alone would report all six as
# tagging no segment revenue.
CONSOLIDATION_ITEMS_AXIS = 'srt:ConsolidationItemsAxis'
OPERATING_SEGMENTS_MEMBER = 'us-gaap:OperatingSegmentsMember'

SEGMENT_AXIS_ONLY = 'segment axis only'
SEGMENT_OPERATING_COLUMN = (f'segment axis + {CONSOLIDATION_ITEMS_AXIS}='
                            f'{OPERATING_SEGMENTS_MEMBER}')
# Order is the preference used to break a coverage tie, best first.
_SEGMENT_CONTEXT_BASES = (SEGMENT_AXIS_ONLY, SEGMENT_OPERATING_COLUMN)

# A 10-K's segment note carries three comparative years; only annual durations
# are wanted from it, as before.
_SEGMENT_SPAN_DAYS = (350, None)

# Members can nest. CAT tags us-gaap:ReportableSegmentAggregationBeforeOther
# OperatingSegmentMember at 73,955m, exactly Construction 25,060 + Resource
# 12,474 + Power & Energy 32,201 + Financial Products 4,220, on the same axis as
# those four; AMT tags amt:PropertyMember above its five property regions; LEN
# tags one member that is the sum of five Homebuilding regions. Detected by
# comparing the sum against the consolidated fact already in hand rather than by
# trying to recognise a parent from its tag name, which cannot be done -- the
# same mechanism `forward_metrics.get_geographic_revenue` uses for geography.
#
# The band absorbs a filer that tags an intersegment-revenue member on the
# segment axis, which is genuinely additive to the parts and eliminated from the
# total. Measured on the identity-sweep basket the largest legitimate overshoot
# is WFC's 1.4% (GOOGL 0.03%, BA 0.2%); the overlapping filers run from 194% to
# 219%.
_SEGMENT_OVERLAP_TOLERANCE = 0.02


def _same_axis(left: str, right: str) -> bool:
  """Whether two axis spellings name the same axis.

  `get_unique_dimensions()` keys axes with '_' where a context's dimensions use
  ':'. Comparing the raw strings silently resolves nothing for every filer.
  """
  return str(left).replace('_', ':', 1) == str(right).replace('_', ':', 1)


def _segment_context_basis(dimensions: Dict[str, str], axis: str) -> Optional[str]:
  """Which selection basis a fact belongs to, or None if it is not a segment's
  own figure.

  A fact carrying the segment axis *plus another* axis is a piece of a segment
  or an adjustment to it -- a product line, a geography, an intersegment
  elimination, a corporate reconciling item -- and never the segment. GE's
  largest segment reported -62,000,000 of revenue because the elimination
  context answered a query for the segment; the segment-only context in the
  same filing carries 33,252,000,000.
  """
  # Both spellings of a prefix compare equal, as they do everywhere else in
  # this package: the same axis reaches us as 'srt:ConsolidationItemsAxis' from
  # a context and 'srt_ConsolidationItemsAxis' from the dimension index.
  extra = {str(a).replace('_', ':', 1): str(m).replace('_', ':', 1)
           for a, m in dimensions.items() if not _same_axis(a, axis)}
  if not extra:
    return SEGMENT_AXIS_ONLY
  if (len(extra) == 1
      and extra.get(CONSOLIDATION_ITEMS_AXIS) == OPERATING_SEGMENTS_MEMBER):
    return SEGMENT_OPERATING_COLUMN
  return None


def _segment_facts(xbrl, concept: str, axis: str) -> Dict[str, Dict[str, Dict[str, set]]]:
  """{basis: {member: {period_end: {values}}}} for one concept.

  Reads through `sec_series.concept_point`, so the exact-concept filter and the
  dimension resolution are the ones every other reader in this module uses
  rather than a third copy. `by_dimension(axis, member)`, which this replaces,
  returns the facts carrying a second axis alongside the segment's own.
  """
  from .sec_series import _in_span, _period_rank, concept_point
  try:
    point = concept_point(xbrl, concept, filing_date='', form='')
  except Exception:  # noqa: BLE001 - an unreadable concept is not a segment
    return {}
  if point is None:
    return {}

  out: Dict[str, Dict[str, Dict[str, set]]] = {}
  for fact in point.deduplicated():
    member = None
    for a, m in fact.dimensions.items():
      if _same_axis(a, axis):
        member = m
        break
    if member is None:
      continue
    period_end, days = _period_rank(fact.period)
    if not _in_span(days, *_SEGMENT_SPAN_DAYS):
      continue
    basis = _segment_context_basis(fact.dimensions, axis)
    if basis is None:
      continue
    (out.setdefault(basis, {}).setdefault(member, {})
        .setdefault(period_end, set()).add(fact.value))
  return out


def _resolve_segment_basis(xbrl, concepts, axis: str):
  """(concept, basis, {member: {period_end: {values}}}) or None.

  The concept and the basis are chosen **once for the filing**, by which pair
  resolves the most members, so every segment of one filer is measured the same
  way. Per-member preference mixes them: AMT tags five members' non-lease
  revenue on the segment axis alone and all seven members' total revenue on the
  operating-segments column, and taking each member's most specific fact puts
  935,900,000 beside 10,305,000,000 in one total.

  Ties break to the earlier concept -- the chain is broadest-first, for the
  reason `REVENUE_CONCEPTS` documents -- and then to the segment-only basis.
  """
  best = None
  for rank, concept in enumerate(concepts):
    bases = _segment_facts(xbrl, concept, axis)
    for preference, basis in enumerate(_SEGMENT_CONTEXT_BASES):
      by_member = bases.get(basis)
      if not by_member:
        continue
      key = (len(by_member), -rank, -preference)
      if best is None or key > best[0]:
        best = (key, concept, basis, by_member)
  return None if best is None else (best[1], best[2], best[3])


def _segment_period_series(by_period: Dict[str, set]) -> tuple:
  """([{period_end, value}] newest first, [(period_end, values)] in conflict).

  Two different values tagged for one member, one period and one basis cannot
  both be that segment's figure, and there is nothing in the frame to choose
  between them, so the period is dropped and named rather than guessed at.
  """
  rows, conflicts = [], []
  for period_end in sorted(by_period, reverse=True):
    values = by_period[period_end]
    if len(values) > 1:
      conflicts.append((period_end, sorted(values)))
      continue
    rows.append({'period_end': period_end, 'value': float(next(iter(values)))})
  return rows, conflicts




def get_segment_financials(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract per-segment revenue and operating income from latest 10-K XBRL.

  Uses the `us-gaap:StatementBusinessSegmentsAxis` (or any axis whose name
  contains 'Segment') and pulls the company-defined segment members. For each
  member the fact returned is the one that is the segment's own figure --
  qualified by the segment axis alone, or by the segment axis plus the
  operating-segments column of the reconciliation table where the filer tags
  nothing else. See `_segment_context_basis` and `_resolve_segment_basis`.

  Returns up to 5 years of history per segment plus the most recent YoY growth
  and operating margin. Critical for resolving the variant-perception question
  on multi-segment companies -- e.g. MSFT's Intelligent Cloud (Azure) growth vs.
  Productivity & Business Processes margin.

  Two things the caller has to read:

  * `members_overlap` -- true when the members sum to more than consolidated
    revenue, which means at least one of them aggregates the others. The values
    are still each correct; their sum is not a total, and the percentages are
    then of consolidated revenue rather than of the sum.
  * `segments_without_revenue` -- members whose revenue is not extractable as
    tagged. XOM tags every segment revenue fact in combination with
    `srt:StatementGeographicalAxis` and no segment-only fact exists, so its
    segment revenue is a real absence rather than a number to approximate.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'ticker': ticker, 'success': False,
              'error': _filing_miss(
                  ticker, form_type,
                  f'No {form_type} filing or XBRL data found for {ticker}')}

    xbrl = filing_data['xbrl_data']

    # Discover the segment axis. edgartools normalizes ':' to '_' in the
    # unique_dimensions dict keys, but the contexts carry the colon form.
    unique_dims = xbrl.facts.get_unique_dimensions()
    segment_axis_key = None
    segment_axis = None
    for key in unique_dims.keys():
      if 'StatementBusinessSegments' in key or key.endswith('SegmentsAxis'):
        segment_axis_key = key
        segment_axis = key.replace('_', ':', 1)
        break

    if not segment_axis_key:
      return {'ticker': ticker, 'success': False,
              'error': 'No business-segment axis in XBRL — company may not have reportable segments',
              'axes_available': list(unique_dims.keys())[:10]}

    members = sorted(unique_dims.get(segment_axis_key, set()))
    if not members:
      return {'ticker': ticker, 'success': False,
              'error': f'No segment members under {segment_axis_key}'}

    # Broadest element first, and shared with the consolidated reader below so
    # the two cannot disagree about what a filer tags. Trying the ASC 606
    # element first returns AMT's 935,900,000 of non-lease revenue against
    # 10,644,600,000 of revenue -- tower rent is lease income under ASC 842.
    resolved = _resolve_segment_basis(xbrl, REVENUE_CONCEPTS, segment_axis)
    if resolved is None:
      return {
        'ticker': ticker, 'success': False,
        'error': (
          f'{ticker} tags no segment revenue fact that is a segment\'s own '
          f'figure. Every fact on {segment_axis} carries a further dimension '
          f'-- a product, a geography, an intersegment elimination or a '
          f'corporate reconciling item -- or no revenue element is tagged on '
          f'that axis at all. Summing across the further axis is not something '
          f'this tool does, so segment revenue is not extractable as tagged '
          f'for {members}.'),
        'segment_axis': segment_axis,
        'segments_available': members,
        'total_latest_segment_revenue': None,
        'filing_date': filing_data.get('filing_date'),
      }
    revenue_concept, revenue_basis, revenue_by_member = resolved

    op_resolved = _resolve_segment_basis(
      xbrl, ('us-gaap:OperatingIncomeLoss',), segment_axis)
    op_basis = op_resolved[1] if op_resolved else None
    op_by_member = op_resolved[2] if op_resolved else {}

    # The consolidated fact to reconcile against, read from the same filing
    # through the same undimensioned selection get_revenue_base uses.
    consolidated = None
    consolidated_concept = None
    for concept in REVENUE_CONCEPTS:
      fact = _consolidated_fact(xbrl, concept, span_days=_SEGMENT_SPAN_DAYS)
      if fact is not None:
        consolidated = fact.value
        consolidated_concept = fact.concept or concept
        break

    segments_out = []
    unresolved = []
    for member in members:
      # Pretty name: "msft:ProductivityAndBusinessProcessesMember"
      # -> "Productivity And Business Processes"
      seg_short = member.split(':')[-1]
      if seg_short.endswith('Member'):
        seg_short = seg_short[:-len('Member')]
      seg_display = re.sub(r'([a-z])([A-Z])', r'\1 \2', seg_short).strip()

      rev_series, rev_conflicts = _segment_period_series(
        revenue_by_member.get(member, {}))
      op_series, _ = _segment_period_series(op_by_member.get(member, {}))

      unresolved_reason = None
      if not rev_series:
        if rev_conflicts:
          period, values = rev_conflicts[0]
          unresolved_reason = (
            f'{revenue_concept} is tagged more than once for {period} on the '
            f'{revenue_basis} basis ({values}); which of them is the segment\'s '
            f'revenue cannot be determined from the filing')
        else:
          unresolved_reason = (
            f'no {revenue_concept} fact for this member on the {revenue_basis} '
            f'basis this filing is read on; the facts it does tag carry a '
            f'further dimension')
        unresolved.append({'segment': seg_display, 'segment_member': member,
                           'reason': unresolved_reason})

      latest_rev = rev_series[0]['value'] if rev_series else None
      prev_rev = rev_series[1]['value'] if len(rev_series) > 1 else None
      rev_yoy_pct = round(((latest_rev / prev_rev) - 1) * 100, 2) \
        if (latest_rev and prev_rev) else None

      latest_op = op_series[0]['value'] if op_series else None
      prev_op = op_series[1]['value'] if len(op_series) > 1 else None
      op_yoy_pct = round(((latest_op / prev_op) - 1) * 100, 2) \
        if (latest_op and prev_op) else None
      op_margin_pct = round((latest_op / latest_rev) * 100, 2) \
        if (latest_op and latest_rev) else None


      segments_out.append({
        'segment': seg_display,
        'segment_member': member,
        'latest_period_end': rev_series[0]['period_end'] if rev_series else None,
        'revenue': rev_series[:5],
        'operating_income': op_series[:5],
        'revenue_yoy_pct': rev_yoy_pct,
        'op_income_yoy_pct': op_yoy_pct,
        'op_margin_pct': op_margin_pct,
        'unresolved_reason': unresolved_reason,
      })

    total_seg_rev = sum(s['revenue'][0]['value'] for s in segments_out
                        if s['revenue'])


    members_overlap = None
    if consolidated:
      members_overlap = (
        total_seg_rev > consolidated * (1 + _SEGMENT_OVERLAP_TOLERANCE))
    denominator = consolidated if members_overlap else total_seg_rev

    for segment in segments_out:
      latest = segment['revenue'][0]['value'] if segment['revenue'] else None
      segment['pct_of_revenue'] = (
        latest / denominator * 100.0
        if (latest is not None and denominator) else None)

    if members_overlap:
      note = (
        f"This filer's segment members OVERLAP: they sum to "
        f"{total_seg_rev:,.0f} against consolidated revenue of "
        f"{consolidated:,.0f}, so at least one member aggregates the others "
        f"(CAT tags ReportableSegmentAggregationBeforeOtherOperatingSegment"
        f"Member beside the four segments it is the sum of). Percentages are "
        f"of consolidated revenue and therefore do NOT sum to 100. Read "
        f"individual segments, not the total, and do not add them together.")
    else:
      note = (
        "Percentages are of the disclosed segment total, which may sit below "
        "consolidated revenue: segments reconcile to it through unallocated "
        "corporate costs, intersegment eliminations and an 'all other' bucket "
        "a filer may not tag on the segment axis.")
    if unresolved:
      note += (
        f" {len(unresolved)} of {len(members)} members report no revenue "
        f"this tool will stand behind and are listed in "
        f"segments_without_revenue; the total excludes them.")

    return {
      'ticker': ticker,
      'success': True,
      'error': None,
      'segments': segments_out,
      'segment_axis': segment_axis,
      'segment_count': len(segments_out),
      'total_latest_segment_revenue': total_seg_rev,
      'revenue_concept_used': revenue_concept,
      'revenue_basis': revenue_basis,
      'operating_income_basis': op_basis,
      'consolidated_revenue': consolidated,
      'consolidated_concept_used': consolidated_concept,
      'members_overlap': members_overlap,
      'segments_without_revenue': unresolved,
      'note': note,
      'filing_date': filing_data.get('filing_date'),
    }

  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'get_segment_financials failed: {type(e).__name__}: {e}'}


def get_buyback_history(ticker: str, form_type: str = '10-K', max_years: int = 5) -> Dict[str, Any]:
  """Extract share repurchase (buyback) history from the latest 10-K XBRL.

  Mirrors the get_capex_pct_revenue / get_depreciation pattern. Returns the
  latest annual buyback as `ttm_repurchase` (the calculate_capital_returns
  consumer expects that field name, even though the value is the latest
  fiscal year's repurchases, not a rolling 4-quarter sum — comparable 10-Ks
  publish annual figures only).

  Concept priority follows GAAP usage:
    1. PaymentsForRepurchaseOfCommonStock  - cash-flow statement, most common
    2. StockRepurchasedAndRetiredDuringPeriodValue - equity statement variant
    3. TreasuryStockAcquiredCostOfSharesAcquired - treasury accounting variant
    4. PaymentsForRepurchaseOfEquity - generic equity buyback (preferred + common)
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'ticker': ticker,
              'error': _filing_miss(ticker, form_type,
                                    f'No filing found for {ticker}'),
              'success': False}

    xbrl = filing_data['xbrl_data']

    concepts = [
      'us-gaap:PaymentsForRepurchaseOfCommonStock',
      'us-gaap:StockRepurchasedAndRetiredDuringPeriodValue',
      'us-gaap:TreasuryStockAcquiredCostOfSharesAcquired',
      'us-gaap:PaymentsForRepurchaseOfEquity',
    ]

    concept_used = None
    annual_history: list = []  # [{period_end, value}, ...] sorted newest-first

    for concept in concepts:
      try:
        facts = xbrl.facts.query().by_concept(concept).to_dataframe()
      except Exception:
        continue
      if facts.empty:
        continue

      # Mirror filter_annual_data's period filter (10-K annual: 350+ days)
      facts['period_start_dt'] = pd.to_datetime(facts['period_start'])
      facts['period_end_dt'] = pd.to_datetime(facts['period_end'])
      facts['duration_days'] = (facts['period_end_dt'] - facts['period_start_dt']).dt.days
      annual = facts[facts['duration_days'] >= 350]
      if annual.empty:
        continue

      # For each unique period_end, take the largest absolute value to capture
      # the consolidated total (XBRL has segment + consolidated rows).
      rows = []
      for end_dt, group in annual.groupby('period_end_dt'):
        # Buybacks are reported positive on CF statement (outflow) per Finnhub
        # convention, but XBRL filings sometimes use negative. abs() normalizes.
        consolidated = group.loc[group['numeric_value'].abs().idxmax()]
        rows.append({
          'period_end': consolidated['period_end'],
          'value': abs(float(consolidated['numeric_value'])),
        })

      rows.sort(key=lambda r: r['period_end'], reverse=True)
      if rows:
        annual_history = rows[:max_years]
        concept_used = concept
        break

    if not annual_history:
      return {
        'ticker': ticker,
        'error': 'No buyback concept matched in latest filing — '
                 'company may have no repurchase program or uses a '
                 'non-standard XBRL concept',
        'success': False,
        'concepts_tried': concepts,
      }

    return {
      'ticker': ticker,
      'success': True,
      'error': None,
      'ttm_repurchase': annual_history[0]['value'],  # most recent FY total
      'annual_repurchases': annual_history,
      'concept_used': concept_used,
      'period_end': annual_history[0]['period_end'],
      'filing_date': filing_data.get('filing_date'),
    }

  except Exception as e:
    return {
      'ticker': ticker,
      'error': f'get_buyback_history failed: {type(e).__name__}: {e}',
      'success': False,
    }


# ---------------------------------------------------------------------------
# Forward-looking signal extractor
# ---------------------------------------------------------------------------
#
# Goal: scan recent earnings releases + 10-K MD&A for forward-looking language
# (guidance, capacity adds, multi-year plans) and surface structured excerpts.
# The regex layer is deterministic and exposed as a module-level helper
# `_scan_forward_signals` so it can be unit-tested in isolation.

FORWARD_PATTERNS: Dict[str, list] = {
  'guidance': [
    r'we (?:expect|anticipate|estimate|project|forecast)\s+[^.]{20,300}',
    r'(?:guidance|outlook) (?:for|of|in)\s+[^.]{20,300}',
    r'(?:we|management) believe[s]?\s+[^.]{20,300}',
  ],
  'capacity_addition': [
    r'(?:capacity|fab|plant)\s+(?:addition|expansion|build-out|ramp)[^.]{10,300}',
    r'new\s+(?:facility|factory|fab|plant|data\s+center)[^.]{10,300}',
  ],
  'capex_plan': [
    r'capex\s+(?:plan|guidance|commitment|outlook)[^.]{10,300}',
    r'capital expenditures? (?:will|expected|planned)[^.]{10,300}',
  ],
  'multi_year_commitment': [
    r'multi-?year[^.]{10,300}',
    r'long-?term (?:commitment|agreement|contract|plan)[^.]{10,300}',
    r'by (?:FY|fiscal\s+year\s+)?(?:20[2-3][0-9])[^.]{10,300}',
    r'over the next\s+(?:three|four|five|several|\d+)\s+(?:years|quarters)[^.]{10,300}',
  ],
  'backlog_orderbook': [
    r'backlog (?:grew|increased|reached|stood at|of|is now)[^.]{10,300}',
    r'orders?\s+(?:received|booked|pipeline|backlog)[^.]{10,300}',
    r'remaining performance obligation[^.]{10,300}',
  ],
  'product_roadmap': [
    r'next-gen[^.]{10,300}',
    r'(?:will|plan to)\s+(?:launch|introduce|release|ship)\s+[^.]{10,300}',
    r'in development[^.]{10,300}',
  ],
}

# Compile once at module load.
_FORWARD_COMPILED: Dict[str, list] = {
  cat: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats]
  for cat, pats in FORWARD_PATTERNS.items()
}


def _normalize_excerpt(text: str) -> str:
  """Collapse whitespace and strip — used both for excerpts and dedup keys."""
  return re.sub(r'\s+', ' ', text or '').strip()


def _scan_forward_signals(text: str, source: str,
                          filing_date: Optional[str] = None,
                          accession: Optional[str] = None,
                          context_chars: int = 200) -> list:
  """Run the FORWARD_PATTERNS regexes over a piece of text and return a list
  of signal dicts. Pure function; no I/O. Exposed so tests can hit the
  regex layer with synthetic strings.

  Each signal dict has: category, source, filing_date, accession, excerpt,
  match_text. `excerpt` is +/- context_chars around the match, whitespace-
  normalized.
  """
  if not text:
    return []

  signals: list = []
  text_len = len(text)
  for category, compiled_list in _FORWARD_COMPILED.items():
    for pat in compiled_list:
      for m in pat.finditer(text):
        start = max(0, m.start() - context_chars)
        end = min(text_len, m.end() + context_chars)
        excerpt = _normalize_excerpt(text[start:end])
        match_text = _normalize_excerpt(m.group(0))
        if not excerpt or not match_text:
          continue
        signals.append({
          'category':    category,
          'source':      source,
          'filing_date': filing_date,
          'accession':   accession,
          'excerpt':     excerpt,
          'match_text':  match_text,
        })
  return signals


def _dedupe_signals(signals: list, overlap_threshold: float = 0.8) -> list:
  """Drop signals whose excerpts overlap (substring containment) by more
  than `overlap_threshold` of the shorter excerpt with one we've already
  kept. O(N^2) in worst case but N is small (typically <500).
  """
  kept: list = []
  for s in signals:
    exc = s.get('excerpt') or ''
    if not exc:
      continue
    is_dup = False
    for k in kept:
      kexc = k.get('excerpt') or ''
      if not kexc:
        continue
      shorter, longer = (exc, kexc) if len(exc) <= len(kexc) else (kexc, exc)
      if not shorter:
        continue
      # Containment-style overlap: if 80%+ of the shorter excerpt appears
      # inside the longer one, treat as duplicate. We approximate by
      # sliding-window substring match on a leading prefix of the shorter
      # excerpt — cheap and good enough for typical guidance prose.
      probe_len = max(20, int(len(shorter) * overlap_threshold))
      probe = shorter[:probe_len]
      if probe and probe in longer:
        is_dup = True
        break
    if not is_dup:
      kept.append(s)
  return kept


def _ingest_signals_to_rag(ticker: str, signals: list) -> int:
  """Best-effort RAG ingest: chunk each excerpt, embed, store. Returns the
  number of chunks successfully inserted. Wrapped in try/except so any
  failure (missing sentence-transformers, sqlite-vec extension, etc.) is
  silent — extraction must not break because RAG is offline.
  """
  inserted = 0
  try:
    from agent.rag import chunker, embedder, store
  except Exception:
    return 0

  for idx, sig in enumerate(signals):
    try:
      excerpt = sig.get('excerpt') or ''
      if not excerpt:
        continue
      filing_date = sig.get('filing_date')
      source = sig.get('source') or 'forward_signal'
      accession = sig.get('accession') or ''
      # Stable, human-readable doc_id so re-runs are idempotent enough that
      # repeated ingests don't fan out chunk_ids forever. Include the index
      # because two excerpts in the same release can share metadata.
      doc_id = f'forward_signal_{ticker}_{source}_{accession or filing_date or "unknown"}_{idx}'

      chunks = chunker.chunk_text(
        excerpt,
        target_tokens=500,
        overlap_tokens=50,
        section_heading=sig.get('category'),
      )
      if not chunks:
        # Excerpts are short — chunker may return [] for very short text.
        # In that case ingest the excerpt itself as a single mini-chunk.
        chunks = [{
          'chunk_text':      excerpt,
          'chunk_offset':    0,
          'chunk_sequence':  0,
          'section_heading': sig.get('category'),
        }]

      for ch in chunks:
        try:
          vec = embedder.embed(ch['chunk_text'])
          store.insert_chunk({
            'doc_id':          doc_id,
            'ticker':          ticker,
            'source_tool':     'extract_forward_signals',
            'doc_type':        'forward_signal',
            'filing_date':     filing_date,
            'section_heading': ch.get('section_heading'),
            'chunk_text':      ch['chunk_text'],
            'chunk_offset':    ch.get('chunk_offset', 0),
            'chunk_sequence':  ch.get('chunk_sequence', 0),
          }, vec)
          inserted += 1
        except Exception:
          # Single-chunk failure: keep going.
          continue
    except Exception:
      continue
  return inserted


def extract_forward_signals(ticker: str,
                            lookback_quarters: int = 4) -> Dict[str, Any]:
  """Scan recent earnings releases + the latest 10-K MD&A for forward-
  looking language and return structured excerpts.

  Pipeline:
    1. get_earnings_releases(ticker, max_quarters=lookback_quarters)
    2. extract_mda(ticker)
    3. For each text source run FORWARD_PATTERNS regexes
    4. Capture +/- 200 chars of surrounding context, normalize whitespace
    5. Deduplicate near-identical excerpts (containment overlap > 80%)
    6. Best-effort ingest each signal into RAG with doc_type='forward_signal'

  Failures in any one source are non-fatal — the function returns whatever
  it managed to scan. Only completely empty input returns success=False.
  """
  try:
    sources_scanned: list = []
    raw_signals: list = []

    # --- Earnings releases ---
    try:
      er = get_earnings_releases(ticker, max_quarters=lookback_quarters)
    except Exception as exc:
      er = {'success': False, 'error': f'get_earnings_releases raised: {exc}'}

    if er.get('success') and er.get('releases'):
      for rel in er['releases']:
        rel_text = rel.get('text')
        if not rel_text:
          continue
        filing_date = rel.get('filing_date')
        accession = rel.get('accession_number')
        label = f'earnings_release:{filing_date or accession or "unknown"}'
        sources_scanned.append(label)
        raw_signals.extend(_scan_forward_signals(
          rel_text,
          source='earnings_release',
          filing_date=filing_date,
          accession=accession,
        ))

    # --- MD&A ---
    try:
      mda = extract_mda(ticker)
    except Exception as exc:
      mda = {'success': False, 'error': f'extract_mda raised: {exc}'}

    if mda.get('success') and mda.get('text'):
      filing_date = mda.get('filing_date')
      filing_date_str = str(filing_date) if filing_date is not None else None
      sources_scanned.append(f'mda:{filing_date_str or "latest_10k"}')
      raw_signals.extend(_scan_forward_signals(
        mda['text'],
        source='mda',
        filing_date=filing_date_str,
        accession=None,
      ))

    if not sources_scanned:
      # Both sources reported why they failed. Discarding those reasons left a
      # sentence that cannot tell a filer with no MD&A from an SEC outage, a
      # missing SEC_EMAIL, or a ticker that does not exist -- and a caller
      # retrying needs to know which source to retry.
      source_failures = {}
      if not er.get('success'):
        source_failures['earnings_releases'] = str(
          er.get('error') or 'no reason reported')
      if not mda.get('success'):
        source_failures['mda'] = str(mda.get('error') or 'no reason reported')

      detail = ('; '.join(f'{name}: {why}'
                          for name, why in source_failures.items())
                or 'both sources returned successfully but carried no text')
      return {
        'ticker':            ticker,
        'success':           False,
        'lookback_quarters': lookback_quarters,
        'sources_scanned':   [],
        'signal_count':      0,
        'signals':           [],
        'by_category':       {},
        'source_failures':   source_failures,
        'error':             f'No text sources available -- {detail}',
      }

    # Dedupe
    deduped = _dedupe_signals(raw_signals)

    # by_category tally
    by_category: Dict[str, int] = {}
    for s in deduped:
      c = s.get('category', 'unknown')
      by_category[c] = by_category.get(c, 0) + 1

    # Bonus: RAG ingest (silent on failure)
    rag_inserted = 0
    try:
      rag_inserted = _ingest_signals_to_rag(ticker, deduped)
    except Exception:
      rag_inserted = 0

    return {
      'ticker':            ticker,
      'success':           True,
      'error':             None,
      'lookback_quarters': lookback_quarters,
      'sources_scanned':   sources_scanned,
      'signal_count':      len(deduped),
      'signals':           deduped,
      'by_category':       by_category,
      'rag_chunks_inserted': rag_inserted,
    }

  except Exception as exc:
    return {
      'ticker':            ticker,
      'success':           False,
      'lookback_quarters': lookback_quarters,
      'sources_scanned':   [],
      'signal_count':      0,
      'signals':           [],
      'by_category':       {},
      'error':             f'extract_forward_signals failed: {type(exc).__name__}: {exc}',
    }


if __name__ == "__main__":
  # Test diverse companies across different industries
  test_companies = [
    "AAPL",  # Tech/Manufacturing (Apple)
    "GOOGL", # Tech/Services (Google)
    "JPM",   # Banking (JPMorgan Chase)
    "JNJ",   # Healthcare/Pharma (Johnson & Johnson)
    "WMT",   # Retail (Walmart)
    "XOM",   # Energy (ExxonMobil)
    "BAC",   # Banking (Bank of America)
    "MSFT"   # Tech/Software (Microsoft)
  ]

  # Test SEC form type support
  print("Testing SEC Form Type Support:")
  print("=" * 60)

  test_ticker = "AAPL"
  forms_to_test = ['10-K', '10-Q', '8-K', 'S-1', 'DEF 14A', '13F']

  for form in forms_to_test:
    try:
      filing_data = get_latest_filing(test_ticker, form)
      if filing_data:
        print(f"✓ {form}: SUCCESS - Found filing dated {filing_data['filing_date']}")
        # Check if XBRL is available
        if filing_data['xbrl_data']:
          print(f"  XBRL: Available")
        else:
          print(f"  XBRL: Not available")
      else:
        print(f"✗ {form}: No filings found")
    except Exception as e:
      print(f"✗ {form}: ERROR - {str(e)}")
    print("-" * 40)

  print("Testing Different Form Types - Revenue Comparison:")
  print("=" * 60)

  test_ticker = "AAPL"
  form_types = ['10-K', '10-Q']

  for form_type in form_types:
    print(f"\n{form_type} DATA:")
    print("-" * 30)

    try:
      # Test revenue
      revenue_result = get_revenue_base(test_ticker, form_type)
      if revenue_result['success']:
        print(f"Revenue: ${revenue_result['revenue_base']/1e9:.1f}B")
        print(f"Period End: {revenue_result['period_end']}")
        print(f"Concept: {revenue_result['concept_used']}")

      # Test EBITDA
      ebitda_result = get_ebitda_margin(test_ticker, form_type)
      if ebitda_result['success']:
        print(f"EBITDA Margin: {ebitda_result['ebitda_margin_percent']:.2f}%")
        print(f"EBITDA Amount: ${ebitda_result['ebitda_amount']/1e9:.1f}B")

      # Test CapEx
      capex_result = get_capex_pct_revenue(test_ticker, form_type)
      if capex_result['success']:
        print(f"CapEx % of Revenue: {capex_result['capex_pct_revenue']:.2f}%")
        print(f"Total CapEx: ${capex_result['total_capex']/1e9:.2f}B")

      # Test Tax Rate
      tax_result = get_tax_rate(test_ticker, form_type)
      if tax_result['success']:
        print(f"Effective Tax Rate: {tax_result['effective_tax_rate']:.2f}%")

    except Exception as e:
      print(f"ERROR with {form_type}: {str(e)}")

  print("\n" + "=" * 60)
  print("Testing 8-K, DEF 14A Disclosure Data:")
  print("=" * 60)

  test_ticker = "AAPL"
  special_forms = ['8-K', 'DEF 14A']

  for form_type in special_forms:
    print(f"\n{form_type} DISCLOSURES:")
    print("-" * 40)

    try:
      # Get disclosure names first
      disclosures_result = get_disclosures_names(test_ticker, form_type)
      if disclosures_result['success']:
        print(f"Found {len(disclosures_result['disclosure_names'])} disclosures:")
        for i, disclosure in enumerate(disclosures_result['disclosure_names'][:5]):  # Show first 5
          print(f"  {i+1}. {disclosure}")

        # Try to extract data from first disclosure
        if disclosures_result['disclosure_names']:
          first_disclosure = disclosures_result['disclosure_names'][0]
          print(f"\nExtracting data from: {first_disclosure}")
          disclosure_data = extract_disclosure_data(test_ticker, first_disclosure, form_type)
          if 'clean_text' in str(disclosure_data):
            print("Found text-based disclosure data")
          elif 'sample_data' in str(disclosure_data):
            print("Found structured disclosure data")
          else:
            print("No structured data found")
      else:
        print(f"Error getting disclosures: {disclosures_result['error']}")

    except Exception as e:
      print(f"ERROR with {form_type}: {str(e)}")

  print("\n" + "=" * 60)
  print("Investigating Filing Structure and Content:")
  print("=" * 60)

  test_ticker = "AAPL"
  investigation_forms = ['8-K', 'DEF 14A']

  for form_type in investigation_forms:
    print(f"\n{form_type} FILING STRUCTURE:")
    print("-" * 50)

    try:
      # Get the raw filing object
      filing_data = get_latest_filing(test_ticker, form_type)
      if filing_data:
        filing = filing_data['filing_object']
        print(f"Filing Date: {filing_data['filing_date']}")
        print(f"Accession Number: {filing_data['accession_number']}")
        print(f"URL: {filing_data['url']}")

        # Check what attributes the filing object has
        print(f"\nFiling Object Attributes:")
        attrs = [attr for attr in dir(filing) if not attr.startswith('_')]
        for attr in attrs[:10]:  # Show first 10 attributes
          print(f"  - {attr}")

        # Try to get the actual document content
        try:
          # Check if filing has documents
          if hasattr(filing, 'documents'):
            docs = filing.documents
            print(f"\nNumber of Documents: {len(docs) if docs else 'None'}")
            if docs:
              for i, doc in enumerate(docs[:3]):  # Show first 3 docs
                print(f"  Doc {i+1}: {doc.document if hasattr(doc, 'document') else 'Unknown'}")

          # Check if filing has html content
          if hasattr(filing, 'html'):
            html_content = filing.html()
            print(f"\nHTML Content Length: {len(html_content)} characters")
            print(f"HTML Preview (first 500 chars):\n{html_content[:500]}...")

          # Check if filing has text content
          if hasattr(filing, 'text'):
            text_content = filing.text()
            print(f"\nText Content Length: {len(text_content)} characters")
            print(f"Text Preview (first 500 chars):\n{text_content[:500]}...")

        except Exception as e:
          print(f"Error accessing content: {e}")

        # For DEF 14A, check for tables
        if form_type == 'DEF 14A':
          try:
            if hasattr(filing, 'tables'):
              tables = filing.tables()
              print(f"\nTables found: {len(tables) if tables else 0}")
              if tables:
                for i, table in enumerate(tables[:2]):  # Show first 2 tables
                  print(f"  Table {i+1} shape: {table.shape if hasattr(table, 'shape') else 'Unknown'}")
          except Exception as e:
            print(f"Error accessing tables: {e}")

        # Check XBRL structure
        if filing_data['xbrl_data']:
          xbrl = filing_data['xbrl_data']
          print(f"\nXBRL Structure:")

          # Check statements
          if hasattr(xbrl, 'statements'):
            statements = xbrl.statements
            print(f"  Statements available: {len(statements) if statements else 0}")

          # Check facts
          if hasattr(xbrl, 'facts'):
            facts = xbrl.facts
            print(f"  Facts available: {len(facts) if hasattr(facts, '__len__') else 'Unknown'}")

          # Check concepts
          if hasattr(xbrl, 'concepts'):
            concepts = xbrl.concepts
            print(f"  Concepts available: {len(concepts) if hasattr(concepts, '__len__') else 'Unknown'}")

      else:
        print(f"No {form_type} filing found")

    except Exception as e:
      print(f"ERROR investigating {form_type}: {str(e)}")

  print("\n" + "=" * 60)
  print("Testing EBITDA margins across different industries:")
  print("=" * 60)

  for ticker in test_companies:
    try:
      result = get_ebitda_margin(ticker, '10-K')
      if result['success']:
        print(f"{ticker}: {result['ebitda_margin_percent']:.2f}% EBITDA margin "
              f"(Revenue: ${result['revenue']/1e9:.1f}B, Concept: {result['operating_income_concept_used']})")
      else:
        print(f"{ticker}: ERROR - {result['error']}")
    except Exception as e:
      print(f"{ticker}: EXCEPTION - {str(e)}")
    print("-" * 50)


# ---------------------------------------------------------------------------
# Item-section extraction and customer concentration.
#
# Item 3 (Legal Proceedings) and major-customer disclosure both had zero
# coverage. Litigation is a numbered item and reuses the heading-boundary
# approach already used for MD&A and risk factors. Customer concentration is
# not an item at all -- filers put it in Item 1, Item 1A, or the
# concentration-of-credit-risk footnote -- so it is found by disclosure
# language instead.
# ---------------------------------------------------------------------------

def _locate_item_section(text: str, header_pattern: str,
                         next_item_pattern: str) -> Optional[str]:
  """Return the body of a numbered item, or None if it is not present.

  A filing names each item twice: once in the table of contents and once in the
  body. Taking the first match returns a page number rather than the
  disclosure, so when several matches exist and the first is early in the
  document, the last one is used.
  """
  matches = list(re.finditer(header_pattern, text, re.IGNORECASE))
  if not matches:
    return None

  match = matches[0]
  if len(matches) > 1 and match.start() < 30000:
    match = matches[-1]

  start = match.start()
  search_from = match.end()
  end_match = re.search(next_item_pattern, text[search_from:], re.IGNORECASE)
  end = (search_from + end_match.start()) if end_match else min(start + 250000,
                                                                len(text))
  return text[start:end]


# "No single customer accounted for more than 10%" is a disclosure of LOW
# concentration. Without excluding these spans first, the 10% inside the denial
# gets scraped as though it were a real customer share.
# Filers phrase the denial many ways: "No single customer accounted for...",
# "No customer represented more than...", and Microsoft's "No sales to an
# individual customer or country other than the United States accounted for
# more than 10%". Allowing a few words between "no" and "customer" catches all
# three. Getting this wrong inverts the meaning of the disclosure.
_NO_CONCENTRATION_RE = re.compile(
  r'no\s+(?:\w+\s+){0,5}?(?:customers?|clients?)\b[^.]{0,140}?\d{1,2}(?:\.\d+)?\s*%',
  re.IGNORECASE)

# Either order: "customer ... 19%" or "19% ... from one customer".
_CUSTOMER_PCT_RE = re.compile(
  r'(?:customer|client)[^.]{0,100}?(\d{1,2}(?:\.\d+)?)\s*%',
  re.IGNORECASE)

_CUSTOMER_NAME_RE = re.compile(
  r'(?:customer|client),?\s+([A-Z][A-Za-z.&\- ]{2,40}?),?\s+'
  r'(?:accounted|represented|comprised)',
  re.IGNORECASE)


def _clean_customer_name(candidate: str) -> Optional[str]:
  """Accept a proper-noun name, reject a sentence fragment.

  The name pattern happily matched "or country other than the United States"
  out of Microsoft's denial sentence. A genuine company name capitalises every
  word; a fragment carries lowercase connectives, so any all-lowercase word
  disqualifies the match.
  """
  name = (candidate or "").strip().strip(",")
  if not name:
    return None
  words = name.split()
  if not words:
    return None
  for word in words:
    stripped = word.strip(".,&-")
    if stripped and stripped[0].islower():
      return None
  return name


# "customers headquartered outside of the United States accounted for 31%" is
# revenue by geography, not one buyer's share. NVDA's 10-K carries that
# sentence and it was the single largest contributor to a customer list that
# summed to 159% of revenue.
_GEOGRAPHIC_ATTRIBUTION_RE = re.compile(
  r'(?:customers?|clients?)\s+(?:headquartered|located|domiciled|based|residing)\b',
  re.IGNORECASE)

# A share of receivables is not a share of revenue. AVGO discloses both one
# paragraph apart -- one customer at 32% of net revenue and 44% of net accounts
# receivable -- and both landed in pct_of_revenue, so the tool reported 44%
# when the filing says 32%. The larger, wronger number sorts first.
#
# The row is kept rather than dropped: a receivables concentration is a real
# credit-risk disclosure, and the gap between the two says that customer pays
# slower than the rest.
# "Top five end customers accounted for 40%" and "one customer accounted for
# 32%" are both true and cannot be added. Summed, AVGO's fiscal 2025 rows came
# to 144% of revenue. The aggregate is worth keeping -- concentration across a
# handful of buyers is the point of the disclosure -- but a caller iterating
# rows reads an unlabelled 40% as a single buyer.
_AGGREGATE_SCOPE_RE = re.compile(
  r'\b(?:aggregate|combined|together|collectively)\b|'
  r'\b(?:top|largest|five|four|three|ten)\s+'
  r'(?:\w+\s+){0,2}?(?:customers|clients|end\s+customers)\b|'
  r'\bcustomers\s+(?:accounted|represented|comprised)\b',
  re.IGNORECASE)

_RECEIVABLES_BASIS_RE = re.compile(
  r'\b(?:accounts?\s+receivable|receivables?\s+balance|trade\s+receivables?)\b',
  re.IGNORECASE)

# The fiscal year the disclosure describes. Anchored on "fiscal ... <year>" or
# "year(s) ended <month> <day>, <year>" so a year that merely appears in the
# sentence -- an acquisition date, a contract term -- is not mistaken for the
# reporting period.
_FISCAL_YEAR_RE = re.compile(
  r'(?:fiscal\s+(?:years?\s+)?(?:ended\s+\w+\s+\d{1,2},?\s*)?(19|20)(\d{2})'
  r'|years?\s+ended\s+\w+\s+\d{1,2},\s*(19|20)(\d{2}))',
  re.IGNORECASE)


# "in 2023, 2024 and 2025 accounted for 25%, 22% and 19%" -- one sentence, one
# customer, three years. Taking the first percentage reported the OLDEST figure,
# which for TSMC's second-largest customer (11 -> 12 -> 17) inverted the trend
# and reported a rising customer at its smallest.
# Not \b: filing text extracted from PDF loses spaces, so "in2023,2024"
# has no word boundary before the year. Digit lookarounds instead.
_YEAR_TOKEN_RE = re.compile(r'(?<!\d)(20\d{2})(?!\d)')
_PCT_TOKEN_RE = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')

# "Major customers representing at least 10% of net revenue" -- the rule for
# which customers must be named, not anybody's share.
# Scoped to the words immediately before a percentage, not the whole sentence.
# "No single customer accounted for more than 10% of revenue, except one which
# represented 12%" carries a threshold AND a real disclosure; matching anywhere
# in the sentence discarded the 12% along with the 10%.
_DISCLOSURE_THRESHOLD_RE = re.compile(
  r'(?:at\s+least|or\s+more|exceed(?:ing|s|ed)?|greater\s+than|more\s+than|'
  r'in\s+excess\s+of|representing)\s*$', re.IGNORECASE)


def _multi_year_series(sentence: str) -> Optional[Dict[int, float]]:
  """{year: percent} when a sentence pairs a run of years with a run of
  percentages, else None.

  Requires equal counts and at least two of each: that is what makes the
  pairing positional rather than a guess. A sentence naming one year and one
  percentage is the ordinary case and is left to the existing path.
  """
  years = [int(y) for y in _YEAR_TOKEN_RE.findall(sentence)]
  pcts = [float(v) for v in _PCT_TOKEN_RE.findall(sentence)]
  # A year may be repeated by a column header; keep first appearances in order.
  seen: List[int] = []
  for year in years:
    if year not in seen:
      seen.append(year)
  if len(seen) < 2 or len(seen) != len(pcts):
    return None
  return dict(zip(seen, pcts))


def _sentence_around(text: str, index: int) -> str:
  """The sentence containing `index`, whitespace-normalised.

  Dedup and fiscal-year detection both need the whole statement, not the
  fixed +/-120 character window used for the excerpt: the window clipped the
  same sentence differently depending on where in it the percentage sat, so
  two views of one disclosure never compared equal.
  """
  start = max(text.rfind('. ', 0, index), text.rfind('\n\n', 0, index))
  start = 0 if start < 0 else start + 2
  terminator = re.search(r'\.(?=\s|$)', text[index:])
  end = index + (terminator.end() if terminator else 200)
  return _normalize_excerpt(text[start:end])


def _fiscal_year_in(sentence: str, offset: int) -> Optional[int]:
  """The fiscal year a disclosure describes, or None if it does not say.

  Prefers the last mention at or before the percentage -- filers write "For
  fiscal year 2026, sales to one direct customer represented 22%" -- and falls
  back to the first mention after it.

  Without this, NVDA's FY2026, FY2025 and FY2024 disclosures arrived in one
  flat list with nothing to tell them apart, so 12% (a FY2025 customer) read
  as a second current-year customer.
  """
  before, after = None, None
  for match in _FISCAL_YEAR_RE.finditer(sentence):
    century, year = (match.group(1), match.group(2)) if match.group(1) \
      else (match.group(3), match.group(4))
    value = int(f"{century}{year}")
    if match.start() <= offset:
      before = value
    elif after is None:
      after = value
  return before if before is not None else after


def _tokens(sentence: str) -> set:
  return {w for w in re.findall(r'[a-z0-9%]+', sentence.lower()) if w}


def _first_repeat_index(sentence: str, seen) -> Optional[int]:
  """The row index of a kept sentence that already says all `sentence` says.

  `seen` is a list of (sentence, row index) for one (percentage, fiscal year,
  name) claim. Returning the index rather than a bool matters when two
  genuinely different sentences share a claim -- AVGO words its "top five end
  customers accounted for 40%" disclosure two ways -- so a third copy is
  counted against the sentence it repeats, not against whichever row happened
  to come first.

  Exact-string matching is not enough. NVDA prints the direct-customer
  sentence twice, the second time behind a "Direct Customers - " lead-in, and
  states the indirect-customer fact twice with "and we estimate" inserted in
  one copy. Both pairs are one disclosure printed twice, and both survived
  exact matching -- which is how 22% and 14% each appeared twice in a list
  that summed to 159% of revenue.

  The test is token-set containment in either direction, not a similarity
  score. A score would have to be tuned, and any threshold loose enough to
  merge those two pairs also merges "Customer A accounted for 15%" with
  "Customer B accounted for 15%" -- two real customers collapsed into one,
  which is a worse error than the duplication it fixes. Containment says
  precisely what is meant: one sentence adds no word the other lacks, so it
  cannot be naming a different party.
  """
  candidate = _tokens(sentence)
  if not candidate:
    return None
  for prior, index in seen:
    other = _tokens(prior)
    if other and (candidate <= other or other <= candidate):
      return index
  return None


def _scan_customer_concentration(text: str) -> Dict[str, Any]:
  """Find major-customer disclosure in filing text.

  Returns `explicitly_none` when the filer states no customer crosses the
  threshold. That is a real and useful disclosure, distinct from finding
  nothing at all, and the two must not be conflated.

  Each row carries the fiscal year it describes and is deduplicated against
  the disclosures already kept for that (percentage, year), so a sentence the
  filing prints twice is reported once. `periods` totals each year's disclosed
  shares; a year totalling more than 100% of revenue is impossible and is
  reported as a warning rather than returned as fact.
  """
  denial_spans = [m.span() for m in _NO_CONCENTRATION_RE.finditer(text)]

  def _inside_denial(position: int) -> bool:
    return any(start <= position < end for start, end in denial_spans)

  customers: List[Dict[str, Any]] = []
  seen_sentences: Dict[tuple, list] = {}
  for match in _CUSTOMER_PCT_RE.finditer(text):
    if _inside_denial(match.start()):
      continue
    try:
      pct = float(match.group(1))
    except (TypeError, ValueError):
      continue
    # A "customer" sentence quoting 90%+ is nearly always describing something
    # other than one buyer's share of revenue.
    if not 0 < pct <= 100:
      continue
    if _GEOGRAPHIC_ATTRIBUTION_RE.search(match.group(0)):
      continue

    sentence = _sentence_around(text, match.start())

    # A naming threshold is a rule about which customers must be disclosed,
    # not a share anybody holds. Judged on the words leading into THIS
    # percentage, since one sentence can carry both.
    lead_in = text[max(0, match.start()):match.end()]
    lead_in = lead_in[:lead_in.rfind('%')] if '%' in lead_in else lead_in
    if _DISCLOSURE_THRESHOLD_RE.search(_normalize_excerpt(lead_in).rstrip('0123456789. ')):
      continue

    window = text[max(0, match.start() - 120):match.end() + 40]
    name_match = _CUSTOMER_NAME_RE.search(window)
    name = _clean_customer_name(name_match.group(1)) if name_match else None

    # One sentence covering several years is one customer, and the figure that
    # matters is the most recent. Reporting the first printed percentage gave
    # the oldest year and, for a customer rising 11 -> 12 -> 17, inverted the
    # trend the caller came for.
    by_year = _multi_year_series(sentence)
    if by_year:
      # The context pattern only anchors on the first percentage in the
      # sentence, so `pct` is whichever year is printed first -- the oldest.
      # Take the latest instead. Repeats collapse through the dedupe below,
      # which now sees identical (pct, year, name) for the same sentence.
      latest = max(by_year)
      pct = by_year[latest]
      fiscal_year = latest
    else:
      fiscal_year = _fiscal_year_in(
        sentence, sentence.find(_normalize_excerpt(match.group(0))))

    # Group by the claim, not by the sentence alone: two different sentences
    # each disclosing 15% for the same year are two customers.
    key = (pct, fiscal_year, name)
    prior = seen_sentences.setdefault(key, [])
    duplicate_of = _first_repeat_index(sentence, prior)
    if duplicate_of is not None:
      customers[duplicate_of]['occurrences'] += 1
      continue
    prior.append((sentence, len(customers)))

    measures_receivables = bool(_RECEIVABLES_BASIS_RE.search(sentence))
    is_aggregate = bool(_AGGREGATE_SCOPE_RE.search(sentence))
    customers.append({
      "name": name,
      # None on a receivables row: the field names revenue, so a number in it
      # is a claim about revenue.
      "pct_of_revenue": None if measures_receivables else pct,
      "pct_of_receivables": pct if measures_receivables else None,
      "basis": "accounts_receivable" if measures_receivables else "revenue",
      # The whole series when the filing gave one -- the trend is the point,
      # and losing it is what made the stale figure dangerous.
      "by_year": by_year,
      "scope": "aggregate" if is_aggregate else "single_customer",
      "fiscal_year": fiscal_year,
      "occurrences": 1,
      "excerpt": window.strip(),
      "disclosure": sentence,
    })

  periods: List[Dict[str, Any]] = []
  warnings: List[Dict[str, Any]] = []
  for year in sorted({row['fiscal_year'] for row in customers},
                     key=lambda y: (y is None, -(y or 0))):
    rows = [r for r in customers if r['fiscal_year'] == year]
    # total_pct exists to warn when disclosed shares exceed 100% of revenue,
    # so only revenue rows belong in it.
    revenue_rows = [r for r in rows
                    if r['pct_of_revenue'] is not None
                    and r['scope'] == 'single_customer']
    total = round(sum(r['pct_of_revenue'] for r in revenue_rows), 4)
    periods.append({
      'fiscal_year': year,
      'total_pct': total,
      'disclosure_count': len(rows),
      'revenue_disclosure_count': len(revenue_rows),
    })
    # Two unnamed single-customer rows disclosing the same share in the same
    # year are usually one customer stated twice, in Risk Factors and again in
    # MD&A. Usually is not always: NVDA's 10-K reports two different direct
    # customers at the same percentage, so merging them would erase a real
    # one. The tool cannot tell which case this is and says so instead of
    # choosing. Before aggregates were excluded from this total, AVGO's
    # double-count showed up as an impossible 144%; at 64% nothing else would
    # mention it.
    shares = {}
    for row in revenue_rows:
      if row.get('name') is None:
        shares.setdefault(row['pct_of_revenue'], []).append(row)
    repeated = {pct: rows_ for pct, rows_ in shares.items() if len(rows_) > 1}
    for pct, rows_ in sorted(repeated.items()):
      label = f"fiscal year {year}" if year is not None else "an unstated period"
      warnings.append(warning(
        'possible_duplicate_disclosure',
        f"{len(rows_)} unnamed customer disclosures for {label} each report "
        f"{pct}% of revenue. Filings often state one customer's share twice, "
        f"in Risk Factors and again in MD&A, and these may be one customer "
        f"rather than {len(rows_)}. They are also genuinely two in some "
        f"filings, so they are counted separately and total_pct may "
        f"double-count. Read the `disclosure` text on each row to decide.",
        fiscal_year=year, pct_of_revenue=pct, rows=len(rows_)))

    if total > 100:
      label = f"fiscal year {year}" if year is not None else "an unstated period"
      warnings.append(warning(
        'concentration_exceeds_total_revenue',
        f"Disclosed customer shares for {label} sum to {total:.1f}% of "
        f"revenue, which is impossible. The extraction has picked up "
        f"something that is not a single customer's share, or has attributed "
        f"disclosures from more than one period to this one. Treat these rows "
        f"as unreliable.",
        fiscal_year=year, total_pct=total))

  return {
    # Canonical key. The rows are disclosures of a share of revenue and are
    # usually anonymous -- "one direct customer" -- so calling the array
    # `named_customers` promised a name that is null on every NVDA row.
    "customer_disclosures": customers,
    # Back-compatible alias for callers written against the old key.
    "named_customers": customers,
    "periods": periods,
    "has_concentration": bool(customers),
    "explicitly_none": bool(denial_spans) and not customers,
    "warnings": warnings,
  }


def extract_litigation(ticker: str, form_type: str = '10-K',
                       max_chars: int = 40000) -> Dict[str, Any]:
  """Extract Item 3, Legal Proceedings.

  Many filers cross-reference a note rather than restating the detail here, so
  a short section is normal and is not an extraction failure.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data:
      return {'ticker': ticker, 'success': False,
              'error': _filing_miss(
                  ticker, form_type,
                  f'No {form_type} filing found for {ticker}')}

    filing_obj = filing_data.get('filing_object')
    if filing_obj is None:
      return {'ticker': ticker, 'success': False,
              'error': 'No filing object in cache'}

    text = filing_obj.text()
    if not text:
      return {'ticker': ticker, 'success': False, 'error': 'Empty filing text'}

    section = _locate_item_section(
      text, r'ITEM\s+3\.?\s*[-–—]?\s*LEGAL\s+PROCEEDINGS', r'ITEM\s+4\b')
    if section is None:
      return {'ticker': ticker, 'success': False,
              'error': 'Could not locate Item 3 (Legal Proceedings) header'}

    section = section[:max_chars]
    return {
      'ticker': ticker,
      'success': True,
      'form_type': form_type,
      'filing_date': filing_data.get('filing_date'),
      'text': section,
      'char_count': len(section),
      'cross_referenced_only': len(section) < 1500,
    }
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'{type(e).__name__}: {e}'}


def _concentration_failure(ticker: str, message: str) -> Dict[str, Any]:
  """Failure carrying the same keys as success.

  A caller reading `has_concentration` should get False, not a KeyError,
  when the filing could not be read. Changing the shape on failure pushes
  error handling onto every consumer.
  """
  return {
    'ticker': ticker,
    'success': False,
    'error': message,
    'customer_disclosures': [],
    'named_customers': [],
    'periods': [],
    'has_concentration': False,
    'explicitly_none': False,
    'warnings': [],
  }


def extract_customer_concentration(ticker: str,
                                   form_type: str = '10-K') -> Dict[str, Any]:
  """Find major-customer disclosure anywhere in the filing.

  Not a numbered item: filers place this in Item 1, Item 1A, or the
  concentration-of-credit-risk footnote, so the whole document is scanned.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data:
      return _concentration_failure(
        ticker, _filing_miss(ticker, form_type,
                             f'No {form_type} filing found for {ticker}'))

    filing_obj = filing_data.get('filing_object')
    if filing_obj is None:
      return _concentration_failure(ticker, 'No filing object in cache')

    text = filing_obj.text()
    if not text:
      return _concentration_failure(ticker, 'Empty filing text')

    found = _scan_customer_concentration(text)
    return {
      'ticker': ticker,
      'success': True,
      'form_type': form_type,
      'filing_date': filing_data.get('filing_date'),
      **found,
    }
  except Exception as e:
    return _concentration_failure(ticker, f'{type(e).__name__}: {e}')
