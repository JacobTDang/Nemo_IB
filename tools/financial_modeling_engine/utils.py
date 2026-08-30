import yfinance as yf

from tools.ticker import normalize_ticker
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import re
import sys
from datetime import datetime, timedelta, timezone

def _json_safe(value):
  """Replace NaN and infinity with None, recursively.

  RFC 8259 has no NaN literal. Python emits a bare `NaN` and accepts it back,
  which is why `"interestExpense": NaN` survived every Python test -- but a
  JavaScript, Go or Rust client loses the ENTIRE response, not the one field.
  A tool that silently fails for every non-Python caller is worse than one
  returning a null.
  """
  import math
  if isinstance(value, dict):
    return {k: _json_safe(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_json_safe(v) for v in value]
  if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
    return None
  return value


def _balance_or_none(info: dict, key: str):
    """A balance-sheet figure, or None when the provider did not report one.

    Defaulting to 0 says the company holds none of it. For an unknown ticker
    yfinance returns an info dict with a single key, so `info.get(key, 0)`
    reported a company that does not exist as holding no cash and carrying no
    debt -- and those two fields feed calculate_wacc and
    calculate_credit_profile, which would then value it as debt-free.

    A genuine zero survives: the key is present and its value is 0.
    """
    value = info.get(key)
    return None if value is None else value


def ratio_is_comparable(quote_currency, filing_currency) -> bool:
    """Whether a price-derived figure and a filer-reported one can be divided.

    They can only be divided when they are denominated in the same money. An
    ADR quotes in USD while the filer reports in its own currency, so every
    multiple built that way is wrong by the exchange rate -- TSM's P/E came out
    at 0.977 against a true 30.7.

    An unknown currency is NOT treated as a match. Assuming they agree is the
    failure this exists to prevent, and it would reproduce the bug for exactly
    the securities whose metadata is thinnest.
    """
    if not quote_currency or not filing_currency:
        return False
    return str(quote_currency).strip().upper() == str(filing_currency).strip().upper()


def cross_currency_note(quote_currency, filing_currency) -> str:
    """Why a multiple was refused, naming both currencies."""
    return (f"Multiples are not reported: the quote is in "
            f"{quote_currency or 'an unknown currency'} while the financials "
            f"are reported in {filing_currency or 'an unknown currency'}. "
            f"Dividing one by the other yields a figure wrong by the exchange "
            f"rate rather than a valuation multiple. Use "
            f"provider_trailing_pe, which the data provider computes in a "
            f"single currency, or the SEC tools, which report the filer's "
            f"figures with an explicit currency.")


def missing_inputs_note(ticker: str, multiples, missing: List[str]) -> str:
    """Why a multiple is absent, naming the inputs the provider did not report.

    `multiples` is one name or several. Several share a sentence when they
    share a cause -- a missing market cap takes pe_ratio and pb_ratio together,
    and saying it twice reads as two separate problems.

    `multiples_suppressed_reason` was wired to the cross-currency and
    negative-EV paths only, so a filer whose provider record is merely thin
    lost its ratios with the explanation left null. MU is a US filer reporting
    in USD; `marketCap` and `sharesOutstanding` both came back null, pe_ratio
    and pb_ratio were dropped, and the field built to say why said nothing --
    while provider_trailing_pe 21.249 sat in the same payload. A null reason
    reads as "we could not obtain this multiple", which is the one thing it
    was not: we declined to build it on an input that is missing rather than
    zero.
    """
    names = [multiples] if isinstance(multiples, str) else list(multiples)
    subject = (names[0] if len(names) == 1
               else f"{', '.join(names[:-1])} and {names[-1]}")
    verb = "is" if len(names) == 1 else "are"
    inputs = " and no ".join(missing)
    return (f"{subject} {verb} not reported for {ticker}: the market-data "
            f"provider returned no {inputs}. An absent input is not a zero, "
            f"so the ratio is refused rather than computed against a "
            f"substitute.")


def non_positive_ev_note(ticker: str, enterprise_value, market_cap,
                         total_debt, cash) -> str:
    """Why EV multiples are refused on an enterprise value at or below zero.

    The provider reports -233.9bn for BRK-B while its own market cap plus debt
    less cash is +842.7bn, because it counts Berkshire's insurance investments
    as cash. Left alone that produced ev_revenue -0.61 and ev_ebitda -1.79 --
    not cheap valuations but non-valuations, which sort below zero and read as
    the cheapest name in any comp table they land in.
    """
    substitute = (market_cap or 0) + (total_debt or 0) - (cash or 0)
    return (f"EV multiples are not reported: the provider's enterprise value "
            f"for {ticker} is {enterprise_value:,.0f}, which is not positive. "
            f"A multiple over a non-positive numerator has no ordering and "
            f"sorts below every real one. Market cap plus debt less cash is "
            f"{substitute:,.0f} if you need a substitute.")


def _absent(value) -> bool:
    """Whether a provider figure is missing. NaN counts as missing.

    A NaN reaching a ratio produces a NaN, which `_json_safe` turns into a
    null at the very end -- indistinguishable from a field the provider never
    sent, and therefore just as much in need of an explanation.
    """
    if value is None:
        return True
    return isinstance(value, float) and value != value


# Two share counts in one response are on the same basis or they are not. Below
# this the difference is the provider rounding its own figures -- NVDA, AAPL and
# MSFT all reconcile to within 1e-7% -- and above it the two numbers are
# measuring different things. Nothing real sits between: the smallest genuine
# basis gap measured is BRK-B's 52%.
_SHARE_BASIS_TOLERANCE_PCT = 0.5


def market_cap_implied_shares(market_cap, price) -> Optional[float]:
    """The share count `marketCap` was built on, or None.

    The only count in the payload guaranteed to be consistent with the payload:
    it is derived from two of its own fields. An absent market cap or an absent
    or zero price yields None rather than a division that would either raise or
    invent a count.
    """
    if _absent(market_cap) or _absent(price) or not price:
        return None
    return market_cap / price


def _within(value: float, reference: float, tolerance_pct: float) -> bool:
    """Whether two figures of the same thing agree to within `tolerance_pct`."""
    if not reference:
        return False
    return abs(value - reference) / abs(reference) * 100.0 <= tolerance_pct


def share_count_basis(market_cap, price, shares_outstanding,
                      provider_implied=None) -> Dict[str, Any]:
    """How `sharesOutstanding` relates to the `marketCap` sitting beside it.

    yfinance reports `marketCap` across every share class and
    `sharesOutstanding` for one of them, with nothing distinguishing the two.
    Side by side that is a 2.0845x error for GOOGL and a 1.5204x error for
    BRK-B in any figure divided by the share count -- a DCF value per share, a
    book value per share, an insider ownership percentage -- while `pe_ratio`
    and `pb_ratio`, which divide `marketCap`, stay correct. One field in the
    response disagrees with the rest of it.

    `basis` is what was observed, not what caused it. A gap is usually a
    multi-class filer, but a cover page a quarter stale makes the same shape,
    and asserting the cause would be a guess where the observation is a fact.

    `shares_outstanding_all_classes` is the count that reproduces `marketCap`,
    so a caller has something with a valid counterpart to divide by. It prefers
    the provider's own `impliedSharesOutstanding` where that agrees with
    market cap over price, and falls back to the quotient -- which is
    internally consistent by construction -- where it does not.
    """
    implied = market_cap_implied_shares(market_cap, price)
    reported = None if _absent(shares_outstanding) else float(shares_outstanding)
    provider = None if _absent(provider_implied) or not provider_implied \
        else float(provider_implied)

    all_classes, source = None, None
    if implied is not None:
        if provider is not None and _within(provider, implied,
                                            _SHARE_BASIS_TOLERANCE_PCT):
            all_classes, source = provider, 'provider_implied'
        else:
            all_classes, source = implied, 'market_cap_over_price'
        # A share count is a whole number. `market_cap_implied_shares` keeps
        # the fraction, because it is a quotient and saying so is the point.
        all_classes = int(round(all_classes))

    gap_pct = None
    basis = 'unverified'
    if implied is not None and reported:
        gap_pct = (implied - reported) / reported * 100.0
        if abs(gap_pct) <= _SHARE_BASIS_TOLERANCE_PCT:
            basis = 'matches_market_cap'
        elif gap_pct > 0:
            basis = 'narrower_than_market_cap'
        else:
            basis = 'wider_than_market_cap'

    return {
        'basis': basis,
        'shares_outstanding': reported,
        'market_cap_implied_shares': implied,
        'provider_implied_shares_outstanding': provider,
        'shares_outstanding_all_classes': all_classes,
        'all_classes_source': source,
        'gap_pct': gap_pct,
    }


def share_basis_warning(ticker: str, basis: Dict[str, Any]
                        ) -> Optional[Dict[str, Any]]:
    """The warning for a share count that does not match its own market cap.

    None when the two reconcile, which is every single-class filer. A caveat
    that fires on the responses it has nothing to say about is a caveat
    attached to the tool rather than to the answer, and a reader learns to skip
    the array.

    The message states the multiple, not only the percentage gap. "52% apart"
    does not tell an analyst that their value per share is 2.0845x too high,
    and the multiple is the entire reason this class of defect matters.
    """
    if basis['basis'] not in ('narrower_than_market_cap',
                             'wider_than_market_cap'):
        return None
    reported = basis['shares_outstanding']
    implied = basis['market_cap_implied_shares']
    multiple = implied / reported
    return {
        'code': 'share_count_basis_mismatch',
        'message': (
            f"{ticker}: sharesOutstanding ({reported:,.0f}) and marketCap "
            f"are on different share bases. marketCap divided by the price "
            f"in this same response implies {implied:,.0f} shares, "
            f"{basis['gap_pct']:+.2f}% away. The usual cause is a multi-class "
            f"filer: the provider counts one class in sharesOutstanding and "
            f"every class in marketCap. Any per-share figure divided by "
            f"sharesOutstanding is therefore {multiple:.4g}x too high. Use "
            f"shares_outstanding_all_classes for per-share arithmetic; "
            f"pe_ratio and pb_ratio are computed from marketCap and are "
            f"unaffected."),
        'shares_outstanding': reported,
        'shares_outstanding_all_classes': basis['shares_outstanding_all_classes'],
        'gap_pct': basis['gap_pct'],
        'multiple': multiple,
    }


# The two inputs each multiple divides. Ordered so the report reads the way an
# analyst would list them.
_MULTIPLE_INPUTS = ('pe_ratio', 'pb_ratio', 'ev_revenue', 'ev_ebitda', 'ev_ebit')

# An input more than one multiple depends on. Its absence takes a whole block
# of ratios with it, which is what makes it worth stating at the top level
# rather than only per-multiple.
_SHARED_INPUTS = ('marketCap', 'enterpriseValue')


# Identity fields a resolved quote carries. Measured against live yfinance: a
# symbol that does not exist comes back as `{'trailingPegRatio': None}` with an
# empty `history_metadata`, while a real filer carries 189 info keys and 28
# metadata keys. An emptiness check on `info` does not catch the former.
_SYMBOL_IDENTITY_KEYS = ('symbol', 'quoteType', 'shortName', 'longName',
                         'regularMarketPrice', 'currency', 'marketCap')


def symbol_is_resolved(handle: Any, info: Optional[Dict[str, Any]] = None) -> bool:
    """Did the market-data provider recognise this symbol at all?

    yfinance never raises for an unknown symbol -- it logs `HTTP Error 404` and
    returns empty structures, which every downstream field then reads as None.
    Without this check the response is a well-formed answer about a company
    that does not exist: `pays_dividend: false`, `split_count: 0`,
    `shares_short: null`, all reported as a success.
    """
    if getattr(handle, 'history_metadata', None):
        return True
    if info is None:
        try:
            info = handle.info or {}
        except AttributeError:
            # Not a yfinance handle at all -- a test double, or a wrapper that
            # exposes only the series a caller asked for. We did not observe a
            # missing symbol, we failed to look, and those are different.
            # Refusing needs positive evidence of non-existence.
            return True
        except Exception:      # noqa: BLE001 - a failed quote is not an answer
            return False
    return any(info.get(key) is not None for key in _SYMBOL_IDENTITY_KEYS)


def unresolved_symbol_error(ticker: str, handle: Any,
                            info: Optional[Dict[str, Any]] = None
                            ) -> Optional[Dict[str, Any]]:
    """The refusal to return when a symbol did not resolve, or None."""
    if symbol_is_resolved(handle, info):
        return None
    return {
        'ticker': ticker,
        'success': False,
        'error': (f"{ticker!r} did not resolve to a listed security at the "
                  f"market-data provider. This is a failed lookup, not a "
                  f"company without financials -- check the symbol, or use "
                  f"the SEC tools if it is an SEC registrant without a quote."),
    }


def get_data(ticker: str) -> Dict[str, Any]:
  data = {}

  # Providers spell class shares with a dash; filings use a dot. Resolve to
  # the provider's form so a caller can pass either and still join the two.
  requested_ticker = ticker
  ticker = normalize_ticker(ticker) or ticker
  company = yf.Ticker(ticker)
  book_value = None
  operating_income = None

  # CONSTANT POSSIBLE KEYS
  BOOK_VAL_KEYS : List = ['Stockholders Equity', 'Total Stockholder Equity','Total Equity Gross Minority Interest', 'Common Stock Equity',]
  OPERATING_INCOME_KEYS : List = ['Operating Income','Ebit']

  # get necessary information
  info = company.info

  # yfinance does not raise for a symbol it cannot find: it logs `HTTP Error
  # 404` and hands back an empty info dict. Every field below then reads as
  # None and the response looks like a real company that discloses nothing --
  # including a cross-currency note blaming two unknown currencies for the
  # missing multiples, which explains the absence with something that is not
  # the reason. `history_metadata` is populated only when the provider
  # actually answered, so it separates "no such symbol" from "a real company
  # whose fields we could not read".
  unresolved = unresolved_symbol_error(ticker, company, info)
  if unresolved is not None:
    return unresolved

  data['ticker'] = requested_ticker
  data['ticker_resolved'] = ticker
  data['marketCap'] = info.get('marketCap')
  data['currentPrice'] = info.get('currentPrice') or info.get('regularMarketPrice')
  # yfinance `totalRevenue` and `ebitda` ARE trailing-twelve-month values.
  # Expose them under explicit `*_ttm` aliases so the analyst prompt can
  # distinguish TTM (these) from latest-annual values (from SEC tools like
  # get_revenue_base / get_ebitda_margin which return last 10-K's FY total).
  # The legacy `revenue` / `EBITDA` keys are kept for back-compat with the
  # DCF/credit/LBO callers that already read them.
  data['revenue'] = info.get('totalRevenue')
  data['revenue_ttm'] = data['revenue']
  data['EBITDA'] = info.get('ebitda')
  data['ebitda_ttm'] = data['EBITDA']
  data['netIncomeToCommon'] = info.get('netIncomeToCommon')
  data['net_income_ttm'] = data['netIncomeToCommon']
  data['enterpriseValue'] = info.get('enterpriseValue')
  data['cash'] = _balance_or_none(info, 'totalCash')
  data['totalDebt'] = _balance_or_none(info, 'totalDebt')
  data['sharesOutstanding'] = info.get('sharesOutstanding')
  data['beta'] = info.get('beta')

  # `sharesOutstanding` is left exactly as the provider sent it -- callers
  # already read it and silently redefining it would move the error rather than
  # remove it. What is added is the basis it is on and the count that does
  # reproduce the marketCap beside it, so the two can no longer be multiplied
  # together into a 2.08x error with nothing in the payload objecting.
  share_basis = share_count_basis(
      data['marketCap'], data['currentPrice'], data['sharesOutstanding'],
      provider_implied=info.get('impliedSharesOutstanding'))
  data['shares_outstanding_basis'] = share_basis['basis']
  data['shares_outstanding_all_classes'] = share_basis['shares_outstanding_all_classes']
  data['shares_outstanding_all_classes_source'] = share_basis['all_classes_source']
  data['market_cap_implied_shares'] = share_basis['market_cap_implied_shares']
  data['shares_outstanding_gap_pct'] = share_basis['gap_pct']
  basis_mismatch = share_basis_warning(ticker.upper(), share_basis)
  data['warnings'] = [basis_mismatch] if basis_mismatch else []

  # safely get interest expense from income statement
  INTEREST_EXPENSE_KEYS: List[str] = ['Interest Expense', 'Interest Expense Non Operating', 'Net Interest Income']
  try:
    income_stmt = company.income_stmt
    ie_key = find_key(INTEREST_EXPENSE_KEYS, income_stmt.index)
    if ie_key:
      interest_expense = income_stmt.loc[ie_key].iloc[0]
      data['interestExpense'] = abs(float(interest_expense)) if interest_expense is not None else None
    else:
      data['interestExpense'] = None
  except Exception as e:
    print(f'Could not get interest expense for {ticker}: {str(e)}', file=sys.stderr)
    data['interestExpense'] = None

  # safely get the balancesheet and book_value
  try:
    balance_sheet = company.balance_sheet
    key = find_key(BOOK_VAL_KEYS, balance_sheet.index)
    # .loc finds the row with key and .iloc will get the first col / most recent year
    book_value = balance_sheet.loc[key].iloc[0]
  except Exception as e:
    print(f'Could not get book value for {ticker} : {str(e)}', file=sys.stderr)

  data['EBIT'] = None

  # safely get the operating_income from the income statement
  try:
    income_statement = company.income_stmt
    key = find_key(OPERATING_INCOME_KEYS, income_statement.index)
    operating_income = income_statement.loc[key].iloc[0]
    data['EBIT'] = operating_income
    # EBIT comes from the annual statement while revenue_ttm, ebitda_ttm and
    # net_income_ttm come from the trailing-twelve-month feed. Both are right;
    # only their pairing is wrong, and unlabelled it reads as TTM like
    # everything around it. Subtracting one from the other gave NVDA an
    # implied D&A of $35.1bn against an actual $2.84bn.
    data['ebit_basis'] = 'fiscal_year'
    try:
      data['ebit_period_end'] = str(income_statement.columns[0].date())
    except Exception:
      data['ebit_period_end'] = str(income_statement.columns[0])

  except Exception as e:
    print(f"Could not get the operating income from income statement for {ticker} : {str(e)}", file=sys.stderr)


  # The quote currency and the reporting currency are different things for an
  # ADR, and every multiple below divides one by the other.
  data['currency'] = info.get('currency')
  data['financial_currency'] = info.get('financialCurrency')
  data['provider_trailing_pe'] = info.get('trailingPE')
  data['multiples_suppressed_reason'] = None

  comparable = ratio_is_comparable(data['currency'], data['financial_currency'])
  if not comparable:
    # Refuse rather than emit a plausible wrong number. SAP's cross-currency
    # P/E was 32.12 against a true 29.7 -- an 8% error with no smell at all,
    # which is worse than an obviously broken one because it gets acted on.
    data['multiples_suppressed_reason'] = cross_currency_note(
        data['currency'], data['financial_currency'])
    for key in ('pe_ratio', 'pb_ratio', 'ev_revenue', 'ev_ebitda', 'ev_ebit'):
      data[key] = None

  # An enterprise value at or below zero cannot carry a multiple. The provider
  # reports -233.9bn for BRK-B while its own market cap plus debt less cash is
  # +842.7bn, because it counts Berkshire's insurance investments as cash. Left
  # alone, that produced ev_revenue -0.61 and ev_ebitda -1.79 -- not cheap
  # valuations but non-valuations, which sort below zero and read as the
  # cheapest name in any comp table they land in. The provider's own figure is
  # kept; only the ratios built on it go.
  ev_usable = (data['enterpriseValue'] is not None
               and data['enterpriseValue'] > 0)
  ev_reason = None
  if comparable and not ev_usable and data['enterpriseValue'] is not None:
    ev_reason = non_positive_ev_note(
        ticker, data['enterpriseValue'], data['marketCap'],
        data['totalDebt'], data['cash'])
    existing = data.get('multiples_suppressed_reason')
    data['multiples_suppressed_reason'] = (
        f"{existing} {ev_reason}" if existing else ev_reason)

  # calculate multiples -- each wrapped independently so one failure doesn't skip all
  try:
    if comparable and data['marketCap'] is not None and data['netIncomeToCommon'] is not None:
      data['pe_ratio'] = data['marketCap'] / data['netIncomeToCommon']
  except Exception as e:
    print(f'Error calculating P/E ratio for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if comparable and data['marketCap'] is not None and book_value is not None:
      data['pb_ratio'] = data['marketCap'] / book_value
  except Exception as e:
    print(f'Error calculating P/B ratio for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if comparable and ev_usable and data['revenue'] is not None:
      data['ev_revenue'] = data['enterpriseValue'] / data['revenue']
  except Exception as e:
    print(f'Error calculating EV/Revenue for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if comparable and ev_usable and data['EBITDA'] is not None:
      data['ev_ebitda'] = data['enterpriseValue'] / data['EBITDA']
  except Exception as e:
    print(f'Error calculating EV/EBITDA for {ticker}: {str(e)}', file=sys.stderr)

  if data.get('EBIT') is not None and data.get('ebitda_ttm') is not None:
    data['basis_warning'] = (
        f"EBIT is {data.get('ebit_basis', 'fiscal-year')} "
        f"(period ending {data.get('ebit_period_end')}) while revenue_ttm, "
        f"ebitda_ttm and net_income_ttm are trailing twelve months. Do not "
        f"subtract one from the other -- the difference is not D&A -- and read "
        f"ev_ebit as a current enterprise value over a fiscal-year EBIT.")

  try:
    if comparable and ev_usable and data['EBIT'] is not None:
      data['ev_ebit'] = data['enterpriseValue'] / data['EBIT']
      data['ev_ebit_basis'] = (
          f"current enterprise value / {data.get('ebit_basis','fiscal_year')} "
          f"EBIT ending {data.get('ebit_period_end')}")
  except Exception as e:
    print(f'Error calculating EV/EBIT for {ticker}: {str(e)}', file=sys.stderr)

  # Every multiple that ended up absent, with the input that made it absent.
  # Suppression used to be explained only where a whole block went at once
  # (cross-currency, non-positive EV); anything narrower left the ratio null
  # and the reason null beside it. Downstream, comparable_company_analysis
  # then had nothing to attribute its exclusions to and asserted one fixed
  # cause for all of them.
  divisors = {
    'pe_ratio':   {'marketCap': data['marketCap'],
                   'netIncomeToCommon': data['netIncomeToCommon']},
    'pb_ratio':   {'marketCap': data['marketCap'],
                   'book value': book_value},
    'ev_revenue': {'enterpriseValue': data['enterpriseValue'],
                   'revenue': data['revenue']},
    'ev_ebitda':  {'enterpriseValue': data['enterpriseValue'],
                   'EBITDA': data['EBITDA']},
    'ev_ebit':    {'enterpriseValue': data['enterpriseValue'],
                   'EBIT': data['EBIT']},
  }
  detail = {}
  shared_absences = []
  for name in _MULTIPLE_INPUTS:
    if not _absent(data.get(name)):
      continue
    if not comparable:
      detail[name] = data['multiples_suppressed_reason']
      continue
    if ev_reason is not None and name.startswith('ev_'):
      detail[name] = ev_reason
      continue
    missing = [label for label, value in divisors[name].items()
               if _absent(value)]
    if missing:
      detail[name] = missing_inputs_note(ticker, name, missing)
      shared = tuple(label for label in missing if label in _SHARED_INPUTS)
      if shared:
        shared_absences.append((shared, name))
    else:
      # Both divisors present and the ratio still absent: the calculation
      # itself raised and was logged to stderr. Say that rather than blame an
      # input that was there.
      detail[name] = (
          f"{name} is not reported for {ticker}: both inputs were present but "
          f"the ratio could not be computed from them.")

  # An input several multiples share takes a block of them with it, which is
  # what earns a top-level statement. A single missing line item does not: it
  # would turn multiples_suppressed_reason non-null for healthy filers and
  # cost the field its meaning.
  if detail and data['multiples_suppressed_reason'] is None and shared_absences:
    grouped = {}
    for shared, name in shared_absences:
      grouped.setdefault(shared, []).append(name)
    data['multiples_suppressed_reason'] = " ".join(
        missing_inputs_note(ticker, names, list(shared))
        for shared, names in grouped.items())
  data['multiples_suppressed_detail'] = detail or None

  return _json_safe(data)


def find_key(possible_key: List[str], indexes: pd.Index) -> Optional[str]:
  """The first candidate label present in `indexes`, or None.

  None means this statement does not carry the line under any of the names we
  know. That is the common case for banks, which report no 'Operating Income'
  at all, and the caller turns it into an absent field rather than a zero.

  It used to announce "using llm to compare indexes to possible keys" and then
  "complete failure, key DNE" on stderr. No LLM was ever called -- the fallback
  was a TODO that printed and returned None -- so every bank lookup logged a
  claim about work that did not happen.
  """
  for key in possible_key:
    if key in indexes:
      return str(key)
  return None

# Curated theme-to-ETF map. Each theme maps to one or more ETF tickers that
# offer concentrated exposure. ETFs chosen by AUM + holdings purity.
_THEME_TO_ETFS: Dict[str, List[str]] = {
  # Tech / AI / semis
  'semiconductors':    ['SMH', 'SOXX', 'XSD'],
  'semis':             ['SMH', 'SOXX', 'XSD'],
  'ai':                ['BOTZ', 'AIQ', 'ROBO', 'ARTY'],
  'artificial intelligence': ['BOTZ', 'AIQ', 'ROBO'],
  'robotics':          ['BOTZ', 'ROBO', 'IRBO'],
  'cloud':             ['SKYY', 'WCLD', 'CLOU'],
  'cybersecurity':     ['HACK', 'CIBR', 'BUG'],
  'fintech':           ['FINX', 'IPAY', 'KFIN'],
  'software':          ['IGV', 'XSW'],
  'internet':          ['FDN', 'PNQI'],
  'tech':              ['XLK', 'VGT', 'QQQ'],
  # Energy / commodities
  'energy':            ['XLE', 'XOP', 'IEO'],
  'oil':               ['XOP', 'OIH', 'IEZ'],
  'natural gas':       ['UNG', 'FCG'],
  'uranium':           ['URA', 'URNM'],
  'lithium':           ['LIT', 'BATT'],
  'battery':           ['LIT', 'BATT'],
  'clean energy':      ['ICLN', 'QCLN', 'TAN'],
  'solar':             ['TAN', 'PBW'],
  'wind':              ['FAN'],
  'utilities':         ['XLU', 'VPU'],
  # Financials
  'banks':             ['KBE', 'KRE', 'XLF'],
  'financials':        ['XLF', 'VFH'],
  # Healthcare / biotech
  'healthcare':        ['XLV', 'VHT'],
  'biotech':           ['IBB', 'XBI', 'BBH'],
  'pharma':            ['IHE', 'PJP'],
  'medical devices':   ['IHI'],
  # Consumer
  'consumer discretionary': ['XLY', 'VCR'],
  'consumer staples':       ['XLP', 'VDC'],
  'retail':                 ['XRT', 'RTH'],
  # Industrials / infra
  'industrials':       ['XLI', 'VIS'],
  'aerospace':         ['ITA', 'PPA'],
  'defense':           ['ITA', 'PPA', 'XAR'],
  'infrastructure':    ['PAVE', 'IFRA'],
  'rare earth':        ['REMX'],
  # Macro / geographic
  'gold':              ['GLD', 'IAU', 'GDX'],
  'silver':            ['SLV', 'SIL'],
  'china':             ['FXI', 'KWEB', 'MCHI'],
  'india':             ['INDA', 'EPI'],
  'japan':             ['EWJ'],
  'emerging markets':  ['EEM', 'VWO', 'IEMG'],
  # Real estate
  'reit':              ['VNQ', 'IYR', 'XLRE'],
  'real estate':       ['VNQ', 'IYR', 'XLRE'],
  'data centers':      ['DTCR', 'SRVR'],
  # Themes
  'electric vehicles': ['DRIV', 'KARS', 'IDRV'],
  'ev':                ['DRIV', 'KARS', 'IDRV'],
  'genomics':          ['ARKG'],
  'space':             ['UFO', 'ARKX'],
  'esports':           ['HERO', 'GAMR'],
  'metaverse':         ['META', 'METV'],
}


_ANALOGUES_CACHE: List[Dict[str, Any]] = []


def _parse_magnitude(body: str):
  """Extract the **Magnitude:** line per the schema documented in
  knowledge/analogues.md and return (drawdown_pct, duration_months,
  direction). drawdown_pct is a signed int (negative for bear analogues
  that lose value, positive for bull analogues that gain) or None when
  the entry is a setup with no completed move yet.
  """
  import re as _re
  m = _re.search(r'\*\*Magnitude:\*\*\s*([^\n]+(?:\n[^\n*]+)*)', body)
  if not m:
    return None, None, None
  line = m.group(1).strip()

  # Direction: trailing "(bear analogue)" / "(bull analogue)" / "(setup)".
  # Permit continuation text inside the parens (e.g. "(bear analogue —
  # use for the post-pull-forward case)").
  direction = None
  if _re.search(r'\(bear analogue\b[^)]*\)', line, _re.IGNORECASE):
    direction = 'bear'
  elif _re.search(r'\(bull analogue\b[^)]*\)', line, _re.IGNORECASE):
    direction = 'bull'
  elif _re.search(r'\(setup\b[^)]*\)|\(setup\)', line, _re.IGNORECASE):
    direction = 'setup'

  # If setup or explicit N/A, magnitude is None
  if direction == 'setup' or line.lstrip().startswith('N/A'):
    return None, None, direction

  # Drawdown percentage: first signed number followed by %
  pct_match = _re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', line)
  drawdown_pct = float(pct_match.group(1)) if pct_match else None

  # If no explicit sign but direction is bear, treat as negative
  if drawdown_pct is not None and direction == 'bear' and drawdown_pct > 0:
    # Catalog convention is to write "-86%" explicitly but be defensive
    if not _re.search(r'[+-]\d', line):
      drawdown_pct = -drawdown_pct

  # Duration in months
  dur_match = _re.search(r'(\d+)\s*months?', line)
  duration_months = int(dur_match.group(1)) if dur_match else None

  return drawdown_pct, duration_months, direction


def _load_analogues() -> List[Dict[str, Any]]:
  """Load the analogues knowledge file into a list of
  {name, tags, body, drawdown_pct, duration_months, direction}.
  Cached in-process — file is checked into the repo, doesn't change at
  runtime."""
  global _ANALOGUES_CACHE
  if _ANALOGUES_CACHE:
    return _ANALOGUES_CACHE
  import re as _re
  from pathlib import Path as _P
  path = _P(__file__).resolve().parents[2] / 'knowledge' / 'analogues.md'
  if not path.exists():
    return []
  text = path.read_text(encoding='utf-8')
  # Split on numbered section headers: ## N. Title
  sections = _re.split(r'\n##\s+\d+\.\s+', text)
  out = []
  for sec in sections[1:]:  # skip preamble
    # First line is the title
    lines = sec.split('\n', 1)
    title = lines[0].strip()
    body = lines[1] if len(lines) > 1 else ''
    # Extract tags line
    tag_match = _re.search(r'\*\*Tags:\*\*\s*([^\n]+)', body)
    tags = []
    if tag_match:
      tags = [t.strip().rstrip(',').lower()
              for t in tag_match.group(1).split(',')]
      tags = [t for t in tags if t]
    drawdown_pct, duration_months, direction = _parse_magnitude(body)
    out.append({
      'name': title,
      'tags': tags,
      'body': body.strip(),
      'drawdown_pct': drawdown_pct,
      'duration_months': duration_months,
      'direction': direction,
    })
  _ANALOGUES_CACHE = out
  return out


def get_historical_analogue(thesis_description: str,
                            top_n: int = 3) -> Dict[str, Any]:
  """Match a current thesis description against the curated catalog of
  historical investment periods. Returns top N matches by tag-overlap
  score plus the catalog's lesson for each match.

  The thesis description should contain structural keywords matching
  the analogue tag schema (capex_cycle, valuation_expansion,
  margin_compression, supply_constrained, etc.). The tool scans the
  description case-insensitively for tag tokens and ranks analogues by
  how many tags they share with the thesis.
  """
  analogues = _load_analogues()
  if not analogues:
    return {
      'success': False,
      'error': 'analogues.md not found or empty',
    }

  td = (thesis_description or '').lower()

  # Score each analogue by token overlap with the description.
  # Token = a tag name, but we also treat each whitespace word as a token
  # so descriptive phrases like "AI capex peak" can match the
  # 'capex_peak' tag too.
  description_tokens = set(re.findall(r'[a-z_]+', td))
  # Convert multi-word phrases (e.g. "capex peak") to underscore form for
  # tag matching
  td_underscored = re.sub(r'\s+', '_', td)
  underscored_tokens = set(re.findall(r'[a-z_]{3,}', td_underscored))
  description_tokens.update(underscored_tokens)

  scored = []
  for a in analogues:
    matches = []
    for tag in a['tags']:
      # Tag like "capex_peak" — check if any component matches
      tag_norm = tag.strip().lower()
      if tag_norm in td_underscored or tag_norm in td:
        matches.append(tag_norm)
        continue
      parts = tag_norm.split('_')
      if all(p in description_tokens for p in parts if p):
        matches.append(tag_norm)
    # Also boost when sector name appears in description
    sector_tags = {'tech', 'energy', 'financials', 'commodities',
                   'consumer', 'real_estate', 'biotech'}
    sector_matched = any(t in sector_tags for t in matches)
    score = len(matches) + (2 if sector_matched else 0)
    if score > 0:
      scored.append({
        'name': a['name'],
        'score': score,
        'matched_tags': matches,
        'all_tags': a['tags'],
        'body_excerpt': a['body'][:1500],
        'drawdown_pct': a.get('drawdown_pct'),
        'duration_months': a.get('duration_months'),
        'direction': a.get('direction'),
      })

  scored.sort(key=lambda r: r['score'], reverse=True)

  return {
    'success': True,
    'thesis_description': thesis_description,
    'top_matches': scored[:top_n],
    'analogues_catalog_size': len(analogues),
    'note': "Matching is by tag-token overlap, not semantic similarity. Tag taxonomy is documented in knowledge/analogues.md preamble. drawdown_pct is signed (-X for bear analogues, +X for bull, None for setup) and feeds the /equity-deep-research Step 15 analogue-calibration rule. Adjust thesis description to include structural tags (capex_cycle, valuation_expansion, supply_constrained, etc.) for better matches.",
  }


def get_industry_etfs(theme: str, top_holdings_per_etf: int = 10) -> Dict[str, Any]:
  """Map a research theme (e.g. 'AI semis', 'energy', 'cloud') to relevant
  ETFs and return their top holdings + weights.

  Acts as the bridge from top-down thematic conviction to bottom-up
  ticker selection. Searches the theme map; if no exact match, tries
  substring match across theme keys. Returns up to 3 ETFs per theme with
  their top N holdings.
  """
  out: Dict[str, Any] = {
    'theme_query': theme,
    'success': True,
    'error': None,
  }

  theme_norm = theme.lower().strip()
  # Exact match first
  etfs = _THEME_TO_ETFS.get(theme_norm)
  matched_themes = [theme_norm] if etfs else []

  # Fuzzy / substring match across keys
  if not etfs:
    matches = {}
    for key, etf_list in _THEME_TO_ETFS.items():
      if theme_norm in key or key in theme_norm:
        matches[key] = etf_list
    if matches:
      # Combine ETF lists (dedup, preserve order)
      seen = set()
      etfs = []
      for k, lst in matches.items():
        for e in lst:
          if e not in seen:
            seen.add(e)
            etfs.append(e)
      matched_themes = list(matches.keys())

  if not etfs:
    return {
      'theme_query': theme,
      'success': False,
      'error': f'No ETF mapping for theme {theme!r}',
      'available_themes': sorted(_THEME_TO_ETFS.keys()),
    }

  out['matched_themes'] = matched_themes
  out['etfs_matched'] = etfs[:5]  # cap at 5 to keep payload bounded

  # Fetch holdings for each ETF
  etf_details = []
  for etf_symbol in out['etfs_matched']:
    detail: Dict[str, Any] = {'symbol': etf_symbol}
    try:
      t = yf.Ticker(etf_symbol)
      info = t.info
      detail['name'] = info.get('longName') or info.get('shortName')
      detail['category'] = info.get('category')
      detail['total_assets'] = info.get('totalAssets')
      detail['expense_ratio'] = info.get('annualReportExpenseRatio')

      fd = t.funds_data
      if fd:
        try:
          th = fd.top_holdings
          holdings = []
          if hasattr(th, 'iterrows'):
            for sym, row in th.head(top_holdings_per_etf).iterrows():
              holdings.append({
                'symbol': str(sym),
                'name': str(row.get('Name', '')),
                'weight_pct': round(float(row.get('Holding Percent', 0)) * 100, 2),
              })
          detail['top_holdings'] = holdings
        except Exception as exc:
          detail['holdings_error'] = f'{type(exc).__name__}: {exc}'

        try:
          sw = fd.sector_weightings
          if isinstance(sw, dict) and sw:
            detail['sector_weightings'] = {k: round(float(v) * 100, 2) for k, v in sw.items()}
        except Exception:
          pass
    except Exception as exc:
      detail['error'] = f'{type(exc).__name__}: {exc}'
    etf_details.append(detail)

  out['etfs'] = etf_details
  return out


def fetch_daily_bars(ticker: str, period: str = '2y') -> "pd.DataFrame":
  """The one daily-OHLCV fetch every price-derived tool goes through.

  Split out so that price-history summaries and the trading/timing metrics
  read the same bars. Two independent yf.Ticker().history() calls with
  different periods or adjustment settings can disagree about the same
  session, and nothing downstream would show which one was wrong.

  auto_adjust=True back-adjusts OHLC for splits and dividends. Same-day
  ratios built from these bars (true range against close, close times
  volume) stay internally consistent because both legs carry the same
  adjustment factor.
  """
  return yf.Ticker(ticker).history(period=period, auto_adjust=True)


# Below this the dividends stripped out of the price path are smaller than the
# rounding on the closes themselves. Above it they are a real understatement of
# any historical market cap built from these bars -- AAPL's dividends after
# 2020-08-28 are 4.75% of that bar.
_DIVIDEND_ADJUSTMENT_MATERIAL_PCT = 1.0

PRICE_BASIS = 'split_and_dividend_adjusted'


def price_adjustment(df: "pd.DataFrame") -> Dict[str, Any]:
  """What `auto_adjust=True` did to the closes in `df`.

  The bars come back back-adjusted for splits and dividends onto the newest
  session's basis, and until now the response said so nowhere -- the basis
  appeared only in a tool description. Paired with a cover-page share count
  that is stated as filed, that is an exact-multiple error:

      NVDA 2024-05-24  close 106.29 x total 2,460,000,000        = $261bn
                       close 106.29 x total_split_adjusted 24.6bn = $2,615bn

  10.0x, which is the split ratio. AAPL on 2020-08-28 is 4.0x.

  Read off the frame rather than fetched: yfinance returns the `Dividends` and
  `Stock Splits` columns from the same request as the prices, so naming the
  adjustment costs nothing and cannot disagree with the bars it describes.
  """
  splits: List[Dict[str, Any]] = []
  factor = 1.0
  if 'Stock Splits' in df:
    for stamp, value in df['Stock Splits'].items():
      try:
        ratio = float(value)
      except (TypeError, ValueError):
        continue
      # A ratio of zero is "no split on this bar"; a ratio of one is a split
      # that changed nothing, which is a different claim and not one this
      # toolset makes anywhere else.
      if ratio != ratio or ratio <= 0 or ratio == 1.0:
        continue
      splits.append({'date': pd.Timestamp(stamp).strftime('%Y-%m-%d'),
                     'ratio': ratio})
      factor *= ratio

  dividends: List[float] = []
  if 'Dividends' in df:
    for value in df['Dividends']:
      try:
        amount = float(value)
      except (TypeError, ValueError):
        continue
      if amount == amount and amount > 0:
        dividends.append(amount)

  oldest_close = None
  if 'Close' in df and len(df):
    try:
      oldest_close = float(df['Close'].iloc[0])
    except (TypeError, ValueError):
      oldest_close = None

  removed = float(sum(dividends))
  pct = (removed / oldest_close * 100.0
         if oldest_close and removed else (0.0 if oldest_close else None))

  return {
    'auto_adjust': True,
    'basis': PRICE_BASIS,
    'splits_in_window': splits,
    'cumulative_split_factor': factor,
    'dividends_in_window': len(dividends),
    'dividends_per_share_removed': removed,
    'dividends_pct_of_oldest_close': pct,
  }


def price_adjustment_warnings(ticker: str,
                              adjustment: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Warnings for an adjustment a caller could pair something wrong with.

  Only when there is something to say. `price_basis` is a field on every
  response because it is true of every bar; a warning that fired on every
  response would be a property of the tool wearing a warning's clothes.
  """
  entries: List[Dict[str, Any]] = []
  splits = adjustment['splits_in_window']
  if splits:
    described = ", ".join(
        (f"{s['ratio']:g}-for-1" if s['ratio'] >= 1.0
         else f"1-for-{1.0 / s['ratio']:g}") + f" on {s['date']}"
        for s in splits)
    factor = adjustment['cumulative_split_factor']
    entries.append({
      'code': 'prices_split_adjusted',
      'message': (
        f"{ticker}: every close here is back-adjusted onto the newest "
        f"session's basis for {described}, so a close from before the "
        f"earliest of those dates is stated in {factor:g}x as many shares as "
        f"the company had at the time. Do not multiply it by a cover-page "
        f"share count: get_share_count_series reports `total` as filed and "
        f"`total_split_adjusted` on this basis, and only the second one "
        f"pairs with these prices."),
      'splits_in_window': splits,
      'cumulative_split_factor': factor,
    })
  pct = adjustment['dividends_pct_of_oldest_close']
  if pct is not None and pct >= _DIVIDEND_ADJUSTMENT_MATERIAL_PCT:
    entries.append({
      'code': 'prices_dividend_adjusted',
      'message': (
        f"{ticker}: {adjustment['dividends_in_window']} dividends totalling "
        f"{adjustment['dividends_per_share_removed']:.4g} per share have also "
        f"been taken out of the price path, {pct:.2f}% of the oldest close in "
        f"this window. A historical market capitalisation built from these "
        f"closes understates the price actually quoted at the time by that "
        f"much; get_corporate_actions carries the payments themselves."),
      'dividends_pct_of_oldest_close': pct,
    })
  return entries


def get_price_history(ticker: str, period: str = '2y',
                      include_recent_bars: int = 20) -> Dict[str, Any]:
  """Historical OHLCV summary from yfinance.

  Returns aggregate metrics rather than the raw bars (which would blow out
  MCP payload size): returns over 1M/3M/6M/YTD/1Y/3Y, realized volatility
  over the same windows, 52-week high/low with dates, max drawdown from
  trailing-12-month peak, and the most recent N daily OHLCV bars for
  technical reference.
  """
  from datetime import datetime, timedelta

  try:
    df = fetch_daily_bars(ticker, period)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'yfinance history fetch failed: {type(e).__name__}: {e}'}

  if df is None or df.empty:
    return {'ticker': ticker, 'success': False, 'error': 'no price history returned'}

  close = df['Close']
  returns = close.pct_change().dropna()

  def _ret_over(days: int):
    if len(close) <= days:
      return None
    return round(((close.iloc[-1] / close.iloc[-days - 1]) - 1) * 100, 2)

  def _vol_over(days: int):
    if len(returns) < days:
      return None
    sub = returns.iloc[-days:]
    return round(float(sub.std() * (252 ** 0.5) * 100), 2)

  # Fields whose window the request could not support. Named rather than
  # silently substituted or silently dropped.
  window_short_fields: List[str] = []

  # YTD return: from first trading day of this calendar year
  today = datetime.now()
  jan1 = datetime(today.year, 1, 1)
  ytd_idx = df.index[df.index >= pd.Timestamp(jan1, tz=df.index.tz)] if df.index.tz else \
            df.index[df.index >= jan1]
  ytd_ret = None
  since_listing_ret = None
  if len(ytd_idx) > 0:
    # Two things have to be true, and an earlier version checked only one.
    #
    # The REQUEST must reach back before 1 January. Comparing the frame start
    # to the first in-frame January bar is trivially true, which is why a
    # six-month window reported AMD's "YTD" as 140.21% against a true 115.21%
    # -- on that frame both dates are late February.
    #
    # And the STOCK must have traded then. Letting a young listing keep its
    # YTD looked reasonable -- the frame starts late because the stock did --
    # and produced -41.44% for CBRS, listed 2026-05-14. That is a return since
    # listing wearing a year-to-date label, and a caller ranking names on YTD
    # compares one company's five months against another's eight.
    _PERIOD_DAYS = {'1mo': 31, '3mo': 92, '6mo': 183, 'ytd': 366, '1y': 366,
                    '2y': 731, '5y': 1827, '10y': 3653, 'max': 10 ** 5}
    requested_days = _PERIOD_DAYS.get(str(period).lower(), 0)
    days_since_jan1 = (df.index[-1] - pd.Timestamp(jan1, tz=df.index.tz)).days \
        if df.index.tz else (df.index[-1] - jan1).days
    year_start = pd.Timestamp(jan1, tz=df.index.tz) if df.index.tz else jan1
    traded_in_january = df.index[0] < year_start
    window_reaches_back = requested_days >= days_since_jan1

    if window_reaches_back and traded_in_january:
      ytd_start = close.loc[ytd_idx[0]]
      if ytd_start:
        ytd_ret = round(((close.iloc[-1] / ytd_start) - 1) * 100, 2)
    elif window_reaches_back:
      # Not a short window -- a short life. The honest figure is the return
      # since it listed, under its own name.
      first_close = close.iloc[0]
      if first_close:
        since_listing_ret = round(((close.iloc[-1] / first_close) - 1) * 100, 2)
      window_short_fields.append('returns_pct.ytd (listed mid-year)')
    else:
      window_short_fields.append('returns_pct.ytd')

  # 52w stats. Checked as a DATE SPAN, not a row count: period="1y" returns
  # 251 bars, so a 252-row test would null the field on the most natural
  # request for it.
  #
  # The old `df.iloc[-252:] if len(df) >= 252 else df` silently substituted
  # whatever the request fetched and kept the field name, so period="6mo"
  # reported a "52-week low" of 188.22 for AMD against a true 149.22 -- 26%
  # too high, no warning. The same frame carries max_drawdown_12m.
  window_days = (df.index[-1] - df.index[0]).days
  has_52w = window_days >= 350
  win52 = df.iloc[-252:] if has_52w else None
  high_idx = low_idx = None
  max_dd = max_dd_date = max_dd_peak_close = None
  if win52 is not None:
    high_idx = win52['High'].idxmax()
    low_idx = win52['Low'].idxmin()
    running_max = win52['Close'].expanding().max()
    drawdown = (win52['Close'] / running_max - 1) * 100
    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin()
    max_dd_peak_close = float(running_max.loc[max_dd_date])
  else:
    window_short_fields.extend(['fifty_two_week', 'max_drawdown_12m'])

  # Recent bars
  recent = df.tail(include_recent_bars).copy()
  recent_bars = []
  for ts, row in recent.iterrows():
    recent_bars.append({
      'date': ts.strftime('%Y-%m-%d'),
      'open': round(float(row['Open']), 2),
      'high': round(float(row['High']), 2),
      'low': round(float(row['Low']), 2),
      'close': round(float(row['Close']), 2),
      'volume': int(row['Volume']),
    })

  adjustment = price_adjustment(df)

  return {
    'ticker':            ticker.upper(),
    'success':           True,
    'error':             None,
    'period_requested':  period,
    'bars_returned':     len(df),
    'price_basis':       adjustment['basis'],
    'price_adjustment':  adjustment,
    'warnings':          price_adjustment_warnings(ticker.upper(), adjustment)
                         + ([{
                             'code': 'window_too_short_for_field',
                             'message': (
                               f"{', '.join(window_short_fields)} "
                               f"{'are' if len(window_short_fields) > 1 else 'is'} "
                               f"not reported: period={period!r} covers "
                               f"{window_days} days, which cannot answer a "
                               f"52-week or year-to-date question. Request a "
                               f"longer period for {'them' if len(window_short_fields) > 1 else 'it'}."),
                             'fields': window_short_fields,
                           }] if window_short_fields else []),
    'date_range': {
      'start': df.index[0].strftime('%Y-%m-%d'),
      'end':   df.index[-1].strftime('%Y-%m-%d'),
    },
    'current_close':     round(float(close.iloc[-1]), 2),
    'returns_pct': {
      '1m':  _ret_over(21),
      '3m':  _ret_over(63),
      '6m':  _ret_over(126),
      'ytd': ytd_ret,
      # Present only when the company listed after 1 January, where it is the
      # answer year-to-date cannot give.
      'since_listing': since_listing_ret,
      '1y':  _ret_over(252),
      '3y':  _ret_over(252 * 3),
    },
    'realized_vol_annualized_pct': {
      '30d':  _vol_over(30),
      '90d':  _vol_over(90),
      '180d': _vol_over(180),
      '1y':   _vol_over(252),
    },
    'fifty_two_week': None if win52 is None else {
      'high':     round(float(win52['High'].max()), 2),
      'high_date': high_idx.strftime('%Y-%m-%d'),
      'low':      round(float(win52['Low'].min()), 2),
      'low_date': low_idx.strftime('%Y-%m-%d'),
    },
    'max_drawdown_12m': None if win52 is None else {
      'drawdown_pct':       round(max_dd, 2),
      'trough_date':        max_dd_date.strftime('%Y-%m-%d'),
      'peak_close_before':  round(max_dd_peak_close, 2),
      'trough_close':       round(float(win52['Close'].loc[max_dd_date]), 2),
    },
    'recent_bars':       recent_bars,
  }


def get_short_interest(ticker: str) -> Dict[str, Any]:
  """Short-interest snapshot from yfinance (underlying source: FINRA biweekly).

  Returns shares short, short ratio (days to cover), percent of float, and
  MoM trend (current vs prior-month shares short). Crowded shorts can
  signal squeeze risk or strong bear conviction; low shorts on a high-
  quality name indicate institutional acceptance.
  """
  from datetime import datetime, timezone

  try:
    t = yf.Ticker(ticker)
    info = t.info
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'yfinance init failed: {type(e).__name__}: {e}'}

  unresolved = unresolved_symbol_error(ticker, t, info)
  if unresolved is not None:
    return unresolved

  def _epoch_to_iso(v):
    try:
      return datetime.fromtimestamp(int(v), tz=timezone.utc).strftime('%Y-%m-%d')
    except (TypeError, ValueError):
      return None

  shares_short = info.get('sharesShort')
  shares_short_prior = info.get('sharesShortPriorMonth')
  mom_change_pct = None
  if shares_short and shares_short_prior:
    mom_change_pct = round(((shares_short / shares_short_prior) - 1) * 100, 2)

  short_pct_float = info.get('shortPercentOfFloat')
  short_ratio = info.get('shortRatio')

  # The same provider field on the same footing as in get_market_data, plus the
  # two impossibilities it produces here. GOOGL's float_shares (10.88bn) comes
  # back larger than its shares_outstanding (5.87bn) because the float covers
  # every class and the share count covers one; insider ownership taken as
  # 1 - float/shares computes to -85%. BRK-B's shares_short (11.9m) exceeds its
  # float_shares (1.23m) for the mirror-image reason. Neither is a state a
  # company can be in, and both were reported without comment.
  float_shares = info.get('floatShares') or None
  shares_out = info.get('sharesOutstanding') or None
  basis = share_count_basis(info.get('marketCap'),
                            info.get('currentPrice')
                            or info.get('regularMarketPrice'),
                            shares_out,
                            provider_implied=info.get('impliedSharesOutstanding'))
  warnings: List[Dict[str, Any]] = []
  mismatch = share_basis_warning(ticker.upper(), basis)
  if mismatch is not None:
    warnings.append(mismatch)
  if float_shares and shares_out and float_shares > shares_out:
    all_classes = basis['shares_outstanding_all_classes']
    remedy = (f" Compare the float against shares_outstanding_all_classes "
              f"({all_classes:,.0f}), which is counted the same way."
              if all_classes else "")
    warnings.append({
      'code': 'float_exceeds_shares_outstanding',
      'message': (
        f"{ticker.upper()}: float_shares ({float_shares:,.0f}) exceeds "
        f"shares_outstanding ({shares_out:,.0f}), which no company can do. "
        f"The float is counted across every share class and the share count "
        f"covers one of them, so the two do not subtract: insider ownership "
        f"taken as 1 - float_shares / shares_outstanding comes out at "
        f"{(1 - float_shares / shares_out) * 100:.0f}%." + remedy),
    })
  if shares_short and float_shares and shares_short > float_shares:
    warnings.append({
      'code': 'short_interest_exceeds_float',
      'message': (
        f"{ticker.upper()}: shares_short ({shares_short:,.0f}) exceeds "
        f"float_shares ({float_shares:,.0f}), so the two are not counted on "
        f"the same share class. short_pct_of_float is the provider's own "
        f"figure and is not shares_short / float_shares -- dividing them here "
        f"gives {shares_short / float_shares * 100:,.0f}%."),
    })

  # Sentiment label based on % of float
  signal = 'unknown'
  if short_pct_float is not None:
    if short_pct_float < 0.02:
      signal = 'low_short_interest'
    elif short_pct_float < 0.05:
      signal = 'moderate_short_interest'
    elif short_pct_float < 0.10:
      signal = 'elevated_short_interest'
    else:
      signal = 'crowded_short_squeeze_risk'

  return {
    'ticker': ticker.upper(),
    'success': True,
    'error': None,
    'shares_short':              int(shares_short) if shares_short else None,
    'shares_short_prior_month':  int(shares_short_prior) if shares_short_prior else None,
    'mom_change_pct':            mom_change_pct,
    'short_ratio_days_to_cover': float(short_ratio) if short_ratio else None,
    'short_pct_of_float':        round(float(short_pct_float) * 100, 3) if short_pct_float else None,
    'float_shares':              int(float_shares) if float_shares else None,
    'shares_outstanding':        int(shares_out) if shares_out else None,
    'shares_outstanding_basis':  basis['basis'],
    'shares_outstanding_all_classes': basis['shares_outstanding_all_classes'],
    'as_of_date':                _epoch_to_iso(info.get('dateShortInterest')),
    'prior_month_date':          _epoch_to_iso(info.get('sharesShortPreviousMonthDate')),
    'signal':                    signal,
    'source':                    'yfinance (underlying: FINRA biweekly short interest)',
    'warnings':                  warnings,
  }


def _safe_float(v: Any, default: float = 0.0) -> float:
  """float() that maps None, non-numeric, and NaN to a default.

  yfinance returns NaN (a truthy float) for ask/impliedVolatility on illiquid
  strikes; `float(x or 0)` leaves NaN intact, which then poisons the straddle
  math and serializes to invalid JSON. NaN != NaN, so `f == f` catches it."""
  try:
    f = float(v)
  except (TypeError, ValueError):
    return default
  return f if f == f else default


def _leg_price(opt: Dict) -> Tuple[float, bool]:
  """Price of one ATM leg for the straddle. Prefer the live ask; when ask is 0
  (market closed) fall back to last_price then bid. Returns (price, used_fallback);
  used_fallback=True means the quote is stale (after-hours / illiquid)."""
  # _safe_float each key separately: a NaN ask is truthy, so `ask or ask_price`
  # would swallow a valid OpenBB ask_price behind a NaN yfinance ask.
  ask = _safe_float(opt.get("ask")) or _safe_float(opt.get("ask_price"))
  if ask > 0:
    return ask, False
  for k in ("last_price", "bid"):
    v = _safe_float(opt.get(k))
    if v > 0:
      return v, True
  return 0.0, True


def compute_implied_move(spot: float, atm_call_ask: float,
            atm_put_ask: float) -> Dict[str, Any]:
  straddle = _safe_float(atm_call_ask) + _safe_float(atm_put_ask)
  implied_move_pct = straddle / spot if spot > 0 else 0.0
  return {"implied_move_pct": round(implied_move_pct, 4), "straddle_cost": straddle}


_ATM_GAP_THRESHOLD = 0.08  # nearest strike >8% from spot → treat as ATM missing
_PARITY_TOLERANCE = 0.05   # |C-P-(S-K)|/S above this → ask quotes are junk


def _us_market_today():
  """Today's date in US-market terms. UTC is a day ahead of ET every evening
  (00:00-05:00 UTC) — exactly when after-hours pre-earnings research runs —
  which made a tomorrow-ET front expiry look like 'today' and get dropped."""
  try:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()
  except Exception:
    # tzdata unavailable: fixed ET-standard offset (off by 1h in summer,
    # which only matters in the 04:00-05:00 UTC sliver)
    return datetime.now(timezone(timedelta(hours=-5))).date()


def _find_atm_options(rows: List[Dict], spot: float,
           target_expiry: Optional[str] = None):
  """Find the nearest ATM call and put. Returns (None, None, None) if ATM gap > 8%."""
  if not rows:
    return None, None, None

  expiries = sorted({r.get("expiration") or r.get("expiration_date", "") for r in rows
           if r.get("expiration") or r.get("expiration_date")})
  if not expiries:
    return None, None, None

  today = _us_market_today().isoformat()
  future = [e for e in expiries if e > today]
  chosen_expiry = future[0] if future else expiries[-1]
  if target_expiry:
    chosen_expiry = target_expiry

  chain = [r for r in rows
      if (r.get("expiration") or r.get("expiration_date", "")) == chosen_expiry]
  calls = [r for r in chain if (r.get("option_type") or r.get("optionType", "")).lower() == "call"]
  puts  = [r for r in chain if (r.get("option_type") or r.get("optionType", "")).lower() == "put"]

  def _effective_price(r):
    # Ask is the right straddle cost when the market is open; after hours
    # ask is 0, so fall back to last_price then bid for ATM selection.
    for k in ("ask", "ask_price", "last_price", "bid"):
      v = _safe_float(r.get(k))
      if v > 0:
        return v
    return 0.0

  def nearest_atm(options):
    if not options:
      return None
    # Prefer strikes with a positive price signal; a 0-price ATM strike
    # would yield a garbage straddle.
    quoted = [o for o in options if _effective_price(o) > 0]
    pool = quoted or options
    best = min(pool, key=lambda r: abs(_safe_float(r.get("strike")) - spot))
    # Guard: reject if gap is too large (truncated chain)
    if spot > 0 and abs(_safe_float(best.get("strike")) - spot) / spot > _ATM_GAP_THRESHOLD:
      return None
    return best

  atm_call = nearest_atm(calls)
  atm_put  = nearest_atm(puts)
  return atm_call, atm_put, chosen_expiry


def _chain_to_rows(chain, expiry: str) -> List[Dict]:
  """Flatten one yfinance option_chain result into the row shape the ATM
  helpers expect. yfinance uses camelCase DataFrame columns; the helpers use
  snake_case keys, so the rename happens here and nowhere else."""
  rows: List[Dict] = []
  for df, otype in ((chain.calls, 'call'), (chain.puts, 'put')):
    for _, row in df.iterrows():
      rows.append({
        'expiration': expiry,
        'option_type': otype,
        'strike': _safe_float(row.get('strike')),
        'ask': _safe_float(row.get('ask')),
        'bid': _safe_float(row.get('bid')),
        'last_price': _safe_float(row.get('lastPrice')),
        'implied_volatility': _safe_float(row.get('impliedVolatility')),
      })
  return rows


def _straddle_legs(atm_call: Dict, atm_put: Dict,
                   spot: float) -> Tuple[float, float, bool]:
  """Price both straddle legs, with a put-call parity sanity check.

  C - P should approximate S - K for same-strike legs. A nonzero ask is not
  necessarily a SANE ask -- junk wide quotes left at the close pass a bare >0
  check (live ORCL: call 6.75 / put 28.35 at strike 237.5 with spot 236.34, a
  $21 violation). On gross violation both legs are rebuilt from last_price/bid
  and the result is flagged stale.

  Returns (call_price, put_price, quotes_stale)."""
  call_px, call_stale = _leg_price(atm_call)
  put_px, put_stale = _leg_price(atm_put)
  stale = call_stale or put_stale

  strike = _safe_float(atm_call.get('strike'))
  if (call_px > 0 and put_px > 0 and spot > 0
      and abs((call_px - put_px) - (spot - strike)) / spot > _PARITY_TOLERANCE):
    def _no_ask(opt):
      return {k: v for k, v in opt.items() if k not in ('ask', 'ask_price')}
    c2, _ = _leg_price(_no_ask(atm_call))
    p2, _ = _leg_price(_no_ask(atm_put))
    # Only rebuild if a fallback actually exists -- otherwise keep the asks
    # rather than returning a fabricated zero.
    if c2 > 0 and p2 > 0:
      call_px, put_px = c2, p2
      stale = True

  return call_px, put_px, stale


def _find_expiry(exp_dates: List[Tuple[str, int]], target_dte: int) -> Tuple[str, int]:
  """Pick the listed expiry closest to `target_dte`, preferring at-or-beyond.

  Among expirations with DTE >= target_dte, returns the one nearest to
  target_dte. If none are at or beyond the target, falls back to the
  overall nearest expiry (nearer-than-target).

  Consequence: this floors the selection at (approximately) target_dte
  rather than snapping to whatever is nearest overall. For the 7d term-
  structure bucket, that means a 1-DTE Friday weekly is skipped in favor
  of e.g. an 8-DTE expiry -- the old select_expiries() had no such floor
  and would have picked the 1-DTE contract. That floor is intentional:
  get_options_metrics reuses this same 7d selection as the front expiry
  for the ATM straddle, and implied_move_pct from a 1-DTE straddle
  understates the move enough to let the `implied_move_pct > 0.20` risk
  gate under-fire (see .claude/skills/preearnings-research/SKILL.md).
  """
  future = [(e, d) for e, d in exp_dates if d >= target_dte]
  pool = future if future else exp_dates
  return min(pool, key=lambda x: abs(x[1] - target_dte))


def get_options_metrics(ticker: str) -> Dict[str, Any]:
  """Compute key options-market metrics from yfinance option chains.

  Returns:
    - term structure (ATM IV at ~7d/30d/60d/90d expirations)
    - 30d put/call skew (downside vs. upside IV)
    - nearest-expiry open interest + volume put/call ratios

  Filters out illiquid contracts (bid == 0) to avoid yfinance's garbage IV
  values on deep ITM/OTM strikes. ATM is defined as the strike closest to
  spot. Skew compares 0.9*spot put IV to 1.1*spot call IV.
  """
  from datetime import datetime

  out: Dict[str, Any] = {'ticker': ticker.upper(), 'success': True, 'error': None}

  try:
    t = yf.Ticker(ticker)
    info = t.info
    spot = info.get('currentPrice') or info.get('regularMarketPrice')
    if not spot:
      return {'ticker': ticker, 'success': False, 'error': 'no spot price'}
    out['spot_price'] = float(spot)

    exps = list(t.options)
    if not exps:
      return {'ticker': ticker, 'success': False, 'error': 'no options listed'}
    out['expirations_available'] = len(exps)

    today = _us_market_today()
    exp_dates = []
    for e in exps:
      try:
        d = (datetime.strptime(e, '%Y-%m-%d').date() - today).days
        exp_dates.append((e, d))
      except ValueError:
        continue

    def _atm_iv(chain, side: str) -> Optional[float]:
      df = chain.calls if side == 'call' else chain.puts
      # Filter out yfinance's IV=0.00001 sentinel for inactive contracts but
      # keep bid==0 contracts (some ATM strikes show bid=0 but valid IV).
      df = df[df['impliedVolatility'] > 0.01]
      if df.empty:
        return None
      idx = (df['strike'] - spot).abs().idxmin()
      return float(df.loc[idx, 'impliedVolatility'])

    # Term structure
    term_structure = {}
    front_chain = None
    front_expiry = None
    for label, target_days in [('7d', 7), ('30d', 30), ('60d', 60), ('90d', 90)]:
      exp, dte = _find_expiry(exp_dates, target_days)
      try:
        chain = t.option_chain(exp)
      except Exception:
        term_structure[label] = {'expiry': exp, 'dte': dte, 'error': 'chain fetch failed'}
        continue
      if label == '7d':
        front_chain = chain
        front_expiry = exp
      call_iv = _atm_iv(chain, 'call')
      put_iv = _atm_iv(chain, 'put')
      atm_iv = None
      if call_iv is not None and put_iv is not None:
        atm_iv = (call_iv + put_iv) / 2
      elif call_iv is not None:
        atm_iv = call_iv
      elif put_iv is not None:
        atm_iv = put_iv
      term_structure[label] = {
        'expiry':         exp,
        'dte':            dte,
        'atm_call_iv':    round(call_iv, 4) if call_iv is not None else None,
        'atm_put_iv':     round(put_iv, 4) if put_iv is not None else None,
        'atm_iv':         round(atm_iv, 4) if atm_iv is not None else None,
      }
    out['term_structure'] = term_structure

    # ATM straddle from the front expiry — reuses the 7d chain already
    # fetched for the term structure rather than refetching.
    if front_chain is not None:
      rows = _chain_to_rows(front_chain, front_expiry)
      call, put, chosen_expiry = _find_atm_options(rows, spot)
      if call is None or put is None:
        out['implied_move'] = {'error': 'no ATM strike within threshold'}
      else:
        call_px, put_px, stale = _straddle_legs(call, put, spot)
        moved = compute_implied_move(spot, call_px, put_px)
        out['implied_move'] = {
          **moved,
          'front_expiry': chosen_expiry,
          'quotes_stale': stale,
        }
    else:
      out['implied_move'] = {'error': 'front expiry chain unavailable'}

    # 30d skew
    exp_30, _ = _find_expiry(exp_dates, 30)
    try:
      chain30 = t.option_chain(exp_30)
      calls = chain30.calls[chain30.calls['impliedVolatility'] > 0.01]
      puts = chain30.puts[chain30.puts['impliedVolatility'] > 0.01]
      otm_put_idx = (puts['strike'] - 0.9 * spot).abs().idxmin() if not puts.empty else None
      otm_call_idx = (calls['strike'] - 1.1 * spot).abs().idxmin() if not calls.empty else None
      put_iv_90 = float(puts.loc[otm_put_idx, 'impliedVolatility']) if otm_put_idx is not None else None
      call_iv_110 = float(calls.loc[otm_call_idx, 'impliedVolatility']) if otm_call_idx is not None else None
      skew = None
      if put_iv_90 is not None and call_iv_110 is not None:
        skew = round(put_iv_90 - call_iv_110, 4)
      out['put_call_skew_30d'] = {
        'value':       skew,
        'put_iv_90pct': round(put_iv_90, 4) if put_iv_90 is not None else None,
        'call_iv_110pct': round(call_iv_110, 4) if call_iv_110 is not None else None,
        'expiry':      exp_30,
        'note':        '0.9*spot put IV minus 1.1*spot call IV; positive=downside fear, negative=upside speculation',
      }
    except Exception as exc:
      out['put_call_skew_30d'] = {'error': str(exc)}

    # Nearest-expiry OI and volume aggregates
    try:
      chain_near = t.option_chain(exps[0])
      c_oi = int(chain_near.calls['openInterest'].fillna(0).sum())
      p_oi = int(chain_near.puts['openInterest'].fillna(0).sum())
      c_vol = int(chain_near.calls['volume'].fillna(0).sum())
      p_vol = int(chain_near.puts['volume'].fillna(0).sum())
      out['nearest_expiry_activity'] = {
        'expiry':                  exps[0],
        'call_open_interest':      c_oi,
        'put_open_interest':       p_oi,
        'put_call_oi_ratio':       round(p_oi / c_oi, 3) if c_oi else None,
        'call_volume':             c_vol,
        'put_volume':              p_vol,
        'put_call_volume_ratio':   round(p_vol / c_vol, 3) if c_vol else None,
      }
    except Exception as exc:
      out['nearest_expiry_activity'] = {'error': str(exc)}

    # Data-quality check. yfinance sometimes returns sentinel IV values
    # (powers of 2 fractions like 0.0156, 0.0625, 0.125) when the underlying
    # market snapshot is stale or contracts are illiquid. Real ATM IVs for
    # US large-caps live in the 0.15-0.60 range; anything below 0.08 is
    # almost certainly bad data.
    iv_vals = [
      entry.get('atm_iv') for entry in term_structure.values()
      if isinstance(entry, dict) and entry.get('atm_iv') is not None
    ]
    iv_quality = 'ok'
    iv_quality_notes = []
    if iv_vals:
      if max(iv_vals) < 0.08:
        iv_quality = 'suspect_iv_sentinel'
        iv_quality_notes.append(
          'all ATM IVs < 0.08 — likely yfinance sentinel values, not real implied volatility')
      elif len(set(round(v, 4) for v in iv_vals)) == 1:
        iv_quality = 'suspect_iv_constant'
        iv_quality_notes.append('all ATM IVs identical across tenors — suspect data')
    else:
      iv_quality = 'no_iv_data'
      iv_quality_notes.append('no ATM IV could be extracted')

    # Volume ratios are usually still reliable even when IV is sentinel
    out['data_quality'] = {
      'iv_status':   iv_quality,
      'notes':       iv_quality_notes,
      'volume_data_usable': True,
    }

    return out

  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'get_options_metrics failed: {type(e).__name__}: {e}'}


def get_institutional_holdings(ticker: str, top_n: int = 10) -> Dict[str, Any]:
  """Pull aggregated 13F institutional holdings via yfinance.

  Source layering: Yahoo aggregates SEC 13F-HR filings server-side, so the
  underlying data is SEC-tier but the aggregation/freshness is Yahoo's.
  Tagged as vendor-tier in the playbook hierarchy.

  Returns top N institutional holders + top N mutual fund holders with
  shares, market value, percent of shares outstanding, and quarter-over-
  quarter percent change in position. Also returns aggregate institutional
  ownership stats (institutions %, insiders %, total institution count).
  """
  out: Dict[str, Any] = {
    "ticker": ticker.upper(),
    "success": True,
    "error": None,
    "source": "yfinance (aggregates SEC 13F-HR)",
  }

  _handle = yf.Ticker(ticker)
  unresolved = unresolved_symbol_error(ticker.upper(), _handle)
  if unresolved is not None:
    return unresolved

  try:
    t = yf.Ticker(ticker)
  except Exception as exc:
    return {"ticker": ticker, "success": False,
            "error": f"yfinance Ticker init failed: {type(exc).__name__}: {exc}"}

  # Aggregate stats (institutions %, insiders %, count)
  try:
    mh = t.major_holders
    if mh is not None and not mh.empty:
      vals = {}
      for idx, row in mh.iterrows():
        try:
          vals[str(idx)] = float(row['Value'])
        except (KeyError, TypeError, ValueError):
          continue
      out['aggregate'] = {
        'insiders_pct': vals.get('insidersPercentHeld'),
        'institutions_pct': vals.get('institutionsPercentHeld'),
        'institutions_float_pct': vals.get('institutionsFloatPercentHeld'),
        'institutions_count': int(vals['institutionsCount']) if 'institutionsCount' in vals else None,
      }
  except Exception as exc:
    out['aggregate_error'] = f"{type(exc).__name__}: {exc}"

  def _holders_to_list(df, cap: int) -> List[Dict[str, Any]]:
    rows = []
    for _, r in df.head(cap).iterrows():
      pct_change = r.get('pctChange')
      try:
        pct_change_f = float(pct_change) if pd.notna(pct_change) else None
      except (TypeError, ValueError):
        pct_change_f = None
      rows.append({
        'holder':         str(r.get('Holder', '')),
        'date_reported':  str(r.get('Date Reported', '')),
        'pct_held':       float(r.get('pctHeld', 0)) if pd.notna(r.get('pctHeld', 0)) else None,
        'shares':         int(r.get('Shares', 0)) if pd.notna(r.get('Shares', 0)) else None,
        'value_usd':      float(r.get('Value', 0)) if pd.notna(r.get('Value', 0)) else None,
        'pct_change_qoq': pct_change_f,
      })
    return rows

  # Institutional (13F) holders
  try:
    ih = t.institutional_holders
    if ih is not None and not ih.empty:
      out['institutional_holders'] = _holders_to_list(ih, top_n)
  except Exception as exc:
    out['institutional_error'] = f"{type(exc).__name__}: {exc}"

  # Mutual fund holders (NPORT-P filings, also aggregated by Yahoo)
  try:
    mf = t.mutualfund_holders
    if mf is not None and not mf.empty:
      out['mutualfund_holders'] = _holders_to_list(mf, top_n)
  except Exception as exc:
    out['mutualfund_error'] = f"{type(exc).__name__}: {exc}"

  return out


def calculate_percentiles(data: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
  percentiles = {}
  # build the list of values based on key
  values = [d[key] for d in data if d.get(key) is not None]

  if not values:
    print(f'No valid data found for key: {str(key)}', file=sys.stderr)
    return {}

  # calcaute statistics
  percentiles['mean'] = np.mean(values)
  percentiles['median'] = np.median(values)
  percentiles['q1'] = np.percentile(values, 25)
  percentiles['q3'] = np.percentile(values, 75)
  percentiles['low'] = np.min(values)
  percentiles['high'] = np.max(values)

  return percentiles
if __name__ == "__main__":
  data = get_data("MSFT")
  print(data['cash'])
  print(data['totalDebt'])
  print(data['sharesOutstanding'])
