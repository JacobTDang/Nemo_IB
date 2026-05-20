import yfinance as yf
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys

def get_data(ticker: str) -> Dict[str, Any]:
  data = {}

  company = yf.Ticker(ticker)
  book_value = None
  operating_income = None

  # CONSTANT POSSIBLE KEYS
  BOOK_VAL_KEYS : List = ['Stockholders Equity', 'Total Stockholder Equity','Total Equity Gross Minority Interest', 'Common Stock Equity',]
  OPERATING_INCOME_KEYS : List = ['Operating Income','Ebit']
  REVENUE_KEYS : List[str] = ['Total Revenue','Revenue','Net Sales']
  NET_INCOME_KEYS: List[str] = ['Net Income','Net Income To Common', 'Net Income From Continuing Ops']

  # get necessary information
  info = company.info
  data['ticker'] = ticker
  data['marketCap'] = info.get('marketCap')
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
  data['cash'] = info.get('totalCash', 0)
  data['totalDebt'] = info.get('totalDebt', 0)
  data['sharesOutstanding'] = info.get('sharesOutstanding')
  data['beta'] = info.get('beta')

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

  except Exception as e:
    print(f"Could not get the operating income from income statement for {ticker} : {str(e)}", file=sys.stderr)


  # calculate multiples -- each wrapped independently so one failure doesn't skip all
  try:
    if data['marketCap'] is not None and data['netIncomeToCommon'] is not None:
      data['pe_ratio'] = data['marketCap'] / data['netIncomeToCommon']
  except Exception as e:
    print(f'Error calculating P/E ratio for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['marketCap'] is not None and book_value is not None:
      data['pb_ratio'] = data['marketCap'] / book_value
  except Exception as e:
    print(f'Error calculating P/B ratio for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['enterpriseValue'] is not None and data['revenue'] is not None:
      data['ev_revenue'] = data['enterpriseValue'] / data['revenue']
  except Exception as e:
    print(f'Error calculating EV/Revenue for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['enterpriseValue'] is not None and data['EBITDA'] is not None:
      data['ev_ebitda'] = data['enterpriseValue'] / data['EBITDA']
  except Exception as e:
    print(f'Error calculating EV/EBITDA for {ticker}: {str(e)}', file=sys.stderr)

  try:
    if data['EBIT'] is not None and data['enterpriseValue'] is not None:
      data['ev_ebit'] = data['enterpriseValue'] / data['EBIT']
  except Exception as e:
    print(f'Error calculating EV/EBIT for {ticker}: {str(e)}', file=sys.stderr)

  return data


def find_key(possible_key : List[str], indexes: pd.Index) -> Optional[str]:
  # find key with in the list of indexes
  for key in possible_key:
    if key in indexes:
      return str(key)

  # if that fails then we use a llm fallback - TODO later when I implement quantiazation for models
  print(f'Unable to find key, using llm to compare indexes to possible keys: {possible_key}', file=sys.stderr)

  # complete failure
  print('complete failure, key DNE', file=sys.stderr)
  return None

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
