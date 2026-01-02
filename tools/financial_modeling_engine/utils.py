import yfinance as yf
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
async def get_data(ticker: str) -> Dict[str, Any]:
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
  data['marketCap'] = info.get('marketCap')
  data['revenue'] = info.get('totalRevenue') # gets the trailing revenue over the last 12 months
  data['EBITDA'] = info.get('ebitda', 'N/A')
  data['netIncomeToCommon'] = info.get('netIncomeToCommon')
  data['enterpriseValue'] = info.get('enterpriseValue')

  # safely get the balancesheet and book_value
  try:
    balance_sheet = company.balance_sheet
    key = await find_key(BOOK_VAL_KEYS, balance_sheet.index)
    # .loc finds the row with key and .iloc will get the first col / most recent year
    book_value = balance_sheet.loc[key].iloc[0]
  except Exception as e:
    print(f'Could not get book value for {ticker} : {str(e)}')

  # safely get the operating_income from the income statement
  try:
    income_statement = company.income_stmt
    key = await find_key(OPERATING_INCOME_KEYS, income_statement.index)
    operating_income = income_statement.loc[key].iloc[0]
    data[key] = operating_income

  except Exception as e:
    print(f"Could not get the operating income from income statement for {ticker} : {str(e)}")


  # verify that both market cap and net income are valid before calculating multiples
  try:
    # calculate multiples
    if data['marketCap'] is not None and data['netIncomeToCommon'] is not None:
      pe_ratio = data['marketCap'] / data['netIncomeToCommon']
      data['pe_ratio'] = pe_ratio

    if data['marketCap'] is not None and book_value is not None:
      pb_ratio = data['marketCap'] / book_value
      data['pb_ratio'] = pb_ratio

    if data['enterpriseValue'] is not None and data['revenue'] is not None:
      ev_revenue = data['enterpriseValue'] / data['revenue']
      data['ev_revenue'] = ev_revenue

    if data['enterpriseValue'] is not None and data['EBITDA'] is not None:
      ev_ebitda = data['enterpriseValue'] / data['EBITDA']
      data['ev_ebitda'] = ev_ebitda

    if data['EBIT'] is not None and data['enterpriseValue'] is not None:
      ev_ebit = data['enterpriseValue'] / data['EBIT']
      data['ev_ebit'] = ev_ebit

  except Exception as e:
    print(f'Error occured when calculaing multiples: {str(e)}')

  return data


async def find_key(possible_key : List[str], indexes: pd.Index) -> Optional[str]:
  # find key with in the list of indexes
  for key in possible_key:
    if key in indexes:
      return str(key)

  # if that fails then we use a llm fallback - TODO later when I implement quantiazation for models
  print(f'Unable to find key, using llm to compare indexes to possible keys: {possible_key}')

  # complete failure
  print('complete failure, key DNE')
  return None

if __name__ == "__main__":
  data = asyncio.run(get_data("MSFT"))
  print(data.items())
