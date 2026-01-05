
from typing import Any, Dict, Optional
from edgar import Company, set_identity
from edgar.xbrl import XBRL
import pandas as pd
import os


# set identity first
NAME = os.getenv('NAME', 'Investment Analyst')
SEC_EMAIL = os.getenv('SEC_EMAIL', 'analyst@example.com')

def get_latest_filing(ticker: str, form_type: str = '10-K') -> Optional[Dict[str, Any]]:
  # get the latest SEC filing with XBRL data for a company
  try:
    # Set identity globally first
    set_identity(f"{NAME} {SEC_EMAIL}")

    # Create company object and get filings
    company = Company(ticker)
    filings = company.get_filings(form=form_type)

    if filings:
      latest_filing = filings[0]

      # Get XBRL data
      try:
        xbrl_data = latest_filing.xbrl()
      except Exception:
        xbrl_data = None

      # Get filing URL
      url = None
      for attr in ['filing_url', 'url', 'filing_details_url', 'document_url']:
        if hasattr(latest_filing, attr):
          url = getattr(latest_filing, attr)
          break

      return {
        'filing_date': latest_filing.filing_date,
        'url': url,
        'accession_number': latest_filing.accession_number,
        'filing_object': latest_filing,
        'xbrl_data': xbrl_data
      }

    return None

  except Exception:
    return None

def get_disclosures():
  # for later, might be useful information for my agent later
  return None

def get_revenue_base(ticker: str):
  # this is the company's recurring revenue from its primary business operations. It will be the starting point for nearly all financial analysis
  try:
    filing_data = get_latest_filing(ticker)

    if filing_data and filing_data['xbrl_data']:
      xbrl = filing_data['xbrl_data']

      # Try different revenue concept names
      revenue_concepts = ['Revenue', 'Revenues', 'TotalRevenues', 'SalesRevenueNet']

      for concept in revenue_concepts:
        try:
          # Query for this specific concept
          revenue_facts = xbrl.facts.query().by_concept(concept).to_dataframe()

          if not revenue_facts.empty:
            # Filter for annual periods dynamically (350+ days for fiscal year variations)
            revenue_facts['period_start_dt'] = pd.to_datetime(revenue_facts['period_start'])
            revenue_facts['period_end_dt'] = pd.to_datetime(revenue_facts['period_end'])
            revenue_facts['duration_days'] = (revenue_facts['period_end_dt'] - revenue_facts['period_start_dt']).dt.days

            # Get annual periods first, fallback to most recent
            annual_periods = revenue_facts[revenue_facts['duration_days'] >= 350]

            if not annual_periods.empty:
              annual_data = annual_periods[annual_periods['period_end_dt'] == annual_periods['period_end_dt'].max()]
            else:
              annual_data = revenue_facts[revenue_facts['period_end_dt'] == revenue_facts['period_end_dt'].max()]

            if not annual_data.empty:
              latest_row = annual_data.iloc[0]
              revenue_value = latest_row['numeric_value']

              return {
                'ticker': ticker,
                'revenue_base': revenue_value / 1_000_000,  # convert to millions
                'concept_used': concept,
                'period_end': latest_row['period_end'],
                'filing_date': filing_data['filing_date'],
                'success': True
              }
        except Exception:
          continue

      return {
        'ticker': ticker,
        'error': 'No revenue concept found',
        'success': False
      }

    return {
      'ticker': ticker,
      'error': 'No XBRL data available',
      'success': False
    }

  except Exception as e:
    return {
      'ticker': ticker,
      'error': f"Unable to get filing data: {str(e)}",
      'success': False
    }

if __name__ == "__main__":
  print("Testing revenue extraction...")
  result = get_revenue_base("MSFT")
  print(result)
