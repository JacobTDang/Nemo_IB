from typing import Any, Dict, Optional
from edgar import Company, set_identity
from edgar.xbrl import XBRL
import pandas as pd
import os

# useful documentation for edgartools xbrl: https://edgartools.readthedocs.io/en/latest/getting-xbrl/

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

def get_disclosures_names(ticker) -> Dict[str, Any]:
  # get the disclosure name for agent to use
  try:
    filing_data = get_latest_filing(ticker)
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
    'error': f'Unable to get disclosures for {ticker}',
    'disclosure_names': None
  }

def extract_disclosure_data(ticker: str, disclosure_name: str) -> Dict[str, Any]:

  try:
    latest_filing = get_latest_filing(ticker)
    if latest_filing and latest_filing['xbrl_data']:
      xbrl = latest_filing['xbrl_data']

      try:
        statement = xbrl.statements
        disclosures =  statement.disclosures()

        # Find the specific disclosure by name
        target_disclosure = None
        for disclosure in disclosures:
          if hasattr(disclosure, 'role_or_type'):
            role = disclosure.role_or_type
            # Extract the disclosure name from the URL
            if '/' in role:
              current_name = role.split('/')[-1]
            else:
              current_name = role

            if current_name == disclosure_name:
              target_disclosure = disclosure
              break

        if target_disclosure:
          print(f'Found disclosure: {disclosure_name}')

          # Get summary info about the disclosure
          disclosure_summary = {
            'name': disclosure_name,
            'role_or_type': target_disclosure.role_or_type if hasattr(target_disclosure, 'role_or_type') else None,
            'primary_concept': target_disclosure.primary_concept if hasattr(target_disclosure, 'primary_concept') else None
          }

          # Try to get DataFrame but filter out text-heavy data
          if hasattr(target_disclosure, 'to_dataframe'):
            try:
              df = target_disclosure.to_dataframe()
              print(f'DataFrame shape: {df.shape if df is not None else None}')

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
                  print("This disclosure contains mostly text/HTML data - extracting clean text")
                  disclosure_summary['data_type'] = 'text_heavy'

                  # Extract clean text from HTML
                  import re
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
                        print(f"Extracted clean text length: {len(clean_text)} characters")

                else:
                  # Only include actual data for numerical/structured disclosures
                  disclosure_summary['data_type'] = 'structured'
                  disclosure_summary['sample_data'] = df.head(3).to_dict("records")

              print(f'Disclosure summary: {disclosure_summary}')

            except Exception as e:
              print(f'Error converting to dataframe: {e}')

        else:
          print(f'debug: Unable to find disclosure: {disclosure_name}')
          print(f'Available disclosures: {[d.role_or_type.split("/")[-1] if hasattr(d, "role_or_type") and "/" in d.role_or_type else str(d) for d in disclosures[:5]]}...')


      except Exception as e:
        return {
          'error': f'Unable to get statement'
        }

    else:
      # failed to get the filing data
      return {
        'error': f"Unable to get latest filing for {ticker}"
      }

  except Exception as e:
    return {
      'error': f"Unable to get {disclosure_name} for {ticker}: {str(e)}",
      'success': False
    }


  return {}

def get_revenue_base(ticker: str) -> Dict[str, Any]:
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
           # facts -> pretty much a collection of the revenue data
          # query -> start the query procses
          # by_concept -> searching for revenue using keywords from concept
          # to_dataframes -> organize all of the search results into a dataframe
          revenue_facts = xbrl.facts.query().by_concept(concept).to_dataframe()

          if not revenue_facts.empty:
            # Filter for annual periods dynamically (350+ days for fiscal year variations)
            revenue_facts['period_start_dt'] = pd.to_datetime(revenue_facts['period_start'])
            revenue_facts['period_end_dt'] = pd.to_datetime(revenue_facts['period_end'])
            revenue_facts['duration_days'] = (revenue_facts['period_end_dt'] - revenue_facts['period_start_dt']).dt.days

            # Get annual periods first, fallback to most recent
            # filter out quaterly data to get only annual
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
  print("Testing disclosure extraction...")
  result = extract_disclosure_data("MSFT", 'Role_DisclosureUNEARNEDREVENUE')
  print(result)
