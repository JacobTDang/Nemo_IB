from typing import Any, Dict, Optional
from edgar import Company, set_identity
from edgar.xbrl import XBRL
import pandas as pd
import os

# useful documentation for edgartools xbrl: https://edgartools.readthedocs.io/en/latest/getting-xbrl/

# set identity first
NAME = os.getenv('NAME', 'Investment Analyst')
SEC_EMAIL = os.getenv('SEC_EMAIL', 'analyst@example.com')

def filter_annual_data(xbrl, concept: str, form_type: str = '10-K') -> Optional[Dict[str, Any]]:
  """
  Helper function to filter XBRL facts for period data based on form type
  Returns latest period data or None if not found
  - 10-K: Annual data (350+ days)
  - 10-Q: Quarterly data (80-95 days)
  - Others: Most recent available
  """
  try:
    facts = xbrl.facts.query().by_concept(concept).to_dataframe()

    if facts.empty:
      return None

    # Filter for appropriate periods based on form type
    facts['period_start_dt'] = pd.to_datetime(facts['period_start'])
    facts['period_end_dt'] = pd.to_datetime(facts['period_end'])
    facts['duration_days'] = (facts['period_end_dt'] - facts['period_start_dt']).dt.days

    # Set period filters based on form type
    if form_type == '10-K':
      # Annual data (350+ days for fiscal year variations)
      target_periods = facts[facts['duration_days'] >= 350]
    elif form_type == '10-Q':
      # Quarterly data (80-95 days typically)
      target_periods = facts[(facts['duration_days'] >= 80) & (facts['duration_days'] <= 95)]
    else:
      # For other forms (8-K, S-1, etc.), get most recent regardless of duration
      target_periods = facts

    if not target_periods.empty:
      period_data = target_periods[target_periods['period_end_dt'] == target_periods['period_end_dt'].max()]
      # For revenue, take the highest value (total revenue vs segment revenue)
      if 'evenue' in concept:
        period_data = period_data.loc[period_data['numeric_value'].idxmax()]
      else:
        period_data = period_data.iloc[0]
    else:
      # Fallback to most recent data if no target periods found
      period_data = facts[facts['period_end_dt'] == facts['period_end_dt'].max()]
      if 'evenue' in concept and len(period_data) > 1:
        period_data = period_data.loc[period_data['numeric_value'].idxmax()]
      else:
        period_data = period_data.iloc[0]

    if period_data is not None and not (hasattr(period_data, 'empty') and period_data.empty):
      latest_row = period_data

      return {
        'value': latest_row['numeric_value'],
        'concept_used': concept,
        'period_end': latest_row['period_end'],
        'duration_days': latest_row['duration_days']
      }

    return None

  except Exception:
    return None

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

def get_disclosures_names(ticker:str, form_type: str = '10-k') -> Dict[str, Any]:
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
    'error': f'Unable to get disclosures for {ticker}',
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

def get_revenue_base(ticker: str, form_type: str= "10-K") -> Dict[str, Any]:
  # this is the company's recurring revenue from its primary business operations. It will be the starting point for nearly all financial analysis
  try:
    filing_data = get_latest_filing(ticker, form_type)

    if filing_data and filing_data['xbrl_data']:
      xbrl = filing_data['xbrl_data']


      # Try different revenue concept names - prioritize Google's specific concepts
      revenue_concepts = [
        'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',  # Google's main revenue concept
        'us-gaap:Revenues',  # Total revenues (most comprehensive)
        'us-gaap:SalesRevenueNet',
        'RevenueFromContractWithCustomerExcludingAssessedTax',  # Without prefix
        'Revenues',
        'Revenue',
        'TotalRevenues',
        'SalesRevenueNet'
      ]

      for concept in revenue_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          return {
            'ticker': ticker,
            'revenue_base': float(result['value']),  # keep in raw dollars
            'concept_used': result['concept_used'],
            'period_end': result['period_end'],
            'filing_date': filing_data['filing_date'],
            'success': True
          }

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

def get_ebitda_margin(ticker: str, form_type: str = '10-k') -> Dict[str, Any]:
  # Ignores interst, taxes, and non-cash expenses, so it allows you to compare the underlying profit generation of a company from there core operations
  # how to ususally calculate it
  # 1. Find the operating income from income statement
  # 2. Find the depreciation & amorization on the cash flow statement
  # 3. Calcualte ebitda = operating income + depreication & amorization, and then divide the sum by revenue
  try:
    ebitda = {} # use to store operating income and depreciation & amorization
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      # get the operating income from income statement
      operating_income_concepts = [
      'us-gaap:OperatingIncomeLoss',        # 95% of companies
      'OperatingIncomeLoss',                # Fallback
      'IncomeLossFromContinuingOperations'  # Edge cases
      ]

      for concept in operating_income_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          ebitda['operating_income'] = result['value']
          ebitda['operating_income_concept_used'] = result['concept_used']
          ebitda['operating_income_period_end'] = result['period_end']
          break


      # get the depreciation & amorization from the cash flow statement
      cashflow_statement_concepts = [
      'us-gaap:DepreciationDepletionAndAmortization',  # First choice - combined total
      'us-gaap:DepreciationAndAmortization',           # Alternative combined
      'DepreciationDepletionAndAmortization'           # Fallback without prefix
      ]


      for concept in cashflow_statement_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          ebitda['d&a'] = result['value']
          ebitda['d&a_concept_used'] = result['concept_used']
          ebitda['d&a_period_end'] = result['period_end']
          break

      if 'd&a' not in ebitda or not ebitda['d&a']:
        # Try to get individual components and add them together
        individual_concepts = [
          'us-gaap:Depreciation',
          'us-gaap:AmortizationOfIntangibleAssets'
        ]

        depreciation_value = 0
        amortization_value = 0
        concepts_used = []
        period_end = None

        for concept in individual_concepts:
          result = filter_annual_data(xbrl, concept, form_type)
          if result:
            if 'Depreciation' in concept:
              depreciation_value = result['value']
            elif 'Amortization' in concept:
              amortization_value = result['value']
            concepts_used.append(result['concept_used'])
            if not period_end:
              period_end = result['period_end']

        # Only set D&A if we found at least one component
        if depreciation_value > 0 or amortization_value > 0:
          ebitda['d&a'] = depreciation_value + amortization_value
          ebitda['d&a_concept_used'] = ' + '.join(concepts_used)
          ebitda['d&a_period_end'] = period_end

      # Get revenue for margin calculation
      revenue_data = get_revenue_base(ticker, form_type)
      if not revenue_data['success']:
        return revenue_data  # Return the error from revenue function

      revenue = revenue_data['revenue_base']  # already in raw dollars

      # Calculate EBITDA and EBITDA margin
      ebitda_amount = ebitda['operating_income'] + ebitda['d&a']
      ebitda_margin_percent = (ebitda_amount / revenue) * 100

      return {
        'error': None,
        'success': True,
        'ticker': ticker,
        'ebitda_margin_percent': float(ebitda_margin_percent),
        'ebitda_amount': float(ebitda_amount),
        'operating_income': float(ebitda['operating_income']),
        'd&a': float(ebitda['d&a']),
        'revenue': float(revenue),
        'operating_income_concept_used': ebitda.get('operating_income_concept_used'),
        'd&a_concept_used': ebitda.get('d&a_concept_used'),
        'period_end': ebitda.get('operating_income_period_end')
      }
    else:
      return {
        'error': f'Unable to get latest filing for {ticker}',
        'success': False
      }


  except Exception as e:
    return {
      'error': f'Failed to get latest file for {ticker}',
      'success': False
    }

def get_capex_pct_revenue(ticker: str, form_type: str = '10-k') -> Dict[str, Any]:
  # function to get capital expenditures: Capex is the money that the company spends to buy, maintain, or upgrade physical assets
  # this metric will show CapEX as percentage of revenue, it shows how much a company is reinvesting back into its assets
  # CapEx % of revenue = capital expedeitures / total revenue
  # can find it on cash flow statement under 'cash flow from investing activities'
  try:
    filing = get_latest_filing(ticker, form_type)

    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      primary_capex_concepts = [
      'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',  # Total PP&E (most common)
      'us-gaap:PaymentsForCapitalExpenditures',              # Direct total CapEx
      'PaymentsToAcquirePropertyPlantAndEquipment',          # Fallback
      'CapitalExpenditures'                                  # Basic total
      ]
      total_capex = 0

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
        print(f'WARNING for {ticker}: Might not account for all capital expenditures. Possible overlap of capital expenditures. Using concepts: {component_concepts}')
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
        'error': f"Unable to get xbrl data for: {ticker}",
        'success': False
      }

  except:
    return{
      'error': f'Unable to get filing for {ticker}',
      'success': False
    }


def get_tax_rate(ticker: str, form_type: str = '10-k') -> Dict[str, Any]:
  # returns the effective/actual tax rate that the company pays on its profits
  # can find it on the income statement in 'income before provision for income taxes or similar wording' and 'provision for income taxes
  # formula: Effective tax rate = provision for income taxes / earnings before taxes
  try:
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      tax_expense_concepts = [
      'us-gaap:IncomeTaxExpenseBenefit',                           # Most common - total tax expense
      'us-gaap:ProvisionForIncomeTaxes',                           # Alternative provision concept
      'us-gaap:IncomeTaxesPaid',                                   # Cash taxes paid
      'us-gaap:CurrentIncomeTaxExpense',                           # Current year tax expense
      'IncomeTaxExpenseBenefit',                                   # Without prefix
      'ProvisionForIncomeTaxes',                                   # Without prefix
      'IncomeTaxExpense'                                           # Basic form
      ]
      tax_expense = 0.0 # bc panda dataframe return np.float64
      for concept in tax_expense_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          tax_expense = float(result['value'])
          tax_concept_used = concept
          break

      pretax_income_concepts = [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',  # Full concept
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes',                                          # Most common
        'us-gaap:EarningsBeforeIncomeTaxes',                                                                    # Alternative
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',                                                  # Without prefix
        'EarningsBeforeIncomeTaxes',                                                                            # Without prefix
        'IncomeBeforeTaxes'                                                                                     # Basic form
      ]

      pretax_income = 0.0
      for concept in pretax_income_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          pretax_income = float(result['value'])
          pretax_concept_used = concept
          break

      # calulate the effective tax rate: Effective tax rate = provision for income taxes / earnings before taxes
      if tax_expense != 0: # prevent divide by 0 error
        effective_tax_rate = (tax_expense / pretax_income) * 100
      else:
        return{
          'error': f"tax expense is 0, unable to divide by 0",
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
        'error': f'Unable to get xbrl data',
        'success': False
      }
  except Exception as e:
    return{
      'error': f'Unable to get filing for {ticker}: {str(e)}',
      'success': False
    }


def get_depreciation(ticker: str, form_type: str = '10-k') -> Dict[str, Any]:
  # this is the accounting method of allocating the cost of a physical asset over its uselife. It is a non cash expense is will be expressed as a percentage of revenue
  # formula: depreication % of revenue = depreication & amorization / total revenue
  # this will be helpful beacuse it helps us find the age and cost structure of a company's assets
  # find it on the cash flow statement, usually under "cash flow from operating activities"
  try:
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      depreciation_concepts = [
      'us-gaap:DepreciationDepletionAndAmortization',           # Combined D&A (most common)
      'us-gaap:Depreciation',                                   # Depreciation only
      'us-gaap:DepreciationAndAmortization',                    # Alternative combined
      'us-gaap:DepreciationAmortizationAndAccretionNet',        # With accretion
      'us-gaap:DepreciationDepletionAndAmortizationExcludingAmortizationOfDebtIssuanceCosts',  # Excluding debt costs
      'DepreciationDepletionAndAmortization',                   # Without prefix
      'Depreciation',                                           # Basic depreciation
      'DepreciationAndAmortization'                            # Basic combined
      ]

      d_a_value = 0.0
      for concept in depreciation_concepts:
        results = filter_annual_data(xbrl, concept, form_type)
        if results:
          d_a_value = float(results['value'])
          d_a_concept = concept
          break

      if d_a_value == 0.0:
        return{
          'error': f"Unable to find concept for {ticker}: Concepts used = {depreciation_concepts}",
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
        'error': f"Unable to get filing for {ticker}",
        'success': False
      }

  except Exception as e:
    return{
      'error': f"Unable to get filing for {ticker}",
      'success':False
    }
if __name__ == "__main__":
  pass
