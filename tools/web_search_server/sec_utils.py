from typing import Any, Dict, Optional
from edgar import Company, set_identity
from edgar.xbrl import XBRL
import pandas as pd
import os
import re
import sys
# useful documentation for edgartools xbrl: https://edgartools.readthedocs.io/en/latest/getting-xbrl/

# set identity first
NAME = os.getenv('NAME', 'Investment Analyst')
SEC_EMAIL = os.getenv('SEC_EMAIL', 'analyst@example.com')

# Module-level cache: prevents the 4 SEC tools (ebitda, capex, tax, depreciation) from each
# downloading and parsing the same 10-K XBRL independently. Within one MCP server process
# lifetime the filing content doesn't change, so caching is safe.
_filing_cache: Dict[tuple, Any] = {}


def filter_instant_data(xbrl, concept: str) -> Optional[Dict[str, Any]]:
  """Filter XBRL for instant (point-in-time) facts -- balance sheet items.

  Balance sheet items use 'period_instant' rather than 'period_start'/'period_end'.
  Returns the most recent fact and its as-of date.
  """
  try:
    facts = xbrl.facts.query().by_concept(concept).to_dataframe()
    if facts.empty:
      return None
    # Filter to instant-type rows (balance sheet); period_instant has the as-of date
    if 'period_type' in facts.columns:
      facts = facts[facts['period_type'] == 'instant']
    if facts.empty:
      return None
    facts['instant_dt'] = pd.to_datetime(facts['period_instant'])
    latest = facts[facts['instant_dt'] == facts['instant_dt'].max()]
    if len(latest) > 1:
      # Consolidated total = largest absolute value among segments
      latest = latest.loc[latest['numeric_value'].abs().idxmax()]
    else:
      latest = latest.iloc[0]
    return {
      'value': latest['numeric_value'],
      'concept_used': concept,
      'period_end': latest['period_instant'],
    }
  except Exception:
    return None


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
      # XBRL often has multiple facts for the same concept and period
      # (segments + consolidated total). The consolidated total is always
      # the largest positive value, so take the max for any concept that
      # can have segment breakdowns.
      if len(period_data) > 1:
        period_data = period_data.loc[period_data['numeric_value'].idxmax()]
      else:
        period_data = period_data.iloc[0]
    else:
      # Fallback to most recent data if no target periods found
      period_data = facts[facts['period_end_dt'] == facts['period_end_dt'].max()]
      if len(period_data) > 1:
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
  """Get latest SEC filing with XBRL data.

  Results are cached in-process by (ticker, form_type) key.
  When multiple SEC tools run in the same parallel batch (e.g. get_ebitda_margin,
  get_capex_pct_revenue, get_tax_rate, get_depreciation all called concurrently),
  the cache prevents 4 duplicate 10-K downloads that would trigger SEC rate limiting.
  """
  cache_key = (ticker.upper(), form_type)
  if cache_key in _filing_cache:
    return _filing_cache[cache_key]

  result = None
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

      result = {
        'filing_date': latest_filing.filing_date,
        'url': url,
        'accession_number': latest_filing.accession_number,
        'filing_object': latest_filing,
        'xbrl_data': xbrl_data
      }

  except Exception:
    result = None

  _filing_cache[cache_key] = result
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

        else:
          print(f'debug: Unable to find disclosure: {disclosure_name}', file=sys.stderr, flush=True)
          print(f'Available disclosures: {[d.role_or_type.split("/")[-1] if hasattr(d, "role_or_type") and "/" in d.role_or_type else str(d) for d in disclosures[:5]]}...', file=sys.stderr, flush=True)


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

def get_ebitda_margin(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
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
      if 'operating_income' not in ebitda or 'd&a' not in ebitda:
        missing = []
        if 'operating_income' not in ebitda: missing.append('operating_income')
        if 'd&a' not in ebitda: missing.append('d&a')
        return {
          'error': f"Missing EBITDA components for {ticker}: {', '.join(missing)}",
          'success': False
        }
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

def get_capex_pct_revenue(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
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
        'error': f"Unable to get xbrl data for: {ticker}",
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
      tax_concept_used = None
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
      pretax_concept_used = None
      for concept in pretax_income_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          pretax_income = float(result['value'])
          pretax_concept_used = concept
          break

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
        'error': f'Unable to get xbrl data',
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
      d_a_concept = None
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


def _get_revenue_from_xbrl(xbrl, form_type: str):
  """Helper: try the two most common revenue concepts; return (value, period_end) or None."""
  for concept in ('us-gaap:Revenues',
                  'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
                  'us-gaap:SalesRevenueNet'):
    d = filter_annual_data(xbrl, concept, form_type)
    if d:
      return d['value'], d['period_end']
  return None


def get_margin_breakdown(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract gross margin, SG&A %, R&D % from the latest filing.

  Returns ticker, revenue, gross_profit, sga, rnd, *_pct_revenue values, plus
  concepts_used for traceability. Banks (no COGS) typically have no GrossProfit
  XBRL concept; absence is expected, not an error.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': f'No filing found for {ticker}', 'success': False}

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
        concepts_used['gross_profit'] = c
        break

    for c in ('us-gaap:SellingGeneralAndAdministrativeExpense',
              'us-gaap:GeneralAndAdministrativeExpense'):
      sga = filter_annual_data(xbrl, c, form_type)
      if sga:
        result['sga'] = sga['value']
        result['sga_pct_revenue'] = (sga['value'] / revenue) * 100
        concepts_used['sga'] = c
        break

    for c in ('us-gaap:ResearchAndDevelopmentExpense',):
      rnd = filter_annual_data(xbrl, c, form_type)
      if rnd:
        result['rnd'] = rnd['value']
        result['rnd_pct_revenue'] = (rnd['value'] / revenue) * 100
        concepts_used['rnd'] = c
        break

    result['concepts_used'] = concepts_used

    if 'gross_profit' not in result:
      print(f"[Validate SEC] {ticker}: gross_profit XBRL concept not found (expected for banks/financials)",
            file=sys.stderr, flush=True)
    return result

  except Exception as e:
    return {'ticker': ticker, 'error': f'get_margin_breakdown failed: {e}', 'success': False}


def get_historical_fcf(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract operating cash flow, capex, and compute FCF and FCF margin from latest filing."""
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': f'No filing found for {ticker}', 'success': False}

    xbrl = filing_data['xbrl_data']
    ocf = None
    for c in ('us-gaap:NetCashProvidedByUsedInOperatingActivities',
              'us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'):
      d = filter_annual_data(xbrl, c, form_type)
      if d:
        ocf = d['value']
        break

    capex = None
    for c in ('us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
              'us-gaap:PaymentsForCapitalImprovements'):
      d = filter_annual_data(xbrl, c, form_type)
      if d:
        capex = abs(d['value'])  # capex usually reported negative on CF statement
        break

    if ocf is None:
      return {'ticker': ticker, 'error': 'OCF concept not found', 'success': False}

    fcf = ocf - (capex or 0)
    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)
    revenue = rev_tuple[0] if rev_tuple else None
    period_end = rev_tuple[1] if rev_tuple else None

    return {
      'ticker': ticker,
      'operating_cash_flow': ocf,
      'capex': capex,
      'free_cash_flow': fcf,
      'fcf_margin_pct': (fcf / revenue * 100) if revenue else None,
      'period_end': period_end,
      'success': True
    }
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
      return {'error': 'No filing', 'success': False}

    xbrl = filing_data['xbrl_data']
    ca = filter_instant_data(xbrl, 'us-gaap:AssetsCurrent')
    cl = filter_instant_data(xbrl, 'us-gaap:LiabilitiesCurrent')
    ar = filter_instant_data(xbrl, 'us-gaap:AccountsReceivableNetCurrent')
    inv = filter_instant_data(xbrl, 'us-gaap:InventoryNet')
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
  try:
    set_identity(f"{NAME} {SEC_EMAIL}")
    company = Company(ticker)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'Company lookup failed: {type(e).__name__}: {e}'}

  forms_to_pull = ['SC 13D', 'SC 13D/A']
  if include_passive:
    forms_to_pull.extend(['SC 13G', 'SC 13G/A'])

  rows: list = []
  for form in forms_to_pull:
    try:
      filings = company.get_filings(form=form).head(limit)
    except Exception:
      continue
    for f in filings:
      filer_name = None
      filer_cik = None
      try:
        if f.header and f.header.filers:
          ci = f.header.filers[0].company_information
          filer_name = ci.name
          filer_cik = ci.cik
      except Exception:
        pass

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
  rows = rows[:limit]

  activist_count = sum(1 for r in rows if r['is_activist'])
  passive_count = sum(1 for r in rows if not r['is_activist'])

  return {
    'ticker':         ticker,
    'success':        True,
    'error':          None,
    'filings':        rows,
    'count':          len(rows),
    'activist_count': activist_count,
    'passive_count':  passive_count,
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
  try:
    set_identity(f"{NAME} {SEC_EMAIL}")
    company = Company(ticker)
    filings = list(company.get_filings(form='10-K').head(10))
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'company/filings lookup failed: {type(e).__name__}: {e}'}

  if len(filings) < 2:
    return {'ticker': ticker, 'success': False,
            'error': f'Need 2+ 10-K filings to diff, found {len(filings)}'}

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
      return {'ticker': ticker, 'success': False, 'error': 'no filing'}
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
  try:
    set_identity(f"{NAME} {SEC_EMAIL}")
    company = Company(ticker)
    filings = company.get_filings(form=form_type)
  except Exception as e:
    return {'ticker': ticker, 'success': False,
            'error': f'company/filings fetch failed: {type(e).__name__}: {e}'}

  if not filings:
    return {'ticker': ticker, 'success': False,
            'error': f'No {form_type} filings found for {ticker}'}

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

  def _query(qs: str) -> Optional[Dict[str, Any]]:
    try:
      r = _req.get(url, params={'url': qs, 'exp': ''},
                   headers=headers, timeout=20)
      if r.status_code != 200:
        return None
      return r.json()
    except Exception:
      return None

  # Total count for assignee
  base_q = f'assignee={company_name}'
  total_payload = _query(base_q + '&num=10')
  if total_payload is None:
    return {'company_name': company_name, 'success': False,
            'error': 'Google Patents query failed (network or 4xx response)'}

  total_results = total_payload.get('results', {}).get('total_num_results', 0)

  # Year-by-year breakdown (last N years of grant dates)
  this_year = _dt.now().year
  year_counts = []
  for y in range(this_year - years_back, this_year + 1):
    # Patents granted within calendar year y
    yq = f'assignee={company_name}&after=publication:{y}0101&before=publication:{y}1231&num=1'
    pl = _query(yq)
    if pl:
      year_counts.append({
        'year': y,
        'count': pl.get('results', {}).get('total_num_results', 0),
      })

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
  try:
    set_identity(f"{NAME} {SEC_EMAIL}")
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
              'error': f'No {form_type} filing found for {ticker}'}

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
              'error': f'No {form_type} filing found for {ticker}'}

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


def get_segment_financials(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract per-segment revenue and operating income from latest 10-K XBRL.

  Uses the `us-gaap:StatementBusinessSegmentsAxis` (or any axis whose name
  contains 'Segment') and pulls the company-defined segment members. For
  each segment, fetches annual revenue (RevenueFromContractWithCustomer
  ExcludingAssessedTax / Revenues / SalesRevenueNet) and operating income
  (OperatingIncomeLoss) by joining the concept to the segment dimension.

  Returns up to 5 years of history per segment plus the most recent YoY
  growth and operating margin. Critical for resolving the variant-
  perception question on multi-segment companies — e.g. MSFT's Intelligent
  Cloud (Azure) growth vs. Productivity & Business Processes margin.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'ticker': ticker, 'success': False,
              'error': f'No {form_type} filing or XBRL data found for {ticker}'}

    xbrl = filing_data['xbrl_data']

    # Discover the segment axis. edgartools normalizes ':' to '_' in the
    # unique_dimensions dict keys, but by_dimension() accepts the colon form.
    unique_dims = xbrl.facts.get_unique_dimensions()
    segment_axis_key = None
    segment_axis_for_query = None
    for key in unique_dims.keys():
      if 'StatementBusinessSegments' in key or key.endswith('SegmentsAxis'):
        segment_axis_key = key
        # Normalize to colon form for by_dimension query
        segment_axis_for_query = key.replace('_', ':', 1)
        break

    if not segment_axis_key:
      return {'ticker': ticker, 'success': False,
              'error': 'No business-segment axis in XBRL — company may not have reportable segments',
              'axes_available': list(unique_dims.keys())[:10]}

    members = unique_dims.get(segment_axis_key, set())
    if not members:
      return {'ticker': ticker, 'success': False,
              'error': f'No segment members under {segment_axis_key}'}

    revenue_concepts = [
      'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
      'us-gaap:Revenues',
      'us-gaap:SalesRevenueNet',
      'us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax',
    ]
    op_income_concept = 'us-gaap:OperatingIncomeLoss'

    def _annual_series(concept: str, member: str) -> tuple:
      """Return (concept_used, [{period_end, value}, ...]) sorted newest-first."""
      try:
        q = xbrl.facts.query().by_concept(concept).by_dimension(
          segment_axis_for_query, member)
        df = q.to_dataframe()
      except Exception:
        return (None, [])
      if df.empty:
        return (None, [])
      df = df.copy()
      df['period_start_dt'] = pd.to_datetime(df['period_start'])
      df['period_end_dt'] = pd.to_datetime(df['period_end'])
      df['duration_days'] = (df['period_end_dt'] - df['period_start_dt']).dt.days
      annual = df[df['duration_days'] >= 350].sort_values('period_end_dt', ascending=False)
      series = [{'period_end': r['period_end'], 'value': float(r['numeric_value'])}
                for _, r in annual.iterrows()]
      return (concept, series)

    segments_out = []
    revenue_concept_used = None
    for member in sorted(members):
      # Pretty name: "msft:ProductivityAndBusinessProcessesMember"
      # -> "Productivity And Business Processes"
      seg_short = member.split(':')[-1]
      if seg_short.endswith('Member'):
        seg_short = seg_short[:-len('Member')]
      seg_display = re.sub(r'([a-z])([A-Z])', r'\1 \2', seg_short).strip()

      # Revenue (try each concept in priority order)
      rev_series: list = []
      for c in revenue_concepts:
        used, series = _annual_series(c, member)
        if series:
          rev_series = series
          revenue_concept_used = revenue_concept_used or used
          break

      # Operating income
      _, op_series = _annual_series(op_income_concept, member)

      # Derived metrics on the latest period
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
      })

    # Total check: sum latest-period revenues vs. consolidated revenue base
    total_seg_rev = sum(s['revenue'][0]['value'] for s in segments_out if s['revenue'])

    return {
      'ticker': ticker,
      'success': True,
      'error': None,
      'segments': segments_out,
      'segment_axis': segment_axis_for_query,
      'segment_count': len(segments_out),
      'total_latest_segment_revenue': total_seg_rev,
      'revenue_concept_used': revenue_concept_used,
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
      return {'ticker': ticker, 'error': f'No filing found for {ticker}',
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
      return {
        'ticker':            ticker,
        'success':           False,
        'lookback_quarters': lookback_quarters,
        'sources_scanned':   [],
        'signal_count':      0,
        'signals':           [],
        'by_category':       {},
        'error':             'No text sources available (earnings releases + MD&A both failed)',
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
