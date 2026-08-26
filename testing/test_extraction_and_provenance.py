"""Three extraction/provenance defects that all produced a plausible answer.

Every one of these returned a well-formed response that a reader would have
believed:

1. `get_margin_breakdown` took `GeneralAndAdministrativeExpense` when a filer
   splits Sales & Marketing onto its own line, understating MSFT's SG&A by
   $26.7bn and making an AAPL-vs-MSFT comparison say the opposite of the
   truth. Nothing in the response said the number was partial.
2. `extract_customer_concentration` returned the same 10-K sentence twice,
   interleaved three fiscal years in one flat list with no year on any row,
   and called the array `named_customers` while every `name` was null.
3. `get_forward_estimates` said `provider: "Finnhub"` and `errors: []` while
   serving yfinance, and attached an analyst count to an EBITDA figure the
   tool multiplied out itself.

The unit tests below run against synthetic inputs so the assertions are
deterministic; the `network`-marked tests confirm the same rules hold against
live filings and the live APIs.
"""
import os

import pytest

from tools.web_search_server import sec_utils
from tools.news_agregator import finnhub_server

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Real `network` marker plus the offline skip, so `-m network` selects it."""
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live network test")(func)


@pytest.fixture(scope="module", autouse=True)
def _load_env():
  from dotenv import load_dotenv
  load_dotenv()


# =====================================================================
# Defect 1 -- SG&A concept chain and the operating-income reconciliation
# =====================================================================

def _lookup(values):
  """A stand-in for `filter_annual_data` backed by a concept -> value dict."""
  def read(concept):
    if concept not in values:
      return None
    return {'value': float(values[concept]), 'concept_used': concept,
            'period_end': '2026-06-30', 'duration_days': 365, 'currency': 'USD'}
  return read


# Read from data.sec.gov 2026-08-26, CIK 0000789019, FY ending 2026-06-30.
# SellingGeneralAndAdministrativeExpense is not tagged by Microsoft at all.
MSFT_FY2026 = {
  'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax': 331_839_000_000,
  'us-gaap:GrossProfit': 225_465_000_000,
  'us-gaap:GeneralAndAdministrativeExpense': 7_956_000_000,
  'us-gaap:SellingAndMarketingExpense': 26_710_000_000,
  'us-gaap:ResearchAndDevelopmentExpense': 35_562_000_000,
  'us-gaap:OperatingIncomeLoss': 155_237_000_000,
}

# CIK 0000320193, FY ending 2025-09-27. Apple tags the combined element AND
# both halves, and the halves sum to the combined element exactly.
AAPL_FY2025 = {
  'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax': 416_161_000_000,
  'us-gaap:GrossProfit': 195_201_000_000,
  'us-gaap:SellingGeneralAndAdministrativeExpense': 27_601_000_000,
  'us-gaap:GeneralAndAdministrativeExpense': 8_077_000_000,
  'us-gaap:SellingAndMarketingExpense': 19_524_000_000,
  'us-gaap:ResearchAndDevelopmentExpense': 34_550_000_000,
  'us-gaap:OperatingIncomeLoss': 133_050_000_000,
}


def test_split_selling_and_administrative_lines_are_summed():
  """MSFT does not tag SellingGeneralAndAdministrativeExpense. Taking G&A
  alone reported $7.956bn where selling+G&A is $34.666bn -- a 2.4% SG&A ratio
  against a true 10.4%, which inverts an AAPL-vs-MSFT cost comparison."""
  selected = sec_utils._select_sga(_lookup(MSFT_FY2026))
  assert selected is not None
  assert selected['value'] == 34_666_000_000
  assert {c['concept'] for c in selected['components']} == {
    'us-gaap:SellingAndMarketingExpense',
    'us-gaap:GeneralAndAdministrativeExpense',
  }


def test_the_combined_element_wins_over_its_own_components():
  """Apple tags all three. Summing the halves on top of the combined element
  would double-count to $55.2bn, so the combined element must be tried first
  and must stop the chain."""
  selected = sec_utils._select_sga(_lookup(AAPL_FY2025))
  assert selected['value'] == 27_601_000_000
  assert selected['concept_used'] == 'us-gaap:SellingGeneralAndAdministrativeExpense'


def test_a_selling_line_with_no_administrative_line_is_not_sga():
  """JPM tags MarketingAndAdvertisingExpense and no G&A. A bank's advertising
  spend is not its SG&A, and reporting $5.5bn as SG&A for a filer whose
  noninterest expense is ~$100bn would be worse than reporting nothing."""
  selected = sec_utils._select_sga(_lookup({
    'us-gaap:MarketingExpense': 5_531_000_000,
  }))
  assert selected is None


def test_an_administrative_line_with_no_selling_line_is_still_reported():
  """Not every filer has a selling line to split out. G&A alone is the whole
  of SG&A for them, so it must still be returned -- the reconciliation check
  is what distinguishes that case from the MSFT one."""
  selected = sec_utils._select_sga(_lookup({
    'us-gaap:GeneralAndAdministrativeExpense': 3_241_000_000,
  }))
  assert selected['value'] == 3_241_000_000


def test_reconciliation_catches_the_missing_selling_line():
  """The proof the old number was wrong is internal and needed no outside
  source: gross profit - SG&A - R&D must land on operating income, and with
  G&A alone it missed by exactly the omitted $26.71bn of Sales & Marketing.
  Without this check the understatement passes silently."""
  recon = sec_utils._reconcile_operating_income(
    revenue=331_839_000_000, gross_profit=225_465_000_000,
    sga=7_956_000_000, rnd=35_562_000_000,
    operating_income=155_237_000_000)
  assert recon['reconciles'] is False
  assert round(recon['residual']) == 26_710_000_000


def test_reconciliation_passes_once_the_selling_line_is_included():
  """MSFT's income statement has no other operating expense line, so the
  corrected SG&A must reconcile to the dollar. A check that never passes
  would be ignored."""
  recon = sec_utils._reconcile_operating_income(
    revenue=331_839_000_000, gross_profit=225_465_000_000,
    sga=34_666_000_000, rnd=35_562_000_000,
    operating_income=155_237_000_000)
  assert recon['reconciles'] is True
  assert round(recon['residual']) == 0


def test_margin_breakdown_reports_the_summed_sga_and_reconciles(monkeypatch):
  """End to end on the MSFT shape: the response must carry $34.666bn, name
  both concepts it summed, and show the reconciliation so a reader can check
  the arithmetic without leaving the response."""
  monkeypatch.setattr(sec_utils, 'get_latest_filing',
                      lambda *a, **k: {'xbrl_data': object()})
  read = _lookup(MSFT_FY2026)
  monkeypatch.setattr(sec_utils, 'filter_annual_data',
                      lambda xbrl, concept, form_type='10-K': read(concept))

  result = sec_utils.get_margin_breakdown('MSFT')
  assert result['success'] is True
  assert result['sga'] == 34_666_000_000
  assert round(result['sga_pct_revenue'], 1) == 10.4
  assert 'us-gaap:SellingAndMarketingExpense' in result['concepts_used']['sga']
  assert 'us-gaap:GeneralAndAdministrativeExpense' in result['concepts_used']['sga']
  assert result['reconciliation']['reconciles'] is True
  assert result['warnings'] == []


def test_margin_breakdown_warns_when_it_cannot_reconcile(monkeypatch):
  """A filer whose operating income is nowhere near gross profit minus SG&A
  and R&D has an expense line this tool does not see. Saying so is the whole
  point: silence is what let the MSFT gap sit unnoticed."""
  broken = dict(MSFT_FY2026)
  broken.pop('us-gaap:SellingAndMarketingExpense')
  monkeypatch.setattr(sec_utils, 'get_latest_filing',
                      lambda *a, **k: {'xbrl_data': object()})
  read = _lookup(broken)
  monkeypatch.setattr(sec_utils, 'filter_annual_data',
                      lambda xbrl, concept, form_type='10-K': read(concept))

  result = sec_utils.get_margin_breakdown('MSFT')
  codes = [w['code'] for w in result['warnings']]
  assert 'operating_income_does_not_reconcile' in codes
  assert result['reconciliation']['reconciles'] is False


@network
def test_msft_margin_breakdown_reconciles_against_the_filing():
  """The live regression. 225,465 - 34,666 - 35,562 = 155,237, which is the
  operating income MSFT tags."""
  result = sec_utils.get_margin_breakdown('MSFT')
  assert result['success'] is True, result.get('error')
  assert result['sga'] == 34_666_000_000, result.get('concepts_used')
  assert result['reconciliation']['reconciles'] is True, result['reconciliation']


@network
@pytest.mark.parametrize('ticker', ['AAPL', 'NVDA'])
def test_filers_that_tag_the_combined_element_still_reconcile(ticker):
  """AAPL and NVDA reconciled to the dollar before the change and must still
  do so after it -- the fix must not move a number that was already right."""
  result = sec_utils.get_margin_breakdown(ticker)
  assert result['success'] is True, result.get('error')
  assert result['reconciliation']['reconciles'] is True, result['reconciliation']


@network
@pytest.mark.parametrize('ticker', ['GOOGL', 'META', 'ADBE', 'ORCL', 'CRM', 'AMZN'])
def test_filers_that_split_the_line_report_both_halves(ticker):
  """Six more filers that tag no combined element. Each must report SG&A as
  the sum of its two halves rather than the administrative half alone. AMZN is
  in the list because its selling line is tagged `MarketingExpense`, not
  `SellingAndMarketingExpense`."""
  result = sec_utils.get_margin_breakdown(ticker)
  assert result['success'] is True, result.get('error')
  components = result.get('sga_components') or []
  assert len(components) == 2, components
  assert result['sga'] == pytest.approx(sum(c['value'] for c in components))


@network
def test_a_selling_line_alone_is_not_reported_as_sga():
  """JPM tags MarketingAndAdvertisingExpense and no G&A. The live check that
  the widened chain did not start calling a bank's advertising budget SG&A."""
  result = sec_utils.get_margin_breakdown('JPM')
  assert result['success'] is True, result.get('error')
  assert 'sga' not in result, result.get('concepts_used')


@network
def test_adbe_research_and_development_is_found():
  """Adobe tags only ResearchAndDevelopmentExpenseSoftwareExcludingAcquired-
  InProcessCost, so R&D was absent entirely and the margin breakdown implied
  an operating income 4.5bn above the one Adobe reports. The reconciliation
  check is what surfaced it."""
  result = sec_utils.get_margin_breakdown('ADBE')
  assert result['success'] is True, result.get('error')
  assert result['rnd'] == 4_294_000_000, result.get('concepts_used')
  assert result['reconciliation']['reconciles'] is True, result['reconciliation']


# =====================================================================
# Defect 2 -- customer concentration: duplicates, periods, names
# =====================================================================

# NVDA FY2026 10-K. The identical direct-customer sentence appears twice --
# once in the risk-factor summary, once in the risk-factor body with a
# "Direct Customers - " lead-in -- which is why the flat list carried 22% and
# 14% twice each.
NVDA_TEXT = (
  "We generate a significant amount of our revenue from a limited number of "
  "customers, and this trend may continue. For fiscal year 2026, sales to one "
  "direct customer represented 22% of total revenue and sales to another "
  "direct customer represented 14% of total revenue, all of which were "
  "primarily attributable to the Compute & Networking segment.\n\n"
  "Direct Customers - For fiscal year 2026, sales to one direct customer "
  "represented 22% of total revenue and sales to another direct customer "
  "represented 14% of total revenue, all of which were primarily attributable "
  "to the Compute & Networking segment. For fiscal year 2025, sales to one "
  "direct customer represented 12% of total revenue and sales to two direct "
  "customers each represented 11% of total revenue, all of which were "
  "primarily attributable to the Compute & Networking segment. For fiscal "
  "year 2024, sales to one direct customer represented 13% of total revenue, "
  "and were primarily attributable to the Compute & Networking segment.\n\n"
  "Revenue from sales to customers headquartered outside of the United States "
  "accounted for 31% and 41% of total revenue for fiscal years 2026 and 2025, "
  "respectively."
)


def test_the_same_sentence_is_not_counted_twice():
  """The 22% and 14% figures each appeared twice because the same disclosure
  is printed twice in the filing. A reader summing the list got 159% of
  revenue from a company whose largest customer is 22%."""
  found = sec_utils._scan_customer_concentration(NVDA_TEXT)
  rows = found['customer_disclosures']
  keys = [(r['pct_of_revenue'], r['fiscal_year']) for r in rows]
  assert len(keys) == len(set(keys)), keys


def test_each_disclosure_carries_the_fiscal_year_it_describes():
  """Three fiscal years were interleaved in one flat list with no year field,
  so 12% (FY2025) read as a second FY2026 customer. Without the year the rows
  cannot be compared or summed at all."""
  found = sec_utils._scan_customer_concentration(NVDA_TEXT)
  by_year = {}
  for row in found['customer_disclosures']:
    by_year.setdefault(row['fiscal_year'], set()).add(row['pct_of_revenue'])
  assert by_year.get(2026) == {22.0, 14.0}
  assert by_year.get(2025) == {12.0, 11.0}
  assert by_year.get(2024) == {13.0}


def test_periods_are_reported_separately_with_their_totals():
  """A caller asking 'how concentrated is this company now' needs the current
  year's rows, not three years stacked together."""
  found = sec_utils._scan_customer_concentration(NVDA_TEXT)
  periods = {p['fiscal_year']: p for p in found['periods']}
  assert periods[2026]['total_pct'] == 36.0
  assert periods[2026]['disclosure_count'] == 2


def test_revenue_by_geography_is_not_a_customer():
  """'customers headquartered outside of the United States accounted for 31%'
  is a geographic split, not one buyer's share. It was the single largest
  contributor to the impossible 159% total."""
  found = sec_utils._scan_customer_concentration(NVDA_TEXT)
  assert 31.0 not in {r['pct_of_revenue'] for r in found['customer_disclosures']}


def test_shares_summing_over_100_percent_in_one_period_are_flagged():
  """Customer shares of revenue in a single period cannot exceed 100%. When
  they do, the extraction is wrong, and the response must say so instead of
  handing back an arithmetic impossibility as fact."""
  text = ("For fiscal year 2026, sales to customer A represented 60% of total "
          "revenue. For fiscal year 2026, sales to customer B represented 55% "
          "of total revenue.")
  found = sec_utils._scan_customer_concentration(text)
  codes = [w['code'] for w in found['warnings']]
  assert 'concentration_exceeds_total_revenue' in codes


def test_an_unnamed_disclosure_is_not_advertised_as_a_named_customer():
  """Every entry in an array called `named_customers` had `name: null`. The
  canonical key now says what the rows are -- a disclosure of a share of
  revenue, usually anonymous."""
  found = sec_utils._scan_customer_concentration(NVDA_TEXT)
  assert 'customer_disclosures' in found
  assert all(r['name'] is None for r in found['customer_disclosures'])
  # `named_customers` is kept as a back-compatible alias for existing callers.
  assert found['named_customers'] == found['customer_disclosures']


def test_a_real_name_is_still_captured():
  """Where the filer does name the buyer, the name must survive dedup."""
  text = "One customer, Acme Holdings Inc, accounted for 19% of revenue in fiscal 2026."
  found = sec_utils._scan_customer_concentration(text)
  assert [r['name'] for r in found['customer_disclosures']] == ['Acme Holdings Inc']
  assert found['customer_disclosures'][0]['fiscal_year'] == 2026


def test_two_customers_at_the_same_share_in_the_same_year_both_survive():
  """Dedup must key on the disclosure, not on the percentage: two distinct
  sentences each reporting 15% are two customers, not one counted twice."""
  text = ("Customer A accounted for 15% of revenue in fiscal 2026. "
          "Customer B accounted for 15% of revenue in fiscal 2026.")
  found = sec_utils._scan_customer_concentration(text)
  assert len(found['customer_disclosures']) == 2


@network
def test_nvda_live_concentration_has_no_duplicates_and_no_impossible_period():
  """The live regression: 10 rows summing to 159% of revenue."""
  result = sec_utils.extract_customer_concentration('NVDA', '10-K')
  assert result['success'] is True, result.get('error')
  rows = result['customer_disclosures']
  keys = [(r['pct_of_revenue'], r['fiscal_year']) for r in rows]
  assert len(keys) == len(set(keys)), keys
  for period in result['periods']:
    assert period['total_pct'] <= 100, period
  assert {r['fiscal_year'] for r in rows} & {2026}


# =====================================================================
# Defect 3 -- forward estimates provenance
# =====================================================================

def test_the_reason_finnhub_returned_nothing_is_kept():
  """Finnhub answers these three endpoints with HTTP 403 on the free tier.
  Collapsing that to 'no data' hid a fixable entitlement problem behind what
  looked like a company with no analyst coverage."""
  condensed = finnhub_server._condense_forward_estimates(
    {"error": "HTTP 403: {\"error\":\"You don't have access to this resource.\"}"},
    {"error": "HTTP 403: {\"error\":\"You don't have access to this resource.\"}"},
    {"error": "HTTP 403: {\"error\":\"You don't have access to this resource.\"}"},
  )
  assert '403' in condensed['eps']['error']


def test_a_derived_ebitda_carries_no_analyst_count():
  """`ebitda_B` is revenue x the current TTM margin held flat. Reporting
  `analysts: 54` beside it claims 54 analysts published an EBITDA estimate
  when none did -- the figure came out of this process."""
  periods = finnhub_server._infer_ebitda_periods(
    [{"period": "0q", "avg": 92.1766, "low": 90.302, "high": 97.7852,
      "analysts": 44}],
    margin=0.6529)
  assert periods[0]['avg'] == pytest.approx(92.1766 * 0.6529, rel=1e-6)
  assert 'analysts' not in periods[0]
  assert periods[0]['_derived'] is True


def test_provider_names_the_source_that_actually_answered():
  """The response said `provider: "Finnhub"` on every call while serving
  yfinance. A downstream reader crediting Finnhub cannot audit the number."""
  condensed = {
    'eps': {'periods': [], '_source': 'yfinance_fallback'},
    'revenue_B': {'periods': [], '_source': 'yfinance_fallback'},
    'ebitda_B': {'periods': [], '_source': 'yfinance_fallback_inferred'},
  }
  provider, sources = finnhub_server._forward_estimates_provenance(condensed)
  assert 'Finnhub' not in provider
  assert 'yfinance' in provider
  assert sources['eps'] == 'yfinance_fallback'


def test_provider_still_names_finnhub_when_finnhub_answered():
  condensed = {
    'eps': {'periods': []},
    'revenue_B': {'periods': []},
    'ebitda_B': {'periods': []},
  }
  provider, _ = finnhub_server._forward_estimates_provenance(condensed)
  assert provider == 'Finnhub'


async def test_forward_estimates_envelope_tells_the_truth(monkeypatch):
  """All three defects in one response: the provider, the empty error list,
  and the analyst count on a figure nobody published."""
  server = finnhub_server.FinnhubServer.__new__(finnhub_server.FinnhubServer)

  class _Client:
    async def get(self, endpoint, params):
      return {"error": "HTTP 403: {\"error\":\"You don't have access to this resource.\"}"}

  server.client = _Client()

  def _fake_yf(ticker):
    rev = [{"period": "0q", "avg": 92.1766, "low": 90.302, "high": 97.7852,
            "analysts": 44}]
    return {
      "eps": {"periods": [{"period": "0q", "avg": 2.09161, "analysts": 41}],
              "_source": "yfinance_fallback"},
      "revenue_B": {"periods": rev, "_source": "yfinance_fallback"},
      "ebitda_B": {"periods": finnhub_server._infer_ebitda_periods(rev, 0.6529),
                   "_source": "yfinance_fallback_inferred",
                   "_inferred_margin": 0.6529},
    }

  monkeypatch.setattr(finnhub_server, '_yf_forward_estimates', _fake_yf)

  contents = await server.get_forward_estimates('NVDA')
  import json
  envelope = json.loads(contents[0].text)

  assert envelope['provider'] != 'Finnhub'
  assert 'yfinance' in envelope['provider']
  errors = envelope['metadata']['errors']
  assert errors and any('403' in str(e) for e in errors)
  codes = [w['code'] for w in envelope['warnings']]
  assert 'primary_source_unavailable' in codes
  assert 'derived_not_an_analyst_estimate' in codes
  for period in envelope['data']['ebitda_B']['periods']:
    assert 'analysts' not in period


@network
@pytest.mark.parametrize('ticker', ['NVDA', 'AAPL', 'KO'])
async def test_forward_estimates_live_provenance(ticker):
  """Against the live stack, on the three tickers the audit used."""
  import json
  from testing._gates import requires_finnhub  # noqa: F401 - import guard

  server = finnhub_server.FinnhubServer.__new__(finnhub_server.FinnhubServer)
  from tools.news_agregator.finnhub_utils import FinnhubClient
  server.client = FinnhubClient()
  try:
    contents = await server.get_forward_estimates(ticker)
  finally:
    await server.client.close()
  envelope = json.loads(contents[0].text)

  data = envelope['data']
  sources = {k: (data[k].get('_source') if isinstance(data[k], dict) else None)
             for k in ('eps', 'revenue_B', 'ebitda_B')}
  if any(s and 'yfinance' in s for s in sources.values()):
    assert 'yfinance' in envelope['provider'], envelope['provider']
    assert envelope['metadata']['errors'], 'yfinance served but no reason recorded'
  ebitda = data.get('ebitda_B') or {}
  if str(ebitda.get('_source', '')).endswith('_inferred'):
    for period in ebitda.get('periods', []):
      assert 'analysts' not in period, period
