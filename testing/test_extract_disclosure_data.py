"""extract_disclosure_data must say when the disclosure name did not match.

The disclosure names are per-filing taxonomy roles ('Role_DisclosureINVESTMENTS',
'DisclosureCybersecurityRiskManagementStrategyAndGovernance'), so a caller
guessing one -- or reusing a name from a different issuer -- misses routinely.
On a miss the function logged the available names to stderr and fell off the
end to `return {}`, so the MCP client received an empty object: no error, no
success flag, no list of what it should have asked for. An empty result reads
as "this company discloses nothing" rather than "wrong name".

Run:
  SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_extract_disclosure_data.py
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.web_search_server.sec_utils as sec_utils


class _Disclosure:
  def __init__(self, role, frame=None):
    self.role_or_type = role
    self.primary_concept = 'us-gaap_Abstract'
    self._frame = frame

  def to_dataframe(self):
    if self._frame is None:
      raise AssertionError('to_dataframe should not be called on a non-match')
    return self._frame


class _Statements:
  def __init__(self, items):
    self._items = items

  def disclosures(self):
    return self._items


class _Xbrl:
  def __init__(self, items):
    self.statements = _Statements(items)


def _install_filing(monkeypatch, items):
  monkeypatch.setattr(sec_utils, 'get_latest_filing',
                      lambda ticker, form_type='10-K': {'xbrl_data': _Xbrl(items)})


def test_unknown_disclosure_name_is_an_explicit_miss(monkeypatch):
  _install_filing(monkeypatch, [
    _Disclosure('http://www.nvidia.com/2026/role/Role_DisclosureINVESTMENTS'),
    _Disclosure('http://www.nvidia.com/2026/role/Role_DisclosureLEASES'),
  ])
  result = sec_utils.extract_disclosure_data('NVDA', 'Role_DisclosureIncomeTaxes')

  assert result != {}, 'a miss must not come back as an empty object'
  assert result['success'] is False
  assert 'Role_DisclosureIncomeTaxes' in result['error']
  # The caller needs to know what it could have asked for.
  assert 'Role_DisclosureINVESTMENTS' in result['available_disclosure_names']
  assert 'Role_DisclosureLEASES' in result['available_disclosure_names']


def test_matching_disclosure_still_returns_its_data(monkeypatch):
  frame = pd.DataFrame({'concept': ['us-gaap:Leases'], 'label': ['Leases'],
                        '2026-06-30': [1234.0]})
  _install_filing(monkeypatch, [
    _Disclosure('http://www.nvidia.com/2026/role/Role_DisclosureLEASES', frame),
  ])
  result = sec_utils.extract_disclosure_data('NVDA', 'Role_DisclosureLEASES')

  assert result['success'] is True
  assert result['name'] == 'Role_DisclosureLEASES'
  assert result['data_type'] == 'structured'
  assert result['sample_data'][0]['concept'] == 'us-gaap:Leases'


def test_a_broken_statements_call_carries_its_cause(monkeypatch):
  class _Exploding:
    @property
    def statements(self):
      raise RuntimeError('xbrl statements index is corrupt')

  monkeypatch.setattr(sec_utils, 'get_latest_filing',
                      lambda ticker, form_type='10-K': {'xbrl_data': _Exploding()})
  result = sec_utils.extract_disclosure_data('NVDA', 'Role_DisclosureLEASES')

  assert result['success'] is False
  assert 'xbrl statements index is corrupt' in result['error'], result['error']


def test_missing_filing_is_reported(monkeypatch):
  monkeypatch.setattr(sec_utils, 'get_latest_filing',
                      lambda ticker, form_type='10-K': None)
  result = sec_utils.extract_disclosure_data('NVDA', 'Role_DisclosureLEASES')

  assert result['success'] is False
  assert 'NVDA' in result['error']
