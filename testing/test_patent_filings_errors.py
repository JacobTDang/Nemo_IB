"""get_patent_filings must say why a Google Patents query failed.

Google throttles the public /xhr/query endpoint with a 503 "Sorry..." page once
a host makes a handful of requests. The helper collapsed that 503, a 404, a
timeout and a DNS failure into a bare None, and the tool reported all four as
"Google Patents query failed (network or 4xx response)" -- indistinguishable
from a misspelled assignee, and undiagnosable without reproducing the raw HTTP
call by hand.

The per-year queries had the same problem in a worse place: a throttled year
was skipped silently, so a truncated year_counts series still came back with
success True. The year-over-year trend is the whole point of the tool and a
gap in it must be visible.

Run:
  SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_patent_filings_errors.py
"""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.web_search_server.sec_utils import get_patent_filings


class _Response:
  def __init__(self, status_code, payload=None):
    self.status_code = status_code
    self._payload = payload

  def json(self):
    if self._payload is None:
      raise ValueError("Expecting value: line 1 column 1 (char 0)")
    return self._payload


def _payload(total):
  return {"results": {"total_num_results": total, "cluster": []}}


def test_http_status_is_named_in_the_error(monkeypatch):
  monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(503))
  result = get_patent_filings("Microsoft", years_back=1, sample_count=1)
  assert result["success"] is False
  assert "503" in result["error"], result["error"]


def test_transport_exception_is_named_in_the_error(monkeypatch):
  def _boom(*a, **k):
    raise requests.exceptions.ConnectTimeout("connect timed out")
  monkeypatch.setattr(requests, "get", _boom)
  result = get_patent_filings("Microsoft", years_back=1, sample_count=1)
  assert result["success"] is False
  assert "ConnectTimeout" in result["error"], result["error"]


def test_a_throttled_year_is_reported_not_silently_dropped(monkeypatch):
  """First call (the assignee total) succeeds, the year queries get 503."""
  calls = {"n": 0}

  def _get(*a, **k):
    calls["n"] += 1
    if calls["n"] == 1:
      return _Response(200, _payload(125048))
    return _Response(503)

  monkeypatch.setattr(requests, "get", _get)
  result = get_patent_filings("Microsoft", years_back=2, sample_count=1)
  assert result["success"] is True
  assert result["year_counts"] == []
  failed = result["failed_years"]
  assert len(failed) == 3, failed          # years_back=2 spans 3 calendar years
  assert all("503" in f["reason"] for f in failed), failed


def test_a_clean_run_reports_no_failed_years(monkeypatch):
  monkeypatch.setattr(requests, "get",
                      lambda *a, **k: _Response(200, _payload(4242)))
  result = get_patent_filings("Microsoft", years_back=2, sample_count=1)
  assert result["success"] is True
  assert result["error"] is None
  assert result["total_patents"] == 4242
  assert len(result["year_counts"]) == 3
  assert result["failed_years"] == []


def test_non_json_body_is_reported_as_such(monkeypatch):
  monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(200, None))
  result = get_patent_filings("Microsoft", years_back=1, sample_count=1)
  assert result["success"] is False
  assert "json" in result["error"].lower(), result["error"]


def test_empty_company_name_still_refuses():
  result = get_patent_filings("", years_back=1, sample_count=1)
  assert result["success"] is False
  assert "company_name" in result["error"]
