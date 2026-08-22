"""A FinMind code with no rows must not be reported as 'status 200: success'.

FinMind answers an unknown or non-Taiwan-listed code with HTTP 200,
status 200, msg 'success' and an empty data list. The fetcher folded the
no-data case into the bad-status branch, so the caller received
{"error": "FinMind status 200: success"} -- an error string that literally
says success and never mentions the code that produced it.

Run:
  SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_taiwan_revenue_errors.py
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.altdata_server.server import _fetch_taiwan_revenue_finmind


class _Response:
  def __init__(self, payload, status_code=200):
    self._payload = payload
    self.status_code = status_code

  def raise_for_status(self):
    if self.status_code >= 400:
      raise requests.exceptions.HTTPError(f"{self.status_code} error")

  def json(self):
    return self._payload


def test_empty_data_names_the_code_and_does_not_claim_success(monkeypatch):
  monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(
    {"status": 200, "msg": "success", "data": []}))

  out = _fetch_taiwan_revenue_finmind(["9999"], months=3)
  error = out["companies"]["9999"]["error"]

  assert "9999" in error, error
  assert "success" not in error.lower(), error


def test_a_real_finmind_failure_still_reports_status_and_message(monkeypatch):
  monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(
    {"status": 402, "msg": "Requests reach the upper limit.", "data": []}))

  error = _fetch_taiwan_revenue_finmind(["2330"], months=3)["companies"]["2330"]["error"]

  assert "402" in error, error
  assert "upper limit" in error, error


def test_a_good_response_still_parses(monkeypatch):
  rows = [
    {"date": "2025-08-01", "revenue_year": 2025, "revenue_month": 7,
     "revenue": 200_000_000_000},
    {"date": "2026-08-01", "revenue_year": 2026, "revenue_month": 7,
     "revenue": 250_000_000_000},
  ]
  monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(
    {"status": 200, "msg": "success", "data": rows}))

  company = _fetch_taiwan_revenue_finmind(["2330"], months=1)["companies"]["2330"]

  assert "error" not in company
  assert company["months_returned"] == 1
  assert company["months"][0]["revenue_ntd_m"] == 250_000.0
  assert company["months"][0]["yoy_pct"] == 25.0


def test_a_transport_failure_names_its_exception(monkeypatch):
  def _boom(*a, **k):
    raise requests.exceptions.ConnectTimeout("connect timed out")
  monkeypatch.setattr(requests, "get", _boom)

  error = _fetch_taiwan_revenue_finmind(["2330"], months=3)["companies"]["2330"]["error"]
  assert "ConnectTimeout" in error, error
