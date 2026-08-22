"""A failed alt-data lookup must not read as a genuine zero.

Every tool on nemo_altdata answered `success: true` with an empty payload when
the lookup could not be performed at all -- no ATS reachable, an unknown
ticker, a provider that 404s. A caller could not tell "this company has no open
roles" from "we could not reach any job board", and those describe different
companies.

The shape follows tools/web_search_server/debt_maturity.py: `success: false`
plus a `coverage` of "full" / "partial" / "not_covered" and an `error` naming
what was tried. `degraded` names credential-driven narrowing separately, since
a missing token narrows the sources rather than the answer.

Run:
  SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_altdata_lookup_failures.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.altdata_server import server as alt

SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "0") == "1"


def network(func):
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="network tests skipped")(func)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.url = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


class _RoutedGet:
    """requests.get double that answers by URL substring and records calls."""

    def __init__(self, routes, default=(404, {})):
        self.routes = routes
        self.default = default
        self.urls = []

    def __call__(self, url, *a, **k):
        self.urls.append(url)
        for fragment, (status, payload) in self.routes.items():
            if fragment in url:
                return _Response(payload, status)
        return _Response(self.default[1], self.default[0])


def _envelope(text_contents):
    return json.loads(text_contents[0].text)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# get_job_postings_count
# ---------------------------------------------------------------------------

def _no_providers(monkeypatch):
    monkeypatch.setattr(alt, "_try_greenhouse_norm", lambda *a, **k: None)
    monkeypatch.setattr(alt, "_try_lever_norm", lambda *a, **k: None)
    monkeypatch.setattr(alt, "_try_workday_discovery", lambda *a, **k: None)
    monkeypatch.setattr(alt, "_detect_ats_from_website", lambda *a, **k: None)


def test_jobs_no_provider_reachable_is_a_failure_not_an_empty_count(monkeypatch):
    _no_providers(monkeypatch)

    out = alt._fetch_job_postings("definitely-not-a-real-company-xyz",
                                  "greenhouse", None)

    assert out["success"] is False
    assert out["coverage"] == "not_covered"
    assert "greenhouse" in out["error"].lower()
    assert "lever" in out["error"].lower()
    assert "workday" in out["error"].lower()
    assert out["total_postings"] is None, "a failed lookup has no count"


def test_jobs_handler_reports_the_failure_in_the_envelope(monkeypatch):
    _no_providers(monkeypatch)

    payload = _envelope(_run(alt.AltDataServer().job_postings_count(
        {"company_slug": "definitely-not-a-real-company-xyz"})))

    assert payload["success"] is False
    assert payload["metadata"]["errors"], "no error named for a failed lookup"
    assert payload["data"] is not None, "the diagnostic payload is dropped"


def test_jobs_genuine_zero_is_a_success(monkeypatch):
    """A reachable board with no open roles is a real answer, not a failure."""
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": []})})
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(alt, "_try_workday_discovery", lambda *a, **k: None)
    monkeypatch.setattr(alt, "_detect_ats_from_website", lambda *a, **k: None)

    out = alt._fetch_job_postings("emptyco", "greenhouse", None)

    assert out["success"] is True
    assert out["coverage"] == "full"
    assert out["total_postings"] == 0
    assert out["ats"] == "greenhouse"


def test_jobs_count_is_reported_under_the_name_the_tool_advertises(monkeypatch):
    jobs = [{"id": i, "departments": [{"name": "Engineering"}]} for i in range(7)]
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": jobs})})
    monkeypatch.setattr(requests, "get", get)

    out = alt._fetch_job_postings("someco", "greenhouse", None)

    assert out["total_postings"] == 7
    assert out["source"] == "greenhouse"


def test_jobs_greenhouse_asks_for_the_department_carrying_response(monkeypatch):
    """boards-api /jobs omits departments entirely; only content=true has them.

    Every Stripe posting bucketed as 'Unknown' because the plain listing has no
    `departments` key at all.
    """
    jobs = [{"id": 1, "departments": [{"name": "Engineering"}]}]
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": jobs})})
    monkeypatch.setattr(requests, "get", get)

    alt._try_greenhouse_norm("someco", None)

    assert any("content=true" in u for u in get.urls), get.urls


def test_jobs_missing_departments_are_flagged_not_bucketed_as_unknown(monkeypatch):
    jobs = [{"id": i} for i in range(575)]  # no departments key -- the Stripe case
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": jobs})})
    monkeypatch.setattr(requests, "get", get)

    out = alt._fetch_job_postings("someco", "greenhouse", None)

    assert out["success"] is True
    assert out["total_postings"] == 575
    assert out["by_department"] != {"Unknown": 575}
    assert out["department_coverage"] == "not_covered"
    assert "575" in out["department_coverage_reason"]
    assert out["coverage"] == "partial"


def test_jobs_departments_present_are_bucketed(monkeypatch):
    jobs = ([{"id": i, "departments": [{"name": "Engineering"}]} for i in range(3)]
            + [{"id": 9, "departments": [{"name": "Sales"}]}])
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": jobs})})
    monkeypatch.setattr(requests, "get", get)

    out = alt._fetch_job_postings("someco", "greenhouse", None)

    assert out["by_department"] == {"Engineering": 3, "Sales": 1}
    assert out["department_coverage"] == "full"
    assert out["coverage"] == "full"


def test_jobs_partial_departments_are_reported_as_partial(monkeypatch):
    jobs = [{"id": 1, "departments": [{"name": "Engineering"}]},
            {"id": 2},
            {"id": 3, "departments": []}]
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": jobs})})
    monkeypatch.setattr(requests, "get", get)

    out = alt._fetch_job_postings("someco", "greenhouse", None)

    assert out["department_coverage"] == "partial"
    assert "2 of 3" in out["department_coverage_reason"]


def test_jobs_department_filter_without_departments_fails_loudly(monkeypatch):
    """Filtering on a breakdown the provider never returned silently gave the
    unfiltered total, which reads as 'all 575 roles are in Engineering'."""
    jobs = [{"id": i} for i in range(575)]
    get = _RoutedGet({"boards-api.greenhouse.io": (200, {"jobs": jobs})})
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(alt, "_try_workday_discovery", lambda *a, **k: None)
    monkeypatch.setattr(alt, "_detect_ats_from_website", lambda *a, **k: None)

    out = alt._fetch_job_postings("someco", "greenhouse", "engineering")

    assert out["success"] is False
    assert out["coverage"] == "not_covered"
    assert "engineering" in out["error"].lower()


def test_jobs_lever_departments_come_from_categories(monkeypatch):
    postings = [{"id": "a", "categories": {"department": "Design"}},
                {"id": "b", "categories": {"department": "Design"}}]
    get = _RoutedGet({"api.lever.co": (200, postings)})
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(alt, "_try_greenhouse_norm", lambda *a, **k: None)

    out = alt._fetch_job_postings("someco", "lever", None)

    assert out["ats"] == "lever"
    assert out["by_department"] == {"Design": 2}
    assert out["department_coverage"] == "full"


def test_jobs_workday_fetch_respects_the_provider_limit_cap(monkeypatch):
    """Workday rejects limit > 20 with HTTP 400.

    The full fetch asked for 50, so every Workday board 400'd and the silent
    `except: return None` turned a discovered, reachable tenant into "no
    provider answered".
    """
    seen = {}

    def fake_post(url, json=None, **k):
        seen["limit"] = (json or {}).get("limit")
        if (json or {}).get("limit", 0) > 20:
            return _Response({"errorCode": "HTTP_400"}, 400)
        return _Response({"total": 1527, "facets": []}, 200)

    monkeypatch.setattr(requests, "post", fake_post)

    out = alt._workday_fetch_full("salesforce", 12, "External_Career_Site", None)

    assert seen["limit"] <= 20, f"asked Workday for limit={seen['limit']}"
    assert out is not None, "a reachable Workday tenant returned nothing"
    assert out["total_postings"] == 1527


def test_jobs_workday_without_facets_says_so(monkeypatch):
    """Workday often returns facets with no values. That is a missing
    breakdown, not a company whose every role is 'Unknown'."""
    monkeypatch.setattr(requests, "post", lambda url, json=None, **k: _Response(
        {"total": 1527, "facets": [{"facetParameter": "jobFamilyGroup",
                                    "facetValues": []}]}, 200))

    out = alt._workday_fetch_full("salesforce", 12, "External_Career_Site", None)

    assert out["success"] is True
    assert out["total_postings"] == 1527
    assert out["by_department"] is None
    assert out["department_coverage"] == "not_covered"
    assert out["coverage"] == "partial"


@network
def test_jobs_live_workday_tenant_is_counted():
    """Salesforce is a discoverable Workday tenant with ~1500 open roles."""
    out = alt._fetch_job_postings("salesforce", "workday", None)

    assert out["success"] is True, out.get("error")
    assert out["ats"] == "workday"
    assert out["total_postings"] > 0


@network
def test_jobs_live_stripe_reports_real_departments():
    out = alt._fetch_job_postings("stripe", "greenhouse", None)

    assert out["success"] is True, out.get("error")
    assert out["total_postings"] > 0
    assert out["department_coverage"] == "full", out.get(
        "department_coverage_reason")
    assert out["by_department"], "no department breakdown"
    assert list(out["by_department"]) != ["Unknown"]
    assert out["departments_found"] > 1


@network
def test_jobs_live_unknown_company_is_an_explicit_failure():
    out = alt._fetch_job_postings("definitely-not-a-real-company-xyz",
                                  "greenhouse", None)

    assert out["success"] is False
    assert out["coverage"] == "not_covered"
    assert out["total_postings"] is None
