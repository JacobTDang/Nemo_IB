"""Tier 2: web-traffic demand signal via SimilarWeb (Phase G).

Paid data — gated behind the SIMILARWEB_API_KEY env var (same pattern as
CONGRESS_API_KEY). When present this supersedes Google Trends as the demand
proxy for digital/e-commerce/SaaS names, because traffic panels carry real
information that the free proxy only gestures at.

The signal math (`compute_traffic_signal`) is pure and unit-tested. The fetch and
the ticker->domain resolution are dynamic (no hardcoded domains) and require the
key, so they are verified live only when a key is configured.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional


def compute_traffic_signal(
    current_visits: Optional[float],
    prior_visits: Optional[float],
    growth_tol: float = 0.10,
) -> Dict[str, Any]:
    """YoY web-traffic demand signal. Pure.

    > +growth_tol -> bullish, < -growth_tol -> bearish, else neutral.
    Missing/zero history -> data_gap.
    """
    if not current_visits or not prior_visits or prior_visits <= 0:
        return {"signal": "data_gap", "yoy_pct": None,
                "current_visits": current_visits, "prior_visits": prior_visits}
    yoy = (current_visits - prior_visits) / prior_visits
    if yoy > growth_tol:
        signal = "bullish"
    elif yoy < -growth_tol:
        signal = "bearish"
    else:
        signal = "neutral"
    return {"signal": signal, "yoy_pct": round(yoy * 100, 1),
            "current_visits": current_visits, "prior_visits": prior_visits}


def ticker_to_domain(ticker: str) -> Optional[str]:
    """Resolve a ticker to its primary web domain via yfinance (dynamic).
    Returns the bare host (e.g. 'nvidia.com') or None."""
    try:
        import yfinance as yf
        website = (yf.Ticker(ticker).info or {}).get("website") or ""
    except Exception:
        return None
    website = website.strip().lower()
    if not website:
        return None
    for prefix in ("https://", "http://", "www."):
        if website.startswith(prefix):
            website = website[len(prefix):]
    return website.split("/")[0] or None


def _month_str(d: date) -> str:
    return d.strftime("%Y-%m")


def align_yoy_months(months: List[Dict[str, Any]],
                      today: Optional[date] = None) -> Dict[str, Any]:
    """Pick the latest COMPLETE month and its exact same-month-last-year
    comparator by date-string match (never by index). Pure/testable.

    Index-based selection was wrong three ways: SimilarWeb lags 1-2 months
    (11 buckets -> permanent data_gap), an exactly-12-bucket window compared
    adjacent months (off by one), and a partial current month produced a
    massive false-bearish YoY."""
    today = today or date.today()
    cur_month = today.strftime("%Y-%m")

    def _ym(m) -> str:
        return str(m.get("date", ""))[:7]

    complete = [m for m in months
                if _ym(m) and _ym(m) != cur_month and m.get("visits")]
    if not complete:
        return {"current": None, "prior": None}
    current = max(complete, key=_ym)
    cy, cm = int(_ym(current)[:4]), int(_ym(current)[5:7])
    target = f"{cy - 1:04d}-{cm:02d}"
    prior = next((m for m in months if _ym(m) == target), None)
    return {"current": current, "prior": prior}


def fetch_similarweb_traffic(domain: str, api_key: str,
                              timeout: int = 20) -> Dict[str, Any]:
    """Fetch monthly visits for a domain from SimilarWeb. Requires api_key.

    Returns {"months": [...], "current_visits", "prior_visits"} where prior is
    the SAME CALENDAR MONTH one year before the latest complete month — pulled
    over a ~16-month window so the comparator exists even with vendor lag.
    """
    import requests

    today = date.today()
    # First day of the month 16 months back (covers latest-complete + its
    # year-ago month under 1-2 months of vendor lag).
    total_m = today.year * 12 + (today.month - 1) - 16
    sy, sm = divmod(total_m, 12)
    start = date(sy, sm + 1, 1)
    url = (f"https://api.similarweb.com/v1/website/{domain}/"
           f"total-traffic-and-engagement/visits")
    params = {
        "api_key": api_key,
        "start_date": _month_str(start),
        "end_date": _month_str(today),
        "granularity": "monthly",
        "main_domain_only": "true",
    }
    resp = requests.get(url, params=params, timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    data = resp.json()
    visits = data.get("visits", []) or []
    months = [{"date": v.get("date"), "visits": v.get("visits")} for v in visits]

    aligned = align_yoy_months(months, today)
    current, prior = aligned["current"], aligned["prior"]
    return {"domain": domain, "months": months,
            "current_month": current.get("date") if current else None,
            "prior_month": prior.get("date") if prior else None,
            "current_visits": current.get("visits") if current else None,
            "prior_visits": prior.get("visits") if prior else None}


def web_traffic_signal(ticker: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """Top-level: resolve domain (dynamic), fetch, compute the signal.
    Clean error when the key is absent (tier-2 paid data)."""
    api_key = os.environ.get("SIMILARWEB_API_KEY", "")
    if not api_key:
        return {"error": "SIMILARWEB_API_KEY not set — get_web_traffic_signal is "
                         "tier-2 paid data (SimilarWeb). Configure the key to enable.",
                "ticker": ticker, "tier": 2}
    dom = domain or ticker_to_domain(ticker)
    if not dom:
        return {"error": f"could not resolve a web domain for {ticker}",
                "ticker": ticker}
    try:
        raw = fetch_similarweb_traffic(dom, api_key)
    except Exception as exc:
        return {"error": f"SimilarWeb request failed: {type(exc).__name__}: {str(exc)[:160]}",
                "ticker": ticker, "domain": dom}
    sig = compute_traffic_signal(raw.get("current_visits"), raw.get("prior_visits"))
    return {"ticker": ticker, "domain": dom, "source": "similarweb",
            "supersedes": "google_trends", **sig, "months": raw.get("months", [])[-14:]}
