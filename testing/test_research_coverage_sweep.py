"""Tier 2: broad coverage sweep across filer types.

The golden tests pin exact values for a handful of companies. They cannot tell
you whether a tool works for a REIT, a bank, a recent IPO, or a company with no
debt. This sweep answers that, and records the coverage rate per tool so a
partial extractor is known to be partial rather than assumed complete.

The load-bearing assertion is the market-cap reconciliation. Shape checks pass
happily on a share count missing an entire class; price times shares does not.
That is the assertion that would have caught the GOOGL error.

Network-gated and rate-limited. Never runs in the offline suite.
"""
import os
from collections import defaultdict

import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
pytestmark = pytest.mark.skipif(SKIP_NETWORK, reason="live SEC/yfinance sweep")


# Chosen for structural variety rather than familiarity: each group exercises a
# code path the megacaps do not.
BASKET = {
    "megacap":      ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA"],
    "multi_class":  ["GOOGL", "META", "PARA"],
    "bank":         ["JPM", "BAC", "WFC", "GS"],
    "reit":         ["O", "SPG", "PLD", "AMT"],
    "biotech":      ["SAVA", "MRNA", "BIIB"],
    "serial_issuer": ["PLUG", "NKLA", "RIOT"],
    "industrial":   ["CAT", "GE", "BA", "F"],
    "energy":       ["XOM", "CVX", "OXY"],
    "consumer":     ["KO", "PG", "WMT", "COST"],
    "recent_ipo":   ["ARM", "RDDT", "CART"],
}

ALL_TICKERS = sorted({t for group in BASKET.values() for t in group})

# Share count is as of the last filing; market cap is current. A quarter of
# drift is ordinary. A dropped share class is 50%+ off, so this band separates
# staleness from a real extraction bug.
_MARKET_CAP_TOLERANCE = 0.35


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@pytest.fixture(scope="module")
def sweep_results():
    """Run every tool over the basket once and share the results."""
    from tools.web_search_server.dilution import get_share_count_series
    from tools.web_search_server.sbc import get_sbc_series
    from tools.web_search_server.debt_maturity import get_debt_maturity_schedule
    from tools.web_search_server.sec_utils import extract_customer_concentration
    from tools.financial_modeling_engine.corporate_actions import get_corporate_actions

    tools = {
        "share_count": lambda t: get_share_count_series(t, limit=3),
        "sbc": lambda t: get_sbc_series(t, limit=2),
        "debt_maturity": lambda t: get_debt_maturity_schedule(t),
        "customer_concentration": lambda t: extract_customer_concentration(t),
        "corporate_actions": lambda t: get_corporate_actions(t, years=6),
    }

    results = defaultdict(dict)
    errors = defaultdict(list)
    for ticker in ALL_TICKERS:
        for name, run in tools.items():
            try:
                results[name][ticker] = run(ticker)
            except Exception as exc:  # noqa: BLE001 - a raise IS the finding
                errors[name].append((ticker, f"{type(exc).__name__}: {exc}"))
                results[name][ticker] = None
    return {"results": results, "errors": errors}


def _coverage(results_for_tool):
    ok = sum(1 for r in results_for_tool.values()
             if r is not None and r.get("success"))
    return ok, len(results_for_tool)


def test_no_tool_raises_on_any_ticker(sweep_results):
    """A tool must return a not-covered result, never propagate an exception.
    An exception is the one outcome the caller cannot reason about."""
    errors = sweep_results["errors"]
    assert not any(errors.values()), (
        "tools raised instead of returning a result:\n" +
        "\n".join(f"  {tool}: {items}" for tool, items in errors.items() if items))


def test_every_result_has_the_documented_shape(sweep_results):
    required = {
        "share_count": ("success", "latest_total", "by_class", "classes_found"),
        "sbc": ("success", "series"),
        "debt_maturity": ("success", "coverage", "by_year"),
        "customer_concentration": ("success", "has_concentration"),
        "corporate_actions": ("success", "splits", "dividends"),
    }
    for tool, keys in required.items():
        for ticker, result in sweep_results["results"][tool].items():
            if result is None:
                continue
            missing = [k for k in keys if k not in result]
            assert not missing, f"{tool}/{ticker} missing keys {missing}"


def test_share_count_reconciles_with_market_cap(sweep_results):
    """The assertion that catches a dropped share class.

    A shape check passes on a GOOGL total that omits Class B and C. Price times
    shares does not -- it lands 52% away from the real market cap.
    """
    import yfinance as yf

    failures = []
    checked = 0
    for ticker, result in sweep_results["results"]["share_count"].items():
        if not result or not result.get("success") or not result.get("latest_total"):
            continue
        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        except Exception:
            continue
        if not market_cap or not price:
            continue

        implied = result["latest_total"] * price
        drift = abs(implied - market_cap) / market_cap
        checked += 1
        if drift > _MARKET_CAP_TOLERANCE:
            failures.append(
                f"{ticker}: shares={result['latest_total']:,.0f} x ${price:,.2f} "
                f"= ${implied/1e9:,.1f}B vs market cap ${market_cap/1e9:,.1f}B "
                f"({drift:.0%} off, classes={result['classes_found']})")

    assert checked >= 10, f"only {checked} tickers could be reconciled"
    assert not failures, (
        "share count does not reconcile with market cap -- likely a missing "
        "share class:\n" + "\n".join(f"  {f}" for f in failures))


def test_multi_class_filers_report_more_than_one_class(sweep_results):
    """GOOGL, META, and PARA all have multiple share classes. Reporting one
    means the others were silently dropped."""
    single = []
    for ticker in BASKET["multi_class"]:
        result = sweep_results["results"]["share_count"].get(ticker)
        if not result or not result.get("success"):
            continue
        if len(result["classes_found"]) < 2:
            single.append(f"{ticker}: {result['classes_found']}")
    assert not single, (
        "known multi-class filers reported a single class:\n" +
        "\n".join(f"  {s}" for s in single))


def test_coverage_rates_are_recorded(sweep_results, capsys):
    """Not a pass/fail gate on every tool -- a record of what actually works.

    A tool covering 60% of filers is useful. A tool silently covering 60% is
    not, which is why this prints rather than hides the number.
    """
    lines = ["", "coverage by tool:"]
    for tool, results in sweep_results["results"].items():
        ok, total = _coverage(results)
        lines.append(f"  {tool:24s} {ok:3d}/{total:3d}  {ok / total:6.1%}")

    debt = sweep_results["results"]["debt_maturity"]
    buckets = defaultdict(int)
    for result in debt.values():
        if result:
            buckets[result.get("coverage", "error")] += 1
    lines.append(f"  debt_maturity detail:    {dict(buckets)}")

    with capsys.disabled():
        print("\n".join(lines))

    share_ok, share_total = _coverage(sweep_results["results"]["share_count"])
    assert share_ok / share_total > 0.80, (
        f"share count resolved for only {share_ok}/{share_total} filers; it is "
        f"the cover-page tag every filer must report")
