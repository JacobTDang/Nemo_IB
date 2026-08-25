"""Queries over the ingested congressional disclosures.

Two rules run through all of it.

**Coverage travels with the answer.** An empty result from a store holding
four of five hundred filings is not a finding about the ticker; it is a gap in
what has been read. Nothing distinguishes the two except the coverage printed
beside the result, so every response carries it and says plainly when the
answer is not drawn from the complete record.

**Brackets stay brackets.** The disclosures give ranges, so a total is the sum
of the lower bounds and the sum of the upper bounds -- a wider bracket, not a
number. There is no midpoint anywhere in this module. A midpoint would turn
"somewhere between $16,002 and $65,000" into a figure that reads as measured,
and that figure would then travel without the range that qualifies it.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from . import congress_store as store

_TXN_COLUMNS = """t.ticker, t.cusip, t.asset_name, t.asset_type_code, t.owner,
                  t.transaction_type, t.transaction_date, t.notification_date,
                  t.amount_min, t.amount_max, f.chamber, f.filed_date,
                  f.source_url, m.full_name AS member, m.state, m.district"""

_INCOMPLETE = ("This is not the complete record: {unparsed} of {total} filings "
               "in the store could not be parsed or have not been ingested, so "
               "an absent result does not mean it did not happen. Run "
               "`python -m tools.altdata_server.congress_sync --status` to see "
               "the gap.")

_BRACKET_NOTE = ("Amounts are the brackets the filer disclosed. Totals are the "
                 "sum of the lower bounds and the sum of the upper bounds; "
                 "there is no midpoint, because the filings do not contain one.")


def _rows(conn: sqlite3.Connection, sql: str, params: tuple) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _totals(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A bracketed total, plus the direction counts that give it meaning."""
    return {
        "amount_min_total": sum(t.get("amount_min") or 0 for t in transactions),
        "amount_max_total": sum(t.get("amount_max") or 0 for t in transactions),
        "purchase_count": sum(1 for t in transactions
                              if (t.get("transaction_type") or "").startswith("purchase")),
        "sale_count": sum(1 for t in transactions
                          if (t.get("transaction_type") or "").startswith("sale")),
        "exchange_count": sum(1 for t in transactions
                              if (t.get("transaction_type") or "") == "exchange"),
    }


def _note(coverage: Dict[str, Any], extra: str = "") -> str:
    parts = [_BRACKET_NOTE]
    if not coverage.get("complete"):
        parts.append(_INCOMPLETE.format(unparsed=coverage.get("unparsed", 0),
                                        total=coverage.get("total", 0)))
    if extra:
        parts.append(extra)
    return " ".join(parts)


# ------------------------------------------------------------------ by ticker

def ticker_activity(ticker: str, since: Optional[str] = None,
                    until: Optional[str] = None,
                    chamber: Optional[str] = None,
                    limit: int = 500) -> Dict[str, Any]:
    """Every disclosed trade in one ticker, across both chambers."""
    wanted = (ticker or "").upper().strip()
    where = ["t.ticker = ?"]
    params: List[Any] = [wanted]
    if since:
        where.append("t.transaction_date >= ?")
        params.append(since)
    if until:
        where.append("t.transaction_date <= ?")
        params.append(until)
    if chamber:
        where.append("f.chamber = ?")
        params.append(chamber)

    with store.connect() as conn:
        transactions = _rows(conn, f"""
            SELECT {_TXN_COLUMNS} FROM transactions t
            JOIN filings f ON f.filing_id = t.filing_id
            LEFT JOIN members m ON m.member_id = t.member_id
            WHERE {' AND '.join(where)}
            ORDER BY t.transaction_date DESC
            LIMIT ?""", (*params, limit))

    coverage = store.coverage()
    return {
        "success": True,
        "ticker": wanted,
        "query": {"since": since, "until": until, "chamber": chamber},
        "transaction_count": len(transactions),
        "member_count": len({t["member"] for t in transactions if t["member"]}),
        "totals": _totals(transactions),
        "transactions": transactions,
        "coverage": coverage,
        "note": _note(coverage),
    }


# ------------------------------------------------------------------ by member

def member_activity(name: str, since: Optional[str] = None,
                    limit: int = 500) -> Dict[str, Any]:
    """One member's disclosed trades, matched loosely on name.

    The members actually matched are returned alongside the trades. A loose
    match that does not say who it hit produces numbers nobody can attribute,
    and two members share a surname often enough that it matters.
    """
    pattern = f"%{(name or '').strip().lower()}%"
    with store.connect() as conn:
        matched = _rows(conn, """
            SELECT member_id, full_name, chamber, state, district
            FROM members
            WHERE LOWER(full_name) LIKE ? OR LOWER(last) LIKE ?
            ORDER BY full_name""", (pattern, pattern))

        transactions: List[Dict[str, Any]] = []
        if matched:
            ids = [m["member_id"] for m in matched]
            placeholders = ",".join("?" * len(ids))
            params: List[Any] = list(ids)
            date_clause = ""
            if since:
                date_clause = "AND t.transaction_date >= ?"
                params.append(since)
            transactions = _rows(conn, f"""
                SELECT {_TXN_COLUMNS} FROM transactions t
                JOIN filings f ON f.filing_id = t.filing_id
                LEFT JOIN members m ON m.member_id = t.member_id
                WHERE t.member_id IN ({placeholders}) {date_clause}
                ORDER BY t.transaction_date DESC
                LIMIT ?""", (*params, limit))

    by_ticker: Dict[str, int] = {}
    for txn in transactions:
        if txn.get("ticker"):
            by_ticker[txn["ticker"]] = by_ticker.get(txn["ticker"], 0) + 1
    top = sorted(by_ticker.items(), key=lambda kv: -kv[1])[:10]

    coverage = store.coverage()
    extra = "" if matched else (
        f"'{name}' matched no member in the store. Either the name is spelled "
        f"differently in the filings or that member's filings have not been "
        f"ingested.")
    return {
        "success": True,
        "query": {"name": name, "since": since},
        "matched_members": matched,
        "transaction_count": len(transactions),
        "totals": _totals(transactions),
        "most_traded": [{"ticker": t, "transaction_count": c} for t, c in top],
        "transactions": transactions,
        "coverage": coverage,
        "note": _note(coverage, extra),
    }


# --------------------------------------------------------------- leaderboards

def most_traded_tickers(since: Optional[str] = None, chamber: Optional[str] = None,
                        limit: int = 25) -> Dict[str, Any]:
    """Tickers by disclosed trading activity.

    Rows with no ticker are excluded rather than grouped: bonds, treasuries and
    private funds have no symbol, and bucketing them together would invent a
    heavily traded security called nothing.
    """
    where = ["t.ticker IS NOT NULL", "t.ticker != ''"]
    params: List[Any] = []
    if since:
        where.append("t.transaction_date >= ?")
        params.append(since)
    if chamber:
        where.append("f.chamber = ?")
        params.append(chamber)

    with store.connect() as conn:
        tickers = _rows(conn, f"""
            SELECT t.ticker,
                   COUNT(*)                        AS transaction_count,
                   COUNT(DISTINCT t.member_id)     AS member_count,
                   SUM(COALESCE(t.amount_min, 0))  AS amount_min_total,
                   SUM(COALESCE(t.amount_max, 0))  AS amount_max_total,
                   SUM(CASE WHEN t.transaction_type LIKE 'purchase%'
                            THEN 1 ELSE 0 END)     AS purchase_count,
                   SUM(CASE WHEN t.transaction_type LIKE 'sale%'
                            THEN 1 ELSE 0 END)     AS sale_count
            FROM transactions t
            JOIN filings f ON f.filing_id = t.filing_id
            WHERE {' AND '.join(where)}
            GROUP BY t.ticker
            ORDER BY transaction_count DESC, member_count DESC
            LIMIT ?""", (*params, limit))

    coverage = store.coverage()
    return {
        "success": True,
        "query": {"since": since, "chamber": chamber, "limit": limit},
        "tickers": tickers,
        "coverage": coverage,
        "note": _note(coverage),
    }


def most_active_members(since: Optional[str] = None, limit: int = 25
                        ) -> Dict[str, Any]:
    where = ["1=1"]
    params: List[Any] = []
    if since:
        where.append("t.transaction_date >= ?")
        params.append(since)

    with store.connect() as conn:
        members = _rows(conn, f"""
            SELECT m.full_name AS member, m.chamber, m.state, m.district,
                   COUNT(*)                       AS transaction_count,
                   COUNT(DISTINCT t.ticker)       AS distinct_tickers,
                   SUM(COALESCE(t.amount_min, 0)) AS amount_min_total,
                   SUM(COALESCE(t.amount_max, 0)) AS amount_max_total
            FROM transactions t
            JOIN filings f ON f.filing_id = t.filing_id
            JOIN members m ON m.member_id = t.member_id
            WHERE {' AND '.join(where)}
            GROUP BY t.member_id
            ORDER BY transaction_count DESC
            LIMIT ?""", (*params, limit))

    coverage = store.coverage()
    return {"success": True, "query": {"since": since, "limit": limit},
            "members": members, "coverage": coverage, "note": _note(coverage)}


# ---------------------------------------------------------------- holdings

_HOLDING_NOTE = (
    "An annual report discloses assets held at some point DURING the calendar "
    "year it covers, valued in brackets, and is filed months after that year "
    "ends. A row here is not a current position: the member may have exited it "
    "before filing, and any trade disclosed since is not reflected in it. "
    "Values are brackets; totals sum the lower and upper bounds separately and "
    "there is no midpoint. Holdings the filer could not price "
    "(state pensions, family trusts) carry no bounds at all and are counted in "
    "unpriced_count rather than summed as zero."
)

_HOLDING_COLUMNS = """h.ticker, h.cusip, h.asset_name, h.asset_type_code, h.owner,
                      h.value_min, h.value_max, h.income_min, h.income_max,
                      h.income_type, h.as_of, f.chamber, f.filed_date,
                      f.source_url, m.full_name AS member, m.state"""


def _holding_totals(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    priced = [h for h in holdings if h.get("value_min") is not None]
    return {
        "value_min_total": sum(h["value_min"] or 0 for h in priced),
        "value_max_total": sum(h["value_max"] or 0 for h in priced),
        "priced_count": len(priced),
        # Never folded into the totals: a holding nobody could price and a
        # holding worth nothing are different disclosures.
        "unpriced_count": len(holdings) - len(priced),
        "open_ended_count": sum(1 for h in priced if h.get("value_max") is None),
    }


def ticker_holdings(ticker: str, limit: int = 500) -> Dict[str, Any]:
    """Who disclosed holding one ticker, from the annual reports."""
    wanted = (ticker or "").upper().strip()
    with store.connect() as conn:
        holdings = _rows(conn, f"""
            SELECT {_HOLDING_COLUMNS} FROM holdings h
            JOIN filings f ON f.filing_id = h.filing_id
            LEFT JOIN members m ON m.member_id = h.member_id
            WHERE h.ticker = ?
            ORDER BY h.as_of DESC, h.value_min DESC
            LIMIT ?""", (wanted, limit))

    coverage = store.coverage()
    return {"success": True, "ticker": wanted,
            "holding_count": len(holdings),
            "member_count": len({h["member"] for h in holdings if h["member"]}),
            "totals": _holding_totals(holdings), "holdings": holdings,
            "coverage": coverage, "note": _note(coverage, _HOLDING_NOTE)}


def member_holdings(name: str, limit: int = 1000) -> Dict[str, Any]:
    """One member's disclosed holdings, matched loosely on name."""
    pattern = f"%{(name or '').strip().lower()}%"
    with store.connect() as conn:
        matched = _rows(conn, """
            SELECT member_id, full_name, chamber, state, district FROM members
            WHERE LOWER(full_name) LIKE ? OR LOWER(last) LIKE ?
            ORDER BY full_name""", (pattern, pattern))

        holdings: List[Dict[str, Any]] = []
        if matched:
            ids = [m["member_id"] for m in matched]
            placeholders = ",".join("?" * len(ids))
            holdings = _rows(conn, f"""
                SELECT {_HOLDING_COLUMNS} FROM holdings h
                JOIN filings f ON f.filing_id = h.filing_id
                LEFT JOIN members m ON m.member_id = h.member_id
                WHERE h.member_id IN ({placeholders})
                ORDER BY h.as_of DESC, h.value_min DESC
                LIMIT ?""", (*ids, limit))

    coverage = store.coverage()
    extra = _HOLDING_NOTE if matched else (
        f"'{name}' matched no member in the store. {_HOLDING_NOTE}")
    return {"success": True, "query": {"name": name},
            "matched_members": matched, "holding_count": len(holdings),
            "totals": _holding_totals(holdings), "holdings": holdings,
            "coverage": coverage, "note": _note(coverage, extra)}
