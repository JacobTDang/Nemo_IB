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


def _open_ended(rows: List[Dict[str, Any]], low: str, high: str) -> bool:
    """Whether any row has a floor but no disclosed ceiling.

    "Over $50,000,000" and the EIGA spouse cap have no upper bound. Summing
    them as though the ceiling were zero produced a maximum below its own
    minimum -- live, three NVDA holdings totalled 1,600,002 to 1,250,000.
    Once such a row is in the sum the total has no ceiling either.
    """
    return any(r.get(low) is not None and r.get(high) is None for r in rows)


def _totals(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A bracketed total, plus the direction counts that give it meaning."""
    unbounded = _open_ended(transactions, "amount_min", "amount_max")
    return {
        "amount_min_total": sum(t.get("amount_min") or 0 for t in transactions),
        "amount_max_total": None if unbounded else sum(
            t.get("amount_max") or 0 for t in transactions),
        "open_ended_count": sum(1 for t in transactions
                                if t.get("amount_min") is not None
                                and t.get("amount_max") is None),
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


_TRUNCATED_NOTE = (
    "The row list is truncated: this response hit its limit, so the rows "
    "shown are a subset of those that matched. "
    "The counts and totals are computed over every matching row, not over the "
    "subset -- raise `limit` to see more rows, not to change the numbers."
)


def _transaction_aggregate(conn: sqlite3.Connection, where: str,
                           params: tuple) -> Dict[str, Any]:
    """Counts and bracketed totals over EVERY matching trade, not the page.

    `transaction_count` always described the full matching set while
    `member_count` and `totals` were folded up from the rows that fitted under
    `limit`, and the three sat adjacent with nothing to say which was which.
    Live, NVDA at limit=10 read "226 transactions, 7 members, $137,010" where
    the record held 42 members and $12,963,226 -- the dollars understated 98x
    by a paging artefact.

    So the numbers come from the same WHERE clause as the row list and are
    computed independently of how many rows that list is allowed to carry.
    """
    row = _rows(conn, f"""
        SELECT COUNT(*)                            AS row_count,
               COUNT(DISTINCT m.member_id)         AS member_count,
               SUM(COALESCE(t.amount_min, 0))      AS amount_min_total,
               SUM(COALESCE(t.amount_max, 0))      AS amount_max_total,
               SUM(CASE WHEN t.amount_min IS NOT NULL AND t.amount_max IS NULL
                        THEN 1 ELSE 0 END)         AS open_ended_count,
               SUM(CASE WHEN t.transaction_type LIKE 'purchase%'
                        THEN 1 ELSE 0 END)         AS purchase_count,
               SUM(CASE WHEN t.transaction_type LIKE 'sale%'
                        THEN 1 ELSE 0 END)         AS sale_count,
               SUM(CASE WHEN t.transaction_type = 'exchange'
                        THEN 1 ELSE 0 END)         AS exchange_count
        FROM transactions t
        JOIN filings f ON f.filing_id = t.filing_id
        LEFT JOIN members m ON m.member_id = t.member_id
        WHERE {where}""", params)[0]
    return {
        "row_count": row["row_count"] or 0,
        "member_count": row["member_count"] or 0,
        "totals": {
            "amount_min_total": row["amount_min_total"] or 0,
            # None, not a smaller number: SUM(COALESCE(max, 0)) closes an
            # open-ended bracket at zero, which is how a maximum once landed
            # below its own minimum. With such a row in the sum the disclosure
            # states no ceiling, so neither does the total.
            "amount_max_total": None if row["open_ended_count"]
            else (row["amount_max_total"] or 0),
            "open_ended_count": row["open_ended_count"] or 0,
            "purchase_count": row["purchase_count"] or 0,
            "sale_count": row["sale_count"] or 0,
            "exchange_count": row["exchange_count"] or 0,
        },
    }


def _holding_aggregate(conn: sqlite3.Connection, where: str,
                       params: tuple) -> Dict[str, Any]:
    """The same guarantee on the holdings side: totals over all matching rows.

    Holdings the filer could not price carry no bounds at all. They are
    counted, never summed as zero -- a holding nobody could value and a
    holding worth nothing are different disclosures.
    """
    row = _rows(conn, f"""
        SELECT COUNT(*)                            AS row_count,
               COUNT(DISTINCT m.member_id)         AS member_count,
               SUM(COALESCE(h.value_min, 0))       AS value_min_total,
               SUM(CASE WHEN h.value_min IS NOT NULL
                        THEN COALESCE(h.value_max, 0) ELSE 0 END)
                                                   AS value_max_total,
               SUM(CASE WHEN h.value_min IS NOT NULL
                        THEN 1 ELSE 0 END)         AS priced_count,
               SUM(CASE WHEN h.value_min IS NULL
                        THEN 1 ELSE 0 END)         AS unpriced_count,
               SUM(CASE WHEN h.value_min IS NOT NULL AND h.value_max IS NULL
                        THEN 1 ELSE 0 END)         AS open_ended_count
        FROM holdings h
        LEFT JOIN members m ON m.member_id = h.member_id
        WHERE {where}""", params)[0]
    return {
        "row_count": row["row_count"] or 0,
        "member_count": row["member_count"] or 0,
        "totals": {
            "value_min_total": row["value_min_total"] or 0,
            "value_max_total": None if row["open_ended_count"]
            else (row["value_max_total"] or 0),
            "priced_count": row["priced_count"] or 0,
            "unpriced_count": row["unpriced_count"] or 0,
            "open_ended_count": row["open_ended_count"] or 0,
        },
    }


def _aggregate_by_member(conn: sqlite3.Connection, table: str,
                         member_ids: List[str], low: str, high: str,
                         date_clause: str = "", params: tuple = ()
                         ) -> Dict[str, Dict[str, Any]]:
    """Per-member counts and bounds over ALL matching rows.

    Computed separately from the rows returned. A shared LIMIT across several
    members hands its slots to whoever sorts first -- ordered by value, Rick
    Scott's positions took 199 of 200 and Tim Scott's 213 holdings were
    reported as 1. A paging artefact must not read as a fact about a filer.
    """
    if not member_ids:
        return {}
    placeholders = ",".join("?" * len(member_ids))
    rows = _rows(conn, f"""
        SELECT x.member_id,
               COUNT(*)                       AS count,
               SUM(COALESCE(x.{low}, 0))      AS low_total,
               SUM(COALESCE(x.{high}, 0))     AS high_total,
               SUM(CASE WHEN x.{low} IS NOT NULL AND x.{high} IS NULL
                        THEN 1 ELSE 0 END)    AS open_ended_count,
               SUM(CASE WHEN x.{low} IS NULL THEN 1 ELSE 0 END) AS unpriced_count
        FROM {table} x
        WHERE x.member_id IN ({placeholders}) {date_clause}
        GROUP BY x.member_id""", (*member_ids, *params))
    return {r["member_id"]: r for r in rows}


def _match_members(conn: sqlite3.Connection, name: str) -> List[Dict[str, Any]]:
    """Members matching `name`, without sweeping in other people's first names.

    A single token is matched against the surname only. Matching it against
    the full name too made "Scott" return Scott DesJarlais and C. Scott
    Franklin alongside Rick and Tim Scott, and their holdings were then
    attributed to a query nobody meant to include them in.

    Two or more tokens are matched against the full name, so "Rick Scott"
    still resolves to one person.
    """
    query = (name or "").strip().lower()
    if not query:
        return []
    if len(query.split()) == 1:
        return _rows(conn, """
            SELECT member_id, full_name, chamber, state, district FROM members
            WHERE LOWER(last) = ? OR LOWER(last) LIKE ?
            ORDER BY full_name""", (query, f"%{query}%"))
    return _rows(conn, """
        SELECT member_id, full_name, chamber, state, district FROM members
        WHERE LOWER(full_name) LIKE ? ORDER BY full_name""", (f"%{query}%",))


def _per_member(matched: List[Dict[str, Any]],
                aggregate: Dict[str, Dict[str, Any]],
                low_key: str, high_key: str) -> List[Dict[str, Any]]:
    """Break the numbers out by person, from the aggregate not the page.

    One total covering several filers is not a fact about any of them, and a
    per-member figure taken from a truncated page is not one either.
    """
    out = []
    for member in matched:
        agg = aggregate.get(member["member_id"], {})
        unbounded = bool(agg.get("open_ended_count"))
        out.append({
            "member": member["full_name"], "chamber": member["chamber"],
            "state": member["state"], "district": member["district"],
            "count": agg.get("count", 0),
            "totals": {
                f"{low_key}_total": agg.get("low_total", 0) or 0,
                f"{high_key}_total": None if unbounded else (agg.get("high_total", 0) or 0),
                "open_ended_count": agg.get("open_ended_count", 0) or 0,
                "unpriced_count": agg.get("unpriced_count", 0) or 0,
            }})
    return out


# ------------------------------------------------------------------ by ticker

def _uncap_open_ended(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """SQL SUM cannot express "no ceiling"; COALESCE(max, 0) closes it at zero.

    An aggregate containing an open-ended bracket has no upper bound, so the
    ceiling is dropped rather than reported as a figure below its own floor.
    """
    for row in rows:
        if row.get("open_ended_count"):
            row[key] = None
    return rows


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

    clause = " AND ".join(where)
    with store.connect() as conn:
        transactions = _rows(conn, f"""
            SELECT {_TXN_COLUMNS} FROM transactions t
            JOIN filings f ON f.filing_id = t.filing_id
            LEFT JOIN members m ON m.member_id = t.member_id
            WHERE {clause}
            ORDER BY t.transaction_date DESC
            LIMIT ?""", (*params, limit))
        aggregate = _transaction_aggregate(conn, clause, tuple(params))

    coverage = store.coverage()
    matching = aggregate["row_count"]
    truncated = len(transactions) >= limit and matching > len(transactions)
    return {
        "success": True,
        "ticker": wanted,
        "query": {"since": since, "until": until, "chamber": chamber},
        "truncated": truncated,
        "transaction_count": matching,
        "rows_returned": len(transactions),
        "member_count": aggregate["member_count"],
        "totals": aggregate["totals"],
        "transactions": transactions,
        "coverage": coverage,
        "note": _note(coverage, _TRUNCATED_NOTE if truncated else ""),
    }


# ------------------------------------------------------------------ by member

def member_activity(name: str, since: Optional[str] = None,
                    limit: int = 500,
                    ticker: Optional[str] = None) -> Dict[str, Any]:
    """One member's disclosed trades, matched loosely on name.

    The members actually matched are returned alongside the trades. A loose
    match that does not say who it hit produces numbers nobody can attribute,
    and two members share a surname often enough that it matters.

    `ticker` narrows the query itself rather than the page. Filtering the
    returned rows afterwards left `totals` describing every trade the member
    made in anything, beside a `transaction_count` describing one symbol.
    """
    wanted = (ticker or "").upper().strip() or None
    aggregate: Dict[str, Any] = {"row_count": 0, "member_count": 0,
                                 "totals": _totals([])}
    per_member_agg: Dict[str, Dict[str, Any]] = {}
    top: List[Dict[str, Any]] = []
    with store.connect() as conn:
        matched = _match_members(conn, name)

        transactions: List[Dict[str, Any]] = []
        if matched:
            ids = [m["member_id"] for m in matched]
            placeholders = ",".join("?" * len(ids))
            params: List[Any] = list(ids)
            narrowing = ""
            if since:
                narrowing += " AND t.transaction_date >= ?"
                params.append(since)
            if wanted:
                narrowing += " AND t.ticker = ?"
                params.append(wanted)
            where = f"t.member_id IN ({placeholders}){narrowing}"
            transactions = _rows(conn, f"""
                SELECT {_TXN_COLUMNS} FROM transactions t
                JOIN filings f ON f.filing_id = t.filing_id
                LEFT JOIN members m ON m.member_id = t.member_id
                WHERE {where}
                ORDER BY t.transaction_date DESC
                LIMIT ?""", (*params, limit))
            aggregate = _transaction_aggregate(conn, where, tuple(params))
            per_member_agg = _aggregate_by_member(
                conn, "transactions", ids, "amount_min", "amount_max",
                narrowing.replace("t.", "x."), tuple(params[len(ids):]))
            # Ranked over every matching row too: a leaderboard built from the
            # page ranks whichever trades sorted onto it.
            top = _rows(conn, f"""
                SELECT t.ticker, COUNT(*) AS transaction_count
                FROM transactions t
                JOIN filings f ON f.filing_id = t.filing_id
                WHERE {where} AND t.ticker IS NOT NULL AND t.ticker != ''
                GROUP BY t.ticker
                ORDER BY transaction_count DESC, t.ticker
                LIMIT 10""", tuple(params))

    coverage = store.coverage()
    ambiguous = len(matched) > 1
    total_rows = aggregate["row_count"]
    truncated = len(transactions) >= limit and total_rows > len(transactions)
    extra = "" if matched else (
        f"'{name}' matched no member in the store. Either the name is spelled "
        f"differently in the filings or that member's filings have not been "
        f"ingested.")
    if ambiguous:
        extra = (f"'{name}' matched more than one member "
                 f"({', '.join(m['full_name'] for m in matched)}). The totals "
                 f"below cover all of them together; per_member breaks them "
                 f"apart. " + extra)
    return {
        "success": True,
        "query": {"name": name, "since": since, "ticker": wanted},
        "matched_members": matched,
        "ambiguous": ambiguous,
        "truncated": truncated,
        "per_member": _per_member(matched, per_member_agg,
                                  "amount_min", "amount_max"),
        "transaction_count": total_rows,
        "rows_returned": len(transactions),
        "totals": aggregate["totals"],
        "most_traded": top,
        "transactions": transactions,
        "coverage": coverage,
        "note": _note(coverage, (_TRUNCATED_NOTE + " " + extra) if truncated else extra),
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
                   SUM(CASE WHEN t.amount_min IS NOT NULL
                             AND t.amount_max IS NULL
                            THEN 1 ELSE 0 END)     AS open_ended_count,
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
    _uncap_open_ended(tickers, "amount_max_total")

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
                   SUM(COALESCE(t.amount_max, 0)) AS amount_max_total,
                   SUM(CASE WHEN t.amount_min IS NOT NULL
                             AND t.amount_max IS NULL
                            THEN 1 ELSE 0 END)    AS open_ended_count
            FROM transactions t
            JOIN filings f ON f.filing_id = t.filing_id
            JOIN members m ON m.member_id = t.member_id
            WHERE {' AND '.join(where)}
            GROUP BY t.member_id
            ORDER BY transaction_count DESC
            LIMIT ?""", (*params, limit))
    _uncap_open_ended(members, "amount_max_total")

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
    unbounded = _open_ended(priced, "value_min", "value_max")
    return {
        "value_min_total": sum(h["value_min"] or 0 for h in priced),
        # None, not a smaller number: with an open-ended row in the sum the
        # disclosure states no ceiling, and closing it at one inverts the range.
        "value_max_total": None if unbounded else sum(
            h["value_max"] or 0 for h in priced),
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
        aggregate = _holding_aggregate(conn, "h.ticker = ?", (wanted,))

    coverage = store.coverage()
    matching = aggregate["row_count"]
    truncated = len(holdings) >= limit and matching > len(holdings)
    return {"success": True, "ticker": wanted,
            "truncated": truncated,
            "holding_count": matching,
            "rows_returned": len(holdings),
            "member_count": aggregate["member_count"],
            "totals": aggregate["totals"], "holdings": holdings,
            "coverage": coverage,
            "note": _note(coverage, (_TRUNCATED_NOTE + " " + _HOLDING_NOTE)
                          if truncated else _HOLDING_NOTE)}


def member_holdings(name: str, limit: int = 1000) -> Dict[str, Any]:
    """One member's disclosed holdings, matched loosely on name."""
    aggregate: Dict[str, Any] = {"row_count": 0, "member_count": 0,
                                 "totals": _holding_totals([])}
    per_member_agg: Dict[str, Dict[str, Any]] = {}
    with store.connect() as conn:
        matched = _match_members(conn, name)

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
            aggregate = _holding_aggregate(
                conn, f"h.member_id IN ({placeholders})", tuple(ids))
            per_member_agg = _aggregate_by_member(
                conn, "holdings", ids, "value_min", "value_max")

    coverage = store.coverage()
    ambiguous = len(matched) > 1
    total_rows = aggregate["row_count"]
    truncated = len(holdings) >= limit and total_rows > len(holdings)
    extra = _HOLDING_NOTE if matched else (
        f"'{name}' matched no member in the store. {_HOLDING_NOTE}")
    if ambiguous:
        extra = (f"'{name}' matched more than one member "
                 f"({', '.join(m['full_name'] for m in matched)}). The totals "
                 f"below cover all of them together; per_member breaks them "
                 f"apart. " + extra)
    return {"success": True, "query": {"name": name},
            "matched_members": matched, "ambiguous": ambiguous,
            "truncated": truncated,
            "per_member": _per_member(matched, per_member_agg,
                                      "value_min", "value_max"),
            "holding_count": total_rows,
            "rows_returned": len(holdings),
            "totals": aggregate["totals"], "holdings": holdings,
            "coverage": coverage,
            "note": _note(coverage,
                          (_TRUNCATED_NOTE + " " + extra) if truncated else extra)}
