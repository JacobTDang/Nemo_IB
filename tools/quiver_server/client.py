"""QuiverQuant web-app client — typed access to quiverquant.com's pages.

QuiverQuant's public web application is a server-rendered Django site.
Its "API" is therefore not one protocol but three surfaces, all discovered
by reading the app's own JavaScript (see tools/quiver_server/server.py for
the tool layer):

  1. JSON endpoints (GET, no auth):
       /get_current_price/<ticker>
       /get_bulls_bears_data/<ticker>/
       /get_tv_lazy_data/<ticker>/
       /_get_ticker_insiders/<ticker>/
       /_get_ticker_institutions/<ticker>/
       /_get_ticker_forecast/<ticker>/
       /get_politician_page_tab_data/<bioguide_id>
       /get_politician_page_tab_data_2/<bioguide_id>
  2. JavaScript literals embedded in page HTML (server-rendered data):
       recentTradesData   -- /congresstrading/  (300 most recent trades)
       tradeData          -- /congresstrading/politician/<Name>-<BioguideID>
       allCongressMembers -- /congresstrading/  (politician directory)
       Plotly.newPlot calls -- /wallstreetbets/ (mention time series,
                             calls/puts comment counts, sentiment scatter)
  3. HTML tables (server-rendered):
       /wallstreetbets/          -- mention counts per ticker
       /lobbying/                 -- recent filings + top spenders
       /sources/govcontracts      -- contract awards

Everything here is public data the site serves to anonymous visitors. A
logged-in session cookie is optional (raises rate ceilings slightly) and
is attached when QUIVER_SESSION_COOKIE is set.

Caching: the wallstreetbets dashboard is ~28 MB of HTML, so page fetches
are cached with a TTL (default 15 min) to keep repeated tool calls cheap.
"""
from __future__ import annotations

import ast
import codecs
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

BASE_URL = "https://www.quiverquant.com"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/157.0 Safari/537.36"
)

# The congress page embeds ~300 trades, the WSB dashboard ~28 MB. A short TTL
# keeps a burst of tool calls from re-downloading them, while staying short
# enough that "recent trades" stays recent.
PAGE_CACHE_TTL_S = 900.0

# Cloudflare sits in front of quiverquant.com and answers 429 to a burst of
# anonymous requests. The window is not published: twelve page fetches back to
# back pass, while roughly eighty requests inside a few seconds -- which one
# ticker snapshot contributes six of, and a client looping tickers reaches
# without trying -- does not. Requests are therefore spaced a little, and a
# 429 is retried once, which keeps an honest burst from turning into a wrong
# answer. The spacing costs nothing in practice: the pages are cached for
# fifteen minutes, so only the first call of a dataset pays it.
_MIN_REQUEST_INTERVAL_S = 0.2
_RATE_LIMIT_RETRY_S = 1.5
_RATE_LIMIT_RETRY_CAP_S = 5.0

_PACE_LOCK = threading.Lock()
_last_request_at = 0.0


def _pace() -> None:
    """Hold requests at least _MIN_REQUEST_INTERVAL_S apart, across threads."""
    global _last_request_at
    with _PACE_LOCK:
        gap = _MIN_REQUEST_INTERVAL_S - (time.time() - _last_request_at)
        if gap > 0:
            time.sleep(gap)
        _last_request_at = time.time()


def _retry_delay(resp: Any) -> float:
    """How long to wait before retrying, honouring Retry-After if sent."""
    raw = resp.headers.get("retry-after")
    if raw:
        try:
            return min(max(float(raw), 0.0), _RATE_LIMIT_RETRY_CAP_S)
        except ValueError:
            pass  # an HTTP-date, not a delay in seconds
    return _RATE_LIMIT_RETRY_S


def _snapshot_is_hollow(snapshot: Dict[str, Any]) -> bool:
    """True when every section answered with nothing.

    Quiver answers HTTP 200 for a ticker it does not cover: current_price
    "N/A", empty arrays everywhere. Reporting that as a covered ticker that
    happened to have nothing to report is the silent-absence mistake -- a
    caller reads six empty sections and concludes the company has no
    insiders, no institutions and no analyst coverage.
    """
    price = (snapshot.get("price") or {}).get("price_data") or {}
    quoted = str(price.get("current_price") or "").strip().upper()
    if quoted not in ("", "N/A"):
        return False
    if (snapshot.get("insiders") or {}).get("insider_trades"):
        return False
    for key in ("institutions", "etf_holders"):
        if snapshot.get(key):
            return False
    forecast = snapshot.get("forecast") or {}
    if forecast.get("data") or forecast.get("price_targets"):
        return False
    bulls = snapshot.get("bulls_bears") or {}
    if bulls and bulls.get("response") not in (None, "", "empty"):
        return False
    return True


class QuiverError(Exception):
    """A fetch or parse failure with a machine-readable reason code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


# ---------------------------------------------------------------------------
# Page cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    body: str
    fetched_at: float


_PAGE_CACHE: Dict[str, _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(path: str, ttl: float) -> Optional[str]:
    with _CACHE_LOCK:
        entry = _PAGE_CACHE.get(path)
        if entry and (time.time() - entry.fetched_at) < ttl:
            return entry.body
    return None


def _cache_put(path: str, body: str) -> None:
    with _CACHE_LOCK:
        _PAGE_CACHE[path] = _CacheEntry(body=body, fetched_at=time.time())


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class QuiverWebClient:
    """Fetches and parses quiverquant.com's web pages and JSON endpoints."""

    def __init__(self, session_cookie: Optional[str] = None,
                 timeout: float = 30.0, base_url: str = BASE_URL):
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if session_cookie:
            headers["Cookie"] = f"sessionid={session_cookie}"
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout,
            follow_redirects=True,
        )
        self._xsrf_headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
        }

    # -- transport ----------------------------------------------------------

    def _fetch(self, path: str, *, json_headers: bool = False) -> str:
        """GET a URL as text, streamed and decoded incrementally.

        The WSB dashboard is ~28 MB of HTML; reading it through
        resp.text buffers the body and the decoded string whole, which
        measures at ~350 MB of RSS. Streaming the bytes and decoding in
        chunks halves that, which is the difference between a 512 MB
        container limit and an OOM kill.
        """
        extra = self._xsrf_headers if json_headers else {}
        rate_limited_once = False
        while True:
            _pace()
            try:
                with self._client.stream("GET", path,
                                         headers=extra) as resp:
                    if resp.status_code in (403, 429):
                        # A single retry turns a self-inflicted burst into a
                        # slow answer instead of a wrong one. More than one
                        # would hold an MCP call open past its budget.
                        if resp.status_code == 429 and not rate_limited_once:
                            rate_limited_once = True
                            time.sleep(_retry_delay(resp))
                            continue
                        raise QuiverError(
                            "rate_limited",
                            f"{path} returned HTTP {resp.status_code} "
                            f"(Cloudflare or rate limit). Retry later or set "
                            f"QUIVER_SESSION_COOKIE.")
                    if resp.status_code == 404:
                        raise QuiverError(
                            "not_found", f"{path} returned HTTP 404")
                    if resp.status_code != 200:
                        raise QuiverError(
                            "provider_unavailable",
                            f"{path} returned HTTP {resp.status_code}")
                    decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    parts: List[str] = []
                    for chunk in resp.iter_bytes(65536):
                        piece = decoder.decode(chunk)
                        if piece:
                            parts.append(piece)
                    return "".join(parts)
            except httpx.HTTPError as exc:
                raise QuiverError(
                    "provider_unavailable",
                    f"request to {path} failed: {exc}") from exc

    def _page(self, path: str, *, ttl: float = PAGE_CACHE_TTL_S) -> str:
        cached = _cache_get(path, ttl)
        if cached is not None:
            return cached
        body = self._fetch(path)
        _cache_put(path, body)
        return body

    def _json(self, path: str) -> Any:
        body = self._fetch(path, json_headers=True)
        try:
            return json.loads(body)
        except ValueError as exc:
            raise QuiverError(
                "parse_failed", f"{path} returned non-JSON body") from exc

    def close(self) -> None:
        self._client.close()

    # -- embedded-JS parsing helpers ---------------------------------------

    @staticmethod
    def _extract_js_array(page: str, var_name: str,
                          literal: bool = False) -> List[Any]:
        """Pull `var_name = [...]` (or `var_name = JSON.parse(`[...]`)`) out
        of a page's inline scripts.

        recentTradesData uses single-quoted JS literals (parsed with
        ast.literal_eval); tradeData / JSON.parse blocks are double-quoted
        (parsed with json.loads, after NaN -> null).
        """
        # JSON.parse(`[...]`) form
        m = re.search(
            re.escape(var_name) + r"\s*=\s*JSON\.parse\(`(.*?)`\)",
            page, re.DOTALL)
        if m:
            raw = m.group(1)
            raw = re.sub(r"\bNaN\b", "null", raw)
            return json.loads(raw)
        # plain literal form: scan to the matching close bracket
        m = re.search(r"(?:let|const|var)\s+" + re.escape(var_name) +
                      r"\s*=\s*", page)
        if not m:
            raise QuiverError(
                "parse_failed",
                f"{var_name} not found on page (page layout changed?)")
        start = m.end()
        if start >= len(page) or page[start] != "[":
            raise QuiverError(
                "parse_failed", f"{var_name} is not an array literal")
        depth = 0
        in_str = False
        quote = ""
        i = start
        while i < len(page):
            c = page[i]
            if in_str:
                if c == "\\":
                    i += 2
                    continue
                if c == quote:
                    in_str = False
            elif c in "'\"`":
                in_str = True
                quote = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    raw = page[start:i + 1]
                    if literal:
                        return ast.literal_eval(raw)
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        # JS literal with single quotes / NaN
                        return ast.literal_eval(raw)
            i += 1
        raise QuiverError(
            "parse_failed", f"{var_name} array is truncated on page")

    @staticmethod
    def _extract_newplot(page: str) -> List[List[Dict[str, Any]]]:
        """Every Plotly.newPlot(div, traces, layout, ...) call's traces.

        The WSB dashboard renders its charts server-side: the data arrays
        live in the page as JSON. Returns a list of trace lists.

        Scanned in place. Slicing the page per match copies up to the
        whole 28 MB eight times over, and the freed copies stay in RSS as
        a high-water mark -- ~260 MB of it, measured.
        """
        charts: List[List[Dict[str, Any]]] = []
        n = len(page)
        for m in re.finditer(r"Plotly\.newPlot\(\s*[\"']", page):
            # skip the div id string to the traces array
            start = m.end()
            while start < n and page[start] not in "[{":
                start += 1
            if start >= n or start - m.end() > 120:
                continue  # too far -> not a normal call (library code)
            depth = 0
            in_str = False
            quote = ""
            i = start
            end = None
            while i < n:
                c = page[i]
                if in_str:
                    if c == "\\":
                        i += 2
                        continue
                    if c == quote:
                        in_str = False
                elif c in "'\"":
                    in_str = True
                    quote = c
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
                i += 1
            if end is None:
                continue
            raw = page[start:end + 1]
            try:
                traces = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (isinstance(traces, list) and traces
                    and isinstance(traces[0], dict)
                    and "type" in traces[0]):
                charts.append(traces)
        return charts

    # -- HTML table parsing --------------------------------------------------

    @staticmethod
    def _parse_tables(page: str) -> List[List[List[str]]]:
        """Every <table> as row/cell plain text.

        Quiver's WSB dashboard is ~28 MB of HTML, almost all of it inline
        chart JavaScript around one small table. Building a DOM for the
        whole page to reach it costs hundreds of MB of RSS, so the table
        markup is cut out with a regex first and each fragment parsed on
        its own. The pages served here nest no tables.
        """
        from bs4 import BeautifulSoup
        out: List[List[List[str]]] = []
        for frag in re.findall(r"<table\b.*?</table>", page, re.DOTALL):
            soup = BeautifulSoup(frag, "html.parser")
            rows: List[List[str]] = []
            for tr in soup.find_all("tr"):
                cells = [c.get_text(" ", strip=True)
                         for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                out.append(rows)
        return out

    # -----------------------------------------------------------------------
    # Congressional trading
    # -----------------------------------------------------------------------

    def get_congress_trades(self, *, ticker: Optional[str] = None,
                            politician: Optional[str] = None,
                            tx_type: Optional[str] = None,
                            chamber: Optional[str] = None,
                            party: Optional[str] = None,
                            days: Optional[int] = None,
                            limit: Optional[int] = None) -> Dict[str, Any]:
        """The ~300 most recent congressional trades, filtered.

        Fields (16): ticker, security_name, asset_type, tx_type, amount_range,
        politician_display, chamber, party, filed_date, traded_date, _,
        tx_id, _, politician_name, image_url, bioguide_id.
        """
        page = self._page("/congresstrading/")
        rows = self._extract_js_array(page, "recentTradesData")
        trades = []
        for r in rows:
            if not isinstance(r, list) or len(r) < 16:
                continue
            trades.append({
                "ticker": r[0] if r[0] and r[0] != "-" else None,
                "security_name": r[1],
                "asset_type": r[2],
                "tx_type": r[3],
                "amount_range": r[4],
                "politician": r[13] or r[5],
                "chamber": r[6],
                "party": r[7],
                "filed_date": (r[8] or "")[:10],
                "traded_date": (r[9] or "")[:10],
                "tx_id": r[11] if r[11] and r[11] != "-" else None,
                "bioguide_id": r[15],
            })
        total = len(trades)
        if ticker:
            ticker = ticker.upper()
            trades = [t for t in trades if t["ticker"] == ticker]
        if politician:
            needle = politician.lower()
            trades = [t for t in trades
                      if needle in (t["politician"] or "").lower()
                      or needle in (t["bioguide_id"] or "").lower()]
        if tx_type:
            needle = tx_type.lower()
            trades = [t for t in trades
                      if needle in (t["tx_type"] or "").lower()]
        if chamber:
            trades = [t for t in trades
                      if (t["chamber"] or "").lower() == chamber.lower()]
        if party:
            trades = [t for t in trades
                      if (t["party"] or "").upper() == party.upper()]
        if days is not None:
            # filed_date is the freshest timestamp; filter on it.
            cutoff = time.time() - days * 86400
            keep = []
            for t in trades:
                try:
                    if time.mktime(time.strptime(t["filed_date"],
                                                 "%Y-%m-%d")) >= cutoff:
                        keep.append(t)
                except ValueError:
                    keep.append(t)  # unparseable date: keep, don't drop
            trades = keep
        if limit is not None and limit > 0:
            trades = trades[:limit]
        return {"trades": trades, "total_embedded": total,
                "returned": len(trades)}

    def resolve_politician(self, query: str) -> List[Dict[str, Any]]:
        """Search the embedded congressional directory by name or ID."""
        page = self._page("/congresstrading/")
        members = self._extract_js_array(page, "allCongressMembers")
        needle = query.lower()
        out = []
        for m in members:
            if not isinstance(m, dict):
                continue
            if (needle in (m.get("name") or "").lower()
                    or needle == (m.get("bioguideID") or "").lower()):
                out.append({
                    "name": m.get("name"),
                    "bioguide_id": m.get("bioguideID"),
                    "chamber": m.get("chamber"),
                    "party": m.get("party"),
                    "district": m.get("district"),
                })
        return out

    @staticmethod
    def _politician_slug(name: str, bioguide_id: str) -> str:
        """'Nancy Pelosi' + 'P000197' -> 'Nancy%20Pelosi-P000197'."""
        return f"{name.replace(' ', '%20')}-{bioguide_id}"

    def get_politician_trades(self, politician: str,
                              limit: Optional[int] = None
                              ) -> Dict[str, Any]:
        """A politician's full disclosed trade history.

        `politician` is a name ('Nancy Pelosi'), a bioguide ID ('P000197'),
        or the full slug. The page embeds tradeData with 15 fields:
        ticker, tx_type, filed, traded, description, excess_return,
        politician, tx_id, security_name, asset_type, amount_range,
        chamber, party, sector, amount_midpoint.
        """
        slug = politician
        if "/" not in politician and "-" not in politician.split(" ")[-1]:
            # resolve name or bare bioguide id to the page slug
            matches = self.resolve_politician(politician)
            exact = [m for m in matches
                     if m["bioguide_id"].lower() == politician.lower()]
            chosen = (exact or matches)
            if not chosen:
                raise QuiverError(
                    "unknown_politician",
                    f"no congress member matches {politician!r}")
            if len(chosen) > 1:
                # prefer an exact name match, else the first — page trade
                # history is name-scoped, and partial names that hit two
                # members are the caller's ambiguity to resolve.
                by_name = [m for m in chosen
                           if m["name"].lower() == politician.lower()]
                chosen = by_name or chosen[:1]
            m0 = chosen[0]
            slug = self._politician_slug(m0["name"], m0["bioguide_id"])
        page = self._page(f"/congresstrading/politician/{slug}")
        rows = self._extract_js_array(page, "tradeData")
        trades = []
        for r in rows:
            if not isinstance(r, list) or len(r) < 15:
                continue
            trades.append({
                "ticker": r[0],
                "tx_type": r[1],
                "filed_date": (r[2] or "")[:10],
                "traded_date": (r[3] or "")[:10],
                "description": r[4],
                "security_name": r[8],
                "asset_type": r[9],
                "amount_range": r[10],
                "amount_midpoint": r[14],
                "chamber": r[11],
                "party": r[12],
                "sector": r[13],
                "tx_id": r[7],
            })
        trades.sort(key=lambda t: t["filed_date"] or "", reverse=True)
        total = len(trades)
        if limit is not None and limit > 0:
            trades = trades[:limit]
        return {"politician": slug, "trades": trades,
                "total": total, "returned": len(trades)}

    def get_politician_donors(self, bioguide_id: str) -> Dict[str, Any]:
        """A politician's top donors (FEC data via the politician page JSON)."""
        data = self._json(f"/get_politician_page_tab_data/{bioguide_id}")
        out: Dict[str, Any] = {"bioguide_id": bioguide_id}
        raw = data.get("politician_donors", {})
        if isinstance(raw, dict):
            raw = raw.get("politician_donors")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        donors = []
        for d in raw or []:
            if isinstance(d, list) and len(d) >= 5:
                donors.append({
                    "ticker": d[0],
                    "donor": d[1],
                    "recipient": d[2],
                    "amount": d[3],
                    "date": (d[4] or "")[:10],
                    "party": d[5] if len(d) > 5 else None,
                    "cycle": d[6] if len(d) > 6 else None,
                })
        out["donors"] = donors
        return out

    # -----------------------------------------------------------------------
    # WallStreetBets / Reddit sentiment
    # -----------------------------------------------------------------------

    def get_wsb_sentiment(self, *, ticker: Optional[str] = None,
                          include_history: bool = False,
                          limit: Optional[int] = None) -> Dict[str, Any]:
        """WSB daily-discussion mention counts and, for top tickers,
        mention time series, calls/puts comments and a sentiment score.

        The dashboard page embeds everything server-side. The mention table
        covers every discussed ticker (~600); time series and sentiment
        cover only the ~25 tickers Quiver charts.
        """
        page = self._page("/wallstreetbets/")
        tables = self._parse_tables(page)
        mentions: List[Dict[str, Any]] = []
        for table in tables:
            if not table or not table[0]:
                continue
            header = [h.lower() for h in table[0]]
            if "mentions" not in " ".join(header):
                continue
            for row in table[1:]:
                if len(row) < 3:
                    continue
                mentions.append({
                    "ticker": row[0].split()[0] if row[0] else None,
                    "mentions": row[1],
                    "change": row[2],
                })
        total = len(mentions)
        result: Dict[str, Any] = {"mentions": mentions,
                                  "tickers_embedded": total}

        charts = self._extract_newplot(page)
        # Quiver's WSB dashboard embeds a mention time series whose last
        # point can trail "today" by months when their Reddit scrape
        # pipeline stalls. The honest thing is to date the data: every
        # chart x-axis max, reported as data_horizon.
        horizon = ""
        for traces in charts:
            for t in traces:
                for x in (t.get("x") or []):
                    # bar charts use ticker symbols on x; only date-like
                    # values (2025-01-22T...) count as a horizon.
                    if (isinstance(x, str) and x[:2] == "20"
                            and len(x) >= 10 and x[4] == "-"
                            and x[:10] > horizon):
                        horizon = x[:10]
        if horizon:
            result["data_horizon"] = horizon


        # The sentiment scatter: customdata rows are
        # [ticker, sentiment, est_volume, change_in_mentions, volume_str].
        sentiment: Dict[str, Dict[str, Any]] = {}
        for traces in charts:
            for t in traces:
                if t.get("type") == "scatter" and t.get("customdata"):
                    for cd in t["customdata"]:
                        if isinstance(cd, list) and len(cd) >= 2:
                            sentiment[str(cd[0])] = {
                                "sentiment": cd[1],
                                "mentions_change": cd[3] if len(cd) > 3
                                else None,
                            }
        if sentiment:
            result["sentiment_by_ticker"] = sentiment

        if ticker:
            ticker = ticker.upper()
            row = next((m for m in mentions
                        if (m["ticker"] or "").upper() == ticker), None)
            result["ticker_row"] = row
            result["ticker_sentiment"] = sentiment.get(ticker)
            if row is None:
                # The dashboard carries ~600 tickers. One that is absent was
                # not mentioned in the window -- which is a different answer
                # from "mentioned zero times" and must be named as such.
                result["coverage"] = "not_covered"
            else:
                result["coverage"] = "full"
            if include_history:
                # The dashboard stacks three mention charts (1-day hourly,
                # ~5-day, ~30-day). Pick the LONGEST series for the ticker
                # and aggregate it to daily counts.
                best: List[Tuple[str, Any]] = []
                for traces in charts:
                    for t in traces:
                        if t.get("name", "").upper() != ticker:
                            continue
                        xs, ys = t.get("x") or [], t.get("y") or []
                        pairs = [(str(x), y) for x, y in zip(xs, ys)
                                 if y is not None]
                        if len(pairs) > len(best):
                            best = pairs
                if best:
                    best.sort(key=lambda p: p[0])
                    daily: Dict[str, int] = {}
                    for x, y in best:
                        day = x[:10]
                        daily[day] = daily.get(day, 0) + int(y)
                    history = [{"date": d, "mentions": v}
                               for d, v in sorted(daily.items())]
                    result["history"] = history
                    result["history_days"] = len(history)
                    result["history_window"] = {
                        "start": best[0][0], "end": best[-1][0]}

        if limit is not None and limit > 0 and not ticker:
            result["mentions"] = mentions[:limit]
            result["returned"] = len(result["mentions"])
        return result

    # -----------------------------------------------------------------------
    # Corporate lobbying
    # -----------------------------------------------------------------------

    # "Nov. 9, 2026, 3:10 p.m." -> "2026-11-09". Kept as a small map rather
    # than dateparser so the parse stays pure-stdlib and cheap per row.
    _MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
               "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

    @classmethod
    def _normalize_date(cls, raw: str) -> str:
        m = re.match(r"([A-Za-z]{3})\.?\s+(\d{1,2}),\s+(\d{4})", raw or "")
        if m:
            mon = cls._MONTHS.get(m.group(1).lower())
            if mon:
                return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"
        return raw

    def get_lobbying_filings(self, *, ticker: Optional[str] = None,
                             limit: Optional[int] = None) -> Dict[str, Any]:
        """Recent federal lobbying filings and the top-spender leaderboard."""
        page = self._page("/lobbying/")
        tables = self._parse_tables(page)
        filings: List[Dict[str, Any]] = []
        top_spenders: List[Dict[str, Any]] = []
        for table in tables:
            if not table or not table[0]:
                continue
            header = " ".join(table[0]).lower()
            if "date" in header:
                for row in table[1:]:
                    if len(row) < 3:
                        continue
                    filings.append({
                        "ticker": row[0].split()[0] if row[0] else None,
                        "amount": row[1],
                        "date": self._normalize_date(row[2]),
                    })
            elif header.startswith("ticker") or "ticker amount" == header:
                for row in table[1:]:
                    if len(row) < 2:
                        continue
                    top_spenders.append({
                        "ticker": row[0].split()[0] if row[0] else None,
                        "amount": row[1],
                    })
        total = len(filings)
        if ticker:
            ticker = ticker.upper()
            filings = [f for f in filings
                       if (f["ticker"] or "").upper() == ticker]
            top_spenders = [f for f in top_spenders
                            if (f["ticker"] or "").upper() == ticker]
        if limit is not None and limit > 0:
            filings = filings[:limit]
        return {"filings": filings, "top_spenders": top_spenders,
                "total_filings": total, "returned": len(filings)}

    # -----------------------------------------------------------------------
    # Government contracts
    # -----------------------------------------------------------------------

    def get_gov_contract_awards(self, *, ticker: Optional[str] = None,
                                days: Optional[int] = None,
                                limit: Optional[int] = None
                                ) -> Dict[str, Any]:
        """Recent federal contract awards (recipient, agency, amount)."""
        page = self._page("/sources/govcontracts")
        tables = self._parse_tables(page)
        awards: List[Dict[str, Any]] = []
        for table in tables:
            if not table or not table[0]:
                continue
            header = " ".join(table[0]).lower()
            if "awarded" not in header:
                continue
            for row in table[1:]:
                if len(row) < 5:
                    continue
                awards.append({
                    "recipient": row[0].split()[0] if row[0] else None,
                    "date_awarded": row[1],
                    "amount": row[2],
                    "agency": row[3],
                    "description": row[4],
                })
        total = len(awards)
        if ticker:
            ticker = ticker.upper()
            awards = [a for a in awards
                      if (a["recipient"] or "").upper() == ticker]
        if days is not None:
            cutoff = time.time() - days * 86400
            keep = []
            for a in awards:
                d = a["date_awarded"]
                try:
                    if time.mktime(time.strptime(d[:10], "%Y-%m-%d")) >= cutoff:
                        keep.append(a)
                except ValueError:
                    keep.append(a)
            awards = keep
        if limit is not None and limit > 0:
            awards = awards[:limit]
        return {"awards": awards, "total_embedded": total,
                "returned": len(awards)}

    # -----------------------------------------------------------------------
    # Per-ticker JSON endpoints
    # -----------------------------------------------------------------------

    def get_ticker_snapshot(self, ticker: str,
                            trade_limit: int = 25) -> Dict[str, Any]:
        """Per-ticker data from Quiver's JSON endpoints.

        Sections: price, insiders, institutions, forecast, etf_holders,
        bulls_bears (session-only). Each section degrades independently —
        a failed or auth-gated endpoint never drops the whole snapshot.
        """
        ticker = ticker.upper()
        out: Dict[str, Any] = {"ticker": ticker}
        degraded: List[str] = []

        def safe(name: str, fn) -> None:
            try:
                out[name] = fn()
            except QuiverError as exc:
                degraded.append(f"{name}:{exc.reason}")
            except Exception:  # endpoint-specific surprises
                degraded.append(f"{name}:unexpected")

        def _insiders() -> Dict[str, Any]:
            data = self._json(f"/_get_ticker_insiders/{ticker}/")
            trades = data.get("insider_trades") or []
            # A busy ticker carries 1,600+ rows; the report period's
            # detail is in the tail, so keep the most recent N.
            trades = trades[-trade_limit:]
            return {"insider_trades": trades, "total_trades":
                    len(data.get("insider_trades") or [])}

        def _institutions() -> List[Any]:
            data = self._json(f"/_get_ticker_institutions/{ticker}/")
            return data.get("response", {}).get("heatmap_data")

        def _bulls_bears() -> Dict[str, Any]:
            data = self._json(f"/get_bulls_bears_data/{ticker}/")
            if data.get("response") == "no-auth":
                # Session-gated: reachable only with QUIVER_SESSION_COOKIE.
                raise QuiverError("no_auth",
                                  "bulls/bears requires a logged-in session")
            return data

        safe("price", lambda: self._json(f"/get_current_price/{ticker}"))
        safe("bulls_bears", _bulls_bears)
        safe("insiders", _insiders)
        safe("institutions", _institutions)
        safe("forecast", lambda: self._json(
            f"/_get_ticker_forecast/{ticker}/"))
        safe("etf_holders", lambda: (self._json(
            f"/get_tv_lazy_data/{ticker}/") or {}).get("etf_data"))
        if degraded:
            out["degraded_sections"] = degraded
        sections = [k for k in out
                    if k not in ("ticker", "degraded_sections")]
        if not sections:
            # Every section failed. Which reason that is decides what it
            # means: a ticker Quiver does not cover answers 404 throughout,
            # while a rate limit or an outage says nothing about the ticker.
            # Blaming the ticker for a 429 turns "slow down" into "no such
            # company", which is the wrong answer to act on.
            reasons = {entry.split(":", 1)[1] for entry in degraded
                       if ":" in entry}
            for transport in ("rate_limited", "provider_unavailable",
                              "unexpected"):
                if transport in reasons:
                    raise QuiverError(
                        transport,
                        f"every section for {ticker} failed: "
                        + ", ".join(sorted(degraded)))
            raise QuiverError(
                "unknown_ticker",
                f"no Quiver endpoint answered for {ticker}")
        if _snapshot_is_hollow(out):
            raise QuiverError(
                "unknown_ticker",
                f"Quiver returned no price, insiders, institutions, "
                f"forecast or ETF holders for {ticker}; the ticker is "
                f"most likely not covered")
        out["coverage"] = "partial" if degraded else "full"
        return out
