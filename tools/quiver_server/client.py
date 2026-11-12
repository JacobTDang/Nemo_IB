"""QuiverQuant web-app client — typed access to quiverquant.com's pages.

QuiverQuant's public web application is a server-rendered Django site.
Its "API" is therefore not one protocol but five surfaces, all discovered
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
       /get_gov_spending_data/     (500 ticker-tagged transactions)
       /get_fund_graph_13f/<fund>  (a Plotly treemap of 13F positions)
  2. JavaScript literals embedded in page HTML (server-rendered data):
       recentTradesData   -- /congresstrading/  (300 most recent trades)
       tradeData          -- /congresstrading/politician/<Name>-<BioguideID>
       allCongressMembers -- /congresstrading/  (politician directory)
       Plotly.newPlot calls -- /wallstreetbets/ (mention time series,
                             calls/puts comment counts, sentiment scatter)
  3. JSON blocks in `<script id="...">` tags -- dashboards that render their
     table client-side, so the rows appear nowhere in the HTML:
       layoffsRowsData       -- /layoffs/  (2,000 WARN notices)
       revolvingDoorRowsData -- /revolving-door/  (1,500 registrations)
  4. HTML tables (server-rendered):
       /wallstreetbets/          -- mention counts per ticker
       /lobbying/                 -- recent filings + top spenders
       /sources/govcontracts      -- contract awards
       /sec13f/                   -- 13F filers
  5. POST search forms -- no GET route exists, so the query is submitted with
     the page's own CSRF token, exactly as the site's search box does:
       /patentsearch/            -- recent patent grants by search term

Everything here is public data the site serves to anonymous visitors. A
logged-in session cookie is optional (raises rate ceilings slightly) and
is attached when QUIVER_SESSION_COOKIE is set.

Caching: the wallstreetbets dashboard is ~28 MB of HTML, so page fetches
are cached with a TTL (default 15 min) to keep repeated tool calls cheap.
"""
from __future__ import annotations

import ast
import codecs
import datetime as _dt
import json
import re
import threading
import time
import urllib.parse
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


_CSRF_RE = re.compile(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"')


def _csrf_token(page: str) -> Optional[str]:
    """The hidden token the site's search forms submit alongside the query."""
    m = _CSRF_RE.search(page)
    return m.group(1) if m else None


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

    def _post(self, path: str, data: Dict[str, str],
              *, referer: str, cache_key: Optional[str] = None) -> str:
        """Submit one of the site's own search forms, with its CSRF token.

        Two dashboards -- patent search and ETF holdings -- have no GET form
        at all: the page is a form that POSTs. Nothing here is privileged; it
        is the same request the search box makes, and the token is the one the
        site hands out with the cookie jar this client already holds.

        Results are cached under `cache_key` because a search is expensive on
        the site's side and deterministic for a day: repeating it within the
        TTL costs them nothing.
        """
        if cache_key:
            cached = _cache_get(cache_key, PAGE_CACHE_TTL_S)
            if cached is not None:
                return cached

        page = self._page(path)
        token = _csrf_token(page)
        if token is None:
            raise QuiverError(
                "parse_failed",
                f"no CSRF token on {path} (page layout changed?)")

        form = dict(data)
        form["csrfmiddlewaretoken"] = token
        headers = {
            "Referer": str(self._client.base_url.join(referer)),
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": str(self._client.base_url).rstrip("/"),
        }
        rate_limited_once = False
        while True:
            _pace()
            try:
                resp = self._client.post(path, data=form, headers=headers)
            except httpx.HTTPError as exc:
                raise QuiverError(
                    "provider_unavailable",
                    f"POST {path} failed: {exc}") from exc
            if resp.status_code in (403, 429):
                if resp.status_code == 429 and not rate_limited_once:
                    rate_limited_once = True
                    time.sleep(_retry_delay(resp))
                    continue
                raise QuiverError(
                    "rate_limited",
                    f"POST {path} returned HTTP {resp.status_code}")
            if resp.status_code == 404:
                raise QuiverError("not_found", f"POST {path} returned HTTP 404")
            if resp.status_code != 200:
                raise QuiverError(
                    "provider_unavailable",
                    f"POST {path} returned HTTP {resp.status_code}")
            body = resp.text
            if cache_key:
                _cache_put(cache_key, body)
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

    # -- Embedded JSON payloads ----------------------------------------------

    @staticmethod
    def _extract_plotly_json(raw: str) -> Dict[str, Any]:
        """The first trace of a `Plotly.newPlot` call as a dict.

        The fund-graph endpoint returns its payload with the quotes escaped,
        as a JS string rather than an object literal, so neither json.loads
        nor the treemap-shaped reader in _extract_newplot can take it
        directly. Decode the escapes, then parse the array.
        """
        call = raw.find("Plotly.newPlot(")
        if call < 0:
            raise QuiverError(
                "parse_failed", "no Plotly.newPlot payload in the response")
        start = raw.find("[", call)
        if start < 0:
            raise QuiverError(
                "parse_failed", "Plotly payload has no data array")
        depth = 0
        escaped = False
        in_str = False
        quote = ""
        end = None
        i = start
        while i < len(raw):
            c = raw[i]
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif in_str:
                if c == quote:
                    in_str = False
            elif c in "\"'":
                in_str = True
                quote = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end is None:
            raise QuiverError("parse_failed", "Plotly payload is truncated")

        chunk = raw[start:end]
        traces = None
        for candidate in (chunk, chunk.replace('\\"', '"')):
            try:
                traces = json.loads(candidate)
                break
            except ValueError:
                continue
        if traces is None:
            try:
                traces = ast.literal_eval(chunk.replace('\\"', '"'))
            except (ValueError, SyntaxError) as exc:
                raise QuiverError(
                    "parse_failed",
                    "Plotly payload is not valid JSON") from exc
        if not isinstance(traces, list) or not traces:
            raise QuiverError("parse_failed", "Plotly payload has no traces")
        first = traces[0]
        if not isinstance(first, dict):
            raise QuiverError("parse_failed", "Plotly trace is not an object")
        return first

    @staticmethod
    def _extract_json_script(page: str, element_id: str) -> List[Any]:
        """Rows carried in `<script id="..." type="application/json">`.

        Several dashboards -- layoffs, revolving door -- do not put their rows
        in a JS literal or a rendered table. They ship them as a JSON block
        and let the page's own table class render them, so both of the older
        readers walked straight past 2,000 rows. A missing block is a layout
        change, not an empty dataset, and says so.
        """
        m = re.search(
            r'<script[^>]*\bid="' + re.escape(element_id) + r'"[^>]*>(.*?)'
            r'</script>', page, re.DOTALL)
        if m is None:
            # Attribute order is not guaranteed; try type-first as well.
            m = re.search(
                r'<script[^>]*type="application/json"[^>]*\bid="'
                + re.escape(element_id) + r'"[^>]*>(.*?)</script>',
                page, re.DOTALL)
        if m is None:
            raise QuiverError(
                "parse_failed",
                f"{element_id} not found on page (page layout changed?)")
        try:
            rows = json.loads(m.group(1))
        except ValueError as exc:
            raise QuiverError(
                "parse_failed",
                f"{element_id} is not valid JSON") from exc
        if not isinstance(rows, list):
            raise QuiverError(
                "parse_failed", f"{element_id} is not an array")
        return rows

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
                "returned": len(trades),
                # The dashboard embeds the latest ~300 trades, not history:
                # a question about a quiet ticker can fall outside the window.
                "coverage": "partial"}

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
                "total": total, "returned": len(trades),
                # A politician page carries their whole disclosed history.
                "coverage": "full"}

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
        # Without a ticker this is the dashboard's whole table; with one, the
        # ticker branch above already said whether it was found.
        result.setdefault("coverage", "full")
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
                "total_filings": total, "returned": len(filings),
                # The page publishes the latest 100 filings, not the register.
                "coverage": "partial"}

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
                "returned": len(awards),
                # The page embeds the latest 100 awards, not the register.
                "coverage": "partial"}

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

    # -----------------------------------------------------------------------
    # Insider transactions (the dashboard's embedded arrays)
    # -----------------------------------------------------------------------

    @staticmethod
    def _insider_row(row: Dict[str, Any], source: str,
                     normalize) -> Optional[Dict[str, Any]]:
        """One dashboard record as a flat dict, or None if it is unusable."""
        if not isinstance(row, dict):
            return None
        shares = row.get("transactionShares")
        price = row.get("transactionPricePerShare")
        value = row.get("transactionValue")
        worth = row.get("netWorthEstimate")
        return {
            "ticker": (row.get("issuerTradingSymbol") or "").strip() or None,
            "owner": (row.get("rptOwnerName") or "").strip() or None,
            "title": (row.get("officerTitle") or "").strip() or None,
            "tx_type": (row.get("transactionCode") or "").strip() or None,
            "shares": shares if isinstance(shares, (int, float)) else None,
            "price_per_share": (price if isinstance(price, (int, float))
                                else None),
            "value_usd": (value if isinstance(value, (int, float))
                          else None),
            "transaction_date": normalize(row.get("transactionDate") or ""),
            "filed_date": normalize(
                (row.get("fileDate") or "").split(" (")[0]),
            "insider_cik": row.get("rptOwnerCik"),
            # "-" is the site's stand-in for "not disclosed", not a number.
            "net_worth_estimate": (worth if isinstance(worth, (int, float))
                                   else None),
            "list": source,
        }

    def get_insider_trades(self, *, ticker: Optional[str] = None,
                           owner: Optional[str] = None,
                           tx_type: Optional[str] = None,
                           min_value: Optional[float] = None,
                           days: Optional[int] = None,
                           limit: Optional[int] = None
                           ) -> Dict[str, Any]:
        """Corporate insider purchases and sales from `/insiders/`.

        The page embeds two 1,000-row arrays: the latest filings and the
        month's largest. They overlap by about 165 rows, so the union is
        ~1,824 distinct transactions -- roughly twice what either array
        holds on its own. Rows are deduplicated on owner, ticker, date,
        shares and code, and each row names the list it came from.
        """
        page = self._page("/insiders/")
        recent = self._extract_js_array(page, "recentInsiderTransactionsData")
        monthly = self._extract_js_array(page,
                                         "topMonthlyInsiderTransactionsData")

        seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for source, rows in (("recent", recent), ("top_monthly", monthly)):
            for row in rows or []:
                trade = self._insider_row(row, source, self._normalize_date)
                if not trade or not trade["ticker"]:
                    continue
                key = (trade["owner"], trade["ticker"],
                       trade["transaction_date"], trade["shares"],
                       trade["tx_type"])
                keep = seen.get(key)
                # Keep the first-seen row but promote it to "both" when the
                # second list carries it too, so the overlap stays visible.
                if keep is None:
                    seen[key] = trade
                elif keep["list"] != source:
                    keep["list"] = "both"

        trades = list(seen.values())
        for trade in trades:
            if trade["list"] == "both":
                trade["list"] = "recent+top_monthly"
        total = len(trades)

        if ticker:
            needle = ticker.upper()
            trades = [t for t in trades if (t["ticker"] or "") == needle]
        if owner:
            needle = owner.lower()
            trades = [t for t in trades
                      if needle in (t["owner"] or "").lower()]
        if tx_type:
            needle = tx_type.lower()
            trades = [t for t in trades
                      if needle in (t["tx_type"] or "").lower()]
        if min_value is not None:
            trades = [t for t in trades
                      if (t["value_usd"] or 0) >= float(min_value)]
        if days is not None:
            cutoff = (_dt.date.today()
                      - _dt.timedelta(days=int(days))).isoformat()
            trades = [t for t in trades
                      if (t["transaction_date"] or "") >= cutoff]

        matching = len(trades)
        total_value = sum(t["value_usd"] or 0 for t in trades)
        # Two rows in the union carry a transaction date years after the
        # filing date (TMUS, filed 2024-05-28, dated 2028-05-24). They are
        # kept -- dropping rows silently is worse -- but flagged, and they
        # do not get to define the date range a caller reads as freshness.
        today = _dt.date.today().isoformat()
        for trade in trades:
            if (trade["transaction_date"] or "") > today:
                trade["date_anomaly"] = "transaction_date_after_today"
        anomalous = sum(1 for t in trades if "date_anomaly" in t)
        clean = sorted((t["transaction_date"] for t in trades
                        if t["transaction_date"]
                        and "date_anomaly" not in t), reverse=True)
        # Newest first, with the flagged rows last rather than on top.
        trades = (sorted([t for t in trades if "date_anomaly" not in t],
                         key=lambda t: t["transaction_date"] or "",
                         reverse=True)
                  + [t for t in trades if "date_anomaly" in t])
        if limit is not None and limit > 0:
            trades = trades[:limit]
        return {
            "trades": trades,
            "total_available": total,
            "returned": len(trades),
            "matching": matching,
            "matching_value_usd": round(total_value, 2),
            "date_range": {"newest": clean[0] if clean else None,
                           "oldest": clean[-1] if clean else None},
            "anomalous_rows": anomalous,
            "coverage": "partial",
        }

    # -----------------------------------------------------------------------
    # Congressional net worth (embedded on the congress dashboard)
    # -----------------------------------------------------------------------

    def get_congress_net_worth(self, *, name: Optional[str] = None,
                               chamber: Optional[str] = None,
                               party: Optional[str] = None,
                               state: Optional[str] = None,
                               min_net_worth: Optional[float] = None,
                               limit: Optional[int] = None
                               ) -> Dict[str, Any]:
        """Disclosed net worth per sitting member, ranked.

        `netWorthData` rides on the congress dashboard, so this costs
        nothing extra when that page is already cached. Columns are rank,
        name, net worth, two fields that are zero or float noise on every
        row, bioguide id, portrait, chamber, party and state; the two
        empty ones are dropped rather than passed off as measurements.
        """
        page = self._page("/congresstrading/")
        rows = self._extract_js_array(page, "netWorthData")
        members: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, list) or len(row) < 10:
                continue
            worth = row[2]
            members.append({
                "rank": row[0],
                "name": row[1],
                "net_worth_usd": (round(float(worth), 2)
                                  if isinstance(worth, (int, float)) else None),
                "bioguide_id": row[5],
                "chamber": row[7],
                "party": row[8],
                "state": row[9],
            })
        total = len(members)

        if name:
            needle = name.lower()
            members = [m for m in members
                       if needle in (m["name"] or "").lower()]
        if chamber:
            members = [m for m in members
                       if (m["chamber"] or "").lower() == chamber.lower()]
        if party:
            members = [m for m in members
                       if (m["party"] or "").lower() == party.lower()]
        if state:
            members = [m for m in members
                       if (m["state"] or "").lower() == state.lower()]
        if min_net_worth is not None:
            members = [m for m in members
                       if (m["net_worth_usd"] or 0) >= float(min_net_worth)]

        members.sort(key=lambda m: -(m["net_worth_usd"] or 0))
        matching = len(members)
        if limit is not None and limit > 0:
            members = members[:limit]
        return {
            "members": members,
            "total_members": total,
            "matching": matching,
            "returned": len(members),
            "coverage": "full",
        }

    # -----------------------------------------------------------------------
    # Bills (the legislation search index)
    # -----------------------------------------------------------------------

    def get_bills(self, *, query: Optional[str] = None,
                  bill_type: Optional[str] = None,
                  congress: Optional[str] = None,
                  limit: Optional[int] = None,
                  offset: int = 0) -> Dict[str, Any]:
        """Bills from `/bills/`, which embeds the whole 119th/118th index.

        The array is ~33,000 rows of {congress, number, type, clean_title}:
        a title index, not the bill text. `query` matches the title
        case-insensitively, which is how a theme is turned into a list of
        bill numbers to follow up on congress.gov.
        """
        page = self._page("/bills/")
        rows = self._extract_js_array(page, "searchBills")
        bills: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict) or not row.get("clean_title"):
                continue
            bills.append({
                "congress": str(row.get("congress") or ""),
                "number": row.get("number"),
                "type": row.get("type"),
                "title": row.get("clean_title"),
            })
        total = len(bills)

        if query:
            needle = query.lower()
            bills = [b for b in bills if needle in (b["title"] or "").lower()]
        if bill_type:
            needle = bill_type.lower()
            bills = [b for b in bills
                     if (b["type"] or "").lower() == needle]
        if congress:
            needle = str(congress)
            bills = [b for b in bills if b["congress"] == needle]

        matching = len(bills)
        offset = max(int(offset or 0), 0)
        window = bills[offset:]
        if limit is not None and limit > 0:
            window = window[:limit]
        return {
            "bills": window,
            "total_indexed": total,
            "matching": matching,
            "offset": offset,
            "returned": len(window),
            "more": offset + len(window) < matching,
            "coverage": "full",
        }

    # -----------------------------------------------------------------------
    # Corporate election contributions
    # -----------------------------------------------------------------------

    _MONEY = re.compile(r"-?\$?\s*([\d,]+(?:\.\d+)?)")
    _SHARE = re.compile(r"\(([\d.]+)\s*%\)")
    _TICKER_PREFIX = re.compile(r"^([A-Z][A-Z0-9.\-]{0,6})\s+(\S.*)$")

    @classmethod
    def _money(cls, raw: str) -> Optional[float]:
        """`'$4,455,171'` -> 4455171.0. Anything unparseable is None."""
        m = cls._MONEY.search(raw or "")
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None

    @classmethod
    def _money_and_share(cls, raw: str
                         ) -> Tuple[Optional[float], Optional[float]]:
        """`'$2,006,000 (45.20%)'` -> (2006000.0, 45.2)."""
        m = cls._SHARE.search(raw or "")
        return cls._money(raw), (float(m.group(1)) if m else None)

    @classmethod
    def _ticker_and_name(cls, cell: str) -> Tuple[Optional[str], str]:
        """`'HON Honeywell International Inc'` -> ('HON', 'Honeywell ...').

        The first column is a ticker and a company name in one cell. A cell
        whose leading word is not shaped like a ticker keeps its full text as
        the name rather than guessing.
        """
        m = cls._TICKER_PREFIX.match((cell or "").strip())
        if m:
            return m.group(1), m.group(2).strip()
        return None, (cell or "").strip()

    def get_election_contributions(self, *, ticker: Optional[str] = None,
                                   source: Optional[str] = None,
                                   leans: Optional[str] = None,
                                   min_total: Optional[float] = None,
                                   limit: Optional[int] = None
                                   ) -> Dict[str, Any]:
        """Corporate PAC and employee donations per company, current cycle.

        `/election-contributions/` renders two tables of 100 rows each, both
        for the 2026 cycle: company PAC donations and company employee
        donations. They are separate money — a company's PAC and its
        employees file separately — so a ticker may appear once in each, and
        every row names which table it came from. Each row carries the total
        and the Democratic/Republican split with its share.
        """
        page = self._page("/election-contributions/")
        pac_at = page.find("Company PAC Donations")
        emp_at = page.find("Company Employee Donations")
        labelled: List[Tuple[str, List[List[str]]]] = []
        if 0 <= pac_at < emp_at:
            # The headings are what name each table; take the table under
            # each one instead of trusting document order.
            for kind, chunk in (("pac", page[pac_at:emp_at]),
                                ("employee", page[emp_at:])):
                tables = self._parse_tables(chunk)
                if tables:
                    labelled.append((kind, tables[0]))
        else:
            # Headings gone: the page's order is PAC then employee.
            for kind, table in zip(("pac", "employee"),
                                   self._parse_tables(page)):
                labelled.append((kind, table))

        contributions: List[Dict[str, Any]] = []
        by_source: Dict[str, int] = {}
        for kind, table in labelled:
            if len(table) < 2 or "ticker" not in " ".join(table[0]).lower():
                continue
            for row in table[1:]:
                if len(row) < 4:
                    continue
                symbol, company = self._ticker_and_name(row[0])
                democrat, democrat_pct = self._money_and_share(row[2])
                republican, republican_pct = self._money_and_share(row[3])
                contributions.append({
                    "ticker": symbol,
                    "company": company,
                    "source": kind,
                    "total_usd": self._money(row[1]),
                    "democrat_usd": democrat,
                    "democrat_pct": democrat_pct,
                    "republican_usd": republican,
                    "republican_pct": republican_pct,
                    "cycle": row[4] if len(row) > 4 else None,
                })
            by_source[kind] = by_source.get(kind, 0) + len(table) - 1
        total = len(contributions)

        if ticker:
            needle = ticker.upper()
            contributions = [c for c in contributions
                             if (c["ticker"] or "") == needle]
        if source:
            needle = source.lower()
            contributions = [c for c in contributions
                             if (c["source"] or "") == needle]
        if min_total is not None:
            contributions = [c for c in contributions
                             if (c["total_usd"] or 0) >= float(min_total)]
        if leans:
            needle = leans.lower()
            if needle.startswith("d"):
                contributions = [c for c in contributions
                                 if (c["democrat_pct"] or 0)
                                 > (c["republican_pct"] or 0)]
            elif needle.startswith("r"):
                contributions = [c for c in contributions
                                 if (c["republican_pct"] or 0)
                                 > (c["democrat_pct"] or 0)]

        contributions.sort(key=lambda c: -(c["total_usd"] or 0))
        matching = len(contributions)
        if limit is not None and limit > 0:
            contributions = contributions[:limit]
        return {
            "contributions": contributions,
            "total_embedded": total,
            "by_source": by_source,
            "matching": matching,
            "returned": len(contributions),
            # The two tables are the cycle's largest donors, not the register.
            "coverage": "partial",
        }

    # -----------------------------------------------------------------------
    # WARN layoff notices (embedded as a JSON script block)
    # -----------------------------------------------------------------------

    # The dashboard's own table class reads rows positionally against this
    # map, which is the only statement of what each column means.
    _LAYOFF_FIELDS = {
        "employer": 0, "ticker": 1, "ticker_match": 2, "city": 3,
        "state": 4, "kind": 5, "temporary": 6, "workers": 7,
        "notice_date": 8, "date_source": 9, "effective_date": 10,
        "raw_type": 11, "county": 12, "industry": 13,
    }

    def get_layoffs(self, *, ticker: Optional[str] = None,
                    state: Optional[str] = None,
                    kind: Optional[str] = None,
                    industry: Optional[str] = None,
                    min_workers: Optional[int] = None,
                    days: Optional[int] = None,
                    limit: Optional[int] = None) -> Dict[str, Any]:
        """WARN Act layoff and closure notices, 2,000 rows.

        State workforce agencies publish these; Quiver collects them and
        matches each employer to a ticker. The match is heuristic, so every
        row carries `ticker_match` (exact / fuzzy / search / unmatched) and a
        fuzzy row is a guess about which company filed -- check it before
        attributing 900 layoffs to a ticker. Dates are the state's notice date
        and the employer's effective date, which are not the same thing.
        """
        page = self._page("/layoffs/")
        rows = self._extract_json_script(page, "layoffsRowsData")
        f = self._LAYOFF_FIELDS
        notices: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) <= f["industry"]:
                continue                # a row in another layout; skip it
            workers = row[f["workers"]]
            notices.append({
                "employer": (row[f["employer"]] or "").strip() or None,
                "ticker": (row[f["ticker"]] or "").strip() or None,
                "ticker_match": (row[f["ticker_match"]] or "").strip() or None,
                "city": (row[f["city"]] or "").strip() or None,
                "state": (row[f["state"]] or "").strip() or None,
                "kind": (row[f["kind"]] or "").strip() or None,
                "temporary": bool(row[f["temporary"]]),
                "workers": workers if isinstance(workers, (int, float)) else None,
                "notice_date": row[f["notice_date"]] or None,
                "date_source": (row[f["date_source"]] or "").strip() or None,
                "effective_date": row[f["effective_date"]] or None,
                "raw_type": (row[f["raw_type"]] or "").strip() or None,
                "county": (row[f["county"]] or "").strip() or None,
                "industry": (row[f["industry"]] or "").strip() or None,
            })
        total = len(notices)

        if ticker:
            needle = ticker.upper()
            notices = [n for n in notices if (n["ticker"] or "") == needle]
        if state:
            needle = state.upper()
            notices = [n for n in notices if (n["state"] or "") == needle]
        if kind:
            needle = kind.lower()
            notices = [n for n in notices
                       if (n["kind"] or "").lower() == needle]
        if industry:
            needle = industry.lower()
            notices = [n for n in notices
                       if needle in (n["industry"] or "").lower()]
        if min_workers is not None:
            notices = [n for n in notices
                       if (n["workers"] or 0) >= int(min_workers)]
        if days is not None:
            cutoff = (_dt.date.today()
                      - _dt.timedelta(days=int(days))).isoformat()
            notices = [n for n in notices
                       if (n["notice_date"] or "") >= cutoff]

        matching = len(notices)
        workers_total = sum(n["workers"] or 0 for n in notices)
        mapped = sum(1 for n in notices if n["ticker"])
        dates = sorted((n["notice_date"] for n in notices if n["notice_date"]),
                       reverse=True)
        notices.sort(key=lambda n: ((n["notice_date"] or ""),
                                    n["workers"] or 0), reverse=True)
        if limit is not None and limit > 0:
            notices = notices[:limit]
        return {
            "notices": notices,
            "total_embedded": total,
            "matching": matching,
            "returned": len(notices),
            "workers_matching": workers_total,
            "date_range": {"newest": dates[0] if dates else None,
                           "oldest": dates[-1] if dates else None},
            # Across the matching set, not the returned slice: a caller asking
            # "how many of these notices have a ticker?" is asking about the
            # set, and answering from the first ten rows would be a guess.
            "tickers_mapped": mapped,
            # A 2,000-row window of state notices, not every WARN filing.
            "coverage": "partial",
        }

    # -----------------------------------------------------------------------
    # Revolving door (lobbyists' former government offices)
    # -----------------------------------------------------------------------

    _REVOLVING_FIELDS = {
        "lobbyist": 0, "position": 1, "office": 2, "bioguide_id": 3,
        "member_name": 4, "chamber": 5, "ticker": 6, "ticker_match": 11,
        "client": 7, "firm": 8, "filed_date": 9, "is_new": 10,
        "filing_type": 12, "party": 13, "period": 14,
    }

    def get_revolving_door(self, *, name: Optional[str] = None,
                           chamber: Optional[str] = None,
                           party: Optional[str] = None,
                           client: Optional[str] = None,
                           firm: Optional[str] = None,
                           ticker: Optional[str] = None,
                           days: Optional[int] = None,
                           limit: Optional[int] = None) -> Dict[str, Any]:
        """Lobbyists who previously held government or congressional posts.

        Rows come from lobbying disclosure filings: who registered, the office
        they left, the client paying them and the firm they registered under.
        1,500 rows covering the most recent filings, so this is a window --
        `period` names the quarter each filing covers.
        """
        page = self._page("/revolving-door/")
        rows = self._extract_json_script(page, "revolvingDoorRowsData")
        f = self._REVOLVING_FIELDS
        records: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) <= f["period"]:
                continue
            records.append({
                "lobbyist": (row[f["lobbyist"]] or "").strip() or None,
                "former_position": (row[f["position"]] or "").strip() or None,
                "former_office": (row[f["office"]] or "").strip() or None,
                "bioguide_id": (row[f["bioguide_id"]] or "").strip() or None,
                "member_name": (row[f["member_name"]] or "").strip() or None,
                "chamber": (row[f["chamber"]] or "").strip() or None,
                "party": (row[f["party"]] or "").strip() or None,
                "ticker": (row[f["ticker"]] or "").strip() or None,
                "ticker_match": (row[f["ticker_match"]] or "").strip() or None,
                "client": (row[f["client"]] or "").strip() or None,
                "firm": (row[f["firm"]] or "").strip() or None,
                "filed_date": row[f["filed_date"]] or None,
                "is_new": bool(row[f["is_new"]]),
                "filing_type": (row[f["filing_type"]] or "").strip() or None,
                "period": (row[f["period"]] or "").strip() or None,
            })
        total = len(records)

        if name:
            needle = name.lower()
            records = [r for r in records
                       if needle in (r["lobbyist"] or "").lower()]
        if chamber:
            needle = chamber.lower()
            records = [r for r in records
                       if (r["chamber"] or "").lower() == needle]
        if party:
            needle = party.upper()
            records = [r for r in records
                       if (r["party"] or "").upper() == needle]
        if client:
            needle = client.lower()
            records = [r for r in records
                       if needle in (r["client"] or "").lower()]
        if firm:
            needle = firm.lower()
            records = [r for r in records
                       if needle in (r["firm"] or "").lower()]
        if ticker:
            needle = ticker.upper()
            records = [r for r in records
                       if (r["ticker"] or "") == needle]
        if days is not None:
            cutoff = (_dt.date.today()
                      - _dt.timedelta(days=int(days))).isoformat()
            records = [r for r in records
                       if (r["filed_date"] or "") >= cutoff]

        matching = len(records)
        dates = sorted((r["filed_date"] for r in records if r["filed_date"]),
                       reverse=True)
        records.sort(key=lambda r: r["filed_date"] or "", reverse=True)
        if limit is not None and limit > 0:
            records = records[:limit]
        return {
            "registrations": records,
            "total_embedded": total,
            "matching": matching,
            "returned": len(records),
            "date_range": {"newest": dates[0] if dates else None,
                           "oldest": dates[-1] if dates else None},
            "coverage": "partial",
        }

    # -----------------------------------------------------------------------
    # Government spending (transaction-level JSON endpoint)
    # -----------------------------------------------------------------------

    def get_gov_spending(self, *, ticker: Optional[str] = None,
                         agency: Optional[str] = None,
                         query: Optional[str] = None,
                         min_amount: Optional[float] = None,
                         days: Optional[int] = None,
                         limit: Optional[int] = None) -> Dict[str, Any]:
        """Individual federal transactions, with the recipient's ticker.

        The dashboard renders these from `/get_gov_spending_data/` -- the one
        JSON endpoint on the site that answers anonymously without a session.
        500 transactions, each naming the awarding agency, the recipient, the
        date, the ticker, the amount, the award id and a description. This is
        transaction level: for a company's obligation flow use the spending
        tools in the altdata server instead.
        """
        data = self._json("/get_gov_spending_data/")
        rows = data.get("gov_transactions") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise QuiverError(
                "parse_failed",
                "gov_transactions not found in the spending response")
        transactions: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 5:
                continue
            amount = row[4]
            transactions.append({
                "agency": (row[0] or "").strip() or None,
                "recipient": (row[1] or "").strip() or None,
                "date": row[2] or None,
                "ticker": (row[3] or "").strip() or None,
                "amount_usd": (float(amount)
                               if isinstance(amount, (int, float)) else None),
                "award_id": (row[5] if len(row) > 5 else None),
                "description": (row[6] if len(row) > 6 else None),
            })
        total = len(transactions)

        if ticker:
            needle = ticker.upper()
            transactions = [t for t in transactions
                            if (t["ticker"] or "") == needle]
        if agency:
            needle = agency.lower()
            transactions = [t for t in transactions
                            if needle in (t["agency"] or "").lower()]
        if query:
            needle = query.lower()
            transactions = [t for t in transactions
                            if needle in (t["description"] or "").lower()
                            or needle in (t["recipient"] or "").lower()]
        if min_amount is not None:
            transactions = [t for t in transactions
                            if (t["amount_usd"] or 0) >= float(min_amount)]
        if days is not None:
            cutoff = (_dt.date.today()
                      - _dt.timedelta(days=int(days))).isoformat()
            transactions = [t for t in transactions
                            if (t["date"] or "") >= cutoff]

        matching = len(transactions)
        amount_total = sum(t["amount_usd"] or 0 for t in transactions)
        tickers = len({t["ticker"] for t in transactions if t["ticker"]})
        dates = sorted((t["date"] for t in transactions if t["date"]),
                       reverse=True)
        transactions.sort(key=lambda t: -(t["amount_usd"] or 0))
        if limit is not None and limit > 0:
            transactions = transactions[:limit]
        return {
            "transactions": transactions,
            "total_embedded": total,
            "matching": matching,
            "returned": len(transactions),
            "amount_matching_usd": round(amount_total, 2),
            "date_range": {"newest": dates[0] if dates else None,
                           "oldest": dates[-1] if dates else None},
            # Companies in the matching set, not in the returned slice.
            "tickers": tickers,
            "coverage": "partial",
        }

    # -----------------------------------------------------------------------
    # Patent search (POST-only search form)
    # -----------------------------------------------------------------------

    def get_patent_search(self, query: str,
                          limit: Optional[int] = None) -> Dict[str, Any]:
        """Recently granted patents matching a description or a theme.

        `/patentsearch/` is a form, not a GET route: the query is POSTed with
        the page's CSRF token, exactly as the search box submits it. Results
        come back server-rendered, 100 at a time, each with the assignee's
        ticker, the grant date and the patent title. It is a title/abstract
        search over recent grants -- not a full patent database, and not a
        measure of quality.
        """
        term = (query or "").strip()
        if not term:
            raise QuiverError(
                "parse_failed", "a patent search needs a query term")
        body = self._post(
            "/patentsearch/",
            {"searchterm": term},
            referer="/patentsearch/",
            cache_key=f"/patentsearch/?searchterm={term.lower()}",
        )
        tables = self._parse_tables(body)
        patents: List[Dict[str, Any]] = []
        for table in tables:
            if not table or len(table) < 2:
                continue
            header = " ".join(table[0]).lower()
            if "ticker" not in header or "patent" not in header:
                continue
            for row in table[1:]:
                if len(row) < 3:
                    continue
                patents.append({
                    "ticker": (row[0] or "").strip().upper() or None,
                    "grant_date": self._normalize_us_date(row[1]),
                    "title": (row[2] or "").strip() or None,
                })
        total = len(patents)
        tickers = len({p["ticker"] for p in patents if p["ticker"]})
        if limit is not None and limit > 0:
            patents = patents[:limit]
        return {
            "query": term,
            "patents": patents,
            "total_matched": total,
            "returned": len(patents),
            # Companies in the matching set, not in the returned slice.
            "tickers": tickers,
            # The search returns its first 100 matches and offers no paging.
            "coverage": "partial" if total >= 100 else "full",
        }

    # -----------------------------------------------------------------------
    # 13F fund holdings (a Plotly treemap served per fund)
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_us_date(raw: str) -> Optional[str]:
        """'09/22/2026' -> '2026-09-22'."""
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", (raw or "").strip())
        if m:
            return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        return raw or None

    def get_fund_filers(self) -> List[Dict[str, Any]]:
        """The 13F filers listed on `/sec13f/`, newest first.

        Each row carries a fund name and the site's own relative timestamp
        ("1 day, 8 hours ago"), which is kept as written: it is a freshness
        hint, not a filed date.
        """
        page = self._page("/sec13f/")
        out: List[Dict[str, Any]] = []
        for table in self._parse_tables(page):
            if not table or len(table) < 2:
                continue
            if "fund" not in " ".join(table[0]).lower():
                continue
            for row in table[1:]:
                if len(row) < 2 or not row[0].strip():
                    continue
                out.append({"fund": row[0].strip(),
                            "filed_ago": row[1].strip() or None})
            break
        return out

    def get_fund_holdings(self, fund: str,
                          limit: Optional[int] = None) -> Dict[str, Any]:
        """One institution's reported 13F positions.

        `/get_fund_graph_13f/<fund>` answers with a Plotly treemap: each leaf
        is a ticker, its parent names the sector, and the value is the
        end-of-period position in dollars. The same payload carries the change
        in shares and an estimate of the amount bought, so the answer includes
        what a fund added and cut, not only what it holds.

        Positions are what the last filing reports -- a 45-day-old snapshot of
        a quarter that has already ended.
        """
        name = (fund or "").strip()
        if not name:
            raise QuiverError("parse_failed", "a fund name is required")
        filers = self.get_fund_filers()
        # Resolve the way a reader would: exact, then prefix, then substring.
        match = next((f for f in filers if f["fund"].lower() == name.lower()),
                     None)
        if match is None:
            match = next((f for f in filers
                          if f["fund"].lower().startswith(name.lower())), None)
        if match is None:
            match = next((f for f in filers
                          if name.lower() in f["fund"].lower()), None)
        if match is None:
            raise QuiverError(
                "unknown_fund",
                f"no 13F filer on the dashboard matches '{name}'")
        resolved = match["fund"]

        raw = self._fetch("/get_fund_graph_13f/"
                          + urllib.parse.quote(resolved, safe=""))
        trace = self._extract_plotly_json(raw)
        labels = trace.get("labels") or []
        parents = trace.get("parents") or []
        values = trace.get("values") or []
        extra = trace.get("customdata") or []
        positions: List[Dict[str, Any]] = []
        for i, label in enumerate(labels):
            parent = parents[i] if i < len(parents) else None
            # The treemap's own root and sector rows are not positions.
            if not parent or parent == "" or parent == label:
                continue
            if "/" not in str(parent):
                continue
            sector = str(parent).rsplit("/", 1)[-1]
            row = extra[i] if i < len(extra) else []
            value = values[i] if i < len(values) else None
            shares_change = row[1] if isinstance(row, list) and len(row) > 1 else None
            est_bought = row[2] if isinstance(row, list) and len(row) > 2 else None
            positions.append({
                "ticker": str(label).strip().upper() or None,
                "sector": sector or None,
                "value_usd": (round(float(value), 2)
                              if isinstance(value, (int, float)) else None),
                "shares_change": (shares_change
                                  if isinstance(shares_change, (int, float))
                                  else None),
                "est_bought_usd": (round(float(est_bought), 2)
                                   if isinstance(est_bought, (int, float))
                                   else None),
            })
        total = len(positions)
        invested = sum(p["value_usd"] or 0 for p in positions)
        positions.sort(key=lambda p: -(p["value_usd"] or 0))
        if limit is not None and limit > 0:
            positions = positions[:limit]
        for p in positions:
            if p["value_usd"] and invested:
                p["portfolio_pct"] = round(100 * p["value_usd"] / invested, 2)
        return {
            "fund": resolved,
            "filed_ago": match["filed_ago"],
            "positions": positions,
            "total_positions": total,
            "returned": len(positions),
            "invested_usd": round(invested, 2),
            "sectors": len({p["sector"] for p in positions if p["sector"]}),
            "coverage": "full",
        }
