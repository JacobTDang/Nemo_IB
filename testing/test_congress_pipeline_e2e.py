"""The whole pipeline, end to end: search -> fetch -> parse -> store -> tool.

Every other congress test pins one seam. This one runs all of them together
with only the network replaced, using the real search, the real parsers and
the real SQLite store, and finishes at the JSON an MCP client actually
receives. It is the test that catches the breaks the unit tests cannot see:
a parser whose output shape drifts from what the store writes, a store column
the query never selects, a query key the tool does not pass through.

The fixtures are verbatim captures -- a House PTR's extracted PDF text, a
Senate annual report's HTML, and the paginated JSON the Senate search
returns -- so the assertions are about real filings rather than invented
ones.
"""
import json

import pytest

from tools.altdata_server import congress_store as store
from tools.altdata_server import congress_sync as sync
from tools.altdata_server import congress_queries as q
from tools.altdata_server.congress_trades import parse_house_ptr
from tools.altdata_server.server import AltDataServer

from testing.test_congress_trades import HOUSE_WRAPPED
from testing.test_senate_annual_holdings import REPORT as SENATE_ANNUAL


HOUSE_INDEX = [
    {"last": "Allen", "first": "Richard W.", "filing_type": "P",
     "state_district": "GA12", "filing_date": "01/16/2025",
     "doc_id": "20026537", "year": "2025"},
    # A scan, recognisable from the DocID without being downloaded.
    {"last": "McCaul", "first": "Michael", "filing_type": "P",
     "state_district": "TX10", "filing_date": "02/03/2025",
     "doc_id": "8221263", "year": "2025"},
    # Not a PTR; must not be ingested as one.
    {"last": "Foxx", "first": "Virginia", "filing_type": "O",
     "state_district": "NC05", "filing_date": "05/12/2025",
     "doc_id": "10077643", "year": "2025"},
]


class _SenateResponse:
    def __init__(self, payload=None, text=""):
        self._payload, self.text, self.status_code = payload, text, 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _SenateSession:
    """Answers the search over two pages, then serves the report HTML."""

    def __init__(self):
        self.cookies = {"csrftoken": "t"}
        self.headers = {}
        self.pages = 0

    def post(self, url, **kwargs):
        data = kwargs.get("data") or {}
        start = int(data.get("start", 0))
        rows = []
        if start == 0:
            self.pages += 1
            rows = [["John", "Boozman", "Boozman, John (Senator)",
                     '<a href="/search/view/annual/aaaa-1111/">'
                     'Annual Report for CY 2025 (Amendment 2)</a>',
                     "08/24/2026"]]
        return _SenateResponse({"recordsTotal": 1, "data": rows})

    def get(self, url, **kwargs):
        return _SenateResponse(text=SENATE_ANNUAL)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "e2e.db"))
    monkeypatch.setenv("SEC_EMAIL", "e2e@example.invalid")
    monkeypatch.setattr(sync, "_throttle", lambda: None)
    monkeypatch.setattr(sync, "fetch_house_index",
                        lambda year, session=None: HOUSE_INDEX)

    def house_ptr(doc_id, year, session=None):
        # The real parser, on a real filing's extracted text.
        filing = parse_house_ptr(HOUSE_WRAPPED)
        filing["source_url"] = (f"https://disclosures-clerk.house.gov/"
                                f"public_disc/ptr-pdfs/{year}/{doc_id}.pdf")
        return filing

    monkeypatch.setattr(sync, "fetch_house_ptr", house_ptr)
    monkeypatch.setattr(sync, "senate_session", _SenateSession)

    store.init_schema()
    house = sync.sync_house_ptrs(2025, quiet=True)
    senate = sync.sync_senate_annuals(since="01/01/2026", quiet=True)
    return {"house": house, "senate": senate}


# ------------------------------------------------------------------ ingestion

def test_the_run_reports_what_it_did(pipeline):
    house = pipeline["house"]
    assert house["filings_seen"] == 2, "an annual filing was ingested as a PTR"
    assert house["filings_parsed"] == 1
    assert house["scans"] == 1
    assert house["complete"] is True


def test_a_scan_is_recorded_without_being_downloaded(pipeline):
    coverage = store.coverage("house")
    assert coverage["by_status"]["scanned"] == 1
    assert coverage["complete"] is False, (
        "a store holding an unreadable filing is not complete coverage")


def test_both_chambers_landed_in_one_store(pipeline):
    with store.connect() as conn:
        chambers = {r[0] for r in conn.execute("SELECT DISTINCT chamber FROM filings")}
        txns = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        holds = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
    assert chambers == {"house", "senate"}
    assert txns == 3, f"expected the filing's three transactions, got {txns}"
    assert holds == 7, f"expected seven leaf holdings, got {holds}"


def test_containers_never_reach_the_holdings_table(pipeline):
    """The Senate fixture has two accounts with their contents beneath them."""
    with store.connect() as conn:
        names = {r[0] for r in conn.execute("SELECT asset_name FROM holdings")}
    assert not any("Merrill Lynch Account" == n for n in names)
    assert not any("John Boozman IRA" == n for n in names)


# --------------------------------------------------------------------- values

def test_a_wrapped_bracket_survives_to_the_database(pipeline):
    """$15,001 - / $50,000 wrapped across two PDF lines."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT amount_min, amount_max, owner FROM transactions "
            "WHERE ticker = 'ROL'").fetchone()
    assert row == (15001, 50000, "spouse")


def test_a_cusip_never_lands_in_the_ticker_column(pipeline):
    with store.connect() as conn:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM transactions WHERE ticker IS NOT NULL")]
    assert all(not any(c.isdigit() for c in t) for t in tickers), tickers


def test_open_ended_and_unpriced_holdings_keep_their_meaning(pipeline):
    with store.connect() as conn:
        rows = dict(conn.execute(
            "SELECT ticker, value_max FROM holdings WHERE ticker IS NOT NULL"))
        unpriced = conn.execute(
            "SELECT COUNT(*) FROM holdings "
            "WHERE value_min IS NULL AND value_max IS NULL").fetchone()[0]
    assert rows.get("SBUX") is None, "'Over $50,000,000' was given a ceiling"
    assert unpriced == 1, "the unascertainable holding was priced at zero"


# --------------------------------------------------------------------- queries

def test_a_ticker_query_finds_the_ingested_trade(pipeline):
    result = q.ticker_activity("ROL")
    assert result["transaction_count"] == 1
    assert result["transactions"][0]["member"] == "Richard W. Allen"
    assert result["totals"]["amount_min_total"] == 15001


def test_a_holdings_query_finds_the_ingested_position(pipeline):
    result = q.ticker_holdings("TBLL")
    assert result["holding_count"] == 1
    holding = result["holdings"][0]
    assert holding["member"] == "John Boozman"
    assert holding["as_of"] == "2025-12-31"
    assert holding["value_min"] == 50001


def test_an_absent_ticker_is_qualified_by_coverage(pipeline):
    result = q.ticker_activity("NVDA")
    assert result["transaction_count"] == 0
    assert result["coverage"]["complete"] is False
    assert "not the complete record" in result["note"]


# ----------------------------------------------------------------- MCP surface

async def _call(server, method, args):
    return json.loads((await getattr(server, method)(args))[0].text)


async def test_the_tools_serve_what_was_ingested(pipeline):
    server = AltDataServer()

    trades = await _call(server, "congress_trades", {"ticker": "ROL"})
    trades = trades.get("data", trades)
    assert trades["transaction_count"] == 1
    assert trades.get("store_empty") is not True

    holdings = await _call(server, "congress_holdings", {"ticker": "TBLL"})
    holdings = holdings.get("data", holdings)
    assert holdings["holding_count"] == 1
    assert "not a current position" in holdings["note"].lower() or \
        "during" in holdings["note"].lower()

    coverage = await _call(server, "congress_coverage", {})
    coverage = coverage.get("data", coverage)
    assert coverage["total"] == 3
    assert coverage["by_chamber"]["house"]["by_status"]["scanned"] == 1


async def test_the_leaderboard_reflects_the_ingested_rows(pipeline):
    server = AltDataServer()
    body = await _call(server, "congress_leaderboard", {"kind": "tickers"})
    body = body.get("data", body)
    tickers = {t["ticker"] for t in body["tickers"]}
    assert "ROL" in tickers
    assert all(t["ticker"] for t in body["tickers"]), "a null ticker became a row"


# ------------------------------------------------------------------ resumption

def test_a_second_run_changes_nothing(pipeline, monkeypatch):
    """The pipeline's core promise: re-running is free and non-destructive."""
    with store.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    again = sync.sync_house_ptrs(2025, quiet=True)
    assert again["filings_parsed"] == 0
    assert again["already_held"] == 2

    with store.connect() as conn:
        after = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert after == before, "a re-run duplicated rows"
