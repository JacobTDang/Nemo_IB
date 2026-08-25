"""The Senate search returns 100 rows a page whatever you ask for.

`length=400` returns 100 and reports `recordsTotal: 234`. The page cap is the
server's, not ours, so a single request silently sees 43% of the record and
reports nothing about the rest -- and every count downstream inherits that,
including the "seen" figure the ingest logs and the coverage a query prints.

This is the same failure the rest of this pipeline exists to avoid: a partial
answer that is indistinguishable from a complete one.
"""
import pytest

from tools.altdata_server import congress_trades as ct
from tools.altdata_server import senate_annual as sa


class _Session:
    """A search endpoint that caps every page at 100 rows, as the real one does."""

    PAGE_CAP = 100

    def __init__(self, total, kind="ptr"):
        self.total = total
        self.kind = kind
        self.calls = []
        self.cookies = {}

    def post(self, url, **kwargs):
        data = kwargs.get("data") or {}
        start = int(data.get("start", 0))
        length = min(int(data.get("length", 100)), self.PAGE_CAP)
        self.calls.append((start, length))
        rows = []
        for i in range(start, min(start + length, self.total)):
            link = (f'<a href="/search/view/{self.kind}/'
                    f'{i:08d}-0000-0000-0000-000000000000/">'
                    f'Annual Report for CY 2025</a>')
            rows.append(["First", f"Last{i}", "Office (Senator)", link,
                         "05/15/2026"])
        return _Response({"recordsTotal": self.total, "data": rows})

    def get(self, url, **kwargs):  # pragma: no cover - unused here
        raise AssertionError("no GET expected")


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.parametrize("search,kind", [
    (ct.search_senate_ptrs, "ptr"),
    (sa.search_senate_annuals, "annual"),
])
def test_every_page_is_followed(search, kind):
    session = _Session(total=234, kind=kind)
    results = search(session, "01/01/2025")

    assert len(results) == 234, (
        f"got {len(results)} of 234; the server caps a page at 100 and only "
        f"the first was read")
    assert len(session.calls) >= 3, (
        f"made {len(session.calls)} request(s); 234 rows cannot arrive in "
        f"fewer than three pages")


@pytest.mark.parametrize("search,kind", [
    (ct.search_senate_ptrs, "ptr"),
    (sa.search_senate_annuals, "annual"),
])
def test_an_explicit_limit_is_still_honoured(search, kind):
    """A caller asking for 40 gets 40, and only pays for one page."""
    session = _Session(total=234, kind=kind)
    results = search(session, "01/01/2025", limit=40)

    assert len(results) == 40
    assert len(session.calls) == 1


@pytest.mark.parametrize("search,kind", [
    (ct.search_senate_ptrs, "ptr"),
    (sa.search_senate_annuals, "annual"),
])
def test_a_short_result_set_stops_early(search, kind):
    session = _Session(total=12, kind=kind)
    results = search(session, "01/01/2025")

    assert len(results) == 12
    assert len(session.calls) == 1, "paged past the end of the result set"


def test_pagination_does_not_loop_forever_on_a_stuck_page():
    """A server that stops returning rows must end the walk, not spin."""
    class _Stuck(_Session):
        def post(self, url, **kwargs):
            self.calls.append(("stuck", 0))
            return _Response({"recordsTotal": 500, "data": []})

    session = _Stuck(total=500)
    results = ct.search_senate_ptrs(session, "01/01/2025")
    assert results == []
    assert len(session.calls) < 10, "kept requesting empty pages"
