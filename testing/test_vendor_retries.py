"""The retry policy has to match what is recoverable, and it did not.

A cold bootstrap on a clean volume failed on its first run: 1.12M bars written,
10,391 registrants screened, and then

    "consensus": {"status": "failed",
                  "error": "earnings calendar: HTTP 503: <!doctype html> ..."}

The 503 was transient -- five retries twenty seconds later all returned 200,
and every other Finnhub endpoint answered throughout. But `_fetch_bars` retried
three times with backoff while `_fetch_calendar` did not retry at all, and
`FinnhubClient.get` retried only on 429. So the fetch that can be repeated
tomorrow was protected and the one that cannot -- consensus accrues forward
only, and the vendor serves four quarters however many you ask for -- was not.

One blip therefore cost a permanent hole in the unrecoverable series, and the
`&&` in the cron line cost the whole night's decisions with it.
"""
import pytest

from research import daily_job


class _Response:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _Session:
    """A session that answers a scripted sequence of statuses."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        status = self.statuses.pop(0) if self.statuses else 200
        if status == 200:
            return _Response(200, {"earningsCalendar": []})
        return _Response(status, text=f"<html>error {status}</html>")


@pytest.fixture
def client(monkeypatch):
    from tools.news_agregator import finnhub_utils

    monkeypatch.setenv("FINNHUB_API_KEY", "not-a-real-key")
    made = finnhub_utils.FinnhubClient()

    class _NoWait:
        async def acquire(self):
            return None

    made._rate_limiter = _NoWait()
    return made


def _get(client, session, endpoint="/calendar/earnings"):
    import asyncio

    async def go():
        async def session_for():
            return session
        client._get_session = session_for
        return await client.get(endpoint, {})

    return asyncio.run(go())


# --- the client ------------------------------------------------------------

def test_a_transient_503_is_retried_rather_than_returned(client, monkeypatch):
    """The exact failure seen live: one 503, then fine."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep())
    session = _Session([503, 200])

    out = _get(client, session)

    assert "error" not in out, f"a retryable 503 was surfaced as an error: {out}"
    assert out == {"earningsCalendar": []}
    assert session.calls == 2


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_every_5xx_is_treated_as_retryable(client, monkeypatch, status):
    monkeypatch.setattr("asyncio.sleep", _no_sleep())
    session = _Session([status, 200])

    assert "error" not in _get(client, session)
    assert session.calls == 2


def test_a_429_is_still_retried(client, monkeypatch):
    """The behaviour that already existed must survive."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep())
    session = _Session([429, 200])

    assert "error" not in _get(client, session)


def test_a_404_is_not_retried_because_it_will_not_change(client, monkeypatch):
    """Retrying a client error burns the rate limit to get the same answer."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep())
    session = _Session([404, 200])

    out = _get(client, session)

    assert "error" in out and "404" in out["error"]
    assert session.calls == 1


def test_a_persistent_5xx_gives_up_and_says_so(client, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep())
    session = _Session([503] * 10)

    out = _get(client, session)

    assert "error" in out and "503" in out["error"]
    assert session.calls <= 4, "a dead vendor must not be hammered"


def test_the_key_is_never_rendered_in_a_retry_exhausted_error(client,
                                                              monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep())
    out = _get(client, _Session([503] * 10))

    assert "not-a-real-key" not in str(out)


# --- the calendar fetch ----------------------------------------------------

def test_the_calendar_fetch_retries_as_hard_as_the_bars_fetch(monkeypatch):
    """The asymmetry that caused this. Bars can be re-fetched tomorrow;
    consensus cannot, and it was the one with no retries."""
    calls = {"n": 0}

    def flaky(start, end):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("earnings calendar: HTTP 503: <!doctype html>")
        return [{"ticker": "AAA", "fiscal_period": "2026Q1",
                 "eps_estimate": 1.0, "eps_actual": None, "analyst_count": 3}]

    monkeypatch.setattr(daily_job, "_fetch_calendar_once", flaky)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    rows = daily_job._fetch_calendar("2026-03-01", "2026-03-10")

    assert calls["n"] == 3
    assert rows and rows[0]["ticker"] == "AAA"


def test_the_calendar_fetch_gives_up_after_the_same_number_of_attempts(
        monkeypatch):
    def dead(start, end):
        raise RuntimeError("earnings calendar: HTTP 503")

    seen = {"n": 0}

    def counted(start, end):
        seen["n"] += 1
        return dead(start, end)

    monkeypatch.setattr(daily_job, "_fetch_calendar_once", counted)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    with pytest.raises(RuntimeError):
        daily_job._fetch_calendar("2026-03-01", "2026-03-10")

    assert seen["n"] == daily_job.FETCH_RETRIES


def test_the_final_failure_still_names_the_vendor_status(monkeypatch):
    """A run_log entry saying "failed" without saying what the vendor said is
    an alert nobody can act on."""
    monkeypatch.setattr(daily_job, "_fetch_calendar_once",
                        _raises("earnings calendar: HTTP 503: gateway"))
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    with pytest.raises(RuntimeError) as raised:
        daily_job._fetch_calendar("2026-03-01", "2026-03-10")

    assert "503" in str(raised.value)
    assert str(daily_job.FETCH_RETRIES) in str(raised.value)


def _raises(message):
    def go(start, end):
        raise RuntimeError(message)
    return go


def _no_sleep():
    async def go(seconds):
        return None
    return go
