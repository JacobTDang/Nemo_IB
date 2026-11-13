"""Reconstructing consensus history from a vendor that sells the history.

Finnhub serves four quarters however many you ask for, so the analyst surprise
waits two years for its eight-quarter window. This vendor sells the window.
That makes the question here narrow and sharp: what does a reconstruction have
to carry so nobody mistakes it for an observation, and what does it have to
refuse rather than guess?

Three things it must not do, and the tests are mostly those:

  * attach a vendor row to a fiscal quarter by arithmetic on the vendor's
    date, when the filer's own period end is on record;
  * write a surprise whose release has not happened yet at the `as_of` it was
    asked about;
  * report `ok` when the payload changed shape and it read nothing.

The live test at the bottom needs `FMP_API_KEY`. There is no key in the
repository and none is needed: everything above runs offline, and a
deployment without the key reports `no_api_key` and writes nothing.
"""
import json
import os

import pytest

from research import fmp_consensus as fmp
from research import pit_store
from testing._gates import skip_if_provider_unavailable

SKIP_NETWORK = os.getenv("SKIP_NETWORK_TESTS", "0") == "1"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    monkeypatch.setenv(fmp.KEY_ENV, "test-key")
    # A retry that sleeps is a test that takes three seconds to say the same
    # thing twice.
    monkeypatch.setattr(fmp, "RETRY_BACKOFF_SECONDS", 0.0)
    pit_store.init_schema()
    return pit_store


class _Resp:
    def __init__(self, status, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = (json.dumps(payload) if text is None and payload is not None
                     else (text or ""))


class _Fmp:
    """The vendor's endpoint: one GET per symbol, JSON back.

    `payload` is what it answers with, `status` is the HTTP status, and
    `calls` is every request that reached it -- which is how "no key means no
    request" gets asserted rather than assumed.
    """

    def __init__(self, payload, status=200, text=None):
        self.payload, self.status, self._text = payload, status, text
        self.calls = []

    def get(self, url, params=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return _Resp(self.status, self.payload, self._text)

    def close(self):
        pass


# One name's history in the payload shape the vendor documents. The two
# spellings are both live: v3 shipped `actualEarningResult`, the stable path
# ships `actualEPS`, and a key issued today gets the second while a payload
# recorded last year carries the first.
def _v3_rows(*rows):
    return [{"symbol": "AAA", "date": day,
             "actualEarningResult": actual, "estimatedEarningResult": estimate}
            for day, actual, estimate in rows]


def _stable_rows(*rows):
    return [{"symbol": "AAA", "date": day, "actualEPS": actual,
             "estimatedEPS": estimate, "surprise": actual - estimate}
            for day, actual, estimate in rows]


# Two quarters the filer has already reported, as `sue.eps_series` and
# `announcements.for_quarters` would hand them over.
QUARTERS = {
    "2026Q1": {"period_end": "2026-03-31", "known_at": "2026-05-06T00:00:00Z",
               "announced": "2026-05-05"},
    "2026Q2": {"period_end": "2026-06-30", "known_at": "2026-08-06T00:00:00Z",
               "announced": "2026-08-05"},
}


def _series(quarters=QUARTERS):
    return {"success": True, "ticker": "AAA", "cik": "1",
            "quarters": [{"fiscal_period": period, "eps": 1.0,
                          "basis_factor": 1.0, "source": "xbrl", **dates}
                         for period, dates in quarters.items()]}


def _releases(quarters=QUARTERS):
    return {period: {"announced_date": dates["announced"],
                     "period_end": dates["period_end"],
                     "known_at": dates["known_at"],
                     "accession": f"{period}-8k", "fiscal_period": period}
            for period, dates in quarters.items()}


# ------------------------------------------------------------------ the payload

def test_both_field_spellings_read_the_same(store):
    """The vendor renamed the two fields between API versions. A reader that
    knows one spelling reports "this company has never reported" for every
    company the other one serves."""
    old = fmp.fetch_surprises(
        "AAA", client=_Fmp(_v3_rows(("2026-05-05", 1.2, 1.0))))
    new = fmp.fetch_surprises(
        "AAA", client=_Fmp(_stable_rows(("2026-05-05", 1.2, 1.0))))

    assert old == new == [{"announced_date": "2026-05-05", "estimate": 1.0,
                           "actual": 1.2}]


def test_a_row_with_one_leg_is_dropped(store):
    """Half a pair cannot make a surprise. Writing the estimate alone would
    look like coverage this does not have."""
    rows = fmp.fetch_surprises("AAA", client=_Fmp([
        {"date": "2026-05-05", "actualEPS": 1.2, "estimatedEPS": None},
        {"date": "2026-08-05", "actualEPS": 1.1, "estimatedEPS": 0.9},
    ]))

    assert [r["announced_date"] for r in rows] == ["2026-08-05"], rows


def test_a_date_that_is_not_a_date_is_dropped_not_raised(store):
    """Everything downstream compares the date to a release and to `as_of`,
    both as strings. A value that is not a date has to be counted here rather
    than raised from the middle of a run over six hundred names."""
    rows = fmp.fetch_surprises("AAA", client=_Fmp([
        {"date": "2026/05/05", "actualEPS": 1.2, "estimatedEPS": 1.0},
        {"date": "2026-08-05", "actualEPS": 1.1, "estimatedEPS": 0.9},
    ]))

    assert [r["announced_date"] for r in rows] == ["2026-08-05"], rows


def test_an_empty_list_is_a_name_with_no_history(store):
    assert fmp.fetch_surprises("AAA", client=_Fmp([])) == []


def test_the_history_comes_back_oldest_first(store):
    rows = fmp.fetch_surprises("AAA", client=_Fmp(_stable_rows(
        ("2026-08-05", 1.1, 0.9), ("2026-05-05", 1.2, 1.0))))

    assert [r["announced_date"] for r in rows] == ["2026-05-05", "2026-08-05"]


def test_a_payload_nobody_can_read_names_the_keys_it_got(store):
    """The first run with a real key is where a layout change shows up. An
    operator needs the keys that arrived, not "parse failed"."""
    payload = [{"symbol": "AAA", "fiscalDateEnding": "2026-05-05",
                "reportedEPS": "1.2", "estimatedEPS": "1.0"}]

    with pytest.raises(fmp.FmpError) as caught:
        fmp.fetch_surprises("AAA", client=_Fmp(payload))

    assert caught.value.reason == "parse_failed"
    assert "reportedEPS" in caught.value.message, caught.value.message
    assert "estimatedEPS" in caught.value.message, caught.value.message


def test_an_object_answer_is_a_refusal_not_an_empty_history(store):
    """A quota error arrives as a 200 with an object. Reading it as "no
    history" would say every company in the universe has never reported."""
    with pytest.raises(fmp.FmpError) as caught:
        fmp.fetch_surprises("AAA", client=_Fmp(
            {"Error Message": "Limit Reach. Please upgrade to get access."}))

    assert caught.value.reason == "bad_request"
    assert "Limit Reach" in caught.value.message, caught.value.message


def test_a_refused_key_says_the_free_tier_does_not_cover_it(store):
    with pytest.raises(fmp.FmpError) as caught:
        fmp.fetch_surprises("AAA", client=_Fmp([], status=401))

    assert caught.value.reason == "bad_request"
    assert "free tier" in caught.value.message, caught.value.message


def test_a_rate_limited_answer_is_named(store):
    with pytest.raises(fmp.FmpError) as caught:
        fmp.fetch_surprises("AAA", client=_Fmp([], status=429))

    assert caught.value.reason == "rate_limited", caught.value


def test_a_server_error_is_retried_and_then_reported(store):
    api = _Fmp([], status=503)

    with pytest.raises(fmp.FmpError) as caught:
        fmp.fetch_surprises("AAA", client=api)

    assert caught.value.reason == "provider_unavailable"
    assert len(api.calls) == fmp.RETRIES + 1, api.calls


def test_no_key_means_no_request(monkeypatch):
    """The vendor rejects `demo`, so a placeholder turns "not configured" into
    "failed". Unconfigured is `no_api_key` and it must not spend a request."""
    monkeypatch.delenv(fmp.KEY_ENV, raising=False)

    class _NeverAsked:
        def get(self, url, params=None):  # pragma: no cover - must not run
            raise AssertionError("the module called a vendor it has no key for")

    with pytest.raises(fmp.FmpError) as caught:
        fmp.fetch_surprises("AAA", client=_NeverAsked())

    assert caught.value.reason == "no_api_key"
    assert not fmp.available()


# ----------------------------------------------------------------- the quarters

def test_a_row_lands_on_the_quarter_whose_release_it_is(store):
    api = _Fmp(_stable_rows(("2026-05-05", 1.2, 1.0)))
    result = fmp.record("AAA", "2026-07-01", client=api, series=_series(),
                        releases=_releases())

    assert result["status"] == "ok", result
    assert result["written"] == 2, result
    row = store.consensus_as_of("AAA", "2026Q1", as_of="2026-04-01")
    assert row["eps_estimate"] == 1.0, row
    assert row["source"] == "fmp", row
    # The estimate is stamped at the filer's own period end, which is before
    # the print by construction.
    assert row["as_of_date"] == "2026-03-31", row
    # And the actual at the release, because it had not happened before then.
    assert store.actual_as_of("AAA", "2026Q1", "2026-05-05") == 1.2
    assert store.actual_as_of("AAA", "2026Q1", "2026-05-04") is None


def test_a_vendor_date_a_day_off_the_release_is_the_same_print(store):
    """The two sources read the same 8-K, so they agree to the day. A holiday
    rollover or a vendor dating a release in its own timezone moves it by one,
    and a quarter is ninety days away, so a week cannot reach the next one."""
    result = fmp.record("AAA", "2026-07-01",
                        client=_Fmp(_stable_rows(("2026-05-04", 1.2, 1.0))),
                        series=_series(), releases=_releases())

    assert result["written"] == 2, result


def test_a_row_that_matches_no_release_is_counted_not_attached(store):
    """Attaching it to the nearest quarter is how a surprise ends up in a
    window it does not belong to. It is counted and dropped."""
    result = fmp.record("AAA", "2026-07-01",
                        client=_Fmp(_stable_rows(("2026-01-15", 1.2, 1.0))),
                        series=_series(), releases=_releases())

    assert result["written"] == 0, result
    assert result["unmatched"] == 1, result
    assert result["unmatched_dates"] == ["2026-01-15"], result
    assert store.consensus_as_of("AAA", "2026Q1", as_of="2026-07-01") is None


def test_two_releases_within_a_week_is_ambiguous_and_neither_is_written(store):
    """A quarter is ninety days from its neighbour, so this only happens when
    the record itself is odd. Which one the row is, is not answerable from the
    payload, so it is a refusal rather than a coin flip."""
    quarters = dict(QUARTERS)
    quarters["2026Q2"] = {**quarters["2026Q2"], "announced": "2026-05-08"}
    result = fmp.record("AAA", "2026-09-01",
                        client=_Fmp(_stable_rows(("2026-05-06", 1.2, 1.0))),
                        series=_series(quarters), releases=_releases(quarters))

    assert result["written"] == 0, result
    assert result["ambiguous"] == 1, result


def test_a_surprise_that_has_not_been_released_is_not_written(store):
    """The actual of a print that has not happened is the future. Replaying
    last quarter may not learn this quarter's number."""
    result = fmp.record("AAA", "2026-06-01",
                        client=_Fmp(_stable_rows(
                            ("2026-05-05", 1.2, 1.0), ("2026-08-05", 1.1, 0.9))),
                        series=_series(), releases=_releases())

    assert result["written"] == 2, result
    assert result["future"] == 1, result
    assert store.actual_as_of("AAA", "2026Q2", "2026-09-01") is None


def test_a_quarter_already_on_record_is_never_restated(store):
    """A row that was watched outranks a reconstruction of it, and
    INSERT OR IGNORE would leave the two disagreeing about which to prefer."""
    store.record_consensus("2026-03-31", "AAA", "2026Q1", eps_estimate=0.97,
                           recorded_at="2026-03-31T21:00:00Z")
    store.record_consensus("2026-05-05", "AAA", "2026Q1", eps_estimate=0.97,
                           eps_actual=1.21, recorded_at="2026-05-05T21:00:00Z")

    result = fmp.record("AAA", "2026-07-01",
                        client=_Fmp(_stable_rows(("2026-05-05", 1.2, 1.0))),
                        series=_series(), releases=_releases())

    assert result["written"] == 0, result
    assert result["already"] == 1, result
    assert store.actual_as_of("AAA", "2026Q1", "2026-07-01") == 1.21
    row = store.consensus_as_of("AAA", "2026Q1", as_of="2026-04-01")
    assert row["eps_estimate"] == 0.97, row
    assert row["source"] == "recorded", row


def test_a_name_with_no_eps_series_says_which_kind_of_empty_it_is(store):
    result = fmp.record("AAA", "2026-07-01", client=_Fmp([]),
                        series={"success": False, "error": "no US-GAAP EPS"},
                        releases={})

    assert result["status"] == "failed", result
    assert result["reason"] == "no US-GAAP EPS", result
    assert result["written"] == 0, result


# ---------------------------------------------------------------------- the job

def test_seeding_without_a_key_writes_nothing_and_says_so(store, monkeypatch):
    monkeypatch.delenv(fmp.KEY_ENV, raising=False)

    result = fmp.seed(["AAA", "BBB"], as_of="2026-07-01")

    assert result["status"] == "skipped", result
    assert result["reason"] == "no_api_key", result
    assert result["written"] == 0, result
    assert not store.has_consensus_history("2026-07-01")


def test_the_entry_point_exits_zero_when_the_vendor_is_not_configured(
        store, monkeypatch, capsys):
    """An unconfigured paid vendor is the expected state on a deployment that
    has not bought one. It is not a failure, and cron must not mail it."""
    monkeypatch.delenv(fmp.KEY_ENV, raising=False)

    assert fmp.main(["--tickers", "AAA", "--as-of", "2026-07-01"]) == 0
    assert "no_api_key" in capsys.readouterr().out
    # It still logged a run, because a night with no row is a night nobody can
    # tell from the job not having been scheduled.
    assert store.has_consensus_history("2026-07-01") is False
    with store.connect() as conn:
        run = conn.execute(
            "SELECT * FROM run_log WHERE job = 'fmp_consensus'").fetchone()
    assert run is not None and run["status"] == "ok", dict(run or {})


def test_a_run_whose_names_all_failed_exits_non_zero(store, monkeypatch):
    monkeypatch.setattr(fmp, "record", lambda ticker, as_of=None, **kw: {
        "ticker": ticker, "status": "failed", "reason": "parse_failed",
        "error": "the payload changed", "written": 0})

    assert fmp.main(["--tickers", "AAA", "--as-of", "2026-07-01"]) == 1


# ------------------------------------------------------------------- live

@pytest.mark.network
@pytest.mark.skipif(SKIP_NETWORK, reason=(
    "SKIP_NETWORK_TESTS=1: this test calls FMP's API with a paid key."))
def test_the_vendor_answers_with_a_history_this_client_can_read(store):
    """The shape, not the numbers: rows come back, oldest first, and the
    fields read. Skips when there is no key rather than failing, because a
    machine without one is the ordinary case here."""
    if not fmp.available():
        pytest.skip(f"{fmp.KEY_ENV} is not set")
    try:
        rows = fmp.fetch_surprises("AAPL")
    except fmp.FmpError as exc:
        skip_if_provider_unavailable(
            {"reason": exc.reason, "error": exc.message}, "FMP")
        raise

    assert len(rows) > 4, rows
    assert rows == sorted(rows, key=lambda r: r["announced_date"])
    latest = rows[-1]
    assert latest["announced_date"] < fmp._today(), latest
    assert isinstance(latest["estimate"], float), latest
    assert isinstance(latest["actual"], float), latest
