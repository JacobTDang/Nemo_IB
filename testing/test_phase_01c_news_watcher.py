"""Phase 1c: news_watcher daemon integration with mocked feeds.

Tests the daemon's loop logic end-to-end:
- can it parse a Finnhub response
- does it skip already-seen events (dedup at fetch time)
- does it write classified events to the DB
- does it survive a 500 from the feed
- does it shut down cleanly on the _running flag flipping

Live network tests (1-hour daemon, real Finnhub) are calendar-time gates and
live in the Phase 1 checkpoint runbook, not in CI.
"""
import sys, os
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.events_store import seen, store_event
import daemons.news_watcher as nw


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE source LIKE 'finnhub:%' OR source LIKE 'sec:%' OR source LIKE 'finnhub-general:%'")
    conn.commit()
  finally:
    conn.close()


def _install_permissive_limiter():
  """Replace the module-level rate limiter with a permissive one for tests
  that don't care about pacing. Production min_interval is window/max_calls
  = 60/25 = 2.4s, which would block fast tests. Tests that need to assert
  pacing (test_rate_limiter_paces_evenly, test_classifier_rate_limit_enforced)
  construct their own limiter and don't use the singleton."""
  nw._LIMITER = nw.MaterialityRateLimiter(max_calls=10_000, window=1.0)


class FakeMaterialityResult:
  def __init__(self, is_material=True, category='earnings', tickers=None,
               primary='AAPL', signal='bullish', urgency='immediate',
               conf=0.9, reason='test'):
    self.is_material = is_material
    self.category = category
    self.affected_tickers = tickers or ['AAPL']
    self.primary_ticker = primary
    self.directional_signal = signal
    self.urgency = urgency
    self.confidence = conf
    self.one_line_reason = reason


class FakeClassifier:
  """Mock classifier that returns canned results based on headline keywords."""
  def __init__(self):
    self.calls = 0

  def classify(self, headline, summary='', source=''):
    self.calls += 1
    if 'NOISE' in headline:
      return FakeMaterialityResult(is_material=False, category='other',
                                    signal='neutral', urgency='watch')
    if 'CRASH' in headline:
      return None  # simulate parse failure
    if 'TSLA' in headline:
      return FakeMaterialityResult(tickers=['TSLA'], primary='TSLA', signal='bearish')
    return FakeMaterialityResult()


def test_finnhub_loop_one_round_writes_classified_events():
  """One pass through watch_finnhub_company_news should classify + store articles."""
  init_schema(); _clean(); _install_permissive_limiter()
  # Speed the daemon up so the test finishes
  nw.POLL_FINNHUB = 1

  fake_articles = [
    {'headline': 'AAPL Q3 beats consensus', 'summary': 'beat',
     'datetime': 1715000000, 'source': 'reuters', 'url': 'http://x.com/1'},
    {'headline': 'AAPL NOISE: routine coverage', 'summary': 'meh',
     'datetime': 1715000010, 'source': 'cnbc', 'url': 'http://x.com/2'},
    {'headline': 'AAPL CRASH parser', 'summary': 'crash',
     'datetime': 1715000020, 'source': 'wsj', 'url': 'http://x.com/3'},
  ]

  class FakeResp:
    status = 200
    async def json(self):
      return fake_articles
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  class FakeSession:
    def get(self, url, params, timeout):
      return FakeResp()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  with patch('daemons.news_watcher.get_watchlist', return_value=['AAPL']), \
       patch('daemons.news_watcher.FinnhubClient') as MockClient, \
       patch('aiohttp.ClientSession', return_value=FakeSession()):
    MockClient.return_value._api_key = 'fake'
    classifier = FakeClassifier()

    async def run_one_round():
      nw._running = True
      task = asyncio.create_task(nw.watch_finnhub_company_news(classifier))
      await asyncio.sleep(0.3)  # let one iteration complete
      nw._running = False
      try:
        await asyncio.wait_for(task, timeout=3)
      except asyncio.TimeoutError:
        task.cancel()

    asyncio.run(run_one_round())

  # Inspect DB: should have 2 stored events (1 material + 1 noise), CRASH skipped
  conn = get_connection()
  try:
    rows = conn.execute(
      "SELECT headline, materiality FROM events WHERE source LIKE 'finnhub:%'"
    ).fetchall()
  finally:
    conn.close()
  headlines = [r['headline'] for r in rows]
  assert any('Q3 beats' in h for h in headlines), f"material event missing. Got: {headlines}"
  assert not any('CRASH' in h for h in headlines), "CRASH event should be skipped on None"
  # Classifier should have been called 3x but only 2 events written (CRASH returned None)
  assert classifier.calls == 3, f"classifier should be called for all 3 unseen articles, got {classifier.calls}"
  print(f"PASS: finnhub loop classified {classifier.calls} articles, "
        f"wrote {len(rows)} (CRASH skipped on None)")


def test_finnhub_loop_dedups_on_second_pass():
  """Same article seen twice should only be classified once."""
  init_schema(); _clean(); _install_permissive_limiter()
  nw.POLL_FINNHUB = 1
  fake_articles = [
    {'headline': 'AAPL repeat article', 'summary': 'dup',
     'datetime': 1715000000, 'source': 'reuters', 'url': 'http://x.com/1'},
  ]

  class FakeResp:
    status = 200
    async def json(self):
      return fake_articles
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  class FakeSession:
    def get(self, url, params, timeout):
      return FakeResp()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  classifier = FakeClassifier()
  with patch('daemons.news_watcher.get_watchlist', return_value=['AAPL']), \
       patch('daemons.news_watcher.FinnhubClient') as MockClient, \
       patch('aiohttp.ClientSession', return_value=FakeSession()):
    MockClient.return_value._api_key = 'fake'

    async def run_two_rounds():
      nw._running = True
      task = asyncio.create_task(nw.watch_finnhub_company_news(classifier))
      await asyncio.sleep(2.5)  # > 1 poll interval, < 2.5
      nw._running = False
      try:
        await asyncio.wait_for(task, timeout=3)
      except asyncio.TimeoutError:
        task.cancel()

    asyncio.run(run_two_rounds())

  # Classifier should have been called ONCE despite being polled twice
  assert classifier.calls == 1, \
    f"classifier should be called exactly once for repeat content, got {classifier.calls}"
  print(f"PASS: dedup at fetch time prevents re-classification (calls={classifier.calls})")


def test_finnhub_loop_recovers_from_http_error():
  """A 500 from the API shouldn't kill the daemon."""
  init_schema(); _clean(); _install_permissive_limiter()
  nw.POLL_FINNHUB = 1

  class FakeResp500:
    status = 500
    async def json(self): return []
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  class FakeSession:
    def get(self, url, params, timeout):
      return FakeResp500()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  classifier = FakeClassifier()
  with patch('daemons.news_watcher.get_watchlist', return_value=['AAPL']), \
       patch('daemons.news_watcher.FinnhubClient') as MockClient, \
       patch('aiohttp.ClientSession', return_value=FakeSession()):
    MockClient.return_value._api_key = 'fake'

    async def run():
      nw._running = True
      task = asyncio.create_task(nw.watch_finnhub_company_news(classifier))
      await asyncio.sleep(1.5)
      nw._running = False
      try:
        await asyncio.wait_for(task, timeout=3)
        return True
      except asyncio.TimeoutError:
        task.cancel()
        return False
      except Exception:
        return False

    result = asyncio.run(run())
  assert result, "daemon crashed on 500 instead of logging and continuing"
  assert classifier.calls == 0, "no classifier calls expected on 500"
  print("PASS: daemon survives HTTP 500 from feed")


def test_finnhub_loop_recovers_from_connection_exception():
  """A connection error should be caught, logged, and the loop continues."""
  init_schema(); _clean(); _install_permissive_limiter()
  nw.POLL_FINNHUB = 1

  class FakeSession:
    def get(self, url, params, timeout):
      raise ConnectionError("DNS failure")
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  classifier = FakeClassifier()
  with patch('daemons.news_watcher.get_watchlist', return_value=['AAPL', 'MSFT']), \
       patch('daemons.news_watcher.FinnhubClient') as MockClient, \
       patch('aiohttp.ClientSession', return_value=FakeSession()):
    MockClient.return_value._api_key = 'fake'

    async def run():
      nw._running = True
      task = asyncio.create_task(nw.watch_finnhub_company_news(classifier))
      await asyncio.sleep(1.5)
      nw._running = False
      try:
        await asyncio.wait_for(task, timeout=3)
        return True
      except asyncio.TimeoutError:
        task.cancel()
        return False
      except Exception:
        return False

    result = asyncio.run(run())
  assert result, "daemon should catch ConnectionError and keep running"
  print("PASS: daemon survives connection exception")


def test_rate_limiter_paces_evenly():
  """With max_calls=10 and window=60s, consecutive acquires must be spaced
  at least 60/10 = 6 seconds apart. Pre-fix the token bucket let bursts
  through then dumped all waiters into a single backoff cycle (thundering
  herd). Even pacing eliminates that."""
  import asyncio as _asyncio
  from daemons.news_watcher import MaterialityRateLimiter

  current = [0.0]
  def clock():
    return current[0]

  async def fake_sleep(seconds):
    current[0] += seconds

  acquire_times = []

  async def run():
    limiter = MaterialityRateLimiter(max_calls=10, window=60.0, clock=clock)
    for _ in range(20):
      await limiter.acquire()
      acquire_times.append(current[0])

  with patch('daemons.news_watcher.asyncio.sleep', fake_sleep):
    _asyncio.run(run())

  assert len(acquire_times) == 20
  min_interval = 60.0 / 10
  # Within ~1e-6 tolerance, every consecutive pair must be at least 6s apart
  gaps = [acquire_times[i] - acquire_times[i - 1] for i in range(1, len(acquire_times))]
  too_close = [g for g in gaps if g < min_interval - 1e-6]
  assert not too_close, \
    f"all gaps must be >= {min_interval}s; violations: {too_close}"
  # And the average gap should be close to min_interval (proves the pacer is
  # not idle-padding more than necessary)
  avg = sum(gaps) / len(gaps)
  assert abs(avg - min_interval) < 0.01, \
    f"avg gap should be ~{min_interval}s, got {avg}"
  print(f"PASS: pacer spaced 20 acquires at exactly {min_interval}s "
        f"(min_gap={min(gaps):.2f}, avg={avg:.2f})")


def test_classifier_rate_limit_enforced():
  """Token bucket throttles classifier to at most max_calls per window seconds.
  Uses a virtual clock + patched asyncio.sleep so the test stays under 3s real
  time even when simulating minutes of throttling."""
  import asyncio as _asyncio
  from daemons.news_watcher import MaterialityRateLimiter

  current = [0.0]
  def clock():
    return current[0]

  async def fake_sleep(seconds):
    current[0] += seconds

  call_times = []

  async def run():
    limiter = MaterialityRateLimiter(max_calls=10, window=60.0, clock=clock)
    for _ in range(15):
      await limiter.acquire()
      call_times.append(current[0])
      current[0] += 1.0  # simulate ~1s of work per call

  with patch('daemons.news_watcher.asyncio.sleep', fake_sleep):
    _asyncio.run(run())

  in_first_minute = sum(1 for t in call_times if t < 60.0)
  assert in_first_minute <= 10, \
    f"max_calls=10 in window=60s but {in_first_minute} calls landed in first 60s"
  assert len(call_times) == 15, "all 15 calls eventually acquired"
  print(f"PASS: rate limiter capped first-minute calls to {in_first_minute} "
        f"(15 total over {call_times[-1]:.0f}s virtual)")


def test_max_per_ticker_caps_articles():
  """Even if Finnhub returns 30 articles for a ticker, only MAX_PER_TICKER
  should be classified (lower load on Groq)."""
  import daemons.news_watcher as nw
  init_schema(); _clean(); _install_permissive_limiter()
  nw.POLL_FINNHUB = 1
  fake_articles = [
    {'headline': f'AAPL article {i}', 'summary': f'b{i}',
     'datetime': 1715000000 + i, 'source': 'reuters', 'url': f'http://x/{i}'}
    for i in range(30)
  ]

  class FakeResp:
    status = 200
    async def json(self): return fake_articles
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
  class FakeSession:
    def get(self, url, params, timeout): return FakeResp()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  classifier = FakeClassifier()
  with patch('daemons.news_watcher.get_watchlist', return_value=['AAPL']), \
       patch('daemons.news_watcher.FinnhubClient') as MockClient, \
       patch('aiohttp.ClientSession', return_value=FakeSession()):
    MockClient.return_value._api_key = 'fake'

    async def run_one_round():
      nw._running = True
      task = asyncio.create_task(nw.watch_finnhub_company_news(classifier))
      await asyncio.sleep(0.4)
      nw._running = False
      try:
        await asyncio.wait_for(task, timeout=3)
      except asyncio.TimeoutError:
        task.cancel()
    asyncio.run(run_one_round())

  assert classifier.calls <= nw.MAX_PER_TICKER, \
    f"classifier called {classifier.calls} times; cap is {nw.MAX_PER_TICKER}"
  print(f"PASS: per-ticker cap honored ({classifier.calls} ≤ {nw.MAX_PER_TICKER})")


def test_shutdown_flag_terminates_loop_promptly():
  """Setting _running=False should make the loop return within ~1s."""
  init_schema(); _clean(); _install_permissive_limiter()
  nw.POLL_FINNHUB = 60  # set high so we know shutdown is what stopped it

  class FakeResp:
    status = 200
    async def json(self): return []
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  class FakeSession:
    def get(self, url, params, timeout):
      return FakeResp()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

  classifier = FakeClassifier()
  import time
  with patch('daemons.news_watcher.get_watchlist', return_value=['AAPL']), \
       patch('daemons.news_watcher.FinnhubClient') as MockClient, \
       patch('aiohttp.ClientSession', return_value=FakeSession()):
    MockClient.return_value._api_key = 'fake'

    async def run():
      nw._running = True
      task = asyncio.create_task(nw.watch_finnhub_company_news(classifier))
      await asyncio.sleep(0.5)
      t0 = time.monotonic()
      nw._running = False
      await task
      elapsed = time.monotonic() - t0
      return elapsed

    elapsed = asyncio.run(run())
  assert elapsed < 2.5, f"shutdown took too long: {elapsed:.2f}s"
  print(f"PASS: shutdown propagates promptly ({elapsed:.2f}s after flag flip)")


if __name__ == "__main__":
  test_finnhub_loop_one_round_writes_classified_events()
  test_finnhub_loop_dedups_on_second_pass()
  test_finnhub_loop_recovers_from_http_error()
  test_finnhub_loop_recovers_from_connection_exception()
  test_rate_limiter_paces_evenly()
  test_classifier_rate_limit_enforced()
  test_max_per_ticker_caps_articles()
  test_shutdown_flag_terminates_loop_promptly()
  _clean()
  print("\nAll Phase 1c news_watcher tests passed.")
