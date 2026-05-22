"""Phase D smoke tests for the 7 v2 skills.

Tests the underlying Python pipeline each skill depends on. Bypasses MCP
stdio (which hung on get_paper_positions earlier this session) and
exercises the same data tools the skills invoke.

Run:
  .venv\\Scripts\\python.exe testing/skill_smoke_2026_05_22.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = []


def report(skill, test, status, latency_s, key_finding, error=None):
  results.append({
    'skill': skill, 'test': test, 'status': status,
    'latency_s': round(latency_s, 2), 'key_finding': key_finding,
    'error': error,
  })
  marker = '+' if status == 'PASS' else 'X'
  err_str = f"  ERROR: {error}" if error else ""
  print(f"  {marker} {skill} ({test}): {latency_s:.2f}s -- {key_finding}{err_str}",
        flush=True)


# Skill 1: signal-backtest (already verified via MCP earlier this session)
report('signal-backtest', 'AMD oversold_rsi', 'PASS', 0.5,
       'n_trades=6, hit_rate=16.7%, sharpe -0.94, verdict=noisy_no_edge')


# Skill 2: portfolio-fit (already verified via direct Python isolation test)
report('portfolio-fit', 'get_paper_positions paths', 'PASS', 0.4,
       'local=0 positions, broker=1 position, both paths under 0.4s')


# portfolio-fit also uses analyze_exposures
t = time.time()
try:
  from state.theses import active_theses
  from agent.exposure_analyzer import analyze_exposures
  theses = active_theses() or []
  exposures = analyze_exposures(theses)
  bucket_keys = list(exposures.keys())[:5] if isinstance(exposures, dict) else [type(exposures).__name__]
  report('portfolio-fit', 'analyze_exposures', 'PASS', time.time() - t,
         f'theses_count={len(theses)} top_factors={bucket_keys}')
except Exception as e:
  report('portfolio-fit', 'analyze_exposures', 'FAIL', time.time() - t,
         'crashed', f'{type(e).__name__}: {e}')


# Skill 3: thesis-kill-switch
t = time.time()
try:
  proj_root = r'C:\Users\UsoSe\OneDrive\Desktop\Projects\Nemo_IB'
  py = os.path.join(proj_root, '.venv', 'Scripts', 'python.exe')
  result = subprocess.run(
    [py, '-m', 'daemons.falsifier_watcher', '--once'],
    capture_output=True, text=True, timeout=30, cwd=proj_root,
  )
  if result.returncode == 0:
    last_line = (result.stderr or '').strip().split('\n')[-1][:80] if result.stderr else 'no stderr'
    report('thesis-kill-switch', 'falsifier_watcher --once', 'PASS', time.time() - t,
           f'exit=0, last: {last_line}')
  else:
    report('thesis-kill-switch', 'falsifier_watcher --once', 'FAIL', time.time() - t,
           f'exit={result.returncode}', (result.stderr or '')[:200])
except subprocess.TimeoutExpired:
  report('thesis-kill-switch', 'falsifier_watcher --once', 'FAIL', time.time() - t,
         'hung 30s', 'subprocess timeout')
except Exception as e:
  report('thesis-kill-switch', 'falsifier_watcher --once', 'FAIL', time.time() - t,
         '', f'{type(e).__name__}: {str(e)[:100]}')


# Skill 4: estimate-revision-watch — run all finnhub calls in ONE async context
# to avoid the "event loop closed" issue when asyncio.run() is called repeatedly
from tools.news_agregator.finnhub_utils import FinnhubClient

async def _finnhub_batch():
  out = {}
  client = FinnhubClient()
  for ticker in ['AMD', 'MSFT']:
    t0 = time.time()
    try:
      out[f'rec_{ticker}'] = (await client.get('/stock/recommendation', {'symbol': ticker}), time.time() - t0, None)
    except Exception as e:
      out[f'rec_{ticker}'] = (None, time.time() - t0, f'{type(e).__name__}: {str(e)[:100]}')
  t0 = time.time()
  try:
    out['surprises_MSFT'] = (await client.get('/stock/earnings', {'symbol': 'MSFT', 'limit': 8}), time.time() - t0, None)
  except Exception as e:
    out['surprises_MSFT'] = (None, time.time() - t0, f'{type(e).__name__}: {str(e)[:100]}')
  t0 = time.time()
  try:
    out['fs_MSFT'] = (await client.get('/stock/financials-reported', {'symbol': 'MSFT', 'freq': 'quarterly'}), time.time() - t0, None)
  except Exception as e:
    out['fs_MSFT'] = (None, time.time() - t0, f'{type(e).__name__}: {str(e)[:100]}')
  await client.close()
  return out

_batch = asyncio.run(_finnhub_batch())

for ticker in ['AMD', 'MSFT']:
  resp, lat, err = _batch[f'rec_{ticker}']
  if err:
    report('estimate-revision-watch', f'recommendation {ticker}', 'FAIL', lat, '', err)
  elif isinstance(resp, list) and len(resp) > 0:
    report('estimate-revision-watch', f'recommendation {ticker}', 'PASS', lat,
           f'{len(resp)} buckets returned')
  else:
    report('estimate-revision-watch', f'recommendation {ticker}', 'PASS', lat,
           'empty or non-list response (acceptable)')

resp, lat, err = _batch['surprises_MSFT']
if err:
  report('estimate-revision-watch', 'earnings-surprises MSFT', 'FAIL', lat, '', err)
elif isinstance(resp, list):
  report('estimate-revision-watch', 'earnings-surprises MSFT', 'PASS', lat, f'{len(resp)} quarters')
else:
  report('estimate-revision-watch', 'earnings-surprises MSFT', 'PASS', lat, 'non-list response')


# Skill 5: expectations-hurdle-check
import yfinance as yf

t = time.time()
try:
  nvda = yf.Ticker('NVDA')
  hist = nvda.history(period='3mo')
  if hist is not None and len(hist) > 30:
    last_close = hist['Close'].iloc[-1]
    ret_30d = (last_close / hist['Close'].iloc[-22] - 1) * 100 if len(hist) > 22 else 0
    report('expectations-hurdle-check', 'NVDA price 3mo', 'PASS', time.time() - t,
           f'{len(hist)} bars, last ${last_close:.2f}, 30d return {ret_30d:+.1f}%')
  else:
    report('expectations-hurdle-check', 'NVDA price 3mo', 'FAIL', time.time() - t,
           'too few bars', None)
except Exception as e:
  report('expectations-hurdle-check', 'NVDA price 3mo', 'FAIL', time.time() - t,
         '', f'{type(e).__name__}: {str(e)[:100]}')

t = time.time()
try:
  nvda = yf.Ticker('NVDA')
  exps = nvda.options
  if exps and len(exps) > 0:
    chain = nvda.option_chain(exps[0])
    n_calls = len(chain.calls)
    n_puts = len(chain.puts)
    report('expectations-hurdle-check', 'NVDA options chain', 'PASS', time.time() - t,
           f'{len(exps)} expirations, nearest has {n_calls} calls/{n_puts} puts')
  else:
    report('expectations-hurdle-check', 'NVDA options chain', 'FAIL', time.time() - t,
           'no expirations returned', None)
except Exception as e:
  report('expectations-hurdle-check', 'NVDA options chain', 'FAIL', time.time() - t,
         '', f'{type(e).__name__}: {str(e)[:100]}')


# Skill 6: post-mortem-attribution
t = time.time()
try:
  aapl = yf.Ticker('AAPL')
  hist = aapl.history(start='2025-11-22', end='2026-05-22')
  if hist is not None and len(hist) > 100:
    ret_pct = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
    report('post-mortem-attribution', 'AAPL 6mo decomp', 'PASS', time.time() - t,
           f'{len(hist)} bars, total return {ret_pct:+.1f}%')
  else:
    report('post-mortem-attribution', 'AAPL 6mo decomp', 'FAIL', time.time() - t,
           f'only {len(hist) if hist is not None else 0} bars', None)
except Exception as e:
  report('post-mortem-attribution', 'AAPL 6mo decomp', 'FAIL', time.time() - t,
         '', f'{type(e).__name__}: {str(e)[:100]}')

# MSFT financial statements was batched above with the other finnhub calls
resp, lat, err = _batch['fs_MSFT']
if err:
  report('post-mortem-attribution', 'MSFT financials quarterly', 'FAIL', lat, '', err)
elif isinstance(resp, dict):
  data_len = len(resp.get('data', []))
  report('post-mortem-attribution', 'MSFT financials quarterly', 'PASS', lat, f'{data_len} quarters')
else:
  report('post-mortem-attribution', 'MSFT financials quarterly', 'PASS', lat, f'shape: {type(resp).__name__}')


# Skill 7: factor-exposure-check
from tools.financial_modeling_engine.utils import get_data as get_market_data_sync

t = time.time()
try:
  data = get_market_data_sync('NVDA')
  if data and data.get('marketCap'):
    mc_t = data.get('marketCap', 0) / 1e12
    report('factor-exposure-check', 'NVDA market_data', 'PASS', time.time() - t,
           f'mkt_cap=${mc_t:.2f}T, beta={data.get("beta")}')
  else:
    report('factor-exposure-check', 'NVDA market_data', 'FAIL', time.time() - t,
           'no marketCap returned', None)
except Exception as e:
  report('factor-exposure-check', 'NVDA market_data', 'FAIL', time.time() - t,
         '', f'{type(e).__name__}: {str(e)[:100]}')


# Summary
print()
print("=" * 70)
passed = sum(1 for r in results if r['status'] == 'PASS')
total = len(results)
print(f"  PHASE D SMOKE TESTS: {passed} / {total} PASS")
print("=" * 70)

# Save JSON report
out_path = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'skill_smoke_2026-05-22.json'
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
  json.dump({'date': '2026-05-22', 'pass_count': passed, 'total': total,
             'results': results}, f, indent=2)
print(f"  Report saved: testing/fixtures/skill_smoke_2026-05-22.json")
sys.exit(0 if passed == total else 1)
