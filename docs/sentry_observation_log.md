# Sentry observation log

Three-ET-day window after Sentry Phases 1-3 shipped to main
(2026-05-22). Goal: see what the system actually does on real
data, capture what's surprising, surface 3-5 concrete tuning
recommendations that feed Phase 5.

No code changes during this window — pure observation. If
something looks obviously broken, file it in `known_issues.md`
and keep going. Threshold tuning happens after Day 3 in a
separate plan.

## How to take the daily snapshot

End of each ET trading day, run this from the repo root:

```powershell
.venv\Scripts\python.exe -c @"
import sqlite3
from datetime import datetime, timezone, timedelta
conn = sqlite3.connect('db_cache/session.db')
conn.row_factory = sqlite3.Row
today_et = (datetime.now(timezone.utc) + timedelta(hours=-5)).strftime('%Y-%m-%d')

print(f'== day {today_et} ==')

print('-- sentry_queue by status --')
for r in conn.execute('SELECT status, COUNT(*) c FROM sentry_queue GROUP BY status'):
  print(f'  {r[\"status\"]:12s} {r[\"c\"]}')

print('-- sentry_evaluation_log today by decision --')
for r in conn.execute(
  \"SELECT decision, COUNT(*) c FROM sentry_evaluation_log \"
  \"WHERE date(evaluated_at) = ? GROUP BY decision\",
  (today_et,)):
  print(f'  {r[\"decision\"]:24s} {r[\"c\"]}')

print('-- sentry_discovery_runs today --')
row = conn.execute('SELECT * FROM sentry_discovery_runs WHERE day=?', (today_et,)).fetchone()
if row:
  print(f'  ran_at={row[\"ran_at\"]}')
  print(f'  catalyst={row[\"catalyst_enqueued\"]} insider={row[\"insider_enqueued\"]} '
        f'activist={row[\"activist_enqueued\"]} theme_flow={row[\"theme_flow_enqueued\"]} '
        f'total={row[\"total_enqueued\"]}')
  if row['errors']:
    print(f'  errors: {row[\"errors\"]}')
else:
  print('  (no row — discovery has not run today)')
conn.close()
"@
Get-ChildItem db_cache\daemon_logs\*.err.log | ForEach-Object {
  $lines = (Get-Content $_.FullName | Measure-Object -Line).Lines
  Write-Host "  $($_.Name): $lines err lines"
}
```

Then append 2-4 sentences below in the day's section. The goal
is qualitative read, not a dashboard.

## Day 1 (2026-05-22)

(observation pending — first end-of-day snapshot)

## Day 2 (2026-05-23)

(observation pending)

## Day 3 (2026-05-24)

(observation pending)

## Day 3 summary — tuning recommendations

(filled in after Day 3)

Format: 3-5 bullet points, each one sentence on the observation +
one sentence on the proposed tuning. Example shape:

- Insider cluster scan returned zero matches across 7 watchlist
  tickers over 3 days. **Tune:** lower the insider buy threshold
  from $100k → $50k, or widen the 30-day window to 60 days.

This summary becomes the input to the Phase 5 tuning plan.
