# Manual cleanup for Nemo IB Python daemons.
#
# Use when the launcher crashed and left orphaned daemons running, or when
# you want to stop the daemons without exiting Claude Code. Reads PIDs from
# db_cache/daemon_pids.txt (populated by nemo-launch.ps1) and stops them.
#
# If the PID file is missing, falls back to killing any Python process
# whose command line includes 'daemons.' — heavy-handed but works when
# the PID file got nuked.

$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile     = Join-Path $projectRoot 'db_cache\daemon_pids.txt'

Write-Host '== Stopping Nemo IB daemons ==' -ForegroundColor Cyan

if (Test-Path $pidFile) {
  Write-Host "Reading PIDs from $pidFile"
  foreach ($line in (Get-Content $pidFile)) {
    if (-not $line) { continue }
    $parts = $line.Split(':')
    if ($parts.Count -lt 2) { continue }
    $name = $parts[0]
    $procId = $parts[1]
    try {
      Stop-Process -Id $procId -Force -ErrorAction Stop
      Write-Host "  - $name (PID $procId) stopped" -ForegroundColor Green
    } catch {
      Write-Host "  - $name (PID $procId) already gone" -ForegroundColor DarkGray
    }
  }
  Remove-Item $pidFile -ErrorAction SilentlyContinue
} else {
  Write-Host 'PID file not found. Scanning for daemon processes by command line...'
  $killed = 0
  $procs = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue
  foreach ($p in $procs) {
    if ($p.CommandLine -match 'daemons\.(edgar_firehose|news_watcher|falsifier_watcher|sentry_triage)') {
      $name = ($p.CommandLine | Select-String -Pattern 'daemons\.(\w+)').Matches.Groups[1].Value
      try {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        Write-Host "  - $name (PID $($p.ProcessId)) stopped" -ForegroundColor Green
        $killed += 1
      } catch {
        Write-Host "  - $name (PID $($p.ProcessId)) not killable" -ForegroundColor DarkGray
      }
    }
  }
  if ($killed -eq 0) {
    Write-Host '  no daemon processes found.' -ForegroundColor DarkGray
  }
}

Write-Host 'Done.' -ForegroundColor Cyan
