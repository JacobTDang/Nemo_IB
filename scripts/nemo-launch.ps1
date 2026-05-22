# Single-shot launcher for the Nemo IB workflow.
#
# Brings up the required external service (SearxNG container), starts the
# four always-on Python daemons (edgar_firehose, news_watcher,
# falsifier_watcher, sentry_triage), then hands control to Claude Code in
# the project directory. When Claude Code exits, the daemons are stopped.
#
# Designed for double-click invocation via nemo.bat at the project root.

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$bootstrap   = Join-Path $PSScriptRoot 'start_searxng.ps1'
$healthUrl   = 'http://localhost:8888/healthz'

$pythonExe   = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDir      = Join-Path $projectRoot 'db_cache\daemon_logs'
$pidFile     = Join-Path $projectRoot 'db_cache\daemon_pids.txt'

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# Daemons to start in the background. Each is a Python module run via
# `python -m <module>`. Output goes to db_cache/daemon_logs/<name>.{out,err}.log
# so the user can tail them when debugging.
$daemons = @(
  @{ Name = 'edgar_firehose';    Module = 'daemons.edgar_firehose' },
  @{ Name = 'news_watcher';      Module = 'daemons.news_watcher' },
  @{ Name = 'falsifier_watcher'; Module = 'daemons.falsifier_watcher' },
  @{ Name = 'sentry_triage';     Module = 'daemons.sentry_triage' }
)

function Stop-StalePids {
  # If a previous launch crashed, daemon_pids.txt may have stale entries.
  # Try to stop each — silently ignore PIDs that no longer exist (Windows
  # reuses them quickly, so an existence check isn't enough; just attempt
  # the kill and swallow errors).
  if (-not (Test-Path $pidFile)) { return }
  Write-Host 'Cleaning up daemons from previous session...' -ForegroundColor Yellow
  foreach ($line in (Get-Content $pidFile -ErrorAction SilentlyContinue)) {
    if (-not $line) { continue }
    $parts = $line.Split(':')
    if ($parts.Count -lt 2) { continue }
    $name = $parts[0]
    $procId = $parts[1]
    try {
      $proc = Get-Process -Id $procId -ErrorAction Stop
      # Verify it's actually one of our Python processes before killing —
      # PID could have been recycled to something unrelated.
      if ($proc.ProcessName -match 'python') {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "  - stopped stale $name (PID $procId)" -ForegroundColor DarkYellow
      }
    } catch {
      # Process not found or not killable — ignore
    }
  }
  Remove-Item $pidFile -ErrorAction SilentlyContinue
}

function Start-Daemons {
  $startedPids = @()
  foreach ($d in $daemons) {
    $outLog = Join-Path $logDir "$($d.Name).out.log"
    $errLog = Join-Path $logDir "$($d.Name).err.log"
    try {
      $proc = Start-Process -FilePath $pythonExe `
                            -ArgumentList @('-m', $d.Module) `
                            -WorkingDirectory $projectRoot `
                            -RedirectStandardOutput $outLog `
                            -RedirectStandardError $errLog `
                            -WindowStyle Hidden `
                            -PassThru
      $startedPids += "$($d.Name):$($proc.Id)"
      Write-Host "  - $($d.Name) started (PID $($proc.Id))" -ForegroundColor Green
    } catch {
      Write-Host "  - $($d.Name) FAILED to start: $_" -ForegroundColor Red
    }
  }
  if ($startedPids.Count -gt 0) {
    $startedPids | Out-File -FilePath $pidFile -Encoding utf8
  }
}

function Stop-Daemons {
  if (-not (Test-Path $pidFile)) { return }
  Write-Host ''
  Write-Host 'Stopping Python daemons...' -ForegroundColor Yellow
  foreach ($line in (Get-Content $pidFile)) {
    if (-not $line) { continue }
    $parts = $line.Split(':')
    if ($parts.Count -lt 2) { continue }
    $name = $parts[0]
    $procId = $parts[1]
    try {
      Stop-Process -Id $procId -Force -ErrorAction Stop
      Write-Host "  - $name (PID $procId) stopped" -ForegroundColor DarkGreen
    } catch {
      Write-Host "  - $name (PID $procId) already gone" -ForegroundColor DarkGray
    }
  }
  Remove-Item $pidFile -ErrorAction SilentlyContinue
}

Write-Host '== Nemo IB launcher ==' -ForegroundColor Cyan
Write-Host "Project: $projectRoot"
Write-Host ''

# Step 1: ensure Docker + SearxNG are up. start_searxng.ps1 is idempotent —
# fast path is ~50ms when SearxNG already responds.
Write-Host 'Starting Docker + SearxNG (idempotent)...' -ForegroundColor Yellow
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bootstrap
$bootstrapExit = $LASTEXITCODE

if ($bootstrapExit -ne 0) {
  Write-Host "WARNING: SearxNG bootstrap exited $bootstrapExit." -ForegroundColor Red
  Write-Host '         Search MCP tool will return [] until SearxNG is reachable.'
  Write-Host '         Continuing to launch Claude Code anyway.' -ForegroundColor Yellow
} else {
  try {
    $r = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 5 -UseBasicParsing
    if ($r.StatusCode -eq 200) {
      Write-Host 'SearxNG: healthy' -ForegroundColor Green
    } else {
      Write-Host "SearxNG: HTTP $($r.StatusCode) (may still work)" -ForegroundColor Yellow
    }
  } catch {
    Write-Host 'SearxNG: NOT responding (search tool will return [])' -ForegroundColor Red
  }
}

# Step 2: ensure the venv exists and python is available
if (-not (Test-Path $pythonExe)) {
  Write-Host ''
  Write-Host "ERROR: venv python not found at $pythonExe" -ForegroundColor Red
  Write-Host '       Run `python -m venv .venv` then `pip install -r requirements.txt`.'
  exit 2
}

# Step 3: clean up any stale daemons from a crashed previous session
Stop-StalePids

# Step 4: start the four Python daemons in the background
Write-Host ''
Write-Host 'Starting Python daemons (background)...' -ForegroundColor Yellow
Start-Daemons
Write-Host "Daemon logs: $logDir" -ForegroundColor DarkCyan

# Step 5: launch Claude Code in the project directory. Stay in the
# foreground so the user has a normal interactive session in this window.
# When the `& claude` call returns (user closes Claude Code), we drop
# through to Step 6 and stop the daemons.
#   --dangerously-skip-permissions:           trusted personal box, skip per-tool prompts
#   --remote-control:                         enable monitoring/driving from claude.ai/code
#   --dangerously-load-development-channels:  load the local slack channel plugin
Write-Host ''
Write-Host 'Launching Claude Code (--remote-control + slack channel)...' -ForegroundColor Cyan
Set-Location $projectRoot

try {
  & claude --dangerously-skip-permissions --remote-control --dangerously-load-development-channels server:slack
} finally {
  # Step 6: Claude Code has exited (or was interrupted). Stop the daemons
  # so we don't leave orphaned Python processes consuming resources.
  Stop-Daemons
}
