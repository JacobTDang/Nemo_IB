# Single-shot launcher for the Nemo IB workflow.
#
# Brings up the required external service (SearxNG container) and then
# hands control to Claude Code in the project directory. Reuses the
# bootstrap logic in start_searxng.ps1 so a developer can still run the
# Docker piece on its own without launching Claude.
#
# Designed for double-click invocation via nemo.bat at the project root.

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$bootstrap   = Join-Path $PSScriptRoot 'start_searxng.ps1'
$healthUrl   = 'http://localhost:8888/healthz'

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
  # Double-check the container actually answers before handing off.
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

# Step 2: launch Claude Code in the project directory. Stay in the foreground
# so the user has a normal interactive session in this window.
#   --dangerously-skip-permissions: trusted personal box, skip per-tool prompts
#   --remote-control:               enable monitoring/driving from claude.ai/code
Write-Host ''
Write-Host 'Launching Claude Code (--remote-control --dangerously-skip-permissions)...' -ForegroundColor Cyan
Set-Location $projectRoot
& claude --dangerously-skip-permissions --remote-control
