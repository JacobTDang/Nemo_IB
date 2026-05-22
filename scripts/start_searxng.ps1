# Idempotently bring up Docker Desktop + the SearxNG container.
#
# Designed for fire-and-forget invocation from a Claude Code SessionStart hook.
# Exits 0 on success or when SearxNG was already healthy; non-zero only when
# Docker fails to come up within the timeout. Output is suppressed so the hook
# doesn't leak text into the Claude Code transcript.

$ErrorActionPreference = 'SilentlyContinue'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerCli   = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$dockerApp   = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Fast path: SearxNG is already responding — nothing to do.
try {
  $r = Invoke-WebRequest -Uri 'http://localhost:8888/healthz' `
                         -TimeoutSec 2 -UseBasicParsing
  if ($r.StatusCode -eq 200) { exit 0 }
} catch {}

# Launch Docker Desktop if its tray process is absent. Hidden window keeps it
# from grabbing focus when a new shell starts.
if (-not (Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue)) {
  if (Test-Path $dockerApp) {
    Start-Process -FilePath $dockerApp -WindowStyle Hidden
  } else {
    exit 1  # Docker Desktop not installed at the expected path
  }
}

# Wait for the Docker daemon to start accepting commands. Cold boot of Docker
# Desktop on Windows typically takes 20-45s; cap at 120s to avoid hanging
# forever if WSL2/Hyper-V is broken.
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
  & $dockerCli info 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) { break }
  Start-Sleep -Seconds 2
}
if ($LASTEXITCODE -ne 0) { exit 2 }

# Bring up SearxNG (idempotent — `compose up -d` is a no-op when the container
# is already running). Run from the project root so docker-compose.yml resolves.
Push-Location $projectRoot
try {
  & $dockerCli compose up -d searxng 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { exit 3 }
} finally {
  Pop-Location
}

# `compose up -d` returns as soon as the container starts, but SearxNG takes
# another 5-15 seconds to bind to port 8888 (Python imports + uwsgi spin-up).
# Poll /healthz until it responds or we hit a 45s cap. Without this poll,
# downstream consumers (search MCP tool, launcher health check) race the
# in-container startup and see "not responding" right after a successful bring-up.
$deadline = (Get-Date).AddSeconds(45)
while ((Get-Date) -lt $deadline) {
  try {
    $r = Invoke-WebRequest -Uri 'http://localhost:8888/healthz' `
                           -TimeoutSec 2 -UseBasicParsing
    if ($r.StatusCode -eq 200) { exit 0 }
  } catch {}
  Start-Sleep -Seconds 1
}
exit 4  # container is up but SearxNG never started answering
