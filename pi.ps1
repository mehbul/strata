<#
.SYNOPSIS
    Serve the local model if nothing is serving yet, then open pi on it.

.DESCRIPTION
    With Strata set as pi's default provider, this is what makes `pi` mean
    something wherever it is typed. It brings the engine to a usable state and
    hands off:

      nothing on the port    start strata.ps1 in its own window and wait
      up, model unloaded     POST /model/load - a second strata could not bind
      up and loaded          straight through

    Closing pi does not stop the engine. That is deliberate: the window keeps
    serving so the next `pi` opens instantly. Close it, or press Unload in the
    console at http://127.0.0.1:8080, to have the memory back.

    Every argument goes to pi untouched, so `pi -p "..."`, `pi --resume` and the
    rest behave exactly as they do without the wrapper. Utility commands that
    never reach a model - update, config, --help, --version, --list-models -
    skip the engine entirely.

    Two environment variables steer the wrapper itself:

      PI_NO_ENGINE=1    open pi without starting anything, for a cloud model
      STRATA_PORT=8080  where the engine listens; match models.json if changed

.EXAMPLE
    pi
    pi -p "explain src/placement.rs" --approve
    $env:PI_NO_ENGINE=1; pi        # a cloud model, no local load
#>
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$port = if ($env:STRATA_PORT) { [int]$env:STRATA_PORT } else { 8080 }
$timeout = if ($env:STRATA_TIMEOUT) { [int]$env:STRATA_TIMEOUT } else { 900 }

# Commands that only read local files or talk to pi.dev. Loading the weights for
# `pi update` would be a few minutes spent on nothing.
$offline = @("update", "config", "--help", "-h", "--version", "-v", "--list-models", "doctor")
$skip = $env:PI_NO_ENGINE -or ($args.Count -gt 0 -and $offline -contains $args[0])

# down | unloaded | ready. The middle one is real: strata keeps serving after
# the model is unloaded, so a listening port is not the same as a usable model.
function Get-EngineState {
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3
        if ($h.model.loaded) { "ready" } else { "unloaded" }
    } catch { "down" }
}

if (-not $skip) {
    if ((Get-EngineState) -eq "unloaded") {
        Write-Host "engine is serving but the model is unloaded - loading..." -ForegroundColor DarkGray
        try {
            Invoke-RestMethod "http://127.0.0.1:$port/model/load" -Method Post -TimeoutSec $timeout | Out-Null
        } catch {
            throw "the engine refused to load the model: $($_.Exception.Message)"
        }
    }
}

if (-not $skip -and (Get-EngineState) -eq "down") {
    Write-Host "starting the model on port $port..." -ForegroundColor DarkGray
    # The quotes around the path are not decoration. Start-Process joins
    # -ArgumentList with spaces and quotes nothing, so an unquoted path under
    # "Mehbul Islam" reaches powershell as `-File C:\Users\Mehbul` and the
    # window dies on the space while this loop dots away for a quarter of an
    # hour waiting for a server that was never started.
    $server = Start-Process powershell -PassThru -WorkingDirectory $root -ArgumentList @(
        "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "`"$root\strata.ps1`"",
        "-Listen", "127.0.0.1:$port"
    )

    # The server binds only after the weights are in, so the port answering is
    # the model being ready - there is no half-open state to poll around.
    $deadline = (Get-Date).AddSeconds($timeout)
    $ticks = 0
    while ((Get-EngineState) -ne "ready") {
        if ($server.HasExited) {
            throw "the engine window exited (code $($server.ExitCode)) - run .\strata.ps1 in $root to see why"
        }
        if ((Get-Date) -gt $deadline) {
            throw "the engine did not come up within $timeout s - see the window it opened"
        }
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline -ForegroundColor DarkGray
        # Dots alone cannot tell a slow load from a window sitting on an error,
        # so say every half minute whether the engine process actually exists.
        if ((++$ticks % 15) -eq 0) {
            $up = [bool](Get-Process strata -ErrorAction SilentlyContinue)
            Write-Host (" engine process: " + $(if ($up) { "running, loading weights" } else { "not started yet - check the other window" })) -ForegroundColor DarkGray
        }
    }
    Write-Host "`nready" -ForegroundColor DarkGray
}

# pi.cmd, not pi: `pi` is the profile function that calls this script.
& pi.cmd @args
