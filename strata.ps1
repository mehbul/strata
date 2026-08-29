<#
.SYNOPSIS
    Build what is out of date, then serve. The one command for daily use.

.DESCRIPTION
    Rebuilds the engine when a source file is newer than the binary, builds the
    web console if it is missing, and starts the server. Nothing here is magic:
    it is `cargo build`, `npm run build` and `strata serve`, skipped when they
    would be no-ops.

.EXAMPLE
    .\strata.ps1
    .\strata.ps1 -Ctx 16384
    .\strata.ps1 -Model ornith-1.5_35b -NoCompact
#>
[CmdletBinding()]
param(
    # Model directory under rocm/models. Defaults to the only one present.
    [string] $Model,
    # Context window to load. `strata tune --ctx N --save` first if you change it.
    [int]    $Ctx = 65536,
    # Address to listen on.
    [string] $Listen = "127.0.0.1:8080",
    # Let a long conversation overflow instead of summarising its older turns.
    [switch] $NoCompact,
    # Rebuild even if nothing changed.
    [switch] $Rebuild
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$exe = "target\release\strata.exe"

# --- engine ---------------------------------------------------------------
$newestSource = Get-ChildItem src, Cargo.toml, build.rs -Recurse -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$stale = $Rebuild -or -not (Test-Path $exe) -or
         $newestSource.LastWriteTime -gt (Get-Item $exe).LastWriteTime
if ($stale) {
    Write-Host "building the engine..." -ForegroundColor DarkGray
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} else {
    Write-Host "engine up to date" -ForegroundColor DarkGray
}

# --- console --------------------------------------------------------------
# Only built when absent: the console changes far less often than the engine,
# and `npm run build` is slower than checking.
if (-not (Test-Path "web\dist\index.html")) {
    Write-Host "building the console..." -ForegroundColor DarkGray
    Push-Location web
    try {
        if (-not (Test-Path node_modules)) { npm install }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
    } finally { Pop-Location }
}

# --- model ----------------------------------------------------------------
if (-not $Model) {
    $found = @(Get-ChildItem "rocm\models" -Directory -ErrorAction SilentlyContinue)
    if ($found.Count -eq 1) {
        $Model = $found[0].Name
    } elseif ($found.Count -eq 0) {
        throw "no models in rocm\models. Run: $exe setup   (it explains what is missing)"
    } else {
        throw "several models present; pick one with -Model: $($found.Name -join ', ')"
    }
}

# --- context --------------------------------------------------------------
# A tuned split is worth ~2x, and it is context-specific: the KV cache reserved
# at load time is VRAM the experts cannot have, so each context keeps its own
# measurement in tuned-<ctx>.json.
function Get-TunedContexts {
    @(Get-ChildItem "rocm\models\$Model\tuned-*.json" -ErrorAction SilentlyContinue |
        ForEach-Object { if ($_.BaseName -match '^tuned-(\d+)$') { [int]$Matches[1] } } |
        Sort-Object)
}

$explicitCtx = $PSBoundParameters.ContainsKey('Ctx')
if (-not $explicitCtx) {
    $measured = Get-TunedContexts
    if ($measured.Count -eq 0) {
        # Nothing has been measured on this machine yet. `setup` inspects the
        # hardware, picks a context that fits it, and measures the split - the
        # same steps that were done by hand for the first machine.
        Write-Host "no measurement for this machine yet - running setup" -ForegroundColor Cyan
        & $exe setup --model $Model
        if ($LASTEXITCODE -ne 0) { throw "setup did not complete" }
        $measured = Get-TunedContexts
        if ($measured.Count -eq 0) { throw "setup produced no tuned configuration" }
    }
    # Prefer the default when it has been measured, otherwise the largest that has.
    $Ctx = if ($measured -contains $Ctx) { $Ctx } else { $measured[-1] }
}

if (-not (Test-Path "rocm\models\$Model\tuned-$Ctx.json")) {
    Write-Host ("no measured split for ctx {0} - run: {1} tune --model {2} --ctx {0} --save" `
                -f $Ctx, $exe, $Model) -ForegroundColor Yellow
}

# --- serve ----------------------------------------------------------------
$serveArgs = @("serve", "--model", $Model, "--ctx", $Ctx, "--listen", $Listen)
if ($NoCompact) { $serveArgs += "--no-compact" }
& $exe @serveArgs
