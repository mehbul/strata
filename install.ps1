# One-command setup for Strata on Windows.
#
#   irm https://raw.githubusercontent.com/mehbul/strata/main/install.ps1 | iex
#
# Fetches the llama.cpp ROCm runtime, builds the release binary, and runs setup.
# It does not download a model: those are large and the choice is yours.

$ErrorActionPreference = "Stop"

$LlamaBuild = "b10679"
$LlamaZip = "llama-$LlamaBuild-bin-win-rocm-7.14-x64.zip"
$LlamaUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaBuild/$LlamaZip"

function Say($msg) { Write-Host "  $msg" }
function Step($msg) { Write-Host "`n$msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host "`n  $msg" -ForegroundColor Red
    exit 1
}

Write-Host "`nStrata setup" -ForegroundColor White

# --- prerequisites ----------------------------------------------------------
Step "Checking prerequisites"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not installed. Get it from https://git-scm.com/download/win"
}
Say "git      ok"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Fail @"
cargo is not installed. Install Rust from https://rustup.rs and run this again.

  winget install Rustlang.Rustup
"@
}
Say "cargo    $((cargo --version) -replace 'cargo ','')"

# --- source -----------------------------------------------------------------
Step "Fetching Strata"

if (Test-Path "strata\Cargo.toml") {
    Say "strata\ already present, using it"
    Set-Location strata
} else {
    git clone --depth 1 https://github.com/mehbul/strata
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
    Set-Location strata
    Say "cloned into .\strata"
}

# --- runtime ----------------------------------------------------------------
Step "Fetching the compute runtime"

if (Test-Path "runtime\llama-server.exe") {
    Say "runtime\ already present, skipping"
} else {
    Say "downloading llama.cpp $LlamaBuild for ROCm (about 1 GB)"
    $zip = Join-Path $env:TEMP $LlamaZip
    curl.exe -L --fail -o "$zip" "$LlamaUrl"
    if ($LASTEXITCODE -ne 0) { Fail "download failed: $LlamaUrl" }
    Expand-Archive -Path "$zip" -DestinationPath "runtime" -Force
    Remove-Item "$zip" -ErrorAction SilentlyContinue
    Say "extracted to runtime\"
}

# The whole point of this check: a missing backend library shows up as an empty
# device list with nothing in the log, and everything silently falls back to CPU.
Step "Checking the GPU is visible"
$devices = & ".\runtime\llama-server.exe" --list-devices 2>&1 | Out-String
Write-Host $devices.TrimEnd()
if ($devices -notmatch "ROCm|Vulkan|CUDA") {
    Write-Host @"

  No GPU was listed. Strata will still build, but inference will run on the CPU
  and be far slower. A missing backend library is the usual cause.
"@ -ForegroundColor Yellow
}

# --- build ------------------------------------------------------------------
Step "Building (this takes a few minutes the first time)"
cargo build --release
if ($LASTEXITCODE -ne 0) { Fail "cargo build failed" }
Say "built target\release\strata.exe"

# --- configure --------------------------------------------------------------
Step "Configuring for this machine"
$models = Get-ChildItem "rocm\models" -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "model.gguf") }

if ($models.Count -eq 0) {
    Write-Host @"

  No model found in rocm\models\. Fetch one, then run setup:

    .\target\release\strata.exe pull --model ornith-1.5:35b
    .\target\release\strata.exe setup

  Then serve:

    .\strata.ps1
"@ -ForegroundColor Yellow
    exit 0
}

.\target\release\strata.exe setup
if ($LASTEXITCODE -ne 0) { Fail "setup failed" }

Write-Host @"

  Ready. Start it with:

    cd strata
    .\strata.ps1

  Then open http://localhost:8080
"@ -ForegroundColor Green
