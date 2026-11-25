# One-command setup for Strata on Windows.
#
#   irm https://raw.githubusercontent.com/mehbul/strata/main/install.ps1 | iex
#
# Fetches the llama.cpp ROCm runtime, builds the release binary, and runs setup.
# It does not download a model: those are large and the choice is yours.

$ErrorActionPreference = "Stop"

$LlamaBuild = "b12704"
$LlamaZip = "llama-$LlamaBuild-bin-win-rocm-10.0-x64.zip"
$LlamaUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaBuild/$LlamaZip"

# Windows PowerShell 5.1 turns any native command's stderr into an error
# record. With ErrorActionPreference=Stop, cargo's ordinary progress output
# would abort the install. Run native commands with the preference relaxed.
function Invoke-Native($file, [string[]] $arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $file @arguments }
    finally { $ErrorActionPreference = $previous }
    return $LASTEXITCODE
}

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
    $code = Invoke-Native git @('clone', '--depth', '1', 'https://github.com/mehbul/strata')
    if ($code -ne 0) { Fail "git clone failed" }
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
    $code = Invoke-Native 'curl.exe' @('-L', '--fail', '-o', $zip, $LlamaUrl)
    if ($code -ne 0) { Fail "download failed: $LlamaUrl" }
    Expand-Archive -Path "$zip" -DestinationPath "runtime" -Force
    Remove-Item "$zip" -ErrorAction SilentlyContinue
    Say "extracted to runtime\"
}

# The ROCm zip ships ggml-hip.dll but NOT the BLAS libraries it imports. Without
# them LoadLibrary fails with error 126, --list-devices prints "(none)" and every
# benchmark silently runs on the CPU. Provision them from a local ROCm install
# before checking the device list.
Step "Checking ROCm support libraries"
$runtimeDir = Join-Path (Get-Location).Path "runtime"
$sources = @()
if ($env:STRATA_ROCM_LIBS) { $sources += $env:STRATA_ROCM_LIBS }
foreach ($root in @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama"),
    (Join-Path $env:ProgramFiles "AMD\ROCm"),
    (Join-Path ${env:ProgramFiles(x86)} "AMD\ROCm")
)) {
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem $root -Directory -Recurse -Depth 2 -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "rocblas.dll") } |
        ForEach-Object { $sources += $_.FullName }
}
# (target name, source name) pairs. Never overwrite amdhip64_7.dll or
# amd_comgr.dll: the runtime's own 7.14 copies are the ones ggml-hip needs.
$pairs = @(
    @("hipblas.dll", "libhipblas.dll"),
    @("hipblas.dll", "hipblas.dll"),
    @("rocblas.dll", "rocblas.dll"),
    @("hipblaslt.dll", "libhipblaslt.dll"),
    @("libhipblaslt.dll", "libhipblaslt.dll"),
    @("msvcp140.dll", "msvcp140.dll"),
    @("msvcp140_1.dll", "msvcp140_1.dll"),
    @("msvcp140_2.dll", "msvcp140_2.dll"),
    @("vcruntime140.dll", "vcruntime140.dll"),
    @("vcruntime140_1.dll", "vcruntime140_1.dll"),
    @("concrt140.dll", "concrt140.dll"),
    @("amd_comgr_3.dll", "amd_comgr_3.dll")
)
foreach ($pair in $pairs) {
    $target = Join-Path $runtimeDir $pair[0]
    if (Test-Path $target) { continue }
    foreach ($src in $sources) {
        $candidate = Join-Path $src $pair[1]
        if (Test-Path $candidate) {
            Copy-Item $candidate $target -Force
            Say "copied $($pair[0]) from $src"
            break
        }
    }
}

# b12704's ROCm build also loads Tensile kernels from rocblas/library beside the
# binary. The release zip does not contain them: without this folder the server
# crashes during model load with "Cannot read .../rocblas/library/TensileLibrary.dat".
$tensile = Join-Path $runtimeDir "rocblas\library"
if (-not (Test-Path $tensile)) {
    foreach ($src in $sources) {
        $candidate = Join-Path $src "rocblas\library"
        if (Test-Path $candidate) {
            New-Item -ItemType Directory -Path (Split-Path $tensile -Parent) -Force | Out-Null
            Copy-Item -LiteralPath $candidate -Destination $tensile -Recurse -Force
            Say "copied rocblas\library from $src"
            break
        }
    }
}

# The whole point of this check: a missing backend library shows up as an empty
# device list with nothing in the log, and everything silently falls back to CPU.
Step "Checking the GPU is visible"
$devices = Invoke-Native '.\runtime\llama-server.exe' @('--list-devices') 2>&1 | Out-String
Write-Host $devices.TrimEnd()
if ($devices -notmatch "ROCm|Vulkan|CUDA") {
    Fail @"

  No GPU was listed. Inference would run on the CPU, which is not what this
  engine is for. The ROCm runtime needs hipblas.dll, rocblas.dll and their
  support DLLs next to llama-server.exe. Set STRATA_ROCM_LIBS to a folder
  containing them, or install Ollama (which ships a compatible set), then
  run install.ps1 again.
"@
}

# --- build ------------------------------------------------------------------
Step "Building (this takes a few minutes the first time)"
$code = Invoke-Native cargo @('build', '--release')
if ($code -ne 0) { Fail "cargo build failed" }
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

$code = Invoke-Native '.\target\release\strata.exe' @('setup')
if ($code -ne 0) { Fail "setup failed" }

# --- global command ---------------------------------------------------------
# The engine discovers `runtime\` and `rocm\models\` relative to the working
# directory, so a bare PATH entry to target\release would silently lose the GPU
# runtime. A small shim changes directory first.
Step "Installing the 'strata' command"
$binDir = Join-Path $env:USERPROFILE ".strata\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$exePath = (Resolve-Path "target\release\strata.exe").Path
$projectDir = (Get-Location).Path
Set-Content -Path (Join-Path $binDir "strata.cmd") -Encoding ASCII -Value @(
    "@echo off"
    "cd /d `"$projectDir`""
    'if not exist "target\release\strata.exe" cargo build --release'
    'if errorlevel 1 exit /b 1'
    'for /f %%i in (''powershell -NoProfile -Command "(Get-ChildItem src -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime -gt (Get-Item target\release\strata.exe).LastWriteTime"'') do if "%%i"=="True" cargo build --release'
    'if errorlevel 1 exit /b 1'
    "`"$exePath`" %*"
)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$binDir*") {
    $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $binDir } else { "$userPath;$binDir" }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = "$env:Path;$binDir"
}
Say "strata.cmd in $binDir (open a new terminal to pick up PATH)"

Write-Host @"

  Ready. Start it with:

    strata serve --model <name>

  or, from this directory:

    .\strata.ps1

  Then open http://localhost:8080
"@ -ForegroundColor Green
