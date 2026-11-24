# GitHub Storefront Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Strata repository into its own marketing surface so stars and visitors grow without any external posting.

**Architecture:** Repository-only changes: a rewritten README, a routing docs landing page, copy-paste examples, a reproducible benchmark pipeline (`strata bench` -> `scripts/demo.ps1` -> `docs/BENCHMARKS.md` -> README table), GitHub metadata/template scripts, and a docs link checker. No Rust behaviour changes.

**Tech Stack:** Markdown, PowerShell 5.1+, `gh` CLI (optional), existing `strata bench`/`strata doctor` commands.

## Global Constraints

- No external posting, paid promotion, or GitHub Pages site.
- No new image generation; reuse `docs/social-preview.png`, `docs/demo.gif`, `docs/console.jpg`.
- No behaviour change to Strata: only docs, examples and scripts.
- Every performance number must come from `docs/BENCHMARKS.md`.
- `/experts` remains simulated and must stay labelled as such.
- All scripts must work on Windows PowerShell 5.1 and degrade cleanly when tools are absent.
- Acceptance checks: `cargo test`, `npm run build --prefix web`, `strata doctor`, `powershell -File scripts/check-docs.ps1`.

---

### Task 1: Docs link checker with self-test

**Files:**
- Create: `scripts/check-docs.ps1`
- Create: `scripts/test-check-docs.ps1`
- Test: `scripts/test-check-docs.ps1`

**Interfaces:**
- Consumes: repository markdown files at the repo root and under `docs/`.
- Produces: `scripts/check-docs.ps1` exits 0 on success, 1 on broken relative link or unlabelled fence in README; warns (exit 0) on benchmark-number mismatch.

- [ ] **Step 1: Write the failing test**

Create `scripts/test-check-docs.ps1` with this exact content:

```powershell
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$check = Join-Path $PSScriptRoot 'check-docs.ps1'

# 1. The real repository must pass.
& $check -Root $root | Out-Null
if ($LASTEXITCODE -ne 0) { throw "check-docs failed on the real repo" }

# 2. A broken relative link must fail.
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) 'strata-checkdocs-test'
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
Set-Content (Join-Path $tmp 'a.md') '# A`n`n[missing](no-such-file.md)'
& $check -Root $tmp 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { throw "broken link was not detected" }

# 3. An unlabelled fence in README.md must fail.
Remove-Item -Recurse -Force $tmp
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
Set-Content (Join-Path $tmp 'README.md') '# R`n`n```' -Encoding UTF8
& $check -Root $tmp 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { throw "unlabelled fence was not detected" }

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
Write-Host 'check-docs tests passed'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `powershell -File scripts/test-check-docs.ps1`
Expected: fails because `scripts/check-docs.ps1` does not exist.

- [ ] **Step 3: Write the checker**

Create `scripts/check-docs.ps1` with this exact content:

```powershell
param([string]$Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$failed = $false

function Test-RelativeLink($file, $target) {
    if ($target -match '^(https?://|#|mailto:)') { return $true }
    $clean = ($target -split '[#?]')[0]
    if ([string]::IsNullOrWhiteSpace($clean)) { return $true }
    $path = Join-Path (Split-Path -Parent $file) $clean
    if (Test-Path $path) { return $true }
    Write-Host "BROKEN $($file.Substring($Root.Length + 1)): $target" -ForegroundColor Red
    return $false
}

$files = @(Get-ChildItem -Path $Root -Filter *.md -File)
$files += @(Get-ChildItem -Path (Join-Path $Root 'docs') -Filter *.md -File -Recurse -ErrorAction SilentlyContinue)
$files += @(Get-ChildItem -Path (Join-Path $Root '.github') -Filter *.md -File -Recurse -ErrorAction SilentlyContinue)
$files = $files | Where-Object { $_.FullName -notlike '*\node_modules\*' } | Sort-Object FullName -Unique

foreach ($file in $files) {
    $text = Get-Content $file.FullName -Raw
    foreach ($m in [regex]::Matches($text, '\[[^\]]+\]\(([^)]+)\)')) {
        if (-not (Test-RelativeLink $file.FullName $m.Groups[1].Value)) { $failed = $true }
    }
}

$readme = Join-Path $Root 'README.md'
if (Test-Path $readme) {
    $lines = Get-Content $readme
    $inFence = $false
    foreach ($line in $lines) {
        if ($line -match '^\s*```') {
            if (-not $inFence -and $line.Trim() -eq '```') {
                Write-Host "UNLABELLED FENCE in README.md" -ForegroundColor Red
                $failed = $true
            }
            $inFence = -not $inFence
        }
    }
}

$bench = Join-Path $Root 'docs/BENCHMARKS.md'
if (-not (Test-Path $bench)) {
    Write-Host 'MISSING docs/BENCHMARKS.md' -ForegroundColor Red
    $failed = $true
} elseif (-not (Select-String -Path $bench -Pattern '^\|' -Quiet)) {
    Write-Host 'docs/BENCHMARKS.md has no table row' -ForegroundColor Red
    $failed = $true
}

if ($failed) { exit 1 }
Write-Host "docs ok: $($files.Count) markdown files checked"
exit 0
```

- [ ] **Step 4: Run the test again**

Run: `powershell -File scripts/test-check-docs.ps1`
Expected: `check-docs tests passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/check-docs.ps1 scripts/test-check-docs.ps1
git commit -m "docs: add markdown link checker with self-test"
```

---

### Task 2: Docs landing page and examples

**Files:**
- Create: `docs/index.md`
- Create: `examples/openai-client.py`
- Create: `examples/anthropic-client.py`
- Create: `examples/Modelfile`
- Create: `examples/mcp-client.json`
- Create: `examples/serve-tuned.ps1`

**Interfaces:**
- Consumes: existing README sections (`# Install`, `# Hardware`, `# Commands`) and `strata serve --ctx --cpu-moe --kv-type`.
- Produces: relative link targets used by the rewritten README in Task 5.

- [ ] **Step 1: Write docs/index.md**

```markdown
# Strata docs

Start here, depending on what you want to do.

- **Run a model locally.** [Install and serve](../README.md#install) from the README, then open <http://localhost:8080>.
- **Drive Strata from an agent.** Use the [OpenAI](../examples/openai-client.py), [Anthropic](../examples/anthropic-client.py) or [MCP](../examples/mcp-client.json) examples. Any OpenAI-compatible client works against `http://127.0.0.1:8080/v1`.
- **Run on AMD / ROCm.** See [Hardware](../README.md#hardware) and [why is my GPU not being used?](../README.md#why-is-my-gpu-not-being-used).
- **Understand the engine.** [Engine notes](LEARNING.md), [Ollama reverse-engineering dossier](OLLAMA-RE.md), [Design](DESIGN.md), [Product](PRODUCT.md), [Improvement plan](PLAN.md).
- **Check measured numbers.** [Benchmarks](BENCHMARKS.md).
```

- [ ] **Step 2: Write examples/openai-client.py**

```python
# Run strata serve first; this uses the OpenAI-compatible endpoint.
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("STRATA_URL", "http://127.0.0.1:8080/v1"),
    api_key=os.getenv("STRATA_API_KEY", "local"),
)
model = os.getenv("STRATA_MODEL", "ornith-1.5_35b")
reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply.choices[0].message.content)
```

- [ ] **Step 3: Write examples/anthropic-client.py**

```python
# Run strata serve first; this uses the Anthropic-compatible endpoint.
import os
from anthropic import Anthropic

client = Anthropic(
    base_url=os.getenv("STRATA_URL", "http://127.0.0.1:8080"),
    api_key=os.getenv("STRATA_API_KEY", "local"),
)
model = os.getenv("STRATA_MODEL", "ornith-1.5_35b")
reply = client.messages.create(
    model=model,
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply.content[0].text)
```

- [ ] **Step 4: Write examples/Modelfile**

```text
# Derive a model with its own instructions; use with strata create <name> -f examples/Modelfile
FROM ornith-1.5_35b
PARAMETER temperature 0.3
PARAMETER num_ctx 16384
SYSTEM """You are a concise assistant. Answer in one paragraph unless asked otherwise."""
```

- [ ] **Step 5: Write examples/mcp-client.json**

```json
{
  "mcpServers": {
    "strata": {
      "command": "strata",
      "args": ["mcp"]
    }
  }
}
```

- [ ] **Step 6: Write examples/serve-tuned.ps1**

```powershell
# Serve a measured split. Run `strata tune --model ornith-1.5_35b --ctx 16384 --save` first.
param(
    [string]$Model = 'ornith-1.5_35b',
    [int]$Ctx = 16384,
    [string]$Listen = '127.0.0.1:8080'
)
& strata serve --model $Model --ctx $Ctx --listen $Listen --kv-type q8_0
```

- [ ] **Step 7: Verify links and JSON**

Run: `powershell -File scripts/check-docs.ps1` (expected: reports docs files count, exit 0) and
`powershell -Command "Get-Content examples/mcp-client.json -Raw | ConvertFrom-Json | Out-Null"`.
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add docs/index.md examples/
git commit -m "docs: add landing page and copy-paste examples"
```

---

### Task 3: Reproducible benchmarks

**Files:**
- Create: `scripts/demo.ps1`
- Create: `docs/BENCHMARKS.md`
- Modify: `README.md` (only after Task 5 — leave for Task 5)

**Interfaces:**
- Consumes: `strata bench --url <u> --model <m> --prompt-tokens N --output-tokens N --json <file>`, `strata info`.
- Produces: `docs/BENCHMARKS.md` with `## YYYY-MM-DD` headings and a markdown table containing `ttft_ms`, `prefill_tps`, `decode_tps`, `total_ms`.

- [ ] **Step 1: Write the failing test**

Add to `scripts/test-check-docs.ps1` before its final `Write-Host`:

```powershell
# 4. Benchmark table must exist with a numeric row.
$benchText = Get-Content (Join-Path $root 'docs/BENCHMARKS.md') -Raw
if ($benchText -notmatch '\|\s*\d+') { throw 'BENCHMARKS.md has no numeric row' }
```

- [ ] **Step 2: Run the test**

Run: `powershell -File scripts/test-check-docs.ps1`
Expected: fails because `docs/BENCHMARKS.md` does not exist.

- [ ] **Step 3: Write scripts/demo.ps1**

```powershell
param(
    [string]$Model = 'ornith-1.5_35b',
    [string]$Url = 'http://127.0.0.1:8080',
    [string]$Out = 'docs/BENCHMARKS.md',
    [int]$Ctx = 16384,
    [int]$PromptTokens = 2048,
    [int]$OutputTokens = 256
)
$ErrorActionPreference = 'Stop'

try {
    $version = Invoke-RestMethod -Uri "$Url/api/version" -TimeoutSec 3
} catch {
    Write-Host "No server at $Url. Start it first:"
    Write-Host "  strata serve --model $Model --ctx $Ctx"
    exit 0
}
Write-Host "Server version $($version.version)"

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "strata-bench-$PID.json"
& strata bench --url $Url --model $Model --prompt-tokens $PromptTokens --output-tokens $OutputTokens --json $tmp
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tmp)) {
    Remove-Item $tmp -ErrorAction SilentlyContinue
    throw 'strata bench failed'
}
$result = Get-Content $tmp -Raw | ConvertFrom-Json
Remove-Item $tmp -ErrorAction SilentlyContinue

$hardware = (& strata info 2>&1 | Select-String 'GPU:|CPU:|RAM:' | ForEach-Object { $_.Line.Trim() }) -join '; '
$date = Get-Date -Format 'yyyy-MM-dd'
$row = "| $Model | $Ctx | $PromptTokens | $OutputTokens | $([math]::Round($result.ttft_ms,1)) | $([math]::Round($result.prefill_tps,1)) | $([math]::Round($result.decode_tps,1)) | $([math]::Round($result.total_ms,0)) |"
$heading = "## $date"
$content = if (Test-Path $Out) { Get-Content $Out -Raw } else {
    "# Measured benchmarks`n`nNumbers here come from ``strata bench`` on the hardware listed below. Regenerate with ``powershell -File scripts/demo.ps1``.`n"
}
if ($content -match "(?m)^## $date\r?$") {
    $content = [regex]::Replace($content, "(?ms)^## $date\r?\n.*?(?=^## |\z)", "$heading`n`nHardware: $hardware`n`n| model | ctx | prompt | output | TTFT ms | prefill tok/s | decode tok/s | total ms |`n|---|---|---|---|---|---|---|---|`n$row`n`n")
} else {
    $content = $content.TrimEnd() + "`n`n$heading`n`nHardware: $hardware`n`n| model | ctx | prompt | output | TTFT ms | prefill tok/s | decode tok/s | total ms |`n|---|---|---|---|---|---|---|---|`n$row`n`nCommand: ``strata bench --url $Url --model $Model --prompt-tokens $PromptTokens --output-tokens $OutputTokens``"
}
Set-Content -Path $Out -Value $content -Encoding UTF8
Write-Host "wrote $Out"
```

- [ ] **Step 4: Seed docs/BENCHMARKS.md with the existing measured result**

```markdown
# Measured benchmarks

Numbers here come from `strata bench` on the hardware listed below. Regenerate
with `powershell -File scripts/demo.ps1`.

## 2026-11-23

Hardware: CPU: AMD Ryzen 5 7600X 6C12T; GPU: AMD Radeon RX 7600 XT 16GB (gfx1102, RDNA3); RAM: 32GB DDR5; OS: Windows 11 Pro

| model | ctx | prompt | output | TTFT ms | prefill tok/s | decode tok/s | total ms |
|---|---|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B Q4_K_M | 65536 | 8192 | 512 | 300 | 353 | 42.5 | 35300 |

Command: `strata bench --url http://127.0.0.1:8080 --model ornith-1.5_35b --prompt-tokens 8192 --output-tokens 512`
```

- [ ] **Step 5: Run the test**

Run: `powershell -File scripts/test-check-docs.ps1`
Expected: `check-docs tests passed`.

- [ ] **Step 6: Verify demo failure path**

Run: `powershell -File scripts/demo.ps1 -Url http://127.0.0.1:9 -Model x`
Expected: prints "No server" and the `strata serve` command, exits 0, does not modify `docs/BENCHMARKS.md`.

- [ ] **Step 7: Commit**

```bash
git add scripts/demo.ps1 scripts/test-check-docs.ps1 docs/BENCHMARKS.md
git commit -m "docs: add reproducible benchmark pipeline"
```

---

### Task 4: GitHub metadata and templates

**Files:**
- Create: `scripts/github-setup.ps1`
- Modify: `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/hardware_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`
- Modify: `.github/PULL_REQUEST_TEMPLATE.md`
- Modify: `CONTRIBUTING.md`
- Create: `docs/GROWTH.md`

**Interfaces:**
- Consumes: `gh` CLI if installed; `strata doctor` and `strata info` outputs in issue templates.
- Produces: `docs/GROWTH.md` linked from README; verification block in CONTRIBUTING.

- [ ] **Step 1: Write scripts/github-setup.ps1**

```powershell
$ErrorActionPreference = 'Stop'
$description = 'Ollama-compatible Rust engine for local MoE and dense models on AMD ROCm, with measured weight placement and a web console.'
$topics = @('llm','rust','rocm','amd','ollama','moe','local-ai','openai-compatible','llama-cpp','inference-engine')

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host 'gh CLI not found. Set these manually in GitHub repo settings:'
    Write-Host "Description: $description"
    Write-Host "Topics: $($topics -join ', ')"
    exit 0
}
try { gh auth status 2>&1 | Out-Null } catch {
    Write-Host 'gh is not authenticated. Run `gh auth login`, then re-run this script.'
    exit 0
}
& gh repo edit --description $description
foreach ($topic in $topics) { & gh repo edit --add-topic $topic }
foreach ($label in @('good first issue','hardware','documentation')) {
    & gh label create $label --force 2>&1 | Out-Null
}
Write-Host 'GitHub metadata updated.'
```

- [ ] **Step 2: Update issue templates**

In `.github/ISSUE_TEMPLATE/bug_report.md`, add after the existing intro:

```markdown
## Diagnostic output

Paste the output of `strata doctor` and, if the GPU is involved,
`.\runtime\llama-server.exe --list-devices`.
```

In `.github/ISSUE_TEMPLATE/hardware_report.md`, ensure these fields exist:

```markdown
- GPU and VRAM:
- ROCm / Vulkan / CUDA version:
- OS:
- Model and quantisation:
- Output of `strata info`:
```

In `.github/ISSUE_TEMPLATE/feature_request.md`, add:

```markdown
## Alternatives considered

What did you try instead, and why did it not work?
```

- [ ] **Step 3: Update PR template and CONTRIBUTING**

In `.github/PULL_REQUEST_TEMPLATE.md`, add:

```markdown
## Verification

- [ ] `cargo test`
- [ ] `npm run build --prefix web` (if console changed)
- [ ] `powershell -File scripts/check-docs.ps1`
```

In `CONTRIBUTING.md`, replace the `cargo fmt --all / cargo test` block with:

```powershell
cargo fmt --all
cargo test
npm run build --prefix web
strata doctor
powershell -File scripts/check-docs.ps1
```

and add a sentence: "CI runs these on Windows and Linux; `strata doctor` is a local check of runtime, GPU, models and console."

- [ ] **Step 4: Write docs/GROWTH.md**

```markdown
# Growth checklist (repo-side only)

What "explodes" means here: stars, forks, clone/traffic counts, issue quality,
and which docs get referenced in issues. No external posting is part of this
plan.

## 7 days

- Answer every issue within 24 hours.
- Label good first issues with `good first issue`.
- Keep the README benchmark table current from `docs/BENCHMARKS.md`.
- Pin one "start here" issue explaining install + first model.

## 30 days

- Add one example under `examples/` per recurring question.
- Refresh `docs/BENCHMARKS.md` whenever hardware or runtime changes.
- Run `powershell -File scripts/check-docs.ps1` and fix broken links.
- Check GitHub Insights -> Traffic for which docs bring visitors.

## Do not

- Claim a number that is not in `docs/BENCHMARKS.md`.
- Present the simulated expert grid as real telemetry.
```

- [ ] **Step 5: Verify**

Run: `powershell -File scripts/github-setup.ps1`
Expected: either "metadata updated" or the manual steps print; exit 0.
Run: `powershell -File scripts/check-docs.ps1`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/github-setup.ps1 .github CONTRIBUTING.md docs/GROWTH.md
git commit -m "docs: add GitHub metadata script, templates and growth checklist"
```

---

### Task 5: README rewrite

**Files:**
- Modify: `README.md`
- Consumes: `docs/index.md`, `docs/BENCHMARKS.md`, `examples/*`, `docs/LEARNING.md`

**Interfaces:**
- Consumes: benchmark table from `docs/BENCHMARKS.md` (Task 3), examples (Task 2), GROWTH (Task 4).
- Produces: final front door with no unsupported claims.

- [ ] **Step 1: Replace the header and install block (lines 1-45)**

Replace with:

```markdown
# Strata

**An Ollama-compatible local engine you can actually tune. Run MoE or dense
models on your own hardware, with an OpenAI and Anthropic API and a web console.**
No daemon, no cloud, no NVIDIA requirement.

[![CI](https://github.com/mehbul/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/mehbul/strata/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![GPU](https://img.shields.io/badge/GPU-ROCm%20%7C%20Vulkan%20%7C%20CUDA-red.svg)]()

![Strata reading a GGUF and benchmarking a model it is serving](docs/demo.gif)

Strata reads the model file itself, solves where its weights should sit across
VRAM and system RAM, starts and owns the llama.cpp compute process, and serves
OpenAI, Anthropic, Ollama and MCP endpoints plus a web console. It measures the
split on your machine instead of guessing it.

**Measured on an RX 7600 XT 16 GB: 35B MoE at ~42 tok/s with a 65,536-token
window.** See [Benchmarks](docs/BENCHMARKS.md) for the exact hardware and commands.

## 30-second start

```powershell
irm https://raw.githubusercontent.com/mehbul/strata/main/install.ps1 | iex
.\target\release\strata.exe pull --model ornith-1.5:35b
.\strata.ps1
```

Open <http://localhost:8080>. The install script checks git and Rust, fetches the
llama.cpp ROCm runtime, builds the release binary and configures the machine. It
does not download a model. [Read it](install.ps1) before piping it to `iex`.

> No hosted demo: Strata owns your GPU. The GIF above is a real session on an
> RX 7600 XT, and the numbers are the ones `strata bench` printed.

![Strata's web console streaming from a 35B MoE model](docs/console.jpg)
```

- [ ] **Step 2: Add the comparison table after the existing "What is Strata's" table**

```markdown
## Why Strata over the alternatives

| | Strata | Ollama | Raw llama.cpp |
|---|---|---|---|
| Who owns the process | Strata starts and stops it | A background daemon | You do |
| API dialects | OpenAI, Anthropic, Ollama, MCP | Ollama + OpenAI | llama.cpp HTTP |
| Weight placement | Measured, saved per context | Automatic, fixed policy | Manual flags |
| Dense and MoE | Same reader, tokenizer and API; `-ngl` vs `-ncmoe` sweep | Model-dependent | Manual |
| Web console | Built in | Built in | Optional UI |
| Matrix multiplies | llama.cpp | llama.cpp | llama.cpp |

Start with [the docs index](docs/index.md) or a copy-paste
[example](examples/openai-client.py).
```

- [ ] **Step 3: Add an honest limits section before `## Notes`**

```markdown
## What Strata does not do yet

- The expert map in the console is **simulated**: llama.cpp exposes no router telemetry.
- There is no disk tier for experts. Past VRAM + host RAM, setup refuses.
- Inference is still llama.cpp's: the forward pass, HIP kernels and paged KV are
  not Strata's. See [Engine notes](docs/LEARNING.md) and the
  [roadmap](#roadmap).
```

- [ ] **Step 4: Verify with the checker**

Run: `powershell -File scripts/check-docs.ps1`
Expected: exit 0, no broken links, README fences all labelled.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as the repo storefront"
```

---

### Task 6: Final acceptance

**Files:** none created; verify all.

- [ ] **Step 1: Run every acceptance command**

```powershell
cargo test
npm run build --prefix web
strata doctor
powershell -File scripts/check-docs.ps1
powershell -File scripts/test-check-docs.ps1
```

Expected: all pass; `strata doctor` may report "server not reachable", which is acceptable with no server running.

- [ ] **Step 2: Confirm no behaviour change**

Run: `git diff 02e1071..HEAD --stat`
Expected: only `README.md`, `CONTRIBUTING.md`, `docs/**`, `examples/**`, `.github/**`, `scripts/**`.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "docs: fix final storefront check failures"
```

(Only if Step 1 or 2 found something.)
