# GitHub storefront design

Goal: increase stars and visitors to `mehbul/strata` using **repository work only**.
No external posting, no paid promotion, no new hosted property.

## Scope

In scope:
- README rewrite (front door)
- `docs/index.md` landing page
- `examples/` copy-paste assets
- GitHub metadata script, issue/PR templates, CONTRIBUTING refresh
- `scripts/demo.ps1` reproducible benchmarks
- `docs/BENCHMARKS.md` single source of truth for numbers
- `docs/GROWTH.md` measurement and repo-side checklist
- `scripts/check-docs.ps1` link/code-fence check

Out of scope:
- Posting to HN/Reddit/Product Hunt/X
- GitHub Pages site
- New image generation (reuse `docs/social-preview.png`)
- Engine changes, new CLI features, behaviour changes

## Audience

Broad ("everyone, equally"), so the front door leads with one promise that works
for any local-AI reader: an Ollama-compatible local engine you can actually
tune, with an OpenAI/Anthropic API and a real web console. Secondary entry
points in the landing page route by intent (run a model, drive an agent, AMD
hardware, understand the engine) so specific audiences still find their path
without diluting the headline.

## Components

### 1. README.md

Top-to-bottom structure:
1. Title + one-sentence promise + badges (keep existing CI/license/Rust/platform/GPU).
2. 30-second block: install one-liner, `strata pull`, `strata serve`, console URL.
3. "Why Strata" — one short paragraph and a 3-column comparison table
   (Strata / Ollama / raw llama.cpp) covering control plane, API dialects,
   placement tuning and compute ownership. No claims without a doc link.
4. Measured performance: small table sourced from `docs/BENCHMARKS.md`, with
   hardware and exact command stated. Link to the demo GIF and console shot.
5. "What it can't do yet": simulated expert map, no disk expert tier, single
   KV slot, forward pass still llama.cpp's. Link to `docs/LEARNING.md`.
6. Navigation: docs landing, examples, commands, hardware, security.

Constraints:
- Keep the existing honesty about simulated experts and "kernels are not Strata's".
- Do not claim a number that is not in `docs/BENCHMARKS.md`.
- No new external links that cannot be checked offline by `check-docs.ps1`.

### 2. docs/index.md

A short routing page, not a duplicate README. Four links with one sentence each:
- "Run a model" -> README install section
- "Drive an agent" -> `examples/` OpenAI + Anthropic + MCP configs
- "AMD / ROCm" -> README hardware section
- "Understand the engine" -> `LEARNING.md`, `OLLAMA-RE.md`, `DESIGN.md`, `PRODUCT.md`, `PLAN.md`

### 3. examples/

Files (small, copy-pasteable, no secrets):
- `openai-client.py` — chat completion against `http://127.0.0.1:8080/v1`
- `anthropic-client.py` — messages API against the same server
- `Modelfile` — derived model with SYSTEM, PARAMETER, TEMPLATE
- `mcp-client.json` — stdio MCP config pointing at `strata mcp`
- `serve-tuned.ps1` — `strata serve` with tuned ctx/cpu-moe/kv-type and comments

Each file starts with a one-line comment stating what it assumes (server running,
model name, port).

### 4. GitHub metadata and templates

`scripts/github-setup.ps1` (idempotent):
- requires `gh` on PATH and an authenticated user; otherwise prints manual steps
- `gh repo edit --description <one-line promise>`
- `gh repo edit --add-topic ...` for: llm, rust, rocm, amd, ollama, moe,
  local-ai, openai-compatible, llama-cpp, inference-engine
- creates labels `good first issue`, `hardware`, `documentation`

`.github/ISSUE_TEMPLATE/`:
- `bug.yml` — reproduction, expected/actual, `strata doctor` output
- `hardware.yml` — GPU, VRAM, ROCm/Vulkan/CUDA, OS, model, `strata info` output
- `feature.yml` — problem, proposed change, alternatives

`.github/PULL_REQUEST_TEMPLATE.md` — links to CONTRIBUTING verification block.

`CONTRIBUTING.md` — add verification block:
`cargo test`, `npm run build --prefix web`, `strata doctor`,
`powershell -File scripts/check-docs.ps1`.

### 5. scripts/demo.ps1 and docs/BENCHMARKS.md

`scripts/demo.ps1` parameters: `-Model`, `-Url`, `-Out` (default
`docs/BENCHMARKS.md`), `-Ctx`, `-PromptTokens`, `-OutputTokens`.
Steps:
1. Check the server answers `GET /api/version`; if not, print the serve command
   and stop (no implicit GPU load).
2. Run `strata bench --url <url> --model <model> ... --json <tempfile>`.
3. Read the JSON, write a markdown table with: date, hardware line from
   `strata info`, model, ctx, prompt/output tokens, TTFT, prefill tok/s,
   decode tok/s, total ms, and the exact command used.
4. Append to `docs/BENCHMARKS.md` under a `## YYYY-MM-DD` heading, so the history
   is visible. Re-running on the same date replaces that date's heading rather
   than creating a duplicate.

`docs/BENCHMARKS.md` is the only place numbers live; README links to it and
embeds a short table copied from the latest entry.

### 6. scripts/check-docs.ps1

No dependencies. Checks:
- every relative markdown link target exists
- every fenced code block in README has a language tag
- `docs/BENCHMARKS.md` exists and has at least one table row
- README benchmark numbers appear in `docs/BENCHMARKS.md` (warn only)

Exit non-zero on hard failures; warnings printed but do not fail.

### 7. docs/GROWTH.md

Short document:
- Definition of "explodes": stars, forks, traffic sources, clones, issue quality,
  which docs are cited in issues.
- 7-day repo-side checklist: answer issues within 24h, tag good first issues,
  keep README numbers current, pin a "start here" issue.
- 30-day checklist: refresh benchmarks on hardware change, add one example per
  recurring question, prune dead links with `check-docs.ps1`.
- Explicitly states this is measurement, not promotion.

## Data flow

```
README / docs / examples  -->  check-docs.ps1  -->  CI-quality gate
strata bench  -->  scripts/demo.ps1  -->  docs/BENCHMARKS.md  -->  README table
gh CLI  -->  scripts/github-setup.ps1  -->  GitHub description/topics/labels
```

No runtime code is changed. No network calls except optional `gh` metadata setup.

## Error handling

- `demo.ps1`: server absent -> print command and exit 0 (no hidden GPU work);
  bench failure -> no markdown written, temp JSON removed.
- `check-docs.ps1`: missing target -> hard failure with file:line; number
  mismatch -> warning.
- `github-setup.ps1`: no `gh` or unauthenticated -> print exact manual commands
  and exit 0, never fail the repo.

## Verification

- `cargo test`
- `npm run build --prefix web`
- `strata doctor` (with no server running is acceptable)
- `powershell -File scripts/check-docs.ps1`
- `scripts/demo.ps1` against a running server (manual; the script refuses if absent)
- Review README first 80 lines as a first-time visitor would.

## Acceptance criteria

1. A new visitor can install and serve a model from the first screen.
2. Every claim in README has a link or a reproducible command.
3. Benchmark numbers exist in exactly one authoritative file.
4. Running check-docs reports no broken relative links.
5. No behaviour changes to Strata itself; only docs, examples and scripts.
