# Strata Improvement Plan

Based on research across PolyStrata, Niko1221/Strata, AirLLM, PowerInfer,
ExLlamaV2, rvllm, and llm-runtime-optimizer. Filtered to items that fit
Strata's architecture (Rust control plane + llama.cpp compute process on AMD ROCm)
and its MoE-on-consumer-GPU positioning.

## Guiding constraints (do not violate)

- Compute is llama.cpp (fetched at setup). We do NOT rewrite the forward pass.
- No disk tier for experts (design decision; `--allow-disk` is the escape hatch).
- Single-slot KV pool (`-np 1`) is intentional — see runner.rs comment. Do not
  flip to multi-slot without fixing the prefix-cache failure mode.
- Expert map stays honestly labeled simulated until real router telemetry exists.

## Phase 1 — High value, low risk (Tier 1)

### 1.1 Metrics / observability endpoint
- Add `GET /metrics` (Prometheus text format) exposing: request count, in-flight,
  prompt/completion tokens, TTFT, decode tok/s, KV usage, VRAM/RAM, model name,
  expert-cache hit rate (if reported by compute).
- Add counters updated in chat_completions admission/completion path.
- Files: server.rs (new handler + state fields), runner.rs (parse compute metrics
  if available). Tests: unit test for format.

### 1.2 Built-in benchmark command
- Add `strata bench --model <m> [--prompt-tokens N] [--output-tokens N]` subcommand.
- Reuse chat_completions with greedy sampling; measure TTFT, prefill tok/s,
  decode tok/s, peak VRAM/RAM, cache hit.
- Print table + optional --json. PolyStrata's `benchmark` is the reference.
- Files: new src/bench.rs, main.rs (Bench variant), reuses server client.

### 1.3 Multi-model registry + switching
- Registry: scan rocm/models (and config dir) for `strata-<name>.json` run configs;
  parse GGUF header for architecture/params/quant.
- `GET /v1/models` (already) lists all installed; mark current.
- New: `GET /api/models`, `GET /api/models/current`, `POST /api/models/load`,
  `POST /api/models/unload` (PolyStrata contract).
- Service::switch_model: wait for in-flight (2 min timeout), stop compute, start
  new, swap tokenizer/template/stop/sampling, rollback on failure.
- CLI: `strata models`, `strata switch <name>`. Web console: model picker in header.
- Files: new src/registry.rs, server.rs (handlers + Shared state), web/src.

### 1.4 MCP server
- Expose tools: install/start/stop/status/list/bench over MCP (stdio or HTTP).
- PolyStrata ships one; reuse its tool surface.
- Files: new src/mcp.rs (feature-gated), main.rs (Mcp subcommand).

## Phase 2 — Hardware/model-gated (Tier 2)

### 2.1 Speculative decoding (optional, auto-select)
- Add `--draft <path>` and `--spec N` to Serve; forward to llama.cpp
  (`--model-draft`, `-sd`).
- Auto-select per card: benchmark MTP vs DFlash2 on first run, store choice in
  tuned-<ctx>.json. Default OFF (benchmarks show regressions on some cards).
- Files: runner.rs (flags), setup.rs (auto-select), tune.rs.

### 2.2 Vision input
- Detect mmproj in model dir; spawn `strata-vision` (llama.cpp mtmd) child process
  like PolyStrata. Accept image content parts in chat_completions.
- Only enable for models with vision encoder (MiniCPM-V etc.).
- Files: runner.rs (vision child), server.rs (content parts parsing).

### 2.3 Multi-GPU
- Add `--gpus` / `--layer-split` to Serve; map to llama.cpp `-dev` / tensor split.
- PolyStrata's GPUS env var pattern.

### 2.4 Hot-expert caching
- Move from layer-level cpu-moe to expert-level hot tracking in VRAM.
- Requires profiling which experts fire; store hot set per model.
- Files: expert.rs, placement.rs, tune.rs.

## Phase 3 — Defer (Tier 3, not now)

- Continuous batching: blocked on compute engine (llama.cpp single-slot).
- Layer streaming: contradicts "no disk tier" design.
- Faster prompt processing: upstream llama.cpp.
- Desktop app: separate project.
- Embedding endpoint: off-focus.
- Thinking-level control: trivial, low value; fold into 1.3 if cheap.

## Execution order

Start with 1.1 (metrics) + 1.2 (bench) — smallest, self-contained, high DX.
Then 1.3 (multi-model) — biggest Tier-1 payoff.
Then 1.4 (MCP).
Tier 2 only after Tier 1 ships and is tested.

## Verification

- `cargo build --release` + `cargo test` after each step.
- `strata bench` must produce stable numbers across 3 runs.
- Multi-model switch must roll back on failure (test by pointing at bad model).
  Verified against a live server: a failed switch leaves the previous model resident.
- Metrics endpoint must parse with `promtool check metrics` (if available).

## Status

- Phase 1.1 metrics endpoint: DONE (GET /metrics, /v1/metrics)
- Phase 1.2 bench command: DONE (strata bench)
- Phase 1.3 multi-model registry + API + CLI: DONE (registry.rs, /api/models*, strata models, strata switch).
  `switch` rolls back to the previously loaded model when the new one fails to load.
- Phase 1.4 MCP server: DONE (strata mcp)
- Phase 2 (speculative decoding, vision, multi-GPU, hot-expert caching): NOT STARTED

## Status update (Phase 2)

- Phase 2.1 speculative decoding: DONE (--draft, --spec flags forwarded to llama.cpp)
- Phase 2.2 vision input: DONE (--mmproj flag + auto-detect + content parts preserved)
- Phase 2.3 multi-GPU: DONE (--devices, --tensor-split flags)
- Phase 2.4 hot-expert caching: BLOCKED - requires router telemetry; llama.cpp exposes none (same reason /experts is simulated). Out of scope per 'no forward-pass rewrite' constraint.

## Phase 3 - specialized backends (DONE)

- Backend abstraction (src/backend.rs), auto-selection by GGUF arch, --backend override.
- Runtime discovery with fallback to llama.cpp when specialized binary absent.
- Compute process uses selected backend; prints backend name.
- Specialized binaries not bundled (license/size); drop into runtime/<backend>/ to enable.
