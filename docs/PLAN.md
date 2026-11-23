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

## Ollama parity pass (REA-driven, 2026-11-23)

Evidence and gap analysis: `docs/OLLAMA-RE.md`.

- Honest Ollama metadata: DONE — `/api/tags`, `/api/ps` and `/api/show` report
  GGUF-derived parameter size, quantisation level, families, context, embedding
  length and file mtime. `strata show` prints the same object.
- Dialects: DONE — `/v1/completions` and `/v1/responses` (including streaming
  events) translate over the chat endpoint; `/api/embed` and `/api/embeddings`
  forward to llama.cpp with a `/embedding` fallback.
- Request semantics: DONE — `think`, `message.thinking`, `tool_name`, `format`
  and `keep_alive` are handled on the Ollama path. Keep-alive unloading drains
  in-flight turns before killing the compute process.
- Model management: DONE — `/api/copy`, `/api/delete`, `/api/create`,
  `/api/pull`, `HEAD/POST /api/blobs/:digest` and CLI `cp`/`rm`/`show`.
- CLI parity: DONE — `create -f Modelfile`, `run` (one-shot and interactive with
  slash commands and `--verbose` timings), `stop`, `ps`, `push`, `signin`,
  `signout`, `launch`, positional names and global `--verbose`/`--nowordwrap`.
  Modelfile SYSTEM/PARAMETER entries are applied at the API boundary, and
  `show` reports them back in a generated Modelfile.
- Response timings: DONE — Ollama's `total_duration`, `eval_duration` and
  friends are emitted in nanoseconds on both streaming and non-streaming paths.
- Second parity pass: DONE — Modelfile TEMPLATE is passed to llama.cpp via
  `--chat-template`; `create` with `files` builds a model from uploaded blobs;
  Ollama env vars (`OLLAMA_MODELS`, `OLLAMA_CONTEXT_LENGTH`,
  `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_API_KEY`, `OLLAMA_DEBUG`,
  `OLLAMA_HOST`) are honoured; capabilities are derived from the chat template;
  `/api/ps` reports `expires_at`; `run` gained `/set system`; `generate`
  accepts `suffix` and `raw`; logical `team/model:tag` names are stored in
  `tags.json`; `launch` knows common integrations.
- Multi-model residency: DONE — with `OLLAMA_MAX_LOADED_MODELS>1`, additional
  models are loaded into a resident pool, each with its own llama.cpp process
  and port. Requests are routed by model name, the least-recently-used resident
  is evicted when the pool is full, and `/api/ps` lists every resident with its
  expiry. Default capacity is 1, so single-GPU behaviour is unchanged.
- Final local-use pass: DONE — resident capacity is derived from VRAM when
  `OLLAMA_MAX_LOADED_MODELS` is unset; keep-alive deadlines are refreshed on
  use and can expire an individual resident; `/api/pull` streams real NDJSON
  progress from the hub; `tune` now measures a family of non-prefix `-ot`
  expert placements generated from `ExpertPlan`, not a single regex.
- Dequantizer: DONE — `src/dequant.rs` decodes Q4_K and Q6_K blocks following
  llama.cpp b10679, with reference vectors and a `strata dequant` command that
  reads real tensors out of a GGUF.
- Tokenizer: DONE — `src/tokenizer.rs` builds a byte-level BPE from the file's
  own tokens, merges and BOS/EOS flags. Verified against the bundled
  `llama-tokenize.exe` on several prompts (identical token ids).
- Web console: DONE — the inspector has a Models pane for listing, showing,
  copying, deleting, pulling, pushing, creating from a Modelfile and embedding.
  The composer now controls think, reasoning effort, response format and
  keep-alive per request.
- Final gap pass: DONE — cancellation tokens on upstream calls, full JSON
  Schema validation with one retry, true `raw`/`suffix` via llama.cpp's
  `/completion` and `/infill`, device-code signin, system-tray controller,
  daemon watchdog with logon tasks, web metric meter, TLS crypto provider fix,
  wider safetensors metadata, and more route tests. TLS verified with HTTPS
  curl against a self-signed certificate; raw/suffix verified live.
- Final audit items: DONE — converter now emits tokenizer arrays from
  `tokenizer.json` and normalises llama-style architectures; POST/validation
  route tests added; tray process launch and device-code flow verified against
  a local mock; a real Windows SCM service is available via
  `strata service-windows --install` (requires Administrator). Tray icon itself
  cannot be visually verified in a headless session.
- Final fidelity pass: DONE — Ollama endpoints now stream raw NDJSON, default
  keep-alive is 5m, empty prompts implement load/unload, more Ollama options
  are mapped, generate accepts images, reasoning streams in `message.thinking`,
  `DELETE /api/connect` exists, create/push stream status lines, embed accepts
  multimodal objects, `OLLAMA_MAX_QUEUE`/`NUM_PARALLEL` are honoured, compaction
  is reported in responses, and push logs layer progress. The web console has
  Modelfile validation, resident metrics and JSON-schema input. TLS cert
  generation, safetensors conversion and daemon log rotation are included.
- Final completion pass: DONE — TLS via `STRATA_TLS_CERT`/`STRATA_TLS_KEY`,
  shell completions, Windows logon task, `export`/`import` bundles, upload
  timeout, `/api/keys`, schema validation marker, extra install flags.
- API/model polish pass: DONE — `/api/connect` remote endpoints, cached-token
  accounting, `vram` and chat `context` fields, authenticated registry pulls,
  upstream retry/backoff, structured-layer push, LoRA adapter and `--embeddings`
  flags, batch env vars, `list --json`, `models --verify`, `pull --dry-run`,
  `gc`/`rm --purge`, `run --image`, and `/set temperature|max-tokens|top-p`.
- Architecture-generic placement: DONE — dense models no longer get a hardcoded
  `-ngl 99`. `Placement::gpu_layers()` solves whole-layer offload from measured
  bytes, `strata tune` sweeps `gpu_layers` for dense models and `cpu_moe_layers`
  for MoE, and the saved value is applied at serve time. Tested live with a
  qwen2 dense model and the qwen35 hybrid models.
- Ollama registry Go templates are detected and not passed to llama.cpp's Jinja
  engine; the model's embedded template is used instead.
- Background daemon: DONE — `strata daemon start|stop|status` runs the server
  detached with a PID file and log under `~/.strata`; stop uses `taskkill /T`
  so no compute child is orphaned.
- Diagnostics and self-update: DONE — `strata doctor` checks binary freshness,
  runtime, GPU, models, console and server; `strata update` refuses dirty trees
  and otherwise pulls, rebuilds and refreshes the shim. The shim itself now
  rebuilds before running when sources are newer.
- Extra Ollama env vars: DONE — `OLLAMA_NUM_THREADS`, `OLLAMA_FLASH_ATTENTION`,
  `OLLAMA_GPU_OVERHEAD` and `OLLAMA_SCHED_SPREAD` map to runner flags/budgets.
- HTTP handler tests: DONE — three integration tests exercise version, ps and
  health handlers through axum's router with a constructed test state.
- Direct registry pull: DONE — `strata pull <model>` now downloads Ollama
  manifests and content-addressed layers without an Ollama install, verifies
  every blob's SHA-256, and reconstructs system/template/parameters/license
  into a derived model. `ollama:` prefix forces the registry for namespaced
  models; `OLLAMA_REGISTRY` overrides the endpoint.
- Final polish: DONE — richer `/v1/responses` streaming events, `/api/generate`
  context token ids, `projector_info`, numeric `model_info`, GGUF licence in
  `show --license`, Ollama-style `list`/`ps`, local tokenizer for compaction,
  conversation `/save`/`/load` in `run`, and MCP embed/version/details tools.
- Not built: real router telemetry (llama.cpp exposes no activations), the
  forward pass (gated-DeltaNet SSM scan + 256-expert routing) and HIP kernels
  (both are a separate engine project, not a missing Strata feature). Push
  talks Docker Registry v2 and needs a token. Device-code sign-in is built via
  `strata signin --device` with configurable OAuth endpoints.
