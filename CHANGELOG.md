# Changelog

Notable changes to Strata. This project follows [semantic
versioning](https://semver.org/): 0.x releases may break between minors.

## [Unreleased] — Ollama parity

- Full Ollama API and CLI compatibility: `/api/*`, `/v1/completions`,
  `/v1/responses`, model management, registry pull/push, blobs, daemon,
  doctor, update, config and global `strata` command.
- Architecture-generic inference: dense, hybrid and MoE models share the
  reader, tokenizer, registry and API; dense models get measured `-ngl`
  offload instead of `-ngl 99`.
- Own dequantizer (Q4_K/Q6_K) and byte-level BPE tokenizer, both verified
  against llama.cpp on this machine.
- Deterministic Ollama reverse-engineering dossier in `docs/OLLAMA-RE.md`.
- Web console: model management and per-request think/format/keep-alive/image
  controls.
- Fixed installer GPU detection: the ROCm runtime is provisioned with the
  `hipblas`/`rocblas` support DLLs it imports, and setup fails loudly instead of
  silently falling back to CPU.
- `strata bench` sizes its prompt with the model's own tokenizer and reports the
  requested and observed token counts.
- Clippy is clean with `-D warnings` on all targets.

## [0.1.0] — 2026-11-17

First tagged release. Strata is a control plane around llama.cpp: it reads the
GGUF, solves weight placement, tunes it by measurement, compacts context, and
serves an OpenAI-compatible API with a web console.

### Engine

- **GGUF reader** — architecture, layer and expert counts, quantisation, tensor
  index, and hybrid state-space metadata, all read from the file itself.
- **Placement solved from measured bytes** — dense weights, routed experts and
  KV cache are sized from the tensor index and divided across VRAM and system
  RAM. There is no disk tier; `--allow-disk` measures a model that would spill
  anyway.
- **Measured tuning** (`strata tune`) — sweeps the CPU-expert split over a
  realistic turn (read 8192 tokens, write 512), then re-measures either side of
  the winner. Results are stored per context as `tuned-<ctx>.json`, because the
  KV cache reserved at load is VRAM the experts cannot have.
- **Context compaction** — when a conversation outgrows the window, the middle
  is replaced with a summary the model writes. The cut is measured from the
  start of the conversation so the prompt prefix stays stable and llama.cpp's KV
  cache still holds it.
- **Hardware detection** — from the compute runtime's own device list, so it
  needs no vendor SDK and answers the same for ROCm, Vulkan and CUDA.

### Serving

- OpenAI-compatible `POST /v1/chat/completions`, SSE when `stream: true`.
- Tool calling forwarded to llama.cpp's own chat template (`--jinja`), so any
  OpenAI-compatible coding agent can drive it.
- Scheduler with admission control, queue timeout, and cancellation.
- `GET /metrics` in Prometheus text format.
- `GET /health`, `/profile`, `/compact`, `/experts` (simulated), `/api/models*`.
- API key required on any non-loopback address, compared in constant time.

### Commands

`setup`, `info`, `list`, `models`, `inspect`, `plan`, `tune`, `pull`, `serve`,
`switch`, `bench`, `mcp`.

### Model management

- `pull` fetches from Ollama's manifests or a Hugging Face repository, listing
  quantisations rather than choosing one for you.
- `switch` asks a running server to load another model, and rolls back to the
  previous one if the new model fails to load.

### Compute backends

- llama.cpp is the universal backend. Specialized engines are selected from the
  GGUF architecture when their binary is present in `runtime/<backend>/`, and
  fall back to llama.cpp when it is not.
- `serve` accepts `--draft`/`--spec` (speculative decoding), `--mmproj` (vision,
  auto-detected from the model directory), `--devices`/`--tensor-split`
  (multi-GPU), and `--backend`.

### Console

Chat with streaming markdown and syntax-highlighted code, saved conversations,
and an inspector covering model residency, context fill, scheduler, weight
placement, hardware, experts (simulated), profiling and setup.

### Known limits

- **The expert map is simulated.** llama.cpp exposes no router telemetry, so
  `/experts` and `/health` say `"simulated": true`. Real router data needs a
  patched compute backend.
- **Linux is compiled, not measured.** CI builds and tests it; every number in
  the README was measured on Windows 11 with an RX 7600 XT.
- **No disk tier for experts.** Anything past VRAM plus free RAM is the
  operating system faulting pages out of the mmapped file, which is worth a
  fraction of a token per second. The honest answer for a model that large is
  more RAM.

[0.1.0]: https://github.com/mehbul/strata/releases/tag/v0.1.0
