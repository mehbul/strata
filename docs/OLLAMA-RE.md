# Ollama reverse-engineering dossier

What Strata can learn from the Ollama installed on this machine. Everything
below is either REA evidence from the shipped artifact, an observation of the
running local server, or a public API declaration. Observation, inference and
unknown are kept separate on purpose.

Target: `C:\Users\Mehbul Islam\AppData\Local\Programs\Ollama\ollama.exe`
SHA-256 `4ccefcbfecc50fb6526471adabbc20869dd614e29e351120b15083cb174b0b8a`
(41,268,104 bytes). REA was run through its CLI fallback because the MCP
server is not registered; the CLI was run with Node 24.22.1 via npx.

## 1. Evidence ledger

| Operation | REA Evidence ID | What it establishes |
|---|---|---|
| `inspect_go_binary` | `ev_674af54e4c8509a3fc3319e4e8961e0fffc69f30dcb2cd9e7d5f9f97aff7a6f0` | Go 1.26.10, module `github.com/ollama/ollama` v0.40.16, 72 deps, build flags, VCS revision `d63b7026…` (2026-11-08) |
| `inspect_pe_imports` | `ev_a9aa30a129c622bfb58528d6a1363aa90db7c0ab053435f48cb49b349965a20f` | 10 PE imports (KERNEL32 + CRT only), 117 KERNEL32 functions; `CreateFileMappingW`, `MapViewOfFile`, `LockFileEx`, named-pipe-capable file APIs |
| `inspect_go_binary --include-string-literals` | `ev_fd1f761e952f71b4f62a0f7c535bb59cc9343794e32e8304fd4dd67daed95595` | 14,626 inferred static strings; route strings, env names, registry/manifest/compaction/thinking vocabulary |
| `inspect_model_artifact` on a local GGUF blob | exported locally as `ollama-model.json` | header keys: `general.architecture=qwen35`, `general.size_label=9B`, `general.file_type=15`, `qwen35.context_length=262144`, `qwen35.embedding_length=4096`, `general.license`, no `general.parameter_count` |

REA limitations preserved: string literals are *inferred* from static
`(ptr, len)` headers, so the route list is a lower bound; compiler-inlined
strings and strings in the companion app/runners are absent. PE imports do
not include Winsock because Go's `net` package resolves `ws2_32.dll` at
runtime — the absence of `WS2_32` in the import table is not evidence of no
network server. Static analysis did not execute the binary.

Live observations (not REA): `ollama --version` → 0.40.16;
`GET /api/version`, `GET /api/tags`, `GET /api/ps`, `POST /api/show` answered
on `127.0.0.1:11434`; three models are installed. Public API declaration:
`docs/api.md` at `github.com/ollama/ollama`.

## 2. What Ollama actually is

- **Go 1.26.10** main module with **Gin** (`gin-gonic/gin` v1.12.0) as the
  HTTP framework, `gin-contrib/sse` for streaming, `mattn/go-sqlite3` for
  state, `quic-go` for remote/transport traffic, `mongo-driver/v2`, Cobra for
  the CLI, Bubble Tea for the interactive prompt, `goccy/go-yaml`,
  `pelletier/go-toml`, `golang.org/x/image`, and `x448/float16`.
- **CGO_ENABLED=1**, so the matrix work is native code loaded beside the Go
  process, not a separate `llama-server` child. Strata instead owns a
  llama.cpp compute process. This is the single biggest architectural
  difference; it is deliberate and documented in `README.md`.
- **Content-addressed store**: `~/.ollama/models/blobs/sha256-<digest>`,
  `manifests/`, `manifests-v2/`, `metadata/`. A model is a manifest naming
  layers; the GGUF is one layer. `strata hub` already reads this store to
  copy a blob out, but Strata has no writable manifest/blob layer.
- **Scheduler**: `OLLAMA_MAX_LOADED_MODELS`, keep-alive based residency, and
  `/api/ps` expose loaded models. Strata is single-slot by design; `/api/ps`
  reports one model and `/model/unload` gives the GPU back.

## 3. Route surface (observed + declared)

The table below is the **gap analysis at the start of the work**. The current
state is in `docs/PLAN.md`: tags/show/ps metadata, `/v1/completions`,
`/v1/responses`, embed, copy/delete/create/pull/push, blobs and the CLI are now
implemented. Rows describing them as missing are historical.

Ollama 0.40.16, union of recovered strings, live probes and public docs:

| Endpoint | Ollama | Strata today | Gap |
|---|---|---|---|
| `POST /api/generate` | yes | yes | `think`, `format`, `suffix`, `raw`, durations, `context`, `keep_alive` not handled |
| `POST /api/chat` | yes | yes | `think`, `message.thinking`, `tool_name`, `keep_alive` not handled |
| `GET /api/tags` | yes | yes | Strata invents `"MoE"`, `"see \`strata inspect\`"`, `strata-<id>`; no families/quant/context/embedding |
| `POST /api/show` | yes | yes | Strata omits `modelfile`, `template`, `parameters`, real `details`, `model_info`, `projector_info`, `capabilities` |
| `GET /api/ps` | yes | yes | Strata's details are thinner, digest is `"strata"` |
| `GET /api/version` | yes | yes | Strata reports Cargo version (fine) |
| `POST /api/create` | yes | no | no Modelfile/derived-model creation |
| `POST /api/copy`, `DELETE /api/delete` | yes | no | no tag mutation |
| `POST /api/pull`, `POST /api/push` | yes | CLI `pull` only | no API surface; push needs credentials |
| `HEAD/POST /api/blobs/:digest` | yes | no | no blob store |
| `POST /api/embed` (and legacy `/api/embeddings`) | yes | no | no embedding endpoint |
| `POST /v1/chat/completions` | yes | yes | — |
| `POST /v1/completions` | yes | no | legacy OpenAI clients break |
| `POST /v1/messages` | yes | yes | Anthropic-compatible |
| `POST /v1/responses` | yes (recovered string) | no | OpenAI Responses API clients break |

Debug strings also show `ollama_compaction` / `ollama_compaction_summary`,
`relax_thinking`, `response.compaction`, `RENDERER`/`PARSER` fields, and
`capabilities` entries including `vision`, `tools`, `thinking`, `decision`.
Those are design targets, not proof of a stable API contract.

## 4. Metadata: what the GGUF really offers

REA header inspection of a real installed model shows no
`general.parameter_count`. The fields that can be reported honestly are:

- architecture, name, basename, size label, license from `general.*`;
- context length, embedding length, heads, block count from `<arch>.*`;
- quantisation from the tensor type distribution (`by_type()`), not from a
  guessed `file_type` mapping — this model declares `file_type=15` but its
  tensors are Q4_K + Q6_K, which is what actually makes it Q4_K_M;
- parameter count by summing tensor element counts — this is an estimate of
  stored parameters and must be labelled as derived.

Ollama's live `/api/show` for this model reports `9.2B`, `Q4_K_M`,
`families: ["qwen35"]`, `embedding_length: 4096`, `context_length: 262144`,
`runner: "llamacpp"`, `capabilities: ["completion","vision","tools","thinking"]`.
Strata must not copy the capability list: it has no projector unless
`--mmproj` was given and no verified thinking renderer. Derive it.

## 5. Gap priority for Strata

1. **Honest metadata** in `/api/tags`, `/api/ps`, `/api/show` — clients gate
   features on these fields. Currently Strata returns fabricated values,
   which is worse than an omission because it is believed.
2. **Dialect coverage** — `/v1/completions`, `/v1/responses`, `/api/embed`.
3. **Request semantics** — `think` forwarding, `keep_alive` (including the
   `0` unload idiom), `format` forwarded to the compute process.
4. **Model management** — `/api/copy`, `/api/delete`, `/api/create`,
   `/api/blobs/:digest`, Modelfile parsing, CLI `show`/`cp`/`rm`.
5. **Registry push/pull API** — large; pull can reuse `hub::Hub`, push needs
   credentials and is deliberately deferred.

Constraints that stay: no forward-pass rewrite, no disk expert tier, single
KV slot, and the simulated expert map stays labelled simulated.

## 6. Late findings from parity testing

Two details were only established by running Strata against the live Ollama
OpenAI endpoint on this machine:

- A thinking model can put its entire answer in `reasoning` and leave
  `content` empty. Translators that only know `reasoning_content` report an
  empty answer. Strata now checks `content`, `reasoning_content`, then
  `reasoning` and skips empty strings.
- `general.file_type` is the reliable quantisation label. Ollama's
  `fs/ggml/type.go` numbering was checked against two installed models:
  `15 → Q4_K_M` and `27 → IQ3_M`. Both match Ollama's live `/api/show`.
  Strata uses that table first and falls back to tensor-type distribution.
- The chat templates on this machine accept `low`, `medium` and `xhigh` for
  `reasoning_effort`; Ollama's public `high`/`max` are mapped to `xhigh`
  instead of being passed through and raising a template error.

## 7. Verification used

REA CLI: `npx -y -p node@24.22.1 -p rea-agents@latest -c "rea <op> <path> --format json"`.
Strata: `cargo test` and `cargo build --release` after each implementation stage.
