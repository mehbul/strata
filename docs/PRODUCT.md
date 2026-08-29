# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite (TypeScript), retained from the existing build pipeline: `npm run build`
emits `web/dist`, which the Rust server serves as static files. User-chosen.

## Users

One developer — the author — running the engine on their own Windows workstation
for local coding assistance. That is the case the product is designed around, and
the same person is the developer, the operator and the only user, so the UI is
simultaneously a chat client and an instrument panel for an engine that is still
being built.

It is no longer the only case the code admits. `serve` binds whatever address it
is given, and any address that is not loopback is refused unless an API key is
set, so a second machine on the same network is a supported configuration rather
than an accident. What that does *not* buy is tenancy: there are no accounts and
no per-user state on the server — one key, one engine, one four-slot queue, and
conversations stored in whichever browser opened the console. Anyone holding the
key holds the GPU.

## Product Purpose

Strata is a local LLM inference engine targeting ROCm on consumer AMD hardware.
It reads the model file itself, solves where the weights should sit across VRAM
and system RAM, starts and owns the compute process, and serves an
OpenAI-compatible API and a web interface, plus live telemetry on how it is
placing weights and spending time. Success is running a large mixture-of-experts
model on a 16GB consumer card at usable speed, and being able to see exactly why
it is fast or slow.

## Positioning

Treats VRAM and system RAM as one addressable hierarchy — *strata* — and places
mixture-of-experts weights across those tiers rather than requiring the model to
fit in VRAM, choosing the split by measurement instead of by rule of thumb. Runs
without a daemon and without Ollama: it owns the compute process directly.
Reuses the ROCm redistributables already installed with ComfyUI instead of
requiring a second ROCm installation, and detects the GPU by asking the compute
runtime rather than a vendor SDK.

## Operating Context

- Windows 11 Pro, AMD Ryzen 5 7600X (6C/12T), Radeon RX 7600 XT 16GB (gfx1102,
  RDNA3), 32GB DDR5, WD_BLACK SN770 NVMe.
- Compute runs on a llama.cpp ROCm build vendored under `runtime/` (~1.3 GB,
  fetched at setup, not committed), linked against rocBLAS/hipBLAS found in
  `STRATA_ROCM_BIN`, a ComfyUI ROCm SDK, or `rocm/hip-sdk`.
- Runs alongside a browser, ComfyUI and other desktop apps. Memory pressure is a
  live constraint: the 35B model puts the machine near its commit limit, so the
  UI must surface memory state rather than hide it.
- Target model: Ornith-1.5-35B-A3B — 41 layers, 256 experts per layer, top-8
  routing, one shared expert per layer, hybrid attention (full attention every
  4th layer, gated-DeltaNet elsewhere), Q4_K/Q6_K, 20.22 GiB on disk.
- Default context is 65536, chosen so the KV cache stays under a tenth of the
  card; the model declares 262144.

## Capabilities and Constraints

Working today:
- GGUF parsing and model facts read from the file, placement solved from measured
  bytes, measured tuning stored per context, compute-process lifecycle and flags,
  OpenAI-compatible API with SSE streaming, a 4-slot scheduler with real
  admission control, context compaction with a model-written summary, telemetry
  endpoints (`/health`, `/profile`, `/experts`, `/compact`), and API-key auth with
  constant-time comparison, a loopback-only default, and CORS limited to
  localhost origins.

Not Strata's, and the UI must not imply otherwise:
- The matrix multiplies belong to llama.cpp, vendored under `runtime/`. Strata
  owns everything around the forward pass and none of it.
- Expert routing shown in the UI is **simulated** — the compute process does not
  report which experts a token activated. The API marks it `"simulated": true`
  and the interface carries that distinction visibly.
- Profiling phase timers report zero until Strata's own kernels populate them,
  and the panel says so rather than drawing a breakdown of nothing.

Terminology: *tier* (VRAM/RAM placement), *expert* (one MoE FFN block), *routing*
(which experts a token activates), *slot* (a KV/scheduler session), *split* (how
many layers keep their experts on the CPU), *compaction* (replacing the oldest
part of a conversation with a summary).

## Brand Commitments

- Name: **Strata**. Renamed from the working name; the name refers to the memory
  hierarchy.
- Hard constraint: every part of the interface is original to this project. This
  is a correctness requirement, not a preference.
- Standing visual preference (user-set): the console follows the conventional
  developer-chat canon — Codex and ChatGPT are the named references and their
  craft level is the bar. Neutral near-black, minimal chrome, system font stack,
  conversation centred, secondary instrumentation collapsed out of the way rather
  than arranged around the conversation. Execute the canon straight: no metaphor
  layer, no smuggled quirk.
- Voice: precise and unembellished. The engine's own CLI states plainly what is
  not implemented; the UI holds the same standard, and so does the README.

## Evidence on Hand

Measured in this environment, not estimated:
- Detected hardware: RX 7600 XT, gfx1102, 16.0 GB VRAM, 6C/12T, 31.1 GB RAM.
- Engine process: ~17 MB RSS, 13 threads, flat under 8 concurrent requests. The
  compute process is what grows: ~17 GB resident for the 35B, against a system
  commit of 43.9/49.8 GB.
- Tuned splits, stored beside the model: at 8192 context, 16 CPU-expert layers →
  38.5 tok/s, 13.47 GB VRAM; at 65536, 15 layers → 352.5 tok/s prefill, 42.5
  tok/s decode, 35.3s turn, 14.96 GB VRAM; at 262144, 12 layers → 32.6 tok/s.
- Forcing every layer onto the GPU measures 18.9 tok/s — the slowest of the
  sweep.
- Compaction on a 128-message conversation at 65536: the turn that compacts goes
  56,916 → 24,198 prompt tokens and costs 199s to first token; every turn after
  it holds 24,198 tokens and answers in 0.3s.
- Ceiling reported by `setup` on this machine: ~35 GB of weights fit without
  touching disk.
