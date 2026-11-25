# PolyStrata study — what transfers to Strata

Cloned `https://github.com/simplexai-labs/PolyStrata` and read the "How fast is
it?" README section, `docs/BACKENDS.md` (engine architecture, "What was tuned",
"Less of the PC's RAM and VRAM"), `docs/models/inkling.md`, and the design notes
`BATCHED_DMA.md`, `KV_PREFETCH.md`, `DISJOINT_EXPERT_CACHE.md`, `EXPERTDEC.md`,
`VRAM_ELASTIC.md`.

## What they actually built

PolyStrata is two engines behind one OpenAI-compatible server:

- `strata` (Niko's) for Qwen3.8 / Hy4.
- `strata-poly`, a custom C++/CUDA/HIP engine for thirteen other architectures,
  including `inkling2_loop`. It reads a GGUF in place, keeps dense weights in
  VRAM, keeps routed experts in a **usage-driven VRAM cache** with the rest on
  host, computes cache misses on the CPU concurrently with GPU work, loads a
  prompt batch through a staging buffer, uses CUDA graphs for decode launches,
  and keeps conversation checkpoints so follow-ups skip re-prefill. An
  architecture supplies hooks (`load_layer`, `route`, `layer_head/tail`, …);
  the shared engine asks no model-specific questions.

Their Inkling numbers (RTX 4070 Laptop 8 GB): **53 tok/s decode, 892 tok/s
prompt**. Our RX 7600 XT 16 GB: 23.4 / 111 (Strata's tune) or ~23 / 137
(`bench.py` at ~16K). Different cards and harnesses, but the gap is large enough
that the mechanisms matter.

## Technique → what we can do here

| PolyStrata technique | Transferable to our stack? | Action |
|---|---|---|
| Usage-driven VRAM expert cache, misses computed on CPU | **Yes**, llama.cpp has `--moe-cache-mib` and `--moe-cache-cpu-share` | Added to `strata serve`; A/B next |
| Experts read by page fault from mmap | **Yes**, llama.cpp `--load-mode none` (our own serve log already warns about mmap + CPU expert overrides) | Added to `strata serve`; A/B next |
| Separate prompt/batch threads | **Yes**, `-tb/--threads-batch` | Added to `strata serve`; A/B next |
| `--kv q8_0` / fp8 KV | **Already done** | Measured +16.7% c1, +163% 16K prefill vs f16 |
| Prompt/conversation cache, warm start | **Partly** — llama.cpp prompt caching is on (`cached_tokens > 0`); we have compaction | No code change; keep measuring cache hits |
| `--attn-requant q5_0`, `--loop-bf16`, `--tail-skip` | **No** — custom kernels inside their forward pass; `--tail-skip` deliberately omits low-ranked experts (lossy) | Out of scope: Strata does not own the forward pass |
| int8 / NVFP4 prefill tensor cores | **No** on llama.cpp ROCm for this model | Kernel project, explicitly out of scope |
| Batched DMA submission, KV prefetch, async expert swaps | **Partly** conceptually (overlap host→device with compute), not as a patch | Future only; their DMA path is CUDA 13 |
| `--ram-experts-mib`, pinned/locked experts, disjoint multi-GPU caches | **No** on llama.cpp/ROCm; we have one GPU and mmap/no-mmap | Not applicable |
| `POST /v1/vram` elastic cache | **No** — NVIDIA-only | Not applicable |
| Several conversations/parallel requests | Their Inkling forces one request; ours already uses `-np 1` | Already aligned |

## What changed in Strata

`serve` now accepts `--threads-batch`, `--load-mode`, `--moe-cache-mib` and
`--moe-cache-cpu-share`. They are threaded through `ServeOptions` → `Flags` →
the llama.cpp command line, with defaults of `None` so existing command lines
and tuned splits are unchanged. `runner.rs` has a test that all four reach the
command line. `cargo test` (115) and clippy `-D warnings` pass.

## What is not yet known

Nothing above is a proven speedup on this machine. The next step is the skill's
control/candidate/control protocol against the current best config
(`--ctx 32768 --kv-type q8_0`, split 7), changing one flag at a time. PolyStrata's
own documentation is careful about this: several of their experiments improved
submission latency or micro-benchmarks without improving model tokens/s.

Their published numbers are from different hardware and a different engine.
They are priors, not predictions for Strata.
