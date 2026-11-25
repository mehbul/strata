# Inference optimisation result — Strata / Inkling-2-Mini

Measured 2026-11-25 with the `inference-optimisation` skill's `bench.py` against
Strata's OpenAI-compatible endpoint at `127.0.0.1:8080`.

| Metric | Baseline (f16 KV, warm ctl-b) | Now (q8_0 KV) | Ceiling | How measured |
|---|---:|---:|---:|---|
| c1 decode | 19.86 tok/s | **23.17 tok/s** | 42 tok/s | median of 6 samples, 512 max tokens, temp 0.6 |
| c2 aggregate | 19.48 tok/s | **22.57 tok/s** | 99 tok/s | 2 streams, wall clock over both |
| Prefill ~4K | 218.5 tok/s | 218.3 tok/s | 2069 tok/s | server `prompt_tokens` / TTFT, cold unique doc |
| Prefill ~16K | 120.1 tok/s | **136.8 tok/s** | 1709 tok/s | as above, 12,900 actual tokens |
| Long prefill | 104.7 tok/s | **133.7 tok/s** | — | 12,837-token doc, 3/3 needles found |
| Startup | 7 s | 7 s | ~10 s (weights from page cache) | `strata serve` log |

Ceilings come from `THEORY.md`: 250 GB/s measured device copy, 26.7 TFLOPS
sustained FP16, 50% MFU assumption. Decode is bandwidth bound; prefill is
compute bound.

## What got the gain

One change: KV cache dtype f16 → `q8_0`, with the expert split held at 7 CPU
layers. Halving the KV cache (3.0 GiB → 1.5 GiB at ctx 32768) leaves more VRAM
for routed experts, so decode gains and long-context prefill gains. Control,
candidate, control were run in that order; the candidate beat both controls.

## What limits further gains

- Decode is ~55% of the bandwidth ceiling. The rest is quantized GEMV kernel
  efficiency and host-RAM expert traffic; Strata does not own the forward pass.
- Prefill is ~8–10% of the compute ceiling. At ctx 32768 most of the 13.76 GiB
  of routed experts cannot be VRAM-resident on a 16 GB card.

## Follow-up: q8_0-specific re-tune

`strata tune --ctx 32768 --kv-type q8_0 --save` confirmed the same 7-layer split
is optimal and persisted it as `tuned-32768-q8_0.json` (24.0 tok/s decode,
234.9 tok/s prefill, 56.2 s turn under tune's harness, versus 23.4 / 111.4 /
95.4 s for the f16 tune). No config change resulted, so no new A/B was needed.
The server now loads that file automatically without an explicit `--cpu-moe`.

## PolyStrata study — later A/Bs (all on the q8_0 base)

| Change | Leading control | Candidate | Trailing control | Verdict |
|---|---:|---:|---:|---|
| `--moe-cache-mib 512` | 19.28 | 13.18 | 19.36 | **Rejected** (−32% c1, −22% 16K prefill) |
| `--load-mode none` | 19.36 | 21.05* | 16.34 | **Rejected as inconclusive** — bimodal samples, c2 −9% vs leading control, controls drifted −15.6% |

\* median of a bimodal set (22.9, 22.9, 22.8, 19.3, 18.9, 19.0). Not a stable win.

Their technique that *did* transfer was already applied before the study: q8_0
KV. Their usage-driven expert cache, when expressed through llama.cpp's
`--moe-cache-mib`, made this model much slower on RX 7600; their host-resident
expert idea, expressed through `--load-mode none`, could not be measured
reliably because the host drifted. The custom kernel levers
(`--attn-requant`, `--loop-bf16`, `--tail-skip`, int8/FP4 prefill) have no
equivalent in llama.cpp and remain out of scope — Strata does not own the
forward pass.

## Known limitations / not verified

- `q8_0` KV is a **lossy** change. One greedy prompt matched f16 token-for-token
  (84 tokens) and all 3 needles were retrieved exactly, but broad output
  equivalence was not tested.
- Controls drifted 21.32 → 19.86 tok/s on c1 between the first and last run
  (cold start on the first); conclusions use the warm trailing control.
- No concurrent full-KV soak, no speculative decoding candidate exists for
  this model, and no kernel-level profiling was done.
- The first lever from the skill's list (re-tune the split for the new KV type)
  is queued, not done: `--kv-type q8_0` changes the VRAM budget, so 7 CPU layers
  may no longer be optimal.

## System state changed

- `strata/hillclimb/` added (PLAN.md, THEORY.md, RESULT.md, ledger.jsonl,
  configs/, best-config.json, model-hardware.json).
- Server running with `--ctx 32768 --kv-type q8_0` on port 8080, using
  `tuned-32768-q8_0.json` (split 7) automatically.
- No power-profile or driver settings changed. Old runtime backup remains at
  `%TEMP%\strata-runtime-b10679` from the earlier upgrade.
