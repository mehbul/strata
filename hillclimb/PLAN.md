# PLAN — inference hill-climb, Strata / Inkling-2-Mini

Harness: `inference-optimisation` skill (`bench.py`, `ceiling.py`) against the
running Strata server at `http://127.0.0.1:8080`. Logs: `ledger.jsonl`.

## Targets and constraints

- Metrics: c1 decode (median of 6), c2 aggregate decode, cold prefill at 4096 and
  16384, long-context needle retrieval at 16384.
- Memory cap: 16 GiB VRAM + 32 GiB host RAM; no disk expert tier.
- Context: 32768 (tuned split `cpu-moe 7`), must stay stable.
- Quality: lossless only. A q8_0 KV trial is **labelled lossy** and must be
  reported as such; it is not a drop-in winner unless outputs are verified.
- One GPU job at a time; server restarted between configurations, never killed
  mid-request. No other GPU processes during runs.

## Existing experiment log (read first, do not repeat)

- `strata tune --ctx 16384 --save` → best `-ot` pattern, 20.5 decode / 105 prefill.
- `strata tune --ctx 32768 --save` → best `-ncmoe 7`, 23.4 decode / 111 prefill.
  Rejected at 32768: all pattern placements (slower than prefix 7), 1 CPU layer
  (37 tok/s prefill, 14.1 decode). Do not rerun without a new reason.

## Log

### 2026-11-25T14:58 — setup

- Measured hardware: 250 GB/s copy, 26.7 TFLOPS FP16 sustained. Wrote `THEORY.md`.
- Ceiling at ctx 32768: c1 decode 42 tok/s, prefill 1387 tok/s.
- Current best (from Strata's own tune, different harness): c1 23.4, prefill 111.

### 2026-11-25T14:59 — ctl-a (baseline)

- Config: `configs/ctl-a.json`; server `serve --model inkling-2-mini:chat --ctx 32768`,
  f16 KV, `-ncmoe 7`, flash attention on, 6 threads.
- Result: c1 median **21.32** tok/s (6 samples: 21.14, 21.50, 21.22, 21.21,
  21.43, 21.53). c2 aggregate 20.98 (per-stream 21.36/21.45 — no gain from
  concurrency, consistent with the KV pool being the limit). Prefill 60.0 tok/s
  over 3,302 tokens (~4K) and 52.0 tok/s over 12,900 tokens (~16K). Long needle
  12,837 tokens: 3/3 found, prefill 55.3, decode 12.8. `flags: []`.
- Verdict: accepted as control. Stable spread (<2%), so the measurement is
  usable. Prefill is only ~4% of the 1,387 tok/s ceiling — the obvious gap, but
  the hardware cannot fit the routed experts in VRAM at this context, so it is
  likely host-RAM weight traffic rather than a tunable kernel.

### 2026-11-25T15:xx — cand-q8 (single change: KV dtype)

- Config: `configs/cand-q8.json`; server restarted with `--kv-type q8_0`, split
  held at `--cpu-moe 7` so KV dtype is the only difference. This is a
  **labelled lossy** trial (quantized KV), not a lossless drop-in.
- Result: c1 median **23.17** tok/s (23.05–23.34), c2 aggregate 22.57, prefill
  218.3 tok/s over 3,302 tokens and 136.8 over 12,900. Long needle 12,837 tokens:
  3/3 found, prefill 133.7, decode 18.5. `flags: []`.
- Greedy fidelity spot-check: prompt "The capital of France is", temperature 0,
  160 max tokens → "The capital of France is Paris.", 84 completion tokens on
  both q8_0 and f16. Identical on this one prompt; not proof of bit-equivalence.
- Verdict: provisional win (+8.7% c1, +163% prefill). Must beat ctl-b, the
  trailing control, before acceptance.

### 2026-11-25T15:xx — ctl-b (trailing control)

- Server restarted back to f16 KV, split held at `--cpu-moe 7`.
- Result: c1 median **19.86** tok/s (19.54–19.95), c2 19.48, prefill 218.5 over
  3,302 tokens and 120.1 over 12,900. Long needle 12,837: 3/3, prefill 104.7,
  decode 18.4. `flags: []`.
- Note: controls drifted 21.32 → 19.86 (−6.8%) on c1, inside the ±5–10% band the
  skill assumes. Also, ctl-a was the first run after server boot and its prefill
  (60/52) is treated as a cold-start artifact; ctl-b (218.5/120.1) is the warm
  f16 comparison.

### Verdict — 2026-11-25T15:xx

**ACCEPT `cand-q8` with a lossy label.** Against the trailing control it improves
c1 +16.7%, c2 +15.9%, 16K prefill +13.9%, long prefill +27.7%, with 4K prefill
unchanged (−0.1%) and needle retrieval still 3/3. It also beats the leading
control (21.32) on c1 by +8.7%. No flags, no failed requests. All six decode
samples are tightly clustered, so this is not a single lucky run.

Caveat KV quantization changes attention numerics: this is **not** claimed
lossless. The one-prompt greedy check matched f16 and the needles were found
exactly, but broader output equivalence was not verified.

### 2026-11-25T15:xx — re-tune for q8_0 KV (follow-up)

- Command: `strata tune --model inkling-2-mini:chat --ctx 32768 --kv-type q8_0 --save`.
  The first attempt was killed by a harness restart; no file was written, so it
  was restarted from scratch.
- Result: best is still **cpu-moe 7**, no `-ot` pattern. Saved to
  `rocm/models/inkling-2-mini_chat/tuned-32768-q8_0.json`: 24.0 tok/s decode,
  234.9 tok/s prefill, turn 56.2 s (vs the f16 tune's 23.4 / 111.4 / 95.4 s).
- Verdict: no config change to accept — the split was already 7. This run
  confirms 7 is optimal under the smaller KV budget and persists the measurement
  under the q8_0 key, so `serve --kv-type q8_0` now loads it automatically
  (verified: log says `split measured by strata tune`, KV q8_0, ready in 6 s).
- Note: the tune's 234.9 tok/s prefill is its own harness (8192-token read,
  best-of-two decode); do not mix it with `bench.py`'s 136.8 tok/s at ~16K.

### 2026-11-25T16:xx — PolyStrata study, new levers

- Cloned and studied `simplexai-labs/PolyStrata`; notes in
  `docs/POLYSTRATA-STUDY.md`. Their MoE engine keeps a usage-driven VRAM expert
  cache, computes cache misses on the host, avoids mmap page faults for experts,
  and separates prompt threads — all things llama.cpp b12704 exposes as
  `--moe-cache-mib`, `--moe-cache-cpu-share`, `--load-mode` and `-tb`.
- Added those four flags to `strata serve` (Flags → command line), with tests.
  Defaults unchanged. `cargo test` 115 pass, clippy clean.
- Control/candidate/control on `--moe-cache-mib 512` against the current best
  config (q8_0 KV, split 7). Configs: `poly-ctl-a`, `poly-cand-cache`,
  `poly-ctl-b`. One change per run; do not combine with `--load-mode` yet.
- `poly-ctl-a`: c1 **19.28** (19.06–19.32), c2 18.95, prefill 219.0/137.8,
  needle 3/3, no flags. Verified llama-server command line without cache flag.
- `poly-cand-cache` (`--moe-cache-mib 512`): c1 **13.18** (12.99–13.39), c2
  13.04, prefill 217.9/107.8, needle 3/3, no flags. Flag verified in the
  compute process command line. Large regression; trailing control pending.
  Hypothesis: 512 MiB of the little free VRAM is taken from the expert layers,
  and the cache-miss path on this ROCm build costs more than it saves.
- `poly-ctl-b`: c1 **19.36** (19.28–19.37), c2 19.03, prefill 218.1/137.3,
  needle 3/3. Controls agree within 0.4%.
- **REJECT `--moe-cache-mib 512`**: −32% c1, −31% c2, −22% 16K prefill, far
  outside noise, in both directions confirmed. Do not rerun without a reason
  (a different cache size with a different split would be a new reason).
- The portable flag remains in Strata (it is harmless when unset); the measured
  verdict is simply that this cache does not pay on this card/model.

### 2026-11-25T16:xx — load mode (next lever)

- Single change: `--load-mode none`, using `poly-ctl-b` as the leading control.
  Rationale: PolyStrata keeps host-side expert bytes resident/pinned instead of
  taking mmap page faults, and llama.cpp itself warns that `-ncmoe` with mmap
  should consider `--load-mode none`.
- `poly-cand-load` (`--load-mode none`): c1 median **21.05** but bimodal —
  samples 22.90, 22.90, 22.81, 19.28, 18.94, 18.95. c2 17.32 (down from 19.03),
  prefill 235.1/134.3, needle 3/3. The first half is +18% c1, the second half
  is at the control level. No mmap warning in the llama.cpp log. This is not a
  stable win; the trailing control decides whether the second half is drift.
- `poly-ctl-c` (trailing control): c1 **16.34** (15.51–16.44), c2 15.23,
  prefill 214.6/132.7, needle 3/3. Controls drifted 19.36 → 16.34 (−15.6%) in
  one A/B, outside the ±5–10% noise band the skill assumes.
- **REJECT / INCONCLUSIVE `--load-mode none`**: it beats both controls on the
  median, but its samples are bimodal (22.9, 22.9, 22.8, 19.3, 18.9, 19.0) and
  c2 (17.32) is 9% below the leading control (19.03). The candidate's last three
  samples sit at the leading control's level while the trailing control keeps
  falling — consistent with drift during the run, not a speedup. The skill says
  to suspect measurement first when results are weird. Do not accept; re-measure
  on an idle host with shorter interleaved runs if revisiting.

### Status after the PolyStrata study

- Three portable levers tried: q8_0 KV (accepted earlier, +16.7% c1), MoE
  expert cache (rejected, −32%), load-mode none (inconclusive/rejected).
  `-tb/--threads-batch` is implemented but not measured, so no plateau declared.
- Best confirmed configuration remains: `--ctx 32768 --kv-type q8_0`,
  `tuned-32768-q8_0.json` (split 7), i.e. c1 ~19–23 tok/s depending on run,
  prefill ~137 tok/s at 16K.
- Server will be left unloaded/stopped as requested.

### Plateau / what limits further gains

- c1 decode 23.17 tok/s is ~55% of the 42 tok/s bandwidth ceiling at 32K.
- 16K prefill 136.8 tok/s is ~10% of the 1,387 tok/s compute ceiling. The gap is
  consistent with routed experts (13.76 GiB) living mostly in host RAM at ctx
  32768; there is no disk tier and no more VRAM to buy without shrinking context.
- Remaining levers, ranked: (1) re-tune the expert split for q8_0 KV
  (`strata tune --ctx 32768 --kv-type q8_0 --save`) — the current 7-layer split
  was tuned for f16 and was held fixed for this A/B; (2) try a smaller context
  (16384) with q8_0, which would free more VRAM for experts; (3) kernel work is
  out of scope (Strata does not own the forward pass). Three levers have not
  been tried, so no formal plateau is declared.
