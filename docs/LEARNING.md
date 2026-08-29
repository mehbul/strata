# Engine notes

Why Strata is built the way it is. These are the principles the design follows —
see the table at the top of `README.md` for which parts are Strata's and which
are llama.cpp's, and the roadmap for what has not been written yet.

## The model these decisions were made against

Read from the file's own header with `strata inspect --path <gguf>`, not from a
model card:

| | |
|---|---|
| architecture | `qwen35moe` — hybrid: full attention every 4th layer, gated-DeltaNet in the rest |
| layers | 41 |
| routed experts | 256 per layer, top-8, 10,496 in the file |
| shared expert | one per layer, always active |
| parameters | 35B total, ~3B active per token |
| stored quantisation | Q4_K (379 tensors) + Q6_K (64), 20.22 GiB on disk |
| context / vocab | 262144 declared, 248,320 tokens |

Every number below follows from that table, so it is quoted from the file rather
than remembered.

## Principles

- **MoE sparsity makes placement the problem, not capacity.** Eight of 256
  experts fire per layer per token, and roughly 3B of 35B parameters are touched.
  Where a weight *lives* matters more than whether the whole model fits.

- **The routed experts are the only thing worth moving.** Attention, the SSM
  state, the shared experts, the embeddings and the LM head are small and touched
  on every token, so they stay on the card. The routed experts are most of the
  20 GiB and each one is touched a fraction of the time, so a placement decision
  is really a decision about them and nothing else.

- **Placement is measured, not reasoned about.** "Put as much on the GPU as
  possible" is the *slowest* configuration measured here: everything resident
  gives 18.9 tok/s, and the tuned split — fifteen layers keeping their experts in
  host memory — gives 42.5, because the experts that moved off the card freed
  VRAM for the layers that stayed. The capacity estimate in `Flags::derive`
  predicted 17 layers at 8192 context and measurement said 16. An estimate is a
  starting point for a sweep, not an answer.

- **Two rates, not one.** Prefill is compute-bound and wants experts on the GPU;
  decode is memory-bound and tolerates them in host RAM. A single tok/s figure
  hides the conflict, so `tune` minimises the wall time of a whole turn — read
  8192 unseen tokens, write 512 — and reports both rates. The surface is not
  smooth, so it re-measures either side of the winner.

- **Depth in the window costs more than the size of the window.** Allocating
  262144 instead of 65536 costs about 15%; *filling* it costs far more, because
  the 5 GiB KV cache leaves too little VRAM for experts. Compaction exists to
  keep a conversation on the flat part of that curve, not to fit more tokens.
  And because the cut is measured from the start of the conversation, the prompt
  prefix does not move, so the runtime's KV cache survives the next turn.

- **There is no disk tier, and saying so is part of the design.** Expert tensors
  are placed in VRAM or in host memory and nowhere else. Past that, what happens
  is the operating system faulting pages out of the mmapped file — and decode
  routes a different handful of experts per layer per token, which is a scattered
  read with almost no locality, bound by random reads rather than bandwidth and
  worth a fraction of a token per second. So `setup` refuses when more than half
  the experts would spill instead of spending hours loading the file once per
  configuration to arrive at the same conclusion. `--allow-disk` measures it
  anyway.

- **Never silently change precision or routing.** The file is stored Q4_K and
  Q6_K and Strata does not requantise it. When Strata grows its own
  dequantisation it is to be checked against the current runtime's output on the
  same machine, token for token, rather than against a similarity score — a
  plausible-looking drift is the failure that is hardest to notice afterwards.

- **The GPU is speed, not a requirement.** The same binary serves whatever split
  the machine measured, and the split is stored per model and per context
  (`tuned-<ctx>.json`), because the KV cache reserved at load time is VRAM the
  experts cannot have: a split measured at 8192 is wrong at 65536.

## Choices specific to this machine

- **Detect through the process that does the work.** `setup` asks the compute
  runtime itself for the device list — no vendor SDK, no Python, and the same
  view of the hardware the process doing the work will have. Anything the runtime
  cannot see does not exist, however confidently something else reports it.
- **Reuse the ROCm that is already installed.** The runtime links rocBLAS/hipBLAS
  but does not ship them, so Strata looks in `STRATA_ROCM_BIN`, then a ComfyUI
  ROCm SDK, then `rocm/hip-sdk`, rather than requiring a second ROCm
  installation.
- **Native HIP rather than Vulkan, once the kernels exist.** On RDNA3, ROCm gives
  access to WMMA and hipBLASLt that a Vulkan path cannot reach.
- **Context chosen so the KV cache stays under a tenth of the card,** because the
  cache is allocated in full at load time and is VRAM the experts cannot have for
  the whole run. On 16 GB that rule gives 65536.
- **Planning is automatic.** The plan is computed from measured hardware, not
  from environment variables the user has to tune by hand.

## Intended, not built

These are design positions the engine has not earned yet. They are recorded here
so the roadmap has reasons attached, not because the code does any of it:

- **Per-layer routing prediction.** A single global hot-set overfits whatever the
  last prompt happened to exercise; heat belongs per layer, over a sliding
  window. Nothing tracks it today — the expert map in the console is simulated,
  and labelled as such, because the compute process does not report which experts
  a token activated.
- **GPU-side dequantisation.** Unpacking on the CPU would spend PCIe bandwidth on
  every expert fetch; doing it in the kernel roughly halves the transfer.
- **Paged KV.** A single conversation slot forces a cold start on every context
  switch; paged blocks would let sessions share prefix pages.

## Why this is worth doing at this size

A 20 GiB model on a 16 GB card with 32 GB of system RAM does not need streaming.
It needs a good split, and the difference between a good split and an obvious one
is a factor of two. `setup` reports the ceiling before it measures anything —
here, about 35 GB of weights fit without touching disk — and the same mechanism
scales down to a smaller card unchanged.

It does not scale up indefinitely, and the honest version of that is worth
writing down: a 200B MoE at Q4_K is about 112 GB, of which ~74% spills on this
machine; Q2_K brings it to 58 GB and still spills ~42%. Fitting a 200B into 35 GB
needs under 1.5 bits per weight. The answer for that model is more RAM, not a
different setting.
