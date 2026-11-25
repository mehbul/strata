# Theory — Inkling-2-Mini on RX 7600 XT

Measured on 2026-11-25. Sources: HIP device-to-device copy via `ctypes` (256/512 MiB),
FP16 GEMM via rocBLAS `rocblas_hgemm` (4096³/8192³), GGUF tensor index via
`gguf_tensors.py` and `strata plan`.

## Hardware (measured, not spec)

| Quantity | Value |
|---|---|
| Device | AMD Radeon RX 7600 XT, gfx1102, 16.0 GiB discrete GDDR6 |
| Host | Ryzen 5 7600X, 32 GiB RAM, 12 threads |
| Device copy bandwidth | 246.1 GB/s (256 MiB), 250.7 GB/s (512 MiB) |
| FP16 GEMM | 32.5 TFLOPS @ 4096³, 26.7 TFLOPS @ 8192³ |
| Practical ceiling assumption | 100% of measured bandwidth, 50% of sustained GEMM (13.4 TFLOPS) |
| Compute runtime | llama.cpp b12704 (commit 72cda4a22), ROCm backend, flash attention on |

## Model (from the GGUF, not the model card)

| Component | Value |
|---|---|
| Architecture | `inkling2_loop`, 24 layers, 128 experts/layer, top-8 |
| Stored weights | 15.47 GiB total; dense 1.71 GiB, routed 13.76 GiB |
| LM head (`output.weight`) | 492,106,752 B (0.492 GB), 201,024 vocab × 2304 |
| Token embedding | 492,106,752 B |
| KV | 98,304 B/token (all 24 layers, 8 KV heads) |
| Declared context | 1,048,576 tokens; practical ceiling here 131,072 |

`dense_gb` in `model-hardware.json` is total-dense minus `output.weight`, so the
head is not counted twice.

## Ceilings (from `ceiling.py`)

```
Decode ceiling, concurrency 1 (aggregate tok/s)
  drafts accept       4K      16K      32K      64K
       0    1.0       79       57       42       27

Decode ceiling, concurrency 2 (aggregate tok/s)
       0    1.0      226      146       99       61

Prefill ceiling (tok/s, practical):  4K 2069 | 16K 1709 | 32K 1387 | 64K 1008
KV: 25.77 GB per 256K-token stream
```

No speculative decoding candidate exists for this model here, so the only row
is k=0 drafts.

## Where we stand (2026-11-25)

From `strata tune --ctx 32768 --save` (not the same harness as `bench.py`, do not
mix the numbers): 111 tok/s prefill at 8192 tokens, 23.4 tok/s decode c1.
Against the ceilings at 32K: decode ~56% of 42 tok/s; prefill ~8% of 1387 tok/s.

The prefill gap is the largest remaining opportunity, but `tune`'s prefill
number is a wall-clock measurement over a client-side doc build and may include
per-request overhead. The `bench.py` baseline will give the comparable number.
