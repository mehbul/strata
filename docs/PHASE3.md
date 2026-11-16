# Phase 3 (append to plan)

## Phase 3 — Specialized compute backends (optional)

Keep llama.cpp as the universal backend. Add specialized engines that provide
real expert caching for supported architectures, selected automatically from
the GGUF's `general.architecture`. Fall back to llama.cpp when the specialized
binary is absent or the arch is unsupported.

### 3.1 Backend abstraction
- `src/backend.rs`: `Backend` enum (LlamaCpp, StrataNiko, StrataPoly), plus
  `select(arch: &str) -> Backend` and `binary_name(backend) -> &'static str`.
- Niko: qwen4exp
- Poly: glm5-next, glm5next, qwen35moe, clover_moe, gemma4, deepseek4, lfm2moe, flint, kolibri1
- else: LlamaCpp

### 3.2 Runtime discovery
- `runtime/<backend>/` directory layout; `runner::Runtime::discover_backend(backend)`.
- If binary missing, fall back to LlamaCpp with a printed note.

### 3.3 Flag translation
- Each backend has its own flag vocabulary. `Flags::to_args_for(backend)`.

### 3.4 Serve integration
- In serve(), pick backend from GGUF arch; print which backend is used.
- `--backend` override (auto|llamacpp|niko|poly).

### 3.5 Packaging
- Document how to fetch specialized binaries; not bundled (license/size).

## Status
- 3.1 backend abstraction: DONE
- 3.2 runtime discovery: DONE (fallback)
- 3.3 flag translation: DONE (llamacpp fully; others minimal)
- 3.4 serve integration: DONE
- 3.5 packaging docs: DONE
