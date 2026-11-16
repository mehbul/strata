# Contributing to Strata

Thanks for looking. Strata is a small, opinionated project, so a few things are
worth knowing before you open a pull request.

## What Strata is, and what it is not

Strata owns everything *around* the matrix multiplies: reading the GGUF, solving
where weights sit across VRAM and system RAM, tuning that placement by
measurement, compacting context, and serving an OpenAI-compatible API. The
kernels themselves are llama.cpp's, fetched at setup.

That boundary is deliberate. A pull request that reimplements the forward pass
will not be merged; one that improves how Strata *places* weights almost
certainly will.

## Getting set up

```powershell
git clone https://github.com/mehbul/strata
cd strata
cargo build --release
```

You also need a llama.cpp ROCm build in `runtime/` and a MoE GGUF in
`rocm/models/`. Both are gitignored and both are fetched, not committed. See the
README for the exact commands.

Run `.\strata.ps1` to build what is stale, configure the machine if it has never
been measured, and serve.

## Before you send a pull request

```powershell
cargo fmt --all
cargo test
```

Both must pass. CI runs them on Windows and Linux, plus the web console's
typecheck and build. Clippy runs too but is advisory: the crate carries
pre-existing style lints that are not defects.

If your change touches the web console:

```powershell
cd web
npm ci
npm run build
```

## What makes a good change here

- **Measure, do not guess.** This project's whole character is that its numbers
  were measured on real hardware. If you claim a speedup, say what machine, what
  model, what quantisation, and what the before and after were.
- **Be honest about what is simulated.** The expert map is simulated because
  llama.cpp exposes no router telemetry. Anything that reports a number must
  either measure it or label it.
- **Keep the fallbacks.** Specialized backends fall back to llama.cpp; a failed
  model switch rolls back to the previous one. Do not remove those paths.
- **One concern per pull request.** Small and reviewable beats large and clever.

## Reporting a bug

Include:

- Your GPU and driver, and the output of `strata info`
- The model file and quantisation
- The command you ran and what happened instead
- Whether `.\runtime\llama-server.exe --list-devices` lists your GPU

That last one matters more than it looks: a missing backend library shows up as
an empty device list with nothing in the log, and everything silently falls back
to CPU.

## Questions

Open an issue. There is no support contract, but issues get read.
